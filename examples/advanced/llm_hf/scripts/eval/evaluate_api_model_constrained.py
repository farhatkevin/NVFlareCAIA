#!/usr/bin/env python3
"""
Constrained evaluation script for API-based models (OpenAI, Anthropic, etc).
Evaluates closed-source models on JSONL datasets with constrained generation (A, B, C, D, E only).
"""

import argparse
import json
import csv
import os
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import datasets
import asyncio
import aiohttp
from dataclasses import dataclass
# import tiktoken  # Optional - will use rough estimation if not available


@dataclass
class APIConfig:
    """Configuration for different API providers."""
    name: str
    base_url: str
    headers: Dict[str, str]
    request_format: callable
    response_parser: callable
    rate_limit_delay: float = 1.0  # seconds between requests


def load_dataset_files(dataset_paths: List[str], max_tokens: int = 4000):
    """Load dataset from one or more JSONL files using datasets library."""
    print(f"Loading datasets from: {dataset_paths}")
    dataset = datasets.load_dataset("json", data_files=dataset_paths, split="train").shuffle(seed=42).select(range(2500))

    # Use a simple tokenizer for length filtering (GPT-style tokenizer)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")  # Use GPT-2 tokenizer as proxy
        print(f"Using tokenizer for prompt length filtering (max {max_tokens} tokens)")

        def filter_by_length(example):
            try:
                tokens = tokenizer.encode(example['prompt'])
                return len(tokens) <= max_tokens
            except:
                # Fallback to character-based estimation if tokenization fails
                return len(example['prompt']) <= max_tokens * 4

    except ImportError:
        print("Transformers not available, using character-based estimation")
        def filter_by_length(example):
            # Rough estimation: ~4 characters per token
            return len(example['prompt']) <= max_tokens * 4

    # Filter out prompts that are too long
    original_size = len(dataset)
    dataset = dataset.filter(filter_by_length)
    filtered_size = len(dataset)

    print(f"Filtered dataset: {original_size} -> {filtered_size} examples ({original_size - filtered_size} removed due to length)")
    print(f"Total examples loaded: {filtered_size}")
    return dataset


def openai_request_format(model_name: str, prompt: str, choices: List[str]) -> Dict[str, Any]:
    """Format request for OpenAI API."""
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 4096,  # Give more space for reasoning + answer
        "temperature": 1,  # Make it deterministic
        "top_p": 1.0
    }


def anthropic_request_format(model_name: str, prompt: str, choices: List[str]) -> Dict[str, Any]:
    """Format request for Anthropic API."""
    return {
        "model": model_name,
        "max_tokens": 4096,  # Give more space for reasoning + answer
        "temperature": 1,
        "messages": [{"role": "user", "content": prompt}]
    }


def extract_choice_from_response(response_text: str, choices: List[str]) -> str:
    """Extract the choice letter from response text."""
    response_text = response_text.strip()

    # First, try exact match
    if response_text in choices:
        return response_text

    # Then try to find any of the choice letters in the response
    for choice in choices:
        if choice in response_text:
            return choice

    return response_text  # Return as-is if no match found


def openai_response_parser(response_data: Dict[str, Any]) -> str:
    """Parse OpenAI API response."""
    try:
        content = response_data["choices"][0]["message"]["content"].strip()
        return extract_choice_from_response(content, ["A", "B", "C", "D", "E"])
    except (KeyError, IndexError):
        return ""


def anthropic_response_parser(response_data: Dict[str, Any]) -> str:
    """Parse Anthropic API response."""
    try:
        content = response_data["content"][0]["text"].strip()
        return extract_choice_from_response(content, ["A", "B", "C", "D", "E"])
    except (KeyError, IndexError):
        return ""


def get_api_config(provider: str, model_name: str, api_key: str) -> APIConfig:
    """Get API configuration for different providers."""

    if provider.lower() == "openai":
        return APIConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            request_format=lambda prompt, choices: openai_request_format(model_name, prompt, choices),
            response_parser=openai_response_parser,
            rate_limit_delay=0.1
        )

    elif provider.lower() == "anthropic":
        return APIConfig(
            name="Anthropic",
            base_url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            request_format=lambda prompt, choices: anthropic_request_format(model_name, prompt, choices),
            response_parser=anthropic_response_parser,
            rate_limit_delay=0.5
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def make_api_request(
    session: aiohttp.ClientSession,
    config: APIConfig,
    prompt: str,
    choices: List[str],
    max_retries: int = 3,
    debug: bool = False
) -> Optional[str]:
    """Make an API request with retry logic."""

    request_data = config.request_format(prompt, choices)

    if debug:
        print(f"Making API request to {config.base_url}")
        print(f"Prompt length: {len(prompt)} characters")

        # Try to estimate actual token count for this specific prompt
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
            actual_tokens = len(tokenizer.encode(prompt))
            print(f"Estimated token count: {actual_tokens} tokens")
        except:
            print("Could not estimate token count")

        print(f"Request data: {json.dumps(request_data, indent=2)[:500]}...")

    for attempt in range(max_retries):
        try:
            async with session.post(
                config.base_url,
                headers=config.headers,
                json=request_data
            ) as response:

                if response.status == 200:
                    response_data = await response.json()
                    parsed_response = config.response_parser(response_data)

                    if debug:
                        print(f"Response status: {response.status}")
                        print(f"Parsed response: '{parsed_response}'")

                        # Show full generation content and usage stats
                        if 'choices' in response_data and len(response_data['choices']) > 0:
                            choice = response_data['choices'][0]
                            if 'message' in choice and 'content' in choice['message']:
                                full_content = choice['message']['content']
                                print(f"FULL GENERATION CONTENT:")
                                print(f"=========================")
                                print(f"'{full_content}'")
                                print(f"=========================")
                                print(f"Content length: {len(full_content)} characters")
                            if 'finish_reason' in choice:
                                print(f"Finish reason: {choice['finish_reason']}")

                        # Show token usage
                        if 'usage' in response_data:
                            usage = response_data['usage']
                            print(f"Token usage: prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')}, total={usage.get('total_tokens', 'N/A')}")

                            # Flag unusual token usage
                            completion_tokens = usage.get('completion_tokens', 0)
                            if completion_tokens > 50:
                                print(f"WARNING: High completion token usage ({completion_tokens}) for short response!")

                        print("-" * 50)

                    return parsed_response

                elif response.status == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                else:
                    error_text = await response.text()
                    print(f"API error {response.status}: {error_text}")
                    return None

        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}): {type(e).__name__}: {str(e)}")
            if debug:
                import traceback
                print("Full error traceback:")
                traceback.print_exc()
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    print(f"All {max_retries} attempts failed")
    return None


async def evaluate_batch(
    session: aiohttp.ClientSession,
    config: APIConfig,
    batch: List[Dict[str, Any]],
    choices: List[str]
) -> List[Dict[str, Any]]:
    """Evaluate a batch of examples."""

    tasks = []
    for i, item in enumerate(batch):
        # Enable debug for first request to see what's happening
        debug = (i == 0)
        task = make_api_request(session, config, item['prompt'], choices, debug=debug)
        tasks.append(task)

    # Add delay between batches to respect rate limits
    await asyncio.sleep(config.rate_limit_delay * len(batch))

    predictions = await asyncio.gather(*tasks)

    results = []
    for i, (item, predicted_text) in enumerate(zip(batch, predictions)):
        # Map prediction to choice index
        if predicted_text and predicted_text in choices:
            predicted_idx = choices.index(predicted_text)
        else:
            predicted_idx = -1

        # Get ground truth
        ground_truth_idx = item['answer_idx']
        is_correct = (predicted_idx == ground_truth_idx)

        # Store result
        result = {
            'id': item.get('id', f"item_{i}"),
            'prompt': item['prompt'][:100] + "..." if len(item['prompt']) > 100 else item['prompt'],
            'choices': item.get('choices', []),
            'ground_truth_idx': ground_truth_idx,
            'ground_truth_choice': choices[ground_truth_idx] if 0 <= ground_truth_idx < len(choices) else "Unknown",
            'predicted_idx': predicted_idx,
            'predicted_choice': predicted_text or "NO_RESPONSE",
            'is_correct': is_correct
        }
        results.append(result)

    return results


async def evaluate_model_on_dataset(
    provider: str,
    model_name: str,
    api_key: str,
    dataset_paths: List[str],
    batch_size: int = 5,
    output_dir: str = "results",
    choices: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate an API-based model on constrained multiple choice dataset.

    Args:
        provider: API provider ("openai", "anthropic")
        model_name: Model name/ID for the API
        api_key: API key for the provider
        dataset_paths: List of paths to JSONL dataset files
        batch_size: Batch size for processing (smaller for API rate limits)
        output_dir: Directory to save results
        choices: List of choice tokens (default: ["A", "B", "C", "D", "E"])

    Returns:
        Dictionary with evaluation results
    """
    if choices is None:
        choices = ["A", "B", "C", "D", "E"]

    print(f"Evaluating {provider} model: {model_name}")

    # Get API configuration
    config = get_api_config(provider, model_name, api_key)

    # Load dataset
    dataset = load_dataset_files(dataset_paths, max_tokens=4000)
    dataset_size = len(dataset)

    # Create evaluation_output parent directory if it doesn't exist
    parent_dir = "evaluation_output"
    os.makedirs(parent_dir, exist_ok=True)

    # Create the specific output directory inside evaluation_output
    full_output_dir = os.path.join(parent_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    # Update output_dir to use the full path
    output_dir = full_output_dir

    # Prepare results storage
    all_predictions = []
    correct = 0
    total = 0

    # Setup incremental JSON writing
    predictions_file = os.path.join(output_dir, "predictions.json")
    progress_file = os.path.join(output_dir, "progress.txt")

    # Write opening bracket for JSON array
    with open(predictions_file, 'w', encoding='utf-8') as f:
        f.write('[\n')

    # Initialize progress tracking file
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Provider: {provider}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total examples: {dataset_size}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("=" * 50 + "\n")
        f.write("Progress updates:\n")

    first_result = True

    # Create async session
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Process in batches
        print(f"Processing {dataset_size} examples in batches of {batch_size}...")

        for batch_start in tqdm(range(0, dataset_size, batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, dataset_size)
            batch_indices = list(range(batch_start, batch_end))
            batch = dataset.select(batch_indices)

            # Convert batch to list of dicts
            batch_items = []
            batch_prompts = list(batch['prompt'])
            batch_answer_idx = list(batch['answer_idx'])

            # Handle optional fields
            batch_ids = list(batch['id']) if 'id' in batch else [f"item_{batch_start+i}" for i in range(len(batch_prompts))]
            batch_choices = list(batch['choices']) if 'choices' in batch else [[] for _ in range(len(batch_prompts))]

            for i in range(len(batch_prompts)):
                item = {
                    'prompt': batch_prompts[i],
                    'answer_idx': batch_answer_idx[i],
                    'id': batch_ids[i],
                    'choices': batch_choices[i]
                }
                batch_items.append(item)

            # Evaluate batch
            batch_results = await evaluate_batch(session, config, batch_items, choices)

            # Update counters and write results incrementally
            for result in batch_results:
                if result['is_correct']:
                    correct += 1
                total += 1
                all_predictions.append(result)

                # Write result to JSON file immediately
                with open(predictions_file, 'a', encoding='utf-8') as f:
                    if not first_result:
                        f.write(',\n')
                    else:
                        first_result = False
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Save progress updates frequently
                if total % 10 == 0 or total == 1:
                    current_accuracy = correct / total if total > 0 else 0.0
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                    # Print to console
                    print(f"Progress: {total} examples processed, current accuracy: {current_accuracy:.4f}")

                    # Append to progress file
                    with open(progress_file, 'a', encoding='utf-8') as f:
                        f.write(f"{timestamp}: {total}/{dataset_size} examples, accuracy: {current_accuracy:.4f}\n")

                # Also save a quick metrics snapshot every 50 examples
                if total % 50 == 0:
                    temp_metrics_file = os.path.join(output_dir, "metrics_temp.csv")
                    with open(temp_metrics_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['metric', 'value'])
                        writer.writerow(['provider', provider])
                        writer.writerow(['model_name', model_name])
                        writer.writerow(['total_examples_so_far', total])
                        writer.writerow(['correct_predictions', correct])
                        writer.writerow(['current_accuracy', f"{current_accuracy:.4f}"])
                        writer.writerow(['last_updated', time.strftime('%Y-%m-%d %H:%M:%S')])

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Prepare results summary
    results = {
        'provider': provider,
        'model_name': model_name,
        'dataset_paths': dataset_paths,
        'total_examples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'batch_size': batch_size,
        'choices': choices
    }

    print(f"\nEvaluation Results:")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    # Close the JSON array
    with open(predictions_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    print(f"Predictions saved to: {predictions_file}")

    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, "metrics.csv")
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['provider', provider])
        writer.writerow(['model_name', model_name])
        writer.writerow(['dataset_paths', ' '.join(dataset_paths)])
        writer.writerow(['total_examples', total])
        writer.writerow(['correct_predictions', correct])
        writer.writerow(['accuracy', f"{accuracy:.4f}"])
        writer.writerow(['batch_size', batch_size])
    print(f"Metrics saved to: {metrics_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate API-based models on constrained MCQ datasets")
    parser.add_argument("--provider", "-p", required=True, choices=["openai", "anthropic"],
                        help="API provider")
    parser.add_argument("--model", "-m", required=True, help="Model name/ID for the API")
    parser.add_argument("--api-key", "-k", required=True, help="API key for the provider")
    parser.add_argument("--dataset", "-d", nargs='+', required=True, help="Path(s) to JSONL dataset file(s)")
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Batch size for evaluation (default: 5)")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory for results (default: results)")
    parser.add_argument("--choices", nargs="+", default=["A", "B", "C", "D", "E"],
                        help="Choice tokens (default: A B C D E)")

    args = parser.parse_args()

    # Validate inputs
    for dataset_file in args.dataset:
        if not os.path.exists(dataset_file):
            print(f"Error: Dataset file not found: {dataset_file}")
            return 1

    try:
        results = asyncio.run(evaluate_model_on_dataset(
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            dataset_paths=args.dataset,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            choices=args.choices
        ))
        print(f"\nEvaluation completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())