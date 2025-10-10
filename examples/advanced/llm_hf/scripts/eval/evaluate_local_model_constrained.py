#!/usr/bin/env python3
"""
Offline evaluation script that EXACTLY matches the in-loop evaluation during training.
This replicates the same evaluation logic used in ConstrainedSFTTrainer.
"""

import argparse
import csv
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

# Add the src directory to the path to import training_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training_utils import (
    format_instruction,
    filter_by_length_messages,
)


class ConstrainedLogitsProcessor(LogitsProcessor):
    """Logits processor that constrains generation to only allowed tokens."""

    def __init__(self, allowed_token_ids):
        """
        Args:
            allowed_token_ids: List of token IDs that are allowed (e.g., for 'A', 'B', 'C', 'D', 'E').
        """
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create mask with -inf for all tokens
        mask = torch.full_like(scores, float("-inf"))

        # Set allowed tokens to 0 (no penalty)
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0

        scores = scores + mask
        return scores

# Set CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_SEQ_LENGTH = 4096

def evaluate_model_on_dataset(
    model_name: str,
    dataset_paths: List[str],
    batch_size: int = 8,
    output_dir: str = "results",
    seed: int = 42,
    num_examples: int = -1,
    choices: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a HuggingFace model using EXACTLY the same methodology as training evaluation.

    Args:
        model_name: HuggingFace model name or local path
        dataset_paths: List of paths to JSONL dataset files
        batch_size: Batch size for processing
        output_dir: Directory to save results
        seed: Random seed (should match training)
        num_examples: Number of examples to evaluate (-1 for all examples)
        choices: List of choice tokens (default: ["A", "B", "C", "D", "E"])

    Returns:
        Dictionary with evaluation results
    """
    if choices is None:
        choices = ["A", "B", "C", "D", "E"]

    print(f"Starting evaluation with exact training match methodology...")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_paths}")
    print(f"Seed: {seed}")

    # Set deterministic seeds (same as training)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with EXACT same parameters as training
    if device == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs")
        # Match training exactly: set default dtype and use same model loading params
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # EXACT match to training
            use_cache=False,  # EXACT match to training
            device_map="auto",  # Let it auto-assign devices
            low_cpu_mem_usage=True,
        )

        torch.set_default_dtype(default_dtype)  # Restore default dtype
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

    if device == "cpu":
        model = model.to(device)

    # Set model configuration to match training exactly
    model.config.pretraining_tp = 1

    # Set up padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if tokenizer has a chat template
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    print(f"Chat template available: {has_chat_template}")
    if has_chat_template:
        print(f"Chat template preview: {tokenizer.chat_template[:200]}...")

    # Only set fallback if no template exists (EXACTLY like training)
    if not has_chat_template:
        # Fallback chat template for models without one (SAME as training)
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    # Load dataset EXACTLY like training
    print(f"Loading datasets from: {dataset_paths}")
    dataset_raw = datasets.load_dataset("json", data_files=dataset_paths, split="train")
    print(f"Raw dataset size: {len(dataset_raw)}")

    # Apply EXACT same processing as training
    # 1. Shuffle with same seed
    dataset_shuffled = dataset_raw.shuffle(seed=seed)

    # 2. Select examples based on num_examples parameter
    if num_examples > 0:
        dataset_selected = dataset_shuffled.select(range(min(num_examples, len(dataset_shuffled))))
        print(f"Selected first {len(dataset_selected)} examples after shuffling with seed={seed}")
    else:
        dataset_selected = dataset_shuffled
        print(f"Using all {len(dataset_selected)} examples after shuffling with seed={seed}")

    # 3. Apply format_instruction (EXACT same as training)
    dataset_formatted = dataset_selected.map(format_instruction, batched=True, remove_columns=dataset_selected.column_names)
    print(f"Applied format_instruction, dataset size: {len(dataset_formatted)}")

    # 4. Apply filtering (EXACT same as training)
    original_size = len(dataset_formatted)
    dataset_filtered = dataset_formatted.filter(lambda x: filter_by_length_messages(x, tokenizer, MAX_SEQ_LENGTH))
    print(f"Applied filtering: {original_size} -> {len(dataset_filtered)} examples")

    # Show a sample of the processed data
    if len(dataset_filtered) > 0:
        sample = dataset_filtered[0]
        print(f"Sample processed data structure: {sample.keys()}")
        if "messages" in sample:
            print(f"Sample messages: {sample['messages']}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get allowed token IDs for constrained generation
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in choices]

    # Verify all tokens are in vocabulary
    if any(tid == tokenizer.unk_token_id for tid in allowed_token_ids):
        missing_tokens = [choices[i] for i, tid in enumerate(allowed_token_ids) if tid == tokenizer.unk_token_id]
        raise ValueError(f"Some choice tokens not in vocabulary: {missing_tokens}")

    print(f"Choice tokens mapped to IDs: {dict(zip(choices, allowed_token_ids))}")

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

    # Create constrained logits processor
    logits_processors = LogitsProcessorList([ConstrainedLogitsProcessor(allowed_token_ids)])

    # Setup incremental JSON writing
    predictions_file = os.path.join(output_dir, "predictions.json")
    progress_file = os.path.join(output_dir, "progress.txt")

    # Write opening bracket for JSON array
    with open(predictions_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    # Initialize progress tracking file
    with open(progress_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset paths: {dataset_paths}\n")
        f.write(f"Total examples: {len(dataset_filtered)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("=" * 50 + "\n")
        f.write("Progress updates:\n")

    first_result = True

    # Process in batches
    dataset_size = len(dataset_filtered)
    print(f"Processing {dataset_size} examples in batches of {batch_size}...")

    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, dataset_size, batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, dataset_size)
            batch_indices = list(range(batch_start, batch_end))
            batch = dataset_filtered.select(batch_indices)

            # Process each example in the batch
            for i, example in enumerate(batch):
                # Get the input prompt (everything before the answer)
                messages = example["messages"]

                # Extract the user prompt (before assistant answer)
                prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]

                # Apply chat template to get input prompt
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                    formatted_prompt = tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Fallback if no chat template
                    formatted_prompt = prompt_messages[-1]["content"] if prompt_messages else ""

                # Tokenize input
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=4096,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate with constraints
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=1,
                        logits_processor=logits_processors,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Extract generated token
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Map prediction to choice index
                try:
                    predicted_idx = choices.index(predicted_text)
                except ValueError:
                    predicted_idx = -1

                # Get ground truth from messages
                ground_truth_text = None
                for msg in messages:
                    if msg["role"] == "assistant":
                        ground_truth_text = msg["content"].strip()
                        break

                if ground_truth_text and ground_truth_text in choices:
                    ground_truth_idx = choices.index(ground_truth_text)
                else:
                    ground_truth_idx = -1

                # Check if correct
                is_correct = predicted_idx == ground_truth_idx and predicted_idx != -1
                if is_correct:
                    correct += 1
                total += 1

                # Store prediction
                prediction_record = {
                    "id": f"item_{total-1}",
                    "predicted_choice": predicted_text,
                    "predicted_idx": predicted_idx,
                    "ground_truth_choice": ground_truth_text,
                    "ground_truth_idx": ground_truth_idx,
                    "is_correct": is_correct,
                }
                all_predictions.append(prediction_record)

                # Write result to JSON file immediately
                with open(predictions_file, "a", encoding="utf-8") as f:
                    if not first_result:
                        f.write(",\n")
                    else:
                        first_result = False
                    json.dump(prediction_record, f, indent=2, ensure_ascii=False)

                # Save progress updates frequently
                if total % 10 == 0 or total == 1:
                    current_accuracy = correct / total if total > 0 else 0.0
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    # Print to console
                    print(f"Progress: {total} examples processed, current accuracy: {current_accuracy:.4f}")

                    # Append to progress file
                    with open(progress_file, "a", encoding="utf-8") as f:
                        f.write(f"{timestamp}: {total}/{dataset_size} examples, accuracy: {current_accuracy:.4f}\n")

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Prepare results summary
    results = {
        "model_name": model_name,
        "dataset_paths": dataset_paths,
        "total_examples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "batch_size": batch_size,
        "choices": choices,
        "seed": seed,
        "methodology": "exact_training_match",
    }

    print(f"\nEvaluation Results:")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    # Close the JSON array
    with open(predictions_file, "a", encoding="utf-8") as f:
        f.write("\n]")
    print(f"Predictions saved to: {predictions_file}")

    # Save metrics to CSV
    import csv
    metrics_file = os.path.join(output_dir, "metrics.csv")
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["model_name", model_name])
        writer.writerow(["dataset_paths", " ".join(dataset_paths)])
        writer.writerow(["total_examples", total])
        writer.writerow(["correct_predictions", correct])
        writer.writerow(["accuracy", f"{accuracy:.4f}"])
        writer.writerow(["batch_size", batch_size])
        writer.writerow(["seed", seed])
        writer.writerow(["methodology", "exact_training_match"])
    print(f"Metrics saved to: {metrics_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation that EXACTLY matches training evaluation")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--dataset", "-d", required=True, help="Path to JSONL dataset file")
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument("--output-dir", "-o", default="results_exact_match", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (should match training)")
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=-1,
        help="Number of examples to evaluate (-1 for all examples, 200 to match training)",
    )
    parser.add_argument(
        "--choices",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="Choice tokens (default: A B C D E)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1

    try:
        results = evaluate_model_on_dataset(
            model_name=args.model,
            dataset_paths=[args.dataset],
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            seed=args.seed,
            num_examples=args.num_examples,
            choices=args.choices,
        )

        print(f"\n✅ Evaluation completed successfully!")
        print(f"This evaluation uses EXACTLY the same methodology as training evaluation.")
        print(f"Accuracy should now match the training evaluation results.")
        return 0

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())