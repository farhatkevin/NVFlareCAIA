#!/usr/bin/env python3
"""
Constrained evaluation script for multiple choice questions.
Evaluates any HuggingFace model on JSONL datasets with constrained generation (A, B, C, D, E only).
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import datasets
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

# Set CUDA memory allocation configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


def load_dataset_files(dataset_paths: List[str]):
    """Load dataset from one or more JSONL files using datasets library."""
    print(f"Loading datasets from: {dataset_paths}")

    # Use datasets library to load and concatenate files automatically
    dataset = datasets.load_dataset("json", data_files=dataset_paths, split="train")

    print(f"Total examples loaded: {len(dataset)}")
    return dataset


def evaluate_model_on_dataset(
    model_name: str,
    dataset_paths: List[str],
    batch_size: int = 8,
    output_dir: str = "results",
    choices: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a HuggingFace model on constrained multiple choice dataset.

    Args:
        model_name: HuggingFace model name or local path
        dataset_paths: List of paths to JSONL dataset files
        batch_size: Batch size for processing
        output_dir: Directory to save results
        choices: List of choice tokens (default: ["A", "B", "C", "D", "E"])

    Returns:
        Dictionary with evaluation results
    """
    if choices is None:
        choices = ["A", "B", "C", "D", "E"]

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with multi-GPU support and memory optimization
    if device == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # Let the model use its preferred dtype (likely BFloat16)
            device_map="balanced",  # Balanced memory usage across GPUs
            max_memory={i: "78GB" for i in range(torch.cuda.device_count())},
            low_cpu_mem_usage=True,
            # tensor_parallel=num_gpus := torch.cuda.device_count(
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

    if device == "cpu":
        model = model.to(device)

    # Set up padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for decoder-only models (better for generation)
    tokenizer.padding_side = "left"

    # Get allowed token IDs
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in choices]

    # Verify all tokens are in vocabulary
    if any(tid == tokenizer.unk_token_id for tid in allowed_token_ids):
        missing_tokens = [choices[i] for i, tid in enumerate(allowed_token_ids) if tid == tokenizer.unk_token_id]
        raise ValueError(f"Some choice tokens not in vocabulary: {missing_tokens}")

    print(f"Choice tokens mapped to IDs: {dict(zip(choices, allowed_token_ids))}")

    # Load dataset
    dataset = load_dataset_files(dataset_paths)

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
        f.write(f"Total examples: {len(dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write("=" * 50 + "\n")
        f.write("Progress updates:\n")

    first_result = True

    # Process in batches
    dataset_size = len(dataset)
    print(f"Processing {dataset_size} examples in batches of {batch_size}...")

    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, dataset_size, batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, dataset_size)
            batch_indices = list(range(batch_start, batch_end))
            batch = dataset.select(batch_indices)

            # Prepare batch prompts and apply chat template
            raw_prompts = list(batch["prompt"])

            # Apply chat template if available
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompts = []
                for prompt in raw_prompts:
                    # Format as a user message
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(formatted_prompt)
            else:
                prompts = raw_prompts

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,  # Adjust based on your needs
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate predictions
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=1,
                    logits_processor=logits_processors,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Process batch results
            for i, output in enumerate(outputs):
                # Extract generated token
                input_length = inputs["input_ids"][i].shape[0]
                generated_tokens = output[input_length:]
                predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Map prediction to choice index
                try:
                    predicted_idx = choices.index(predicted_text)
                except ValueError:
                    # Fallback: if prediction not in choices, mark as -1
                    predicted_idx = -1

                # Get ground truth
                ground_truth_idx = list(batch["answer_idx"])[i]

                # Check if correct
                is_correct = predicted_idx == ground_truth_idx
                if is_correct:
                    correct += 1
                total += 1

                # Store prediction
                batch_prompts = list(batch["prompt"])
                batch_answer_idx = list(batch["answer_idx"])

                # Handle optional fields
                batch_ids = (
                    list(batch["id"]) if "id" in batch else [f"item_{total - 1 + j}" for j in range(len(batch_prompts))]
                )
                batch_choices = (
                    list(batch["choices"]) if "choices" in batch else [[] for _ in range(len(batch_prompts))]
                )

                item_id = batch_ids[i]
                item_prompt = batch_prompts[i]
                item_choices = batch_choices[i]

                prediction_record = {
                    "id": item_id,
                    "prompt": item_prompt[:100] + "..." if len(item_prompt) > 100 else item_prompt,
                    "choices": item_choices,
                    "ground_truth_idx": ground_truth_idx,
                    "ground_truth_choice": (
                        choices[ground_truth_idx] if 0 <= ground_truth_idx < len(choices) else "Unknown"
                    ),
                    "predicted_idx": predicted_idx,
                    "predicted_choice": predicted_text,
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

                # Also save a quick metrics snapshot every 50 examples
                if total % 50 == 0:
                    temp_metrics_file = os.path.join(output_dir, "metrics_temp.csv")
                    with open(temp_metrics_file, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["metric", "value"])
                        writer.writerow(["model_name", model_name])
                        writer.writerow(["dataset_paths", " ".join(dataset_paths)])
                        writer.writerow(["total_examples_so_far", total])
                        writer.writerow(["correct_predictions", correct])
                        writer.writerow(["current_accuracy", f"{current_accuracy:.4f}"])
                        writer.writerow(["last_updated", time.strftime("%Y-%m-%d %H:%M:%S")])
                        writer.writerow(["batch_size", batch_size])

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
    print(f"Metrics saved to: {metrics_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace models on constrained MCQ datasets")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name or local path")
    parser.add_argument(
        "--dataset",
        "-d",
        nargs="+",
        required=True,
        help="Path(s) to JSONL dataset file(s)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--choices",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="Choice tokens (default: A B C D E)",
    )

    args = parser.parse_args()

    # Validate inputs
    for dataset_file in args.dataset:
        if not os.path.exists(dataset_file):
            print(f"Error: Dataset file not found: {dataset_file}")
            return 1

    try:
        results = evaluate_model_on_dataset(
            model_name=args.model,
            dataset_paths=args.dataset,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            choices=args.choices,
        )
        print(f"\nEvaluation completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
