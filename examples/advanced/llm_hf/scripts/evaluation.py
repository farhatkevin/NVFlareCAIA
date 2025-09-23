import argparse
import datasets
import torch
import torch.distributed as dist
import os
import shutil
import sys
from functools import partial

# Add the parent directory to the Python path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training_utils import (
    compute_metrics,
    preprocess_logits_for_metrics,
    format_instruction,
    filter_by_length_messages,
)
from peft import PeftModel, LoraConfig
from trl import SFTConfig, SFTTrainer

MAX_SEQ_LENGTH = 4096

# def filter_by_length_messages(example, tokenizer, max_length):
#     try:
#         # Now we want a generation prompt
#         text = tokenizer.apply_chat_template(
#             example["messages"],
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         return len(tokenizer.encode(text)) <= max_length
#     except Exception as e:
#         print(f"Error in filter_by_length_messages: {e}")
#         return False


# def format_instruction(example):
#     """Convert to messages format that TRL expects"""
#     system_prompt = "You are a medical AI expert."

#     if isinstance(example["prompt"], list):
#         # Batch processing
#         formatted_messages = []
#         for i, prompt in enumerate(example["prompt"]):
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt},
#             ]
#             formatted_messages.append(messages)
#         return {"messages": formatted_messages}
#     else:
#         # Single example
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": example["prompt"]},
#         ]
#         return {"messages": messages}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument(
        "--data_path_valid",
        type=str,
        nargs="+",                         # <-- allow multiple
        required=True,
        help="Path(s) to validation dataset JSONL file(s)."
    )
    parser.add_argument("--peft", action="store_true", help="Set this flag if the model uses PEFT.")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading and preparing dataset...")
    data_files = {"train": args.data_path_valid}  # list works
    dataset_valid = datasets.load_dataset("json", data_files=data_files, split="train")
    
    # 1. Format the data into 'messages' format (same as training)
    dataset_valid = dataset_valid.map(format_instruction, batched=True, remove_columns=dataset_valid.column_names)
    
    # 2. Filter out sequences that are too long (same as training)
    dataset_valid = dataset_valid.filter(lambda x: filter_by_length_messages(x, tokenizer, MAX_SEQ_LENGTH))
    dataset_valid = dataset_valid.select(range(min(100, len(dataset_valid))))  # Limit to 1000 samples for quick eval
    
    print("Dataset prepared.")

    # Load model
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if args.peft:
        # Load the PEFT adapter
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        print("PEFT model loaded from:", args.model_name_or_path)
    else:
        # Load the full SFT model
        model = base_model
        print("Full SFT model loaded from:", args.model_name_or_path)

    # Use SFTTrainer for evaluation (same as training) - just don't call train()
    eval_args = SFTConfig(
        output_dir="./eval_output",
        per_device_eval_batch_size=8,
        bf16=True,
        report_to="none",  # Do not log to wandb
        max_length=MAX_SEQ_LENGTH,
    )
    
    # Reuse your existing compute_metrics functions
    eval_compute_metrics = partial(compute_metrics, tokenizer=tokenizer, verbose=True)
    
    # SFTTrainer needs a train_dataset for initialization, so we provide a small dummy one
    # We'll use a small subset of the validation data as dummy training data
    dummy_train_dataset = dataset_valid.select(range(min(10, len(dataset_valid))))
    
    # Use SFTTrainer (same as training) - this ensures identical data processing
    trainer = SFTTrainer(
        model=model,
        train_dataset=dummy_train_dataset,  # Small dummy dataset for initialization
        eval_dataset=dataset_valid,
        processing_class=tokenizer,  # Same as training - lets SFTTrainer handle tokenization
        compute_metrics=eval_compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, tokenizer),
        args=eval_args,
    )
    
    # Run the evaluation
    print("Starting evaluation...")
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()