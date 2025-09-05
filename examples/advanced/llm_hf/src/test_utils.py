from training_utils import (
        get_answer_token_ids,
        preprocess_logits_for_metrics,
        compute_metrics,
    )
    
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys
from typing import List, Dict, Any

def setup(model_name: str):
    """Load the model and tokenizer."""
    print(f"Loading model and tokenizer: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def tokenize_input(tokenizer, text: str):
    """Tokenize the input text."""
    tok = tokenizer(text, return_tensors="pt")
    return tok.input_ids, tok.attention_mask

def build_example(tokenizer): 
    """Build a simple example for testing."""
    example = {
        "prompt": "What is the best letter out of these 5? 1) A 2) B 3) C 4) D 5) E\n",
        "completion": "C",
    }
    # mask = 
    tokenize_input = tokenize_input(tokenizer, example["prompt"] + example["completion"])
    print(example)
    return example

def main():
    model, tokenizer = setup("meta-llama/llama-3.1-8B-Instruct")
    print("Model and tokenizer loaded successfully.")
    build_example(tokenizer)


if __name__ == "__main__":
    main()