import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_input_text(tokenizer, prompt: str, use_chat_template: bool):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Quick test inference for a local HF model folder")
    parser.add_argument("model_dir", type=str, help="Path to local HF model folder (e.g., .../round_1_model_hf)")
    parser.add_argument("--prompt", type=str, default="Hello!", help="Prompt text to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (ignored if do_sample=False)"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (default off -> greedy)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code when loading model/tokenizer"
    )
    parser.add_argument(
        "--use_chat_template", action="store_true", help="Use tokenizer.apply_chat_template if available"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Torch dtype for model"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Model dir not found: {args.model_dir}")
        sys.exit(1)

    # Select dtype
    if args.dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"Loading tokenizer from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=args.trust_remote_code)

    print(f"Loading model from: {args.model_dir}")
    model = None
    # Prefer device_map='auto' if available (accelerate installed), else fallback to .to(device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        device = model.device
    except Exception as e:
        print(f"  ⚠️ device_map='auto' failed ({e}); falling back to single device load")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)

    # Prepare input
    input_text = build_input_text(tokenizer, args.prompt, args.use_chat_template)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Ensure pad_token_id exists for generation on some models
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("Generating...\n")
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("===== PROMPT =====")
    print(input_text)
    print("===== OUTPUT =====")
    print(text)


if __name__ == "__main__":
    main()
