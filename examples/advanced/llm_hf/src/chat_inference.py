#!/usr/bin/env python3

import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_answer_token_ids(tokenizer, choices=("A", "B", "C", "D", "E")) -> List[int]:
    """Get token IDs for multiple choice answer letters A, B, C, D, E"""
    ids = set()
    for letter in choices:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        ids.add(enc[-1])
    return sorted(ids)


def get_restricted_token_probabilities(logits, tokenizer, choices=("A", "B", "C", "D", "E")):
    """
    Get probabilities for only the allowed choice tokens (A, B, C, D, E)
    Returns: dict with choice -> probability mapping and the most likely choice
    """
    print("incoming logits: ", logits)
    print("shape: ", logits.shape)

    answer_token_ids = get_answer_token_ids(tokenizer, choices)

    # Get the last token's logits (the prediction for the next token)
    last_logits = logits[0, -1, :]  # [vocab_size]
    print("last logits: ", last_logits)
    print("shape: ", last_logits.shape)

    # Create a mask that only allows answer token IDs
    vocab_size = last_logits.size(0)
    mask = torch.full((vocab_size,), float("-inf"), device=last_logits.device, dtype=last_logits.dtype)
    mask[answer_token_ids] = 0  # Set allowed tokens to 0 (no penalty)

    # Apply mask and get probabilities
    masked_logits = last_logits + mask
    probabilities = F.softmax(masked_logits, dim=-1)

    # Extract probabilities for each choice and find the most likely
    choice_probs = {}
    max_prob = 0
    best_choice = choices[0]

    for choice in choices:
        choice_token_ids = tokenizer.encode(choice, add_special_tokens=False)
        token_id = choice_token_ids[-1]
        prob = probabilities[token_id].item()
        choice_probs[choice] = prob
        if prob > max_prob:
            max_prob = prob
            best_choice = choice

    return {
        "probabilities": choice_probs,
        "best_choice": best_choice,
        "best_probability": max_prob,
    }


class ChatInference:
    """
    Chat-based inference class with configurable system prompt for Llama models.
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the chat inference model.

        Args:
            model_name_or_path: Path or name of the model to load
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights
            system_prompt: Custom system prompt (if None, uses default)
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        # Default system prompt
        self.default_system_prompt = (
            # "You are a medical AI expert. You must answer with only one letter, one of A, B, C, D, or E."
            # "You are a medical AI expert. You will be asked to answer a medical multiple choice question, and you must answer with 1 letter A, B, C, D, or E."
        )

        self.system_prompt = system_prompt or self.default_system_prompt

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        device_map = "auto" if self.device == "auto" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=True,
        )

        if device_map is None and self.device != "auto":
            self.model = self.model.to(self.device)

        print(f"Model loaded on device: {self.model.device}")

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = system_prompt
        print(f"System prompt updated: {system_prompt[:100]}...")

    def format_chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Format a chat conversation using the model's chat template.

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages (optional)

        Returns:
            Formatted chat string ready for tokenization
        """
        messages = []

        # Add system message
        messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Use the tokenizer's chat template
        formatted_chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return formatted_chat

    def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float = 0.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response to a user message.

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing the response and metadata
        """
        # Format the chat
        formatted_input = self.format_chat(user_message, conversation_history)

        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096,  # Adjust based on model's context length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the generated part (exclude input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Prepare response
        result = {
            "response": response_text.strip(),
            "formatted_input": formatted_input,
            "input_tokens": input_length,
            "generated_tokens": len(generated_tokens),
            "total_tokens": len(outputs[0]),
        }

        return result

    def generate_multiple_choice_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float = 0.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        choices: tuple = ("A", "B", "C", "D", "E"),
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response restricted to multiple choice answers (A, B, C, D, E).
        Returns the most likely choice along with probabilities for all choices.

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
            choices: Tuple of valid choices (default: A, B, C, D, E)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing the best choice, probabilities, and metadata
        """
        # Format the chat
        formatted_input = self.format_chat(user_message, conversation_history)

        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096,  # Adjust based on model's context length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate logits (we only need one step to get the next token probabilities)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get restricted probabilities for multiple choice answers
        mc_result = get_restricted_token_probabilities(logits, self.tokenizer, choices)

        # Also generate a regular response for comparison
        with torch.no_grad():
            generation_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the generated part (exclude input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generation_outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Prepare comprehensive response
        result = {
            "best_choice": mc_result["best_choice"],
            "choice_probabilities": mc_result["probabilities"],
            "best_probability": mc_result["best_probability"],
            "raw_response": response_text.strip(),
            "formatted_input": formatted_input,
            "input_tokens": input_length,
            "generated_tokens": len(generated_tokens),
            "total_tokens": len(generation_outputs[0]),
        }

        return result

    def chat_interactive(self):
        """Start an interactive chat session."""
        print("=" * 50)
        print("Interactive Chat Mode")
        print("Type 'quit' to exit, 'reset' to clear history, 'system <prompt>' to change system prompt")
        print("=" * 50)
        print(f"Current system prompt: {self.system_prompt}")
        print("=" * 50)

        conversation_history = []

        while True:
            try:
                user_input = input("\nUser: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    break
                elif user_input.lower() == "reset":
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                elif user_input.lower().startswith("system "):
                    new_system_prompt = user_input[7:].strip()
                    self.set_system_prompt(new_system_prompt)
                    continue
                elif not user_input:
                    continue

                # Generate multiple choice response
                result = self.generate_multiple_choice_response(user_input, conversation_history)

                print(f"\nMost Likely Answer: {result['best_choice']} (probability: {result['best_probability']:.4f})")
                print(f"\nAll Choice Probabilities:")
                for choice, prob in result["choice_probabilities"].items():
                    print(f"  {choice}: {prob:.4f}")
                print(f"\nRaw Response: {result['raw_response']}")
                print(f"[Tokens - Input: {result['input_tokens']}, Generated: {result['generated_tokens']}]")

                response = result["best_choice"]  # Use best choice for conversation history

                # Update conversation history
                conversation_history.extend(
                    [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response},
                    ]
                )

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Chat session ended.")


def main():
    parser = argparse.ArgumentParser(description="Chat inference with configurable system prompt")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="PyTorch data type (bfloat16, float16, float32)",
    )
    parser.add_argument("--system_prompt", type=str, help="Custom system prompt")
    parser.add_argument("--message", type=str, help="Single message to process (non-interactive mode)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")

    args = parser.parse_args()

    # Initialize chat inference
    chat = ChatInference(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        system_prompt=args.system_prompt,
    )

    if args.message:
        # Single message mode
        result = chat.generate_multiple_choice_response(
            args.message,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"User: {args.message}")
        print(f"Most Likely Answer: {result['best_choice']} (probability: {result['best_probability']:.4f})")
        print(f"\nAll Choice Probabilities:")
        for choice, prob in result["choice_probabilities"].items():
            print(f"  {choice}: {prob:.4f}")
        print(f"Raw Response: {result['raw_response']}")
        print(f"Tokens used: {result['total_tokens']}")
    else:
        # Interactive mode
        chat.chat_interactive()


def test_training_utils_functions():
    """
    Test the functions from training_utils.py in the inference context
    to see if they work properly with normal model predictions.
    """
    import numpy as np
    from training_utils import get_answer_token_ids, preprocess_logits_for_metrics

    print("=== Testing training_utils functions in inference context ===")

    # Initialize chat inference
    chat = ChatInference(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        device="auto",
        torch_dtype="bfloat16",
    )

    # Test with a few different medical questions
    test_questions = [
        "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST-elevation. What is the most likely diagnosis? A) Pneumonia B) Myocardial infarction C) Anxiety D) GERD E) Asthma",
        "A 30-year-old woman has a rash and joint pain. Lab shows positive ANA. What is the most likely diagnosis? A) Lupus B) Diabetes C) Hypertension D) Depression E) Pneumonia",
        "A child has fever and a barking cough. What is the most likely diagnosis? A) Pneumonia B) Asthma C) Croup D) Bronchitis E) COPD",
    ]

    for i, question in enumerate(test_questions):
        print(f"\n--- Test Question {i + 1} ---")
        print(f"Question: {question[:100]}...")

        # Format the input
        formatted_input = chat.format_chat(question)
        inputs = chat.tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(chat.model.device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = chat.model(**inputs)
            logits = outputs.logits  # Shape: [1, seq_len, vocab_size]

        # Create fake labels for testing (simulate the training scenario)
        seq_len = logits.size(1)
        labels = torch.full((1, seq_len), -100, dtype=torch.long, device=logits.device)

        # Simulate having an answer at the end (let's say position seq_len-2)
        # We'll put different answer tokens for different questions
        answer_tokens = [33, 32, 34]  # B, A, C respectively
        labels[0, seq_len - 2] = answer_tokens[i % 3]

        print(f"Simulated true answer token: {answer_tokens[i % 3]} ({chat.tokenizer.decode([answer_tokens[i % 3]])})")

        # Test our training_utils functions
        print("\n--- Testing training_utils functions ---")

        # Test get_answer_token_ids
        answer_ids = get_answer_token_ids(chat.tokenizer)
        print(f"Answer token IDs (A,B,C,D,E): {answer_ids}")

        # Test preprocess_logits_for_metrics
        predictions = preprocess_logits_for_metrics(logits, labels, chat.tokenizer)
        print(f"Predictions from preprocess_logits_for_metrics: {predictions}")
        print(f"Predicted token decoded: {chat.tokenizer.decode(predictions)}")

        # Compare with regular inference (what the model would normally predict)
        last_logits = logits[0, -1, :]  # Last position logits
        unrestricted_pred = last_logits.argmax().item()
        print(f"Unrestricted prediction: {unrestricted_pred} ({chat.tokenizer.decode([unrestricted_pred])})")

        # Test the existing chat_inference function for comparison
        mc_result = get_restricted_token_probabilities(logits, chat.tokenizer)
        print(
            f"get_restricted_token_probabilities result: {mc_result['best_choice']} (prob: {mc_result['best_probability']:.4f})"
        )
        print(f"All probabilities: {mc_result['probabilities']}")

        print("-" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-training-utils":
        test_training_utils_functions()
    else:
        main()
