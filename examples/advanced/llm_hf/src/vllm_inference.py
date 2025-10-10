#!/usr/bin/env python3
"""
vLLM Offline Batched Inference Script

This script demonstrates offline batched inference using vLLM for high-throughput text generation.
Supports both basic prompts and chat-based inference with proper chat templates.
"""

import argparse
import json
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def basic_inference(model_name: str, prompts: List[str], sampling_params: SamplingParams) -> None:
    """Perform basic offline inference with vLLM."""
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name)

    print(f"Generating text for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 80)
    print("BASIC INFERENCE RESULTS")
    print("=" * 80)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 40)


def chat_inference(
    model_name: str,
    messages_list: List[List[Dict[str, str]]],
    sampling_params: SamplingParams,
) -> None:
    """Perform chat-based inference with proper chat template application."""
    print(f"Loading model and tokenizer: {model_name}")
    llm = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply chat template
    texts = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"Generating chat responses for {len(messages_list)} conversations...")
    outputs = llm.generate(texts, sampling_params)

    print("\n" + "=" * 80)
    print("CHAT INFERENCE RESULTS")
    print("=" * 80)

    for idx, output in enumerate(outputs):
        user_message = messages_list[idx][0]["content"]
        generated_text = output.outputs[0].text
        print(f"User: {user_message}")
        print(f"Assistant: {generated_text}")
        print("-" * 40)


def chat_interface_inference(
    model_name: str,
    messages_list: List[List[Dict[str, str]]],
    sampling_params: SamplingParams,
) -> None:
    """Perform chat inference using vLLM's built-in chat interface."""
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name)

    print(f"Generating chat responses using chat interface for {len(messages_list)} conversations...")
    outputs = llm.chat(messages_list, sampling_params)

    print("\n" + "=" * 80)
    print("CHAT INTERFACE RESULTS")
    print("=" * 80)

    for idx, output in enumerate(outputs):
        user_message = messages_list[idx][0]["content"]
        generated_text = output.outputs[0].text
        print(f"User: {user_message}")
        print(f"Assistant: {generated_text}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="vLLM Offline Batched Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Model name or path (default: OLMo-2-1124-7B-Instruct)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "chat", "chat_interface"],
        default="basic",
        help="Inference mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument("--prompts_file", type=str, help="JSON file containing prompts or messages")

    args = parser.parse_args()

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    print(f"Sampling parameters: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    # Load prompts/messages from file or use defaults
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            data = json.load(f)

        if args.mode == "basic":
            prompts = data["prompts"] if "prompts" in data else data
        else:
            messages_list = data["messages"] if "messages" in data else data
    else:
        # Default prompts
        if args.mode == "basic":
            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
                "Explain quantum computing in simple terms:",
            ]
        else:
            messages_list = [
                [
                    {
                        "role": "user",
                        "content": "You are an assistant tasked with creating patient vignettes to help train\nmedical students...\n\nFirst, you must demonstrate a strong understanding of the underlying clinical presentation in the\nstructured data. To do this you will think carefully to identify the diagnosis in the multiple\nchoice question below that the patient most likely has.\n\nYou will be given five options, 'A', 'B', 'C', 'D', and 'E'. Provide your answer within tags, for\nexample <answer>one_letter_here</answer>. Ensure that your answer inside the tags is only a single\nletter.\n\nHere is the patient's structured data:\n<Patient Data>\n## Patient Summary\nThis report details the medical history of Arden553 McKenzie361, a 10-year-old female born on 2015-02-10.\n\n**Allergies:**\n- No known allergies.\n\n**Immunization History:**\n- Hep B  adolescent or pediatric (on 2015-02-10)\n- Hep B  adolescent or pediatric (on 2015-03-14)\n- rotavirus  monovalent (on 2015-04-12)\n- DTaP (on 2015-04-12)\n- Hib (PRP-OMP) (on 2015-04-12)\n- Pneumococcal conjugate PCV 13 (on 2015-04-12)\n- IPV (on 2015-04-12)\n- Hep B  adolescent or pediatric (on 2015-08-20)\n- rotavirus  monovalent (on 2015-08-20)\n- DTaP (on 2015-08-20)\n- Hib (PRP-OMP) (on 2015-08-20)\n- Pneumococcal conjugate PCV 13 (on 2015-08-20)\n- IPV (on 2015-08-20)\n- Influenza  seasonal  injectable  preservative free (on 2015-08-20)\n- DTaP (on 2015-11-24)\n- Pneumococcal conjugate PCV 13 (on 2015-11-24)\n- IPV (on 2015-11-24)\n\n**Ongoing Care Plans:**\n- No active care plans found.\n\n---\n\n## Chronological History\n### Encounter on 2015-02-10: Outpatient Encounter\n\n**Immunizations:**\n- Hep B  adolescent or pediatric\n### Encounter on 2015-03-14: Outpatient Encounter\n\n**Procedures:**\n- Documentation of current medications\n\n**Observations:**\n- Diastolic Blood Pressure: 70.0 mmHg\n- Body Weight: 3.11 kg\n- Body Height: 50.81 cm\n- Systolic Blood Pressure: 117.0 mmHg\n\n**Immunizations:**\n- Hep B  adolescent or pediatric\n### Encounter on 2015-04-12: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 78.0 mmHg\n- Body Height: 54.41 cm\n- Body Weight: 3.78 kg\n- Systolic Blood Pressure: 121.0 mmHg\n\n**Immunizations:**\n- IPV\n- DTaP\n- Pneumococcal conjugate PCV 13\n- rotavirus  monovalent\n- Hib (PRP-OMP)\n### Encounter on 2015-06-02: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 71.0 mmHg\n- Systolic Blood Pressure: 112.0 mmHg\n- Body Weight: 4.97 kg\n- Body Height: 59.56 cm\n### Encounter on 2015-08-20: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 70.0 mmHg\n- Systolic Blood Pressure: 118.0 mmHg\n- Body Height: 65.16 cm\n- Body Weight: 6.42 kg\n\n**Immunizations:**\n- Hep B  adolescent or pediatric\n- IPV\n- DTaP\n- Influenza  seasonal  injectable  preservative free\n- Pneumococcal conjugate PCV 13\n- rotavirus  monovalent\n- Hib (PRP-OMP)\n### Encounter on 2015-11-24: Outpatient Encounter\n\n**Procedures:**\n- Documentation of current medications\n\n**Observations:**\n- Diastolic Blood Pressure: 74.0 mmHg\n- Systolic Blood Pressure: 117.0 mmHg\n- Body Weight: 7.56 kg\n- Body Height: 69.57 cm\n\n**Immunizations:**\n- IPV\n- Pneumococcal conjugate PCV 13\n- DTaP\n\n---\n</Patient Data>\n\nHere are the possible diagnoses that could be associated with the patient's history:\nA. Non-small cell lung cancer\nB. Neuropathy due to type 2 diabetes mellitus\nC. Sprain of ankle\nD. Otitis media\nE. First degree burn\n",
                    }
                ],
            ]

    # Run inference based on mode
    try:
        if args.mode == "basic":
            basic_inference(args.model, prompts, sampling_params)
        elif args.mode == "chat":
            chat_inference(args.model, messages_list, sampling_params)
        elif args.mode == "chat_interface":
            chat_interface_inference(args.model, messages_list, sampling_params)
    except Exception as e:
        print(f"Error during inference: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure vLLM is properly installed: uv pip install vllm --torch-backend=auto")
        print("2. Check if the model is supported: https://docs.vllm.ai/en/latest/models/supported_models.html")
        print("3. For GPU issues, verify CUDA installation and GPU memory availability")
        print("4. For chat models, ensure the model supports chat templates")


if __name__ == "__main__":
    main()
