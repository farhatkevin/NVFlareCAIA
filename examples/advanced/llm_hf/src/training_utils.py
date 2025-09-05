import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Callable, List
from transformers import EvalPrediction

def get_answer_token_ids(tokenizer, choices=("A","B","C","D","E")) -> List[int]:
    """Get token IDs for multiple choice answer letters A, B, C, D, E"""
    ids = set()
    for letter in choices:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        ids.add(enc[-1])
    return sorted(ids)

def preprocess_logits_for_metrics(logits, labels, tokenizer=None):
    """
    For each sample, select the logits at the FIRST non-masked target position
    (i.e., first index where labels != -100), then restrict predictions to A..E.

    - Aligns eval prediction position with compute_metrics' use of the first
      non-ignored label.
    - If a row has no valid label positions, we fall back to the last timestep.
    - Uses get_answer_token_ids(tokenizer) as-is (no space/newline variants).
    """
    if isinstance(logits, tuple):
        logits = logits[0]  # [B, S, V]

    with torch.no_grad():
        B, S, V = logits.shape

        if not torch.is_tensor(labels):
            labels_t = torch.tensor(labels, device=logits.device)
        else:
            labels_t = labels.to(logits.device)

        # First non-ignored position per row (where answer token is)
        mask = (labels_t != -100)                      # [B, S] bool
        has_any = mask.any(dim=1)                      # [B]
        first_pos = mask.float().argmax(dim=1)         # [B] position of answer token
        
        # logits[t-1] predicts labels[t], so use first_pos-1
        pred_pos = torch.clamp(first_pos - 1 , min=0)   # [B] position that predicts the answer
        
        # Fallback to last position for rows with no valid labels
        pos = torch.where(has_any, pred_pos, torch.full_like(pred_pos, S - 1))

        row = torch.arange(B, device=logits.device)
        selected_logits = logits[row, pos, :]          # [B, V] - logits that predict answer

        debug_count = getattr(preprocess_logits_for_metrics, 'debug_count', 0) + 1
        preprocess_logits_for_metrics.debug_count = debug_count
        
        # Always print for the first eval batch (we reset debug_count before each eval)
        if debug_count == 1 and tokenizer is not None:
            print(f"=== POSITION DEBUG (call #{debug_count}) ===")
            sample_0_labels = labels_t[0].cpu().numpy()
            print(f"Sample 0 label structure around answer:")
            answer_pos = first_pos[0].item()
            pred_position = pos[0].item()
            
            # Show 5 positions around the answer
            start_pos = max(0, answer_pos - 3)
            end_pos = min(len(sample_0_labels), answer_pos + 3)
            # start_pos = 0
            # end_pos = len(sample_0_labels)
            print("end_pos:", end_pos)
            for i in range(start_pos, end_pos):
                label_val = sample_0_labels[i]
                token_str = tokenizer.decode([label_val]) if label_val != -100 else "IGNORED"
                marker = " <-- ANSWER" if i == answer_pos else " <-- PRED" if i == pred_position else ""
                print(f"  pos {i}: label={label_val} ({token_str}){marker}")
            
            print(f"Using prediction position: {pred_position} (answer_pos: {answer_pos})")
            print("=" * 30)

        if tokenizer is not None:
            allowed_ids = get_answer_token_ids(tokenizer)  # token ids for A..E
            mask_vec = torch.full((V,), float("-inf"),
                                  device=selected_logits.device,
                                  dtype=selected_logits.dtype)
            mask_vec[allowed_ids] = 0
            selected_logits = selected_logits + mask_vec

            if debug_count == 1:
                try:
                    letters = [tokenizer.decode([tid]) for tid in allowed_ids]
                except Exception:
                    letters = [str(tid) for tid in allowed_ids]
                ae_scores = selected_logits[:, allowed_ids].detach().float().cpu()
                print("Choice logits at prediction position (A–E):")
                for i in range(B):
                    pos_i = int(pos[i].item())
                    # True answer if available
                    true_ans = None
                    if bool(has_any[i].item()):
                        true_token = int(labels_t[i, first_pos[i]].item())
                        if true_token in allowed_ids:
                            true_ans = tokenizer.decode([true_token])
                    row_scores = ae_scores[i].tolist()
                    per_choice = ", ".join(f"{l}:{s:.3f}" for l, s in zip(letters, row_scores))
                    print(f"  sample {i} (pred_pos={pos_i}): {per_choice}"
                          + (f" | true={true_ans}" if true_ans is not None else ""))

        preds = selected_logits.argmax(dim=-1)         # [B] token IDs
        return preds


def compute_metrics(eval_preds: EvalPrediction, tokenizer, verbose=True) -> Dict[str, float]:
    """Compare argmax predictions to true labels for accuracy"""
    pred_ids, labels = eval_preds

    if verbose:
        # Debug: Show label structure for MULTIPLE samples to check variety
        print(f"=== LABEL DEBUG - BATCH ANALYSIS ===")
        print(f"Batch size: {labels.shape[0]}")
        
        # Check first 3 samples
        for i in range(min(3, labels.shape[0])):
            sample_labels = labels[i]
            non_ignored = (sample_labels != -100)
            answer_tokens = sample_labels[non_ignored]
            answer_letters = [tokenizer.decode([t]) for t in answer_tokens if t not in [128001]]  # Exclude end_of_text
            
            print(f"Sample {i}: {answer_letters} (tokens: {answer_tokens.tolist()})")
        
        # Check ALL samples for answer distribution
        all_answers = []
        answer_ids = get_answer_token_ids(tokenizer)
        
        for i in range(labels.shape[0]):
            sample_labels = labels[i]
            non_ignored = (sample_labels != -100)
            answer_tokens = sample_labels[non_ignored]
            # Find A,B,C,D,E tokens (exclude end_of_text)
            valid_answers = [t for t in answer_tokens if t in answer_ids]
            if valid_answers:
                all_answers.extend(valid_answers)
        
        # Show distribution
        from collections import Counter
        answer_counts = Counter(all_answers)
        print(f"Answer distribution in batch:")
        for token_id in answer_ids:
            letter = tokenizer.decode([token_id])
            count = answer_counts.get(token_id, 0)
            print(f"  {letter}: {count} samples")
        
        # Also show what the model predicted vs true answers
        print(f"Model predictions vs true answers (first 5 samples):")
        pred_ids_debug = np.asarray(pred_ids).reshape(-1)
        for i in range(min(5, len(pred_ids_debug), labels.shape[0])):
            # Get prediction
            pred_token = pred_ids_debug[i]
            pred_letter = tokenizer.decode([pred_token]) if pred_token in answer_ids else f"UNKNOWN({pred_token})"
            
            # Get true answer
            sample_labels = labels[i]
            non_ignored = (sample_labels != -100)
            true_tokens = sample_labels[non_ignored]
            true_answers = [t for t in true_tokens if t in answer_ids]
            true_letter = tokenizer.decode([true_answers[0]]) if true_answers else "UNKNOWN"
            
            match = "✓" if pred_token in true_answers else "✗"
            print(f"  Sample {i}: predicted {pred_letter} vs true {true_letter} {match}")
        
        # Show distribution of all predictions 
        pred_counts = Counter(pred_ids_debug[:labels.shape[0]])  # Only count valid predictions
        print(f"Prediction distribution: A:{pred_counts.get(32,0)} B:{pred_counts.get(33,0)} C:{pred_counts.get(34,0)} D:{pred_counts.get(35,0)} E:{pred_counts.get(36,0)}")
        print(f"True answer distribution:   A:{answer_counts.get(32,0)} B:{answer_counts.get(33,0)} C:{answer_counts.get(34,0)} D:{answer_counts.get(35,0)} E:{answer_counts.get(36,0)}")
        
        print("=" * 40)


    pred_ids = np.asarray(pred_ids).reshape(-1)
    labels = np.asarray(labels)
    
    if labels.ndim != 2 or labels.shape[0] == 0:
        return {"accuracy": 0.0}
    
    # Find valid samples (have context before answer)
    mask = labels != -100
    has_any = mask.any(axis=1)
    first_idx = np.where(has_any, mask.argmax(axis=1), 0)
    valid = has_any & (first_idx > 0)
    
    if not valid.any():
        return {"accuracy": 0.0}
    
    # Compare predictions to true labels
    y_true = labels[valid, first_idx[valid]]
    y_pred = pred_ids[valid]
    accuracy = float((y_pred == y_true).mean())
    
    if verbose:
        print(f"Evaluated {valid.sum()} samples, accuracy: {accuracy:.3f}")
    
    return {"accuracy": accuracy}


def filter_by_length(example, tokenizer, format_instruction: Callable, max_seq_length: int):
    """
    Filter dataset examples that exceed the maximum sequence length.
    
    Args:
        example: Dataset example with 'prompt' and 'completion' fields
        tokenizer: The tokenizer used for the model
        format_instruction: Function to format examples into instruction format
        max_seq_length: Maximum sequence length in tokens
        
    Returns:
        Boolean indicating whether to keep the example (True if under limit)
    """
    # Since format_instruction now returns a dict, manually concatenate for length check
    full_text = example["prompt"] + "\n\n" + example["completion"]
    tokens = tokenizer(full_text, truncation=False, return_tensors=None)
    return len(tokens["input_ids"]) < max_seq_length - 1

def test_generation_with_pipeline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    print_result: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate continuations using Hugging Face pipeline and return only the generated part
    (excluding the prompt) as text and token IDs.

    Args:
        model: HF model instance already loaded (will be used by pipeline)
        tokenizer: HF tokenizer instance
        prompt: input prompt string
        max_new_tokens: number of new tokens to generate (continuation length)
        do_sample: whether to sample
        temperature: sampling temperature
        top_p: nucleus sampling probability
        num_return_sequences: number of generations to return
        print_result: whether to print the generation(s)

    Returns:
        List of dicts per sequence: { 'generated_text': str, 'generated_ids': List[int] }
    """
    from transformers import pipeline

    # Create a generation pipeline with the given model/tokenizer.
    # return_full_text=False ensures we only get the continuation text, not the prompt.
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        # device is inferred from model; avoid forcing device index to prevent re-loading
    )

    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Normalize to list
    if isinstance(outputs, dict):
        outputs = [outputs]

    results: List[Dict[str, Any]] = []
    for i, out in enumerate(outputs):
        gen_text = out.get("generated_text", "")
        # Tokenize only the generated continuation
        gen_ids = tokenizer(gen_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        item = {
            "generated_text": gen_text,
            "generated_ids": gen_ids.tolist(),
        }
        results.append(item)

        if print_result:
            print(f"--- Generation {i+1} with pipeline---")
            print("Generated text:")
            print(gen_text)
            print(f"Generated token IDs: {item['generated_ids']}")

    return results


def test_generation_with_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    print_result: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate continuations using model.generate and return both the full text and
    the continuation-only text and token IDs (excluding the prompt part).

    Args are similar to test_generation_with_pipeline.
    """
    model.eval()

    # Determine special token ids
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    # Normalize to list of sequences
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)

    results: List[Dict[str, Any]] = []
    for i in range(outputs.shape[0]):
        out_ids = outputs[i].to("cpu")
        gen_ids = out_ids[prompt_len:]

        full_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        item = {
            "full_text": full_text,
            "generated_text": generated_text,
            "generated_ids": gen_ids.tolist(),
        }
        results.append(item)

        if print_result:
            print(f"--- Generation {i+1} with generate ---")
            print("Generated text:")
            print(generated_text)
            print(f"Generated token IDs: {item['generated_ids']}")

    return results

from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "allenai/OLMo-2-1124-7B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# prompt = """You are an assistant tasked with creating patient vignettes to help train\nmedical students...\n\nFirst, you must demonstrate a strong understanding of the underlying clinical presentation in the\nstructured data. To do this you will think carefully to identify the diagnosis in the multiple\nchoice question below that the patient most likely has.\n\nYou will be given five options, 'A', 'B', 'C', 'D', and 'E'. Ensure that your answer is only a single letter.\n\nHere is the patient's structured data:\n<Patient Data>\n## Patient Summary\nThis report details the medical history of Columbus276 Murray55, a 15-year-old female born on 2009-09-06.\n\n**Allergies:**\n- No known allergies.\n\n**Immunization History:**\n- Pneumococcal conjugate PCV 13 (on 2010-07-10)\n- Hib (PRP-OMP) (on 2010-10-15)\n- Pneumococcal conjugate PCV 13 (on 2010-10-15)\n- MMR (on 2010-10-15)\n- varicella (on 2010-10-15)\n- Hep A  ped/adol  2 dose (on 2010-10-15)\n- DTaP (on 2011-01-16)\n- Influenza  seasonal  injectable  preservative free (on 2011-04-27)\n- Hep A  ped/adol  2 dose (on 2011-11-09)\n- Influenza  seasonal  injectable  preservative free (on 2012-05-13)\n\n**Ongoing Care Plans:**\n- No active care plans found.\n\n---\n\n## Chronological History\n### Encounter on 2010-07-10: Outpatient Encounter\n\n**Observations:**\n- Systolic Blood Pressure: 107.0 mmHg\n- Body Weight: 7.89 kg\n- Body Height: 65.28 cm\n- Diastolic Blood Pressure: 75.0 mmHg\n\n**Immunizations:**\n- Pneumococcal conjugate PCV 13\n### Encounter on 2010-10-15: Outpatient Encounter\n\n**Observations:**\n- Body Weight: 9.06 kg\n- Systolic Blood Pressure: 114.0 mmHg\n- Diastolic Blood Pressure: 89.0 mmHg\n- Body Height: 69.84 cm\n\n**Immunizations:**\n- Hep A  ped/adol  2 dose\n- varicella\n- Pneumococcal conjugate PCV 13\n- Hib (PRP-OMP)\n- MMR\n### Encounter on 2011-01-16: Outpatient Encounter\n\n**Procedures:**\n- Documentation of current medications\n\n**Observations:**\n- Body Height: 72.81 cm\n- Systolic Blood Pressure: 110.0 mmHg\n- Diastolic Blood Pressure: 88.0 mmHg\n- Body Weight: 9.74 kg\n\n**Immunizations:**\n- DTaP\n### Encounter on 2011-04-27: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 78.0 mmHg\n- Systolic Blood Pressure: 133.0 mmHg\n- Body Height: 75.51 cm\n- Body Weight: 10.3 kg\n\n**Immunizations:**\n- Influenza  seasonal  injectable  preservative free\n### Encounter on 2011-11-09: Outpatient Encounter\n\n**Observations:**\n- Systolic Blood Pressure: 138.0 mmHg\n- Body Mass Index: 17.68 kg/m2\n- Body Height: 79.6 cm\n- Diastolic Blood Pressure: 70.0 mmHg\n- Body Weight: 11.21 kg\n\n**Immunizations:**\n- Hep A  ped/adol  2 dose\n### Encounter on 2012-05-13: Outpatient Encounter\n\n**Procedures:**\n- Documentation of current medications\n\n**Observations:**\n- Body Height: 84.6 cm\n- Diastolic Blood Pressure: 78.0 mmHg\n- Body Mass Index: 16.91 kg/m2\n- Body Weight: 12.11 kg\n- Systolic Blood Pressure: 112.0 mmHg\n\n**Immunizations:**\n- Influenza  seasonal  injectable  preservative free\n\n---\n</Patient Data>\n\nHere are the possible diagnoses that could be associated with the patient's history:\nA. Viral sinusitis\nB. Acute bronchitis\nC. Laceration of foot\nD. Acute bacterial sinusitis\nE. Sprain of wrist\n. nWhat is the diagnosis that the patient most likely has? """

prompt = """You are an assistant tasked with creating patient vignettes to help train\nmedical students...\n\nFirst, you must demonstrate a strong understanding of the underlying clinical presentation in the\nstructured data. To do this you will think carefully to identify the diagnosis in the multiple\nchoice question below that the patient most likely has.\n\nYou will be given five options, 'A', 'B', 'C', 'D', and 'E'. Provide your answer as only a single uppercase letter.\n\nHere is the patient's structured data:\n<Patient Data>\n## Patient Summary\nThis report details the medical history of Arden553 McKenzie361, a 10-year-old female born on 2015-02-10.\n\n**Allergies:**\n- No known allergies.\n\n**Immunization History:**\n- Hep B  adolescent or pediatric (on 2015-02-10)\n- Hep B  adolescent or pediatric (on 2015-03-14)\n- rotavirus  monovalent (on 2015-04-12)\n- DTaP (on 2015-04-12)\n- Hib (PRP-OMP) (on 2015-04-12)\n- Pneumococcal conjugate PCV 13 (on 2015-04-12)\n- IPV (on 2015-04-12)\n- Hep B  adolescent or pediatric (on 2015-08-20)\n- rotavirus  monovalent (on 2015-08-20)\n- DTaP (on 2015-08-20)\n- Hib (PRP-OMP) (on 2015-08-20)\n- Pneumococcal conjugate PCV 13 (on 2015-08-20)\n- IPV (on 2015-08-20)\n- Influenza  seasonal  injectable  preservative free (on 2015-08-20)\n- DTaP (on 2015-11-24)\n- Pneumococcal conjugate PCV 13 (on 2015-11-24)\n- IPV (on 2015-11-24)\n\n**Ongoing Care Plans:**\n- No active care plans found.\n\n---\n\n## Chronological History\n### Encounter on 2015-02-10: Outpatient Encounter\n\n**Immunizations:**\n- Hep B  adolescent or pediatric\n### Encounter on 2015-03-14: Outpatient Encounter\n\n**Procedures:**\n- Documentation of current medications\n\n**Observations:**\n- Diastolic Blood Pressure: 70.0 mmHg\n- Body Weight: 3.11 kg\n- Body Height: 50.81 cm\n- Systolic Blood Pressure: 117.0 mmHg\n\n**Immunizations:**\n- Hep B  adolescent or pediatric\n### Encounter on 2015-04-12: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 78.0 mmHg\n- Body Height: 54.41 cm\n- Body Weight: 3.78 kg\n- Systolic Blood Pressure: 121.0 mmHg\n\n**Immunizations:**\n- IPV\n- DTaP\n- Pneumococcal conjugate PCV 13\n- rotavirus  monovalent\n- Hib (PRP-OMP)\n### Encounter on 2015-06-02: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 71.0 mmHg\n- Systolic Blood Pressure: 112.0 mmHg\n- Body Weight: 4.97 kg\n- Body Height: 59.56 cm\n### Encounter on 2015-08-20: Outpatient Encounter\n\n**Observations:**\n- Diastolic Blood Pressure: 70.0 mmHg\n- Systolic Blood Pressure: 118.0 mmHg\n- Body Height: 65.16 cm\n- Body Weight: 6.42 kg\n\n**Immunizations:**\n- Hep B  adolescent or pediatric\n- IPV\n- DTaP\n- Influenza  seasonal  injectable  preservative free\n- Pneumococcal conjugate PCV 13\n- rotavirus  monovalent\n- Hib (PRP-OMP)\n### Encounter on 2015-11-24: Outpatient Encounter\n\n**Procedures:**\n- Documentation of current medications\n\n**Observations:**\n- Diastolic Blood Pressure: 74.0 mmHg\n- Systolic Blood Pressure: 117.0 mmHg\n- Body Weight: 7.56 kg\n- Body Height: 69.57 cm\n\n**Immunizations:**\n- IPV\n- Pneumococcal conjugate PCV 13\n- DTaP\n\n---\n</Patient Data>\n\nHere are the possible diagnoses that could be associated with the patient's history:\nA. Non-small cell lung cancer\nB. Neuropathy due to type 2 diabetes mellitus\nC. Sprain of ankle\nD. Otitis media\nE. First degree burn\n. Answer only with the single letter corresponding to the diagnosis that the patient most likely has. """

# print(test_generation_with_pipeline(model, tokenizer, prompt))  
# print(test_generation_with_generate(model, tokenizer, prompt))

from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
message = [prompt]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
inputs = {k: v.to('cuda') for k,v in inputs.items()}
olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
