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
