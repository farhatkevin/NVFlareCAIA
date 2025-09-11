#!/usr/bin/env python3

import random

import numpy as np
import torch
from training_utils import get_answer_token_ids, preprocess_logits_for_metrics
from transformers import AutoTokenizer


def simple_toy_example():
    """
    Simplest possible test: manually create tokens and labels.
    """
    print("=== SIMPLE TOY EXAMPLE ===")

    # Load tokenizer just to get A,B,C,D,E token IDs
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    answer_ids = get_answer_token_ids(tokenizer)
    print(f"A,B,C,D,E token IDs: {answer_ids}")

    # Create simple sequence: 5 tokens total
    # [prompt_token1, prompt_token2, answer_token, eos_token, pad_token]
    sequence = [100, 200, 33, 128001, 0]  # 33 = "B" token
    labels = [-100, -100, 33, 128001, -100]  # Only train on "B" and EOS

    print(f"Sequence: {sequence}")
    print(f"Labels:   {labels}")
    print("Position alignment:")
    for i, (seq_token, label_token) in enumerate(zip(sequence, labels)):
        status = "COMPLETION" if label_token != -100 else "IGNORED"
        print(f"  pos {i}: seq={seq_token} | label={label_token} ({status})")

    # Create fake logits: [1, 5, vocab_size]
    vocab_size = tokenizer.vocab_size
    batch_size, seq_len = 1, 5
    logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1

    # Make position 1 predict "B" (token 33) at position 2
    # Causal LM: logits[1] should predict labels[2]
    logits[0, 1, 33] = 10.0  # High logit for "B"
    for aid in answer_ids:
        if aid != 33:
            logits[0, 1, aid] = -5.0  # Low logit for A,C,D,E

    print(f"\nBoosted logits[0, 1, 33] (position 1 predicting 'B' at position 2)")
    print("logits: ", logits)
    # Test the function
    labels_tensor = torch.tensor([labels], dtype=torch.long)
    predictions = preprocess_logits_for_metrics(logits, labels_tensor, tokenizer)

    print(f"\nFunction output:")
    print(f"Predictions: {predictions}")
    print(f"Expected: [33] (token for 'B')")
    print(f"Correct: {predictions[0].item() == 33}")

    return predictions[0].item() == 33


def test_batch():
    """
    Test with 3 samples: A, C, E answers
    """
    print("\n=== BATCH TEST ===")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    answer_ids = get_answer_token_ids(tokenizer)
    A_token, B_token, C_token, D_token, E_token = answer_ids

    # 3 samples, each 4 tokens: [prompt, prompt, answer, eos]
    sequences = [
        [100, 200, A_token, 128001],  # Answer A
        [300, 400, C_token, 128001],  # Answer C
        [500, 600, E_token, 128001],  # Answer E
    ]

    labels = [
        [-100, -100, A_token, 128001],
        [-100, -100, C_token, 128001],
        [-100, -100, E_token, 128001],
    ]

    print("Batch structure:")
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        answer_token = [t for t in lab if t != -100 and t != 128001][0]
        answer_letter = tokenizer.decode([answer_token])
        print(f"  Sample {i}: answer = {answer_letter} (token {answer_token})")

    # Create batch logits
    vocab_size = tokenizer.vocab_size
    batch_size, seq_len = 3, 4
    logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1

    # Set up causal predictions: logits[i, 1] predicts labels[i, 2]
    expected_answers = [A_token, C_token, E_token]
    for i, answer_token in enumerate(expected_answers):
        logits[i, 1, answer_token] = 10.0  # High for correct
        for aid in answer_ids:
            if aid != answer_token:
                logits[i, 1, aid] = -5.0  # Low for wrong

    # Test function
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    predictions = preprocess_logits_for_metrics(logits, labels_tensor, tokenizer)

    print(f"\nPredictions: {predictions.tolist()}")
    print(f"Expected:    {expected_answers}")

    predicted_letters = [tokenizer.decode([p]) for p in predictions]
    expected_letters = [tokenizer.decode([e]) for e in expected_answers]

    print(f"Predicted: {predicted_letters}")
    print(f"Expected:  {expected_letters}")

    correct = sum(p == e for p, e in zip(predictions.tolist(), expected_answers))
    print(f"Correct: {correct}/3")

    return correct == 3


def main():
    print("SIMPLE CAUSAL EVALUATION TEST")
    print("=" * 40)

    # Test single sample
    single_ok = simple_toy_example()

    # Test batch
    batch_ok = test_batch()

    # Larger randomized batch test
    def test_large_batch(num_samples: int = 100, max_prompt_len: int = 8):
        print("\n=== LARGE RANDOMIZED BATCH TEST ===")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        answer_ids = get_answer_token_ids(tokenizer)
        A_token, B_token, C_token, D_token, E_token = answer_ids

        rng = random.Random(1234)

        # Create random prompt lengths in [1, max_prompt_len]
        prompt_lens = [rng.randint(1, max_prompt_len) for _ in range(num_samples)]
        # Fixed eos token used in earlier examples (arbitrary but consistent)
        EOS = 128001

        # Build sequences and labels with variable lengths, then pad to uniform length
        sequences = []
        labels = []
        answers = []
        for pl in prompt_lens:
            # random prompt tokens that are not A–E or EOS, using an offset range
            prompt = [1000 + rng.randint(0, 500) for _ in range(pl)]
            ans = rng.choice(answer_ids)
            answers.append(ans)
            seq = prompt + [ans, EOS]
            lab = [-100] * pl + [ans, EOS]
            sequences.append(seq)
            labels.append(lab)

        max_len = max(len(s) for s in sequences)

        # Pad sequences and labels
        PAD = 0
        sequences_padded = [s + [PAD] * (max_len - len(s)) for s in sequences]
        labels_padded = [l + [-100] * (max_len - len(l)) for l in labels]

        # Create logits tensor
        vocab_size = tokenizer.vocab_size
        B = num_samples
        S = max_len
        logits = torch.randn(B, S, vocab_size) * 0.05  # smaller noise

        # For each sample, set causal prediction at pred_pos = first_pos - 1
        for i, pl in enumerate(prompt_lens):
            first_pos = pl  # position of answer token
            pred_pos = max(0, first_pos - 1)  # position whose logits predict the answer
            correct = answers[i]

            # Make the correct answer very likely at pred_pos
            logits[i, pred_pos, correct] = 10.0
            for aid in answer_ids:
                if aid != correct:
                    logits[i, pred_pos, aid] = -5.0

            # Also make a WRONG token high at the answer position itself to ensure
            # the function is not mistakenly reading t instead of t-1.
            wrong = rng.choice([aid for aid in answer_ids if aid != correct])
            logits[i, first_pos, wrong] = 9.0

        # Run preprocess
        labels_tensor = torch.tensor(labels_padded, dtype=torch.long)

        # Reset debug counter so we see A–E logits once for this larger test
        try:
            preprocess_logits_for_metrics.debug_count = 0
        except Exception:
            pass

        preds = preprocess_logits_for_metrics(logits, labels_tensor, tokenizer)
        preds_list = preds.tolist()

        # Compare
        correct_count = sum(int(p == a) for p, a in zip(preds_list, answers))
        pred_letters = [tokenizer.decode([p]) for p in preds_list]
        true_letters = [tokenizer.decode([a]) for a in answers]

        # Distribution summary
        from collections import Counter

        pc = Counter(preds_list)
        tc = Counter(answers)
        print("\nPrediction distribution:", {tokenizer.decode([k]): pc[k] for k in answer_ids})
        print("True distribution:", {tokenizer.decode([k]): tc[k] for k in answer_ids})

        # Show a few examples
        print("Samples (first 10):")
        for i in range(min(10, B)):
            print(f"  {i}: pred={pred_letters[i]} true={true_letters[i]} {'✓' if preds_list[i]==answers[i] else '✗'}")

        print(f"Correct: {correct_count}/{B}")
        return correct_count == B

    large_ok = test_large_batch()

    print(f"\n{'='*40}")
    print("RESULTS:")
    print(f"Single sample: {'✅' if single_ok else '❌'}")
    print(f"Batch test: {'✅' if batch_ok else '❌'}")

    if single_ok and batch_ok and large_ok:
        print("🎉 All tests passed! Causal evaluation works!")
    else:
        print("❌ Some tests failed. Check the logic.")


if __name__ == "__main__":
    main()
