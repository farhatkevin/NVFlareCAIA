#!/usr/bin/env python3
"""
Check if answer tokens in labels_raw follow the expected pattern:
- Token at position -3 (relative to first -100): should be 271
- Token at position -2: should be one of 32-36 (A-E answer tokens)
- Token at position -1: should be 128009
"""

import json


def check_answer_pattern(labels_raw_file):
    """Check the answer token pattern in eval dump labels."""

    valid_answer_tokens = {32, 33, 34, 35, 36}  # A, B, C, D, E
    expected_before = 271
    expected_after = 128009

    results = {
        "total_samples": 0,
        "valid_pattern": 0,
        "invalid_pattern": 0,
        "pattern_details": [],
        "answer_token_counts": {32: 0, 33: 0, 34: 0, 35: 0, 36: 0},
    }

    with open(labels_raw_file, "r") as f:
        data = json.loads(f.readline())
        labels_raw = data["labels_raw"]

        results["total_samples"] = len(labels_raw)

        for i, label_seq in enumerate(labels_raw):
            # Find the first occurrence of -100
            try:
                first_neg100_idx = label_seq.index(-100)

                if first_neg100_idx >= 3:
                    token_minus_3 = label_seq[first_neg100_idx - 3]
                    token_minus_2 = label_seq[first_neg100_idx - 2]
                    token_minus_1 = label_seq[first_neg100_idx - 1]

                    # Check the pattern
                    pattern_valid = (
                        token_minus_3 == expected_before
                        and token_minus_2 in valid_answer_tokens
                        and token_minus_1 == expected_after
                    )

                    if pattern_valid:
                        results["valid_pattern"] += 1
                        if token_minus_2 in results["answer_token_counts"]:
                            results["answer_token_counts"][token_minus_2] += 1
                    else:
                        results["invalid_pattern"] += 1

                    # Store details for first 10 samples and any invalid ones
                    if i < 10 or not pattern_valid:
                        results["pattern_details"].append(
                            {
                                "sample": i,
                                "first_neg100_at": first_neg100_idx,
                                "token_-3": token_minus_3,
                                "token_-2": token_minus_2,
                                "token_-1": token_minus_1,
                                "pattern_valid": pattern_valid,
                                "expected_pattern": f"271, {token_minus_2 if token_minus_2 in valid_answer_tokens else '[INVALID]'}, 128009",
                            }
                        )
                else:
                    results["invalid_pattern"] += 1
                    results["pattern_details"].append(
                        {
                            "sample": i,
                            "error": f"Not enough tokens before first -100 (position {first_neg100_idx})",
                        }
                    )

            except ValueError:
                # No -100 found
                results["invalid_pattern"] += 1
                results["pattern_details"].append(
                    {"sample": i, "error": "No -100 tokens found in sequence"}
                )

    return results


def print_results(results):
    """Print the analysis results."""
    print("=" * 60)
    print("ANSWER TOKEN PATTERN ANALYSIS")
    print("=" * 60)
    print(f"Total samples analyzed: {results['total_samples']}")
    print(f"Samples with valid pattern: {results['valid_pattern']}")
    print(f"Samples with invalid pattern: {results['invalid_pattern']}")
    print(
        f"Success rate: {results['valid_pattern'] / results['total_samples'] * 100:.1f}%"
    )

    print("\nAnswer token distribution:")
    token_to_letter = {32: "A", 33: "B", 34: "C", 35: "D", 36: "E"}
    for token, count in results["answer_token_counts"].items():
        letter = token_to_letter[token]
        print(f"  {letter} (token {token}): {count} samples")

    print(f"\nPattern details (first 10 samples + any invalid):")
    print("-" * 60)
    for detail in results["pattern_details"]:
        if "error" in detail:
            print(f"Sample {detail['sample']:2d}: ERROR - {detail['error']}")
        else:
            status = "✓" if detail["pattern_valid"] else "✗"
            print(
                f"Sample {detail['sample']:2d}: {status} [{detail['token_-3']}, {detail['token_-2']}, {detail['token_-1']}] "
                f"at position {detail['first_neg100_at'] - 2}"
            )


if __name__ == "__main__":
    labels_file = "/data/input/kf/NVFlare/examples/advanced/llm_hf/workspace/hf_sft_multi/site-train_1/simulate_job/app_site-train_1/peft/eval_dump.jsonl"

    try:
        results = check_answer_pattern(labels_file)
        print_results(results)
    except Exception as e:
        print(f"Error analyzing file: {e}")
