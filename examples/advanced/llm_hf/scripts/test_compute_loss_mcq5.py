#!/usr/bin/env python3

"""
Quick, self-contained test for compute_loss_mcq5 with dummy inputs.

It constructs:
  - A dummy tokenizer that maps A–E to fixed token IDs
  - Labels with exactly two supervised positions per sample where the
    second-to-last supervised token is the answer (as expected by the loss)
  - Logits with clear peaks at the answer positions for some samples

Run from repo root:
  python3 examples/advanced/llm_hf/scripts/test_compute_loss_mcq5.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def add_src_to_path():
    # Add ../src to import path so we can import training_utils
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(this_dir, "..", "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


add_src_to_path()
from training_utils import compute_loss_mcq5, get_answer_token_ids  # noqa: E402


class DummyTokenizer:
    """Minimal tokenizer that provides encode() for letters A–E.

    The training_utils.get_answer_token_ids() uses tokenizer.encode("A")[-1]
    to obtain the token ID. We map:
        A->10, B->11, C->12, D->13, E->14
    """

    def __init__(self):
        self.map = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}

    def encode(self, text, add_special_tokens=False):  # signature used by the helper
        if text not in self.map:
            # For any unexpected text, return a single fallback token id
            return [0]
        return [self.map[text]]


def build_dummy_batch():
    """Construct dummy logits and labels for 3 samples.

    Each sample has exactly two supervised positions (labels != -100):
        - the second-to-last supervised position is the answer token (A–E)
        - the last supervised position is a throwaway token (not used by the loss)
    """
    torch.manual_seed(7)

    B, T, V = 3, 6, 37  # batch size, sequence length, vocab size
    tok = DummyTokenizer()
    choice_ids = get_answer_token_ids(tok, ("A", "B", "C", "D", "E"))

    print("Choice token IDs (A–E):", choice_ids)

    # Initialize labels with ignore index (-100)
    labels = torch.full((B, T), -100, dtype=torch.long)

    # Define per-sample answers and positions
    # We ensure last two supervised positions are [answer_pos, tail_pos]
    #   Sample 0: answer=C at pos=2, tail at pos=5
    #   Sample 1: answer=B at pos=2, tail at pos=5
    #   Sample 2: answer=E at pos=3, tail at pos=5

    ans_tokens = ["C", "B", "E"]
    ans_positions = [2, 2, 3]

    for i in range(B):
        labels[i, ans_positions[i]] = tok.encode(ans_tokens[i])[0]
        labels[i, : ans_positions[i]] = torch.randint(0, V, (ans_positions[i],), dtype=torch.long)
        labels[i, ans_positions[i] + 1] = 120098
        # labels[i, tail_positions[i]] = 0  # tail token id (can be anything, not used)

    print("Labels tensor (shape BxT):\n", labels)

    # Build logits tensor and place strong signals at answer positions
    logits = torch.randn(B, T, V)
    A, B, C, D, E = choice_ids  # unpack for readability
    print("logits:", logits)
    print("Logits shape:", logits.shape)
    # Force peaks at the answer positions for samples 0 and 2 (correct predictions)
    logits[0, ans_positions[0], C] = 9.0  # sample 0, answer C
    logits[1, ans_positions[1], B] = 8.5  # sample 1, target=B but peak at D
    logits[2, ans_positions[2], E] = 11.0  # sample 2, answer E

    # For sample 1, intentionally make it predict the wrong class (D)
    print("logits:", logits)
    return SimpleNamespace(logits=logits), labels, tok


def compute_loss_mcq5(model_outputs, labels, num_items_in_batch=None, tokenizer=None):
    """
    Compute multiple-choice loss (A–E) using 5-class CE:
      - Each sequence has exactly one supervised answer token
      - We slice logits to only the 5 choice IDs
      - Target label is mapped to class index 0..4
    """
    logits = model_outputs.logits  # (B, T, V)
    B, T, V = logits.shape

    # Get the 5 choice token IDs
    choice_ids = torch.tensor(
        get_answer_token_ids(tokenizer, ("A", "B", "C", "D", "E")),
        device=logits.device,
        dtype=torch.long,
    )
    id2cls = {int(tid): i for i, tid in enumerate(choice_ids.tolist())}

    print(f"{id2cls=}")
    # Collect per-sample answer positions and target classes
    target_classes = []
    pos_idx = []
    for i in range(B):
        non_ignored = torch.nonzero(labels[i] != -100, as_tuple=True)[0]
        print(f"{non_ignored=}")
        print(f"non_ignored.numel()={non_ignored.numel()}")
        if non_ignored.numel() >= 2:
            ans_pos = non_ignored[-2].item()
            print(f"{ans_pos=}")
            print(f"{labels[i]=}")
            ans_id = int(labels[i, ans_pos].item())
            print(f"{ans_id=}")
            if ans_id not in id2cls:
                raise ValueError(f"Label id {ans_id} not in choice IDs {choice_ids.tolist()}")
            target_classes.append(id2cls[ans_id])
            pos_idx.append(ans_pos)
            print(f"[sample {i}] answer_pos={ans_pos}, token_id={ans_id}, class={id2cls[ans_id]}")
        else:
            print(f"[sample {i}] no valid answer found in labels")
            target_classes.append(-100)
            pos_idx.append(None)

    target_classes = torch.tensor(target_classes, device=logits.device, dtype=torch.long)

    # Gather logits at answer positions
    # shape: (B, V)
    answer_logits = []
    for i, pos in enumerate(pos_idx):
        if pos is not None:
            answer_logits.append(logits[i, pos])
    if len(answer_logits) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    answer_logits = torch.stack(answer_logits, dim=0)  # (B, V)

    # Slice to the 5 choices: (B, 5)
    print(f"{choice_ids=}")
    logits_5 = answer_logits[:, choice_ids]
    print(f"{logits_5=}")

    probs = F.softmax(logits_5, dim=-1)
    pred_classes = probs.argmax(dim=-1)

    for i in range(probs.size(0)):
        p = probs[i].detach().cpu().numpy().round(6).tolist()
        print(f"[sample {i}] probs={p}, target={target_classes[i].item()}, pred={pred_classes[i].item()}")

    # Cross-entropy loss
    loss_sum = F.cross_entropy(logits_5, target_classes, reduction="sum")

    print(f"len(answer_logits)={len(answer_logits)}")
    # Denom = trainer’s count or batch size
    denom = num_items_in_batch if num_items_in_batch is not None else len(answer_logits)
    loss = loss_sum / denom

    print(f"[compute_loss_mcq5] target_classes={target_classes.tolist()}, loss={loss.item():.10f}")
    return loss


def main():
    model_outputs, labels, tok = build_dummy_batch()

    print("\n=== Running compute_loss_mcq5 (denom inferred) ===")
    loss = compute_loss_mcq5(model_outputs, labels, tokenizer=tok)
    print("Returned loss (tensor):", loss)
    print("Returned loss (float):", float(loss))

    print("\n=== Running compute_loss_mcq5 (denom = batch size) ===")
    loss_b = compute_loss_mcq5(model_outputs, labels, num_items_in_batch=labels.size(0), tokenizer=tok)
    print("Returned loss (tensor):", loss_b)
    print("Returned loss (float):", float(loss_b))

    print("\nDone.")


if __name__ == "__main__":
    main()
