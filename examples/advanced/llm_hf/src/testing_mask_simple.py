import argparse
import json
import os
import sys
from typing import List
import numpy as np  # used by compute_metrics
import torch        # used by preprocess_logits_for_metrics


def try_load_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_name_or_path)
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        print(f"Loaded HF tokenizer from '{model_name_or_path}'.")
        return tok
    except Exception as e:
        print(f"Could not load HF tokenizer ('{model_name_or_path}'): {e}")
        print("Falling back to ToyTokenizer (A..E only).")

        class ToyTokenizer:
            def __init__(self):
                self.map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
                self.inv = {v: k for k, v in self.map.items()}
                self.eos_token_id = 128001

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                ids: List[int] = []
                for ch in text:
                    if ch in self.map:
                        ids.append(self.map[ch])
                if not ids:
                    return [self.eos_token_id]
                return ids

            def decode(self, ids: List[int]) -> str:
                if isinstance(ids, int):
                    ids = [ids]
                out: List[str] = []
                for i in ids:
                    if i in self.inv:
                        out.append(self.inv[i])
                    else:
                        out.append(f"<{i}>")
                return "".join(out)

        return ToyTokenizer()


def main():
    parser = argparse.ArgumentParser(description="Simplest prompt/completion masking debugger")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "synthea_data",
            "train_1",
            "training.jsonl",
        ),
        help="Path to training.jsonl",
    )
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/llama-3.1-8B-Instruct",
        help="Optional local tokenizer path; falls back to ToyTokenizer.",
    )
    args = parser.parse_args()

    # import training_utils for get_answer_token_ids and metric helpers
    src_dir = os.path.dirname(__file__)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from training_utils import (
        get_answer_token_ids,
        preprocess_logits_for_metrics,
        compute_metrics,
    )

    tokenizer = try_load_tokenizer(args.model_name_or_path)

    # Show token variants for A..E
    letters = ["A", "B", "C", "D", "E"]
    print("=== Token variants for A..E ===")
    for l in letters:
        plain = tokenizer.encode(l, add_special_tokens=False)
        sp = tokenizer.encode(" " + l, add_special_tokens=False)
        nl = tokenizer.encode("\n" + l, add_special_tokens=False)
        print(f"{l}: plain={plain}  space={sp}  newline={nl}")
    allowed = get_answer_token_ids(tokenizer)
    print("Allowed (plain) answer ids:", allowed)
    print("Decoded allowed:", [tokenizer.decode([i]) for i in allowed])
    print("===============================\n")

    # Read a few samples
    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = str(obj.get("prompt", ""))
            completion = str(obj.get("completion", "")).strip()
            if completion in {"A", "B", "C", "D", "E"}:
                rows.append((prompt, completion))
            if len(rows) >= args.num_samples:
                break

    if not rows:
        print("No A..E completions found; nothing to show.")
        return

    # Tokenize all rows and prepare per-sample info
    per = []
    for idx, (prompt, completion) in enumerate(rows):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        comp_ids = tokenizer.encode(completion, add_special_tokens=False)
        # Keep only the FIRST completion token as the target (labels for others ignored)
        if not comp_ids:
            continue
        answer_id = comp_ids[0]
        first_pos = len(prompt_ids)
        pred_pos = max(first_pos - 1, 0)
        per.append({
            "prompt_ids": prompt_ids,
            "comp_ids": comp_ids,
            "answer_id": answer_id,
            "first_pos": first_pos,
            "pred_pos": pred_pos,
        })

    if not per:
        print("No suitable samples with non-empty completion tokenization.")
        return

    # Build a batch with minimal length up to first_pos+1 per sample
    B = len(per)
    S = max(x["first_pos"] for x in per) + 1
    V = max(max(allowed) + 1 if allowed else 10, getattr(tokenizer, "vocab_size", 50), 50)

    labels = torch.full((B, S), -100, dtype=torch.long)
    logits = torch.zeros((B, S, V), dtype=torch.float32)

    # For each sample, place the true answer at first_pos in labels.
    # In logits, make the correct answer high at pred_pos (t-1), and a wrong one high at t (first_pos)
    for i, info in enumerate(per):
        fp = info["first_pos"]
        pp = info["pred_pos"]
        ans = info["answer_id"]
        labels[i, fp] = ans

        # enforce causal alignment: correct at t-1
        logits[i, pp, ans] = 10.0

        # at time t, push a different allowed id high to show misalignment clearly
        if ans in allowed and len(allowed) > 1:
            idx = allowed.index(ans)
            wrong = allowed[(idx + 1) % len(allowed)]
        else:
            wrong = (ans + 1) % V
        logits[i, fp, wrong] = 9.0

    # Use the provided preprocess and compute_metrics
    preds_current = preprocess_logits_for_metrics(logits.clone(), labels.clone(), tokenizer)
    preds_current_np = preds_current.cpu().numpy()

    print("\n=== Batch Boundary View ===")
    for i, info in enumerate(per):
        fp = info["first_pos"]
        pp = info["pred_pos"]
        answer_id = info["answer_id"]
        print(f"Sample {i}: first_pos(t)={fp}, pred_pos(t-1)={pp}, true_id={answer_id} ('{tokenizer.decode([answer_id])}')")
        # Show the logits focus at t-1 and t
        top_t_1 = int(torch.argmax(logits[i, pp, :]).item())
        top_t = int(torch.argmax(logits[i, fp, :]).item())
        print(f"  top@t-1: id={top_t_1} dec='{tokenizer.decode([top_t_1])}'  | top@t: id={top_t} dec='{tokenizer.decode([top_t])}'")

    print("\nPredictions with CURRENT preprocess (no -1 shift inside function):")
    print("  ids:", preds_current_np.tolist())
    print("  dec:", [tokenizer.decode([int(x)]) for x in preds_current_np])

    acc_current = compute_metrics((preds_current_np, labels.cpu().numpy()), tokenizer, verbose=True)
    print(f"Accuracy (current): {acc_current}")

    # Provide a local shifted version for comparison (first_pos - 1)
    def preprocess_shifted(logits_in, labels_in, tokenizer_in=None):
        with torch.no_grad():
            if isinstance(logits_in, tuple):
                logits_in = logits_in[0]
            B_, S_, V_ = logits_in.shape
            labels_t = labels_in if torch.is_tensor(labels_in) else torch.tensor(labels_in, device=logits_in.device)
            labels_t = labels_t.to(logits_in.device)
            mask_ = labels_t != -100
            has_any = mask_.any(dim=1)
            first_pos_ = mask_.float().argmax(dim=1)
            pred_pos_ = torch.clamp(first_pos_ - 1, min=0)
            pos_ = torch.where(has_any, pred_pos_, torch.full_like(pred_pos_, S_ - 1))
            row_ = torch.arange(B_, device=logits_in.device)
            sel = logits_in[row_, pos_, :]
            if tokenizer_in is not None:
                allowed_ = get_answer_token_ids(tokenizer_in)
                mask_vec = torch.full((V_,), float("-inf"), device=sel.device, dtype=sel.dtype)
                mask_vec[allowed_] = 0
                sel = sel + mask_vec
            return sel.argmax(dim=-1)

    preds_shift = preprocess_shifted(logits.clone(), labels.clone(), tokenizer)
    preds_shift_np = preds_shift.cpu().numpy()
    print("\nPredictions with SHIFTED preprocess (first_pos - 1):")
    print("  ids:", preds_shift_np.tolist())
    print("  dec:", [tokenizer.decode([int(x)]) for x in preds_shift_np])

    acc_shift = compute_metrics((preds_shift_np, labels.cpu().numpy()), tokenizer, verbose=True)
    print(f"Accuracy (shifted): {acc_shift}")

    print("Summary: For causal LM, logits at pred_pos (t-1) predict the first non-ignored label at first_pos (t).\n"
          "Current function reads at t; shifted version reads at t-1. If the first completion token is not in plain A..E,\n"
          "consider adding space/newline variants to get_answer_token_ids for your tokenizer.")


if __name__ == "__main__":
    main()
