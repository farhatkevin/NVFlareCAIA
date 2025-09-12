import json
import os
import re
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction, TrainerCallback


def get_answer_token_ids(tokenizer, choices=("A", "B", "C", "D", "E")) -> List[int]:
    """Return token IDs for choices in the given order (A–E by default).

    This function preserves the order of `choices` so callers can map
    class indices 0..len(choices)-1 directly to these IDs.
    """
    return [tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in choices]


# NOTE: get_answer_token_ids_ordered is no longer needed since
# get_answer_token_ids already preserves order.


def preprocess_logits_for_metrics(logits, labels, tokenizer):
    """
    Preprocess logits for multiple choice evaluation where answers are single letters A-E.

    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        labels: Ground truth labels of shape (batch_size, seq_len)

    Returns:
        Processed logits for metric calculation
    """
    import torch

    # Token IDs for A–E in order for slicing logits consistently
    choice_token_ids = torch.tensor(
        get_answer_token_ids(tokenizer, ("A", "B", "C", "D", "E")),
        device=logits.device,
        dtype=torch.long,
    )

    # Get logits for the last non-padding token (where the answer should be)
    batch_size = logits.shape[0]
    last_token_logits = []

    for i in range(batch_size):
        # Find the last non-padding token position
        non_pad_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
        #   print(f"Sample {i} non-pad positions: {non_pad_positions.tolist()}")
        if len(non_pad_positions) > 0:
            # TODO: this is assuming eos token before answer token
            last_pos = non_pad_positions[-2]
            # print(f"Sample {i}: last non-pad position for prediction is {last_pos.item()}")
            # print(labels[i, last_pos-3:last_pos+3])  # Show context around prediction
            #   last_token_logits.append(logits[i, last_pos, choice_token_ids])
            # TODO: double check off-by-one here for last_pos
            last_token_logits.append(logits[i, last_pos - 1, choice_token_ids].argmax(dim=-1))
            # print("last token logits: ", last_token_logits)
        else:
            print("preprocessing logits is not finding any non-pad positions")
            last_token_logits.append(torch.zeros(choice_token_ids.shape[0], device=logits.device))

    return torch.stack(last_token_logits)


def compute_metrics(eval_preds: EvalPrediction, tokenizer, verbose=True) -> Dict[str, float]:
    """Compare argmax predictions to true labels for accuracy"""
    pred_ids, labels = eval_preds

    pred_ids = np.asarray(pred_ids).reshape(-1)
    labels = np.asarray(labels)

    if labels.ndim != 2 or labels.shape[0] == 0:
        return {"accuracy": 0.0}

    # Build y_true as the LAST non-ignored A–E token per sample (assistant answer at end)
    # Ordered IDs; use set for membership checks
    answer_ids = get_answer_token_ids(tokenizer)
    true_ids = []
    for i in range(labels.shape[0]):
        row = labels[i]
        idxs = np.where(row != -100)[0]
        # print(f"row: {row}, idxs: {idxs}")
        t_id = -1
        for j in reversed(idxs.tolist()):
            if row[j] in answer_ids:
                # print(f"found answer token {row[j]} at position {j}")
                t_id = int(row[j])
                # print("position where token is found: ", j)
                # print("idk", row[j-3:j+4])
                # print("token_id:", t_id)
                break
        true_ids.append(t_id)

    true_ids = np.array(true_ids, dtype=np.int64)
    # print("true_ids: ", true_ids)
    valid_idx = np.where(true_ids != -1)[0]
    # print("valid idx: ", valid_idx)
    if valid_idx.size == 0:
        return {"accuracy": 0.0}
    y_true = true_ids[valid_idx]
    # print("y_true: ", y_true)
    y_pred = pred_ids[valid_idx]
    # Map predicted class indices (0..4) to tokenizer-specific token IDs for A–E
    index_to_token = {i: tid for i, tid in enumerate(answer_ids)}
    for i in range(len(y_pred)):
        if y_pred[i] in index_to_token:
            y_pred[i] = index_to_token[y_pred[i]]

    # print("y_pred: ", y_pred)
    # Truncate to common length if needed (defensive)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return {"accuracy": 0.0}
    accuracy = float((y_pred[:m] == y_true[:m]).mean())

    if verbose:
        print(f"Evaluated {len(valid_idx)} samples, accuracy: {accuracy:.3f}")

    return {"accuracy": accuracy}


def format_instruction(example):
    """Convert to messages format that TRL expects"""
    system_prompt = "You are a medical AI expert."

    if isinstance(example["prompt"], list):
        # Batch processing
        formatted_messages = []
        for i, prompt in enumerate(example["prompt"]):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["completion"][i]},
            ]
            formatted_messages.append(messages)
        return {"messages": formatted_messages}
    else:
        # Single example
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        return {"messages": messages}


def filter_by_length_messages(example, tokenizer, max_length):
    """Filter function for messages format"""
    try:
        # Apply chat template to get the full formatted text
        formatted_text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(formatted_text)
        return len(tokens) <= max_length
    except Exception as e:
        # Log the error for debugging
        print(f"Error in filter_by_length_messages: {e}")
        print(f"Example keys: {example.keys() if hasattr(example, 'keys') else 'Not a dict'}")
        if "messages" in example:
            print(f"Messages type: {type(example['messages'])}")
        return False


def print_first_k_eval_predictions(k: int = 2):
    """Preview the first k recorded eval predictions without regenerating.

    Relies on predictions captured by `wrap_compute_metrics`. Prints the
    choice index (0-4) and decoded letter (A-E) for quick inspection.
    """
    try:
        preds = _last_eval_data.get("predictions")
        labels = _last_eval_data.get("label_ids")
        if preds is None:
            print("No recorded eval predictions found. Ensure compute_metrics is wrapped with wrap_compute_metrics.")
            return
        # Convert to list for uniform handling
        preds_list = _to_list(preds) or []
        labels_list = _to_list(labels) if labels is not None else None
        n = min(k, len(preds_list))
        to_choice = ["A", "B", "C", "D", "E"]
        print(f"Recorded eval preview for {n} sample(s):")
        for i in range(n):
            # Each pred is expected to be an int in [0..4]
            p_idx = preds_list[i]
            try:
                p_choice = to_choice[p_idx]
            except Exception:
                p_choice = str(p_idx)
            row = f"pred_index={p_idx}, pred_choice={p_choice}"
            if labels_list is not None:
                row += f", label_row_len={len(labels_list[i]) if hasattr(labels_list[i], '__len__') else 'NA'}"
            print(f"- [{i}] {row}")
    except Exception as e:
        print(f"print_first_k_eval_predictions failed: {e}")


class EvalPreviewCallback(TrainerCallback):
    """Callback to print K recorded predictions at every evaluation (no regeneration)."""

    def __init__(self, k: int = 2, local_rank: int = 0):
        self.k = k
        self.local_rank = local_rank

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.local_rank != 0:
            return
        try:
            print_first_k_eval_predictions(k=self.k)
        except Exception as e:
            print(f"EvalPreviewCallback failed: {e}")


# --- Recorded predictions/labels from evaluate() ---
_last_eval_data = {"predictions": None, "label_ids": None}


def wrap_compute_metrics(base_compute_fn):
    """Wrap compute_metrics to record raw predictions/labels without altering scoring.

    The returned function takes only `eval_preds` since other args can be bound via partial.
    """

    def new_compute(eval_preds):
        try:
            preds, labels = eval_preds
            _last_eval_data["predictions"] = preds
            _last_eval_data["label_ids"] = labels
        except Exception as e:
            print(f"wrap_compute_metrics: failed to record eval preds: {e}")
        return base_compute_fn(eval_preds)

    return new_compute


class EvalDumpCallback(TrainerCallback):
    """Append recorded eval predictions to a JSONL file (no regeneration).

    Each evaluation appends one JSON object with fields:
      - step, epoch, eval_loss, eval_accuracy (if available)
      - predictions_raw: list of ints (indices 0-4)
      - predictions_choice: list of letters (A-E)
      - labels_raw: optional raw labels if available and include_labels=True
    """

    def __init__(self, out_path: str, local_rank: int = 0, include_labels: bool = True):
        self.out_path = out_path
        self.local_rank = local_rank
        self.include_labels = include_labels
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.local_rank != 0:
            return
        preds = _last_eval_data.get("predictions")
        labels = _last_eval_data.get("label_ids")
        if preds is None:
            print("EvalDumpCallback: no recorded predictions found; ensure wrap_compute_metrics is used.")
            return

        preds_list = _to_list(preds) or []
        to_choice = ["A", "B", "C", "D", "E"]
        preds_choice = [to_choice[p] if isinstance(p, int) and 0 <= p < 5 else str(p) for p in preds_list]
        rec = {
            "step": getattr(state, "global_step", None),
            "epoch": metrics.get("epoch") if isinstance(metrics, dict) else None,
            "eval_loss": metrics.get("eval_loss") if isinstance(metrics, dict) else None,
            "eval_accuracy": metrics.get("eval_accuracy") if isinstance(metrics, dict) else None,
            "predictions_raw": preds_list,
            "predictions_choice": preds_choice,
        }
        if self.include_labels and labels is not None:
            rec["labels_raw"] = _to_list(labels)
        try:
            with open(self.out_path, "a") as f:
                json.dump(rec, f)
                f.write("\n")
        except Exception as e:
            print(f"EvalDumpCallback write failed: {e}")


class EvalDumpRecordedCallback(TrainerCallback):
    """Write the raw and decoded predictions used during evaluation to JSONL (no regeneration)."""

    def __init__(self, out_path: str, local_rank: int = 0, include_labels: bool = True, tokenizer=None):
        self.out_path = out_path
        self.local_rank = local_rank
        self.include_labels = include_labels
        self.tokenizer = tokenizer
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.local_rank != 0:
            return
        preds = _last_eval_data.get("predictions")
        labels = _last_eval_data.get("label_ids")
        if preds is None:
            return
        preds_list = _to_list(preds) or []
        to_choice = ["A", "B", "C", "D", "E"]
        preds_choice = [to_choice[p] if isinstance(p, int) and 0 <= p < 5 else str(p) for p in preds_list]
        rec = {
            "step": getattr(state, "global_step", None),
            "epoch": metrics.get("epoch") if isinstance(metrics, dict) else None,
            "eval_loss": metrics.get("eval_loss") if isinstance(metrics, dict) else None,
            "eval_accuracy": metrics.get("eval_accuracy") if isinstance(metrics, dict) else None,
            "predictions_raw": preds_list,
            "predictions_choice": preds_choice,
        }
        if self.include_labels and labels is not None:
            # Build mapping based on tokenizer if available, else fallback
            ordered_ids = (
                get_answer_token_ids(self.tokenizer) if self.tokenizer is not None else [32, 33, 34, 35, 36]
            )
            token_to_choice = {tid: ch for tid, ch in zip(ordered_ids, ["A", "B", "C", "D", "E"])}

            # Extract compact answer tokens instead of full sequences
            answer_tokens = _extract_answer_tokens(labels, valid_answer_tokens=set(ordered_ids))
            if answer_tokens is not None:
                rec["labels_answer_tokens"] = answer_tokens
                # Also provide letter choices for convenience
                rec["labels_answer_choices"] = [
                    token_to_choice.get(token, "?" if token != -1 else "INVALID") for token in answer_tokens
                ]
            else:
                rec["labels_answer_tokens"] = None
                rec["labels_answer_choices"] = None
        try:
            with open(self.out_path, "a") as f:
                json.dump(rec, f)
                f.write("\n")
        except Exception as e:
            print(f"EvalDumpRecordedCallback write failed: {e}")


def _extract_answer_tokens(labels, valid_answer_tokens: set[int]):
    """Extract just the answer tokens (A–E) from labels_raw sequences.

    For each sequence, finds the token 2 positions before the first -100,
    which should be the answer token. Valid answers are provided via
    `valid_answer_tokens` (derived from the tokenizer).

    Args:
        labels: Raw label sequences from evaluation

    Returns:
        List of answer token IDs, or None if extraction fails
    """
    try:
        labels_list = _to_list(labels)
        if not labels_list:
            return None

        answer_tokens = []

        for label_seq in labels_list:
            try:
                # Find first -100 token
                first_neg100_idx = label_seq.index(-100)
                if first_neg100_idx >= 2:
                    # Get token 2 positions before first -100
                    answer_token = label_seq[first_neg100_idx - 2]
                    if answer_token in valid_answer_tokens:
                        answer_tokens.append(answer_token)
                    else:
                        # Invalid answer token, mark as -1
                        answer_tokens.append(-1)
                else:
                    # Not enough tokens before -100
                    answer_tokens.append(-1)
            except ValueError:
                # No -100 found - check if it ends with [271, answer_token, 128009] pattern
                if len(label_seq) >= 3:
                    # Check last 3 tokens for pattern
                    if label_seq[-3] == 271 and label_seq[-2] in valid_answer_tokens and label_seq[-1] == 128009:
                        answer_tokens.append(label_seq[-2])
                    else:
                        answer_tokens.append(-1)
                else:
                    answer_tokens.append(-1)

        return answer_tokens

    except Exception as e:
        print(f"_extract_answer_tokens failed: {e}")
        return None


def _to_list(x):
    try:
        import numpy as _np
        import torch as _torch

        if isinstance(x, _np.ndarray):
            return x.tolist()
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return list(x)
    except Exception:
        try:
            return list(x)
        except Exception:
            return None


class LossFileLogger(TrainerCallback):
    """Minimal logger that writes only loss values to a file.

    - Logs training loss on `on_log` (controlled by HF `logging_steps`).
    - Logs eval loss on `on_evaluate`.
    - Writes only on local_rank 0 to avoid duplicates in DDP.
    - Includes site name and FL round for easy aggregation.
    """

    def __init__(self, log_file: str, site_name: str, local_rank: int = 0):
        self.log_file = log_file
        self.site_name = site_name
        self.local_rank = local_rank
        self.current_round = 0
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def set_round(self, r: int | None):
        try:
            self.current_round = int(r) if r is not None else 0
        except Exception:
            self.current_round = 0

    def _write(self, line: str):
        # Only rank 0 writes to avoid duplicates in multi-GPU
        if self.local_rank != 0:
            return
        try:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            # Best-effort logging; avoid crashing training on I/O issues
            print(f"LossFileLogger write failed: {e}")

    # Called during training according to Trainer's logging frequency
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Write the raw dict exactly as received from HF Trainer
        self._write(str(logs))

    # Called after evaluation with metrics
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        # Write the raw dict exactly as received from HF Trainer
        self._write(str(metrics))


def create_loss_logger(site_name: str, output_path: str, loss_log_file: str | None, local_rank: int) -> LossFileLogger:
    """Factory to create a LossFileLogger with a sensible default filepath.

    If `loss_log_file` is None, uses `<output_path>/loss_<site_name>.log`.
    """
    # If user provides a path, and it's relative, place it under output_path
    if loss_log_file:
        log_file = loss_log_file if os.path.isabs(loss_log_file) else os.path.join(output_path, loss_log_file)
    else:
        log_file = os.path.join(output_path, f"loss_{site_name}.log")
    return LossFileLogger(log_file=log_file, site_name=site_name, local_rank=local_rank)
