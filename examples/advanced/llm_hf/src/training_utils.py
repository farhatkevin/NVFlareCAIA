import hashlib
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction, TrainerCallback, LogitsProcessor, LogitsProcessorList

# ConstrainedLogitsProcessor - copied from evaluate_local_model_constrained.py for consistency
class ConstrainedLogitsProcessor(LogitsProcessor):
    """Logits processor that constrains generation to only allowed tokens."""

    def __init__(self, allowed_token_ids):
        """
        Args:
            allowed_token_ids: List of token IDs that are allowed (e.g., for 'A', 'B', 'C', 'D', 'E').
        """
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create mask with -inf for all tokens
        mask = torch.full_like(scores, float("-inf"))

        # Set allowed tokens to 0 (no penalty)
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0

        scores = scores + mask
        return scores

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
    """Compare argmax predictions to true labels for accuracy using constrained generation"""
    pred_ids, labels = eval_preds

    pred_ids = np.asarray(pred_ids).reshape(-1)
    labels = np.asarray(labels)

    if labels.ndim != 2 or labels.shape[0] == 0:
        return {"accuracy": 0.0}

    # Build y_true as the LAST non-ignored A–E token per sample (assistant answer at end)
    answer_ids = get_answer_token_ids(tokenizer)
    true_ids = []
    for i in range(labels.shape[0]):
        row = labels[i]
        idxs = np.where(row != -100)[0]
        t_id = -1
        for j in reversed(idxs.tolist()):
            if row[j] in answer_ids:
                t_id = int(row[j])
                break
        true_ids.append(t_id)

    true_ids = np.array(true_ids, dtype=np.int64)
    valid_idx = np.where(true_ids != -1)[0]
    if valid_idx.size == 0:
        return {"accuracy": 0.0}

    y_true = true_ids[valid_idx]
    y_pred = pred_ids[valid_idx]

    # Map predicted class indices (0..4) to tokenizer-specific token IDs for A–E
    index_to_token = {i: tid for i, tid in enumerate(answer_ids)}
    for i in range(len(y_pred)):
        if y_pred[i] in index_to_token:
            y_pred[i] = index_to_token[y_pred[i]]

    # Truncate to common length if needed (defensive)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return {"accuracy": 0.0}
    accuracy = float((y_pred[:m] == y_true[:m]).mean())

    if verbose:
        print(f"Evaluated {len(valid_idx)} samples, accuracy: {accuracy:.3f}")

        # Log first few examples to see actual generations
        print("\n--- Sample generations from evaluation ---")
        for i in range(min(3, len(valid_idx))):
            idx = valid_idx[i]
            label_row = labels[idx] if labels.ndim > 1 else labels

            non_pad_mask = label_row != -100
            if non_pad_mask.any():
                valid_tokens = label_row[non_pad_mask]
                decoded_text = tokenizer.decode(valid_tokens, skip_special_tokens=True)

                pred_token = tokenizer.decode([y_pred[i]], skip_special_tokens=True) if i < len(y_pred) else "N/A"
                true_token = tokenizer.decode([y_true[i]], skip_special_tokens=True) if i < len(y_true) else "N/A"

                print(f"Example {i + 1}:")
                print(f"  Full text: {decoded_text[-200:]}")
                print(f"  Predicted: {pred_token}, True: {true_token}")
                print()

    return {"accuracy": accuracy}


def compute_metrics_constrained(eval_dataset, model, tokenizer, choices=("A", "B", "C", "D", "E"), verbose=True) -> Dict[str, float]:
    """
    Compute metrics using constrained generation (like offline eval).

    This function performs actual constrained generation during evaluation,
    matching the methodology used in the offline evaluation script.
    """
    device = next(model.parameters()).device

    # Get allowed token IDs for constrained generation
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in choices]

    # Verify all tokens are in vocabulary
    if any(tid == tokenizer.unk_token_id for tid in allowed_token_ids):
        missing_tokens = [choices[i] for i, tid in enumerate(allowed_token_ids) if tid == tokenizer.unk_token_id]
        raise ValueError(f"Some choice tokens not in vocabulary: {missing_tokens}")

    if verbose:
        print(f"Choice tokens mapped to IDs: {dict(zip(choices, allowed_token_ids))}")

    # Create constrained logits processor
    logits_processors = LogitsProcessorList([ConstrainedLogitsProcessor(allowed_token_ids)])

    correct = 0
    total = 0
    predictions = []

    model.eval()
    with torch.no_grad():
        for i, example in enumerate(eval_dataset):
            # Get the input prompt (everything before the answer)
            messages = example["messages"]

            # Extract the user prompt (before assistant answer)
            prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]

            # Apply chat template to get input prompt
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback if no chat template
                formatted_prompt = prompt_messages[-1]["content"] if prompt_messages else ""

            # Tokenize input
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate with constraints
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1,
                logits_processor=logits_processors,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )

            # Extract generated token
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Map prediction to choice index
            try:
                predicted_idx = choices.index(predicted_text)
            except ValueError:
                predicted_idx = -1

            # Get ground truth from messages
            ground_truth_text = None
            for msg in messages:
                if msg["role"] == "assistant":
                    ground_truth_text = msg["content"].strip()
                    break

            if ground_truth_text and ground_truth_text in choices:
                ground_truth_idx = choices.index(ground_truth_text)
            else:
                ground_truth_idx = -1

            # Check if correct
            is_correct = predicted_idx == ground_truth_idx and predicted_idx != -1
            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                "predicted_choice": predicted_text,
                "predicted_idx": predicted_idx,
                "ground_truth_choice": ground_truth_text,
                "ground_truth_idx": ground_truth_idx,
                "is_correct": is_correct
            })

    accuracy = correct / total if total > 0 else 0.0

    if verbose:
        print(f"Constrained generation evaluation:")
        print(f"Total examples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.4f}")

        # Show first few examples
        print("\n--- Sample constrained generations ---")
        for i, pred in enumerate(predictions[:3]):
            print(f"Example {i + 1}:")
            print(f"  Predicted: {pred['predicted_choice']}")
            print(f"  Ground truth: {pred['ground_truth_choice']}")
            print(f"  Correct: {pred['is_correct']}")
            print()

    return {"accuracy": accuracy}




from trl import SFTTrainer

class ConstrainedSFTTrainer(SFTTrainer):
    """
    Direct subclass of SFTTrainer that runs constrained evaluation within the evaluation loop.
    This ensures eval_accuracy is available for metric_for_best_model selection.
    """

    def __init__(self, constrained_eval_dataset=None, constrained_eval_tokenizer=None,
                 constrained_eval_choices=("A", "B", "C", "D", "E"), **kwargs):

        # Store constrained eval parameters
        self.constrained_eval_dataset = constrained_eval_dataset
        self.constrained_eval_tokenizer = constrained_eval_tokenizer
        self.constrained_eval_choices = constrained_eval_choices

        # Initialize parent SFTTrainer
        super().__init__(**kwargs)

    def evaluation_loop(self, dataloader, description: str, prediction_loss_only=None,
                       ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation_loop to add constrained metrics before best model checking.
        """
        # Run the normal evaluation first (gets eval_loss)
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # Add constrained accuracy for eval runs (not train subset evals)
        if (metric_key_prefix == "eval" and
            self.constrained_eval_dataset is not None):

            try:
                constrained_results = compute_metrics_constrained(
                    eval_dataset=self.constrained_eval_dataset,
                    model=self.model,
                    tokenizer=self.constrained_eval_tokenizer,
                    choices=self.constrained_eval_choices,
                    verbose=False  # Reduce verbosity
                )

                # Add to the metrics that will be used for model selection
                if "accuracy" in constrained_results:
                    output.metrics["eval_accuracy"] = constrained_results["accuracy"]

            except Exception as e:
                print(f"Constrained evaluation failed: {e}")

        return output


def compute_loss_mcq5(
    model_outputs,
    labels,
    gradient_accumulation_steps,
    num_items_in_batch=None,
    tokenizer=None,
):
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

    # Collect per-sample answer positions and target classes
    target_classes = []
    pos_idx = []
    for i in range(B):
        non_ignored = torch.nonzero(labels[i] != -100, as_tuple=True)[0]
        if non_ignored.numel() >= 2:
            ans_pos = non_ignored[-2].item()
            ans_id = int(labels[i, ans_pos].item())
            if ans_id not in id2cls:
                raise ValueError(f"Label id {ans_id} not in choice IDs {choice_ids.tolist()}")
            target_classes.append(id2cls[ans_id])
            pos_idx.append(ans_pos)
            # print(f"[sample {i}] answer_pos={ans_pos}, token_id={ans_id}, class={id2cls[ans_id]}")
        else:
            print(f"[sample {i}] no valid answer found in labels")
            target_classes.append(-100)
            pos_idx.append(None)

    target_classes = torch.tensor(target_classes, device=logits.device, dtype=torch.long)

    # Gather logits at answer positions
    # shape: (B, V)
    answer_logits = []
    for i, pos in enumerate(pos_idx):
        if pos is not None and pos > 0:
            answer_logits.append(logits[i, pos - 1])
        else:
            print(f"[sample {i}] skipping because answer_pos={pos}")
    if len(answer_logits) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    answer_logits = torch.stack(answer_logits, dim=0)  # (B, V)

    # Slice to the 5 choices: (B, 5)
    logits_5 = answer_logits[:, choice_ids]

    probs = F.softmax(logits_5, dim=-1)
    pred_classes = probs.argmax(dim=-1)

    for i in range(probs.size(0)):
        p = probs[i].detach().cpu().numpy().round(6).tolist()
        # print(f"[sample {i}] probs={p}, target={target_classes[i].item()}, pred={pred_classes[i].item()}")

    # Cross-entropy loss
    # TODO: check whether we need to use num_items_in_batch, sometimes it is None
    loss = F.cross_entropy(logits_5, target_classes, reduction="mean")

    if gradient_accumulation_steps and gradient_accumulation_steps > 1:
        loss = loss / float(gradient_accumulation_steps)

    # print(f"[compute_loss_mcq5] target_classes={target_classes.tolist()}, loss={loss.item():.10f}")
    return loss
    # return torch.tensor(0.7, device=model_outputs.logits.device, requires_grad=True)


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

    def __init__(
        self,
        out_path: str,
        local_rank: int = 0,
        include_labels: bool = True,
        tokenizer=None,
    ):
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
            ordered_ids = get_answer_token_ids(self.tokenizer) if self.tokenizer is not None else [32, 33, 34, 35, 36]
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
        # Print just a simple site tag before the framework prints the dict
        try:
            print(f"[site={self.site_name}]")
        except Exception:
            pass
        prefix = f"[site={self.site_name}][round={self.current_round}]"
        self._write(f"{prefix} train: {logs}")

    # Called after evaluation with metrics
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        # Print just a simple site tag before the framework prints the dict
        try:
            print(f"[site={self.site_name}]")
        except Exception:
            pass
        prefix = f"[site={self.site_name}][round={self.current_round}]"
        # Always write full metrics dict to file
        self._write(f"{prefix} eval: {metrics}")


class WandbMetricsLogger(TrainerCallback):
    """Logs key metrics explicitly to Weights & Biases.

    - Only logs on local_rank 0 to avoid duplication.
    - Focuses on eval metrics like eval_accuracy/eval_loss that may not appear by default.
    - Stores immutable metadata (site) in config/summary instead of logging as a metric.
    """

    def __init__(self, site_name: str, local_rank: int = 0):
        self.site_name = site_name
        self.local_rank = local_rank
        self.current_round = 0

    def set_round(self, r: int | None):
        try:
            self.current_round = int(r) if r is not None else 0
        except Exception:
            self.current_round = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Save site info once as run metadata to avoid W&B media panel warnings
        try:
            import wandb  # type: ignore

            if wandb.run is None:
                return
            wandb.config.update({"site": self.site_name}, allow_val_change=True)
            wandb.run.summary["site"] = self.site_name
        except Exception:
            pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.local_rank != 0 or not isinstance(metrics, dict):
            return
        try:
            import wandb  # type: ignore

            if wandb.run is None:
                return

            log_items = {}
            # Accuracy: log both grouped and flat keys
            if "eval_accuracy" in metrics and metrics["eval_accuracy"] is not None:
                log_items["eval/accuracy"] = metrics["eval_accuracy"]
                log_items["eval_accuracy"] = metrics["eval_accuracy"]

            # Loss: log both grouped and flat keys
            if "eval_loss" in metrics and metrics["eval_loss"] is not None:
                log_items["eval/loss"] = metrics["eval_loss"]
                log_items["eval_loss"] = metrics["eval_loss"]
            # Training metrics if produced via evaluate(..., metric_key_prefix="train")
            if "train_accuracy" in metrics and metrics["train_accuracy"] is not None:
                print("Logging train_accuracy to W&B")
                log_items["train/accuracy"] = metrics["train_accuracy"]
                log_items["train_accuracy"] = metrics["train_accuracy"]
            if "train_loss" in metrics and metrics["train_loss"] is not None:
                print("Logging train_loss to W&B")
                log_items["train/loss"] = metrics["train_loss"]
                log_items["train_loss"] = metrics["train_loss"]

            # Also log current training loss from trainer state if available
            if hasattr(state, "log_history") and state.log_history:
                # Get the most recent training loss
                for log_entry in reversed(state.log_history):
                    if "loss" in log_entry:
                        print("Logging current train/loss to W&B")
                        log_items["train/loss"] = log_entry["loss"]
                        log_items["train_loss"] = log_entry["loss"]
                        break

            if not log_items:
                return

            # Add context (numeric only)
            log_items["fl_round"] = self.current_round
            wandb.log(log_items)
        except Exception as e:
            print(f"WandbMetricsLogger failed: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Mirror training loss to W&B under train/* keys for consistency."""
        if self.local_rank != 0 or not isinstance(logs, dict):
            return
        try:
            import wandb  # type: ignore

            if wandb.run is None:
                return
            log_items = {}
            if "loss" in logs and logs["loss"] is not None:
                print("Logging train/loss to W&B")
                log_items["train/loss"] = logs["loss"]
                log_items["train_loss"] = logs["loss"]
            if "accuracy" in logs and logs["accuracy"] is not None:
                print("Logging train/accuracy to W&B")
                log_items["train/accuracy"] = logs["accuracy"]
                log_items["train_accuracy"] = logs["accuracy"]
            if "learning_rate" in logs:
                log_items["learning_rate"] = logs["learning_rate"]
            if "epoch" in logs:
                log_items["epoch"] = logs["epoch"]
            if not log_items:
                return
            log_items["fl_round"] = self.current_round
            wandb.log(log_items)
        except Exception as e:
            print(f"WandbMetricsLogger on_log failed: {e}")


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


class TrainSubsetEvalCallback(TrainerCallback):
    """Evaluate a small subset of the training set to log train accuracy/loss.

    - Triggers at the end of each training epoch (local epoch in an FL round).
    - Uses the same compute_metrics/preprocess as evaluation.
    - Writes metrics with the `train_` prefix (Trainer adds it); WandbMetricsLogger mirrors to train/*.
    - Must run on all processes in DDP; only rank 0 prints/logs to console.
    """

    def __init__(self, train_dataset, sample_size: int = 256, local_rank: int = 0):
        self.local_rank = local_rank
        self.sample_size = max(1, int(sample_size))
        self.trainer = None
        try:
            n = len(train_dataset)
        except Exception:
            n = 0
        if n and hasattr(train_dataset, "select"):
            self.train_subset = train_dataset.select(range(min(self.sample_size, n)))
        else:
            self.train_subset = train_dataset
        self._in_progress = False

    def bind_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or self._in_progress:
            return
        try:
            self._in_progress = True
            # Check if we have a processed train dataset to work with
            if hasattr(self.trainer, "train_dataset") and hasattr(self.trainer.train_dataset, "select"):
                # Try to use the trainer's processed train dataset if available
                try:
                    # Create a small subset from the trainer's processed dataset
                    n_processed = len(self.trainer.train_dataset)
                    sample_size = min(self.sample_size, n_processed)
                    if sample_size > 0:
                        subset_indices = list(range(sample_size))
                        processed_subset = self.trainer.train_dataset.select(subset_indices)
                        metrics = self.trainer.evaluate(eval_dataset=processed_subset, metric_key_prefix="train")
                    else:
                        if self.local_rank == 0:
                            print("[train subset eval] No processed training data available")
                        return
                except Exception as subset_error:
                    if self.local_rank == 0:
                        print(f"[train subset eval] Could not create processed subset: {subset_error}")
                    return
            else:
                # Fallback: Skip evaluation if we can't access processed data
                if self.local_rank == 0:
                    print("[train subset eval] Skipping - processed training dataset not accessible")
                return

            if self.local_rank == 0:
                try:
                    ta = metrics.get("train_accuracy")
                    tl = metrics.get("train_loss")
                    print(f"[train subset eval] train_accuracy={ta} train_loss={tl}")
                except Exception:
                    print(f"[train subset eval] metrics: {metrics}")
        except Exception as e:
            if self.local_rank == 0:
                print(f"TrainSubsetEvalCallback failed: {e}")
        finally:
            self._in_progress = False


# ------------------------------
# Model/weights fingerprinting
# ------------------------------


def _tensor_sample_bytes(t: torch.Tensor, max_elems: int = 4096) -> bytes:
    """Return a stable byte sample from a tensor for hashing.

    - Moves to CPU, casts to float32 for dtype invariance.
    - Flattens and takes the first `max_elems` values (or all if smaller).
    - Returns raw bytes for hashing.
    """
    if not torch.is_tensor(t):
        return b""
    with torch.no_grad():
        flat = t.detach().to(dtype=torch.float32, device="cpu").view(-1)
        n = min(max_elems, flat.numel())
        if n == 0:
            return b""
        return flat[:n].numpy().tobytes()


def state_dict_fingerprint(
    sd: Dict[str, Any],
    sample_per_tensor: int = 4096,
    name_hint: str | None = None,
    include_names_in_hash: bool = False,
) -> Dict[str, Any]:
    """Compute a lightweight fingerprint for a (name -> Tensor) dict.

    Returns a dict with:
      - name: optional tag
      - sha256: first 16 hex chars of SHA256 over sampled bytes
      - num_tensors: count of tensor entries
      - num_elems: total number of elements across tensors
      - sample_keys: a few representative keys (first/last)
    """
    h = hashlib.sha256()
    num_tensors = 0
    num_elems = 0
    keys = sorted([k for k in sd.keys()])
    for k in keys:
        v = sd[k]
        if not torch.is_tensor(v):
            try:
                import numpy as _np

                if isinstance(v, _np.ndarray):
                    v = torch.from_numpy(v)
                else:
                    continue
            except Exception:
                continue
        num_tensors += 1
        try:
            num_elems += int(v.numel())
        except Exception:
            pass
        if include_names_in_hash:
            h.update(k.encode("utf-8"))
        h.update(_tensor_sample_bytes(v, max_elems=sample_per_tensor))
    sample_keys = keys[:2] + (keys[-2:] if len(keys) > 3 else [])
    return {
        "name": name_hint,
        "sha256": h.hexdigest()[:16],
        "num_tensors": num_tensors,
        "num_elems": int(num_elems),
        "sample_keys": sample_keys,
    }


def model_fingerprint(model: torch.nn.Module, peft_only: bool = False, sample_per_tensor: int = 4096) -> Dict[str, Any]:
    """Fingerprint a model's weights.

    - If `peft_only` is True and the model is a PEFT model, fingerprints only the PEFT adapter
      parameters via `get_peft_model_state_dict`.
    - Otherwise fingerprints the full `state_dict()`.
    """
    try:
        from peft import get_peft_model_state_dict
    except Exception:
        get_peft_model_state_dict = None

    if peft_only and get_peft_model_state_dict is not None:
        try:
            sd = get_peft_model_state_dict(model)
            return state_dict_fingerprint(sd, sample_per_tensor=sample_per_tensor, name_hint="peft_state")
        except Exception:
            # Fallback to full model if PEFT access fails
            pass

    sd = model.state_dict()
    return state_dict_fingerprint(sd, sample_per_tensor=sample_per_tensor, name_hint="full_state")
