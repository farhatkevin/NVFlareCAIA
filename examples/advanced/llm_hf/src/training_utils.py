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

def preprocess_logits_for_metrics(logits, labels):
      """
      Preprocess logits for multiple choice evaluation where answers are single letters A-E.
      
      Args:
          logits: Model logits of shape (batch_size, seq_len, vocab_size)
          labels: Ground truth labels of shape (batch_size, seq_len)
      
      Returns:
          Processed logits for metric calculation
      """
      import torch

      # Get tokenizer to find token IDs for A, B, C, D, E
      # Assuming these are the token IDs - you may need to adjust based on your tokenizer
      choice_tokens = {
          'A': 32, 'B': 33, 'C': 34, 'D': 35, 'E': 36  # Example token IDs
      }

      # Extract logits only for the choice tokens
      choice_token_ids = torch.tensor([32, 33, 34, 35, 36], device=logits.device)

      # Get logits for the last non-padding token (where the answer should be)
      batch_size = logits.shape[0]
      last_token_logits = []

      for i in range(batch_size):
          # Find the last non-padding token position
          non_pad_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
        #   print(f"Sample {i} non-pad positions: {non_pad_positions.tolist()}")
          if len(non_pad_positions) > 0:
              #TODO: this is assuming eos token before answer token
              last_pos = non_pad_positions[-2]
            #   print(f"Sample {i}: last non-pad position for prediction is {last_pos.item()}")
            #   print(labels[i, last_pos-3:last_pos+3])  # Show context around prediction
            #   last_token_logits.append(logits[i, last_pos, choice_token_ids])
              last_token_logits.append(logits[i, last_pos, choice_token_ids].argmax(dim=-1))
            #   print("last token logits: ", last_token_logits)
          else:
              # If no valid positions, use zeros
              print("failling")
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
    mapping = { 0:32, 1:33, 2:34, 3:35, 4:36 }
    for i in range(len(y_pred)):
        if y_pred[i] in mapping:
            y_pred[i] = mapping[y_pred[i]]


    # print("y_pred: ", y_pred)
    # Truncate to common length if needed (defensive)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return {"accuracy": 0.0}
    accuracy = float((y_pred[:m] == y_true[:m]).mean())
    
    if verbose:
        print(f"Evaluated {len(valid_idx)} samples, accuracy: {accuracy:.3f}")
    
    return {"accuracy": accuracy}


# def filter_by_length(example, tokenizer, format_instruction: Callable, max_seq_length: int):
#     """
#     Filter dataset examples that exceed the maximum sequence length.
    
#     Args:
#         example: Dataset example with 'prompt' and 'completion' fields
#         tokenizer: The tokenizer used for the model
#         format_instruction: Function to format examples into instruction format
#         max_seq_length: Maximum sequence length in tokens
        
#     Returns:
#         Boolean indicating whether to keep the example (True if under limit)
#     """
#     # Since format_instruction now returns a dict, manually concatenate for length check
#     full_text = example["prompt"] + "\n\n" + example["completion"]
#     tokens = tokenizer(full_text, truncation=False, return_tensors=None)
#     return len(tokens["input_ids"]) < max_seq_length - 1

def filter_by_length_messages(example, tokenizer, max_length):
    """Filter function for messages format"""
    try:
        # Apply chat template to get the full formatted text
        formatted_text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        tokens = tokenizer.encode(formatted_text)
        return len(tokens) <= max_length
    except Exception as e:
        # Log the error for debugging
        print(f"Error in filter_by_length_messages: {e}")
        print(f"Example keys: {example.keys() if hasattr(example, 'keys') else 'Not a dict'}")
        if "messages" in example:
            print(f"Messages type: {type(example['messages'])}")
        return False

