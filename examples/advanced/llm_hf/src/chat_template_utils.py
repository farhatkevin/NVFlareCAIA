"""
Shared chat formatting and scoring helpers used to align
training/eval behavior with chat inference without introducing
cross-module dependencies.

Functions:
- build_chat_input: construct a chat-formatted prompt using the tokenizer's
  chat template, mirroring chat_inference behavior.
- restricted_choice_logits_at_positions: read logits at specified positions
  and restrict to choices (A–E), returning probabilities or logits.
"""

from typing import Tuple

import torch


def build_chat_input(
    tokenizer,
    user_prompt: str,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Build a chat-formatted input string using the tokenizer's chat template.

    Mirrors chat_inference: apply_chat_template(messages, add_generation_prompt=True).
    Does not import chat_inference to avoid cross-dependencies.
    """
    default_system = z
    sys_prompt = system_prompt or default_system

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def restricted_choice_logits_at_positions(
    logits: torch.Tensor,
    tokenizer,
    positions: torch.Tensor,
    choices: Tuple[str, ...] = ("A", "B", "C", "D", "E"),
    return_probs: bool = True,
) -> torch.Tensor:
    """
    Compute logits/probabilities restricted to the provided choices (A–E) at
    specified positions for each sample.
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    B, S, V = logits.shape
    device = logits.device

    # Map choices to token ids (use last sub-token id for single-letter tokens)
    choice_ids = []
    for ch in choices:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        choice_ids.append(enc[-1])

    row_idx = torch.arange(B, device=device)
    pos = positions.to(device)
    selected = logits[row_idx, pos, :]  # [B, V]

    # Mask everything except the choice ids
    mask_vec = torch.full((V,), float("-inf"), device=device, dtype=selected.dtype)
    mask_vec[choice_ids] = 0
    masked = selected + mask_vec  # [B, V]

    # Gather columns corresponding to choices in given order
    choice_ids_t = torch.tensor(choice_ids, device=device)
    masked_choices = masked.index_select(dim=1, index=choice_ids_t)  # [B, C]

    if return_probs:
        return torch.softmax(masked_choices, dim=-1)
    else:
        return masked_choices
