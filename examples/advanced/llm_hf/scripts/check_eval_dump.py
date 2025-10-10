#!/usr/bin/env python3
"""
Quick checker for eval_dump.jsonl produced by EvalDumpRecordedCallback.

Compares predictions_choice (A–E) to labels_answer_choices (A–E) for each
evaluation entry and prints per-step accuracy and overall accuracy.

Usage:
  python3 scripts/check_eval_dump.py [path/to/eval_dump.jsonl]

If no path is provided, defaults to:
  workspace/hf_sft_multi/site-train_1/simulate_job/app_site-train_1/peft/eval_dump.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {ln}: {e}")


def compare(preds: Iterable[str], labels: Iterable[str]) -> Tuple[int, int]:
    valid = {"A", "B", "C", "D", "E"}
    corr = 0
    cnt = 0
    for p, l in zip(preds, labels):
        if p in valid and l in valid:
            cnt += 1
            if p == l:
                corr += 1
    return corr, cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="workspace/hf_sft_multi/site-train_2/simulate_job/app_site-train_2/peft/eval_dump.jsonl",
        help="Path to eval_dump.jsonl",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    total_corr = 0
    total_cnt = 0

    for rec in iter_jsonl(path):
        preds = rec.get("predictions_choice") or []
        labs = rec.get("labels_answer_choices") or []
        corr, cnt = compare(preds, labs)
        total_corr += corr
        total_cnt += cnt
        step = rec.get("step")
        epoch = rec.get("epoch")
        acc = (corr / cnt) if cnt else 0.0
        print(f"step={step}, epoch={epoch}, acc={acc:.3f} ({corr}/{cnt})")

    overall = (total_corr / total_cnt) if total_cnt else 0.0
    print(f"OVERALL acc={overall:.3f} ({total_corr}/{total_cnt})")


if __name__ == "__main__":
    main()
