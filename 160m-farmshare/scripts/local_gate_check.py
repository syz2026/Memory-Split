#!/usr/bin/env python
"""Preliminary gate-A check on a pulled snapshot, independent of the
cluster queue. Scores a subsample of the held-out reasoning sets locally.

Usage:
  python scripts/local_gate_check.py --snapshot outputs/pulled/step0001520.pt \
      --eval-dir outputs/pulled --limit 250 [--device auto]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from corpusgen.records import QAItem
from evals.scorers import score_items
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok
from train.trainer import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    state = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = get_tok()
    print(f"snapshot step {state['step']}, device {device}")

    for task in ("igsm", "deduction"):
        path = Path(args.eval_dir) / f"{task}.jsonl"
        items = [QAItem(**json.loads(l)) for l in open(path)][: args.limit]
        rows, _stats = score_items(
            model, tok, items, None, device,
            max_new=args.max_new, batch_size=args.batch_size,
        )
        acc = sum(r["correct"] for r in rows) / len(rows)
        n_parsed = sum(1 for r in rows if r["pred"] is not None)
        print(f"{task}: acc={acc:.4f} n={len(rows)} parsed={n_parsed}")
        if task == "igsm":
            by_op: dict[int, list[int]] = {}
            for r in rows:
                by_op.setdefault(r["meta"]["op"], []).append(int(r["correct"]))
            print("  by op:", {k: round(sum(v) / len(v), 3) for k, v in sorted(by_op.items())})


if __name__ == "__main__":
    main()
