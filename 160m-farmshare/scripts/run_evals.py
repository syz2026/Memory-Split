#!/usr/bin/env python
"""Run the eval battery for one trained run.

Usage:
  python scripts/run_evals.py --run outputs/d160m_split_n200k_s0 \
      [--ckpt snapshots/step0001234.pt] [--natural] [--limit 0]

Reads the run's config.yaml for arm/data paths; writes JSONLs into
<run>/evals/: igsm.jsonl, deduction.jsonl, factqa.jsonl, factqa_fresh.jsonl,
recall_{closed,on,off}.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen.records import QAItem
from evals.recall import bits_in_weights, recall_accuracy
from evals.scorers import save_results, score_items
from evals.stats import composite
from organizer.store import Organizer
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device

POOL_SIZES = {
    "birth_date": 27_759.0,  # days in 1930-01-01..2005-12-31
    "birth_city": 200.0,
    "university": 300.0,
    "major": 100.0,
    "employer": 263.0,
    "current_city": 200.0,
}


def load_items(path: Path) -> list[QAItem]:
    items = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            items.append(QAItem(**row))
    return items


def load_model(run_dir: Path, ckpt_rel: str | None, device: str) -> GPT:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model_cfg = (
        PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    )
    model = GPT(model_cfg)
    ckpt_path = run_dir / (ckpt_rel or "ckpt.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap items per task (0 = all)")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    arm = cfg.get("arm", "dense")
    data_dir = Path(cfg["data_dir"])
    device = pick_device("auto")
    tok = get_tok()
    model = load_model(run_dir, args.ckpt, device)
    organizer = Organizer.load(data_dir / "organizer.jsonl")
    fresh_path = data_dir / "organizer_fresh.jsonl"
    organizer_fresh = Organizer.load(fresh_path) if fresh_path.exists() else organizer

    out = run_dir / "evals"
    out.mkdir(exist_ok=True)
    summary: dict = {"run": run_dir.name, "arm": arm, "ckpt": args.ckpt or "ckpt.pt"}

    def cap(items):
        return items[: args.limit] if args.limit else items

    rows_by_task = {}
    for task, use_store in [
        ("igsm", False),
        ("deduction", False),
        ("factqa", arm == "split"),
        ("factqa_fresh", arm == "split"),
    ]:
        path = data_dir / "eval" / f"{task}.jsonl"
        if not path.exists():
            continue
        items = cap(load_items(path))
        store = organizer_fresh if task == "factqa_fresh" else organizer
        rows, stats = score_items(
            model, tok, items, store if use_store else None, device,
            batch_size=args.batch_size,
        )
        save_results(rows, out / f"{task}.jsonl")
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        summary[task] = {"acc": acc, "n": len(rows), "lookup_stats": stats}
        rows_by_task[task] = rows
        print(f"{task}: acc={acc:.4f} n={len(rows)} stats={stats}")

    probes = cap(load_items(data_dir / "eval" / "recall.jsonl"))
    recall = {}
    modes = [("closed", None)] if arm == "dense" else [("on", organizer), ("off", None)]
    for mode, org in modes:
        res = recall_accuracy(model, tok, probes, mode, org, device)
        recall[mode] = res
        print(f"recall[{mode}]: {res['overall']:.4f}")
    summary["recall"] = {m: r["overall"] for m, r in recall.items()}
    weight_mode = "closed" if arm == "dense" else "off"
    if weight_mode in recall:
        summary["bits_in_weights"] = bits_in_weights(
            recall[weight_mode]["per_attribute"], cfg["n_entities"], POOL_SIZES
        )
    with open(out / "recall.json", "w") as f:
        json.dump(recall, f, indent=2)

    if "igsm" in rows_by_task and "deduction" in rows_by_task:
        summary["composite_knowledge_free"] = composite(rows_by_task)

    if args.natural:
        from evals.natural import run_natural_suite

        summary["natural"] = run_natural_suite(model, tok, device)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
