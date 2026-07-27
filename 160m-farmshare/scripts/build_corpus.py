#!/usr/bin/env python
"""Build per-load corpora (both arms) + organizer + eval sets.

Usage:
  python scripts/build_corpus.py --out-root DATA_ROOT --stage gates|full|full1b \
      [--loads n50k,n200k,n800k] [--bed-file local_text.txt] [--total-tokens N]

Stages: gates = 0.8B tokens (160M pilots), full = 3.2B (160M sweep),
full1b = 10B tokens for the 1B confirmation — defaults to loads n800k,n4m
and writes into {load}_1b/ directories (larger bed, same generators).

Bed text: FineWeb-Edu sample-10BT streamed via HF datasets (deterministic
shard order), or --bed-file (one doc per paragraph split on blank lines)
for offline/smoke use.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

from corpusgen.build import LOADS, BuildCfg, build_corpus
from train.tokenizer import get_tok

STAGE_TOKENS = {
    "gates": 800_000_000,
    "full": 3_200_000_000,
    "full1b": 10_000_000_000,
    # exploratory overtraining tier: 80 tokens/param at 160M (4x Chinchilla)
    "over": 12_800_000_000,
}
STAGE_DEFAULT_LOADS = {
    "gates": "n50k,n200k,n800k",
    "full": "n50k,n200k,n800k",
    "full1b": "n800k,n4m",
    "over": "n200k",
}
# each stage owns its directories — gates and full builds may run
# concurrently and must never share output paths (learned the hard way)
STAGE_DIR_TAG = {"gates": "_gate", "full": "", "full1b": "_1b", "over": "_over"}


def bed_iter_hf():
    from datasets import load_dataset

    # token=False: fineweb-edu is public; a stale ambient HF token would
    # otherwise turn every request into a 401.
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        streaming=True, token=False,
    )
    for row in ds:
        yield row["text"]


def bed_iter_file(path: str):
    text = Path(path).read_text()
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    # cycle forever; builder stops at its token budget
    yield from itertools.cycle(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stage", default="full", choices=list(STAGE_TOKENS))
    ap.add_argument("--loads", default=None)
    ap.add_argument("--bed-file", default=None)
    ap.add_argument("--total-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()

    tok = get_tok()
    total = args.total_tokens or STAGE_TOKENS[args.stage]
    loads = args.loads or STAGE_DEFAULT_LOADS[args.stage]
    tag = STAGE_DIR_TAG[args.stage]
    for load in loads.split(","):
        # Each spawned worker regenerates the full record set at init;
        # at multi-million entities that is ~1-2 GB per worker, which
        # OOM-killed a 16-worker n4m build (job 1649440). Cap workers so
        # worker-resident records stay bounded.
        workers = args.workers
        if LOADS[load] >= 2_000_000:
            workers = min(workers, 6)
        out_dir = Path(args.out_root) / (load + tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = BuildCfg(
            n_entities=LOADS[load],
            total_tokens=total,
            seed=args.seed,
            workers=workers,
        )
        bed = bed_iter_file(args.bed_file) if args.bed_file else bed_iter_hf()
        report = build_corpus(cfg, tok, bed, out_dir)
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"{load}: done -> {out_dir}")
        print(json.dumps({k: v for k, v in report.items() if k != "bed_hashes"}, indent=2)[:800])


if __name__ == "__main__":
    main()
    # Force exit: HF datasets' multiprocess resource tracker can hang the
    # interpreter at shutdown (observed zombifying Slurm job 1647852 for 3h
    # after all work completed). Everything is flushed/closed by here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
