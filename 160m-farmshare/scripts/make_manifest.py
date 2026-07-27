#!/usr/bin/env python
"""Generate run configs + a manifest TSV for a battery stage.

Stages (schedule option B, decided 2026-07-18):
  gates   - 160M short-budget pilots for gates A-C
  sweep   - 160M dose-response: 3 loads x 2 arms x 2 seeds
  confirm - 1B confirmation: top load x 2 arms x 2 seeds (submit each
            config via cluster/submit_chain.sh — runs outlast the 2-day wall)
  mid410  - OPTIONAL 410M tier (2 arms x 3 seeds), add-back if calendar allows

Usage:
  python scripts/make_manifest.py --stage gates|sweep|confirm|mid410 \
      --data-root /scratch/users/syz/memorysplit_data [--top-load n800k]

Writes configs/gen/{stage}_{arm}_{load}_s{seed}.yaml and
outputs/manifests/{stage}.tsv (one config path per line).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from corpusgen.build import LOADS

SWEEP_LOADS = ("n50k", "n200k", "n800k")

SCALE = {
    # preset: (total_tokens, tokens/step, micro_bs, lr, warmup)
    "d160m": (3_200_000_000, 524_288, 16, 1.5e-3, 300),
    "d410m": (8_000_000_000, 524_288, 8, 1.0e-3, 300),
    "d1b": (10_000_000_000, 524_288, 4, 6.0e-4, 300),
}

GATE_TOKENS = 800_000_000  # short-budget pilots for gates A-C


def make_cfg(preset, arm, load, seed, data_root, out_root, total_tokens=None,
             data_tag="", run_tag=None, snap_frac=0.10):
    tokens, tps, mbs, lr, warmup = SCALE[preset]
    if total_tokens is not None:
        tokens = total_tokens
    if run_tag is None:
        run_tag = "" if total_tokens is None else "_gate"
    run_id = f"{preset}_{arm}_{load}_s{seed}" + run_tag
    data_dir = Path(data_root) / (load + data_tag)
    return run_id, {
        "run_id": run_id,
        "model": preset,
        "arm": arm,
        "load": load,
        "n_entities": LOADS[load],
        "train_bin": str(data_dir / arm / "train.bin"),
        "train_mask": str(data_dir / arm / "train.mask.bin") if arm == "split" else None,
        "data_dir": str(data_dir),
        "total_tokens": tokens,
        "tokens_per_step": tps,
        "micro_batch_size": mbs,
        "lr": lr,
        "warmup_steps": warmup,
        "weight_decay": 0.1,
        "seed": seed,
        "compile": True,
        "device": "auto",
        "out_dir": str(Path(out_root) / run_id),
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": snap_frac,
        "ckpt_minutes": 30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gates", "sweep", "calib1b", "confirm", "mid410",
                             "overtrain"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--top-load", default="n800k", choices=list(LOADS))
    ap.add_argument("--gate-tokens", type=int, default=GATE_TOKENS)
    ap.add_argument("--gate-loads", default=None,
                    help="comma list; default all sweep loads + split n200k")
    args = ap.parse_args()

    jobs: list[tuple[str, dict]] = []
    if args.stage == "gates":
        gate_loads = (args.gate_loads.split(",") if args.gate_loads
                      else list(SWEEP_LOADS) + ["split:n200k"])
        for spec_load in gate_loads:
            arm, load = (spec_load.split(":") if ":" in spec_load
                         else ("dense", spec_load))
            jobs.append(make_cfg("d160m", arm, load, 0, args.data_root,
                                 args.out_root, args.gate_tokens, data_tag="_gate"))
    elif args.stage == "sweep":
        for load in SWEEP_LOADS:
            for arm in ("dense", "split"):
                for seed in (0, 1):
                    jobs.append(make_cfg("d160m", arm, load, seed, args.data_root, args.out_root))
    elif args.stage == "calib1b":
        # gate B at scale: short dense 1B runs on the two candidate doses
        # (1.5B tokens ~ 20 L40S-h each) to pick the load that binds at 1B
        for load in ("n800k", "n4m"):
            jobs.append(make_cfg("d1b", "dense", load, 0, args.data_root,
                                 args.out_root, 1_500_000_000, data_tag="_1b"))
    elif args.stage == "confirm":
        for arm in ("dense", "split"):
            for seed in (0, 1):
                jobs.append(make_cfg("d1b", arm, args.top_load, seed,
                                     args.data_root, args.out_root, data_tag="_1b"))
    elif args.stage == "mid410":
        for arm in ("dense", "split"):
            for seed in (0, 1, 2):
                jobs.append(make_cfg("d410m", arm, args.top_load, seed, args.data_root, args.out_root))
    elif args.stage == "overtrain":
        # EXPLORATORY tier (outside the frozen preregistration): one 160M
        # pair at 80 tokens/param (12.8B tokens, 4x Chinchilla) on the
        # n200k dose, dense snapshots (every 5%) so per-checkpoint evals
        # trace recall / bits-in-weights / fact-use through the
        # overtraining trajectory. See specs/2026-07-21-overtrain-exploratory.md.
        for arm in ("dense", "split"):
            jobs.append(make_cfg("d160m", arm, "n200k", 0, args.data_root,
                                 args.out_root, total_tokens=12_800_000_000,
                                 data_tag="_over", run_tag="_over",
                                 snap_frac=0.05))

    gen_dir = Path("configs/gen")
    gen_dir.mkdir(parents=True, exist_ok=True)
    man_dir = Path("outputs/manifests")
    man_dir.mkdir(parents=True, exist_ok=True)
    manifest = man_dir / f"{args.stage}.tsv"
    with open(manifest, "w") as mf:
        for run_id, cfg in jobs:
            cfg_path = gen_dir / f"{run_id}.yaml"
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            mf.write(str(cfg_path) + "\n")
    print(f"{len(jobs)} configs -> {manifest}")


if __name__ == "__main__":
    main()
