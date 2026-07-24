#!/usr/bin/env python3
"""Generate or verify the frozen exploratory 135M reasoning-v3 AWS matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.reasoning_cohort import (
    ARMS,
    CHECKPOINT_UPDATES,
    COHORT_ID,
    DATASET_CONTRACT_ID,
    MODEL_PARAMETERS,
    PACKED_TARGETS,
    POINTER_PATH,
    PROVIDER,
    RAW_TARGETS,
    ROLE,
    SCIENTIFIC_SCOPE,
    SEEDS,
    TARGET_WEIGHTS,
    TARGETS_PER_UPDATE,
    TERMINAL_UPDATES,
    canonical_assignment,
    config_path,
    pair_id,
    run_id,
)


def config_document(arm: str, seed: int) -> dict:
    return {
        "schema_version": 1,
        "cohort_id": COHORT_ID,
        "run_id": run_id(arm, seed),
        "pair_id": pair_id(seed),
        "model": "d135m",
        "model_parameters": MODEL_PARAMETERS,
        "arm": arm,
        "seed": seed,
        "operator": ROLE,
        "provider": PROVIDER,
        "ctx": 1024,
        "train_bin": list(PACKED_TARGETS),
        "train_mask": list(TARGET_WEIGHTS[arm]),
        "total_tokens": RAW_TARGETS,
        "tokens_per_step": TARGETS_PER_UPDATE,
        "max_steps": TERMINAL_UPDATES,
        "micro_batch_size": 8,
        "lr": 0.0015,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "out_dir": f"outputs/135m-v3/{run_id(arm, seed)}",
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": 0.1,
        "ckpt_minutes": 30,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "dataset": {
            "contract_id": DATASET_CONTRACT_ID,
            "pointer": POINTER_PATH,
            "complete_dataset": True,
            "packed_targets": list(PACKED_TARGETS),
            "target_weights": list(TARGET_WEIGHTS[arm]),
            "raw_target_tokens": RAW_TARGETS,
            "scientific_scope": SCIENTIFIC_SCOPE,
        },
    }


def expected_files() -> dict[str, str]:
    files = {
        "configs/cohort-assignment-135m-v3-aws-n10.json": (
            json.dumps(canonical_assignment(), indent=2, sort_keys=True) + "\n"
        )
    }
    for seed in SEEDS:
        for arm in ARMS:
            files[config_path(arm, seed)] = yaml.safe_dump(
                config_document(arm, seed),
                sort_keys=False,
            )
    return files


def generate(repository_root: Path, *, write: bool) -> list[str]:
    differences = []
    for relative, expected in expected_files().items():
        path = repository_root / relative
        current = (
            path.read_text(encoding="utf-8")
            if path.is_file() and not path.is_symlink()
            else None
        )
        if current == expected:
            continue
        differences.append(relative)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    differences = generate(Path(args.repository_root).resolve(), write=args.write)
    report = {
        "differences": differences,
        "mode": "write" if args.write else "check",
        "ok": not differences or args.write,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
