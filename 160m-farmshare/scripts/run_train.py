#!/usr/bin/env python
"""Train one arm from a YAML config.

Usage: python scripts/train.py --config configs/foo.yaml [--resume auto|none]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from train.trainer import train


def _runtime_root(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required for a relative run config")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a canonical absolute path")
    if path.is_symlink() or path.resolve(strict=False) != path:
        raise ValueError(f"{name} cannot contain a symlink")
    return path


def resolve_relative_config(
    raw: Mapping,
    *,
    run_manifest,
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Resolve a validated Task-9 config at the sole training boundary."""

    from scripts.make_relational_manifest import (
        RunConfig,
        RunManifest,
        require_launchable,
    )

    config = RunConfig.from_dict(raw)
    manifest = (
        RunManifest.from_dict(run_manifest.to_dict())
        if isinstance(run_manifest, RunManifest)
        else RunManifest.from_dict(run_manifest)
    )
    require_launchable(manifest)
    expected = [
        run for run in manifest.runs if run.run_id == config.run_id
    ]
    if (
        len(expected) != 1
        or expected[0].to_dict() != config.to_dict()
    ):
        raise ValueError(
            "run config is not an exact member of the launchable run manifest"
        )
    environment = os.environ if environ is None else environ
    data_root = _runtime_root(environment.get("DATA_ROOT"), "DATA_ROOT")
    out_root = _runtime_root(environment.get("OUT_ROOT"), "OUT_ROOT")
    train_bin = data_root / config.data_rel
    train_weights = data_root / config.weights_rel
    out_dir = out_root / config.out_rel
    if (
        not train_bin.is_relative_to(data_root)
        or not train_weights.is_relative_to(data_root)
        or not out_dir.is_relative_to(out_root)
    ):
        raise ValueError("relative config path escaped its runtime root")
    resolved = dict(config.to_dict())
    resolved.update(
        train_bin=str(train_bin),
        train_weights=str(train_weights),
        out_dir=str(out_dir),
        data_root=str(data_root),
        out_root=str(out_root),
        ledger_root=str(out_root),
        source_tree_sha256=(
            manifest.freeze.source_provenance.source_tree_sha256
        ),
        model=config.model,
        condition=config.condition,
        seed=config.seed,
        micro_batch_size=8,
        tokens_per_step=config.tokens_per_step,
        max_steps=config.steps,
        total_tokens=config.actual_raw_positions,
        lr=config.optimizer["lr"],
        weight_decay=config.optimizer["weight_decay"],
        warmup_steps=config.scheduler["warmup_steps"],
        ctx=config.architecture["ctx"],
    )
    return resolved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--run-manifest",
        help="required freeze-bound manifest for relative Task-9 configs",
    )
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    args = ap.parse_args(argv)
    try:
        with open(args.config) as stream:
            cfg = yaml.safe_load(stream)
        if isinstance(cfg, Mapping) and {
            "data_rel",
            "weights_rel",
            "out_rel",
        } <= set(cfg):
            if args.run_manifest is None:
                raise ValueError(
                    "--run-manifest is required for relative Task-9 configs"
                )
            from experiment.provenance import verify_source_provenance
            from scripts.make_relational_manifest import (
                load_run_manifest,
                require_launchable,
            )

            manifest = require_launchable(
                load_run_manifest(args.run_manifest)
            )
            verify_source_provenance(
                Path(__file__).resolve().parents[1],
                manifest.freeze.source_provenance,
                require_clean=True,
            )
            cfg = resolve_relative_config(
                cfg,
                run_manifest=manifest,
            )
        trainer = train(cfg, resume=args.resume)
    except Exception as exc:
        print(
            f"training failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"done: step={trainer.step} out={trainer.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
