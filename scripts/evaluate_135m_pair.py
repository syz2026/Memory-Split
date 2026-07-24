#!/usr/bin/env python3
"""Evaluate or finalize one paired 135M Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.corpus_contract import sha256_file
from msctl.adapters.slurm import load_pair_manifest
from msctl.profile import load_profile
from msctl.reasoning_cohort import DATASET_CONTRACT_ID as REASONING_V3


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return "unavailable"
    return "unavailable"


def run_arm(
    manifest_path: Path,
    profile_path: Path,
    *,
    arm: str,
    evidence_root: Path,
) -> int:
    pair = load_pair_manifest(manifest_path)
    profile = load_profile(profile_path)
    if pair["profile_sha256"] != profile.sha256:
        raise ValueError("pair/profile hash mismatch")
    record = next(item for item in pair["arms"] if item["arm"] == arm)
    run_dir = Path(record["out_dir"])
    checkpoint = run_dir / "ckpt.pt"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError(f"{arm} checkpoint is missing or unsafe")
    runtime_config = Path(record["runtime_config"])
    if (
        not runtime_config.is_file()
        or runtime_config.is_symlink()
        or sha256_file(runtime_config) != record["runtime_config_sha256"]
    ):
        raise ValueError(f"{arm} runtime config is missing, unsafe, or mismatched")
    cfg = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    dataset = cfg.get("dataset") if isinstance(cfg, dict) else None
    evaluation_script = (
        "evaluate_reasoning_v3_run.py"
        if isinstance(dataset, dict)
        and dataset.get("contract_id") == REASONING_V3
        else "run_evals.py"
    )
    log = evidence_root / "logs" / f"{pair['pair_id']}-{arm}-evaluate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / evaluation_script),
                "--run",
                str(run_dir),
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary = run_dir / "evals" / "summary.json"
    summary_value = (
        json.loads(summary.read_text(encoding="utf-8"))
        if result.returncode == 0 and summary.is_file() and not summary.is_symlink()
        else {}
    )
    evidence = {
        "arm": arm,
        "evaluation_scope": summary_value.get("evaluation_scope", "scientific"),
        "gpu_name": _gpu_name(),
        "pair_id": pair["pair_id"],
        "profile_sha256": profile.sha256,
        "returncode": int(result.returncode),
        "schema_version": 1,
        "status": (
            "completed"
            if result.returncode == 0 and summary.is_file() and not summary.is_symlink()
            else "failed"
        ),
        "summary": str(summary),
    }
    _atomic_json(
        evidence_root / "arms" / f"{pair['pair_id']}-{arm}-evaluate.json",
        evidence,
    )
    return 0 if evidence["status"] == "completed" else 1


def finalize(
    manifest_path: Path,
    *,
    evidence_root: Path,
    dense_returncode: int,
    split_returncode: int,
) -> int:
    pair = load_pair_manifest(manifest_path)
    arms = {}
    for arm in ("dense", "split90"):
        path = evidence_root / "arms" / f"{pair['pair_id']}-{arm}-evaluate.json"
        arms[arm] = (
            json.loads(path.read_text())
            if path.is_file() and not path.is_symlink()
            else {"status": "missing"}
        )
    passed = (
        dense_returncode == 0
        and split_returncode == 0
        and all(item.get("status") == "completed" for item in arms.values())
    )
    _atomic_json(
        evidence_root / f"{pair['pair_id']}-evaluate-evidence.json",
        {
            "arms": arms,
            "dataset": pair["dataset"],
            "pair_id": pair["pair_id"],
            "profile_sha256": pair["profile_sha256"],
            "schema_version": 1,
            "seed": pair["seed"],
            "status": "completed" if passed else "failed",
        },
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--arm", choices=("dense", "split90"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--dense-returncode", type=int, default=1)
    parser.add_argument("--split-returncode", type=int, default=1)
    args = parser.parse_args(argv)
    if args.finalize:
        return finalize(
            Path(args.manifest),
            evidence_root=Path(args.evidence_root),
            dense_returncode=args.dense_returncode,
            split_returncode=args.split_returncode,
        )
    if args.profile is None or args.arm is None:
        parser.error("--profile and --arm are required unless --finalize is used")
    return run_arm(
        Path(args.manifest),
        Path(args.profile),
        arm=args.arm,
        evidence_root=Path(args.evidence_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
