#!/usr/bin/env python3
"""Run or finalize one arm of a supervised paired Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.corpus_contract import sha256_file  # noqa: E402
from msctl.adapters.slurm import load_pair_manifest  # noqa: E402
from msctl.manifest import write_json_no_replace  # noqa: E402
from msctl.profile import load_profile  # noqa: E402


STEP_LIMITS = {
    "functional": 1,
    "resume": 2,
    "throughput": 100,
    "protected": None,
}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_config(
    pair: dict,
    arm: str,
    *,
    mode: str,
    evidence_root: Path,
) -> tuple[Path, dict]:
    record = next(item for item in pair["arms"] if item["arm"] == arm)
    source = Path(record["runtime_config"])
    if (
        not source.is_file()
        or source.is_symlink()
        or sha256_file(source) != record["runtime_config_sha256"]
    ):
        raise ValueError(f"{arm} runtime config is missing, unsafe, or mismatched")
    cfg = yaml.safe_load(source.read_text())
    if not isinstance(cfg, dict) or cfg.get("run_id") != record["run_id"]:
        raise ValueError(f"{arm} runtime config identity differs")
    limit = STEP_LIMITS[mode]
    if limit is not None:
        cfg["max_steps"] = limit
        cfg["out_dir"] = str(
            evidence_root / "canaries" / mode / record["run_id"]
        )
    destination = evidence_root / "runtime" / f"{record['run_id']}-{mode}.yaml"
    data = yaml.safe_dump(cfg, sort_keys=False)
    if destination.exists():
        if destination.is_symlink() or destination.read_text() != data:
            raise ValueError("existing launch config differs from frozen runtime config")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
        try:
            temporary.write_text(data)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination, cfg


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "unavailable"


def _checkpoint_step(path: Path) -> int | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        return int(state["step"])
    except Exception:
        return None


def _throughput(path: Path) -> float | None:
    if not path.is_file() or path.is_symlink():
        return None
    values = []
    try:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            value = row.get("tok_s")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return sum(values) / len(values) if values else None


def _checkpoint_record(record: dict) -> dict:
    runtime_path = Path(record["runtime_config"])
    if (
        not runtime_path.is_file()
        or runtime_path.is_symlink()
        or sha256_file(runtime_path) != record["runtime_config_sha256"]
    ):
        raise ValueError(
            f"{record['arm']} runtime config is missing, unsafe, or mismatched"
        )
    cfg = yaml.safe_load(runtime_path.read_text())
    if (
        not isinstance(cfg, dict)
        or cfg.get("arm") != record["arm"]
        or cfg.get("run_id") != record["run_id"]
    ):
        raise ValueError(f"{record['arm']} runtime config identity differs")
    expected_step = cfg.get("max_steps")
    if (
        isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step <= 0
    ):
        raise ValueError(f"{record['arm']} runtime config has no terminal update")

    checkpoint = Path(record["out_dir"]) / "ckpt.pt"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError(f"{record['arm']} terminal checkpoint is missing or unsafe")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("step") != expected_step:
        raise ValueError(
            f"{record['arm']} checkpoint is not at terminal update {expected_step}"
        )
    data = state.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"{record['arm']} checkpoint has no data cursor")
    cursor = data.get("cursor")
    epoch = data.get("epoch", 0)
    if (
        isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError(f"{record['arm']} checkpoint data cursor is invalid")
    checkpoint_cfg = state.get("cfg")
    if (
        not isinstance(checkpoint_cfg, dict)
        or checkpoint_cfg.get("arm") != record["arm"]
        or checkpoint_cfg.get("run_id") != record["run_id"]
    ):
        raise ValueError(f"{record['arm']} checkpoint config identity differs")
    del state
    return {
        "arm": record["arm"],
        "bytes": checkpoint.stat().st_size,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cursor": cursor,
        "epoch": epoch,
        "run_id": record["run_id"],
        "step": expected_step,
    }


def _write_pair_checkpoint_receipt(
    pair: dict,
    *,
    evidence_root: Path,
) -> Path:
    checkpoints = {
        record["arm"]: _checkpoint_record(record)
        for record in pair["arms"]
    }
    dense = checkpoints["dense"]
    split90 = checkpoints["split90"]
    for field in ("step", "cursor", "epoch"):
        if dense[field] != split90[field]:
            raise ValueError(f"paired checkpoints do not share the same {field}")
    receipt = {
        "checkpoints": checkpoints,
        "cohort_id": pair["cohort_id"],
        "dataset": pair["dataset"],
        "pair_id": pair["pair_id"],
        "profile_sha256": pair["profile_sha256"],
        "provider": pair["provider"],
        "schema_version": 1,
        "seed": pair["seed"],
        "terminal_cursor": dense["cursor"],
        "terminal_epoch": dense["epoch"],
        "terminal_step": dense["step"],
    }
    return write_json_no_replace(
        evidence_root / f"{pair['pair_id']}-pair-checkpoint-receipt.json",
        receipt,
    )


def run_arm(
    manifest_path: Path,
    profile_path: Path,
    *,
    arm: str,
    mode: str,
    evidence_root: Path,
) -> int:
    pair = load_pair_manifest(manifest_path)
    profile = load_profile(profile_path)
    if pair["profile_sha256"] != profile.sha256:
        raise ValueError("pair/profile hash mismatch")
    runtime_path, cfg = _runtime_config(
        pair,
        arm,
        mode=mode,
        evidence_root=evidence_root,
    )
    record = next(item for item in pair["arms"] if item["arm"] == arm)
    log_path = evidence_root / "logs" / f"{pair['pair_id']}-{arm}-{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_train.py"),
        "--config",
        str(runtime_path),
        "--resume",
        "none" if mode == "resume" else "auto",
    ]
    stages = [command]
    if mode == "resume":
        first_cfg = dict(cfg)
        first_cfg["max_steps"] = 1
        first_path = runtime_path.with_name(f"{runtime_path.stem}-first.yaml")
        first_path.write_text(yaml.safe_dump(first_cfg, sort_keys=False))
        stages = [
            [
                sys.executable,
                str(ROOT / "scripts" / "run_train.py"),
                "--config",
                str(first_path),
                "--resume",
                "none",
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "run_train.py"),
                "--config",
                str(runtime_path),
                "--resume",
                "auto",
            ],
        ]
    returncode = 0
    with log_path.open("w") as log:
        for stage in stages:
            completed = subprocess.run(
                stage,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = int(completed.returncode)
            if returncode:
                break
    output = Path(cfg["out_dir"])
    text = log_path.read_text(errors="replace")
    checkpoint = output / "ckpt.pt"
    step = _checkpoint_step(checkpoint)
    expected = STEP_LIMITS[mode] or int(cfg["max_steps"])
    gpu_name = _gpu_name()
    gpu_supported = bool(re.search(profile.gpu_name_regex, gpu_name))
    evidence = {
        "arm": arm,
        "checkpoint_present": checkpoint.is_file() and not checkpoint.is_symlink(),
        "config_sha256": record["config_sha256"],
        "gpu_name": gpu_name,
        "gpu_supported": gpu_supported,
        "mode": mode,
        "oom_detected": "out of memory" in text.lower(),
        "pair_id": pair["pair_id"],
        "profile_sha256": profile.sha256,
        "resume_exact": mode != "resume" or len(stages) == 2 and step == 2,
        "returncode": returncode,
        "runtime_config": str(runtime_path),
        "runtime_config_sha256": sha256_file(runtime_path),
        "schema_version": 1,
        "status": (
            "completed"
            if returncode == 0
            and step == expected
            and "out of memory" not in text.lower()
            and gpu_supported
            else "failed"
        ),
        "step": step,
        "tokens_per_second": _throughput(output / "log.jsonl"),
    }
    _atomic_json(
        evidence_root / "arms" / f"{pair['pair_id']}-{arm}-{mode}.json",
        evidence,
    )
    return 0 if evidence["status"] == "completed" else 1


def finalize(
    manifest_path: Path,
    *,
    mode: str,
    evidence_root: Path,
    dense_returncode: int,
    split_returncode: int,
) -> int:
    pair = load_pair_manifest(manifest_path)
    arms = {}
    for arm in ("dense", "split90"):
        path = evidence_root / "arms" / f"{pair['pair_id']}-{arm}-{mode}.json"
        if not path.is_file() or path.is_symlink():
            arms[arm] = {"status": "missing"}
        else:
            arms[arm] = json.loads(path.read_text())
    names = [record.get("gpu_name") for record in arms.values()]
    passed = (
        dense_returncode == 0
        and split_returncode == 0
        and all(record.get("status") == "completed" for record in arms.values())
        and len(set(names)) == 1
        and names[0] not in (None, "unavailable")
    )
    pair_checkpoint = None
    if passed and mode == "protected":
        pair_checkpoint = _write_pair_checkpoint_receipt(
            pair,
            evidence_root=evidence_root,
        )
    evidence = {
        "arms": arms,
        "dataset": pair["dataset"],
        "environment": {
            "python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "gpu_model_match": len(set(names)) == 1,
        "mode": mode,
        "pair_id": pair["pair_id"],
        "profile_sha256": pair["profile_sha256"],
        "schema_version": 1,
        "seed": pair["seed"],
        "status": "completed" if passed else "failed",
    }
    if pair_checkpoint is not None:
        evidence["pair_checkpoint_receipt"] = {
            "path": str(pair_checkpoint.resolve()),
            "sha256": sha256_file(pair_checkpoint),
        }
    name = (
        f"{pair['pair_id']}-train-evidence.json"
        if mode == "protected"
        else f"{pair['pair_id']}-{mode}-train-evidence.json"
    )
    _atomic_json(evidence_root / name, evidence)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--mode", choices=tuple(STEP_LIMITS), required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--arm", choices=("dense", "split90"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--dense-returncode", type=int, default=1)
    parser.add_argument("--split-returncode", type=int, default=1)
    args = parser.parse_args(argv)
    if args.finalize:
        return finalize(
            Path(args.manifest),
            mode=args.mode,
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
        mode=args.mode,
        evidence_root=Path(args.evidence_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
