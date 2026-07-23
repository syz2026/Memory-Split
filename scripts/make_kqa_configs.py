#!/usr/bin/env python
"""Create dense/split continued-training configs from existing run folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml


_PAIRED_SOURCE_KEYS = (
    "model",
    "ctx",
    "seed",
    "total_tokens",
    "tokens_per_step",
    "micro_batch_size",
    "lr",
    "warmup_steps",
    "weight_decay",
    "precision",
    "max_steps",
    "load",
    "n_entities",
)


def _source_run(path: str) -> tuple[Path, Path, dict]:
    supplied = Path(path).expanduser().resolve()
    if supplied.is_dir():
        run_dir = supplied
        checkpoint = (
            run_dir / "model.pt"
            if (run_dir / "model.pt").exists()
            else run_dir / "ckpt.pt"
        )
    else:
        checkpoint = supplied
        run_dir = checkpoint.parent
    config_path = run_dir / "config.yaml"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"source config not found: {config_path}; provide a full run directory"
        )
    with open(config_path) as handle:
        return run_dir, checkpoint, yaml.safe_load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(checkpoint: Path, source_cfg: dict) -> dict:
    try:
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "step" not in state:
        raise ValueError(
            f"source checkpoint lacks required training-step metadata: {checkpoint}"
        )
    step = int(state["step"])
    saved_cfg = state.get("cfg")
    if isinstance(saved_cfg, dict):
        stale = [
            key
            for key in (*_PAIRED_SOURCE_KEYS, "arm")
            if key in saved_cfg
            and key in source_cfg
            and saved_cfg[key] != source_cfg[key]
        ]
        if stale:
            raise ValueError(
                f"source config does not match checkpoint {checkpoint.name}: "
                + ", ".join(stale)
            )
    del state
    return {
        "path": str(checkpoint),
        "sha256": _sha256_file(checkpoint),
        "step": step,
    }


def _git_identity(repo_root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    if not status:
        return commit
    digest = hashlib.sha256(status.encode())
    digest.update(
        subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repo_root
        )
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
    )
    for raw_path in filter(None, untracked.split(b"\0")):
        path = repo_root / raw_path.decode()
        digest.update(raw_path)
        if path.is_file():
            digest.update(path.read_bytes())
    return f"{commit}-dirty-{digest.hexdigest()[:12]}"


def _resolve_precision(precision: str) -> str:
    if precision != "auto":
        return precision
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def validate_source_pair(dense_path: str, split_path: str) -> dict:
    _, dense_checkpoint, dense = _source_run(dense_path)
    _, split_checkpoint, split = _source_run(split_path)
    if dense.get("arm") != "dense" or split.get("arm") != "split":
        raise ValueError(
            "source configs must identify a dense arm and a split arm respectively"
        )
    mismatches = [
        key for key in _PAIRED_SOURCE_KEYS if dense.get(key) != split.get(key)
    ]
    if mismatches:
        raise ValueError(
            "source runs are not a matched pair; mismatched config fields: "
            + ", ".join(mismatches)
        )
    identities = {
        "dense": _checkpoint_identity(dense_checkpoint, dense),
        "split": _checkpoint_identity(split_checkpoint, split),
    }
    if identities["dense"]["step"] != identities["split"]["step"]:
        raise ValueError(
            "source runs are not a matched pair; checkpoint steps differ: "
            f"{identities['dense']['step']} vs {identities['split']['step']}"
        )
    pair_payload = {
        "shared_config": {key: dense.get(key) for key in _PAIRED_SOURCE_KEYS},
        "dense": {
            "sha256": identities["dense"]["sha256"],
            "step": identities["dense"]["step"],
        },
        "split": {
            "sha256": identities["split"]["sha256"],
            "step": identities["split"]["step"],
        },
    }
    pair_id = hashlib.sha256(
        json.dumps(pair_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return {"pair_id": pair_id, "arms": identities}


def make_continuation_config(
    source_path: str,
    arm: str,
    corpus_dir: str | Path,
    out_root: str | Path,
    continuation_tokens: int,
    *,
    lr: float | None = None,
    warmup_steps: int | None = None,
    tokens_per_step: int | None = None,
    micro_batch_size: int | None = None,
    precision: str = "auto",
    source_identity: dict | None = None,
    source_pair_id: str | None = None,
) -> dict:
    source_dir, checkpoint, source = _source_run(source_path)
    source_identity = source_identity or _checkpoint_identity(checkpoint, source)
    if Path(source_identity["path"]).resolve() != checkpoint:
        raise ValueError("source identity does not match the selected checkpoint")
    corpus_dir = Path(corpus_dir).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    report_path = corpus_dir / "continuation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"continuation report not found: {report_path}")
    with open(report_path) as handle:
        corpus_report = json.load(handle)
    data_fingerprint = corpus_report["data_fingerprint"]
    artifact_hashes = corpus_report["arms"][arm]["artifact_sha256"]
    train_sha256 = artifact_hashes["train_bin"]
    train_mask_sha256 = (
        artifact_hashes["train_mask"] if arm == "split" else None
    )
    repo_root = Path(__file__).resolve().parents[1]
    code_commit = _git_identity(repo_root)
    source_pair_id = source_pair_id or hashlib.sha256(
        f"{source_identity['sha256']}:{source_identity['step']}".encode()
    ).hexdigest()
    run_prefix = (
        f"{source_dir.name}_kqa_{arm}_s{source_identity['sha256'][:8]}_"
        f"{data_fingerprint[:8]}_"
        f"t{continuation_tokens}"
    )
    config = dict(source)
    config.update(
        {
            "run_id": "",
            "arm": arm,
            "train_bin": str(corpus_dir / arm / "train.bin"),
            "train_mask": (
                str(corpus_dir / arm / "train.mask.bin") if arm == "split" else None
            ),
            "data_dir": str(corpus_dir),
            "out_dir": "",
            "total_tokens": continuation_tokens,
            "init_from": str(checkpoint),
            "source_run": str(source_dir),
            "source_checkpoint_sha256": source_identity["sha256"],
            "source_checkpoint_step": source_identity["step"],
            "source_pair_id": source_pair_id,
            "data_fingerprint": data_fingerprint,
            "train_sha256": train_sha256,
            "train_mask_sha256": train_mask_sha256,
            "code_commit": code_commit,
            "compile": False,
            "precision": _resolve_precision(precision),
        }
    )
    for stale in ("max_steps",):
        config.pop(stale, None)
    if lr is not None:
        config["lr"] = lr
    if tokens_per_step is not None:
        config["tokens_per_step"] = tokens_per_step
    if micro_batch_size is not None:
        config["micro_batch_size"] = micro_batch_size
    max_steps = max(1, continuation_tokens // config["tokens_per_step"])
    if warmup_steps is not None:
        config["warmup_steps"] = warmup_steps
    else:
        config["warmup_steps"] = min(
            config.get("warmup_steps", 300), max(1, max_steps // 10)
        )
    context = config.get(
        "ctx",
        config["model"].get("ctx", 2048)
        if isinstance(config["model"], dict)
        else 2048,
    )
    accumulation = max(
        1,
        config["tokens_per_step"] // (config["micro_batch_size"] * context),
    )
    effective_tokens_per_step = accumulation * config["micro_batch_size"] * context
    shard_tokens = max(
        corpus_report["arms"]["dense"]["total_tokens"],
        corpus_report["arms"]["split"]["total_tokens"],
    )
    config["max_steps"] = (
        shard_tokens + effective_tokens_per_step - 1
    ) // effective_tokens_per_step
    config["planned_consumed_tokens"] = (
        config["max_steps"] * effective_tokens_per_step
    )
    identity_keys = (
        "arm",
        "source_checkpoint_sha256",
        "source_checkpoint_step",
        "source_pair_id",
        "data_fingerprint",
        "train_sha256",
        "train_mask_sha256",
        "code_commit",
        "total_tokens",
        "tokens_per_step",
        "micro_batch_size",
        "max_steps",
        "seed",
        "precision",
        "lr",
        "warmup_steps",
        "weight_decay",
        "planned_consumed_tokens",
    )
    continuation_id = hashlib.sha256(
        json.dumps(
            {key: config.get(key) for key in identity_keys},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    run_id = f"{run_prefix}_c{continuation_id[:8]}"
    config["continuation_id"] = continuation_id
    config["run_id"] = run_id
    config["out_dir"] = str(out_root / run_id)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--split-run", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--continuation-tokens", type=int, required=True)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--tokens-per-step", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    args = parser.parse_args()

    config_out = Path(args.config_out)
    config_out.mkdir(parents=True, exist_ok=True)
    source_pair = validate_source_pair(args.dense_run, args.split_run)
    configs = {}
    for arm, source in (("dense", args.dense_run), ("split", args.split_run)):
        config = make_continuation_config(
            source,
            arm,
            args.corpus_dir,
            args.out_root,
            args.continuation_tokens,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            tokens_per_step=args.tokens_per_step,
            micro_batch_size=args.micro_batch_size,
            precision=args.precision,
            source_identity=source_pair["arms"][arm],
            source_pair_id=source_pair["pair_id"],
        )
        path = config_out / f"kqa_{arm}.yaml"
        with open(path, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        configs[arm] = {"config": str(path), "out_dir": config["out_dir"]}
        print(f"wrote {path}")
    configs["provenance"] = {
        "source_pair_id": source_pair["pair_id"],
        "source_checkpoints": {
            arm: {
                "sha256": source_pair["arms"][arm]["sha256"],
                "step": source_pair["arms"][arm]["step"],
            }
            for arm in ("dense", "split")
        },
        "data_fingerprint": config["data_fingerprint"],
    }
    with open(config_out / "kqa_manifest.json", "w") as handle:
        json.dump(configs, handle, indent=2)


if __name__ == "__main__":
    main()
