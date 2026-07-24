#!/usr/bin/env python3
"""Run one operational checkpoint/data-boundary evaluation for reasoning-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.reasoning_cohort import (
    DATASET_CONTRACT_ID,
    RAW_TARGETS,
    TARGETS_PER_UPDATE,
    TERMINAL_UPDATES,
    VIRTUAL_RECEIPT_SHA256,
)
from train.data import PackedShards
from train.model import GPT, PRESETS, GPTConfig
from train.trainer import pick_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_run(run_dir: Path, *, device: str = "auto") -> dict:
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "ckpt.pt"
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
    ):
        raise ValueError("reasoning-v3 evaluation requires safe config and checkpoint files")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(cfg, dict)
        or not isinstance(cfg.get("dataset"), dict)
        or cfg["dataset"].get("contract_id") != DATASET_CONTRACT_ID
        or cfg.get("dataset_receipt_sha256") != VIRTUAL_RECEIPT_SHA256
        or cfg.get("total_tokens") != RAW_TARGETS
        or cfg.get("tokens_per_step") != TARGETS_PER_UPDATE
        or cfg.get("max_steps") != TERMINAL_UPDATES
    ):
        raise ValueError("reasoning-v3 evaluation config identity differs")
    selected_device = pick_device(device)
    state = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=False,
    )
    if (
        not isinstance(state, dict)
        or state.get("step") != TERMINAL_UPDATES
        or state.get("data") != {"cursor": RAW_TARGETS, "epoch": 1}
    ):
        raise ValueError("reasoning-v3 checkpoint is not at the exact terminal cursor")
    model_cfg = (
        PRESETS[cfg["model"]]
        if isinstance(cfg["model"], str)
        else GPTConfig(**cfg["model"])
    )
    if "ctx" in cfg:
        model_cfg.ctx = cfg["ctx"]
    model = GPT(model_cfg).to(selected_device)
    model.load_state_dict(state["model"])
    model.eval()
    target_count = cfg["micro_batch_size"] * model_cfg.ctx
    token_paths = cfg.get("train_bin")
    if (
        not isinstance(token_paths, list)
        or len(token_paths) != 2
        or any(not isinstance(path, str) or not path for path in token_paths)
    ):
        raise ValueError("reasoning-v3 evaluation requires two packed token segments")
    base_path = Path(token_paths[0])
    if (
        not base_path.is_file()
        or base_path.is_symlink()
        or base_path.stat().st_size % 2
    ):
        raise ValueError("reasoning-v3 base token segment is missing, unsafe, or invalid")
    base_target_tokens = base_path.stat().st_size // 2
    if target_count < 2 or base_target_tokens <= target_count // 2:
        raise ValueError("reasoning-v3 base segment is too short for boundary evaluation")
    boundary_cursor = base_target_tokens - target_count // 2
    data = PackedShards(
        cfg["train_bin"],
        cfg["train_mask"],
        ctx=model_cfg.ctx,
        batch_size=cfg["micro_batch_size"],
        device=selected_device,
        start_cursor=boundary_cursor,
    )
    with torch.no_grad():
        x, y, weights = data.next_weighted_batch()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if selected_device == "cuda"
            else torch.no_grad()
        )
        with context:
            _, loss_sum = model(
                x,
                y,
                target_weights=weights,
                loss_reduction="sum",
            )
    loss = float(loss_sum.item() / y.numel())
    if not math.isfinite(loss):
        raise ValueError("reasoning-v3 boundary evaluation produced non-finite loss")
    return {
        "boundary_cursor": boundary_cursor,
        "base_target_tokens": base_target_tokens,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_receipt_sha256": VIRTUAL_RECEIPT_SHA256,
        "evaluation_scope": "operational_integrity_only",
        "loss": loss,
        "raw_target_tokens": RAW_TARGETS,
        "run_id": cfg["run_id"],
        "schema_version": 1,
        "step": TERMINAL_UPDATES,
        "target_weight_sum": float(weights.sum().item()),
        "targets_evaluated": y.numel(),
    }


def _publish(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace evaluation summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    summary = evaluate_run(run_dir, device=args.device)
    _publish(run_dir / "evals" / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
