"""Immutable run/checkpoint identity binding for robustness evaluation."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import yaml


CLAIM_BEARING_CONDITIONS = {"dense", "split", "random"}


class CheckpointValidationPolicy(StrEnum):
    CLAIM_BEARING = "claim_bearing"
    RELATIONAL_EXPLORATORY = "relational_exploratory"


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run_checkpoint(
    run: str | Path,
    checkpoint: str | Path,
) -> Path:
    run_path = Path(run)
    if run_path.is_symlink() or not run_path.is_dir():
        raise ValueError("run must be a regular non-symlink directory")
    supplied = Path(checkpoint)
    if not supplied.is_absolute() and ".." in supplied.parts:
        raise ValueError("checkpoint path cannot contain parent traversal")
    candidate = supplied if supplied.is_absolute() else run_path / supplied
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    resolved_run = run_path.resolve(strict=True)
    resolved_checkpoint = candidate.resolve(strict=True)
    if not resolved_checkpoint.is_relative_to(resolved_run):
        raise ValueError(
            "checkpoint path must be contained under the run directory"
        )
    return resolved_checkpoint


def require_claim_bearing_checkpoint(
    path: str | Path,
) -> Mapping[str, Any]:
    checkpoint = Path(path)
    checkpoint_sha256(checkpoint)
    try:
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError("cannot load claim-bearing checkpoint") from exc
    if not isinstance(state, Mapping):
        raise ValueError("claim-bearing checkpoint must contain a mapping")
    model = state.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError(
            "claim-bearing checkpoint requires nonempty model weights"
        )
    if not isinstance(state.get("cfg"), Mapping):
        raise ValueError(
            "claim-bearing checkpoint requires embedded cfg mapping"
        )
    return state


def verify_checkpoint_unchanged(
    path: str | Path,
    expected_sha256: str,
) -> dict[str, str | bool]:
    after = checkpoint_sha256(path)
    if after != expected_sha256:
        raise RuntimeError("checkpoint changed during evaluation")
    return {
        "checkpoint_sha256": expected_sha256,
        "checkpoint_sha256_after": after,
        "checkpoint_mutated": False,
        "fine_tuning_performed": False,
    }


def _normalize_config(value: Any, path: str = "cfg") -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be nonempty strings")
            normalized[key] = _normalize_config(item, f"{path}.{key}")
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return [
            _normalize_config(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains a non-canonical value")


def _identity_sha256(value: Any) -> str:
    payload = json.dumps(
        _normalize_config(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_configuration_sha256(value: Mapping[str, Any]) -> str:
    """Hash the complete normalized run configuration."""

    if not isinstance(value, Mapping):
        raise ValueError("run configuration must be a mapping")
    return _identity_sha256(value)


PAIRING_ARM_SPECIFIC_CONFIG_FIELDS = (
    "config_sha256",
    "condition",
    "out_dir",
    "out_rel",
    "run_id",
    "train_weights",
    "weights_commitment_sha256",
    "weights_sha256",
    "weights_rel",
)


def canonical_shared_configuration_sha256(
    value: Mapping[str, Any],
) -> str:
    """Hash all config evidence except explicitly arm-specific fields."""

    if not isinstance(value, Mapping):
        raise ValueError("run configuration must be a mapping")
    normalized = _normalize_config(value)
    condition = normalized.get("condition")
    if not isinstance(condition, str) or not condition:
        raise ValueError("run configuration requires condition identity")
    present_arm_fields = tuple(
        field
        for field in PAIRING_ARM_SPECIFIC_CONFIG_FIELDS
        if field in normalized
    )
    shared = {
        key: item
        for key, item in normalized.items()
        if key not in PAIRING_ARM_SPECIFIC_CONFIG_FIELDS
    }
    return _identity_sha256(
        {
            "record_type": "paired_shared_configuration",
            "schema_version": 1,
            "excluded_arm_specific_fields": present_arm_fields,
            "shared_configuration": shared,
        }
    )


def _identity_parts(
    cfg: Mapping[str, Any],
    *,
    policy: CheckpointValidationPolicy,
) -> dict[str, Any]:
    normalized = _normalize_config(cfg)
    condition = normalized.get("condition")
    model = normalized.get("model")
    seed = normalized.get("seed")
    if policy == CheckpointValidationPolicy.CLAIM_BEARING:
        if condition not in CLAIM_BEARING_CONDITIONS:
            raise ValueError(
                "claim-bearing evaluation requires a dense, split, "
                "or random checkpoint"
            )
    elif policy == CheckpointValidationPolicy.RELATIONAL_EXPLORATORY:
        if condition != "selective":
            raise ValueError(
                "relational exploratory evaluation requires a Selective "
                "checkpoint"
            )
    else:
        raise ValueError("unknown checkpoint validation policy")
    if model is None:
        raise ValueError("checkpoint config requires model identity")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("checkpoint config requires integer seed identity")
    model_identity = {
        "model": model,
        "ctx": normalized.get("ctx"),
    }
    data_identity = {
        "train_bin": normalized.get("train_bin"),
        "train_mask": normalized.get("train_mask"),
    }
    weights_identity = {
        "train_weights": normalized.get("train_weights"),
    }
    return {
        "configuration_sha256": _identity_sha256(normalized),
        "condition": condition,
        "condition_sha256": _identity_sha256(condition),
        "model": model,
        "model_sha256": _identity_sha256(model_identity),
        "seed": seed,
        "seed_sha256": _identity_sha256(seed),
        "data_sha256": _identity_sha256(data_identity),
        "weights_sha256": _identity_sha256(weights_identity),
    }


def load_run_configuration(
    run: str | Path,
    *,
    policy: CheckpointValidationPolicy = CheckpointValidationPolicy.CLAIM_BEARING,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_path = Path(run)
    if run_path.is_symlink() or not run_path.is_dir():
        raise ValueError("run must be a regular non-symlink directory")
    if run_path.resolve(strict=True) != run_path.absolute():
        raise ValueError("run path cannot traverse symlink components")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "no-follow run configuration validation is unsupported"
        )
    config_path = run_path / "config.yaml"
    try:
        descriptor = os.open(config_path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK, errno.ENOTDIR}:
            raise ValueError(
                "run config must be a regular non-symlink file"
            ) from exc
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("run config must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read()
        entry = os.stat(config_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_dev != opened.st_dev
            or entry.st_ino != opened.st_ino
        ):
            raise ValueError("run config changed during validation")
    finally:
        os.close(descriptor)
    try:
        run_cfg = yaml.safe_load(content)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("run config is not valid YAML") from exc
    if not isinstance(run_cfg, Mapping):
        raise ValueError("run config must contain a mapping")
    try:
        selected_policy = CheckpointValidationPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown checkpoint validation policy") from exc
    normalized = _normalize_config(run_cfg)
    return normalized, _identity_parts(normalized, policy=selected_policy)


def verify_checkpoint_config(
    run: str | Path,
    state: Mapping[str, Any],
    *,
    policy: CheckpointValidationPolicy = (
        CheckpointValidationPolicy.CLAIM_BEARING
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_cfg = state.get("cfg")
    if not isinstance(checkpoint_cfg, Mapping):
        raise ValueError("checkpoint requires embedded cfg mapping")

    normalized_checkpoint = _normalize_config(checkpoint_cfg)
    try:
        selected_policy = CheckpointValidationPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown checkpoint validation policy") from exc
    normalized_run, run_identities = load_run_configuration(
        run,
        policy=selected_policy,
    )
    checkpoint_identities = _identity_parts(
        normalized_checkpoint,
        policy=selected_policy,
    )
    for name in ("condition", "model", "seed"):
        if run_identities[name] != checkpoint_identities[name]:
            raise ValueError(f"checkpoint cfg {name} identity mismatch")
    for name in ("data_sha256", "weights_sha256"):
        if run_identities[name] != checkpoint_identities[name]:
            label = name.removesuffix("_sha256")
            raise ValueError(f"checkpoint cfg {label} identity mismatch")
    if (
        run_identities["model_sha256"]
        != checkpoint_identities["model_sha256"]
    ):
        raise ValueError("checkpoint cfg model identity mismatch")
    if (
        run_identities["configuration_sha256"]
        != checkpoint_identities["configuration_sha256"]
    ):
        raise ValueError("checkpoint cfg configuration identity mismatch")
    return normalized_checkpoint, checkpoint_identities
