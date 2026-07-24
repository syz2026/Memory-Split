"""Exploratory 135M N=10 AWS contract for the frozen reasoning-v3 corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

COHORT_ID = "memorysplit-exploratory-v3-135m-aws-n10"
DATASET_CONTRACT_ID = "memorysplit-reasoning-dataset-v3"
MODEL_PARAMETERS = 134_660_880
SEEDS = tuple(range(10))
ARMS = ("dense", "split90")
RAW_TARGETS = 8_169_455_616
TARGETS_PER_UPDATE = 524_288
TERMINAL_UPDATES = 15_582
CHECKPOINT_UPDATES = (1_558, 3_896, 7_791, 11_687, 15_582)
ROLE = "aws-operator"
PROVIDER = "aws-parallelcluster"
SCIENTIFIC_SCOPE = "successor_exploratory_unpreregistered"
POINTER_PATH = "DATASET-POINTER-AWS-135M-V3.json"
TRANSFER_MANIFEST = "cluster/aws/reasoning-v3-corpus-manifest.json"
TRANSFER_MANIFEST_SHA256 = (
    "84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb"
)
VIRTUAL_RECEIPT_SHA256 = (
    "b1eabb1719f66876ab54cc0791b857ccdbbbddb0ffb8c5986ac2aaa7bf33b80d"
)
COMPOSITE_STREAM_SHA256 = {
    "dense_target_weights": (
        "917768b13ec169728cec51dc8294d118a113aee3c370ecd8c16ef0529f63f56e"
    ),
    "packed_targets": (
        "035ee111c329eb615c642eae9b9a7075314932ff8175e989aabb3317d6a4ef6f"
    ),
    "split90_target_weights": (
        "8a9c84c900e503d1742342b6a21092292c2968313087d0873e429b4268757144"
    ),
}
PACKED_TARGETS = (
    "dataset/base/packed/targets.bin",
    "dataset/extension/packed/targets.bin",
)
TARGET_WEIGHTS = {
    "dense": (
        "dataset/base/sidecars/dense_target_weights.bin",
        "dataset/extension/sidecars/shared_target_weights.bin",
    ),
    "split90": (
        "dataset/base/sidecars/split90_target_weights.bin",
        "dataset/extension/sidecars/shared_target_weights.bin",
    ),
}
ROLES: dict[str, dict[str, Any]] = {
    ROLE: {
        "platform": "aws",
        "provider": PROVIDER,
        "seeds": SEEDS,
    }
}

_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "cohort_id",
        "run_id",
        "pair_id",
        "model",
        "model_parameters",
        "arm",
        "seed",
        "operator",
        "provider",
        "ctx",
        "train_bin",
        "train_mask",
        "total_tokens",
        "tokens_per_step",
        "max_steps",
        "micro_batch_size",
        "lr",
        "warmup_steps",
        "weight_decay",
        "compile",
        "device",
        "out_dir",
        "log_every",
        "eval_every",
        "snap_frac",
        "ckpt_minutes",
        "checkpoint_updates",
        "dataset",
    }
)
_DATASET_FIELDS = frozenset(
    {
        "contract_id",
        "pointer",
        "complete_dataset",
        "packed_targets",
        "target_weights",
        "raw_target_tokens",
        "scientific_scope",
    }
)


def pair_id(seed: int) -> str:
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError(f"unknown reasoning-v3 seed: {seed!r}")
    return f"d135m_reasoning_v3_s{seed}"


def run_id(arm: str, seed: int) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown reasoning-v3 arm: {arm!r}")
    pair_id(seed)
    return f"d135m_{arm}_reasoning_v3_s{seed}"


def config_path(arm: str, seed: int) -> str:
    run_id(arm, seed)
    return f"configs/135m-v3/{arm}-s{seed}.yaml"


def role_config_paths(role: str) -> list[str]:
    if role != ROLE:
        raise ValueError(f"unknown reasoning-v3 AWS role: {role!r}")
    return [
        config_path(arm, seed)
        for seed in SEEDS
        for arm in ARMS
    ]


def canonical_assignment() -> dict[str, Any]:
    return {
        "arms": list(ARMS),
        "cohort_id": COHORT_ID,
        "dataset_contract_id": DATASET_CONTRACT_ID,
        "model": "d135m",
        "model_parameters": MODEL_PARAMETERS,
        "operators": [
            {
                "config_paths": role_config_paths(ROLE),
                "id": ROLE,
                "platform": "aws",
                "provider": PROVIDER,
                "seeds": list(SEEDS),
            }
        ],
        "provider_seeds": {PROVIDER: list(SEEDS)},
        "raw_target_tokens": RAW_TARGETS,
        "schema_version": 1,
        "scientific_scope": SCIENTIFIC_SCOPE,
        "seeds": list(SEEDS),
        "targets_per_update": TARGETS_PER_UPDATE,
        "terminal_updates": TERMINAL_UPDATES,
    }


def canonical_pointer() -> dict[str, Any]:
    return {
        "contract_id": DATASET_CONTRACT_ID,
        "format": "memorysplit-aws-segmented-pointer-v1",
        "launch_gate_status": "frozen",
        "raw_target_tokens": RAW_TARGETS,
        "root_environment": "MS135_AWS_DATASET_ROOT",
        "schema_version": 1,
        "scientific_scope": SCIENTIFIC_SCOPE,
        "streams": {
            name: {
                "paths": [
                    value.removeprefix("dataset/")
                    for value in (
                        PACKED_TARGETS
                        if name == "packed_targets"
                        else TARGET_WEIGHTS[name.removesuffix("_target_weights")]
                    )
                ],
                "sha256": COMPOSITE_STREAM_SHA256[name],
            }
            for name in (
                "dense_target_weights",
                "packed_targets",
                "split90_target_weights",
            )
        },
        "transfer_manifest": TRANSFER_MANIFEST,
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA256,
        "virtual_receipt_sha256": VIRTUAL_RECEIPT_SHA256,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_exact_json(path: str | Path, expected: object, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 1 << 20:
        raise ValueError(f"{label} is missing, unsafe, or oversized")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if raw != expected:
        raise ValueError(f"{label} differs from the frozen reasoning-v3 contract")
    return raw


def load_cohort_assignment(path: str | Path) -> dict[str, Any]:
    return _load_exact_json(
        path,
        canonical_assignment(),
        label="reasoning-v3 cohort assignment",
    )


def load_dataset_pointer(path: str | Path) -> dict[str, Any]:
    return _load_exact_json(
        path,
        canonical_pointer(),
        label="reasoning-v3 AWS dataset pointer",
    )


def _exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields differ from the reasoning-v3 schema")
    return value


def validate_run_config(
    raw: object,
    *,
    relative_path: str | None = None,
) -> dict[str, Any]:
    cfg = dict(_exact_fields(raw, _RUN_FIELDS, label="run config"))
    if cfg["schema_version"] != 1 or isinstance(cfg["schema_version"], bool):
        raise ValueError("run config schema_version must be exactly 1")
    arm, seed = cfg["arm"], cfg["seed"]
    if (
        cfg["cohort_id"] != COHORT_ID
        or arm not in ARMS
        or isinstance(seed, bool)
        or seed not in SEEDS
    ):
        raise ValueError("run config contains an unknown reasoning-v3 cell")
    expected_path = config_path(arm, seed)
    if relative_path is not None and relative_path != expected_path:
        raise ValueError("run config path is not canonical")
    expected_scalars = {
        "run_id": run_id(arm, seed),
        "pair_id": pair_id(seed),
        "model": "d135m",
        "model_parameters": MODEL_PARAMETERS,
        "operator": ROLE,
        "provider": PROVIDER,
        "ctx": 1024,
        "total_tokens": RAW_TARGETS,
        "tokens_per_step": TARGETS_PER_UPDATE,
        "max_steps": TERMINAL_UPDATES,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "out_dir": f"outputs/135m-v3/{run_id(arm, seed)}",
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
    }
    for field, expected in expected_scalars.items():
        if cfg[field] != expected:
            raise ValueError(f"run config {field} differs from {expected!r}")
    if cfg["train_bin"] != list(PACKED_TARGETS):
        raise ValueError("run config must use both frozen packed segments")
    if cfg["train_mask"] != list(TARGET_WEIGHTS[arm]):
        raise ValueError("run config must use both frozen weight segments")
    if (
        isinstance(cfg["micro_batch_size"], bool)
        or not isinstance(cfg["micro_batch_size"], int)
        or cfg["micro_batch_size"] <= 0
    ):
        raise ValueError("micro_batch_size must be a positive integer")
    for field in ("lr", "snap_frac"):
        if isinstance(cfg[field], bool) or not isinstance(cfg[field], (int, float)):
            raise ValueError(f"{field} must be numeric")  # noqa: TRY004
    for field in ("log_every", "eval_every", "ckpt_minutes"):
        if (
            isinstance(cfg[field], bool)
            or not isinstance(cfg[field], int)
            or cfg[field] <= 0
        ):
            raise ValueError(f"{field} must be a positive integer")
    dataset = _exact_fields(cfg["dataset"], _DATASET_FIELDS, label="run dataset")
    expected_dataset = {
        "contract_id": DATASET_CONTRACT_ID,
        "pointer": POINTER_PATH,
        "complete_dataset": True,
        "packed_targets": list(PACKED_TARGETS),
        "target_weights": list(TARGET_WEIGHTS[arm]),
        "raw_target_tokens": RAW_TARGETS,
        "scientific_scope": SCIENTIFIC_SCOPE,
    }
    if dict(dataset) != expected_dataset:
        raise ValueError("run config dataset binding is not canonical")
    return cfg


def load_run_config(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    config = Path(path)
    if not config.is_file() or config.is_symlink():
        raise ValueError(f"run config is missing or unsafe: {config}")
    try:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("run config must be valid UTF-8 YAML") from error
    relative = None
    if root is not None:
        try:
            relative = config.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError as error:
            raise ValueError("run config must remain under the repository root") from error
    return validate_run_config(raw, relative_path=relative)
