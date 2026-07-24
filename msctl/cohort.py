"""Frozen scientific and ownership contract for the 135M N=10 cohort."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


COHORT_ID = "memorysplit-confirmatory-v2-135m-n10"
DATASET_CONTRACT_ID = "memorysplit-parallel-corpus-v2"
MODEL_PARAMETERS = 134_660_880
SEEDS = tuple(range(10))
ARMS = ("dense", "split90")
RAW_TARGETS = 7_120_879_616
TARGETS_PER_UPDATE = 524_288
TERMINAL_UPDATES = 13_582
CHECKPOINT_UPDATES = (1_358, 3_396, 6_791, 10_187, 13_582)

ROLES: dict[str, dict[str, Any]] = {
    "farmshare-lead": {
        "platform": "farmshare",
        "provider": "farmshare-l40s",
        "seeds": (0, 5),
    },
    "farmshare-collaborator-1": {
        "platform": "farmshare",
        "provider": "farmshare-l40s",
        "seeds": (1, 6),
    },
    "farmshare-collaborator-2": {
        "platform": "farmshare",
        "provider": "farmshare-l40s",
        "seeds": (2, 7),
    },
    "mit-collaborator-a": {
        "platform": "mit",
        "provider": "mit-slurm",
        "seeds": (3, 8),
    },
    "mit-collaborator-b": {
        "platform": "mit",
        "provider": "mit-slurm",
        "seeds": (4, 9),
    },
}

EXPECTED_CONFIG_PATHS = frozenset(
    f"configs/135m-v2/{arm}-s{seed}.yaml"
    for seed in SEEDS
    for arm in ARMS
)

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
    }
)


def role_for_seed(seed: int) -> str:
    """Return the sole frozen owner for ``seed``."""

    owners = [role for role, spec in ROLES.items() if seed in spec["seeds"]]
    if len(owners) != 1:
        raise ValueError(f"seed {seed!r} does not have exactly one owner")
    return owners[0]


def provider_for_seed(seed: int) -> str:
    return str(ROLES[role_for_seed(seed)]["provider"])


def config_path(arm: str, seed: int) -> str:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError(f"unknown frozen cell: arm={arm!r}, seed={seed!r}")
    return f"configs/135m-v2/{arm}-s{seed}.yaml"


def role_config_paths(role: str) -> list[str]:
    try:
        seeds = ROLES[role]["seeds"]
    except KeyError as error:
        raise ValueError(f"unknown operator role: {role!r}") from error
    return [
        config_path(arm, seed)
        for seed in seeds
        for arm in ARMS
    ]


def canonical_assignment() -> dict[str, Any]:
    """Return the canonical JSON-serializable assignment document."""

    return {
        "arms": list(ARMS),
        "cohort_id": COHORT_ID,
        "dataset_contract_id": DATASET_CONTRACT_ID,
        "model": "d135m",
        "model_parameters": MODEL_PARAMETERS,
        "operators": [
            {
                "config_paths": role_config_paths(role),
                "id": role,
                "platform": spec["platform"],
                "provider": spec["provider"],
                "seeds": list(spec["seeds"]),
            }
            for role, spec in ROLES.items()
        ],
        "provider_seeds": {
            "farmshare-l40s": [0, 1, 2, 5, 6, 7],
            "mit-slurm": [3, 4, 8, 9],
        },
        "raw_target_tokens": RAW_TARGETS,
        "schema_version": 1,
        "seeds": list(SEEDS),
        "targets_per_update": TARGETS_PER_UPDATE,
        "terminal_updates": TERMINAL_UPDATES,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_cohort_assignment(path: str | Path) -> dict[str, Any]:
    assignment_path = Path(path)
    if not assignment_path.is_file() or assignment_path.is_symlink():
        raise ValueError(f"assignment is missing or unsafe: {assignment_path}")
    try:
        raw = json.loads(
            assignment_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("assignment must be valid UTF-8 JSON") from error
    if raw != canonical_assignment():
        raise ValueError("assignment differs from the frozen 135M N=10 matrix")
    return raw


def _require_exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the frozen schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def validate_run_config(
    raw: object,
    *,
    relative_path: str | None = None,
) -> dict[str, Any]:
    """Validate one protected run cell without coercing values."""

    cfg = dict(_require_exact_fields(raw, _RUN_FIELDS, label="run config"))
    if cfg["schema_version"] != 1 or isinstance(cfg["schema_version"], bool):
        raise ValueError("run config schema_version must be exactly 1")
    if cfg["cohort_id"] != COHORT_ID:
        raise ValueError("run config cohort_id is not the frozen 135M cohort")
    arm, seed = cfg["arm"], cfg["seed"]
    if arm not in ARMS or isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError("run config contains an unknown arm or seed")
    expected_path = config_path(arm, seed)
    if relative_path is not None and relative_path != expected_path:
        raise ValueError(
            f"run config path is not canonical: {relative_path!r} != {expected_path!r}"
        )
    role = role_for_seed(seed)
    provider = provider_for_seed(seed)
    expected_scalars = {
        "run_id": f"d135m_{arm}_full_s{seed}",
        "pair_id": f"d135m_full_s{seed}",
        "model": "d135m",
        "model_parameters": MODEL_PARAMETERS,
        "operator": role,
        "provider": provider,
        "ctx": 1024,
        "total_tokens": RAW_TARGETS,
        "tokens_per_step": TARGETS_PER_UPDATE,
        "max_steps": TERMINAL_UPDATES,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "out_dir": f"outputs/135m-v2/d135m_{arm}_full_s{seed}",
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
    }
    for field, expected in expected_scalars.items():
        if cfg[field] != expected:
            raise ValueError(
                f"run config {field} differs from frozen value {expected!r}"
            )
    if (
        isinstance(cfg["micro_batch_size"], bool)
        or not isinstance(cfg["micro_batch_size"], int)
        or cfg["micro_batch_size"] <= 0
    ):
        raise ValueError("micro_batch_size must be a positive integer")
    for field in ("lr", "snap_frac"):
        if isinstance(cfg[field], bool) or not isinstance(cfg[field], (int, float)):
            raise ValueError(f"{field} must be numeric")
    for field in ("log_every", "eval_every", "ckpt_minutes"):
        if (
            isinstance(cfg[field], bool)
            or not isinstance(cfg[field], int)
            or cfg[field] <= 0
        ):
            raise ValueError(f"{field} must be a positive integer")

    weights = f"dataset/sidecars/{arm}_target_weights.bin"
    if cfg["train_bin"] != "dataset/packed/targets.bin":
        raise ValueError("run config must use the canonical packed target stream")
    if cfg["train_mask"] != weights:
        raise ValueError("run config must use its canonical target-weight sidecar")
    dataset = _require_exact_fields(
        cfg["dataset"],
        _DATASET_FIELDS,
        label="run config dataset",
    )
    expected_dataset = {
        "contract_id": DATASET_CONTRACT_ID,
        "pointer": "DATASET-POINTER-SLURM-135M.json",
        "complete_dataset": True,
        "packed_targets": "dataset/packed/targets.bin",
        "target_weights": weights,
        "raw_target_tokens": RAW_TARGETS,
    }
    if dict(dataset) != expected_dataset:
        raise ValueError("run config dataset binding is not canonical")
    return cfg


def load_run_config(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    config_path_value = Path(path)
    if not config_path_value.is_file() or config_path_value.is_symlink():
        raise ValueError(f"run config is missing or unsafe: {config_path_value}")
    try:
        raw = yaml.safe_load(config_path_value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("run config must be valid UTF-8 YAML") from error
    relative = None
    if root is not None:
        try:
            relative = config_path_value.resolve().relative_to(
                Path(root).resolve()
            ).as_posix()
        except ValueError as error:
            raise ValueError("run config must remain under the repository root") from error
    return validate_run_config(raw, relative_path=relative)
