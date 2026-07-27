"""Training loop: AdamW + cosine, bf16 autocast (CUDA), grad accumulation,
atomic checkpoint/resume (model+opt+data cursor+RNG), model-only snapshots,
JSONL logging including the split-arm mechanism metric `loss_masked_values`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment.artifacts import (
    atomic_write_bytes,
    atomic_write_stream,
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
    require_regular_file,
    sha256_file,
    validate_sha256,
)
from experiment.ledger import RunLedger
from train.data import PackedShards
from train.model import GPT, GPTConfig, PRESETS


class ProvenanceError(ValueError):
    """Raised before training when run identity or resume state is unsafe."""


_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_RUN_CONFIG_FIELDS = {
    "record_type",
    "schema_version",
    "run_id",
    "status",
    "watermark",
    "launchable",
    "model",
    "condition",
    "load",
    "load_role",
    "entities",
    "seed",
    "initialization_seed",
    "data_seed",
    "parameter_count",
    "architecture",
    "data_rel",
    "weights_rel",
    "out_rel",
    "stream_sha256",
    "weights_sha256",
    "stream_commitment_sha256",
    "weights_commitment_sha256",
    "freeze_sha256",
    "selected_mixture",
    "tokens_per_parameter",
    "tokens_per_step",
    "requested_raw_positions",
    "actual_raw_positions",
    "steps",
    "rounding_error_fraction",
    "packing",
    "optimizer",
    "scheduler",
    "decode_budget",
    "pair_fingerprint",
    "config_sha256",
}
_PROVENANCE_FIELDS = {
    "record_type",
    "schema_version",
    "run_id",
    "freeze_sha256",
    "config_sha256",
    "source_tree_sha256",
    "corpus_sha256",
    "mask_sha256",
    "weights_sha256",
    "initialization_sha256",
    "architecture",
    "initialization_seed",
    "data_seed",
    "optimizer",
    "scheduler",
    "packing",
    "raw_positions",
    "steps",
    "tokens_per_step",
    "pair_fingerprint",
    "provenance_sha256",
}
_CHECKPOINT_FIELDS = {
    "schema_version",
    "provenance",
    "model",
    "opt",
    "data",
    "step",
    "scheduler",
    "rng_python",
    "rng_numpy",
    "rng_torch",
    "rng_cuda",
    "rng_mps",
    "cfg",
    "running_loss",
    "last_step_loss",
}


def _normalize_json(value: Any, path: str = "$") -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ProvenanceError(
                    f"{path} contains a non-string or empty key"
                )
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ProvenanceError(f"{path} contains a non-canonical value")


def _canonical_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise ProvenanceError("training configuration must be a mapping")
    normalized = _normalize_json(cfg, "cfg")
    try:
        return json.loads(canonical_json_bytes(normalized))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProvenanceError("training configuration is not canonical") from exc


def _validate_hash(value: object, name: str) -> str:
    try:
        return validate_sha256(value, name)
    except ValueError as exc:
        raise ProvenanceError(str(exc)) from exc


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProvenanceError(f"{name} must be a positive integer")
    return value


def _seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProvenanceError(f"{name} must be a nonnegative integer")
    return value


def mps_rng_available() -> bool:
    return bool(
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
        and hasattr(torch.mps, "manual_seed")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


_NUMPY_RNG_FIELDS = {
    "schema_version",
    "bit_generator",
    "keys",
    "position",
    "has_gauss",
    "cached_gaussian",
}


def _decode_numpy_rng_state(
    value: object,
) -> tuple[str, np.ndarray, int, int, float]:
    if not isinstance(value, Mapping) or set(value) != _NUMPY_RNG_FIELDS:
        raise ProvenanceError("checkpoint NumPy RNG state fields are not exact")
    if value["schema_version"] != 1 or value["bit_generator"] != "MT19937":
        raise ProvenanceError("checkpoint NumPy RNG state protocol is invalid")
    keys = value["keys"]
    if (
        not isinstance(keys, torch.Tensor)
        or keys.dtype != torch.int64
        or keys.device.type != "cpu"
        or keys.ndim != 1
        or keys.numel() != 624
        or not bool(((keys >= 0) & (keys <= 2**32 - 1)).all())
    ):
        raise ProvenanceError("checkpoint NumPy RNG keys are invalid")
    position = value["position"]
    has_gauss = value["has_gauss"]
    cached_gaussian = value["cached_gaussian"]
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position <= keys.numel()
        or isinstance(has_gauss, bool)
        or not isinstance(has_gauss, int)
        or has_gauss not in {0, 1}
        or isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(float(cached_gaussian))
    ):
        raise ProvenanceError("checkpoint NumPy RNG position is invalid")
    decoded = (
        "MT19937",
        keys.numpy().astype(np.uint32, copy=True),
        position,
        has_gauss,
        float(cached_gaussian),
    )
    try:
        probe = np.random.RandomState()
        probe.set_state(decoded)
        roundtrip = probe.get_state()
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("checkpoint NumPy RNG state is invalid") from exc
    if (
        roundtrip[0] != decoded[0]
        or not np.array_equal(roundtrip[1], decoded[1])
        or roundtrip[2:] != decoded[2:]
    ):
        raise ProvenanceError(
            "checkpoint NumPy RNG state does not round-trip exactly"
        )
    return decoded


def _encode_numpy_rng_state() -> dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = (
        np.random.get_state()
    )
    encoded = {
        "schema_version": 1,
        "bit_generator": bit_generator,
        "keys": torch.tensor(
            keys.astype(np.int64, copy=False),
            dtype=torch.int64,
        ),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }
    _decode_numpy_rng_state(encoded)
    return encoded


def capture_rng_state() -> dict[str, Any]:
    return {
        "rng_python": random.getstate(),
        "rng_numpy": _encode_numpy_rng_state(),
        "rng_torch": torch.get_rng_state().clone(),
        "rng_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "rng_mps": (
            torch.mps.get_rng_state().clone()
            if mps_rng_available()
            else None
        ),
    }


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    expected = {
        "rng_python",
        "rng_numpy",
        "rng_torch",
        "rng_cuda",
        "rng_mps",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ProvenanceError("checkpoint RNG state fields are not exact")
    try:
        random.Random().setstate(state["rng_python"])
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("checkpoint Python RNG state is invalid") from exc
    _decode_numpy_rng_state(state["rng_numpy"])
    torch_state = state["rng_torch"]
    if (
        not isinstance(torch_state, torch.Tensor)
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
    ):
        raise ProvenanceError("checkpoint Torch CPU RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(torch_state.cpu())
    except RuntimeError as exc:
        raise ProvenanceError("checkpoint Torch CPU RNG state is invalid") from exc
    cuda_state = state["rng_cuda"]
    if cuda_state is not None and (
        not isinstance(cuda_state, list)
        or any(
            not isinstance(item, torch.Tensor)
            or item.dtype != torch.uint8
            or item.ndim != 1
            for item in cuda_state
        )
    ):
        raise ProvenanceError("checkpoint CUDA RNG state is invalid")
    if torch.cuda.is_available() and (
        cuda_state is None or len(cuda_state) != torch.cuda.device_count()
    ):
        raise ProvenanceError(
            "checkpoint CUDA RNG state does not match available devices"
        )
    if torch.cuda.is_available():
        try:
            current_cuda_state = torch.cuda.get_rng_state_all()
        except RuntimeError as exc:
            raise ProvenanceError(
                "available CUDA RNG state cannot be inspected"
            ) from exc
        if any(
            saved.shape != current.shape
            for saved, current in zip(cuda_state, current_cuda_state)
        ):
            raise ProvenanceError(
                "checkpoint CUDA RNG state does not match available devices"
            )
    mps_state = state["rng_mps"]
    if mps_state is not None and (
        not isinstance(mps_state, torch.Tensor)
        or mps_state.dtype != torch.uint8
        or mps_state.ndim != 1
    ):
        raise ProvenanceError("checkpoint MPS RNG state is invalid")
    if mps_rng_available() and mps_state is None:
        raise ProvenanceError(
            "checkpoint MPS RNG state is missing on an available MPS device"
        )
    if mps_rng_available():
        try:
            current_mps_state = torch.mps.get_rng_state()
        except RuntimeError as exc:
            raise ProvenanceError(
                "available MPS RNG state cannot be inspected"
            ) from exc
        if mps_state.shape != current_mps_state.shape:
            raise ProvenanceError(
                "checkpoint MPS RNG state does not match the available device"
            )


def restore_rng_state(state: Mapping[str, Any]) -> None:
    _validate_rng_state(state)
    random.setstate(state["rng_python"])
    np.random.set_state(_decode_numpy_rng_state(state["rng_numpy"]))
    torch.set_rng_state(state["rng_torch"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["rng_cuda"]]
        )
    if mps_rng_available():
        torch.mps.set_rng_state(state["rng_mps"].cpu())


def _model_config(cfg: Mapping[str, Any]) -> GPTConfig:
    model = cfg.get("model")
    if isinstance(model, str):
        if model not in PRESETS:
            raise ProvenanceError(f"unknown model preset: {model}")
        result = replace(PRESETS[model])
    elif isinstance(model, Mapping):
        try:
            result = GPTConfig(**dict(model))
        except (TypeError, ValueError) as exc:
            raise ProvenanceError("model configuration is invalid") from exc
    else:
        raise ProvenanceError("model configuration is required")
    if "ctx" in cfg:
        ctx = _positive_integer(cfg["ctx"], "context length")
        result = replace(result, ctx=ctx)
    if result.d_model % result.n_head:
        raise ProvenanceError("model width must be divisible by attention heads")
    return result


def _architecture(
    cfg: Mapping[str, Any],
    model_cfg: GPTConfig,
) -> dict[str, Any]:
    expected = {
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head,
        "d_model": model_cfg.d_model,
        "vocab_size": model_cfg.vocab_size,
        "ctx": model_cfg.ctx,
    }
    supplied = cfg.get("architecture")
    if supplied is None:
        return _normalize_json(asdict(model_cfg), "architecture")
    if not isinstance(supplied, Mapping):
        raise ProvenanceError("architecture identity must be an object")
    normalized = _normalize_json(supplied, "architecture")
    for field, value in expected.items():
        if normalized.get(field) != value:
            raise ProvenanceError(
                f"runtime architecture {field} does not match model"
            )
    return normalized


def _optimizer_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    supplied = cfg.get("optimizer")
    if supplied is None:
        identity = {
            "name": "adamw",
            "lr": cfg.get("lr"),
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": cfg.get("weight_decay", 0.1),
            "gradient_clip": 1.0,
        }
    elif isinstance(supplied, Mapping):
        identity = _normalize_json(supplied, "optimizer")
    else:
        raise ProvenanceError("optimizer identity must be an object")
    required = {
        "name",
        "lr",
        "betas",
        "epsilon",
        "weight_decay",
        "gradient_clip",
    }
    if set(identity) != required or identity["name"] != "adamw":
        raise ProvenanceError("optimizer identity is not exact AdamW")
    betas = identity["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in betas
        )
    ):
        raise ProvenanceError("optimizer betas are invalid")
    for name in ("lr", "epsilon", "weight_decay", "gradient_clip"):
        value = identity[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ProvenanceError(f"optimizer {name} is invalid")
    for runtime, identity_name in (
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
    ):
        if runtime in cfg and float(cfg[runtime]) != float(identity[identity_name]):
            raise ProvenanceError(
                f"runtime {runtime} does not match optimizer identity"
            )
    return identity


def _scheduler_identity(
    cfg: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    supplied = cfg.get("scheduler")
    if supplied is None:
        identity = {
            "name": "cosine",
            "warmup_steps": cfg.get("warmup_steps", 300),
            "minimum_learning_rate_fraction": 0.1,
        }
    elif isinstance(supplied, Mapping):
        identity = _normalize_json(supplied, "scheduler")
    else:
        raise ProvenanceError("scheduler identity must be an object")
    if set(identity) != {
        "name",
        "warmup_steps",
        "minimum_learning_rate_fraction",
    } or identity["name"] != "cosine":
        raise ProvenanceError("scheduler identity is not exact cosine")
    warmup = identity["warmup_steps"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ProvenanceError("scheduler warmup steps are invalid")
    minimum = identity["minimum_learning_rate_fraction"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or not 0 <= float(minimum) <= 1
    ):
        raise ProvenanceError("scheduler minimum learning-rate fraction is invalid")
    if "warmup_steps" in cfg and cfg["warmup_steps"] != warmup:
        raise ProvenanceError(
            "runtime warmup_steps does not match scheduler identity"
        )
    return identity


def _packing_identity(
    cfg: Mapping[str, Any],
    model_cfg: GPTConfig,
) -> dict[str, Any]:
    supplied = cfg.get("packing")
    if supplied is None:
        return {
            "format": "packed-u16-v1",
            "context_length": model_cfg.ctx,
            "boundary_policy": "contiguous-v1",
        }
    if not isinstance(supplied, Mapping):
        raise ProvenanceError("packing identity must be an object")
    identity = _normalize_json(supplied, "packing")
    if identity.get("context_length") != model_cfg.ctx:
        raise ProvenanceError(
            "packing context length does not match model architecture"
        )
    return identity


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    freeze_sha256: str
    config_sha256: str
    source_tree_sha256: str
    corpus_sha256: str
    mask_sha256: str | None
    weights_sha256: str | None
    initialization_sha256: str
    architecture: Mapping[str, Any]
    initialization_seed: int
    data_seed: int
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    packing: Mapping[str, Any]
    raw_positions: int
    steps: int
    tokens_per_step: int
    pair_fingerprint: str
    provenance_sha256: str

    def _without_hash(self) -> dict[str, Any]:
        return {
            "record_type": "training_run_provenance",
            "schema_version": 1,
            "run_id": self.run_id,
            "freeze_sha256": self.freeze_sha256,
            "config_sha256": self.config_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "corpus_sha256": self.corpus_sha256,
            "mask_sha256": self.mask_sha256,
            "weights_sha256": self.weights_sha256,
            "initialization_sha256": self.initialization_sha256,
            "architecture": _normalize_json(
                self.architecture,
                "provenance.architecture",
            ),
            "initialization_seed": self.initialization_seed,
            "data_seed": self.data_seed,
            "optimizer": _normalize_json(
                self.optimizer,
                "provenance.optimizer",
            ),
            "scheduler": _normalize_json(
                self.scheduler,
                "provenance.scheduler",
            ),
            "packing": _normalize_json(
                self.packing,
                "provenance.packing",
            ),
            "raw_positions": self.raw_positions,
            "steps": self.steps,
            "tokens_per_step": self.tokens_per_step,
            "pair_fingerprint": self.pair_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._without_hash(),
            "provenance_sha256": self.provenance_sha256,
        }

    as_dict = to_dict

    @classmethod
    def create(cls, **values: Any) -> "RunProvenance":
        provisional = cls(provenance_sha256="0" * 64, **values)
        material = provisional._without_hash()
        return cls(
            **values,
            provenance_sha256=canonical_sha256(material),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunProvenance":
        if not isinstance(raw, Mapping) or set(raw) != _PROVENANCE_FIELDS:
            raise ProvenanceError("run provenance fields are not exact")
        if (
            raw["record_type"] != "training_run_provenance"
            or raw["schema_version"] != 1
        ):
            raise ProvenanceError("run provenance protocol is invalid")
        try:
            provenance = cls(
                run_id=raw["run_id"],
                freeze_sha256=raw["freeze_sha256"],
                config_sha256=raw["config_sha256"],
                source_tree_sha256=raw["source_tree_sha256"],
                corpus_sha256=raw["corpus_sha256"],
                mask_sha256=raw["mask_sha256"],
                weights_sha256=raw["weights_sha256"],
                initialization_sha256=raw["initialization_sha256"],
                architecture=raw["architecture"],
                initialization_seed=raw["initialization_seed"],
                data_seed=raw["data_seed"],
                optimizer=raw["optimizer"],
                scheduler=raw["scheduler"],
                packing=raw["packing"],
                raw_positions=raw["raw_positions"],
                steps=raw["steps"],
                tokens_per_step=raw["tokens_per_step"],
                pair_fingerprint=raw["pair_fingerprint"],
                provenance_sha256=raw["provenance_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ProvenanceError):
                raise
            raise ProvenanceError("run provenance values are invalid") from exc
        provenance.validate()
        return provenance

    def validate(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or _RUN_ID_RE.fullmatch(self.run_id) is None
        ):
            raise ProvenanceError("run ID is not canonical")
        for name, value in (
            ("freeze SHA-256", self.freeze_sha256),
            ("config SHA-256", self.config_sha256),
            ("source tree SHA-256", self.source_tree_sha256),
            ("corpus SHA-256", self.corpus_sha256),
            ("initialization SHA-256", self.initialization_sha256),
            ("pair fingerprint", self.pair_fingerprint),
            ("provenance SHA-256", self.provenance_sha256),
        ):
            _validate_hash(value, name)
        for name, value in (
            ("mask SHA-256", self.mask_sha256),
            ("weights SHA-256", self.weights_sha256),
        ):
            if value is not None:
                _validate_hash(value, name)
        _seed(self.initialization_seed, "initialization seed")
        _seed(self.data_seed, "data seed")
        _positive_integer(self.raw_positions, "raw positions")
        _positive_integer(self.steps, "optimizer steps")
        _positive_integer(self.tokens_per_step, "tokens per step")
        if self.raw_positions != self.steps * self.tokens_per_step:
            raise ProvenanceError(
                "raw positions do not match optimizer-step budget"
            )
        for name, value in (
            ("architecture", self.architecture),
            ("optimizer", self.optimizer),
            ("scheduler", self.scheduler),
            ("packing", self.packing),
        ):
            if not isinstance(value, Mapping):
                raise ProvenanceError(f"provenance {name} must be an object")
            _normalize_json(value, f"provenance.{name}")
        if canonical_sha256(self._without_hash()) != self.provenance_sha256:
            raise ProvenanceError("run provenance hash mismatch")


@dataclass(frozen=True)
class ValidatedRunStart:
    cfg: Mapping[str, Any]
    provenance: RunProvenance
    resume: str
    checkpoint_path: Path | None
    checkpoint_state: Mapping[str, Any] | None

    @property
    def is_resume(self) -> bool:
        return self.checkpoint_state is not None


RunStart = ValidatedRunStart


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return peak * min_frac
    ratio = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * ratio)))


def _validated_frozen_config(cfg: Mapping[str, Any]):
    if cfg.get("record_type") != "relational_run_config":
        return None
    if not _RUN_CONFIG_FIELDS <= set(cfg):
        raise ProvenanceError("resolved run config is missing frozen fields")
    from scripts.make_relational_manifest import RunConfig

    raw = {field: cfg[field] for field in _RUN_CONFIG_FIELDS}
    try:
        run = RunConfig.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("frozen run configuration is invalid") from exc
    if not run.launchable:
        raise ProvenanceError("frozen run configuration is not launchable")
    expected_runtime = {
        "model": run.model,
        "condition": run.condition,
        "seed": run.seed,
        "initialization_seed": run.initialization_seed,
        "data_seed": run.data_seed,
        "tokens_per_step": run.tokens_per_step,
        "max_steps": run.steps,
        "total_tokens": run.actual_raw_positions,
        "actual_raw_positions": run.actual_raw_positions,
        "lr": run.optimizer["lr"],
        "weight_decay": run.optimizer["weight_decay"],
        "warmup_steps": run.scheduler["warmup_steps"],
        "ctx": run.architecture["ctx"],
    }
    for field, expected in expected_runtime.items():
        if cfg.get(field) != expected:
            raise ProvenanceError(
                f"runtime {field} does not match frozen run configuration"
            )
    if cfg.get("architecture") != run.to_dict()["architecture"]:
        raise ProvenanceError(
            "runtime architecture does not match frozen run configuration"
        )
    if cfg.get("optimizer") != run.to_dict()["optimizer"]:
        raise ProvenanceError(
            "runtime optimizer does not match frozen run configuration"
        )
    if cfg.get("scheduler") != run.to_dict()["scheduler"]:
        raise ProvenanceError(
            "runtime scheduler does not match frozen run configuration"
        )
    if cfg.get("packing") != run.to_dict()["packing"]:
        raise ProvenanceError(
            "runtime packing does not match frozen run configuration"
        )
    if "source_tree_sha256" not in cfg:
        raise ProvenanceError(
            "frozen runtime config requires source tree SHA-256"
        )
    if not cfg.get("train_weights"):
        raise ProvenanceError(
            "frozen runtime config requires a weights sidecar"
        )
    if "data_root" in cfg:
        expected_data = Path(cfg["data_root"]) / run.data_rel
        expected_weights = Path(cfg["data_root"]) / run.weights_rel
        if Path(cfg["train_bin"]) != expected_data:
            raise ProvenanceError(
                "runtime corpus path does not match frozen relative path"
            )
        if Path(cfg["train_weights"]) != expected_weights:
            raise ProvenanceError(
                "runtime weights path does not match frozen relative path"
            )
    if "out_root" in cfg:
        expected_output = Path(cfg["out_root"]) / run.out_rel
        if Path(cfg["out_dir"]) != expected_output:
            raise ProvenanceError(
                "runtime output path does not match frozen relative path"
            )
    return run


def _canonical_output_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ProvenanceError("out_dir must be a nonempty path string")
    supplied = Path(value)
    if ".." in supplied.parts:
        raise ProvenanceError("out_dir cannot contain traversal")
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    current = absolute
    missing: list[Path] = []
    while not os.path.lexists(current):
        missing.append(current)
        if current == current.parent:
            break
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ProvenanceError(
            "out_dir ancestor must be a regular non-symlink directory"
        )
    if current.resolve(strict=True) != current:
        raise ProvenanceError("out_dir traverses a symlink")
    for path in reversed(missing):
        if path.name in {"", ".", ".."}:
            raise ProvenanceError("out_dir is not canonical")
    if os.path.lexists(absolute):
        if absolute.is_symlink() or not absolute.is_dir():
            raise ProvenanceError(
                "out_dir must be a regular non-symlink directory"
            )
        if absolute.resolve(strict=True) != absolute:
            raise ProvenanceError("out_dir traverses a symlink")
    return absolute


def _input_file(
    value: object,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{name} path is required")
    try:
        return require_regular_file(value, name=name)
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"{name} is missing or invalid: {exc}") from exc


def _validate_sidecar_lengths(
    corpus: Path,
    mask: Path | None,
    weights: Path | None,
) -> int:
    corpus_bytes = corpus.stat(follow_symlinks=False).st_size
    if corpus_bytes % np.dtype(np.uint16).itemsize:
        raise ProvenanceError("corpus byte length is not aligned to uint16")
    token_count = corpus_bytes // np.dtype(np.uint16).itemsize
    if token_count < 1:
        raise ProvenanceError("corpus is empty")
    for name, path in (("mask", mask), ("weights", weights)):
        if path is not None and path.stat(follow_symlinks=False).st_size != token_count:
            raise ProvenanceError(
                f"{name} sidecar length does not match corpus tokens"
            )
    return token_count


def _generic_config_sha256(cfg: Mapping[str, Any]) -> str:
    portable = {
        key: value
        for key, value in cfg.items()
        if key
        not in {
            "out_dir",
            "out_root",
            "ledger_root",
            "data_root",
        }
    }
    return canonical_sha256(
        {
            "record_type": "runtime_training_config",
            "schema_version": 1,
            "config": portable,
        }
    )


def _checkpoint_config_identity(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _canonical_config(cfg)
    return {
        key: value
        for key, value in normalized.items()
        if key
        not in {
            "train_bin",
            "train_mask",
            "train_weights",
            "data_root",
            "out_dir",
            "out_root",
            "ledger_root",
        }
    }


def _validate_configured_asset_hash(
    cfg: Mapping[str, Any],
    *,
    canonical_field: str,
    alias_field: str,
    actual_sha256: str | None,
    name: str,
) -> None:
    configured: dict[str, str] = {}
    for field in (canonical_field, alias_field):
        if field in cfg:
            configured[field] = _validate_hash(
                cfg[field],
                f"configured {name} SHA-256",
            )
    if (
        canonical_field in configured
        and alias_field in configured
        and configured[canonical_field] != configured[alias_field]
    ):
        raise ProvenanceError(
            f"conflicting {name} SHA-256 fields "
            f"{canonical_field} and {alias_field}"
        )
    expected = configured.get(
        canonical_field,
        configured.get(alias_field),
    )
    if expected is not None and expected != actual_sha256:
        raise ProvenanceError(
            f"configured {name} SHA-256 does not match bytes"
        )


def _build_run_provenance(
    cfg: Mapping[str, Any],
    *,
    corpus_sha256: str,
    mask_sha256: str | None,
    weights_sha256: str | None,
) -> RunProvenance:
    frozen = _validated_frozen_config(cfg)
    model_cfg = _model_config(cfg)
    architecture = _architecture(cfg, model_cfg)
    tokens_per_step = _positive_integer(
        cfg.get("tokens_per_step"),
        "tokens per step",
    )
    micro_batch_size = _positive_integer(
        cfg.get("micro_batch_size"),
        "micro batch size",
    )
    positions_per_microbatch = micro_batch_size * model_cfg.ctx
    if tokens_per_step % positions_per_microbatch:
        raise ProvenanceError(
            "tokens_per_step must be divisible by micro_batch_size * context"
        )
    max_steps_value = cfg.get("max_steps")
    if max_steps_value is None:
        total_tokens = _positive_integer(
            cfg.get("total_tokens"),
            "total tokens",
        )
        if total_tokens % tokens_per_step:
            raise ProvenanceError(
                "total tokens must be divisible by tokens_per_step"
            )
        max_steps = total_tokens // tokens_per_step
    else:
        max_steps = _positive_integer(max_steps_value, "optimizer steps")
    raw_positions = cfg.get(
        "actual_raw_positions",
        cfg.get("total_tokens", max_steps * tokens_per_step),
    )
    raw_positions = _positive_integer(raw_positions, "raw positions")
    if raw_positions != max_steps * tokens_per_step:
        raise ProvenanceError(
            "raw positions must equal optimizer steps times tokens_per_step"
        )
    optimizer = _optimizer_identity(cfg)
    scheduler = _scheduler_identity(cfg, max_steps)
    packing = _packing_identity(cfg, model_cfg)
    initialization_seed = _seed(
        cfg.get("initialization_seed", cfg.get("seed")),
        "initialization seed",
    )
    data_seed = _seed(
        cfg.get("data_seed", cfg.get("seed")),
        "data seed",
    )
    initialization_sha256 = canonical_sha256(
        {
            "record_type": "model_initialization",
            "schema_version": 1,
            "architecture": architecture,
            "initialization_seed": initialization_seed,
        }
    )
    if "initialization_sha256" in cfg and (
        _validate_hash(
            cfg["initialization_sha256"],
            "configured initialization SHA-256",
        )
        != initialization_sha256
    ):
        raise ProvenanceError(
            "configured initialization SHA-256 does not match architecture "
            "and seed"
        )
    _validate_configured_asset_hash(
        cfg,
        canonical_field="stream_sha256",
        alias_field="corpus_sha256",
        actual_sha256=corpus_sha256,
        name="stream",
    )
    _validate_configured_asset_hash(
        cfg,
        canonical_field="weights_sha256",
        alias_field="weights_file_sha256",
        actual_sha256=weights_sha256,
        name="weights",
    )
    if frozen is not None:
        run_id = frozen.run_id
        freeze_sha256 = frozen.freeze_sha256
        config_sha256 = frozen.config_sha256
        pair_fingerprint = frozen.pair_fingerprint
        source_tree_sha256 = _validate_hash(
            cfg["source_tree_sha256"],
            "source tree SHA-256",
        )
    else:
        config_sha256 = (
            _validate_hash(cfg["config_sha256"], "config SHA-256")
            if "config_sha256" in cfg
            else _generic_config_sha256(cfg)
        )
        run_id = cfg.get("run_id", f"run-{config_sha256[:16]}")
        if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
            raise ProvenanceError("run ID is not canonical")
        freeze_sha256 = (
            _validate_hash(cfg["freeze_sha256"], "freeze SHA-256")
            if "freeze_sha256" in cfg
            else canonical_sha256(
                {
                    "record_type": "unfrozen_training",
                    "schema_version": 1,
                }
            )
        )
        source_tree_sha256 = (
            _validate_hash(
                cfg["source_tree_sha256"],
                "source tree SHA-256",
            )
            if "source_tree_sha256" in cfg
            else canonical_sha256(
                {
                    "record_type": "unbound_source_tree",
                    "schema_version": 1,
                }
            )
        )
        pair_fingerprint = (
            _validate_hash(cfg["pair_fingerprint"], "pair fingerprint")
            if "pair_fingerprint" in cfg
            else canonical_sha256(
                {
                    "record_type": "runtime_pair_fingerprint",
                    "schema_version": 1,
                    "architecture": architecture,
                    "initialization_seed": initialization_seed,
                    "data_seed": data_seed,
                    "corpus_sha256": corpus_sha256,
                    "packing": packing,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "steps": max_steps,
                    "raw_positions": raw_positions,
                }
            )
        )
    provenance = RunProvenance.create(
        run_id=run_id,
        freeze_sha256=freeze_sha256,
        config_sha256=config_sha256,
        source_tree_sha256=source_tree_sha256,
        corpus_sha256=corpus_sha256,
        mask_sha256=mask_sha256,
        weights_sha256=weights_sha256,
        initialization_sha256=initialization_sha256,
        architecture=architecture,
        initialization_seed=initialization_seed,
        data_seed=data_seed,
        optimizer=optimizer,
        scheduler=scheduler,
        packing=packing,
        raw_positions=raw_positions,
        steps=max_steps,
        tokens_per_step=tokens_per_step,
        pair_fingerprint=pair_fingerprint,
    )
    provenance.validate()
    return provenance


def _expected_model_shapes(
    architecture: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    n_layer = _positive_integer(architecture.get("n_layer"), "model layers")
    d_model = _positive_integer(architecture.get("d_model"), "model width")
    vocab_size = _positive_integer(
        architecture.get("vocab_size"),
        "vocabulary size",
    )
    hidden = ((int(8 * d_model / 3) + 63) // 64) * 64
    shapes: dict[str, tuple[int, ...]] = {
        "wte.weight": (vocab_size, d_model),
    }
    for index in range(n_layer):
        prefix = f"blocks.{index}"
        shapes[f"{prefix}.ln1.weight"] = (d_model,)
        for projection in ("wq", "wk", "wv", "wo"):
            shapes[f"{prefix}.attn.{projection}.weight"] = (
                d_model,
                d_model,
            )
        shapes[f"{prefix}.ln2.weight"] = (d_model,)
        shapes[f"{prefix}.mlp.w1.weight"] = (hidden, d_model)
        shapes[f"{prefix}.mlp.w3.weight"] = (hidden, d_model)
        shapes[f"{prefix}.mlp.w2.weight"] = (d_model, hidden)
    shapes["ln_f.weight"] = (d_model,)
    shapes["lm_head.weight"] = (vocab_size, d_model)
    return shapes


def _validate_model_state(
    state: object,
    architecture: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    expected = _expected_model_shapes(architecture)
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ProvenanceError(
            "checkpoint model state keys do not match architecture"
        )
    for name, shape in expected.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.layout != torch.strided
            or not torch.isfinite(tensor).all()
        ):
            raise ProvenanceError(
                f"checkpoint model tensor {name} is incompatible"
            )
    return expected


def _validate_optimizer_state(
    state: object,
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    step: int,
    provenance: RunProvenance,
    fused: bool,
) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "state",
        "param_groups",
    }:
        raise ProvenanceError("checkpoint optimizer state fields are invalid")
    groups = state["param_groups"]
    slots = state["state"]
    if (
        not isinstance(groups, list)
        or len(groups) != 2
        or not isinstance(slots, Mapping)
    ):
        raise ProvenanceError("checkpoint optimizer parameter groups are invalid")
    decay_shapes = [
        shape for shape in expected_shapes.values() if len(shape) >= 2
    ]
    no_decay_shapes = [
        shape for shape in expected_shapes.values() if len(shape) < 2
    ]
    ordered_shapes = decay_shapes + no_decay_shapes
    expected_ids = list(range(len(ordered_shapes)))
    group_ids = [group.get("params") for group in groups]
    if group_ids != [
        expected_ids[: len(decay_shapes)],
        expected_ids[len(decay_shapes) :],
    ]:
        raise ProvenanceError(
            "checkpoint optimizer parameters do not match architecture"
        )
    optimizer = provenance.optimizer
    required_group_fields = {
        "weight_decay",
        "lr",
        "betas",
        "eps",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "params",
    }
    for index, group in enumerate(groups):
        group_fields = set(group)
        if (
            not required_group_fields <= group_fields
            or not (
                group_fields - required_group_fields
                <= {"decoupled_weight_decay"}
            )
        ):
            raise ProvenanceError(
                "checkpoint optimizer parameter group fields are invalid"
            )
        if tuple(group.get("betas", ())) != tuple(optimizer["betas"]):
            raise ProvenanceError("checkpoint optimizer betas mismatch")
        if float(group.get("eps", -1)) != float(optimizer["epsilon"]):
            raise ProvenanceError("checkpoint optimizer epsilon mismatch")
        expected_decay = (
            float(optimizer["weight_decay"]) if index == 0 else 0.0
        )
        if float(group.get("weight_decay", -1)) != expected_decay:
            raise ProvenanceError("checkpoint optimizer weight decay mismatch")
        if (
            group["amsgrad"] is not False
            or group["maximize"] is not False
            or group["foreach"] is not None
            or group["capturable"] is not False
            or group["differentiable"] is not False
            or group["fused"] is not fused
            or (
                "decoupled_weight_decay" in group
                and group["decoupled_weight_decay"] is not True
            )
        ):
            raise ProvenanceError(
                "checkpoint optimizer execution mode is incompatible"
            )
    if step == 0:
        if slots:
            raise ProvenanceError("step-zero checkpoint has optimizer moments")
        return
    if set(slots) != set(expected_ids):
        raise ProvenanceError("checkpoint optimizer moments are incomplete")
    for parameter_id, shape in enumerate(ordered_shapes):
        entry = slots[parameter_id]
        if not isinstance(entry, Mapping) or set(entry) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ProvenanceError(
                "checkpoint AdamW state fields are incompatible"
            )
        if (
            not isinstance(entry["step"], torch.Tensor)
            or entry["step"].numel() != 1
            or not torch.isfinite(entry["step"]).all()
            or int(entry["step"].item()) != step
        ):
            raise ProvenanceError("checkpoint optimizer step is invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = entry[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or tensor.layout != torch.strided
                or not torch.isfinite(tensor).all()
            ):
                raise ProvenanceError(
                    f"checkpoint optimizer {name} shape is incompatible"
                )


def _validate_data_state(
    state: object,
    *,
    step: int,
    provenance: RunProvenance,
    token_count: int,
    micro_batch_size: int,
) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "schema_version",
        "cursor",
        "epoch",
        "raw_positions",
    }:
        raise ProvenanceError("checkpoint data state fields are not exact")
    if state["schema_version"] != 2:
        raise ProvenanceError("checkpoint data state schema is incompatible")
    for name in ("cursor", "epoch", "raw_positions"):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProvenanceError(
                f"checkpoint data {name.replace('_', ' ')} is invalid"
            )
    if state["cursor"] > token_count:
        raise ProvenanceError("checkpoint data cursor is outside the corpus")
    expected_positions = step * provenance.tokens_per_step
    if state["raw_positions"] != expected_positions:
        raise ProvenanceError(
            "checkpoint data raw positions do not match optimizer step"
        )
    context = provenance.architecture["ctx"]
    stride = micro_batch_size * context
    span = micro_batch_size * (context + 1)
    accumulation = provenance.tokens_per_step // stride
    batches = step * accumulation
    batches_per_epoch = ((token_count - span - 1) // stride) + 1
    if batches == 0:
        expected_cursor = 0
        expected_epoch = 0
    else:
        expected_epoch = (batches - 1) // batches_per_epoch
        within_epoch = ((batches - 1) % batches_per_epoch) + 1
        expected_cursor = within_epoch * stride
    if state["cursor"] != expected_cursor:
        raise ProvenanceError(
            "checkpoint data cursor does not match optimizer step"
        )
    if state["epoch"] != expected_epoch:
        raise ProvenanceError(
            "checkpoint data epoch does not match optimizer step"
        )


def _validate_checkpoint_state(
    state: object,
    *,
    expected: RunProvenance,
    cfg: Mapping[str, Any],
    token_count: int,
) -> Mapping[str, Any]:
    if not isinstance(state, Mapping) or set(state) != _CHECKPOINT_FIELDS:
        raise ProvenanceError("checkpoint fields are not exact schema version 2")
    if state["schema_version"] != 2:
        raise ProvenanceError("checkpoint schema version is incompatible")
    raw_provenance = state["provenance"]
    if not isinstance(raw_provenance, Mapping) or set(raw_provenance) != (
        _PROVENANCE_FIELDS
    ):
        raise ProvenanceError("checkpoint run provenance fields are not exact")
    expected_raw = expected.to_dict()
    for field in _PROVENANCE_FIELDS:
        if field in {"record_type", "schema_version", "provenance_sha256"}:
            continue
        if raw_provenance[field] != expected_raw[field]:
            raise ProvenanceError(
                "checkpoint provenance "
                f"{field.replace('_', ' ')} mismatch"
            )
    checkpoint_provenance = RunProvenance.from_dict(raw_provenance)
    if checkpoint_provenance.to_dict() != expected_raw:
        raise ProvenanceError("checkpoint run provenance mismatch")
    if _checkpoint_config_identity(
        state["cfg"]
    ) != _checkpoint_config_identity(cfg):
        raise ProvenanceError("checkpoint embedded configuration mismatch")
    step = state["step"]
    if isinstance(step, bool) or not isinstance(step, int) or not (
        0 <= step <= expected.steps
    ):
        raise ProvenanceError("checkpoint optimizer step is invalid")
    scheduler = state["scheduler"]
    if not isinstance(scheduler, Mapping) or set(scheduler) != {
        "step",
        "max_steps",
        "last_lr",
    }:
        raise ProvenanceError("checkpoint scheduler state is invalid")
    if scheduler["step"] != step or scheduler["max_steps"] != expected.steps:
        raise ProvenanceError("checkpoint scheduler position mismatch")
    last_lr = scheduler["last_lr"]
    if (
        not isinstance(last_lr, list)
        or len(last_lr) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in last_lr
        )
    ):
        raise ProvenanceError("checkpoint scheduler learning rates are invalid")
    if step == 0:
        expected_lr = float(expected.optimizer["lr"])
    else:
        expected_lr = cosine_lr(
            step - 1,
            expected.optimizer["lr"],
            expected.scheduler["warmup_steps"],
            expected.steps,
            expected.scheduler["minimum_learning_rate_fraction"],
        )
    if any(float(value) != expected_lr for value in last_lr):
        raise ProvenanceError(
            "checkpoint scheduler learning rate position mismatch"
        )
    if not isinstance(state["opt"], Mapping):
        raise ProvenanceError("checkpoint optimizer state is invalid")
    optimizer_groups = state["opt"].get("param_groups", [])
    if any(
        float(group.get("lr", float("nan"))) != expected_lr
        for group in optimizer_groups
    ):
        raise ProvenanceError(
            "checkpoint optimizer learning rate disagrees with scheduler"
        )
    expected_shapes = _validate_model_state(
        state["model"],
        expected.architecture,
    )
    _validate_optimizer_state(
        state["opt"],
        expected_shapes,
        step=step,
        provenance=expected,
        fused=pick_device(cfg.get("device", "auto")) == "cuda",
    )
    _validate_data_state(
        state["data"],
        step=step,
        provenance=expected,
        token_count=token_count,
        micro_batch_size=cfg["micro_batch_size"],
    )
    for name in ("running_loss", "last_step_loss"):
        value = state[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ProvenanceError(
                f"checkpoint {name.replace('_', ' ')} is invalid"
            )
    _validate_rng_state(
        {
            "rng_python": state["rng_python"],
            "rng_numpy": state["rng_numpy"],
            "rng_torch": state["rng_torch"],
            "rng_cuda": state["rng_cuda"],
            "rng_mps": state["rng_mps"],
        }
    )
    return state


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    descriptor: int | None = None
    try:
        checkpoint = require_regular_file(path, name="checkpoint")
        if not hasattr(os, "O_NOFOLLOW"):
            raise ProvenanceError(
                "checkpoint loading requires O_NOFOLLOW support"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(checkpoint, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ProvenanceError("checkpoint must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            state = torch.load(
                stream,
                map_location="cpu",
                weights_only=True,
            )
        after_read = os.fstat(descriptor)
        current = os.stat(checkpoint, follow_symlinks=False)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(after_read.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (
                after_read.st_dev,
                after_read.st_ino,
                after_read.st_size,
                after_read.st_mtime_ns,
            )
            != opened_identity
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            != opened_identity
        ):
            raise ProvenanceError("checkpoint changed while it was being read")
        return state
    except ProvenanceError:
        raise
    except Exception as exc:
        raise ProvenanceError("checkpoint cannot be loaded safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_log(path: Path, checkpoint_step: int) -> None:
    try:
        content = require_regular_file(path, name="training log").read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ProvenanceError("training log is invalid") from exc
    prior = -1
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProvenanceError("training log contains a partial row") from exc
        step = row.get("step") if isinstance(row, Mapping) else None
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= prior
            or step > checkpoint_step
        ):
            raise ProvenanceError(
                "training log is inconsistent with checkpoint step"
            )
        prior = step


def _validate_resume_inventory(
    out_dir: Path,
    *,
    resume: str,
    expected: RunProvenance,
    cfg: Mapping[str, Any],
    token_count: int,
) -> tuple[Path | None, Mapping[str, Any] | None]:
    if not os.path.lexists(out_dir):
        return None, None
    entries = {entry.name: entry for entry in out_dir.iterdir()}
    if resume == "none":
        if entries:
            raise ProvenanceError(
                "resume=none refuses a nonempty output directory"
            )
        return None, None
    if not entries:
        return None, None
    allowed = {"config.yaml", "ckpt.pt", "log.jsonl", "snapshots"}
    if set(entries) - allowed:
        raise ProvenanceError(
            "partial output contains unexpected training artifacts"
        )
    has_config = "config.yaml" in entries
    has_checkpoint = "ckpt.pt" in entries
    if not (has_config and has_checkpoint):
        raise ProvenanceError(
            "partial output requires both config and checkpoint"
        )
    try:
        stored_config = load_canonical_json(entries["config.yaml"])
    except (OSError, TypeError, ValueError) as exc:
        raise ProvenanceError(
            "output config is missing, partial, or noncanonical"
        ) from exc
    if stored_config != _canonical_config(cfg):
        raise ProvenanceError("output config does not match requested run")
    checkpoint_path = entries["ckpt.pt"]
    state = _validate_checkpoint_state(
        _load_checkpoint(checkpoint_path),
        expected=expected,
        cfg=cfg,
        token_count=token_count,
    )
    snapshots = entries.get("snapshots")
    if snapshots is not None:
        if snapshots.is_symlink() or not snapshots.is_dir():
            raise ProvenanceError(
                "snapshot output must be a regular non-symlink directory"
            )
        for snapshot in snapshots.iterdir():
            match = re.fullmatch(r"step([0-9]{7})\.pt", snapshot.name)
            if (
                match is None
                or snapshot.is_symlink()
                or not snapshot.is_file()
                or int(match.group(1)) > state["step"]
            ):
                raise ProvenanceError(
                    "partial snapshot output is inconsistent with checkpoint"
                )
    if "log.jsonl" in entries:
        _validate_log(entries["log.jsonl"], state["step"])
    return checkpoint_path, state


def validate_run_start(
    cfg: Mapping[str, Any],
    resume: str = "auto",
) -> ValidatedRunStart:
    """Read-only validation of all input, output, and checkpoint identity."""

    if resume not in {"auto", "none"}:
        raise ProvenanceError("resume policy must be auto or none")
    resolved = _canonical_config(cfg)
    corpus = _input_file(resolved.get("train_bin"), "corpus")
    mask = (
        _input_file(resolved["train_mask"], "mask sidecar")
        if resolved.get("train_mask") is not None
        else None
    )
    weights = (
        _input_file(resolved["train_weights"], "weights sidecar")
        if resolved.get("train_weights") is not None
        else None
    )
    resolved["train_bin"] = str(corpus)
    if mask is not None:
        resolved["train_mask"] = str(mask)
    if weights is not None:
        resolved["train_weights"] = str(weights)
    out_dir = _canonical_output_path(resolved.get("out_dir"))
    resolved["out_dir"] = str(out_dir)
    token_count = _validate_sidecar_lengths(corpus, mask, weights)
    corpus_hash = sha256_file(corpus)
    mask_hash = sha256_file(mask) if mask is not None else None
    weights_hash = sha256_file(weights) if weights is not None else None
    provenance = _build_run_provenance(
        resolved,
        corpus_sha256=corpus_hash,
        mask_sha256=mask_hash,
        weights_sha256=weights_hash,
    )
    minimum_batch_tokens = (
        resolved["micro_batch_size"]
        * (provenance.architecture["ctx"] + 1)
    )
    if token_count <= minimum_batch_tokens:
        raise ProvenanceError("corpus is smaller than one packed batch")
    checkpoint_path, checkpoint_state = _validate_resume_inventory(
        out_dir,
        resume=resume,
        expected=provenance,
        cfg=resolved,
        token_count=token_count,
    )
    return ValidatedRunStart(
        cfg=resolved,
        provenance=provenance,
        resume=resume,
        checkpoint_path=checkpoint_path,
        checkpoint_state=checkpoint_state,
    )


class Trainer:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        run_start: ValidatedRunStart | None = None,
        ledger: RunLedger | None = None,
    ):
        start = (
            validate_run_start(cfg, resume="none")
            if run_start is None
            else run_start
        )
        self.cfg = _canonical_config(start.cfg)
        self.run_provenance = RunProvenance.from_dict(
            start.provenance.to_dict()
        )
        self.ledger = ledger
        self.device = pick_device(self.cfg.get("device", "auto"))

        random.seed(self.run_provenance.data_seed)
        np.random.seed(self.run_provenance.data_seed % (2**32))
        torch.manual_seed(self.run_provenance.initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.run_provenance.initialization_seed
            )
        if mps_rng_available():
            torch.mps.manual_seed(self.run_provenance.initialization_seed)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_cfg = _model_config(self.cfg)
        self.model = GPT(model_cfg).to(self.device)
        if self.cfg.get("compile", False) and self.device == "cuda":
            self.model = torch.compile(self.model)

        self.micro_bs = self.cfg["micro_batch_size"]
        self.accum = (
            self.cfg["tokens_per_step"]
            // (self.micro_bs * model_cfg.ctx)
        )
        self.data = PackedShards(
            self.cfg["train_bin"],
            self.cfg.get("train_mask"),
            ctx=model_cfg.ctx,
            batch_size=self.micro_bs,
            device=self.device,
            seed=self.run_provenance.data_seed,
            weights_path=self.cfg.get("train_weights"),
        )
        self.max_steps = self.run_provenance.steps

        optimizer = self.run_provenance.optimizer
        decay, no_decay = [], []
        for _, parameter in self.model.named_parameters():
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        self.opt = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "weight_decay": optimizer["weight_decay"],
                },
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=optimizer["lr"],
            betas=tuple(optimizer["betas"]),
            eps=optimizer["epsilon"],
            fused=self.device == "cuda",
        )

        self.step = 0
        self.running_loss: float | None = None
        self.last_step_loss: float | None = None
        self.out_dir = Path(self.cfg["out_dir"])
        self.snapshot_dir = self.out_dir / "snapshots"
        self.ckpt_path = self.out_dir / "ckpt.pt"
        self.log_path = self.out_dir / "log.jsonl"
        self.config_path = self.out_dir / "config.yaml"
        self.snap_every = max(
            1,
            int(self.max_steps * self.cfg.get("snap_frac", 0.10)),
        )
        self.ckpt_seconds = self.cfg.get("ckpt_minutes", 30) * 60
        self.log_every = self.cfg.get("log_every", 20)
        self.eval_every = self.cfg.get("eval_every", 250)
        self._probe = None
        self._output_ready = False
        self._checkpoint_loaded = False

    def prepare_output(self) -> None:
        if self._output_ready:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.out_dir.is_symlink() or not self.out_dir.is_dir():
            raise ProvenanceError(
                "output directory became invalid after preflight"
            )
        if self.out_dir.resolve(strict=True) != self.out_dir:
            raise ProvenanceError(
                "output directory traverses a symlink after preflight"
            )
        if os.path.lexists(self.snapshot_dir):
            if self.snapshot_dir.is_symlink() or not self.snapshot_dir.is_dir():
                raise ProvenanceError(
                    "snapshot directory is not a regular directory"
                )
        else:
            self.snapshot_dir.mkdir()
        if os.path.lexists(self.config_path):
            try:
                stored = load_canonical_json(self.config_path)
            except (OSError, TypeError, ValueError) as exc:
                raise ProvenanceError(
                    "existing output config is invalid"
                ) from exc
            if stored != self.cfg:
                raise ProvenanceError(
                    "existing output config does not match validated config"
                )
        else:
            atomic_write_bytes(
                self.config_path,
                canonical_json_bytes(self.cfg),
            )
        self._output_ready = True

    # --- checkpointing -----------------------------------------------------

    def _scheduler_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "last_lr": [
                float(group["lr"]) for group in self.opt.param_groups
            ],
        }

    def _checkpoint_state(self) -> dict[str, Any]:
        raw = getattr(self.model, "_orig_mod", self.model)
        rng = capture_rng_state()
        return {
            "schema_version": 2,
            "provenance": self.run_provenance.to_dict(),
            "model": raw.state_dict(),
            "opt": self.opt.state_dict(),
            "data": self.data.state_dict(),
            "step": self.step,
            "scheduler": self._scheduler_state(),
            **rng,
            "cfg": self.cfg,
            "running_loss": self.running_loss,
            "last_step_loss": self.last_step_loss,
        }

    def save_ckpt(self) -> None:
        self.prepare_output()
        state = self._checkpoint_state()
        atomic_write_stream(
            self.ckpt_path,
            lambda stream: torch.save(state, stream),
        )
        if self.ledger is not None:
            self.ledger.append(
                "checkpointed",
                details={
                    "step": self.step,
                    "raw_positions": self.data.raw_positions,
                    "provenance_sha256": (
                        self.run_provenance.provenance_sha256
                    ),
                },
            )

    def load_ckpt(self, path: str | Path | None = None) -> None:
        checkpoint_path = Path(path) if path is not None else self.ckpt_path
        state = _validate_checkpoint_state(
            _load_checkpoint(checkpoint_path),
            expected=self.run_provenance,
            cfg=self.cfg,
            token_count=self.data.n_tokens,
        )
        self._apply_checkpoint_state(state)

    def _apply_checkpoint_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        raw = getattr(self.model, "_orig_mod", self.model)
        raw.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.data.load_state_dict(state["data"])
        self.step = state["step"]
        self.running_loss = state["running_loss"]
        self.last_step_loss = state["last_step_loss"]
        restore_rng_state(
            {
                "rng_python": state["rng_python"],
                "rng_numpy": state["rng_numpy"],
                "rng_torch": state["rng_torch"],
                "rng_cuda": state["rng_cuda"],
                "rng_mps": state["rng_mps"],
            }
        )
        self._checkpoint_loaded = True

    def load_validated_start(self, start: ValidatedRunStart) -> None:
        if start.checkpoint_state is None:
            raise ProvenanceError("validated run start has no checkpoint")
        if start.provenance.to_dict() != self.run_provenance.to_dict():
            raise ProvenanceError(
                "validated checkpoint provenance changed before restore"
            )
        state = _validate_checkpoint_state(
            start.checkpoint_state,
            expected=self.run_provenance,
            cfg=self.cfg,
            token_count=self.data.n_tokens,
        )
        self._apply_checkpoint_state(state)

    def save_snapshot(self) -> None:
        self.prepare_output()
        raw = getattr(self.model, "_orig_mod", self.model)
        destination = self.snapshot_dir / f"step{self.step:07d}.pt"
        if os.path.lexists(destination):
            raise ProvenanceError(
                f"immutable snapshot already exists: {destination.name}"
            )
        snapshot = {
            "schema_version": 2,
            "provenance": self.run_provenance.to_dict(),
            "model": raw.state_dict(),
            "step": self.step,
            "raw_positions": self.data.raw_positions,
            "model_cfg": asdict(raw.cfg),
        }
        atomic_write_stream(
            destination,
            lambda stream: torch.save(snapshot, stream),
        )

    # --- metrics -----------------------------------------------------------

    @torch.no_grad()
    def loss_masked_values(self) -> float | None:
        """CE at loss-masked positions (fact values). The gate-0 mechanism
        metric: stays high in the split arm, falls in the dense arm's bio text."""
        if self._probe is None:
            self._probe = self.data.masked_value_batch() or "none"
        if self._probe == "none":
            return None
        x, y = self._probe
        losses = []
        was_training = self.model.training
        self.model.eval()
        for i in range(0, x.size(0), self.micro_bs):
            xb = x[i : i + self.micro_bs].to(self.device)
            yb = y[i : i + self.micro_bs].to(self.device)
            with self._autocast():
                _, loss = self.model(xb, yb)
            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())
        if was_training:
            self.model.train()
        return sum(losses) / len(losses) if losses else None

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --- loop ----------------------------------------------------------------

    def _append_log(self, row: Mapping[str, Any]) -> None:
        self.prepare_output()
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.log_path, flags, 0o644)
        except OSError as exc:
            raise ProvenanceError("cannot append training log safely") from exc
        try:
            payload = canonical_json_bytes(row)
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise ProvenanceError("training log append was partial")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def train_steps(self, n_steps: int | None = None) -> float:
        if n_steps is not None and (
            isinstance(n_steps, bool)
            or not isinstance(n_steps, int)
            or n_steps < 0
        ):
            raise ValueError("n_steps must be a nonnegative integer")
        self.prepare_output()
        target = self.step + n_steps if n_steps is not None else self.max_steps
        target = min(target, self.max_steps)
        self.model.train()
        last_ckpt = time.time()
        t0 = time.time()
        tokens_seen = 0
        running = self.running_loss
        optimizer = self.run_provenance.optimizer
        scheduler = self.run_provenance.scheduler
        while self.step < target:
            lr = cosine_lr(
                self.step,
                optimizer["lr"],
                scheduler["warmup_steps"],
                self.max_steps,
                scheduler["minimum_learning_rate_fraction"],
            )
            for group in self.opt.param_groups:
                group["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(self.accum):
                if self.cfg.get("train_weights"):
                    x, y, weights = self.data.next_weighted_batch()
                    with self._autocast():
                        _, loss = self.model(x, y, target_weights=weights)
                else:
                    x, y = self.data.next_batch()
                    with self._autocast():
                        _, loss = self.model(x, y)
                (loss / self.accum).backward()
                micro_losses.append(loss.item())
                tokens_seen += x.numel()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                optimizer["gradient_clip"],
            )
            self.opt.step()
            self.step += 1
            step_loss = sum(micro_losses) / len(micro_losses)
            running = step_loss if running is None else 0.95 * running + 0.05 * step_loss
            self.last_step_loss = step_loss
            self.running_loss = running

            if self.step % self.log_every == 0 or self.step == target:
                row = {
                    "step": self.step,
                    "loss": round(step_loss, 4),
                    "loss_ema": round(running, 4),
                    "lr": lr,
                    "tok_s": round(tokens_seen / max(1e-9, time.time() - t0), 1),
                    "epoch": self.data.epoch,
                }
                if self.step % self.eval_every == 0 or self.step == target:
                    mv = self.loss_masked_values()
                    if mv is not None:
                        row["loss_masked_values"] = round(mv, 4)
                self._append_log(row)
                t0 = time.time()
                tokens_seen = 0
            if self.step % self.snap_every == 0:
                self.save_snapshot()
            if time.time() - last_ckpt > self.ckpt_seconds:
                self.save_ckpt()
                last_ckpt = time.time()
        self.save_ckpt()
        return (
            self.running_loss
            if self.running_loss is not None
            else float("nan")
        )


def _prepare_training_ledger(
    start: ValidatedRunStart,
) -> RunLedger | None:
    root = start.cfg.get("ledger_root", start.cfg.get("out_root"))
    if root is None:
        return None
    ledger = RunLedger(root, start.provenance.run_id)
    events = ledger.events()
    details = {
        "provenance_sha256": start.provenance.provenance_sha256,
    }
    checkpoint_steps: list[int] = []
    for event in events:
        if event.details.get("provenance_sha256") != details[
            "provenance_sha256"
        ]:
            raise ProvenanceError(
                "run ledger provenance does not match the validated run"
            )
        if event.event_type == "checkpointed":
            step = event.details.get("step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or step > start.provenance.steps
                or (checkpoint_steps and step < checkpoint_steps[-1])
            ):
                raise ProvenanceError(
                    "run ledger checkpoint positions are invalid"
                )
            checkpoint_steps.append(step)

    original_latest = events[-1].event_type if events else None
    if original_latest is None:
        ledger.append("planned", details=details)
        latest = "planned"
    else:
        latest = original_latest
    if latest in {"completed", "excluded"}:
        raise ProvenanceError(
            f"run ledger is terminal with status {latest}"
        )
    if not start.is_resume and checkpoint_steps:
        raise ProvenanceError(
            "run ledger checkpoint history requires an exact checkpoint resume"
        )
    if latest == "planned":
        ledger.append("preflight_passed", details=details)
        latest = "preflight_passed"
    if latest == "preflight_passed":
        ledger.append("launch_requested", details=details)
        latest = "launch_requested"
    if latest == "launch_requested":
        ledger.append("started", details=details)
        latest = "started"

    if start.is_resume:
        checkpoint_step = start.checkpoint_state["step"]
        recorded_step = checkpoint_steps[-1] if checkpoint_steps else None
        if recorded_step is not None and checkpoint_step < recorded_step:
            raise ProvenanceError(
                "validated checkpoint is behind the run ledger; "
                "rollback is forbidden"
            )
        import_checkpoint = (
            recorded_step is None
            or checkpoint_step > recorded_step
            or latest in {"started", "resumed"}
        )
        if import_checkpoint:
            ledger.append(
                "checkpointed",
                details={
                    **details,
                    "step": checkpoint_step,
                    "imported": True,
                },
            )
            latest = "checkpointed"
        if latest not in {"checkpointed", "failed"}:
            raise ProvenanceError(
                "run ledger cannot resume from its current lifecycle state"
            )
        ledger.append(
            "resumed",
            details={
                **details,
                "step": checkpoint_step,
            },
        )
    else:
        if original_latest == "started":
            ledger.append(
                "failed",
                details={
                    **details,
                    "error_type": "InterruptedRun",
                    "message": "run stopped before publishing a checkpoint",
                },
            )
            latest = "failed"
        if latest == "failed":
            ledger.append("preflight_passed", details=details)
            ledger.append("launch_requested", details=details)
            ledger.append("started", details=details)
        elif original_latest not in {
            None,
            "planned",
            "preflight_passed",
            "launch_requested",
        }:
            raise ProvenanceError(
                "existing run ledger requires an exact checkpoint resume"
            )
    return ledger


def _retain_training_failure(
    ledger: RunLedger | None,
    exc: BaseException,
    *,
    provenance_sha256: str,
) -> None:
    if ledger is None:
        return
    try:
        events = ledger.events()
        if events and events[-1].event_type not in {
            "failed",
            "excluded",
            "completed",
        }:
            ledger.append(
                "failed",
                details={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "provenance_sha256": provenance_sha256,
                },
            )
    except Exception:
        # Preserve the original training failure; ledger validation will expose
        # any independently broken lifecycle on the next invocation.
        return


def train(
    cfg: Mapping[str, Any],
    resume: str = "auto",
) -> Trainer:
    start = validate_run_start(cfg, resume=resume)
    ledger: RunLedger | None = None
    try:
        ledger = _prepare_training_ledger(start)
        trainer = Trainer(start.cfg, run_start=start, ledger=ledger)
        if start.is_resume:
            trainer.load_validated_start(start)
            print(f"resumed from step {trainer.step}")
        trainer.prepare_output()
        trainer.train_steps()
        if ledger is not None:
            ledger.append(
                "completed",
                details={
                    "step": trainer.step,
                    "raw_positions": trainer.data.raw_positions,
                    "provenance_sha256": (
                        trainer.run_provenance.provenance_sha256
                    ),
                },
            )
        return trainer
    except BaseException as exc:
        _retain_training_failure(
            ledger,
            exc,
            provenance_sha256=start.provenance.provenance_sha256,
        )
        raise
