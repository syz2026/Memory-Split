#!/usr/bin/env python
"""Generate the exact portable 35-run relational configuration matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.artifacts import (
    atomic_write_json,
    canonical_sha256,
    load_canonical_json,
    validate_relative_path,
    validate_sha256,
)
from experiment.relational_assets import (
    AssetReceipt,
    StagedAssetSpec,
    load_asset_receipt,
    validate_asset_receipt,
)
from scripts.freeze_relational_study import (
    FIXTURE_WATERMARK,
    FreezeManifest,
    validate_freeze_manifest,
)


MAX_ROUNDING_ERROR = 0.0002
_CONDITIONS = {"dense", "split", "random"}
_LOAD_ROLES = {"low", "high", "confirmation"}
_ARCHITECTURES = {
    "d160m": {
        "n_layer": 12,
        "n_head": 12,
        "d_model": 768,
        "vocab_size": 50_304,
        "ctx": 1_024,
        "parameter_count": 162_220_800,
    },
    "d360m": {
        "n_layer": 20,
        "n_head": 16,
        "d_model": 1_024,
        "vocab_size": 50_304,
        "ctx": 1_024,
        "parameter_count": 356_033_536,
    },
}
_PACKING = {
    "format": "packed-u16-v1",
    "context_length": 1_024,
    "boundary_policy": "shared-record-boundaries-v1",
}
_OPTIMIZER = {
    "name": "adamw",
    "lr": 0.0003,
    "betas": [0.9, 0.95],
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
}
_SCHEDULER = {
    "name": "cosine",
    "warmup_steps": 300,
    "minimum_learning_rate_fraction": 0.1,
}
_RUN_FIELDS = {
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
_MANIFEST_FIELDS = {
    "record_type",
    "schema_version",
    "status",
    "watermark",
    "launchable",
    "freeze",
    "freeze_sha256",
    "freeze_decision_sha256",
    "load_labels",
    "entities",
    "seeds",
    "runs",
    "matrix_plan_sha256",
    "asset_receipt",
    "decision_sha256",
    "manifest_sha256",
}


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {key: freeze(child) for key, child in sorted(item.items())}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, name: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class RoundedBudget:
    requested_raw_positions: int
    actual_raw_positions: int
    steps: int
    rounding_error_fraction: float


def round_raw_positions(
    requested_raw_positions: int,
    *,
    tokens_per_step: int,
) -> RoundedBudget:
    requested = _integer(
        requested_raw_positions,
        "requested raw positions",
        minimum=1,
    )
    per_step = _integer(tokens_per_step, "tokens per step", minimum=1)
    steps = (requested + per_step // 2) // per_step
    steps = max(1, steps)
    actual = steps * per_step
    error = abs(actual - requested) / requested
    if error >= MAX_ROUNDING_ERROR:
        raise ValueError(
            "optimizer-step rounding error must remain below "
            "the frozen 0.02% limit"
        )
    return RoundedBudget(
        requested_raw_positions=requested,
        actual_raw_positions=actual,
        steps=steps,
        rounding_error_fraction=error,
    )


def entity_load_label(entities: int) -> str:
    value = _integer(entities, "entity load", minimum=1)
    if value < 1_000_000 and value % 1_000 == 0:
        return f"n{value // 1_000}k"
    if value >= 1_000_000:
        whole, remainder = divmod(value, 1_000_000)
        if remainder == 0:
            return f"n{whole}m"
        digits = f"{remainder:06d}".rstrip("0")
        return f"n{whole}p{digits}m"
    return f"n{value}"


def _exact_mapping(
    value: object,
    expected: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or _thaw(value) != _thaw(expected):
        raise ValueError(f"{name} does not match the frozen contract")
    return _freeze_mapping(dict(value))


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    status: str
    watermark: str | None
    launchable: bool
    model: str
    condition: str
    load: str
    load_label: str
    entities: int
    seed: int
    initialization_seed: int
    data_seed: int
    parameter_count: int
    architecture: Mapping[str, Any]
    data_rel: str
    weights_rel: str
    out_rel: str
    stream_sha256: str | None
    weights_sha256: str | None
    stream_commitment_sha256: str
    weights_commitment_sha256: str
    freeze_sha256: str
    selected_mixture: tuple[float, float, float]
    tokens_per_parameter: int
    tokens_per_step: int
    requested_raw_positions: int
    actual_raw_positions: int
    steps: int
    rounding_error_fraction: float
    packing: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    decode_budget: int
    pair_fingerprint: str
    config_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.run_id) is None
        ):
            raise ValueError("run_id is not a canonical portable identifier")
        if self.status not in {"fixture", "frozen"}:
            raise ValueError("run status must be fixture or frozen")
        if not isinstance(self.launchable, bool):
            raise ValueError("run launchable must be Boolean")
        if self.status == "fixture":
            if self.watermark != FIXTURE_WATERMARK or self.launchable:
                raise ValueError("fixture run must be explicitly nonlaunchable")
        elif self.watermark is not None or not self.launchable:
            raise ValueError("frozen run launch state is invalid")
        if self.model not in _ARCHITECTURES:
            raise ValueError("run model is not a protected preset")
        if self.condition not in _CONDITIONS:
            raise ValueError("run condition is invalid")
        if self.load not in _LOAD_ROLES:
            raise ValueError("run load role is invalid")
        if self.load_label != entity_load_label(self.entities):
            raise ValueError("run load label does not match entity count")
        for name, value in (
            ("entities", self.entities),
            ("seed", self.seed),
            ("initialization_seed", self.initialization_seed),
            ("data_seed", self.data_seed),
            ("parameter_count", self.parameter_count),
            ("tokens_per_parameter", self.tokens_per_parameter),
            ("tokens_per_step", self.tokens_per_step),
            ("requested_raw_positions", self.requested_raw_positions),
            ("actual_raw_positions", self.actual_raw_positions),
            ("steps", self.steps),
            ("decode_budget", self.decode_budget),
        ):
            _integer(value, name, minimum=1)
        if (
            self.seed != self.initialization_seed
            or self.seed != self.data_seed
        ):
            raise ValueError("paired initialization and data seeds must match")
        expected_architecture = _ARCHITECTURES[self.model]
        if self.parameter_count != expected_architecture["parameter_count"]:
            raise ValueError("run parameter count does not match model")
        object.__setattr__(
            self,
            "architecture",
            _exact_mapping(
                self.architecture,
                expected_architecture,
                "architecture",
            ),
        )
        for field in ("data_rel", "weights_rel", "out_rel"):
            validate_relative_path(getattr(self, field), field)
        if self.status == "fixture":
            if self.stream_sha256 is not None or self.weights_sha256 is not None:
                raise ValueError(
                    "fixture run cannot label recipe commitments as file hashes"
                )
        else:
            validate_sha256(self.stream_sha256, "stream SHA-256")
            validate_sha256(self.weights_sha256, "weights SHA-256")
        validate_sha256(
            self.stream_commitment_sha256,
            "stream commitment SHA-256",
        )
        validate_sha256(
            self.weights_commitment_sha256,
            "weights commitment SHA-256",
        )
        validate_sha256(self.freeze_sha256, "freeze SHA-256")
        mixture = tuple(self.selected_mixture)
        if len(mixture) != 3 or sum(mixture) != 1.0:
            raise ValueError("selected mixture must contain three shares")
        object.__setattr__(self, "selected_mixture", mixture)
        if self.tokens_per_parameter not in {10, 20}:
            raise ValueError("tokens per parameter must be 10 or 20")
        expected_requested = self.parameter_count * self.tokens_per_parameter
        if self.requested_raw_positions != expected_requested:
            raise ValueError("requested raw positions do not match budget")
        budget = round_raw_positions(
            self.requested_raw_positions,
            tokens_per_step=self.tokens_per_step,
        )
        if (
            self.actual_raw_positions != budget.actual_raw_positions
            or self.steps != budget.steps
            or self.rounding_error_fraction != budget.rounding_error_fraction
        ):
            raise ValueError("rounded optimizer-step budget is inconsistent")
        object.__setattr__(
            self,
            "packing",
            _exact_mapping(self.packing, _PACKING, "packing"),
        )
        object.__setattr__(
            self,
            "optimizer",
            _exact_mapping(self.optimizer, _OPTIMIZER, "optimizer"),
        )
        object.__setattr__(
            self,
            "scheduler",
            _exact_mapping(self.scheduler, _SCHEDULER, "scheduler"),
        )
        if self.decode_budget != 6:
            raise ValueError("decode budget must remain six")
        validate_sha256(self.pair_fingerprint, "pair fingerprint")
        validate_sha256(self.config_sha256, "configuration SHA-256")

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.model, self.condition, self.load_label, self.seed

    def relative_paths(self) -> tuple[str, str, str]:
        return self.data_rel, self.weights_rel, self.out_rel

    def pair_material(self) -> dict[str, Any]:
        return {
            "record_type": "relational_pair_fingerprint",
            "schema_version": 1,
            "architecture": _thaw(self.architecture),
            "initialization_seed": self.initialization_seed,
            "data_seed": self.data_seed,
            "stream_sha256": self.stream_sha256,
            "stream_commitment_sha256": self.stream_commitment_sha256,
            "packing": _thaw(self.packing),
            "optimizer": _thaw(self.optimizer),
            "scheduler": _thaw(self.scheduler),
            "steps": self.steps,
            "raw_positions": self.actual_raw_positions,
            "decode_budget": self.decode_budget,
        }

    def _without_config_hash(self) -> dict[str, Any]:
        return {
            "record_type": "relational_run_config",
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "watermark": self.watermark,
            "launchable": self.launchable,
            "model": self.model,
            "condition": self.condition,
            "load": self.load_label,
            "load_role": self.load,
            "entities": self.entities,
            "seed": self.seed,
            "initialization_seed": self.initialization_seed,
            "data_seed": self.data_seed,
            "parameter_count": self.parameter_count,
            "architecture": _thaw(self.architecture),
            "data_rel": self.data_rel,
            "weights_rel": self.weights_rel,
            "out_rel": self.out_rel,
            "stream_sha256": self.stream_sha256,
            "weights_sha256": self.weights_sha256,
            "stream_commitment_sha256": self.stream_commitment_sha256,
            "weights_commitment_sha256": self.weights_commitment_sha256,
            "freeze_sha256": self.freeze_sha256,
            "selected_mixture": list(self.selected_mixture),
            "tokens_per_parameter": self.tokens_per_parameter,
            "tokens_per_step": self.tokens_per_step,
            "requested_raw_positions": self.requested_raw_positions,
            "actual_raw_positions": self.actual_raw_positions,
            "steps": self.steps,
            "rounding_error_fraction": self.rounding_error_fraction,
            "packing": _thaw(self.packing),
            "optimizer": _thaw(self.optimizer),
            "scheduler": _thaw(self.scheduler),
            "decode_budget": self.decode_budget,
            "pair_fingerprint": self.pair_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._without_config_hash(),
            "config_sha256": self.config_sha256,
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunConfig":
        if not isinstance(raw, Mapping) or set(raw) != _RUN_FIELDS:
            raise ValueError("run config fields are not exact")
        if (
            raw["record_type"] != "relational_run_config"
            or raw["schema_version"] != 1
            or not isinstance(raw["selected_mixture"], list)
        ):
            raise ValueError("run config protocol mismatch")
        config = cls(
            run_id=raw["run_id"],
            status=raw["status"],
            watermark=raw["watermark"],
            launchable=raw["launchable"],
            model=raw["model"],
            condition=raw["condition"],
            load=raw["load_role"],
            load_label=raw["load"],
            entities=raw["entities"],
            seed=raw["seed"],
            initialization_seed=raw["initialization_seed"],
            data_seed=raw["data_seed"],
            parameter_count=raw["parameter_count"],
            architecture=raw["architecture"],
            data_rel=raw["data_rel"],
            weights_rel=raw["weights_rel"],
            out_rel=raw["out_rel"],
            stream_sha256=raw["stream_sha256"],
            weights_sha256=raw["weights_sha256"],
            stream_commitment_sha256=raw["stream_commitment_sha256"],
            weights_commitment_sha256=raw["weights_commitment_sha256"],
            freeze_sha256=raw["freeze_sha256"],
            selected_mixture=tuple(raw["selected_mixture"]),
            tokens_per_parameter=raw["tokens_per_parameter"],
            tokens_per_step=raw["tokens_per_step"],
            requested_raw_positions=raw["requested_raw_positions"],
            actual_raw_positions=raw["actual_raw_positions"],
            steps=raw["steps"],
            rounding_error_fraction=raw["rounding_error_fraction"],
            packing=raw["packing"],
            optimizer=raw["optimizer"],
            scheduler=raw["scheduler"],
            decode_budget=raw["decode_budget"],
            pair_fingerprint=raw["pair_fingerprint"],
            config_sha256=raw["config_sha256"],
        )
        if config.pair_fingerprint != canonical_sha256(
            config.pair_material()
        ):
            raise ValueError("run pair fingerprint mismatch")
        if config.config_sha256 != canonical_sha256(
            config._without_config_hash()
        ):
            raise ValueError("run config hash mismatch")
        return config


def _run_specs(
    freeze: FreezeManifest,
) -> tuple[tuple[str, str, str, int, int], ...]:
    specs: list[tuple[str, str, str, int, int]] = []
    loads = (
        ("d160m", "low", freeze.low_entities, ("dense", "split")),
        (
            "d160m",
            "high",
            freeze.high_entities,
            ("dense", "split", "random"),
        ),
        (
            "d360m",
            "confirmation",
            freeze.confirmation_entities,
            ("dense", "split"),
        ),
    )
    for model, load, entities, conditions in loads:
        for condition in conditions:
            for seed in freeze.seeds:
                specs.append((model, condition, load, entities, seed))
    return tuple(specs)


def _stream_commitment_sha256(
    freeze: FreezeManifest,
    model: str,
    load: str,
    entities: int,
    seed: int,
) -> str:
    requested = freeze.model_parameters[model] * freeze.tokens_per_parameter
    budget = round_raw_positions(
        requested,
        tokens_per_step=freeze.tokens_per_step,
    )
    return canonical_sha256(
        {
            "record_type": "relational_stream_commitment",
            "schema_version": 1,
            "freeze_sha256": freeze.freeze_sha256,
            "corpus_recipe_sha256": freeze.artifact_sha256["corpus_recipe"],
            "model": model,
            "load_role": load,
            "entities": entities,
            "data_seed": seed,
            "raw_positions": budget.actual_raw_positions,
        }
    )


def _weights_commitment_sha256(
    freeze: FreezeManifest,
    *,
    stream_commitment_sha256: str,
    condition: str,
) -> str:
    recipe = freeze.artifact_sha256[f"{condition}_sidecar_recipe"]
    return canonical_sha256(
        {
            "record_type": "relational_sidecar_commitment",
            "schema_version": 1,
            # Preserve the frozen Task-9 commitment preimage byte-for-byte.
            "stream_sha256": stream_commitment_sha256,
            "condition": condition,
            "sidecar_recipe_sha256": recipe,
        }
    )


def _planned_run_material(
    freeze: FreezeManifest,
    model: str,
    condition: str,
    load: str,
    entities: int,
    seed: int,
) -> dict[str, Any]:
    load_label = entity_load_label(entities)
    requested = freeze.model_parameters[model] * freeze.tokens_per_parameter
    budget = round_raw_positions(
        requested,
        tokens_per_step=freeze.tokens_per_step,
    )
    base_dir = f"relational/{model}/{load_label}/seed-{seed}"
    stream_commitment = _stream_commitment_sha256(
        freeze,
        model,
        load,
        entities,
        seed,
    )
    weights_commitment = _weights_commitment_sha256(
        freeze,
        stream_commitment_sha256=stream_commitment,
        condition=condition,
    )
    return {
        "record_type": "relational_planned_run",
        "schema_version": 1,
        "run_id": f"{model}-{load_label}-{condition}-s{seed}",
        "model": model,
        "condition": condition,
        "load_role": load,
        "load": load_label,
        "entities": entities,
        "seed": seed,
        "initialization_seed": seed,
        "data_seed": seed,
        "parameter_count": freeze.model_parameters[model],
        "architecture": _ARCHITECTURES[model],
        "data_rel": f"{base_dir}/train.bin",
        "weights_rel": f"{base_dir}/{condition}.weights.bin",
        "out_rel": f"relational/{model}/{load_label}/{condition}/seed-{seed}",
        "stream_commitment_sha256": stream_commitment,
        "weights_commitment_sha256": weights_commitment,
        "freeze_sha256": freeze.freeze_sha256,
        "selected_mixture": list(freeze.selected_mixture),
        "tokens_per_parameter": freeze.tokens_per_parameter,
        "tokens_per_step": freeze.tokens_per_step,
        "requested_raw_positions": requested,
        "actual_raw_positions": budget.actual_raw_positions,
        "steps": budget.steps,
        "rounding_error_fraction": budget.rounding_error_fraction,
        "packing": _PACKING,
        "optimizer": _OPTIMIZER,
        "scheduler": _SCHEDULER,
        "decode_budget": freeze.decode_budget,
    }


def _matrix_plan_material(freeze: FreezeManifest) -> dict[str, Any]:
    return {
        "record_type": "relational_run_matrix_plan",
        "schema_version": 1,
        "freeze_sha256": freeze.freeze_sha256,
        "freeze_decision_sha256": freeze.decision_sha256,
        "runs": [
            _planned_run_material(freeze, *spec)
            for spec in _run_specs(freeze)
        ],
    }


def matrix_plan_sha256(
    freeze: FreezeManifest | Mapping[str, Any],
) -> str:
    validated = validate_freeze_manifest(freeze)
    return canonical_sha256(_matrix_plan_material(validated))


def protected_build_metadata(
    freeze: FreezeManifest | Mapping[str, Any],
    *,
    model: str,
    load: str,
    entities: int,
    seed: int,
) -> dict[str, Any]:
    validated = validate_freeze_manifest(freeze)
    identity = (model, load, entities, seed)
    if identity not in {
        (item_model, item_load, item_entities, item_seed)
        for item_model, _condition, item_load, item_entities, item_seed
        in _run_specs(validated)
    }:
        raise ValueError("protected build is not in the frozen matrix plan")
    stream_commitment = _stream_commitment_sha256(
        validated,
        model,
        load,
        entities,
        seed,
    )
    requested = validated.model_parameters[model] * validated.tokens_per_parameter
    budget = round_raw_positions(
        requested,
        tokens_per_step=validated.tokens_per_step,
    )
    return {
        "record_type": "relational_protected_build",
        "schema_version": 1,
        "freeze_sha256": validated.freeze_sha256,
        "matrix_plan_sha256": matrix_plan_sha256(validated),
        "model": model,
        "load_role": load,
        "entities": entities,
        "data_seed": seed,
        "raw_positions": budget.actual_raw_positions,
        "stream_commitment_sha256": stream_commitment,
        "weights_commitment_sha256": {
            condition: _weights_commitment_sha256(
                validated,
                stream_commitment_sha256=stream_commitment,
                condition=condition,
            )
            for condition in ("dense", "random", "split")
        },
    }


def build_asset_specs(
    freeze: FreezeManifest | Mapping[str, Any],
) -> tuple[StagedAssetSpec, ...]:
    validated = validate_freeze_manifest(freeze)
    specs: list[StagedAssetSpec] = []
    for run_spec in _run_specs(validated):
        model, condition, load, entities, seed = run_spec
        plan = _planned_run_material(validated, *run_spec)
        build_rel = Path(plan["data_rel"]).parent.as_posix()
        metadata = protected_build_metadata(
            validated,
            model=model,
            load=load,
            entities=entities,
            seed=seed,
        )
        specs.extend(
            (
                StagedAssetSpec(
                    kind="stream",
                    path=plan["data_rel"],
                    commitment_sha256=plan["stream_commitment_sha256"],
                    build_rel=build_rel,
                    build_metadata=metadata,
                ),
                StagedAssetSpec(
                    kind="weights",
                    path=plan["weights_rel"],
                    commitment_sha256=plan["weights_commitment_sha256"],
                    build_rel=build_rel,
                    build_metadata=metadata,
                ),
            )
        )
    return tuple(specs)


def _make_run(
    freeze: FreezeManifest,
    model: str,
    condition: str,
    load: str,
    entities: int,
    seed: int,
    *,
    asset_hashes: Mapping[tuple[str, str], str] | None = None,
) -> RunConfig:
    plan = _planned_run_material(
        freeze,
        model,
        condition,
        load,
        entities,
        seed,
    )
    stream_sha256 = (
        None
        if asset_hashes is None
        else asset_hashes[("stream", plan["data_rel"])]
    )
    weights_sha256 = (
        None
        if asset_hashes is None
        else asset_hashes[("weights", plan["weights_rel"])]
    )
    provisional = RunConfig(
        run_id=plan["run_id"],
        status=freeze.status,
        watermark=freeze.watermark,
        launchable=freeze.launchable,
        model=model,
        condition=condition,
        load=load,
        load_label=plan["load"],
        entities=entities,
        seed=seed,
        initialization_seed=seed,
        data_seed=seed,
        parameter_count=freeze.model_parameters[model],
        architecture=plan["architecture"],
        data_rel=plan["data_rel"],
        weights_rel=plan["weights_rel"],
        out_rel=plan["out_rel"],
        stream_sha256=stream_sha256,
        weights_sha256=weights_sha256,
        stream_commitment_sha256=plan["stream_commitment_sha256"],
        weights_commitment_sha256=plan["weights_commitment_sha256"],
        freeze_sha256=freeze.freeze_sha256,
        selected_mixture=freeze.selected_mixture,
        tokens_per_parameter=freeze.tokens_per_parameter,
        tokens_per_step=freeze.tokens_per_step,
        requested_raw_positions=plan["requested_raw_positions"],
        actual_raw_positions=plan["actual_raw_positions"],
        steps=plan["steps"],
        rounding_error_fraction=plan["rounding_error_fraction"],
        packing=_PACKING,
        optimizer=_OPTIMIZER,
        scheduler=_SCHEDULER,
        decode_budget=freeze.decode_budget,
        pair_fingerprint="0" * 64,
        config_sha256="0" * 64,
    )
    with_pair = replace(
        provisional,
        pair_fingerprint=canonical_sha256(provisional.pair_material()),
    )
    return replace(
        with_pair,
        config_sha256=canonical_sha256(with_pair._without_config_hash()),
    )


def _build_runs(
    freeze: FreezeManifest,
    *,
    asset_hashes: Mapping[tuple[str, str], str] | None = None,
) -> tuple[RunConfig, ...]:
    return tuple(
        _make_run(freeze, *spec, asset_hashes=asset_hashes)
        for spec in _run_specs(freeze)
    )


@dataclass(frozen=True)
class RunManifest:
    status: str
    watermark: str | None
    launchable: bool
    freeze: FreezeManifest
    freeze_sha256: str
    freeze_decision_sha256: str
    load_labels: Mapping[str, str]
    entities: Mapping[str, int]
    seeds: tuple[int, ...]
    runs: tuple[RunConfig, ...]
    matrix_plan_sha256: str
    asset_receipt: AssetReceipt | None
    decision_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.status != self.freeze.status:
            raise ValueError("run manifest status does not match freeze")
        if self.watermark != self.freeze.watermark:
            raise ValueError("run manifest watermark does not match freeze")
        if self.launchable != self.freeze.launchable:
            raise ValueError(
                "run manifest launchable state does not match freeze"
            )
        validate_sha256(self.freeze_sha256, "manifest freeze SHA-256")
        validate_sha256(
            self.freeze_decision_sha256,
            "manifest freeze decision SHA-256",
        )
        labels = dict(self.load_labels)
        entities = dict(self.entities)
        if set(labels) != _LOAD_ROLES or set(entities) != _LOAD_ROLES:
            raise ValueError("run manifest load roles are not exact")
        for role in _LOAD_ROLES:
            if labels[role] != entity_load_label(entities[role]):
                raise ValueError("run manifest load label mismatch")
        object.__setattr__(
            self,
            "load_labels",
            MappingProxyType(dict(sorted(labels.items()))),
        )
        object.__setattr__(
            self,
            "entities",
            MappingProxyType(dict(sorted(entities.items()))),
        )
        seeds = tuple(self.seeds)
        if seeds != self.freeze.seeds:
            raise ValueError("run manifest seeds do not match freeze")
        object.__setattr__(self, "seeds", seeds)
        runs = tuple(self.runs)
        if len(runs) != 35 or any(
            not isinstance(run, RunConfig) for run in runs
        ):
            raise ValueError("run manifest must contain exactly 35 configs")
        object.__setattr__(self, "runs", runs)
        validate_sha256(
            self.matrix_plan_sha256,
            "manifest matrix plan SHA-256",
        )
        expected_plan_sha256 = matrix_plan_sha256(self.freeze)
        if self.matrix_plan_sha256 != expected_plan_sha256:
            raise ValueError("run manifest matrix plan hash mismatch")
        if self.launchable:
            if self.asset_receipt is None:
                raise ValueError(
                    "launchable run manifest requires an asset receipt"
                )
            receipt = validate_asset_receipt(
                self.asset_receipt,
                freeze_sha256=self.freeze_sha256,
                matrix_plan_sha256=self.matrix_plan_sha256,
                specs=build_asset_specs(self.freeze),
            )
            object.__setattr__(self, "asset_receipt", receipt)
        elif self.asset_receipt is not None:
            raise ValueError(
                "nonlaunchable fixture cannot contain an asset receipt"
            )
        validate_sha256(self.decision_sha256, "manifest decision SHA-256")
        validate_sha256(self.manifest_sha256, "run manifest SHA-256")

    def _decision_material(self) -> dict[str, Any]:
        return {
            "record_type": "relational_run_matrix_decision",
            "schema_version": 1,
            "freeze_sha256": self.freeze_sha256,
            "freeze_decision_sha256": self.freeze_decision_sha256,
            "status": self.status,
            "load_labels": dict(self.load_labels),
            "entities": dict(self.entities),
            "seeds": list(self.seeds),
            "matrix_plan_sha256": self.matrix_plan_sha256,
            "asset_receipt_sha256": (
                None
                if self.asset_receipt is None
                else self.asset_receipt.receipt_sha256
            ),
            "run_config_sha256": [
                run.config_sha256 for run in self.runs
            ],
        }

    def _without_manifest_hash(self) -> dict[str, Any]:
        return {
            "record_type": "relational_run_manifest",
            "schema_version": 1,
            "status": self.status,
            "watermark": self.watermark,
            "launchable": self.launchable,
            "freeze": self.freeze.to_dict(),
            "freeze_sha256": self.freeze_sha256,
            "freeze_decision_sha256": self.freeze_decision_sha256,
            "load_labels": dict(self.load_labels),
            "entities": dict(self.entities),
            "seeds": list(self.seeds),
            "runs": [run.to_dict() for run in self.runs],
            "matrix_plan_sha256": self.matrix_plan_sha256,
            "asset_receipt": (
                None
                if self.asset_receipt is None
                else self.asset_receipt.to_dict()
            ),
            "decision_sha256": self.decision_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._without_manifest_hash(),
            "manifest_sha256": self.manifest_sha256,
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(raw, Mapping) or set(raw) != _MANIFEST_FIELDS:
            raise ValueError("run manifest fields are not exact")
        if (
            raw["record_type"] != "relational_run_manifest"
            or raw["schema_version"] != 1
            or not isinstance(raw["runs"], list)
            or not isinstance(raw["seeds"], list)
        ):
            raise ValueError("run manifest protocol mismatch")
        freeze = validate_freeze_manifest(raw["freeze"])
        manifest = cls(
            status=raw["status"],
            watermark=raw["watermark"],
            launchable=raw["launchable"],
            freeze=freeze,
            freeze_sha256=raw["freeze_sha256"],
            freeze_decision_sha256=raw["freeze_decision_sha256"],
            load_labels=raw["load_labels"],
            entities=raw["entities"],
            seeds=tuple(raw["seeds"]),
            runs=tuple(RunConfig.from_dict(run) for run in raw["runs"]),
            matrix_plan_sha256=raw["matrix_plan_sha256"],
            asset_receipt=(
                None
                if raw["asset_receipt"] is None
                else AssetReceipt.from_dict(raw["asset_receipt"])
            ),
            decision_sha256=raw["decision_sha256"],
            manifest_sha256=raw["manifest_sha256"],
        )
        if (
            manifest.freeze_sha256 != freeze.freeze_sha256
            or manifest.freeze_decision_sha256 != freeze.decision_sha256
        ):
            raise ValueError("run manifest freeze hash mismatch")
        asset_hashes = (
            None
            if manifest.asset_receipt is None
            else {
                (asset.kind, asset.path): asset.sha256
                for asset in manifest.asset_receipt.assets
            }
        )
        expected_runs = _build_runs(freeze, asset_hashes=asset_hashes)
        if [run.to_dict() for run in manifest.runs] != [
            run.to_dict() for run in expected_runs
        ]:
            raise ValueError("run configs do not match the frozen matrix")
        assert_pair_fingerprints(manifest)
        if manifest.decision_sha256 != canonical_sha256(
            manifest._decision_material()
        ):
            raise ValueError("run manifest decision hash mismatch")
        if manifest.manifest_sha256 != canonical_sha256(
            manifest._without_manifest_hash()
        ):
            raise ValueError("run manifest hash mismatch")
        return manifest


def assert_pair_fingerprints(manifest: RunManifest) -> None:
    if not isinstance(manifest, RunManifest):
        raise TypeError("pair validation requires a RunManifest")
    grouped: dict[tuple[str, str, int], list[RunConfig]] = defaultdict(list)
    for run in manifest.runs:
        grouped[(run.model, run.load, run.seed)].append(run)
    expected_group_count = 3 * len(manifest.seeds)
    if len(grouped) != expected_group_count:
        raise ValueError("paired run grouping is incomplete")
    for (model, load, _seed), runs in grouped.items():
        expected_conditions = (
            {"dense", "split", "random"}
            if model == "d160m" and load == "high"
            else {"dense", "split"}
        )
        if {run.condition for run in runs} != expected_conditions:
            raise ValueError("paired condition set is incomplete")
        if len({run.pair_fingerprint for run in runs}) != 1:
            raise ValueError("pair fingerprints do not match")
        if any(
            run.pair_fingerprint != canonical_sha256(run.pair_material())
            for run in runs
        ):
            raise ValueError("pair fingerprint hash mismatch")
        baseline = runs[0].pair_material()
        if any(run.pair_material() != baseline for run in runs[1:]):
            raise ValueError("paired scientific configuration mismatch")


def build_manifest(
    freeze: FreezeManifest | Mapping[str, Any],
    *,
    asset_receipt: AssetReceipt | Mapping[str, Any] | None = None,
) -> RunManifest:
    validated_freeze = validate_freeze_manifest(freeze)
    plan_sha256 = matrix_plan_sha256(validated_freeze)
    if validated_freeze.launchable and asset_receipt is None:
        raise ValueError(
            "launchable run manifest requires a post-build asset receipt"
        )
    if not validated_freeze.launchable and asset_receipt is not None:
        raise ValueError("fixture matrix cannot consume an asset receipt")
    validated_receipt = (
        None
        if asset_receipt is None
        else validate_asset_receipt(
            asset_receipt,
            freeze_sha256=validated_freeze.freeze_sha256,
            matrix_plan_sha256=plan_sha256,
            specs=build_asset_specs(validated_freeze),
        )
    )
    asset_hashes = (
        None
        if validated_receipt is None
        else {
            (asset.kind, asset.path): asset.sha256
            for asset in validated_receipt.assets
        }
    )
    runs = _build_runs(validated_freeze, asset_hashes=asset_hashes)
    load_labels = {
        "low": entity_load_label(validated_freeze.low_entities),
        "high": entity_load_label(validated_freeze.high_entities),
        "confirmation": entity_load_label(
            validated_freeze.confirmation_entities
        ),
    }
    entities = {
        "low": validated_freeze.low_entities,
        "high": validated_freeze.high_entities,
        "confirmation": validated_freeze.confirmation_entities,
    }
    provisional = RunManifest(
        status=validated_freeze.status,
        watermark=validated_freeze.watermark,
        launchable=validated_freeze.launchable,
        freeze=validated_freeze,
        freeze_sha256=validated_freeze.freeze_sha256,
        freeze_decision_sha256=validated_freeze.decision_sha256,
        load_labels=load_labels,
        entities=entities,
        seeds=validated_freeze.seeds,
        runs=runs,
        matrix_plan_sha256=plan_sha256,
        asset_receipt=validated_receipt,
        decision_sha256="0" * 64,
        manifest_sha256="0" * 64,
    )
    with_decision = replace(
        provisional,
        decision_sha256=canonical_sha256(provisional._decision_material()),
    )
    manifest = replace(
        with_decision,
        manifest_sha256=canonical_sha256(
            with_decision._without_manifest_hash()
        ),
    )
    return RunManifest.from_dict(manifest.to_dict())


def require_launchable(manifest: RunManifest) -> RunManifest:
    if not isinstance(manifest, RunManifest):
        raise TypeError("launch validation requires a RunManifest")
    validated = RunManifest.from_dict(manifest.to_dict())
    if not validated.launchable:
        raise ValueError("nonlaunchable fixture cannot be launched")
    return validated


def write_run_manifest(
    path: str | Path,
    manifest: RunManifest,
) -> Path:
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("run manifest destination cannot be a symlink")
    if os.path.lexists(destination):
        raise FileExistsError(
            f"run manifest destination already exists: {destination}"
        )
    validated = RunManifest.from_dict(manifest.to_dict())
    return atomic_write_json(destination, validated.to_dict())


def load_run_manifest(path: str | Path) -> RunManifest:
    return RunManifest.from_dict(load_canonical_json(path))


def publish_run_configs(output: str | Path, manifest: RunManifest) -> Path:
    root = Path(output)
    if root.is_symlink() or os.path.lexists(root):
        raise FileExistsError(f"config output already exists: {root}")
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise ValueError("config output parent must be a regular directory")
    root.mkdir()
    try:
        configs = root / "configs"
        configs.mkdir()
        write_run_manifest(root / "run-manifest.json", manifest)
        if manifest.asset_receipt is not None:
            atomic_write_json(
                root / "asset-receipt.json",
                manifest.asset_receipt.to_dict(),
            )
        for run in manifest.runs:
            atomic_write_json(configs / f"{run.run_id}.json", run.to_dict())
    except BaseException:
        import shutil

        shutil.rmtree(root)
        raise
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the exact 35-run relational matrix."
    )
    parser.add_argument("--freeze", required=True)
    parser.add_argument(
        "--asset-receipt",
        help="canonical post-build receipt required for launchable freezes",
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from scripts.freeze_relational_study import load_freeze_manifest

    freeze = load_freeze_manifest(args.freeze)
    if freeze.launchable:
        from experiment.provenance import verify_source_provenance

        verify_source_provenance(
            Path(__file__).resolve().parents[1],
            freeze.source_provenance,
        )
    receipt = (
        None
        if args.asset_receipt is None
        else load_asset_receipt(args.asset_receipt)
    )
    manifest = build_manifest(freeze, asset_receipt=receipt)
    publish_run_configs(args.out, manifest)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
