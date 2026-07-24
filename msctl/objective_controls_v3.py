"""Strict prospective V3 objective-control contracts.

This module is intentionally standalone. It parses the append-only amendment,
its eight 29M development configs, and future admission evidence without
registering any lifecycle, launch, evaluation, or study-lock behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from .errors import MsctlError


AMENDMENT_ID = "memorysplit-v3-objective-controls-amendment-1"
MANIFEST_ID = "memorysplit-v3-objective-controls-29m"
SOURCE_COMMIT = "b3471e0969ca2a997d33acf60d2e777720afa1c4"
PREREGISTRATION_SHA256 = (
    "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
)
COHORT_ASSIGNMENT_SHA256 = (
    "47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c"
)
AMENDMENT_SHA256 = (
    "376e6a2234fc89aac3baea52e09d40ac08f841102e33fde9daf473a4ca589bf8"
)
MANIFEST_SHA256 = (
    "a03194591977fccaec4acd0c74c2bc13484f40854208948c7172f2bea4fe8781"
)
CONFIG_SHA256S = (
    (
        "configs/29m-v3/full_corpus_dense.yaml",
        "653d65470f862babe78cbeda9e6200f0eaa98428435fd5b6935f115e3f59d472",
    ),
    (
        "configs/29m-v3/full_corpus_split90.yaml",
        "54042c4964ba5a6614827f1dda45381c8d86203f5a69c31fb9bdb753ec871664",
    ),
    (
        "configs/29m-v3/no_arc_conceptarc_dense.yaml",
        "fe3aabae2e32f24f20f128ac6e880cca8dc4450b6b99ab6df418b8a740de6494",
    ),
    (
        "configs/29m-v3/no_arc_conceptarc_split90.yaml",
        "f86fec7060f0f157dd95cf46e25825ddb733ad45cd779d2e2637cb10b4dac2ad",
    ),
    (
        "configs/29m-v3/no_refinement_dense.yaml",
        "fbb95b381e46947df76b02511adca053b16c7b0c153a9f5fd9b1b2098e782a90",
    ),
    (
        "configs/29m-v3/no_refinement_split90.yaml",
        "6269209cd552c2421f782139f6168da371a5a5500fdf38007e2f9b176edff61b",
    ),
    (
        "configs/29m-v3/full_corpus_random_fact90.yaml",
        "1df281116cb31430dcef63feb5da761506753ecce220a54add24632660cfe96e",
    ),
    (
        "configs/29m-v3/full_corpus_matched_nonfactual_mask.yaml",
        "3c0bd664988c682b251e5bae9a2902efa138946726cc40eb3fc33f85e4ae975a",
    ),
)
OBJECTIVE_CONTROLS_CONTRACT_SHA256 = (
    "222844dbf68ad9ca48be2069b5eb3b771b90166252af7eb0a4a5d8db3631adb6"
)
PROTECTED_COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
PROTECTED_MODEL_PARAMETERS = 356_033_536
MODEL_PARAMETERS = 28_969_216
TARGETS_PER_UPDATE = 524_288
OPTIMIZER_STEPS = 1_106
RAW_TARGET_TOKENS = 579_862_528
SHARED_SEED = 0
SHARED_INITIALIZATION_ID = "memorysplit-v3-29m-shared-init-s0"
PRIMARY_CELLS = (
    "graph__composition_ood",
    "graph__joint_ood",
    "non_path__composition_ood",
    "non_path__joint_ood",
)
STATISTICAL_EXCLUSIONS = (
    "primary_exact_test",
    "fixed_checkpoint_aulc",
    "practical_equivalence_test",
    "continuation_decisions",
    "effect_direction_gate",
)
CHECKPOINT_RESUME_MAX_ABS_DELTA = 1e-5
LANGUAGE_MAX_RELATIVE_DEGRADATION = 0.01
LEARNABILITY_FLOOR = 0.75

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_PLACEHOLDER = re.compile(r"\$|%\{|{{")
_AMENDMENT_FILENAME = "objective-controls-amendment-v3.yaml"
_MANIFEST_RELATIVE = "configs/29m-v3/manifest.json"
_PREREGISTRATION_RELATIVE = "configs/preregistration-v3.yaml"
_COHORT_RELATIVE = "configs/cohort-assignment-v3.json"
_CONFIG_FIELDS = {
    "schema_version",
    "amendment_id",
    "manifest_id",
    "run_id",
    "role",
    "pair_id",
    "condition",
    "seed",
    "initialization_id",
    "model",
    "model_parameters",
    "ctx",
    "corpus_variant_id",
    "train_corpus",
    "sidecar_id",
    "sidecar_name",
    "provenance_id",
    "token_matched_replacement",
    "out_dir",
    "micro_batch_size",
    "tokens_per_step",
    "max_steps",
    "total_tokens",
    "lr",
    "warmup_steps",
    "weight_decay",
    "compile",
    "device",
    "log_every",
    "eval_every",
    "ckpt_minutes",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "amendment_id",
    "manifest_id",
    "preregistration_sha256",
    "cohort_assignment_sha256",
    "run_count",
    "model_parameters",
    "targets_per_update",
    "optimizer_steps",
    "raw_target_tokens",
    "shared_seed",
    "shared_initialization_id",
    "runs",
}
_MANIFEST_RUN_FIELDS = {
    "run_id",
    "config",
    "config_sha256",
    "role",
    "pair_id",
    "condition",
    "seed",
    "initialization_id",
    "corpus_variant_id",
    "sidecar_id",
    "provenance_id",
    "token_matched_replacement",
}
_ADMISSION_FIELDS = {
    "schema_version",
    "amendment_sha256",
    "manifest_sha256",
    "objective_controls_contract_sha256",
    "runs",
}
_ADMISSION_RUN_FIELDS = {
    "run_id",
    "config_sha256",
    "corpus_variant_id",
    "corpus_sha256",
    "sidecar_id",
    "sidecar_sha256",
    "provenance_id",
    "provenance_sha256",
    "seed",
    "initialization_id",
    "model_parameters",
    "targets_per_update",
    "optimizer_steps",
    "raw_target_tokens",
    "optimization_finite",
    "checkpoint_resume_exact",
    "checkpoint_resume_max_abs_delta",
    "language_relative_degradation",
    "route_audit_passed",
    "mask_audit_passed",
    "semantic_closure_audit_passed",
    "primary_cell_accuracy",
}


@dataclass(frozen=True)
class _RunSpec:
    run_id: str
    role: str
    pair_id: str
    condition: str
    corpus_variant_id: str
    train_corpus: str
    sidecar_id: str
    sidecar_name: str
    provenance_id: str
    token_matched_replacement: str

    @property
    def config_path(self) -> str:
        return f"configs/29m-v3/{self.run_id}.yaml"


_RUN_SPECS = (
    _RunSpec(
        "full_corpus_dense",
        "learnability",
        "full_corpus",
        "dense",
        "corpus-v3-29m-full-v1",
        "dataset/29m-v3/full-corpus/receipt.json",
        "sidecar-v3-29m-full-corpus-dense-v1",
        "dense_target_weights",
        "provenance-v3-29m-full-corpus-v1",
        "none",
    ),
    _RunSpec(
        "full_corpus_split90",
        "learnability",
        "full_corpus",
        "split90",
        "corpus-v3-29m-full-v1",
        "dataset/29m-v3/full-corpus/receipt.json",
        "sidecar-v3-29m-full-corpus-split90-v1",
        "split90_target_weights",
        "provenance-v3-29m-full-corpus-v1",
        "none",
    ),
    _RunSpec(
        "no_arc_conceptarc_dense",
        "learnability",
        "no_arc_conceptarc",
        "dense",
        (
            "corpus-v3-29m-no-arc-conceptarc-"
            "fineweb-edu-token-matched-v1"
        ),
        "dataset/29m-v3/no-arc-conceptarc/receipt.json",
        "sidecar-v3-29m-no-arc-conceptarc-dense-v1",
        "dense_target_weights",
        "provenance-v3-29m-no-arc-conceptarc-v1",
        "fineweb_edu",
    ),
    _RunSpec(
        "no_arc_conceptarc_split90",
        "learnability",
        "no_arc_conceptarc",
        "split90",
        (
            "corpus-v3-29m-no-arc-conceptarc-"
            "fineweb-edu-token-matched-v1"
        ),
        "dataset/29m-v3/no-arc-conceptarc/receipt.json",
        "sidecar-v3-29m-no-arc-conceptarc-split90-v1",
        "split90_target_weights",
        "provenance-v3-29m-no-arc-conceptarc-v1",
        "fineweb_edu",
    ),
    _RunSpec(
        "no_refinement_dense",
        "learnability",
        "no_refinement",
        "dense",
        (
            "corpus-v3-29m-no-refinement-"
            "verified-standard-relational-v1"
        ),
        "dataset/29m-v3/no-refinement/receipt.json",
        "sidecar-v3-29m-no-refinement-dense-v1",
        "dense_target_weights",
        "provenance-v3-29m-no-refinement-v1",
        "verified_standard_relational_records",
    ),
    _RunSpec(
        "no_refinement_split90",
        "learnability",
        "no_refinement",
        "split90",
        (
            "corpus-v3-29m-no-refinement-"
            "verified-standard-relational-v1"
        ),
        "dataset/29m-v3/no-refinement/receipt.json",
        "sidecar-v3-29m-no-refinement-split90-v1",
        "split90_target_weights",
        "provenance-v3-29m-no-refinement-v1",
        "verified_standard_relational_records",
    ),
    _RunSpec(
        "full_corpus_random_fact90",
        "integrity_only",
        "full_corpus",
        "random_fact90",
        "corpus-v3-29m-full-v1",
        "dataset/29m-v3/full-corpus/receipt.json",
        "sidecar-v3-29m-full-corpus-random-fact90-v1",
        "random_fact90_target_weights",
        "provenance-v3-29m-full-corpus-v1",
        "none",
    ),
    _RunSpec(
        "full_corpus_matched_nonfactual_mask",
        "integrity_only",
        "full_corpus",
        "matched_nonfactual_mask",
        "corpus-v3-29m-full-v1",
        "dataset/29m-v3/full-corpus/receipt.json",
        "sidecar-v3-29m-full-corpus-matched-nonfactual-mask-v1",
        "matched_nonfactual_mask_target_weights",
        "provenance-v3-29m-full-corpus-v1",
        "none",
    ),
)
RUN_IDS = tuple(spec.run_id for spec in _RUN_SPECS)
LEARNABILITY_RUN_IDS = tuple(
    spec.run_id for spec in _RUN_SPECS if spec.role == "learnability"
)
INTEGRITY_ONLY_RUN_IDS = tuple(
    spec.run_id for spec in _RUN_SPECS if spec.role == "integrity_only"
)
PROTECTED_CELLS = tuple(
    f"memorysplit-v3-360m-s{seed}-{arm}"
    for seed in range(10)
    for arm in ("dense", "split90")
)


def _contract_commitment_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cohort_assignment_sha256": COHORT_ASSIGNMENT_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "config_sha256s": dict(CONFIG_SHA256S),
    }


def _computed_contract_commitment_sha256() -> str:
    data = json.dumps(
        _contract_commitment_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ObjectiveControlRun:
    """Data-only view of one authenticated, non-claim-bearing 29M run."""

    run_id: str
    role: str
    pair_id: str
    condition: str
    seed: int
    initialization_id: str
    config_path: str
    config_sha256: str
    corpus_variant_id: str
    sidecar_id: str
    provenance_id: str
    token_matched_replacement: str


@dataclass(frozen=True)
class ObjectiveControlsContract:
    """Data-only loader result; never accepted as authority by public APIs."""

    amendment_id: str
    amendment_sha256: str
    manifest_sha256: str
    objective_controls_contract_sha256: str
    preregistration_sha256: str
    cohort_assignment_sha256: str
    protected_outcomes_inspected: bool
    protected_cells: tuple[str, ...]
    added_360m_controls: tuple[str, ...]
    statistical_exclusions: tuple[str, ...]
    replicated_360m_selectivity_claim_disclaimed: bool
    runs: tuple[ObjectiveControlRun, ...]


@dataclass(frozen=True)
class ObjectiveControlsAdmission:
    """Data-only result of canonical-path admission authentication."""

    run_ids: tuple[str, ...]
    learnability_run_ids: tuple[str, ...]
    integrity_only_run_ids: tuple[str, ...]
    directional_control_thresholds_applied: tuple[str, ...]


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise MsctlError(
                "OBJECTIVE_CONTROLS_INVALID",
                "objective-controls YAML mapping keys must be scalar",
            ) from error
        if duplicate:
            _fail("objective-controls YAML contains a duplicate field")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _fail(
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> None:
    raise MsctlError(
        "OBJECTIVE_CONTROLS_INVALID",
        message,
        details=details,
    )


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise MsctlError(
            "OBJECTIVE_CONTROLS_INVALID",
            f"{label} cannot be read",
        ) from error


def _resolve_inside(root: Path, relative: str, *, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or relative.startswith("~")
        or _ENVIRONMENT_PLACEHOLDER.search(relative)
    ):
        _fail(f"{label} must be a resolved portable relative path")
    logical = PurePosixPath(relative)
    if logical.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        _fail(f"{label} must be a resolved portable relative path")
    candidate = root
    for part in logical.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            _fail(f"{label} must not traverse a symlink")
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise MsctlError(
            "OBJECTIVE_CONTROLS_INVALID",
            f"{label} must remain inside the repository",
        ) from error
    return candidate


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains a duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda _: _fail(
                f"{label} contains a non-finite number"
            ),
        )
    except MsctlError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "OBJECTIVE_CONTROLS_INVALID",
            f"{label} must contain one valid UTF-8 JSON value",
        ) from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object with string fields")
    return value


def _yaml_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        text = data.decode("utf-8")
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.events.AliasEvent) or getattr(
                event, "anchor", None
            ):
                _fail(f"{label} must not contain YAML anchors or aliases")
        value = yaml.load(text, Loader=_UniqueSafeLoader)
    except MsctlError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise MsctlError(
            "OBJECTIVE_CONTROLS_INVALID",
            f"{label} must contain one valid UTF-8 YAML object",
        ) from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object with string fields")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} has missing or unknown fields",
            details={
                "missing": sorted(expected - actual),
                "unknown": sorted(actual - expected),
            },
        )


def _require_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        _fail(f"{label} must be an integer without bool/float aliases")
    return value


def _require_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{label} must be a finite float without bool/integer aliases")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a boolean")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _expected_amendment() -> dict[str, object]:
    return {
        "schema_version": 1,
        "amendment_id": AMENDMENT_ID,
        "append_only": True,
        "status": "prospective",
        "adopted_date": "2026-07-24",
        "parent_contract": {
            "source_commit": SOURCE_COMMIT,
            "preregistration_path": _PREREGISTRATION_RELATIVE,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "cohort_assignment_path": _COHORT_RELATIVE,
            "cohort_assignment_sha256": COHORT_ASSIGNMENT_SHA256,
        },
        "outcome_inspection_before_amendment": {
            "protected_outcomes_inspected": 0,
            "seed_level_outcomes_inspected": False,
            "arm_level_outcomes_inspected": False,
            "aggregate_outcomes_inspected": False,
        },
        "protected_primary_cohort": {
            "cohort_id": PROTECTED_COHORT_ID,
            "unchanged": True,
            "model_parameters": PROTECTED_MODEL_PARAMETERS,
            "conditions": ["dense", "split90"],
            "seeds": list(range(10)),
            "cell_count": 20,
            "cells": list(PROTECTED_CELLS),
            "claim_bearing_conditions": ["dense", "split90"],
            "added_360m_controls": [],
            "replicated_360m_selectivity_vs_random_claim": "disclaimed",
        },
        "development_diagnostics_29m": {
            "manifest_path": _MANIFEST_RELATIVE,
            "manifest_sha256": MANIFEST_SHA256,
            "run_count": 8,
            "model_parameters": MODEL_PARAMETERS,
            "targets_per_update": TARGETS_PER_UPDATE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "raw_target_tokens": RAW_TARGET_TOKENS,
            "shared_seed": SHARED_SEED,
            "shared_initialization_id": SHARED_INITIALIZATION_ID,
            "run_ids": list(RUN_IDS),
            "non_claim_bearing": True,
            "selection_use_forbidden": True,
            "statistical_exclusions": list(STATISTICAL_EXCLUSIONS),
        },
        "admission": {
            "non_directional": True,
            "required_identity_bindings": [
                "config_sha256",
                "corpus_variant_id",
                "corpus_sha256",
                "sidecar_id",
                "sidecar_sha256",
                "provenance_id",
                "provenance_sha256",
            ],
            "finite_optimization_required": True,
            "checkpoint_resume": {
                "exact_required": True,
                "max_abs_delta": CHECKPOINT_RESUME_MAX_ABS_DELTA,
                "threshold_rule": "less_than_or_equal",
            },
            "language_noninferiority": {
                "max_relative_degradation": (
                    LANGUAGE_MAX_RELATIVE_DEGRADATION
                ),
                "threshold_rule": "less_than_or_equal",
            },
            "required_audits": ["route", "mask", "semantic_closure"],
            "learnability": {
                "run_ids": list(LEARNABILITY_RUN_IDS),
                "primary_cells": list(PRIMARY_CELLS),
                "accuracy_floor": LEARNABILITY_FLOOR,
                "floor_rule": "strictly_greater_than",
            },
            "controls": {
                "run_ids": list(INTEGRITY_ONLY_RUN_IDS),
                "role": "integrity_only",
                "directional_accuracy_threshold": None,
            },
            "forbidden_uses": list(STATISTICAL_EXCLUSIONS),
        },
    }


def _validate_amendment(value: dict[str, object]) -> None:
    expected = _expected_amendment()
    if value != expected:
        _fail(
            "objective-controls amendment does not match its prospective "
            "canonical contract"
        )

    _require_int(value["schema_version"], label="amendment.schema_version")
    _require_bool(value["append_only"], label="amendment.append_only")
    outcomes = value["outcome_inspection_before_amendment"]
    protected = value["protected_primary_cohort"]
    diagnostics = value["development_diagnostics_29m"]
    admission = value["admission"]
    assert isinstance(outcomes, dict)
    assert isinstance(protected, dict)
    assert isinstance(diagnostics, dict)
    assert isinstance(admission, dict)
    _require_int(
        outcomes["protected_outcomes_inspected"],
        label="amendment.protected_outcomes_inspected",
    )
    for field in (
        "seed_level_outcomes_inspected",
        "arm_level_outcomes_inspected",
        "aggregate_outcomes_inspected",
    ):
        _require_bool(outcomes[field], label=f"amendment.{field}")
    _require_bool(protected["unchanged"], label="amendment.protected.unchanged")
    for field in ("model_parameters", "cell_count"):
        _require_int(
            protected[field],
            label=f"amendment.protected.{field}",
        )
    for index, seed in enumerate(protected["seeds"]):
        _require_int(seed, label=f"amendment.protected.seeds[{index}]")
    for field in (
        "run_count",
        "model_parameters",
        "targets_per_update",
        "optimizer_steps",
        "raw_target_tokens",
        "shared_seed",
    ):
        _require_int(
            diagnostics[field],
            label=f"amendment.diagnostics.{field}",
        )
    for field in ("non_claim_bearing", "selection_use_forbidden"):
        _require_bool(
            diagnostics[field],
            label=f"amendment.diagnostics.{field}",
        )
    _require_bool(
        admission["non_directional"],
        label="amendment.admission.non_directional",
    )
    _require_bool(
        admission["finite_optimization_required"],
        label="amendment.admission.finite_optimization_required",
    )
    checkpoint = admission["checkpoint_resume"]
    language = admission["language_noninferiority"]
    controls = admission["controls"]
    assert isinstance(checkpoint, dict)
    assert isinstance(language, dict)
    assert isinstance(controls, dict)
    _require_bool(
        checkpoint["exact_required"],
        label="amendment.admission.checkpoint_resume.exact_required",
    )
    _require_float(
        checkpoint["max_abs_delta"],
        label="amendment.admission.checkpoint_resume.max_abs_delta",
    )
    _require_float(
        language["max_relative_degradation"],
        label="amendment.admission.language.max_relative_degradation",
    )
    learnability = admission["learnability"]
    assert isinstance(learnability, dict)
    _require_float(
        learnability["accuracy_floor"],
        label="amendment.admission.learnability.accuracy_floor",
    )
    if controls["directional_accuracy_threshold"] is not None:
        _fail("integrity-only controls cannot carry a directional threshold")


def _expected_config(spec: _RunSpec) -> dict[str, object]:
    return {
        "schema_version": 3,
        "amendment_id": AMENDMENT_ID,
        "manifest_id": MANIFEST_ID,
        "run_id": spec.run_id,
        "role": spec.role,
        "pair_id": spec.pair_id,
        "condition": spec.condition,
        "seed": SHARED_SEED,
        "initialization_id": SHARED_INITIALIZATION_ID,
        "model": "toy",
        "model_parameters": MODEL_PARAMETERS,
        "ctx": 1024,
        "corpus_variant_id": spec.corpus_variant_id,
        "train_corpus": spec.train_corpus,
        "sidecar_id": spec.sidecar_id,
        "sidecar_name": spec.sidecar_name,
        "provenance_id": spec.provenance_id,
        "token_matched_replacement": spec.token_matched_replacement,
        "out_dir": f"runs/29m-v3/{spec.run_id}",
        "micro_batch_size": 16,
        "tokens_per_step": TARGETS_PER_UPDATE,
        "max_steps": OPTIMIZER_STEPS,
        "total_tokens": RAW_TARGET_TOKENS,
        "lr": 0.0015,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "ckpt_minutes": 30,
    }


def _validate_config(value: dict[str, object], spec: _RunSpec) -> None:
    _require_exact_fields(value, _CONFIG_FIELDS, label=f"config {spec.run_id}")
    for field in (
        "schema_version",
        "seed",
        "model_parameters",
        "ctx",
        "micro_batch_size",
        "tokens_per_step",
        "max_steps",
        "total_tokens",
        "warmup_steps",
        "log_every",
        "eval_every",
        "ckpt_minutes",
    ):
        _require_int(value[field], label=f"config {spec.run_id}.{field}")
    for field in ("lr", "weight_decay"):
        _require_float(value[field], label=f"config {spec.run_id}.{field}")
    _require_bool(value["compile"], label=f"config {spec.run_id}.compile")
    for field in ("train_corpus", "out_dir"):
        logical = value[field]
        if not isinstance(logical, str):
            _fail(f"config {spec.run_id}.{field} must be a string")
        if (
            PurePosixPath(logical).is_absolute()
            or "\\" in logical
            or logical.startswith("~")
            or _ENVIRONMENT_PLACEHOLDER.search(logical)
            or any(part in {"", ".", ".."} for part in logical.split("/"))
        ):
            _fail(
                f"config {spec.run_id}.{field} must be a resolved portable "
                "relative path"
            )
    if value != _expected_config(spec):
        _fail(f"config {spec.run_id} violates the exact 29M contract")
    if value["tokens_per_step"] * value["max_steps"] != value["total_tokens"]:
        _fail(f"config {spec.run_id} has non-integral target-budget math")


def _expected_manifest_row(
    spec: _RunSpec,
    *,
    config_sha256: str,
) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "config": spec.config_path,
        "config_sha256": config_sha256,
        "role": spec.role,
        "pair_id": spec.pair_id,
        "condition": spec.condition,
        "seed": SHARED_SEED,
        "initialization_id": SHARED_INITIALIZATION_ID,
        "corpus_variant_id": spec.corpus_variant_id,
        "sidecar_id": spec.sidecar_id,
        "provenance_id": spec.provenance_id,
        "token_matched_replacement": spec.token_matched_replacement,
    }


def _load_runs(
    repo_root: Path,
    manifest_value: dict[str, object],
) -> tuple[ObjectiveControlRun, ...]:
    _require_exact_fields(
        manifest_value,
        _MANIFEST_FIELDS,
        label="objective-controls manifest",
    )
    for field in (
        "schema_version",
        "run_count",
        "model_parameters",
        "targets_per_update",
        "optimizer_steps",
        "raw_target_tokens",
        "shared_seed",
    ):
        _require_int(
            manifest_value[field],
            label=f"objective-controls manifest.{field}",
        )
    if (
        manifest_value["schema_version"] != 1
        or manifest_value["amendment_id"] != AMENDMENT_ID
        or manifest_value["manifest_id"] != MANIFEST_ID
        or manifest_value["preregistration_sha256"]
        != PREREGISTRATION_SHA256
        or manifest_value["cohort_assignment_sha256"]
        != COHORT_ASSIGNMENT_SHA256
        or manifest_value["run_count"] != len(_RUN_SPECS)
        or manifest_value["model_parameters"] != MODEL_PARAMETERS
        or manifest_value["targets_per_update"] != TARGETS_PER_UPDATE
        or manifest_value["optimizer_steps"] != OPTIMIZER_STEPS
        or manifest_value["raw_target_tokens"] != RAW_TARGET_TOKENS
        or manifest_value["shared_seed"] != SHARED_SEED
        or manifest_value["shared_initialization_id"]
        != SHARED_INITIALIZATION_ID
    ):
        _fail("objective-controls manifest violates the frozen 29M contract")

    rows = manifest_value["runs"]
    if not isinstance(rows, list) or len(rows) != len(_RUN_SPECS):
        _fail("objective-controls manifest must contain exactly eight runs")
    run_root = _resolve_inside(
        repo_root,
        "configs/29m-v3",
        label="objective-controls config directory",
    )
    if not run_root.is_dir():
        _fail("objective-controls config directory must be a directory")
    expected_names = {
        "manifest.json",
        *(f"{spec.run_id}.yaml" for spec in _RUN_SPECS),
    }
    try:
        actual_names = {entry.name for entry in run_root.iterdir()}
    except OSError as error:
        raise MsctlError(
            "OBJECTIVE_CONTROLS_INVALID",
            "objective-controls config directory cannot be read",
        ) from error
    if actual_names != expected_names:
        _fail(
            "objective-controls directory must contain exactly the manifest "
            "and eight configs",
            details={
                "missing": sorted(expected_names - actual_names),
                "unknown": sorted(actual_names - expected_names),
            },
        )

    reviewed_config_sha256s = dict(CONFIG_SHA256S)
    expected_config_paths = {spec.config_path for spec in _RUN_SPECS}
    if set(reviewed_config_sha256s) != expected_config_paths:
        _fail("reviewed config commitments do not match the eight-run scope")
    parsed = []
    expected_rows = []
    for index, spec in enumerate(_RUN_SPECS):
        row = rows[index]
        if not isinstance(row, dict) or not all(
            isinstance(key, str) for key in row
        ):
            _fail(f"objective-controls manifest run[{index}] must be an object")
        _require_exact_fields(
            row,
            _MANIFEST_RUN_FIELDS,
            label=f"objective-controls manifest run[{index}]",
        )
        _require_int(
            row["seed"],
            label=f"objective-controls manifest run[{index}].seed",
        )
        config_sha256 = _require_sha256(
            row["config_sha256"],
            label=f"objective-controls manifest run[{index}].config_sha256",
        )
        config_path = _resolve_inside(
            repo_root,
            spec.config_path,
            label=f"config {spec.run_id}",
        )
        config_data = _read_regular(config_path, label=f"config {spec.run_id}")
        actual_sha256 = hashlib.sha256(config_data).hexdigest()
        reviewed_sha256 = reviewed_config_sha256s[spec.config_path]
        if (
            actual_sha256 != reviewed_sha256
            or config_sha256 != reviewed_sha256
        ):
            _fail(
                f"config {spec.run_id} does not match its reviewed byte "
                "commitment"
            )
        config_value = _yaml_object(config_data, label=f"config {spec.run_id}")
        _validate_config(config_value, spec)
        expected_row = _expected_manifest_row(
            spec,
            config_sha256=reviewed_sha256,
        )
        if row != expected_row:
            _fail(
                f"objective-controls manifest run {spec.run_id} has a "
                "cross-variant or cross-sidecar binding"
            )
        expected_rows.append(expected_row)
        parsed.append(
            ObjectiveControlRun(
                run_id=spec.run_id,
                role=spec.role,
                pair_id=spec.pair_id,
                condition=spec.condition,
                seed=SHARED_SEED,
                initialization_id=SHARED_INITIALIZATION_ID,
                config_path=spec.config_path,
                config_sha256=reviewed_sha256,
                corpus_variant_id=spec.corpus_variant_id,
                sidecar_id=spec.sidecar_id,
                provenance_id=spec.provenance_id,
                token_matched_replacement=(
                    spec.token_matched_replacement
                ),
            )
        )

    expected_manifest = {
        "schema_version": 1,
        "amendment_id": AMENDMENT_ID,
        "manifest_id": MANIFEST_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "cohort_assignment_sha256": COHORT_ASSIGNMENT_SHA256,
        "run_count": len(_RUN_SPECS),
        "model_parameters": MODEL_PARAMETERS,
        "targets_per_update": TARGETS_PER_UPDATE,
        "optimizer_steps": OPTIMIZER_STEPS,
        "raw_target_tokens": RAW_TARGET_TOKENS,
        "shared_seed": SHARED_SEED,
        "shared_initialization_id": SHARED_INITIALIZATION_ID,
        "runs": expected_rows,
    }
    if manifest_value != expected_manifest:
        _fail("objective-controls manifest is not the exact canonical matrix")
    return tuple(parsed)


def load_objective_controls_contract(
    amendment_path: Path | str,
) -> ObjectiveControlsContract:
    """Load and validate the append-only amendment and all eight configs."""

    path = Path(amendment_path)
    if path.name != _AMENDMENT_FILENAME or path.parent.name != "configs":
        _fail("objective-controls amendment must use its canonical configs path")
    repo_root = path.parent.parent
    amendment_data = _read_regular(path, label="objective-controls amendment")
    if hashlib.sha256(amendment_data).hexdigest() != AMENDMENT_SHA256:
        _fail("objective-controls amendment bytes are not the reviewed commitment")
    if (
        _computed_contract_commitment_sha256()
        != OBJECTIVE_CONTROLS_CONTRACT_SHA256
    ):
        _fail("objective-controls aggregate commitment constant is inconsistent")
    amendment_value = _yaml_object(
        amendment_data,
        label="objective-controls amendment",
    )
    _validate_amendment(amendment_value)

    preregistration = _resolve_inside(
        repo_root,
        _PREREGISTRATION_RELATIVE,
        label="frozen V3 preregistration",
    )
    cohort = _resolve_inside(
        repo_root,
        _COHORT_RELATIVE,
        label="frozen V3 cohort assignment",
    )
    preregistration_data = _read_regular(
        preregistration,
        label="frozen V3 preregistration",
    )
    cohort_data = _read_regular(cohort, label="frozen V3 cohort assignment")
    if hashlib.sha256(preregistration_data).hexdigest() != (
        PREREGISTRATION_SHA256
    ):
        _fail("frozen V3 preregistration bytes do not match the amendment")
    if hashlib.sha256(cohort_data).hexdigest() != COHORT_ASSIGNMENT_SHA256:
        _fail("frozen V3 cohort-assignment bytes do not match the amendment")

    manifest_path = _resolve_inside(
        repo_root,
        _MANIFEST_RELATIVE,
        label="objective-controls manifest",
    )
    manifest_data = _read_regular(
        manifest_path,
        label="objective-controls manifest",
    )
    if hashlib.sha256(manifest_data).hexdigest() != MANIFEST_SHA256:
        _fail("objective-controls manifest bytes are not the reviewed commitment")
    manifest_value = _json_object(
        manifest_data,
        label="objective-controls manifest",
    )
    canonical_manifest = (
        json.dumps(
            manifest_value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if manifest_data != canonical_manifest:
        _fail("objective-controls manifest must be canonical JSON")
    runs = _load_runs(repo_root, manifest_value)

    return ObjectiveControlsContract(
        amendment_id=AMENDMENT_ID,
        amendment_sha256=AMENDMENT_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        objective_controls_contract_sha256=(
            OBJECTIVE_CONTROLS_CONTRACT_SHA256
        ),
        preregistration_sha256=PREREGISTRATION_SHA256,
        cohort_assignment_sha256=COHORT_ASSIGNMENT_SHA256,
        protected_outcomes_inspected=False,
        protected_cells=PROTECTED_CELLS,
        added_360m_controls=(),
        statistical_exclusions=STATISTICAL_EXCLUSIONS,
        replicated_360m_selectivity_claim_disclaimed=True,
        runs=runs,
    )


def _require_same_binding(
    bindings: dict[str, str],
    identity: str,
    sha256: str,
    *,
    label: str,
) -> None:
    prior = bindings.setdefault(identity, sha256)
    if prior != sha256:
        _fail(f"{label} identity maps to multiple hashes")


def _require_injective(bindings: Mapping[str, str], *, label: str) -> None:
    if len(set(bindings.values())) != len(bindings):
        _fail(f"distinct {label} identities cannot share one hash")


def _validate_objective_controls_admission(
    contract: ObjectiveControlsContract,
    value: Mapping[str, object],
) -> ObjectiveControlsAdmission:
    """Validate evidence after the public path authenticated the contract."""

    if not isinstance(contract, ObjectiveControlsContract):
        _fail("admission requires a validated objective-controls contract")
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail("objective-controls admission must be an object")
    _require_exact_fields(
        value,
        _ADMISSION_FIELDS,
        label="objective-controls admission",
    )
    schema_version = _require_int(
        value["schema_version"],
        label="objective-controls admission.schema_version",
    )
    if schema_version != 1:
        _fail("objective-controls admission schema version must be 1")
    if (
        _require_sha256(
            value["amendment_sha256"],
            label="objective-controls admission.amendment_sha256",
        )
        != contract.amendment_sha256
        or _require_sha256(
            value["manifest_sha256"],
            label="objective-controls admission.manifest_sha256",
        )
        != contract.manifest_sha256
        or _require_sha256(
            value["objective_controls_contract_sha256"],
            label=(
                "objective-controls admission."
                "objective_controls_contract_sha256"
            ),
        )
        != contract.objective_controls_contract_sha256
    ):
        _fail("objective-controls admission has the wrong contract hashes")
    rows = value["runs"]
    if not isinstance(rows, list) or len(rows) != len(contract.runs):
        _fail("objective-controls admission must contain exactly eight runs")

    corpus_bindings: dict[str, str] = {}
    sidecar_bindings: dict[str, str] = {}
    provenance_bindings: dict[str, str] = {}
    seen: set[str] = set()
    for index, expected in enumerate(contract.runs):
        row = rows[index]
        if not isinstance(row, dict) or not all(
            isinstance(key, str) for key in row
        ):
            _fail(f"objective-controls admission run[{index}] must be an object")
        _require_exact_fields(
            row,
            _ADMISSION_RUN_FIELDS,
            label=f"objective-controls admission run[{index}]",
        )
        run_id = row["run_id"]
        if not isinstance(run_id, str) or run_id != expected.run_id:
            _fail("objective-controls admission has missing, extra, or reordered runs")
        if run_id in seen:
            _fail("objective-controls admission contains a duplicate run")
        seen.add(run_id)

        config_sha256 = _require_sha256(
            row["config_sha256"],
            label=f"admission {run_id}.config_sha256",
        )
        corpus_sha256 = _require_sha256(
            row["corpus_sha256"],
            label=f"admission {run_id}.corpus_sha256",
        )
        sidecar_sha256 = _require_sha256(
            row["sidecar_sha256"],
            label=f"admission {run_id}.sidecar_sha256",
        )
        provenance_sha256 = _require_sha256(
            row["provenance_sha256"],
            label=f"admission {run_id}.provenance_sha256",
        )
        if (
            config_sha256 != expected.config_sha256
            or row["corpus_variant_id"] != expected.corpus_variant_id
            or row["sidecar_id"] != expected.sidecar_id
            or row["provenance_id"] != expected.provenance_id
            or row["initialization_id"] != expected.initialization_id
        ):
            _fail(
                f"admission {run_id} has a cross-config, cross-variant, "
                "cross-sidecar, or cross-provenance binding"
            )
        for field, exact in (
            ("seed", expected.seed),
            ("model_parameters", MODEL_PARAMETERS),
            ("targets_per_update", TARGETS_PER_UPDATE),
            ("optimizer_steps", OPTIMIZER_STEPS),
            ("raw_target_tokens", RAW_TARGET_TOKENS),
        ):
            parsed = _require_int(
                row[field],
                label=f"admission {run_id}.{field}",
            )
            if parsed != exact:
                _fail(f"admission {run_id}.{field} violates the exact budget")

        for field in (
            "optimization_finite",
            "checkpoint_resume_exact",
            "route_audit_passed",
            "mask_audit_passed",
            "semantic_closure_audit_passed",
        ):
            if not _require_bool(
                row[field],
                label=f"admission {run_id}.{field}",
            ):
                _fail(f"admission {run_id}.{field} must pass")
        resume_delta = _require_float(
            row["checkpoint_resume_max_abs_delta"],
            label=f"admission {run_id}.checkpoint_resume_max_abs_delta",
        )
        if not 0.0 <= resume_delta <= CHECKPOINT_RESUME_MAX_ABS_DELTA:
            _fail(f"admission {run_id} exceeds the checkpoint/resume delta")
        language_degradation = _require_float(
            row["language_relative_degradation"],
            label=f"admission {run_id}.language_relative_degradation",
        )
        if not -1.0 <= language_degradation <= (
            LANGUAGE_MAX_RELATIVE_DEGRADATION
        ):
            _fail(f"admission {run_id} fails language non-inferiority")

        accuracy = row["primary_cell_accuracy"]
        if run_id in LEARNABILITY_RUN_IDS:
            if not isinstance(accuracy, dict) or not all(
                isinstance(key, str) for key in accuracy
            ):
                _fail(f"admission {run_id} requires primary-cell accuracies")
            _require_exact_fields(
                accuracy,
                set(PRIMARY_CELLS),
                label=f"admission {run_id}.primary_cell_accuracy",
            )
            for cell in PRIMARY_CELLS:
                score = _require_float(
                    accuracy[cell],
                    label=f"admission {run_id}.primary_cell_accuracy.{cell}",
                )
                if not LEARNABILITY_FLOOR < score <= 1.0:
                    _fail(
                        f"admission {run_id}.{cell} must be strictly greater "
                        "than the 0.75 learnability floor"
                    )
        elif accuracy is not None:
            _fail(
                f"integrity-only control {run_id} cannot carry a directional "
                "accuracy threshold"
            )

        corpus_id = row["corpus_variant_id"]
        sidecar_id = row["sidecar_id"]
        provenance_id = row["provenance_id"]
        assert isinstance(corpus_id, str)
        assert isinstance(sidecar_id, str)
        assert isinstance(provenance_id, str)
        _require_same_binding(
            corpus_bindings,
            corpus_id,
            corpus_sha256,
            label="corpus",
        )
        _require_same_binding(
            sidecar_bindings,
            sidecar_id,
            sidecar_sha256,
            label="sidecar",
        )
        _require_same_binding(
            provenance_bindings,
            provenance_id,
            provenance_sha256,
            label="provenance",
        )

    if seen != set(RUN_IDS):
        _fail("objective-controls admission has missing or extra runs")
    _require_injective(corpus_bindings, label="corpus")
    _require_injective(sidecar_bindings, label="sidecar")
    _require_injective(provenance_bindings, label="provenance")
    return ObjectiveControlsAdmission(
        run_ids=RUN_IDS,
        learnability_run_ids=LEARNABILITY_RUN_IDS,
        integrity_only_run_ids=INTEGRITY_ONLY_RUN_IDS,
        directional_control_thresholds_applied=(),
    )


def _load_admission_authority(
    amendment_path: Path | str,
) -> ObjectiveControlsContract:
    if not isinstance(amendment_path, (str, Path)):
        _fail(
            "admission authority requires a canonical amendment path, "
            "not a caller-constructed contract"
        )
    return load_objective_controls_contract(amendment_path)


def validate_objective_controls_admission(
    amendment_path: Path | str,
    value: Mapping[str, object],
) -> ObjectiveControlsAdmission:
    """Authenticate canonical contract bytes, then validate admission evidence."""

    contract = _load_admission_authority(amendment_path)
    return _validate_objective_controls_admission(contract, value)


def load_objective_controls_admission(
    path: Path | str,
    amendment_path: Path | str,
) -> ObjectiveControlsAdmission:
    """Load evidence and authenticate the canonical contract before admission."""

    contract = _load_admission_authority(amendment_path)
    data = _read_regular(Path(path), label="objective-controls admission")
    value = _json_object(data, label="objective-controls admission")
    return _validate_objective_controls_admission(contract, value)


__all__ = [
    "AMENDMENT_SHA256",
    "CONFIG_SHA256S",
    "MANIFEST_SHA256",
    "OBJECTIVE_CONTROLS_CONTRACT_SHA256",
    "ObjectiveControlRun",
    "ObjectiveControlsAdmission",
    "ObjectiveControlsContract",
    "load_objective_controls_admission",
    "load_objective_controls_contract",
    "validate_objective_controls_admission",
]
