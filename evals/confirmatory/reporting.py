"""Hash-bound, fail-closed confirmatory artifact reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from types import MappingProxyType
from typing import Any

from evals.confirmatory.contracts import (
    Arm,
    CheckpointRecord,
    CONTRACT_VERSION,
    Control,
    ItemRecord,
    MemoryMode,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.inference import ExactTestResult, exact_sign_flip_test
from evals.confirmatory.metrics import (
    ItemOutcome,
    PairMetricSummary,
    balanced_counterfactual_pair_metric,
    validate_item_outcome_binding,
)
from evals.confirmatory.status import (
    FinalInferenceConclusion,
    InterimEvidenceLabel,
    ScientificStatus,
    StatusAxes,
    classify_status,
)


ARTIFACT_REPORT_SCHEMA = "memorysplit.confirmatory.artifact-report.v2"
INFERENCE_EVIDENCE_SCHEMA = "memorysplit.confirmatory.inference-evidence.v2"
METRICS_SCHEMA = "memorysplit.confirmatory.metrics.v2"
PRIMARY_CONTRAST_ID = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)
PRIMARY_TEST_METHOD = "exact_one_sided_exhaustive_sign_flip"
PRACTICAL_NULL_REPLAY_STATUS = "unavailable_unfrozen_rng_seed"
PRACTICAL_NULL_REPLAY_GAP = (
    "The frozen practical-null procedure requires 20,000 hierarchical-bootstrap "
    "draws, but preregistration-v2 does not freeze an RNG seed; "
    "supports_practical_null cannot be replayed before amendment."
)
_FROZEN_SEEDS = tuple(range(5))
_FROZEN_ARMS = (Arm.DENSE, Arm.SPLIT)
REQUIRED_ARTIFACTS = (
    "checkpoints.jsonl",
    "inference.json",
    "items.jsonl",
    "metrics.json",
    "outcomes.jsonl",
    "sealed-gold.jsonl",
    "stores.jsonl",
)
_REPORT_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "scientific_status",
        "interim_evidence_label",
        "final_inference_conclusion",
        "practical_null_replay_status",
        "paired_seed_bundle_deltas",
        "expected_cells",
        "observed_cells",
        "artifacts",
        "report_sha256",
    }
)
_BINDING_FIELDS = frozenset({"sha256", "size_bytes"})
_INFERENCE_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "measured_validity_failure",
        "primary_test",
        "paired_seed_bundle_deltas",
        "exact_test_result",
    }
)
_PRIMARY_TEST_FIELDS = frozenset(
    {
        "contrast_id",
        "method",
        "alternative",
        "alpha",
        "n_pairs",
        "sign_assignments",
        "equality_counted",
    }
)
_EXACT_RESULT_FIELDS = frozenset(
    {"statistic", "extreme_count", "p_value", "reject_null"}
)
_METRICS_FIELDS = frozenset(
    {"record_type", "schema_version", "summaries"}
)
_HEX = frozenset("0123456789abcdef")
_DIR_RELATIVE_PUBLICATION_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.link in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.link in os.supports_follow_symlinks
)


def _hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _count(value: object, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _schema_version(value: object, name: str) -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ValueError(f"{name} schema_version is invalid")
    return value


def _strict_fields(
    value: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{name} fields are not exact")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class PrimaryTestIdentity:
    contrast_id: str
    method: str
    alternative: str
    alpha: float
    n_pairs: int
    sign_assignments: int
    equality_counted: bool

    def __post_init__(self) -> None:
        expected = (
            self.contrast_id == PRIMARY_CONTRAST_ID
            and self.method == PRIMARY_TEST_METHOD
            and self.alternative == "greater"
            and _number(self.alpha, "primary test alpha") == 0.05
            and type(self.n_pairs) is int
            and self.n_pairs == 5
            and type(self.sign_assignments) is int
            and self.sign_assignments == 32
            and self.equality_counted is True
        )
        if not expected:
            raise ValueError("primary test identity disagrees with frozen contract")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PrimaryTestIdentity":
        value = _strict_fields(raw, _PRIMARY_TEST_FIELDS, "primary test")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "method": self.method,
            "alternative": self.alternative,
            "alpha": self.alpha,
            "n_pairs": self.n_pairs,
            "sign_assignments": self.sign_assignments,
            "equality_counted": self.equality_counted,
        }


@dataclass(frozen=True)
class PersistedExactTestResult:
    statistic: float
    extreme_count: int
    p_value: float
    reject_null: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statistic",
            _number(self.statistic, "exact test statistic"),
        )
        if (
            type(self.extreme_count) is not int
            or not 0 <= self.extreme_count <= 32
        ):
            raise ValueError("exact test extreme_count is invalid")
        object.__setattr__(
            self,
            "p_value",
            _number(self.p_value, "exact test p_value"),
        )
        if (
            not 0.0 <= self.p_value <= 1.0
            or self.p_value != self.extreme_count / 32
        ):
            raise ValueError(
                "exact test p_value disagrees with exhaustive assignments"
            )
        _boolean(self.reject_null, "exact test reject_null")

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "PersistedExactTestResult":
        value = _strict_fields(
            raw,
            _EXACT_RESULT_FIELDS,
            "exact test result",
        )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "extreme_count": self.extreme_count,
            "p_value": self.p_value,
            "reject_null": self.reject_null,
        }


@dataclass(frozen=True)
class InferenceEvidence:
    record_type: str
    schema_version: int
    measured_validity_failure: bool
    primary_test: PrimaryTestIdentity
    paired_seed_bundle_deltas: tuple[float, ...]
    exact_test_result: PersistedExactTestResult | None

    def __post_init__(self) -> None:
        if self.record_type != INFERENCE_EVIDENCE_SCHEMA:
            raise ValueError("inference evidence schema identity is invalid")
        _schema_version(self.schema_version, "inference evidence")
        _boolean(
            self.measured_validity_failure,
            "measured_validity_failure",
        )
        if not isinstance(self.primary_test, PrimaryTestIdentity):
            raise ValueError("inference evidence primary_test is invalid")
        if not isinstance(self.paired_seed_bundle_deltas, (list, tuple)):
            raise ValueError("paired seed-bundle deltas must be ordered")
        deltas = tuple(
            _number(value, f"paired seed-bundle delta {index}")
            for index, value in enumerate(self.paired_seed_bundle_deltas)
        )
        if len(deltas) > 5:
            raise ValueError("paired seed-bundle deltas exceed frozen N=5")
        object.__setattr__(self, "paired_seed_bundle_deltas", deltas)
        if len(deltas) == 5:
            if not isinstance(
                self.exact_test_result,
                PersistedExactTestResult,
            ):
                raise ValueError("complete N=5 evidence requires an exact test result")
        elif self.exact_test_result is not None:
            raise ValueError("pre-terminal evidence cannot persist a final exact test")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InferenceEvidence":
        value = _strict_fields(
            raw,
            _INFERENCE_FIELDS,
            "inference evidence",
        )
        raw_result = value["exact_test_result"]
        return cls(
            record_type=value["record_type"],
            schema_version=value["schema_version"],
            measured_validity_failure=value["measured_validity_failure"],
            primary_test=PrimaryTestIdentity.from_dict(value["primary_test"]),
            paired_seed_bundle_deltas=value["paired_seed_bundle_deltas"],
            exact_test_result=(
                None
                if raw_result is None
                else PersistedExactTestResult.from_dict(raw_result)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "measured_validity_failure": self.measured_validity_failure,
            "primary_test": self.primary_test.to_dict(),
            "paired_seed_bundle_deltas": list(
                self.paired_seed_bundle_deltas
            ),
            "exact_test_result": (
                None
                if self.exact_test_result is None
                else self.exact_test_result.to_dict()
            ),
        }


def _inference_evidence(content: bytes) -> InferenceEvidence:
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("inference evidence must be canonical UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("inference evidence must be an object")
    try:
        if canonical_json_bytes(raw) != content:
            raise ValueError("inference evidence must use canonical JSON encoding")
        return InferenceEvidence.from_dict(raw)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("inference evidence is invalid") from exc


def _canonical_jsonl(
    content: bytes,
    *,
    name: str,
    parser: Callable[[Mapping[str, Any]], Any],
) -> tuple[Any, ...]:
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"{name} must be non-empty canonical JSONL")
    records = []
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} line {index} is invalid JSON") from exc
        if not isinstance(raw, Mapping) or canonical_json_bytes(raw) != line:
            raise ValueError(f"{name} line {index} is not canonical")
        records.append(parser(raw))
    return tuple(records)


def _parse_items(content: bytes) -> tuple[ItemRecord, ...]:
    records = _canonical_jsonl(
        content,
        name="items.jsonl",
        parser=ItemRecord.from_dict,
    )
    identities = tuple(record.item_id for record in records)
    if len(set(identities)) != len(identities):
        raise ValueError("items.jsonl contains a duplicate item_id")
    if identities != tuple(sorted(identities)):
        raise ValueError("items.jsonl is not ordered by item_id")
    return records


def _parse_checkpoints(content: bytes) -> tuple[CheckpointRecord, ...]:
    records = _canonical_jsonl(
        content,
        name="checkpoints.jsonl",
        parser=CheckpointRecord.from_dict,
    )
    identities = tuple(record.checkpoint_sha256 for record in records)
    if len(set(identities)) != len(identities):
        raise ValueError("checkpoints.jsonl contains a duplicate hash")
    slots = tuple((record.seed, record.arm) for record in records)
    allowed = {
        (seed, arm)
        for seed in _FROZEN_SEEDS
        for arm in _FROZEN_ARMS
    }
    if len(set(slots)) != len(slots) or not set(slots) <= allowed:
        raise ValueError("checkpoints.jsonl has an invalid frozen seed/arm slot")
    if slots != tuple(sorted(slots, key=lambda slot: (slot[0], slot[1].value))):
        raise ValueError("checkpoints.jsonl is not ordered by seed and arm")
    return records


def _parse_outcomes(content: bytes) -> tuple[ItemOutcome, ...]:
    records = _canonical_jsonl(
        content,
        name="outcomes.jsonl",
        parser=ItemOutcome.from_dict,
    )
    order = tuple(
        (record.seed, record.arm.value, record.item_id)
        for record in records
    )
    if order != tuple(sorted(order)):
        raise ValueError("outcomes.jsonl is not ordered by seed, arm, and item")
    return records


def _parse_metrics(content: bytes) -> tuple[PairMetricSummary, ...]:
    try:
        raw = json.loads(content)
        if not isinstance(raw, Mapping) or canonical_json_bytes(raw) != content:
            raise ValueError("metrics.json must be canonical JSON")
        value = _strict_fields(raw, _METRICS_FIELDS, "metrics artifact")
        if value["record_type"] != METRICS_SCHEMA:
            raise ValueError("metrics artifact record_type is invalid")
        _schema_version(value["schema_version"], "metrics artifact")
        if not isinstance(value["summaries"], list):
            raise ValueError("metrics summaries must be an ordered list")
        return tuple(
            PairMetricSummary.from_dict(summary)
            for summary in value["summaries"]
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"metrics artifact is invalid: {exc}") from exc


@dataclass(frozen=True)
class ArtifactReplay:
    axes: StatusAxes
    expected_cells: int
    observed_cells: int
    seed_deltas: tuple[float, ...]
    exact_test_result: ExactTestResult | None


def _metric_sort_key(
    summary: PairMetricSummary,
    checkpoints: Mapping[str, CheckpointRecord],
) -> tuple[int, str, str, str]:
    checkpoint = checkpoints[summary.checkpoint_sha256]
    return (
        checkpoint.seed,
        checkpoint.arm.value,
        summary.memory_mode.value,
        summary.control.value,
    )


def _recomputed_metrics(
    *,
    items: Mapping[str, ItemRecord],
    checkpoints: Mapping[str, CheckpointRecord],
    outcomes: Sequence[ItemOutcome],
) -> tuple[PairMetricSummary, ...]:
    item_cells: dict[tuple[MemoryMode, Control], set[str]] = defaultdict(set)
    for item in items.values():
        item_cells[(item.memory_mode, item.control)].add(item.item_id)
    grouped: dict[
        tuple[str, MemoryMode, Control],
        list[ItemOutcome],
    ] = defaultdict(list)
    for outcome in outcomes:
        grouped[
            (
                outcome.checkpoint_sha256,
                outcome.memory_mode,
                outcome.control,
            )
        ].append(outcome)

    summaries = []
    for (checkpoint_hash, memory_mode, control), rows in grouped.items():
        expected_items = item_cells[(memory_mode, control)]
        if (
            {row.item_id for row in rows} != expected_items
            or any(not row.complete or not row.valid for row in rows)
        ):
            continue
        summaries.append(
            balanced_counterfactual_pair_metric(
                rows,
                items={item_id: items[item_id] for item_id in expected_items},
                checkpoints={
                    checkpoint_hash: checkpoints[checkpoint_hash],
                },
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: _metric_sort_key(summary, checkpoints),
        )
    )


def _metrics_bytes(summaries: Sequence[PairMetricSummary]) -> bytes:
    return canonical_json_bytes(
        {
            "record_type": METRICS_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "summaries": [summary.to_dict() for summary in summaries],
        }
    )


def _seed_deltas(
    summaries: Sequence[PairMetricSummary],
    checkpoints: Mapping[str, CheckpointRecord],
) -> tuple[float, ...]:
    by_slot: dict[tuple[int, Arm], float] = {}
    for summary in summaries:
        if (
            summary.memory_mode is not MemoryMode.MEMORY_ON
            or summary.control is not Control.CORRECT
        ):
            continue
        checkpoint = checkpoints[summary.checkpoint_sha256]
        slot = checkpoint.seed, checkpoint.arm
        if slot in by_slot:
            raise ValueError("metrics contain a duplicate primary seed/arm cell")
        by_slot[slot] = summary.primary_accuracy
    return tuple(
        by_slot[(seed, Arm.SPLIT)] - by_slot[(seed, Arm.DENSE)]
        for seed in _FROZEN_SEEDS
        if (seed, Arm.SPLIT) in by_slot and (seed, Arm.DENSE) in by_slot
    )


def _validate_exact_result(
    evidence: InferenceEvidence,
    replayed: ExactTestResult | None,
) -> bool:
    if replayed is None:
        if evidence.exact_test_result is not None:
            raise ValueError("exact test result exists without replayable N=5 data")
        return False
    persisted = evidence.exact_test_result
    if persisted is None:
        raise ValueError("replayable N=5 data is missing an exact test result")
    decision = replayed.statistic > 0.0 and replayed.p_value <= 0.05
    if (
        persisted.statistic != replayed.statistic
        or persisted.extreme_count != replayed.extreme_count
        or persisted.p_value != replayed.p_value
        or persisted.reject_null is not decision
    ):
        raise ValueError("persisted exact test result disagrees with replay")
    return decision


def _replay_artifacts(
    content: Mapping[str, bytes],
) -> ArtifactReplay:
    items = _parse_items(content["items.jsonl"])
    checkpoints = _parse_checkpoints(content["checkpoints.jsonl"])
    outcomes = _parse_outcomes(content["outcomes.jsonl"])
    persisted_metrics = _parse_metrics(content["metrics.json"])
    evidence = _inference_evidence(content["inference.json"])

    item_map = {item.item_id: item for item in items}
    checkpoint_map = {
        checkpoint.checkpoint_sha256: checkpoint
        for checkpoint in checkpoints
    }
    observed_keys: set[tuple[str, int, Arm]] = set()
    for outcome in outcomes:
        try:
            item = item_map[outcome.item_id]
            checkpoint = checkpoint_map[outcome.checkpoint_sha256]
        except KeyError as exc:
            raise ValueError("outcome references an unbound record") from exc
        validate_item_outcome_binding(
            outcome=outcome,
            item=item,
            checkpoint=checkpoint,
        )
        key = outcome.item_id, outcome.seed, outcome.arm
        if key in observed_keys:
            raise ValueError("outcomes.jsonl contains a duplicate evaluation cell")
        observed_keys.add(key)

    expected_keys = {
        (item.item_id, seed, arm)
        for item in items
        for seed in _FROZEN_SEEDS
        for arm in _FROZEN_ARMS
    }
    if not observed_keys <= expected_keys:
        raise ValueError("outcomes contain a cell outside the frozen registry")

    recomputed_metrics = _recomputed_metrics(
        items=item_map,
        checkpoints=checkpoint_map,
        outcomes=outcomes,
    )
    if _metrics_bytes(persisted_metrics) != content["metrics.json"]:
        raise ValueError("metrics artifact is not canonical")
    if _metrics_bytes(recomputed_metrics) != content["metrics.json"]:
        raise ValueError("metrics artifact disagrees with recomputed outcomes")

    deltas = _seed_deltas(recomputed_metrics, checkpoint_map)
    if deltas != evidence.paired_seed_bundle_deltas:
        raise ValueError("paired seed-bundle deltas disagree with replay")
    replayed_test = (
        exact_sign_flip_test(deltas, alternative="greater")
        if len(deltas) == 5
        else None
    )
    effect_decision = _validate_exact_result(evidence, replayed_test)

    complete = (
        observed_keys == expected_keys
        and set((checkpoint.seed, checkpoint.arm) for checkpoint in checkpoints)
        == {
            (seed, arm)
            for seed in _FROZEN_SEEDS
            for arm in _FROZEN_ARMS
        }
        and all(outcome.complete for outcome in outcomes)
    )
    measured_failure = (
        evidence.measured_validity_failure
        or any(not outcome.valid for outcome in outcomes)
    )
    same_sign = bool(deltas) and (
        all(delta > 0.0 for delta in deltas)
        or all(delta < 0.0 for delta in deltas)
    )
    axes = classify_status(
        complete=complete,
        valid=not measured_failure,
        observed_seeds=len(deltas),
        required_seeds=5,
        sign_consistent=same_sign,
        supports_effect=effect_decision,
        supports_practical_null=False,
    )
    return ArtifactReplay(
        axes=axes,
        expected_cells=len(expected_keys),
        observed_cells=len(observed_keys),
        seed_deltas=deltas,
        exact_test_result=replayed_test,
    )


def _artifact_bindings(
    value: object,
) -> dict[str, dict[str, str | int]]:
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_ARTIFACTS):
        raise ValueError("artifact set is not exact")
    bindings = {}
    for name in REQUIRED_ARTIFACTS:
        raw = _strict_fields(
            value[name],
            _BINDING_FIELDS,
            f"artifact binding {name}",
        )
        bindings[name] = {
            "sha256": _hash(raw["sha256"], f"artifact {name} hash"),
            "size_bytes": _count(
                raw["size_bytes"],
                f"artifact {name} size",
                positive=True,
            ),
        }
    return bindings


def _report_payload(
    *,
    axes: StatusAxes,
    seed_deltas: Sequence[float],
    expected_cells: int,
    observed_cells: int,
    artifacts: Mapping[str, Mapping[str, str | int]],
) -> dict[str, Any]:
    return {
        "record_type": ARTIFACT_REPORT_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        **axes.to_dict(),
        "practical_null_replay_status": PRACTICAL_NULL_REPLAY_STATUS,
        "paired_seed_bundle_deltas": list(seed_deltas),
        "expected_cells": expected_cells,
        "observed_cells": observed_cells,
        "artifacts": {
            name: dict(artifacts[name]) for name in REQUIRED_ARTIFACTS
        },
    }


@dataclass(frozen=True)
class ArtifactReport:
    record_type: str
    schema_version: int
    scientific_status: ScientificStatus
    interim_evidence_label: InterimEvidenceLabel
    final_inference_conclusion: FinalInferenceConclusion
    practical_null_replay_status: str
    paired_seed_bundle_deltas: tuple[float, ...]
    expected_cells: int
    observed_cells: int
    artifacts: Mapping[str, Mapping[str, str | int]]
    report_sha256: str

    def __post_init__(self) -> None:
        if self.record_type != ARTIFACT_REPORT_SCHEMA:
            raise ValueError("artifact report schema identity is invalid")
        _schema_version(self.schema_version, "artifact report")
        axes = StatusAxes(
            self.scientific_status,
            self.interim_evidence_label,
            self.final_inference_conclusion,
        )
        if self.practical_null_replay_status != PRACTICAL_NULL_REPLAY_STATUS:
            raise ValueError("practical-null replay status is invalid")
        if not isinstance(self.paired_seed_bundle_deltas, (list, tuple)):
            raise ValueError("report seed-bundle deltas must be ordered")
        seed_deltas = tuple(
            _number(value, f"report seed-bundle delta {index}")
            for index, value in enumerate(self.paired_seed_bundle_deltas)
        )
        if len(seed_deltas) > 5:
            raise ValueError("report seed-bundle deltas exceed frozen N=5")
        expected = _count(self.expected_cells, "expected_cells", positive=True)
        observed = _count(self.observed_cells, "observed_cells")
        if observed > expected:
            raise ValueError("observed_cells exceeds expected_cells")
        if (
            observed < expected
            and axes.scientific_status is ScientificStatus.COMPLETE
        ):
            raise ValueError("missing cells cannot have complete scientific status")
        bindings = _artifact_bindings(self.artifacts)
        payload = _report_payload(
            axes=axes,
            seed_deltas=seed_deltas,
            expected_cells=expected,
            observed_cells=observed,
            artifacts=bindings,
        )
        claimed = _hash(self.report_sha256, "report_sha256")
        if claimed != canonical_sha256(payload):
            raise ValueError("artifact report_sha256 does not match report")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(binding))
                    for name, binding in bindings.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "scientific_status",
            axes.scientific_status,
        )
        object.__setattr__(
            self,
            "interim_evidence_label",
            axes.interim_evidence_label,
        )
        object.__setattr__(
            self,
            "final_inference_conclusion",
            axes.final_inference_conclusion,
        )
        object.__setattr__(
            self,
            "paired_seed_bundle_deltas",
            seed_deltas,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactReport":
        value = _strict_fields(raw, _REPORT_FIELDS, "artifact report")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            **_report_payload(
                axes=StatusAxes(
                    self.scientific_status,
                    self.interim_evidence_label,
                    self.final_inference_conclusion,
                ),
                seed_deltas=self.paired_seed_bundle_deltas,
                expected_cells=self.expected_cells,
                observed_cells=self.observed_cells,
                artifacts=self.artifacts,
            ),
            "report_sha256": self.report_sha256,
        }


def _artifact_bytes(
    artifacts: Mapping[str, bytes],
) -> dict[str, bytes]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        REQUIRED_ARTIFACTS
    ):
        raise ValueError("artifact set is not exact")
    result = {}
    for name in REQUIRED_ARTIFACTS:
        content = artifacts[name]
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"artifact {name} must contain bytes")
        result[name] = content
    return result


def build_artifact_report(
    *,
    artifacts: Mapping[str, bytes],
) -> ArtifactReport:
    content = _artifact_bytes(artifacts)
    replay = _replay_artifacts(content)
    bindings = {
        name: {
            "sha256": hashlib.sha256(content[name]).hexdigest(),
            "size_bytes": len(content[name]),
        }
        for name in REQUIRED_ARTIFACTS
    }
    payload = _report_payload(
        axes=replay.axes,
        seed_deltas=replay.seed_deltas,
        expected_cells=replay.expected_cells,
        observed_cells=replay.observed_cells,
        artifacts=bindings,
    )
    return ArtifactReport.from_dict(
        {
            **payload,
            "report_sha256": canonical_sha256(payload),
        }
    )


def validate_artifact_report(
    report: ArtifactReport | Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> ArtifactReport:
    typed = (
        ArtifactReport.from_dict(report.to_dict())
        if isinstance(report, ArtifactReport)
        else ArtifactReport.from_dict(report)
    )
    content = _artifact_bytes(artifacts)
    for name in REQUIRED_ARTIFACTS:
        binding = typed.artifacts[name]
        if len(content[name]) != binding["size_bytes"]:
            raise ValueError(f"artifact {name} size mismatch")
        if hashlib.sha256(content[name]).hexdigest() != binding["sha256"]:
            raise ValueError(f"artifact {name} hash mismatch")
    replay = _replay_artifacts(content)
    reported_axes = StatusAxes(
        typed.scientific_status,
        typed.interim_evidence_label,
        typed.final_inference_conclusion,
    )
    if reported_axes != replay.axes:
        raise ValueError("artifact report status axes disagree with bound evidence")
    if (
        typed.expected_cells != replay.expected_cells
        or typed.observed_cells != replay.observed_cells
    ):
        raise ValueError("artifact report counts disagree with bound evidence")
    if typed.paired_seed_bundle_deltas != replay.seed_deltas:
        raise ValueError("artifact report seed effects disagree with bound evidence")
    return typed


def _secure_directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(
        not hasattr(os, name) or not getattr(os, name)
        for name in required
    ):
        raise RuntimeError(
            "secure directory-relative publication is unsupported"
        )
    if not _DIR_RELATIVE_PUBLICATION_SUPPORTED:
        raise RuntimeError(
            "secure directory-relative publication is unsupported"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_parent_directory(output: Path) -> int:
    if any(part == ".." for part in output.parts):
        raise ValueError("artifact report path cannot contain traversal")
    if output.name in {"", ".", ".."}:
        raise ValueError("artifact report path must name a file")

    flags = _secure_directory_flags()
    parent = output.parent
    if parent.is_absolute():
        descriptor = os.open(parent.anchor, flags)
        components = parent.parts[1:]
    else:
        descriptor = os.open(".", flags)
        components = parent.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            try:
                child = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ValueError(
                    "artifact report parent cannot contain a symlink "
                    "or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(
                "artifact report parent must be a regular directory"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_temporary(
    directory_fd: int,
    content: bytes,
) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        name = f".confirmatory-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            raise
        return name
    raise FileExistsError("could not allocate a secure temporary report")


def publish_artifact_report(
    path: str | Path,
    report: ArtifactReport | Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> Path:
    """Atomically publish a canonical report without replacing existing data."""

    typed = validate_artifact_report(report, artifacts)
    output = Path(path)
    directory_fd = _open_parent_directory(output)
    temporary_name: str | None = None
    try:
        temporary_name = _create_temporary(
            directory_fd,
            canonical_json_bytes(typed),
        )
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"artifact report already exists: {output}"
            ) from None
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    return output
