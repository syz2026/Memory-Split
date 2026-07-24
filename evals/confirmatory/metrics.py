"""Balanced counterfactual-pair metrics for one explicit evaluation cell."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar

from evals.confirmatory.contracts import (
    Arm,
    CheckpointRecord,
    ConditionId,
    CONTRACT_VERSION,
    Control,
    ItemRecord,
    MemoryMode,
    ReasoningFamily,
    SealedGoldRecord,
    StoreRecord,
    Stratum,
    STUDY_CONTRACT_VERSION,
    StudyArm,
    StudyCheckpointRecord,
    Twin,
    validate_study_record_identity,
)
from evals.confirmatory.actions import ActionSlot, validate_action_slots
from evals.confirmatory.study_lock import StudyLockV3, StudySnapshotBinding
from evals.confirmatory.solver import (
    ProofAnswerVerification,
    registered_solver,
    verify_proof_and_answer,
)
from msctl.aws_contracts import checkpoint_object_key


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
OUTCOME_SCHEMA = "memorysplit.confirmatory.outcome.v2"
STUDY_OUTCOME_SCHEMA = "memorysplit.confirmatory.outcome.v3"
STUDY_METRICS_SCHEMA = "memorysplit.confirmatory.metrics.v3"
PRIMARY_CELLS = (
    (ReasoningFamily.GRAPH, Stratum.COMPOSITION_OOD),
    (ReasoningFamily.GRAPH, Stratum.JOINT_OOD),
    (ReasoningFamily.NON_PATH, Stratum.COMPOSITION_OOD),
    (ReasoningFamily.NON_PATH, Stratum.JOINT_OOD),
)
PRIMARY_CELL_IDS = tuple(
    f"{family.value}__{stratum.value}" for family, stratum in PRIMARY_CELLS
)


def _enum(value, enum_type, name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an approved value") from exc


def _legacy_arm(arm: StudyArm) -> Arm:
    return {
        StudyArm.DENSE: Arm.DENSE,
        StudyArm.SPLIT90: Arm.SPLIT,
    }[arm]


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
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


def _schema_version(value: object, name: str) -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ValueError(f"{name} schema_version is invalid")
    return value


def _study_schema_version(value: object, name: str) -> int:
    if type(value) is not int or value != STUDY_CONTRACT_VERSION:
        raise ValueError(f"{name} schema_version is invalid")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ItemOutcome:
    item_id: str
    pair_id: str
    twin: Twin
    stratum: Stratum
    family: ReasoningFamily
    seed: int
    world_id: str
    checkpoint_sha256: str
    arm: Arm
    condition_id: ConditionId
    memory_mode: MemoryMode
    control: Control
    submitted_answer: str
    submitted_proof: tuple[ActionSlot, ...]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "item_id",
            "pair_id",
            "twin",
            "stratum",
            "family",
            "seed",
            "world_id",
            "checkpoint_sha256",
            "arm",
            "condition_id",
            "memory_mode",
            "control",
            "submitted_answer",
            "submitted_proof",
        }
    )

    def __post_init__(self) -> None:
        for field in ("item_id", "pair_id", "world_id"):
            object.__setattr__(
                self,
                field,
                _nonempty(getattr(self, field), field),
            )
        object.__setattr__(self, "twin", _enum(self.twin, Twin, "twin"))
        object.__setattr__(
            self,
            "stratum",
            _enum(self.stratum, Stratum, "stratum"),
        )
        object.__setattr__(
            self,
            "family",
            _enum(self.family, ReasoningFamily, "family"),
        )
        object.__setattr__(self, "arm", _enum(self.arm, Arm, "arm"))
        object.__setattr__(
            self,
            "condition_id",
            _enum(self.condition_id, ConditionId, "condition_id"),
        )
        object.__setattr__(
            self,
            "memory_mode",
            _enum(self.memory_mode, MemoryMode, "memory_mode"),
        )
        object.__setattr__(
            self,
            "control",
            _enum(self.control, Control, "control"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or _SHA256_RE.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
        if not isinstance(self.submitted_answer, str):
            raise ValueError("submitted_answer must be a string")
        object.__setattr__(
            self,
            "submitted_proof",
            validate_action_slots(self.submitted_proof),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ItemOutcome":
        value = _strict_fields(raw, cls.FIELDS, "ItemOutcome")
        if value["record_type"] != OUTCOME_SCHEMA:
            raise ValueError(f"outcome record_type must be {OUTCOME_SCHEMA}")
        _schema_version(value["schema_version"], "outcome")
        return cls(
            **{
                key: field_value
                for key, field_value in value.items()
                if key not in {"record_type", "schema_version"}
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": OUTCOME_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "item_id": self.item_id,
            "pair_id": self.pair_id,
            "twin": self.twin.value,
            "stratum": self.stratum.value,
            "family": self.family.value,
            "seed": self.seed,
            "world_id": self.world_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "arm": self.arm.value,
            "condition_id": self.condition_id.value,
            "memory_mode": self.memory_mode.value,
            "control": self.control.value,
            "submitted_answer": self.submitted_answer,
            "submitted_proof": [
                action.to_dict() for action in self.submitted_proof
            ],
        }


@dataclass(frozen=True)
class StudyOutcomeRecord:
    """Step-aware v3 outcome bound to one protected study checkpoint."""

    item_id: str
    pair_id: str
    twin: Twin
    stratum: Stratum
    family: ReasoningFamily
    seed: int
    world_id: str
    checkpoint_sha256: str
    arm: StudyArm
    condition_id: ConditionId
    optimizer_step: int
    raw_token_count: int
    memory_mode: MemoryMode
    control: Control
    submitted_answer: str
    submitted_proof: tuple[ActionSlot, ...]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "item_id",
            "pair_id",
            "twin",
            "stratum",
            "family",
            "seed",
            "world_id",
            "checkpoint_sha256",
            "arm",
            "condition_id",
            "optimizer_step",
            "raw_token_count",
            "memory_mode",
            "control",
            "submitted_answer",
            "submitted_proof",
        }
    )

    def __post_init__(self) -> None:
        (
            seed,
            arm,
            condition_id,
            optimizer_step,
            raw_token_count,
        ) = validate_study_record_identity(
            seed=self.seed,
            arm=self.arm,
            condition_id=self.condition_id,
            optimizer_step=self.optimizer_step,
            raw_token_count=self.raw_token_count,
        )
        base = ItemOutcome(
            item_id=self.item_id,
            pair_id=self.pair_id,
            twin=self.twin,
            stratum=self.stratum,
            family=self.family,
            seed=seed,
            world_id=self.world_id,
            checkpoint_sha256=self.checkpoint_sha256,
            arm=_legacy_arm(arm),
            condition_id=condition_id,
            memory_mode=self.memory_mode,
            control=self.control,
            submitted_answer=self.submitted_answer,
            submitted_proof=self.submitted_proof,
        )
        for field in (
            "item_id",
            "pair_id",
            "twin",
            "stratum",
            "family",
            "world_id",
            "checkpoint_sha256",
            "memory_mode",
            "control",
            "submitted_answer",
            "submitted_proof",
        ):
            object.__setattr__(self, field, getattr(base, field))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "condition_id", condition_id)
        object.__setattr__(self, "optimizer_step", optimizer_step)
        object.__setattr__(self, "raw_token_count", raw_token_count)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudyOutcomeRecord":
        value = _strict_fields(raw, cls.FIELDS, "StudyOutcomeRecord")
        if value["record_type"] != STUDY_OUTCOME_SCHEMA:
            raise ValueError(
                f"outcome record_type must be {STUDY_OUTCOME_SCHEMA}"
            )
        _study_schema_version(value["schema_version"], "study outcome")
        return cls(
            **{
                key: field_value
                for key, field_value in value.items()
                if key not in {"record_type", "schema_version"}
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": STUDY_OUTCOME_SCHEMA,
            "schema_version": STUDY_CONTRACT_VERSION,
            "item_id": self.item_id,
            "pair_id": self.pair_id,
            "twin": self.twin.value,
            "stratum": self.stratum.value,
            "family": self.family.value,
            "seed": self.seed,
            "world_id": self.world_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "arm": self.arm.value,
            "condition_id": self.condition_id.value,
            "optimizer_step": self.optimizer_step,
            "raw_token_count": self.raw_token_count,
            "memory_mode": self.memory_mode.value,
            "control": self.control.value,
            "submitted_answer": self.submitted_answer,
            "submitted_proof": [
                action.to_dict() for action in self.submitted_proof
            ],
        }


def validate_study_outcome_binding(
    *,
    outcome: StudyOutcomeRecord,
    item: ItemRecord,
    checkpoint: StudyCheckpointRecord,
    snapshot: StudySnapshotBinding,
    lock: StudyLockV3,
) -> StudyOutcomeRecord:
    """Authenticate one v3 outcome across item, checkpoint, and locked object."""

    if not isinstance(outcome, StudyOutcomeRecord):
        raise TypeError("study outcome binding requires a StudyOutcomeRecord")
    if not isinstance(item, ItemRecord):
        raise TypeError("study outcome binding requires an ItemRecord")
    if not isinstance(checkpoint, StudyCheckpointRecord):
        raise TypeError(
            "study outcome binding requires a StudyCheckpointRecord"
        )
    if not isinstance(snapshot, StudySnapshotBinding):
        raise TypeError(
            "study outcome binding requires a StudySnapshotBinding"
        )
    if not isinstance(lock, StudyLockV3):
        raise TypeError("study outcome binding requires a StudyLockV3")
    for field in (
        "item_id",
        "pair_id",
        "twin",
        "world_id",
        "family",
        "stratum",
        "memory_mode",
        "control",
    ):
        if getattr(outcome, field) != getattr(item, field):
            raise ValueError(f"study outcome item mismatch: {field}")
    for field in (
        "checkpoint_sha256",
        "seed",
        "arm",
        "condition_id",
        "optimizer_step",
        "raw_token_count",
    ):
        if getattr(outcome, field) != getattr(checkpoint, field):
            raise ValueError(f"study outcome checkpoint mismatch: {field}")
    for field in (
        "checkpoint_sha256",
        "seed",
        "arm",
        "optimizer_step",
    ):
        if getattr(snapshot, field) != getattr(checkpoint, field):
            raise ValueError(f"study lock snapshot mismatch: {field}")
    expected_key = checkpoint_object_key(
        checkpoint.seed,
        checkpoint.arm.value,
        checkpoint.checkpoint_sha256,
    )
    if snapshot.s3_object_key != expected_key:
        raise ValueError("study lock snapshot object key mismatch")
    locked_snapshot = next(
        (
            value
            for value in lock.snapshots
            if (
                value.seed,
                value.arm,
                value.optimizer_step,
            )
            == (
                checkpoint.seed,
                checkpoint.arm,
                checkpoint.optimizer_step,
            )
        ),
        None,
    )
    if locked_snapshot != snapshot:
        raise ValueError("study lock snapshot object identity was replaced")
    return outcome


@dataclass(frozen=True, init=False)
class _ScoredItemOutcome:
    submission: ItemOutcome
    proof_valid: bool
    answer_valid: bool

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "scored outcomes are internal solver-replay results"
        )

    @classmethod
    def _from_solver_replay(
        cls,
        submission: ItemOutcome,
        verification: ProofAnswerVerification,
    ) -> "_ScoredItemOutcome":
        if not isinstance(submission, ItemOutcome) or not isinstance(
            verification,
            ProofAnswerVerification,
        ):
            raise TypeError("solver replay produced an invalid scored outcome")
        result = object.__new__(cls)
        object.__setattr__(result, "submission", submission)
        object.__setattr__(result, "proof_valid", verification.proof_valid)
        object.__setattr__(result, "answer_valid", verification.answer_valid)
        return result

    def __getattr__(self, name: str):
        return getattr(self.submission, name)

    @property
    def verified_correct(self) -> bool:
        return self.proof_valid and self.answer_valid


def _score_item_outcome(
    *,
    outcome: ItemOutcome,
    item: ItemRecord,
    checkpoint: CheckpointRecord,
    gold: SealedGoldRecord,
    store: StoreRecord,
) -> _ScoredItemOutcome:
    validate_item_outcome_binding(
        outcome=outcome,
        item=item,
        checkpoint=checkpoint,
    )
    verification = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=outcome.submitted_proof,
        answer=outcome.submitted_answer,
        solver=registered_solver(gold.solver_id),
    )
    return _ScoredItemOutcome._from_solver_replay(
        outcome,
        verification,
    )


def validate_item_outcome_binding(
    *,
    outcome: ItemOutcome,
    item: ItemRecord,
    checkpoint: CheckpointRecord,
) -> ItemOutcome:
    """Authenticate every item and checkpoint attribution on one outcome."""

    if not isinstance(outcome, ItemOutcome):
        raise TypeError("outcome binding requires an ItemOutcome")
    if not isinstance(item, ItemRecord):
        raise TypeError("outcome binding requires an ItemRecord")
    if not isinstance(checkpoint, CheckpointRecord):
        raise TypeError("outcome binding requires a CheckpointRecord")

    for field in (
        "item_id",
        "pair_id",
        "twin",
        "world_id",
        "family",
        "stratum",
        "memory_mode",
        "control",
    ):
        if getattr(outcome, field) != getattr(item, field):
            raise ValueError(f"outcome item binding mismatch: {field}")
    for field in ("checkpoint_sha256", "seed", "arm", "condition_id"):
        if getattr(outcome, field) != getattr(checkpoint, field):
            raise ValueError(f"outcome checkpoint binding mismatch: {field}")
    return outcome


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.numerator, bool)
            or not isinstance(self.numerator, int)
            or isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or not 0 <= self.numerator <= self.denominator
            or self.denominator == 0
        ):
            raise ValueError("rate counts must satisfy 0 <= numerator <= denominator")

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    @property
    def exact_value(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], name: str = "rate") -> "Rate":
        value = _strict_fields(
            raw,
            frozenset({"value", "numerator", "denominator"}),
            name,
        )
        rate = cls(value["numerator"], value["denominator"])
        claimed = _finite(value["value"], f"{name} value")
        if claimed != rate.value:
            raise ValueError(f"{name} value disagrees with its exact counts")
        return rate

    def to_dict(self) -> dict[str, int | float]:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


@dataclass(frozen=True)
class PairMetricSummary:
    primary_accuracy: float
    primary_cells: Mapping[str, Rate]
    overall_pair_accuracy: Rate
    by_stratum: Mapping[Stratum, Rate]
    by_family: Mapping[ReasoningFamily, Rate]
    checkpoint_sha256: str
    arm: Arm
    condition_id: ConditionId
    memory_mode: MemoryMode
    control: Control

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "primary_accuracy",
            "primary_cells",
            "overall_pair_accuracy",
            "by_stratum",
            "by_family",
            "checkpoint_sha256",
            "arm",
            "condition_id",
            "memory_mode",
            "control",
        }
    )

    def __post_init__(self) -> None:
        primary_accuracy = _finite(
            self.primary_accuracy,
            "primary_accuracy",
        )
        if not 0.0 <= primary_accuracy <= 1.0:
            raise ValueError("primary accuracy must be in [0, 1]")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or _SHA256_RE.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
        object.__setattr__(self, "arm", _enum(self.arm, Arm, "arm"))
        object.__setattr__(
            self,
            "condition_id",
            _enum(self.condition_id, ConditionId, "condition_id"),
        )
        object.__setattr__(
            self,
            "memory_mode",
            _enum(self.memory_mode, MemoryMode, "memory_mode"),
        )
        object.__setattr__(
            self,
            "control",
            _enum(self.control, Control, "control"),
        )
        if not isinstance(self.primary_cells, Mapping):
            raise ValueError("primary_cells must be an object")
        if tuple(self.primary_cells) != PRIMARY_CELL_IDS:
            raise ValueError("primary metric must contain the four frozen cells")
        if any(not isinstance(rate, Rate) for rate in self.primary_cells.values()):
            raise ValueError("primary cells must contain exact rates")
        if not isinstance(self.overall_pair_accuracy, Rate):
            raise ValueError("overall_pair_accuracy must be an exact rate")
        if (
            not isinstance(self.by_stratum, Mapping)
            or not self.by_stratum
            or not set(self.by_stratum) <= set(Stratum)
            or any(not isinstance(rate, Rate) for rate in self.by_stratum.values())
        ):
            raise ValueError("stratum diagnostics are invalid")
        if (
            not isinstance(self.by_family, Mapping)
            or set(self.by_family) != set(ReasoningFamily)
            or any(not isinstance(rate, Rate) for rate in self.by_family.values())
        ):
            raise ValueError("family diagnostics must contain both families")
        expected_primary = (
            sum(rate.value for rate in self.primary_cells.values())
            / len(PRIMARY_CELLS)
        )
        if primary_accuracy != expected_primary:
            raise ValueError("primary_accuracy disagrees with primary cells")
        for name, rates in (
            ("stratum", self.by_stratum.values()),
            ("family", self.by_family.values()),
        ):
            materialized = tuple(rates)
            if (
                sum(rate.numerator for rate in materialized)
                != self.overall_pair_accuracy.numerator
                or sum(rate.denominator for rate in materialized)
                != self.overall_pair_accuracy.denominator
            ):
                raise ValueError(
                    f"{name} diagnostics disagree with overall pair accuracy"
                )
        object.__setattr__(self, "primary_accuracy", primary_accuracy)
        object.__setattr__(
            self,
            "primary_cells",
            MappingProxyType(dict(self.primary_cells)),
        )
        object.__setattr__(
            self,
            "by_stratum",
            MappingProxyType(dict(self.by_stratum)),
        )
        object.__setattr__(
            self,
            "by_family",
            MappingProxyType(dict(self.by_family)),
        )

    @property
    def balanced_accuracy(self) -> float:
        """Compatibility name for the frozen equal-primary-cell accuracy."""

        return self.primary_accuracy

    @property
    def exact_primary_accuracy(self) -> Fraction:
        return (
            sum(
                (
                    rate.exact_value
                    for rate in self.primary_cells.values()
                ),
                Fraction(),
            )
            / len(PRIMARY_CELLS)
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PairMetricSummary":
        value = _strict_fields(raw, cls.FIELDS, "PairMetricSummary")
        primary_raw = _strict_fields(
            value["primary_cells"],
            frozenset(PRIMARY_CELL_IDS),
            "primary_cells",
        )
        if not isinstance(value["by_stratum"], Mapping):
            raise ValueError("by_stratum must be an object")
        if not isinstance(value["by_family"], Mapping):
            raise ValueError("by_family must be an object")
        try:
            by_stratum = {
                Stratum(name): Rate.from_dict(rate, f"by_stratum.{name}")
                for name, rate in value["by_stratum"].items()
            }
            by_family = {
                ReasoningFamily(name): Rate.from_dict(
                    rate,
                    f"by_family.{name}",
                )
                for name, rate in value["by_family"].items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("metric diagnostics contain an unknown key") from exc
        return cls(
            primary_accuracy=value["primary_accuracy"],
            primary_cells={
                name: Rate.from_dict(primary_raw[name], f"primary_cells.{name}")
                for name in PRIMARY_CELL_IDS
            },
            overall_pair_accuracy=Rate.from_dict(
                value["overall_pair_accuracy"],
                "overall_pair_accuracy",
            ),
            by_stratum=by_stratum,
            by_family=by_family,
            checkpoint_sha256=value["checkpoint_sha256"],
            arm=value["arm"],
            condition_id=value["condition_id"],
            memory_mode=value["memory_mode"],
            control=value["control"],
        )

    def to_dict(self) -> dict:
        return {
            "primary_accuracy": self.primary_accuracy,
            "primary_cells": {
                name: self.primary_cells[name].to_dict()
                for name in PRIMARY_CELL_IDS
            },
            "overall_pair_accuracy": self.overall_pair_accuracy.to_dict(),
            "by_stratum": {
                stratum.value: self.by_stratum[stratum].to_dict()
                for stratum in Stratum
                if stratum in self.by_stratum
            },
            "by_family": {
                family.value: self.by_family[family].to_dict()
                for family in ReasoningFamily
            },
            "checkpoint_sha256": self.checkpoint_sha256,
            "arm": self.arm.value,
            "condition_id": self.condition_id.value,
            "memory_mode": self.memory_mode.value,
            "control": self.control.value,
        }


@dataclass(frozen=True)
class StudyMetricsRecord:
    """One step-aware v3 metric summary for a protected evaluation cell."""

    primary_accuracy: float
    primary_cells: Mapping[str, Rate]
    overall_pair_accuracy: Rate
    by_stratum: Mapping[Stratum, Rate]
    by_family: Mapping[ReasoningFamily, Rate]
    checkpoint_sha256: str
    seed: int
    arm: StudyArm
    condition_id: ConditionId
    optimizer_step: int
    raw_token_count: int
    memory_mode: MemoryMode
    control: Control

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            *PairMetricSummary.FIELDS,
            "seed",
            "optimizer_step",
            "raw_token_count",
        }
    )

    def __post_init__(self) -> None:
        (
            seed,
            arm,
            condition_id,
            optimizer_step,
            raw_token_count,
        ) = validate_study_record_identity(
            seed=self.seed,
            arm=self.arm,
            condition_id=self.condition_id,
            optimizer_step=self.optimizer_step,
            raw_token_count=self.raw_token_count,
        )
        summary = PairMetricSummary(
            primary_accuracy=self.primary_accuracy,
            primary_cells=self.primary_cells,
            overall_pair_accuracy=self.overall_pair_accuracy,
            by_stratum=self.by_stratum,
            by_family=self.by_family,
            checkpoint_sha256=self.checkpoint_sha256,
            arm=_legacy_arm(arm),
            condition_id=condition_id,
            memory_mode=self.memory_mode,
            control=self.control,
        )
        for field in PairMetricSummary.FIELDS:
            object.__setattr__(self, field, getattr(summary, field))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "condition_id", condition_id)
        object.__setattr__(self, "optimizer_step", optimizer_step)
        object.__setattr__(self, "raw_token_count", raw_token_count)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudyMetricsRecord":
        value = _strict_fields(raw, cls.FIELDS, "StudyMetricsRecord")
        if value["record_type"] != STUDY_METRICS_SCHEMA:
            raise ValueError(
                f"metrics record_type must be {STUDY_METRICS_SCHEMA}"
            )
        _study_schema_version(value["schema_version"], "study metrics")
        study_arm = _enum(value["arm"], StudyArm, "arm")
        summary = PairMetricSummary.from_dict(
            {
                field: (
                    _legacy_arm(study_arm).value
                    if field == "arm"
                    else value[field]
                )
                for field in PairMetricSummary.FIELDS
            }
        )
        return cls(
            **{
                field: getattr(summary, field)
                for field in PairMetricSummary.FIELDS
                if field != "arm"
            },
            arm=study_arm,
            seed=value["seed"],
            optimizer_step=value["optimizer_step"],
            raw_token_count=value["raw_token_count"],
        )

    def to_dict(self) -> dict[str, Any]:
        summary = PairMetricSummary(
            **{
                field: (
                    _legacy_arm(self.arm)
                    if field == "arm"
                    else getattr(self, field)
                )
                for field in PairMetricSummary.FIELDS
            }
        )
        value = summary.to_dict()
        return {
            "record_type": STUDY_METRICS_SCHEMA,
            "schema_version": STUDY_CONTRACT_VERSION,
            "primary_accuracy": value["primary_accuracy"],
            "primary_cells": value["primary_cells"],
            "overall_pair_accuracy": value["overall_pair_accuracy"],
            "by_stratum": value["by_stratum"],
            "by_family": value["by_family"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "seed": self.seed,
            "arm": self.arm.value,
            "condition_id": value["condition_id"],
            "optimizer_step": self.optimizer_step,
            "raw_token_count": self.raw_token_count,
            "memory_mode": value["memory_mode"],
            "control": value["control"],
        }


def exact_study_paired_delta(
    *,
    split90: StudyMetricsRecord,
    dense: StudyMetricsRecord,
) -> Fraction:
    """Return the exact Split90-minus-Dense primary metric contrast."""

    if not isinstance(split90, StudyMetricsRecord) or not isinstance(
        dense,
        StudyMetricsRecord,
    ):
        raise TypeError("paired delta requires two StudyMetricsRecord values")
    if (
        split90.arm is not StudyArm.SPLIT90
        or split90.condition_id is not ConditionId.SPLIT90
        or dense.arm is not StudyArm.DENSE
        or dense.condition_id is not ConditionId.DENSE
    ):
        raise ValueError("paired delta requires Split90 and Dense records")
    for field in (
        "seed",
        "optimizer_step",
        "raw_token_count",
        "memory_mode",
        "control",
    ):
        if getattr(split90, field) != getattr(dense, field):
            raise ValueError(f"paired metric records disagree on {field}")
    split_exact = PairMetricSummary(
        **{
            field: (
                _legacy_arm(split90.arm)
                if field == "arm"
                else getattr(split90, field)
            )
            for field in PairMetricSummary.FIELDS
        }
    ).exact_primary_accuracy
    dense_exact = PairMetricSummary(
        **{
            field: (
                _legacy_arm(dense.arm)
                if field == "arm"
                else getattr(dense, field)
            )
            for field in PairMetricSummary.FIELDS
        }
    ).exact_primary_accuracy
    return split_exact - dense_exact


_PAIR_METADATA = (
    "pair_id",
    "stratum",
    "family",
    "seed",
    "world_id",
    "checkpoint_sha256",
    "arm",
    "condition_id",
    "memory_mode",
    "control",
)


def _aggregate_scored_pair_metric(
    outcomes: Iterable[_ScoredItemOutcome],
    *,
    items: Mapping[str, ItemRecord],
    checkpoints: Mapping[str, CheckpointRecord],
) -> PairMetricSummary:
    """Score twin conjunctions and equally weight the four primary cells."""

    rows = tuple(outcomes)
    if not rows:
        raise ValueError("counterfactual pair metric requires outcomes")
    if any(not isinstance(row, _ScoredItemOutcome) for row in rows):
        raise TypeError("pair metric outcomes must be solver-scored values")
    if not isinstance(items, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, ItemRecord)
        or key != value.item_id
        for key, value in items.items()
    ):
        raise ValueError("metric items must map item_id to ItemRecord")
    if not isinstance(checkpoints, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, CheckpointRecord)
        or key != value.checkpoint_sha256
        for key, value in checkpoints.items()
    ):
        raise ValueError(
            "metric checkpoints must map checkpoint_sha256 to CheckpointRecord"
        )
    required_items = {row.item_id for row in rows}
    required_checkpoints = {row.checkpoint_sha256 for row in rows}
    if set(items) != required_items:
        raise ValueError("metric item bindings are not exact")
    if set(checkpoints) != required_checkpoints:
        raise ValueError("metric checkpoint bindings are not exact")
    for row in rows:
        validate_item_outcome_binding(
            outcome=row.submission,
            item=items[row.item_id],
            checkpoint=checkpoints[row.checkpoint_sha256],
        )

    cell_values = {
        (
            row.checkpoint_sha256,
            row.arm,
            row.condition_id,
            row.memory_mode,
            row.control,
        )
        for row in rows
    }
    if len(cell_values) != 1:
        raise ValueError("pair metric requires one explicit evaluation cell")
    checkpoint, arm, condition_id, memory_mode, control = next(
        iter(cell_values)
    )

    seen_items: set[str] = set()
    grouped: dict[
        tuple[int, str, str],
        dict[Twin, _ScoredItemOutcome],
    ] = defaultdict(dict)
    for row in rows:
        if row.item_id in seen_items:
            raise ValueError("duplicate item outcome")
        seen_items.add(row.item_id)
        key = row.seed, row.world_id, row.pair_id
        if row.twin in grouped[key]:
            raise ValueError("duplicate twin in counterfactual pair")
        grouped[key][row.twin] = row

    stratum_successes = {stratum: 0 for stratum in Stratum}
    stratum_totals = {stratum: 0 for stratum in Stratum}
    family_successes = {family: 0 for family in ReasoningFamily}
    family_totals = {family: 0 for family in ReasoningFamily}
    primary_successes = {cell: 0 for cell in PRIMARY_CELLS}
    primary_totals = {cell: 0 for cell in PRIMARY_CELLS}
    for pair in grouped.values():
        if set(pair) != {Twin.ORIGINAL, Twin.COUNTERFACTUAL}:
            raise ValueError("every counterfactual pair requires both twins")
        original = pair[Twin.ORIGINAL]
        counterfactual = pair[Twin.COUNTERFACTUAL]
        if any(
            getattr(original, field) != getattr(counterfactual, field)
            for field in _PAIR_METADATA
        ):
            raise ValueError("counterfactual twins have crossed metadata")
        if original.item_id == counterfactual.item_id:
            raise ValueError("counterfactual twins require distinct item ids")
        stratum = original.stratum
        family = original.family
        success = (
            original.verified_correct and counterfactual.verified_correct
        )
        stratum_totals[stratum] += 1
        stratum_successes[stratum] += success
        family_totals[family] += 1
        family_successes[family] += success
        cell = family, stratum
        if cell in primary_totals:
            primary_totals[cell] += 1
            primary_successes[cell] += success

    missing = [
        f"{family.value}__{stratum.value}"
        for family, stratum in PRIMARY_CELLS
        if primary_totals[(family, stratum)] == 0
    ]
    if missing:
        raise ValueError(f"pair metric is missing required primary cells: {missing}")
    primary_rates = {
        f"{family.value}__{stratum.value}": Rate(
            primary_successes[(family, stratum)],
            primary_totals[(family, stratum)],
        )
        for family, stratum in PRIMARY_CELLS
    }
    stratum_rates = {
        stratum: Rate(stratum_successes[stratum], stratum_totals[stratum])
        for stratum in Stratum
        if stratum_totals[stratum]
    }
    family_rates = {
        family: Rate(family_successes[family], family_totals[family])
        for family in ReasoningFamily
    }
    overall = Rate(
        sum(stratum_successes.values()),
        sum(stratum_totals.values()),
    )
    return PairMetricSummary(
        primary_accuracy=sum(rate.value for rate in primary_rates.values())
        / len(PRIMARY_CELLS),
        primary_cells=primary_rates,
        overall_pair_accuracy=overall,
        by_stratum=stratum_rates,
        by_family=family_rates,
        checkpoint_sha256=checkpoint,
        arm=arm,
        condition_id=condition_id,
        memory_mode=memory_mode,
        control=control,
    )
