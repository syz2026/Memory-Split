"""Balanced counterfactual-pair metrics for one explicit evaluation cell."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from evals.confirmatory.contracts import (
    Arm,
    CheckpointRecord,
    Control,
    ItemRecord,
    MemoryMode,
    ReasoningFamily,
    Stratum,
    Twin,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
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


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


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
    memory_mode: MemoryMode
    control: Control
    proof_valid: bool
    answer_valid: bool
    complete: bool
    valid: bool

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
        for field in ("proof_valid", "answer_valid", "complete", "valid"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be Boolean")

    @property
    def verified_correct(self) -> bool:
        return (
            self.complete
            and self.valid
            and self.proof_valid
            and self.answer_valid
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
    for field in ("checkpoint_sha256", "seed", "arm"):
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
    memory_mode: MemoryMode
    control: Control

    def __post_init__(self) -> None:
        if not 0.0 <= self.primary_accuracy <= 1.0:
            raise ValueError("primary accuracy must be in [0, 1]")
        if tuple(self.primary_cells) != PRIMARY_CELL_IDS:
            raise ValueError("primary metric must contain the four frozen cells")
        if not self.by_stratum or not set(self.by_stratum) <= set(Stratum):
            raise ValueError("stratum diagnostics are invalid")
        if set(self.by_family) != set(ReasoningFamily):
            raise ValueError("family diagnostics must contain both families")
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
            "memory_mode": self.memory_mode.value,
            "control": self.control.value,
        }


_PAIR_METADATA = (
    "pair_id",
    "stratum",
    "family",
    "seed",
    "world_id",
    "checkpoint_sha256",
    "arm",
    "memory_mode",
    "control",
)


def balanced_counterfactual_pair_metric(
    outcomes: Iterable[ItemOutcome],
    *,
    items: Mapping[str, ItemRecord],
    checkpoints: Mapping[str, CheckpointRecord],
) -> PairMetricSummary:
    """Score twin conjunctions and equally weight the four primary cells."""

    rows = tuple(outcomes)
    if not rows:
        raise ValueError("counterfactual pair metric requires outcomes")
    if any(not isinstance(row, ItemOutcome) for row in rows):
        raise TypeError("pair metric outcomes must be ItemOutcome values")
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
            outcome=row,
            item=items[row.item_id],
            checkpoint=checkpoints[row.checkpoint_sha256],
        )
    if any(not row.complete for row in rows):
        raise ValueError("pair metric cannot consume incomplete outcomes")
    if any(not row.valid for row in rows):
        raise ValueError("pair metric cannot consume invalid outcomes")

    cell_values = {
        (
            row.checkpoint_sha256,
            row.arm,
            row.memory_mode,
            row.control,
        )
        for row in rows
    }
    if len(cell_values) != 1:
        raise ValueError("pair metric requires one explicit evaluation cell")
    checkpoint, arm, memory_mode, control = next(iter(cell_values))

    seen_items: set[str] = set()
    grouped: dict[
        tuple[int, str, str],
        dict[Twin, ItemOutcome],
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
        memory_mode=memory_mode,
        control=control,
    )


counterfactual_pair_metric = balanced_counterfactual_pair_metric
