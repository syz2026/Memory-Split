"""Balanced counterfactual-pair metrics for one explicit evaluation cell."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from evals.confirmatory.contracts import (
    Arm,
    Control,
    MemoryMode,
    Stratum,
    Twin,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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
    balanced_accuracy: float
    overall_pair_accuracy: Rate
    by_stratum: Mapping[Stratum, Rate]
    checkpoint_sha256: str
    arm: Arm
    memory_mode: MemoryMode
    control: Control

    def __post_init__(self) -> None:
        if not 0.0 <= self.balanced_accuracy <= 1.0:
            raise ValueError("balanced accuracy must be in [0, 1]")
        if set(self.by_stratum) != set(Stratum):
            raise ValueError("pair metric must contain all four strata")
        object.__setattr__(
            self,
            "by_stratum",
            MappingProxyType(dict(self.by_stratum)),
        )

    def to_dict(self) -> dict:
        return {
            "balanced_accuracy": self.balanced_accuracy,
            "overall_pair_accuracy": self.overall_pair_accuracy.to_dict(),
            "by_stratum": {
                stratum.value: self.by_stratum[stratum].to_dict()
                for stratum in Stratum
            },
            "checkpoint_sha256": self.checkpoint_sha256,
            "arm": self.arm.value,
            "memory_mode": self.memory_mode.value,
            "control": self.control.value,
        }


_PAIR_METADATA = (
    "pair_id",
    "stratum",
    "seed",
    "world_id",
    "checkpoint_sha256",
    "arm",
    "memory_mode",
    "control",
)


def balanced_counterfactual_pair_metric(
    outcomes: Iterable[ItemOutcome],
) -> PairMetricSummary:
    """Score both-twin conjunctions, then equal-weight all four strata."""

    rows = tuple(outcomes)
    if not rows:
        raise ValueError("counterfactual pair metric requires outcomes")
    if any(not isinstance(row, ItemOutcome) for row in rows):
        raise TypeError("pair metric outcomes must be ItemOutcome values")
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
        tuple[int, str],
        dict[Twin, ItemOutcome],
    ] = defaultdict(dict)
    for row in rows:
        if row.item_id in seen_items:
            raise ValueError("duplicate item outcome")
        seen_items.add(row.item_id)
        key = row.seed, row.pair_id
        if row.twin in grouped[key]:
            raise ValueError("duplicate twin in counterfactual pair")
        grouped[key][row.twin] = row

    successes = {stratum: 0 for stratum in Stratum}
    totals = {stratum: 0 for stratum in Stratum}
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
        totals[stratum] += 1
        successes[stratum] += (
            original.verified_correct and counterfactual.verified_correct
        )

    missing = [stratum.value for stratum, total in totals.items() if total == 0]
    if missing:
        raise ValueError(f"pair metric is missing required strata: {missing}")
    rates = {
        stratum: Rate(successes[stratum], totals[stratum])
        for stratum in Stratum
    }
    overall = Rate(sum(successes.values()), sum(totals.values()))
    return PairMetricSummary(
        balanced_accuracy=sum(rate.value for rate in rates.values())
        / len(Stratum),
        overall_pair_accuracy=overall,
        by_stratum=rates,
        checkpoint_sha256=checkpoint,
        arm=arm,
        memory_mode=memory_mode,
        control=control,
    )


counterfactual_pair_metric = balanced_counterfactual_pair_metric
