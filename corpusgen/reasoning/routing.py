"""Deterministic, train-only routing for fixed intervention doses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


ROUTE_POLICY = "train-score-ranked-quota-v1"
METADATA_SCOPE = "training-only"
SPLIT_FRACTIONS = {
    "Split50": Fraction(1, 2),
    "Split90": Fraction(9, 10),
}
TRAIN_ONLY_FEATURES = (
    "payload_entropy_bits",
    "scheduled_exposures",
    "expected_reads",
    "expected_hops",
    "information_burden_bits",
)


def _fraction(value: object, field: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative finite number")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} must be a non-negative finite number") from error
    if result < 0:
        raise ValueError(f"{field} must be a non-negative finite number")
    return result


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True)
class FactMetadata:
    """The complete feature row allowed to influence routing.

    Every feature is computed from the frozen training schedule or training
    source statistics. The type deliberately has no validation/test or model
    outcome fields.
    """

    fact_id: str
    source: str
    record_type: str
    payload_entropy_bits: Fraction
    scheduled_exposures: int
    expected_reads: Fraction
    expected_hops: Fraction
    surfaces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise ValueError("fact_id must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if not self.record_type:
            raise ValueError("record_type must be non-empty")
        if (
            isinstance(self.scheduled_exposures, bool)
            or not isinstance(self.scheduled_exposures, int)
            or self.scheduled_exposures < 0
        ):
            raise ValueError("scheduled_exposures must be a non-negative integer")
        object.__setattr__(
            self,
            "payload_entropy_bits",
            _fraction(self.payload_entropy_bits, "payload_entropy_bits"),
        )
        object.__setattr__(
            self,
            "expected_reads",
            _fraction(self.expected_reads, "expected_reads"),
        )
        object.__setattr__(
            self,
            "expected_hops",
            _fraction(self.expected_hops, "expected_hops"),
        )
        surfaces = tuple(self.surfaces)
        if any(not isinstance(surface, str) or not surface for surface in surfaces):
            raise ValueError("surfaces must contain non-empty strings")
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("surfaces must be distinct")
        object.__setattr__(self, "surfaces", surfaces)

    @property
    def information_burden_bits(self) -> Fraction:
        """Train-only self-information repeated over scheduled exposures."""

        return self.payload_entropy_bits * max(self.scheduled_exposures, 1)

    def as_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "source": self.source,
            "record_type": self.record_type,
            "payload_entropy_bits": _fraction_json(self.payload_entropy_bits),
            "scheduled_exposures": self.scheduled_exposures,
            "expected_reads": _fraction_json(self.expected_reads),
            "expected_hops": _fraction_json(self.expected_hops),
            "information_burden_bits": _fraction_json(
                self.information_burden_bits
            ),
            "surfaces": list(self.surfaces),
        }


def route_score(fact: FactMetadata) -> Fraction:
    """Return ``predict_cost - external_cost`` from train-only metadata."""

    predict_cost = fact.payload_entropy_bits / max(fact.scheduled_exposures, 1)
    external_cost = (
        Fraction(1)
        + fact.expected_reads * Fraction(1, 4)
        + fact.expected_hops * Fraction(1, 4)
    )
    return predict_cost - external_cost


def minimally_rounded_quota(total: int, fraction: Fraction) -> int:
    """Return the smallest integer count that preserves a minimum dose."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    target = total * fraction
    return (target.numerator + target.denominator - 1) // target.denominator


@dataclass(frozen=True)
class RouteDecision:
    fact: FactMetadata
    score: Fraction
    rank: int
    route: str
    reason: str

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.route not in {"external", "internal"}:
            raise ValueError("route must be external or internal")

    def as_dict(self) -> dict:
        return {
            **self.fact.as_dict(),
            "score": _fraction_json(self.score),
            "rank": self.rank,
            "route": self.route,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouteManifest:
    split: str
    target_fraction: Fraction
    quota_count: int
    decisions: tuple[RouteDecision, ...]
    policy: str = ROUTE_POLICY
    metadata_scope: str = METADATA_SCOPE

    @property
    def total_facts(self) -> int:
        return len(self.decisions)

    @property
    def external_count(self) -> int:
        return sum(decision.route == "external" for decision in self.decisions)

    @property
    def external_fact_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.fact.fact_id
            for decision in self.decisions
            if decision.route == "external"
        )

    @property
    def rounding_error(self) -> Fraction:
        return Fraction(self.external_count) - self.total_facts * self.target_fraction

    @property
    def total_information_burden(self) -> Fraction:
        return sum(
            (
                decision.fact.information_burden_bits
                for decision in self.decisions
            ),
            Fraction(),
        )

    @property
    def external_information_burden(self) -> Fraction:
        return sum(
            (
                decision.fact.information_burden_bits
                for decision in self.decisions
                if decision.route == "external"
            ),
            Fraction(),
        )

    @property
    def information_burden_fraction(self) -> Fraction:
        if self.total_information_burden == 0:
            return Fraction(1)
        return self.external_information_burden / self.total_information_burden

    @property
    def information_burden_quota_met(self) -> bool:
        return self.information_burden_fraction >= self.target_fraction

    def as_dict(self) -> dict:
        return {
            "format": "memorysplit-route-manifest-v2",
            "split": self.split,
            "policy": self.policy,
            "metadata_scope": self.metadata_scope,
            "train_only_features": list(TRAIN_ONLY_FEATURES),
            "target_external_fraction": _fraction_json(self.target_fraction),
            "total_facts": self.total_facts,
            "quota_facts": self.quota_count,
            "external_facts": self.external_count,
            "rounding_error": _fraction_json(self.rounding_error),
            "total_information_burden_bits": _fraction_json(
                self.total_information_burden
            ),
            "external_information_burden_bits": _fraction_json(
                self.external_information_burden
            ),
            "information_burden_fraction": _fraction_json(
                self.information_burden_fraction
            ),
            "information_burden_quota_met": self.information_burden_quota_met,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }

    def to_bytes(self) -> bytes:
        return _canonical_bytes(self.as_dict())


def build_route_manifest(
    facts: Iterable[FactMetadata],
    split: str,
) -> RouteManifest:
    """Rank facts by frozen cost score and take the fixed dose quota."""

    try:
        target_fraction = SPLIT_FRACTIONS[split]
    except KeyError as error:
        raise ValueError(f"split must be one of {sorted(SPLIT_FRACTIONS)}") from error
    fact_rows = tuple(facts)
    if not fact_rows:
        raise ValueError("routing requires at least one fact")
    if any(not isinstance(fact, FactMetadata) for fact in fact_rows):
        raise TypeError("facts must contain FactMetadata rows")
    fact_ids = [fact.fact_id for fact in fact_rows]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("routing fact ids must be unique")

    ranked = sorted(
        fact_rows,
        key=lambda fact: (-route_score(fact), fact.fact_id),
    )
    quota = minimally_rounded_quota(len(ranked), target_fraction)
    selected_ids = {fact.fact_id for fact in ranked[:quota]}
    total_burden = sum(
        (fact.information_burden_bits for fact in ranked),
        Fraction(),
    )
    target_burden = total_burden * target_fraction

    def selected_burden() -> Fraction:
        return sum(
            (
                fact.information_burden_bits
                for fact in ranked
                if fact.fact_id in selected_ids
            ),
            Fraction(),
        )

    while selected_burden() < target_burden:
        swaps = [
            (incoming, outgoing)
            for incoming in ranked
            if incoming.fact_id not in selected_ids
            for outgoing in ranked
            if outgoing.fact_id in selected_ids
            and (
                incoming.information_burden_bits
                > outgoing.information_burden_bits
            )
        ]
        if not swaps:
            raise ValueError(
                f"{split} fact quota cannot meet information-burden dose"
            )
        incoming, outgoing = min(
            swaps,
            key=lambda pair: (
                -(
                    pair[0].information_burden_bits
                    - pair[1].information_burden_bits
                ),
                -(route_score(pair[0]) - route_score(pair[1])),
                pair[0].fact_id,
                pair[1].fact_id,
            ),
        )
        selected_ids.remove(outgoing.fact_id)
        selected_ids.add(incoming.fact_id)

    decisions = tuple(
        RouteDecision(
            fact=fact,
            score=route_score(fact),
            rank=rank,
            route="external" if fact.fact_id in selected_ids else "internal",
            reason=(
                "selected by descending train score at fixed quota"
                if fact.fact_id in selected_ids and rank < quota
                else "selected to satisfy train-only information-burden quota"
                if fact.fact_id in selected_ids
                else "displaced by train-only information-burden quota"
                if rank < quota
                else "retained after fixed train-score quota"
            ),
        )
        for rank, fact in enumerate(ranked)
    )
    return RouteManifest(
        split=split,
        target_fraction=target_fraction,
        quota_count=quota,
        decisions=decisions,
    )


def build_route_manifests(
    facts: Iterable[FactMetadata],
) -> dict[str, RouteManifest]:
    rows = tuple(facts)
    return {
        split: build_route_manifest(rows, split)
        for split in SPLIT_FRACTIONS
    }
