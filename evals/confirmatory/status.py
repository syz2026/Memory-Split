"""Frozen confirmatory status vocabulary and precedence."""

from __future__ import annotations

from collections.abc import Iterable


STATUS_PRECEDENCE = (
    "incomplete",
    "invalid",
    "directional_only",
    "sign_consistent_only",
    "inconclusive",
    "supports_effect",
    "supports_practical_null",
)
_STATUS_RANK = {status: rank for rank, status in enumerate(STATUS_PRECEDENCE)}


def resolve_status(statuses: Iterable[str]) -> str:
    """Return the highest-precedence (earliest) status."""

    values = tuple(statuses)
    if not values:
        raise ValueError("at least one status is required")
    unknown = sorted(set(values) - set(STATUS_PRECEDENCE))
    if unknown:
        raise ValueError(f"unknown confirmatory status: {unknown}")
    return min(values, key=_STATUS_RANK.__getitem__)


def classify_status(
    *,
    complete: bool,
    valid: bool,
    observed_seeds: int,
    required_seeds: int,
    sign_consistent: bool,
    supports_effect: bool,
    supports_practical_null: bool,
) -> str:
    """Classify one study without allowing stronger evidence to mask blockers."""

    for name, value in (
        ("complete", complete),
        ("valid", valid),
        ("sign_consistent", sign_consistent),
        ("supports_effect", supports_effect),
        ("supports_practical_null", supports_practical_null),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be Boolean")
    if (
        isinstance(observed_seeds, bool)
        or not isinstance(observed_seeds, int)
        or observed_seeds < 0
    ):
        raise ValueError("observed_seeds must be a non-negative integer")
    if (
        isinstance(required_seeds, bool)
        or not isinstance(required_seeds, int)
        or required_seeds < 1
    ):
        raise ValueError("required_seeds must be a positive integer")
    if observed_seeds > required_seeds:
        raise ValueError("observed_seeds cannot exceed required_seeds")

    if not complete or observed_seeds == 0:
        return "incomplete"
    if not valid:
        return "invalid"
    if observed_seeds < required_seeds:
        if observed_seeds >= 2 and sign_consistent:
            return "sign_consistent_only"
        return "directional_only"
    if supports_effect:
        return "supports_effect"
    if supports_practical_null:
        return "supports_practical_null"
    if sign_consistent:
        return "sign_consistent_only"
    return "inconclusive"
