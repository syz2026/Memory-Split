"""Small deterministic positive, practical-null, and invalid study fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from evals.confirmatory.inference import PairedObservation
from evals.confirmatory.status import STATUS_PRECEDENCE


@dataclass(frozen=True)
class DeterministicFixture:
    name: str
    observations: tuple[PairedObservation, ...]
    complete: bool
    valid: bool
    sign_consistent: bool
    supports_effect: bool
    supports_practical_null: bool
    expected_status: str

    def __post_init__(self) -> None:
        if self.name not in {"positive", "null", "invalid"}:
            raise ValueError("unknown deterministic fixture name")
        if self.expected_status not in STATUS_PRECEDENCE:
            raise ValueError("unknown deterministic fixture status")
        if not self.observations:
            raise ValueError("deterministic fixture requires observations")
        for field in (
            "complete",
            "valid",
            "sign_consistent",
            "supports_effect",
            "supports_practical_null",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be Boolean")


def _rows(*, treatment: float, control: float) -> tuple[PairedObservation, ...]:
    return tuple(
        PairedObservation(
            seed=seed,
            world_id=f"fixture-world-{world}",
            pair_id=f"seed-{seed}-world-{world}-pair-{pair}",
            treatment=treatment,
            control=control,
        )
        for seed in range(1001, 1006)
        for world in range(2)
        for pair in range(2)
    )


def positive_fixture() -> DeterministicFixture:
    return DeterministicFixture(
        name="positive",
        observations=_rows(treatment=0.75, control=0.50),
        complete=True,
        valid=True,
        sign_consistent=True,
        supports_effect=True,
        supports_practical_null=False,
        expected_status="supports_effect",
    )


def null_fixture() -> DeterministicFixture:
    return DeterministicFixture(
        name="null",
        observations=_rows(treatment=0.50, control=0.50),
        complete=True,
        valid=True,
        sign_consistent=False,
        supports_effect=False,
        supports_practical_null=True,
        expected_status="supports_practical_null",
    )


def invalid_fixture() -> DeterministicFixture:
    return DeterministicFixture(
        name="invalid",
        observations=_rows(treatment=0.75, control=0.50),
        complete=True,
        valid=False,
        sign_consistent=True,
        supports_effect=False,
        supports_practical_null=False,
        expected_status="invalid",
    )


def fixture_by_name(name: str) -> DeterministicFixture:
    fixtures = {
        "positive": positive_fixture,
        "null": null_fixture,
        "invalid": invalid_fixture,
    }
    try:
        factory = fixtures[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown deterministic fixture: {name!r}") from exc
    return factory()


practical_null_fixture = null_fixture
