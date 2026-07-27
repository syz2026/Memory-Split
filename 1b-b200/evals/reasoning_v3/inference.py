"""Frozen exact paired inference and seed-family-item hierarchical bootstrap."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from evals.reasoning_v3.contracts import ACCEPTED_ITEMS_PER_FAMILY, FAMILY_ORDER
from msctl.reasoning_cohort import SEEDS


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_RNG_SEED = 0
BOOTSTRAP_RNG_ALGORITHM = "PCG64"
PRIMARY_ALPHA = Fraction(1, 20)
PRACTICAL_NULL_MARGIN = Fraction(1, 100)


@dataclass(frozen=True)
class PairedItemOutcome:
    seed: int
    family: str
    item_id: str
    dense_correct: bool
    split90_correct: bool

    @property
    def difference(self) -> int:
        return int(self.split90_correct) - int(self.dense_correct)


@dataclass(frozen=True)
class ExactTestResult:
    statistic: Fraction
    p_value: Fraction
    alternative: str
    n: int
    permutations: int
    zero_deltas_retained: int
    equality_included: bool
    add_one_correction: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "add_one_correction": self.add_one_correction,
            "alternative": self.alternative,
            "equality_included": self.equality_included,
            "n": self.n,
            "p_value": float(self.p_value),
            "p_value_fraction": _fraction_dict(self.p_value),
            "permutations": self.permutations,
            "statistic": float(self.statistic),
            "statistic_fraction": _fraction_dict(self.statistic),
            "zero_deltas_retained": self.zero_deltas_retained,
        }


@dataclass(frozen=True)
class ConfidenceInterval:
    confidence: Fraction
    lower: Fraction
    upper: Fraction
    method: str = "nearest_rank"

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence": float(self.confidence),
            "confidence_fraction": _fraction_dict(self.confidence),
            "lower": float(self.lower),
            "lower_fraction": _fraction_dict(self.lower),
            "method": self.method,
            "upper": float(self.upper),
            "upper_fraction": _fraction_dict(self.upper),
        }


@dataclass(frozen=True)
class BootstrapResult:
    estimate: Fraction
    interval_95: ConfidenceInterval
    interval_90: ConfidenceInterval
    standard_error: float
    n_draws: int
    rng_seed: int
    rng_algorithm: str
    hierarchy: tuple[str, str, str]
    seed_deltas: tuple[Fraction, ...]
    draws_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "draws_sha256": self.draws_sha256,
            "estimate": float(self.estimate),
            "estimate_fraction": _fraction_dict(self.estimate),
            "hierarchy": list(self.hierarchy),
            "interval_90": self.interval_90.as_dict(),
            "interval_95": self.interval_95.as_dict(),
            "n_draws": self.n_draws,
            "rng_algorithm": self.rng_algorithm,
            "rng_seed": self.rng_seed,
            "seed_delta_fractions": [
                _fraction_dict(value) for value in self.seed_deltas
            ],
            "seed_deltas": [float(value) for value in self.seed_deltas],
            "standard_error": self.standard_error,
        }


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _as_fraction(value: object, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label} must contain only numeric values")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(str(value))
    raise ValueError(f"{label} must contain only finite numeric values")


def _finite_numbers(
    values: Sequence[int | float | Fraction],
    label: str,
) -> list[Fraction]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric sequence")
    return [_as_fraction(value, label) for value in values]


def exact_paired_sign_test(
    seed_deltas: Sequence[int | float | Fraction],
) -> ExactTestResult:
    """Enumerate all 2^10 signs for the frozen greater-than-zero mean test."""

    values = _finite_numbers(seed_deltas, "seed deltas")
    if len(values) != len(SEEDS):
        raise ValueError("exact paired test requires exactly ten frozen seed deltas")
    observed_sum = sum(values, start=Fraction())
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        permuted_sum = sum(
            (sign * value for sign, value in zip(signs, values, strict=True)),
            start=Fraction(),
        )
        extreme += int(permuted_sum >= observed_sum)
    permutations = 1 << len(values)
    return ExactTestResult(
        statistic=observed_sum / len(values),
        p_value=Fraction(extreme, permutations),
        alternative="greater",
        n=len(values),
        permutations=permutations,
        zero_deltas_retained=sum(value == 0 for value in values),
        equality_included=True,
        add_one_correction=False,
    )


def _validate_bootstrap_dimensions(
    expected_seeds: Sequence[int],
    family_order: Sequence[str],
    items_per_family: int,
    n_draws: int,
    rng_seed: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    seeds = tuple(expected_seeds)
    families = tuple(family_order)
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("expected seeds must be unique integers")
    if (
        not families
        or len(families) != len(set(families))
        or any(not isinstance(family, str) or not family for family in families)
    ):
        raise ValueError("family order must contain unique non-empty names")
    for value, label in (
        (items_per_family, "items_per_family"),
        (n_draws, "n_draws"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or rng_seed < 0:
        raise ValueError("rng_seed must be a non-negative integer")
    return seeds, families


def _paired_delta_array(
    outcomes: Sequence[PairedItemOutcome],
    *,
    expected_seeds: Sequence[int],
    family_order: Sequence[str],
    items_per_family: int,
) -> np.ndarray:
    seeds, families = _validate_bootstrap_dimensions(
        expected_seeds,
        family_order,
        items_per_family,
        1,
        0,
    )
    if isinstance(outcomes, (str, bytes, bytearray)):
        raise ValueError("paired outcomes must be a sequence")
    seed_position = {seed: index for index, seed in enumerate(seeds)}
    family_position = {
        family: index for index, family in enumerate(families)
    }
    grouped: dict[tuple[int, str], dict[str, int]] = {
        (seed, family): {} for seed in seeds for family in families
    }
    for row in outcomes:
        if not isinstance(row, PairedItemOutcome):
            raise ValueError("paired outcomes must be PairedItemOutcome instances")
        if row.seed not in seed_position or row.family not in family_position:
            raise ValueError("paired outcome is outside the frozen strata")
        if (
            not isinstance(row.item_id, str)
            or not row.item_id
            or type(row.dense_correct) is not bool
            or type(row.split90_correct) is not bool
        ):
            raise ValueError("paired outcome identities and correctness must be boolean")
        stratum = grouped[(row.seed, row.family)]
        if row.item_id in stratum:
            raise ValueError(
                f"duplicate paired item outcome: {row.seed}/{row.family}/{row.item_id}"
            )
        stratum[row.item_id] = row.difference

    expected_total = len(seeds) * len(families) * items_per_family
    if len(outcomes) != expected_total:
        raise ValueError(
            f"paired outcome count differs: {len(outcomes)}/{expected_total}"
        )
    result = np.empty(
        (len(seeds), len(families), items_per_family),
        dtype=np.int8,
    )
    identities_by_family: dict[str, tuple[str, ...]] = {}
    for seed in seeds:
        for family in families:
            stratum = grouped[(seed, family)]
            if len(stratum) != items_per_family:
                raise ValueError(
                    f"paired item count differs for seed={seed}, family={family}"
                )
            identities = tuple(sorted(stratum))
            prior = identities_by_family.setdefault(family, identities)
            if prior != identities:
                raise ValueError(
                    f"paired item identities differ across seeds for {family}"
                )
            result[
                seed_position[seed],
                family_position[family],
                :,
            ] = [stratum[item_id] for item_id in identities]
    return result


def _nearest_rank_interval(
    draws: Sequence[int | float | Fraction] | np.ndarray,
    confidence: int | float | Fraction,
    *,
    denominator: int = 1,
) -> tuple[Fraction, Fraction]:
    """Return two-sided nearest-rank bounds using one-indexed ceil ranks."""

    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("nearest-rank interval inputs are invalid")
    if isinstance(draws, np.ndarray):
        if draws.ndim != 1 or draws.size == 0:
            raise ValueError("nearest-rank interval inputs are invalid")
        raw_values = draws.tolist()
    else:
        if isinstance(draws, (str, bytes, bytearray)):
            raise ValueError("nearest-rank interval inputs are invalid")
        raw_values = list(draws)
    if not raw_values:
        raise ValueError("nearest-rank interval inputs are invalid")
    try:
        confidence_fraction = _as_fraction(confidence, "confidence")
        values = [
            _as_fraction(value, "nearest-rank draws") / denominator
            for value in raw_values
        ]
    except ValueError as error:
        raise ValueError("nearest-rank interval inputs are invalid") from error
    if not 0 < confidence_fraction < 1:
        raise ValueError("nearest-rank interval inputs are invalid")
    values.sort()
    alpha = (1 - confidence_fraction) / 2

    def ceil_fraction(value: Fraction) -> int:
        return -(-value.numerator // value.denominator)

    lower_rank = max(1, ceil_fraction(alpha * len(values)))
    upper_rank = min(
        len(values),
        ceil_fraction((1 - alpha) * len(values)),
    )
    return values[lower_rank - 1], values[upper_rank - 1]


def _hierarchical_paired_bootstrap(
    outcomes: Sequence[PairedItemOutcome],
    *,
    expected_seeds: Sequence[int],
    family_order: Sequence[str],
    items_per_family: int,
    n_draws: int,
    rng_seed: int,
) -> BootstrapResult:
    """Resample paired deltas at seed, then family, then item levels."""

    seeds, families = _validate_bootstrap_dimensions(
        expected_seeds,
        family_order,
        items_per_family,
        n_draws,
        rng_seed,
    )
    deltas = _paired_delta_array(
        outcomes,
        expected_seeds=seeds,
        family_order=families,
        items_per_family=items_per_family,
    )
    scientific_denominator = len(seeds) * len(families) * items_per_family
    seed_denominator = len(families) * items_per_family
    seed_deltas = tuple(
        Fraction(int(deltas[index].sum()), seed_denominator)
        for index in range(len(seeds))
    )
    observed = Fraction(int(deltas.sum()), scientific_denominator)

    negative_probability = (deltas == -1).mean(axis=2)
    positive_probability = (deltas == 1).mean(axis=2)
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    draw_numerators = np.empty(n_draws, dtype=np.int64)
    seed_count = len(seeds)
    family_count = len(families)
    chunk_size = 512
    for start in range(0, n_draws, chunk_size):
        size = min(chunk_size, n_draws - start)
        sampled_seeds = rng.integers(
            0,
            seed_count,
            size=(size, seed_count),
        )
        sampled_families = rng.integers(
            0,
            family_count,
            size=(size, seed_count, family_count),
        )
        p_negative = negative_probability[
            sampled_seeds[:, :, None],
            sampled_families,
        ]
        p_positive = positive_probability[
            sampled_seeds[:, :, None],
            sampled_families,
        ]
        negative_counts = rng.binomial(items_per_family, p_negative)
        remaining = items_per_family - negative_counts
        denominator = 1.0 - p_negative
        conditional_positive = np.divide(
            p_positive,
            denominator,
            out=np.zeros_like(p_positive),
            where=denominator > 0,
        )
        conditional_positive = np.clip(conditional_positive, 0.0, 1.0)
        positive_counts = rng.binomial(remaining, conditional_positive)
        draw_numerators[start : start + size] = (
            positive_counts.astype(np.int64)
            - negative_counts.astype(np.int64)
        ).sum(axis=(1, 2))

    lower_95, upper_95 = _nearest_rank_interval(
        draw_numerators,
        Fraction(95, 100),
        denominator=scientific_denominator,
    )
    lower_90, upper_90 = _nearest_rank_interval(
        draw_numerators,
        Fraction(90, 100),
        denominator=scientific_denominator,
    )
    floating_draws = draw_numerators.astype(np.float64) / scientific_denominator
    return BootstrapResult(
        estimate=observed,
        interval_95=ConfidenceInterval(Fraction(95, 100), lower_95, upper_95),
        interval_90=ConfidenceInterval(Fraction(90, 100), lower_90, upper_90),
        standard_error=(
            float(np.std(floating_draws, ddof=1)) if n_draws > 1 else 0.0
        ),
        n_draws=n_draws,
        rng_seed=rng_seed,
        rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
        hierarchy=("seed", "family", "item"),
        seed_deltas=seed_deltas,
        draws_sha256=hashlib.sha256(
            draw_numerators.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
    )


def frozen_hierarchical_bootstrap(
    outcomes: Sequence[PairedItemOutcome],
) -> BootstrapResult:
    """Run the non-overridable 20,000-draw PCG64(0) frozen bootstrap."""

    return _hierarchical_paired_bootstrap(
        outcomes,
        expected_seeds=SEEDS,
        family_order=FAMILY_ORDER,
        items_per_family=ACCEPTED_ITEMS_PER_FAMILY,
        n_draws=BOOTSTRAP_DRAWS,
        rng_seed=BOOTSTRAP_RNG_SEED,
    )


def right_step_aulc(
    checkpoints: Sequence[int],
    values: Sequence[int | float | Fraction],
) -> Fraction:
    """Average a no-interpolation right-step trajectory over frozen intervals."""

    steps = tuple(checkpoints)
    scores = _finite_numbers(values, "AULC values")
    if (
        len(steps) != len(scores)
        or len(steps) < 2
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in steps
        )
        or tuple(sorted(set(steps))) != steps
    ):
        raise ValueError("AULC requires matching sorted unique positive checkpoints")
    width = steps[-1] - steps[0]
    area = sum(
        scores[index] * (steps[index] - steps[index - 1])
        for index in range(1, len(steps))
    )
    return area / width


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_RNG_SEED",
    "BootstrapResult",
    "ConfidenceInterval",
    "ExactTestResult",
    "PairedItemOutcome",
    "exact_paired_sign_test",
    "frozen_hierarchical_bootstrap",
    "right_step_aulc",
]
