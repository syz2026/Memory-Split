"""Exact paired tests and provider-stratified hierarchical bootstrap."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


CONFIRMATORY_SEED_COUNT = 5  # retained for exact legacy 360M replay
COHORT_135M_SEED_COUNT = 10


@dataclass(frozen=True)
class PairedObservation:
    seed: int
    provider: str
    world_id: str
    pair_id: str
    dense_score: float
    split90_score: float

    @property
    def difference(self) -> float:
        return float(self.split90_score - self.dense_score)


@dataclass(frozen=True)
class ExactTestResult:
    statistic: float
    p_value: float
    alternative: str
    n: int
    permutations: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    lower: float
    upper: float
    standard_error: float
    confidence: float
    n_resamples: int
    rng_seed: int
    seed_count: int
    provider_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provider_counts"] = dict(self.provider_counts)
        return value


@dataclass(frozen=True)
class PracticalEquivalenceResult:
    equivalent: bool
    lower: float
    upper: float
    margin: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_values(values: Sequence[float], *, label: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain only numeric values")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{label} must contain only finite values")
        result.append(converted)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def exact_sign_flip_test(
    differences: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> ExactTestResult:
    """Exhaust the paired randomization distribution for up to 20 seeds."""

    values = _finite_values(differences, label="differences")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    n = len(values)
    if n > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 pairs")
    observed = float(np.mean(values))
    extreme = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        statistic = sum(sign * value for sign, value in zip(signs, values)) / n
        if alternative == "greater":
            selected = statistic >= observed - tolerance
        elif alternative == "less":
            selected = statistic <= observed + tolerance
        else:
            selected = abs(statistic) >= abs(observed) - tolerance
        extreme += int(selected)
    permutations = 1 << n
    return ExactTestResult(
        statistic=observed,
        p_value=extreme / permutations,
        alternative=alternative,
        n=n,
        permutations=permutations,
    )


def _normalize_observations(
    observations: Sequence[PairedObservation],
    *,
    expected_seeds: Sequence[int],
) -> tuple[
    dict[str, dict[int, dict[str, list[float]]]],
    dict[int, float],
]:
    seeds = tuple(expected_seeds)
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("expected_seeds must be unique integers")
    grouped: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    identities = set()
    provider_by_seed = {}
    ordered = sorted(
        observations,
        key=lambda row: (row.seed, row.provider, row.world_id, row.pair_id),
    )
    for row in ordered:
        if not isinstance(row, PairedObservation):
            raise ValueError("observations must be PairedObservation instances")
        if row.seed not in seeds:
            raise ValueError(f"observation has an unexpected seed: {row.seed}")
        if (
            not row.provider
            or not row.world_id
            or not row.pair_id
            or any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0 <= float(score) <= 1
                for score in (row.dense_score, row.split90_score)
            )
        ):
            raise ValueError("observation fields are invalid")
        identity = (row.seed, row.world_id, row.pair_id)
        if identity in identities:
            raise ValueError(f"duplicate counterfactual pair observation: {identity}")
        identities.add(identity)
        prior = provider_by_seed.setdefault(row.seed, row.provider)
        if prior != row.provider:
            raise ValueError(f"seed {row.seed} spans more than one provider")
        grouped[row.provider][row.seed][row.world_id].append(row.difference)
    missing = sorted(set(seeds) - set(provider_by_seed))
    if missing:
        raise ValueError(f"observations are missing frozen seeds: {missing}")

    seed_estimates = {}
    for provider in sorted(grouped):
        for seed in sorted(grouped[provider]):
            worlds = grouped[provider][seed]
            if not worlds or any(not values for values in worlds.values()):
                raise ValueError(f"seed {seed} has an empty world or pair stratum")
            seed_estimates[seed] = float(
                np.mean(
                    [
                        np.mean(worlds[world])
                        for world in sorted(worlds)
                    ]
                )
            )
    return {
        provider: {
            seed: {
                world: list(grouped[provider][seed][world])
                for world in sorted(grouped[provider][seed])
            }
            for seed in sorted(grouped[provider])
        }
        for provider in sorted(grouped)
    }, seed_estimates


def seed_mean_differences(
    observations: Sequence[PairedObservation],
    *,
    expected_seeds: Sequence[int],
) -> dict[int, float]:
    _, estimates = _normalize_observations(
        observations,
        expected_seeds=expected_seeds,
    )
    return estimates


def provider_mean_deltas(
    observations: Sequence[PairedObservation],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, float]:
    grouped, estimates = _normalize_observations(
        observations,
        expected_seeds=expected_seeds,
    )
    return {
        provider: float(np.mean([estimates[seed] for seed in seeds]))
        for provider, seeds in (
            (provider, sorted(grouped[provider]))
            for provider in sorted(grouped)
        )
    }


def hierarchical_paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    expected_seeds: Sequence[int],
    n_resamples: int,
    rng_seed: int,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Resample provider-fixed seed → world → counterfactual-pair strata."""

    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(rng_seed, bool)
        or not isinstance(rng_seed, int)
        or rng_seed < 0
    ):
        raise ValueError("rng_seed must be a non-negative integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 < float(confidence) < 1
    ):
        raise ValueError("confidence must lie strictly between zero and one")
    grouped, estimates = _normalize_observations(
        observations,
        expected_seeds=expected_seeds,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    draws = np.empty(n_resamples, dtype=np.float64)
    providers = sorted(grouped)
    for draw_index in range(n_resamples):
        sampled_seed_values = []
        for provider in providers:
            seeds = sorted(grouped[provider])
            sampled_seed_indices = rng.integers(0, len(seeds), size=len(seeds))
            for seed_index in sampled_seed_indices:
                worlds = grouped[provider][seeds[int(seed_index)]]
                world_names = sorted(worlds)
                sampled_world_indices = rng.integers(
                    0,
                    len(world_names),
                    size=len(world_names),
                )
                sampled_world_values = []
                for world_index in sampled_world_indices:
                    pairs = worlds[world_names[int(world_index)]]
                    pair_indices = rng.integers(0, len(pairs), size=len(pairs))
                    sampled_world_values.append(
                        float(np.mean([pairs[int(index)] for index in pair_indices]))
                    )
                sampled_seed_values.append(float(np.mean(sampled_world_values)))
        draws[draw_index] = float(np.mean(sampled_seed_values))
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return BootstrapEstimate(
        estimate=float(np.mean([estimates[seed] for seed in expected_seeds])),
        lower=float(lower),
        upper=float(upper),
        standard_error=float(np.std(draws, ddof=1)) if n_resamples > 1 else 0.0,
        confidence=float(confidence),
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        seed_count=len(tuple(expected_seeds)),
        provider_counts={
            provider: len(grouped[provider])
            for provider in providers
        },
    )


def practical_equivalence(
    interval: BootstrapEstimate,
    *,
    margin: float = 0.01,
) -> PracticalEquivalenceResult:
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or margin <= 0
    ):
        raise ValueError("equivalence margin must be positive and finite")
    return PracticalEquivalenceResult(
        equivalent=interval.lower > -float(margin)
        and interval.upper < float(margin),
        lower=interval.lower,
        upper=interval.upper,
        margin=float(margin),
        confidence=interval.confidence,
    )


def holm_adjust(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    if not p_values:
        return {}
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    checked = {}
    for name, value in p_values.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError("Holm inputs must be named p-values in [0, 1]")
        checked[name] = float(value)
    ordered = sorted(checked.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = {
            "adjusted_p_value": running,
            "raw_p_value": value,
            "reject": running <= alpha,
        }
    return {name: adjusted[name] for name in p_values}


def fixed_checkpoint_aulc(
    checkpoints: Sequence[int],
    deltas: Sequence[float],
) -> float:
    if len(checkpoints) != len(deltas) or len(checkpoints) < 2:
        raise ValueError("AULC requires matching checkpoint and delta sequences")
    if (
        any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in checkpoints
        )
        or list(checkpoints) != sorted(set(checkpoints))
    ):
        raise ValueError("AULC checkpoints must be sorted unique positive integers")
    values = _finite_values(deltas, label="AULC deltas")
    area = sum(
        (values[index] + values[index + 1])
        * (checkpoints[index + 1] - checkpoints[index])
        / 2
        for index in range(len(values) - 1)
    )
    return area / (checkpoints[-1] - checkpoints[0])
