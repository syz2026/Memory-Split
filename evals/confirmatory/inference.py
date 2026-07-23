"""Deterministic paired inference over seed, world, and pair levels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from statistics import fmean

import numpy as np


HIERARCHY = ("seed", "world", "pair")
CONFIRMATORY_SEED_COUNT = 5
_ALTERNATIVES = {"two-sided", "greater", "less"}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rng_seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > (1 << 128) - 1
    ):
        raise ValueError("rng_seed must be a non-negative 128-bit integer")
    return value


def _alternative(value: object) -> str:
    if value not in _ALTERNATIVES:
        raise ValueError(
            "alternative must be two-sided, greater, or less"
        )
    return str(value)


@dataclass(frozen=True)
class PairedObservation:
    seed: int
    world_id: str
    pair_id: str
    treatment: float
    control: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("world_id", "pair_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "treatment",
            _finite(self.treatment, "treatment"),
        )
        object.__setattr__(
            self,
            "control",
            _finite(self.control, "control"),
        )

    @property
    def difference(self) -> float:
        return self.treatment - self.control


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    seed_effects: tuple[float, ...]
    replicates: tuple[float, ...]
    n_seeds: int
    n_worlds: int
    n_pairs: int
    n_resamples: int
    rng_seed: int
    hierarchy: tuple[str, ...] = HIERARCHY

    def __post_init__(self) -> None:
        if self.hierarchy != HIERARCHY:
            raise ValueError("bootstrap hierarchy must be seed/world/pair")
        for name in ("estimate", "ci_low", "ci_high"):
            _finite(getattr(self, name), name)
        if self.ci_low > self.ci_high:
            raise ValueError("bootstrap interval bounds are reversed")
        if len(self.seed_effects) != self.n_seeds:
            raise ValueError("seed effect count does not match n_seeds")
        if len(self.replicates) != self.n_resamples:
            raise ValueError("replicate count does not match n_resamples")


def nearest_rank_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Nearest-rank interval using zero-based ``ceil(p*n)-1`` indices."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("interval values must be an ordered sequence")
    materialized = [_finite(value, "interval value") for value in values]
    if not materialized:
        raise ValueError("interval values must not be empty")
    confidence_value = _finite(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    materialized.sort()
    tail = round((1.0 - confidence_value) / 2.0, 15)

    def index(percentile: float) -> int:
        return min(
            len(materialized) - 1,
            max(0, math.ceil(percentile * len(materialized)) - 1),
        )

    return (
        materialized[index(tail)],
        materialized[index(1.0 - tail)],
    )


def _panel(
    observations: Sequence[PairedObservation],
) -> dict[int, dict[str, tuple[PairedObservation, ...]]]:
    if isinstance(observations, (str, bytes)) or not isinstance(
        observations,
        Sequence,
    ):
        raise ValueError("paired observations must be an ordered sequence")
    if not observations:
        raise ValueError("paired observations must not be empty")
    if any(not isinstance(row, PairedObservation) for row in observations):
        raise TypeError("paired observations must be PairedObservation values")
    seen: set[tuple[int, str, str]] = set()
    grouped: dict[int, dict[str, list[PairedObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in observations:
        identity = row.seed, row.world_id, row.pair_id
        if identity in seen:
            raise ValueError("duplicate seed/world/pair observation")
        seen.add(identity)
        grouped[row.seed][row.world_id].append(row)
    if len(grouped) != CONFIRMATORY_SEED_COUNT:
        raise ValueError(
            "confirmatory bootstrap requires exactly five paired seeds"
        )
    return {
        seed: {
            world: tuple(
                sorted(world_rows, key=lambda row: row.pair_id)
            )
            for world, world_rows in sorted(worlds.items())
        }
        for seed, worlds in sorted(grouped.items())
    }


def _seed_effect(
    worlds: Mapping[str, Sequence[PairedObservation]],
) -> float:
    return fmean(
        fmean(row.difference for row in rows)
        for rows in worlds.values()
    )


def _sample(
    values: tuple,
    rng: np.random.Generator,
) -> tuple:
    indices = rng.integers(0, len(values), size=len(values))
    return tuple(values[int(index)] for index in indices)


def hierarchical_paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    n_resamples: int,
    rng_seed: int,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Use one shared paired effect and resample seed → world → pair."""

    count = _positive_integer(n_resamples, "n_resamples")
    seed_value = _rng_seed(rng_seed)
    confidence_value = _finite(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    panel = _panel(observations)
    seeds = tuple(panel)
    seed_effects = tuple(_seed_effect(panel[seed]) for seed in seeds)
    estimate = fmean(seed_effects)

    rng = np.random.Generator(np.random.PCG64(seed_value))
    replicates: list[float] = []
    for _ in range(count):
        sampled_seed_effects = []
        for seed in _sample(seeds, rng):
            world_names = tuple(panel[seed])
            sampled_world_effects = []
            for world in _sample(world_names, rng):
                pairs = panel[seed][world]
                sampled_world_effects.append(
                    fmean(row.difference for row in _sample(pairs, rng))
                )
            sampled_seed_effects.append(fmean(sampled_world_effects))
        replicates.append(fmean(sampled_seed_effects))
    low, high = nearest_rank_interval(replicates, confidence_value)
    return BootstrapEstimate(
        estimate=estimate,
        ci_low=low,
        ci_high=high,
        seed_effects=seed_effects,
        replicates=tuple(replicates),
        n_seeds=len(seeds),
        n_worlds=sum(len(worlds) for worlds in panel.values()),
        n_pairs=len(observations),
        n_resamples=count,
        rng_seed=seed_value,
    )


@dataclass(frozen=True)
class ExactTestResult:
    method: str
    alternative: str
    n: int
    statistic: float | int
    p_value: float

    def __post_init__(self) -> None:
        if self.method not in {"paired_sign_flip", "paired_sign"}:
            raise ValueError("unknown exact paired test method")
        _alternative(self.alternative)
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("exact test n must be non-negative")
        _finite(self.statistic, "exact test statistic")
        p_value = _finite(self.p_value, "exact test p-value")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("exact test p-value must be in [0, 1]")


def _differences(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("paired differences must be an ordered sequence")
    result = tuple(_finite(value, "paired difference") for value in values)
    if not result:
        raise ValueError("paired differences must not be empty")
    return result


def exact_sign_flip_test(
    differences: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> ExactTestResult:
    """Enumerate every sign assignment; intended for small paired seed panels."""

    values = _differences(differences)
    alternative_value = _alternative(alternative)
    if len(values) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 pairs")
    observed = fmean(values)
    exact_values = tuple(Fraction.from_float(value) for value in values)
    denominator = max(value.denominator for value in exact_values)
    scaled_values = tuple(
        value.numerator * (denominator // value.denominator)
        for value in exact_values
    )
    observed_sum = sum(scaled_values)
    magnitudes = tuple(abs(value) for value in scaled_values)
    total = 1 << len(values)
    extreme = 0
    for mask in range(total):
        statistic_sum = sum(
            magnitude if mask & (1 << index) else -magnitude
            for index, magnitude in enumerate(magnitudes)
        )
        if alternative_value == "greater":
            is_extreme = statistic_sum >= observed_sum
        elif alternative_value == "less":
            is_extreme = statistic_sum <= observed_sum
        else:
            is_extreme = abs(statistic_sum) >= abs(observed_sum)
        extreme += is_extreme
    return ExactTestResult(
        method="paired_sign_flip",
        alternative=alternative_value,
        n=len(values),
        statistic=observed,
        p_value=extreme / total,
    )


def exact_paired_sign_test(
    differences: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> ExactTestResult:
    """Exact binomial sign test with zero differences removed."""

    values = _differences(differences)
    alternative_value = _alternative(alternative)
    nonzero = tuple(value for value in values if value != 0.0)
    n = len(nonzero)
    positives = sum(value > 0.0 for value in nonzero)
    if n == 0:
        p_value = 1.0
    else:
        denominator = 1 << n
        lower = sum(math.comb(n, count) for count in range(positives + 1))
        upper = sum(math.comb(n, count) for count in range(positives, n + 1))
        if alternative_value == "greater":
            p_value = upper / denominator
        elif alternative_value == "less":
            p_value = lower / denominator
        else:
            p_value = min(1.0, 2.0 * min(lower, upper) / denominator)
    return ExactTestResult(
        method="paired_sign",
        alternative=alternative_value,
        n=n,
        statistic=positives,
        p_value=p_value,
    )


def exact_paired_test(
    differences: Sequence[float],
    *,
    method: str = "sign_flip",
    alternative: str = "two-sided",
) -> ExactTestResult:
    if method in {"sign_flip", "sign-flip"}:
        return exact_sign_flip_test(differences, alternative=alternative)
    if method == "sign":
        return exact_paired_sign_test(differences, alternative=alternative)
    raise ValueError("method must be sign_flip or sign")


def _p_values(raw: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("p-values must be a non-empty mapping")
    result = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("p-value names must be non-empty strings")
        number = _finite(value, f"p-value {name}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"p-value {name} must be in [0, 1]")
        result[name] = number
    return result


def holm_correction(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values keyed deterministically."""

    values = _p_values(p_values)
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in sorted(adjusted)}


def holm_rejections(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, bool]:
    threshold = _finite(alpha, "alpha")
    if not 0.0 < threshold < 1.0:
        raise ValueError("alpha must be between zero and one")
    adjusted = holm_correction(p_values)
    return {name: value <= threshold for name, value in adjusted.items()}


paired_hierarchical_bootstrap = hierarchical_paired_bootstrap
holm_adjust = holm_correction
