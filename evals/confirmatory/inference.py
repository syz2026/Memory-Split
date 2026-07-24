"""Deterministic paired inference over seed, world, and pair levels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from statistics import fmean

import numpy as np

from msctl.aws_contracts import SNAPSHOT_STEPS


HIERARCHY = ("seed", "world", "pair")
CONFIRMATORY_SEED_COUNT = 5
V3_CONFIRMATORY_SEED_COUNT = 10
V3_SIGN_ASSIGNMENTS = 1 << V3_CONFIRMATORY_SEED_COUNT
V3_BOOTSTRAP_RNG_SEED = 0
V3_BOOTSTRAP_DRAWS = 20_000
V3_BOOTSTRAP_CONFIDENCE = 0.90
V3_EQUIVALENCE_MARGIN = 0.01
V3_SNAPSHOT_STEPS = SNAPSHOT_STEPS
_V3_SEEDS = tuple(range(V3_CONFIRMATORY_SEED_COUNT))
_V3_EQUIVALENCE_MARGIN_EXACT = Fraction(1, 100)
_ALTERNATIVES = {"two-sided", "greater", "less"}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_rational(value: object, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact rational number")
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, Decimal) and value.is_finite():
        return Fraction(value)
    exact_value = getattr(value, "exact_value", None)
    if isinstance(exact_value, Fraction):
        return exact_value
    raise ValueError(
        f"{name} must be an exact rational number, not a binary float"
    )


def _finite_number(value: object, name: str) -> float | Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        return Fraction(value)
    exact_value = getattr(value, "exact_value", None)
    if isinstance(exact_value, Fraction):
        return exact_value
    return _finite(value, name)


def _mean(
    values: Sequence[float | Fraction],
    *,
    exact: bool,
) -> float | Fraction:
    if exact:
        exact_values = tuple(
            _exact_rational(value, "arithmetic-mean value") for value in values
        )
        return sum(exact_values, Fraction()) / len(exact_values)
    return fmean(values)


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
    treatment: float | Fraction
    control: float | Fraction

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
            _finite_number(self.treatment, "treatment"),
        )
        object.__setattr__(
            self,
            "control",
            _finite_number(self.control, "control"),
        )

    @property
    def difference(self) -> float | Fraction:
        return self.treatment - self.control


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float | Fraction
    ci_low: float | Fraction
    ci_high: float | Fraction
    seed_effects: tuple[float | Fraction, ...]
    replicates: tuple[float | Fraction, ...]
    n_seeds: int
    n_worlds: int
    n_pairs: int
    n_resamples: int
    rng_seed: int
    hierarchy: tuple[str, ...] = HIERARCHY

    def __post_init__(self) -> None:
        if self.hierarchy != HIERARCHY:
            raise ValueError("bootstrap hierarchy must be seed/world/pair")
        estimate = _finite_number(self.estimate, "estimate")
        ci_low = _finite_number(self.ci_low, "ci_low")
        ci_high = _finite_number(self.ci_high, "ci_high")
        if ci_low > ci_high:
            raise ValueError("bootstrap interval bounds are reversed")
        if not isinstance(self.seed_effects, (list, tuple)):
            raise ValueError("seed effects must be an ordered sequence")
        if not isinstance(self.replicates, (list, tuple)):
            raise ValueError("bootstrap replicates must be an ordered sequence")
        seed_effects = tuple(
            _finite_number(value, "seed effect") for value in self.seed_effects
        )
        replicates = tuple(
            _finite_number(value, "bootstrap replicate")
            for value in self.replicates
        )
        n_seeds = _positive_integer(self.n_seeds, "n_seeds")
        n_worlds = _positive_integer(self.n_worlds, "n_worlds")
        n_pairs = _positive_integer(self.n_pairs, "n_pairs")
        n_resamples = _positive_integer(self.n_resamples, "n_resamples")
        rng_seed = _rng_seed(self.rng_seed)
        if len(seed_effects) != n_seeds:
            raise ValueError("seed effect count does not match n_seeds")
        if len(replicates) != n_resamples:
            raise ValueError("replicate count does not match n_resamples")
        if not n_seeds <= n_worlds <= n_pairs:
            raise ValueError(
                "bootstrap counts must satisfy n_seeds <= n_worlds <= n_pairs"
            )
        exact = all(isinstance(value, Fraction) for value in seed_effects)
        if estimate != _mean(seed_effects, exact=exact):
            raise ValueError("bootstrap estimate disagrees with seed effects")
        if ci_low < min(replicates) or ci_high > max(replicates):
            raise ValueError("bootstrap interval lies outside its replicates")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "ci_low", ci_low)
        object.__setattr__(self, "ci_high", ci_high)
        object.__setattr__(self, "seed_effects", seed_effects)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "n_seeds", n_seeds)
        object.__setattr__(self, "n_worlds", n_worlds)
        object.__setattr__(self, "n_pairs", n_pairs)
        object.__setattr__(self, "n_resamples", n_resamples)
        object.__setattr__(self, "rng_seed", rng_seed)


def _nearest_rank_bounds(
    values: Sequence[float | Fraction],
    confidence: float,
) -> tuple[float | Fraction, float | Fraction]:
    materialized = sorted(values)
    tail = round((1.0 - confidence) / 2.0, 15)

    def index(percentile: float) -> int:
        return min(
            len(materialized) - 1,
            max(0, math.ceil(percentile * len(materialized)) - 1),
        )

    return (
        materialized[index(tail)],
        materialized[index(1.0 - tail)],
    )


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
    low, high = _nearest_rank_bounds(materialized, confidence_value)
    return float(low), float(high)


def _panel(
    observations: Sequence[PairedObservation],
    *,
    required_seed_count: int,
    required_seeds: tuple[int, ...] | None = None,
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
    if required_seeds is not None and tuple(sorted(grouped)) != required_seeds:
        raise ValueError(
            "v3 confirmatory bootstrap requires exactly seeds 0 through 9"
        )
    if len(grouped) != required_seed_count:
        count_name = {
            CONFIRMATORY_SEED_COUNT: "five",
            V3_CONFIRMATORY_SEED_COUNT: "ten",
        }.get(required_seed_count, str(required_seed_count))
        raise ValueError(
            f"confirmatory bootstrap requires exactly {count_name} paired seeds"
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
    *,
    exact: bool,
) -> float | Fraction:
    world_effects = tuple(
        _mean(
            tuple(row.difference for row in rows),
            exact=exact,
        )
        for rows in worlds.values()
    )
    return _mean(world_effects, exact=exact)


def _sample(
    values: tuple,
    rng: np.random.Generator,
) -> tuple:
    indices = rng.integers(0, len(values), size=len(values))
    return tuple(values[int(index)] for index in indices)


def _hierarchical_paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    n_resamples: int,
    rng_seed: int,
    confidence: float,
    required_seed_count: int,
    required_seeds: tuple[int, ...] | None = None,
    exact: bool = False,
) -> BootstrapEstimate:
    count = _positive_integer(n_resamples, "n_resamples")
    seed_value = _rng_seed(rng_seed)
    confidence_value = _finite(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    panel = _panel(
        observations,
        required_seed_count=required_seed_count,
        required_seeds=required_seeds,
    )
    seeds = tuple(panel)
    seed_effects = tuple(
        _seed_effect(panel[seed], exact=exact) for seed in seeds
    )
    estimate = _mean(seed_effects, exact=exact)

    rng = np.random.Generator(np.random.PCG64(seed_value))
    replicates: list[float | Fraction] = []
    for _ in range(count):
        sampled_seed_effects: list[float | Fraction] = []
        for seed in _sample(seeds, rng):
            world_names = tuple(panel[seed])
            sampled_world_effects: list[float | Fraction] = []
            for world in _sample(world_names, rng):
                pairs = panel[seed][world]
                sampled_world_effects.append(
                    _mean(
                        tuple(
                            row.difference for row in _sample(pairs, rng)
                        ),
                        exact=exact,
                    )
                )
            sampled_seed_effects.append(
                _mean(sampled_world_effects, exact=exact)
            )
        replicates.append(_mean(sampled_seed_effects, exact=exact))
    if exact:
        low, high = _nearest_rank_bounds(
            replicates,
            confidence_value,
        )
    else:
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


def hierarchical_paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    n_resamples: int,
    rng_seed: int,
    confidence: float = 0.95,
) -> BootstrapEstimate:
    """Use one shared paired effect and resample seed → world → pair."""

    return _hierarchical_paired_bootstrap(
        observations,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        confidence=confidence,
        required_seed_count=CONFIRMATORY_SEED_COUNT,
    )


def v3_supports_practical_equivalence(
    ci_low: object,
    ci_high: object,
) -> bool:
    """Apply the frozen strict two-one-sided ±0.01 equivalence rule."""

    low = _exact_rational(
        ci_low,
        "practical-equivalence lower bound",
    )
    high = _exact_rational(
        ci_high,
        "practical-equivalence upper bound",
    )
    if low > high:
        raise ValueError("practical-equivalence interval bounds are reversed")
    return (
        low > -_V3_EQUIVALENCE_MARGIN_EXACT
        and high < _V3_EQUIVALENCE_MARGIN_EXACT
    )


@dataclass(frozen=True)
class PracticalEquivalenceBounds:
    bootstrap: BootstrapEstimate
    confidence: float = V3_BOOTSTRAP_CONFIDENCE
    margin: float = V3_EQUIVALENCE_MARGIN
    bit_generator: str = "PCG64"

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap, BootstrapEstimate):
            raise TypeError("practical-equivalence bootstrap is invalid")
        _exact_rational(
            self.bootstrap.ci_low,
            "practical-equivalence lower bound",
        )
        _exact_rational(
            self.bootstrap.ci_high,
            "practical-equivalence upper bound",
        )
        for value in (
            *self.bootstrap.seed_effects,
            *self.bootstrap.replicates,
        ):
            _exact_rational(value, "practical-equivalence bootstrap value")
        expected_bounds = _nearest_rank_bounds(
            self.bootstrap.replicates,
            V3_BOOTSTRAP_CONFIDENCE,
        )
        if (
            self.bootstrap.ci_low,
            self.bootstrap.ci_high,
        ) != expected_bounds:
            raise ValueError(
                "practical-equivalence bounds disagree with frozen "
                "nearest-rank quantiles"
            )
        if (
            self.bootstrap.n_seeds != V3_CONFIRMATORY_SEED_COUNT
            or self.bootstrap.n_resamples != V3_BOOTSTRAP_DRAWS
            or self.bootstrap.rng_seed != V3_BOOTSTRAP_RNG_SEED
            or self.confidence != V3_BOOTSTRAP_CONFIDENCE
            or self.margin != V3_EQUIVALENCE_MARGIN
            or self.bit_generator != "PCG64"
        ):
            raise ValueError(
                "practical-equivalence bounds disagree with the v3 contract"
            )

    @property
    def supports_equivalence(self) -> bool:
        return v3_supports_practical_equivalence(
            self.bootstrap.ci_low,
            self.bootstrap.ci_high,
        )


def v3_practical_equivalence_bounds(
    observations: Sequence[PairedObservation],
) -> PracticalEquivalenceBounds:
    """Replay the frozen N=10 PCG64 hierarchical practical-null bounds."""

    bootstrap = _hierarchical_paired_bootstrap(
        observations,
        n_resamples=V3_BOOTSTRAP_DRAWS,
        rng_seed=V3_BOOTSTRAP_RNG_SEED,
        confidence=V3_BOOTSTRAP_CONFIDENCE,
        required_seed_count=V3_CONFIRMATORY_SEED_COUNT,
        required_seeds=_V3_SEEDS,
        exact=True,
    )
    return PracticalEquivalenceBounds(bootstrap=bootstrap)


def v3_right_step_aulc(
    points: Sequence[tuple[int, object]],
) -> Fraction:
    """Integrate the five frozen checkpoint values as a right-step curve."""

    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise ValueError("AULC points must be an ordered sequence")
    if len(points) != len(V3_SNAPSHOT_STEPS):
        raise ValueError("AULC requires the five frozen optimizer steps")
    materialized: list[tuple[int, Fraction]] = []
    for index, point in enumerate(points):
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or type(point[0]) is not int
        ):
            raise ValueError(f"AULC point {index} is invalid")
        materialized.append(
            (
                point[0],
                _exact_rational(point[1], f"AULC value {index}"),
            )
        )
    steps = tuple(step for step, _ in materialized)
    if steps != V3_SNAPSHOT_STEPS:
        raise ValueError("AULC points disagree with the ordered optimizer steps")
    previous = 0
    area = Fraction()
    for step, value in materialized:
        area += (step - previous) * value
        previous = step
    return area


@dataclass(frozen=True)
class ExactTestResult:
    method: str
    alternative: str
    n: int
    statistic: float | int | Fraction
    extreme_count: int
    assignments: int
    p_value: float

    def __post_init__(self) -> None:
        if self.method not in {"paired_sign_flip", "paired_sign"}:
            raise ValueError("unknown exact paired test method")
        _alternative(self.alternative)
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("exact test n must be non-negative")
        _finite_number(self.statistic, "exact test statistic")
        if (
            isinstance(self.extreme_count, bool)
            or not isinstance(self.extreme_count, int)
            or isinstance(self.assignments, bool)
            or not isinstance(self.assignments, int)
            or self.assignments < 1
            or not 0 <= self.extreme_count <= self.assignments
        ):
            raise ValueError("exact test assignment counts are invalid")
        p_value = _finite(self.p_value, "exact test p-value")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("exact test p-value must be in [0, 1]")
        if p_value != self.extreme_count / self.assignments:
            raise ValueError("exact test p-value disagrees with assignment counts")

    @property
    def exact_p_value(self) -> Fraction:
        return Fraction(self.extreme_count, self.assignments)


def _differences(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("paired differences must be an ordered sequence")
    result = tuple(_finite(value, "paired difference") for value in values)
    if not result:
        raise ValueError("paired differences must not be empty")
    return result


def _exact_differences(values: Sequence[object]) -> tuple[Fraction, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("paired differences must be an ordered sequence")
    result = tuple(
        _exact_rational(value, "paired difference") for value in values
    )
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
        extreme_count=extreme,
        assignments=total,
        p_value=extreme / total,
    )


def v3_exact_sign_flip_test(
    differences: Sequence[object],
) -> ExactTestResult:
    """Run the frozen one-sided exhaustive sign flip over exactly ten pairs."""

    values = _exact_differences(differences)
    if len(values) != V3_CONFIRMATORY_SEED_COUNT:
        raise ValueError("v3 exact sign-flip test requires exactly ten pairs")
    observed_sum = sum(values, Fraction())
    observed = observed_sum / len(values)
    magnitudes = tuple(abs(value) for value in values)
    extreme = 0
    for mask in range(V3_SIGN_ASSIGNMENTS):
        statistic = sum(
            (
                magnitude if mask & (1 << index) else -magnitude
                for index, magnitude in enumerate(magnitudes)
            ),
            Fraction(),
        )
        extreme += statistic >= observed_sum
    return ExactTestResult(
        method="paired_sign_flip",
        alternative="greater",
        n=len(values),
        statistic=observed,
        extreme_count=extreme,
        assignments=V3_SIGN_ASSIGNMENTS,
        p_value=extreme / V3_SIGN_ASSIGNMENTS,
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
        assignments = 1
        extreme_count = 1
    else:
        assignments = 1 << n
        lower = sum(math.comb(n, count) for count in range(positives + 1))
        upper = sum(math.comb(n, count) for count in range(positives, n + 1))
        if alternative_value == "greater":
            extreme_count = upper
        elif alternative_value == "less":
            extreme_count = lower
        else:
            extreme_count = min(assignments, 2 * min(lower, upper))
    return ExactTestResult(
        method="paired_sign",
        alternative=alternative_value,
        n=n,
        statistic=positives,
        extreme_count=extreme_count,
        assignments=assignments,
        p_value=extreme_count / assignments,
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
