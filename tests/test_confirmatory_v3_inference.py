from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from evals.confirmatory.inference import (
    BootstrapEstimate,
    PairedObservation,
    PracticalEquivalenceBounds,
    V3_BOOTSTRAP_CONFIDENCE,
    V3_BOOTSTRAP_DRAWS,
    V3_BOOTSTRAP_RNG_SEED,
    V3_EQUIVALENCE_MARGIN,
    V3_SIGN_ASSIGNMENTS,
    V3_SNAPSHOT_STEPS,
    v3_exact_sign_flip_test,
    v3_practical_equivalence_bounds,
    v3_right_step_aulc,
    v3_supports_practical_equivalence,
)


def _n10_observations() -> list[PairedObservation]:
    return [
        PairedObservation(
            seed=seed,
            world_id=f"world-{seed}",
            pair_id=f"pair-{seed}",
            treatment=Fraction(2 * seed - 9, 2_000),
            control=Fraction(0),
        )
        for seed in range(10)
    ]


def test_v3_exact_test_is_the_frozen_n10_one_sided_exhaustive_test():
    result = v3_exact_sign_flip_test([Fraction(1)] * 10)

    assert result.alternative == "greater"
    assert result.n == 10
    assert result.assignments == V3_SIGN_ASSIGNMENTS == 1_024
    assert result.extreme_count == 1
    assert result.p_value == 1 / 1_024

    zeros = v3_exact_sign_flip_test([Fraction(0)] * 10)
    assert zeros.assignments == 1_024
    assert zeros.p_value == 1.0

    with pytest.raises(ValueError, match="exactly ten"):
        v3_exact_sign_flip_test([Fraction(1)] * 9)


def test_v3_exact_test_retains_rational_ties_for_the_inclusive_tail():
    differences = (
        Fraction(1, 10),
        Fraction(1, 5),
        Fraction(-3, 10),
        *(Fraction(0) for _ in range(7)),
    )

    result = v3_exact_sign_flip_test(differences)

    assert isinstance(result.statistic, Fraction)
    assert result.statistic == 0
    assert result.extreme_count == 640
    assert result.assignments == 1_024
    assert result.exact_p_value == Fraction(640, 1_024)

    with pytest.raises(ValueError, match="exact rational"):
        v3_exact_sign_flip_test([0.1, 0.2, -0.3, *([0.0] * 7)])


def test_v3_practical_equivalence_bounds_freeze_pcg64_seed_draws_and_confidence():
    first = v3_practical_equivalence_bounds(_n10_observations())
    replay = v3_practical_equivalence_bounds(
        list(reversed(_n10_observations()))
    )

    assert replay == first
    assert first.bit_generator == "PCG64"
    assert first.bootstrap.rng_seed == V3_BOOTSTRAP_RNG_SEED == 0
    assert first.bootstrap.n_resamples == V3_BOOTSTRAP_DRAWS == 20_000
    assert first.confidence == V3_BOOTSTRAP_CONFIDENCE == 0.90
    assert first.margin == V3_EQUIVALENCE_MARGIN == 0.01
    assert first.bootstrap.n_seeds == 10
    assert isinstance(first.bootstrap.ci_low, Fraction)
    assert isinstance(first.bootstrap.ci_high, Fraction)
    assert first.supports_equivalence is True


def test_v3_practical_equivalence_uses_strict_margin_boundaries():
    assert v3_supports_practical_equivalence(
        Fraction(-9, 1_000),
        Fraction(9, 1_000),
    )
    assert not v3_supports_practical_equivalence(
        Fraction(-1, 100),
        Fraction(9, 1_000),
    )
    assert not v3_supports_practical_equivalence(
        Fraction(-9, 1_000),
        Fraction(1, 100),
    )
    assert not v3_supports_practical_equivalence(
        Decimal("-0.01"),
        Decimal("0.01"),
    )

    with pytest.raises(ValueError, match="reversed"):
        v3_supports_practical_equivalence(
            Fraction(1, 1_000),
            Fraction(-1, 1_000),
        )
    with pytest.raises(ValueError, match="exact rational"):
        v3_supports_practical_equivalence(-0.01, 0.01)


def _bootstrap_estimate() -> BootstrapEstimate:
    return BootstrapEstimate(
        estimate=Fraction(0),
        ci_low=Fraction(-1, 10),
        ci_high=Fraction(1, 10),
        seed_effects=(Fraction(-1, 10), Fraction(1, 10)),
        replicates=(Fraction(-1, 10), Fraction(0), Fraction(1, 10)),
        n_seeds=2,
        n_worlds=2,
        n_pairs=2,
        n_resamples=3,
        rng_seed=0,
    )


def test_bootstrap_estimate_accepts_and_validates_exact_finite_values():
    estimate = _bootstrap_estimate()

    assert estimate.estimate == Fraction(0)
    assert estimate.seed_effects == (
        Fraction(-1, 10),
        Fraction(1, 10),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"seed_effects": (float("nan"), 0.0)},
        {"replicates": (-0.1, float("inf"), 0.1)},
        {"n_seeds": 2.0},
        {"n_worlds": True},
        {"n_pairs": 2.0},
        {"n_pairs": 1},
        {"n_resamples": 3.0},
        {"rng_seed": True},
        {"estimate": Fraction(1, 10)},
    ],
)
def test_bootstrap_estimate_rejects_nonfinite_mistyped_or_inconsistent_values(
    changes,
):
    with pytest.raises(ValueError):
        replace(_bootstrap_estimate(), **changes)


def _claimed_v3_bootstrap(
    *,
    ci_low: Fraction,
    ci_high: Fraction,
) -> BootstrapEstimate:
    replicates = (
        *(Fraction(-1, 10) for _ in range(10_000)),
        *(Fraction(1, 10) for _ in range(10_000)),
    )
    return BootstrapEstimate(
        estimate=Fraction(0),
        ci_low=ci_low,
        ci_high=ci_high,
        seed_effects=tuple(Fraction(0) for _ in range(10)),
        replicates=replicates,
        n_seeds=10,
        n_worlds=10,
        n_pairs=10,
        n_resamples=20_000,
        rng_seed=0,
    )


def test_v3_practical_bounds_recompute_frozen_nearest_rank_quantiles():
    with pytest.raises(ValueError, match="nearest-rank|quantile|bounds"):
        PracticalEquivalenceBounds(
            bootstrap=_claimed_v3_bootstrap(
                ci_low=Fraction(-9, 1_000),
                ci_high=Fraction(9, 1_000),
            )
        )

    validated = PracticalEquivalenceBounds(
        bootstrap=_claimed_v3_bootstrap(
            ci_low=Fraction(-1, 10),
            ci_high=Fraction(1, 10),
        )
    )
    assert validated.supports_equivalence is False


def test_v3_aulc_is_the_frozen_right_step_integral():
    values = tuple(Fraction(value) for value in range(1, 6))
    points = tuple(zip(V3_SNAPSHOT_STEPS, values, strict=True))
    expected = sum(
        (step - previous) * value
        for previous, (step, value) in zip(
            (0, *V3_SNAPSHOT_STEPS[:-1]),
            points,
            strict=True,
        )
    )

    assert V3_SNAPSHOT_STEPS == (1_358, 3_396, 6_791, 10_187, 13_582)
    assert v3_right_step_aulc(points) == expected
    assert v3_right_step_aulc(
        tuple((step, Fraction(1)) for step in V3_SNAPSHOT_STEPS)
    ) == 13_582

    with pytest.raises(ValueError, match="five frozen optimizer steps"):
        v3_right_step_aulc(points[:-1])
    with pytest.raises(ValueError, match="ordered optimizer steps"):
        v3_right_step_aulc((points[1], points[0], *points[2:]))
