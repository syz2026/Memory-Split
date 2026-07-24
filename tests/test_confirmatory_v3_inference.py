from __future__ import annotations

import pytest

from evals.confirmatory.inference import (
    PairedObservation,
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
            treatment=(seed - 4.5) / 1_000,
            control=0.0,
        )
        for seed in range(10)
    ]


def test_v3_exact_test_is_the_frozen_n10_one_sided_exhaustive_test():
    result = v3_exact_sign_flip_test([1.0] * 10)

    assert result.alternative == "greater"
    assert result.n == 10
    assert result.assignments == V3_SIGN_ASSIGNMENTS == 1_024
    assert result.extreme_count == 1
    assert result.p_value == 1 / 1_024

    zeros = v3_exact_sign_flip_test([0.0] * 10)
    assert zeros.assignments == 1_024
    assert zeros.p_value == 1.0

    with pytest.raises(ValueError, match="exactly ten"):
        v3_exact_sign_flip_test([1.0] * 9)


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
    assert first.supports_equivalence is True


def test_v3_practical_equivalence_uses_strict_margin_boundaries():
    assert v3_supports_practical_equivalence(-0.009, 0.009)
    assert not v3_supports_practical_equivalence(-0.01, 0.009)
    assert not v3_supports_practical_equivalence(-0.009, 0.01)
    assert not v3_supports_practical_equivalence(-0.01, 0.01)

    with pytest.raises(ValueError, match="reversed"):
        v3_supports_practical_equivalence(0.001, -0.001)


def test_v3_aulc_is_the_frozen_right_step_integral():
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
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
        tuple((step, 1.0) for step in V3_SNAPSHOT_STEPS)
    ) == 13_582

    with pytest.raises(ValueError, match="five frozen optimizer steps"):
        v3_right_step_aulc(points[:-1])
    with pytest.raises(ValueError, match="ordered optimizer steps"):
        v3_right_step_aulc((points[1], points[0], *points[2:]))
