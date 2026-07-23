from __future__ import annotations

from dataclasses import replace
import math

import pytest

from evals.confirmatory.inference import (
    PairedObservation,
    exact_paired_sign_test,
    exact_paired_test,
    exact_sign_flip_test,
    hierarchical_paired_bootstrap,
    holm_correction,
    holm_rejections,
    nearest_rank_interval,
)


def _observations() -> list[PairedObservation]:
    rows = []
    for seed in range(1, 6):
        for index in range(2):
            rows.append(
                PairedObservation(
                    seed=seed,
                    world_id=f"seed-{seed}-small",
                    pair_id=f"small-{index}",
                    treatment=1.0,
                    control=0.0,
                )
            )
        for index in range(8):
            rows.append(
                PairedObservation(
                    seed=seed,
                    world_id=f"seed-{seed}-large",
                    pair_id=f"large-{index}",
                    treatment=0.0,
                    control=0.0,
                )
            )
    return rows


def test_hierarchical_bootstrap_equal_weights_seed_world_and_pair_levels():
    result = hierarchical_paired_bootstrap(
        _observations(),
        n_resamples=500,
        rng_seed=20260723,
    )

    assert result.hierarchy == ("seed", "world", "pair")
    assert result.estimate == pytest.approx(0.5)
    assert result.seed_effects == pytest.approx((0.5,) * 5)
    assert result.n_seeds == 5
    assert result.n_worlds == 10
    assert result.n_pairs == 50
    assert len(result.replicates) == 500
    assert result.ci_low <= result.estimate <= result.ci_high


def test_hierarchical_bootstrap_is_order_independent_and_rng_deterministic():
    forward = hierarchical_paired_bootstrap(
        _observations(),
        n_resamples=250,
        rng_seed=91,
    )
    reverse = hierarchical_paired_bootstrap(
        list(reversed(_observations())),
        n_resamples=250,
        rng_seed=91,
    )
    changed_seed = hierarchical_paired_bootstrap(
        _observations(),
        n_resamples=250,
        rng_seed=92,
    )

    assert reverse == forward
    assert changed_seed.replicates != forward.replicates


def test_hierarchical_bootstrap_rejects_nonfive_seed_and_duplicate_panels():
    rows = _observations()
    with pytest.raises(ValueError, match="five"):
        hierarchical_paired_bootstrap(
            [row for row in rows if row.seed != 5],
            n_resamples=20,
            rng_seed=1,
        )
    with pytest.raises(ValueError, match="duplicate"):
        hierarchical_paired_bootstrap(
            [*rows, rows[0]],
            n_resamples=20,
            rng_seed=1,
        )
    with pytest.raises(ValueError, match="finite"):
        hierarchical_paired_bootstrap(
            [replace(rows[0], treatment=math.nan), *rows[1:]],
            n_resamples=20,
            rng_seed=1,
        )


def test_nearest_rank_interval_has_frozen_indices():
    values = tuple(float(index) for index in reversed(range(10_000)))
    assert nearest_rank_interval(values) == (249.0, 9749.0)
    with pytest.raises(ValueError, match="finite"):
        nearest_rank_interval([0.0, math.nan])


def test_exact_sign_flip_is_valid_for_five_paired_seeds():
    differences = [1.0] * 5

    two_sided = exact_sign_flip_test(differences, alternative="two-sided")
    greater = exact_sign_flip_test(differences, alternative="greater")

    assert two_sided.method == "paired_sign_flip"
    assert two_sided.n == 5
    assert two_sided.statistic == 1.0
    assert two_sided.p_value == pytest.approx(2 / 32)
    assert greater.p_value == pytest.approx(1 / 32)
    assert exact_paired_test(
        differences,
        method="sign_flip",
        alternative="greater",
    ) == greater


def test_exact_sign_test_ignores_ties_and_handles_five_seeds_without_scipy():
    all_positive = exact_paired_sign_test(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        alternative="two-sided",
    )
    with_tie = exact_paired_sign_test(
        [0.1, 0.2, 0.0, -0.1, -0.2],
        alternative="two-sided",
    )

    assert all_positive.method == "paired_sign"
    assert all_positive.n == 5
    assert all_positive.p_value == pytest.approx(2 / 32)
    assert with_tie.n == 4
    assert with_tie.statistic == 2
    assert with_tie.p_value == 1.0
    assert exact_paired_test(
        [0.1] * 5,
        method="sign",
        alternative="greater",
    ).p_value == pytest.approx(1 / 32)


def test_holm_correction_is_monotone_and_key_order_independent():
    expected = {"a": 0.03, "b": 0.06, "c": 0.06}

    assert holm_correction({"c": 0.04, "a": 0.01, "b": 0.03}) == expected
    assert holm_rejections(
        {"c": 0.04, "a": 0.01, "b": 0.03},
        alpha=0.05,
    ) == {"a": True, "b": False, "c": False}

    with pytest.raises(ValueError, match="p-value"):
        holm_correction({"bad": 1.1})
