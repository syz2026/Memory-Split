from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from evals.reasoning_v3.inference import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_RNG_SEED,
    FAMILY_ORDER,
    PairedItemOutcome,
    _hierarchical_paired_bootstrap,
    _nearest_rank_interval,
    exact_paired_sign_test,
    frozen_hierarchical_bootstrap,
    right_step_aulc,
)


def _small_outcomes() -> list[PairedItemOutcome]:
    rows: list[PairedItemOutcome] = []
    patterns = {
        (0, "a"): ((False, True), (False, True), (True, True)),
        (0, "b"): ((True, True), (False, False), (True, False)),
        (1, "a"): ((False, False), (True, True), (False, True)),
        (1, "b"): ((True, False), (False, True), (False, True)),
    }
    for seed in (0, 1):
        for family in ("a", "b"):
            for index, (dense, split90) in enumerate(patterns[(seed, family)]):
                rows.append(
                    PairedItemOutcome(
                        seed=seed,
                        family=family,
                        item_id=f"{family}/{index}",
                        dense_correct=dense,
                        split90_correct=split90,
                    )
                )
    return rows


def test_exact_test_enumerates_all_signs_retains_zeros_and_includes_equality():
    positive = exact_paired_sign_test([Fraction(1, 10)] * 10)
    assert positive.statistic == Fraction(1, 10)
    assert positive.p_value == Fraction(1, 1024)
    assert positive.permutations == 1024
    assert positive.alternative == "greater"

    zeros_retained = exact_paired_sign_test([1] + [0] * 9)
    assert zeros_retained.p_value == Fraction(1, 2)
    assert zeros_retained.n == 10
    assert zeros_retained.permutations == 1024

    equality_in_upper_tail = exact_paired_sign_test(
        [Fraction(1, 7), Fraction(-1, 7)] + [0] * 8
    )
    assert equality_in_upper_tail.statistic == 0
    assert equality_in_upper_tail.p_value == Fraction(3, 4)
    assert equality_in_upper_tail.zero_deltas_retained == 8


@pytest.mark.parametrize(
    "values",
    [
        [0.0] * 9,
        [0.0] * 11,
        [0.0] * 9 + [float("nan")],
        [0.0] * 9 + [True],
    ],
)
def test_exact_test_rejects_non_frozen_or_non_finite_inputs(values):
    with pytest.raises(ValueError):
        exact_paired_sign_test(values)


def test_hierarchical_bootstrap_is_paired_deterministic_and_nearest_rank():
    rows = _small_outcomes()
    first = _hierarchical_paired_bootstrap(
        rows,
        expected_seeds=(0, 1),
        family_order=("a", "b"),
        items_per_family=3,
        n_draws=2_000,
        rng_seed=17,
    )
    second = _hierarchical_paired_bootstrap(
        list(reversed(rows)),
        expected_seeds=(0, 1),
        family_order=("a", "b"),
        items_per_family=3,
        n_draws=2_000,
        rng_seed=17,
    )
    assert first == second
    assert first.n_draws == 2_000
    assert first.rng_algorithm == "PCG64"
    assert first.hierarchy == ("seed", "family", "item")
    assert -1.0 <= first.interval_95.lower <= first.interval_95.upper <= 1.0
    assert -1.0 <= first.interval_90.lower <= first.interval_90.upper <= 1.0
    assert (
        first.interval_95.lower
        <= first.interval_90.lower
        <= first.interval_90.upper
        <= first.interval_95.upper
    )

    assert _nearest_rank_interval(
        [0, 1, 2, 3],
        Fraction(1, 2),
    ) == (Fraction(0), Fraction(2))


def test_nearest_rank_uses_exact_ranks_at_twenty_thousand_draw_boundary():
    lower, upper = _nearest_rank_interval(
        range(20_000),
        Fraction(95, 100),
    )
    assert lower == 499
    assert upper == 19_499


def test_frozen_pcg64_vector_digest_and_exact_ranks_are_golden():
    rows = [
        PairedItemOutcome(
            seed=seed,
            family=family,
            item_id=f"{family}/{item:03d}",
            dense_correct=(seed + family_index + item) % 5 == 0,
            split90_correct=(3 * seed + 5 * family_index + item) % 7 == 0,
        )
        for seed in range(10)
        for family_index, family in enumerate(FAMILY_ORDER)
        for item in range(512)
    ]
    result = frozen_hierarchical_bootstrap(rows)
    assert result.draws_sha256 == (
        "e91f8758ebc132a987baa6443cb7a4cb98951b33816ceff2b1216d3c423d353c"
    )
    assert result.estimate == Fraction(-4096, 71_680)
    assert result.interval_95.lower == Fraction(-4373, 71_680)
    assert result.interval_95.upper == Fraction(-3812, 71_680)
    assert result.interval_90.lower == Fraction(-4329, 71_680)
    assert result.interval_90.upper == Fraction(-3859, 71_680)


def test_hierarchical_bootstrap_rejects_duplicate_missing_and_unpaired_rows():
    rows = _small_outcomes()
    with pytest.raises(ValueError, match="duplicate"):
        _hierarchical_paired_bootstrap(
            rows + [rows[0]],
            expected_seeds=(0, 1),
            family_order=("a", "b"),
            items_per_family=3,
            n_draws=10,
            rng_seed=0,
        )
    with pytest.raises(ValueError, match="count"):
        _hierarchical_paired_bootstrap(
            rows[:-1],
            expected_seeds=(0, 1),
            family_order=("a", "b"),
            items_per_family=3,
            n_draws=10,
            rng_seed=0,
        )
    with pytest.raises(ValueError, match="boolean"):
        _hierarchical_paired_bootstrap(
            [replace(rows[0], dense_correct=1), *rows[1:]],
            expected_seeds=(0, 1),
            family_order=("a", "b"),
            items_per_family=3,
            n_draws=10,
            rng_seed=0,
        )


def test_public_bootstrap_constants_are_not_caller_overridable():
    assert BOOTSTRAP_DRAWS == 20_000
    assert BOOTSTRAP_RNG_SEED == 0
    assert len(FAMILY_ORDER) == 14
    assert frozen_hierarchical_bootstrap.__kwdefaults__ is None


def test_right_step_aulc_uses_no_interpolation_and_frozen_order():
    assert right_step_aulc((1, 3, 6), (1.0, 2.0, 4.0)) == pytest.approx(3.2)
    with pytest.raises(ValueError):
        right_step_aulc((1, 1, 6), (1.0, 2.0, 4.0))
    with pytest.raises(ValueError):
        right_step_aulc((1, 3), (1.0,))
