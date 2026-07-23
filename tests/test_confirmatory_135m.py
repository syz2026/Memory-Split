from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.confirmatory.inference import (
    BootstrapEstimate,
    ExactTestResult,
    PairedObservation,
    exact_sign_flip_test,
    fixed_checkpoint_aulc,
    hierarchical_paired_bootstrap,
    holm_adjust,
    practical_equivalence,
    provider_mean_deltas,
)
from evals.confirmatory.reporting import (
    ConfirmatoryEvidence,
    ValidityGates,
    assign_verdict,
    build_confirmatory_report,
)
from evals.confirmatory.study_lock import (
    LEGACY_360M_SEEDS,
    StudyLock,
    load_135m_study_lock,
)
from msctl.cohort import COHORT_ID, SEEDS


ROOT = Path(__file__).resolve().parents[1]


def _observations(deltas: list[float]) -> list[PairedObservation]:
    rows = []
    for seed, delta in enumerate(deltas):
        provider = "farmshare-l40s" if seed in {0, 1, 2, 5, 6, 7} else "mit-slurm"
        for world in range(3):
            for pair in range(4):
                rows.append(
                    PairedObservation(
                        seed=seed,
                        provider=provider,
                        world_id=f"w{world}",
                        pair_id=f"w{world}-p{pair}",
                        dense_score=0.50,
                        split90_score=0.50 + delta,
                    )
                )
    return rows


def _bootstrap(lower: float, upper: float, confidence: float) -> BootstrapEstimate:
    return BootstrapEstimate(
        estimate=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        standard_error=(upper - lower) / 4,
        confidence=confidence,
        n_resamples=20_000,
        rng_seed=1,
        seed_count=10,
        provider_counts={"farmshare-l40s": 6, "mit-slurm": 4},
    )


def _exact(p: float, statistic: float = 0.02) -> ExactTestResult:
    return ExactTestResult(
        statistic=statistic,
        p_value=p,
        alternative="greater",
        n=10,
        permutations=1024,
    )


def test_exact_sign_flip_preserves_n5_and_supports_n10():
    legacy = exact_sign_flip_test([0.1] * 5)
    assert legacy.n == len(LEGACY_360M_SEEDS)
    assert legacy.p_value == pytest.approx(2 / 32)
    one_sided = exact_sign_flip_test([0.1] * 5, alternative="greater")
    assert one_sided.p_value == pytest.approx(1 / 32)
    n10 = exact_sign_flip_test([0.02] * 10, alternative="greater")
    assert n10.p_value == pytest.approx(1 / 1024)
    assert n10.permutations == 1024


def test_provider_stratified_hierarchical_bootstrap_is_deterministic():
    rows = _observations([0.03] * 10)
    first = hierarchical_paired_bootstrap(
        rows,
        expected_seeds=SEEDS,
        n_resamples=500,
        rng_seed=17,
    )
    second = hierarchical_paired_bootstrap(
        list(reversed(rows)),
        expected_seeds=SEEDS,
        n_resamples=500,
        rng_seed=17,
    )
    assert first == second
    assert first.estimate == pytest.approx(0.03)
    assert first.lower > 0
    assert first.provider_counts == {"farmshare-l40s": 6, "mit-slurm": 4}


def test_provider_guardrail_uses_equal_seed_means():
    rows = _observations([0.01, 0.02, 0.03, -0.01, -0.02, 0.04, 0.05, 0.06, -0.03, -0.04])
    means = provider_mean_deltas(rows, expected_seeds=SEEDS)
    assert means["farmshare-l40s"] > 0
    assert means["mit-slurm"] < 0


def test_equivalence_is_strictly_inside_margin():
    assert practical_equivalence(_bootstrap(-0.009, 0.009, 0.90)).equivalent
    assert not practical_equivalence(_bootstrap(-0.01, 0.009, 0.90)).equivalent
    assert not practical_equivalence(_bootstrap(-0.009, 0.01, 0.90)).equivalent


def test_holm_adjustment_is_monotone_and_family_complete():
    adjusted = holm_adjust({"graph": 0.01, "non_path": 0.02}, alpha=0.05)
    assert adjusted["graph"]["adjusted_p_value"] == pytest.approx(0.02)
    assert adjusted["non_path"]["adjusted_p_value"] == pytest.approx(0.02)
    assert all(item["reject"] for item in adjusted.values())


def test_fixed_checkpoint_aulc_is_secondary_and_deterministic():
    assert fixed_checkpoint_aulc(
        [1358, 3396, 6791, 10187, 13582],
        [0.02, 0.02, 0.02, 0.02, 0.02],
    ) == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("evidence", "verdict"),
    [
        (
            ConfirmatoryEvidence(
                cohort_complete=True,
                validity=ValidityGates(),
                primary_exact=_exact(0.05),
                primary_interval_95=_bootstrap(0.000001, 0.04, 0.95),
                equivalence_interval_90=_bootstrap(0.001, 0.03, 0.90),
                provider_means={"farmshare-l40s": 0.02, "mit-slurm": 0.01},
                family_tests={},
            ),
            "supports_effect",
        ),
        (
            ConfirmatoryEvidence(
                cohort_complete=True,
                validity=ValidityGates(),
                primary_exact=_exact(0.001, statistic=0.002),
                primary_interval_95=_bootstrap(-0.001, 0.006, 0.95),
                equivalence_interval_90=_bootstrap(-0.009, 0.009, 0.90),
                provider_means={"farmshare-l40s": 0.003, "mit-slurm": 0.001},
                family_tests={},
            ),
            "supports_practical_null",
        ),
        (
            ConfirmatoryEvidence(
                cohort_complete=True,
                validity=ValidityGates(),
                primary_exact=_exact(0.20),
                primary_interval_95=_bootstrap(-0.02, 0.03, 0.95),
                equivalence_interval_90=_bootstrap(-0.02, 0.02, 0.90),
                provider_means={"farmshare-l40s": 0.01, "mit-slurm": -0.01},
                family_tests={},
            ),
            "inconclusive",
        ),
        (
            ConfirmatoryEvidence(
                cohort_complete=False,
                validity=ValidityGates(),
                primary_exact=_exact(0.001),
                primary_interval_95=_bootstrap(0.01, 0.03, 0.95),
                equivalence_interval_90=_bootstrap(0.01, 0.03, 0.90),
                provider_means={"farmshare-l40s": 0.02, "mit-slurm": 0.02},
                family_tests={},
            ),
            "inconclusive",
        ),
        (
            ConfirmatoryEvidence(
                cohort_complete=True,
                validity=ValidityGates(protocol_valid=False, failures=("seed replaced",)),
                primary_exact=_exact(0.001),
                primary_interval_95=_bootstrap(0.01, 0.03, 0.95),
                equivalence_interval_90=_bootstrap(0.01, 0.03, 0.90),
                provider_means={"farmshare-l40s": 0.02, "mit-slurm": 0.02},
                family_tests={},
            ),
            "invalid",
        ),
    ],
)
def test_verdict_boundaries(evidence, verdict):
    assert assign_verdict(evidence).verdict == verdict


def test_effect_requires_strictly_positive_provider_means():
    evidence = ConfirmatoryEvidence(
        cohort_complete=True,
        validity=ValidityGates(),
        primary_exact=_exact(0.01),
        primary_interval_95=_bootstrap(0.01, 0.03, 0.95),
        equivalence_interval_90=_bootstrap(0.01, 0.03, 0.90),
        provider_means={"farmshare-l40s": 0.02, "mit-slurm": 0.0},
        family_tests={},
    )
    assert assign_verdict(evidence).verdict == "inconclusive"


def test_study_lock_reads_explicit_n10_and_keeps_legacy_constructor():
    lock = load_135m_study_lock(ROOT)
    assert lock.cohort_id == COHORT_ID
    assert lock.seeds == SEEDS
    assert lock.providers["farmshare-l40s"] == (0, 1, 2, 5, 6, 7)
    legacy = StudyLock.legacy_360m()
    assert legacy.seeds == LEGACY_360M_SEEDS
    assert legacy.terminal_n_pairs == 5


def test_end_to_end_effect_null_and_inconclusive_reports():
    lock = load_135m_study_lock(ROOT)
    effect = build_confirmatory_report(
        _observations([0.03] * 10),
        study_lock=lock,
        validity=ValidityGates(),
        n_resamples=500,
        rng_seed=3,
        family_seed_differences={
            "graph": [0.03] * 10,
            "non_path": [0.02] * 10,
        },
    )
    assert effect["decision"]["verdict"] == "supports_effect"
    assert effect["decision"]["broader_family_claim"] is True

    practical_null = build_confirmatory_report(
        _observations([0.0] * 10),
        study_lock=lock,
        validity=ValidityGates(),
        n_resamples=500,
        rng_seed=4,
    )
    assert practical_null["decision"]["verdict"] == "supports_practical_null"

    inconclusive = build_confirmatory_report(
        _observations([0.03, -0.03] * 5),
        study_lock=lock,
        validity=ValidityGates(),
        n_resamples=500,
        rng_seed=5,
    )
    assert inconclusive["decision"]["verdict"] == "inconclusive"


def test_incomplete_collected_cohort_reports_inconclusive_without_testing():
    report = build_confirmatory_report(
        _observations([0.03] * 9),
        study_lock=load_135m_study_lock(ROOT),
        validity=ValidityGates(),
        n_resamples=100,
        rng_seed=6,
    )
    assert report["cohort_complete"] is False
    assert report["primary"] is None
    assert report["decision"]["verdict"] == "inconclusive"


def test_power_sensitivity_receipt_is_preregistered_not_a_90pct_claim():
    receipt = json.loads(
        (ROOT / "configs" / "power-sensitivity-135m-n10.json").read_text()
    )
    assert receipt["n_pairs"] == 10
    assert receipt["approximate_power"] == 0.8
    assert receipt["claim_90_percent_power"] is False
    assert receipt["independent_pilot_required_for_90_percent_claim"] is True
