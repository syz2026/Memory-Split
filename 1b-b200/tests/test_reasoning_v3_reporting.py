from __future__ import annotations

import inspect
import json
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

import evals.reasoning_v3.reporting as reporting_module
from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.aws_authority import VerifiedAwsAuthority
from evals.reasoning_v3.reporting import (
    CHECKPOINT_STEPS,
    CLAIM_LIMITATION,
    LEARNABILITY_THRESHOLD,
    PairingIdentity,
    ReplayedCheckpoint,
    ReportingError,
    ValiditySummary,
    VerdictInputs,
    _build_scientific_report,
    _validate_complete_matrix,
    assign_frozen_verdict,
    run_frozen_scientific_inference,
)


def _checkpoint(
    arm: str,
    seed: int,
    step: int,
    *,
    dense_score: float,
    split_score: float,
    valid: bool = True,
) -> ReplayedCheckpoint:
    score = dense_score if arm == "dense" else split_score
    family_rows = {}
    for family in ("a", "b"):
        correct_count = round(score * 2)
        family_rows[family] = tuple(
            (f"{family}/{index}", index < correct_count)
            for index in range(2)
        )
    seed_hex = format(seed, "x")
    pairing = PairingIdentity(
        initialization_sha256=seed_hex * 64,
        data_order_sha256="a" * 64,
        runtime_sha256="b" * 64,
        corpus_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    failures = () if valid else ("memory_off_leakage",)
    return ReplayedCheckpoint(
        arm=arm,
        seed=seed,
        step=step,
        family_items=family_rows,
        pairing=pairing,
        validity=ValiditySummary(passed=valid, failures=failures),
        release_identity_sha256="e" * 64,
        evaluator_code_sha256="f" * 64,
        result_key=f"results/{arm}/{seed}/{step}.json",
        result_version_id=f"version-{arm}-{seed}-{step}",
        result_sha256=(format((seed + step) % 16, "x") * 64),
    )


def _matrix(*, valid: bool = True) -> list[ReplayedCheckpoint]:
    scores = {
        CHECKPOINT_STEPS[0]: (1.0, 0.0),
        CHECKPOINT_STEPS[1]: (0.0, 0.0),
        CHECKPOINT_STEPS[2]: (0.0, 0.5),
        CHECKPOINT_STEPS[3]: (0.0, 1.0),
        CHECKPOINT_STEPS[4]: (0.0, 1.0),
    }
    return [
        _checkpoint(
            arm,
            seed,
            step,
            dense_score=scores[step][0],
            split_score=scores[step][1],
            valid=valid,
        )
        for seed in range(10)
        for arm in ("dense", "split90")
        for step in CHECKPOINT_STEPS
    ]


def _verdict(
    *,
    validity: bool = True,
    observed: float = 0.02,
    p_value: float = 0.05,
    lower_95: float = 0.001,
    upper_95: float = 0.04,
    lower_90: float = 0.001,
    upper_90: float = 0.03,
    dense: float = 0.10,
    split90: float = 0.12,
) -> VerdictInputs:
    return VerdictInputs(
        validity=ValiditySummary(
            passed=validity,
            failures=() if validity else ("no_substitution",),
        ),
        observed_mean_delta=observed,
        exact_p_value=p_value,
        interval_95=(lower_95, upper_95),
        interval_90=(lower_90, upper_90),
        dense_mean_accuracy=dense,
        split90_mean_accuracy=split90,
    )


def test_frozen_verdict_boundaries_are_mutually_exclusive():
    effect = assign_frozen_verdict(_verdict())
    assert effect.label == "exploratory_supports_effect"
    assert effect.supports_effect is True
    assert effect.supports_practical_null is False

    practical_null = assign_frozen_verdict(
        _verdict(
            observed=0.0,
            p_value=1.0,
            lower_95=-0.02,
            upper_95=0.02,
            lower_90=-0.009,
            upper_90=0.009,
            dense=LEARNABILITY_THRESHOLD,
            split90=LEARNABILITY_THRESHOLD,
        )
    )
    assert practical_null.label == "exploratory_supports_practical_null"
    assert practical_null.supports_effect is False
    assert practical_null.supports_practical_null is True

    for lower, upper in ((-0.01, 0.009), (-0.009, 0.01)):
        boundary = assign_frozen_verdict(
            _verdict(
                observed=0.0,
                p_value=1.0,
                lower_95=-0.02,
                upper_95=0.02,
                lower_90=lower,
                upper_90=upper,
            )
        )
        assert boundary.label == "inconclusive"

    invalid = assign_frozen_verdict(_verdict(validity=False))
    assert invalid.label == "invalid"
    assert not invalid.supports_effect
    assert not invalid.supports_practical_null


@pytest.mark.parametrize(
    ("observed", "p_value", "lower_95", "expected"),
    [
        (Fraction(0), Fraction(1, 1024), Fraction(1, 1000), "inconclusive"),
        (
            Fraction(1, 100),
            Fraction(51, 1000),
            Fraction(1, 1000),
            "inconclusive",
        ),
        (Fraction(1, 100), Fraction(1, 1024), Fraction(0), "inconclusive"),
    ],
)
def test_effect_boundaries_use_exact_raw_count_fractions(
    observed: Fraction,
    p_value: Fraction,
    lower_95: Fraction,
    expected: str,
):
    decision = assign_frozen_verdict(
        _verdict(
            observed=observed,
            p_value=p_value,
            lower_95=lower_95,
            upper_95=Fraction(1, 50),
            lower_90=Fraction(-1, 50),
            upper_90=Fraction(1, 50),
            dense=Fraction(1, 10),
            split90=Fraction(11, 100),
        )
    )
    assert decision.label == expected
    assert decision.supports_effect is False


def test_effect_takes_priority_when_effect_and_practical_interval_overlap():
    decision = assign_frozen_verdict(
        _verdict(
            observed=Fraction(1, 100),
            p_value=Fraction(1, 1024),
            lower_95=Fraction(1, 10_000),
            upper_95=Fraction(1, 100),
            lower_90=Fraction(1, 1000),
            upper_90=Fraction(9, 1000),
            dense=Fraction(1, 10),
            split90=Fraction(11, 100),
        )
    )
    assert decision.label == "exploratory_supports_effect"
    assert decision.supports_effect is True
    assert decision.supports_practical_null is False


def test_floor_rule_is_absolute_arm_symmetric_and_exact_at_boundary():
    for dense, split90 in (
        (LEARNABILITY_THRESHOLD, 0.0),
        (0.0, LEARNABILITY_THRESHOLD),
    ):
        decision = assign_frozen_verdict(
            _verdict(
                observed=0.0,
                p_value=1.0,
                lower_95=-0.02,
                upper_95=0.02,
                lower_90=-0.009,
                upper_90=0.009,
                dense=dense,
                split90=split90,
            )
        )
        assert decision.learnable is True
        assert decision.label == "exploratory_supports_practical_null"

    below = LEARNABILITY_THRESHOLD - 1e-12
    floor = assign_frozen_verdict(
        _verdict(
            observed=0.0,
            p_value=1.0,
            lower_95=-0.02,
            upper_95=0.02,
            lower_90=-0.009,
            upper_90=0.009,
            dense=below,
            split90=below,
        )
    )
    assert floor.learnable is False
    assert floor.label == "inconclusive_floor"
    assert floor.supports_practical_null is False


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "wrong_arm"])
def test_complete_matrix_rejects_missing_extra_duplicate_or_wrong_cells(mode):
    rows = _matrix()
    if mode == "missing":
        rows.pop()
    elif mode == "extra":
        rows.append(replace(rows[-1], step=99))
    elif mode == "duplicate":
        rows.append(rows[-1])
    else:
        rows[-1] = replace(rows[-1], arm="other")
    with pytest.raises(ReportingError):
        _validate_complete_matrix(rows)


def test_report_replays_full_matrix_preserves_trajectory_and_limits_claim():
    report = _build_scientific_report(
        _matrix(),
        family_order=("a", "b"),
        items_per_family=2,
    )
    assert report["matrix"]["run_count"] == 20
    assert report["matrix"]["checkpoint_count"] == 100
    assert report["primary"]["exact"]["permutations"] == 1024
    assert report["primary"]["bootstrap"]["n_draws"] == 20_000
    assert report["primary"]["bootstrap"]["rng_seed"] == 0
    assert report["primary"]["exact"]["statistic_fraction"] == {
        "denominator": 1,
        "numerator": 1,
    }
    assert report["primary"]["seed_delta_fractions"]["0"] == {
        "denominator": 1,
        "numerator": 1,
    }
    assert report["decision"]["label"] == "exploratory_supports_effect"
    assert report["decision"]["supports_effect"] is True
    assert report["decision"]["supports_practical_null"] is False
    assert report["claim_limitation"] == CLAIM_LIMITATION

    trajectory = report["secondary_trajectory"]
    assert trajectory["steps"] == list(CHECKPOINT_STEPS)
    assert trajectory["seed_macro_deltas"]["0"][0] == -1.0
    widths = [
        CHECKPOINT_STEPS[index] - CHECKPOINT_STEPS[index - 1]
        for index in range(1, len(CHECKPOINT_STEPS))
    ]
    expected_aulc = (
        0.0 * widths[0] + 0.5 * widths[1] + 1.0 * widths[2] + 1.0 * widths[3]
    ) / sum(widths)
    assert trajectory["seed_macro_aulc_deltas"]["0"] == pytest.approx(
        expected_aulc
    )
    assert len(report["source_results"]) == 100


def test_failed_validity_gate_forces_invalid_without_erasing_outcomes():
    report = _build_scientific_report(
        _matrix(valid=False),
        family_order=("a", "b"),
        items_per_family=2,
    )
    assert report["decision"]["label"] == "invalid"
    assert report["primary"]["seed_deltas"]
    assert report["validity"]["passed"] is False
    assert "memory_off_leakage" in report["validity"]["failures"]


def test_claim_text_forbids_fact_specific_mechanism_and_public_api_is_closed():
    lowered = CLAIM_LIMITATION.lower()
    assert "no equal-mass protected control arm" in lowered
    assert "less active target-gradient mass" in lowered
    assert "complete dense-versus-split90 intervention" in lowered
    assert "does not identify a fact-specific mechanism" in lowered
    assert not inspect.signature(run_frozen_scientific_inference).parameters


def test_aggregate_report_uses_fixed_signed_kms_publication_method():
    verified = VerifiedAwsAuthority(
        contract_sha256="1" * 64,
        record_sha256="2" * 64,
        record_version_id="record-version",
        signature_version_id="signature-version",
        signer_key_arn=(
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "11111111-1111-4111-8111-111111111111"
        ),
        sealed_gold_kms_key_arn=(
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "22222222-2222-4222-8222-222222222222"
        ),
        checkpoint_kms_key_arn=(
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "33333333-3333-4333-8333-333333333333"
        ),
    )
    ref = SimpleNamespace(version_id="report-version")

    class Authority:
        def __init__(self):
            self.calls = []

        def put_scientific_report(self, payload, authority):
            self.calls.append((payload, authority))
            return SimpleNamespace(object_ref=ref, payload_bytes=payload)

    authority = Authority()
    report = {"decision": {"label": "inconclusive"}, "schema_version": 1}
    published = reporting_module._publish_scientific_report(
        report,
        verified,
        authority,
    )
    assert published.object_ref is ref
    assert published.report == report
    assert authority.calls == [(canonical_json_bytes(report), verified)]
    assert json.loads(authority.calls[0][0]) == report
