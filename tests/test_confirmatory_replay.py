from __future__ import annotations

import hashlib
import json

import pytest

from evals.confirmatory.contracts import CONTRACT_VERSION, canonical_json_bytes
from evals.confirmatory.fixtures import (
    invalid_fixture,
    null_fixture,
    positive_fixture,
)
from evals.confirmatory.reporting import build_artifact_report
from evals.confirmatory import reporting as reporting_module


def _bundle(
    *,
    observed_seed_count: int = 5,
    split_correct: bool = True,
    dense_correct: bool = False,
    failed_readiness: bool = False,
) -> dict[str, bytes]:
    if failed_readiness:
        fixture = invalid_fixture()
    elif split_correct == dense_correct:
        fixture = null_fixture()
    else:
        fixture = positive_fixture()
    artifacts = dict(fixture.artifacts)
    if observed_seed_count < 5:
        outcome_rows = [
            line
            for line in artifacts["outcomes.jsonl"].splitlines(keepends=True)
            if json.loads(line)["seed"] < observed_seed_count
        ]
        artifacts["outcomes.jsonl"] = b"".join(outcome_rows)
        checkpoints = {
            raw["checkpoint_sha256"]: raw["seed"]
            for raw in (
                json.loads(line)
                for line in artifacts["checkpoints.jsonl"].splitlines()
            )
        }
        metrics = json.loads(artifacts["metrics.json"])
        metrics["summaries"] = [
            summary
            for summary in metrics["summaries"]
            if checkpoints[summary["checkpoint_sha256"]]
            < observed_seed_count
        ]
        artifacts["metrics.json"] = canonical_json_bytes(metrics)
        inference = json.loads(artifacts["inference.json"])
        inference["paired_seed_bundle_deltas"] = inference[
            "paired_seed_bundle_deltas"
        ][:observed_seed_count]
        inference["exact_test_result"] = None
        artifacts["inference.json"] = canonical_json_bytes(inference)
    return artifacts


def _build(artifacts):
    return build_artifact_report(
        artifacts=artifacts,
        expected_study_lock_sha256=hashlib.sha256(
            artifacts["study-lock.json"]
        ).hexdigest(),
    )


def _mutate_json(artifacts, name, mutate):
    changed = dict(artifacts)
    raw = json.loads(changed[name])
    mutate(raw)
    changed[name] = canonical_json_bytes(raw)
    return changed


def test_complete_effect_conclusion_is_replayed_from_exact_bound_evidence():
    report = _build(_bundle())

    assert report.expected_cells == 1600
    assert report.observed_cells == 1600
    assert report.scientific_status == "complete"
    assert report.interim_evidence_label == "none"
    assert report.final_inference_conclusion == "supports_effect"
    assert report.paired_seed_bundle_deltas == (1.0,) * 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contrast_id", "wrong-contrast"),
        ("alternative", "less"),
        ("alpha", 0.1),
        ("sign_assignments", 31),
        ("equality_counted", False),
    ],
)
def test_primary_test_identity_is_frozen_and_strict(field, value):
    artifacts = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw["primary_test"].__setitem__(field, value),
    )

    with pytest.raises(ValueError, match="primary test|frozen"):
        _build(artifacts)


def test_asserted_conclusions_and_dishonest_exact_results_are_rejected():
    asserted = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw.__setitem__("supports_effect", True),
    )
    with pytest.raises(ValueError, match="fields"):
        _build(asserted)

    dishonest = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw["exact_test_result"].__setitem__("extreme_count", 2),
    )
    with pytest.raises(ValueError, match="exact test"):
        _build(dishonest)


def test_seed_deltas_and_metrics_are_recomputed_from_bound_outcomes():
    dishonest_delta = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw["paired_seed_bundle_deltas"].__setitem__(0, 0.5),
    )
    with pytest.raises(ValueError, match="seed.*delta|replay"):
        _build(dishonest_delta)

    dishonest_metric = _mutate_json(
        _bundle(),
        "metrics.json",
        lambda raw: raw["summaries"][0].__setitem__(
            "primary_accuracy",
            1.0,
        ),
    )
    with pytest.raises(ValueError, match="metrics"):
        _build(dishonest_metric)


def test_completeness_and_counts_are_derived_from_the_frozen_registry():
    one_seed = _build(_bundle(observed_seed_count=1))
    three_seeds = _build(_bundle(observed_seed_count=3))

    assert (one_seed.expected_cells, one_seed.observed_cells) == (1600, 320)
    assert one_seed.paired_seed_bundle_deltas == (1.0,)
    assert one_seed.scientific_status == "incomplete"
    assert one_seed.interim_evidence_label == "directional_only"
    assert (three_seeds.expected_cells, three_seeds.observed_cells) == (
        1600,
        960,
    )
    assert three_seeds.paired_seed_bundle_deltas == (1.0,) * 3
    assert three_seeds.scientific_status == "incomplete"
    assert three_seeds.interim_evidence_label == "sign_consistent_only"

    with pytest.raises(TypeError):
        build_artifact_report(
            artifacts=_bundle(observed_seed_count=1),
            expected_study_lock_sha256="0" * 64,
            expected_cells=10_000,
            observed_cells=10_000,
        )


def test_outcome_bindings_and_canonical_metrics_are_mandatory():
    artifacts = _bundle()
    raw_lines = artifacts["outcomes.jsonl"].splitlines()
    first = json.loads(raw_lines[0])
    checkpoints = [
        json.loads(line)
        for line in artifacts["checkpoints.jsonl"].splitlines()
    ]
    first["checkpoint_sha256"] = checkpoints[1]["checkpoint_sha256"]
    raw_lines[0] = canonical_json_bytes(first).rstrip(b"\n")
    crossed = dict(artifacts)
    crossed["outcomes.jsonl"] = b"\n".join(raw_lines) + b"\n"

    with pytest.raises(ValueError, match="checkpoint binding"):
        _build(crossed)

    one_row = dict(artifacts)
    one_row["outcomes.jsonl"] = (
        artifacts["outcomes.jsonl"].splitlines(keepends=True)[0]
    )
    one_row["metrics.json"] = canonical_json_bytes(
        {
            "record_type": "memorysplit.confirmatory.metrics.v2",
            "schema_version": CONTRACT_VERSION,
            "summaries": [json.loads(artifacts["metrics.json"])["summaries"][0]],
        }
    )
    with pytest.raises(ValueError, match="metrics|summary|primary"):
        _build(one_row)


@pytest.mark.parametrize("observed_seed_count", [1, 3])
def test_invalid_replay_forces_none_and_not_evaluated(observed_seed_count):
    report = _build(
        _bundle(
            observed_seed_count=observed_seed_count,
            failed_readiness=True,
        )
    )

    assert report.scientific_status == "invalid"
    assert report.interim_evidence_label == "none"
    assert report.final_inference_conclusion == "not_evaluated"


def test_practical_null_fails_closed_without_a_frozen_bootstrap_rng():
    artifacts = _bundle(split_correct=True, dense_correct=True)
    report = _build(artifacts)

    assert report.scientific_status == "complete"
    assert report.final_inference_conclusion == "inconclusive"
    assert (
        report.practical_null_replay_status
        == "unavailable_unfrozen_rng_seed"
    )
    assert "20,000" in reporting_module.PRACTICAL_NULL_REPLAY_GAP
    assert "RNG seed" in reporting_module.PRACTICAL_NULL_REPLAY_GAP

    asserted = _mutate_json(
        artifacts,
        "inference.json",
        lambda raw: raw.__setitem__("supports_practical_null", True),
    )
    with pytest.raises(ValueError, match="fields"):
        _build(asserted)


def test_deterministic_fixtures_are_genuinely_replayable_artifact_bundles():
    fixtures = (positive_fixture(), null_fixture(), invalid_fixture())
    assert all(not hasattr(fixture, "supports_effect") for fixture in fixtures)
    assert all(
        not hasattr(fixture, "supports_practical_null")
        for fixture in fixtures
    )
    positive = _build(fixtures[0].artifacts)
    practical_null = _build(fixtures[1].artifacts)
    invalid = _build(fixtures[2].artifacts)

    assert positive.final_inference_conclusion == "supports_effect"
    assert practical_null.final_inference_conclusion == "inconclusive"
    assert (
        practical_null.practical_null_replay_status
        == "unavailable_unfrozen_rng_seed"
    )
    assert invalid.scientific_status == "invalid"
    assert invalid.interim_evidence_label == "none"
    assert invalid.final_inference_conclusion == "not_evaluated"
