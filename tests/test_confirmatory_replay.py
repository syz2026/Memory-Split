from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    Arm,
    CheckpointRecord,
    ItemRecord,
    canonical_json_bytes,
)
from evals.confirmatory.inference import exact_sign_flip_test
from evals.confirmatory.fixtures import (
    invalid_fixture,
    null_fixture,
    positive_fixture,
)
from evals.confirmatory.metrics import (
    ItemOutcome,
    balanced_counterfactual_pair_metric,
)
from evals.confirmatory.reporting import (
    INFERENCE_EVIDENCE_SCHEMA,
    REQUIRED_ARTIFACTS,
    build_artifact_report,
)
from evals.confirmatory import reporting as reporting_module


PRIMARY_CONTRAST_ID = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)


def _item_records() -> tuple[ItemRecord, ...]:
    records = []
    shapes = {
        "composition_ood": (2, "heldout"),
        "joint_ood": (7, "heldout"),
    }
    for family in ("graph", "non_path"):
        for stratum, (path_length, composition_split) in shapes.items():
            pair_id = f"{family}-{stratum}"
            for twin in ("original", "counterfactual"):
                records.append(
                    ItemRecord.from_dict(
                        {
                            "record_type": ITEM_SCHEMA,
                            "schema_version": CONTRACT_VERSION,
                            "item_id": f"{pair_id}-{twin}",
                            "pair_id": pair_id,
                            "twin": twin,
                            "stratum": stratum,
                            "family": family,
                            "world_id": f"world-{pair_id}",
                            "task": "fixture_task",
                            "path_length": path_length,
                            "composition_split": composition_split,
                            "composition_id": f"composition-{pair_id}",
                            "prompt": "fixture prompt",
                            "initial_slots": ["Q1", None, None, None],
                            "store_id": f"store-{pair_id}",
                            "memory_mode": "memory_on",
                            "control": "correct",
                        }
                    )
                )
    return tuple(sorted(records, key=lambda record: record.item_id))


def _checkpoint_records() -> tuple[CheckpointRecord, ...]:
    records = []
    for seed in range(5):
        for arm in (Arm.DENSE, Arm.SPLIT):
            digest = hashlib.sha256(f"{seed}:{arm.value}".encode()).hexdigest()
            records.append(
                CheckpointRecord.from_dict(
                    {
                        "record_type": CHECKPOINT_SCHEMA,
                        "schema_version": CONTRACT_VERSION,
                        "checkpoint_sha256": digest,
                        "model_id": "fixture-model",
                        "arm": arm.value,
                        "seed": seed,
                        "raw_token_count": 1,
                        "configuration_sha256": "b" * 64,
                        "corpus_sha256": "c" * 64,
                        "code_sha256": "d" * 64,
                    }
                )
            )
    return tuple(sorted(records, key=lambda record: (record.seed, record.arm.value)))


def _outcome_dict(outcome: ItemOutcome) -> dict:
    return {
        "record_type": "memorysplit.confirmatory.outcome.v2",
        "schema_version": CONTRACT_VERSION,
        "item_id": outcome.item_id,
        "pair_id": outcome.pair_id,
        "twin": outcome.twin.value,
        "stratum": outcome.stratum.value,
        "family": outcome.family.value,
        "seed": outcome.seed,
        "world_id": outcome.world_id,
        "checkpoint_sha256": outcome.checkpoint_sha256,
        "arm": outcome.arm.value,
        "memory_mode": outcome.memory_mode.value,
        "control": outcome.control.value,
        "proof_valid": outcome.proof_valid,
        "answer_valid": outcome.answer_valid,
        "complete": outcome.complete,
        "valid": outcome.valid,
    }


def _jsonl(records) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _bundle(
    *,
    observed_seed_count: int = 5,
    split_correct: bool = True,
    dense_correct: bool = False,
    measured_validity_failure: bool = False,
) -> dict[str, bytes]:
    items = _item_records()
    checkpoints = _checkpoint_records()
    item_map = {item.item_id: item for item in items}
    outcomes = []
    summaries = []
    deltas = []
    for checkpoint in checkpoints:
        if checkpoint.seed >= observed_seed_count:
            continue
        correct = (
            split_correct
            if checkpoint.arm is Arm.SPLIT
            else dense_correct
        )
        checkpoint_outcomes = [
            ItemOutcome(
                item_id=item.item_id,
                pair_id=item.pair_id,
                twin=item.twin,
                stratum=item.stratum,
                family=item.family,
                seed=checkpoint.seed,
                world_id=item.world_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                arm=checkpoint.arm,
                memory_mode=item.memory_mode,
                control=item.control,
                proof_valid=correct,
                answer_valid=correct,
                complete=True,
                valid=True,
            )
            for item in items
        ]
        outcomes.extend(checkpoint_outcomes)
        summaries.append(
            (
                checkpoint.seed,
                checkpoint.arm,
                balanced_counterfactual_pair_metric(
                    checkpoint_outcomes,
                    items=item_map,
                    checkpoints={
                        checkpoint.checkpoint_sha256: checkpoint,
                    },
                ),
            )
        )
    summaries.sort(key=lambda entry: (entry[0], entry[1].value))
    for seed in range(observed_seed_count):
        by_arm = {
            arm: summary.primary_accuracy
            for summary_seed, arm, summary in summaries
            if summary_seed == seed
        }
        deltas.append(by_arm[Arm.SPLIT] - by_arm[Arm.DENSE])

    test_result = None
    if len(deltas) == 5:
        replayed = exact_sign_flip_test(deltas, alternative="greater")
        test_result = {
            "statistic": replayed.statistic,
            "extreme_count": replayed.extreme_count,
            "p_value": replayed.p_value,
            "reject_null": (
                replayed.statistic > 0.0 and replayed.p_value <= 0.05
            ),
        }
    inference = {
        "record_type": INFERENCE_EVIDENCE_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "measured_validity_failure": measured_validity_failure,
        "primary_test": {
            "contrast_id": PRIMARY_CONTRAST_ID,
            "method": "exact_one_sided_exhaustive_sign_flip",
            "alternative": "greater",
            "alpha": 0.05,
            "n_pairs": 5,
            "sign_assignments": 32,
            "equality_counted": True,
        },
        "paired_seed_bundle_deltas": deltas,
        "exact_test_result": test_result,
    }
    metrics = {
        "record_type": "memorysplit.confirmatory.metrics.v2",
        "schema_version": CONTRACT_VERSION,
        "summaries": [summary.to_dict() for _, _, summary in summaries],
    }
    outcome_records = sorted(
        outcomes,
        key=lambda row: (row.seed, row.arm.value, row.item_id),
    )
    artifacts = {
        "checkpoints.jsonl": _jsonl(
            checkpoint.to_dict() for checkpoint in checkpoints
        ),
        "inference.json": canonical_json_bytes(inference),
        "items.jsonl": _jsonl(item.to_dict() for item in items),
        "metrics.json": canonical_json_bytes(metrics),
        "outcomes.jsonl": _jsonl(
            _outcome_dict(outcome) for outcome in outcome_records
        ),
        "sealed-gold.jsonl": canonical_json_bytes(
            {"fixture": "sealed-gold"}
        ),
        "stores.jsonl": canonical_json_bytes({"fixture": "stores"}),
    }
    assert set(artifacts) == set(REQUIRED_ARTIFACTS)
    return artifacts


def _mutate_json(artifacts, name, mutate):
    changed = dict(artifacts)
    raw = json.loads(changed[name])
    mutate(raw)
    changed[name] = canonical_json_bytes(raw)
    return changed


def test_complete_effect_conclusion_is_replayed_from_exact_bound_evidence():
    report = build_artifact_report(artifacts=_bundle())

    assert report.expected_cells == 80
    assert report.observed_cells == 80
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
        build_artifact_report(artifacts=artifacts)


def test_asserted_conclusions_and_dishonest_exact_results_are_rejected():
    asserted = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw.__setitem__("supports_effect", True),
    )
    with pytest.raises(ValueError, match="fields"):
        build_artifact_report(artifacts=asserted)

    dishonest = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw["exact_test_result"].__setitem__("extreme_count", 2),
    )
    with pytest.raises(ValueError, match="exact test"):
        build_artifact_report(artifacts=dishonest)


def test_seed_deltas_and_metrics_are_recomputed_from_bound_outcomes():
    dishonest_delta = _mutate_json(
        _bundle(),
        "inference.json",
        lambda raw: raw["paired_seed_bundle_deltas"].__setitem__(0, 0.5),
    )
    with pytest.raises(ValueError, match="seed.*delta|replay"):
        build_artifact_report(artifacts=dishonest_delta)

    dishonest_metric = _mutate_json(
        _bundle(),
        "metrics.json",
        lambda raw: raw["summaries"][0].__setitem__(
            "primary_accuracy",
            1.0,
        ),
    )
    with pytest.raises(ValueError, match="metrics"):
        build_artifact_report(artifacts=dishonest_metric)


def test_completeness_and_counts_are_derived_from_the_frozen_registry():
    one_seed = build_artifact_report(
        artifacts=_bundle(observed_seed_count=1),
    )
    three_seeds = build_artifact_report(
        artifacts=_bundle(observed_seed_count=3),
    )

    assert (one_seed.expected_cells, one_seed.observed_cells) == (80, 16)
    assert one_seed.paired_seed_bundle_deltas == (1.0,)
    assert one_seed.scientific_status == "incomplete"
    assert one_seed.interim_evidence_label == "directional_only"
    assert (three_seeds.expected_cells, three_seeds.observed_cells) == (80, 48)
    assert three_seeds.paired_seed_bundle_deltas == (1.0,) * 3
    assert three_seeds.scientific_status == "incomplete"
    assert three_seeds.interim_evidence_label == "sign_consistent_only"

    with pytest.raises(TypeError):
        build_artifact_report(
            artifacts=_bundle(observed_seed_count=1),
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
        build_artifact_report(artifacts=crossed)

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
        build_artifact_report(artifacts=one_row)


@pytest.mark.parametrize("observed_seed_count", [1, 3])
def test_invalid_replay_forces_none_and_not_evaluated(observed_seed_count):
    report = build_artifact_report(
        artifacts=_bundle(
            observed_seed_count=observed_seed_count,
            measured_validity_failure=True,
        )
    )

    assert report.scientific_status == "invalid"
    assert report.interim_evidence_label == "none"
    assert report.final_inference_conclusion == "not_evaluated"


def test_practical_null_fails_closed_without_a_frozen_bootstrap_rng():
    artifacts = _bundle(split_correct=True, dense_correct=True)
    report = build_artifact_report(artifacts=artifacts)

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
        build_artifact_report(artifacts=asserted)


def test_deterministic_fixtures_are_genuinely_replayable_artifact_bundles():
    fixtures = (positive_fixture(), null_fixture(), invalid_fixture())
    assert all(not hasattr(fixture, "supports_effect") for fixture in fixtures)
    assert all(
        not hasattr(fixture, "supports_practical_null")
        for fixture in fixtures
    )
    positive = build_artifact_report(artifacts=fixtures[0].artifacts)
    practical_null = build_artifact_report(artifacts=fixtures[1].artifacts)
    invalid = build_artifact_report(artifacts=fixtures[2].artifacts)

    assert positive.final_inference_conclusion == "supports_effect"
    assert practical_null.final_inference_conclusion == "inconclusive"
    assert (
        practical_null.practical_null_replay_status
        == "unavailable_unfrozen_rng_seed"
    )
    assert invalid.scientific_status == "invalid"
    assert invalid.interim_evidence_label == "none"
    assert invalid.final_inference_conclusion == "not_evaluated"
