from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from evals.relational_contracts import (
    CHECKPOINT_SUMMARY_FIELDS,
    EVAL_ROW_FIELDS,
    GUARDRAIL_REPORT_FIELDS,
    CheckpointSummary,
    EvalRow,
    GuardrailReport,
    StreamingEvaluationPublisher,
    canonical_json_bytes,
    cluster_id_for,
    load_result_schema,
    publish_evaluation,
    rows_sha256,
    validate_eval_rows,
    validate_published_evaluation,
    validate_result_payload,
)
from evals.relational_metrics import (
    PREREGISTERED_REASONING_MILESTONES,
    wilson_interval,
)


HASHES = {
    "checkpoint_sha256": "1" * 64,
    "evaluator_sha256": "2" * 64,
    "data_sha256": "3" * 64,
    "relation_schema_sha256": "4" * 64,
    "provenance_sha256": "5" * 64,
    "configuration_sha256": "c" * 64,
    "result_schema_sha256": "d" * 64,
}


def _actions() -> list[list]:
    return [
        [0, "r0", "out", True, False],
        [0, "r0", "out", False, True],
        *[[0, "r0", "out", False, False] for _ in range(4)],
    ]


def _row(
    variant: str = "original",
    *,
    control_id: str = "correct",
    qid: str | None = None,
) -> dict:
    qid = qid or ("pair-o" if variant == "original" else "pair-c")
    relation_path_hash = "a" * 64
    value = {
        "record_type": "eval_row",
        "schema_version": 1,
        "qid": qid,
        "pair_id": "pair",
        "variant": variant,
        "task": "path_composition",
        "world_id": 17,
        "provenance_id": "world-17",
        "relation_path_hash": relation_path_hash,
        "template_id": "path_composition:v1",
        "composition_split": "seen",
        "hop": 1,
        "seed": 1001,
        "model_id": "d160m",
        "arm": "split",
        **HASHES,
        "raw_token_count": 811_104_000,
        "memory_mode": "memory_on",
        "control_id": control_id,
        "cluster_id": cluster_id_for(
            seed=1001,
            world_id=17,
            relation_path_hash=relation_path_hash,
            template_id="path_composition:v1",
        ),
        "prediction": "r1",
        "answer": "r1",
        "correct": True,
        "prediction_source": "model",
        "all_actions": _actions(),
        "gold_all_actions": _actions(),
        "returned_addresses": [[17, "r0", "out"], None, None, None, None, None],
        "gold_addresses": [[17, "r0", "out"]],
        "correct_referents": [True],
        "misses": 0,
        "malformed": 0,
        "abstained": False,
        "excess_reads": 0,
        "halt_step": 2,
        "answer_logits": [[-0.1] for _ in range(6)],
        "lookup_latency_ns": 31,
        "lookup_count": 1,
        "store_rows": 10,
        "store_bytes": 1000,
        "control_seed": 77,
        "transformation_id": None,
        "source_store_sha256": None,
        "transformed_store_sha256": None,
        "transformation_metadata_sha256": None,
        "changed_addresses": [],
        "oracle_before": "r1",
        "oracle_after": "r1",
        "oracle_effect": "unchanged",
        "edit_locality_correct": None,
    }
    return value


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _slice_metrics(denominator: int = 0) -> dict:
    return {
        "item_accuracy": _rate(0, denominator),
        "pair_accuracy": _rate(0, denominator),
        "exact_action_path": _rate(0, denominator),
    }


def _metrics() -> dict:
    return {
        "item_accuracy": _rate(2, 2),
        "pair_accuracy": _rate(1, 1),
        "all_six_action_exact": _rate(2, 2),
        "exact_action_path": _rate(2, 2),
        "answer_given_correct_retrieval": _rate(2, 2),
        "gold_path_answer_accuracy": _rate(0, 0),
        "malformed_rate": _rate(0, 12),
        "miss_rate": _rate(0, 2),
        "abstention_rate": _rate(0, 2),
        "excess_read_rate": _rate(0, 12),
        "per_hop": {
            str(hop): {
                name: _rate(2, 2) if hop == 1 else _rate(0, 0)
                for name in ("relation", "direction", "action", "referent")
            }
            for hop in range(1, 7)
        },
        "by_hop": {
            str(hop): (
                {
                    "item_accuracy": _rate(2, 2),
                    "pair_accuracy": _rate(1, 1),
                    "exact_action_path": _rate(2, 2),
                }
                if hop == 1
                else _slice_metrics()
            )
            for hop in range(1, 7)
        },
        "by_composition": {
            "seen": {
                "item_accuracy": _rate(2, 2),
                "pair_accuracy": _rate(1, 1),
                "exact_action_path": _rate(2, 2),
            },
            "heldout": _slice_metrics(),
        },
        "by_task": {
            "path_composition": {
                "item_accuracy": _rate(2, 2),
                "pair_accuracy": _rate(1, 1),
                "exact_action_path": _rate(2, 2),
            },
            "date_ordering": _slice_metrics(),
            "balanced_equality": _slice_metrics(),
        },
        "store": {
            "rows": 10,
            "bytes": 1000,
            "lookup_latency_ns": 31.0,
            "lookup_count": 2,
        },
        "edit_locality": _rate(0, 0),
        "milestone_crossings": {
            name: (
                811_104_000
                if name == "path_composition_pair_accuracy_0.75"
                else None
            )
            for name in PREREGISTERED_REASONING_MILESTONES
        },
    }


def _summary(rows: list[EvalRow]) -> dict:
    return {
        "record_type": "checkpoint_summary",
        "schema_version": 1,
        "checkpoint_sha256": HASHES["checkpoint_sha256"],
        "model_id": "d160m",
        "arm": "split",
        "seed": 1001,
        "raw_token_count": 811_104_000,
        "memory_mode": "memory_on",
        "control_id": "correct",
        "evaluator_sha256": HASHES["evaluator_sha256"],
        "data_sha256": HASHES["data_sha256"],
        "relation_schema_sha256": HASHES["relation_schema_sha256"],
        "configuration_sha256": HASHES["configuration_sha256"],
        "result_schema_sha256": HASHES["result_schema_sha256"],
        "provenance_sha256": HASHES["provenance_sha256"],
        "rows_sha256": rows_sha256(rows),
        "n_rows": 2,
        "n_pairs": 1,
        "metrics": _metrics(),
    }


def _check(
    check_id: str,
    *,
    value: float | bool = 1.0,
    reference_value: float | None = None,
    threshold: float | bool = 1.0,
    comparison: str = ">=",
    numerator: int | None = 1,
    denominator: int | None = 1,
) -> dict:
    return {
        "check_id": check_id,
        "value": value,
        "reference_value": reference_value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": True,
        "numerator": numerator,
        "denominator": denominator,
    }


def _guardrail_report() -> dict:
    guards = {
        "factual_job": {
            "passed": True,
            "checks": [
                _check("split_on_recall_floor", threshold=0.95),
                _check(
                    "split_on_recall_noninferiority",
                    reference_value=0.0,
                    threshold=-0.02,
                ),
            ],
        },
        "split_off_leakage": {
            "passed": True,
            "checks": [
                _check(
                    "split_off_recall",
                    value=0.0,
                    threshold=0.05,
                    comparison="<",
                    numerator=0,
                ),
                _check(
                    "split_off_recognition_wilson_hi",
                    value=wilson_interval(0, 20)[1],
                    threshold=0.30,
                    comparison="<",
                    numerator=0,
                    denominator=20,
                ),
            ],
        },
        "retrieval_procedure": {
            "passed": True,
            "checks": [
                _check("split_off_first_hop", threshold=0.75),
                _check(
                    "gold_return_path_noninferiority",
                    reference_value=0.0,
                    threshold=-0.05,
                ),
            ],
        },
        "relation_rule": {
            "passed": True,
            "checks": [
                _check(
                    "rule_noninferiority",
                    reference_value=0.0,
                    threshold=-0.02,
                ),
            ],
        },
        "natural_text": {
            "passed": True,
            "checks": [
                _check(
                    "bpb_noninferiority",
                    value=1.0,
                    reference_value=1.0,
                    threshold=1.01,
                    comparison="<=",
                    numerator=None,
                    denominator=None,
                ),
            ],
        },
        "instrument_integrity": {
            "passed": True,
            "checks": [
                _check(
                    "integrity_conjunction",
                    value=True,
                    threshold=True,
                    comparison="==",
                    numerator=None,
                    denominator=None,
                ),
            ],
        },
    }
    return {
        "record_type": "guardrail_report",
        "schema_version": 1,
        "split_checkpoint_sha256": "1" * 64,
        "dense_checkpoint_sha256": "6" * 64,
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 811_104_000,
        "evaluator_sha256": HASHES["evaluator_sha256"],
        "data_sha256": HASHES["data_sha256"],
        "relation_schema_sha256": HASHES["relation_schema_sha256"],
        "split_configuration_sha256": HASHES["configuration_sha256"],
        "dense_configuration_sha256": "e" * 64,
        "result_schema_sha256": HASHES["result_schema_sha256"],
        "split_result_provenance_sha256": HASHES["provenance_sha256"],
        "dense_result_provenance_sha256": "f" * 64,
        "study_provenance_sha256": "0" * 64,
        "pairing_receipt_sha256": "7" * 64,
        "split_guardrail_source_sha256": "8" * 64,
        "dense_guardrail_source_sha256": "9" * 64,
        "split_matrix_manifest_sha256": "a" * 64,
        "dense_matrix_manifest_sha256": "b" * 64,
        "guards": guards,
        "confirmatory_passed": True,
    }


def test_eval_row_round_trips_canonically_and_has_exact_join_key():
    value = _row()
    row = EvalRow.from_dict(value)

    assert row.to_dict() == value
    assert json.loads(canonical_json_bytes(row)) == value
    assert row.paired_join_key() == (
        1001,
        "pair-o",
        "pair",
        "original",
        "path_composition",
        17,
        "a" * 64,
        "path_composition:v1",
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.pop("relation_path_hash"), "relation_path_hash"),
        (lambda value: value.update(extra=True), "unknown"),
        (lambda value: value.update(seed=True), "seed"),
        (lambda value: value.update(hop=0), "hop"),
        (lambda value: value.update(control_id="../escape"), "control_id"),
        (lambda value: value.update(correct=False), "correct"),
        (lambda value: value.update(misses=1), "misses"),
        (lambda value: value.update(lookup_count=0), "lookup latency"),
        (
            lambda value: value.update(
                memory_mode="memory_off",
                lookup_count=1,
            ),
            "memory-off lookup_count",
        ),
        (lambda value: value.update(lookup_count=2), "read attempts"),
        (
            lambda value: value["answer_logits"][0].__setitem__(
                0, float("nan")
            ),
            "finite",
        ),
        (
            lambda value: value.update(
                changed_addresses=[[17, "r0", "out"], [17, "r0", "out"]]
            ),
            "duplicate",
        ),
        (
            lambda value: value.update(cluster_id="0" * 64),
            "cluster_id",
        ),
    ],
)
def test_eval_row_rejects_unknown_missing_mistyped_and_nonfinite_fields(
    mutation,
    message,
):
    value = _row()
    mutation(value)
    with pytest.raises(ValueError, match=message):
        EvalRow.from_dict(value)


def test_row_collection_rejects_duplicates_incomplete_pairs_and_crossed_metadata():
    original = EvalRow.from_dict(_row("original"))
    counterfactual = EvalRow.from_dict(_row("counterfactual"))
    assert validate_eval_rows([original, counterfactual]) == (
        original,
        counterfactual,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_eval_rows([original, original])
    with pytest.raises(ValueError, match="both variants"):
        validate_eval_rows([original])

    crossed = _row("counterfactual")
    crossed["template_id"] = "crossed:v1"
    crossed["cluster_id"] = cluster_id_for(
        seed=1001,
        world_id=17,
        relation_path_hash="a" * 64,
        template_id="crossed:v1",
    )
    with pytest.raises(ValueError, match="crossed pair metadata"):
        validate_eval_rows([original, EvalRow.from_dict(crossed)])


def test_cyclic_paths_may_repeat_gold_addresses_but_not_changed_addresses():
    value = _row()
    value["hop"] = 2
    value["gold_addresses"] = [
        [17, "r0", "out"],
        [17, "r0", "out"],
    ]
    value["correct_referents"] = [True, True]
    value["gold_all_actions"] = [
        [0, "r0", "out", True, False],
        [0, "r0", "out", True, False],
        [0, "r0", "out", False, True],
        *[[0, "r0", "out", False, False] for _ in range(3)],
    ]
    value["all_actions"] = [
        list(action) for action in value["gold_all_actions"]
    ]
    value["returned_addresses"] = [
        [17, "r0", "out"],
        [17, "r0", "out"],
        None,
        None,
        None,
        None,
    ]
    value["halt_step"] = 3
    value["lookup_count"] = 2

    assert len(EvalRow.from_dict(value).gold_addresses) == 2


def test_checkpoint_summary_rejects_invalid_rates_and_row_binding():
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))
    assert summary.to_dict()["metrics"]["by_hop"]["6"][
        "item_accuracy"
    ] == _rate(0, 0)

    bad_rate = _summary(rows)
    bad_rate["metrics"]["pair_accuracy"] = {
        "value": 0.0,
        "numerator": 2,
        "denominator": 1,
    }
    with pytest.raises(ValueError, match="numerator"):
        CheckpointSummary.from_dict(bad_rate)

    bad_null = _summary(rows)
    bad_null["metrics"]["by_hop"]["6"]["item_accuracy"]["value"] = 0.0
    with pytest.raises(ValueError, match="null"):
        CheckpointSummary.from_dict(bad_null)

    bad_counts = _summary(rows)
    bad_counts["n_rows"] = 3
    bad_counts["metrics"]["item_accuracy"] = _rate(2, 3)
    with pytest.raises(ValueError, match="exactly two rows per pair"):
        CheckpointSummary.from_dict(bad_counts)

    bad_denominator = _summary(rows)
    bad_denominator["metrics"]["malformed_rate"] = _rate(0, 11)
    with pytest.raises(ValueError, match="malformed_rate denominator"):
        CheckpointSummary.from_dict(bad_denominator)

    future_crossing = _summary(rows)
    future_crossing["metrics"]["milestone_crossings"][
        "path_composition_pair_accuracy_0.75"
    ] = future_crossing["raw_token_count"] + 1
    with pytest.raises(ValueError, match="future raw-token"):
        CheckpointSummary.from_dict(future_crossing)

    unknown_crossing = _summary(rows)
    unknown_crossing["metrics"]["milestone_crossings"]["not_preregistered"] = None
    with pytest.raises(ValueError, match="milestone.*set"):
        CheckpointSummary.from_dict(unknown_crossing)

    crossed = _summary(rows)
    crossed["checkpoint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        crossed_summary = CheckpointSummary.from_dict(crossed)
        crossed_summary.validate_rows(rows)

    dishonest = _summary(rows)
    dishonest["metrics"]["item_accuracy"] = _rate(0, 2)
    with pytest.raises(ValueError, match="recomputed"):
        CheckpointSummary.from_dict(dishonest).validate_rows(rows)


def test_guardrail_report_is_exact_conjunctive_and_excludes_routes():
    report = GuardrailReport.from_dict(_guardrail_report())
    assert report.confirmatory_passed
    assert set(report.guards) == {
        "factual_job",
        "split_off_leakage",
        "retrieval_procedure",
        "relation_rule",
        "natural_text",
        "instrument_integrity",
    }

    extra = _guardrail_report()
    extra["route_guardrails"] = {"route_rate": 0.5}
    with pytest.raises(ValueError, match="unknown"):
        GuardrailReport.from_dict(extra)

    inconsistent = _guardrail_report()
    inconsistent["guards"]["natural_text"]["passed"] = False
    with pytest.raises(ValueError, match="conjunction"):
        GuardrailReport.from_dict(inconsistent)

    weakened = _guardrail_report()
    check = weakened["guards"]["split_off_leakage"]["checks"][0]
    check.update(value=1.0, threshold=0.0, comparison=">=", passed=True)
    with pytest.raises(ValueError, match="comparison is not preregistered"):
        GuardrailReport.from_dict(weakened)

    dishonest_counts = _guardrail_report()
    dishonest_counts["guards"]["factual_job"]["checks"][0].update(
        numerator=0,
        denominator=1,
    )
    with pytest.raises(ValueError, match="counts"):
        GuardrailReport.from_dict(dishonest_counts)

    dishonest_wilson = _guardrail_report()
    dishonest_wilson["guards"]["split_off_leakage"]["checks"][1][
        "value"
    ] = 0.1
    with pytest.raises(ValueError, match="Wilson"):
        GuardrailReport.from_dict(dishonest_wilson)

    same_checkpoint = _guardrail_report()
    same_checkpoint["dense_checkpoint_sha256"] = same_checkpoint[
        "split_checkpoint_sha256"
    ]
    with pytest.raises(ValueError, match="distinct checkpoints"):
        GuardrailReport.from_dict(same_checkpoint)

    same_configuration = _guardrail_report()
    same_configuration["dense_configuration_sha256"] = same_configuration[
        "split_configuration_sha256"
    ]
    with pytest.raises(ValueError, match="configuration.*distinct"):
        GuardrailReport.from_dict(same_configuration)

    same_result_provenance = _guardrail_report()
    same_result_provenance[
        "dense_result_provenance_sha256"
    ] = same_result_provenance["split_result_provenance_sha256"]
    with pytest.raises(ValueError, match="provenance.*distinct"):
        GuardrailReport.from_dict(same_result_provenance)


def test_typed_contracts_and_dataclass_replace_cannot_bypass_validation():
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    row = rows[0]
    summary = CheckpointSummary.from_dict(_summary(rows))
    report = GuardrailReport.from_dict(_guardrail_report())

    with pytest.raises(ValueError, match="correct disagrees"):
        replace(row, correct=False)
    with pytest.raises(ValueError, match="two rows per pair"):
        replace(summary, n_rows=3)
    with pytest.raises(ValueError, match="six-guard conjunction"):
        replace(report, confirmatory_passed=False)


def test_nested_contract_data_is_deeply_immutable():
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))
    report = GuardrailReport.from_dict(_guardrail_report())

    with pytest.raises(TypeError):
        summary.metrics["item_accuracy"]["numerator"] = 0
    with pytest.raises(TypeError):
        report.guards["factual_job"]["passed"] = False
    with pytest.raises(TypeError):
        report.guards["factual_job"]["checks"][0]["passed"] = False


def test_summary_boundary_rejects_forced_post_validation_mutation():
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))
    dishonest_metrics = summary.to_dict()["metrics"]
    dishonest_metrics["milestone_crossings"][
        "path_composition_pair_accuracy_0.75"
    ] = summary.raw_token_count + 1
    object.__setattr__(summary, "metrics", dishonest_metrics)

    with pytest.raises(ValueError, match="future raw-token"):
        summary.validate_rows(rows)


def test_canonical_serialization_revalidates_typed_contracts():
    report = GuardrailReport.from_dict(_guardrail_report())
    dishonest_guards = report.to_dict()["guards"]
    dishonest_guards["factual_job"]["passed"] = False
    object.__setattr__(report, "guards", dishonest_guards)

    with pytest.raises(ValueError, match="conjunction"):
        canonical_json_bytes(report)


def test_semantic_validation_rejects_schema_shaped_dishonest_payloads():
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    dishonest_rate = _summary(rows)
    dishonest_rate["metrics"]["item_accuracy"]["value"] = 0.5
    with pytest.raises(ValueError, match="counts"):
        validate_result_payload(dishonest_rate)

    dishonest_report = _guardrail_report()
    dishonest_report["guards"]["factual_job"]["passed"] = False
    with pytest.raises(ValueError, match="conjunction"):
        validate_result_payload(dishonest_report)


def test_task8_guardrail_collection_fails_closed_on_legacy_measurements():
    from scripts.analyze_relational import _collect_guardrails

    key = ("d160m", "split", "n50k", 1001)
    with pytest.raises(ValueError, match="strict GuardrailReport"):
        _collect_guardrails(
            {key: {"guardrails": {"within_run_guardrails": {}}}}
        )

    report = GuardrailReport.from_dict(_guardrail_report())
    dense_key = ("d160m", "dense", "n50k", 1001)
    with pytest.raises(ValueError, match="provenance-bound"):
        _collect_guardrails(
            {
                key: {"guardrail_report": report},
                dense_key: {},
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "split_checkpoint_sha256",
        "dense_checkpoint_sha256",
        "model_id",
        "seed",
        "raw_token_count",
        "split_configuration_sha256",
        "dense_configuration_sha256",
        "result_schema_sha256",
        "split_result_provenance_sha256",
        "dense_result_provenance_sha256",
        "study_provenance_sha256",
    ],
)
def test_task8_rejects_copied_or_cross_identity_guardrail_report(
    field,
    tmp_path,
):
    from scripts.analyze_relational import _validate_bound_guardrail_report
    from evals.checkpoint_binding import canonical_configuration_sha256
    from evals.relational_pairing import build_pairing_receipt
    from scripts.run_relational_evals import _ValidatedCheckpointMatrix

    common_config = {
        "model": "d160m",
        "seed": 1001,
        "load": "n50k",
        "ctx": 128,
        "initialization_seed": 1001,
        "data_seed": 17,
        "packing": {"block_size": 128},
        "optimizer": {"name": "adamw"},
        "scheduler": {"name": "cosine"},
        "raw_positions": {"start": 0},
        "decode_budget": 6,
        "checkpoint_schedule": [5, 10, 20],
    }
    split_config = {
        **common_config,
        "condition": "split",
        "train_weights": "split.weights.bin",
    }
    dense_config = {
        **common_config,
        "condition": "dense",
        "train_weights": "dense.weights.bin",
    }
    split_configuration_sha256 = canonical_configuration_sha256(
        split_config
    )
    dense_configuration_sha256 = canonical_configuration_sha256(
        dense_config
    )
    row_values = [_row("original"), _row("counterfactual")]
    for value in row_values:
        value["configuration_sha256"] = split_configuration_sha256
    rows = [EvalRow.from_dict(value) for value in row_values]
    split_anchor = CheckpointSummary.from_dict(_summary(rows))
    split_value = split_anchor.to_dict()
    split_value["configuration_sha256"] = split_configuration_sha256
    split_anchor = CheckpointSummary.from_dict(split_value)
    dense_value = split_anchor.to_dict()
    dense_value["checkpoint_sha256"] = "6" * 64
    dense_value["arm"] = "dense"
    dense_value["configuration_sha256"] = dense_configuration_sha256
    dense_value["provenance_sha256"] = "f" * 64
    dense_anchor = CheckpointSummary.from_dict(dense_value)

    receipt = build_pairing_receipt(
        split_anchor,
        dense_anchor,
        split_config,
        dense_config,
    ).to_dict()
    report_value = _guardrail_report()
    report_value["split_configuration_sha256"] = (
        split_anchor.configuration_sha256
    )
    report_value["dense_configuration_sha256"] = (
        dense_anchor.configuration_sha256
    )
    report_value["split_result_provenance_sha256"] = (
        split_anchor.provenance_sha256
    )
    report_value["dense_result_provenance_sha256"] = (
        dense_anchor.provenance_sha256
    )
    report_value["study_provenance_sha256"] = receipt[
        "study_provenance_sha256"
    ]
    report_value["pairing_receipt_sha256"] = receipt["receipt_sha256"]
    valid_report = GuardrailReport.from_dict(report_value)
    split_directory = tmp_path / "split-run"
    dense_directory = tmp_path / "dense-run"
    split_checkpoint_dir = (
        split_directory / "evals" / split_anchor.checkpoint_sha256
    )
    dense_checkpoint_dir = (
        dense_directory / "evals" / dense_anchor.checkpoint_sha256
    )
    split_checkpoint_dir.mkdir(parents=True)
    dense_checkpoint_dir.mkdir(parents=True)
    (split_directory / "config.yaml").write_text(json.dumps(split_config))
    (dense_directory / "config.yaml").write_text(json.dumps(dense_config))
    split_matrix = _ValidatedCheckpointMatrix(
        summaries={},
        guardrail_artifact={},
        guardrail_artifact_sha256=report_value[
            "split_guardrail_source_sha256"
        ],
        matrix_manifest_sha256=report_value[
            "split_matrix_manifest_sha256"
        ],
        control_transformations_sha256="9" * 64,
        identity_sha256="b" * 64,
        checkpoint_dir=split_checkpoint_dir,
    )
    dense_matrix = _ValidatedCheckpointMatrix(
        summaries={},
        guardrail_artifact={},
        guardrail_artifact_sha256=report_value[
            "dense_guardrail_source_sha256"
        ],
        matrix_manifest_sha256=report_value[
            "dense_matrix_manifest_sha256"
        ],
        control_transformations_sha256="a" * 64,
        identity_sha256="c" * 64,
        checkpoint_dir=dense_checkpoint_dir,
    )
    report_path = split_directory / "evals" / "guardrail-report.json"
    receipt_path = split_directory / "evals" / "pairing-receipt.json"
    report_path.write_bytes(canonical_json_bytes(valid_report))
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    split_run = {
        "on": split_anchor,
        "matrix": split_matrix,
        "pairing_receipt": receipt,
        "pairing_receipt_content": canonical_json_bytes(receipt),
        "pairing_receipt_path": receipt_path,
        "guardrail_report": valid_report,
        "guardrail_report_content": canonical_json_bytes(valid_report),
        "guardrail_report_path": report_path,
        "directory": str(split_directory),
        "cfg": split_config,
    }
    dense_run = {
        "on": dense_anchor,
        "matrix": dense_matrix,
        "directory": str(dense_directory),
        "cfg": dense_config,
    }
    assert (
        _validate_bound_guardrail_report(split_run, dense_run)
        == valid_report
    )

    copied_path = dense_directory / "evals" / "guardrail-report.json"
    copied_path.write_bytes(canonical_json_bytes(valid_report))
    split_run["guardrail_report_path"] = copied_path
    with pytest.raises(ValueError, match="canonical report location"):
        _validate_bound_guardrail_report(split_run, dense_run)
    split_run["guardrail_report_path"] = report_path

    crossed = valid_report.to_dict()
    crossed[field] = {
        "split_checkpoint_sha256": "b" * 64,
        "dense_checkpoint_sha256": "c" * 64,
        "model_id": "copied-model",
        "seed": crossed["seed"] + 1,
        "raw_token_count": crossed["raw_token_count"] + 1,
            "split_configuration_sha256": "a" * 64,
            "dense_configuration_sha256": "b" * 64,
            "result_schema_sha256": "c" * 64,
            "split_result_provenance_sha256": "a" * 64,
            "dense_result_provenance_sha256": "b" * 64,
            "study_provenance_sha256": "c" * 64,
    }[field]
    crossed_report = GuardrailReport.from_dict(crossed)
    crossed_content = canonical_json_bytes(crossed_report)
    report_path.write_bytes(crossed_content)
    split_run["guardrail_report"] = crossed_report
    split_run["guardrail_report_content"] = crossed_content
    with pytest.raises(ValueError, match=field):
        _validate_bound_guardrail_report(split_run, dense_run)


def test_python_contract_and_json_schema_have_identical_required_fields():
    schema = load_result_schema()

    assert "MUST invoke" in schema["description"]
    assert set(schema["$defs"]["EvalRow"]["required"]) == set(EVAL_ROW_FIELDS)
    assert set(schema["$defs"]["CheckpointSummary"]["required"]) == set(
        CHECKPOINT_SUMMARY_FIELDS
    )
    assert set(schema["$defs"]["GuardrailReport"]["required"]) == set(
        GUARDRAIL_REPORT_FIELDS
    )
    assert schema["$defs"]["EvalRow"]["additionalProperties"] is False
    assert set(
        schema["$defs"]["EvalRow"]["properties"]["control_id"]["enum"]
    ) == {
        "correct",
        "shuffled_returns",
        "relevant_edge",
        "irrelevant_edge",
        "gold_path",
        "gold_returns",
        "no_query",
        "explicit_miss",
        "handle_swap",
        "entity_rename",
        "graph_isomorphism",
    }
    assert set(
        schema["$defs"]["GuardCheck"]["properties"]["check_id"]["enum"]
    ) == {
        check["check_id"]
        for guard in _guardrail_report()["guards"].values()
        for check in guard["checks"]
    }


def test_atomic_publication_uses_bound_nonoverwriting_checkpoint_path(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))

    output = publish_evaluation(run, rows, summary)

    assert output == (
        run
        / "evals"
        / HASHES["checkpoint_sha256"]
        / "memory_on"
        / "correct"
    )
    assert not any(".tmp-" in entry.name for entry in output.parent.iterdir())
    assert json.loads((output / "summary.json").read_text()) == summary.to_dict()
    assert (output / "manifest.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        publish_evaluation(run, rows, summary)


def test_failed_cell_contender_does_not_remove_owned_lock(tmp_path):
    run = tmp_path / "run"
    mode = (
        run
        / "evals"
        / HASHES["checkpoint_sha256"]
        / "memory_on"
    )
    mode.mkdir(parents=True)
    lock = mode / ".correct.publish.lock"
    lock.write_text("owner")
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))

    with pytest.raises(FileExistsError):
        publish_evaluation(run, rows, summary)
    assert lock.read_text() == "owner"

    lock.unlink()
    assert publish_evaluation(run, rows, summary).is_dir()


def test_cell_publication_fails_closed_on_parent_symlink_swap(
    tmp_path,
    monkeypatch,
):
    import evals.relational_contracts as contracts

    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = tmp_path / "moved-mode"
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))
    original_rename = contracts._rename_directory_noreplace_at

    def swap_then_rename(parent_fd, source, destination):
        mode = (
            run
            / "evals"
            / HASHES["checkpoint_sha256"]
            / "memory_on"
        )
        mode.rename(moved)
        mode.symlink_to(outside, target_is_directory=True)
        return original_rename(parent_fd, source, destination)

    monkeypatch.setattr(
        contracts,
        "_rename_directory_noreplace_at",
        swap_then_rename,
    )
    with pytest.raises(ValueError, match="changed during publication"):
        publish_evaluation(run, rows, summary)
    assert not any(outside.iterdir())
    assert not (moved / "correct").exists()

    monkeypatch.setattr(
        contracts,
        "_rename_directory_noreplace_at",
        original_rename,
    )
    (run / "evals" / HASHES["checkpoint_sha256"] / "memory_on").unlink()
    moved.rename(
        run / "evals" / HASHES["checkpoint_sha256"] / "memory_on"
    )
    assert publish_evaluation(run, rows, summary).is_dir()


def test_streaming_publication_spools_pairs_without_buffering_rows(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    publisher = StreamingEvaluationPublisher(run)

    for index in range(25):
        pair = []
        for variant in ("original", "counterfactual"):
            value = _row(variant)
            suffix = "o" if variant == "original" else "c"
            value.update(
                qid=f"pair-{index:03d}-{suffix}",
                pair_id=f"pair-{index:03d}",
            )
            pair.append(EvalRow.from_dict(value))
        publisher.add_pair(pair)
        assert publisher.buffered_rows == 0

    output, summary = publisher.finish()

    assert summary.n_rows == 50
    assert summary.n_pairs == 25
    assert output.is_dir()
    assert sum(1 for _ in (output / "rows.jsonl").open()) == 50
    assert publisher.closed


@pytest.mark.parametrize("tampered", ["summary", "rows", "manifest"])
def test_historical_cell_validation_rejects_tampering(tmp_path, tampered):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))
    output = publish_evaluation(run, rows, summary)
    assert validate_published_evaluation(output) == summary

    if tampered == "summary":
        value = json.loads((output / "summary.json").read_text())
        value["metrics"]["item_accuracy"]["value"] = 0.5
        (output / "summary.json").write_text(json.dumps(value))
    elif tampered == "rows":
        with (output / "rows.jsonl").open("ab") as stream:
            stream.write(b" \n")
    else:
        value = json.loads((output / "manifest.json").read_text())
        value["n_rows"] = 999
        (output / "manifest.json").write_text(json.dumps(value))

    with pytest.raises(ValueError):
        validate_published_evaluation(output)


def test_publication_rejects_symlink_and_cross_checkpoint_contamination(
    tmp_path,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    (run / "evals").symlink_to(outside, target_is_directory=True)
    rows = [
        EvalRow.from_dict(_row("original")),
        EvalRow.from_dict(_row("counterfactual")),
    ]
    summary = CheckpointSummary.from_dict(_summary(rows))
    with pytest.raises(ValueError, match="symlink"):
        publish_evaluation(run, rows, summary)
    assert not any(outside.iterdir())

    (run / "evals").unlink()
    crossed = _summary(rows)
    crossed["checkpoint_sha256"] = "9" * 64
    crossed["rows_sha256"] = rows_sha256(rows)
    with pytest.raises(ValueError, match="checkpoint"):
        publish_evaluation(
            run,
            rows,
            CheckpointSummary.from_dict(crossed),
        )
