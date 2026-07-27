import copy
import hashlib
import json
import math

import pytest

from evals.relational_contracts import EvalRow, cluster_id_for
from evals.relational_metrics import (
    CheckpointMetricAccumulator,
    assert_expected_counts,
    compute_checkpoint_metrics,
    counterfactual_pair_accuracy,
    evaluate_confirmatory_guardrails,
    exact_accuracy,
    first_frozen_milestone_crossings,
    internal_knowledge_guardrail,
    language_bpb_guardrail,
    mask_ledger_guardrail,
    path_diagnostics,
    path_metrics,
    recognition_accuracy,
    recognition_guardrails,
    route_guardrails,
    shared_text_bpb,
    wilson_interval,
)


def _result(pair_id, variant, correct, task="path_composition"):
    suffix = "o" if variant == "original" else "c"
    return {
        "qid": f"{pair_id}-{suffix}",
        "pair_id": pair_id,
        "variant": variant,
        "task": task,
        "correct": correct,
    }


def test_counterfactual_pair_accuracy_requires_both_distinct_variants():
    rows = [
        _result("p0", "original", True),
        _result("p0", "counterfactual", True),
        _result("p1", "original", True),
        _result("p1", "counterfactual", False),
    ]
    assert counterfactual_pair_accuracy(rows, expected_pairs=2) == 0.5

    duplicate = [
        _result("p0", "original", True),
        _result("p0", "original", True),
    ]
    with pytest.raises(ValueError, match="original and counterfactual"):
        counterfactual_pair_accuracy(duplicate)


def test_counterfactual_pair_accuracy_rejects_missing_or_unexpected_counts():
    with pytest.raises(ValueError, match="requires original and counterfactual"):
        counterfactual_pair_accuracy([_result("p0", "original", True)])

    complete = [
        _result("p0", "original", True),
        _result("p0", "counterfactual", True),
    ]
    with pytest.raises(ValueError, match="expected 2 pairs"):
        counterfactual_pair_accuracy(complete, expected_pairs=2)
    invalid_boolean = [
        _result("p0", "original", 1),
        _result("p0", "counterfactual", True),
    ]
    with pytest.raises(ValueError, match="correct.*Boolean"):
        counterfactual_pair_accuracy(invalid_boolean)


def test_expected_counts_are_exact_per_frozen_stratum():
    rows_by_task = {}
    for task in (
        "path_composition",
        "date_ordering",
        "balanced_equality",
    ):
        rows_by_task[task] = [
            _result(f"{task}-{pair}", variant, True, task)
            for pair in range(2)
            for variant in ("original", "counterfactual")
        ]
    assert_expected_counts(rows_by_task, n_pairs=2)

    rows_by_task["date_ordering"].pop()
    with pytest.raises(ValueError, match="date_ordering: expected 4 rows"):
        assert_expected_counts(rows_by_task, n_pairs=2)


def test_path_metrics_report_exact_hop_referent_and_failure_rates():
    rows = [
        {
            "actions": ["a", "b", "c"],
            "gold_actions": ["a", "b", "x"],
            "correct_referents": [True, True, False],
            "misses": 1,
            "malformed": 1,
            "excess_reads": 2,
            "halt_step": 4,
            "n_steps": 6,
        }
    ]

    out = path_diagnostics(rows)

    assert out["full_path_exact"] == 0.0
    assert out["per_hop_accuracy"] == pytest.approx(2 / 3)
    assert out["correct_referent_rate"] == pytest.approx(2 / 3)
    assert out["miss_rate"] == pytest.approx(1 / 3)
    assert out["excess_read_rate"] == pytest.approx(2 / 3)
    assert out["malformed_rate"] == pytest.approx(1 / 6)
    assert out["mean_halt_step"] == 4.0


def test_path_metrics_reject_missing_hop_evidence():
    with pytest.raises(ValueError, match="correct_referents"):
        path_diagnostics(
            [
                {
                    "actions": ["a"],
                    "gold_actions": ["a", "b"],
                    "correct_referents": [True],
                    "misses": 0,
                    "malformed": 0,
                    "excess_reads": 0,
                    "halt_step": None,
                    "n_steps": 6,
                }
            ]
        )


def test_core_path_metrics_require_only_the_frozen_action_and_miss_fields():
    out = path_metrics(
        [
            {
                "actions": ["a", "b", "c"],
                "gold_actions": ["a", "b", "x"],
                "misses": 1,
            }
        ]
    )
    assert out == {
        "full_path_exact": 0.0,
        "per_hop_accuracy": pytest.approx(2 / 3),
        "miss_rate": pytest.approx(1 / 3),
    }


def test_wilson_and_four_way_recognition_are_exact():
    items = [
        {
            "prompt": "first",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 2,
        },
        {
            "prompt": "second",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 0,
        },
    ]

    def scores(prompt, choices):
        del choices
        return [0.0, 0.1, 0.9, 0.2] if prompt == "first" else [0.0, 1.0, 0.5, 0.2]

    result = recognition_accuracy(scores, items, expected_count=2)
    expected_lo, expected_hi = wilson_interval(1, 2)

    assert result == {
        "accuracy": 0.5,
        "ci_lo": expected_lo,
        "ci_hi": expected_hi,
        "n": 2,
        "correct": 1,
    }


def test_recognition_rejects_non_four_way_or_missing_items():
    with pytest.raises(ValueError, match="four choices"):
        recognition_accuracy(
            lambda prompt, choices: [1.0, 0.0],
            [{"prompt": "x", "choices": ["a", "b"], "answer_index": 0}],
        )
    with pytest.raises(ValueError, match="requires at least one item"):
        recognition_accuracy(lambda prompt, choices: [], [])


def test_shared_text_bpb_uses_utf8_bytes_and_natural_log_units():
    assert shared_text_bpb(8 * math.log(2), 4) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="must contain bytes"):
        shared_text_bpb(0.0, 0)


def test_burden_and_leakage_use_wilson_bounds_not_point_estimates():
    dense = exact_accuracy(
        [{"correct": index < 40} for index in range(100)],
        expected_count=100,
    )
    split = exact_accuracy(
        [{"correct": index < 20} for index in range(100)],
        expected_count=100,
    )

    result = recognition_guardrails(dense, split)

    assert result["burden"]["passed"]
    assert result["burden"]["value"] == dense["ci_lo"]
    assert result["leakage"]["passed"]
    assert result["leakage"]["value"] == split["ci_hi"]


def test_internal_and_language_guardrails_measure_paired_deltas():
    split = {"accuracy": 0.78, "n": 100}
    dense = {"accuracy": 0.80, "n": 100}

    internal = internal_knowledge_guardrail(split, dense)
    language = language_bpb_guardrail(
        {"bpb": 1.01, "total_utf8_bytes": 10_000},
        {"bpb": 1.00, "total_utf8_bytes": 10_000},
    )

    assert internal["passed"] and internal["value"] == pytest.approx(-0.02)
    assert internal["rule"] == "split >= dense - 0.02"
    assert language["passed"] and language["value"] == pytest.approx(1.01)


def test_paired_guardrails_reject_unshared_evaluation_denominators():
    with pytest.raises(ValueError, match="same item count"):
        internal_knowledge_guardrail(
            {"accuracy": 0.8, "n": 99},
            {"accuracy": 0.8, "n": 100},
        )
    with pytest.raises(ValueError, match="same shared text"):
        language_bpb_guardrail(
            {"bpb": 1.0, "total_utf8_bytes": 99},
            {"bpb": 1.0, "total_utf8_bytes": 100},
        )


def test_route_and_mask_guardrails_require_measured_counts():
    route = route_guardrails(
        {
            "route_rate": 0.5,
            "route_total": 100,
            "low_use_high_entropy_external_rate": 0.8,
            "low_use_high_entropy_total": 25,
            "rules_top_centrality_internal_rate": 0.9,
            "rules_top_centrality_total": 20,
        }
    )
    mask = mask_ledger_guardrail(
        {
            "unmasked_external_payloads": 0,
            "external_payload_occurrences": 50,
            "masked_rule_action_answer_targets": 0,
            "rule_action_answer_targets": 200,
        },
        condition="split",
    )

    assert all(value["passed"] for value in route.values())
    assert mask["passed"]

    with pytest.raises(KeyError):
        route_guardrails({"route_rate": 0.5})


@pytest.mark.parametrize("condition", ["dense", "random"])
def test_non_split_ledgers_allow_factual_targets_to_remain_unmasked(condition):
    faithful = mask_ledger_guardrail(
        {
            "unmasked_external_payloads": 50,
            "external_payload_occurrences": 50,
            "masked_rule_action_answer_targets": 0,
            "rule_action_answer_targets": 200,
        },
        condition=condition,
    )
    split = mask_ledger_guardrail(
        {
            "unmasked_external_payloads": 50,
            "external_payload_occurrences": 50,
            "masked_rule_action_answer_targets": 0,
            "rule_action_answer_targets": 200,
        },
        condition="split",
    )

    assert faithful["passed"] and not faithful["external_mask_applicable"]
    assert not split["passed"] and split["external_mask_applicable"]


def _strict_actions(hop: int = 1) -> list[list]:
    actions = [
        [0, f"r{index}", "out", True, False]
        for index in range(hop)
    ]
    if hop < 6:
        actions.append([0, "r0", "out", False, True])
        actions.extend(
            [[0, "r0", "out", False, False] for _ in range(5 - hop)]
        )
    return actions


def _strict_row(
    pair: str,
    variant: str,
    *,
    correct: bool,
    hop: int,
    composition_split: str,
    control_id: str = "correct",
    wrong_hop: int | None = None,
    misses: int = 0,
    malformed: int = 0,
    abstained: bool = False,
    excess_reads: int = 0,
    locality: bool | None = None,
) -> EvalRow:
    qid = f"{pair}-{'o' if variant == 'original' else 'c'}"
    path_hash = (
        "a" * 63 + str(hop)
        if hop < 10
        else "a" * 64
    )
    gold = _strict_actions(hop)
    predicted = copy.deepcopy(gold)
    if excess_reads:
        if excess_reads != 1 or hop >= 5:
            raise ValueError("test fixture supports one excess read")
        predicted[hop] = [0, "r0", "out", True, False]
        predicted[hop + 1] = [0, "r0", "out", False, True]
    if wrong_hop is not None:
        predicted[wrong_hop - 1][1] = "wrong-relation"
    returned = [
        (
            [100 + index, predicted[index][1], "out"]
            if index < hop + excess_reads and index >= misses
            else None
        )
        for index in range(6)
    ]
    referents = [
        wrong_hop != index + 1 and index >= misses
        for index in range(hop)
    ]
    oracle_effect = (
        "changed"
        if control_id == "relevant_edge"
        else "unchanged"
    )
    changed = (
        [[100, "r0", "out"]]
        if control_id in {"relevant_edge", "irrelevant_edge"}
        else []
    )
    value = {
        "record_type": "eval_row",
        "schema_version": 1,
        "qid": qid,
        "pair_id": pair,
        "variant": variant,
        "task": "path_composition",
        "world_id": 4,
        "provenance_id": "world-4",
        "relation_path_hash": path_hash,
        "template_id": "path:v1",
        "composition_split": composition_split,
        "hop": hop,
        "seed": 1001,
        "model_id": "d160m",
        "arm": "split",
        "checkpoint_sha256": "1" * 64,
        "raw_token_count": 500,
        "memory_mode": "memory_on",
        "control_id": control_id,
        "evaluator_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "relation_schema_sha256": "4" * 64,
        "configuration_sha256": "6" * 64,
        "result_schema_sha256": "7" * 64,
        "provenance_sha256": "5" * 64,
        "cluster_id": cluster_id_for(
            seed=1001,
            world_id=4,
            relation_path_hash=path_hash,
            template_id="path:v1",
        ),
        "prediction": "" if abstained else "r1" if correct else "r0",
        "answer": "r1",
        "correct": correct,
        "prediction_source": "model",
        "all_actions": predicted,
        "gold_all_actions": gold,
        "returned_addresses": returned,
        "gold_addresses": [
            [100 + index, f"r{index}", "out"]
            for index in range(hop)
        ],
        "correct_referents": referents,
        "misses": misses,
        "malformed": malformed,
        "abstained": abstained,
        "excess_reads": excess_reads,
        "halt_step": (
            hop + excess_reads + 1
            if hop + excess_reads < 6
            else None
        ),
        "answer_logits": [[-0.1] for _ in range(6)],
        "lookup_latency_ns": 10 + hop,
        "lookup_count": hop + excess_reads,
        "store_rows": 20,
        "store_bytes": 2000,
        "control_seed": 8,
        "transformation_id": None,
        "source_store_sha256": None,
        "transformed_store_sha256": None,
        "transformation_metadata_sha256": None,
        "changed_addresses": changed,
        "oracle_before": "r0" if oracle_effect == "changed" else "r1",
        "oracle_after": "r1",
        "oracle_effect": oracle_effect,
        "edit_locality_correct": locality,
    }
    return EvalRow.from_dict(value)


def test_complete_checkpoint_metrics_cover_paths_slices_store_and_empty_denominators():
    rows = [
        _strict_row(
            "p0",
            variant,
            correct=True,
            hop=1,
            composition_split="seen",
        )
        for variant in ("original", "counterfactual")
    ] + [
        _strict_row(
            "p1",
            "original",
            correct=True,
            hop=2,
            composition_split="heldout",
        ),
        _strict_row(
            "p1",
            "counterfactual",
            correct=False,
            hop=2,
            composition_split="heldout",
            wrong_hop=2,
            misses=1,
            malformed=1,
            abstained=True,
            excess_reads=1,
        ),
    ]

    metrics = compute_checkpoint_metrics(rows)

    assert metrics["item_accuracy"] == {
        "value": 0.75,
        "numerator": 3,
        "denominator": 4,
    }
    assert metrics["pair_accuracy"] == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert metrics["all_six_action_exact"]["numerator"] == 3
    assert metrics["per_hop"]["1"]["action"]["value"] == 1.0
    assert metrics["per_hop"]["2"]["relation"]["value"] == 0.5
    assert metrics["answer_given_correct_retrieval"] == {
        "value": 1.0,
        "numerator": 3,
        "denominator": 3,
    }
    assert metrics["gold_path_answer_accuracy"] == {
        "value": None,
        "numerator": 0,
        "denominator": 0,
    }
    assert metrics["by_hop"]["6"]["item_accuracy"]["value"] is None
    assert metrics["by_composition"]["seen"]["pair_accuracy"]["value"] == 1.0
    assert metrics["by_composition"]["heldout"]["pair_accuracy"]["value"] == 0.0
    assert set(metrics["by_task"]) == {
        "path_composition",
        "date_ordering",
        "balanced_equality",
    }
    assert metrics["by_task"]["path_composition"]["pair_accuracy"][
        "value"
    ] == 0.5
    assert metrics["by_task"]["date_ordering"]["pair_accuracy"][
        "value"
    ] is None
    assert metrics["store"] == {
        "rows": 20,
        "bytes": 2000,
        "lookup_latency_ns": pytest.approx(46 / 7),
        "lookup_count": 7,
    }
    assert metrics["malformed_rate"]["numerator"] == 1
    assert metrics["miss_rate"] == {
        "value": pytest.approx(1 / 7),
        "numerator": 1,
        "denominator": 7,
    }
    assert metrics["abstention_rate"]["numerator"] == 1
    assert metrics["excess_read_rate"]["denominator"] == 24
    assert metrics["edit_locality"]["value"] is None


def test_checkpoint_metric_accumulator_matches_reference_pair_by_pair():
    pairs = [
        [
            _strict_row(
                f"p{pair}",
                variant,
                correct=not (pair == 1 and variant == "counterfactual"),
                hop=pair + 1,
                composition_split="seen" if pair == 0 else "heldout",
                wrong_hop=2
                if pair == 1 and variant == "counterfactual"
                else None,
            )
            for variant in ("original", "counterfactual")
        ]
        for pair in range(2)
    ]
    expected = compute_checkpoint_metrics(
        [row for pair in pairs for row in pair]
    )
    accumulator = CheckpointMetricAccumulator()

    for pair in pairs:
        accumulator.add_pair(pair)

    assert accumulator.finalize() == expected


def test_gold_path_and_edit_locality_have_explicit_denominators():
    gold_rows = [
        _strict_row(
            "gold",
            variant,
            correct=variant == "original",
            hop=1,
            composition_split="seen",
            control_id="gold_path",
        )
        for variant in ("original", "counterfactual")
    ]
    locality_rows = [
        _strict_row(
            "edit",
            variant,
            correct=True,
            hop=1,
            composition_split="seen",
            control_id="relevant_edge",
            locality=variant == "original",
        )
        for variant in ("original", "counterfactual")
    ]

    gold = compute_checkpoint_metrics(gold_rows)
    locality = compute_checkpoint_metrics(locality_rows)

    assert gold["gold_path_answer_accuracy"] == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert locality["edit_locality"] == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }


def test_checkpoint_metrics_reject_incomplete_duplicate_or_crossed_pairs():
    original = _strict_row(
        "p",
        "original",
        correct=True,
        hop=1,
        composition_split="seen",
    )
    with pytest.raises(ValueError, match="both variants"):
        compute_checkpoint_metrics([original])
    with pytest.raises(ValueError, match="duplicate"):
        compute_checkpoint_metrics([original, original])

    counterfactual = _strict_row(
        "p",
        "counterfactual",
        correct=True,
        hop=2,
        composition_split="seen",
    )
    with pytest.raises(ValueError, match="crossed pair metadata"):
        compute_checkpoint_metrics([original, counterfactual])


def test_frozen_crossings_use_first_raw_checkpoint_without_interpolation():
    history = [
        {
            "tokens_per_parameter": 5,
            "raw_token_count": 500,
            "metrics": {"m1": 0.74, "m2": 0.90},
        },
        {
            "tokens_per_parameter": 10,
            "raw_token_count": 1000,
            "metrics": {"m1": 0.75, "m2": 0.94},
        },
        {
            "tokens_per_parameter": 20,
            "raw_token_count": 2000,
            "metrics": {"m1": 0.90, "m2": 0.949},
        },
    ]

    assert first_frozen_milestone_crossings(
        history,
        {"m1": 0.75, "m2": 0.95},
    ) == {"m1": 1000, "m2": None}

    invalid = copy.deepcopy(history)
    invalid[1]["tokens_per_parameter"] = 7
    with pytest.raises(ValueError, match="5, 10, and 20"):
        first_frozen_milestone_crossings(
            invalid,
            {"m1": 0.75, "m2": 0.95},
        )
    invalid = copy.deepcopy(history)
    invalid[2]["metrics"]["m1"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        first_frozen_milestone_crossings(
            invalid,
            {"m1": 0.75, "m2": 0.95},
        )


def _bound_rate(
    numerator: int,
    denominator: int,
    *,
    arm: str,
    memory_mode: str,
    control_id: str = "correct",
) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "arm": arm,
        "memory_mode": memory_mode,
        "control_id": control_id,
        "checkpoint_sha256": (
            "1" * 64 if arm == "split" else "6" * 64
        ),
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 500,
        "evaluator_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "relation_schema_sha256": "4" * 64,
        "configuration_sha256": (
            "c" * 64 if arm == "split" else "d" * 64
        ),
        "result_schema_sha256": "7" * 64,
        "provenance_sha256": (
            "5" * 64 if arm == "split" else "6" * 64
        ),
    }


def _bound_scalar(
    value: float,
    *,
    arm: str,
    memory_mode: str,
    denominator: int = 1000,
) -> dict:
    return {
        "value": value,
        "denominator": denominator,
        "arm": arm,
        "memory_mode": memory_mode,
        "control_id": "correct",
        "checkpoint_sha256": (
            "1" * 64 if arm == "split" else "6" * 64
        ),
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 500,
        "evaluator_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "relation_schema_sha256": "4" * 64,
        "configuration_sha256": (
            "c" * 64 if arm == "split" else "d" * 64
        ),
        "result_schema_sha256": "7" * 64,
        "provenance_sha256": (
            "5" * 64 if arm == "split" else "6" * 64
        ),
    }


def _passing_guardrail_inputs() -> dict:
    return {
        "split_checkpoint_sha256": "1" * 64,
        "dense_checkpoint_sha256": "6" * 64,
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 500,
        "evaluator_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "relation_schema_sha256": "4" * 64,
        "split_configuration_sha256": "c" * 64,
        "dense_configuration_sha256": "d" * 64,
        "result_schema_sha256": "7" * 64,
        "split_result_provenance_sha256": "5" * 64,
        "dense_result_provenance_sha256": "6" * 64,
        "study_provenance_sha256": "e" * 64,
        "pairing_receipt_sha256": "7" * 64,
        "split_guardrail_source_sha256": "8" * 64,
        "dense_guardrail_source_sha256": "9" * 64,
        "split_matrix_manifest_sha256": "a" * 64,
        "dense_matrix_manifest_sha256": "b" * 64,
        "measurements": {
            "split_on_exact_recall": _bound_rate(
                95, 100, arm="split", memory_mode="memory_on"
            ),
            "dense_on_exact_recall": _bound_rate(
                97, 100, arm="dense", memory_mode="memory_on"
            ),
            "split_off_exact_recall": _bound_rate(
                4, 100, arm="split", memory_mode="memory_off"
            ),
            "split_off_recognition": _bound_rate(
                0, 20, arm="split", memory_mode="memory_off"
            ),
            "split_off_first_hop_accuracy": _bound_rate(
                75, 100, arm="split", memory_mode="memory_off"
            ),
            "split_gold_return_path_accuracy": _bound_rate(
                90,
                100,
                arm="split",
                memory_mode="memory_on",
                control_id="gold_returns",
            ),
            "split_on_path_accuracy": _bound_rate(
                95, 100, arm="split", memory_mode="memory_on"
            ),
            "split_rule_accuracy": _bound_rate(
                78, 100, arm="split", memory_mode="memory_on"
            ),
            "dense_rule_accuracy": _bound_rate(
                80, 100, arm="dense", memory_mode="memory_on"
            ),
            "split_bpb": _bound_scalar(
                1.01, arm="split", memory_mode="memory_off"
            ),
            "dense_bpb": _bound_scalar(
                1.00, arm="dense", memory_mode="memory_off"
            ),
        },
        "integrity": {
            "mask_ledger": True,
            "corpus_pairing": True,
            "provenance": True,
            "exact_matrix": True,
        },
    }


def test_confirmatory_study_definition_has_canonical_shared_identity():
    from evals.relational_metrics import (
        CONFIRMATORY_STUDY_DEFINITION,
        confirmatory_study_definition_sha256,
    )

    assert CONFIRMATORY_STUDY_DEFINITION["record_type"] == (
        "confirmatory_study_definition"
    )
    assert CONFIRMATORY_STUDY_DEFINITION["schema_version"] == 1
    assert set(CONFIRMATORY_STUDY_DEFINITION["guard_checks"]) == {
        "split_on_recall_floor",
        "split_on_recall_noninferiority",
        "split_off_recall",
        "split_off_recognition_wilson_hi",
        "split_off_first_hop",
        "gold_return_path_noninferiority",
        "rule_noninferiority",
        "bpb_noninferiority",
        "integrity_conjunction",
    }
    expected = hashlib.sha256(
        json.dumps(
            CONFIRMATORY_STUDY_DEFINITION,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert confirmatory_study_definition_sha256() == expected


def test_approved_guardrails_are_six_conjunctive_guards_and_eight_inequalities():
    report = evaluate_confirmatory_guardrails(
        _passing_guardrail_inputs()
    )

    assert report.confirmatory_passed
    assert len(report.guards) == 6
    assert sum(
        len(guard["checks"])
        for name, guard in report.guards.items()
        if name != "instrument_integrity"
    ) == 8
    leakage = report.guards["split_off_leakage"]
    assert leakage["passed"]
    wilson_check = leakage["checks"][1]
    assert wilson_check["value"] == pytest.approx(wilson_interval(0, 20)[1])
    assert wilson_check["comparison"] == "<"
    assert report.split_result_provenance_sha256 == "5" * 64
    assert report.dense_result_provenance_sha256 == "6" * 64
    assert report.study_provenance_sha256 == "e" * 64
    assert "provenance_sha256" not in report.to_dict()

    one_failure = _passing_guardrail_inputs()
    one_failure["measurements"]["split_off_recognition"] = _bound_rate(
        5, 20, arm="split", memory_mode="memory_off"
    )
    failed = evaluate_confirmatory_guardrails(one_failure)
    assert not failed.guards["split_off_leakage"]["passed"]
    assert not failed.confirmatory_passed


def test_guardrail_thresholds_use_exact_strict_and_inclusive_boundaries():
    passing = evaluate_confirmatory_guardrails(
        _passing_guardrail_inputs()
    )
    assert passing.guards["factual_job"]["passed"]
    assert passing.guards["retrieval_procedure"]["passed"]
    assert passing.guards["relation_rule"]["passed"]
    assert passing.guards["natural_text"]["passed"]

    strict = _passing_guardrail_inputs()
    strict["measurements"]["split_off_exact_recall"] = _bound_rate(
        5, 100, arm="split", memory_mode="memory_off"
    )
    report = evaluate_confirmatory_guardrails(strict)
    assert not report.guards["split_off_leakage"]["checks"][0]["passed"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda value: value["measurements"][
                "split_off_first_hop_accuracy"
            ].update(memory_mode="memory_on"),
            "memory_mode",
        ),
        (
            lambda value: value["measurements"][
                "dense_rule_accuracy"
            ].update(checkpoint_sha256="9" * 64),
            "checkpoint",
        ),
        (
            lambda value: value["measurements"][
                "split_bpb"
            ].update(value=float("nan")),
            "finite",
        ),
        (
            lambda value: value["measurements"][
                "dense_bpb"
            ].update(value=0.0),
            "positive",
        ),
        (
            lambda value: value["measurements"][
                "dense_on_exact_recall"
            ].update(denominator=99),
            "denominator",
        ),
        (
            lambda value: value["measurements"][
                "split_on_exact_recall"
            ].update(data_sha256="9" * 64),
            "data_sha256 mismatch",
        ),
        (
            lambda value: value.update(
                exploratory_route_guardrails={"route_rate": 0.5}
            ),
            "unknown",
        ),
    ],
)
def test_confirmatory_guardrails_fail_closed_on_mismatched_inputs(
    mutation,
    message,
):
    inputs = _passing_guardrail_inputs()
    mutation(inputs)
    with pytest.raises(ValueError, match=message):
        evaluate_confirmatory_guardrails(inputs)
