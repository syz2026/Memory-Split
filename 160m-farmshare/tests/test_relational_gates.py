from __future__ import annotations

import copy
import hashlib

import pytest

from evals.relational_gates import (
    CORE_TASKS,
    build_gate_0_receipt,
    evaluate_development_gates,
    load_development_gate_inputs,
    validate_gate_receipt,
)
from experiment.artifacts import atomic_write_json, canonical_json_bytes
from tests.task8_helpers import make_rows


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def common_input_hashes() -> dict[str, str]:
    return {
        "source_lock_sha256": _digest("source-lock"),
        "relation_schema_sha256": _digest("relation-schema"),
        "preregistration_sha256": _digest("preregistration"),
        "evaluator_sha256": _digest("evaluator"),
        "analysis_sha256": _digest("analysis"),
        "source_tree_sha256": _digest("source-tree"),
    }


def _arm_metrics(value: float = 0.80) -> dict:
    return {
        "task_pair_accuracy": {task: value for task in CORE_TASKS},
        "natural_text_bpb": 1.5,
    }


def _development_rows(label: str, arm: str, value: float = 0.80) -> list[dict]:
    pair_count = 10
    passing = round(value * pair_count)
    return [
        row.to_dict()
        for row in make_rows(
            label,
            arm,
            seeds=(201,),
            model_id="d29m",
            namespace=f"development:{label}",
            pair_counts={task: pair_count for task in CORE_TASKS},
            success=lambda _label, _seed, _task, index: index < passing,
        )
    ]


def development_fixture() -> dict:
    mixtures = []
    for index, mixture in enumerate(
        ([0.70, 0.15, 0.15], [0.65, 0.15, 0.20], [0.65, 0.20, 0.15])
    ):
        mixtures.append(
            {
                "mixture": mixture,
                "parameter_count": 28_969_216,
                "manifest_sha256": _digest(f"mixture-{index}"),
                "arms": {
                    "dense": {
                        **_arm_metrics(),
                        "rows": _development_rows(
                            f"mixture-{index}-dense",
                            "dense",
                        ),
                        "fact_recognition": {
                            "successes": 400,
                            "total": 1_000,
                        },
                    },
                    "split": {
                        **_arm_metrics(),
                        "rows": _development_rows(
                            f"mixture-{index}-split",
                            "split",
                        ),
                    },
                },
            }
        )
    intervals = {
        f"{arm}.{metric}": {"ci_lo": -0.005, "ci_hi": 0.005}
        for arm in ("dense", "split")
        for metric in (*CORE_TASKS, "fact_recall")
    }
    return {
        "record_type": "relational_development_gate_inputs",
        "schema_version": 1,
        "scope": "development",
        "input_hashes": common_input_hashes(),
        "mixtures": mixtures,
        "loads": [
            {
                "entities": 50_000,
                "manifest_sha256": _digest("load-50k"),
                "dense_fact_recall": 0.90,
                "dense_reasoning_composite": 0.85,
            },
            {
                "entities": 200_000,
                "manifest_sha256": _digest("load-200k"),
                "dense_fact_recall": 0.70,
                "dense_reasoning_composite": 0.80,
            },
            {
                "entities": 800_000,
                "manifest_sha256": _digest("load-800k"),
                "dense_fact_recall": 0.55,
                "dense_reasoning_composite": 0.60,
            },
        ],
        "budget": {
            "at_10x": {
                "manifest_sha256": _digest("budget-10x"),
                "arms": {
                    "dense": _arm_metrics(),
                    "split": _arm_metrics(),
                },
                "fact_recognition_passed": True,
                "memory_guardrails_passed": True,
                "natural_text_noninferiority_passed": True,
                "paired_change_ci_to_20x": intervals,
            },
            "at_20x": {
                "manifest_sha256": _digest("budget-20x"),
                "available": True,
            },
        },
    }


def test_gate_receipts_apply_frozen_selection_rules():
    receipts = evaluate_development_gates(development_fixture())

    assert set(receipts) == {"gate_1", "gate_2", "gate_3", "gate_4"}
    assert receipts["gate_1"]["passed"]
    assert receipts["gate_2"]["selected_mixture"] == [0.70, 0.15, 0.15]
    assert receipts["gate_3"]["low_entities"] == 50_000
    assert receipts["gate_3"]["high_entities"] == 800_000
    assert receipts["gate_3"]["confirmation_entities"] == 1_800_000
    assert receipts["gate_4"]["tokens_per_parameter"] == 10
    assert receipts["gate_2"]["gate_1_receipt_sha256"] == receipts["gate_1"][
        "receipt_sha256"
    ]
    assert receipts["gate_3"]["gate_2_receipt_sha256"] == receipts["gate_2"][
        "receipt_sha256"
    ]
    assert receipts["gate_4"]["gate_3_receipt_sha256"] == receipts["gate_3"][
        "receipt_sha256"
    ]
    assert len(
        {
            receipt["development_input_sha256"]
            for receipt in receipts.values()
        }
    ) == 1
    for number, receipt in enumerate(receipts.values(), 1):
        assert receipt["gate"] == number
        assert receipt["input_hashes"] == common_input_hashes()
        assert validate_gate_receipt(receipt, expected_gate=number) == receipt


def test_gate_2_uses_first_passing_ordered_mixture():
    fixture = development_fixture()
    fixture["mixtures"][0]["arms"]["dense"]["fact_recognition"] = {
        "successes": 250,
        "total": 1_000,
    }

    receipt = evaluate_development_gates(fixture)["gate_2"]

    assert receipt["passed"]
    assert receipt["selected_mixture"] == [0.65, 0.15, 0.20]
    assert receipt["selected_mixture_index"] == 1


def test_gate_1_recomputes_task_accuracy_from_validated_rows():
    fixture = development_fixture()
    fixture["mixtures"][0]["arms"]["dense"]["task_pair_accuracy"][
        CORE_TASKS[0]
    ] = 0.90

    with pytest.raises(ValueError, match="rows|accuracy|measurement"):
        evaluate_development_gates(fixture)


def test_gate_1_rejects_rows_merged_across_checkpoints():
    fixture = development_fixture()
    rows = fixture["mixtures"][0]["arms"]["dense"]["rows"]
    pair_id = rows[0]["pair_id"]
    for row in rows:
        if row["pair_id"] == pair_id:
            row["checkpoint_sha256"] = _digest(
                "different-development-checkpoint"
            )

    with pytest.raises(ValueError, match="checkpoint|provenance"):
        evaluate_development_gates(fixture)


def test_gate_3_can_select_200k_as_low_without_code_change():
    fixture = development_fixture()
    fixture["loads"][0]["dense_reasoning_composite"] = 0.70
    fixture["loads"][1]["dense_fact_recall"] = 0.85

    receipt = evaluate_development_gates(fixture)["gate_3"]

    assert receipt["passed"]
    assert receipt["low_entities"] == 200_000
    assert receipt["high_entities"] == 800_000
    assert receipt["confirmation_entities"] == 1_800_000


def test_gate_3_fails_closed_without_distinct_supported_pair():
    fixture = development_fixture()
    fixture["loads"][1]["dense_fact_recall"] = 0.10
    fixture["loads"][2]["dense_reasoning_composite"] = 0.39

    receipt = evaluate_development_gates(fixture)["gate_3"]

    assert receipt["passed"] is False
    assert receipt["low_entities"] is None
    assert receipt["high_entities"] is None
    assert receipt["confirmation_entities"] is None


def test_gate_4_uses_20x_if_any_stability_interval_exceeds_one_point():
    fixture = development_fixture()
    fixture["budget"]["at_10x"]["paired_change_ci_to_20x"][
        "split.fact_recall"
    ] = {"ci_lo": -0.011, "ci_hi": 0.004}

    receipt = evaluate_development_gates(fixture)["gate_4"]

    assert receipt["passed"]
    assert receipt["tokens_per_parameter"] == 20


def test_gate_receipt_hash_drift_and_nonfinite_metrics_are_rejected():
    receipt = evaluate_development_gates(development_fixture())["gate_3"]
    changed = copy.deepcopy(receipt)
    changed["high_entities"] = 200_000
    with pytest.raises(ValueError, match="hash|decision"):
        validate_gate_receipt(changed, expected_gate=3)

    fixture = development_fixture()
    fixture["loads"][0]["dense_fact_recall"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_development_gates(fixture)


def test_gate_0_receipt_binds_the_real_smoke_report():
    smoke = {
        "bundle_byte_deterministic": True,
        "bundle_verified": True,
        "controls": [
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
        ],
        "corpus_builds": 2,
        "corpus_byte_deterministic": True,
        "corpus_sha256": _digest("corpus"),
        "dense_steps": 2,
        "eval_cells": 22,
        "extracted_bundle_verified": True,
        "matrix_runs": 35,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
        "resume_compared_next_update": True,
        "resume_exact": True,
        "schemas_validated": [
            "freeze-v1.schema.json",
            "relational-asset-receipt-v1.schema.json",
            "relational-result-v1.schema.json",
            "run-config-v1.schema.json",
            "run-manifest-v1.schema.json",
        ],
        "shared_stream": True,
        "sidecar_sha256": {
            label: _digest(f"{label}-sidecar")
            for label in ("dense", "random", "selective", "split")
        },
        "sidecars": ["dense", "random", "selective", "split"],
        "split_steps": 2,
        "synthetic_run_count": 35,
        "verdict_branches": [
            "validated",
            "practical_null",
            "inconclusive",
            "invalid",
        ],
    }

    receipt = build_gate_0_receipt(
        smoke,
        input_hashes=common_input_hashes(),
    )

    assert receipt["passed"]
    assert receipt["smoke_report_sha256"] == hashlib.sha256(
        canonical_json_bytes(smoke)
    ).hexdigest()
    validate_gate_receipt(receipt, expected_gate=0)


def test_gate_loader_rejects_protected_symlink_and_noncanonical_input(
    tmp_path,
):
    protected = tmp_path / "protected" / "development.json"
    protected.parent.mkdir()
    atomic_write_json(protected, development_fixture())
    with pytest.raises(ValueError, match="protected"):
        load_development_gate_inputs(protected)

    safe = tmp_path / "development.json"
    atomic_write_json(safe, development_fixture())
    link = tmp_path / "linked-development.json"
    link.symlink_to(safe)
    with pytest.raises(ValueError, match="symlink|canonical|regular"):
        load_development_gate_inputs(link)

    pretty = tmp_path / "pretty-development.json"
    pretty.write_text('{\n  "scope": "development"\n}\n')
    with pytest.raises(ValueError, match="canonical|fields"):
        load_development_gate_inputs(pretty)
