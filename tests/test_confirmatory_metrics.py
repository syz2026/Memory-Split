from __future__ import annotations

from dataclasses import replace

import pytest

import evals.confirmatory as confirmatory
from evals.confirmatory import contracts as contracts_module
from evals.confirmatory.contracts import (
    Arm,
    Control,
    MemoryMode,
    Stratum,
    Twin,
    store_content_sha256,
)
from evals.confirmatory.metrics import (
    ItemOutcome,
    _ScoredItemOutcome,
    _aggregate_scored_pair_metric,
    _score_item_outcome,
)
from evals.confirmatory import metrics as metrics_module
from evals.confirmatory import status as status_module


def _pair(
    family: str,
    stratum: str,
    index: int,
    *,
    original: bool = True,
    counterfactual: bool = True,
) -> list[ItemOutcome]:
    common = {
        "pair_id": f"{family}-{stratum}-pair-{index}",
        "family": family,
        "stratum": stratum,
        "seed": 1001,
        "world_id": f"{family}-{stratum}-world-{index // 2}",
        "checkpoint_sha256": "a" * 64,
        "arm": Arm.SPLIT,
        "condition_id": "split90",
        "memory_mode": MemoryMode.MEMORY_ON,
        "control": Control.CORRECT,
    }
    proof = [
        {
            "source_slot": 0,
            "relation_id": "P1",
            "direction": "out",
            "op": "read",
        },
        {
            "source_slot": None,
            "relation_id": None,
            "direction": None,
            "op": "halt",
        },
        *[
            {
                "source_slot": None,
                "relation_id": None,
                "direction": None,
                "op": "noop",
            }
            for _ in range(10)
        ],
    ]
    return [
        ItemOutcome(
            item_id=f"{common['pair_id']}-original",
            twin=Twin.ORIGINAL,
            submitted_answer="done" if original else "wrong",
            submitted_proof=proof,
            **common,
        ),
        ItemOutcome(
            item_id=f"{common['pair_id']}-counterfactual",
            twin=Twin.COUNTERFACTUAL,
            submitted_answer="done" if counterfactual else "wrong",
            submitted_proof=proof,
            **common,
        ),
    ]


def _store_rows() -> list[dict]:
    return [
        {
            "source_id": "Q1",
            "relation_id": "P1",
            "direction": "out",
            "target_kind": "literal",
            "target": "done",
            "qualifiers": {},
        }
    ]


def _records(rows):
    stratum_shape = {
        "iid": (2, "seen"),
        "composition_ood": (2, "heldout"),
        "length_ood": (7, "seen"),
        "joint_ood": (7, "heldout"),
    }
    items = {}
    checkpoints = {}
    gold_records = {}
    stores = {}
    for row in rows:
        path_length, composition_split = stratum_shape[row.stratum.value]
        item = contracts_module.ItemRecord.from_dict(
            {
                "record_type": contracts_module.ITEM_SCHEMA,
                "schema_version": contracts_module.CONTRACT_VERSION,
                "item_id": row.item_id,
                "pair_id": row.pair_id,
                "twin": row.twin.value,
                "stratum": row.stratum.value,
                "family": row.family.value,
                "world_id": row.world_id,
                "task": "fixture_task",
                "path_length": path_length,
                "composition_split": composition_split,
                "composition_id": "fixture-composition",
                "prompt": "fixture prompt",
                "initial_slots": ["Q1", None, None, None],
                "store_id": f"store-{row.world_id}",
                "memory_mode": row.memory_mode.value,
                "control": row.control.value,
            }
        )
        items[row.item_id] = item
        store_rows = _store_rows()
        store_sha256 = store_content_sha256(
            item.store_id,
            item.world_id,
            store_rows,
        )
        stores[item.store_id] = contracts_module.StoreRecord.from_dict(
            {
                "record_type": contracts_module.STORE_SCHEMA,
                "schema_version": contracts_module.CONTRACT_VERSION,
                "store_id": item.store_id,
                "world_id": item.world_id,
                "rows": store_rows,
                "content_sha256": store_sha256,
            }
        )
        gold_records[row.item_id] = (
            contracts_module.SealedGoldRecord.from_dict(
                {
                    "record_type": contracts_module.SEALED_GOLD_SCHEMA,
                    "schema_version": contracts_module.CONTRACT_VERSION,
                    "item_id": row.item_id,
                    "pair_id": row.pair_id,
                    "twin": row.twin.value,
                    "answer": "done",
                    "proof": [
                        action.to_dict() for action in row.submitted_proof
                    ],
                    "solver_id": "lookup-chain-v1",
                    "store_sha256": store_sha256,
                }
            )
        )
        checkpoints[row.checkpoint_sha256] = (
            contracts_module.CheckpointRecord.from_dict(
                {
                    "record_type": contracts_module.CHECKPOINT_SCHEMA,
                    "schema_version": contracts_module.CONTRACT_VERSION,
                    "checkpoint_sha256": row.checkpoint_sha256,
                    "model_id": "fixture-model",
                    "arm": row.arm.value,
                    "condition_id": row.condition_id.value,
                    "seed": row.seed,
                    "raw_token_count": 1,
                    "configuration_sha256": "b" * 64,
                    "route_dose_sha256": "e" * 64,
                    "corpus_sha256": "c" * 64,
                    "code_sha256": "d" * 64,
                }
            )
        )
    return items, checkpoints, gold_records, stores


def _metric(rows):
    items, checkpoints, gold_records, stores = _records(rows)
    scored = [
        _score_item_outcome(
            outcome=row,
            item=items[row.item_id],
            checkpoint=checkpoints[row.checkpoint_sha256],
            gold=gold_records[row.item_id],
            store=stores[items[row.item_id].store_id],
        )
        for row in rows
    ]
    return _aggregate_scored_pair_metric(
        scored,
        items=items,
        checkpoints=checkpoints,
    )


def test_primary_metric_equal_weights_only_four_family_ood_cells():
    rows = []
    rows.extend(_pair("graph", "composition_ood", 0))
    rows.extend(_pair("graph", "joint_ood", 0))
    rows.extend(_pair("graph", "joint_ood", 1, counterfactual=False))
    for index in range(4):
        rows.extend(
            _pair(
                "non_path",
                "composition_ood",
                index,
                counterfactual=False,
            )
        )
    for index in range(8):
        rows.extend(_pair("non_path", "joint_ood", index))
    for index in range(10):
        rows.extend(_pair("graph", "iid", index, counterfactual=False))
        rows.extend(_pair("non_path", "length_ood", index))

    result = _metric(rows)

    assert result.primary_accuracy == pytest.approx(0.625)
    assert {
        name: rate.value for name, rate in result.primary_cells.items()
    } == {
        "graph__composition_ood": 1.0,
        "graph__joint_ood": 0.5,
        "non_path__composition_ood": 0.0,
        "non_path__joint_ood": 1.0,
    }
    assert result.overall_pair_accuracy.numerator == 20
    assert result.overall_pair_accuracy.denominator == 35
    assert result.by_stratum[Stratum.IID].value == 0.0
    assert result.by_stratum[Stratum.LENGTH_OOD].value == 1.0
    assert result.by_stratum[Stratum.COMPOSITION_OOD].value == pytest.approx(
        0.2
    )
    assert result.by_stratum[Stratum.JOINT_OOD].value == pytest.approx(0.9)
    assert result.arm is Arm.SPLIT
    assert result.memory_mode is MemoryMode.MEMORY_ON
    assert result.control is Control.CORRECT


def test_caller_cannot_forge_a_correct_score_for_a_wrong_submission():
    wrong_submission = replace(
        _pair("graph", "composition_ood", 0)[0],
        submitted_answer="wrong",
    )

    with pytest.raises(TypeError, match="internal|solver"):
        _ScoredItemOutcome(
            submission=wrong_submission,
            proof_valid=True,
            answer_valid=True,
        )
    assert not hasattr(confirmatory, "balanced_counterfactual_pair_metric")
    assert not hasattr(confirmatory, "counterfactual_pair_metric")
    assert not hasattr(confirmatory, "score_item_outcome")
    assert not hasattr(metrics_module, "balanced_counterfactual_pair_metric")
    assert not hasattr(metrics_module, "counterfactual_pair_metric")
    assert not hasattr(metrics_module, "score_item_outcome")

    wrong_rows = [
        row
        for family in ("graph", "non_path")
        for stratum in ("composition_ood", "joint_ood")
        for row in _pair(
            family,
            stratum,
            0,
            original=False,
            counterfactual=False,
        )
    ]
    assert _metric(wrong_rows).primary_accuracy == 0.0


def test_pair_metric_rejects_missing_duplicate_crossed_or_unscored_twins():
    complete = [
        row
        for family in ("graph", "non_path")
        for stratum in ("composition_ood", "joint_ood")
        for row in _pair(family, stratum, 0)
    ]

    with pytest.raises(ValueError, match="both twins"):
        _metric(complete[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        _metric([*complete, complete[0]])

    crossed = list(complete)
    crossed[1] = replace(crossed[1], family="non_path")
    with pytest.raises(ValueError, match="metadata"):
        _metric(crossed)

    with pytest.raises(TypeError, match="solver-scored"):
        _aggregate_scored_pair_metric(
            complete,
            items=_records(complete)[0],
            checkpoints=_records(complete)[1],
        )


def test_pair_metric_requires_four_primary_cells_but_not_iid_or_length():
    rows = [
        row
        for family in ("graph", "non_path")
        for stratum in ("composition_ood", "joint_ood")
        for row in _pair(family, stratum, 0)
    ]
    result = _metric(rows)
    assert result.primary_accuracy == 1.0
    assert set(result.by_stratum) == {
        Stratum.COMPOSITION_OOD,
        Stratum.JOINT_OOD,
    }

    missing = [
        row
        for row in rows
        if not (
            row.family == "non_path"
            and row.stratum == "joint_ood"
        )
    ]
    with pytest.raises(ValueError, match="primary"):
        _metric(missing)

    rows[-1] = replace(rows[-1], memory_mode=MemoryMode.MEMORY_OFF)
    with pytest.raises(ValueError, match="cell"):
        _metric(rows)


def test_outcome_binding_authenticates_item_and_checkpoint_identity():
    outcome = _pair("graph", "composition_ood", 0)[0]
    items, checkpoints, _, _ = _records([outcome])
    item = items[outcome.item_id]
    checkpoint = checkpoints[outcome.checkpoint_sha256]

    assert (
        metrics_module.validate_item_outcome_binding(
            outcome=outcome,
            item=item,
            checkpoint=checkpoint,
        )
        is outcome
    )

    for changes in (
        {"pair_id": "crossed-pair"},
        {"twin": Twin.COUNTERFACTUAL},
        {"world_id": "crossed-world"},
        {"family": "non_path"},
        {"stratum": "joint_ood"},
        {"memory_mode": MemoryMode.MEMORY_OFF},
        {"control": Control.NO_QUERY},
    ):
        with pytest.raises(ValueError, match="item binding"):
            metrics_module.validate_item_outcome_binding(
                outcome=replace(outcome, **changes),
                item=item,
                checkpoint=checkpoint,
            )

    seed_two_dense = replace(
        outcome,
        seed=1002,
        arm=Arm.DENSE,
        condition_id="dense",
    )
    with pytest.raises(ValueError, match="checkpoint binding"):
        metrics_module.validate_item_outcome_binding(
            outcome=seed_two_dense,
            item=item,
            checkpoint=checkpoint,
        )


def test_metric_validation_rejects_cross_checkpoint_attribution():
    rows = [
        row
        for family in ("graph", "non_path")
        for stratum in ("composition_ood", "joint_ood")
        for row in _pair(family, stratum, 0)
    ]
    items, checkpoints, gold_records, stores = _records(rows)
    rows[0] = replace(
        rows[0],
        seed=1002,
        arm=Arm.DENSE,
        condition_id="dense",
    )

    with pytest.raises(ValueError, match="checkpoint binding"):
        _score_item_outcome(
            outcome=rows[0],
            item=items[rows[0].item_id],
            checkpoint=checkpoints[rows[0].checkpoint_sha256],
            gold=gold_records[rows[0].item_id],
            store=stores[items[rows[0].item_id].store_id],
        )


def test_families_strata_memory_modes_and_controls_are_closed_enums():
    assert {
        family.value for family in contracts_module.ReasoningFamily
    } == {"graph", "non_path"}
    assert {stratum.value for stratum in Stratum} == {
        "iid",
        "composition_ood",
        "length_ood",
        "joint_ood",
    }
    assert {mode.value for mode in MemoryMode} == {"memory_off", "memory_on"}
    assert {arm.value for arm in Arm} == {"dense", "split", "random"}
    assert {control.value for control in Control} == {
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
        "page_order_permutation",
    }
    with pytest.raises(ValueError, match="family"):
        _pair("unknown", "composition_ood", 0)


def _status(**changes):
    values = {
        "complete": True,
        "valid": True,
        "observed_seeds": 5,
        "required_seeds": 5,
        "sign_consistent": True,
        "supports_effect": False,
        "supports_practical_null": False,
    }
    values.update(changes)
    return status_module.classify_status(**values)


@pytest.mark.parametrize("observed_seeds", [1, 3])
def test_invalid_status_suppresses_all_interim_evidence(observed_seeds):
    result = _status(
        complete=False,
        valid=False,
        observed_seeds=observed_seeds,
        sign_consistent=True,
    )

    assert result.scientific_status == "invalid"
    assert result.interim_evidence_label == "none"
    assert result.final_inference_conclusion == "not_evaluated"


def test_interim_sign_consistency_requires_three_preterminal_pairs():
    two = _status(complete=False, observed_seeds=2, sign_consistent=True)
    three = _status(complete=False, observed_seeds=3, sign_consistent=True)

    assert two.scientific_status == three.scientific_status == "incomplete"
    assert two.interim_evidence_label == "directional_only"
    assert three.interim_evidence_label == "sign_consistent_only"
    assert (
        two.final_inference_conclusion
        == three.final_inference_conclusion
        == "not_evaluated"
    )


@pytest.mark.parametrize(
    ("changes", "scientific", "interim", "conclusion"),
    [
        (
            {"observed_seeds": 0, "complete": False},
            "incomplete",
            "none",
            "not_evaluated",
        ),
        (
            {"observed_seeds": 1, "complete": False},
            "incomplete",
            "directional_only",
            "not_evaluated",
        ),
        (
            {"sign_consistent": False},
            "complete",
            "none",
            "inconclusive",
        ),
        (
            {"supports_effect": True},
            "complete",
            "none",
            "supports_effect",
        ),
        (
            {"supports_practical_null": True},
            "complete",
            "none",
            "supports_practical_null",
        ),
    ],
)
def test_status_classification_returns_three_orthogonal_axes(
    changes,
    scientific,
    interim,
    conclusion,
):
    result = _status(**changes)

    assert result.scientific_status == scientific
    assert result.interim_evidence_label == interim
    assert result.final_inference_conclusion == conclusion
