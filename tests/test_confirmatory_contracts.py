from __future__ import annotations

from dataclasses import replace
import json

import pytest

import evals.confirmatory as confirmatory
from evals.confirmatory import contracts as contracts_module
from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    SEALED_GOLD_SCHEMA,
    STORE_SCHEMA,
    CheckpointRecord,
    ItemRecord,
    SealedGoldRecord,
    StoreRecord,
    canonical_json_bytes,
    store_content_sha256,
    validate_contract_bundle,
)


def test_confirmatory_package_exports_the_supported_core_api():
    assert confirmatory.CONTRACT_VERSION == 2
    assert confirmatory.ItemRecord is ItemRecord
    assert confirmatory.StoreRecord is StoreRecord
    assert confirmatory.CheckpointRecord is CheckpointRecord
    assert {
        condition.value for condition in confirmatory.ConditionId
    } == {"dense", "split90", "random"}
    assert confirmatory.ReasoningFamily is contracts_module.ReasoningFamily
    assert (
        confirmatory.OUTCOME_SCHEMA
        == "memorysplit.confirmatory.outcome.v2"
    )
    assert (
        confirmatory.METRICS_SCHEMA
        == "memorysplit.confirmatory.metrics.v2"
    )
    assert "RNG seed" in confirmatory.PRACTICAL_NULL_REPLAY_GAP
    assert confirmatory.STUDY_LOCK_SCHEMA.endswith("study-lock.v2")
    assert confirmatory.FROZEN_PREREGISTRATION_SHA256 == (
        "fee38e363298d3def46b741320c9d7df4523d0ff3cd249187cf52d54046cbbf0"
    )
    assert callable(confirmatory.score_item_outcome)
    assert callable(confirmatory.balanced_counterfactual_pair_metric)
    assert callable(confirmatory.hierarchical_paired_bootstrap)
    assert callable(confirmatory.verify_proof_and_answer)
    assert callable(confirmatory.build_artifact_report)


def _noop() -> dict:
    return {
        "source_slot": None,
        "relation_id": None,
        "direction": None,
        "op": "noop",
    }


def _proof() -> list[dict]:
    return [
        {
            "source_slot": 0,
            "relation_id": "P1",
            "direction": "out",
            "op": "read",
        },
        {
            "source_slot": 0,
            "relation_id": "P2",
            "direction": "out",
            "op": "read",
        },
        {
            "source_slot": None,
            "relation_id": None,
            "direction": None,
            "op": "halt",
        },
        *[_noop() for _ in range(9)],
    ]


def _item(**changes) -> dict:
    value = {
        "record_type": ITEM_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "item_id": "item-1-original",
        "pair_id": "pair-1",
        "twin": "original",
        "stratum": "iid",
        "family": "graph",
        "world_id": "world-1",
        "task": "path_composition",
        "path_length": 2,
        "composition_split": "seen",
        "composition_id": "P1/P2",
        "prompt": "Start at Q1 and follow P1 then P2.",
        "initial_slots": ["Q1", None, None, None],
        "store_id": "store-1",
        "memory_mode": "memory_on",
        "control": "correct",
    }
    value.update(changes)
    return value


def _store(**changes) -> dict:
    rows = [
        {
            "source_id": "Q1",
            "relation_id": "P1",
            "direction": "out",
            "target_kind": "entity",
            "target": "Q2",
            "qualifiers": {},
        },
        {
            "source_id": "Q2",
            "relation_id": "P2",
            "direction": "out",
            "target_kind": "literal",
            "target": "done",
            "qualifiers": {"compose": "1"},
        },
    ]
    value = {
        "record_type": STORE_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "store_id": "store-1",
        "world_id": "world-1",
        "rows": rows,
        "content_sha256": store_content_sha256("store-1", "world-1", rows),
    }
    value.update(changes)
    return value


def _gold(store_sha256: str, **changes) -> dict:
    value = {
        "record_type": SEALED_GOLD_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "item_id": "item-1-original",
        "pair_id": "pair-1",
        "twin": "original",
        "answer": "done",
        "proof": _proof(),
        "solver_id": "lookup-chain-v1",
        "store_sha256": store_sha256,
    }
    value.update(changes)
    return value


def _checkpoint(**changes) -> dict:
    value = {
        "record_type": CHECKPOINT_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "checkpoint_sha256": "a" * 64,
        "model_id": "memorysplit-160m",
        "arm": "split",
        "condition_id": "split90",
        "seed": 1001,
        "raw_token_count": 3_244_818_432,
        "configuration_sha256": "b" * 64,
        "route_dose_sha256": "e" * 64,
        "corpus_sha256": "c" * 64,
        "code_sha256": "d" * 64,
    }
    value.update(changes)
    return value


def test_versioned_contracts_round_trip_and_keep_gold_sealed():
    item = ItemRecord.from_dict(_item())
    store = StoreRecord.from_dict(_store())
    gold = SealedGoldRecord.from_dict(_gold(store.content_sha256))
    checkpoint = CheckpointRecord.from_dict(_checkpoint())

    assert item.to_dict() == _item()
    assert store.to_dict() == _store()
    assert gold.to_dict() == _gold(store.content_sha256)
    assert checkpoint.to_dict() == _checkpoint()
    assert "answer" not in item.to_dict()
    assert "proof" not in item.to_dict()
    assert "gold" not in canonical_json_bytes(item).decode()
    assert json.loads(canonical_json_bytes(gold))["answer"] == "done"

    bundle = validate_contract_bundle(item, gold, store, checkpoint)
    assert bundle.item is item
    assert bundle.gold is gold
    assert bundle.store is store
    assert bundle.checkpoint is checkpoint


@pytest.mark.parametrize(
    ("factory", "value", "message"),
    [
        (ItemRecord.from_dict, {**_item(), "answer": "leak"}, "unknown"),
        (
            ItemRecord.from_dict,
            {key: value for key, value in _item().items() if key != "prompt"},
            "missing",
        ),
        (
            ItemRecord.from_dict,
            _item(schema_version=1),
            "schema_version",
        ),
        (
            StoreRecord.from_dict,
            _store(content_sha256="0" * 64),
            "content_sha256",
        ),
        (
            CheckpointRecord.from_dict,
            _checkpoint(seed=True),
            "seed",
        ),
    ],
)
def test_contracts_reject_unknown_missing_old_mistyped_or_drifted_fields(
    factory,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        factory(value)


@pytest.mark.parametrize("schema_version", [True, 2.0])
@pytest.mark.parametrize(
    ("factory", "record"),
    [
        (ItemRecord.from_dict, _item()),
        (StoreRecord.from_dict, _store()),
        (
            SealedGoldRecord.from_dict,
            _gold(_store()["content_sha256"]),
        ),
        (CheckpointRecord.from_dict, _checkpoint()),
    ],
)
def test_contract_schema_versions_are_exact_integers(
    factory,
    record,
    schema_version,
):
    with pytest.raises(ValueError, match="schema_version"):
        factory({**record, "schema_version": schema_version})


def test_checkpoint_requires_explicit_condition_and_route_dose_identity():
    checkpoint = CheckpointRecord.from_dict(_checkpoint())

    assert checkpoint.condition_id.value == "split90"
    assert checkpoint.route_dose_sha256 == "e" * 64

    generic_split = _checkpoint(condition_id="split")
    with pytest.raises(ValueError, match="condition"):
        CheckpointRecord.from_dict(generic_split)
    missing_condition = _checkpoint()
    missing_condition.pop("condition_id")
    with pytest.raises(ValueError, match="condition"):
        CheckpointRecord.from_dict(missing_condition)
    with pytest.raises(ValueError, match="condition|arm"):
        CheckpointRecord.from_dict(
            _checkpoint(arm="dense", condition_id="split90")
        )

    random_checkpoint = CheckpointRecord.from_dict(
        _checkpoint(arm="random", condition_id="random")
    )
    assert random_checkpoint.arm.value == "random"


@pytest.mark.parametrize(
    ("stratum", "path_length", "composition_split"),
    [
        ("iid", 2, "heldout"),
        ("composition_ood", 7, "heldout"),
        ("length_ood", 6, "seen"),
        ("joint_ood", 10, "seen"),
    ],
)
def test_item_contract_enforces_the_four_frozen_strata(
    stratum,
    path_length,
    composition_split,
):
    with pytest.raises(ValueError, match="stratum"):
        ItemRecord.from_dict(
            _item(
                stratum=stratum,
                path_length=path_length,
                composition_split=composition_split,
            )
        )


def test_item_contract_requires_a_strict_reasoning_family():
    assert {
        family.value for family in contracts_module.ReasoningFamily
    } == {"graph", "non_path"}

    with pytest.raises(ValueError, match="family"):
        ItemRecord.from_dict(_item(family="path"))


def test_contract_bundle_rejects_crossed_gold_store_and_world_bindings():
    item = ItemRecord.from_dict(_item())
    store = StoreRecord.from_dict(_store())
    gold = SealedGoldRecord.from_dict(_gold(store.content_sha256))
    checkpoint = CheckpointRecord.from_dict(_checkpoint())

    crossed_gold = replace(gold, item_id="different-item")
    with pytest.raises(ValueError, match="item"):
        validate_contract_bundle(item, crossed_gold, store, checkpoint)

    crossed_store = StoreRecord.from_dict(
        _store(store_id="store-2", content_sha256=store_content_sha256(
            "store-2", "world-1", _store()["rows"]
        ))
    )
    with pytest.raises(ValueError, match="store"):
        validate_contract_bundle(item, gold, crossed_store, checkpoint)

    crossed_world = ItemRecord.from_dict(_item(world_id="world-2"))
    with pytest.raises(ValueError, match="world"):
        validate_contract_bundle(crossed_world, gold, store, checkpoint)


def test_nested_store_and_gold_values_are_immutable():
    store = StoreRecord.from_dict(_store())
    gold = SealedGoldRecord.from_dict(_gold(store.content_sha256))

    with pytest.raises(TypeError):
        store.rows[1].qualifiers["compose"] = "9"
    with pytest.raises((AttributeError, TypeError)):
        gold.proof[0].relation_id = "P9"
