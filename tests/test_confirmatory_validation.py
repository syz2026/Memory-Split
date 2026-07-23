from __future__ import annotations

import inspect

import pytest

from evals.confirmatory.actions import (
    ACTION_SLOTS,
    MAX_READS,
    ActionOp,
    ActionSlot,
    validate_action_slots,
)
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
    store_content_sha256,
)
from evals.confirmatory.solver import (
    LookupChainSolver,
    verify_proof_and_answer,
    verify_sealed_gold,
)


def _read(relation: str, slot: int = 0) -> dict:
    return {
        "source_slot": slot,
        "relation_id": relation,
        "direction": "out",
        "op": "read",
    }


def _terminal(op: str = "noop") -> dict:
    return {
        "source_slot": None,
        "relation_id": None,
        "direction": None,
        "op": op,
    }


def _proof(*relations: str) -> list[dict]:
    active = [_read(relation) for relation in relations]
    return [
        *active,
        _terminal("halt"),
        *[_terminal() for _ in range(ACTION_SLOTS - len(active) - 1)],
    ]


def _records():
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
            "qualifiers": {},
        },
    ]
    content_sha256 = store_content_sha256("store-1", "world-1", rows)
    item = ItemRecord.from_dict(
        {
            "record_type": ITEM_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "item_id": "item-1",
            "pair_id": "pair-1",
            "twin": "original",
            "stratum": "iid",
            "world_id": "world-1",
            "task": "path_composition",
            "path_length": 2,
            "composition_split": "seen",
            "composition_id": "P1/P2",
            "prompt": "Follow the path.",
            "initial_slots": ["Q1", None, None, None],
            "store_id": "store-1",
            "memory_mode": "memory_on",
            "control": "correct",
        }
    )
    store = StoreRecord.from_dict(
        {
            "record_type": STORE_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "store_id": "store-1",
            "world_id": "world-1",
            "rows": rows,
            "content_sha256": content_sha256,
        }
    )
    gold = SealedGoldRecord.from_dict(
        {
            "record_type": SEALED_GOLD_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "item_id": "item-1",
            "pair_id": "pair-1",
            "twin": "original",
            "answer": "done",
            "proof": _proof("P1", "P2"),
            "solver_id": "lookup-chain-v1",
            "store_sha256": content_sha256,
        }
    )
    checkpoint = CheckpointRecord.from_dict(
        {
            "record_type": CHECKPOINT_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "checkpoint_sha256": "a" * 64,
            "model_id": "memorysplit-160m",
            "arm": "split",
            "seed": 1001,
            "raw_token_count": 1,
            "configuration_sha256": "b" * 64,
            "corpus_sha256": "c" * 64,
            "code_sha256": "d" * 64,
        }
    )
    return item, store, gold, checkpoint


def test_action_validation_is_twelve_slot_syntax_only():
    signature = inspect.signature(validate_action_slots)
    assert tuple(signature.parameters) == ("raw_actions",)

    actions = validate_action_slots(
        [_read("P_NOT_PRESENT_IN_ANY_STORE"), _terminal("halt")]
        + [_terminal() for _ in range(10)]
    )

    assert len(actions) == ACTION_SLOTS == 12
    assert actions[0].relation_id == "P_NOT_PRESENT_IN_ANY_STORE"
    assert actions[0].op is ActionOp.READ


def test_action_validation_accepts_ten_reads_without_gold_candidates():
    actions = validate_action_slots(
        [_read(f"P{index}") for index in range(MAX_READS)]
        + [_terminal(), _terminal()]
    )
    assert sum(action.op is ActionOp.READ for action in actions) == MAX_READS == 10


def test_typed_action_slots_cannot_bypass_the_syntax_contract():
    noop = ActionSlot(None, None, None, "noop")
    assert noop.op is ActionOp.NOOP

    with pytest.raises(ValueError, match="source_slot"):
        ActionSlot(4, "P1", "out", ActionOp.READ)
    with pytest.raises(ValueError, match="null"):
        ActionSlot(0, "P1", "out", ActionOp.HALT)


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (_proof("P1")[:-1], "12"),
        ([_read(f"P{x}") for x in range(11)] + [_terminal()], "10"),
        (
            [_terminal("halt"), _terminal("halt")]
            + [_terminal() for _ in range(10)],
            "HALT",
        ),
        (
            [_terminal("halt"), _read("P1")]
            + [_terminal() for _ in range(10)],
            "after HALT",
        ),
        (
            [
                {
                    "source_slot": 0,
                    "relation_id": "P1",
                    "direction": "out",
                    "op": "noop",
                },
                *[_terminal() for _ in range(11)],
            ],
            "null",
        ),
    ],
)
def test_action_validation_rejects_only_grammar_and_budget_violations(
    actions,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_action_slots(actions)


def test_lookup_solver_verifies_both_proof_and_answer():
    item, store, gold, _checkpoint = _records()
    solver = LookupChainSolver()

    result = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=_proof("P1", "P2"),
        answer="done",
        solver=solver,
    )
    assert result.valid
    assert result.proof_valid
    assert result.answer_valid
    assert result.derived_answer == "done"
    assert result.reason is None
    assert verify_sealed_gold(item, store, gold, solver).valid


def test_sealed_gold_rejects_action_budget_violations():
    _item, _store, gold, _checkpoint = _records()
    raw = gold.to_dict()
    raw["proof"] = [
        *[_read(f"P{index}") for index in range(MAX_READS + 1)],
        _terminal("halt"),
    ]

    with pytest.raises(ValueError, match="10"):
        SealedGoldRecord.from_dict(raw)


def test_solver_verification_fails_closed_on_wrong_answer_or_unresolved_proof():
    item, store, gold, _checkpoint = _records()
    solver = LookupChainSolver()

    wrong_answer = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=_proof("P1", "P2"),
        answer="wrong",
        solver=solver,
    )
    assert not wrong_answer.valid
    assert wrong_answer.proof_valid
    assert not wrong_answer.answer_valid

    missing = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=_proof("P404"),
        answer="done",
        solver=solver,
    )
    assert not missing.valid
    assert not missing.proof_valid
    assert not missing.answer_valid
    assert "MISS" in missing.reason

    self_consistent_but_wrong = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=_proof("P1"),
        answer="Q2",
        solver=solver,
    )
    assert not self_consistent_but_wrong.valid
    assert not self_consistent_but_wrong.proof_valid
    assert not self_consistent_but_wrong.answer_valid


def test_solver_verification_fails_closed_on_unexpected_solver_errors():
    item, store, gold, _checkpoint = _records()

    class ExplodingSolver:
        solver_id = "lookup-chain-v1"

        def solve(self, visible_item, visible_store, proof):
            raise RuntimeError("unexpected solver failure")

    result = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=_proof("P1", "P2"),
        answer="done",
        solver=ExplodingSolver(),
    )

    assert not result.valid
    assert not result.proof_valid
    assert not result.answer_valid
    assert result.derived_answer is None
    assert "unexpected solver failure" in result.reason


def test_solver_receives_model_visible_records_but_never_sealed_gold():
    item, store, gold, _checkpoint = _records()

    class RecordingSolver:
        solver_id = "lookup-chain-v1"

        def __init__(self):
            self.call = None

        def solve(self, visible_item, visible_store, proof):
            self.call = visible_item, visible_store, proof
            return "done"

    solver = RecordingSolver()
    result = verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=_proof("P_DOES_NOT_NEED_TO_EXIST"),
        answer="done",
        solver=solver,
    )

    assert result.valid
    assert solver.call[0] is item
    assert solver.call[1] is store
    assert len(solver.call) == 3
