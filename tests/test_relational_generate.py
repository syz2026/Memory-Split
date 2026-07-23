from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.records import QAItem
from evals.relational_generate import (
    GraphDecodeState,
    OverlayStore,
    apply_action,
    decode_item,
    decode_items,
    parse_action,
)
from organizer.graph_store import AtomicGraphStore
from scripts.run_relational_evals import (
    _states_to_rows,
    store_for_item,
)
from train.tokenizer import get_tok


def _item(suffix: str = "a") -> QAItem:
    return QAItem(
        qid=f"pair-{suffix}-o",
        task="balanced_equality",
        prompt=(
            f"Slot 0 refers to entity-{suffix}. "
            f"Slot 1 refers to entity-{suffix}. Are they equal?"
        ),
        answer="<|slot_0|>",
        meta={
            "pair_id": f"pair-{suffix}",
            "variant": "original",
            "entity_slots": [7, 8, None, None],
            "gold_addresses": [[7, "r0", "out"]],
            "gold_fact_ids": ["fact-7"],
            "gold_actions": [
                {
                    "source_slot": 0,
                    "relation_id": "r0",
                    "direction": "out",
                    "read": True,
                    "halt": False,
                },
                {
                    "source_slot": 0,
                    "relation_id": "r0",
                    "direction": "out",
                    "read": False,
                    "halt": True,
                },
                *[
                    {
                        "source_slot": 0,
                        "relation_id": "r0",
                        "direction": "out",
                        "read": False,
                        "halt": False,
                    }
                    for _ in range(4)
                ],
            ],
            "answer_choices": ["<|slot_0|>", "<|slot_1|>"],
        },
    )


def _store() -> AtomicGraphStore:
    return AtomicGraphStore(
        [GraphRow(7, "r0", "out", "entity", "9", (), "world")]
    )


@dataclass
class _HistoryCache:
    histories: list[list[int]]

    def select_batch(self, index: int) -> "_HistoryCache":
        return _HistoryCache([list(self.histories[index])])


class ScriptedModel:
    """Select READ, then HALT; choose slot 0 for provisional answers."""

    device = torch.device("cpu")

    def __init__(self, tok):
        self.tok = tok
        self.prefill_shapes: list[tuple[int, int]] = []
        self.step_batch_sizes: list[int] = []

    def _next(self, history: list[int]) -> int:
        tok = self.tok
        last = history[-1]
        if last == tok.GRAPH_START:
            return tok.SLOTS[0]
        if last in tok.SLOTS:
            return tok.RELATIONS["r0"]
        if last in tok.RELATIONS.values():
            return tok.DIR_OUT
        if last in (tok.DIR_OUT, tok.DIR_IN):
            return tok.GRAPH_HALT if tok.GRAPH_READ in history else tok.GRAPH_READ
        if last == tok.ANSWER_STATE:
            return tok.SLOTS[0]
        return tok.GRAPH_START

    def forward_step(self, idx: torch.Tensor, cache: _HistoryCache | None):
        if cache is None:
            self.prefill_shapes.append(tuple(idx.shape))
            histories = [row.tolist() for row in idx]
        else:
            self.step_batch_sizes.append(idx.shape[0])
            histories = [
                history + row.tolist()
                for history, row in zip(cache.histories, idx)
            ]
        logits = torch.full(
            (idx.shape[0], idx.shape[1], self.tok.VOCAB_SIZE),
            -1e9,
            dtype=torch.float32,
        )
        for row, history in enumerate(histories):
            logits[row, -1, self._next(history)] = 0.0
        return logits, _HistoryCache(histories)


def test_parse_fixed_action_tokens():
    tok = get_tok()
    ids = [
        tok.GRAPH_START,
        tok.SLOTS[1],
        tok.RELATIONS["r2"],
        tok.DIR_OUT,
        tok.GRAPH_READ,
        tok.GRAPH_END,
    ]
    assert parse_action(ids, tok) == GraphAction(1, "r2", "out", True, False)


def test_entity_and_literal_slot_updates():
    entity = GraphRow(7, "r0", "out", "entity", "9", (), "w")
    literal = GraphRow(9, "r4", "out", "literal", "1950-01-01", (), "w")
    state = GraphDecodeState([7, None, None, None])
    store = AtomicGraphStore([entity, literal])

    assert apply_action(
        state, GraphAction(0, "r0", "out", True, False), store
    ) == entity
    assert state.slots[0] == 9
    assert apply_action(
        state, GraphAction(0, "r4", "out", True, False), store
    ) == literal
    assert state.slots[0] == 9


def test_memory_off_is_miss_and_halt_keeps_six_slots():
    tok = get_tok()
    result = decode_item(ScriptedModel(tok), tok, _item(), store=None)

    assert result.misses == 1
    assert len(result.actions) == 6
    assert result.actions[1].halt
    assert all(
        not action.read and not action.halt for action in result.actions[2:]
    )
    assert len(result.provisional_answers) == 6
    assert not hasattr(result, "malformed")
    assert not hasattr(result, "excess_reads")


def test_counterfactual_overlay_changes_only_one_row():
    base = _store()
    replacement = GraphRow(7, "r0", "out", "entity", "10", (), "world")
    overlay = OverlayStore(base, replacement)

    assert overlay.lookup(replacement.address) == replacement
    assert overlay.rows() == (replacement,)
    assert base.lookup(replacement.address).target == "9"

    absent = GraphRow(99, "r0", "out", "entity", "10", (), "world")
    with pytest.raises(ValueError, match="existing base address"):
        OverlayStore(base, absent)


def test_equal_length_batch_matches_single_decode():
    tok = get_tok()
    items = [_item("a"), _item("b")]
    assert len({len(tok.encode(item.prompt)) for item in items}) == 1
    singles = [
        decode_item(ScriptedModel(tok), tok, item, _store()) for item in items
    ]
    model = ScriptedModel(tok)

    batched = decode_items(model, tok, items, _store(), batch_size=2)

    assert batched == singles
    assert model.prefill_shapes == [(2, len(tok.encode(items[0].prompt)))]
    assert set(model.step_batch_sizes) == {2}


def test_unequal_prompt_lengths_are_never_padded_together():
    tok = get_tok()
    short = _item("a")
    long = _item("a much longer entity label")
    assert len(tok.encode(short.prompt)) != len(tok.encode(long.prompt))
    model = ScriptedModel(tok)

    decode_items(model, tok, [short, long], _store(), batch_size=2)

    assert [shape[0] for shape in model.prefill_shapes] == [1, 1]


def test_variable_return_lengths_split_caches_without_changing_results():
    tok = get_tok()
    items = [_item("a"), _item("b")]
    stores = {
        items[0].qid: _store(),
        items[1].qid: AtomicGraphStore(
            [
                GraphRow(
                    7,
                    "r0",
                    "out",
                    "entity",
                    "12345678901234567890",
                    (),
                    "world",
                )
            ]
        ),
    }
    singles = [
        decode_item(
            ScriptedModel(tok),
            tok,
            item,
            stores[item.qid],
        )
        for item in items
    ]
    model = ScriptedModel(tok)

    batched = decode_items(
        model,
        tok,
        items,
        lambda item: stores[item.qid],
        batch_size=2,
    )

    assert batched == singles
    assert 2 in model.step_batch_sizes
    assert 1 in model.step_batch_sizes


def test_read_uses_the_model_selected_relation_and_direction():
    state = GraphDecodeState([7, None, None, None])
    store = AtomicGraphStore(
        [GraphRow(7, "r3", "in", "entity", "11", (), "world")]
    )
    action = GraphAction(0, "r3", "in", read=True, halt=False)

    row = apply_action(state, action, store)

    assert row is not None
    assert row.address == GraphAddress(7, "r3", "in")
    assert state.slots[0] == 11


def test_state_rows_keep_six_steps_but_score_only_the_read_path():
    item = _item()
    returned = _store().rows()[0]
    state = GraphDecodeState(
        slots=[9, 8, None, None],
        actions=[
            GraphAction(0, "r0", "out", True, False),
            GraphAction(0, "r0", "out", False, True),
            *[
                GraphAction(0, "r0", "out", False, False)
                for _ in range(4)
            ],
        ],
        rows=[returned, None, None, None, None, None],
        provisional_answers=["<|slot_0|>"] * 6,
        halt_step=2,
    )

    rows = _states_to_rows([item], [state])

    assert rows[0]["n_steps"] == 6
    assert rows[0]["actions"] == [[0, "r0", "out", True, False]]
    assert len(rows[0]["all_actions"]) == 6
    assert rows[0]["correct_referents"] == [True]
    assert rows[0]["correct"]


def test_state_rows_consume_explicit_multihop_gold_slots():
    item = _item()
    item.task = "path_composition"
    item.meta["gold_addresses"] = [
        [7, "r0", "out"],
        [9, "r1", "out"],
    ]
    item.meta["gold_actions"] = [
        {
            "source_slot": 3,
            "relation_id": "r0",
            "direction": "out",
            "read": True,
            "halt": False,
        },
        {
            "source_slot": 2,
            "relation_id": "r1",
            "direction": "out",
            "read": True,
            "halt": False,
        },
        {
            "source_slot": 1,
            "relation_id": "r7",
            "direction": "in",
            "read": False,
            "halt": True,
        },
        *[
            {
                "source_slot": 1,
                "relation_id": "r7",
                "direction": "in",
                "read": False,
                "halt": False,
            }
            for _ in range(3)
        ],
    ]
    returned = [
        GraphRow(7, "r0", "out", "entity", "9", (), "world"),
        GraphRow(9, "r1", "out", "entity", "10", (), "world"),
    ]
    actions = [
        GraphAction(3, "r0", "out", True, False),
        GraphAction(2, "r1", "out", True, False),
        GraphAction(1, "r7", "in", False, True),
        *[GraphAction(1, "r7", "in", False, False) for _ in range(3)],
    ]
    state = GraphDecodeState(
        slots=[7, 8, 10, 9],
        actions=actions,
        rows=[*returned, None, None, None, None],
        provisional_answers=["<|slot_0|>"] * 6,
        halt_step=3,
    )

    row = _states_to_rows([item], [state])[0]

    assert row["gold_actions"] == [
        [3, "r0", "out", True, False],
        [2, "r1", "out", True, False],
    ]
    assert row["gold_all_actions"][2] == [
        1,
        "r7",
        "in",
        False,
        True,
    ]


def test_store_toggle_only_changes_the_store_view():
    item = _item()
    base = _store()
    changed = GraphRow(7, "r0", "out", "entity", "10", (), "world")
    item.meta["variant"] = "counterfactual"
    item.meta["changed_row"] = changed.as_json()

    assert store_for_item(base, item, memory_on=False) is None
    enabled = store_for_item(base, item, memory_on=True)
    assert isinstance(enabled, OverlayStore)
    assert enabled.lookup(changed.address) == changed


def test_relational_eval_command_is_repo_relative():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/run_relational_evals.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--run" in completed.stdout
    assert "--guardrails-json" not in completed.stdout


def test_read_uses_exact_qid_pid_direction_and_page_address():
    page_zero = GraphRow(
        "Q1", "P31", "out", "entity", "Q2", (), "page-zero", page=0
    )
    page_one = GraphRow(
        "Q1", "P31", "out", "entity", "Q3", (), "page-one", page=1
    )
    state = GraphDecodeState(["Q1", None, None, None])
    store = AtomicGraphStore([page_zero, page_one])

    returned = apply_action(
        state,
        GraphAction(0, "P31", "out", True, False, page=1),
        store,
    )

    assert returned == page_one
    assert state.slots[0] == "Q3"
    assert state.rows == [page_one]


def test_current_items_decode_twelve_slots_with_post_halt_noops():
    tok = get_tok()
    item = _item()
    item.meta["action_slots"] = 12

    result = decode_item(ScriptedModel(tok), tok, item, store=None)

    assert len(result.actions) == 12
    assert len(result.rows) == 12
    assert len(result.provisional_answers) == 12
    assert result.actions[1].halt
    assert all(
        action == GraphAction(0, "r0", "out", False, False)
        for action in result.actions[2:]
    )


def test_graph_state_rejects_more_than_ten_reads():
    state = GraphDecodeState([7, None, None, None])
    store = _store()
    action = GraphAction(0, "r0", "out", True, False)

    for _ in range(10):
        state.slots[0] = 7
        apply_action(state, action, store)

    state.slots[0] = 7
    with pytest.raises(ValueError, match="at most ten graph reads"):
        apply_action(state, action, store)


def test_decoders_constrain_the_model_to_at_most_ten_reads():
    tok = get_tok()

    class AlwaysReadModel(ScriptedModel):
        def _next(self, history):
            if history[-1] in (tok.DIR_OUT, tok.DIR_IN):
                return tok.GRAPH_READ
            return super()._next(history)

    items = [_item("a"), _item("b")]
    for item in items:
        item.meta["action_slots"] = 12

    single = decode_item(AlwaysReadModel(tok), tok, items[0], store=None)
    batched = decode_items(
        AlwaysReadModel(tok),
        tok,
        items,
        store=None,
        batch_size=2,
    )

    assert sum(action.read for action in single.actions) == 10
    assert all(
        sum(action.read for action in state.actions) == 10 for state in batched
    )


def test_state_rows_accept_current_twelve_slot_traces():
    item = _item()
    item.meta["action_slots"] = 12
    item.meta["gold_actions"].extend(
        {
            "source_slot": 0,
            "relation_id": "r0",
            "direction": "out",
            "read": False,
            "halt": False,
        }
        for _ in range(6)
    )
    returned = _store().rows()[0]
    state = GraphDecodeState(
        slots=[9, 8, None, None],
        actions=[
            GraphAction(0, "r0", "out", True, False),
            GraphAction(0, "r0", "out", False, True),
            *[
                GraphAction(0, "r0", "out", False, False)
                for _ in range(10)
            ],
        ],
        rows=[returned, *([None] * 11)],
        provisional_answers=["<|slot_0|>"] * 12,
        halt_step=2,
    )

    result = _states_to_rows([item], [state])[0]

    assert result["n_steps"] == 12
    assert len(result["all_actions"]) == 12
    assert len(result["gold_all_actions"]) == 12


def test_decoder_can_emit_delimited_pid_and_select_exact_page():
    tok = get_tok()
    relation_ids = tok.encode("P31")
    page_ids = tok.encode("1")

    class PageModel(ScriptedModel):
        def _next(self, history):
            last = history[-1]
            already_read = tok.GRAPH_READ in history
            if last == tok.GRAPH_START:
                return tok.SLOTS[0]
            if last in tok.SLOTS:
                return tok.RELATIONS["r0"] if already_read else tok.RELATION_START
            if last == tok.RELATION_START:
                return relation_ids[0]
            for index, token_id in enumerate(relation_ids):
                if last == token_id:
                    return (
                        relation_ids[index + 1]
                        if index + 1 < len(relation_ids)
                        else tok.RELATION_END
                    )
            if last == tok.RELATION_END:
                return tok.DIR_OUT
            if last in tok.RELATIONS.values():
                return tok.DIR_OUT
            if last == tok.DIR_OUT:
                return tok.GRAPH_HALT if already_read else tok.PAGE_START
            if last == tok.PAGE_START:
                return page_ids[0]
            for index, token_id in enumerate(page_ids):
                if last == token_id:
                    return (
                        page_ids[index + 1]
                        if index + 1 < len(page_ids)
                        else tok.PAGE_END
                    )
            if last == tok.PAGE_END:
                return tok.GRAPH_READ
            if last == tok.ANSWER_STATE:
                return tok.encode("yes")[0]
            return tok.GRAPH_START

    item = _item()
    item.prompt = "Read exact page one."
    item.answer = "yes"
    item.meta.update(
        {
            "entity_slots": ["Q1", None, None, None],
            "answer_choices": ["yes", "no"],
            "action_slots": 12,
            "gold_addresses": [["Q1", "P31", "out", 1]],
            "gold_actions": [
                {
                    "source_slot": 0,
                    "relation_id": "P31",
                    "direction": "out",
                    "read": True,
                    "halt": False,
                    "page": 1,
                },
                {
                    "source_slot": 0,
                    "relation_id": "r0",
                    "direction": "out",
                    "read": False,
                    "halt": True,
                    "page": 0,
                },
                *[
                    {
                        "source_slot": 0,
                        "relation_id": "r0",
                        "direction": "out",
                        "read": False,
                        "halt": False,
                        "page": 0,
                    }
                    for _ in range(10)
                ],
            ],
        }
    )
    page = GraphRow(
        "Q1",
        "P31",
        "out",
        "entity",
        "Q9",
        (),
        "page-one",
        page=1,
    )

    result = decode_item(PageModel(tok), tok, item, AtomicGraphStore([page]))

    assert result.actions[0] == GraphAction(
        0, "P31", "out", True, False, page=1
    )
    assert result.rows[0] == page
    assert result.slots[0] == "Q9"
    assert result.actions[1].halt
    assert all(not action.read and not action.halt for action in result.actions[2:])
