from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import math
import tempfile
from types import SimpleNamespace

import pytest
import torch

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.relation_schema import RelationSchema, RelationSpec
from corpusgen.records import QAItem
from corpusgen.srgm_worlds import WorldConfig, generate_eval_pairs, generate_world
from evals.checkpoint_binding import (
    CheckpointValidationPolicy,
    canonical_configuration_sha256,
    verify_checkpoint_config,
)
from evals.relational_contracts import (
    CheckpointSummary,
    EvalRow,
    GuardrailReport,
    canonical_json_bytes,
    cluster_id_for,
    publish_evaluation,
)
from evals.relational_controls import (
    ControlID,
    EvalMode,
    build_control_view,
)
from evals.relational_generate import (
    GraphDecodeState,
    OverlayStore,
    apply_action,
    decode_item,
    decode_items,
    parse_action,
)
from organizer.graph_store import AtomicGraphStore, GraphStore, StoreStats
from organizer.packed_graph_store import PackedGraphStore
from scripts.run_relational_evals import (
    build_confirmatory_guardrail_report,
    publish_confirmatory_guardrail_report,
    _bind_edit_locality,
    _build_exploratory_route_artifact,
    _build_guardrail_source,
    _checkpoint_summary,
    _evaluation_matrix,
    _frozen_checkpoint_multiple,
    _gold_actions,
    _iter_eval_item_batches,
    _load_model,
    _ControlTransformationIndex,
    _LocalityIndex,
    _TransformationReferenceIndex,
    _promote_checkpoint_tree,
    _raw_token_count,
    _states_to_eval_rows,
    _states_to_rows,
    _validate_exact_checkpoint_matrix,
    _validate_pairing_receipt,
    _verify_and_promote_checkpoint_tree,
    store_for_item,
)
from train.tokenizer import get_tok


def _codec():
    from corpusgen.relation_codec import RelationCodec

    return RelationCodec(tuple(f"r{i}" for i in range(16)))


def _relation_schema(relation_id: str = "P31") -> RelationSchema:
    return RelationSchema(
        (RelationSpec(relation_id, ("relation",), "entity"),),
        (relation_id,),
    )


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


class ProtocolOnlyStore:
    def __init__(self, rows) -> None:
        self.contents = {row.address: row for row in rows}

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        return self.contents.get(address)

    def __len__(self) -> int:
        return len(self.contents)

    def snapshot_sha256(self) -> str:
        return AtomicGraphStore(self.contents.values()).snapshot_sha256()

    def stats(self) -> StoreStats:
        return StoreStats(len(self), 0, 0, 0)


@dataclass
class _HistoryCache:
    histories: list[list[int]]

    def select_batch(self, index: int) -> "_HistoryCache":
        return _HistoryCache([list(self.histories[index])])


class ScriptedModel:
    """Select READ, then HALT; choose slot 0 for provisional answers."""

    device = torch.device("cpu")

    def __init__(self, tok, codec):
        self.tok = tok
        self.codec = codec
        self.prefill_shapes: list[tuple[int, int]] = []
        self.step_batch_sizes: list[int] = []

    def _next(self, history: list[int]) -> int:
        tok = self.tok
        last = history[-1]
        if last == tok.ANSWER_STATE:
            return tok.SLOTS[0]
        if tok.GRAPH_START in history:
            start = len(history) - 1 - history[::-1].index(tok.GRAPH_START)
            frame_length = len(history) - start
            relation_code = self.codec.encode("r0", tok)
            if frame_length == 1:
                return tok.SLOTS[0]
            if frame_length in (2, 3, 4):
                return relation_code[frame_length - 2]
            if frame_length == 5:
                return tok.DIR_OUT
            if frame_length == 6:
                return (
                    tok.GRAPH_HALT
                    if tok.GRAPH_READ in history
                    else tok.GRAPH_READ
                )
            if frame_length == 7:
                return tok.GRAPH_END
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


class DelayedReadModel(ScriptedModel):
    """Select NOOP, then READ, then HALT."""

    def _next(self, history: list[int]) -> int:
        tok = self.tok
        if tok.GRAPH_START in history:
            start = len(history) - 1 - history[::-1].index(tok.GRAPH_START)
            frame_length = len(history) - start
            if frame_length == 6:
                if tok.GRAPH_NOOP not in history:
                    return tok.GRAPH_NOOP
                if tok.GRAPH_READ not in history:
                    return tok.GRAPH_READ
                return tok.GRAPH_HALT
        return super()._next(history)


def test_parse_fixed_action_tokens():
    tok = get_tok()
    codec = _codec()
    ids = [
        tok.GRAPH_START,
        tok.SLOTS[1],
        *codec.encode("r2", tok),
        tok.DIR_OUT,
        tok.GRAPH_READ,
        tok.GRAPH_END,
    ]
    assert parse_action(ids, tok, codec) == GraphAction(
        1, "r2", "out", True, False
    )


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
    codec = _codec()
    result = decode_item(
        ScriptedModel(tok, codec),
        tok,
        _item(),
        store=None,
        codec=codec,
    )

    assert result.misses == 1
    assert len(result.actions) == 6
    assert result.actions[1].halt
    assert all(
        not action.read and not action.halt for action in result.actions[2:]
    )
    assert len(result.provisional_answers) == 6
    assert result.lookup_latencies_ns == []
    assert not hasattr(result, "malformed")
    assert not hasattr(result, "excess_reads")


def _forced_gold_trace() -> tuple[GraphAction, ...]:
    return (
        GraphAction(0, "r0", "out", True, False),
        GraphAction(0, "r0", "out", False, True),
        *(
            GraphAction(0, "r0", "out", False, False)
            for _ in range(4)
        ),
    )


def test_forced_actions_and_returns_preserve_model_answer_logits_and_source():
    tok = get_tok()
    codec = _codec()
    item = _item()
    natural = decode_item(
        ScriptedModel(tok, codec),
        tok,
        item,
        _store(),
        codec=codec,
    )
    returned = _store().lookup(GraphAddress(7, "r0", "out"))
    forced = decode_item(
        ScriptedModel(tok, codec),
        tok,
        item,
        _store(),
        codec=codec,
        forced_actions=_forced_gold_trace(),
        forced_returns=(returned, None, None, None, None, None),
    )

    assert forced.actions == list(_forced_gold_trace())
    assert forced.rows == [returned, None, None, None, None, None]
    assert forced.provisional_answers == natural.provisional_answers
    assert forced.answer_logits == natural.answer_logits
    assert forced.prediction_source == "model"
    assert len(forced.answer_logits) == 6
    assert all(
        token_logits
        and all(math.isfinite(value) for value in token_logits)
        for token_logits in forced.answer_logits
    )


def test_gold_path_replay_never_substitutes_the_oracle_answer():
    tok = get_tok()
    codec = _codec()
    item = _item()
    item.answer = "<|slot_1|>"

    state = decode_item(
        ScriptedModel(tok, codec),
        tok,
        item,
        _store(),
        codec=codec,
        forced_actions=_forced_gold_trace(),
    )

    assert state.provisional_answers[-1] == "<|slot_0|>"
    assert state.provisional_answers[-1] != item.answer
    assert state.prediction_source == "model"


def test_gold_returns_scores_model_generated_actions():
    tok = get_tok()
    codec = _codec()
    world = generate_world(0, WorldConfig(n_entities=64, seed=17))
    item = next(
        pair.original
        for pair in generate_eval_pairs(
            world,
            n_pairs_per_task=8,
            seed=19,
        )
        if pair.task == "path_composition"
        and pair.original.meta["relations"][0] != "r0"
    )
    store = AtomicGraphStore(fact.row for fact in world.facts)
    view = build_control_view(
        item,
        store,
        ControlID.GOLD_RETURNS,
        seed=7,
    )
    natural = decode_item(
        ScriptedModel(tok, codec),
        tok,
        view.item,
        view.store,
        codec=codec,
    )

    state = decode_item(
        ScriptedModel(tok, codec),
        tok,
        view.item,
        view.store,
        codec=codec,
        forced_actions=view.forced_actions,
        forced_returns=view.forced_returns,
        forced_return_store=view.store,
    )
    gold_actions = [
        GraphAction(
            action["source_slot"],
            action["relation_id"],
            action["direction"],
            action["read"],
            action["halt"],
        )
        for action in item.meta["gold_actions"]
    ]

    assert view.forced_actions is None
    assert view.forced_returns is not None
    assert state.actions == natural.actions
    assert state.actions != gold_actions
    assert state.rows[0] == view.forced_returns[0]
    assert state.prediction_source == "model"


def test_gold_returns_are_consumed_only_when_model_emits_read():
    tok = get_tok()
    codec = _codec()
    returned = _store().lookup(GraphAddress(7, "r0", "out"))

    state = decode_item(
        DelayedReadModel(tok, codec),
        tok,
        _item(),
        _store(),
        codec=codec,
        forced_returns=(returned, None, None, None, None, None),
        forced_return_store=_store(),
    )

    assert not state.actions[0].read
    assert state.actions[1].read
    assert state.rows[:3] == [None, returned, None]


@pytest.mark.parametrize(
    "forced_actions, message",
    [
        (_forced_gold_trace()[:-1], "exactly six"),
        (
            (
                GraphAction(0, "r0", "out", False, True),
                *(
                    GraphAction(0, "r0", "out", False, False)
                    for _ in range(5)
                ),
            ),
            "before HALT",
        ),
        (
            (
                GraphAction(0, "r0", "out", True, False),
                GraphAction(0, "r0", "out", False, True),
                GraphAction(0, "r0", "out", True, False),
                *(
                    GraphAction(0, "r0", "out", False, False)
                    for _ in range(3)
                ),
            ),
            "after HALT",
        ),
    ],
)
def test_malformed_forced_action_traces_fail_closed(forced_actions, message):
    tok = get_tok()
    codec = _codec()
    with pytest.raises(ValueError, match=message):
        decode_item(
            ScriptedModel(tok, codec),
            tok,
            _item(),
            _store(),
            codec=codec,
            forced_actions=forced_actions,
        )


def test_forced_returns_validate_length_action_and_dynamic_address():
    tok = get_tok()
    codec = _codec()
    item = _item()
    valid = _store().lookup(GraphAddress(7, "r0", "out"))
    wrong = GraphRow(8, "r0", "out", "entity", "9", (), "w")
    forged = GraphRow(7, "r0", "out", "entity", "999", (), "w")

    with pytest.raises(ValueError, match="exactly six"):
        decode_item(
            ScriptedModel(tok, codec),
            tok,
            item,
            _store(),
            codec=codec,
            forced_actions=_forced_gold_trace(),
            forced_returns=(valid,),
        )
    with pytest.raises(ValueError, match="address"):
        decode_item(
            ScriptedModel(tok, codec),
            tok,
            item,
            _store(),
            codec=codec,
            forced_actions=_forced_gold_trace(),
            forced_returns=(wrong, None, None, None, None, None),
        )
    with pytest.raises(ValueError, match="payload"):
        decode_item(
            ScriptedModel(tok, codec),
            tok,
            item,
            _store(),
            codec=codec,
            forced_actions=_forced_gold_trace(),
            forced_returns=(forged, None, None, None, None, None),
        )
    with pytest.raises(ValueError, match="non-read"):
        decode_item(
            ScriptedModel(tok, codec),
            tok,
            item,
            _store(),
            codec=codec,
            forced_actions=_forced_gold_trace(),
            forced_returns=(valid, valid, None, None, None, None),
        )


def test_forced_returns_follow_entity_slot_updates_across_hops():
    tok = get_tok()
    codec = _codec()
    item = _item()
    actions = (
        GraphAction(0, "r0", "out", True, False),
        GraphAction(0, "r1", "out", True, False),
        GraphAction(0, "r0", "out", False, True),
        *(
            GraphAction(0, "r0", "out", False, False)
            for _ in range(3)
        ),
    )
    first = GraphRow(7, "r0", "out", "entity", "9", (), "w")
    second = GraphRow(9, "r1", "out", "entity", "11", (), "w")

    state = decode_item(
        ScriptedModel(tok, codec),
        tok,
        item,
        store=None,
        codec=codec,
        forced_actions=actions,
        forced_returns=(first, second, None, None, None, None),
        forced_return_store=AtomicGraphStore([first, second]),
    )

    assert state.rows[:2] == [first, second]
    assert state.slots[0] == 11


def test_counterfactual_overlay_changes_only_one_row():
    base = _store()
    replacement = GraphRow(7, "r0", "out", "entity", "10", (), "world")
    overlay = OverlayStore(base, replacement)

    assert overlay.lookup(replacement.address) == replacement
    assert len(overlay) == len(base)
    assert base.lookup(replacement.address).target == "9"

    absent = GraphRow(99, "r0", "out", "entity", "10", (), "world")
    with pytest.raises(ValueError, match="existing base address"):
        OverlayStore(base, absent)


def test_overlay_depends_only_on_public_graph_store_protocol():
    base: GraphStore = ProtocolOnlyStore(
        [
            GraphRow(7, "r0", "out", "entity", "9", (), "world"),
            GraphRow(8, "r1", "out", "literal", "value", (), "world"),
        ]
    )
    replacement = GraphRow(7, "r0", "out", "entity", "10", (), "changed")
    overlay = OverlayStore(base, {replacement.address: replacement})

    assert overlay.lookup(replacement.address) == replacement
    assert overlay.lookup(GraphAddress(8, "r1", "out")) == base.lookup(
        GraphAddress(8, "r1", "out")
    )


def test_equal_length_batch_matches_single_decode():
    tok = get_tok()
    codec = _codec()
    items = [_item("a"), _item("b")]
    assert len({len(tok.encode(item.prompt)) for item in items}) == 1
    singles = [
        decode_item(
            ScriptedModel(tok, codec),
            tok,
            item,
            _store(),
            codec=codec,
        )
        for item in items
    ]
    model = ScriptedModel(tok, codec)

    batched = decode_items(
        model,
        tok,
        items,
        _store(),
        batch_size=2,
        codec=codec,
    )

    assert batched == singles
    assert model.prefill_shapes == [(2, len(tok.encode(items[0].prompt)))]
    assert set(model.step_batch_sizes) == {2}


def test_unequal_prompt_lengths_are_never_padded_together():
    tok = get_tok()
    codec = _codec()
    short = _item("a")
    long = _item("a much longer entity label")
    assert len(tok.encode(short.prompt)) != len(tok.encode(long.prompt))
    model = ScriptedModel(tok, codec)

    decode_items(
        model,
        tok,
        [short, long],
        _store(),
        batch_size=2,
        codec=codec,
    )

    assert [shape[0] for shape in model.prefill_shapes] == [1, 1]


def test_variable_return_lengths_split_caches_without_changing_results():
    tok = get_tok()
    codec = _codec()
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
            ScriptedModel(tok, codec),
            tok,
            item,
            stores[item.qid],
            codec=codec,
        )
        for item in items
    ]
    model = ScriptedModel(tok, codec)

    batched = decode_items(
        model,
        tok,
        items,
        lambda item: stores[item.qid],
        batch_size=2,
        codec=codec,
    )

    assert batched == singles
    assert 2 in model.step_batch_sizes
    assert 1 in model.step_batch_sizes


def test_forced_replay_batches_mixed_return_lengths_and_matches_singles():
    tok = get_tok()
    codec = _codec()
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
    actions = {item.qid: _forced_gold_trace() for item in items}
    returns = {
        item.qid: (
            stores[item.qid].lookup(GraphAddress(7, "r0", "out")),
            None,
            None,
            None,
            None,
            None,
        )
        for item in items
    }
    single_models = [ScriptedModel(tok, codec) for _ in items]
    singles = [
        decode_item(
            model,
            tok,
            item,
            stores[item.qid],
            codec=codec,
            forced_actions=actions[item.qid],
            forced_returns=returns[item.qid],
        )
        for model, item in zip(single_models, items)
    ]
    batched_model = ScriptedModel(tok, codec)

    batched = decode_items(
        batched_model,
        tok,
        items,
        lambda item: stores[item.qid],
        batch_size=2,
        codec=codec,
        forced_actions=actions,
        forced_returns=returns,
        forced_return_store=lambda item: stores[item.qid],
    )

    assert batched == singles
    assert batched_model.prefill_shapes == [
        (2, len(tok.encode(items[0].prompt)))
    ]
    assert 2 in batched_model.step_batch_sizes
    assert (
        len(batched_model.prefill_shapes) + len(batched_model.step_batch_sizes)
        < sum(
            len(model.prefill_shapes) + len(model.step_batch_sizes)
            for model in single_models
        )
    )


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
    assert len(state.lookup_latencies_ns) == 1
    assert state.lookup_latencies_ns[0] >= 0


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


def _result_identity() -> dict:
    return {
        "model_id": "d160m",
        "arm": "split",
        "seed": 1001,
        "checkpoint_sha256": "1" * 64,
        "raw_token_count": 500,
        "evaluator_sha256": "2" * 64,
        "data_sha256": "3" * 64,
        "relation_schema_sha256": "4" * 64,
        "configuration_sha256": "6" * 64,
        "result_schema_sha256": "7" * 64,
        "provenance_sha256": "5" * 64,
    }


def test_strict_eval_rows_and_summary_bind_control_checkpoint_and_provenance():
    tok = get_tok()
    codec = _codec()
    item = _item()
    item.task = "path_composition"
    item.prompt = (
        "Slot 0 refers to Q7. Start at slot 0 and follow r0. "
        "Return the composed relation."
    )
    item.answer = "r1"
    item.meta["answer_choices"] = ["r0", "r1", "r2", "r3"]
    item.meta.update(
        {
            "world_id": 3,
            "provenance_id": "world-3",
            "relation_path_hash": "a" * 64,
            "template_id": "path_composition:v1",
            "composition_split": "seen",
            "hop_count": 1,
            "graph_rows": 1,
        }
    )
    store = AtomicGraphStore(
        [
            GraphRow(
                7,
                "r0",
                "out",
                "entity",
                "9",
                (("compose", "1"),),
                "world-3",
            )
        ]
    )
    view = build_control_view(item, store, ControlID.CORRECT, seed=9)
    state = decode_item(
        ScriptedModel(tok, codec),
        tok,
        view.item,
        view.store,
        codec=codec,
    )

    rows = _states_to_eval_rows(
        [view],
        [state],
        memory_mode=EvalMode.MEMORY_ON,
        identity=_result_identity(),
    )

    assert len(rows) == 1
    assert isinstance(rows[0], EvalRow)
    assert rows[0].checkpoint_sha256 == "1" * 64
    assert rows[0].control_id == "correct"
    assert rows[0].provenance_id == "world-3"
    assert rows[0].prediction_source == "model"
    assert rows[0].lookup_latency_ns >= 0

    relevant_view = build_control_view(
        item,
        store,
        ControlID.RELEVANT_EDGE,
        seed=9,
    )
    relevant_state = decode_item(
        ScriptedModel(tok, codec),
        tok,
        relevant_view.item,
        relevant_view.store,
        codec=codec,
    )
    relevant_state.provisional_answers[-1] = relevant_view.oracle_after
    relevant_row = _states_to_eval_rows(
        [relevant_view],
        [relevant_state],
        memory_mode=EvalMode.MEMORY_ON,
        identity=_result_identity(),
    )[0]
    assert relevant_row.answer == relevant_view.oracle_after
    assert relevant_row.correct

    counterfactual = rows[0].to_dict()
    counterfactual["qid"] = "pair-a-c"
    counterfactual["variant"] = "counterfactual"
    paired = [rows[0], EvalRow.from_dict(counterfactual)]
    summary = _checkpoint_summary(paired)
    assert isinstance(summary, CheckpointSummary)
    assert summary.checkpoint_sha256 == rows[0].checkpoint_sha256
    assert summary.control_id == rows[0].control_id
    assert summary.n_pairs == 1


def test_edit_locality_joins_controls_one_to_one_and_rejects_crossing():
    base = _strict_locality_row(
        control_id="correct",
        prediction="r1",
        oracle_before="r1",
        oracle_after="r1",
        oracle_effect="unchanged",
        changed_addresses=[],
    )
    relevant = _strict_locality_row(
        control_id="relevant_edge",
        prediction="r2",
        oracle_before="r1",
        oracle_after="r2",
        oracle_effect="changed",
        changed_addresses=[[7, "r0", "out"]],
    )
    irrelevant = _strict_locality_row(
        control_id="irrelevant_edge",
        prediction="r1",
        oracle_before="r1",
        oracle_after="r1",
        oracle_effect="unchanged",
        changed_addresses=[[8, "r0", "out"]],
    )

    bound = _bind_edit_locality(
        {
            ControlID.CORRECT: [base],
            ControlID.RELEVANT_EDGE: [relevant],
            ControlID.IRRELEVANT_EDGE: [irrelevant],
        }
    )

    assert bound[ControlID.RELEVANT_EDGE][0].edit_locality_correct is True
    assert bound[ControlID.IRRELEVANT_EDGE][0].edit_locality_correct is True

    crossed = relevant.to_dict()
    crossed["qid"] = "other-o"
    with pytest.raises(ValueError, match="one-to-one"):
        _bind_edit_locality(
            {
                ControlID.CORRECT: [base],
                ControlID.RELEVANT_EDGE: [EvalRow.from_dict(crossed)],
                ControlID.IRRELEVANT_EDGE: [irrelevant],
            }
        )


def test_streaming_locality_index_joins_without_retaining_rows(tmp_path):
    baseline = _strict_locality_row(
        control_id="correct",
        prediction="r1",
        oracle_before="r1",
        oracle_after="r1",
        oracle_effect="unchanged",
        changed_addresses=[],
    )
    relevant = _strict_locality_row(
        control_id="relevant_edge",
        prediction="r2",
        oracle_before="r1",
        oracle_after="r2",
        oracle_effect="changed",
        changed_addresses=[[7, "r0", "out"]],
    )
    index = _LocalityIndex(tmp_path)
    try:
        index.add_baseline(baseline)
        bound = index.bind(relevant)
        index.require_complete({ControlID.RELEVANT_EDGE})
    finally:
        index.close()

    assert bound.edit_locality_correct is True
    assert index.buffered_rows == 0
    assert index.closed


def test_control_transformation_index_deduplicates_compact_records(tmp_path):
    record = {
        "record_type": "control_transformation",
        "schema_version": 1,
        "control_id": "shuffled_returns",
        "seed": 7,
        "provenance_id": "eval:world:3",
        "source_store_sha256": "a" * 64,
        "transformed_store_sha256": "b" * 64,
        "changed_address_count": 100_000,
        "changed_addresses_sha256": "c" * 64,
        "return_sources_sha256": "d" * 64,
        "entity_bijection_sha256": hashlib.sha256(b"[]").hexdigest(),
    }
    record["transformation_metadata_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "changed_address_count",
                    "changed_addresses_sha256",
                    "return_sources_sha256",
                    "entity_bijection_sha256",
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    record["transformation_id"] = hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    view = SimpleNamespace(transformation_record=lambda: dict(record))
    index = _ControlTransformationIndex(tmp_path)
    try:
        index.add(view)
        index.add(view)
        path, count = index.publish("9" * 64)
    finally:
        index.close()

    assert count == 1
    assert path == (
        tmp_path
        / "evals"
        / ("9" * 64)
        / "control-transformations.jsonl"
    )
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        record
    ]


def _transformation_record(
    *,
    control_id="shuffled_returns",
    seed=7,
    provenance_id="eval:world:3",
    source="a" * 64,
    transformed="b" * 64,
    changed="c" * 64,
    returns="d" * 64,
    entities=hashlib.sha256(b"[]").hexdigest(),
):
    metadata = {
        "changed_address_count": 100_000,
        "changed_addresses_sha256": changed,
        "return_sources_sha256": returns,
        "entity_bijection_sha256": entities,
    }
    record = {
        "record_type": "control_transformation",
        "schema_version": 1,
        "control_id": control_id,
        "seed": seed,
        "provenance_id": provenance_id,
        "source_store_sha256": source,
        "transformed_store_sha256": transformed,
        **metadata,
        "transformation_metadata_sha256": hashlib.sha256(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    record["transformation_id"] = hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return record


def _transformation_row(record, memory_mode):
    return SimpleNamespace(
        qid="pair-a-o",
        memory_mode=memory_mode,
        control_id=record["control_id"],
        control_seed=record["seed"],
        provenance_id=record["provenance_id"],
        source_store_sha256=record["source_store_sha256"],
        transformed_store_sha256=record["transformed_store_sha256"],
        transformation_metadata_sha256=record[
            "transformation_metadata_sha256"
        ],
        transformation_id=record["transformation_id"],
    )


def test_transformation_reference_index_resolves_exact_memory_twins(tmp_path):
    record = _transformation_record()
    index = _TransformationReferenceIndex(tmp_path)
    try:
        index.add_record(record)
        index.add_row(
            _transformation_row(record, EvalMode.MEMORY_ON.value)
        )
        index.add_row(
            _transformation_row(record, EvalMode.MEMORY_OFF.value)
        )
        index.finalize(expected_record_count=1)
    finally:
        index.close()

    assert index.buffered_rows == 0
    assert index.closed


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing", "missing transformation"),
        ("control", "control_id"),
        ("seed", "seed"),
        ("provenance", "provenance_id"),
        ("source", "source_store_sha256"),
        ("transformed", "transformed_store_sha256"),
        ("metadata", "transformation_metadata_sha256"),
        ("duplicate_record", "duplicate transformation"),
        ("orphan", "orphan transformation"),
        ("crossed_memory", "memory modes"),
    ],
)
def test_transformation_reference_index_rejects_all_crossings(
    tmp_path,
    tamper,
    message,
):
    first = _transformation_record()
    second = _transformation_record(source="f" * 64)
    index = _TransformationReferenceIndex(tmp_path)
    try:
        if tamper != "missing":
            index.add_record(first)
        if tamper in {"orphan", "crossed_memory"}:
            index.add_record(second)
        if tamper == "duplicate_record":
            with pytest.raises(ValueError, match=message):
                index.add_record(first)
            return
        on = _transformation_row(first, EvalMode.MEMORY_ON.value)
        off_record = second if tamper == "crossed_memory" else first
        off = _transformation_row(
            off_record,
            EvalMode.MEMORY_OFF.value,
        )
        if tamper in {
            "control",
            "seed",
            "provenance",
            "source",
            "transformed",
            "metadata",
        }:
            field = {
                "control": "control_id",
                "seed": "control_seed",
                "provenance": "provenance_id",
                "source": "source_store_sha256",
                "transformed": "transformed_store_sha256",
                "metadata": "transformation_metadata_sha256",
            }[tamper]
            setattr(on, field, 8 if field == "control_seed" else "wrong")
        if tamper == "missing":
            with pytest.raises(ValueError, match=message):
                index.add_row(on)
            return
        if tamper in {
            "control",
            "seed",
            "provenance",
            "source",
            "transformed",
            "metadata",
        }:
            with pytest.raises(ValueError, match=message):
                index.add_row(on)
            return
        index.add_row(on)
        if tamper == "crossed_memory":
            with pytest.raises(ValueError, match=message):
                index.add_row(off)
            return
        index.add_row(off)
        with pytest.raises(ValueError, match=message):
            index.finalize(expected_record_count=len(
                {first["transformation_id"], second["transformation_id"]}
            ))
    finally:
        index.close()


def _strict_locality_row(
    *,
    control_id: str,
    prediction: str,
    oracle_before: str,
    oracle_after: str,
    oracle_effect: str,
    changed_addresses: list,
) -> EvalRow:
    identity = _result_identity()
    value = {
        "record_type": "eval_row",
        "schema_version": 1,
        "qid": "pair-a-o",
        "pair_id": "pair-a",
        "variant": "original",
        "task": "path_composition",
        "world_id": 3,
        "provenance_id": "world-3",
        "relation_path_hash": "a" * 64,
        "template_id": "path:v1",
        "composition_split": "seen",
        "hop": 1,
        **identity,
        "memory_mode": "memory_on",
        "control_id": control_id,
        "cluster_id": cluster_id_for(
            seed=identity["seed"],
            world_id=3,
            relation_path_hash="a" * 64,
            template_id="path:v1",
        ),
        "prediction": prediction,
        "answer": oracle_after,
        "correct": prediction == oracle_after,
        "prediction_source": "model",
        "all_actions": [
            [0, "r0", "out", True, False],
            [0, "r0", "out", False, True],
            *[[0, "r0", "out", False, False] for _ in range(4)],
        ],
        "gold_all_actions": [
            [0, "r0", "out", True, False],
            [0, "r0", "out", False, True],
            *[[0, "r0", "out", False, False] for _ in range(4)],
        ],
        "returned_addresses": [[7, "r0", "out"], None, None, None, None, None],
        "gold_addresses": [[7, "r0", "out"]],
        "correct_referents": [True],
        "misses": 0,
        "malformed": 0,
        "abstained": False,
        "excess_reads": 0,
        "halt_step": 2,
        "answer_logits": [[-0.1] for _ in range(6)],
        "lookup_latency_ns": 10,
        "lookup_count": 1,
        "store_rows": 10,
        "store_bytes": 100,
        "control_seed": 9,
        "transformation_id": None,
        "source_store_sha256": None,
        "transformed_store_sha256": None,
        "transformation_metadata_sha256": None,
        "changed_addresses": changed_addresses,
        "oracle_before": oracle_before,
        "oracle_after": oracle_after,
        "oracle_effect": oracle_effect,
        "edit_locality_correct": None,
    }
    return EvalRow.from_dict(value)


def test_raw_token_count_is_exact_and_rejects_missing_or_boolean_metadata():
    assert _raw_token_count(
        {"step": 10},
        {"tokens_per_step": 256},
    ) == 2560
    with pytest.raises(ValueError, match="step"):
        _raw_token_count({"step": True}, {"tokens_per_step": 256})
    with pytest.raises(ValueError, match="tokens_per_step"):
        _raw_token_count({"step": 10}, {})


def test_frozen_checkpoint_multiple_enforces_rounding_tolerance():
    assert _frozen_checkpoint_multiple(500_099, 100_000) == 5
    assert _frozen_checkpoint_multiple(1_000_199, 100_000) == 10
    assert _frozen_checkpoint_multiple(2_000_399, 100_000) == 20
    with pytest.raises(ValueError, match="0.02%"):
        _frozen_checkpoint_multiple(500_100, 100_000)
    with pytest.raises(ValueError, match="0.02%"):
        _frozen_checkpoint_multiple(500_101, 100_000)
    with pytest.raises(ValueError, match="parameter_count"):
        _frozen_checkpoint_multiple(500_000, True)


def test_evaluation_matrix_contains_every_memory_control_cell_once():
    matrix = _evaluation_matrix()
    assert len(matrix) == 22
    assert len(set(matrix)) == 22
    assert set(matrix) == {
        (mode, control)
        for mode in EvalMode
        for control in ControlID
    }


def test_eval_item_batches_are_bounded_and_keep_twins_together(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    originals = []
    counterfactuals = []
    for task in ("path_composition", "date_ordering", "balanced_equality"):
        for index in range(3):
            pair_id = f"{task}-{index}"
            original = _item(f"{pair_id}-o")
            original.task = task
            original.meta.update(pair_id=pair_id, variant="original")
            counterfactual = _item(f"{pair_id}-c")
            counterfactual.task = task
            counterfactual.meta.update(
                pair_id=pair_id,
                variant="counterfactual",
            )
            originals.append(original.__dict__)
            counterfactuals.append(counterfactual.__dict__)
    (eval_dir / "original.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in originals)
    )
    (eval_dir / "counterfactual.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in counterfactuals)
    )

    batches = list(
        _iter_eval_item_batches(
            tmp_path,
            expected_pairs=3,
            batch_pairs=2,
        )
    )

    assert max(map(len, batches)) == 4
    assert sum(map(len, batches)) == 18
    for batch in batches:
        assert len(batch) % 2 == 0
        for offset in range(0, len(batch), 2):
            first, second = batch[offset : offset + 2]
            assert first.meta["pair_id"] == second.meta["pair_id"]
            assert [first.meta["variant"], second.meta["variant"]] == [
                "original",
                "counterfactual",
            ]


def test_checkpoint_tree_publication_is_atomic_and_non_overwriting(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint_hash = "a" * 64

    def staged(name: str) -> Path:
        root = tmp_path / name
        checkpoint = root / "evals" / checkpoint_hash
        for mode in EvalMode:
            (checkpoint / mode.value).mkdir(parents=True)
        (checkpoint / "guardrail-source.json").write_text("{}\n")
        (checkpoint / "control-transformations.jsonl").write_text("")
        (checkpoint / "exact-matrix-manifest.json").write_text("{}\n")
        return root

    destination = _promote_checkpoint_tree(
        run,
        staged("first"),
        checkpoint_hash,
    )

    assert destination == run / "evals" / checkpoint_hash
    assert destination.is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        _promote_checkpoint_tree(
            run,
            staged("second"),
            checkpoint_hash,
        )


def test_failed_checkpoint_contender_does_not_remove_owned_lock(tmp_path):
    run = tmp_path / "run"
    evals = run / "evals"
    evals.mkdir(parents=True)
    checkpoint_hash = "b" * 64
    lock = evals / f".{checkpoint_hash}.publish.lock"
    lock.write_text("owner")
    staging = tmp_path / "staging"
    checkpoint = staging / "evals" / checkpoint_hash
    for mode in EvalMode:
        (checkpoint / mode.value).mkdir(parents=True)
    (checkpoint / "guardrail-source.json").write_text("{}\n")
    (checkpoint / "control-transformations.jsonl").write_text("")
    (checkpoint / "exact-matrix-manifest.json").write_text("{}\n")

    with pytest.raises(FileExistsError):
        _promote_checkpoint_tree(run, staging, checkpoint_hash)
    assert lock.read_text() == "owner"

    lock.unlink()
    assert _promote_checkpoint_tree(
        run,
        staging,
        checkpoint_hash,
    ).is_dir()


@pytest.mark.parametrize(
    "mutation",
    [
        "checkpoint",
        "data",
        "evaluator",
        "relation_schema",
        "result_schema",
        "config",
        "provenance",
    ],
)
def test_bound_input_mutation_stays_staged_and_does_not_block_rerun(
    tmp_path,
    monkeypatch,
    mutation,
):
    import scripts.run_relational_evals as evaluator

    run = tmp_path / "run"
    run.mkdir()
    staging = tmp_path / "staging"
    checkpoint_hash = "c" * 64
    checkpoint = staging / "evals" / checkpoint_hash
    for mode in EvalMode:
        (checkpoint / mode.value).mkdir(parents=True)
    (checkpoint / "guardrail-source.json").write_text("{}\n")
    (checkpoint / "control-transformations.jsonl").write_text("")
    (checkpoint / "exact-matrix-manifest.json").write_text("{}\n")
    relation_schema = tmp_path / "relation-schema.json"
    result_schema = tmp_path / "result-schema.json"
    expected = {
        "data": "1" * 64,
        "evaluator": "2" * 64,
        "relation_schema": "3" * 64,
        "result_schema": "4" * 64,
    }
    active = {"mutation": mutation}

    def verify_checkpoint(*_args):
        if active["mutation"] == "checkpoint":
            raise RuntimeError("checkpoint changed")

    def verify_config(*_args, **_kwargs):
        identities = {
            "config": "ok",
            "configuration_sha256": "5" * 64,
        }
        if active["mutation"] == "config":
            identities["config"] = "changed"
        return {"condition": "split"}, identities

    def data_hash(*_args):
        return (
            "9" * 64
            if active["mutation"] == "data"
            else expected["data"]
        )

    def evaluator_hash():
        return (
            "9" * 64
            if active["mutation"] == "evaluator"
            else expected["evaluator"]
        )

    def file_hash(path):
        key = (
            "relation_schema"
            if Path(path) == relation_schema
            else "result_schema"
        )
        return (
            "9" * 64
            if active["mutation"] == key
            else expected[key]
        )

    monkeypatch.setattr(
        evaluator,
        "verify_checkpoint_unchanged",
        verify_checkpoint,
    )
    monkeypatch.setattr(evaluator, "verify_checkpoint_config", verify_config)
    monkeypatch.setattr(evaluator, "_data_sha256", data_hash)
    monkeypatch.setattr(evaluator, "_evaluator_sha256", evaluator_hash)
    monkeypatch.setattr(evaluator, "_regular_file_sha256", file_hash)
    provenance = hashlib.sha256(
        evaluator.canonical_json_bytes(
            {
                "result_contract": "relational-result-v1",
                "data_sha256": expected["data"],
                "relation_schema_sha256": expected["relation_schema"],
                "evaluator_sha256": expected["evaluator"],
                "configuration_sha256": "5" * 64,
                "result_schema_sha256": expected["result_schema"],
            }
        )
    ).hexdigest()
    bindings = {
        "run": run,
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_state": {},
        "config": {"condition": "split"},
        "config_identities": {
            "config": "ok",
            "configuration_sha256": "5" * 64,
        },
        "base_store": object(),
        "factual_store": object(),
        "data_dir": tmp_path,
        "data_sha256": expected["data"],
        "evaluator_sha256": expected["evaluator"],
        "relation_schema_path": relation_schema,
        "relation_codec": SimpleNamespace(
            sha256=lambda: expected["relation_schema"]
        ),
        "relation_schema_sha256": expected["relation_schema"],
        "result_schema_path": result_schema,
        "result_schema_sha256": expected["result_schema"],
        "provenance_sha256": (
            "9" * 64 if mutation == "provenance" else provenance
        ),
    }

    with pytest.raises(RuntimeError, match="changed"):
        _verify_and_promote_checkpoint_tree(
            run,
            staging,
            checkpoint_hash,
            bindings,
        )
    assert checkpoint.is_dir()
    assert not (run / "evals" / checkpoint_hash).exists()

    active["mutation"] = None
    bindings["provenance_sha256"] = provenance
    assert _verify_and_promote_checkpoint_tree(
        run,
        staging,
        checkpoint_hash,
        bindings,
    ).is_dir()


def test_historical_matrix_manifest_tampering_is_rejected(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    checkpoint_hash = "d" * 64
    checkpoint = tmp_path / checkpoint_hash
    guardrail = checkpoint / "guardrail-source.json"
    transformations = checkpoint / "control-transformations.jsonl"
    guardrail.parent.mkdir()
    guardrail_identity = {
        "checkpoint_sha256": checkpoint_hash,
        "model_id": "d160m",
        "arm": "split",
        "seed": 1001,
        "raw_token_count": 100,
        "evaluator_sha256": "1" * 64,
        "data_sha256": "2" * 64,
        "relation_schema_sha256": "3" * 64,
        "provenance_sha256": "4" * 64,
        "configuration_sha256": "5" * 64,
        "result_schema_sha256": "6" * 64,
    }
    guardrail.write_bytes(
        evaluator.canonical_json_bytes(
            {
                "record_type": "guardrail_source",
                **guardrail_identity,
            }
        )
    )
    empty = hashlib.sha256(b"[]").hexdigest()
    transformation_records = {
        "shuffled_returns": _transformation_record(),
        "entity_rename": _transformation_record(
            control_id="entity_rename",
            source="b" * 64,
            transformed="c" * 64,
            returns=empty,
            entities="d" * 64,
        ),
        "graph_isomorphism": _transformation_record(
            control_id="graph_isomorphism",
            source="e" * 64,
            transformed="f" * 64,
            returns=empty,
            entities="1" * 64,
        ),
    }
    transformations.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in sorted(
                transformation_records.values(),
                key=lambda value: value["transformation_id"],
            )
        )
    )
    cells = []
    for mode, control in _evaluation_matrix():
        cell = checkpoint / mode.value / control.value
        cell.mkdir(parents=True)
        for name in ("rows.jsonl", "summary.json", "manifest.json"):
            (cell / name).write_text(name)
        cells.append(
            {
                "memory_mode": mode.value,
                "control_id": control.value,
                "summary_sha256": evaluator._regular_file_sha256(
                    cell / "summary.json"
                ),
                "rows_sha256": evaluator._regular_file_sha256(
                    cell / "rows.jsonl"
                ),
                "manifest_sha256": evaluator._regular_file_sha256(
                    cell / "manifest.json"
                ),
            }
        )
    manifest = {
        "record_type": "exact_evaluation_matrix",
        "schema_version": 2,
        "checkpoint_sha256": checkpoint_hash,
        "cell_count": len(cells),
        "cells": cells,
        "identity": dict(guardrail_identity),
        "identity_sha256": hashlib.sha256(
            evaluator.canonical_json_bytes(guardrail_identity)
        ).hexdigest(),
        "guardrail_artifact": {
            "path": "guardrail-source.json",
            "record_type": "guardrail_source",
            "sha256": evaluator._regular_file_sha256(guardrail),
        },
        "control_transformations_sha256": evaluator._regular_file_sha256(
            transformations
        ),
        "control_transformation_count": len(transformation_records),
    }
    manifest_path = checkpoint / "exact-matrix-manifest.json"
    manifest_path.write_bytes(evaluator.canonical_json_bytes(manifest))

    crossed_field = {"name": None}

    def validated_cell(path, **kwargs):
        cell = Path(path)
        record = transformation_records.get(cell.name)
        if record is not None:
            row = _transformation_row(record, cell.parent.name)
            row.qid = f"qid-{cell.name}"
            kwargs["row_consumer"](row)
        identity = dict(guardrail_identity)
        field = crossed_field["name"]
        if (
            field is not None
            and cell.parent.name == EvalMode.MEMORY_OFF.value
            and cell.name == ControlID.NO_QUERY.value
        ):
            identity[field] = (
                identity[field] + 1
                if field in {"seed", "raw_token_count"}
                else "crossed-model"
                if field == "model_id"
                else "dense"
                if field == "arm"
                else "f" * 64
            )
        return SimpleNamespace(
            **identity,
            memory_mode=cell.parent.name,
            control_id=cell.name,
            rows_sha256=evaluator._regular_file_sha256(
                cell / "rows.jsonl"
            ),
        )

    monkeypatch.setattr(
        evaluator,
        "validate_published_evaluation",
        validated_cell,
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_guardrail_source",
        lambda value: value,
    )
    scratch_directories = []
    real_mkstemp = tempfile.mkstemp
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        scratch_directories.append(Path(name))
        return descriptor, name

    def recording_mkdtemp(*args, **kwargs):
        name = real_mkdtemp(*args, **kwargs)
        scratch_directories.append(Path(name))
        return name

    monkeypatch.setattr(evaluator.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(evaluator.tempfile, "mkdtemp", recording_mkdtemp)
    assert len(_validate_exact_checkpoint_matrix(checkpoint)) == 22
    assert scratch_directories
    assert all(
        not directory.is_relative_to(checkpoint.parent)
        and not directory.exists()
        for directory in scratch_directories
    )

    def tree_fingerprint():
        return [
            (
                path.relative_to(checkpoint).as_posix(),
                path.is_dir(),
                None
                if path.is_dir()
                else hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(checkpoint.rglob("*"))
        ]

    before_read_only_validation = tree_fingerprint()
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    directories = [
        path for path in checkpoint.rglob("*") if path.is_dir()
    ]
    for path in files:
        path.chmod(0o444)
    for path in reversed(directories):
        path.chmod(0o555)
    checkpoint.chmod(0o555)
    try:
        assert len(_validate_exact_checkpoint_matrix(checkpoint)) == 22
        assert tree_fingerprint() == before_read_only_validation
    finally:
        checkpoint.chmod(0o755)
        for path in directories:
            path.chmod(0o755)
        for path in files:
            path.chmod(0o644)

    real_validate_transformations = (
        evaluator._validate_control_transformation_table
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_control_transformation_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected validation crash")
        ),
    )
    with pytest.raises(RuntimeError, match="injected validation crash"):
        _validate_exact_checkpoint_matrix(checkpoint)
    assert all(not directory.exists() for directory in scratch_directories)
    monkeypatch.setattr(
        evaluator,
        "_validate_control_transformation_table",
        real_validate_transformations,
    )
    for field in (
        "checkpoint_sha256",
        "model_id",
        "arm",
        "seed",
        "raw_token_count",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "provenance_sha256",
        "configuration_sha256",
        "result_schema_sha256",
    ):
        crossed_field["name"] = field
        with pytest.raises(ValueError, match="cell identity|immutable identity"):
            _validate_exact_checkpoint_matrix(checkpoint)
    crossed_field["name"] = None

    original_rows_sha256 = manifest["cells"][0]["rows_sha256"]
    manifest["cells"][0]["rows_sha256"] = "0" * 64
    manifest_path.write_bytes(evaluator.canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="cell hash"):
        _validate_exact_checkpoint_matrix(checkpoint)

    manifest["cells"][0]["rows_sha256"] = original_rows_sha256
    manifest_path.write_bytes(evaluator.canonical_json_bytes(manifest))
    external_guardrail = tmp_path / "matching-guardrail-source.json"
    external_guardrail.write_bytes(guardrail.read_bytes())
    guardrail.unlink()
    guardrail.symlink_to(external_guardrail)
    with pytest.raises(ValueError, match="regular|symlink"):
        _validate_exact_checkpoint_matrix(checkpoint)

    guardrail.unlink()
    guardrail.write_bytes(external_guardrail.read_bytes())
    displaced_guardrail = tmp_path / "displaced-guardrail-source.json"
    real_open = evaluator.os.open
    swapped = {"done": False}

    def swap_after_open(name, flags, *args, **kwargs):
        descriptor = real_open(name, flags, *args, **kwargs)
        if (
            name == "guardrail-source.json"
            and kwargs.get("dir_fd") is not None
            and not swapped["done"]
        ):
            swapped["done"] = True
            guardrail.rename(displaced_guardrail)
            guardrail.symlink_to(external_guardrail)
        return descriptor

    monkeypatch.setattr(evaluator.os, "open", swap_after_open)
    with pytest.raises(ValueError, match="changed"):
        _validate_exact_checkpoint_matrix(checkpoint)


def _paired_run_configs():
    common = {
        "model": "d160m",
        "seed": 1001,
        "load": "n800k",
        "ctx": 128,
        "tokens_per_step": 10,
        "initialization_seed": 1001,
        "data_seed": 17,
        "train_bin": "train.bin",
        "train_mask": "train.mask",
        "packing": {"block_size": 128},
        "optimizer": {"name": "adamw"},
        "scheduler": {"name": "cosine"},
        "raw_positions": {"start": 0},
        "decode_budget": 6,
        "checkpoint_schedule": [5, 10, 20],
    }
    return {
        "split": {
            **common,
            "condition": "split",
            "train_weights": "split.weights.bin",
        },
        "dense": {
            **common,
            "condition": "dense",
            "train_weights": "dense.weights.bin",
        },
    }


def _pairing_receipt(split_checkpoint, dense_checkpoint, shared):
    from evals.relational_pairing import build_pairing_receipt

    configs = shared.get("configs", _paired_run_configs())
    common = {
        field: shared[field]
        for field in (
            "model_id",
            "seed",
            "raw_token_count",
            "evaluator_sha256",
            "data_sha256",
            "relation_schema_sha256",
            "result_schema_sha256",
        )
    }
    split = SimpleNamespace(
        **common,
        checkpoint_sha256=split_checkpoint,
        arm="split",
        configuration_sha256=canonical_configuration_sha256(
            configs["split"]
        ),
        provenance_sha256=shared.get(
            "split_result_provenance_sha256",
            "8" * 64,
        ),
    )
    dense = SimpleNamespace(
        **common,
        checkpoint_sha256=dense_checkpoint,
        arm="dense",
        configuration_sha256=canonical_configuration_sha256(
            configs["dense"]
        ),
        provenance_sha256=shared.get(
            "dense_result_provenance_sha256",
            "9" * 64,
        ),
    )
    return build_pairing_receipt(
        split,
        dense,
        configs["split"],
        configs["dense"],
    ).to_dict()


def test_shared_configuration_identity_excludes_only_declared_arm_fields():
    from evals.checkpoint_binding import (
        canonical_shared_configuration_sha256,
    )

    common = {
        "model": "d160m",
        "seed": 1001,
        "load": "n800k",
        "ctx": 128,
        "initialization_seed": 1001,
        "data_seed": 17,
        "train_bin": "train.bin",
        "train_mask": "train.mask",
        "packing": {"block_size": 128},
        "optimizer": {"name": "adamw"},
        "scheduler": {"name": "cosine"},
        "checkpoint_schedule": [5, 10, 20],
    }
    split = {
        **common,
        "condition": "split",
        "train_weights": "split.weights.bin",
    }
    dense = {
        **common,
        "condition": "dense",
        "train_weights": "dense.weights.bin",
    }

    assert canonical_shared_configuration_sha256(
        split
    ) == canonical_shared_configuration_sha256(dense)

    for field, changed in (
        ("load", "n50k"),
        ("data_seed", 18),
        ("optimizer", {"name": "sgd"}),
        ("checkpoint_schedule", [5, 10]),
    ):
        crossed = {**dense, field: changed}
        assert canonical_shared_configuration_sha256(
            crossed
        ) != canonical_shared_configuration_sha256(split)

    differently_declared_sidecar = dict(dense)
    differently_declared_sidecar["weights_rel"] = (
        differently_declared_sidecar.pop("train_weights")
    )
    assert canonical_shared_configuration_sha256(
        differently_declared_sidecar
    ) != canonical_shared_configuration_sha256(split)


def test_pairing_receipt_accepts_authentic_distinct_arm_provenance(
    tmp_path,
):
    import scripts.run_relational_evals as evaluator
    from evals.checkpoint_binding import (
        canonical_shared_configuration_sha256,
        load_run_configuration,
    )

    common = {
        "model": "d160m",
        "seed": 1001,
        "load": "n800k",
        "ctx": 128,
        "tokens_per_step": 10,
        "initialization_seed": 1001,
        "data_seed": 17,
        "train_bin": "train.bin",
        "train_mask": "train.mask",
        "packing": {"block_size": 128},
        "optimizer": {"name": "adamw"},
        "scheduler": {"name": "cosine"},
        "raw_positions": {"start": 0},
        "decode_budget": 6,
        "checkpoint_schedule": [5, 10, 20],
    }
    configs = {
        "split": {
            **common,
            "condition": "split",
            "train_weights": "split.weights.bin",
        },
        "dense": {
            **common,
            "condition": "dense",
            "train_weights": "dense.weights.bin",
        },
    }
    identities = {}
    for condition, config in configs.items():
        run = tmp_path / condition
        run.mkdir()
        (run / "config.yaml").write_text(json.dumps(config))
        normalized, config_identity = load_run_configuration(run)
        identities[condition] = evaluator._result_identity(
            cfg=normalized,
            state={"step": 10},
            checkpoint_hash=("1" if condition == "split" else "2") * 64,
            config_identities=config_identity,
            data_hash="3" * 64,
            relation_schema_hash="4" * 64,
            result_schema_hash="5" * 64,
        )
        configs[condition] = normalized

    assert (
        identities["split"]["configuration_sha256"]
        != identities["dense"]["configuration_sha256"]
    )
    assert (
        identities["split"]["provenance_sha256"]
        != identities["dense"]["provenance_sha256"]
    )

    shared_configuration = canonical_shared_configuration_sha256(
        configs["split"]
    )
    assert shared_configuration == canonical_shared_configuration_sha256(
        configs["dense"]
    )
    shared = {
        "model_id": identities["split"]["model_id"],
        "seed": identities["split"]["seed"],
        "raw_token_count": identities["split"]["raw_token_count"],
        "evaluator_sha256": identities["split"]["evaluator_sha256"],
        "data_sha256": identities["split"]["data_sha256"],
        "relation_schema_sha256": identities["split"][
            "relation_schema_sha256"
        ],
        "result_schema_sha256": identities["split"][
            "result_schema_sha256"
        ],
        "configs": configs,
        "split_result_provenance_sha256": identities["split"][
            "provenance_sha256"
        ],
        "dense_result_provenance_sha256": identities["dense"][
            "provenance_sha256"
        ],
    }
    receipt = _pairing_receipt(
        identities["split"]["checkpoint_sha256"],
        identities["dense"]["checkpoint_sha256"],
        shared,
    )

    validated = _validate_pairing_receipt(
        receipt,
        split_anchor=SimpleNamespace(**identities["split"]),
        dense_anchor=SimpleNamespace(**identities["dense"]),
        split_config=configs["split"],
        dense_config=configs["dense"],
    )
    assert validated["split_result_provenance_sha256"] != validated[
        "dense_result_provenance_sha256"
    ]
    assert validated["study_provenance_sha256"] == (
        receipt["study_provenance_sha256"]
    )

    copied = json.loads(json.dumps(receipt))
    copied["dense_result_provenance_sha256"] = copied[
        "split_result_provenance_sha256"
    ]
    with pytest.raises(ValueError, match="result_provenance|distinct"):
        _validate_pairing_receipt(
            copied,
            split_anchor=SimpleNamespace(**identities["split"]),
            dense_anchor=SimpleNamespace(**identities["dense"]),
            split_config=configs["split"],
            dense_config=configs["dense"],
        )

    tampered_shared = json.loads(json.dumps(receipt))
    tampered_shared["study_provenance_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint|study provenance"):
        _validate_pairing_receipt(
            tampered_shared,
            split_anchor=SimpleNamespace(**identities["split"]),
            dense_anchor=SimpleNamespace(**identities["dense"]),
            split_config=configs["split"],
            dense_config=configs["dense"],
        )


def test_pairing_receipt_rejects_mismatch_and_copied_pair():
    configs = _paired_run_configs()
    shared = {
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 100,
        "evaluator_sha256": "1" * 64,
        "data_sha256": "2" * 64,
        "relation_schema_sha256": "3" * 64,
        "result_schema_sha256": "4" * 64,
        "split_configuration_sha256": canonical_configuration_sha256(
            configs["split"]
        ),
        "dense_configuration_sha256": canonical_configuration_sha256(
            configs["dense"]
        ),
        "split_result_provenance_sha256": "8" * 64,
        "dense_result_provenance_sha256": "9" * 64,
        "configs": configs,
    }
    split = SimpleNamespace(
        **{
            key: value
            for key, value in shared.items()
            if key
            not in {
                "configs",
                "split_configuration_sha256",
                "dense_configuration_sha256",
                "split_result_provenance_sha256",
                "dense_result_provenance_sha256",
            }
        },
        checkpoint_sha256="5" * 64,
        arm="split",
        configuration_sha256=shared["split_configuration_sha256"],
        provenance_sha256=shared["split_result_provenance_sha256"],
    )
    dense = SimpleNamespace(
        **{
            key: value
            for key, value in shared.items()
            if key
            not in {
                "configs",
                "split_configuration_sha256",
                "dense_configuration_sha256",
                "split_result_provenance_sha256",
                "dense_result_provenance_sha256",
            }
        },
        checkpoint_sha256="6" * 64,
        arm="dense",
        configuration_sha256=shared["dense_configuration_sha256"],
        provenance_sha256=shared["dense_result_provenance_sha256"],
    )
    valid = _pairing_receipt(
        split.checkpoint_sha256,
        dense.checkpoint_sha256,
        shared,
    )
    assert _validate_pairing_receipt(
        valid,
        split_anchor=split,
        dense_anchor=dense,
        split_config=configs["split"],
        dense_config=configs["dense"],
    )["receipt_sha256"] == valid["receipt_sha256"]

    mismatched = json.loads(json.dumps(valid))
    mismatched["data_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="study provenance|fingerprint"):
        _validate_pairing_receipt(
            mismatched,
            split_anchor=split,
            dense_anchor=dense,
            split_config=configs["split"],
            dense_config=configs["dense"],
        )

    copied_split = SimpleNamespace(
        **{
            key: value
            for key, value in shared.items()
            if key
            not in {
                "configs",
                "split_configuration_sha256",
                "dense_configuration_sha256",
                "split_result_provenance_sha256",
                "dense_result_provenance_sha256",
            }
        },
        checkpoint_sha256="8" * 64,
        arm="split",
        configuration_sha256=shared["split_configuration_sha256"],
        provenance_sha256=shared["split_result_provenance_sha256"],
    )
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _validate_pairing_receipt(
            valid,
            split_anchor=copied_split,
            dense_anchor=dense,
            split_config=configs["split"],
            dense_config=configs["dense"],
        )

    copied_configuration = json.loads(json.dumps(valid))
    copied_configuration["split_configuration_sha256"] = "9" * 64
    with pytest.raises(
        ValueError,
        match="study provenance|fingerprint|configuration_sha256",
    ):
        _validate_pairing_receipt(
            copied_configuration,
            split_anchor=split,
            dense_anchor=dense,
            split_config=configs["split"],
            dense_config=configs["dense"],
        )


def test_strict_guardrail_report_uses_verified_sources_and_isolates_route(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    split_hash = "5" * 64
    dense_hash = "6" * 64
    configs = _paired_run_configs()
    split_run = tmp_path / "split-run"
    dense_run = tmp_path / "dense-run"
    split_dir = split_run / "evals" / split_hash
    dense_dir = dense_run / "evals" / dense_hash
    split_dir.mkdir(parents=True)
    dense_dir.mkdir(parents=True)
    (split_run / "config.yaml").write_text(json.dumps(configs["split"]))
    (dense_run / "config.yaml").write_text(json.dumps(configs["dense"]))
    shared = {
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 100,
        "evaluator_sha256": "1" * 64,
        "data_sha256": "2" * 64,
        "relation_schema_sha256": "3" * 64,
        "result_schema_sha256": "7" * 64,
    }
    split_configuration_sha256 = canonical_configuration_sha256(
        configs["split"]
    )
    dense_configuration_sha256 = canonical_configuration_sha256(
        configs["dense"]
    )
    split_provenance_sha256 = "4" * 64
    dense_provenance_sha256 = "8" * 64

    def raw_measurements(factual_on):
        def accuracy(correct, n=100):
            return {"accuracy": correct / n, "correct": correct, "n": n}

        return {
            "within_run_guardrails": {"mask": {"passed": True}},
            "factual_recall": {
                "on": accuracy(factual_on),
                "off": accuracy(1),
            },
            "recognition_store_off": accuracy(0),
            "internal_accuracy": {
                "per_kind": {"rule": accuracy(factual_on)}
            },
            "language": {"bpb": 1.0, "total_utf8_bytes": 1_000},
        }

    split_source = _build_guardrail_source(
        raw_measurements(96),
        {
            **shared,
            "arm": "split",
            "checkpoint_sha256": split_hash,
            "configuration_sha256": split_configuration_sha256,
            "provenance_sha256": split_provenance_sha256,
        },
    )
    dense_source = _build_guardrail_source(
        raw_measurements(97),
        {
            **shared,
            "arm": "dense",
            "checkpoint_sha256": dense_hash,
            "configuration_sha256": dense_configuration_sha256,
            "provenance_sha256": dense_provenance_sha256,
        },
    )
    (split_dir / "guardrail-source.json").write_bytes(
        evaluator.canonical_json_bytes(split_source)
    )
    (dense_dir / "guardrail-source.json").write_bytes(
        evaluator.canonical_json_bytes(dense_source)
    )

    def summary(arm, checkpoint, mode, control):
        metrics = {
            "per_hop": {"1": {"action": _rate_dict(80, 100)}},
            "exact_action_path": _rate_dict(96, 100),
            "gold_path_answer_accuracy": _rate_dict(
                95 if control == "gold_path" else 0,
                100 if control == "gold_path" else 0,
            ),
        }
        return SimpleNamespace(
            **shared,
            arm=arm,
            checkpoint_sha256=checkpoint,
            configuration_sha256=(
                split_configuration_sha256
                if arm == "split"
                else dense_configuration_sha256
            ),
            provenance_sha256=(
                split_provenance_sha256
                if arm == "split"
                else dense_provenance_sha256
            ),
            memory_mode=mode,
            control_id=control,
            metrics=metrics,
        )

    split_summaries = {
        (EvalMode.MEMORY_ON, ControlID.CORRECT): summary(
            "split", split_hash, "memory_on", "correct"
        ),
        (EvalMode.MEMORY_OFF, ControlID.CORRECT): summary(
            "split", split_hash, "memory_off", "correct"
        ),
        (EvalMode.MEMORY_ON, ControlID.GOLD_PATH): summary(
            "split", split_hash, "memory_on", "gold_path"
        ),
    }
    dense_summaries = {
        (EvalMode.MEMORY_ON, ControlID.CORRECT): summary(
            "dense", dense_hash, "memory_on", "correct"
        ),
        (EvalMode.MEMORY_OFF, ControlID.CORRECT): summary(
            "dense", dense_hash, "memory_off", "correct"
        ),
    }
    for mode, control in _evaluation_matrix():
        split_summaries.setdefault(
            (mode, control),
            summary(
                "split",
                split_hash,
                mode.value,
                control.value,
            ),
        )
        dense_summaries.setdefault(
            (mode, control),
            summary(
                "dense",
                dense_hash,
                mode.value,
                control.value,
            ),
        )
    split_matrix = evaluator._ValidatedCheckpointMatrix(
        summaries=split_summaries,
        guardrail_artifact=split_source,
        guardrail_artifact_sha256=hashlib.sha256(
            evaluator.canonical_json_bytes(split_source)
        ).hexdigest(),
        matrix_manifest_sha256="8" * 64,
        control_transformations_sha256="9" * 64,
        identity_sha256=hashlib.sha256(
            evaluator.canonical_json_bytes(
                {
                    field: getattr(
                        split_summaries[
                            (EvalMode.MEMORY_ON, ControlID.CORRECT)
                        ],
                        field,
                    )
                    for field in evaluator._MATRIX_IDENTITY_FIELDS
                }
            )
        ).hexdigest(),
        checkpoint_dir=split_dir,
    )
    dense_matrix = evaluator._ValidatedCheckpointMatrix(
        summaries=dense_summaries,
        guardrail_artifact=dense_source,
        guardrail_artifact_sha256=hashlib.sha256(
            evaluator.canonical_json_bytes(dense_source)
        ).hexdigest(),
        matrix_manifest_sha256="a" * 64,
        control_transformations_sha256="b" * 64,
        identity_sha256=hashlib.sha256(
            evaluator.canonical_json_bytes(
                {
                    field: getattr(
                        dense_summaries[
                            (EvalMode.MEMORY_ON, ControlID.CORRECT)
                        ],
                        field,
                    )
                    for field in evaluator._MATRIX_IDENTITY_FIELDS
                }
            )
        ).hexdigest(),
        checkpoint_dir=dense_dir,
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_exact_checkpoint_matrix",
        lambda path: (
            split_matrix if Path(path) == split_dir else dense_matrix
        ),
    )
    (split_dir / "guardrail-source.json").write_text(
        '{"mutated_after_validation":true}\n'
    )

    with pytest.raises(ValueError, match="pairing receipt"):
        build_confirmatory_guardrail_report(split_dir, dense_dir)
    receipt = _pairing_receipt(
        split_hash,
        dense_hash,
        {
            **shared,
            "split_configuration_sha256": split_configuration_sha256,
            "dense_configuration_sha256": dense_configuration_sha256,
            "split_result_provenance_sha256": (
                split_provenance_sha256
            ),
            "dense_result_provenance_sha256": (
                dense_provenance_sha256
            ),
            "configs": configs,
        },
    )
    receipt_path = tmp_path / "pairing-receipt.json"
    receipt_path.write_bytes(evaluator.canonical_json_bytes(receipt))
    report = build_confirmatory_guardrail_report(
        split_dir,
        dense_dir,
        receipt_path,
    )

    assert isinstance(report, GuardrailReport)
    assert report.confirmatory_passed
    assert report.pairing_receipt_sha256 == receipt["receipt_sha256"]
    assert "route" not in report.to_dict()
    with pytest.raises(ValueError):
        GuardrailReport.from_dict(split_source)

    route = _build_exploratory_route_artifact(
        {"within_run_guardrails": {"route": {"route_rate": {}}}},
        {
            **shared,
            "arm": "selective",
            "checkpoint_sha256": "7" * 64,
            "configuration_sha256": "e" * 64,
            "provenance_sha256": "f" * 64,
        },
    )
    assert route["excluded_from_confirmatory_verdict"] is True
    assert "guards" not in route
    with pytest.raises(ValueError):
        GuardrailReport.from_dict(route)


def test_authentic_dual_arm_report_build_publish_and_analyzer_integration(
    tmp_path,
):
    import scripts.run_relational_evals as evaluator
    from evals.checkpoint_binding import load_run_configuration
    from scripts.analyze_relational import _validate_bound_guardrail_report

    configs = _paired_run_configs()
    result_schema = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "relational-result-v1.schema.json"
    )
    result_schema_sha256 = hashlib.sha256(
        result_schema.read_bytes()
    ).hexdigest()
    data_sha256 = "2" * 64
    relation_schema_sha256 = "3" * 64
    checkpoint_hashes = {"split": "5" * 64, "dense": "6" * 64}
    runs = {}
    identities = {}
    for arm in ("split", "dense"):
        run = tmp_path / f"{arm}-run"
        run.mkdir()
        (run / "config.yaml").write_text(json.dumps(configs[arm]))
        cfg, config_identity = load_run_configuration(run)
        configs[arm] = cfg
        runs[arm] = run
        identities[arm] = evaluator._result_identity(
            cfg=cfg,
            state={"step": 10},
            checkpoint_hash=checkpoint_hashes[arm],
            config_identities=config_identity,
            data_hash=data_sha256,
            relation_schema_hash=relation_schema_sha256,
            result_schema_hash=result_schema_sha256,
        )
    assert (
        identities["split"]["provenance_sha256"]
        != identities["dense"]["provenance_sha256"]
    )

    compact_records = {
        ControlID.SHUFFLED_RETURNS: _transformation_record(
            seed=77,
            provenance_id="study:world:17",
        ),
        ControlID.ENTITY_RENAME: _transformation_record(
            control_id=ControlID.ENTITY_RENAME.value,
            seed=77,
            provenance_id="study:world:17",
            source="e" * 64,
            transformed="f" * 64,
            changed="1" * 64,
            returns=hashlib.sha256(b"[]").hexdigest(),
            entities="2" * 64,
        ),
        ControlID.GRAPH_ISOMORPHISM: _transformation_record(
            control_id=ControlID.GRAPH_ISOMORPHISM.value,
            seed=77,
            provenance_id="study:world:17",
            source="3" * 64,
            transformed="4" * 64,
            changed="5" * 64,
            returns=hashlib.sha256(b"[]").hexdigest(),
            entities="6" * 64,
        ),
    }

    def integration_row(identity, mode, control, variant):
        relation_path_hash = "a" * 64
        compact = compact_records.get(control)
        changed_addresses = (
            [[17, "r0", "out"]]
            if control
            in {
                ControlID.RELEVANT_EDGE,
                ControlID.IRRELEVANT_EDGE,
                ControlID.EXPLICIT_MISS,
            }
            else []
        )
        oracle_before = (
            "r0" if control == ControlID.RELEVANT_EDGE else "r1"
        )
        oracle_after = (
            None
            if control == ControlID.EXPLICIT_MISS
            else "r1"
        )
        oracle_effect = (
            "changed"
            if control == ControlID.RELEVANT_EDGE
            else "miss"
            if control == ControlID.EXPLICIT_MISS
            else "unchanged"
        )
        value = {
            "record_type": "eval_row",
            "schema_version": 1,
            "qid": f"{control.value}-{variant}",
            "pair_id": f"pair-{control.value}",
            "variant": variant,
            "task": "path_composition",
            "world_id": 17,
            "provenance_id": "study:world:17",
            "relation_path_hash": relation_path_hash,
            "template_id": "path_composition:v1",
            "composition_split": "seen",
            "hop": 1,
            **identity,
            "memory_mode": mode.value,
            "control_id": control.value,
            "cluster_id": cluster_id_for(
                seed=identity["seed"],
                world_id=17,
                relation_path_hash=relation_path_hash,
                template_id="path_composition:v1",
            ),
            "prediction": "r1",
            "answer": "r1",
            "correct": True,
            "prediction_source": "model",
            "all_actions": [
                [0, "r0", "out", True, False],
                [0, "r0", "out", False, True],
                *[
                    [0, "r0", "out", False, False]
                    for _ in range(4)
                ],
            ],
            "gold_all_actions": [
                [0, "r0", "out", True, False],
                [0, "r0", "out", False, True],
                *[
                    [0, "r0", "out", False, False]
                    for _ in range(4)
                ],
            ],
            "returned_addresses": [
                [17, "r0", "out"],
                None,
                None,
                None,
                None,
                None,
            ],
            "gold_addresses": [[17, "r0", "out"]],
            "correct_referents": [True],
            "misses": 0,
            "malformed": 0,
            "abstained": False,
            "excess_reads": 0,
            "halt_step": 2,
            "answer_logits": [[-0.1] for _ in range(6)],
            "lookup_latency_ns": (
                31 if mode == EvalMode.MEMORY_ON else 0
            ),
            "lookup_count": (
                1 if mode == EvalMode.MEMORY_ON else 0
            ),
            "store_rows": 10,
            "store_bytes": 1000,
            "control_seed": 77,
            "transformation_id": (
                None if compact is None else compact["transformation_id"]
            ),
            "source_store_sha256": (
                None if compact is None else compact["source_store_sha256"]
            ),
            "transformed_store_sha256": (
                None
                if compact is None
                else compact["transformed_store_sha256"]
            ),
            "transformation_metadata_sha256": (
                None
                if compact is None
                else compact["transformation_metadata_sha256"]
            ),
            "changed_addresses": (
                [] if compact is not None else changed_addresses
            ),
            "oracle_before": oracle_before,
            "oracle_after": oracle_after,
            "oracle_effect": oracle_effect,
            "edit_locality_correct": (
                True
                if control
                in {
                    ControlID.RELEVANT_EDGE,
                    ControlID.IRRELEVANT_EDGE,
                }
                else None
            ),
        }
        return EvalRow.from_dict(value)

    def guardrail_measurements():
        def accuracy(correct, n=100):
            return {"accuracy": correct / n, "correct": correct, "n": n}

        return {
            "within_run_guardrails": {"mask": {"passed": True}},
            "factual_recall": {
                "on": accuracy(96),
                "off": accuracy(1),
            },
            "recognition_store_off": accuracy(0, 20),
            "internal_accuracy": {
                "per_kind": {"rule": accuracy(80)}
            },
            "language": {"bpb": 1.0, "total_utf8_bytes": 1_000},
        }

    matrices = {}
    for arm in ("split", "dense"):
        run = runs[arm]
        identity = identities[arm]
        summaries = {
            mode.value: {} for mode in EvalMode
        }
        for mode, control in _evaluation_matrix():
            rows = [
                integration_row(identity, mode, control, variant)
                for variant in ("original", "counterfactual")
            ]
            summary = _checkpoint_summary(rows)
            publish_evaluation(run, rows, summary)
            summaries[mode.value][control.value] = summary.to_dict()
        checkpoint_dir = run / "evals" / checkpoint_hashes[arm]
        transformations = (
            checkpoint_dir / "control-transformations.jsonl"
        )
        transformations.write_text(
            "".join(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for record in sorted(
                    compact_records.values(),
                    key=lambda item: item["transformation_id"],
                )
            )
        )
        source = _build_guardrail_source(
            guardrail_measurements(),
            identity,
        )
        source_path = checkpoint_dir / "guardrail-source.json"
        source_path.write_bytes(canonical_json_bytes(source))
        evaluator._write_exact_matrix_manifest(
            run,
            checkpoint_hashes[arm],
            summaries,
            transformation_count=len(compact_records),
            guardrail_artifact=source_path,
            guardrail_record_type="guardrail_source",
        )
        matrices[arm] = _validate_exact_checkpoint_matrix(checkpoint_dir)

    receipt = _pairing_receipt(
        checkpoint_hashes["split"],
        checkpoint_hashes["dense"],
        {
            **{
                key: identities["split"][key]
                for key in (
                    "model_id",
                    "seed",
                    "raw_token_count",
                    "evaluator_sha256",
                    "data_sha256",
                    "relation_schema_sha256",
                    "result_schema_sha256",
                )
            },
            "split_configuration_sha256": identities["split"][
                "configuration_sha256"
            ],
            "dense_configuration_sha256": identities["dense"][
                "configuration_sha256"
            ],
            "split_result_provenance_sha256": identities["split"][
                "provenance_sha256"
            ],
            "dense_result_provenance_sha256": identities["dense"][
                "provenance_sha256"
            ],
            "configs": configs,
        },
    )
    receipt_path = runs["split"] / "evals" / "pairing-receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    report = build_confirmatory_guardrail_report(
        matrices["split"].checkpoint_dir,
        matrices["dense"].checkpoint_dir,
        receipt_path,
    )
    assert report.split_result_provenance_sha256 == identities["split"][
        "provenance_sha256"
    ]
    assert report.dense_result_provenance_sha256 == identities["dense"][
        "provenance_sha256"
    ]
    assert (
        report.split_result_provenance_sha256
        != report.dense_result_provenance_sha256
    )
    report_path = publish_confirmatory_guardrail_report(
        matrices["split"].checkpoint_dir,
        matrices["dense"].checkpoint_dir,
        "guardrail-report.json",
        pairing_receipt=receipt_path,
    )
    assert report_path == runs["split"] / "evals" / "guardrail-report.json"

    report_content = report_path.read_bytes()
    published_report = GuardrailReport.from_dict(
        json.loads(report_content)
    )
    split_run = {
        "cfg": configs["split"],
        "on": matrices["split"][
            (EvalMode.MEMORY_ON, ControlID.CORRECT)
        ],
        "matrix": matrices["split"],
        "pairing_receipt": receipt,
        "pairing_receipt_content": receipt_path.read_bytes(),
        "pairing_receipt_path": receipt_path,
        "guardrail_report": published_report,
        "guardrail_report_content": report_content,
        "guardrail_report_path": report_path,
        "directory": str(runs["split"]),
    }
    dense_run = {
        "cfg": configs["dense"],
        "on": matrices["dense"][
            (EvalMode.MEMORY_ON, ControlID.CORRECT)
        ],
        "matrix": matrices["dense"],
        "directory": str(runs["dense"]),
    }
    assert (
        _validate_bound_guardrail_report(split_run, dense_run)
        == published_report
    )

    copied = published_report.to_dict()
    copied["dense_result_provenance_sha256"] = copied[
        "split_result_provenance_sha256"
    ]
    with pytest.raises(ValueError, match="distinct"):
        GuardrailReport.from_dict(copied)

    tampered = published_report.to_dict()
    tampered["study_provenance_sha256"] = "f" * 64
    tampered_report = GuardrailReport.from_dict(tampered)
    tampered_content = canonical_json_bytes(tampered_report)
    report_path.write_bytes(tampered_content)
    split_run["guardrail_report"] = tampered_report
    split_run["guardrail_report_content"] = tampered_content
    with pytest.raises(ValueError, match="study_provenance_sha256"):
        _validate_bound_guardrail_report(split_run, dense_run)


def test_selective_checkpoint_uses_relational_exploratory_policy_only(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    run = tmp_path / "run"
    run.mkdir()
    cfg = {
        "condition": "selective",
        "model": "d160m",
        "seed": 1001,
        "ctx": 128,
        "train_bin": "train.bin",
        "train_mask": "train.mask",
        "train_weights": "train.weights",
        "tokens_per_step": 32,
    }
    (run / "config.yaml").write_text(json.dumps(cfg))
    checkpoint = run / "ckpt.pt"
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "cfg": cfg,
            "step": 2,
        },
        checkpoint,
    )
    state = evaluator.require_claim_bearing_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="claim-bearing"):
        verify_checkpoint_config(run, state)
    verified, identities = verify_checkpoint_config(
        run,
        state,
        policy=CheckpointValidationPolicy.RELATIONAL_EXPLORATORY,
    )
    assert verified["condition"] == "selective"
    assert identities["condition"] == "selective"

    class DummyGPT:
        def __init__(self, _cfg):
            self.loaded = None

        def load_state_dict(self, value):
            self.loaded = value

        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluator, "GPT", DummyGPT)
    _, loaded_cfg, _, _, _, _ = _load_model(run, "ckpt.pt", "cpu")
    assert loaded_cfg["condition"] == "selective"

    route = _build_exploratory_route_artifact(
        {"within_run_guardrails": {"route": {"passed": True}}},
        {
            "model_id": "d160m",
            "arm": "selective",
            "seed": 1001,
            "checkpoint_sha256": "7" * 64,
            "raw_token_count": 64,
            "evaluator_sha256": "1" * 64,
            "data_sha256": "2" * 64,
            "relation_schema_sha256": "3" * 64,
            "configuration_sha256": "5" * 64,
            "result_schema_sha256": "6" * 64,
            "provenance_sha256": "4" * 64,
        },
    )
    assert route["analysis_role"] == "exploratory_only"
    with pytest.raises(ValueError):
        GuardrailReport.from_dict(route)
    with pytest.raises(ValueError, match="Selective"):
        _build_guardrail_source(
            {"within_run_guardrails": {"route": {}}},
            {
                "arm": "selective",
            },
        )


def _guardrail_publish_dirs(tmp_path):
    split = tmp_path / "split-run" / "evals" / ("1" * 64)
    dense = tmp_path / "dense-run" / "evals" / ("2" * 64)
    split.mkdir(parents=True)
    dense.mkdir(parents=True)
    return split, dense


def test_guardrail_report_publisher_is_atomic_and_non_overwriting(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    split, dense = _guardrail_publish_dirs(tmp_path)
    output = split.parent / "guardrail-report.json"
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    calls = []

    def build(*_args, **_kwargs):
        calls.append(True)
        return object()

    monkeypatch.setattr(
        evaluator,
        "build_confirmatory_guardrail_report",
        build,
    )
    monkeypatch.setattr(
        evaluator,
        "canonical_json_bytes",
        lambda _value: b'{"strict":true}\n',
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_guardrail_publication_environment",
        lambda *_args, **_kwargs: None,
    )

    assert publish_confirmatory_guardrail_report(
        split,
        dense,
        "guardrail-report.json",
    ) == output
    assert output.read_bytes() == b'{"strict":true}\n'
    assert len(calls) == 2
    assert not list(split.parent.glob(".guardrail-report.*"))

    with pytest.raises(FileExistsError, match="already exists"):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
        )
    assert output.read_bytes() == b'{"strict":true}\n'

    for invalid in (
        output,
        dense.parent / "guardrail-report.json",
        "../guardrail-report.json",
        "nested/guardrail-report.json",
    ):
        with pytest.raises(ValueError, match="canonical|relative"):
            publish_confirmatory_guardrail_report(split, dense, invalid)

    output.unlink()
    external = tmp_path / "external-guardrail-report.json"
    external.write_text("attacker")
    output.symlink_to(external)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
        )
    assert output.is_symlink()
    assert external.read_text() == "attacker"

    output.unlink()
    aliased_run = tmp_path / "aliased-split-run"
    aliased_run.symlink_to(split.parent.parent, target_is_directory=True)
    aliased_split = aliased_run / "evals" / split.name
    with pytest.raises(ValueError, match="canonical|symlink"):
        publish_confirmatory_guardrail_report(
            aliased_split,
            dense,
            "guardrail-report.json",
        )


def test_guardrail_report_publisher_binds_run_configs_and_dependency_hashes(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    split, dense = _guardrail_publish_dirs(tmp_path)
    split_cfg = {
        "condition": "split",
        "model": "d160m",
        "seed": 1001,
        "load": "n800k",
    }
    dense_cfg = {**split_cfg, "condition": "dense"}
    (split.parent.parent / "config.yaml").write_text(json.dumps(split_cfg))
    (dense.parent.parent / "config.yaml").write_text(json.dumps(dense_cfg))
    report = SimpleNamespace(
        model_id="d160m",
        seed=1001,
        split_configuration_sha256="f" * 64,
        dense_configuration_sha256=canonical_configuration_sha256(dense_cfg),
        evaluator_sha256="e" * 64,
        result_schema_sha256="d" * 64,
    )
    monkeypatch.setattr(
        evaluator,
        "build_confirmatory_guardrail_report",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(
        evaluator,
        "canonical_json_bytes",
        lambda _value: b'{"strict":true}\n',
    )
    monkeypatch.setattr(evaluator, "_evaluator_sha256", lambda: "e" * 64)
    monkeypatch.setattr(
        evaluator,
        "_regular_file_sha256",
        lambda _path: "d" * 64,
    )

    with pytest.raises(ValueError, match="configuration"):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
        )
    assert not (split.parent / "guardrail-report.json").exists()


def test_guardrail_report_publisher_preserves_contender_lock_and_cleans_failure(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    split, dense = _guardrail_publish_dirs(tmp_path)
    output = split.parent / "guardrail-report.json"
    lock = split.parent / ".guardrail-report.publish.lock"
    lock.write_text("owner")
    monkeypatch.setattr(
        evaluator,
        "build_confirmatory_guardrail_report",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluator,
        "canonical_json_bytes",
        lambda _value: b'{"strict":true}\n',
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_guardrail_publication_environment",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(FileExistsError):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
        )
    assert lock.read_text() == "owner"
    assert not output.exists()

    lock.unlink()
    monkeypatch.setattr(
        evaluator,
        "_rename_directory_noreplace_between",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected crash")
        ),
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
        )
    assert not output.exists()
    assert not lock.exists()
    assert not list(split.parent.glob(".guardrail-report.*"))


def test_guardrail_report_publisher_fails_closed_on_parent_symlink_swap(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    split, dense = _guardrail_publish_dirs(tmp_path)
    parent = split.parent
    attacker = tmp_path / "attacker"
    displaced = tmp_path / "displaced"
    attacker.mkdir()
    monkeypatch.setattr(
        evaluator,
        "build_confirmatory_guardrail_report",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluator,
        "canonical_json_bytes",
        lambda _value: b'{"strict":true}\n',
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_guardrail_publication_environment",
        lambda *_args, **_kwargs: None,
    )
    original = evaluator._rename_directory_noreplace_between

    def swap_then_rename(*args, **kwargs):
        parent.rename(displaced)
        parent.symlink_to(attacker, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        evaluator,
        "_rename_directory_noreplace_between",
        swap_then_rename,
    )
    with pytest.raises(ValueError, match="parent changed"):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
        )

    assert not (displaced / "guardrail-report.json").exists()
    assert not (attacker / "guardrail-report.json").exists()
    assert not (displaced / ".guardrail-report.publish.lock").exists()
    assert not list(displaced.glob(".guardrail-report.*.tmp"))


@pytest.mark.parametrize("mutation", ["source", "receipt"])
def test_guardrail_report_revalidates_bound_evidence_before_promotion(
    tmp_path,
    monkeypatch,
    mutation,
):
    import scripts.run_relational_evals as evaluator

    split, dense = _guardrail_publish_dirs(tmp_path)
    source = split / "guardrail-source.json"
    receipt = split.parent / "pairing-receipt.json"
    source.write_text("source-v1")
    receipt.write_text("receipt-v1")
    expected = {"source": "source-v1", "receipt": "receipt-v1"}
    mutate = {"enabled": True}

    def build(*_args, **_kwargs):
        if (
            source.read_text() != expected["source"]
            or receipt.read_text() != expected["receipt"]
        ):
            raise RuntimeError("bound evidence changed")
        if mutate["enabled"]:
            mutate["enabled"] = False
            target = source if mutation == "source" else receipt
            target.write_text(f"{mutation}-mutated")
        return object()

    monkeypatch.setattr(
        evaluator,
        "build_confirmatory_guardrail_report",
        build,
    )
    monkeypatch.setattr(
        evaluator,
        "canonical_json_bytes",
        lambda _value: b'{"strict":true}\n',
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_guardrail_publication_environment",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="bound evidence changed"):
        publish_confirmatory_guardrail_report(
            split,
            dense,
            "guardrail-report.json",
            pairing_receipt=receipt,
        )
    output = split.parent / "guardrail-report.json"
    assert not output.exists()
    assert not list(split.parent.glob(".guardrail-report.*"))

    source.write_text(expected["source"])
    receipt.write_text(expected["receipt"])
    assert publish_confirmatory_guardrail_report(
        split,
        dense,
        "guardrail-report.json",
        pairing_receipt=receipt,
    ) == output


def _rate_dict(numerator, denominator):
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def test_evaluator_accepts_six_reads_with_implicit_budget_termination():
    item = _item()
    item.meta["gold_addresses"] = [
        [7 + index, f"r{index}", "out"] for index in range(6)
    ]
    item.meta["gold_actions"] = [
        {
            "source_slot": index % 4,
            "relation_id": f"r{index}",
            "direction": "out",
            "read": True,
            "halt": False,
        }
        for index in range(6)
    ]

    actions = _gold_actions(item)

    assert len(actions) == 6
    assert all(action.read and not action.halt for action in actions)


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


def test_evaluator_store_loader_defaults_to_packed_with_schema_codec(
    tmp_path,
    monkeypatch,
):
    import scripts.run_relational_evals as evaluator

    schema = _relation_schema()
    schema_path = tmp_path / "relation-schema.json"
    store_path = tmp_path / "graph.store"
    schema.write(schema_path)
    row = GraphRow(7, "P31", "out", "entity", "9", (), "world")
    PackedGraphStore.build(store_path, (row,), schema.codec).close()
    real_load = PackedGraphStore.load
    calls = []

    def recording_load(cls, path, codec):
        calls.append((Path(path), codec.sha256()))
        return real_load(path, codec)

    monkeypatch.setattr(
        evaluator.PackedGraphStore,
        "load",
        classmethod(recording_load),
    )
    loaded = evaluator._load_evaluator_store(
        store_path,
        relation_schema_path=schema_path,
    )
    try:
        assert isinstance(loaded, PackedGraphStore)
        assert loaded.lookup(row.address) == row
        assert calls == [(store_path, schema.codec.sha256())]
    finally:
        loaded.close()


def test_evaluator_store_loader_allows_explicit_atomic_fixture(tmp_path):
    import scripts.run_relational_evals as evaluator

    row = GraphRow(7, "P31", "out", "entity", "9", (), "world")
    path = tmp_path / "graph.jsonl"
    AtomicGraphStore((row,)).save(path)

    loaded = evaluator._load_evaluator_store(path, atomic_fixture=True)

    assert isinstance(loaded, GraphStore)
    assert isinstance(loaded, AtomicGraphStore)
    assert loaded.lookup(row.address) == row


def test_evaluator_packed_store_requires_relation_schema(tmp_path):
    import scripts.run_relational_evals as evaluator

    with pytest.raises(ValueError, match="relation schema.*required"):
        evaluator._load_evaluator_store(tmp_path / "graph.store")
    with pytest.raises(FileNotFoundError, match="relation schema"):
        evaluator._load_evaluator_store(
            tmp_path / "graph.store",
            relation_schema_path=tmp_path / "missing-schema.json",
        )


def test_evaluator_rejects_schema_codec_mismatch(tmp_path):
    import scripts.run_relational_evals as evaluator

    built_schema = _relation_schema("P31")
    wrong_schema = _relation_schema("P279")
    store_path = tmp_path / "graph.store"
    schema_path = tmp_path / "relation-schema.json"
    PackedGraphStore.build(
        store_path,
        (GraphRow(7, "P31", "out", "entity", "9", (), "world"),),
        built_schema.codec,
    ).close()
    wrong_schema.write(schema_path)

    with pytest.raises(ValueError, match="codec hash mismatch"):
        evaluator._load_evaluator_store(
            store_path,
            relation_schema_path=schema_path,
        )


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
    assert "--atomic-fixtures" in completed.stdout
    assert "--control-seed" in completed.stdout
    assert "--guardrails-json" not in completed.stdout
