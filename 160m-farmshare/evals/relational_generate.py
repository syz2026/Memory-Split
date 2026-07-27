"""Fixed-grammar decoding over an exact relational graph store.

Only the action frame is constrained. The model chooses the source slot,
relation, direction, and READ/NOOP/HALT terminal. Prompt prefill batches contain
equal-length sequences only; no padding token is ever added to an eval prompt.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.graph_trace import serialize_action, serialize_return
from corpusgen.relation_codec import RelationCodec
from corpusgen.srgm_worlds import SRGM_RELATION_CODEC
from organizer.graph_store import GraphStore, StoreStats

N_GRAPH_STEPS = 6


@dataclass
class GraphDecodeState:
    slots: list[int | None]
    actions: list[GraphAction] = field(default_factory=list)
    rows: list[GraphRow | None] = field(default_factory=list)
    provisional_answers: list[str] = field(default_factory=list)
    answer_logits: list[tuple[float, ...]] = field(default_factory=list)
    prediction_source: str = "model"
    lookup_latencies_ns: list[int] = field(
        default_factory=list,
        compare=False,
    )
    misses: int = 0
    halt_step: int | None = None

    def __post_init__(self) -> None:
        if len(self.slots) != 4:
            raise ValueError("exactly four working slots are required")
        if any(slot is not None and slot < 0 for slot in self.slots):
            raise ValueError("working slots must contain non-negative entity ids")
        if self.prediction_source != "model":
            raise ValueError("relational predictions must come from the model")


class OverlayStore:
    """Read-only view replacing exactly one existing base-store row."""

    def __init__(
        self,
        base: GraphStore,
        replacement: GraphRow | Mapping[GraphAddress, GraphRow],
    ) -> None:
        if not isinstance(base, GraphStore):
            raise TypeError("base must implement GraphStore")
        replacements = (
            {replacement.address: replacement}
            if isinstance(replacement, GraphRow)
            else dict(replacement)
        )
        if len(replacements) != 1:
            raise ValueError("overlay must replace exactly one row")
        address, replacement_row = next(iter(replacements.items()))
        if not isinstance(address, GraphAddress) or not isinstance(
            replacement_row, GraphRow
        ):
            raise TypeError("overlay replacements must map addresses to rows")
        if address != replacement_row.address:
            raise ValueError("replacement key must match the row address")
        base_row = base.lookup(address)
        if base_row is None:
            raise ValueError("replacement must target an existing base address")
        if base_row == replacement_row:
            raise ValueError("replacement row must differ from the base row")
        self.base = base
        self.replacement = replacement_row
        self._replacements = replacements
        self.hits = 0
        self.misses = 0

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        replacement = self._replacements.get(address)
        if replacement is not None:
            self.hits += 1
            return replacement
        row = self.base.lookup(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def snapshot_sha256(self) -> str:
        payload = {
            "base_snapshot_sha256": self.base.snapshot_sha256(),
            "replacement": self.replacement.as_json(),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def stats(self) -> StoreStats:
        return self.base.stats()

    def __len__(self) -> int:
        return len(self.base)

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


def parse_action(
    ids: Iterable[int],
    tok,
    codec: RelationCodec,
) -> GraphAction:
    values = [int(value) for value in ids]
    if len(values) != 8:
        raise ValueError("graph actions require eight tokens")
    if values[0] != tok.GRAPH_START or values[-1] != tok.GRAPH_END:
        raise ValueError("invalid graph action frame")
    try:
        source_slot = tok.SLOTS.index(values[1])
        relation_id = codec.decode(values[2:5], tok)
    except ValueError as error:
        raise ValueError("invalid slot or relation token") from error
    if values[5] == tok.DIR_OUT:
        direction = "out"
    elif values[5] == tok.DIR_IN:
        direction = "in"
    else:
        raise ValueError("invalid direction token")
    terminal = values[6]
    if terminal not in (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT):
        raise ValueError("invalid graph terminal token")
    return GraphAction(
        source_slot=source_slot,
        relation_id=relation_id,
        direction=direction,
        read=terminal == tok.GRAPH_READ,
        halt=terminal == tok.GRAPH_HALT,
    )


def apply_action(
    state: GraphDecodeState,
    action: GraphAction,
    store: GraphStore | None,
) -> GraphRow | None:
    """Apply one model-selected action; ``store=None`` is memory OFF."""

    state.actions.append(action)
    if action.halt:
        if state.halt_step is None:
            state.halt_step = len(state.actions)
        state.rows.append(None)
        return None
    if not action.read:
        state.rows.append(None)
        return None

    source_id = state.slots[action.source_slot]
    if source_id is None or store is None:
        state.misses += 1
        state.rows.append(None)
        return None
    started = time.perf_counter_ns()
    row = store.lookup(
        GraphAddress(source_id, action.relation_id, action.direction)
    )
    state.lookup_latencies_ns.append(time.perf_counter_ns() - started)
    if row is None:
        state.misses += 1
    elif row.target_kind == "entity":
        try:
            target_id = int(row.target)
        except ValueError as error:
            raise ValueError("entity graph targets must be integer ids") from error
        if target_id < 0:
            raise ValueError("entity graph targets must be non-negative")
        state.slots[action.source_slot] = target_id
    state.rows.append(row)
    return row


def _apply_forced_return(
    state: GraphDecodeState,
    action: GraphAction,
    row: GraphRow | None,
    validation_store: GraphStore,
    *,
    bind_to_action: bool,
) -> GraphRow | None:
    """Apply a replayed return sealed to its source-store payload."""

    state.actions.append(action)
    if action.halt:
        if row is not None:
            raise ValueError("forced returns for non-read actions must be MISS")
        if state.halt_step is None:
            state.halt_step = len(state.actions)
        state.rows.append(None)
        return None
    if not action.read:
        if row is not None:
            raise ValueError("forced returns for non-read actions must be MISS")
        state.rows.append(None)
        return None

    if bind_to_action:
        source_id = state.slots[action.source_slot]
        if source_id is None:
            if row is not None:
                raise ValueError(
                    "forced return address cannot resolve from an empty slot"
                )
            state.misses += 1
            state.rows.append(None)
            return None
        expected = GraphAddress(
            source_id,
            action.relation_id,
            action.direction,
        )
        if row is not None and row.address != expected:
            raise ValueError(
                "forced return address does not match the dynamic read address"
            )
        stored = validation_store.lookup(expected)
    else:
        stored = (
            None
            if row is None
            else validation_store.lookup(row.address)
        )
    if row != stored:
        raise ValueError(
            "forced return payload does not match the validation store"
        )
    if row is None:
        state.misses += 1
        state.rows.append(None)
        return None
    if not isinstance(row, GraphRow):
        raise TypeError("forced returns must be GraphRow values or None")
    if row.target_kind == "entity":
        if (
            not row.target.isascii()
            or not row.target.isdecimal()
            or int(row.target) < 0
        ):
            raise ValueError(
                "entity graph targets must be canonical non-negative ids"
            )
        state.slots[action.source_slot] = int(row.target)
    state.rows.append(row)
    return row


def _item_value(item, name: str):
    if isinstance(item, dict):
        return item[name]
    return getattr(item, name)


def _item_meta(item) -> dict:
    value = _item_value(item, "meta")
    if not isinstance(value, dict):
        raise ValueError("eval item meta must be a mapping")
    return value


def _resolve_device(model, device) -> torch.device:
    if device is None:
        device = getattr(model, "device", "cpu")
    return torch.device(device)


def _choose(logits: torch.Tensor, allowed: Iterable[int]) -> int:
    choices = tuple(int(value) for value in allowed)
    if not choices:
        raise ValueError("constrained token class must not be empty")
    index = torch.tensor(choices, dtype=torch.long, device=logits.device)
    return choices[int(logits[index].argmax())]


def _step_token(model, token_id: int, cache, device: torch.device):
    value = torch.tensor([[token_id]], dtype=torch.long, device=device)
    logits, cache = model.forward_step(value, cache)
    return logits[0, -1], cache


def _force_tokens(model, ids, cache, device: torch.device):
    logits = None
    for token_id in ids:
        logits, cache = _step_token(model, int(token_id), cache, device)
    if logits is None:
        raise ValueError("cannot force an empty token sequence")
    return logits, cache


def _allowed_action_tokens(
    frame_position: int,
    frame: list[int],
    tok,
    relation_codes: tuple[tuple[int, int, int], ...],
) -> tuple[int, ...]:
    if frame_position == 1:
        return tuple(tok.SLOTS)
    if 2 <= frame_position <= 4:
        relation_position = frame_position - 2
        prefix = tuple(frame[2:frame_position])
        return tuple(
            sorted(
                {
                    code[relation_position]
                    for code in relation_codes
                    if code[:relation_position] == prefix
                }
            )
        )
    if frame_position == 5:
        return tok.DIR_OUT, tok.DIR_IN
    if frame_position == 6:
        return tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT
    if frame_position == 7:
        return (tok.GRAPH_END,)
    raise ValueError("invalid graph action frame position")


def _generate_action(
    model,
    logits,
    cache,
    tok,
    codec: RelationCodec,
    device: torch.device,
):
    del logits  # GRAPH_START is fixed framing, not a semantic model choice.
    ids = [tok.GRAPH_START]
    logits, cache = _step_token(model, ids[-1], cache, device)
    relation_codes = tuple(
        codec.encode(relation_id, tok)
        for relation_id in codec.relation_ids
    )
    for frame_position in range(1, 8):
        allowed = _allowed_action_tokens(
            frame_position,
            ids,
            tok,
            relation_codes,
        )
        token_id = _choose(logits, allowed)
        ids.append(token_id)
        logits, cache = _step_token(model, token_id, cache, device)
    return parse_action(ids, tok, codec), logits, cache


def _encoded_answer_choices(item, tok) -> tuple[tuple[int, ...], ...]:
    meta = _item_meta(item)
    raw = meta["answer_choices"]
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("answer_choices must contain at least two choices")
    choices = tuple(tuple(tok.encode(str(choice))) for choice in raw)
    if any(not choice for choice in choices):
        raise ValueError("answer choices must encode to at least one token")
    if len(set(choices)) != len(choices):
        raise ValueError("answer choices must have unique tokenizations")
    for index, choice in enumerate(choices):
        for other_index, other in enumerate(choices):
            if index != other_index and len(choice) < len(other):
                if other[: len(choice)] == choice:
                    raise ValueError(
                        "answer choice tokenizations must not be prefixes"
                    )
    return choices


def _generate_answer_choice(
    model,
    logits: torch.Tensor,
    cache,
    choices: tuple[tuple[int, ...], ...],
    tok,
    device: torch.device,
):
    active = list(choices)
    generated: list[int] = []
    chosen_logits: list[float] = []
    while True:
        position = len(generated)
        allowed = sorted({choice[position] for choice in active})
        token_id = _choose(logits, allowed)
        chosen_logits.append(float(logits[token_id].item()))
        generated.append(token_id)
        logits, cache = _step_token(model, token_id, cache, device)
        active = [
            choice
            for choice in active
            if choice[: len(generated)] == tuple(generated)
        ]
        if not active:
            raise AssertionError("constrained answer trie lost every choice")
        complete = [choice for choice in active if len(choice) == len(generated)]
        if complete:
            if len(active) != 1:
                raise AssertionError("ambiguous answer-choice prefix")
            return (
                tok.decode(generated).strip(),
                tuple(chosen_logits),
                logits,
                cache,
            )


def _return_ids(row: GraphRow | None, tok) -> list[int]:
    segments = serialize_return(row, "eval" if row is not None else None)
    ids, _, _ = tok.encode_tagged_segments(segments)
    return ids


def _canonical_noop(codec: RelationCodec) -> GraphAction:
    return GraphAction(
        source_slot=0,
        relation_id=codec.relation_ids[0],
        direction="out",
        read=False,
        halt=False,
    )


def _validate_forced_actions(
    forced_actions,
    codec: RelationCodec,
) -> tuple[GraphAction, ...] | None:
    if forced_actions is None:
        return None
    actions = tuple(forced_actions)
    if len(actions) != N_GRAPH_STEPS:
        raise ValueError("forced action traces require exactly six actions")
    if any(not isinstance(action, GraphAction) for action in actions):
        raise TypeError("forced actions must be GraphAction values")
    unknown = {
        action.relation_id
        for action in actions
        if action.relation_id not in codec.relation_ids
    }
    if unknown:
        raise ValueError(
            f"forced actions use unknown relations: {sorted(unknown)}"
        )
    halt_positions = [
        index for index, action in enumerate(actions) if action.halt
    ]
    if len(halt_positions) > 1:
        raise ValueError("forced action traces allow at most one HALT")
    if halt_positions:
        halt = halt_positions[0]
        if halt == 0 or any(
            not action.read or action.halt for action in actions[:halt]
        ):
            raise ValueError(
                "forced action traces require at least one read before HALT"
            )
        if any(
            action.read or action.halt for action in actions[halt + 1 :]
        ):
            raise ValueError(
                "forced action traces cannot read or halt after HALT"
            )
    elif any(not action.read or action.halt for action in actions):
        raise ValueError(
            "halt-free forced action traces must contain six reads"
        )
    return actions


def _validate_forced_returns(
    forced_returns,
    forced_actions: tuple[GraphAction, ...] | None,
) -> tuple[GraphRow | None, ...] | None:
    if forced_returns is None:
        return None
    rows = tuple(forced_returns)
    if len(rows) != N_GRAPH_STEPS:
        raise ValueError("forced return traces require exactly six returns")
    if any(row is not None and not isinstance(row, GraphRow) for row in rows):
        raise TypeError("forced returns must be GraphRow values or None")
    if forced_actions is not None and any(
        not action.read and row is not None
        for action, row in zip(forced_actions, rows)
    ):
        raise ValueError("forced returns for non-read actions must be MISS")
    return rows


def _decode_prefilled(
    model,
    tok,
    item,
    store: GraphStore | None,
    device: torch.device,
    logits: torch.Tensor,
    cache,
    codec: RelationCodec,
    forced_actions: tuple[GraphAction, ...] | None = None,
    forced_returns: tuple[GraphRow | None, ...] | None = None,
    forced_return_store: GraphStore | None = None,
) -> GraphDecodeState:
    meta = _item_meta(item)
    slots = meta["entity_slots"]
    if not isinstance(slots, list):
        raise ValueError("entity_slots must be a list")
    state = GraphDecodeState(
        [None if value is None else int(value) for value in slots]
    )
    choices = _encoded_answer_choices(item, tok)
    forced_return_index = 0

    with torch.no_grad():
        for step in range(N_GRAPH_STEPS):
            if forced_actions is not None:
                action = forced_actions[step]
                action_ids = serialize_action(action, tok, codec)
                logits, cache = _force_tokens(
                    model, action_ids, cache, device
                )
            elif state.halt_step is None:
                action, logits, cache = _generate_action(
                    model,
                    logits,
                    cache,
                    tok,
                    codec,
                    device,
                )
            else:
                action = _canonical_noop(codec)
                action_ids = serialize_action(action, tok, codec)
                logits, cache = _force_tokens(
                    model, action_ids, cache, device
                )
            if forced_returns is not None:
                if forced_return_store is None:
                    raise ValueError(
                        "forced returns require a validation store"
                    )
                forced_row = None
                if action.read:
                    forced_row = forced_returns[forced_return_index]
                    forced_return_index += 1
                row = _apply_forced_return(
                    state,
                    action,
                    forced_row,
                    forced_return_store,
                    bind_to_action=forced_actions is not None,
                )
            else:
                row = apply_action(state, action, store)
            logits, cache = _force_tokens(
                model, _return_ids(row, tok), cache, device
            )
            logits, cache = _step_token(
                model, tok.ANSWER_STATE, cache, device
            )
            prediction, answer_logits, logits, cache = _generate_answer_choice(
                model, logits, cache, choices, tok, device
            )
            state.provisional_answers.append(prediction)
            state.answer_logits.append(answer_logits)
    return state


def _select_cache(cache, indexes: list[int]):
    if cache is None:
        return None
    selector = getattr(cache, "select_batches", None)
    if callable(selector):
        return selector(indexes)
    selector = getattr(cache, "select_batch", None)
    if callable(selector) and len(indexes) == 1:
        return selector(indexes[0])
    from train.model import KVCache

    if isinstance(cache, KVCache):
        selected = KVCache(len(cache.kv))
        selected.pos = cache.pos
        selected.kv = [
            (
                None if key is None else key[indexes],
                None if value is None else value[indexes],
            )
            for key, value in cache.kv
        ]
        return selected
    if isinstance(cache, torch.Tensor):
        return cache[indexes]
    if isinstance(cache, tuple):
        return tuple(_select_cache(value, indexes) for value in cache)
    if isinstance(cache, list):
        return [_select_cache(value, indexes) for value in cache]
    if isinstance(cache, dict):
        return {
            key: _select_cache(value, indexes)
            for key, value in cache.items()
        }
    raise TypeError(
        "batched forward_step cache must support batch selection"
    )


def _slice_cache(cache, index: int):
    return _select_cache(cache, [index])


@dataclass
class _DecodeBatch:
    indexes: list[int]
    items: list
    stores: list[GraphStore | None]
    states: list[GraphDecodeState]
    choices: list[tuple[tuple[int, ...], ...]]
    forced_actions: list[tuple[GraphAction, ...] | None]
    forced_returns: list[tuple[GraphRow | None, ...] | None]
    forced_return_indexes: list[int]
    validation_stores: list[GraphStore | None]
    logits: torch.Tensor
    cache: object


def _batch_step_tokens(
    model,
    token_ids: list[int],
    cache,
    device: torch.device,
):
    value = torch.tensor(
        token_ids, dtype=torch.long, device=device
    ).unsqueeze(1)
    logits, cache = model.forward_step(value, cache)
    return logits[:, -1], cache


def _batch_actions(
    model,
    group: _DecodeBatch,
    tok,
    codec: RelationCodec,
    device: torch.device,
    step: int,
) -> list[GraphAction]:
    batch = len(group.items)
    frames = [[tok.GRAPH_START] for _ in range(batch)]
    logits, cache = _batch_step_tokens(
        model, [tok.GRAPH_START] * batch, group.cache, device
    )
    noop_ids = serialize_action(_canonical_noop(codec), tok, codec)
    relation_codes = tuple(
        codec.encode(relation_id, tok)
        for relation_id in codec.relation_ids
    )
    for frame_position in range(1, 8):
        token_ids = []
        for row, (state, forced) in enumerate(
            zip(group.states, group.forced_actions)
        ):
            if forced is not None:
                token_id = serialize_action(
                    forced[step],
                    tok,
                    codec,
                )[frame_position]
            else:
                token_id = (
                    noop_ids[frame_position]
                    if state.halt_step is not None
                    else _choose(
                        logits[row],
                        _allowed_action_tokens(
                            frame_position,
                            frames[row],
                            tok,
                            relation_codes,
                        ),
                    )
                )
            frames[row].append(token_id)
            token_ids.append(token_id)
        logits, cache = _batch_step_tokens(
            model, token_ids, cache, device
        )
    group.logits = logits
    group.cache = cache
    return [parse_action(frame, tok, codec) for frame in frames]


def _subgroup(group: _DecodeBatch, positions: list[int]) -> _DecodeBatch:
    if positions == list(range(len(group.items))):
        return group
    return _DecodeBatch(
        indexes=[group.indexes[position] for position in positions],
        items=[group.items[position] for position in positions],
        stores=[group.stores[position] for position in positions],
        states=[group.states[position] for position in positions],
        choices=[group.choices[position] for position in positions],
        forced_actions=[
            group.forced_actions[position] for position in positions
        ],
        forced_returns=[
            group.forced_returns[position] for position in positions
        ],
        forced_return_indexes=[
            group.forced_return_indexes[position] for position in positions
        ],
        validation_stores=[
            group.validation_stores[position] for position in positions
        ],
        logits=group.logits[positions],
        cache=_select_cache(group.cache, positions),
    )


def _uniform_choice_length(
    choices: tuple[tuple[int, ...], ...],
) -> int | None:
    lengths = {len(choice) for choice in choices}
    return next(iter(lengths)) if len(lengths) == 1 else None


def _partition_returns(
    group: _DecodeBatch,
    return_ids: list[list[int]],
) -> list[tuple[_DecodeBatch, list[list[int]], int | None]]:
    partitions: dict[tuple, list[int]] = defaultdict(list)
    for position, (ids, choices) in enumerate(
        zip(return_ids, group.choices)
    ):
        choice_length = _uniform_choice_length(choices)
        key = (
            (len(ids), choice_length)
            if choice_length is not None
            else (len(ids), "single", position)
        )
        partitions[key].append(position)
    if len(partitions) == 1:
        positions = next(iter(partitions.values()))
        return [
            (
                group,
                [return_ids[position] for position in positions],
                _uniform_choice_length(group.choices[0]),
            )
        ]
    output = []
    for positions in partitions.values():
        subgroup = _subgroup(group, positions)
        output.append(
            (
                subgroup,
                [return_ids[position] for position in positions],
                _uniform_choice_length(subgroup.choices[0]),
            )
        )
    return output


def _feed_equal_sequences(
    model,
    group: _DecodeBatch,
    sequences: list[list[int]],
    device: torch.device,
) -> None:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1 or not sequences or not sequences[0]:
        raise ValueError("batched forced sequences must have equal length")
    logits = group.logits
    cache = group.cache
    for position in range(len(sequences[0])):
        logits, cache = _batch_step_tokens(
            model,
            [sequence[position] for sequence in sequences],
            cache,
            device,
        )
    group.logits = logits
    group.cache = cache


def _batch_uniform_answers(
    model,
    group: _DecodeBatch,
    choice_length: int,
    tok,
    device: torch.device,
) -> list[str]:
    active = [list(choices) for choices in group.choices]
    generated: list[list[int]] = [[] for _ in group.items]
    chosen_logits: list[list[float]] = [[] for _ in group.items]
    logits = group.logits
    cache = group.cache
    for position in range(choice_length):
        token_ids = []
        for row in range(len(group.items)):
            allowed = sorted(
                {choice[position] for choice in active[row]}
            )
            token_id = _choose(logits[row], allowed)
            chosen_logits[row].append(float(logits[row, token_id].item()))
            generated[row].append(token_id)
            token_ids.append(token_id)
            prefix = tuple(generated[row])
            active[row] = [
                choice
                for choice in active[row]
                if choice[: len(prefix)] == prefix
            ]
            if not active[row]:
                raise AssertionError(
                    "constrained answer trie lost every choice"
                )
        logits, cache = _batch_step_tokens(
            model, token_ids, cache, device
        )
    if any(
        len(candidates) != 1
        or len(candidates[0]) != choice_length
        for candidates in active
    ):
        raise AssertionError("answer choices did not resolve uniquely")
    group.logits = logits
    group.cache = cache
    return (
        [tok.decode(ids).strip() for ids in generated],
        [tuple(values) for values in chosen_logits],
    )


def _finish_batch_step(
    model,
    group: _DecodeBatch,
    return_ids: list[list[int]],
    choice_length: int | None,
    tok,
    device: torch.device,
) -> None:
    _feed_equal_sequences(model, group, return_ids, device)
    logits, cache = _batch_step_tokens(
        model,
        [tok.ANSWER_STATE] * len(group.items),
        group.cache,
        device,
    )
    group.logits = logits
    group.cache = cache
    if choice_length is not None:
        predictions, answer_logits = _batch_uniform_answers(
            model, group, choice_length, tok, device
        )
    else:
        if len(group.items) != 1:
            raise AssertionError(
                "variable-length answer choices require a singleton batch"
            )
        prediction, token_logits, logits, cache = _generate_answer_choice(
            model,
            group.logits[0],
            group.cache,
            group.choices[0],
            tok,
            device,
        )
        group.logits = logits.unsqueeze(0)
        group.cache = cache
        predictions = [prediction]
        answer_logits = [token_logits]
    for state, prediction, token_logits in zip(
        group.states,
        predictions,
        answer_logits,
    ):
        state.provisional_answers.append(prediction)
        state.answer_logits.append(token_logits)


def _decode_batch_prefilled(
    model,
    tok,
    items: list,
    stores: list[GraphStore | None],
    indexes: list[int],
    device: torch.device,
    logits: torch.Tensor,
    cache,
    codec: RelationCodec,
    forced_actions: list[tuple[GraphAction, ...] | None],
    forced_returns: list[tuple[GraphRow | None, ...] | None],
    validation_stores: list[GraphStore | None],
) -> dict[int, GraphDecodeState]:
    states = []
    choices = []
    for item in items:
        slots = _item_meta(item)["entity_slots"]
        if not isinstance(slots, list):
            raise ValueError("entity_slots must be a list")
        states.append(
            GraphDecodeState(
                [None if value is None else int(value) for value in slots]
            )
        )
        choices.append(_encoded_answer_choices(item, tok))
    groups = [
        _DecodeBatch(
            indexes=indexes,
            items=items,
            stores=stores,
            states=states,
            choices=choices,
            forced_actions=forced_actions,
            forced_returns=forced_returns,
            forced_return_indexes=[0 for _ in items],
            validation_stores=validation_stores,
            logits=logits,
            cache=cache,
        )
    ]
    with torch.no_grad():
        for step in range(N_GRAPH_STEPS):
            next_groups = []
            for group in groups:
                actions = _batch_actions(
                    model,
                    group,
                    tok,
                    codec,
                    device,
                    step,
                )
                returned = []
                for position, (
                    state,
                    action,
                    store,
                    forced_action,
                    forced,
                    validation_store,
                ) in enumerate(zip(
                    group.states,
                    actions,
                    group.stores,
                    group.forced_actions,
                    group.forced_returns,
                    group.validation_stores,
                )):
                    if forced is None:
                        returned.append(apply_action(state, action, store))
                    else:
                        assert validation_store is not None
                        forced_row = None
                        if action.read:
                            forced_row = forced[
                                group.forced_return_indexes[position]
                            ]
                            group.forced_return_indexes[position] += 1
                        returned.append(
                            _apply_forced_return(
                                state,
                                action,
                                forced_row,
                                validation_store,
                                bind_to_action=forced_action is not None,
                            )
                        )
                encoded_returns = [
                    _return_ids(row, tok) for row in returned
                ]
                for subgroup, sequences, choice_length in _partition_returns(
                    group, encoded_returns
                ):
                    _finish_batch_step(
                        model,
                        subgroup,
                        sequences,
                        choice_length,
                        tok,
                        device,
                    )
                    next_groups.append(subgroup)
            groups = next_groups
    return {
        index: state
        for group in groups
        for index, state in zip(group.indexes, group.states)
    }


def decode_item(
    model,
    tok,
    item,
    store=None,
    device=None,
    codec: RelationCodec | None = None,
    forced_actions=None,
    forced_returns=None,
    forced_return_store=None,
) -> GraphDecodeState:
    """Decode one item with memory ON (store) or OFF (``store=None``)."""

    resolved_device = _resolve_device(model, device)
    resolved_codec = SRGM_RELATION_CODEC if codec is None else codec
    validated_actions = _validate_forced_actions(
        forced_actions,
        resolved_codec,
    )
    validated_returns = _validate_forced_returns(
        forced_returns,
        validated_actions,
    )
    validation_store = (
        store if forced_return_store is None else forced_return_store
    )
    if validated_returns is not None:
        if not isinstance(validation_store, GraphStore):
            raise ValueError(
                "forced returns require a GraphStore validation source"
            )
    elif forced_return_store is not None:
        raise ValueError(
            "forced_return_store is only valid with forced_returns"
        )
    prompt_ids = tok.encode(str(_item_value(item, "prompt")))
    if not prompt_ids:
        raise ValueError("eval prompt must encode to at least one token")
    prompt = torch.tensor(
        [prompt_ids], dtype=torch.long, device=resolved_device
    )
    with torch.no_grad():
        logits, cache = model.forward_step(prompt, None)
    return _decode_prefilled(
        model,
        tok,
        item,
        store,
        resolved_device,
        logits[0, -1],
        cache,
        resolved_codec,
        validated_actions,
        validated_returns,
        validation_store,
    )


def decode_items(
    model,
    tok,
    items,
    store=None,
    device=None,
    batch_size: int = 32,
    codec: RelationCodec | None = None,
    forced_actions=None,
    forced_returns=None,
    forced_return_store=None,
) -> list[GraphDecodeState]:
    """Decode in stable order using only equal-length prompt prefill batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    materialized = list(items)
    if not materialized:
        return []
    resolved_device = _resolve_device(model, device)
    resolved_codec = SRGM_RELATION_CODEC if codec is None else codec
    store_for_item: Callable = (
        store if callable(store) else lambda _item: store
    )

    def replay_for_item(value, item):
        if value is None:
            return None
        if callable(value):
            return value(item)
        if isinstance(value, Mapping):
            qid = str(_item_value(item, "qid"))
            if qid not in value:
                raise ValueError(f"forced replay is missing qid {qid}")
            return value[qid]
        return value

    resolved_stores = [store_for_item(item) for item in materialized]
    resolved_actions = [
        _validate_forced_actions(
            replay_for_item(forced_actions, item),
            resolved_codec,
        )
        for item in materialized
    ]
    resolved_returns = [
        _validate_forced_returns(
            replay_for_item(forced_returns, item),
            actions,
        )
        for item, actions in zip(materialized, resolved_actions)
    ]
    resolved_validation_stores: list[GraphStore | None] = []
    for index, item in enumerate(materialized):
        explicit_validation_store = replay_for_item(
            forced_return_store,
            item,
        )
        validation_store = (
            resolved_stores[index]
            if explicit_validation_store is None
            else explicit_validation_store
        )
        if resolved_returns[index] is not None:
            if not isinstance(validation_store, GraphStore):
                raise ValueError(
                    "forced returns require a GraphStore validation source"
                )
        elif explicit_validation_store is not None:
            raise ValueError(
                "forced_return_store is only valid with forced_returns"
            )
        resolved_validation_stores.append(validation_store)

    encoded: list[list[int]] = []
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(materialized):
        prompt_ids = tok.encode(str(_item_value(item, "prompt")))
        if not prompt_ids:
            raise ValueError("eval prompt must encode to at least one token")
        encoded.append(prompt_ids)
        buckets[len(prompt_ids)].append(index)

    results: list[GraphDecodeState | None] = [None] * len(materialized)
    for prompt_length in sorted(buckets):
        indexes = buckets[prompt_length]
        for start in range(0, len(indexes), batch_size):
            chunk_indexes = indexes[start : start + batch_size]
            prompt = torch.tensor(
                [encoded[index] for index in chunk_indexes],
                dtype=torch.long,
                device=resolved_device,
            )
            with torch.no_grad():
                logits, cache = model.forward_step(prompt, None)
            decoded = _decode_batch_prefilled(
                model,
                tok,
                [materialized[index] for index in chunk_indexes],
                [
                    resolved_stores[index] for index in chunk_indexes
                ],
                chunk_indexes,
                resolved_device,
                logits[:, -1],
                cache,
                resolved_codec,
                [resolved_actions[index] for index in chunk_indexes],
                [resolved_returns[index] for index in chunk_indexes],
                [
                    resolved_validation_stores[index]
                    for index in chunk_indexes
                ],
            )
            for item_index, state in decoded.items():
                results[item_index] = state
    if any(result is None for result in results):
        raise AssertionError("decoder failed to produce every result")
    return [result for result in results if result is not None]


decode_graph_item = decode_item
decode_graph_items = decode_items
