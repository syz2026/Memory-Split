#!/usr/bin/env python
"""Evaluate one standard-GPT run in both graph-memory modes."""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.relation_codec import RelationCodec
from corpusgen.relation_schema import RelationSchema
from corpusgen.records import QAItem
from corpusgen.srgm_worlds import SRGM_RELATION_CODEC
from evals.checkpoint_binding import (
    CheckpointValidationPolicy,
    checkpoint_sha256,
    load_run_configuration,
    require_claim_bearing_checkpoint,
    resolve_run_checkpoint,
    verify_checkpoint_config,
    verify_checkpoint_unchanged,
)
from evals.relational_contracts import (
    CheckpointSummary,
    EvalRow,
    GuardrailReport,
    StreamingEvaluationPublisher,
    _directory_entry_matches,
    _directory_flags,
    _open_or_create_directory_at,
    _path_matches_directory,
    _rename_directory_noreplace_between,
    canonical_json_bytes,
    cluster_id_for,
    rows_sha256,
    validate_published_evaluation,
)
from evals.relational_controls import (
    ControlID,
    ControlView,
    EvalMode,
    build_control_view,
)
from evals.relational_generate import (
    GraphDecodeState,
    OverlayStore,
    decode_items,
)
from evals.relational_metrics import (
    EXPECTED_TASKS,
    PREREGISTERED_REASONING_MILESTONES,
    assert_expected_counts,
    compute_checkpoint_metrics,
    counterfactual_pair_accuracy,
    exact_accuracy,
    evaluate_confirmatory_guardrails,
    first_frozen_milestone_crossings,
    mask_ledger_guardrail,
    measure_shared_text_bpb,
    path_diagnostics,
    path_metrics,
    recognition_accuracy,
    route_guardrails,
    score_choice_loglikelihoods,
)
from evals.relational_pairing import (
    PairingReceipt,
    build_pairing_receipt,
    publish_pairing_receipt,
    validate_pairing_receipt,
)
from evals.scorers import normalize_answer
from organizer.graph_store import AtomicGraphStore, GraphStore
from organizer.packed_graph_store import PackedGraphStore
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    found = False
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            found = True
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL row {line_number}: {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            yield value
    if not found:
        raise ValueError(f"JSONL input is empty: {path}")


def _read_jsonl(path: Path) -> list[dict]:
    return list(_iter_jsonl(path))


def _load_relation_codec(path: str | Path) -> RelationCodec:
    schema_path = Path(path)
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"relation schema is required for packed graph stores: "
            f"{schema_path}"
        )
    return RelationSchema.from_path(schema_path).codec


def _load_evaluator_store(
    path: str | Path,
    *,
    relation_schema_path: str | Path | None = None,
    codec: RelationCodec | None = None,
    atomic_fixture: bool = False,
) -> GraphStore:
    store_path = Path(path)
    if not isinstance(atomic_fixture, bool):
        raise TypeError("atomic_fixture must be Boolean")
    if atomic_fixture:
        return AtomicGraphStore.load(store_path)
    if codec is not None and relation_schema_path is not None:
        raise ValueError("provide codec or relation schema path, not both")
    if codec is None:
        if relation_schema_path is None:
            raise ValueError(
                "relation schema path is required for packed graph stores"
            )
        codec = _load_relation_codec(relation_schema_path)
    return PackedGraphStore.load(store_path, codec)


def _load_eval_items(data_dir: Path, expected_pairs: int) -> list[QAItem]:
    originals = [
        QAItem(**row)
        for row in _read_jsonl(data_dir / "eval" / "original.jsonl")
    ]
    counterfactuals = [
        QAItem(**row)
        for row in _read_jsonl(
            data_dir / "eval" / "counterfactual.jsonl"
        )
    ]
    items = originals + counterfactuals
    rows_by_task = {
        task: [item.__dict__ for item in items if item.task == task]
        for task in EXPECTED_TASKS
    }
    assert_expected_counts(rows_by_task, expected_pairs)
    return items


def _iter_eval_item_batches(
    data_dir: Path,
    expected_pairs: int,
    *,
    batch_pairs: int,
):
    """Yield bounded complete-pair batches from the two committed streams."""

    if (
        isinstance(expected_pairs, bool)
        or not isinstance(expected_pairs, int)
        or expected_pairs <= 0
    ):
        raise ValueError("expected_pairs must be positive")
    if (
        isinstance(batch_pairs, bool)
        or not isinstance(batch_pairs, int)
        or batch_pairs <= 0
    ):
        raise ValueError("batch_pairs must be positive")
    original_rows = _iter_jsonl(data_dir / "eval" / "original.jsonl")
    counterfactual_rows = _iter_jsonl(
        data_dir / "eval" / "counterfactual.jsonl"
    )
    sentinel = object()
    counts: Counter[str] = Counter()
    batch: list[QAItem] = []
    for original_raw, counterfactual_raw in zip_longest(
        original_rows,
        counterfactual_rows,
        fillvalue=sentinel,
    ):
        if original_raw is sentinel or counterfactual_raw is sentinel:
            raise ValueError(
                "original and counterfactual item streams have different lengths"
            )
        original = QAItem(**original_raw)
        counterfactual = QAItem(**counterfactual_raw)
        original_meta = _item_meta(original)
        counterfactual_meta = _item_meta(counterfactual)
        if (
            original_meta.get("variant") != "original"
            or counterfactual_meta.get("variant") != "counterfactual"
            or original_meta.get("pair_id")
            != counterfactual_meta.get("pair_id")
            or original.task != counterfactual.task
            or original.task not in EXPECTED_TASKS
        ):
            raise ValueError(
                "eval item streams must align as exact counterfactual pairs"
            )
        counts[original.task] += 1
        batch.extend((original, counterfactual))
        if len(batch) == 2 * batch_pairs:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)
    expected = {task: expected_pairs for task in EXPECTED_TASKS}
    if dict(counts) != expected:
        raise ValueError(
            "eval item pair counts mismatch; "
            f"expected={expected}, actual={dict(counts)}"
        )


def _item_value(item, name: str):
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _item_meta(item) -> dict:
    meta = _item_value(item, "meta")
    if not isinstance(meta, dict):
        raise ValueError("eval item meta must be a mapping")
    return meta


def store_for_item(
    base: GraphStore,
    item,
    *,
    memory_on: bool,
) -> GraphStore | None:
    """Return the only evaluator toggle: base/overlay store, or ``None``."""

    if not isinstance(memory_on, bool):
        raise TypeError("memory_on must be Boolean")
    meta = _item_meta(item)
    variant = meta["variant"]
    changed = meta["changed_row"]
    if variant == "original":
        if changed is not None:
            raise ValueError("original eval items cannot contain a changed row")
        selected = base
    elif variant == "counterfactual":
        if not isinstance(changed, dict):
            raise ValueError(
                "counterfactual eval items require one changed row"
            )
        selected = OverlayStore(base, GraphRow.from_json(changed))
    else:
        raise ValueError(f"unexpected eval variant: {variant}")
    return selected if memory_on else None


def _action_json(action: GraphAction) -> list:
    return [
        action.source_slot,
        action.relation_id,
        action.direction,
        action.read,
        action.halt,
    ]


def _gold_actions(item) -> list[GraphAction]:
    meta = _item_meta(item)
    raw_actions = meta["gold_actions"]
    if not isinstance(raw_actions, list) or len(raw_actions) != 6:
        raise ValueError("gold_actions must contain exactly six actions")
    actions = []
    required = {
        "source_slot",
        "relation_id",
        "direction",
        "read",
        "halt",
    }
    for raw in raw_actions:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("gold action fields do not match the contract")
        actions.append(GraphAction(**raw))
    halt_positions = [
        index for index, action in enumerate(actions) if action.halt
    ]
    if len(halt_positions) > 1:
        raise ValueError("gold_actions allow at most one HALT")
    if not halt_positions:
        if not all(action.read for action in actions):
            raise ValueError("gold_actions without HALT must be six READs")
    else:
        halt = halt_positions[0]
        if halt == 0 or not all(
            action.read for action in actions[:halt]
        ):
            raise ValueError(
                "gold actions before HALT must be one to five reads"
            )
        if any(
            action.read or action.halt for action in actions[halt + 1 :]
        ):
            raise ValueError("gold actions after HALT must be NOOP")
    addresses = meta["gold_addresses"]
    read_actions = [action for action in actions if action.read]
    if len(read_actions) != len(addresses):
        raise ValueError("gold action/address counts differ")
    if any(
        action.relation_id != str(address[1])
        or action.direction != str(address[2])
        for action, address in zip(read_actions, addresses)
    ):
        raise ValueError("gold actions do not match gold addresses")
    return actions


def _states_to_rows(items, states: list[GraphDecodeState]) -> list[dict]:
    materialized = list(items)
    if len(materialized) != len(states):
        raise ValueError("every eval item requires exactly one decoded state")
    rows = []
    for item, state in zip(materialized, states):
        if (
            len(state.actions) != 6
            or len(state.rows) != 6
            or len(state.provisional_answers) != 6
        ):
            raise ValueError("every decoded state must contain six steps")
        meta = _item_meta(item)
        gold_all_actions = _gold_actions(item)
        gold_actions = [
            action for action in gold_all_actions if action.read
        ]
        gold_addresses = [
            GraphAddress(int(source), str(relation), direction)
            for source, relation, direction in meta["gold_addresses"]
        ]
        read_pairs = [
            (action, row)
            for action, row in zip(state.actions, state.rows)
            if action.read
        ]
        correct_referents = []
        for index, address in enumerate(gold_addresses):
            returned = (
                read_pairs[index][1] if index < len(read_pairs) else None
            )
            correct_referents.append(
                returned is not None and returned.address == address
            )
        prediction = state.provisional_answers[-1]
        answer = item["answer"] if isinstance(item, dict) else item.answer
        qid = item["qid"] if isinstance(item, dict) else item.qid
        task = item["task"] if isinstance(item, dict) else item.task
        predicted_reads = [action for action, _ in read_pairs]
        rows.append(
            {
                "qid": qid,
                "task": task,
                "pair_id": str(meta["pair_id"]),
                "variant": str(meta["variant"]),
                "correct": (
                    normalize_answer(prediction)
                    == normalize_answer(str(answer))
                ),
                "pred": prediction,
                "answer": answer,
                "actions": [
                    _action_json(action) for action in predicted_reads
                ],
                "all_actions": [
                    _action_json(action) for action in state.actions
                ],
                "gold_actions": [
                    _action_json(action) for action in gold_actions
                ],
                "gold_all_actions": [
                    _action_json(action) for action in gold_all_actions
                ],
                "correct_referents": correct_referents,
                "misses": state.misses,
                # The constrained action grammar cannot emit malformed frames.
                "malformed": 0,
                "excess_reads": max(
                    0, len(predicted_reads) - len(gold_actions)
                ),
                "halt_step": state.halt_step,
                "n_steps": len(state.actions),
                "meta": meta,
            }
        )
    return rows


_RESULT_IDENTITY_FIELDS = {
    "model_id",
    "arm",
    "seed",
    "checkpoint_sha256",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
}


def _states_to_eval_rows(
    views,
    states: list[GraphDecodeState],
    *,
    memory_mode: EvalMode | str,
    identity: dict,
) -> list[EvalRow]:
    """Bind decoded states to strict item/control/checkpoint provenance."""

    materialized = list(views)
    if len(materialized) != len(states):
        raise ValueError("every control view requires one decoded state")
    if not materialized:
        raise ValueError("strict evaluation rows cannot be empty")
    if not isinstance(identity, dict) or set(identity) != _RESULT_IDENTITY_FIELDS:
        raise ValueError("result identity fields do not match the contract")
    try:
        mode = EvalMode(memory_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid memory mode") from exc
    output: list[EvalRow] = []
    for view, state in zip(materialized, states):
        if not isinstance(view, ControlView):
            raise TypeError("strict rows require ControlView values")
        if (
            len(state.actions) != 6
            or len(state.rows) != 6
            or len(state.provisional_answers) != 6
            or len(state.answer_logits) != 6
        ):
            raise ValueError("every decoded state must contain six steps")
        item = view.item
        meta = _item_meta(item)
        gold_all_actions = _gold_actions(item)
        gold_reads = [
            action for action in gold_all_actions if action.read
        ]
        gold_addresses = list(view.oracle_addresses)
        if view.oracle_after is not None and not gold_addresses:
            raise ValueError("complete control oracles require replayed addresses")
        if len(gold_addresses) != len(gold_reads):
            if view.oracle_after is not None:
                raise ValueError("replayed oracle address/read counts differ")
            gold_addresses = [
                GraphAddress(int(source), str(relation), direction)
                for source, relation, direction in meta["gold_addresses"]
            ]
        predicted_reads = [
            (action, row)
            for action, row in zip(state.actions, state.rows)
            if action.read
        ]
        correct_referents = []
        for index, address in enumerate(gold_addresses):
            returned = (
                predicted_reads[index][1]
                if index < len(predicted_reads)
                else None
            )
            correct_referents.append(
                returned is not None and returned.address == address
            )
        prediction = state.provisional_answers[-1]
        answer = (
            str(_item_value(item, "answer"))
            if view.oracle_after is None
            else view.oracle_after
        )
        transformation = view.transformation_record()
        row_value = {
            "record_type": "eval_row",
            "schema_version": 1,
            "qid": str(_item_value(item, "qid")),
            "pair_id": str(meta["pair_id"]),
            "variant": str(meta["variant"]),
            "task": str(_item_value(item, "task")),
            "world_id": meta["world_id"],
            "provenance_id": view.provenance_id,
            "relation_path_hash": str(meta["relation_path_hash"]),
            "template_id": str(meta["template_id"]),
            "composition_split": str(meta["composition_split"]),
            "hop": meta["hop_count"],
            **identity,
            "memory_mode": mode.value,
            "control_id": view.control_id.value,
            "cluster_id": cluster_id_for(
                seed=identity["seed"],
                world_id=meta["world_id"],
                relation_path_hash=str(meta["relation_path_hash"]),
                template_id=str(meta["template_id"]),
            ),
            "prediction": prediction,
            "answer": answer,
            "correct": (
                normalize_answer(prediction)
                == normalize_answer(answer)
            ),
            "prediction_source": state.prediction_source,
            "all_actions": [
                _action_json(action) for action in state.actions
            ],
            "gold_all_actions": [
                _action_json(action) for action in gold_all_actions
            ],
            "returned_addresses": [
                None
                if row is None
                else [
                    row.address.source_id,
                    row.address.relation_id,
                    row.address.direction,
                ]
                for row in state.rows
            ],
            "gold_addresses": [
                [
                    address.source_id,
                    address.relation_id,
                    address.direction,
                ]
                for address in gold_addresses
            ],
            "correct_referents": correct_referents,
            "misses": state.misses,
            "malformed": 0,
            "abstained": not bool(prediction.strip()),
            "excess_reads": max(
                0,
                len(predicted_reads) - len(gold_reads),
            ),
            "halt_step": state.halt_step,
            "answer_logits": [
                list(values) for values in state.answer_logits
            ],
            "lookup_latency_ns": sum(state.lookup_latencies_ns),
            "lookup_count": len(state.lookup_latencies_ns),
            "store_rows": view.source_store_rows,
            "store_bytes": view.source_store_bytes,
            "control_seed": view.seed,
            "transformation_id": view.transformation_id,
            "source_store_sha256": (
                None
                if transformation is None
                else transformation["source_store_sha256"]
            ),
            "transformed_store_sha256": (
                None
                if transformation is None
                else transformation["transformed_store_sha256"]
            ),
            "transformation_metadata_sha256": (
                None
                if transformation is None
                else transformation["transformation_metadata_sha256"]
            ),
            "changed_addresses": [
                [
                    address.source_id,
                    address.relation_id,
                    address.direction,
                ]
                for address in (
                    ()
                    if view.transformation_id is not None
                    else view.changed_addresses
                )
            ],
            "oracle_before": view.oracle_before,
            "oracle_after": view.oracle_after,
            "oracle_effect": view.oracle_effect,
            "edit_locality_correct": None,
        }
        output.append(EvalRow.from_dict(row_value))
    return output


def _checkpoint_summary(
    rows,
    *,
    milestone_crossings: dict[str, int | None] | None = None,
) -> CheckpointSummary:
    materialized = list(rows)
    if not materialized:
        raise ValueError("checkpoint summary requires rows")
    first = materialized[0]
    if not isinstance(first, EvalRow):
        raise TypeError("checkpoint summaries require strict EvalRow values")
    metrics = compute_checkpoint_metrics(
        materialized,
        milestone_crossings=milestone_crossings,
    )
    return CheckpointSummary.from_dict(
        {
            "record_type": "checkpoint_summary",
            "schema_version": 1,
            "checkpoint_sha256": first.checkpoint_sha256,
            "model_id": first.model_id,
            "arm": first.arm,
            "seed": first.seed,
            "raw_token_count": first.raw_token_count,
            "memory_mode": first.memory_mode,
            "control_id": first.control_id,
            "evaluator_sha256": first.evaluator_sha256,
            "data_sha256": first.data_sha256,
            "relation_schema_sha256": first.relation_schema_sha256,
            "configuration_sha256": first.configuration_sha256,
            "result_schema_sha256": first.result_schema_sha256,
            "provenance_sha256": first.provenance_sha256,
            "rows_sha256": rows_sha256(materialized),
            "n_rows": len(materialized),
            "n_pairs": len({row.pair_id for row in materialized}),
            "metrics": metrics,
        }
    )


def _bind_edit_locality(
    rows_by_control,
) -> dict[ControlID, list[EvalRow]]:
    """Bind relevant/irrelevant effects to the same correct-control qids."""

    normalized = {
        ControlID(control): list(rows)
        for control, rows in rows_by_control.items()
    }
    required = {
        ControlID.CORRECT,
        ControlID.RELEVANT_EDGE,
        ControlID.IRRELEVANT_EDGE,
    }
    if not required.issubset(normalized):
        raise ValueError("edit locality requires all three control rows")
    indexed = {
        control: {row.qid: row for row in normalized[control]}
        for control in required
    }
    qid_sets = {frozenset(values) for values in indexed.values()}
    if len(qid_sets) != 1 or any(
        len(indexed[control]) != len(normalized[control])
        for control in required
    ):
        raise ValueError("edit locality controls require a one-to-one qid join")
    for qid in sorted(next(iter(qid_sets))):
        baseline = indexed[ControlID.CORRECT][qid]
        for control in (
            ControlID.RELEVANT_EDGE,
            ControlID.IRRELEVANT_EDGE,
        ):
            candidate = indexed[control][qid]
            shared = (
                "pair_id",
                "variant",
                "task",
                "world_id",
                "provenance_id",
                "relation_path_hash",
                "template_id",
                "composition_split",
                "hop",
                "seed",
                "model_id",
                "arm",
                "checkpoint_sha256",
                "raw_token_count",
                "memory_mode",
                "evaluator_sha256",
                "data_sha256",
                "relation_schema_sha256",
                "configuration_sha256",
                "result_schema_sha256",
                "provenance_sha256",
                "cluster_id",
            )
            if any(
                getattr(baseline, field) != getattr(candidate, field)
                for field in shared
            ):
                raise ValueError(
                    "edit locality controls contain crossed provenance"
                )
            if control == ControlID.RELEVANT_EDGE:
                locality = (
                    candidate.oracle_after is not None
                    and normalize_answer(candidate.prediction)
                    == normalize_answer(candidate.oracle_after)
                    and normalize_answer(candidate.prediction)
                    != normalize_answer(baseline.prediction)
                )
            else:
                locality = (
                    normalize_answer(candidate.prediction)
                    == normalize_answer(baseline.prediction)
                )
            value = candidate.to_dict()
            value["edit_locality_correct"] = locality
            indexed[control][qid] = EvalRow.from_dict(value)
    for control in (
        ControlID.RELEVANT_EDGE,
        ControlID.IRRELEVANT_EDGE,
    ):
        normalized[control] = [
            indexed[control][row.qid] for row in normalized[control]
        ]
    return normalized


class _LocalityIndex:
    """Disk-backed exact qid join for edit-locality controls."""

    _SHARED_FIELDS = (
        "pair_id",
        "variant",
        "task",
        "world_id",
        "provenance_id",
        "relation_path_hash",
        "template_id",
        "composition_split",
        "hop",
        "seed",
        "model_id",
        "arm",
        "checkpoint_sha256",
        "raw_token_count",
        "memory_mode",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "configuration_sha256",
        "result_schema_sha256",
        "provenance_sha256",
        "cluster_id",
    )

    def __init__(self, directory: Path) -> None:
        descriptor, name = tempfile.mkstemp(
            dir=directory,
            prefix=".edit-locality-",
            suffix=".sqlite3",
        )
        os.close(descriptor)
        self._path = Path(name)
        self._connection = sqlite3.connect(self._path)
        self._connection.execute(
            """
            CREATE TABLE baseline (
                memory_mode TEXT NOT NULL,
                qid TEXT NOT NULL,
                payload BLOB NOT NULL,
                PRIMARY KEY(memory_mode, qid)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE seen (
                control_id TEXT NOT NULL,
                memory_mode TEXT NOT NULL,
                qid TEXT NOT NULL,
                PRIMARY KEY(control_id, memory_mode, qid)
            )
            """
        )
        self._closed = False

    @property
    def buffered_rows(self) -> int:
        return 0

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _validated(row: EvalRow) -> EvalRow:
        if not isinstance(row, EvalRow):
            raise TypeError("edit-locality index requires EvalRow values")
        return EvalRow.from_dict(row.to_dict())

    def add_baseline(self, row: EvalRow) -> None:
        value = self._validated(row)
        if value.control_id != ControlID.CORRECT.value:
            raise ValueError("edit-locality baseline must use correct control")
        try:
            self._connection.execute(
                "INSERT INTO baseline(memory_mode, qid, payload) VALUES (?, ?, ?)",
                (
                    value.memory_mode,
                    value.qid,
                    canonical_json_bytes(value),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate edit-locality baseline qid") from exc

    def bind(self, row: EvalRow) -> EvalRow:
        candidate = self._validated(row)
        control = ControlID(candidate.control_id)
        if control not in {
            ControlID.RELEVANT_EDGE,
            ControlID.IRRELEVANT_EDGE,
        }:
            raise ValueError(
                "edit-locality candidates must use relevant or irrelevant control"
            )
        found = self._connection.execute(
            """
            SELECT payload FROM baseline
            WHERE memory_mode = ? AND qid = ?
            """,
            (candidate.memory_mode, candidate.qid),
        ).fetchone()
        if found is None:
            raise ValueError("edit-locality candidate has no correct baseline")
        baseline = EvalRow.from_dict(json.loads(found[0]))
        if any(
            getattr(baseline, field) != getattr(candidate, field)
            for field in self._SHARED_FIELDS
        ):
            raise ValueError(
                "edit locality controls contain crossed provenance"
            )
        if control == ControlID.RELEVANT_EDGE:
            locality = (
                candidate.oracle_after is not None
                and normalize_answer(candidate.prediction)
                == normalize_answer(candidate.oracle_after)
                and normalize_answer(candidate.prediction)
                != normalize_answer(baseline.prediction)
            )
        else:
            locality = (
                normalize_answer(candidate.prediction)
                == normalize_answer(baseline.prediction)
            )
        try:
            self._connection.execute(
                """
                INSERT INTO seen(control_id, memory_mode, qid)
                VALUES (?, ?, ?)
                """,
                (control.value, candidate.memory_mode, candidate.qid),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate edit-locality candidate qid") from exc
        value = candidate.to_dict()
        value["edit_locality_correct"] = locality
        return EvalRow.from_dict(value)

    def require_complete(self, controls: set[ControlID]) -> None:
        baseline = dict(
            self._connection.execute(
                """
                SELECT memory_mode, COUNT(*) FROM baseline
                GROUP BY memory_mode
                """
            )
        )
        if not baseline:
            raise ValueError("edit-locality baseline is empty")
        for control in controls:
            if control not in {
                ControlID.RELEVANT_EDGE,
                ControlID.IRRELEVANT_EDGE,
            }:
                raise ValueError("invalid edit-locality completion control")
            seen = dict(
                self._connection.execute(
                    """
                    SELECT memory_mode, COUNT(*) FROM seen
                    WHERE control_id = ?
                    GROUP BY memory_mode
                    """,
                    (control.value,),
                )
            )
            if seen != baseline:
                raise ValueError(
                    f"{control.value} does not exactly cover baseline qids"
                )

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._path.unlink(missing_ok=True)
        self._closed = True


_TRANSFORMATION_RECORD_FIELDS = {
    "record_type",
    "schema_version",
    "control_id",
    "seed",
    "provenance_id",
    "source_store_sha256",
    "transformed_store_sha256",
    "changed_address_count",
    "changed_addresses_sha256",
    "return_sources_sha256",
    "entity_bijection_sha256",
    "transformation_metadata_sha256",
    "transformation_id",
}

_COMPACT_TRANSFORMATION_CONTROLS = {
    ControlID.SHUFFLED_RETURNS.value,
    ControlID.ENTITY_RENAME.value,
    ControlID.GRAPH_ISOMORPHISM.value,
}
_EMPTY_TRANSFORMATION_COMPONENT_SHA256 = hashlib.sha256(b"[]").hexdigest()


def _canonical_record_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_regular_file_at(directory_fd: int, name: str) -> bytes:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
    ):
        raise ValueError("invalid descriptor-relative file name")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "no-follow artifact validation is unsupported on this platform"
        )
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK, errno.ENOTDIR}:
            raise ValueError(
                "validated artifact must be a regular non-symlink file"
            ) from exc
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("validated artifact must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_dev != opened.st_dev
            or entry.st_ino != opened.st_ino
        ):
            raise ValueError("validated artifact changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_record_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_transformation_record(record: object) -> dict:
    if (
        not isinstance(record, Mapping)
        or set(record) != _TRANSFORMATION_RECORD_FIELDS
        or record["record_type"] != "control_transformation"
        or record["schema_version"] != 1
        or record["control_id"] not in _COMPACT_TRANSFORMATION_CONTROLS
        or isinstance(record["seed"], bool)
        or not isinstance(record["seed"], int)
        or record["seed"] < 0
        or not isinstance(record["provenance_id"], str)
        or not record["provenance_id"]
        or isinstance(record["changed_address_count"], bool)
        or not isinstance(record["changed_address_count"], int)
        or record["changed_address_count"] <= 0
    ):
        raise ValueError("invalid control transformation contract")
    value = dict(record)
    for field in (
        "source_store_sha256",
        "transformed_store_sha256",
        "changed_addresses_sha256",
        "return_sources_sha256",
        "entity_bijection_sha256",
        "transformation_metadata_sha256",
        "transformation_id",
    ):
        _require_record_sha256(value[field], field)
    if value["source_store_sha256"] == value["transformed_store_sha256"]:
        raise ValueError("transformed store commitment must change")
    metadata = {
        field: value[field]
        for field in (
            "changed_address_count",
            "changed_addresses_sha256",
            "return_sources_sha256",
            "entity_bijection_sha256",
        )
    }
    expected_metadata = hashlib.sha256(
        _canonical_record_bytes(metadata)
    ).hexdigest()
    if value["transformation_metadata_sha256"] != expected_metadata:
        raise ValueError("control transformation metadata hash mismatch")
    if value["control_id"] == ControlID.SHUFFLED_RETURNS.value:
        if (
            value["return_sources_sha256"]
            == _EMPTY_TRANSFORMATION_COMPONENT_SHA256
            or value["entity_bijection_sha256"]
            != _EMPTY_TRANSFORMATION_COMPONENT_SHA256
        ):
            raise ValueError("invalid shuffled-return transformation metadata")
    elif (
        value["return_sources_sha256"]
        != _EMPTY_TRANSFORMATION_COMPONENT_SHA256
        or value["entity_bijection_sha256"]
        == _EMPTY_TRANSFORMATION_COMPONENT_SHA256
    ):
        raise ValueError("invalid entity transformation metadata")
    payload = {
        key: item
        for key, item in value.items()
        if key != "transformation_id"
    }
    expected_id = hashlib.sha256(
        _canonical_record_bytes(payload)
    ).hexdigest()
    if value["transformation_id"] != expected_id:
        raise ValueError("control transformation id mismatch")
    return value


class _ControlTransformationIndex:
    """Disk-backed, deduplicated table for global transformations."""

    def __init__(self, run_dir: Path) -> None:
        descriptor, name = tempfile.mkstemp(
            dir=run_dir,
            prefix=".control-transformations-",
            suffix=".sqlite3",
        )
        os.close(descriptor)
        self._path = Path(name)
        self._connection = sqlite3.connect(self._path)
        self._connection.execute(
            """
            CREATE TABLE transformations (
                transformation_id TEXT PRIMARY KEY,
                record_json TEXT NOT NULL
            )
            """
        )
        self._closed = False

    def add(self, view: object) -> None:
        record = view.transformation_record()
        if record is None:
            return
        record = _validate_transformation_record(record)
        transformation_id = record["transformation_id"]
        record_json = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = self._connection.execute(
            """
            SELECT record_json FROM transformations
            WHERE transformation_id = ?
            """,
            (transformation_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != record_json:
                raise ValueError("control transformation id collision")
            return
        self._connection.execute(
            """
            INSERT INTO transformations (transformation_id, record_json)
            VALUES (?, ?)
            """,
            (transformation_id, record_json),
        )

    def publish(
        self,
        checkpoint_hash: str,
    ) -> tuple[Path, int]:
        if (
            len(checkpoint_hash) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_hash)
        ):
            raise ValueError("invalid checkpoint_sha256")
        target_dir = self._path.parent / "evals" / checkpoint_hash
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "control-transformations.jsonl"
        rows = self._connection.execute(
            """
            SELECT record_json FROM transformations
            ORDER BY transformation_id
            """
        )
        count = 0
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            for (record_json,) in rows:
                handle.write(record_json)
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(target_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return target, count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        self._path.unlink(missing_ok=True)


class _TransformationReferenceIndex:
    """Streaming exact join between compact rows and transformation records."""

    _COMMITMENT_FIELDS = (
        "control_id",
        "control_seed",
        "provenance_id",
        "source_store_sha256",
        "transformed_store_sha256",
        "transformation_metadata_sha256",
    )

    def __init__(self, _directory: Path | None = None) -> None:
        self._scratch = tempfile.TemporaryDirectory(
            prefix=".transformation-references-"
        )
        self._path = Path(self._scratch.name) / "references.sqlite3"
        try:
            self._connection = sqlite3.connect(self._path)
        except BaseException:
            self._scratch.cleanup()
            raise
        self._connection.executescript(
            """
            CREATE TABLE records (
                transformation_id TEXT PRIMARY KEY,
                control_id TEXT NOT NULL,
                control_seed INTEGER NOT NULL,
                provenance_id TEXT NOT NULL,
                source_store_sha256 TEXT NOT NULL,
                transformed_store_sha256 TEXT NOT NULL,
                transformation_metadata_sha256 TEXT NOT NULL,
                reference_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE row_references (
                qid TEXT NOT NULL,
                control_id TEXT NOT NULL,
                memory_mode TEXT NOT NULL,
                transformation_id TEXT NOT NULL,
                PRIMARY KEY(qid, control_id, memory_mode)
            );
            CREATE TABLE control_views (
                qid TEXT NOT NULL,
                control_id TEXT NOT NULL,
                transformation_id TEXT NOT NULL,
                mode_mask INTEGER NOT NULL,
                PRIMARY KEY(qid, control_id)
            );
            """
        )
        self._closed = False

    @property
    def buffered_rows(self) -> int:
        return 0

    @property
    def closed(self) -> bool:
        return self._closed

    def add_record(self, raw: object) -> None:
        record = _validate_transformation_record(raw)
        try:
            self._connection.execute(
                """
                INSERT INTO records(
                    transformation_id,
                    control_id,
                    control_seed,
                    provenance_id,
                    source_store_sha256,
                    transformed_store_sha256,
                    transformation_metadata_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["transformation_id"],
                    record["control_id"],
                    record["seed"],
                    record["provenance_id"],
                    record["source_store_sha256"],
                    record["transformed_store_sha256"],
                    record["transformation_metadata_sha256"],
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate transformation record") from exc

    def add_row(self, row: object) -> None:
        transformation_id = getattr(row, "transformation_id", None)
        commitments = {
            field: getattr(row, field, None)
            for field in self._COMMITMENT_FIELDS
            if field not in {"control_id", "control_seed", "provenance_id"}
        }
        if transformation_id is None:
            if any(value is not None for value in commitments.values()):
                raise ValueError(
                    "noncompact row contains transformation commitments"
                )
            return
        record = self._connection.execute(
            """
            SELECT
                control_id,
                control_seed,
                provenance_id,
                source_store_sha256,
                transformed_store_sha256,
                transformation_metadata_sha256
            FROM records
            WHERE transformation_id = ?
            """,
            (transformation_id,),
        ).fetchone()
        if record is None:
            raise ValueError("row references a missing transformation record")
        expected = dict(zip(self._COMMITMENT_FIELDS, record, strict=True))
        for field, value in expected.items():
            if getattr(row, field, None) != value:
                raise ValueError(
                    f"row {field} does not match transformation record"
                )
        memory_mode = getattr(row, "memory_mode", None)
        mode_bit = {
            EvalMode.MEMORY_OFF.value: 1,
            EvalMode.MEMORY_ON.value: 2,
        }.get(memory_mode)
        if mode_bit is None:
            raise ValueError("transformation row has invalid memory mode")
        qid = getattr(row, "qid", None)
        if not isinstance(qid, str) or not qid:
            raise ValueError("transformation row requires qid")
        try:
            self._connection.execute(
                """
                INSERT INTO row_references(
                    qid, control_id, memory_mode, transformation_id
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    qid,
                    expected["control_id"],
                    memory_mode,
                    transformation_id,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate transformation row reference") from exc
        existing = self._connection.execute(
            """
            SELECT transformation_id, mode_mask
            FROM control_views
            WHERE qid = ? AND control_id = ?
            """,
            (qid, expected["control_id"]),
        ).fetchone()
        if existing is None:
            self._connection.execute(
                """
                INSERT INTO control_views(
                    qid, control_id, transformation_id, mode_mask
                ) VALUES (?, ?, ?, ?)
                """,
                (qid, expected["control_id"], transformation_id, mode_bit),
            )
        else:
            if existing[0] != transformation_id:
                raise ValueError(
                    "same control view uses crossed transformation IDs "
                    "across memory modes"
                )
            self._connection.execute(
                """
                UPDATE control_views
                SET mode_mask = ?
                WHERE qid = ? AND control_id = ?
                """,
                (existing[1] | mode_bit, qid, expected["control_id"]),
            )
        self._connection.execute(
            """
            UPDATE records
            SET reference_count = reference_count + 1
            WHERE transformation_id = ?
            """,
            (transformation_id,),
        )

    def finalize(self, *, expected_record_count: int) -> None:
        record_count = self._connection.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0]
        if record_count != expected_record_count:
            raise ValueError("control transformation count mismatch")
        orphan = self._connection.execute(
            """
            SELECT transformation_id FROM records
            WHERE reference_count = 0
            LIMIT 1
            """
        ).fetchone()
        if orphan is not None:
            raise ValueError("orphan transformation record")
        incomplete = self._connection.execute(
            """
            SELECT qid FROM control_views
            WHERE mode_mask != 3
            LIMIT 1
            """
        ).fetchone()
        if incomplete is not None:
            raise ValueError(
                "transformation references must cover identical memory modes"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        self._scratch.cleanup()


def _raw_token_count(state: dict, cfg: dict) -> int:
    step = state.get("step")
    tokens_per_step = cfg.get("tokens_per_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if (
        isinstance(tokens_per_step, bool)
        or not isinstance(tokens_per_step, int)
        or tokens_per_step <= 0
    ):
        raise ValueError(
            "run config tokens_per_step must be a positive integer"
        )
    return step * tokens_per_step


def _frozen_checkpoint_multiple(
    raw_token_count: int,
    parameter_count: int,
) -> int:
    if (
        isinstance(raw_token_count, bool)
        or not isinstance(raw_token_count, int)
        or raw_token_count < 0
    ):
        raise ValueError("raw_token_count must be a nonnegative integer")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise ValueError("parameter_count must be a positive integer")
    for multiple in (5, 10, 20):
        requested = multiple * parameter_count
        if abs(raw_token_count - requested) / requested < 0.0002:
            return multiple
    raise ValueError(
        "checkpoint is not a frozen 5x, 10x, or 20x checkpoint "
        "with rounding error strictly below 0.02%"
    )


def _reasoning_milestone_values(metrics: Mapping) -> dict[str, float]:
    values: dict[str, float] = {}
    by_task = metrics.get("by_task")
    if not isinstance(by_task, Mapping):
        raise ValueError("checkpoint metrics require by_task slices")
    for task in EXPECTED_TASKS:
        name = f"{task}_pair_accuracy_0.75"
        task_metrics = by_task.get(task)
        if not isinstance(task_metrics, Mapping):
            raise ValueError(f"checkpoint metrics require task {task}")
        pair = task_metrics.get("pair_accuracy")
        if not isinstance(pair, Mapping):
            raise ValueError(f"task {task} requires pair_accuracy")
        value = pair.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"task {task} pair accuracy must be a finite rate"
            )
        values[name] = float(value)
    if set(values) != set(PREREGISTERED_REASONING_MILESTONES):
        raise ValueError("preregistered milestone metric set mismatch")
    return values


def _validate_control_transformation_table(
    path: Path,
    expected_count: int,
    expected_sha256: str,
    reference_index: _TransformationReferenceIndex,
) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("control transformation table must be regular")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    count = 0
    with os.fdopen(descriptor, "rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid control transformation line {line_number}"
                ) from exc
            validated = _validate_transformation_record(record)
            if line != _canonical_record_bytes(validated) + b"\n":
                raise ValueError(
                    "control transformation table is not canonical"
                )
            reference_index.add_record(validated)
            count += 1
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("control transformation table hash mismatch")
    if count != expected_count:
        raise ValueError("control transformation count mismatch")
    return actual_sha256


def _validate_exact_matrix_cells(
    checkpoint_dir: Path,
    checkpoint_hash: str,
    cells: object,
    reference_index: _TransformationReferenceIndex,
) -> dict[tuple[EvalMode, ControlID], CheckpointSummary]:
    if not isinstance(cells, list) or len(cells) != len(
        _evaluation_matrix()
    ):
        raise ValueError("historical exact matrix cells are incomplete")
    indexed: dict[tuple[str, str], Mapping] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or set(cell) != {
            "memory_mode",
            "control_id",
            "summary_sha256",
            "rows_sha256",
            "manifest_sha256",
        }:
            raise ValueError("historical matrix cell contract is invalid")
        key = (cell["memory_mode"], cell["control_id"])
        if key in indexed:
            raise ValueError("duplicate historical matrix cell")
        indexed[key] = cell
    expected_keys = {
        (mode.value, control.value)
        for mode, control in _evaluation_matrix()
    }
    if set(indexed) != expected_keys:
        raise ValueError("historical exact matrix cell set mismatch")
    summaries: dict[
        tuple[EvalMode, ControlID],
        CheckpointSummary,
    ] = {}
    for mode in EvalMode:
        mode_path = checkpoint_dir / mode.value
        if (
            mode_path.is_symlink()
            or not mode_path.is_dir()
            or {entry.name for entry in mode_path.iterdir()}
            != {control.value for control in ControlID}
        ):
            raise ValueError("historical mode cell set is not exact")
    for mode, control in _evaluation_matrix():
        path = checkpoint_dir / mode.value / control.value
        summary = validate_published_evaluation(
            path,
            row_consumer=reference_index.add_row,
        )
        if (
            summary.checkpoint_sha256 != checkpoint_hash
            or summary.memory_mode != mode.value
            or summary.control_id != control.value
        ):
            raise ValueError("historical cell identity mismatch")
        declared = indexed[(mode.value, control.value)]
        if (
            declared["summary_sha256"]
            != _regular_file_sha256(path / "summary.json")
            or declared["rows_sha256"] != summary.rows_sha256
            or declared["manifest_sha256"]
            != _regular_file_sha256(path / "manifest.json")
        ):
            raise ValueError("historical matrix cell hash mismatch")
        summaries[(mode, control)] = summary
    return summaries


@dataclass(frozen=True)
class _ValidatedCheckpointMatrix(Mapping):
    summaries: Mapping[tuple[EvalMode, ControlID], CheckpointSummary]
    guardrail_artifact: Mapping
    guardrail_artifact_sha256: str
    matrix_manifest_sha256: str
    control_transformations_sha256: str
    identity_sha256: str
    checkpoint_dir: Path

    def __getitem__(self, key):
        return self.summaries[key]

    def __iter__(self):
        return iter(self.summaries)

    def __len__(self) -> int:
        return len(self.summaries)


_MATRIX_IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "model_id",
    "arm",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
)


def _validate_exact_checkpoint_matrix(
    checkpoint_dir: Path,
) -> _ValidatedCheckpointMatrix:
    checkpoint_hash = checkpoint_dir.name
    if (
        checkpoint_dir.is_symlink()
        or not checkpoint_dir.is_dir()
        or len(checkpoint_hash) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_hash)
    ):
        raise ValueError("invalid historical checkpoint directory")
    expected_entries = {
        *(mode.value for mode in EvalMode),
        "control-transformations.jsonl",
        "exact-matrix-manifest.json",
    }
    actual_entries = {entry.name for entry in checkpoint_dir.iterdir()}
    artifact_names = actual_entries & {
        "guardrail-source.json",
        "exploratory-route.json",
    }
    if len(artifact_names) != 1:
        raise ValueError("historical guardrail artifact set is not exact")
    expected_entries.update(artifact_names)
    if actual_entries != expected_entries:
        raise ValueError("historical checkpoint tree is not exact")
    checkpoint_fd = os.open(checkpoint_dir, _directory_flags())
    if not _path_matches_directory(checkpoint_dir, checkpoint_fd):
        os.close(checkpoint_fd)
        raise ValueError("historical checkpoint directory changed")
    try:
        manifest_content = _read_regular_file_at(
            checkpoint_fd,
            "exact-matrix-manifest.json",
        )
    finally:
        os.close(checkpoint_fd)
    manifest = json.loads(manifest_content)
    expected_manifest_fields = {
        "record_type",
        "schema_version",
        "checkpoint_sha256",
        "cell_count",
        "cells",
        "identity",
        "identity_sha256",
        "guardrail_artifact",
        "control_transformations_sha256",
        "control_transformation_count",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_manifest_fields
        or manifest["record_type"] != "exact_evaluation_matrix"
        or manifest["schema_version"] != 2
        or manifest["checkpoint_sha256"] != checkpoint_hash
        or manifest["cell_count"] != len(_evaluation_matrix())
        or manifest_content != canonical_json_bytes(manifest)
        or not isinstance(manifest["control_transformation_count"], int)
        or isinstance(manifest["control_transformation_count"], bool)
        or manifest["control_transformation_count"] < 0
    ):
        raise ValueError("historical exact matrix manifest contract is invalid")
    manifest_identity = manifest["identity"]
    if (
        not isinstance(manifest_identity, Mapping)
        or set(manifest_identity) != set(_MATRIX_IDENTITY_FIELDS)
        or _require_record_sha256(
            manifest["identity_sha256"],
            "historical matrix identity_sha256",
        )
        != hashlib.sha256(
            canonical_json_bytes(manifest_identity)
        ).hexdigest()
    ):
        raise ValueError("historical matrix identity binding is invalid")
    artifact_name = next(iter(artifact_names))
    transformations = checkpoint_dir / "control-transformations.jsonl"
    guardrail_binding = manifest["guardrail_artifact"]
    checkpoint_fd = os.open(checkpoint_dir, _directory_flags())
    if not _path_matches_directory(checkpoint_dir, checkpoint_fd):
        os.close(checkpoint_fd)
        raise ValueError("historical checkpoint directory changed")
    try:
        guardrail_content = _read_regular_file_at(
            checkpoint_fd,
            artifact_name,
        )
    finally:
        os.close(checkpoint_fd)
    if (
        not isinstance(guardrail_binding, Mapping)
        or set(guardrail_binding) != {"path", "record_type", "sha256"}
        or guardrail_binding["path"] != artifact_name
        or hashlib.sha256(guardrail_content).hexdigest()
        != guardrail_binding["sha256"]
    ):
        raise ValueError("historical checkpoint artifact hash mismatch")
    guardrail_value = json.loads(guardrail_content)
    if artifact_name == "guardrail-source.json":
        guardrail_value = _validate_guardrail_source(guardrail_value)
        if guardrail_binding["record_type"] != "guardrail_source":
            raise ValueError("historical guardrail source type mismatch")
    elif (
        guardrail_binding["record_type"] != "exploratory_route_report"
        or guardrail_value.get("record_type")
        != "exploratory_route_report"
        or guardrail_value.get("analysis_role") != "exploratory_only"
        or guardrail_value.get("excluded_from_confirmatory_verdict") is not True
        or "guards" in guardrail_value
        or "confirmatory_passed" in guardrail_value
    ):
        raise ValueError("historical exploratory route artifact is invalid")
    if guardrail_content != canonical_json_bytes(guardrail_value):
        raise ValueError("historical guardrail artifact is not canonical")
    reference_index = _TransformationReferenceIndex()
    try:
        transformations_sha256 = _validate_control_transformation_table(
            transformations,
            manifest["control_transformation_count"],
            manifest["control_transformations_sha256"],
            reference_index,
        )
        summaries = _validate_exact_matrix_cells(
            checkpoint_dir,
            checkpoint_hash,
            manifest["cells"],
            reference_index,
        )
        reference_index.finalize(
            expected_record_count=manifest[
                "control_transformation_count"
            ]
        )
    finally:
        reference_index.close()
    anchor = summaries[(EvalMode.MEMORY_ON, ControlID.CORRECT)]
    for summary in summaries.values():
        if any(
            getattr(summary, field) != getattr(anchor, field)
            for field in _MATRIX_IDENTITY_FIELDS
        ):
            raise ValueError(
                "historical matrix cell immutable identity mismatch"
            )
    expected_identity = {
        field: getattr(anchor, field) for field in _MATRIX_IDENTITY_FIELDS
    }
    if dict(manifest_identity) != expected_identity:
        raise ValueError("historical matrix manifest identity mismatch")
    for field in _MATRIX_IDENTITY_FIELDS:
        if guardrail_value.get(field) != getattr(anchor, field):
            raise ValueError(
                f"historical guardrail artifact {field} mismatch"
            )
    if (
        anchor.arm == "selective"
    ) != (artifact_name == "exploratory-route.json"):
        raise ValueError("historical guardrail artifact role mismatch")
    return _ValidatedCheckpointMatrix(
        summaries=summaries,
        guardrail_artifact=dict(guardrail_value),
        guardrail_artifact_sha256=guardrail_binding["sha256"],
        matrix_manifest_sha256=hashlib.sha256(
            manifest_content
        ).hexdigest(),
        control_transformations_sha256=transformations_sha256,
        identity_sha256=manifest["identity_sha256"],
        checkpoint_dir=checkpoint_dir,
    )


def _bound_guardrail_rate(
    rate: Mapping,
    summary: CheckpointSummary,
) -> dict:
    return {
        "numerator": rate["numerator"],
        "denominator": rate["denominator"],
        "arm": summary.arm,
        "memory_mode": summary.memory_mode,
        "control_id": summary.control_id,
        "checkpoint_sha256": summary.checkpoint_sha256,
        "model_id": summary.model_id,
        "seed": summary.seed,
        "raw_token_count": summary.raw_token_count,
        "evaluator_sha256": summary.evaluator_sha256,
        "data_sha256": summary.data_sha256,
        "relation_schema_sha256": summary.relation_schema_sha256,
        "configuration_sha256": summary.configuration_sha256,
        "result_schema_sha256": summary.result_schema_sha256,
        "provenance_sha256": summary.provenance_sha256,
    }


def _validate_pairing_receipt(
    raw: object,
    *,
    split_anchor: object,
    dense_anchor: object,
    split_config: Mapping,
    dense_config: Mapping,
) -> dict:
    receipt = validate_pairing_receipt(raw)
    expected = build_pairing_receipt(
        split_anchor,
        dense_anchor,
        split_config,
        dense_config,
    )
    if receipt != expected:
        for field in PairingReceipt.__dataclass_fields__:
            if getattr(receipt, field) != getattr(expected, field):
                raise ValueError(f"pairing receipt {field} mismatch")
        raise ValueError("pairing receipt mismatch")
    return receipt.to_dict()


def _load_pairing_receipt(
    path: str | Path | None,
    *,
    split_anchor: object,
    dense_anchor: object,
    split_config: Mapping,
    dense_config: Mapping,
) -> dict:
    if path is None:
        raise ValueError("validated pairing receipt is required")
    receipt_path = Path(path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("pairing receipt must be a regular file")
    content = receipt_path.read_bytes()
    value = _validate_pairing_receipt(
        json.loads(content),
        split_anchor=split_anchor,
        dense_anchor=dense_anchor,
        split_config=split_config,
        dense_config=dense_config,
    )
    if content != canonical_json_bytes(value):
        raise ValueError("pairing receipt is not canonical")
    return value


def publish_checkpoint_pairing_receipt(
    split_checkpoint_dir: str | Path,
    dense_checkpoint_dir: str | Path,
) -> Path:
    """Validate both exact matrices and publish their one canonical receipt."""

    split_dir = Path(split_checkpoint_dir)
    dense_dir = Path(dense_checkpoint_dir)
    split = _validate_exact_checkpoint_matrix(split_dir)
    dense = _validate_exact_checkpoint_matrix(dense_dir)
    split_anchor = split[(EvalMode.MEMORY_ON, ControlID.CORRECT)]
    dense_anchor = dense[(EvalMode.MEMORY_ON, ControlID.CORRECT)]
    split_config, _ = load_run_configuration(split_dir.parent.parent)
    dense_config, _ = load_run_configuration(dense_dir.parent.parent)
    receipt = build_pairing_receipt(
        split_anchor,
        dense_anchor,
        split_config,
        dense_config,
    )
    return publish_pairing_receipt(
        (split_dir.parent / "pairing-receipt.json").absolute(),
        receipt,
    )


def build_confirmatory_guardrail_report(
    split_checkpoint_dir: str | Path,
    dense_checkpoint_dir: str | Path,
    pairing_receipt: str | Path | None = None,
) -> GuardrailReport:
    """Build the sole confirmatory artifact from verified checkpoint trees."""

    split_dir = Path(split_checkpoint_dir)
    dense_dir = Path(dense_checkpoint_dir)
    split = _validate_exact_checkpoint_matrix(split_dir)
    dense = _validate_exact_checkpoint_matrix(dense_dir)
    split_anchor = split[(EvalMode.MEMORY_ON, ControlID.CORRECT)]
    dense_anchor = dense[(EvalMode.MEMORY_ON, ControlID.CORRECT)]
    if split_anchor.arm != "split" or dense_anchor.arm != "dense":
        raise ValueError("guardrail report requires Split and Dense checkpoints")
    for field in (
        "model_id",
        "seed",
        "raw_token_count",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "result_schema_sha256",
    ):
        if getattr(split_anchor, field) != getattr(dense_anchor, field):
            raise ValueError(f"guardrail checkpoint {field} mismatch")
    if not isinstance(split, _ValidatedCheckpointMatrix) or not isinstance(
        dense,
        _ValidatedCheckpointMatrix,
    ):
        raise ValueError("guardrail report requires validated exact matrices")
    split_source = _validate_guardrail_source(split.guardrail_artifact)
    dense_source = _validate_guardrail_source(dense.guardrail_artifact)
    split_config, _ = load_run_configuration(split_dir.parent.parent)
    dense_config, _ = load_run_configuration(dense_dir.parent.parent)
    receipt = _load_pairing_receipt(
        pairing_receipt,
        split_anchor=split_anchor,
        dense_anchor=dense_anchor,
        split_config=split_config,
        dense_config=dense_config,
    )
    split_off = split[(EvalMode.MEMORY_OFF, ControlID.CORRECT)]
    split_gold = split[(EvalMode.MEMORY_ON, ControlID.GOLD_RETURNS)]

    def source_rate(
        source: Mapping,
        name: str,
        summary: CheckpointSummary,
    ) -> dict:
        return _bound_guardrail_rate(
            source["measurements"][name],
            summary,
        )

    dense_off = dense[(EvalMode.MEMORY_OFF, ControlID.CORRECT)]
    measurements = {
        "split_on_exact_recall": source_rate(
            split_source, "factual_on", split_anchor
        ),
        "dense_on_exact_recall": source_rate(
            dense_source, "factual_on", dense_anchor
        ),
        "split_off_exact_recall": source_rate(
            split_source, "factual_off", split_off
        ),
        "split_off_recognition": source_rate(
            split_source, "recognition_off", split_off
        ),
        "split_off_first_hop_accuracy": _bound_guardrail_rate(
            split_off.metrics["per_hop"]["1"]["action"],
            split_off,
        ),
        "split_gold_return_path_accuracy": _bound_guardrail_rate(
            split_gold.metrics["exact_action_path"],
            split_gold,
        ),
        "split_on_path_accuracy": _bound_guardrail_rate(
            split_anchor.metrics["exact_action_path"],
            split_anchor,
        ),
        "split_rule_accuracy": source_rate(
            split_source, "rule", split_anchor
        ),
        "dense_rule_accuracy": source_rate(
            dense_source, "rule", dense_anchor
        ),
        "split_bpb": {
            **split_source["measurements"]["bpb"],
            **{
                key: getattr(split_off, key)
                for key in (
                    "arm",
                    "memory_mode",
                    "control_id",
                    "checkpoint_sha256",
                    "model_id",
                    "seed",
                    "raw_token_count",
                    "evaluator_sha256",
                    "data_sha256",
                    "relation_schema_sha256",
                    "configuration_sha256",
                    "result_schema_sha256",
                    "provenance_sha256",
                )
            },
        },
        "dense_bpb": {
            **dense_source["measurements"]["bpb"],
            **{
                key: getattr(dense_off, key)
                for key in (
                    "arm",
                    "memory_mode",
                    "control_id",
                    "checkpoint_sha256",
                    "model_id",
                    "seed",
                    "raw_token_count",
                    "evaluator_sha256",
                    "data_sha256",
                    "relation_schema_sha256",
                    "configuration_sha256",
                    "result_schema_sha256",
                    "provenance_sha256",
                )
            },
        },
    }
    shared_fields = (
        "model_id",
        "seed",
        "raw_token_count",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "result_schema_sha256",
    )
    source_provenance_matches = all(
        source[field] == getattr(anchor, field)
        for source, anchor in (
            (split_source, split_anchor),
            (dense_source, dense_anchor),
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
            "configuration_sha256",
            "result_schema_sha256",
            "provenance_sha256",
        )
    )
    checkpoint_provenance_matches = all(
        getattr(split_anchor, field) == getattr(dense_anchor, field)
        for field in shared_fields
    )
    pairing_matches = all(
        receipt[field] == expected
        for field, expected in {
            "split_checkpoint_sha256": split_anchor.checkpoint_sha256,
            "dense_checkpoint_sha256": dense_anchor.checkpoint_sha256,
            "model_id": split_anchor.model_id,
            "seed": split_anchor.seed,
            "raw_token_count": split_anchor.raw_token_count,
            "evaluator_sha256": split_anchor.evaluator_sha256,
            "data_sha256": split_anchor.data_sha256,
            "relation_schema_sha256": (
                split_anchor.relation_schema_sha256
            ),
            "split_configuration_sha256": (
                split_anchor.configuration_sha256
            ),
            "dense_configuration_sha256": (
                dense_anchor.configuration_sha256
            ),
            "result_schema_sha256": split_anchor.result_schema_sha256,
            "split_result_provenance_sha256": (
                split_anchor.provenance_sha256
            ),
            "dense_result_provenance_sha256": (
                dense_anchor.provenance_sha256
            ),
        }.items()
    )
    exact_matrices_valid = all(
        isinstance(matrix, _ValidatedCheckpointMatrix)
        and len(matrix) == len(_evaluation_matrix())
        and len(matrix.matrix_manifest_sha256) == 64
        and len(matrix.control_transformations_sha256) == 64
        and matrix.identity_sha256
        == hashlib.sha256(
            canonical_json_bytes(
                {
                    field: getattr(
                        matrix[(EvalMode.MEMORY_ON, ControlID.CORRECT)],
                        field,
                    )
                    for field in _MATRIX_IDENTITY_FIELDS
                }
            )
        ).hexdigest()
        for matrix in (split, dense)
    )
    return evaluate_confirmatory_guardrails(
        {
            "split_checkpoint_sha256": split_anchor.checkpoint_sha256,
            "dense_checkpoint_sha256": dense_anchor.checkpoint_sha256,
            "model_id": split_anchor.model_id,
            "seed": split_anchor.seed,
            "raw_token_count": split_anchor.raw_token_count,
            "evaluator_sha256": split_anchor.evaluator_sha256,
            "data_sha256": split_anchor.data_sha256,
            "relation_schema_sha256": (
                split_anchor.relation_schema_sha256
            ),
            "split_configuration_sha256": (
                split_anchor.configuration_sha256
            ),
            "dense_configuration_sha256": (
                dense_anchor.configuration_sha256
            ),
            "result_schema_sha256": split_anchor.result_schema_sha256,
            "split_result_provenance_sha256": (
                split_anchor.provenance_sha256
            ),
            "dense_result_provenance_sha256": (
                dense_anchor.provenance_sha256
            ),
            "study_provenance_sha256": receipt[
                "study_provenance_sha256"
            ],
            "pairing_receipt_sha256": receipt["receipt_sha256"],
            "split_guardrail_source_sha256": (
                split.guardrail_artifact_sha256
            ),
            "dense_guardrail_source_sha256": (
                dense.guardrail_artifact_sha256
            ),
            "split_matrix_manifest_sha256": split.matrix_manifest_sha256,
            "dense_matrix_manifest_sha256": dense.matrix_manifest_sha256,
            "measurements": measurements,
            "integrity": {
                "mask_ledger": (
                    split_source["integrity"]["mask_ledger"]
                    and dense_source["integrity"]["mask_ledger"]
                ),
                "corpus_pairing": pairing_matches,
                "provenance": (
                    source_provenance_matches
                    and checkpoint_provenance_matches
                ),
                "exact_matrix": exact_matrices_valid,
            },
        }
    )


def _validate_guardrail_publication_environment(
    report: GuardrailReport,
    *,
    split_dir: Path,
    dense_dir: Path,
) -> None:
    for condition, directory, field in (
        ("split", split_dir, "split_configuration_sha256"),
        ("dense", dense_dir, "dense_configuration_sha256"),
    ):
        cfg, identities = load_run_configuration(directory.parent.parent)
        if (
            cfg.get("condition") != condition
            or _model_id(cfg, identities) != report.model_id
            or cfg.get("seed") != report.seed
            or identities["configuration_sha256"] != getattr(report, field)
        ):
            raise ValueError(
                f"{condition} run configuration does not match the report"
            )
    if _evaluator_sha256() != report.evaluator_sha256:
        raise RuntimeError(
            "evaluator dependency identity changed before report publication"
        )
    result_schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "relational-result-v1.schema.json"
    )
    if _regular_file_sha256(result_schema_path) != report.result_schema_sha256:
        raise RuntimeError(
            "result schema identity changed before report publication"
        )


def publish_confirmatory_guardrail_report(
    split_checkpoint_dir: str | Path,
    dense_checkpoint_dir: str | Path,
    output: str | Path,
    *,
    pairing_receipt: str | Path | None = None,
) -> Path:
    requested = Path(output)
    if (
        requested.is_absolute()
        or requested.parts != ("guardrail-report.json",)
    ):
        raise ValueError(
            "guardrail report output must be the canonical relative filename"
        )
    split_dir = Path(split_checkpoint_dir)
    dense_dir = Path(dense_checkpoint_dir)
    for name, directory in (("Split", split_dir), ("Dense", dense_dir)):
        if (
            directory.name in {"", ".", ".."}
            or len(directory.name) != 64
            or any(
                character not in "0123456789abcdef"
                for character in directory.name
            )
            or directory.parent.name != "evals"
            or directory.is_symlink()
            or not directory.is_dir()
            or directory.resolve(strict=True) != directory.absolute()
        ):
            raise ValueError(
                f"{name} checkpoint must be a canonical non-symlink "
                "evaluation directory"
            )
    path = split_dir.parent / requested
    canonical_receipt = split_dir.parent / "pairing-receipt.json"
    receipt_path = (
        canonical_receipt
        if pairing_receipt is None
        else Path(pairing_receipt)
    )
    if (
        receipt_path.is_symlink()
        or receipt_path.absolute() != canonical_receipt.absolute()
    ):
        raise ValueError(
            "pairing receipt must use the canonical Split run location"
        )
    report = build_confirmatory_guardrail_report(
        split_dir,
        dense_dir,
        receipt_path,
    )
    _validate_guardrail_publication_environment(
        report,
        split_dir=split_dir,
        dense_dir=dense_dir,
    )
    content = canonical_json_bytes(report)
    parent_fd: int | None = None
    split_fd: int | None = None
    temporary_name: str | None = None
    lock_owned = False
    promoted = False
    lock_name = ".guardrail-report.publish.lock"
    try:
        parent_fd = os.open(path.parent, _directory_flags())
        if not _path_matches_directory(path.parent, parent_fd):
            raise ValueError(
                "guardrail report parent changed during publication"
            )
        split_fd = os.open(
            split_dir.name,
            _directory_flags(),
            dir_fd=parent_fd,
        )
        try:
            os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"guardrail report already exists: {path}"
            )
        lock_fd = os.open(
            lock_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
            dir_fd=parent_fd,
        )
        lock_owned = True
        os.close(lock_fd)
        temporary_name = (
            f".guardrail-report.{os.getpid()}."
            f"{os.urandom(12).hex()}.tmp"
        )
        temporary_fd = os.open(
            temporary_name,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            remaining = memoryview(content)
            while remaining:
                written = os.write(temporary_fd, remaining)
                if written <= 0:
                    raise OSError("guardrail report write made no progress")
                remaining = remaining[written:]
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        final_report = build_confirmatory_guardrail_report(
            split_dir,
            dense_dir,
            receipt_path,
        )
        _validate_guardrail_publication_environment(
            final_report,
            split_dir=split_dir,
            dense_dir=dense_dir,
        )
        if canonical_json_bytes(final_report) != content:
            raise RuntimeError(
                "guardrail report inputs changed before publication"
            )
        if not (
            _path_matches_directory(path.parent, parent_fd)
            and _directory_entry_matches(
                parent_fd,
                split_dir.name,
                split_fd,
            )
        ):
            raise ValueError(
                "guardrail report parent changed during publication"
            )
        _rename_directory_noreplace_between(
            parent_fd,
            temporary_name,
            parent_fd,
            path.name,
        )
        promoted = True
        if not _path_matches_directory(path.parent, parent_fd):
            os.rename(
                path.name,
                temporary_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            promoted = False
            raise ValueError(
                "guardrail report parent changed during publication"
            )
        try:
            os.fsync(parent_fd)
        except BaseException:
            os.rename(
                path.name,
                temporary_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            promoted = False
            os.fsync(parent_fd)
            raise
    finally:
        if parent_fd is not None:
            if temporary_name is not None and not promoted:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if lock_owned:
                try:
                    os.unlink(lock_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if split_fd is not None:
                os.close(split_fd)
            os.close(parent_fd)
    return path


def _milestone_crossings_for_cell(
    run: Path,
    *,
    checkpoint_hash: str,
    parameter_count: int,
    identity: Mapping,
    memory_mode: EvalMode,
    control_id: ControlID,
    current_metrics: Mapping,
) -> dict[str, int | None]:
    history: list[dict] = []
    eval_root = run / "evals"
    if eval_root.is_symlink():
        raise ValueError("evaluation output root cannot be a symlink")
    if eval_root.exists() and not eval_root.is_dir():
        raise ValueError("evaluation output root must be a directory")
    if eval_root.exists():
        for checkpoint_dir in eval_root.iterdir():
            if checkpoint_dir.is_symlink():
                raise ValueError(
                    "evaluation checkpoint directories cannot be symlinks"
                )
            if (
                not checkpoint_dir.is_dir()
                or checkpoint_dir.name == checkpoint_hash
                or len(checkpoint_dir.name) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in checkpoint_dir.name
                )
            ):
                continue
            summaries = _validate_exact_checkpoint_matrix(checkpoint_dir)
            summary = summaries[(memory_mode, control_id)]
            expected = {
                "model_id": identity["model_id"],
                "arm": identity["arm"],
                "seed": identity["seed"],
                "memory_mode": memory_mode.value,
                "control_id": control_id.value,
                "evaluator_sha256": identity["evaluator_sha256"],
                "data_sha256": identity["data_sha256"],
                "relation_schema_sha256": identity[
                    "relation_schema_sha256"
                ],
                "configuration_sha256": identity["configuration_sha256"],
                "result_schema_sha256": identity["result_schema_sha256"],
                "provenance_sha256": identity["provenance_sha256"],
            }
            if any(
                getattr(summary, field) != value
                for field, value in expected.items()
            ):
                continue
            history.append(
                {
                    "tokens_per_parameter": _frozen_checkpoint_multiple(
                        summary.raw_token_count,
                        parameter_count,
                    ),
                    "raw_token_count": summary.raw_token_count,
                    "metrics": _reasoning_milestone_values(
                        summary.metrics
                    ),
                }
            )
    current_multiple = _frozen_checkpoint_multiple(
        identity["raw_token_count"],
        parameter_count,
    )
    prior_multiples = {
        checkpoint["tokens_per_parameter"] for checkpoint in history
    }
    expected_prior = {
        5: set(),
        10: {5},
        20: {5, 10},
    }[current_multiple]
    if prior_multiples != expected_prior:
        raise ValueError(
            "frozen checkpoints must be evaluated in 5x, 10x, 20x order"
        )
    history.append(
        {
            "tokens_per_parameter": current_multiple,
            "raw_token_count": identity["raw_token_count"],
            "metrics": _reasoning_milestone_values(current_metrics),
        }
    )
    multiples = [
        checkpoint["tokens_per_parameter"] for checkpoint in history
    ]
    if len(set(multiples)) != len(multiples):
        raise ValueError("duplicate frozen checkpoint multiple")
    if set(multiples) == {5, 10, 20}:
        return first_frozen_milestone_crossings(
            history,
            PREREGISTERED_REASONING_MILESTONES,
        )
    ordered = sorted(
        history,
        key=lambda checkpoint: checkpoint["tokens_per_parameter"],
    )
    if any(
        current["raw_token_count"] <= previous["raw_token_count"]
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            "raw token counts must increase across frozen checkpoints"
        )
    return {
        name: next(
            (
                checkpoint["raw_token_count"]
                for checkpoint in ordered
                if checkpoint["metrics"][name] >= threshold
            ),
            None,
        )
        for name, threshold in sorted(
            PREREGISTERED_REASONING_MILESTONES.items()
        )
    }


def _evaluation_matrix() -> tuple[tuple[EvalMode, ControlID], ...]:
    return tuple(
        (mode, control)
        for mode in EvalMode
        for control in ControlID
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _summary(rows: list[dict], expected_pairs: int, memory: str) -> dict:
    rows_by_task = {
        task: [row for row in rows if row["task"] == task]
        for task in EXPECTED_TASKS
    }
    assert_expected_counts(rows_by_task, expected_pairs)
    task_summary = {
        task: {
            "counterfactual_pair_accuracy": counterfactual_pair_accuracy(
                task_rows, expected_pairs=expected_pairs
            ),
            "path": path_metrics(task_rows),
            "path_diagnostics": path_diagnostics(task_rows),
            "n_rows": len(task_rows),
            "n_pairs": expected_pairs,
        }
        for task, task_rows in rows_by_task.items()
    }
    composite = sum(
        task_summary[task]["counterfactual_pair_accuracy"]
        for task in EXPECTED_TASKS
    ) / len(EXPECTED_TASKS)
    return {
        "memory": memory,
        "tasks": task_summary,
        "primary_composite": composite,
        "n_rows": len(rows),
        "n_pairs_per_task": expected_pairs,
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def _mask_audit_from_committed_data(
    data_dir: Path,
    condition: str,
) -> dict:
    ledger = _read_jsonl(data_dir / "mask-ledger.jsonl")
    report = _read_json(data_dir / "report.json")
    expected = Counter(
        (row["start"], row["end"], row["fact_id"])
        for row in ledger
        if row["condition"] == "expected_split"
    )
    split = Counter(
        (row["start"], row["end"], row["fact_id"])
        for row in ledger
        if row["condition"] == "split"
    )
    if split - expected:
        raise ValueError("split ledger contains unexpected payload ranges")
    selected_condition = [
        row for row in ledger if row["condition"] == condition
    ]
    protected_roles = {
        "rule",
        "action",
        "provisional_answer",
        "final_answer",
    }
    if condition == "split":
        unmasked_external = sum((expected - split).values())
    else:
        unmasked_external = sum(expected.values())
    return {
        "unmasked_external_payloads": unmasked_external,
        "external_payload_occurrences": sum(expected.values()),
        "masked_rule_action_answer_targets": sum(
            int(row["length"])
            for row in selected_condition
            if row["role"] in protected_roles
        ),
        "rule_action_answer_targets": int(
            report["masks"]["protected_target_tokens"]
        ),
    }


def produce_guardrail_measurements(
    model,
    tok,
    data_dir: Path | str,
    *,
    condition: str,
    device,
    batch_size: int,
    factual_store: GraphStore,
    codec: RelationCodec,
) -> dict:
    data_dir = Path(data_dir)
    if condition not in ("dense", "split", "random", "selective"):
        raise ValueError(f"unexpected training condition: {condition}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(factual_store, GraphStore):
        raise TypeError("factual_store must implement GraphStore")
    if not isinstance(codec, RelationCodec):
        raise TypeError("codec must be a RelationCodec")

    eval_manifest = _read_json(data_dir / "eval-manifest.json")
    expected_items = int(eval_manifest["guardrail_items"])
    expected_shared = int(eval_manifest["shared_text_items"])
    recognition_items = _read_jsonl(
        data_dir / "eval" / "recognition.jsonl"
    )
    factual_items = [
        QAItem(**row)
        for row in _read_jsonl(data_dir / "eval" / "factual.jsonl")
    ]
    internal_items = _read_jsonl(data_dir / "eval" / "internal.jsonl")
    shared_rows = _read_jsonl(
        data_dir / "eval" / "shared_text.jsonl"
    )
    for name, values, expected in (
        ("recognition", recognition_items, expected_items),
        ("factual", factual_items, expected_items),
        ("internal", internal_items, expected_items),
        ("shared_text", shared_rows, expected_shared),
    ):
        if len(values) != expected:
            raise ValueError(
                f"{name}: expected {expected} items, got {len(values)}"
            )

    choice_score_cache = {}

    def score_choices(prompt, choices):
        key = (prompt, tuple(choices))
        if key not in choice_score_cache:
            choice_score_cache[key] = score_choice_loglikelihoods(
                model,
                tok,
                prompt,
                choices,
                device=device,
            )
        return choice_score_cache[key]

    recognition = recognition_accuracy(
        score_choices,
        recognition_items,
        expected_count=expected_items,
    )
    internal = recognition_accuracy(
        score_choices,
        internal_items,
        expected_count=expected_items,
    )
    internal["per_kind"] = {
        kind: recognition_accuracy(
            score_choices,
            [item for item in internal_items if item["kind"] == kind],
        )
        for kind in ("rule", "central_fact")
    }
    language = measure_shared_text_bpb(
        model,
        tok,
        [row["text"] for row in shared_rows],
        device=device,
    )

    factual_measurements = {}
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            model,
            tok,
            factual_items,
            lambda item, enabled=memory_on: store_for_item(
                factual_store,
                item,
                memory_on=enabled,
            ),
            device=device,
            batch_size=batch_size,
            codec=codec,
        )
        factual_measurements[memory] = exact_accuracy(
            _states_to_rows(factual_items, states),
            expected_count=expected_items,
        )

    route = _read_json(data_dir / "eval" / "route-audit.json")
    within_run = (
        {
            "route": route_guardrails(route),
        }
        if condition == "selective"
        else {
            "mask": mask_ledger_guardrail(
                _mask_audit_from_committed_data(data_dir, condition),
                condition=condition,
            )
        }
    )
    return {
        "within_run_guardrails": within_run,
        "recognition_store_off": recognition,
        "factual_recall": factual_measurements,
        "internal_accuracy": internal,
        "language": language,
    }


def _validate_guardrail_schema(value: dict) -> None:
    required = {
        "within_run_guardrails",
        "recognition_store_off",
        "factual_recall",
        "internal_accuracy",
        "language",
    }
    if set(value) != required:
        raise ValueError(
            "guardrail measurement keys mismatch; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )


def _source_rate(measurement: Mapping, name: str) -> dict[str, int]:
    numerator = measurement.get("correct")
    denominator = measurement.get("n")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or not 0 <= numerator <= denominator
    ):
        raise ValueError(f"{name} has invalid count fields")
    accuracy = measurement.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or float(accuracy) != numerator / denominator
    ):
        raise ValueError(f"{name} accuracy disagrees with counts")
    return {"numerator": numerator, "denominator": denominator}


def _build_guardrail_source(
    measurements: Mapping,
    identity: Mapping,
) -> dict:
    arm = identity["arm"]
    if arm == "selective":
        raise ValueError("Selective cannot produce confirmatory source data")
    if arm not in {"dense", "split", "random"}:
        raise ValueError("invalid confirmatory source arm")
    within = measurements["within_run_guardrails"]
    if not isinstance(within, Mapping) or set(within) != {"mask"}:
        raise ValueError(
            "confirmatory source requires mask integrity and excludes route"
        )
    mask = within["mask"]
    if (
        not isinstance(mask, Mapping)
        or not isinstance(mask.get("passed"), bool)
    ):
        raise ValueError("mask integrity measurement is invalid")
    language = measurements["language"]
    bpb = language.get("bpb")
    byte_count = language.get("total_utf8_bytes")
    if (
        isinstance(bpb, bool)
        or not isinstance(bpb, (int, float))
        or not math.isfinite(float(bpb))
        or float(bpb) <= 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise ValueError("shared-text BPB measurement is invalid")
    source = {
        "record_type": "guardrail_source",
        "schema_version": 1,
        **identity,
        "analysis_role": "confirmatory_source_only",
        "cannot_determine_confirmatory_verdict": True,
        "measurements": {
            "factual_on": _source_rate(
                measurements["factual_recall"]["on"],
                "factual_on",
            ),
            "factual_off": _source_rate(
                measurements["factual_recall"]["off"],
                "factual_off",
            ),
            "recognition_off": _source_rate(
                measurements["recognition_store_off"],
                "recognition_off",
            ),
            "rule": _source_rate(
                measurements["internal_accuracy"]["per_kind"]["rule"],
                "rule",
            ),
            "bpb": {
                "value": float(bpb),
                "denominator": byte_count,
            },
        },
        "integrity": {
            "mask_ledger": mask["passed"],
        },
    }
    return _validate_guardrail_source(source)


def _validate_guardrail_source(raw: Mapping) -> dict:
    fields = {
        "record_type",
        "schema_version",
        "model_id",
        "arm",
        "seed",
        "checkpoint_sha256",
        "raw_token_count",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "configuration_sha256",
        "result_schema_sha256",
        "provenance_sha256",
        "analysis_role",
        "cannot_determine_confirmatory_verdict",
        "measurements",
        "integrity",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("guardrail source fields are not exact")
    if (
        raw["record_type"] != "guardrail_source"
        or raw["schema_version"] != 1
        or raw["analysis_role"] != "confirmatory_source_only"
        or raw["cannot_determine_confirmatory_verdict"] is not True
        or raw["arm"] not in {"dense", "split", "random"}
    ):
        raise ValueError("guardrail source contract is invalid")
    for name in (
        "checkpoint_sha256",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "configuration_sha256",
        "result_schema_sha256",
        "provenance_sha256",
    ):
        value = raw[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"guardrail source {name} is invalid")
    if (
        not isinstance(raw["model_id"], str)
        or not raw["model_id"]
        or isinstance(raw["seed"], bool)
        or not isinstance(raw["seed"], int)
        or raw["seed"] < 0
        or isinstance(raw["raw_token_count"], bool)
        or not isinstance(raw["raw_token_count"], int)
        or raw["raw_token_count"] < 0
    ):
        raise ValueError("guardrail source identity is invalid")
    measurements = raw["measurements"]
    if not isinstance(measurements, Mapping) or set(measurements) != {
        "factual_on",
        "factual_off",
        "recognition_off",
        "rule",
        "bpb",
    }:
        raise ValueError("guardrail source measurement set is invalid")
    for name in ("factual_on", "factual_off", "recognition_off", "rule"):
        value = measurements[name]
        if not isinstance(value, Mapping) or set(value) != {
            "numerator",
            "denominator",
        }:
            raise ValueError(f"guardrail source {name} is invalid")
        numerator = value["numerator"]
        denominator = value["denominator"]
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0
            or not 0 <= numerator <= denominator
        ):
            raise ValueError(f"guardrail source {name} counts are invalid")
    bpb = measurements["bpb"]
    if (
        not isinstance(bpb, Mapping)
        or set(bpb) != {"value", "denominator"}
        or isinstance(bpb["value"], bool)
        or not isinstance(bpb["value"], (int, float))
        or not math.isfinite(float(bpb["value"]))
        or float(bpb["value"]) <= 0
        or isinstance(bpb["denominator"], bool)
        or not isinstance(bpb["denominator"], int)
        or bpb["denominator"] <= 0
    ):
        raise ValueError("guardrail source BPB is invalid")
    integrity = raw["integrity"]
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"mask_ledger"}
        or not isinstance(integrity["mask_ledger"], bool)
    ):
        raise ValueError("guardrail source integrity is invalid")
    return copy.deepcopy(dict(raw))


def _build_exploratory_route_artifact(
    measurements: Mapping,
    identity: Mapping,
) -> dict:
    if identity["arm"] != "selective":
        raise ValueError("route diagnostics are Selective-only")
    within = measurements["within_run_guardrails"]
    if not isinstance(within, Mapping) or set(within) != {"route"}:
        raise ValueError("Selective route diagnostics are incomplete")
    return {
        "record_type": "exploratory_route_report",
        "schema_version": 1,
        **identity,
        "analysis_role": "exploratory_only",
        "excluded_from_confirmatory_verdict": True,
        "route": copy.deepcopy(within["route"]),
    }


def _resolve_data_dir(cfg: dict, override: str | None) -> Path:
    if override is not None:
        return Path(override)
    if "data_dir" in cfg:
        return Path(cfg["data_dir"])
    if "data_rel" in cfg:
        root = os.environ.get("DATA_ROOT")
        if root is None:
            raise ValueError("DATA_ROOT is required when config uses data_rel")
        return Path(root) / cfg["data_rel"]
    raise KeyError("run config requires data_dir or data_rel")


def _load_model(
    run: Path,
    checkpoint: str,
    device: str,
) -> tuple[GPT, dict, Mapping, Path, str, dict]:
    checkpoint_path = resolve_run_checkpoint(run, checkpoint)
    checkpoint_hash = checkpoint_sha256(checkpoint_path)
    state = require_claim_bearing_checkpoint(checkpoint_path)
    embedded_cfg = state.get("cfg")
    condition = (
        embedded_cfg.get("condition")
        if isinstance(embedded_cfg, Mapping)
        else None
    )
    policy = (
        CheckpointValidationPolicy.RELATIONAL_EXPLORATORY
        if condition == "selective"
        else CheckpointValidationPolicy.CLAIM_BEARING
    )
    cfg, config_identities = verify_checkpoint_config(
        run,
        state,
        policy=policy,
    )
    model_value = cfg["model"]
    if isinstance(model_value, str):
        if model_value not in PRESETS:
            raise ValueError(f"unknown model preset: {model_value}")
        model_cfg = replace(PRESETS[model_value])
    elif isinstance(model_value, dict):
        model_cfg = GPTConfig(**model_value)
    else:
        raise ValueError("model config must be a preset name or mapping")
    if "ctx" in cfg:
        model_cfg.ctx = int(cfg["ctx"])
    model = GPT(model_cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return (
        model,
        dict(cfg),
        state,
        checkpoint_path,
        checkpoint_hash,
        config_identities,
    )


def _regular_file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.is_symlink() or not file_path.is_file():
        raise ValueError(f"identity input must be a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluator_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    members = (
        "corpusgen/graph_records.py",
        "corpusgen/graph_trace.py",
        "corpusgen/records.py",
        "corpusgen/relation_codec.py",
        "corpusgen/relation_schema.py",
        "corpusgen/srgm_worlds.py",
        "corpusgen/world_splits.py",
        "evals/checkpoint_binding.py",
        "evals/relational_controls.py",
        "evals/relational_contracts.py",
        "evals/relational_generate.py",
        "evals/relational_metrics.py",
        "evals/relational_pairing.py",
        "evals/scorers.py",
        "organizer/graph_store.py",
        "organizer/packed_graph_store.py",
        "scripts/run_relational_evals.py",
        "schemas/relational-result-v1.schema.json",
        "train/data.py",
        "train/model.py",
        "train/tokenizer.py",
        "train/trainer.py",
    )
    digest = hashlib.sha256()
    for member in members:
        path = root / member
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"evaluator identity member is invalid: {member}")
        digest.update(member.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _data_sha256(
    base_store: GraphStore,
    factual_store: GraphStore,
    data_dir: Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "base_store_sha256": base_store.snapshot_sha256(),
                "factual_store_sha256": factual_store.snapshot_sha256(),
            }
        )
    )
    relative_inputs = (
        "eval-manifest.json",
        "mask-ledger.jsonl",
        "report.json",
        "eval/recognition.jsonl",
        "eval/factual.jsonl",
        "eval/internal.jsonl",
        "eval/shared_text.jsonl",
        "eval/route-audit.json",
        "eval/original.jsonl",
        "eval/counterfactual.jsonl",
        *(f"eval/{task}.jsonl" for task in EXPECTED_TASKS),
    )
    for relative in relative_inputs:
        path = data_dir / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_regular_file_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _model_id(cfg: Mapping, config_identities: Mapping) -> str:
    model = cfg.get("model")
    if isinstance(model, str) and model:
        return model
    return f"custom-{config_identities['model_sha256'][:16]}"


def _result_identity(
    *,
    cfg: Mapping,
    state: Mapping,
    checkpoint_hash: str,
    config_identities: Mapping,
    data_hash: str,
    relation_schema_hash: str,
    result_schema_hash: str,
) -> dict:
    """Return an arm-specific result identity bound to the full config."""

    evaluator_hash = _evaluator_sha256()
    result_provenance_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "result_contract": "relational-result-v1",
                "data_sha256": data_hash,
                "relation_schema_sha256": relation_schema_hash,
                "evaluator_sha256": evaluator_hash,
                "configuration_sha256": config_identities[
                    "configuration_sha256"
                ],
                "result_schema_sha256": result_schema_hash,
            }
        )
    ).hexdigest()
    return {
        "model_id": _model_id(cfg, config_identities),
        "arm": cfg["condition"],
        "seed": cfg["seed"],
        "checkpoint_sha256": checkpoint_hash,
        "raw_token_count": _raw_token_count(state, cfg),
        "evaluator_sha256": evaluator_hash,
        "data_sha256": data_hash,
        "relation_schema_sha256": relation_schema_hash,
        "configuration_sha256": config_identities["configuration_sha256"],
        "result_schema_sha256": result_schema_hash,
        "provenance_sha256": result_provenance_hash,
    }


def _verify_bound_inputs(bindings: Mapping) -> None:
    checkpoint_path = Path(bindings["checkpoint_path"])
    checkpoint_hash = str(bindings["checkpoint_sha256"])
    verify_checkpoint_unchanged(checkpoint_path, checkpoint_hash)
    policy = (
        CheckpointValidationPolicy.RELATIONAL_EXPLORATORY
        if bindings["config"].get("condition") == "selective"
        else CheckpointValidationPolicy.CLAIM_BEARING
    )
    fresh_cfg, fresh_config_identities = verify_checkpoint_config(
        Path(bindings["run"]),
        bindings["checkpoint_state"],
        policy=policy,
    )
    if (
        canonical_json_bytes(fresh_cfg)
        != canonical_json_bytes(bindings["config"])
        or dict(fresh_config_identities)
        != dict(bindings["config_identities"])
    ):
        raise RuntimeError("checkpoint config provenance changed")
    data_hash = _data_sha256(
        bindings["base_store"],
        bindings["factual_store"],
        Path(bindings["data_dir"]),
    )
    if data_hash != bindings["data_sha256"]:
        raise RuntimeError("evaluation data changed during evaluation")
    evaluator_hash = _evaluator_sha256()
    if evaluator_hash != bindings["evaluator_sha256"]:
        raise RuntimeError("evaluator dependency closure changed")
    relation_schema_path = bindings["relation_schema_path"]
    relation_schema_hash = (
        bindings["relation_codec"].sha256()
        if relation_schema_path is None
        else _regular_file_sha256(relation_schema_path)
    )
    if relation_schema_hash != bindings["relation_schema_sha256"]:
        raise RuntimeError("relation schema changed during evaluation")
    result_schema_hash = _regular_file_sha256(
        bindings["result_schema_path"]
    )
    if result_schema_hash != bindings["result_schema_sha256"]:
        raise RuntimeError("result schema changed during evaluation")
    result_provenance_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "result_contract": "relational-result-v1",
                "data_sha256": data_hash,
                "relation_schema_sha256": relation_schema_hash,
                "evaluator_sha256": evaluator_hash,
                "configuration_sha256": bindings["config_identities"][
                    "configuration_sha256"
                ],
                "result_schema_sha256": result_schema_hash,
            }
        )
    ).hexdigest()
    if result_provenance_hash != bindings["provenance_sha256"]:
        raise RuntimeError("evaluation provenance binding changed")


def _verify_and_promote_checkpoint_tree(
    run: Path,
    staging_run: Path,
    checkpoint_hash: str,
    bindings: Mapping,
) -> Path:
    _verify_bound_inputs(bindings)
    return _promote_checkpoint_tree(
        run,
        staging_run,
        checkpoint_hash,
    )


def _preflight_output_matrix(run: Path, checkpoint_hash: str) -> None:
    evals = run / "evals"
    if evals.is_symlink():
        raise ValueError("evaluation output root cannot be a symlink")
    destination = evals / checkpoint_hash
    if os.path.lexists(destination):
        raise FileExistsError(
            f"evaluation output already exists: {destination}"
        )


def _write_checkpoint_guardrail_artifact(
    run: Path,
    checkpoint_hash: str,
    value: Mapping,
) -> Path:
    parent = run / "evals" / checkpoint_hash
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("checkpoint evaluation directory is invalid")
    record_type = value.get("record_type")
    filename = {
        "guardrail_source": "guardrail-source.json",
        "exploratory_route_report": "exploratory-route.json",
    }.get(record_type)
    if filename is None:
        raise ValueError("invalid checkpoint guardrail artifact type")
    path = parent / filename
    content = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=".guardrail-artifact.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                "checkpoint guardrail measurements already exist: "
                f"{path}"
            ) from exc
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_exact_matrix_manifest(
    staging_run: Path,
    checkpoint_hash: str,
    summaries: Mapping[str, Mapping[str, Mapping]],
    *,
    transformation_count: int,
    guardrail_artifact: Path,
    guardrail_record_type: str,
) -> Path:
    expected_modes = {mode.value for mode in EvalMode}
    expected_controls = {control.value for control in ControlID}
    if set(summaries) != expected_modes or any(
        set(summaries[mode]) != expected_controls
        for mode in expected_modes
    ):
        raise ValueError("exact evaluation matrix is incomplete")
    checkpoint = staging_run / "evals" / checkpoint_hash
    anchor = CheckpointSummary.from_dict(
        summaries[EvalMode.MEMORY_ON.value][ControlID.CORRECT.value]
    )
    identity = {
        field: getattr(anchor, field) for field in _MATRIX_IDENTITY_FIELDS
    }
    for mode, control in _evaluation_matrix():
        summary = CheckpointSummary.from_dict(
            summaries[mode.value][control.value]
        )
        if any(
            getattr(summary, field) != identity[field]
            for field in _MATRIX_IDENTITY_FIELDS
        ):
            raise ValueError(
                "staged matrix cells cross immutable result identities"
            )
    cells = []
    for mode, control in _evaluation_matrix():
        cell = checkpoint / mode.value / control.value
        manifest_path = cell / "manifest.json"
        if (
            cell.is_symlink()
            or not cell.is_dir()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise ValueError("staged evaluation cell is incomplete")
        cells.append(
            {
                "memory_mode": mode.value,
                "control_id": control.value,
                "summary_sha256": _regular_file_sha256(
                    cell / "summary.json"
                ),
                "rows_sha256": _regular_file_sha256(
                    cell / "rows.jsonl"
                ),
                "manifest_sha256": _regular_file_sha256(manifest_path),
            }
        )
    transformations = checkpoint / "control-transformations.jsonl"
    manifest = {
        "record_type": "exact_evaluation_matrix",
        "schema_version": 2,
        "checkpoint_sha256": checkpoint_hash,
        "cell_count": len(cells),
        "cells": cells,
        "identity": identity,
        "identity_sha256": hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest(),
        "guardrail_artifact": {
            "path": guardrail_artifact.name,
            "record_type": guardrail_record_type,
            "sha256": _regular_file_sha256(guardrail_artifact),
        },
        "control_transformations_sha256": _regular_file_sha256(
            transformations
        ),
        "control_transformation_count": transformation_count,
    }
    path = checkpoint / "exact-matrix-manifest.json"
    content = canonical_json_bytes(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=checkpoint,
        prefix=".exact-matrix.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(checkpoint, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _promote_checkpoint_tree(
    run: Path,
    staging_run: Path,
    checkpoint_hash: str,
) -> Path:
    if (
        len(checkpoint_hash) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_hash)
    ):
        raise ValueError("invalid checkpoint hash path component")
    staged = staging_run / "evals" / checkpoint_hash
    if staged.is_symlink() or not staged.is_dir():
        raise ValueError("staged checkpoint tree is missing")
    expected_entries = {
        *(mode.value for mode in EvalMode),
        "control-transformations.jsonl",
        "exact-matrix-manifest.json",
    }
    expected_entries.add(
        "exploratory-route.json"
        if (
            staged / "exploratory-route.json"
        ).is_file()
        else "guardrail-source.json"
    )
    if {entry.name for entry in staged.iterdir()} != expected_entries:
        raise ValueError("staged checkpoint tree is not exact")
    destination = run / "evals" / checkpoint_hash
    run_fd: int | None = None
    staging_fd: int | None = None
    staging_evals_fd: int | None = None
    staged_fd: int | None = None
    evals_fd: int | None = None
    lock_owned = False
    lock_name = f".{checkpoint_hash}.publish.lock"
    try:
        run_fd = os.open(run, _directory_flags())
        staging_fd = os.open(staging_run, _directory_flags())
        staging_evals_fd = os.open(
            "evals",
            _directory_flags(),
            dir_fd=staging_fd,
        )
        staged_fd = os.open(
            checkpoint_hash,
            _directory_flags(),
            dir_fd=staging_evals_fd,
        )
        evals_fd = _open_or_create_directory_at(run_fd, "evals")
        try:
            os.stat(
                checkpoint_hash,
                dir_fd=evals_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"evaluation output already exists: {destination}"
            )
        descriptor = os.open(
            lock_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
            dir_fd=evals_fd,
        )
        lock_owned = True
        os.close(descriptor)
        if not (
            _path_matches_directory(run, run_fd)
            and _directory_entry_matches(run_fd, "evals", evals_fd)
            and _path_matches_directory(staging_run, staging_fd)
            and _directory_entry_matches(
                staging_fd,
                "evals",
                staging_evals_fd,
            )
            and _directory_entry_matches(
                staging_evals_fd,
                checkpoint_hash,
                staged_fd,
            )
        ):
            raise ValueError(
                "checkpoint publication parent changed during publication"
            )
        _rename_directory_noreplace_between(
            staging_evals_fd,
            checkpoint_hash,
            evals_fd,
            checkpoint_hash,
        )
        if not (
            _path_matches_directory(run, run_fd)
            and _directory_entry_matches(run_fd, "evals", evals_fd)
            and _path_matches_directory(staging_run, staging_fd)
            and _directory_entry_matches(
                staging_fd,
                "evals",
                staging_evals_fd,
            )
            and _directory_entry_matches(
                evals_fd,
                checkpoint_hash,
                staged_fd,
            )
        ):
            os.rename(
                checkpoint_hash,
                checkpoint_hash,
                src_dir_fd=evals_fd,
                dst_dir_fd=staging_evals_fd,
            )
            raise ValueError(
                "checkpoint publication parent changed during publication"
            )
        os.fsync(evals_fd)
        return destination
    finally:
        if evals_fd is not None and lock_owned:
            try:
                os.unlink(lock_name, dir_fd=evals_fd)
            except FileNotFoundError:
                pass
        for descriptor in (
            evals_fd,
            staged_fd,
            staging_evals_fd,
            staging_fd,
            run_fd,
        ):
            if descriptor is not None:
                os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--graph-store",
        help="packed graph directory (default: DATA/eval/graph.store)",
    )
    parser.add_argument(
        "--factual-graph-store",
        help="packed factual graph directory "
        "(default: DATA/eval/factual-graph.store)",
    )
    parser.add_argument(
        "--relation-schema",
        help="frozen relation schema (default: DATA/relation-schema.json)",
    )
    parser.add_argument(
        "--atomic-fixtures",
        action="store_true",
        help="explicitly use legacy JSONL stores and the fixture codec",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--control-seed",
        type=int,
        default=0,
        help="non-negative deterministic seed shared by causal controls",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=10_000,
        help="frozen value is 10000; smaller values are for smoke tests",
    )
    parser.add_argument(
        "--paired-dense-run",
        help="required Dense run root for a Split evaluation",
    )
    parser.add_argument(
        "--paired-dense-checkpoint-dir",
        help="required validated Dense eval checkpoint directory for Split",
    )
    args = parser.parse_args()

    if args.expected_pairs <= 0:
        raise ValueError("expected-pairs must be positive")
    if args.control_seed < 0:
        raise ValueError("control-seed must be non-negative")
    run = Path(args.run)
    device = pick_device(args.device)
    (
        model,
        cfg,
        checkpoint_state,
        checkpoint_path,
        checkpoint_hash,
        config_identities,
    ) = _load_model(run, args.checkpoint, device)
    paired_dense_dir: Path | None = None
    paired_arguments = (
        args.paired_dense_run,
        args.paired_dense_checkpoint_dir,
    )
    if cfg["condition"] == "split":
        if any(value is None for value in paired_arguments):
            raise ValueError(
                "Split evaluation requires paired Dense run and "
                "checkpoint directory arguments"
            )
        dense_run = Path(args.paired_dense_run)
        supplied_dense = Path(args.paired_dense_checkpoint_dir)
        paired_dense_dir = (
            supplied_dense
            if supplied_dense.is_absolute()
            else (
                dense_run / "evals" / supplied_dense
                if len(supplied_dense.parts) == 1
                else dense_run / supplied_dense
            )
        )
        expected_dense_root = dense_run / "evals"
        if (
            dense_run.is_symlink()
            or not dense_run.is_dir()
            or paired_dense_dir.parent.absolute()
            != expected_dense_root.absolute()
        ):
            raise ValueError(
                "paired Dense checkpoint must be under DENSE_RUN/evals"
            )
    elif any(value is not None for value in paired_arguments):
        raise ValueError(
            "paired Dense arguments are valid only for Split evaluations"
        )
    data_dir = _resolve_data_dir(cfg, args.data_dir)
    tok = get_tok()
    if args.atomic_fixtures:
        schema_path = (
            None
            if args.relation_schema is None
            else Path(args.relation_schema)
        )
        relation_codec = (
            SRGM_RELATION_CODEC
            if schema_path is None
            else _load_relation_codec(schema_path)
        )
        base_store_path = (
            Path(args.graph_store)
            if args.graph_store is not None
            else data_dir / "eval" / "graph.jsonl"
        )
        factual_store_path = (
            Path(args.factual_graph_store)
            if args.factual_graph_store is not None
            else data_dir / "eval" / "factual-graph.jsonl"
        )
    else:
        schema_path = (
            Path(args.relation_schema)
            if args.relation_schema is not None
            else data_dir / "relation-schema.json"
        )
        relation_codec = _load_relation_codec(schema_path)
        base_store_path = (
            Path(args.graph_store)
            if args.graph_store is not None
            else data_dir / "eval" / "graph.store"
        )
        factual_store_path = (
            Path(args.factual_graph_store)
            if args.factual_graph_store is not None
            else data_dir / "eval" / "factual-graph.store"
        )
    base_store = _load_evaluator_store(
        base_store_path,
        codec=relation_codec,
        atomic_fixture=args.atomic_fixtures,
    )
    factual_store = _load_evaluator_store(
        factual_store_path,
        codec=relation_codec,
        atomic_fixture=args.atomic_fixtures,
    )
    data_hash: str | None = None
    staging_run: Path | None = None
    try:
        relation_schema_hash = (
            relation_codec.sha256()
            if schema_path is None
            else _regular_file_sha256(schema_path)
        )
        result_schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "relational-result-v1.schema.json"
        )
        result_schema_hash = _regular_file_sha256(result_schema_path)
        data_hash = _data_sha256(
            base_store,
            factual_store,
            data_dir,
        )
        identity = _result_identity(
            cfg=cfg,
            state=checkpoint_state,
            checkpoint_hash=checkpoint_hash,
            config_identities=config_identities,
            data_hash=data_hash,
            relation_schema_hash=relation_schema_hash,
            result_schema_hash=result_schema_hash,
        )
        bound_inputs = {
            "run": run,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_state": checkpoint_state,
            "config": cfg,
            "config_identities": config_identities,
            "base_store": base_store,
            "factual_store": factual_store,
            "data_dir": data_dir,
            "data_sha256": data_hash,
            "evaluator_sha256": identity["evaluator_sha256"],
            "relation_schema_path": schema_path,
            "relation_codec": relation_codec,
            "relation_schema_sha256": relation_schema_hash,
            "result_schema_path": result_schema_path,
            "result_schema_sha256": result_schema_hash,
            "provenance_sha256": identity["provenance_sha256"],
        }
        parameter_count = model.num_params()
        _frozen_checkpoint_multiple(
            identity["raw_token_count"],
            parameter_count,
        )
        _preflight_output_matrix(run, checkpoint_hash)
        staging_run = Path(
            tempfile.mkdtemp(
                dir=run,
                prefix=f".relational-eval-{checkpoint_hash[:12]}-",
            )
        )

        measurements = produce_guardrail_measurements(
            model,
            tok=tok,
            data_dir=data_dir,
            condition=cfg["condition"],
            device=device,
            batch_size=args.batch_size,
            factual_store=factual_store,
            codec=relation_codec,
        )
        _validate_guardrail_schema(measurements)

        summaries: dict[str, dict[str, dict]] = {}
        outputs: dict[str, dict[str, str]] = {}
        arm_crossings: dict[str, int | None] | None = None
        locality_index = _LocalityIndex(staging_run)
        transformation_index = _ControlTransformationIndex(staging_run)
        mode_order = (EvalMode.MEMORY_ON, EvalMode.MEMORY_OFF)
        try:
            for control in ControlID:
                publishers = {
                    mode: StreamingEvaluationPublisher(staging_run)
                    for mode in mode_order
                }
                try:
                    for items in _iter_eval_item_batches(
                        data_dir,
                        args.expected_pairs,
                        batch_pairs=args.batch_size,
                    ):
                        selected_stores = [
                            store_for_item(base_store, item, memory_on=True)
                            for item in items
                        ]
                        transformation_cache: dict[tuple, tuple] = {}
                        views = [
                            build_control_view(
                                item,
                                selected,
                                control,
                                args.control_seed,
                                transformation_cache=transformation_cache,
                            )
                            for item, selected in zip(items, selected_stores)
                        ]
                        for view in views:
                            transformation_index.add(view)
                        view_by_qid = {
                            str(_item_value(view.item, "qid")): view
                            for view in views
                        }
                        if len(view_by_qid) != len(views):
                            raise ValueError(
                                "control views contain duplicate qids"
                            )
                        has_forced_actions = [
                            view.forced_actions is not None for view in views
                        ]
                        if any(has_forced_actions) and not all(
                            has_forced_actions
                        ):
                            raise ValueError(
                                "forced action controls must cover every item"
                            )
                        forced_actions = (
                            {
                                qid: view.forced_actions
                                for qid, view in view_by_qid.items()
                            }
                            if all(has_forced_actions)
                            else None
                        )
                        has_forced_returns = [
                            view.forced_returns is not None for view in views
                        ]
                        if any(has_forced_returns) and not all(
                            has_forced_returns
                        ):
                            raise ValueError(
                                "forced return controls must cover every item"
                            )
                        forced_returns = (
                            {
                                qid: view.forced_returns
                                for qid, view in view_by_qid.items()
                            }
                            if all(has_forced_returns)
                            else None
                        )
                        for mode in mode_order:
                            states = decode_items(
                                model,
                                tok,
                                [view.item for view in views],
                                (
                                    lambda item, lookup=view_by_qid: lookup[
                                        str(_item_value(item, "qid"))
                                    ].store
                                    if mode == EvalMode.MEMORY_ON
                                    else None
                                ),
                                device=device,
                                batch_size=args.batch_size,
                                codec=relation_codec,
                                forced_actions=forced_actions,
                                forced_returns=forced_returns,
                                forced_return_store=(
                                    (
                                        lambda item, lookup=view_by_qid: lookup[
                                            str(_item_value(item, "qid"))
                                        ].store
                                    )
                                    if forced_returns is not None
                                    else None
                                ),
                            )
                            result_rows = _states_to_eval_rows(
                                views,
                                states,
                                memory_mode=mode,
                                identity=identity,
                            )
                            for index, row in enumerate(result_rows):
                                if control == ControlID.CORRECT:
                                    locality_index.add_baseline(row)
                                elif control in {
                                    ControlID.RELEVANT_EDGE,
                                    ControlID.IRRELEVANT_EDGE,
                                }:
                                    result_rows[index] = locality_index.bind(
                                        row
                                    )
                            for start in range(0, len(result_rows), 2):
                                publishers[mode].add_pair(
                                    result_rows[start : start + 2]
                                )

                    if control == ControlID.CORRECT:
                        current = publishers[
                            EvalMode.MEMORY_ON
                        ].preview_summary()
                        arm_crossings = _milestone_crossings_for_cell(
                            run,
                            checkpoint_hash=checkpoint_hash,
                            parameter_count=parameter_count,
                            identity=identity,
                            memory_mode=EvalMode.MEMORY_ON,
                            control_id=ControlID.CORRECT,
                            current_metrics=current.metrics,
                        )
                    if arm_crossings is None:
                        raise AssertionError(
                            "arm-level milestone crossings are unavailable"
                        )
                    for mode in mode_order:
                        _, summary = publishers[mode].finish(
                            milestone_crossings=arm_crossings
                        )
                        summaries.setdefault(mode.value, {})[
                            control.value
                        ] = summary.to_dict()
                        outputs.setdefault(mode.value, {})[
                            control.value
                        ] = str(
                            Path("evals")
                            / checkpoint_hash
                            / mode.value
                            / control.value
                        )
                finally:
                    for publisher in publishers.values():
                        publisher.close()
            locality_index.require_complete(
                {
                    ControlID.RELEVANT_EDGE,
                    ControlID.IRRELEVANT_EDGE,
                }
            )
            _, transformation_count = transformation_index.publish(
                checkpoint_hash
            )
        finally:
            locality_index.close()
            transformation_index.close()
        if any(
            set(summaries.get(mode.value, {}))
            != {control.value for control in ControlID}
            for mode in EvalMode
        ):
            raise AssertionError(
                "evaluation matrix publication is incomplete"
            )

        guardrail_artifact = (
            _build_exploratory_route_artifact(measurements, identity)
            if identity["arm"] == "selective"
            else _build_guardrail_source(measurements, identity)
        )
        guardrail_path = _write_checkpoint_guardrail_artifact(
            staging_run,
            checkpoint_hash,
            guardrail_artifact,
        )
        _write_exact_matrix_manifest(
            staging_run,
            checkpoint_hash,
            summaries,
            transformation_count=transformation_count,
            guardrail_artifact=guardrail_path,
            guardrail_record_type=guardrail_artifact["record_type"],
        )
        _verify_and_promote_checkpoint_tree(
            run,
            staging_run,
            checkpoint_hash,
            bound_inputs,
        )
        if paired_dense_dir is not None:
            split_checkpoint_dir = run / "evals" / checkpoint_hash
            receipt_path = publish_checkpoint_pairing_receipt(
                split_checkpoint_dir,
                paired_dense_dir,
            )
            publish_confirmatory_guardrail_report(
                split_checkpoint_dir,
                paired_dense_dir,
                "guardrail-report.json",
                pairing_receipt=receipt_path,
            )
        guardrail_path = (
            run
            / "evals"
            / checkpoint_hash
            / guardrail_path.name
        )
        combined = {
            "schema_version": 1,
            "condition": cfg["condition"],
            "checkpoint_sha256": checkpoint_hash,
            "identity": identity,
            "outputs": outputs,
            "summaries": summaries,
            "guardrail_artifact": str(
                guardrail_path.relative_to(run)
            ),
        }
        print(
            json.dumps(
                combined,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        for store in (base_store, factual_store):
            close = getattr(store, "close", None)
            if callable(close):
                close()
        if staging_run is not None:
            shutil.rmtree(staging_run, ignore_errors=True)


if __name__ == "__main__":
    main()
