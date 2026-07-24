"""Resumable, external-memory compiler for frozen MemorySplit v2 sources.

The compiler consumes the immutable upstream stage from :mod:`v2_sources`,
materializes each lane exactly once into a private work tree, derives one
global Split90 route over stable fact identities, and atomically publishes the
45-file source-root contract consumed by :mod:`parallel.production`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpusgen.graph_records import GraphAction, GraphRow, TaggedSegment
from corpusgen.graph_trace import serialize_action, serialize_return
from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel.production import (
    DEFAULT_RECIPE_PATH,
    FROZEN_LANES,
    FROZEN_OBJECTIVE_SOURCE_IDS,
    ProductionRecipe,
    expected_production_source_paths,
    load_production_recipe,
    production_preflight,
    seal_production_sources,
)
from corpusgen.reasoning.proofs import (
    CompositionPremise,
    EqualityPremise,
    GraphTraversalPremise,
    solve_graph_composition,
    solve_graph_traversal,
    solve_slot_equality,
)
from corpusgen.reasoning.state import AnswerPointer, serialize_answer_state
from corpusgen.srgm_worlds import WorldConfig, _generate_eval_pairs, generate_world
from corpusgen.v2_objective import PROCEDURAL_PROVIDERS, WORKER_FORMAT
from corpusgen.v2_sources import (
    DEFAULT_V2_SOURCE_LOCK,
    V2SourceSetLock,
    load_v2_source_lock,
    verify_v2_source_stage,
)
from corpusgen.wikidata5m import parse_pid, parse_qid
from train.tokenizer import get_tok

MATERIALIZER_FORMAT = "memorysplit-v2-lane-materializer-v2"
MATERIALIZER_VERSION = "external-memory-exact-quota-v2"
DEFAULT_FINISH_WINDOW = 1 << 18
DEFAULT_FINISH_CANDIDATES = 32_768
DEFAULT_CHECKPOINT_TOKENS = 1 << 22
MAX_RECORD_TOKENS = 2_048
REASONING_ACTION_SLOTS = 6
ARC_CONCEPTARC_MAX_TOKENS = 17_802_199
_OBJECTIVE_PROBE_INDICES = tuple(range(128))
_GENERATOR_LOCK_FORMAT = "memorysplit-v2-source-lock-v1"
# Existing protected SRGM evaluation starts at 2**31.  Training generators use
# widely separated high-ID ranges so graph addresses cannot overlap either the
# legacy low-ID training worlds or that sealed evaluation range.
_GRAPH_WORLD_OFFSET = 1 << 40
_MULTIHOP_WORLD_OFFSET = 1 << 44
_REFINEMENT_WORLD_OFFSET = 1 << 48
_UPSTREAM_LOCK_FILES = {
    "fineweb_edu": "fineweb-edu.lock.json",
    "finemath": "finemath.lock.json",
    "wikidata5m": "wikidata5m-complete-once.lock.json",
    "objective_auxiliary": "objective-auxiliaries.lock.json",
}
_MATERIALIZER_ARTIFACTS = (
    "corpusgen/graph_records.py",
    "corpusgen/graph_trace.py",
    "corpusgen/parallel/canonical.py",
    "corpusgen/parallel/production.py",
    "corpusgen/reasoning/proofs.py",
    "corpusgen/reasoning/state.py",
    "corpusgen/srgm_worlds.py",
    "corpusgen/v2_materialize.py",
    "corpusgen/v2_objective.py",
    "corpusgen/v2_sources.py",
    "corpusgen/wikidata5m.py",
    "train/tokenizer.py",
)
_GENERATED_LOCK_SPECS = {
    "synthetic_graph_generator": {
        "kind": "generator",
        "artifacts": (
            "corpusgen/graph_records.py",
            "corpusgen/graph_trace.py",
            "corpusgen/srgm_worlds.py",
            "corpusgen/v2_materialize.py",
        ),
        "policy": {
            "fact_source": "srgm_worlds.generate_world",
            "candidate_fact_reuse": False,
            "world_id_offset": _GRAPH_WORLD_OFFSET,
            "world_entities": 64,
            "world_schedule": "emit_each_fact_candidate_once_then_advance",
        },
    },
    "verified_synthetic_multihop_generator": {
        "kind": "generator",
        "artifacts": (
            "corpusgen/graph_trace.py",
            "corpusgen/srgm_worlds.py",
            "corpusgen/v2_materialize.py",
        ),
        "policy": {
            "action_slots": REASONING_ACTION_SLOTS,
            "cross_record_edge_reuse": "allowed_within_each_fresh_world",
            "families": ["graph_composition_mod4"],
            "hops": [2, 4],
            "path_edges_unique_within_record": True,
            "record_reuse": False,
            "world_id_offset": _MULTIHOP_WORLD_OFFSET,
            "world_entities": 64,
        },
    },
    "wikidata_path_reasoning_generator": {
        "kind": "generator",
        "artifacts": (
            "corpusgen/graph_trace.py",
            "corpusgen/v2_materialize.py",
            "corpusgen/wikidata5m.py",
        ),
        "policy": {
            "action_slots": REASONING_ACTION_SLOTS,
            "address_policy": "functional_subject_relation_addresses_only",
            "ordering": "first_edge_ordinal_then_second_edge_ordinal",
            "path_length": 2,
            "path_reuse": False,
            "source_edge_reuse_across_paths": True,
            "source": "complete-once Wikidata5M training triples",
        },
    },
    "relational_refinement_generator": {
        "kind": "generator",
        "artifacts": (
            "corpusgen/reasoning/state.py",
            "corpusgen/srgm_worlds.py",
            "corpusgen/v2_materialize.py",
        ),
        "policy": {
            "action_slots": REASONING_ACTION_SLOTS,
            "answer_state": "pointer_only_for_factual_surfaces",
            "candidate": "deterministically_incorrect_boolean",
            "cross_record_fact_reuse": "allowed_within_each_fresh_world",
            "family": "slot_equality",
            "premise_facts_unique_within_record": True,
            "record_reuse": False,
            "world_id_offset": _REFINEMENT_WORLD_OFFSET,
        },
    },
    "reasoning_solver": {
        "kind": "solver",
        "artifacts": (
            "corpusgen/reasoning/proofs.py",
            "corpusgen/v2_materialize.py",
            "corpusgen/v2_objective.py",
        ),
        "policy": {
            "families": [
                "graph_composition_mod4",
                "graph_path_traversal",
                "slot_equality",
            ],
            "objective_determinism_adapters": {
                "deepmind_mathematics_generator": (
                    "ordinal_seed_python_numpy_sympy;"
                    "stable_entity_insertion_order_before_native_seeded_shuffle;"
                    "integral_float_randint_compatibility"
                )
            },
            "objective_validation": "native_generator_then_canonical_exact_match",
            "solver_id": "finite-domain-enumeration-v1",
            "verification": "canonical_deterministic_replay",
        },
    },
}


class V2MaterializationError(RuntimeError):
    """A lane cannot be materialized without violating a frozen input."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        lane: str | None = None,
        action: str | None = None,
    ) -> None:
        self.code = code
        self.lane = lane
        self.action = action
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.lane is not None:
            value["lane"] = self.lane
        if self.action is not None:
            value["action"] = self.action
        return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _u16_bytes(values: Sequence[int]) -> bytes:
    tokens = array("H", values)
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens.tobytes()


def _fraction_json(numerator: int, denominator: int = 1) -> dict[str, int]:
    divisor = math.gcd(numerator, denominator)
    return {
        "denominator": denominator // divisor,
        "numerator": numerator // divisor,
    }


def _write_repeated(handle: Any, value: int, count: int) -> None:
    block = bytes((value,)) * min(1 << 20, max(count, 1))
    remaining = count
    while remaining:
        chunk = block[: min(len(block), remaining)]
        handle.write(chunk)
        remaining -= len(chunk)


@dataclass(frozen=True)
class LaneSegment:
    text: str
    fact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("lane segments must contain non-empty text")
        if self.fact_id is not None and (
            not isinstance(self.fact_id, str) or not self.fact_id
        ):
            raise ValueError("lane segment fact_id must be non-empty")


@dataclass(frozen=True)
class LaneRecord:
    record_id: str
    source_id: str
    segments: tuple[LaneSegment, ...]
    verification: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.record_id or not self.source_id or not self.segments:
            raise ValueError("lane records require identity, source, and segments")


@dataclass(frozen=True)
class FactOccurrence:
    fact_id: str
    start: int
    end: int
    surface_sha256: str


@dataclass(frozen=True)
class EncodedLaneRecord:
    record: LaneRecord
    tokens: tuple[int, ...]
    facts: tuple[FactOccurrence, ...]

    @property
    def token_count(self) -> int:
        return len(self.tokens)


def _encode_record(tok: Any, record: LaneRecord) -> EncodedLaneRecord:
    tokens: list[int] = []
    facts: list[FactOccurrence] = []
    for segment in record.segments:
        encoded = tok.encode(segment.text)
        if not encoded:
            raise V2MaterializationError(
                "empty_encoded_segment",
                f"record {record.record_id!r} contains an empty encoded segment",
            )
        start = len(tokens)
        tokens.extend(encoded)
        if segment.fact_id is not None:
            facts.append(
                FactOccurrence(
                    fact_id=segment.fact_id,
                    start=start,
                    end=len(tokens),
                    surface_sha256=hashlib.sha256(
                        segment.text.encode("utf-8")
                    ).hexdigest(),
                )
            )
    tokens.append(tok.EOT)
    if len(tokens) > MAX_RECORD_TOKENS:
        raise V2MaterializationError(
            "record_exceeds_context",
            f"record {record.record_id!r} has {len(tokens)} tokens; "
            f"maximum is {MAX_RECORD_TOKENS}",
        )
    if any(token < 0 or token >= 1 << 16 for token in tokens):
        raise V2MaterializationError(
            "token_out_of_range",
            f"record {record.record_id!r} contains a non-uint16 token",
        )
    return EncodedLaneRecord(record, tuple(tokens), tuple(facts))


def _solver_bundle(
    premises: (
        Sequence[CompositionPremise]
        | Sequence[EqualityPremise]
        | Sequence[GraphTraversalPremise]
    ),
) -> dict[str, Any]:
    if not premises:
        raise ValueError("solver bundle requires premises")
    first = premises[0]
    if isinstance(first, CompositionPremise):
        proof = solve_graph_composition(premises)
        rows = [
            {
                "compose_code": premise.compose_code,
                "fact_id": premise.fact_id,
                "hop": premise.hop,
                "type": "composition",
            }
            for premise in premises
        ]
    elif isinstance(first, EqualityPremise):
        proof = solve_slot_equality(premises)
        rows = [
            {
                "fact_id": premise.fact_id,
                "slot": premise.slot,
                "type": "equality",
                "value": premise.value,
            }
            for premise in premises
        ]
    elif isinstance(first, GraphTraversalPremise):
        proof = solve_graph_traversal(premises)
        rows = [
            {
                "fact_id": premise.fact_id,
                "hop": premise.hop,
                "relation": premise.relation,
                "source": premise.source,
                "target": premise.target,
                "type": "graph_traversal",
            }
            for premise in premises
        ]
    else:
        raise TypeError("unsupported solver premise type")
    return {
        "family": proof.family,
        "kind": "solver",
        "premises": rows,
        "proof": proof.as_dict(),
    }


class _LaneWriter:
    _STATE_FORMAT = "memorysplit-v2-lane-checkpoint-v3"

    def __init__(
        self,
        work_root: Path,
        lane: str,
        quota: int,
        *,
        checkpoint_tokens: int,
        verification_required: bool = False,
    ) -> None:
        self.root = work_root / "lanes" / lane
        self.root.mkdir(parents=True, exist_ok=True)
        self.lane = lane
        self.quota = quota
        self.checkpoint_tokens = checkpoint_tokens
        self.verification_required = verification_required
        self.token_path = self.root / "tokens.partial"
        self.verification_path = self.root / "verification.partial.jsonl"
        self.state_path = self.root / "checkpoint.json"
        self.facts_path = self.root / "facts.sqlite3"
        self.connection = sqlite3.connect(self.facts_path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                surface_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS occurrences (
                record_index INTEGER NOT NULL,
                occurrence_index INTEGER NOT NULL,
                fact_id TEXT NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                burden_bits INTEGER NOT NULL,
                PRIMARY KEY (record_index, occurrence_index)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS occurrences_fact
                ON occurrences(fact_id);
            CREATE INDEX IF NOT EXISTS occurrences_start
                ON occurrences(start);
            CREATE TABLE IF NOT EXISTS verified_records (
                record_index INTEGER PRIMARY KEY,
                record_id TEXT NOT NULL UNIQUE
            );
            """
        )
        self.connection.commit()
        self.tokens = 0
        self.records = 0
        self.cursor: dict[str, Any] = {}
        self.complete = False
        self._last_checkpoint_tokens = 0
        self._suspend_auto_checkpoint = False
        self._open_from_checkpoint()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "complete": False,
            "cursor": {},
            "files": {"tokens": 0, "verification": 0},
            "format": self._STATE_FORMAT,
            "lane": self.lane,
            "quota": self.quota,
            "records": 0,
            "tokens": 0,
            "verification_required": self.verification_required,
        }

    def _open_from_checkpoint(self) -> None:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise V2MaterializationError(
                    "checkpoint_invalid",
                    f"{self.lane} checkpoint is not valid JSON",
                    lane=self.lane,
                ) from error
            expected_fields = {
                "complete",
                "cursor",
                "files",
                "format",
                "lane",
                "quota",
                "records",
                "tokens",
                "verification_required",
            }
            if (
                not isinstance(state, dict)
                or set(state) != expected_fields
                or not isinstance(state.get("complete"), bool)
                or not isinstance(state.get("cursor"), dict)
                or not isinstance(state.get("files"), dict)
                or set(state["files"]) != {"tokens", "verification"}
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (
                        state.get("records"),
                        state.get("tokens"),
                        *state["files"].values(),
                    )
                )
                or state["tokens"] > self.quota
            ):
                raise V2MaterializationError(
                    "checkpoint_invalid",
                    f"{self.lane} checkpoint fields or counts are invalid",
                    lane=self.lane,
                )
            if (
                state.get("format") != self._STATE_FORMAT
                or state.get("lane") != self.lane
                or state.get("quota") != self.quota
                or state.get("verification_required") is not self.verification_required
            ):
                raise V2MaterializationError(
                    "checkpoint_identity_drift",
                    f"{self.lane} checkpoint identity differs from this build",
                    lane=self.lane,
                )
        else:
            state = self._empty_state()
            _write_json_atomic(self.state_path, state)
        paths = {
            "tokens": self.token_path,
            "verification": self.verification_path,
        }
        for key, path in paths.items():
            path.touch(exist_ok=True)
            expected = int(state["files"][key])
            if path.stat().st_size < expected:
                raise V2MaterializationError(
                    "checkpoint_file_short",
                    f"{self.lane} {key} file is shorter than its checkpoint",
                    lane=self.lane,
                )
            with path.open("r+b") as handle:
                handle.truncate(expected)
        self.tokens = int(state["tokens"])
        self.records = int(state["records"])
        self.cursor = dict(state["cursor"])
        self.complete = bool(state["complete"])
        if self.complete and self.tokens != self.quota:
            raise V2MaterializationError(
                "completed_lane_quota_drift",
                f"{self.lane} completed checkpoint does not meet its quota",
                lane=self.lane,
            )
        if self.token_path.stat().st_size != self.tokens * 2:
            raise V2MaterializationError(
                "checkpoint_token_size_drift",
                f"{self.lane} token bytes do not match its checkpoint count",
                lane=self.lane,
            )
        if (
            not self.verification_required
            and self.verification_path.stat().st_size != 0
        ):
            raise V2MaterializationError(
                "checkpoint_verification_drift",
                f"{self.lane} unexpectedly contains verification rows",
                lane=self.lane,
            )
        self._last_checkpoint_tokens = self.tokens
        with self.connection:
            self.connection.execute(
                "DELETE FROM occurrences WHERE record_index >= ?",
                (self.records,),
            )
            self.connection.execute(
                "DELETE FROM verified_records WHERE record_index >= ?",
                (self.records,),
            )
            self.connection.execute(
                """
                DELETE FROM facts
                WHERE fact_id NOT IN (SELECT DISTINCT fact_id FROM occurrences)
                """
            )
        verified_records = int(
            self.connection.execute("SELECT COUNT(*) FROM verified_records").fetchone()[
                0
            ]
        )
        if verified_records != (self.records if self.verification_required else 0):
            raise V2MaterializationError(
                "checkpoint_verification_drift",
                f"{self.lane} verified-record state does not match its checkpoint",
                lane=self.lane,
            )
        self.token_file = self.token_path.open("ab")
        self.verification_file = self.verification_path.open("ab")

    @property
    def remaining(self) -> int:
        return self.quota - self.tokens

    def add(self, encoded: EncodedLaneRecord, next_cursor: Mapping[str, Any]) -> None:
        count = encoded.token_count
        if self.complete:
            raise ValueError("cannot append to a completed lane")
        if count <= 0 or count > self.remaining:
            raise ValueError("encoded record does not fit the remaining lane quota")
        if (encoded.record.verification is not None) is not self.verification_required:
            raise V2MaterializationError(
                "record_verification_contract_mismatch",
                f"{self.lane} record {encoded.record.record_id!r} "
                f"{'lacks' if self.verification_required else 'unexpectedly has'} "
                "verification",
                lane=self.lane,
            )
        verification = encoded.record.verification
        if verification is not None and verification.get("kind") == "solver":
            premises = verification.get("premises")
            if (
                not isinstance(premises, list)
                or not premises
                or any(
                    not isinstance(premise, dict)
                    or not isinstance(premise.get("fact_id"), str)
                    or not premise["fact_id"]
                    for premise in premises
                )
            ):
                raise V2MaterializationError(
                    "solver_premise_contract_mismatch",
                    f"{self.lane} record {encoded.record.record_id!r} has invalid "
                    "solver premises",
                    lane=self.lane,
                )
            premise_fact_ids = [premise["fact_id"] for premise in premises]
            occurrence_fact_ids = [fact.fact_id for fact in encoded.facts]
            if (
                len(premise_fact_ids) != len(set(premise_fact_ids))
                or len(occurrence_fact_ids) != len(set(occurrence_fact_ids))
                or set(premise_fact_ids) != set(occurrence_fact_ids)
            ):
                raise V2MaterializationError(
                    "solver_premise_occurrence_mismatch",
                    f"{self.lane} record {encoded.record.record_id!r} solver "
                    "premises do not exactly match its factual occurrences",
                    lane=self.lane,
                    action="repair the generator; never verify detached premises",
                )
        start = self.tokens
        if self.verification_required:
            try:
                self.connection.execute(
                    """
                    INSERT INTO verified_records(record_index, record_id)
                    VALUES (?, ?)
                    """,
                    (self.records, encoded.record.record_id),
                )
            except sqlite3.IntegrityError as error:
                raise V2MaterializationError(
                    "reasoning_cycle_fill",
                    f"{self.lane} repeats verified record {encoded.record.record_id!r}",
                    lane=self.lane,
                    action="generate a fresh record; cycle-fill is forbidden",
                ) from error
        for occurrence_index, fact in enumerate(encoded.facts):
            existing = self.connection.execute(
                "SELECT surface_sha256 FROM facts WHERE fact_id = ?",
                (fact.fact_id,),
            ).fetchone()
            if existing is not None and existing[0] != fact.surface_sha256:
                raise V2MaterializationError(
                    "fact_surface_drift",
                    f"{self.lane} fact {fact.fact_id!r} has two payload surfaces",
                    lane=self.lane,
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO facts(fact_id, surface_sha256) VALUES (?, ?)",
                (fact.fact_id, fact.surface_sha256),
            )
            self.connection.execute(
                """
                INSERT INTO occurrences(
                    record_index, occurrence_index, fact_id,
                    start, end, burden_bits
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.records,
                    occurrence_index,
                    fact.fact_id,
                    start + fact.start,
                    start + fact.end,
                    (fact.end - fact.start) * 8,
                ),
            )
        self.token_file.write(_u16_bytes(encoded.tokens))
        if encoded.record.verification is not None:
            self.verification_file.write(
                canonical_json_bytes(
                    {
                        "record_id": encoded.record.record_id,
                        "source_id": encoded.record.source_id,
                        "token_end": start + count,
                        "token_start": start,
                        "verification": encoded.record.verification,
                    }
                )
            )
        self.tokens += count
        self.records += 1
        self.cursor = dict(next_cursor)
        if (
            not self._suspend_auto_checkpoint
            and self.tokens - self._last_checkpoint_tokens >= self.checkpoint_tokens
        ):
            self.checkpoint()

    def checkpoint(self) -> None:
        for handle in (self.token_file, self.verification_file):
            handle.flush()
            os.fsync(handle.fileno())
        self.connection.commit()
        _write_json_atomic(
            self.state_path,
            {
                "complete": self.complete,
                "cursor": self.cursor,
                "files": {
                    "tokens": self.token_file.tell(),
                    "verification": self.verification_file.tell(),
                },
                "format": self._STATE_FORMAT,
                "lane": self.lane,
                "quota": self.quota,
                "records": self.records,
                "tokens": self.tokens,
                "verification_required": self.verification_required,
            },
        )
        self._last_checkpoint_tokens = self.tokens

    def mark_complete(self) -> None:
        if self.tokens != self.quota:
            raise V2MaterializationError(
                "lane_quota_incomplete",
                f"{self.lane} has {self.tokens} of {self.quota} tokens",
                lane=self.lane,
            )
        self.complete = True
        self.checkpoint()

    def close(self, *, checkpoint: bool = True) -> None:
        if checkpoint:
            self.checkpoint()
        else:
            # Leave the last durable checkpoint authoritative.  Token and
            # verification files are truncated back to it on reopen, while
            # SQLite rolls back rows appended after that checkpoint.  This is
            # essential if interruption lands while an exact-finish subset is
            # being installed: its cursor has already scanned past records
            # that must be replayed together.
            self.connection.rollback()
        self.token_file.close()
        self.verification_file.close()
        self.connection.close()


class _ExactSubsetAccumulator:
    """Incrementally solve a bounded exact subset without retaining records."""

    def __init__(self, target: int) -> None:
        if target < 0:
            raise ValueError("subset target must be non-negative")
        self.target = target
        self.reachable = 1
        self.limit_mask = (1 << (target + 1)) - 1
        self.predecessor_sum = array("i", [-1]) * (target + 1)
        self.predecessor_index = array("i", [-1]) * (target + 1)
        self.count = 0

    def add(self, length: int) -> tuple[int, ...] | None:
        index = self.count
        self.count += 1
        if self.target == 0:
            return ()
        if length <= 0 or length > self.target:
            return None
        new = ((self.reachable << length) & self.limit_mask) & ~self.reachable
        bits = new
        while bits:
            bit = bits & -bits
            total = bit.bit_length() - 1
            self.predecessor_sum[total] = total - length
            self.predecessor_index[total] = index
            bits ^= bit
        self.reachable |= new
        if not ((self.reachable >> self.target) & 1):
            return None
        selected = []
        total = self.target
        while total:
            selected.append(self.predecessor_index[total])
            total = self.predecessor_sum[total]
        return tuple(reversed(selected))


def _subset_indices_exact(
    lengths: Sequence[int],
    target: int,
) -> tuple[int, ...] | None:
    """Return the first deterministic source-ordered subset summing to target."""

    accumulator = _ExactSubsetAccumulator(target)
    if target == 0:
        return ()
    for length in lengths:
        selected = accumulator.add(length)
        if selected is not None:
            return selected
    return None


class _RecordSource:
    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


def _source_next(
    source: _RecordSource,
    cursor: Mapping[str, Any],
    lane: str,
) -> tuple[LaneRecord, dict[str, Any]]:
    try:
        return source.next(cursor)
    except StopIteration as error:
        raise V2MaterializationError(
            "fresh_source_exhausted",
            f"{lane} exhausted its locked fresh records before its quota",
            lane=lane,
            action="add no padding or cycle-fill; repair the frozen source recipe",
        ) from error


def _compile_exact_lane(
    writer: _LaneWriter,
    source: _RecordSource,
    *,
    finish_window: int,
    max_finish_candidates: int,
) -> None:
    if writer.complete:
        source.close()
        return
    tok = get_tok()
    cursor = dict(writer.cursor)
    try:
        while writer.remaining > finish_window:
            record, next_cursor = _source_next(source, cursor, writer.lane)
            encoded = _encode_record(tok, record)
            if encoded.token_count > writer.remaining:
                raise V2MaterializationError(
                    "record_exceeds_remaining_quota",
                    f"{writer.lane} record exceeds a non-final quota remainder",
                    lane=writer.lane,
                )
            writer.add(encoded, next_cursor)
            cursor = next_cursor

        writer.checkpoint()
        target = writer.remaining
        candidates: list[tuple[int, int]] = []
        selected: tuple[int, ...] | None = () if target == 0 else None
        subset = _ExactSubsetAccumulator(target)
        # Candidate payloads can be large even though the exact-finish window
        # is bounded.  Spool them beside the lane instead of retaining up to
        # ``max_finish_candidates`` token tuples in RAM.
        with tempfile.TemporaryFile(
            mode="w+b",
            prefix=".finish-candidates-",
            dir=writer.root,
        ) as candidate_file:
            for _ in range(max_finish_candidates):
                if selected is not None:
                    break
                record, next_cursor = _source_next(source, cursor, writer.lane)
                encoded = _encode_record(tok, record)
                cursor = next_cursor
                if encoded.token_count > target:
                    continue
                offset = candidate_file.tell()
                candidate_file.write(
                    canonical_json_bytes(
                        {
                            "record_id": record.record_id,
                            "segments": [
                                {
                                    "fact_id": segment.fact_id,
                                    "text": segment.text,
                                }
                                for segment in record.segments
                            ],
                            "source_id": record.source_id,
                            "verification": record.verification,
                        }
                    )
                )
                candidates.append((offset, encoded.token_count))
                selected = subset.add(encoded.token_count)
            if selected is None:
                raise V2MaterializationError(
                    "exact_quota_unfillable",
                    f"{writer.lane} could not fill its final {target} tokens from "
                    f"{len(candidates)} fresh source records",
                    lane=writer.lane,
                    action=(
                        "increase --finish-window/--max-finish-candidates only; "
                        "do not pad or cycle reasoning records"
                    ),
                )
            final_cursor = dict(cursor)
            writer._suspend_auto_checkpoint = True
            try:
                for index in selected:
                    offset, expected_tokens = candidates[index]
                    candidate_file.seek(offset)
                    value = json.loads(candidate_file.readline())
                    record = LaneRecord(
                        record_id=value["record_id"],
                        source_id=value["source_id"],
                        segments=tuple(
                            LaneSegment(item["text"], item["fact_id"])
                            for item in value["segments"]
                        ),
                        verification=value["verification"],
                    )
                    encoded = _encode_record(tok, record)
                    if encoded.token_count != expected_tokens:
                        raise AssertionError(
                            "spooled exact-finish candidate token count drifted"
                        )
                    writer.add(encoded, final_cursor)
            finally:
                writer._suspend_auto_checkpoint = False
        writer.cursor = final_cursor
        writer.mark_complete()
    finally:
        source.close()


class _ParquetTextSource(_RecordSource):
    def __init__(
        self,
        files: Sequence[tuple[Path, Path | None]],
        *,
        source_id: str,
        record_prefix: str,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise V2MaterializationError(
                "pyarrow_missing",
                "Parquet materialization requires pyarrow",
                lane=source_id,
                action="install the repository data dependencies",
            ) from error
        self._pq = pq
        self.files = tuple(files)
        self.source_id = source_id
        self.record_prefix = record_prefix
        self.tok = get_tok()
        self._file_index = -1
        self._parquet = None
        self._row_group = -1
        self._column = None
        self._row_offsets: tuple[int, ...] = ()
        self._sidecar = None
        self._token_key: tuple[int, int, int] | None = None
        self._row_tokens: list[int] = []

    def _open_file(self, file_index: int) -> None:
        if self._sidecar is not None:
            self._sidecar.close()
            self._sidecar = None
        path, sidecar = self.files[file_index]
        self._parquet = self._pq.ParquetFile(path)
        offsets = [0]
        for group in range(self._parquet.num_row_groups):
            offsets.append(
                offsets[-1] + self._parquet.metadata.row_group(group).num_rows
            )
        self._row_offsets = tuple(offsets)
        if sidecar is not None:
            self._sidecar = sidecar.open("rb")
            if sidecar.stat().st_size != offsets[-1]:
                raise V2MaterializationError(
                    "selection_sidecar_length_drift",
                    f"{sidecar} does not align with its Parquet rows",
                    lane=self.source_id,
                )
        self._file_index = file_index
        self._row_group = -1
        self._column = None
        self._token_key = None

    def _load_group(self, group: int) -> None:
        assert self._parquet is not None
        table = self._parquet.read_row_group(group, columns=["text"])
        if table.column_names != ["text"]:
            raise V2MaterializationError(
                "parquet_text_column_missing",
                f"{self.files[self._file_index][0]} lacks a text column",
                lane=self.source_id,
            )
        self._column = table.column(0).combine_chunks()
        self._row_group = group
        self._token_key = None

    def _kept(self, group: int, row: int) -> bool:
        if self._sidecar is None:
            return True
        self._sidecar.seek(self._row_offsets[group] + row)
        value = self._sidecar.read(1)
        if value not in (b"\x00", b"\x01"):
            raise V2MaterializationError(
                "selection_sidecar_value_invalid",
                "selection sidecars must contain only zero or one",
                lane=self.source_id,
            )
        return value == b"\x01"

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        file_index = int(cursor.get("file", 0))
        group = int(cursor.get("row_group", 0))
        row = int(cursor.get("row", 0))
        token_offset = int(cursor.get("token_offset", 0))
        while file_index < len(self.files):
            if self._file_index != file_index:
                self._open_file(file_index)
            assert self._parquet is not None
            if group >= self._parquet.num_row_groups:
                file_index += 1
                group = row = token_offset = 0
                continue
            if self._row_group != group:
                self._load_group(group)
            assert self._column is not None
            if row >= len(self._column):
                group += 1
                row = token_offset = 0
                continue
            if not self._kept(group, row):
                row += 1
                token_offset = 0
                continue
            key = (file_index, group, row)
            if self._token_key != key:
                text = self._column[row].as_py()
                if not isinstance(text, str):
                    raise V2MaterializationError(
                        "parquet_text_value_invalid",
                        f"{self.files[file_index][0]} contains a non-string text row",
                        lane=self.source_id,
                    )
                self._row_tokens = self.tok.encode(text)
                self._token_key = key
            if not self._row_tokens:
                row += 1
                token_offset = 0
                continue
            candidate_end = min(
                len(self._row_tokens),
                token_offset + MAX_RECORD_TOKENS - 1,
            )
            for end in range(candidate_end, token_offset, -1):
                chunk_ids = self._row_tokens[token_offset:end]
                text = self.tok.decode(chunk_ids)
                if self.tok.encode(text) == chunk_ids:
                    break
            else:
                raise V2MaterializationError(
                    "parquet_chunk_not_lossless",
                    "source text cannot be split at a lossless token boundary",
                    lane=self.source_id,
                )
            record_id = (
                f"{self.record_prefix}:{file_index}:{group}:{row}:{token_offset}"
            )
            if end == len(self._row_tokens):
                next_cursor = {
                    "file": file_index,
                    "row": row + 1,
                    "row_group": group,
                    "token_offset": 0,
                }
            else:
                next_cursor = {
                    "file": file_index,
                    "row": row,
                    "row_group": group,
                    "token_offset": end,
                }
            return (
                LaneRecord(
                    record_id,
                    self.source_id,
                    (LaneSegment(text),),
                ),
                next_cursor,
            )
        raise StopIteration

    def close(self) -> None:
        if self._sidecar is not None:
            self._sidecar.close()


def _fineweb_source(stage_root: Path, lock: V2SourceSetLock) -> _RecordSource:
    source = lock.sources["fineweb_edu"]
    files = [
        (stage_root / "fineweb_edu" / str(item["path"]), None)
        for item in source["files"]
    ]
    return _ParquetTextSource(
        files,
        source_id="fineweb_edu",
        record_prefix="fineweb",
    )


def _finemath_source(stage_root: Path) -> _RecordSource:
    manifest_path = stage_root / "finemath" / "selection-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("format") != "memorysplit-v2-finemath-selection":
        raise V2MaterializationError(
            "finemath_selection_manifest_invalid",
            "FineMath selection manifest identity drifted",
            lane="finemath",
        )
    files = []
    for item in manifest["files"]:
        files.append(
            (
                stage_root / "finemath" / str(item["input_path"]),
                stage_root / "finemath" / str(item["sidecar"]["path"]),
            )
        )
    return _ParquetTextSource(
        files,
        source_id="finemath",
        record_prefix="finemath",
    )


class _WikidataIndex:
    _FORMAT = "memorysplit-v2-wikidata-index-v2"

    def __init__(
        self,
        stage_root: Path,
        work_root: Path,
        source_set_sha256: str,
    ) -> None:
        self.stage_root = stage_root
        self.files_root = stage_root / "wikidata5m" / "files"
        self.selection = json.loads(
            (stage_root / "wikidata5m" / "selection-manifest.json").read_bytes()
        )
        self.path = work_root / "indexes" / "wikidata.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS triples (
                ordinal INTEGER PRIMARY KEY,
                subject INTEGER NOT NULL,
                relation INTEGER NOT NULL,
                target INTEGER NOT NULL,
                UNIQUE(subject, relation, target)
            );
            CREATE TABLE IF NOT EXISTS functional_triples (
                ordinal INTEGER PRIMARY KEY,
                subject INTEGER NOT NULL,
                relation INTEGER NOT NULL,
                target INTEGER NOT NULL,
                UNIQUE(subject, relation)
            ) WITHOUT ROWID;
            """
        )
        identity = json.dumps(
            {
                "format": self._FORMAT,
                "source_set_sha256": source_set_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        existing = self._get("identity")
        if existing is None:
            self._set("identity", identity)
            self.connection.commit()
        elif existing != identity:
            raise V2MaterializationError(
                "wikidata_index_identity_drift",
                "Wikidata external-memory index belongs to another source set",
                lane="wikidata_graph",
            )

    def _get(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def _set(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, str(value)),
        )

    def ensure(self) -> None:
        if self._get("complete") == "1":
            return
        if self.selection.get("format") != (
            "memorysplit-v2-wikidata-complete-once-selection"
        ):
            raise V2MaterializationError(
                "wikidata_selection_manifest_invalid",
                "Wikidata complete-once selection manifest identity drifted",
                lane="wikidata_graph",
            )
        for record in self.selection["files"]:
            self._ingest_split(record)
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS triples_subject_ordinal "
            "ON triples(subject, ordinal)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS triples_address_target "
            "ON triples(subject, relation, target)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS triples_target ON triples(target)"
        )
        expected = int(self.selection["totals"]["distinct_training_triples"])
        observed = int(
            self.connection.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        )
        if observed != expected:
            raise V2MaterializationError(
                "wikidata_index_count_drift",
                f"Wikidata index has {observed} of {expected} selected triples",
                lane="wikidata_graph",
            )
        self.connection.execute("DELETE FROM functional_triples")
        self.connection.execute(
            """
            INSERT INTO functional_triples(ordinal, subject, relation, target)
            SELECT MIN(ordinal), subject, relation, MIN(target)
            FROM triples
            GROUP BY subject, relation
            HAVING COUNT(*) = 1
            ORDER BY MIN(ordinal)
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS functional_triples_subject_ordinal "
            "ON functional_triples(subject, ordinal)"
        )
        if (
            int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM functional_triples"
                ).fetchone()[0]
            )
            <= 0
        ):
            raise V2MaterializationError(
                "wikidata_functional_graph_empty",
                "Wikidata contains no functional subject-relation addresses",
                lane="wikidata_path_reasoning",
            )
        self._set("complete", 1)
        self.connection.commit()

    def _ingest_split(self, record: Mapping[str, Any]) -> None:
        split = str(record["input_path"])
        done_key = f"done:{split}"
        if self._get(done_key) == "1":
            return
        offset_key = f"offset:{split}"
        row_key = f"row:{split}"
        offset = int(self._get(offset_key) or 0)
        row_number = int(self._get(row_key) or 0)
        ordinal = int(
            self.connection.execute(
                "SELECT COALESCE(MAX(ordinal) + 1, 0) FROM triples"
            ).fetchone()[0]
        )
        source_path = self.files_root / split
        sidecar_path = self.stage_root / "wikidata5m" / str(record["sidecar"]["path"])
        kept = 0
        with source_path.open("rb") as source, sidecar_path.open("rb") as sidecar:
            source.seek(offset)
            sidecar.seek(row_number)
            pending = 0
            while line := source.readline():
                marker = sidecar.read(1)
                if marker not in (b"\x00", b"\x01"):
                    raise V2MaterializationError(
                        "wikidata_sidecar_alignment_drift",
                        f"{split} selection sidecar ended or contains invalid bytes",
                        lane="wikidata_graph",
                    )
                row_number += 1
                if marker == b"\x01":
                    try:
                        fields = line.decode("utf-8").rstrip("\r\n").split("\t")
                    except UnicodeDecodeError as error:
                        raise V2MaterializationError(
                            "wikidata_utf8_invalid",
                            f"{split}:{row_number} is not UTF-8",
                            lane="wikidata_graph",
                        ) from error
                    if len(fields) != 3:
                        raise V2MaterializationError(
                            "wikidata_triple_invalid",
                            f"{split}:{row_number} is not a three-column triple",
                            lane="wikidata_graph",
                        )
                    subject = parse_qid(fields[0])
                    relation = int(parse_pid(fields[1])[1:])
                    target = parse_qid(fields[2])
                    try:
                        self.connection.execute(
                            """
                            INSERT INTO triples(ordinal, subject, relation, target)
                            VALUES (?, ?, ?, ?)
                            """,
                            (ordinal, subject, relation, target),
                        )
                    except sqlite3.IntegrityError as error:
                        raise V2MaterializationError(
                            "wikidata_complete_once_duplicate",
                            "a selected complete-once triple is duplicated",
                            lane="wikidata_graph",
                        ) from error
                    ordinal += 1
                    kept += 1
                pending += 1
                if pending == 100_000:
                    self._set(offset_key, source.tell())
                    self._set(row_key, row_number)
                    self.connection.commit()
                    pending = 0
            if sidecar.read(1):
                raise V2MaterializationError(
                    "wikidata_sidecar_alignment_drift",
                    f"{split} selection sidecar exceeds its source rows",
                    lane="wikidata_graph",
                )
            self._set(offset_key, source.tell())
            self._set(row_key, row_number)
            self._set(done_key, 1)
            self.connection.commit()
        if row_number != int(record["rows"]):
            raise V2MaterializationError(
                "wikidata_source_row_count_drift",
                f"{split} has {row_number} rows, expected {record['rows']}",
                lane="wikidata_graph",
            )
        # On resumed ingestion, only the final total is authoritative.
        del kept

    @property
    def triple_count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        )

    def close(self) -> None:
        self.connection.close()


def _wikidata_row(
    subject: int,
    relation: int,
    target: int,
    ordinal: int,
) -> tuple[GraphRow, str]:
    fact_id = f"wikidata:Q{subject}:P{relation}:Q{target}"
    return (
        GraphRow(
            source_id=f"Q{subject}",
            relation_id=f"P{relation}",
            direction="out",
            target_kind="entity",
            target=f"Q{target}",
            provenance_id=f"wikidata-training:{ordinal}",
        ),
        fact_id,
    )


def _lane_segments(values: Sequence[TaggedSegment]) -> tuple[LaneSegment, ...]:
    return tuple(LaneSegment(segment.text, segment.fact_id) for segment in values)


def _graph_record_segments(
    tok: Any,
    row: GraphRow,
    fact_id: str,
) -> tuple[LaneSegment, ...]:
    action = GraphAction(
        source_slot=0,
        relation_id=row.relation_id,
        direction=row.direction,
        read=True,
        halt=False,
        page=row.page,
    )
    return _lane_segments(
        (
            TaggedSegment(f"Graph edge from {row.source_id}. ", "plain"),
            TaggedSegment(tok.decode(serialize_action(action, tok)), "action"),
            *serialize_return(row, fact_id),
        )
    )


def _wikidata_page_segments(
    tok: Any,
    subject: int,
    relation: int,
    page: int,
    members: Sequence[tuple[int, int]],
) -> tuple[LaneSegment, ...]:
    if not members:
        raise ValueError("Wikidata pages require at least one member")
    action = GraphAction(
        source_slot=0,
        relation_id=f"P{relation}",
        direction="out",
        read=True,
        halt=False,
        page=page,
    )
    segments: list[TaggedSegment] = [
        TaggedSegment(
            f"Graph values from Q{subject} via P{relation}, page {page}. ",
            "plain",
        ),
        TaggedSegment(tok.decode(serialize_action(action, tok)), "action"),
        TaggedSegment("<|graph_return|>", "action"),
        TaggedSegment("[", "action"),
    ]
    for index, (ordinal, target) in enumerate(members):
        if index:
            segments.append(TaggedSegment(",", "action"))
        row, fact_id = _wikidata_row(subject, relation, target, ordinal)
        # Keep each complete-once triple as its own atomic factual surface even
        # when several targets share one graph address.  The identical surface
        # is reused by functional Wikidata path records, while page framing
        # remains internal operator syntax.
        segments.append(serialize_return(row, fact_id)[1])
    segments.extend(
        (
            TaggedSegment("]", "action"),
            TaggedSegment("<|graph_end|>", "action"),
        )
    )
    return _lane_segments(segments)


class _WikidataGraphSource(_RecordSource):
    _NEXT_RELATION_SQL = """
        SELECT subject, relation
        FROM triples
        WHERE subject = ? AND relation > ?
        ORDER BY relation, target
        LIMIT 1
    """
    _NEXT_SUBJECT_SQL = """
        SELECT subject, relation
        FROM triples
        WHERE subject > ?
        ORDER BY subject, relation, target
        LIMIT 1
    """

    def __init__(self, index: _WikidataIndex):
        self.index = index
        self.tok = get_tok()

    def _next_group(
        self,
        previous_subject: int,
        previous_relation: int,
    ) -> tuple[int, int] | None:
        # Keep both seeks indexable.  Expressing this as one OR/GROUP BY query
        # makes SQLite scan every preceding address on each call, turning a
        # linear walk into a quadratic one on the production Wikidata index.
        group = self.index.connection.execute(
            self._NEXT_RELATION_SQL,
            (previous_subject, previous_relation),
        ).fetchone()
        if group is None:
            group = self.index.connection.execute(
                self._NEXT_SUBJECT_SQL,
                (previous_subject,),
            ).fetchone()
        if group is None:
            return None
        return tuple(map(int, group))

    def _record(
        self,
        subject: int,
        relation: int,
        page: int,
        members: Sequence[tuple[int, int]],
    ) -> LaneRecord:
        return LaneRecord(
            (
                f"wikidata-graph:{subject}:{relation}:{page}:"
                f"{members[0][0]}:{members[-1][0]}"
            ),
            "wikidata5m",
            _wikidata_page_segments(
                self.tok,
                subject,
                relation,
                page,
                members,
            ),
        )

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        state = dict(cursor)
        while True:
            if "after_target" in state:
                subject = int(state["subject"])
                relation = int(state["relation"])
                after_target = int(state["after_target"])
                page = int(state["page"])
            else:
                previous_subject = int(state.get("subject", -1))
                previous_relation = int(state.get("relation", -1))
                group = self._next_group(
                    previous_subject,
                    previous_relation,
                )
                if group is None:
                    raise StopIteration
                subject, relation = group
                after_target = -1
                page = 0

            rows = self.index.connection.execute(
                """
                SELECT ordinal, target
                FROM triples
                WHERE subject = ? AND relation = ? AND target > ?
                ORDER BY target
                """,
                (subject, relation, after_target),
            )
            members: list[tuple[int, int]] = []
            selected: LaneRecord | None = None
            has_more = False
            for ordinal, target in rows:
                candidate_members = (*members, (int(ordinal), int(target)))
                candidate = self._record(
                    subject,
                    relation,
                    page,
                    candidate_members,
                )
                try:
                    _encode_record(self.tok, candidate)
                except V2MaterializationError as error:
                    if error.code != "record_exceeds_context":
                        raise
                    if not members:
                        raise V2MaterializationError(
                            "wikidata_atomic_fact_exceeds_context",
                            f"Q{subject}/P{relation}/Q{target} cannot fit in "
                            "one graph page",
                            lane="wikidata_graph",
                        ) from error
                    has_more = True
                    break
                members.append((int(ordinal), int(target)))
                selected = candidate
            if selected is None:
                # This is reachable only from a stale hand-written cursor.
                # Advance deterministically rather than replaying a prior page.
                state = {
                    "phase": "graph",
                    "relation": relation,
                    "subject": subject,
                }
                continue
            if has_more:
                next_cursor = {
                    "after_target": members[-1][1],
                    "page": page + 1,
                    "phase": "graph",
                    "relation": relation,
                    "subject": subject,
                }
            else:
                next_cursor = {
                    "phase": "graph",
                    "relation": relation,
                    "subject": subject,
                }
            return selected, next_cursor


class _WikidataAliasSource(_RecordSource):
    def __init__(self, stage_root: Path):
        self.files = (
            stage_root / "wikidata5m" / "files" / "wikidata5m_entity.txt",
            stage_root / "wikidata5m" / "files" / "wikidata5m_relation.txt",
        )
        self.tok = get_tok()
        self._open_index = -1
        self._stream = None

    def _open(self, index: int) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = self.files[index].open("rb")
        self._open_index = index

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        file_index = int(cursor.get("file", 0))
        offset = int(cursor.get("offset", 0))
        alias_index = int(cursor.get("alias", 0))
        while file_index < len(self.files):
            if self._open_index != file_index:
                self._open(file_index)
            assert self._stream is not None
            self._stream.seek(offset)
            line = self._stream.readline()
            if not line:
                file_index += 1
                offset = alias_index = 0
                continue
            next_offset = self._stream.tell()
            fields = line.decode("utf-8").rstrip("\r\n").split("\t")
            if len(fields) < 2:
                raise V2MaterializationError(
                    "wikidata_alias_invalid",
                    f"{self.files[file_index]} contains an invalid alias row",
                    lane="wikidata_graph",
                )
            canonical_id = fields[0]
            if alias_index >= len(fields) - 1:
                offset = next_offset
                alias_index = 0
                continue
            alias = fields[alias_index + 1]
            payload = json.dumps(
                {"alias": alias, "canonical_id": canonical_id},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if len(self.tok.encode(payload)) + 1 > MAX_RECORD_TOKENS:
                alias_index += 1
                continue
            digest = hashlib.sha256(f"{canonical_id}\0{alias}".encode()).hexdigest()
            if alias_index + 2 < len(fields):
                next_cursor = {
                    "alias": alias_index + 1,
                    "file": file_index,
                    "offset": offset,
                    "phase": "aliases",
                }
            else:
                next_cursor = {
                    "alias": 0,
                    "file": file_index,
                    "offset": next_offset,
                    "phase": "aliases",
                }
            return (
                LaneRecord(
                    f"wikidata-alias:{file_index}:{offset}:{alias_index}",
                    "wikidata5m",
                    (
                        LaneSegment("Wikidata alias declaration: "),
                        LaneSegment(payload, f"wikidata-alias:{digest}"),
                    ),
                ),
                next_cursor,
            )
        raise StopIteration

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()


def _compile_wikidata_graph(
    writer: _LaneWriter,
    index: _WikidataIndex,
    stage_root: Path,
    *,
    finish_window: int,
    max_finish_candidates: int,
) -> None:
    if writer.complete:
        return
    tok = get_tok()
    phase = str(writer.cursor.get("phase", "graph"))
    if phase == "graph":
        source = _WikidataGraphSource(index)
        cursor = dict(writer.cursor)
        cursor.setdefault("phase", "graph")
        try:
            while True:
                try:
                    record, next_cursor = source.next(cursor)
                except StopIteration:
                    break
                encoded = _encode_record(tok, record)
                if encoded.token_count > writer.remaining:
                    raise V2MaterializationError(
                        "wikidata_complete_once_exceeds_quota",
                        "the compact complete-once Wikidata graph exceeds its "
                        "frozen lane quota",
                        lane="wikidata_graph",
                        action=(
                            "freeze a scientifically reviewed compact graph "
                            "serialization; do not drop training triples"
                        ),
                    )
                writer.add(encoded, next_cursor)
                cursor = next_cursor
        finally:
            source.close()
        writer.cursor = {
            "alias": 0,
            "file": 0,
            "offset": 0,
            "phase": "aliases",
        }
        writer.checkpoint()
    if writer.remaining == 0:
        writer.mark_complete()
        return
    _compile_exact_lane(
        writer,
        _WikidataAliasSource(stage_root),
        finish_window=finish_window,
        max_finish_candidates=max_finish_candidates,
    )


class _SyntheticGraphSource(_RecordSource):
    def __init__(self):
        self.tok = get_tok()
        self._world_index = -1
        self._world = None

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        world_index = int(cursor.get("world", 0))
        fact_index = int(cursor.get("fact", 0))
        if self._world_index != world_index:
            self._world = generate_world(
                _GRAPH_WORLD_OFFSET + world_index,
                WorldConfig(n_entities=64, seed=0x5A17),
            )
            self._world_index = world_index
        assert self._world is not None
        if fact_index >= len(self._world.facts):
            world_index += 1
            fact_index = 0
            self._world = generate_world(
                _GRAPH_WORLD_OFFSET + world_index,
                WorldConfig(n_entities=64, seed=0x5A17),
            )
            self._world_index = world_index
        fact = self._world.facts[fact_index]
        next_fact = fact_index + 1
        next_world = world_index
        if next_fact == len(self._world.facts):
            next_world += 1
            next_fact = 0
        return (
            LaneRecord(
                f"synthetic-graph:{self._world.world_id}:{fact_index}",
                "synthetic_graph_generator",
                _graph_record_segments(
                    self.tok,
                    fact.row,
                    f"synthetic:{fact.fact_id}",
                ),
            ),
            {"fact": next_fact, "world": next_world},
        )


def _trace_segments(
    tok: Any,
    prompt: str,
    steps: Sequence[tuple[GraphRow, str]],
    *,
    candidate: str,
    final: str,
    source_slots: Sequence[int] | None = None,
) -> tuple[LaneSegment, ...]:
    slots = (0,) * len(steps) if source_slots is None else tuple(source_slots)
    if len(slots) != len(steps):
        raise ValueError("trace source slots must align with graph reads")
    if not 1 <= len(steps) < REASONING_ACTION_SLOTS:
        raise ValueError(
            f"reasoning traces require one through {REASONING_ACTION_SLOTS - 1} reads"
        )
    segments: list[TaggedSegment] = [TaggedSegment(prompt, "query")]
    for index in range(REASONING_ACTION_SLOTS):
        if index < len(steps):
            row, fact_id = steps[index]
            source_slot = slots[index]
            action = GraphAction(
                source_slot=source_slot,
                relation_id=row.relation_id,
                direction=row.direction,
                read=True,
                halt=False,
                page=row.page,
            )
            segments.append(
                TaggedSegment(tok.decode(serialize_action(action, tok)), "action")
            )
            segments.extend(serialize_return(row, fact_id))
            state = serialize_answer_state(
                AnswerPointer(
                    slot=source_slot,
                    read_index=index,
                    member_index=0,
                ),
                phase="candidate",
            )
        else:
            action = GraphAction(
                0,
                "r0",
                "out",
                read=False,
                halt=index == len(steps),
            )
            segments.append(
                TaggedSegment(tok.decode(serialize_action(action, tok)), "action")
            )
            segments.extend(serialize_return(None, None))
            state = candidate
        segments.append(TaggedSegment(state, "candidate_state"))
    segments.append(TaggedSegment(final, "final_answer"))
    return _lane_segments(segments)


class _SyntheticMultihopSource(_RecordSource):
    def __init__(self):
        self.tok = get_tok()
        self._key: tuple[int, int] | None = None
        self._world = None
        self._pairs = ()

    def _load(self, world_index: int, hops: int) -> None:
        world = generate_world(
            _MULTIHOP_WORLD_OFFSET + world_index,
            WorldConfig(n_entities=64, seed=0xC0A5E),
        )
        pairs = _generate_eval_pairs(
            world,
            n_pairs_per_task=8,
            seed=0xA11CE ^ world_index,
            path_hops=hops,
        )
        self._world = world
        composition_pairs = tuple(
            pair for pair in pairs if pair.task == "path_composition"
        )
        self._key = (world_index, hops)
        if len(composition_pairs) != 8:
            raise V2MaterializationError(
                "synthetic_composition_generator_drift",
                "srgm_worlds did not emit eight path-composition pairs",
                lane="verified_synthetic_multihop",
            )
        self._pairs = tuple(
            pair
            for pair in composition_pairs
            if len(set(pair.original.meta["gold_fact_ids"]))
            == len(pair.original.meta["gold_fact_ids"])
        )
        if not self._pairs:
            raise V2MaterializationError(
                "synthetic_composition_fresh_path_unavailable",
                "srgm_worlds emitted no composition path without repeated edges",
                lane="verified_synthetic_multihop",
            )

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        world_index = int(cursor.get("world", 0))
        pair_index = int(cursor.get("pair", 0))
        hops = 2 if world_index % 2 == 0 else 4
        if self._key != (world_index, hops):
            self._load(world_index, hops)
        assert self._world is not None
        if pair_index >= len(self._pairs):
            world_index += 1
            pair_index = 0
            hops = 2 if world_index % 2 == 0 else 4
            self._load(world_index, hops)
        pair = self._pairs[pair_index]
        item = pair.original
        fact_map = {fact.fact_id: fact for fact in self._world.facts}
        facts = [fact_map[str(fact_id)] for fact_id in item.meta["gold_fact_ids"]]
        premises = tuple(
            CompositionPremise(
                fact_id=f"synthetic:{fact.fact_id}",
                hop=hop,
                compose_code=int(dict(fact.row.qualifiers)["compose"]),
            )
            for hop, fact in enumerate(facts)
        )
        proof = solve_graph_composition(premises)
        if dict(proof.conclusion)["relation"] != item.answer:
            raise V2MaterializationError(
                "synthetic_composition_answer_mismatch",
                "srgm_worlds answer disagrees with deterministic solver replay",
                lane="verified_synthetic_multihop",
            )
        steps = tuple((fact.row, f"synthetic:{fact.fact_id}") for fact in facts)
        next_pair = pair_index + 1
        next_world = world_index
        if next_pair == len(self._pairs):
            next_world += 1
            next_pair = 0
        return (
            LaneRecord(
                f"synthetic-multihop:{item.qid}",
                "verified_synthetic_multihop_generator",
                _trace_segments(
                    self.tok,
                    item.prompt,
                    steps,
                    candidate="candidate=unresolved",
                    final=f"composed_relation={item.answer}",
                ),
                _solver_bundle(premises),
            ),
            {"pair": next_pair, "world": next_world},
        )


class _WikidataPathSource(_RecordSource):
    def __init__(self, index: _WikidataIndex):
        self.index = index
        self.tok = get_tok()

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        first_ordinal = int(cursor.get("first", -1))
        second_after = int(cursor.get("second", -1))
        while True:
            if second_after >= 0:
                first = self.index.connection.execute(
                    """
                    SELECT ordinal, subject, relation, target
                    FROM functional_triples
                    WHERE ordinal = ?
                    """,
                    (first_ordinal,),
                ).fetchone()
            else:
                first = self.index.connection.execute(
                    """
                    SELECT ordinal, subject, relation, target
                    FROM functional_triples
                    WHERE ordinal > ?
                    ORDER BY ordinal
                    LIMIT 1
                    """,
                    (first_ordinal,),
                ).fetchone()
            if first is None:
                raise StopIteration
            first_ordinal = int(first[0])
            second = self.index.connection.execute(
                """
                SELECT ordinal, subject, relation, target
                FROM functional_triples
                WHERE subject = ? AND ordinal > ? AND ordinal != ?
                ORDER BY ordinal
                LIMIT 1
                """,
                (int(first[3]), second_after, first_ordinal),
            ).fetchone()
            if second is None:
                # Advance to the next functional first edge.  Persisting the
                # last examined ordinal makes this restart-safe despite gaps
                # in the source triple ordinals.
                second_after = -1
                continue
            second_ordinal = int(second[0])
            row_a, fact_a = _wikidata_row(
                int(first[1]),
                int(first[2]),
                int(first[3]),
                first_ordinal,
            )
            row_b, fact_b = _wikidata_row(
                int(second[1]),
                int(second[2]),
                int(second[3]),
                second_ordinal,
            )
            premises = (
                GraphTraversalPremise(
                    fact_a,
                    0,
                    str(row_a.source_id),
                    row_a.relation_id,
                    row_a.target,
                ),
                GraphTraversalPremise(
                    fact_b,
                    1,
                    str(row_b.source_id),
                    row_b.relation_id,
                    row_b.target,
                ),
            )
            final_pointer = serialize_answer_state(
                AnswerPointer(slot=0, read_index=1, member_index=0),
                phase="final",
            )
            return (
                LaneRecord(
                    f"wikidata-path:{first_ordinal}:{second_ordinal}",
                    "wikidata_path_reasoning_generator",
                    _trace_segments(
                        self.tok,
                        (
                            f"Start at Q{first[1]}; follow P{first[2]} then "
                            f"P{second[2]}. Return the endpoint pointer."
                        ),
                        ((row_a, fact_a), (row_b, fact_b)),
                        candidate="candidate=unresolved",
                        final=final_pointer,
                    ),
                    _solver_bundle(premises),
                ),
                {"first": first_ordinal, "second": second_ordinal},
            )


class _RelationalRefinementSource(_RecordSource):
    def __init__(self):
        self.tok = get_tok()
        self._world_index = -1
        self._world = None
        self._pairs = ()

    def _load(self, world_index: int) -> None:
        world = generate_world(
            _REFINEMENT_WORLD_OFFSET + world_index,
            WorldConfig(n_entities=64, seed=0x5EED),
        )
        pairs = _generate_eval_pairs(
            world,
            n_pairs_per_task=8,
            seed=0xE9A1 ^ world_index,
            path_hops=2,
        )
        self._world = world
        equality_pairs = tuple(
            pair for pair in pairs if pair.task == "balanced_equality"
        )
        self._world_index = world_index
        if len(equality_pairs) != 8:
            raise V2MaterializationError(
                "relational_refinement_generator_drift",
                "srgm_worlds did not emit eight balanced-equality pairs",
                lane="relational_refinement",
            )
        self._pairs = tuple(
            pair
            for pair in equality_pairs
            if len(set(pair.original.meta["gold_fact_ids"])) == 2
        )
        if not self._pairs:
            raise V2MaterializationError(
                "relational_refinement_fresh_pair_unavailable",
                "srgm_worlds emitted no equality pair with two distinct facts",
                lane="relational_refinement",
            )

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        world_index = int(cursor.get("world", 0))
        pair_index = int(cursor.get("pair", 0))
        if self._world_index != world_index:
            self._load(world_index)
        assert self._world is not None
        if pair_index >= len(self._pairs):
            world_index += 1
            pair_index = 0
            self._load(world_index)
        pair = self._pairs[pair_index]
        item = pair.original
        fact_map = {fact.fact_id: fact for fact in self._world.facts}
        facts = [fact_map[str(fact_id)] for fact_id in item.meta["gold_fact_ids"]]
        if len(facts) != 2:
            raise V2MaterializationError(
                "relational_refinement_premise_drift",
                "balanced equality must expose exactly two facts",
                lane="relational_refinement",
            )
        premises = tuple(
            EqualityPremise(
                fact_id=f"refinement:{fact.fact_id}",
                slot=slot,
                value=fact.row.target,
            )
            for slot, fact in enumerate(facts)
        )
        proof = solve_slot_equality(premises)
        expected = bool(dict(proof.conclusion)["equal"])
        if item.answer != ("yes" if expected else "no"):
            raise V2MaterializationError(
                "relational_refinement_answer_mismatch",
                "srgm_worlds answer disagrees with deterministic solver replay",
                lane="relational_refinement",
            )
        candidate_state = json.dumps(
            {
                "equal": not expected,
                "evidence": [
                    AnswerPointer(slot=slot, read_index=slot).as_dict("candidate")
                    for slot in range(2)
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        final_state = json.dumps(
            {
                "equal": expected,
                "evidence": [
                    AnswerPointer(slot=slot, read_index=slot).as_dict("final")
                    for slot in range(2)
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        next_pair = pair_index + 1
        next_world = world_index
        if next_pair == len(self._pairs):
            next_world += 1
            next_pair = 0
        return (
            LaneRecord(
                f"relational-refinement:{item.qid}",
                "relational_refinement_generator",
                _trace_segments(
                    self.tok,
                    item.prompt,
                    tuple((fact.row, f"refinement:{fact.fact_id}") for fact in facts),
                    candidate=f"candidate={candidate_state}",
                    final=f"final={final_state}",
                    source_slots=(0, 1),
                ),
                _solver_bundle(premises),
            ),
            {"pair": next_pair, "world": next_world},
        )


class _ObjectiveWorkerClient:
    def __init__(
        self,
        provider: str,
        source_dir: Path,
        *,
        python: Path,
        log_root: Path,
    ) -> None:
        self.provider = provider
        worker_cwd = log_root / "workers" / provider
        source_resolved = source_dir.resolve()
        worker_cwd_resolved = worker_cwd.resolve()
        if (
            source_resolved == worker_cwd_resolved
            or source_resolved in worker_cwd_resolved.parents
            or worker_cwd_resolved in source_resolved.parents
        ):
            raise V2MaterializationError(
                "objective_work_root_overlaps_source",
                "objective worker scratch directory overlaps its staged source",
                lane="objective_auxiliary",
                action="use a work root disjoint from the immutable source stage",
            )
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_root / f"{provider}.stderr.log").open("ab")
        worker_cwd.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        repository = str(Path(__file__).resolve().parents[1])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = repository
        env.pop("PYTHONHOME", None)
        try:
            self.process = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "corpusgen.v2_objective",
                    "--provider",
                    provider,
                    "--source-dir",
                    str(source_dir),
                ],
                cwd=worker_cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log_handle,
                env=env,
            )
        except OSError as error:
            self.log_handle.close()
            raise V2MaterializationError(
                "objective_interpreter_unavailable",
                f"cannot start {provider} with {python}: {error}",
                lane="objective_auxiliary",
                action="provide a working --objective-python interpreter",
            ) from error
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        try:
            handshake = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.close()
            raise V2MaterializationError(
                "objective_worker_protocol_failed",
                f"{provider} emitted no valid worker handshake",
                lane="objective_auxiliary",
                action=f"inspect {self.log_handle.name}",
            ) from error
        if (
            handshake.get("format") != WORKER_FORMAT
            or handshake.get("provider") != provider
            or handshake.get("ready") is not True
        ):
            self.close()
            detail = handshake.get("error", {})
            raise V2MaterializationError(
                "objective_runtime_missing",
                f"{provider} cannot start: {detail.get('type')}: "
                f"{detail.get('message')}",
                lane="objective_auxiliary",
                action=(
                    "install the pinned generator's declared runtime dependencies "
                    f"into {python}; no substitute generator is permitted"
                ),
            )
        self.runtime = dict(handshake["runtime"])

    def generate(self, index: int) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise V2MaterializationError(
                "objective_worker_exited",
                f"{self.provider} worker exited before record {index}",
                lane="objective_auxiliary",
                action=f"inspect {self.log_handle.name}",
            )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(canonical_json_bytes({"index": index}))
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        try:
            response = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V2MaterializationError(
                "objective_worker_protocol_failed",
                f"{self.provider} returned no valid record {index}",
                lane="objective_auxiliary",
                action=f"inspect {self.log_handle.name}",
            ) from error
        if "error" in response:
            detail = response["error"]
            raise V2MaterializationError(
                "objective_native_generation_failed",
                f"{self.provider} record {index} failed: "
                f"{detail.get('type')}: {detail.get('message')}",
                lane="objective_auxiliary",
            )
        if response.get("index") != index or not isinstance(
            response.get("record"), dict
        ):
            raise V2MaterializationError(
                "objective_worker_protocol_failed",
                f"{self.provider} record identity drifted",
                lane="objective_auxiliary",
            )
        return dict(response["record"])

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        if not self.log_handle.closed:
            self.log_handle.close()


class _PuzzleProvider:
    def __init__(self, source_id: str, files: Sequence[Path]):
        self.source_id = source_id
        self.files = tuple(files)

    def generate(
        self,
        cursor: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, int]] | None:
        file_index = int(cursor.get("file", 0))
        test_index = int(cursor.get("test", 0))
        while file_index < len(self.files):
            task = json.loads(self.files[file_index].read_bytes())
            tests = task.get("test", [])
            if test_index >= len(tests):
                file_index += 1
                test_index = 0
                continue
            example = tests[test_index]
            if not isinstance(example, dict) or not {
                "input",
                "output",
            }.issubset(example):
                raise V2MaterializationError(
                    "objective_training_task_invalid",
                    f"{self.files[file_index]} lacks a test input/output",
                    lane="objective_auxiliary",
                )
            question = json.dumps(
                {
                    "demonstrations": task.get("train", []),
                    "input": example["input"],
                    "task_sha256": _sha256_file(self.files[file_index]),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            return (
                {
                    "answer": example["output"],
                    "metadata": {
                        "file": self.files[file_index].name,
                        "test_index": test_index,
                    },
                    "question": question,
                },
                {"file": file_index, "test": test_index + 1},
            )
        return None


def _objective_component_roots(
    stage_root: Path,
    lock: V2SourceSetLock,
) -> tuple[dict[str, Path], dict[str, tuple[Path, ...]]]:
    procedural: dict[str, Path] = {}
    puzzles: dict[str, tuple[Path, ...]] = {}
    objective_root = stage_root / "objective_auxiliary"
    for source in lock.sources["objective_auxiliary"]["sources"]:
        source_id = str(source["id"])
        component_roots = [
            objective_root / source_id / str(component["id"])
            for component in source["components"]
        ]
        if source_id in PROCEDURAL_PROVIDERS:
            if len(component_roots) != 1:
                raise V2MaterializationError(
                    "objective_component_count_drift",
                    f"{source_id} requires exactly one pinned implementation tree",
                    lane="objective_auxiliary",
                )
            procedural[source_id] = component_roots[0]
        else:
            paths = []
            for component, component_root in zip(source["components"], component_roots):
                for artifact in component["files"]:
                    relative = str(artifact["path"])
                    if relative.endswith(".json"):
                        paths.append(component_root / relative)
            puzzles[source_id] = tuple(sorted(paths))
    if tuple(procedural) != PROCEDURAL_PROVIDERS:
        raise V2MaterializationError(
            "objective_procedural_source_drift",
            "pinned objective procedural source order drifted",
            lane="objective_auxiliary",
        )
    if set(puzzles) != {"arc_agi_training", "conceptarc_training"}:
        raise V2MaterializationError(
            "objective_puzzle_source_drift",
            "ARC/ConceptARC training source identities drifted",
            lane="objective_auxiliary",
        )
    return procedural, puzzles


class _ObjectiveSource(_RecordSource):
    def __init__(
        self,
        stage_root: Path,
        lock: V2SourceSetLock,
        *,
        objective_python: Path,
        work_root: Path,
        expected_runtime: Mapping[str, Any],
    ) -> None:
        procedural, puzzles = _objective_component_roots(stage_root, lock)
        self.clients: dict[str, _ObjectiveWorkerClient] = {}
        try:
            for provider, root in procedural.items():
                self.clients[provider] = _ObjectiveWorkerClient(
                    provider,
                    root,
                    python=objective_python,
                    log_root=work_root / "logs" / "objective",
                )
        except BaseException:
            for client in self.clients.values():
                client.close()
            raise
        self.puzzles = {
            source_id: _PuzzleProvider(source_id, paths)
            for source_id, paths in puzzles.items()
        }
        self.providers = tuple(FROZEN_OBJECTIVE_SOURCE_IDS)
        self.tok = get_tok()
        self.runtime = {
            provider: client.runtime for provider, client in self.clients.items()
        }
        self.expected_runtime = expected_runtime

    def next(self, cursor: Mapping[str, Any]) -> tuple[LaneRecord, dict[str, Any]]:
        state = {
            "indices": {
                provider: int(cursor.get("indices", {}).get(provider, 0))
                for provider in self.providers
            },
            "next_provider": int(cursor.get("next_provider", 0)),
            "puzzle_cursors": {
                provider: dict(cursor.get("puzzle_cursors", {}).get(provider, {}))
                for provider in self.puzzles
            },
            "puzzle_exhausted": list(cursor.get("puzzle_exhausted", [])),
            "puzzle_tokens": int(cursor.get("puzzle_tokens", 0)),
        }
        exhausted = set(state["puzzle_exhausted"])
        for _ in range(100_000):
            provider = self.providers[state["next_provider"] % len(self.providers)]
            state["next_provider"] = (state["next_provider"] + 1) % len(self.providers)
            index = state["indices"][provider]
            if provider in self.clients:
                native = self.clients[provider].generate(index)
                state["indices"][provider] = index + 1
            else:
                if provider in exhausted:
                    continue
                generated = self.puzzles[provider].generate(
                    state["puzzle_cursors"][provider]
                )
                if generated is None:
                    exhausted.add(provider)
                    state["puzzle_exhausted"] = sorted(exhausted)
                    continue
                native, puzzle_cursor = generated
                state["puzzle_cursors"][provider] = puzzle_cursor
                state["indices"][provider] = index + 1
            probe_hashes = self.expected_runtime[provider]["probe_record_sha256s"]
            if index < len(probe_hashes):
                observed_probe = hashlib.sha256(
                    canonical_json_bytes(native)
                ).hexdigest()
                if observed_probe != probe_hashes[index]:
                    raise V2MaterializationError(
                        "objective_probe_replay_drift",
                        f"{provider} record {index} differs from its preflight replay",
                        lane="objective_auxiliary",
                        action=(
                            "do not resume or substitute; repair the "
                            "nondeterministic runtime"
                        ),
                    )
            answer = native["answer"]
            answer_text = json.dumps(
                answer,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            text = (
                f"Objective source={provider}\nQuestion: "
                f"{native['question']}\nAnswer: {answer_text}"
            )
            record = LaneRecord(
                f"objective:{provider}:{index}",
                "objective_auxiliary",
                (LaneSegment(text),),
                {
                    "answer": answer,
                    "kind": "objective_answer",
                    "reference_answer": answer,
                    "validator": "canonical_exact_match",
                },
            )
            try:
                encoded = _encode_record(self.tok, record)
            except V2MaterializationError as error:
                if error.code == "record_exceeds_context":
                    continue
                raise
            if provider in {"arc_agi_training", "conceptarc_training"}:
                if (
                    state["puzzle_tokens"] + encoded.token_count
                    > ARC_CONCEPTARC_MAX_TOKENS
                ):
                    exhausted.update({"arc_agi_training", "conceptarc_training"})
                    state["puzzle_exhausted"] = sorted(exhausted)
                    continue
                state["puzzle_tokens"] += encoded.token_count
            return record, state
        raise V2MaterializationError(
            "objective_valid_record_unavailable",
            "objective providers produced 100000 unusable records",
            lane="objective_auxiliary",
        )

    def close(self) -> None:
        for client in self.clients.values():
            client.close()


def _preflight_objective_runtime(
    stage_root: Path,
    lock: V2SourceSetLock,
    *,
    objective_python: Path,
    work_root: Path,
) -> dict[str, Any]:
    procedural, puzzles = _objective_component_roots(stage_root, lock)
    runtime: dict[str, Any] = {}
    failures = []
    for provider, source_dir in procedural.items():
        try:
            probe_runs = []
            versions = []
            probe_orders = (
                _OBJECTIVE_PROBE_INDICES,
                tuple(reversed(_OBJECTIVE_PROBE_INDICES)),
            )
            for probe_order in probe_orders:
                client = _ObjectiveWorkerClient(
                    provider,
                    source_dir,
                    python=objective_python,
                    log_root=work_root / "logs" / "objective",
                )
                try:
                    versions.append(client.runtime)
                    probe_runs.append(
                        {index: client.generate(index) for index in probe_order}
                    )
                finally:
                    client.close()
            if versions[0] != versions[1] or probe_runs[0] != probe_runs[1]:
                raise V2MaterializationError(
                    "objective_generator_nondeterministic",
                    f"{provider} changed across forward and reverse isolated "
                    "probe replays",
                    lane="objective_auxiliary",
                    action=(
                        "repair mutable generator state or pin the native runtime; "
                        "resuming at an ordinal must reproduce the same record"
                    ),
                )
            runtime[provider] = {
                "probe_indices": list(_OBJECTIVE_PROBE_INDICES),
                "probe_record_sha256s": [
                    hashlib.sha256(
                        canonical_json_bytes(probe_runs[0][index])
                    ).hexdigest()
                    for index in _OBJECTIVE_PROBE_INDICES
                ],
                "versions": versions[0],
            }
        except V2MaterializationError as error:
            failures.append(error.as_dict())
    for provider, files in puzzles.items():
        try:
            generated = _PuzzleProvider(provider, files).generate({})
            if generated is None:
                raise V2MaterializationError(
                    "objective_puzzle_source_empty",
                    f"{provider} contains no training question with a sealed answer",
                    lane="objective_auxiliary",
                )
            probe, _cursor = generated
            runtime[provider] = {
                "files": len(files),
                "probe_indices": [0],
                "probe_record_sha256s": [
                    hashlib.sha256(canonical_json_bytes(probe)).hexdigest()
                ],
                "versions": {"python": ".".join(map(str, sys.version_info[:3]))},
            }
        except (OSError, TypeError, ValueError, V2MaterializationError) as error:
            failures.append(
                error.as_dict()
                if isinstance(error, V2MaterializationError)
                else {
                    "code": "objective_puzzle_preflight_failed",
                    "lane": "objective_auxiliary",
                    "message": f"{provider}: {type(error).__name__}: {error}",
                }
            )
    if failures:
        raise V2MaterializationError(
            "objective_runtime_preflight_failed",
            "one or more pinned objective generators cannot replay: "
            + json.dumps(
                failures,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            lane="objective_auxiliary",
            action=(
                "install each reported pinned runtime dependency into "
                f"{objective_python}; no substitute generator is permitted"
            ),
        )
    _write_json_atomic(work_root / "objective-runtime.json", runtime)
    return runtime


def _route_all_lanes(
    work_root: Path,
    recipe: ProductionRecipe,
) -> dict[str, Any]:
    receipt_path = work_root / "routing-receipt.json"
    input_identity = {}
    for lane in recipe.lanes:
        lane_root = work_root / "lanes" / lane
        checkpoint_path = lane_root / "checkpoint.json"
        facts_path = lane_root / "facts.sqlite3"
        checkpoint = json.loads(checkpoint_path.read_bytes())
        if (
            checkpoint.get("complete") is not True
            or checkpoint.get("tokens") != recipe.quota_by_lane[lane]
        ):
            raise V2MaterializationError(
                "routing_lane_incomplete",
                f"{lane} is not complete at its frozen quota",
                lane=lane,
            )
        facts_stat = facts_path.stat()
        input_identity[lane] = {
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "facts_bytes": facts_stat.st_size,
            "facts_mtime_ns": facts_stat.st_mtime_ns,
        }
    expected_outputs = [
        work_root / "lanes" / lane / name
        for lane in recipe.lanes
        for name in ("routes.jsonl", "masks.jsonl", "split90.weights.bin")
    ]
    if receipt_path.is_file() and all(path.is_file() for path in expected_outputs):
        existing = json.loads(receipt_path.read_bytes())
        split_sizes_match = all(
            (work_root / "lanes" / lane / "split90.weights.bin").stat().st_size
            == recipe.quota_by_lane[lane]
            for lane in recipe.lanes
        )
        output_identity = existing.get("output_identity")
        output_identity_matches = (
            isinstance(output_identity, dict)
            and set(output_identity)
            == {path.relative_to(work_root).as_posix() for path in expected_outputs}
            and all(
                not path.is_symlink()
                and output_identity[path.relative_to(work_root).as_posix()]
                == {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in expected_outputs
            )
        )
        if (
            existing.get("format") == "memorysplit-v2-global-routing-v1"
            and existing.get("input_identity") == input_identity
            and split_sizes_match
            and output_identity_matches
        ):
            return existing
    database = work_root / "routing.sqlite3"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE facts (
            fact_id TEXT PRIMARY KEY,
            surface_sha256 TEXT NOT NULL,
            burden_bits INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE lane_facts (
            lane TEXT NOT NULL,
            fact_id TEXT NOT NULL,
            PRIMARY KEY(lane, fact_id)
        ) WITHOUT ROWID;
        CREATE TABLE routes (
            fact_id TEXT PRIMARY KEY,
            external INTEGER NOT NULL CHECK(external IN (0, 1))
        ) WITHOUT ROWID;
        """
    )
    try:
        for lane in recipe.lanes:
            lane_db = sqlite3.connect(work_root / "lanes" / lane / "facts.sqlite3")
            try:
                rows = lane_db.execute(
                    """
                    SELECT f.fact_id, f.surface_sha256,
                           SUM(o.burden_bits)
                    FROM facts AS f
                    JOIN occurrences AS o USING(fact_id)
                    GROUP BY f.fact_id, f.surface_sha256
                    ORDER BY f.fact_id
                    """
                )
                for fact_id, surface, burden in rows:
                    existing = connection.execute(
                        "SELECT surface_sha256, burden_bits FROM facts "
                        "WHERE fact_id = ?",
                        (fact_id,),
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO facts VALUES (?, ?, ?)",
                            (fact_id, surface, int(burden)),
                        )
                    else:
                        if existing[0] != surface:
                            raise V2MaterializationError(
                                "global_fact_surface_drift",
                                f"fact {fact_id!r} has inconsistent "
                                "cross-lane surfaces",
                            )
                        connection.execute(
                            "UPDATE facts SET burden_bits = ? WHERE fact_id = ?",
                            (int(existing[1]) + int(burden), fact_id),
                        )
                    connection.execute(
                        "INSERT INTO lane_facts VALUES (?, ?)",
                        (lane, fact_id),
                    )
            finally:
                lane_db.close()
            connection.commit()
        fact_count = int(connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        if fact_count <= 0:
            raise V2MaterializationError(
                "route_fact_universe_empty",
                "materialized lanes contain no offloadable atomic facts",
            )
        quota = (9 * fact_count + 9) // 10
        connection.execute("INSERT INTO routes SELECT fact_id, 0 FROM facts")
        connection.execute(
            """
            UPDATE routes SET external = 1
            WHERE fact_id IN (
                SELECT fact_id FROM facts
                ORDER BY burden_bits DESC, fact_id
                LIMIT ?
            )
            """,
            (quota,),
        )
        connection.commit()
        total_burden, external_burden = connection.execute(
            """
            SELECT SUM(f.burden_bits),
                   SUM(CASE WHEN r.external THEN f.burden_bits ELSE 0 END)
            FROM facts AS f JOIN routes AS r USING(fact_id)
            """
        ).fetchone()
        if (
            int(external_burden) * 10 < int(total_burden) * 9
            or quota * 10 < fact_count * 9
        ):
            raise AssertionError("top-burden Split90 selection missed its dose")

        lane_reports = {}
        for lane in recipe.lanes:
            lane_root = work_root / "lanes" / lane
            route_tmp = lane_root / ".routes.partial"
            mask_tmp = lane_root / ".masks.partial"
            split_tmp = lane_root / ".split90.partial"
            for path in (route_tmp, mask_tmp, split_tmp):
                path.unlink(missing_ok=True)
            lane_db = sqlite3.connect(lane_root / "facts.sqlite3")
            lane_db.execute("ATTACH DATABASE ? AS global_route", (str(database),))
            zero_tokens = 0
            position = 0
            route_count = 0
            mask_count = 0
            with (
                route_tmp.open("xb") as route_file,
                mask_tmp.open("xb") as mask_file,
                split_tmp.open("xb") as split_file,
            ):
                route_rows = lane_db.execute(
                    """
                    SELECT lf.fact_id, r.external, f.burden_bits
                    FROM global_route.lane_facts AS lf
                    JOIN global_route.routes AS r USING(fact_id)
                    JOIN global_route.facts AS f USING(fact_id)
                    WHERE lf.lane = ?
                    ORDER BY lf.fact_id
                    """,
                    (lane,),
                )
                for fact_id, external, burden in route_rows:
                    route_file.write(
                        canonical_json_bytes(
                            {
                                "burden_bits": _fraction_json(int(burden)),
                                "external": bool(external),
                                "fact_id": str(fact_id),
                            }
                        )
                    )
                    route_count += 1
                spans = lane_db.execute(
                    """
                    SELECT o.start, o.end, o.fact_id
                    FROM occurrences AS o
                    JOIN global_route.routes AS r USING(fact_id)
                    WHERE r.external = 1
                    ORDER BY o.start, o.end, o.fact_id
                    """
                )
                for start, end, fact_id in spans:
                    start = int(start)
                    end = int(end)
                    if start < position or end <= start:
                        raise V2MaterializationError(
                            "fact_occurrence_overlap",
                            f"{lane} has overlapping factual payload spans",
                            lane=lane,
                        )
                    _write_repeated(split_file, 1, start - position)
                    _write_repeated(split_file, 0, end - start)
                    mask_file.write(
                        canonical_json_bytes(
                            {
                                "end": end,
                                "fact_id": str(fact_id),
                                "start": start,
                            }
                        )
                    )
                    zero_tokens += end - start
                    position = end
                    mask_count += 1
                _write_repeated(
                    split_file,
                    1,
                    recipe.quota_by_lane[lane] - position,
                )
                for handle in (route_file, mask_file, split_file):
                    handle.flush()
                    os.fsync(handle.fileno())
            lane_db.close()
            route_path = lane_root / "routes.jsonl"
            mask_path = lane_root / "masks.jsonl"
            split_path = lane_root / "split90.weights.bin"
            route_tmp.replace(route_path)
            mask_tmp.replace(mask_path)
            split_tmp.replace(split_path)
            lane_reports[lane] = {
                "mask_rows": mask_count,
                "route_rows": route_count,
                "zero_tokens": zero_tokens,
            }
        receipt = {
            "distinct_external_facts": quota,
            "distinct_facts": fact_count,
            "external_burden_bits": int(external_burden),
            "format": "memorysplit-v2-global-routing-v1",
            "input_identity": input_identity,
            "lanes": lane_reports,
            "output_identity": {
                path.relative_to(work_root).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in expected_outputs
            },
            "target_fraction": _fraction_json(9, 10),
            "total_burden_bits": int(total_burden),
        }
        _write_json_atomic(receipt_path, receipt)
        return receipt
    finally:
        connection.close()


def _repository_identity() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    try:
        repository = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        repository = root.as_uri()
        commit = "unavailable"
    if "://" in repository:
        scheme, remainder = repository.split("://", 1)
        authority, separator, suffix = remainder.partition("/")
        if "@" in authority:
            authority = authority.rsplit("@", 1)[1]
        suffix = suffix.split("?", 1)[0].split("#", 1)[0]
        repository = f"{scheme}://{authority}{separator}{suffix}"
    return repository or root.as_uri(), commit


def _materializer_artifact_sha256s() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        relative: _sha256_file(root / relative) for relative in _MATERIALIZER_ARTIFACTS
    }


def _main_runtime_versions() -> dict[str, str]:
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for distribution in ("pyarrow", "tiktoken"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return dict(sorted(versions.items()))


def _generated_lock_payloads(
    objective_runtime: Mapping[str, Any],
    main_runtime: Mapping[str, str],
    materialization_identity: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[1]
    repository, commit = _repository_identity()
    result = {}
    for source_id, spec in _GENERATED_LOCK_SPECS.items():
        artifacts = []
        for relative in sorted(spec["artifacts"]):
            path = root / relative
            artifacts.append(
                {
                    "bytes": path.stat().st_size,
                    "path": relative,
                    "sha256": _sha256_file(path),
                }
            )
        policy = dict(spec["policy"])
        policy["materializer_version"] = MATERIALIZER_VERSION
        policy["main_runtime"] = dict(main_runtime)
        policy["repository_commit_at_materialization"] = commit
        if materialization_identity is not None:
            policy["materialization_identity"] = dict(materialization_identity)
        if source_id == "reasoning_solver":
            policy["objective_runtime"] = dict(objective_runtime)
        revision = hashlib.sha256(
            canonical_json_bytes({"artifacts": artifacts, "policy": policy})
        ).hexdigest()
        result[source_id] = canonical_json_bytes(
            {
                "artifacts": artifacts,
                "format": _GENERATOR_LOCK_FORMAT,
                "kind": spec["kind"],
                "policy": policy,
                "repository": repository,
                "revision": revision,
                "source_id": source_id,
            }
        )
    return result


def _same_regular_bytes(source: Path, destination: Path) -> bool:
    try:
        source_stat = source.stat()
        destination_stat = destination.lstat()
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or not stat.S_ISREG(destination_stat.st_mode)
        or stat.S_ISLNK(destination_stat.st_mode)
        or source_stat.st_size != destination_stat.st_size
    ):
        return False
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        return True
    return _sha256_file(source) == _sha256_file(destination)


def _copy_or_link(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise V2MaterializationError(
            "publication_source_unsafe",
            f"publication source is missing or unsafe: {source}",
        )
    if destination.exists() or destination.is_symlink():
        if _same_regular_bytes(source, destination):
            return
        if destination.is_symlink() or not destination.is_file():
            raise V2MaterializationError(
                "publication_partial_unsafe",
                f"publication destination is unsafe: {destination}",
            )
        destination.unlink()
    temporary = destination.with_name(f".{destination.name}.copying")
    if temporary.exists() or temporary.is_symlink():
        if _same_regular_bytes(source, temporary):
            temporary.replace(destination)
            return
        if temporary.is_symlink() or not temporary.is_file():
            raise V2MaterializationError(
                "publication_partial_unsafe",
                f"publication temporary path is unsafe: {temporary}",
            )
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        with source.open("rb") as input_file, temporary.open("xb") as output:
            shutil.copyfileobj(input_file, output, length=1 << 20)
            output.flush()
            os.fsync(output.fileno())
    temporary.replace(destination)


def _write_publication_bytes(destination: Path, payload: bytes) -> None:
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        if destination.is_symlink() or not destination.is_file():
            raise V2MaterializationError(
                "publication_partial_unsafe",
                f"publication destination is unsafe: {destination}",
            )
        destination.unlink()
    temporary = destination.with_name(f".{destination.name}.copying")
    if temporary.exists() or temporary.is_symlink():
        if (
            temporary.is_file()
            and not temporary.is_symlink()
            and temporary.read_bytes() == payload
        ):
            temporary.replace(destination)
            return
        if temporary.is_symlink() or not temporary.is_file():
            raise V2MaterializationError(
                "publication_partial_unsafe",
                f"publication temporary path is unsafe: {temporary}",
            )
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _ensure_publication_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise V2MaterializationError(
                "publication_partial_unsafe",
                f"publication directory is unsafe: {path}",
            )
        return
    path.mkdir()


def _published_report(
    report: Mapping[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    result = dict(report)
    result["source_manifest_path"] = str(source_root / "source-manifest.json")
    return result


def _publish_source_root(
    work_root: Path,
    source_root: Path,
    recipe: ProductionRecipe,
    lock_payloads: Mapping[str, bytes],
    *,
    seal: bool,
) -> dict[str, Any] | None:
    required = expected_production_source_paths(recipe)
    required_lock_ids = {
        source_id
        for source_ids in recipe.locks_by_lane.values()
        for source_id in source_ids
    }
    if set(lock_payloads) != required_lock_ids:
        raise V2MaterializationError(
            "source_lock_set_drift",
            f"publication locks differ: expected={sorted(required_lock_ids)}, "
            f"actual={sorted(lock_payloads)}",
        )
    if source_root.exists() or source_root.is_symlink():
        if source_root.is_dir() and (source_root / "source-manifest.json").is_file():
            report = production_preflight(source_root, recipe=recipe)
            if report["ready"] is True:
                mismatched_locks = [
                    source_id
                    for source_id, payload in lock_payloads.items()
                    if (source_root / "locks" / f"{source_id}.lock.json").read_bytes()
                    != payload
                ]
                if mismatched_locks:
                    raise V2MaterializationError(
                        "source_root_identity_conflict",
                        "existing sealed source root was produced by a different "
                        f"materialization identity: {sorted(mismatched_locks)}",
                        action=(
                            "use a new output path; never relabel a sealed source root"
                        ),
                    )
                return report
        raise V2MaterializationError(
            "source_root_already_exists",
            f"refusing to replace existing source root: {source_root}",
            action="choose an absent output path or verify the existing sealed root",
        )
    source_root.parent.mkdir(parents=True, exist_ok=True)
    build_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "recipe": recipe.recipe_sha256,
                "work_root": str(work_root.resolve()),
            }
        )
    ).hexdigest()[:16]
    partial = source_root.parent / f".{source_root.name}.partial-{build_id}"
    _ensure_publication_directory(partial)
    partial_manifest = partial / "source-manifest.json"
    if partial_manifest.exists() or partial_manifest.is_symlink():
        if partial_manifest.is_symlink() or not partial_manifest.is_file():
            raise V2MaterializationError(
                "publication_partial_unsafe",
                f"partial source manifest is unsafe: {partial_manifest}",
            )
        report = production_preflight(partial, recipe=recipe)
        if report["ready"] is not True:
            raise V2MaterializationError(
                "publication_partial_invalid",
                "a sealed partial publication failed production preflight",
                action="inspect or remove the named partial publication directory",
            )
        mismatched_locks = [
            source_id
            for source_id, payload in lock_payloads.items()
            if (partial / "locks" / f"{source_id}.lock.json").read_bytes() != payload
        ]
        if mismatched_locks:
            raise V2MaterializationError(
                "publication_partial_identity_conflict",
                "sealed partial publication has different source locks: "
                f"{sorted(mismatched_locks)}",
            )
        partial.rename(source_root)
        return _published_report(report, source_root)
    _ensure_publication_directory(partial / "materialized")
    _ensure_publication_directory(partial / "ledgers")
    _ensure_publication_directory(partial / "locks")
    for lane in recipe.lanes:
        lane_root = work_root / "lanes" / lane
        _copy_or_link(
            lane_root / "tokens.partial",
            partial / "materialized" / f"{lane}.tokens.bin",
        )
        _copy_or_link(
            lane_root / "split90.weights.bin",
            partial / "materialized" / f"{lane}.split90.weights.bin",
        )
        _copy_or_link(
            lane_root / "routes.jsonl",
            partial / "ledgers" / f"{lane}.routes.jsonl",
        )
        _copy_or_link(
            lane_root / "masks.jsonl",
            partial / "ledgers" / f"{lane}.masks.jsonl",
        )
        if lane in recipe.reasoning_lanes or lane == recipe.objective_lane:
            _copy_or_link(
                lane_root / "verification.partial.jsonl",
                partial / "ledgers" / f"{lane}.verification.jsonl",
            )
    for source_id, payload in lock_payloads.items():
        _write_publication_bytes(
            partial / "locks" / f"{source_id}.lock.json",
            payload,
        )
    actual = {
        path.relative_to(partial).as_posix()
        for path in partial.rglob("*")
        if path.is_file()
    }
    if actual != set(required):
        missing = sorted(set(required) - actual)
        extra = sorted(actual - set(required))
        raise V2MaterializationError(
            "published_contract_file_set_drift",
            f"45-file contract differs: missing={missing}, extra={extra}",
        )
    report = seal_production_sources(partial, recipe=recipe) if seal else None
    partial.rename(source_root)
    if report is not None:
        report = _published_report(report, source_root)
    return report


def materialization_status(
    work_root: Path | str,
    *,
    recipe: ProductionRecipe | None = None,
) -> dict[str, Any]:
    root = Path(work_root)
    recipe_value = recipe or load_production_recipe()
    lanes = {}
    for lane in recipe_value.lanes:
        path = root / "lanes" / lane / "checkpoint.json"
        lanes[lane] = (
            json.loads(path.read_bytes())
            if path.is_file()
            else {
                "complete": False,
                "quota": recipe_value.quota_by_lane[lane],
                "records": 0,
                "tokens": 0,
            }
        )
    return {
        "format": MATERIALIZER_FORMAT,
        "lanes": lanes,
        "routing_complete": (root / "routing-receipt.json").is_file(),
    }


def materialize_v2_source_root(
    stage_root: Path | str,
    source_root: Path | str,
    work_root: Path | str,
    *,
    source_lock_path: Path | str = DEFAULT_V2_SOURCE_LOCK,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    objective_python: Path | str = sys.executable,
    checkpoint_tokens: int = DEFAULT_CHECKPOINT_TOKENS,
    finish_window: int = DEFAULT_FINISH_WINDOW,
    max_finish_candidates: int = DEFAULT_FINISH_CANDIDATES,
    seal: bool = True,
) -> dict[str, Any]:
    recipe = load_production_recipe(recipe_path)
    lock = load_v2_source_lock(source_lock_path)
    # Keep the terminal path component unresolved so the source-stage and
    # publication validators can reject symlink roots instead of silently
    # following them.
    stage = Path(os.path.abspath(os.path.expanduser(str(stage_root))))
    destination = Path(os.path.abspath(os.path.expanduser(str(source_root))))
    work = Path(os.path.abspath(os.path.expanduser(str(work_root))))
    # Preserve a virtual-environment launcher path.  Resolving its symlink to
    # the base interpreter would silently discard that environment's packages.
    objective_interpreter = Path(
        os.path.abspath(os.path.expanduser(str(objective_python)))
    )
    if checkpoint_tokens <= 0 or finish_window <= 0 or max_finish_candidates <= 0:
        raise ValueError("materializer checkpoint and finish options must be positive")
    stage_resolved = stage.resolve()
    work_resolved = work.resolve()
    if (
        stage_resolved == work_resolved
        or stage_resolved in work_resolved.parents
        or work_resolved in stage_resolved.parents
    ):
        raise V2MaterializationError(
            "work_root_overlaps_source_stage",
            "materializer work root overlaps the immutable source stage",
            lane="objective_auxiliary",
            action="use a work root disjoint from the immutable source stage",
        )
    work.mkdir(parents=True, exist_ok=True)
    stage_receipt = verify_v2_source_stage(lock, stage)
    objective_runtime = _preflight_objective_runtime(
        stage,
        lock,
        objective_python=objective_interpreter,
        work_root=work,
    )
    main_runtime = _main_runtime_versions()
    publication_identity = {
        "compiler_artifact_sha256s": _materializer_artifact_sha256s(),
        "finish_window": finish_window,
        "format": MATERIALIZER_FORMAT,
        "materializer_version": MATERIALIZER_VERSION,
        "max_finish_candidates": max_finish_candidates,
        "recipe_sha256": recipe.recipe_sha256,
        "source_set_lock_sha256": lock.sha256,
        "stage_inventory_sha256": stage_receipt["inventory_sha256"],
    }
    identity = {
        **publication_identity,
        "checkpoint_tokens": checkpoint_tokens,
        "main_runtime": main_runtime,
        "objective_runtime": objective_runtime,
    }
    identity_path = work / "materialization-identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_bytes()) != identity:
            raise V2MaterializationError(
                "materialization_identity_drift",
                "work directory belongs to different sources or compiler options",
                action="use a new work directory; never resume across identity drift",
            )
    else:
        _write_json_atomic(identity_path, identity)
    wikidata = _WikidataIndex(stage, work, lock.sha256)
    try:
        wikidata.ensure()
        for lane in recipe.lanes:
            writer = _LaneWriter(
                work,
                lane,
                recipe.quota_by_lane[lane],
                checkpoint_tokens=checkpoint_tokens,
                verification_required=(
                    lane in recipe.reasoning_lanes or lane == recipe.objective_lane
                ),
            )
            lane_succeeded = False
            try:
                if writer.complete:
                    lane_succeeded = True
                    continue
                if lane == "fineweb_edu":
                    source = _fineweb_source(stage, lock)
                elif lane == "finemath":
                    source = _finemath_source(stage)
                elif lane == "wikidata_graph":
                    _compile_wikidata_graph(
                        writer,
                        wikidata,
                        stage,
                        finish_window=finish_window,
                        max_finish_candidates=max_finish_candidates,
                    )
                    lane_succeeded = True
                    continue
                elif lane == "synthetic_graph":
                    source = _SyntheticGraphSource()
                elif lane == "verified_synthetic_multihop":
                    source = _SyntheticMultihopSource()
                elif lane == "wikidata_path_reasoning":
                    source = _WikidataPathSource(wikidata)
                elif lane == "relational_refinement":
                    source = _RelationalRefinementSource()
                elif lane == "objective_auxiliary":
                    source = _ObjectiveSource(
                        stage,
                        lock,
                        objective_python=objective_interpreter,
                        work_root=work,
                        expected_runtime=objective_runtime,
                    )
                else:
                    raise AssertionError(f"unknown frozen lane: {lane}")
                _compile_exact_lane(
                    writer,
                    source,
                    finish_window=finish_window,
                    max_finish_candidates=max_finish_candidates,
                )
                lane_succeeded = True
            finally:
                writer.close(checkpoint=lane_succeeded)
    finally:
        wikidata.close()
    routing = _route_all_lanes(work, recipe)
    lock_payloads = {
        source_id: (stage / "locks" / filename).read_bytes()
        for source_id, filename in _UPSTREAM_LOCK_FILES.items()
    }
    lock_payloads.update(
        _generated_lock_payloads(
            objective_runtime,
            main_runtime,
            publication_identity,
        )
    )
    required_lock_ids = {
        source_id
        for _lane, source_ids in recipe.required_source_locks
        for source_id in source_ids
    }
    if set(lock_payloads) != required_lock_ids:
        raise V2MaterializationError(
            "source_lock_set_drift",
            f"generated source locks differ: expected={sorted(required_lock_ids)}, "
            f"actual={sorted(lock_payloads)}",
        )
    published_report = _publish_source_root(
        work,
        destination,
        recipe,
        lock_payloads,
        seal=seal,
    )
    report = (
        published_report
        if published_report is not None
        else {
            "ready": False,
            "required_paths": list(expected_production_source_paths(recipe)),
        }
    )
    receipt = {
        "format": MATERIALIZER_FORMAT,
        "identity": identity,
        "objective_interpreter": str(objective_interpreter),
        "objective_runtime": objective_runtime,
        "publication_identity": publication_identity,
        "production_preflight": report,
        "routing": routing,
        "source_root": str(destination),
    }
    _write_json_atomic(work / "materialization-receipt.json", receipt)
    return receipt


def _smoke_lock(source_id: str) -> bytes:
    payload = source_id.encode()
    digest = hashlib.sha256(payload).hexdigest()
    return canonical_json_bytes(
        {
            "artifacts": [
                {
                    "bytes": len(payload),
                    "path": f"smoke/{source_id}.txt",
                    "sha256": digest,
                }
            ],
            "format": _GENERATOR_LOCK_FORMAT,
            "kind": "generator",
            "policy": {"profile": "non-scientific-smoke"},
            "repository": "https://example.invalid/memorysplit-v2-smoke",
            "revision": digest,
            "source_id": source_id,
        }
    )


def _smoke_records() -> dict[str, tuple[LaneRecord, ...]]:
    shared_a = "smoke shared payload A"
    shared_b = "smoke shared payload B"
    composition = (
        CompositionPremise("smoke:compose:a", 0, 1),
        CompositionPremise("smoke:compose:b", 1, 2),
    )
    traversal = (
        GraphTraversalPremise("smoke:shared:a", 0, "Q1", "P1", "Q2"),
        GraphTraversalPremise("smoke:shared:b", 1, "Q2", "P2", "Q3"),
    )
    equality = (
        EqualityPremise("smoke:eq:a", 0, "same"),
        EqualityPremise("smoke:eq:b", 1, "same"),
    )
    graph_records = tuple(
        LaneRecord(
            f"smoke-wikidata:{index}",
            "smoke_wikidata",
            (
                LaneSegment(f"edge {index}: "),
                LaneSegment(
                    shared_a
                    if index == 0
                    else shared_b
                    if index == 1
                    else f"value {index}",
                    "smoke:shared:"
                    f"{'a' if index == 0 else 'b' if index == 1 else index}",
                ),
            ),
        )
        for index in range(10)
    )
    return {
        "fineweb_edu": (
            LaneRecord(
                "smoke-fineweb", "smoke_fineweb", (LaneSegment("fineweb smoke text"),)
            ),
        ),
        "finemath": (
            LaneRecord("smoke-finemath", "smoke_finemath", (LaneSegment("2 + 2 = 4"),)),
        ),
        "wikidata_graph": graph_records,
        "synthetic_graph": (
            LaneRecord(
                "smoke-synthetic",
                "smoke_synthetic",
                (LaneSegment("synthetic "), LaneSegment("payload", "smoke:synthetic")),
            ),
        ),
        "verified_synthetic_multihop": (
            LaneRecord(
                "smoke-compose",
                "smoke_verified",
                (
                    LaneSegment("compose "),
                    LaneSegment("one", "smoke:compose:a"),
                    LaneSegment(" two", "smoke:compose:b"),
                ),
                _solver_bundle(composition),
            ),
        ),
        "wikidata_path_reasoning": (
            LaneRecord(
                "smoke-traversal",
                "smoke_path",
                (
                    LaneSegment("path "),
                    LaneSegment(shared_a, "smoke:shared:a"),
                    LaneSegment(" then "),
                    LaneSegment(shared_b, "smoke:shared:b"),
                ),
                _solver_bundle(traversal),
            ),
        ),
        "relational_refinement": (
            LaneRecord(
                "smoke-equality",
                "smoke_refinement",
                (
                    LaneSegment("compare "),
                    LaneSegment("same", "smoke:eq:a"),
                    LaneSegment(" and "),
                    LaneSegment("same", "smoke:eq:b"),
                ),
                _solver_bundle(equality),
            ),
        ),
        "objective_auxiliary": (
            LaneRecord(
                "smoke-objective",
                "smoke_objective",
                (LaneSegment("Question: 1+1? Answer: 2"),),
                {
                    "answer": "2",
                    "kind": "objective_answer",
                    "reference_answer": "2",
                    "validator": "canonical_exact_match",
                },
            ),
        ),
    }


def materialize_v2_smoke(
    source_root: Path | str,
    work_root: Path | str,
) -> dict[str, Any]:
    """Materialize and seal a tiny non-scientific 45-file contract."""

    destination = Path(os.path.abspath(os.path.expanduser(str(source_root))))
    work = Path(os.path.abspath(os.path.expanduser(str(work_root))))
    if work.exists():
        raise V2MaterializationError(
            "smoke_work_exists",
            f"smoke work path already exists: {work}",
        )
    work.mkdir(parents=True)
    records = _smoke_records()
    tok = get_tok()
    encoded = {
        lane: tuple(_encode_record(tok, record) for record in lane_records)
        for lane, lane_records in records.items()
    }
    quotas = {
        lane: sum(item.token_count for item in lane_records)
        for lane, lane_records in encoded.items()
    }
    lock_mapping = {
        "fineweb_edu": ("smoke_fineweb",),
        "finemath": ("smoke_finemath",),
        "wikidata_graph": ("smoke_wikidata",),
        "synthetic_graph": ("smoke_synthetic",),
        "verified_synthetic_multihop": ("smoke_verified", "smoke_solver"),
        "wikidata_path_reasoning": (
            "smoke_wikidata",
            "smoke_path",
            "smoke_solver",
        ),
        "relational_refinement": ("smoke_refinement", "smoke_solver"),
        "objective_auxiliary": ("smoke_objective",),
    }
    recipe = ProductionRecipe.for_testing(
        quotas,
        update_tokens=1,
        required_source_locks=lock_mapping,
    )
    for lane in FROZEN_LANES:
        writer = _LaneWriter(
            work,
            lane,
            quotas[lane],
            checkpoint_tokens=1 << 20,
            verification_required=(
                lane in recipe.reasoning_lanes or lane == recipe.objective_lane
            ),
        )
        try:
            for index, item in enumerate(encoded[lane], 1):
                writer.add(item, {"record": index})
            writer.mark_complete()
        finally:
            writer.close()
    routing = _route_all_lanes(work, recipe)
    lock_ids = {
        source_id for source_ids in lock_mapping.values() for source_id in source_ids
    }
    lock_payloads = {source_id: _smoke_lock(source_id) for source_id in lock_ids}
    if len(expected_production_source_paths(recipe)) != 45:
        raise AssertionError("smoke recipe no longer exercises the 45-file contract")
    report = _publish_source_root(
        work,
        destination,
        recipe,
        lock_payloads,
        seal=True,
    )
    if report is None:
        raise AssertionError("sealed smoke publication returned no preflight report")
    receipt = {
        "format": MATERIALIZER_FORMAT,
        "profile": "smoke",
        "production_preflight": report,
        "routing": routing,
        "scientific_result": False,
        "source_root": str(destination),
    }
    _write_json_atomic(work / "materialization-receipt.json", receipt)
    return receipt


__all__ = [
    "ARC_CONCEPTARC_MAX_TOKENS",
    "DEFAULT_CHECKPOINT_TOKENS",
    "DEFAULT_FINISH_CANDIDATES",
    "DEFAULT_FINISH_WINDOW",
    "MATERIALIZER_FORMAT",
    "MATERIALIZER_VERSION",
    "REASONING_ACTION_SLOTS",
    "V2MaterializationError",
    "materialization_status",
    "materialize_v2_smoke",
    "materialize_v2_source_root",
]
