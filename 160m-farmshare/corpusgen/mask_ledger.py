from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from corpusgen.graph_records import relative_position_bin
from corpusgen.payload_inventory import PayloadInventory


WEIGHT_CONDITIONS = ("dense", "split", "random", "selective")
PROTECTED_SCHEMA_ROLES = frozenset(
    {
        "relation_alias",
        "payload",
        "rule",
        "action",
        "provisional_answer",
        "final_answer",
    }
)


class LeakageError(ValueError):
    """Raised when a target-weight or direct-payload invariant is violated."""


class RandomMaskUndersupplyError(ValueError):
    """Raised when Random-mask lacks an exact-key candidate."""


class _Span(Protocol):
    start: int
    end: int
    role: str
    fact_id: str | None
    fact_cost: Any
    payload_field: str | None
    payload_text: str | None


class _RoutePolicy(Protocol):
    def is_external(self, fact: Any) -> bool: ...


def should_mask(condition: str, span: _Span, policy: _RoutePolicy) -> bool:
    if span.role != "payload":
        return False
    if condition == "split":
        return True
    if condition == "selective":
        return (
            span.fact_cost is not None
            and policy.is_external(span.fact_cost)
        )
    if condition in {"dense", "random"}:
        return False
    raise ValueError(f"unknown target-weight condition: {condition}")


def _validated_length(spans: Sequence[_Span]) -> int:
    previous_end = 0
    length = 0
    for span in sorted(spans, key=lambda value: (value.start, value.end)):
        if (
            isinstance(span.start, bool)
            or isinstance(span.end, bool)
            or not isinstance(span.start, int)
            or not isinstance(span.end, int)
            or span.start < 0
            or span.end <= span.start
        ):
            raise ValueError("encoded spans must be nonempty integer ranges")
        if span.start < previous_end:
            raise ValueError("encoded spans must not overlap")
        if span.role == "payload" and not span.fact_id:
            raise ValueError("payload spans require a fact ID")
        previous_end = span.end
        length = max(length, span.end)
    return length


def _key(span: _Span, document_length: int) -> tuple[int, int]:
    return (
        span.end - span.start,
        relative_position_bin(span.start, span.end, document_length),
    )


def derive_weight_sidecars(
    spans: Sequence[_Span],
    policy: _RoutePolicy,
    seed: int,
) -> dict[str, np.ndarray]:
    """Derive all four exact sidecars for one encoded document."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("mask seed must be a nonnegative integer")
    length = _validated_length(spans)
    outputs = {
        condition: np.ones(length, dtype=np.uint8)
        for condition in WEIGHT_CONDITIONS
    }
    demands: Counter[tuple[int, int]] = Counter()
    candidates: dict[tuple[int, int], list[_Span]] = {}
    for span in spans:
        if should_mask("split", span, policy):
            outputs["split"][span.start : span.end] = 0
            demands[_key(span, length)] += 1
        if should_mask("selective", span, policy):
            outputs["selective"][span.start : span.end] = 0
        if span.role == "random_control":
            candidates.setdefault(_key(span, length), []).append(span)

    available = {
        key: sorted(values, key=lambda span: (span.start, span.end))
        for key, values in candidates.items()
    }
    for key, count in sorted(demands.items()):
        supplied = len(available.get(key, ()))
        if supplied < count:
            raise RandomMaskUndersupplyError(
                f"Random-mask exact key {key} requires {count} candidates, "
                f"found {supplied}"
            )

    rng = random.Random(seed)
    for key, count in sorted(demands.items()):
        selected = rng.sample(available[key], count)
        for span in selected:
            outputs["random"][span.start : span.end] = 0
    return outputs


@dataclass(frozen=True)
class ZeroRunSummary:
    count: int
    masked_tokens: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "masked_tokens": self.masked_tokens,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MaskAudit:
    masked_tokens: dict[str, int]
    histograms: dict[str, Counter[tuple[int, int]]]
    pending_random_demands: int
    zero_runs: dict[str, ZeroRunSummary]
    dense_all_ones: bool
    protected_roles_unmasked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "masked_tokens": dict(sorted(self.masked_tokens.items())),
            "histograms": {
                condition: {
                    f"{length}:{position_bin}": count
                    for (length, position_bin), count in sorted(
                        histogram.items()
                    )
                }
                for condition, histogram in sorted(self.histograms.items())
            },
            "pending_random_demands": self.pending_random_demands,
            "zero_runs": {
                condition: summary.to_dict()
                for condition, summary in sorted(self.zero_runs.items())
            },
            "dense_all_ones": self.dense_all_ones,
            "protected_roles_unmasked": self.protected_roles_unmasked,
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def write(self, path: str | Path) -> None:
        Path(path).write_bytes(self.canonical_bytes())


def _create_spool(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE spans (
          span_id INTEGER PRIMARY KEY,
          record_index INTEGER NOT NULL,
          component TEXT NOT NULL,
          start INTEGER NOT NULL,
          end INTEGER NOT NULL,
          length INTEGER NOT NULL,
          position_bin INTEGER NOT NULL,
          role TEXT NOT NULL,
          fact_id TEXT,
          payload_field TEXT,
          payload_text TEXT,
          selective INTEGER NOT NULL CHECK (selective IN (0, 1)),
          token_bytes BLOB
        );
        CREATE TABLE demands (
          span_id INTEGER PRIMARY KEY REFERENCES spans(span_id),
          length INTEGER NOT NULL,
          position_bin INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE candidates (
          span_id INTEGER PRIMARY KEY REFERENCES spans(span_id),
          start INTEGER NOT NULL,
          end INTEGER NOT NULL,
          length INTEGER NOT NULL,
          position_bin INTEGER NOT NULL,
          selected INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0, 1))
        ) WITHOUT ROWID;
        CREATE INDEX demands_key ON demands(length, position_bin);
        CREATE INDEX candidates_key
          ON candidates(length, position_bin, start, end);
        CREATE INDEX spans_range ON spans(start, end);
        """
    )
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        (("schema_version", "2"), ("finalized", "0")),
    )


class OccurrenceSpool:
    """Disk-backed exact Random demand/candidate and occurrence ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"occurrence spool already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute("PRAGMA synchronous = FULL")
        _create_spool(self._connection)
        self._connection.commit()
        self._next_start = 0
        self._next_record = 0
        self._next_span_id = 1
        self._closed = False

    def add_record(
        self,
        *,
        component: str,
        record_index: int,
        global_start: int,
        token_ids: np.ndarray,
        spans: Sequence[_Span],
        policy: _RoutePolicy,
    ) -> None:
        if self._closed:
            raise ValueError("occurrence spool is closed")
        if not isinstance(component, str) or not component:
            raise ValueError("occurrence component must be nonempty")
        if record_index != self._next_record:
            raise ValueError("occurrence records must be contiguous")
        if global_start != self._next_start:
            raise ValueError("occurrence token ranges must be contiguous")
        if token_ids.dtype != np.uint16 or token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional uint16 array")
        document_length = _validated_length(spans)
        if document_length != len(token_ids):
            raise ValueError("encoded spans must cover the full record length")

        rows = []
        demands = []
        candidates = []
        for ordinal, span in enumerate(
            sorted(spans, key=lambda value: (value.start, value.end))
        ):
            span_id = self._next_span_id + ordinal
            start = global_start + span.start
            end = global_start + span.end
            length, position_bin = _key(span, document_length)
            selective = int(should_mask("selective", span, policy))
            token_bytes = (
                token_ids[span.start : span.end].tobytes()
                if span.role == "payload"
                else None
            )
            rows.append(
                (
                    span_id,
                    record_index,
                    component,
                    start,
                    end,
                    length,
                    position_bin,
                    span.role,
                    span.fact_id,
                    span.payload_field,
                    span.payload_text,
                    selective,
                    token_bytes,
                )
            )
            if span.role == "payload":
                demands.append((span_id, length, position_bin))
            elif span.role == "random_control":
                candidates.append(
                    (span_id, start, end, length, position_bin)
                )

        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO spans(
                  span_id, record_index, component, start, end, length,
                  position_bin, role, fact_id, payload_field, payload_text,
                  selective, token_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._connection.executemany(
                """
                INSERT INTO demands(span_id, length, position_bin)
                VALUES (?, ?, ?)
                """,
                demands,
            )
            self._connection.executemany(
                """
                INSERT INTO candidates(
                  span_id, start, end, length, position_bin
                ) VALUES (?, ?, ?, ?, ?)
                """,
                candidates,
            )
        self._next_start += len(token_ids)
        self._next_record += 1
        self._next_span_id += len(rows)

    def _key_counts(self, table: str) -> dict[tuple[int, int], int]:
        if table not in {"demands", "candidates"}:
            raise ValueError("invalid occurrence count table")
        return {
            (length, position_bin): count
            for length, position_bin, count in self._connection.execute(
                f"""
                SELECT length, position_bin, COUNT(*)
                FROM {table}
                GROUP BY length, position_bin
                ORDER BY length, position_bin
                """
            )
        }

    def random_deficits(self) -> dict[tuple[int, int], int]:
        """Return exact-key candidate shortfalls before finalization."""

        if self._closed:
            raise ValueError("occurrence spool is closed")
        demand_counts = self._key_counts("demands")
        candidate_counts = self._key_counts("candidates")
        return {
            key: count - candidate_counts.get(key, 0)
            for key, count in sorted(demand_counts.items())
            if candidate_counts.get(key, 0) < count
        }

    def finalize_random(self, random_sidecar: str | Path, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("mask seed must be a nonnegative integer")
        finalized = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'finalized'"
        ).fetchone()[0]
        if finalized != "0":
            raise ValueError("Random-mask matching is already finalized")
        demand_counts = self._key_counts("demands")
        candidate_counts = self._key_counts("candidates")
        for key, count in sorted(demand_counts.items()):
            supplied = candidate_counts.get(key, 0)
            if supplied < count:
                raise RandomMaskUndersupplyError(
                    f"Random-mask exact key {key} requires {count} "
                    f"candidates, found {supplied}"
                )

        sidecar_path = Path(random_sidecar)
        if sidecar_path.stat().st_size != self._next_start:
            raise ValueError("Random sidecar does not align with occurrence spool")
        rank_prefix = f"{seed}:".encode()

        def selection_rank(span_id: int) -> int:
            digest = hashlib.blake2b(
                rank_prefix + str(span_id).encode(),
                digest_size=8,
                person=b"maskpick",
            ).digest()
            return int.from_bytes(digest, "little") & ((1 << 63) - 1)

        self._connection.create_function(
            "mask_selection_rank",
            1,
            selection_rank,
            deterministic=True,
        )
        with sidecar_path.open("r+b", buffering=0) as stream:
            with self._connection:
                for (length, position_bin), count in sorted(
                    demand_counts.items()
                ):
                    self._connection.execute(
                        """
                        UPDATE candidates
                        SET selected = 1
                        WHERE span_id IN (
                          SELECT span_id
                          FROM candidates
                          WHERE length = ? AND position_bin = ?
                          ORDER BY mask_selection_rank(span_id), span_id
                          LIMIT ?
                        )
                        """,
                        (length, position_bin, count),
                    )
                    selected = 0
                    for start, end in self._connection.execute(
                        """
                        SELECT start, end
                        FROM candidates
                        WHERE length = ? AND position_bin = ? AND selected = 1
                        ORDER BY start, end, span_id
                        """,
                        (length, position_bin),
                    ):
                        stream.seek(start)
                        previous = stream.read(end - start)
                        if previous != bytes([1]) * (end - start):
                            raise LeakageError(
                                "Random sidecar was not all ones before "
                                "finalization"
                            )
                        stream.seek(start)
                        stream.write(bytes(end - start))
                        selected += 1
                    if selected != count:
                        raise LeakageError(
                            f"Random-mask exact key "
                            f"{(length, position_bin)} selected "
                            f"{selected} candidates, expected {count}"
                        )
                stream.flush()
                os.fsync(stream.fileno())
                self._connection.execute(
                    "UPDATE metadata SET value = '1' "
                    "WHERE key = 'finalized'"
                )

    def export_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        with destination.open("w", encoding="utf-8") as stream:
            demand_rows = self._connection.execute(
                """
                SELECT
                  spans.component, spans.record_index, spans.start, spans.end,
                  spans.length, spans.position_bin, spans.role, spans.fact_id,
                  spans.payload_field, spans.payload_text, spans.selective
                FROM demands
                JOIN spans USING(span_id)
                ORDER BY spans.start, spans.end, spans.span_id
                """
            )
            for row in demand_rows:
                (
                    component,
                    record_index,
                    start,
                    end,
                    length,
                    position_bin,
                    role,
                    fact_id,
                    payload_field,
                    payload_text,
                    selective,
                ) = row
                base = {
                    "component": component,
                    "record_index": record_index,
                    "start": start,
                    "end": end,
                    "length": length,
                    "position_bin": position_bin,
                    "role": role,
                    "fact_id": fact_id,
                    "payload_field": payload_field,
                    "payload_text": payload_text,
                }
                for condition in ("expected_split", "split"):
                    stream.write(
                        json.dumps(
                            {**base, "condition": condition},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                if selective:
                    stream.write(
                        json.dumps(
                            {**base, "condition": "selective"},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            random_rows = self._connection.execute(
                """
                SELECT
                  spans.component, spans.record_index, spans.start, spans.end,
                  spans.length, spans.position_bin, spans.role
                FROM candidates
                JOIN spans USING(span_id)
                WHERE candidates.selected = 1
                ORDER BY spans.start, spans.end, spans.span_id
                """
            )
            for (
                component,
                record_index,
                start,
                end,
                length,
                position_bin,
                role,
            ) in random_rows:
                stream.write(
                    json.dumps(
                        {
                            "component": component,
                            "condition": "random",
                            "record_index": record_index,
                            "start": start,
                            "end": end,
                            "length": length,
                            "position_bin": position_bin,
                            "role": role,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    def close(self) -> None:
        if self._closed:
            return
        self._connection.commit()
        self._connection.close()
        self._closed = True

    def __enter__(self) -> OccurrenceSpool:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _iter_weight_zero_runs(path: Path, expected_size: int):
    if path.stat().st_size != expected_size:
        raise LeakageError(f"sidecar length mismatch: {path.name}")
    offset = 0
    run_start: int | None = None
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            for value in chunk:
                if value not in (0, 1):
                    raise LeakageError(
                        f"sidecar contains a nonbinary weight: {path.name}"
                    )
                if value == 0 and run_start is None:
                    run_start = offset
                elif value == 1 and run_start is not None:
                    yield run_start, offset
                    run_start = None
                offset += 1
    if run_start is not None:
        yield run_start, offset


def _iter_merged_ranges(ranges):
    current: tuple[int, int] | None = None
    for start, end in ranges:
        if start < 0 or end <= start:
            raise LeakageError("occurrence spool contains an invalid range")
        if current is not None and start < current[1]:
            raise LeakageError("occurrence spool contains overlapping ranges")
        if current is not None and start == current[1]:
            current = (current[0], end)
        else:
            if current is not None:
                yield current
            current = (start, end)
    if current is not None:
        yield current


def _validate_tagged_payload_bytes(
    tokens: np.memmap,
    rows,
) -> None:
    found = False
    for start, end, raw in rows:
        found = True
        expected = bytes(raw)
        actual = tokens[start:end].tobytes()
        if actual != expected:
            raise LeakageError("tagged payload tokens differ from occurrence spool")
        if not expected:
            raise LeakageError("payload occurrences must contain tokens")
    if not found:
        raise LeakageError("occurrence spool contains no payloads")


def _iter_pattern_matches(
    tokens: np.memmap,
    patterns: Sequence[tuple[int, ...]],
    record_schedule: Path | None = None,
):
    transitions: list[dict[int, int]] = [{}]
    failures = [0]
    outputs: list[list[int]] = [[]]
    for pattern_id, pattern in enumerate(patterns):
        state = 0
        for token in pattern:
            next_state = transitions[state].get(token)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][token] = next_state
                transitions.append({})
                failures.append(0)
                outputs.append([])
            state = next_state
        outputs[state].append(pattern_id)

    pending = deque()
    for state in transitions[0].values():
        pending.append(state)
    while pending:
        state = pending.popleft()
        for token, next_state in transitions[state].items():
            pending.append(next_state)
            failure = failures[state]
            while failure and token not in transitions[failure]:
                failure = failures[failure]
            failures[next_state] = transitions[failure].get(token, 0)
            outputs[next_state].extend(outputs[failures[next_state]])

    chunk_size = 1 << 20
    ranges = (
        ((0, len(tokens)),)
        if record_schedule is None
        else _iter_record_ranges(record_schedule, len(tokens))
    )
    for record_start, record_end in ranges:
        state = 0
        for chunk_start in range(record_start, record_end, chunk_size):
            chunk_end = min(record_end, chunk_start + chunk_size)
            chunk = tokens[chunk_start:chunk_end]
            for local_index, raw_token in enumerate(chunk):
                token = int(raw_token)
                while state and token not in transitions[state]:
                    state = failures[state]
                state = transitions[state].get(token, 0)
                for pattern_id in outputs[state]:
                    end = chunk_start + local_index + 1
                    yield end - len(patterns[pattern_id]), end


def _iter_record_ranges(path: Path, token_count: int):
    previous_end = 0
    rows = 0
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise LeakageError("frozen record schedule is missing") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LeakageError(
                    f"invalid frozen schedule row {line_number}"
                ) from exc
            start = row.get("token_start") if isinstance(row, dict) else None
            end = row.get("token_end") if isinstance(row, dict) else None
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start != previous_end
                or end <= start
                or end > token_count
            ):
                raise LeakageError(
                    "frozen schedule token ranges are not contiguous"
                )
            yield start, end
            previous_end = end
            rows += 1
    if not rows or previous_end != token_count:
        raise LeakageError("frozen schedule does not cover train.bin")


def _raw_pattern(raw: bytes) -> tuple[int, ...]:
    if len(raw) % np.dtype(np.uint16).itemsize:
        raise LeakageError("payload inventory pattern is not uint16-aligned")
    pattern = tuple(
        int(value) for value in np.frombuffer(raw, dtype=np.uint16)
    )
    if not pattern:
        raise LeakageError("payload patterns must be nonempty")
    return pattern


def _spooled_pattern_facts(
    connection: sqlite3.Connection,
) -> dict[tuple[int, ...], set[str]]:
    patterns: dict[tuple[int, ...], set[str]] = {}
    for raw, fact_id in connection.execute(
        """
        SELECT token_bytes, fact_id
        FROM spans
        WHERE role = 'payload'
        GROUP BY token_bytes, fact_id
        ORDER BY token_bytes, fact_id
        """
    ):
        patterns.setdefault(_raw_pattern(bytes(raw)), set()).add(str(fact_id))
    return patterns


def _inventory_pattern_facts(
    inventory: PayloadInventory,
) -> tuple[dict[tuple[int, ...], set[str]], tuple[tuple[int, ...], ...]]:
    train: dict[tuple[int, ...], set[str]] = {}
    protected: set[tuple[int, ...]] = set()
    for entry in inventory.entries:
        if entry.scope == "train":
            train.setdefault(entry.token_ids, set()).add(entry.fact_id)
        else:
            protected.add(entry.token_ids)
    return train, tuple(sorted(protected))


def _require_spooled_payloads_in_inventory(
    connection: sqlite3.Connection,
    inventory: PayloadInventory,
) -> None:
    inventory_keys = {
        (entry.token_ids, entry.fact_id, entry.field, entry.text)
        for entry in inventory.entries
        if entry.scope == "train"
    }
    for raw, fact_id, field, text in connection.execute(
        """
        SELECT token_bytes, fact_id, payload_field, payload_text
        FROM spans
        WHERE role = 'payload'
        GROUP BY token_bytes, fact_id, payload_field, payload_text
        ORDER BY token_bytes, fact_id, payload_field, payload_text
        """
    ):
        key = (_raw_pattern(bytes(raw)), str(fact_id), field, text)
        if key not in inventory_keys:
            raise LeakageError(
                "inventory-unknown tagged payload occurrence "
                f"for fact {fact_id!r}"
            )


def _require_expected_payload_counts(
    connection: sqlite3.Connection,
    inventory: PayloadInventory,
) -> None:
    actual = {
        (_raw_pattern(bytes(raw)), str(fact_id), field, text): int(count)
        for raw, fact_id, field, text, count in connection.execute(
            """
            SELECT
              token_bytes, fact_id, payload_field, payload_text, COUNT(*)
            FROM spans
            WHERE role = 'payload'
            GROUP BY token_bytes, fact_id, payload_field, payload_text
            ORDER BY token_bytes, fact_id, payload_field, payload_text
            """
        )
    }
    expected: Counter[
        tuple[tuple[int, ...], str, str, str]
    ] = Counter()
    for entry in inventory.entries:
        if entry.scope != "train" or entry.expected_occurrences is None:
            continue
        expected[
            (entry.token_ids, entry.fact_id, entry.field, entry.text)
        ] += entry.expected_occurrences
    for key, count in sorted(expected.items()):
        observed = actual.get(key, 0)
        if observed != count:
            raise LeakageError(
                "payload occurrence count mismatch for "
                f"fact {key[1]!r} field {key[2]!r}: "
                f"expected {count}, observed {observed}"
            )


def _require_inventory_matches_tagged(
    tokens: np.memmap,
    connection: sqlite3.Connection,
    pattern_facts: Mapping[tuple[int, ...], set[str]],
    record_schedule: Path | None,
) -> None:
    patterns = tuple(sorted(pattern_facts))
    if not patterns:
        return
    ranges = iter(
        connection.execute(
            """
            SELECT start, end, fact_id
            FROM spans
            WHERE role = 'payload'
            ORDER BY start, end, span_id
            """
        )
    )
    current = next(ranges, None)
    for start, end in _iter_pattern_matches(
        tokens,
        patterns,
        record_schedule,
    ):
        while current is not None and current[1] <= start:
            current = next(ranges, None)
        if current is None or current[0] > start or end > current[1]:
            raise LeakageError(
                f"untagged direct payload occurrence at [{start}, {end})"
            )
        pattern = tuple(int(value) for value in tokens[start:end])
        if str(current[2]) not in pattern_facts[pattern]:
            raise LeakageError(
                f"mislabeled direct payload occurrence at [{start}, {end})"
            )


def _require_no_protected_patterns(
    tokens: np.memmap,
    patterns: Sequence[tuple[int, ...]],
    record_schedule: Path | None,
) -> None:
    if not patterns:
        return
    for start, end in _iter_pattern_matches(
        tokens,
        patterns,
        record_schedule,
    ):
        raise LeakageError(
            f"protected payload occurrence in train.bin at [{start}, {end})"
        )


def _ranges_for_condition(
    connection: sqlite3.Connection,
    condition: str,
) -> Any:
    if condition == "dense":
        return iter(())
    if condition == "split":
        query = """
            SELECT spans.start, spans.end
            FROM demands JOIN spans USING(span_id)
            ORDER BY spans.start, spans.end
        """
        parameters: tuple[object, ...] = ()
    elif condition == "selective":
        query = """
            SELECT spans.start, spans.end
            FROM demands JOIN spans USING(span_id)
            WHERE spans.selective = 1
            ORDER BY spans.start, spans.end
        """
        parameters = ()
    elif condition == "random":
        query = """
            SELECT start, end
            FROM candidates
            WHERE selected = 1
            ORDER BY start, end
        """
        parameters = ()
    else:
        raise ValueError(f"unknown target-weight condition: {condition}")
    return connection.execute(query, parameters)


def _compare_zero_runs(
    *,
    condition: str,
    actual,
    expected,
) -> ZeroRunSummary:
    digest = hashlib.sha256()
    count = 0
    masked_tokens = 0
    sentinel = object()
    for actual_run, expected_run in zip_longest(
        actual,
        expected,
        fillvalue=sentinel,
    ):
        if actual_run is sentinel or expected_run is sentinel:
            raise LeakageError(
                f"{condition} zero runs do not match occurrence spool"
            )
        if actual_run != expected_run:
            raise LeakageError(
                f"{condition} zero runs do not match occurrence spool"
            )
        start, end = actual_run
        digest.update(f"{start}:{end}\n".encode())
        count += 1
        masked_tokens += end - start
    return ZeroRunSummary(
        count=count,
        masked_tokens=masked_tokens,
        sha256=digest.hexdigest(),
    )


def _histogram_for_condition(
    connection: sqlite3.Connection,
    condition: str,
) -> Counter[tuple[int, int]]:
    if condition == "split":
        query = """
            SELECT length, position_bin, COUNT(*)
            FROM demands
            GROUP BY length, position_bin
        """
    elif condition == "selective":
        query = """
            SELECT demands.length, demands.position_bin, COUNT(*)
            FROM demands JOIN spans USING(span_id)
            WHERE spans.selective = 1
            GROUP BY demands.length, demands.position_bin
        """
    elif condition == "random":
        query = """
            SELECT length, position_bin, COUNT(*)
            FROM candidates
            WHERE selected = 1
            GROUP BY length, position_bin
        """
    elif condition == "dense":
        return Counter()
    else:
        raise ValueError(f"unknown target-weight condition: {condition}")
    return Counter(
        {
            (length, position_bin): count
            for length, position_bin, count in connection.execute(query)
        }
    )


def verify_weight_sidecars(
    train_bin: str | Path,
    sidecars: Mapping[str, str | Path],
    occurrence_spool: str | Path,
    *,
    payload_inventory: str | Path | None = None,
    record_schedule: str | Path | None = None,
) -> MaskAudit:
    """Independently reopen and verify token, sidecar, and occurrence artifacts."""

    if set(sidecars) != set(WEIGHT_CONDITIONS):
        raise ValueError("all four target-weight sidecars are required")
    train_path = Path(train_bin)
    if train_path.stat().st_size % np.dtype(np.uint16).itemsize:
        raise LeakageError("train.bin is not aligned to uint16 tokens")
    token_count = train_path.stat().st_size // np.dtype(np.uint16).itemsize
    tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    connection = _read_only_connection(Path(occurrence_spool))
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata != {"schema_version": "2", "finalized": "1"}:
            raise LeakageError("occurrence spool was not finalized")

        zero_runs: dict[str, ZeroRunSummary] = {}
        masked_tokens: dict[str, int] = {}
        histograms: dict[str, Counter[tuple[int, int]]] = {}
        for condition in WEIGHT_CONDITIONS:
            summary = _compare_zero_runs(
                condition=condition,
                actual=_iter_weight_zero_runs(
                    Path(sidecars[condition]),
                    token_count,
                ),
                expected=_iter_merged_ranges(
                    _ranges_for_condition(connection, condition)
                ),
            )
            zero_runs[condition] = summary
            masked_tokens[condition] = summary.masked_tokens
            histograms[condition] = _histogram_for_condition(
                connection,
                condition,
            )

        if histograms["random"] != histograms["split"]:
            raise LeakageError(
                "Random-mask histogram does not exactly match Split"
            )
        if masked_tokens["random"] != masked_tokens["split"]:
            raise LeakageError(
                "Random-mask token mass does not exactly match Split"
            )
        key_counts = {
            (length, position_bin): count
            for length, position_bin, count in connection.execute(
                """
                SELECT length, position_bin, COUNT(*)
                FROM demands
                GROUP BY length, position_bin
                """
            )
        }
        selected_counts = {
            (length, position_bin): count
            for length, position_bin, count in connection.execute(
                """
                SELECT length, position_bin, COUNT(*)
                FROM candidates
                WHERE selected = 1
                GROUP BY length, position_bin
                """
            )
        }
        pending = sum(
            max(count - selected_counts.get(key, 0), 0)
            for key, count in key_counts.items()
        )
        if pending or selected_counts != key_counts:
            raise LeakageError("Random-mask demands remain unmatched")

        random_weights = np.memmap(
            Path(sidecars["random"]),
            dtype=np.uint8,
            mode="r",
        )
        for start, end, role in connection.execute(
            """
            SELECT start, end, role
            FROM spans
            WHERE role IN (
              'relation_alias', 'payload', 'rule', 'action',
              'provisional_answer', 'final_answer'
            )
            ORDER BY start, end
            """
        ):
            if not random_weights[start:end].all():
                raise LeakageError(
                    f"Random-mask intersects protected role {role}"
                )

        _validate_tagged_payload_bytes(
            tokens,
            connection.execute(
                """
                SELECT start, end, token_bytes
                FROM spans
                WHERE role = 'payload'
                ORDER BY start, end, span_id
                """
            ),
        )
        if payload_inventory is None:
            train_patterns = _spooled_pattern_facts(connection)
            protected_patterns: tuple[tuple[int, ...], ...] = ()
        else:
            inventory = PayloadInventory.from_path(payload_inventory)
            train_patterns, protected_patterns = _inventory_pattern_facts(
                inventory
            )
            _require_spooled_payloads_in_inventory(
                connection,
                inventory,
            )
            _require_expected_payload_counts(connection, inventory)
        _require_inventory_matches_tagged(
            tokens,
            connection,
            train_patterns,
            None if record_schedule is None else Path(record_schedule),
        )
        _require_no_protected_patterns(
            tokens,
            protected_patterns,
            None if record_schedule is None else Path(record_schedule),
        )
        return MaskAudit(
            masked_tokens=masked_tokens,
            histograms=histograms,
            pending_random_demands=pending,
            zero_runs=zero_runs,
            dense_all_ones=zero_runs["dense"].count == 0,
            protected_roles_unmasked=True,
        )
    finally:
        connection.close()
