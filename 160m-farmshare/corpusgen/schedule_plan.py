from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from corpusgen.graph_records import relative_position_bin
from corpusgen.mask_ledger import RandomMaskUndersupplyError


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PlannedRecord:
    record_index: int
    component: str
    token_start: int
    token_ids: np.ndarray
    spans: tuple[dict[str, object], ...]
    schedule: dict[str, object]


class SchedulePlanSpool:
    """Disk-backed immutable record plan replayed by the corpus writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"schedule plan already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.executescript(
            """
            CREATE TABLE metadata (
              key TEXT PRIMARY KEY,
              value INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE records (
              record_index INTEGER PRIMARY KEY,
              component TEXT NOT NULL,
              token_start INTEGER NOT NULL,
              token_end INTEGER NOT NULL,
              token_bytes BLOB NOT NULL,
              spans_json TEXT NOT NULL,
              schedule_json TEXT NOT NULL
            );
            INSERT INTO metadata(key, value) VALUES ('sealed', 0);
            """
        )
        self._connection.commit()
        self.total_tokens = 0
        self.total_records = 0
        self._closed = False
        self._sealed = False

    @staticmethod
    def _span_dict(span: Any) -> dict[str, object]:
        fact_cost = span.fact_cost
        if fact_cost is not None:
            if not is_dataclass(fact_cost):
                raise TypeError("planned fact costs must be dataclasses")
            fact_cost = asdict(fact_cost)
        return {
            "start": span.start,
            "end": span.end,
            "role": span.role,
            "fact_id": span.fact_id,
            "fact_cost": fact_cost,
            "payload_field": span.payload_field,
            "payload_text": span.payload_text,
        }

    @staticmethod
    def _require_local_supply(
        spans: Sequence[Any],
        document_length: int,
        record_index: int,
    ) -> None:
        demands: Counter[tuple[int, int]] = Counter()
        candidates: Counter[tuple[int, int]] = Counter()
        for span in spans:
            key = (
                span.end - span.start,
                relative_position_bin(span.start, span.end, document_length),
            )
            if span.role == "payload":
                demands[key] += 1
            elif span.role == "random_control":
                candidates[key] += 1
        for key, count in sorted(demands.items()):
            supplied = candidates.get(key, 0)
            if supplied < count:
                raise RandomMaskUndersupplyError(
                    f"planned record {record_index} Random-mask exact key {key} "
                    f"requires {count} same-record candidates, found {supplied}"
                )

    def add_record(
        self,
        *,
        component: str,
        token_ids: np.ndarray,
        spans: Sequence[Any],
        schedule: dict[str, object],
    ) -> None:
        if self._closed or self._sealed:
            raise ValueError("cannot add to a closed or sealed schedule plan")
        if token_ids.dtype != np.uint16 or token_ids.ndim != 1:
            raise ValueError("planned token_ids must be one-dimensional uint16")
        if not component:
            raise ValueError("planned component must be nonempty")
        self._require_local_supply(
            spans,
            len(token_ids),
            self.total_records,
        )
        span_rows = tuple(self._span_dict(span) for span in spans)
        token_start = self.total_tokens
        token_end = token_start + len(token_ids)
        expected_schedule = {
            **schedule,
            "component": component,
            "token_start": token_start,
            "token_end": token_end,
        }
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO records(
                  record_index, component, token_start, token_end,
                  token_bytes, spans_json, schedule_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.total_records,
                    component,
                    token_start,
                    token_end,
                    token_ids.tobytes(),
                    _canonical_json(span_rows),
                    _canonical_json(expected_schedule),
                ),
            )
        self.total_tokens = token_end
        self.total_records += 1

    def seal(self) -> None:
        if self._closed:
            raise ValueError("schedule plan is closed")
        if self._sealed:
            return
        with self._connection:
            self._connection.execute(
                "UPDATE metadata SET value = 1 WHERE key = 'sealed'"
            )
        self._sealed = True

    def iter_records(self) -> Iterator[PlannedRecord]:
        if not self._sealed or self._closed:
            raise ValueError("schedule plan must be open and sealed")
        for (
            record_index,
            component,
            token_start,
            token_bytes,
            spans_json,
            schedule_json,
        ) in self._connection.execute(
            """
            SELECT
              record_index, component, token_start, token_bytes,
              spans_json, schedule_json
            FROM records
            ORDER BY record_index
            """
        ):
            token_ids = np.frombuffer(
                bytes(token_bytes),
                dtype=np.uint16,
            ).copy()
            yield PlannedRecord(
                record_index=record_index,
                component=component,
                token_start=token_start,
                token_ids=token_ids,
                spans=tuple(json.loads(spans_json)),
                schedule=json.loads(schedule_json),
            )

    def write_schedule(self, path: str | Path) -> None:
        if not self._sealed or self._closed:
            raise ValueError("schedule plan must be open and sealed")
        with Path(path).open("x", encoding="utf-8") as stream:
            for (raw,) in self._connection.execute(
                "SELECT schedule_json FROM records ORDER BY record_index"
            ):
                stream.write(raw + "\n")

    def payload_occurrence_counts(
        self,
    ) -> Counter[tuple[str, str, str, tuple[int, ...]]]:
        if not self._sealed or self._closed:
            raise ValueError("schedule plan must be open and sealed")
        counts: Counter[
            tuple[str, str, str, tuple[int, ...]]
        ] = Counter()
        for record in self.iter_records():
            for span in record.spans:
                if span["role"] != "payload":
                    continue
                fact_id = span["fact_id"]
                field = span["payload_field"]
                text = span["payload_text"]
                start = span["start"]
                end = span["end"]
                if (
                    not isinstance(fact_id, str)
                    or not fact_id
                    or not isinstance(field, str)
                    or not field
                    or not isinstance(text, str)
                    or not text
                    or isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(record.token_ids)
                ):
                    raise ValueError(
                        "planned payload occurrence metadata is invalid"
                    )
                token_ids = tuple(
                    int(value) for value in record.token_ids[start:end]
                )
                counts[(fact_id, field, text, token_ids)] += 1
        return counts

    def close(self) -> None:
        if self._closed:
            return
        self._connection.commit()
        self._connection.close()
        self._closed = True

    def __enter__(self) -> SchedulePlanSpool:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
