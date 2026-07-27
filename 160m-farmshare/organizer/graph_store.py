from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from corpusgen.graph_records import GraphAddress, GraphRow


@dataclass(frozen=True)
class StoreStats:
    rows: int
    index_bytes: int
    row_bytes: int
    blob_bytes: int


@runtime_checkable
class GraphStore(Protocol):
    def lookup(self, address: GraphAddress) -> GraphRow | None: ...

    def __len__(self) -> int: ...

    def snapshot_sha256(self) -> str: ...

    def stats(self) -> StoreStats: ...


class AtomicGraphStore:
    def __init__(self, rows: Iterable[GraphRow] = ()) -> None:
        self._rows: dict[GraphAddress, GraphRow] = {}
        self.hits = 0
        self.misses = 0
        for row in rows:
            self.add(row)

    def add(self, row: GraphRow) -> None:
        if row.address in self._rows:
            raise ValueError(f"duplicate graph address: {row.address}")
        self._rows[row.address] = row

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        row = self._rows.get(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def rows(self) -> tuple[GraphRow, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def rows_for_provenance(self, provenance_id: str) -> tuple[GraphRow, ...]:
        if not isinstance(provenance_id, str) or not provenance_id:
            raise ValueError("provenance_id must be a nonempty string")
        return tuple(
            row
            for row in self.rows()
            if row.provenance_id == provenance_id
        )

    def max_entity_id(self) -> int:
        values = [row.source_id for row in self._rows.values()]
        values.extend(
            int(row.target)
            for row in self._rows.values()
            if row.target_kind == "entity"
        )
        if not values:
            raise ValueError("empty graph store has no entity id")
        return max(values)

    def canonical_bytes(self) -> bytes:
        lines = [
            json.dumps(row.as_json(), sort_keys=True, separators=(",", ":"))
            for row in self.rows()
        ]
        return ("\n".join(lines) + ("\n" if lines else "")).encode()

    def snapshot_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def stats(self) -> StoreStats:
        return StoreStats(
            rows=len(self),
            index_bytes=0,
            row_bytes=len(self.canonical_bytes()),
            blob_bytes=0,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.canonical_bytes())

    @classmethod
    def load(cls, path: str | Path) -> "AtomicGraphStore":
        rows = [
            GraphRow.from_json(json.loads(line))
            for line in Path(path).read_text().splitlines()
            if line
        ]
        return cls(rows)

    def __len__(self) -> int:
        return len(self._rows)
