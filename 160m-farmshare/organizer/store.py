"""The knowledge organizer: an exact-match (entity, relation) -> value store.

This is deliberately minimal (the "fast memory" of the fast/slow split).
Queries are the strings the model decodes between <|db_start|> and
<|db_retrieve|>, i.e. "name, relation". Matching is exact after whitespace
and case normalization; there is no fuzzy matching in v1 (synthetic
entities make exact keys knowable).
"""

from __future__ import annotations

import json
from pathlib import Path


def normalize(query: str) -> str:
    return " ".join(query.lower().split())


class Organizer:
    def __init__(self) -> None:
        self._table: dict[str, str] = {}
        self.misses: int = 0
        self.hits: int = 0

    def add(self, name: str, relation: str, value: str) -> None:
        self._table[normalize(f"{name}, {relation}")] = value

    def lookup(self, query: str) -> str | None:
        value = self._table.get(normalize(query))
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for key, value in self._table.items():
                f.write(json.dumps({"key": key, "value": value}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Organizer":
        store = cls()
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                store._table[row["key"]] = row["value"]
        return store

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, query: str) -> bool:
        return normalize(query) in self._table
