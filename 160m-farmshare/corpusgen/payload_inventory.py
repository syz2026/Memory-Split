from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping

from corpusgen.graph_records import GraphRow, stable_fact_id


PayloadScope = Literal["train", "protected_seen", "protected_heldout"]
_SCOPES = frozenset(("train", "protected_seen", "protected_heldout"))


@dataclass(frozen=True, order=True)
class PayloadInventoryEntry:
    scope: PayloadScope
    fact_id: str
    field: str
    text: str
    token_ids: tuple[int, ...]
    expected_occurrences: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError(f"invalid payload inventory scope: {self.scope!r}")
        if not self.fact_id:
            raise ValueError("payload inventory fact_id must be nonempty")
        if not self.field:
            raise ValueError("payload inventory field must be nonempty")
        if not self.text:
            raise ValueError("payload inventory text must be nonempty")
        if not self.token_ids:
            raise ValueError("payload inventory token_ids must be nonempty")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < 1 << 16
            for token_id in self.token_ids
        ):
            raise ValueError("payload inventory token_ids must fit uint16")
        if (
            self.expected_occurrences is not None
            and (
                isinstance(self.expected_occurrences, bool)
                or not isinstance(self.expected_occurrences, int)
                or self.expected_occurrences < 0
            )
        ):
            raise ValueError(
                "payload inventory expected_occurrences must be nonnegative"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "fact_id": self.fact_id,
            "field": self.field,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "expected_occurrences": self.expected_occurrences,
        }

    @classmethod
    def from_dict(cls, value: object) -> PayloadInventoryEntry:
        if not isinstance(value, dict) or set(value) != {
            "scope",
            "fact_id",
            "field",
            "text",
            "token_ids",
            "expected_occurrences",
        }:
            raise ValueError("invalid payload inventory entry")
        token_ids = value["token_ids"]
        if not isinstance(token_ids, list):
            raise ValueError("payload inventory token_ids must be a list")
        return cls(
            scope=value["scope"],
            fact_id=str(value["fact_id"]),
            field=str(value["field"]),
            text=str(value["text"]),
            token_ids=tuple(token_ids),
            expected_occurrences=value["expected_occurrences"],
        )


@dataclass(frozen=True)
class PayloadInventory:
    entries: tuple[PayloadInventoryEntry, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(entry, PayloadInventoryEntry)
            for entry in self.entries
        ):
            raise TypeError("payload inventory entries must be typed")
        identities = {
            (
                entry.scope,
                entry.fact_id,
                entry.field,
                entry.text,
                entry.token_ids,
            )
            for entry in self.entries
        }
        if len(identities) != len(self.entries):
            raise ValueError("payload inventory entries must be unique")

    @classmethod
    def from_rows(
        cls,
        tok,
        rows_by_scope: Mapping[PayloadScope, Iterable[GraphRow]],
    ) -> PayloadInventory:
        entries: set[PayloadInventoryEntry] = set()
        for scope, rows in rows_by_scope.items():
            if scope not in _SCOPES:
                raise ValueError(f"invalid payload inventory scope: {scope!r}")
            for row in rows:
                if not isinstance(row, GraphRow):
                    raise TypeError("payload inventory rows must be GraphRow values")
                fact_id = stable_fact_id(row)
                values = (("target", row.target),) + tuple(
                    (f"qualifier:{key}", value)
                    for key, value in row.qualifiers
                )
                for field, value in values:
                    text = json.dumps(value)
                    token_ids = tuple(tok.encode(text))
                    entries.add(
                        PayloadInventoryEntry(
                            scope=scope,
                            fact_id=fact_id,
                            field=field,
                            text=text,
                            token_ids=token_ids,
                            expected_occurrences=0,
                        )
                    )
        return cls(entries=tuple(sorted(entries)))

    def bind_expected_occurrences(
        self,
        counts: Mapping[
            tuple[str, str, str, tuple[int, ...]],
            int,
        ],
    ) -> PayloadInventory:
        normalized = dict(counts)
        if any(
            not isinstance(key, tuple)
            or len(key) != 4
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in normalized.items()
        ):
            raise ValueError("invalid payload occurrence count mapping")
        train_identities = {
            (entry.fact_id, entry.field, entry.text, entry.token_ids)
            for entry in self.entries
            if entry.scope == "train"
        }
        unknown = set(normalized) - train_identities
        if unknown:
            raise ValueError(
                "frozen schedule contains inventory-unknown payloads"
            )
        return PayloadInventory(
            entries=tuple(
                replace(
                    entry,
                    expected_occurrences=(
                        normalized.get(
                            (
                                entry.fact_id,
                                entry.field,
                                entry.text,
                                entry.token_ids,
                            ),
                            0,
                        )
                        if entry.scope == "train"
                        else 0
                    ),
                )
                for entry in self.entries
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "entries": [
                entry.to_dict() for entry in sorted(self.entries)
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> PayloadInventory:
        if not isinstance(value, dict) or set(value) != {"version", "entries"}:
            raise ValueError("invalid payload inventory")
        if value["version"] != 2:
            raise ValueError("unsupported payload inventory version")
        entries = value["entries"]
        if not isinstance(entries, list):
            raise ValueError("payload inventory entries must be a list")
        inventory = cls(
            entries=tuple(
                PayloadInventoryEntry.from_dict(entry) for entry in entries
            )
        )
        if inventory.to_dict() != value:
            raise ValueError("payload inventory is not canonical")
        return inventory

    @classmethod
    def from_path(cls, path: str | Path) -> PayloadInventory:
        try:
            value = json.loads(Path(path).read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid payload inventory file") from exc
        return cls.from_dict(value)

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def write(self, path: str | Path) -> None:
        Path(path).write_bytes(self.canonical_bytes())
