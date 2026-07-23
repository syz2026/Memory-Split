"""Immutable, content-addressed input catalogs."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

from .canonical import canonical_json_bytes, sha256_hex

_CATALOG_FIELDS = {
    "flags",
    "lane",
    "ordinal",
    "payload_base64",
    "payload_sha256",
    "record_id",
    "source",
    "source_key",
}


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class CatalogRecord:
    ordinal: int
    record_id: str
    lane: str
    source: str
    source_key: str
    payload: bytes
    flags: tuple[str, ...] = ()
    payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")
        for field_name in ("record_id", "lane", "source", "source_key"):
            _required_text(getattr(self, field_name), field_name)
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("payload must be non-empty bytes")
        if (
            not isinstance(self.flags, tuple)
            or tuple(sorted(set(self.flags))) != self.flags
            or any(not isinstance(flag, str) or not flag for flag in self.flags)
        ):
            raise ValueError("flags must be a sorted tuple of unique strings")
        object.__setattr__(self, "payload_sha256", sha256_hex(self.payload))

    def as_dict(self) -> dict[str, object]:
        return {
            "flags": list(self.flags),
            "lane": self.lane,
            "ordinal": self.ordinal,
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
            "payload_sha256": self.payload_sha256,
            "record_id": self.record_id,
            "source": self.source,
            "source_key": self.source_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> CatalogRecord:
        if not isinstance(value, dict) or set(value) != _CATALOG_FIELDS:
            raise ValueError("catalog record fields do not match the contract")
        try:
            payload = base64.b64decode(value["payload_base64"], validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("catalog payload is not canonical base64") from error
        flags = value["flags"]
        if not isinstance(flags, list):
            raise ValueError("catalog flags must be a list")
        record = cls(
            ordinal=value["ordinal"],
            record_id=value["record_id"],
            lane=value["lane"],
            source=value["source"],
            source_key=value["source_key"],
            payload=payload,
            flags=tuple(flags),
        )
        if value["payload_sha256"] != record.payload_sha256:
            raise ValueError(f"catalog payload digest mismatch: {record.record_id}")
        return record


@dataclass(frozen=True)
class InputCatalog:
    records: tuple[CatalogRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("catalog must contain at least one record")
        if [record.ordinal for record in self.records] != list(range(len(self.records))):
            raise ValueError("catalog ordinals must be contiguous and ordered")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("catalog contains duplicate record ids")
        source_records = [
            (record.source, record.source_key) for record in self.records
        ]
        if len(source_records) != len(set(source_records)):
            raise ValueError("catalog contains duplicate source records")

    def to_bytes(self) -> bytes:
        return b"".join(canonical_json_bytes(record.as_dict()) for record in self.records)

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> InputCatalog:
        if not isinstance(payload, bytes) or not payload or not payload.endswith(b"\n"):
            raise ValueError("catalog must be non-empty newline-terminated bytes")
        records = []
        for line in payload.splitlines(keepends=True):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("catalog contains invalid JSON") from error
            if canonical_json_bytes(value) != line:
                raise ValueError("catalog JSONL is not canonical")
            records.append(CatalogRecord.from_dict(value))
        catalog = cls(tuple(records))
        if catalog.to_bytes() != payload:
            raise ValueError("catalog bytes are not canonical")
        return catalog


def fixture_catalog(record_count: int = 12) -> InputCatalog:
    """Create a deterministic miniature catalog that needs no source archives."""

    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
    ):
        raise ValueError("record_count must be a positive integer")
    lanes = ("natural", "facts", "reasoning")
    records = []
    for ordinal in range(record_count):
        lane = lanes[ordinal % len(lanes)]
        payload = canonical_json_bytes(
            {
                "ordinal": ordinal,
                "text": f"fixture {lane} record {ordinal}",
            }
        )
        records.append(
            CatalogRecord(
                ordinal=ordinal,
                record_id=f"fixture-{ordinal:06d}",
                lane=lane,
                source="fixture",
                source_key=f"row:{ordinal}",
                payload=payload,
                flags=("fixture", f"lane:{lane}"),
            )
        )
    return InputCatalog(tuple(records))
