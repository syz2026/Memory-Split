"""Deterministic rendering and canonical metadata reduction."""

from __future__ import annotations

import json
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol

from .canonical import canonical_json_bytes, sha256_hex
from .catalog import CatalogRecord, InputCatalog

_METADATA_FIELDS = {
    "flags",
    "lane",
    "metadata_sha256",
    "ordinal",
    "record_id",
    "render_sha256",
    "renderer_id",
    "source",
    "source_key",
    "source_sha256",
    "token_length",
}


def token_bytes(token_ids: tuple[int, ...]) -> bytes:
    if not token_ids:
        raise ValueError("rendered records must contain at least one token")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < (1 << 16)
        for token_id in token_ids
    ):
        raise ValueError("rendered token ids must fit uint16")
    return struct.pack(f"<{len(token_ids)}H", *token_ids)


@dataclass(frozen=True)
class RenderedRecord:
    token_ids: tuple[int, ...]
    flags: tuple[str, ...]

    def __post_init__(self) -> None:
        token_bytes(self.token_ids)
        if (
            not isinstance(self.flags, tuple)
            or tuple(sorted(set(self.flags))) != self.flags
            or any(not isinstance(flag, str) or not flag for flag in self.flags)
        ):
            raise ValueError("render flags must be a sorted tuple of unique strings")


class Renderer(Protocol):
    renderer_id: str

    def render(self, record: CatalogRecord) -> RenderedRecord: ...


@dataclass(frozen=True)
class FixtureRenderer:
    """Archive-free renderer used to exercise the complete miniature path."""

    renderer_id: str = field(default="fixture-utf8-u16-v1", init=False)

    def render(self, record: CatalogRecord) -> RenderedRecord:
        try:
            payload = json.loads(record.payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid fixture payload: {record.record_id}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(f"fixture payload lacks text: {record.record_id}")
        encoded = payload["text"].encode("utf-8")
        if not encoded:
            raise ValueError(f"fixture text is empty: {record.record_id}")
        flags = tuple(sorted({*record.flags, f"renderer:{self.renderer_id}"}))
        return RenderedRecord(
            token_ids=tuple(byte + 1 for byte in encoded) + (0,),
            flags=flags,
        )


@dataclass(frozen=True)
class MetadataRecord:
    ordinal: int
    record_id: str
    lane: str
    source: str
    source_key: str
    renderer_id: str
    token_length: int
    flags: tuple[str, ...]
    source_sha256: str
    render_sha256: str
    metadata_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("metadata ordinal must be a non-negative integer")
        for field_name in (
            "record_id",
            "lane",
            "source",
            "source_key",
            "renderer_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"metadata {field_name} must be non-empty")
        if (
            isinstance(self.token_length, bool)
            or not isinstance(self.token_length, int)
            or self.token_length <= 0
        ):
            raise ValueError("metadata token_length must be positive")
        if (
            not isinstance(self.flags, tuple)
            or tuple(sorted(set(self.flags))) != self.flags
            or any(not isinstance(flag, str) or not flag for flag in self.flags)
        ):
            raise ValueError("metadata flags must be sorted and unique")
        for field_name in ("source_sha256", "render_sha256"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"metadata {field_name} must be lowercase SHA-256")
        object.__setattr__(
            self,
            "metadata_sha256",
            sha256_hex(canonical_json_bytes(self._core_dict())),
        )

    def _core_dict(self) -> dict[str, object]:
        return {
            "flags": list(self.flags),
            "lane": self.lane,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "render_sha256": self.render_sha256,
            "renderer_id": self.renderer_id,
            "source": self.source,
            "source_key": self.source_key,
            "source_sha256": self.source_sha256,
            "token_length": self.token_length,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "metadata_sha256": self.metadata_sha256}

    @classmethod
    def from_dict(cls, value: object) -> MetadataRecord:
        if not isinstance(value, dict) or set(value) != _METADATA_FIELDS:
            raise ValueError("metadata record fields do not match the contract")
        flags = value["flags"]
        if not isinstance(flags, list):
            raise ValueError("metadata flags must be a list")
        record = cls(
            ordinal=value["ordinal"],
            record_id=value["record_id"],
            lane=value["lane"],
            source=value["source"],
            source_key=value["source_key"],
            renderer_id=value["renderer_id"],
            token_length=value["token_length"],
            flags=tuple(flags),
            source_sha256=value["source_sha256"],
            render_sha256=value["render_sha256"],
        )
        if value["metadata_sha256"] != record.metadata_sha256:
            raise ValueError(f"metadata digest mismatch: {record.record_id}")
        return record


def _render_one(record: CatalogRecord, renderer: Renderer) -> MetadataRecord:
    rendered = renderer.render(record)
    if not set(record.flags) <= set(rendered.flags):
        raise ValueError(f"renderer dropped catalog flags: {record.record_id}")
    return MetadataRecord(
        ordinal=record.ordinal,
        record_id=record.record_id,
        lane=record.lane,
        source=record.source,
        source_key=record.source_key,
        renderer_id=renderer.renderer_id,
        token_length=len(rendered.token_ids),
        flags=rendered.flags,
        source_sha256=record.payload_sha256,
        render_sha256=sha256_hex(token_bytes(rendered.token_ids)),
    )


def reduce_metadata(
    catalog: InputCatalog,
    candidates: list[MetadataRecord] | tuple[MetadataRecord, ...],
    *,
    expected_renderer_id: str,
) -> tuple[MetadataRecord, ...]:
    """Fail-closed reducer that restores immutable catalog order."""

    if not isinstance(expected_renderer_id, str) or not expected_renderer_id:
        raise ValueError("expected_renderer_id must be a non-empty string")
    expected = {record.record_id: record for record in catalog.records}
    reduced: dict[str, MetadataRecord] = {}
    for metadata in candidates:
        if not isinstance(metadata, MetadataRecord):
            raise TypeError("metadata candidates must be MetadataRecord values")
        if metadata.record_id not in expected:
            raise ValueError(f"unexpected metadata record: {metadata.record_id}")
        if metadata.record_id in reduced:
            raise ValueError(f"duplicate metadata record: {metadata.record_id}")
        if metadata.renderer_id != expected_renderer_id:
            raise ValueError(f"metadata renderer mismatch: {metadata.record_id}")
        source = expected[metadata.record_id]
        if (
            metadata.ordinal != source.ordinal
            or metadata.lane != source.lane
            or metadata.source != source.source
            or metadata.source_key != source.source_key
            or metadata.source_sha256 != source.payload_sha256
            or not set(source.flags) <= set(metadata.flags)
        ):
            raise ValueError(f"metadata disagrees with catalog: {metadata.record_id}")
        reduced[metadata.record_id] = metadata
    missing = [record_id for record_id in expected if record_id not in reduced]
    if missing:
        raise ValueError(f"missing metadata records: {', '.join(missing)}")
    return tuple(reduced[record.record_id] for record in catalog.records)


def render_metadata(
    catalog: InputCatalog,
    renderer: Renderer,
    *,
    workers: int = 1,
) -> tuple[MetadataRecord, ...]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not isinstance(renderer.renderer_id, str) or not renderer.renderer_id:
        raise ValueError("renderer_id must be a non-empty string")
    if workers == 1:
        candidates = [_render_one(record, renderer) for record in catalog.records]
    else:
        candidates = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_render_one, record, renderer)
                for record in catalog.records
            ]
            for future in as_completed(futures):
                candidates.append(future.result())
    return reduce_metadata(
        catalog,
        candidates,
        expected_renderer_id=renderer.renderer_id,
    )


def metadata_to_bytes(records: tuple[MetadataRecord, ...]) -> bytes:
    if not records:
        raise ValueError("metadata stream must not be empty")
    ordinals = [record.ordinal for record in records]
    if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
        raise ValueError("metadata records must be uniquely ordered by ordinal")
    return b"".join(canonical_json_bytes(record.as_dict()) for record in records)


def metadata_from_bytes(payload: bytes) -> tuple[MetadataRecord, ...]:
    if not isinstance(payload, bytes) or not payload or not payload.endswith(b"\n"):
        raise ValueError("metadata must be non-empty newline-terminated bytes")
    records = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("metadata contains invalid JSON") from error
        if canonical_json_bytes(value) != line:
            raise ValueError("metadata JSONL is not canonical")
        records.append(MetadataRecord.from_dict(value))
    result = tuple(records)
    if metadata_to_bytes(result) != payload:
        raise ValueError("metadata bytes are not canonical")
    return result
