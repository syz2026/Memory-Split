"""Canonical external-memory input catalog for the reasoning-v2 corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Protocol, cast

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
    entry_exists,
    entry_lstat,
    fsync_directory,
    list_entries,
    open_directory_at,
    open_parent_directory,
    open_regular_file_at,
)
from corpusgen.reasoning_v2.contracts import (
    LANE_ORDER,
    BuildGeometry,
    LaneId,
    balanced_record_lengths,
)
from corpusgen.reasoning_v2.source_lock import (
    SourceEntry,
    SourceFile,
    SourceLock,
    verify_source_tree,
)


_CATALOG_FORMAT = "memorysplit-reasoning-v2-input-catalog-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SEALED_PATH_TERMS = ("evaluation", "validation", "test", "sealed", "holdout")
_PATH_LOCATOR_KEYS = frozenset({"file", "member", "path", "source_path", "split"})
_RECORD_FIELDS = {
    "lane_id",
    "ordinal",
    "record_id",
    "semantic_facts",
    "semantic_flags",
    "source_byte_sha256",
    "source_id",
    "source_key",
    "source_locator",
    "target_count",
}
_DRAFT_FIELDS = _RECORD_FIELDS - {"ordinal", "record_id", "target_count"}
_FACT_FIELDS = {
    "expected_hops",
    "expected_reads",
    "fact_id",
    "payload_entropy_bits",
    "record_type",
    "scheduled_exposures",
    "source",
    "surfaces",
}


class LaneQuotaShortfall(ValueError):
    def __init__(self, lane_id: LaneId, requested: int, emitted: int) -> None:
        self.lane_id = lane_id
        self.requested = requested
        self.emitted = emitted
        super().__init__(
            f"{lane_id} lacks distinct deterministic records: "
            f"requested={requested}, emitted={emitted}"
        )


@dataclass(frozen=True)
class SemanticFactRow:
    fact_id: str
    source: str
    record_type: str
    payload_entropy_bits: Fraction
    scheduled_exposures: int
    expected_reads: Fraction
    expected_hops: Fraction
    surfaces: tuple[str, ...]


@dataclass(frozen=True)
class CatalogDraft:
    lane_id: LaneId
    source_id: str
    source_key: str
    source_byte_sha256: str
    source_locator: tuple[tuple[str, str | int], ...]
    semantic_flags: tuple[str, ...]
    semantic_facts: tuple[SemanticFactRow, ...]


@dataclass(frozen=True)
class CatalogRecord:
    ordinal: int
    record_id: str
    lane_id: LaneId
    source_id: str
    source_key: str
    source_byte_sha256: str
    source_locator: tuple[tuple[str, str | int], ...]
    target_count: int
    semantic_flags: tuple[str, ...]
    semantic_facts: tuple[SemanticFactRow, ...]


@dataclass(frozen=True)
class CatalogLaneIndex:
    lane_id: LaneId
    first_ordinal: int
    record_count: int
    target_count: int


@dataclass(frozen=True)
class InputCatalog:
    root: Path
    records_path: Path
    index_path: Path
    source_lock_sha256: str
    sha256: str
    record_count: int
    target_count: int
    lanes: tuple[CatalogLaneIndex, ...]

    @property
    def lane_target_counts(self) -> tuple[tuple[LaneId, int], ...]:
        return tuple((lane.lane_id, lane.target_count) for lane in self.lanes)

    def iter_records(self) -> Iterator[CatalogRecord]:
        return _records_from_jsonl(
            self.records_path,
            expected_sha256=self.sha256,
            expected_record_count=self.record_count,
            expected_target_count=self.target_count,
            expected_lanes=self.lanes,
        )

    def to_bytes(self) -> bytes:
        payload = _read_regular_path(self.index_path, "catalog index")
        value = _strict_json_bytes(payload, "catalog index")
        if canonical_json_bytes(value) != payload:
            raise ValueError("catalog index JSON is not canonical")
        return payload


class LaneCatalogSource(Protocol):
    lane_id: LaneId
    finite: bool

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: tuple[int, ...],
    ) -> Iterator[CatalogDraft]:
        raise RuntimeError("lane source must emit deterministic drafts")


def _byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def _require_nfc_string(
    value: object,
    description: str,
    *,
    nonempty: bool = True,
) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ValueError(f"{description} must be a canonical string")
    result = cast(str, value)
    if "\x00" in result or unicodedata.normalize("NFC", result) != result:
        raise ValueError(f"{description} must be NFC-normalized")
    return result


def _require_fraction(
    value: object,
    description: str,
    *,
    nonnegative: bool = True,
) -> Fraction:
    if type(value) is not Fraction:
        raise ValueError(f"{description} must be a finite Fraction")
    result = cast(Fraction, value)
    if nonnegative and result < 0:
        raise ValueError(f"{description} must be non-negative")
    return result


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


def _fact_dict(fact: SemanticFactRow) -> dict[str, object]:
    return {
        "expected_hops": _fraction_dict(fact.expected_hops),
        "expected_reads": _fraction_dict(fact.expected_reads),
        "fact_id": fact.fact_id,
        "payload_entropy_bits": _fraction_dict(fact.payload_entropy_bits),
        "record_type": fact.record_type,
        "scheduled_exposures": fact.scheduled_exposures,
        "source": fact.source,
        "surfaces": list(fact.surfaces),
    }


def _fact_identity_dict(fact: SemanticFactRow) -> dict[str, object]:
    value = _fact_dict(fact)
    del value["scheduled_exposures"]
    return value


def _draft_dict(draft: CatalogDraft) -> dict[str, object]:
    return {
        "lane_id": draft.lane_id,
        "semantic_facts": [_fact_dict(fact) for fact in draft.semantic_facts],
        "semantic_flags": list(draft.semantic_flags),
        "source_byte_sha256": draft.source_byte_sha256,
        "source_id": draft.source_id,
        "source_key": draft.source_key,
        "source_locator": [list(item) for item in draft.source_locator],
    }


def _record_dict(record: CatalogRecord) -> dict[str, object]:
    return {
        **_draft_dict(
            CatalogDraft(
                lane_id=record.lane_id,
                source_id=record.source_id,
                source_key=record.source_key,
                source_byte_sha256=record.source_byte_sha256,
                source_locator=record.source_locator,
                semantic_flags=record.semantic_flags,
                semantic_facts=record.semantic_facts,
            )
        ),
        "ordinal": record.ordinal,
        "record_id": record.record_id,
        "target_count": record.target_count,
    }


def _lane_index_dict(lane: CatalogLaneIndex) -> dict[str, object]:
    return {
        "first_ordinal": lane.first_ordinal,
        "lane_id": lane.lane_id,
        "record_count": lane.record_count,
        "target_count": lane.target_count,
    }


def _validate_fact(fact: object) -> SemanticFactRow:
    if not isinstance(fact, SemanticFactRow):
        raise ValueError("semantic facts must contain SemanticFactRow values")
    _require_nfc_string(fact.fact_id, "semantic fact ID")
    _require_nfc_string(fact.source, "semantic fact source")
    _require_nfc_string(fact.record_type, "semantic fact record type")
    _require_fraction(fact.payload_entropy_bits, "semantic payload entropy")
    _require_fraction(fact.expected_reads, "semantic expected reads")
    _require_fraction(fact.expected_hops, "semantic expected hops")
    if type(fact.scheduled_exposures) is not int or fact.scheduled_exposures <= 0:
        raise ValueError("semantic scheduled exposures must be a positive integer")
    if type(fact.surfaces) is not tuple or not fact.surfaces:
        raise ValueError("semantic fact surfaces must be a nonempty tuple")
    for surface in fact.surfaces:
        _require_nfc_string(surface, "semantic fact surface")
    if len(fact.surfaces) != len(set(fact.surfaces)):
        raise ValueError(f"duplicate semantic fact surface: {fact.fact_id}")
    if fact.surfaces != tuple(sorted(fact.surfaces, key=_byte_key)):
        raise ValueError("semantic fact surfaces must use canonical bytewise order")
    return fact


def _validate_locator(
    locator: object,
) -> tuple[tuple[str, str | int], ...]:
    if type(locator) is not tuple or not locator:
        raise ValueError("source locator must be a nonempty tuple")
    validated: list[tuple[str, str | int]] = []
    for item in locator:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("source locator entries must be key/value tuples")
        key = _require_nfc_string(item[0], "source locator key")
        raw_value = item[1]
        if type(raw_value) is int:
            if raw_value < 0:
                raise ValueError("source locator integers must be non-negative")
            value: str | int = raw_value
        elif type(raw_value) is str:
            value = _require_nfc_string(raw_value, "source locator value")
        else:
            raise ValueError("source locator values must be strings or integers")
        validated.append((key, value))
    keys = tuple(key for key, _value in validated)
    if len(keys) != len(set(keys)):
        raise ValueError("source locator contains duplicate keys")
    if keys != tuple(sorted(keys, key=_byte_key)):
        raise ValueError("source locator keys must use canonical bytewise order")
    return tuple(validated)


def _validate_draft_structure(draft: object) -> CatalogDraft:
    if not isinstance(draft, CatalogDraft):
        raise ValueError("lane source must emit CatalogDraft values")
    if draft.lane_id not in LANE_ORDER:
        raise ValueError(f"catalog draft has unknown lane: {draft.lane_id!r}")
    _require_nfc_string(draft.source_id, "catalog source ID")
    _require_nfc_string(draft.source_key, "catalog source key")
    if (
        type(draft.source_byte_sha256) is not str
        or _SHA256_RE.fullmatch(draft.source_byte_sha256) is None
    ):
        raise ValueError("source byte hash must be a lowercase SHA-256")
    _validate_locator(draft.source_locator)
    if type(draft.semantic_flags) is not tuple:
        raise ValueError("semantic flags must be a tuple")
    for flag in draft.semantic_flags:
        _require_nfc_string(flag, "semantic flag")
    if len(draft.semantic_flags) != len(set(draft.semantic_flags)):
        raise ValueError("semantic flags contain duplicates")
    if draft.semantic_flags != tuple(sorted(draft.semantic_flags, key=_byte_key)):
        raise ValueError("semantic flags must use canonical bytewise order")
    if type(draft.semantic_facts) is not tuple:
        raise ValueError("semantic facts must be a tuple")
    for fact in draft.semantic_facts:
        _validate_fact(fact)
    fact_ids = tuple(fact.fact_id for fact in draft.semantic_facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("duplicate semantic fact ID within one catalog draft")
    if fact_ids != tuple(sorted(fact_ids, key=_byte_key)):
        raise ValueError("semantic facts must use canonical fact-ID order")
    return draft


def _safe_relative_path(value: str, description: str) -> str:
    if "\\" in value:
        raise ValueError(f"{description} is not a canonical POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{description} is not a safe relative path")
    return value


def _path_marks_training(value: str) -> bool:
    parts = tuple(part.casefold() for part in PurePosixPath(value).parts)
    return any(
        part in {"train", "training"} or part.startswith("train-")
        for part in parts
    )


def _reject_sealed_locators(draft: CatalogDraft) -> None:
    for key, value in draft.source_locator:
        if key not in _PATH_LOCATOR_KEYS or type(value) is not str:
            continue
        lowered = value.casefold()
        if not any(term in lowered for term in _SEALED_PATH_TERMS):
            continue
        if key in {"path", "source_path"} and _path_marks_training(value):
            continue
        raise ValueError(f"sealed or evaluation path is forbidden: {value}")


def _manifest_file(
    draft: CatalogDraft,
    entries: Mapping[str, SourceEntry],
) -> SourceFile:
    _reject_sealed_locators(draft)
    entry = entries.get(draft.source_id)
    if entry is None:
        raise ValueError(f"catalog draft references unknown source: {draft.source_id}")
    locator = dict(draft.source_locator)
    path_value = locator.get("path", locator.get("source_path"))
    if type(path_value) is not str:
        raise ValueError("source locator must contain one string path")
    if "path" in locator and "source_path" in locator:
        raise ValueError("source locator has ambiguous source paths")
    path_text = _safe_relative_path(path_value, "source locator path")
    prefix = f"{entry.materialized_path}/"
    relative = path_text[len(prefix) :] if path_text.startswith(prefix) else path_text
    by_path = {row.path: row for row in entry.files}
    source_file = by_path.get(relative)
    if source_file is None:
        raise ValueError(
            f"source locator path is not in the verified manifest: "
            f"{draft.source_id}:{path_text}"
        )
    if draft.source_byte_sha256 != source_file.sha256:
        raise ValueError(
            f"source byte hash disagreement: {draft.source_id}:{source_file.path}"
        )
    return source_file


def catalog_record_id(draft: CatalogDraft, target_count: int) -> str:
    _validate_draft_structure(draft)
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("catalog target_count must be a positive integer")
    return sha256_hex(
        canonical_json_bytes(
            {
                "draft": _draft_dict(draft),
                "target_count": target_count,
            }
        )
    )


def _validate_geometry(
    geometry: object,
) -> tuple[tuple[LaneId, int], ...]:
    if not isinstance(geometry, BuildGeometry):
        raise TypeError("geometry must be a BuildGeometry")
    if (
        type(geometry.total_targets) is not int
        or geometry.total_targets <= 0
        or type(geometry.targets_per_update) is not int
        or geometry.targets_per_update <= 0
        or type(geometry.context_length) is not int
        or geometry.context_length <= 0
        or type(geometry.shard_count) is not int
        or geometry.shard_count <= 0
        or type(geometry.allow_fewer_shards) is not bool
    ):
        raise ValueError("catalog geometry contains invalid scalar values")
    if type(geometry.lane_quotas) is not tuple:
        raise ValueError("catalog lane quotas must be a tuple")
    quotas: list[tuple[LaneId, int]] = []
    for item in geometry.lane_quotas:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("catalog lane quota entries must be tuples")
        lane_id, quota = item
        if lane_id not in LANE_ORDER or type(quota) is not int or quota <= 0:
            raise ValueError("catalog lane quota entry is invalid")
        quotas.append((cast(LaneId, lane_id), quota))
    frozen = tuple(quotas)
    if tuple(lane for lane, _quota in frozen) != LANE_ORDER:
        raise ValueError("catalog lane quotas must use the frozen lane order")
    if sum(quota for _lane, quota in frozen) != geometry.total_targets:
        raise ValueError("catalog lane quotas do not sum to total targets")
    return frozen


def _validated_lane_sources(
    lane_sources: object,
) -> dict[LaneId, LaneCatalogSource]:
    if not isinstance(lane_sources, Mapping):
        raise TypeError("lane_sources must be a Mapping")
    first_keys = tuple(lane_sources)
    second_keys = tuple(lane_sources)
    if first_keys != second_keys:
        raise ValueError("lane_sources has unstable mapping key order")
    for key in first_keys:
        if type(key) is not str:
            raise ValueError("lane source keys must be exact strings")
    known = set(LANE_ORDER)
    actual = set(first_keys)
    unknown = sorted(actual - known, key=_byte_key)
    missing = sorted(known - actual, key=_byte_key)
    if unknown:
        raise ValueError(f"unknown lane source: {unknown[0]}")
    if missing:
        raise ValueError(f"missing lane source: {missing[0]}")
    if len(first_keys) != len(LANE_ORDER):
        raise ValueError("lane source mapping contains duplicate keys")
    result: dict[LaneId, LaneCatalogSource] = {}
    for lane_id in LANE_ORDER:
        source = lane_sources[lane_id]
        if getattr(source, "lane_id", None) != lane_id:
            raise ValueError(f"lane source identity mismatch: {lane_id}")
        if type(getattr(source, "finite", None)) is not bool:
            raise ValueError(f"lane source finite flag is invalid: {lane_id}")
        if not callable(getattr(source, "iter_drafts", None)):
            raise ValueError(f"lane source iterator is missing: {lane_id}")
        result[lane_id] = cast(LaneCatalogSource, source)
    return result


def _validated_lengths(
    quotas: tuple[tuple[LaneId, int], ...],
    context_length: int,
) -> dict[LaneId, tuple[int, ...]]:
    result: dict[LaneId, tuple[int, ...]] = {}
    for lane_id, quota in quotas:
        lengths = balanced_record_lengths(quota, context_length)
        if type(lengths) is not tuple or not lengths:
            raise ValueError(f"{lane_id} target lengths must be a nonempty tuple")
        if any(
            type(length) is not int or length <= 0 or length > context_length
            for length in lengths
        ):
            raise ValueError(f"{lane_id} target lengths are invalid")
        if sum(lengths) != quota:
            raise ValueError(f"{lane_id} target lengths do not sum to lane quota")
        result[lane_id] = lengths
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _strict_json_bytes(payload: bytes, description: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is invalid JSON") from error


def _fraction_from_dict(value: object, description: str) -> Fraction:
    if (
        type(value) is not dict
        or set(cast(dict[object, object], value)) != {"denominator", "numerator"}
    ):
        raise ValueError(f"{description} is not a canonical fraction")
    row = cast(dict[str, object], value)
    numerator = row["numerator"]
    denominator = row["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise ValueError(f"{description} is not a canonical fraction")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ValueError(f"{description} fraction is not reduced")
    return result


def _fact_from_dict(value: object) -> SemanticFactRow:
    if type(value) is not dict or set(cast(dict[object, object], value)) != _FACT_FIELDS:
        raise ValueError("catalog semantic fact fields do not match")
    row = cast(dict[str, object], value)
    surfaces = row["surfaces"]
    if type(surfaces) is not list:
        raise ValueError("catalog semantic fact surfaces must be a list")
    fact = SemanticFactRow(
        fact_id=cast(str, row["fact_id"]),
        source=cast(str, row["source"]),
        record_type=cast(str, row["record_type"]),
        payload_entropy_bits=_fraction_from_dict(
            row["payload_entropy_bits"],
            "semantic payload entropy",
        ),
        scheduled_exposures=cast(int, row["scheduled_exposures"]),
        expected_reads=_fraction_from_dict(
            row["expected_reads"],
            "semantic expected reads",
        ),
        expected_hops=_fraction_from_dict(
            row["expected_hops"],
            "semantic expected hops",
        ),
        surfaces=tuple(cast(list[str], surfaces)),
    )
    return _validate_fact(fact)


def _draft_from_dict(value: object) -> CatalogDraft:
    if type(value) is not dict or set(cast(dict[object, object], value)) != _DRAFT_FIELDS:
        raise ValueError("catalog draft fields do not match")
    row = cast(dict[str, object], value)
    raw_locator = row["source_locator"]
    raw_flags = row["semantic_flags"]
    raw_facts = row["semantic_facts"]
    if (
        type(raw_locator) is not list
        or type(raw_flags) is not list
        or type(raw_facts) is not list
    ):
        raise ValueError("catalog draft arrays do not match")
    locator = []
    for item in raw_locator:
        if type(item) is not list or len(cast(list[object], item)) != 2:
            raise ValueError("catalog source locator JSON is invalid")
        pair = cast(list[object], item)
        locator.append((cast(str, pair[0]), cast(str | int, pair[1])))
    draft = CatalogDraft(
        lane_id=cast(LaneId, row["lane_id"]),
        source_id=cast(str, row["source_id"]),
        source_key=cast(str, row["source_key"]),
        source_byte_sha256=cast(str, row["source_byte_sha256"]),
        source_locator=tuple(locator),
        semantic_flags=tuple(cast(list[str], raw_flags)),
        semantic_facts=tuple(_fact_from_dict(fact) for fact in raw_facts),
    )
    return _validate_draft_structure(draft)


def _record_from_dict(value: object) -> CatalogRecord:
    if type(value) is not dict or set(cast(dict[object, object], value)) != _RECORD_FIELDS:
        raise ValueError("catalog record fields do not match")
    row = cast(dict[str, object], value)
    draft = _draft_from_dict(
        {key: item for key, item in row.items() if key in _DRAFT_FIELDS}
    )
    ordinal = row["ordinal"]
    target_count = row["target_count"]
    record_id = row["record_id"]
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("catalog ordinal must be a non-negative integer")
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("catalog target count must be a positive integer")
    if type(record_id) is not str or _SHA256_RE.fullmatch(record_id) is None:
        raise ValueError("catalog record ID must be a lowercase SHA-256")
    if record_id != catalog_record_id(draft, target_count):
        raise ValueError("catalog record ID commitment drift")
    return CatalogRecord(
        ordinal=ordinal,
        record_id=record_id,
        lane_id=draft.lane_id,
        source_id=draft.source_id,
        source_key=draft.source_key,
        source_byte_sha256=draft.source_byte_sha256,
        source_locator=draft.source_locator,
        target_count=target_count,
        semantic_flags=draft.semantic_flags,
        semantic_facts=draft.semantic_facts,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _require_owned_regular(metadata: os.stat_result, description: str) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError(f"{description} is not an owner-controlled regular file")


def _require_owned_directory(metadata: os.stat_result, description: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError(f"{description} is not an owner-controlled directory")


def _read_regular_path(path: Path, description: str) -> bytes:
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, name = open_parent_directory(path)
        named = entry_lstat(parent_fd, name)
        _require_owned_regular(named, description)
        descriptor, opened = open_regular_file_at(parent_fd, name)
        _require_owned_regular(opened, description)
        if _file_identity(named) != _file_identity(opened):
            raise ValueError(f"{description} identity changed before reading")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = entry_lstat(parent_fd, name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"{description} identity changed while reading")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise ValueError(f"{description} length changed while reading")
        return payload
    except OSError as error:
        raise ValueError(f"{description} is missing or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def _records_from_jsonl(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_record_count: int | None = None,
    expected_target_count: int | None = None,
    expected_lanes: tuple[CatalogLaneIndex, ...] | None = None,
) -> Iterator[CatalogRecord]:
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, name = open_parent_directory(path)
        named = entry_lstat(parent_fd, name)
        _require_owned_regular(named, "catalog records")
        descriptor, opened = open_regular_file_at(parent_fd, name)
        _require_owned_regular(opened, "catalog records")
        if _file_identity(named) != _file_identity(opened):
            raise ValueError("catalog records identity changed before reading")
        digest = hashlib.sha256()
        record_count = 0
        target_count = 0
        lane_rows: dict[LaneId, list[int]] = {
            lane_id: [] for lane_id in LANE_ORDER
        }
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for raw_line in handle:
                if not raw_line.endswith(b"\n"):
                    raise ValueError(
                        "catalog JSONL line is not newline-terminated"
                    )
                digest.update(raw_line)
                value = _strict_json_bytes(raw_line, "catalog JSONL row")
                if canonical_json_bytes(value) != raw_line:
                    raise ValueError("catalog JSONL row is not canonical")
                record = _record_from_dict(value)
                if record.ordinal != record_count:
                    raise ValueError("catalog record ordinals are not contiguous")
                record_count += 1
                target_count += record.target_count
                lane_rows[record.lane_id].append(record.target_count)
                yield record
        after = os.fstat(descriptor)
        named_after = entry_lstat(parent_fd, name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
        ):
            raise ValueError("catalog records identity changed while reading")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError("catalog records SHA-256 disagreement")
        if (
            expected_record_count is not None
            and record_count != expected_record_count
        ):
            raise ValueError("catalog record count disagreement")
        if (
            expected_target_count is not None
            and target_count != expected_target_count
        ):
            raise ValueError("catalog target count disagreement")
        if expected_lanes is not None:
            first = 0
            for lane in expected_lanes:
                values = lane_rows[lane.lane_id]
                if (
                    lane.first_ordinal != first
                    or lane.record_count != len(values)
                    or lane.target_count != sum(values)
                ):
                    raise ValueError("catalog lane index disagreement")
                first += len(values)
    except OSError as error:
        raise ValueError("catalog records are missing or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def records_from_jsonl(path: Path) -> Iterator[CatalogRecord]:
    return _records_from_jsonl(Path(path))


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _open_new_regular(directory_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )


def _verify_regular_at(
    directory_fd: int,
    name: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    named = entry_lstat(directory_fd, name)
    _require_owned_regular(named, f"catalog artifact {name}")
    descriptor, opened = open_regular_file_at(directory_fd, name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        _require_owned_regular(opened, f"catalog artifact {name}")
        if _file_identity(named) != _file_identity(opened):
            raise ValueError(f"catalog artifact identity drift: {name}")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        named_after = entry_lstat(directory_fd, name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"catalog artifact identity drift: {name}")
    finally:
        os.close(descriptor)
    if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError(f"catalog artifact content drift: {name}")


def _new_stage(
    parent_fd: int,
    final_name: str,
) -> tuple[str, int, tuple[int, int, int, int]]:
    for _attempt in range(16):
        stage_name = (
            f".{final_name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
        )
        try:
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        fsync_directory(parent_fd)
        stage_fd, _created = open_directory_at(parent_fd, stage_name)
        metadata = os.fstat(stage_fd)
        named = entry_lstat(parent_fd, stage_name)
        _require_owned_directory(metadata, "private catalog candidate")
        if _directory_identity(metadata) != _directory_identity(named):
            os.close(stage_fd)
            raise ValueError("private catalog candidate identity drift")
        return stage_name, stage_fd, _directory_identity(metadata)
    raise FileExistsError("could not allocate private catalog candidate")


def _cleanup_stage(
    parent_fd: int,
    stage_name: str,
    stage_fd: int,
    stage_identity: tuple[int, int, int, int],
) -> None:
    opened = os.fstat(stage_fd)
    named = entry_lstat(parent_fd, stage_name)
    if (
        _directory_identity(opened) != stage_identity
        or _directory_identity(named) != stage_identity
    ):
        raise ValueError("private catalog candidate cleanup identity drift")
    for name in list_entries(stage_fd):
        metadata = entry_lstat(stage_fd, name)
        _require_owned_regular(metadata, f"private catalog candidate {name}")
        descriptor, opened_file = open_regular_file_at(stage_fd, name)
        try:
            if _file_identity(metadata) != _file_identity(opened_file):
                raise ValueError(
                    f"private catalog candidate file identity drift: {name}"
                )
        finally:
            os.close(descriptor)
        current = entry_lstat(stage_fd, name)
        if _file_identity(current) != _file_identity(metadata):
            raise ValueError(
                f"private catalog candidate file identity drift: {name}"
            )
        os.unlink(name, dir_fd=stage_fd)
    fsync_directory(stage_fd)
    if list_entries(stage_fd):
        raise ValueError("private catalog candidate cleanup is incomplete")
    os.rmdir(stage_name, dir_fd=parent_fd)
    fsync_directory(parent_fd)


def _create_spool(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(database_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE drafts ("
        "lane_rank INTEGER NOT NULL, "
        "source_key BLOB NOT NULL PRIMARY KEY, "
        "payload BLOB NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "CREATE INDEX drafts_lane_order ON drafts(lane_rank, source_key)"
    )
    connection.execute(
        "CREATE TABLE facts ("
        "fact_id BLOB NOT NULL PRIMARY KEY, "
        "metadata BLOB NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE record_ids ("
        "record_id BLOB NOT NULL PRIMARY KEY"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE generated_seeds ("
        "seed_key BLOB NOT NULL PRIMARY KEY, "
        "record_id BLOB NOT NULL"
        ") WITHOUT ROWID"
    )
    return connection


def _spool_fact_metadata(
    connection: sqlite3.Connection,
    fact: SemanticFactRow,
) -> None:
    fact_id = _byte_key(fact.fact_id)
    metadata = canonical_json_bytes(_fact_identity_dict(fact))
    existing = connection.execute(
        "SELECT metadata FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO facts(fact_id, metadata) VALUES (?, ?)",
            (fact_id, metadata),
        )
    elif bytes(existing[0]) != metadata:
        raise ValueError(f"semantic fact ID metadata conflict: {fact.fact_id}")


def _spool_drafts(
    connection: sqlite3.Connection,
    lane_sources: Mapping[LaneId, LaneCatalogSource],
    source_root: Path,
    lengths_by_lane: Mapping[LaneId, tuple[int, ...]],
    entries: Mapping[str, SourceEntry],
) -> None:
    for lane_rank, lane_id in enumerate(LANE_ORDER):
        source = lane_sources[lane_id]
        lengths = lengths_by_lane[lane_id]
        try:
            iterator = iter(source.iter_drafts(source_root, lengths))
        except Exception as error:
            raise ValueError(f"lane source failed to start: {lane_id}") from error
        for emitted in range(len(lengths)):
            try:
                draft = next(iterator)
            except StopIteration:
                if source.finite:
                    raise LaneQuotaShortfall(lane_id, len(lengths), emitted)
                raise ValueError(
                    f"non-finite lane source ended unexpectedly: {lane_id}"
                )
            except Exception as error:
                raise ValueError(f"lane source failed: {lane_id}") from error
            draft = _validate_draft_structure(draft)
            if draft.lane_id != lane_id:
                raise ValueError(
                    f"lane source emitted the wrong lane: "
                    f"expected={lane_id}, actual={draft.lane_id}"
                )
            _manifest_file(draft, entries)
            for fact in draft.semantic_facts:
                _spool_fact_metadata(connection, fact)
            try:
                connection.execute(
                    "INSERT INTO drafts(lane_rank, source_key, payload) "
                    "VALUES (?, ?, ?)",
                    (
                        lane_rank,
                        _byte_key(draft.source_key),
                        canonical_json_bytes(_draft_dict(draft)),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"duplicate source key: {draft.source_key}"
                ) from error
        connection.commit()


def _seed_key(draft: CatalogDraft) -> bytes | None:
    values = dict(draft.source_locator)
    if "seed" not in values:
        return None
    return canonical_json_bytes(
        {
            "lane_id": draft.lane_id,
            "seed": values["seed"],
            "source_id": draft.source_id,
        }
    )


def _write_records(
    connection: sqlite3.Connection,
    descriptor: int,
    lengths_by_lane: Mapping[LaneId, tuple[int, ...]],
    lane_sources: Mapping[LaneId, LaneCatalogSource],
) -> tuple[str, int, int, tuple[CatalogLaneIndex, ...]]:
    digest = hashlib.sha256()
    byte_count = 0
    ordinal = 0
    lanes = []
    for lane_rank, lane_id in enumerate(LANE_ORDER):
        first_ordinal = ordinal
        lengths = lengths_by_lane[lane_id]
        graph_training_keys: list[str] = []
        graph_revisit_seen = False
        rows = connection.execute(
            "SELECT payload FROM drafts "
            "WHERE lane_rank = ? ORDER BY source_key",
            (lane_rank,),
        )
        local_count = 0
        local_targets = 0
        for local_count, (raw_payload,) in enumerate(rows, start=1):
            draft_value = _strict_json_bytes(
                bytes(raw_payload),
                "spooled catalog draft",
            )
            if canonical_json_bytes(draft_value) != bytes(raw_payload):
                raise ValueError("spooled catalog draft is not canonical")
            draft = _draft_from_dict(draft_value)
            target_count = lengths[local_count - 1]
            record_id = catalog_record_id(draft, target_count)
            try:
                connection.execute(
                    "INSERT INTO record_ids(record_id) VALUES (?)",
                    (_byte_key(record_id),),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate catalog record ID: {record_id}") from error
            seed_key = _seed_key(draft)
            if seed_key is not None:
                try:
                    connection.execute(
                        "INSERT INTO generated_seeds(seed_key, record_id) "
                        "VALUES (?, ?)",
                        (seed_key, _byte_key(record_id)),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"generated seed is reused under another record ID: "
                        f"{draft.source_key}"
                    ) from error
            if lane_id == "wikidata_graph":
                revisit = "graph-revisit" in draft.semantic_flags
                if revisit:
                    graph_revisit_seen = True
                elif graph_revisit_seen:
                    raise ValueError(
                        "Wikidata training edge appears after graph revisit"
                    )
                else:
                    graph_training_keys.append(draft.source_key)
            record = CatalogRecord(
                ordinal=ordinal,
                record_id=record_id,
                lane_id=draft.lane_id,
                source_id=draft.source_id,
                source_key=draft.source_key,
                source_byte_sha256=draft.source_byte_sha256,
                source_locator=draft.source_locator,
                target_count=target_count,
                semantic_flags=draft.semantic_flags,
                semantic_facts=draft.semantic_facts,
            )
            payload = canonical_json_bytes(_record_dict(record))
            _write_all(descriptor, payload)
            digest.update(payload)
            byte_count += len(payload)
            ordinal += 1
            local_targets += target_count
        if local_count != len(lengths):
            raise LaneQuotaShortfall(lane_id, len(lengths), local_count)
        if lane_id == "wikidata_graph":
            expected = getattr(
                lane_sources[lane_id],
                "training_edge_keys",
                None,
            )
            if expected is not None:
                if type(expected) not in {tuple, frozenset, set}:
                    raise ValueError(
                        "Wikidata training edge keys must be a finite collection"
                    )
                expected_keys = set(expected)
                if any(type(key) is not str for key in expected_keys):
                    raise ValueError("Wikidata training edge key type drift")
                if set(graph_training_keys) != expected_keys:
                    raise ValueError(
                        "Wikidata graph does not cover every training edge "
                        "before revisit"
                    )
        lanes.append(
            CatalogLaneIndex(
                lane_id=lane_id,
                first_ordinal=first_ordinal,
                record_count=local_count,
                target_count=local_targets,
            )
        )
    connection.commit()
    return digest.hexdigest(), byte_count, ordinal, tuple(lanes)


def build_input_catalog(
    geometry: BuildGeometry,
    source_lock: SourceLock,
    source_root: Path,
    lane_sources: Mapping[LaneId, LaneCatalogSource],
    output_root: Path,
) -> InputCatalog:
    quotas = _validate_geometry(geometry)
    if not isinstance(source_lock, SourceLock):
        raise TypeError("source_lock must be a SourceLock")
    if not isinstance(source_root, Path):
        raise TypeError("source_root must be a pathlib.Path")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    if (
        source_root.name != source_lock.sha256
        or source_root.parent.name != "sources"
    ):
        raise ValueError("source_root is not the content-addressed source-lock path")
    sources = _validated_lane_sources(lane_sources)
    lengths_by_lane = _validated_lengths(quotas, geometry.context_length)
    verify_source_tree(
        source_lock,
        source_root,
        expected_generator_commit=source_lock.generator_commit,
    )
    entries = {entry.source_id: entry for entry in source_lock.sources}

    parent_fd = -1
    stage_fd = -1
    records_fd = -1
    connection: sqlite3.Connection | None = None
    stage_name = ""
    stage_identity: tuple[int, int, int, int] | None = None
    published = False
    try:
        parent_fd, final_name = open_parent_directory(
            output_root,
            create=True,
            mode=0o700,
        )
        parent_metadata = os.fstat(parent_fd)
        _require_owned_directory(parent_metadata, "catalog output parent")
        if entry_exists(parent_fd, final_name):
            raise FileExistsError(
                f"catalog output already exists; no-replace publication required: "
                f"{output_root}"
            )
        stage_name, stage_fd, stage_identity = _new_stage(parent_fd, final_name)
        stage_path = output_root.parent / stage_name
        connection = _create_spool(stage_path / ".catalog-spool.sqlite3")
        _spool_drafts(
            connection,
            sources,
            source_root,
            lengths_by_lane,
            entries,
        )

        records_fd = _open_new_regular(stage_fd, "catalog.jsonl")
        (
            records_sha256,
            records_bytes,
            record_count,
            lanes,
        ) = _write_records(
            connection,
            records_fd,
            lengths_by_lane,
            sources,
        )
        os.fsync(records_fd)
        os.close(records_fd)
        records_fd = -1
        _verify_regular_at(
            stage_fd,
            "catalog.jsonl",
            expected_bytes=records_bytes,
            expected_sha256=records_sha256,
        )

        verify_source_tree(
            source_lock,
            source_root,
            expected_generator_commit=source_lock.generator_commit,
        )
        if tuple((lane.lane_id, lane.target_count) for lane in lanes) != quotas:
            raise ValueError("catalog lane target counts disagree with geometry")
        target_count = sum(lane.target_count for lane in lanes)
        if target_count != geometry.total_targets:
            raise ValueError("catalog target count disagrees with geometry")

        index_bytes = canonical_json_bytes(
            {
                "catalog_sha256": records_sha256,
                "format": _CATALOG_FORMAT,
                "lanes": [_lane_index_dict(lane) for lane in lanes],
                "record_count": record_count,
                "schema_version": 1,
                "source_lock_sha256": source_lock.sha256,
                "target_count": target_count,
            }
        )
        index_fd = _open_new_regular(stage_fd, "catalog-index.json")
        try:
            _write_all(index_fd, index_bytes)
            os.fsync(index_fd)
        finally:
            os.close(index_fd)
        _verify_regular_at(
            stage_fd,
            "catalog-index.json",
            expected_bytes=len(index_bytes),
            expected_sha256=sha256_hex(index_bytes),
        )

        connection.close()
        connection = None
        os.unlink(".catalog-spool.sqlite3", dir_fd=stage_fd)
        fsync_directory(stage_fd)
        if list_entries(stage_fd) != ("catalog-index.json", "catalog.jsonl"):
            raise ValueError("private catalog candidate inventory drift")
        current_stage = os.fstat(stage_fd)
        named_stage = entry_lstat(parent_fd, stage_name)
        if (
            _directory_identity(current_stage) != stage_identity
            or _directory_identity(named_stage) != stage_identity
        ):
            raise ValueError("private catalog candidate identity drift")
        fsync_directory(stage_fd)
        try:
            atomic_rename_noreplace(
                parent_fd,
                stage_name,
                parent_fd,
                final_name,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"catalog no-replace publication lost race: {output_root}"
            ) from error
        fsync_directory(parent_fd)
        final_named = entry_lstat(parent_fd, final_name)
        if _directory_identity(final_named) != stage_identity:
            raise ValueError("published catalog identity drift")
        stage_name = ""
        published = True
        return InputCatalog(
            root=output_root,
            records_path=output_root / "catalog.jsonl",
            index_path=output_root / "catalog-index.json",
            source_lock_sha256=source_lock.sha256,
            sha256=records_sha256,
            record_count=record_count,
            target_count=target_count,
            lanes=lanes,
        )
    except BaseException as build_error:
        cleanup_error: BaseException | None = None
        if connection is not None:
            connection.close()
            connection = None
        if records_fd >= 0:
            os.close(records_fd)
            records_fd = -1
        if (
            not published
            and parent_fd >= 0
            and stage_fd >= 0
            and stage_name
            and stage_identity is not None
        ):
            try:
                _cleanup_stage(
                    parent_fd,
                    stage_name,
                    stage_fd,
                    stage_identity,
                )
                stage_name = ""
            except BaseException as error:
                cleanup_error = error
        if cleanup_error is not None:
            raise ValueError(
                "catalog build failed and private candidate cleanup failed"
            ) from build_error
        raise
    finally:
        if connection is not None:
            connection.close()
        if records_fd >= 0:
            os.close(records_fd)
        if stage_fd >= 0:
            os.close(stage_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


__all__ = (
    "CatalogDraft",
    "CatalogLaneIndex",
    "CatalogRecord",
    "InputCatalog",
    "LaneCatalogSource",
    "LaneQuotaShortfall",
    "SemanticFactRow",
    "build_input_catalog",
    "catalog_record_id",
    "records_from_jsonl",
)
