"""Canonical external-memory input catalog for the reasoning-v2 corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import threading
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Protocol, cast

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
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
)
from corpusgen.reasoning_v2.source_lock import (
    SourceEntry,
    SourceFile,
    SourceLock,
    verify_source_tree,
)
from corpusgen.reasoning_v2.wikidata_source import (
    V2TrainingTriple,
    WikidataDerivedView,
    iter_distinct_training_edges,
    iter_v2_training_triples,
)


_CATALOG_FORMAT = "memorysplit-reasoning-v2-input-catalog-v1"
_CATALOG_QUARANTINE_DIRECTORY = ".memorysplit-catalog-quarantine-v1"
_CATALOG_SPOOL_NAME = ".catalog-spool.sqlite3"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SEALED_PATH_TERMS = ("evaluation", "validation", "test", "sealed", "holdout")
_PATH_LOCATOR_KEYS = frozenset({"file", "member", "path", "source_path", "split"})
_GENERATION_SEED_KEY = "generation_seed"
_GENERATED_LANES = frozenset(
    {
        "synthetic_graph",
        "verified_synthetic_multihop",
    }
)
_LANE_SOURCE_AUTHORITY: dict[LaneId, frozenset[str]] = {
    "fineweb_edu": frozenset({"fineweb_edu"}),
    "finemath": frozenset({"finemath"}),
    "wikidata_graph": frozenset({"wikidata5m"}),
    "synthetic_graph": frozenset(
        {
            "deepmind_mathematics_generator",
            "reasoning_gym_exact_answer",
        }
    ),
    "verified_synthetic_multihop": frozenset(
        {
            "deepmind_mathematics_generator",
            "reasoning_gym_exact_answer",
        }
    ),
    "wikidata_path_reasoning": frozenset({"wikidata5m"}),
    "relational_refinement": frozenset(
        {
            "clrs_text",
            "prontoqa",
            "reasoning_gym_exact_answer",
            "ruletaker",
        }
    ),
    "objective_auxiliary": frozenset(
        {
            "arc_agi_1",
            "arc_agi_2",
            "clrs_text",
            "conceptarc",
            "deepmind_mathematics_generator",
            "prontoqa",
            "reasoning_gym_exact_answer",
            "ruletaker",
        }
    ),
}
_FINEWEB_TRAINING_PATHS = frozenset(
    {
        "sample/10BT/000_00000.parquet",
        "sample/10BT/001_00000.parquet",
        "sample/10BT/002_00000.parquet",
    }
)
_SQLITE_OPEN_LOCK = threading.Lock()
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
    wikidata_view_sha256: str
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
        target_lengths: "TargetLengths",
    ) -> Iterator[CatalogDraft]:
        raise RuntimeError("lane source must emit deterministic drafts")


class TargetLengths(Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[int]: ...

    def __getitem__(self, index: int) -> int: ...


class WikidataGraphCatalogSource:
    lane_id: LaneId = "wikidata_graph"
    finite = False

    __slots__ = (
        "_archive_hashes",
        "_source_lock_sha256",
        "_training_edge_count",
        "_training_rows",
        "_view",
        "_view_sha256",
    )

    def __init__(self, view: WikidataDerivedView) -> None:
        if not isinstance(view, WikidataDerivedView):
            raise TypeError("view must be a WikidataDerivedView")
        distinct_edges = sum(1 for _triple in iter_distinct_training_edges(view))
        if distinct_edges <= 0:
            raise ValueError("Wikidata derived view has no distinct training edges")
        receipt = view.receipt
        self._view = view
        self._view_sha256 = view.receipt_sha256
        self._source_lock_sha256 = receipt.source_lock_sha256
        self._archive_hashes = tuple(
            (record.path, record.sha256) for record in receipt.archives
        )
        self._training_edge_count = distinct_edges
        self._training_rows = receipt.training_rows

    @property
    def training_edge_count(self) -> int:
        return self._training_edge_count

    @property
    def wikidata_view_sha256(self) -> str:
        return self._view_sha256

    @property
    def wikidata_source_lock_sha256(self) -> str:
        return self._source_lock_sha256

    @staticmethod
    def _training_edge_key(triple: V2TrainingTriple) -> str:
        return f"Q{triple.subject}\t{triple.relation}\tQ{triple.object}"

    def _archive_sha256(self, archive_path: str) -> str:
        for path, sha256 in self._archive_hashes:
            if path == archive_path:
                return sha256
        raise ValueError(f"Wikidata archive is absent from the view: {archive_path}")

    def _draft(
        self,
        triple: V2TrainingTriple,
        *,
        phase: str,
        phase_index: int,
    ) -> CatalogDraft:
        edge_key = self._training_edge_key(triple)
        return CatalogDraft(
            lane_id=self.lane_id,
            source_id="wikidata5m",
            source_key=(
                f"wikidata_graph:{phase}:{phase_index:020d}:{edge_key}"
            ),
            source_byte_sha256=self._archive_sha256(triple.archive_path),
            source_locator=(
                ("member", triple.member),
                ("path", triple.archive_path),
                ("row", triple.row),
                ("split", "train"),
                ("training_edge_key", edge_key),
                ("training_split", triple.training_split),
                ("wikidata_view_sha256", self._view_sha256),
            ),
            semantic_flags=(
                ("graph-training-edge",)
                if phase == "edge"
                else ("graph-revisit",)
            ),
            semantic_facts=(),
        )

    def iter_training_edge_keys(
        self,
        source_root: Path | None = None,
    ) -> Iterator[str]:
        del source_root
        emitted = 0
        for triple in iter_distinct_training_edges(self._view):
            emitted += 1
            yield self._training_edge_key(triple)
        if emitted != self._training_edge_count:
            raise ValueError("Wikidata distinct-edge count changed")

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: TargetLengths,
    ) -> Iterator[CatalogDraft]:
        del source_root
        available_records = len(target_lengths)
        if self._training_edge_count > available_records:
            raise ValueError(
                "Wikidata distinct edges exceed allocated records: "
                f"distinct_edges={self._training_edge_count}, "
                f"allocated_records={available_records}"
            )

        emitted = 0
        for edge_index, triple in enumerate(
            iter_distinct_training_edges(self._view)
        ):
            emitted += 1
            yield self._draft(
                triple,
                phase="edge",
                phase_index=edge_index,
            )
        if emitted != self._training_edge_count:
            raise ValueError("Wikidata distinct-edge count changed")

        revisit_index = 0
        while emitted < available_records:
            training_rows = 0
            for triple in iter_v2_training_triples(self._view):
                training_rows += 1
                yield self._draft(
                    triple,
                    phase="revisit",
                    phase_index=revisit_index,
                )
                emitted += 1
                revisit_index += 1
                if emitted == available_records:
                    return
            if training_rows != self._training_rows:
                raise ValueError("Wikidata training-row count changed")


@dataclass(frozen=True, slots=True)
class _BalancedTargetLengths:
    targets: int
    context_length: int
    record_count: int
    longer_count: int
    longer_length: int
    shorter_length: int

    @classmethod
    def create(
        cls,
        targets: int,
        context_length: int,
    ) -> "_BalancedTargetLengths":
        if type(targets) is not int or targets <= 0:
            raise ValueError("targets must be a positive integer")
        if type(context_length) is not int or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        record_count = (targets + context_length - 1) // context_length
        shorter_length, longer_count = divmod(targets, record_count)
        longer_length = shorter_length + (1 if longer_count else 0)
        return cls(
            targets=targets,
            context_length=context_length,
            record_count=record_count,
            longer_count=longer_count,
            longer_length=longer_length,
            shorter_length=shorter_length,
        )

    def __len__(self) -> int:
        return self.record_count

    def __iter__(self) -> Iterator[int]:
        for index in range(self.record_count):
            yield self[index]

    def __getitem__(self, index: int) -> int:
        if type(index) is not int:
            raise TypeError("target length index must be an integer")
        resolved = index + self.record_count if index < 0 else index
        if resolved < 0 or resolved >= self.record_count:
            raise IndexError("target length index out of range")
        return (
            self.longer_length
            if resolved < self.longer_count
            else self.shorter_length
        )


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


def _has_reserved_path_term(value: str) -> bool:
    lowered = value.casefold()
    return any(term in lowered for term in _SEALED_PATH_TERMS)


def _reserved_training_override(entry: SourceEntry, row: SourceFile) -> bool:
    if entry.source_id == "fineweb_edu":
        return row.path in _FINEWEB_TRAINING_PATHS
    if entry.source_id == "finemath":
        proof = entry.finemath_selection
        return proof is not None and row.path in proof.selected_paths
    if entry.source_id in {"arc_agi_1", "arc_agi_2"}:
        return row.path.startswith("data/training/")
    return False


def _reject_sealed_locators(
    draft: CatalogDraft,
    *,
    manifest_path: str,
    reserved_training_override: bool,
) -> None:
    for key, value in draft.source_locator:
        if key not in _PATH_LOCATOR_KEYS or type(value) is not str:
            continue
        if not _has_reserved_path_term(value):
            continue
        if (
            key in {"path", "source_path"}
            and value == manifest_path
            and reserved_training_override
        ):
            continue
        raise ValueError(f"sealed or evaluation path is forbidden: {value}")


def _manifest_file(
    draft: CatalogDraft,
    connection: sqlite3.Connection,
) -> None:
    if draft.source_id not in _LANE_SOURCE_AUTHORITY[draft.lane_id]:
        raise ValueError(
            f"source identity is not authorized for lane {draft.lane_id}: "
            f"{draft.source_id}"
        )
    locator = dict(draft.source_locator)
    path_value = locator.get("path", locator.get("source_path"))
    if type(path_value) is not str:
        raise ValueError("source locator must contain one string path")
    if "path" in locator and "source_path" in locator:
        raise ValueError("source locator has ambiguous source paths")
    path_text = _safe_relative_path(path_value, "source locator path")
    materialized_row = connection.execute(
        "SELECT materialized_path FROM source_entries WHERE source_id = ?",
        (_byte_key(draft.source_id),),
    ).fetchone()
    if materialized_row is None:
        raise ValueError(f"catalog draft references unknown source: {draft.source_id}")
    materialized_path = bytes(materialized_row[0]).decode("utf-8")
    prefix = f"{materialized_path}/"
    relative = path_text[len(prefix) :] if path_text.startswith(prefix) else path_text
    manifest_row = connection.execute(
        "SELECT sha256, reserved_training_override "
        "FROM source_files WHERE source_id = ? AND path = ?",
        (_byte_key(draft.source_id), _byte_key(relative)),
    ).fetchone()
    if manifest_row is None:
        raise ValueError(
            f"source locator path is not in the verified manifest: "
            f"{draft.source_id}:{path_text}"
        )
    expected_sha256 = str(manifest_row[0])
    _reject_sealed_locators(
        draft,
        manifest_path=relative,
        reserved_training_override=bool(manifest_row[1]),
    )
    if draft.source_id == "wikidata5m" and locator.get("split") != "train":
        raise ValueError("Wikidata source locator requires the exact training split")
    if draft.source_byte_sha256 != expected_sha256:
        raise ValueError(
            f"source byte hash disagreement: {draft.source_id}:{relative}"
        )


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
        if lane_id == "wikidata_graph":
            training_edge_count = getattr(source, "training_edge_count", None)
            edge_iterator = getattr(source, "iter_training_edge_keys", None)
            view_sha256 = getattr(source, "wikidata_view_sha256", None)
            view_source_lock_sha256 = getattr(
                source,
                "wikidata_source_lock_sha256",
                None,
            )
            if (
                type(training_edge_count) is not int
                or training_edge_count <= 0
                or not callable(edge_iterator)
            ):
                raise ValueError(
                    "Wikidata graph training-edge authority is required"
                )
            if (
                type(view_sha256) is not str
                or _SHA256_RE.fullmatch(view_sha256) is None
                or type(view_source_lock_sha256) is not str
                or _SHA256_RE.fullmatch(view_source_lock_sha256) is None
            ):
                raise ValueError(
                    "Wikidata graph verified-view authority is required"
                )
        result[lane_id] = cast(LaneCatalogSource, source)
    return result


def _validated_lengths(
    quotas: tuple[tuple[LaneId, int], ...],
    context_length: int,
) -> dict[LaneId, _BalancedTargetLengths]:
    result: dict[LaneId, _BalancedTargetLengths] = {}
    for lane_id, quota in quotas:
        lengths = _BalancedTargetLengths.create(quota, context_length)
        if (
            len(lengths) <= 0
            or lengths.longer_length > context_length
            or lengths.shorter_length <= 0
            or lengths.longer_count * lengths.longer_length
            + (len(lengths) - lengths.longer_count) * lengths.shorter_length
            != quota
        ):
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


def _regular_inode_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
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
        lane_position = 0
        lane_record_count = 0
        lane_target_count = 0
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
                if expected_lanes is not None:
                    if lane_position >= len(expected_lanes):
                        raise ValueError("catalog contains excess lane records")
                    expected_lane = expected_lanes[lane_position]
                    if record.lane_id != expected_lane.lane_id:
                        raise ValueError("catalog lane order disagreement")
                    lane_record_count += 1
                    lane_target_count += record.target_count
                    if lane_record_count == expected_lane.record_count:
                        if (
                            lane_target_count != expected_lane.target_count
                            or expected_lane.first_ordinal
                            != record_count - lane_record_count + 1
                        ):
                            raise ValueError("catalog lane index disagreement")
                        lane_position += 1
                        lane_record_count = 0
                        lane_target_count = 0
                record_count += 1
                target_count += record.target_count
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
            if (
                lane_position != len(expected_lanes)
                or lane_record_count != 0
                or lane_target_count != 0
            ):
                raise ValueError("catalog lane index disagreement")
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


def _read_regular_at(
    directory_fd: int,
    name: str,
    description: str,
) -> bytes:
    named = entry_lstat(directory_fd, name)
    _require_owned_regular(named, description)
    descriptor, opened = open_regular_file_at(directory_fd, name)
    chunks = []
    try:
        _require_owned_regular(opened, description)
        if _file_identity(named) != _file_identity(opened):
            raise ValueError(f"{description} identity drift")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = entry_lstat(directory_fd, name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"{description} identity drift")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise ValueError(f"{description} size drift")
    return payload


def _catalog_result(
    output_root: Path,
    source_lock_sha256: str,
    wikidata_view_sha256: str,
    records_sha256: str,
    record_count: int,
    target_count: int,
    lanes: tuple[CatalogLaneIndex, ...],
) -> InputCatalog:
    return InputCatalog(
        root=output_root,
        records_path=output_root / "catalog.jsonl",
        index_path=output_root / "catalog-index.json",
        source_lock_sha256=source_lock_sha256,
        wikidata_view_sha256=wikidata_view_sha256,
        sha256=records_sha256,
        record_count=record_count,
        target_count=target_count,
        lanes=lanes,
    )


def _verify_exact_catalog_winner(
    parent_fd: int,
    final_name: str,
    output_root: Path,
    *,
    expected_index_bytes: bytes,
    expected_records_bytes: int,
    expected_records_sha256: str,
    source_lock_sha256: str,
    wikidata_view_sha256: str,
    record_count: int,
    target_count: int,
    lanes: tuple[CatalogLaneIndex, ...],
) -> InputCatalog:
    winner_fd = -1
    try:
        winner_fd, _created = open_directory_at(parent_fd, final_name)
        opened = os.fstat(winner_fd)
        named = entry_lstat(parent_fd, final_name)
        _require_owned_directory(opened, "catalog winner")
        winner_identity = _directory_identity(opened)
        if _directory_identity(named) != winner_identity:
            raise ValueError("catalog winner identity drift")
        if list_entries(winner_fd) != ("catalog-index.json", "catalog.jsonl"):
            raise ValueError("catalog winner file inventory differs")
        winner_index = _read_regular_at(
            winner_fd,
            "catalog-index.json",
            "catalog winner index",
        )
        if winner_index != expected_index_bytes:
            raise ValueError("catalog winner index differs")
        _verify_regular_at(
            winner_fd,
            "catalog.jsonl",
            expected_bytes=expected_records_bytes,
            expected_sha256=expected_records_sha256,
        )
        if _directory_identity(os.fstat(winner_fd)) != winner_identity:
            raise ValueError("catalog winner identity drift")
        named_after = entry_lstat(parent_fd, final_name)
        if _directory_identity(named_after) != winner_identity:
            raise ValueError("catalog winner identity drift")
        return _catalog_result(
            output_root,
            source_lock_sha256,
            wikidata_view_sha256,
            expected_records_sha256,
            record_count,
            target_count,
            lanes,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(f"conflicting catalog winner: {output_root}") from error
    finally:
        if winner_fd >= 0:
            os.close(winner_fd)


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


def _open_catalog_quarantine(
    parent_fd: int,
) -> tuple[int, tuple[int, int, int, int]]:
    quarantine_fd, _created = open_directory_at(
        parent_fd,
        _CATALOG_QUARANTINE_DIRECTORY,
        create=True,
        mode=0o700,
    )
    metadata = os.fstat(quarantine_fd)
    named = entry_lstat(parent_fd, _CATALOG_QUARANTINE_DIRECTORY)
    _require_owned_directory(metadata, "catalog quarantine directory")
    identity = _directory_identity(metadata)
    if _directory_identity(named) != identity:
        os.close(quarantine_fd)
        raise ValueError("catalog quarantine directory identity drift")
    return quarantine_fd, identity


def _quarantine_stage(
    parent_fd: int,
    stage_name: str,
    stage_fd: int,
    stage_identity: tuple[int, int, int, int],
    quarantine_fd: int,
) -> str:
    opened = os.fstat(stage_fd)
    named = entry_lstat(parent_fd, stage_name)
    if (
        _directory_identity(opened) != stage_identity
        or _directory_identity(named) != stage_identity
    ):
        raise ValueError("private catalog candidate quarantine identity drift")
    quarantine_name = ""
    for _attempt in range(16):
        candidate = f"stage-{secrets.token_hex(16)}"
        try:
            atomic_rename_noreplace(
                parent_fd,
                stage_name,
                quarantine_fd,
                candidate,
            )
        except FileExistsError:
            continue
        quarantine_name = candidate
        break
    if not quarantine_name:
        raise FileExistsError("could not allocate catalog stage quarantine")
    fsync_directory(parent_fd)
    fsync_directory(quarantine_fd)
    quarantined_fd, _created = open_directory_at(
        quarantine_fd,
        quarantine_name,
    )
    try:
        quarantined = os.fstat(quarantined_fd)
        named_quarantined = entry_lstat(quarantine_fd, quarantine_name)
        pinned_after = os.fstat(stage_fd)
        if (
            _directory_identity(quarantined) != stage_identity
            or _directory_identity(named_quarantined) != stage_identity
            or _directory_identity(pinned_after) != stage_identity
        ):
            raise ValueError("quarantined catalog stage identity drift")
    finally:
        os.close(quarantined_fd)
    return quarantine_name


def _quarantine_spool(
    stage_fd: int,
    spool_name: str,
    spool_fd: int,
    spool_identity: tuple[int, int, int, int, int],
    quarantine_fd: int,
) -> str:
    opened = os.fstat(spool_fd)
    named = entry_lstat(stage_fd, spool_name)
    if (
        _regular_inode_identity(opened) != spool_identity
        or _regular_inode_identity(named) != spool_identity
    ):
        raise ValueError("SQLite spool quarantine identity drift")
    quarantine_name = ""
    for _attempt in range(16):
        candidate = f"spool-{secrets.token_hex(16)}.sqlite3"
        try:
            atomic_rename_noreplace(
                stage_fd,
                spool_name,
                quarantine_fd,
                candidate,
            )
        except FileExistsError:
            continue
        quarantine_name = candidate
        break
    if not quarantine_name:
        raise FileExistsError("could not allocate SQLite spool quarantine")
    fsync_directory(stage_fd)
    fsync_directory(quarantine_fd)
    quarantined_fd, quarantined = open_regular_file_at(
        quarantine_fd,
        quarantine_name,
    )
    try:
        named_quarantined = entry_lstat(quarantine_fd, quarantine_name)
        pinned_after = os.fstat(spool_fd)
        if (
            _regular_inode_identity(quarantined) != spool_identity
            or _regular_inode_identity(named_quarantined) != spool_identity
            or _regular_inode_identity(pinned_after) != spool_identity
        ):
            raise ValueError("quarantined SQLite spool identity drift")
    finally:
        os.close(quarantined_fd)
    return quarantine_name


def _spool_open_hook(
    phase: str,
    stage_fd: int,
    name: str,
    pinned_fd: int,
) -> None:
    del phase, stage_fd, name, pinned_fd


@dataclass
class _PinnedSpool:
    connection: sqlite3.Connection
    descriptor: int
    identity: tuple[int, int, int, int, int]
    name: str

    def close_connection(self, stage_fd: int) -> None:
        if self.connection is None:
            return
        quick_check = self.connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise ValueError("SQLite spool integrity check failed")
        self.connection.commit()
        self.connection.close()
        self.connection = cast(sqlite3.Connection, None)
        pinned = os.fstat(self.descriptor)
        named = entry_lstat(stage_fd, self.name)
        if (
            _regular_inode_identity(pinned) != self.identity
            or _regular_inode_identity(named) != self.identity
        ):
            raise ValueError("SQLite spool identity drift after use")

    def abort_connection(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = cast(sqlite3.Connection, None)

    def close_descriptor(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _create_spool(stage_fd: int) -> _PinnedSpool:
    descriptor = os.open(
        _CATALOG_SPOOL_NAME,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=stage_fd,
    )
    connection: sqlite3.Connection | None = None
    try:
        pinned = os.fstat(descriptor)
        named = entry_lstat(stage_fd, _CATALOG_SPOOL_NAME)
        _require_owned_regular(pinned, "SQLite spool")
        identity = _regular_inode_identity(pinned)
        if _regular_inode_identity(named) != identity:
            raise ValueError("SQLite spool identity drift before open")
        named_before_open = entry_lstat(stage_fd, _CATALOG_SPOOL_NAME)
        if _regular_inode_identity(named_before_open) != identity:
            raise ValueError("SQLite spool namespace identity drift before open")
        with _SQLITE_OPEN_LOCK:
            _spool_open_hook(
                "before_sqlite_open",
                stage_fd,
                _CATALOG_SPOOL_NAME,
                descriptor,
            )
            _spool_open_hook(
                "before_cwd_snapshot",
                stage_fd,
                _CATALOG_SPOOL_NAME,
                descriptor,
            )
            cwd_fd = os.open(
                ".",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0),
            )
            _spool_open_hook(
                "after_cwd_snapshot",
                stage_fd,
                _CATALOG_SPOOL_NAME,
                descriptor,
            )
            try:
                os.fchdir(stage_fd)
                _spool_open_hook(
                    "after_stage_fchdir",
                    stage_fd,
                    _CATALOG_SPOOL_NAME,
                    descriptor,
                )
                connection = sqlite3.connect(
                    f"file:{_CATALOG_SPOOL_NAME}?mode=rw",
                    uri=True,
                )
                _spool_open_hook(
                    "sqlite_opened",
                    stage_fd,
                    _CATALOG_SPOOL_NAME,
                    descriptor,
                )
            finally:
                try:
                    os.fchdir(cwd_fd)
                finally:
                    os.close(cwd_fd)
                _spool_open_hook(
                    "cwd_restored",
                    stage_fd,
                    _CATALOG_SPOOL_NAME,
                    descriptor,
                )
            _spool_open_hook(
                "after_sqlite_open",
                stage_fd,
                _CATALOG_SPOOL_NAME,
                descriptor,
            )
        named_after_open = entry_lstat(stage_fd, _CATALOG_SPOOL_NAME)
        if (
            _regular_inode_identity(os.fstat(descriptor)) != identity
            or _regular_inode_identity(named_after_open) != identity
        ):
            raise ValueError("SQLite spool namespace identity drift after open")
    except BaseException:
        if connection is not None:
            connection.close()
        os.close(descriptor)
        raise
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE source_entries ("
        "source_id BLOB NOT NULL PRIMARY KEY, "
        "materialized_path BLOB NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE source_files ("
        "source_id BLOB NOT NULL, "
        "path BLOB NOT NULL, "
        "sha256 TEXT NOT NULL, "
        "reserved_training_override INTEGER NOT NULL, "
        "PRIMARY KEY (source_id, path)"
        ") WITHOUT ROWID"
    )
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
        "seed INTEGER NOT NULL PRIMARY KEY, "
        "record_id BLOB NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE wikidata_training_edges ("
        "edge_key BLOB NOT NULL PRIMARY KEY, "
        "seen_before_revisit INTEGER NOT NULL DEFAULT 0, "
        "seen_total INTEGER NOT NULL DEFAULT 0"
        ") WITHOUT ROWID"
    )
    return _PinnedSpool(
        connection=connection,
        descriptor=descriptor,
        identity=identity,
        name=_CATALOG_SPOOL_NAME,
    )


def _spool_source_authority(
    connection: sqlite3.Connection,
    source_lock: SourceLock,
) -> None:
    for entry in source_lock.sources:
        connection.execute(
            "INSERT INTO source_entries(source_id, materialized_path) "
            "VALUES (?, ?)",
            (
                _byte_key(entry.source_id),
                _byte_key(entry.materialized_path),
            ),
        )
        for row in entry.files:
            connection.execute(
                "INSERT INTO source_files("
                "source_id, path, sha256, reserved_training_override"
                ") VALUES (?, ?, ?, ?)",
                (
                    _byte_key(entry.source_id),
                    _byte_key(row.path),
                    row.sha256,
                    int(_reserved_training_override(entry, row)),
                ),
            )
    connection.commit()


def _spool_wikidata_training_edge_authority(
    connection: sqlite3.Connection,
    source: LaneCatalogSource,
    source_root: Path,
) -> int:
    expected_count = getattr(source, "training_edge_count")
    try:
        iterator = iter(source.iter_training_edge_keys(source_root))
    except Exception as error:
        raise ValueError(
            "Wikidata graph training-edge authority failed to start"
        ) from error
    for emitted in range(expected_count):
        try:
            edge_key = next(iterator)
        except StopIteration as error:
            raise ValueError(
                "Wikidata graph training-edge authority ended early"
            ) from error
        except Exception as error:
            raise ValueError(
                "Wikidata graph training-edge authority failed"
            ) from error
        edge_key = _require_nfc_string(edge_key, "Wikidata training edge key")
        try:
            connection.execute(
                "INSERT INTO wikidata_training_edges(edge_key) VALUES (?)",
                (_byte_key(edge_key),),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"duplicate Wikidata training edge authority: {edge_key}"
            ) from error
    try:
        next(iterator)
    except StopIteration:
        pass
    except Exception as error:
        raise ValueError(
            "Wikidata graph training-edge authority failed"
        ) from error
    else:
        raise ValueError(
            "Wikidata graph training-edge authority exceeds declared count"
        )
    connection.commit()
    return expected_count


def _generation_seed(draft: CatalogDraft) -> int | None:
    seed_keys = tuple(
        key for key, _value in draft.source_locator if "seed" in key.casefold()
    )
    values = dict(draft.source_locator)
    if draft.lane_id in _GENERATED_LANES:
        if seed_keys != (_GENERATION_SEED_KEY,):
            if not seed_keys:
                raise ValueError(
                    f"generation seed is required for lane {draft.lane_id}"
                )
            raise ValueError(
                f"generation seed must use canonical key "
                f"{_GENERATION_SEED_KEY!r}"
            )
        seed = values[_GENERATION_SEED_KEY]
        if type(seed) is not int or seed < 0 or seed >= 1 << 63:
            raise ValueError(
                "generation seed must be a non-negative signed 64-bit integer"
            )
        return seed
    if seed_keys:
        raise ValueError(
            f"non-generated lane {draft.lane_id} must not carry a seed"
        )
    return None


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
    lengths_by_lane: Mapping[LaneId, _BalancedTargetLengths],
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
            _manifest_file(draft, connection)
            _generation_seed(draft)
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


def _wikidata_training_edge_key(draft: CatalogDraft) -> str:
    values = dict(draft.source_locator)
    return _require_nfc_string(
        values.get("training_edge_key"),
        "Wikidata catalog training edge key",
    )


def _write_records(
    connection: sqlite3.Connection,
    descriptor: int,
    lengths_by_lane: Mapping[LaneId, _BalancedTargetLengths],
) -> tuple[str, int, int, tuple[CatalogLaneIndex, ...]]:
    digest = hashlib.sha256()
    byte_count = 0
    ordinal = 0
    lanes = []
    for lane_rank, lane_id in enumerate(LANE_ORDER):
        first_ordinal = ordinal
        lengths = lengths_by_lane[lane_id]
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
            generation_seed = _generation_seed(draft)
            if generation_seed is not None:
                try:
                    connection.execute(
                        "INSERT INTO generated_seeds(seed, record_id) "
                        "VALUES (?, ?)",
                        (generation_seed, _byte_key(record_id)),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"generated seed is reused under another record ID: "
                        f"{draft.source_key}"
                    ) from error
            if lane_id == "wikidata_graph":
                edge_key = _wikidata_training_edge_key(draft)
                edge_row = connection.execute(
                    "SELECT seen_before_revisit FROM wikidata_training_edges "
                    "WHERE edge_key = ?",
                    (_byte_key(edge_key),),
                ).fetchone()
                if edge_row is None:
                    raise ValueError(
                        f"Wikidata catalog edge is outside training authority: "
                        f"{edge_key}"
                    )
                revisit = "graph-revisit" in draft.semantic_flags
                if revisit:
                    if not graph_revisit_seen:
                        authority_count = connection.execute(
                            "SELECT COUNT(*) FROM wikidata_training_edges"
                        ).fetchone()[0]
                        covered_count = connection.execute(
                            "SELECT COUNT(*) FROM wikidata_training_edges "
                            "WHERE seen_before_revisit = 1"
                        ).fetchone()[0]
                        if covered_count != authority_count:
                            raise ValueError(
                                "Wikidata graph must cover every training edge "
                                "exactly once before the first revisit"
                            )
                    graph_revisit_seen = True
                    connection.execute(
                        "UPDATE wikidata_training_edges "
                        "SET seen_total = seen_total + 1 WHERE edge_key = ?",
                        (_byte_key(edge_key),),
                    )
                elif graph_revisit_seen:
                    raise ValueError(
                        "Wikidata training edge appears after graph revisit"
                    )
                else:
                    if int(edge_row[0]) != 0:
                        raise ValueError(
                            f"Wikidata training edge must appear exactly once "
                            f"before revisit: {edge_key}"
                        )
                    connection.execute(
                        "UPDATE wikidata_training_edges "
                        "SET seen_before_revisit = 1, seen_total = seen_total + 1 "
                        "WHERE edge_key = ?",
                        (_byte_key(edge_key),),
                    )
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
            authority_count = connection.execute(
                "SELECT COUNT(*) FROM wikidata_training_edges"
            ).fetchone()[0]
            covered_count = connection.execute(
                "SELECT COUNT(*) FROM wikidata_training_edges "
                "WHERE seen_before_revisit = 1"
            ).fetchone()[0]
            if covered_count != authority_count:
                raise ValueError(
                    "Wikidata graph does not cover every training edge "
                    "exactly once before revisit"
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
    *,
    expected_generator_commit: str,
) -> InputCatalog:
    quotas = _validate_geometry(geometry)
    if not isinstance(source_lock, SourceLock):
        raise TypeError("source_lock must be a SourceLock")
    if (
        type(expected_generator_commit) is not str
        or _COMMIT_RE.fullmatch(expected_generator_commit) is None
    ):
        raise ValueError(
            "expected generator commit must be a lowercase 40-character commit"
        )
    if source_lock.generator_commit != expected_generator_commit:
        raise ValueError("source lock generator commit does not match authority")
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
        expected_generator_commit=expected_generator_commit,
    )
    wikidata_source = sources["wikidata_graph"]
    wikidata_view_sha256 = cast(
        str,
        getattr(wikidata_source, "wikidata_view_sha256"),
    )
    if (
        getattr(wikidata_source, "wikidata_source_lock_sha256")
        != source_lock.sha256
    ):
        raise ValueError("Wikidata derived view source-lock binding disagrees")
    wikidata_records = len(lengths_by_lane["wikidata_graph"])
    wikidata_distinct_edges = getattr(
        wikidata_source,
        "training_edge_count",
    )
    if wikidata_distinct_edges > wikidata_records:
        raise ValueError(
            "Wikidata distinct edges exceed allocated records: "
            f"distinct_edges={wikidata_distinct_edges}, "
            f"allocated_records={wikidata_records}"
        )
    parent_fd = -1
    quarantine_fd = -1
    stage_fd = -1
    records_fd = -1
    spool: _PinnedSpool | None = None
    stage_name = ""
    stage_identity: tuple[int, int, int, int] | None = None
    try:
        parent_fd, final_name = open_parent_directory(
            output_root,
            create=True,
            mode=0o700,
        )
        parent_metadata = os.fstat(parent_fd)
        _require_owned_directory(parent_metadata, "catalog output parent")
        quarantine_fd, _quarantine_identity = _open_catalog_quarantine(
            parent_fd
        )
        stage_name, stage_fd, stage_identity = _new_stage(parent_fd, final_name)
        spool = _create_spool(stage_fd)
        _spool_source_authority(spool.connection, source_lock)
        _spool_wikidata_training_edge_authority(
            spool.connection,
            sources["wikidata_graph"],
            source_root,
        )
        _spool_drafts(
            spool.connection,
            sources,
            source_root,
            lengths_by_lane,
        )

        records_fd = _open_new_regular(stage_fd, "catalog.jsonl")
        (
            records_sha256,
            records_bytes,
            record_count,
            lanes,
        ) = _write_records(
            spool.connection,
            records_fd,
            lengths_by_lane,
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
            expected_generator_commit=expected_generator_commit,
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
                "wikidata_view_sha256": wikidata_view_sha256,
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

        spool.close_connection(stage_fd)
        _quarantine_spool(
            stage_fd,
            spool.name,
            spool.descriptor,
            spool.identity,
            quarantine_fd,
        )
        spool.close_descriptor()
        spool = None
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
        except FileExistsError:
            winner = _verify_exact_catalog_winner(
                parent_fd,
                final_name,
                output_root,
                expected_index_bytes=index_bytes,
                expected_records_bytes=records_bytes,
                expected_records_sha256=records_sha256,
                source_lock_sha256=source_lock.sha256,
                wikidata_view_sha256=wikidata_view_sha256,
                record_count=record_count,
                target_count=target_count,
                lanes=lanes,
            )
            _quarantine_stage(
                parent_fd,
                stage_name,
                stage_fd,
                stage_identity,
                quarantine_fd,
            )
            stage_name = ""
            return winner
        stage_name = ""
        fsync_directory(parent_fd)
        final_named = entry_lstat(parent_fd, final_name)
        if _directory_identity(final_named) != stage_identity:
            raise ValueError("published catalog identity drift")
        return _catalog_result(
            output_root,
            source_lock.sha256,
            wikidata_view_sha256,
            records_sha256,
            record_count,
            target_count,
            lanes,
        )
    except BaseException as build_error:
        if records_fd >= 0:
            os.close(records_fd)
            records_fd = -1
        if spool is not None:
            spool.abort_connection()
        if (
            parent_fd >= 0
            and quarantine_fd >= 0
            and stage_fd >= 0
            and stage_name
            and stage_identity is not None
        ):
            try:
                _quarantine_stage(
                    parent_fd,
                    stage_name,
                    stage_fd,
                    stage_identity,
                    quarantine_fd,
                )
                stage_name = ""
            except BaseException as quarantine_error:
                raise ValueError(
                    "catalog build failed and exact-inode stage quarantine failed"
                ) from build_error
        raise
    finally:
        if records_fd >= 0:
            os.close(records_fd)
        if spool is not None:
            spool.abort_connection()
            spool.close_descriptor()
        if stage_fd >= 0:
            os.close(stage_fd)
        if quarantine_fd >= 0:
            os.close(quarantine_fd)
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
    "WikidataGraphCatalogSource",
    "build_input_catalog",
    "catalog_record_id",
    "records_from_jsonl",
)
