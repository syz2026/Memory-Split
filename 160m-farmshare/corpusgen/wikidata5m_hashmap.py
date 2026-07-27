from __future__ import annotations

import hashlib
import heapq
import json
import os
import platform
import sqlite3
import stat
import sys
import tempfile
import unicodedata
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from corpusgen.relation_schema import (
    MIN_FUNCTIONALITY_DENOMINATOR,
    MIN_FUNCTIONALITY_NUMERATOR,
    MIN_SUPPORT,
    InstrumentError,
    RelationStats,
    compute_relation_stats,
    select_entity_relations,
)
from corpusgen.wikidata5m import (
    WikidataLock,
    iter_triples,
    normalize_alias,
    parse_pid,
    parse_qid,
    read_aliases,
    safe_extract_archives,
)


ALGORITHM_VERSION = "wikidata5m-hashmap-v1"
ARCHIVE_ROOT = "memorysplit-wikidata5m-real-hashmap-3000"
KEY_COUNT = 3_000
RELATION_COUNT = 32
ROOT = Path(__file__).resolve().parents[1]
HASHMAP_LOCK_PATH = ROOT / "sources" / "wikidata5m_hashmap.lock.json"
CC0_PATH = ROOT / "sources" / "Wikidata-CC0-1.0.txt"
EXPECTED_HASHMAP_LOCK = {
    "repo_id": "intfloat/wikidata5m",
    "repo_type": "dataset",
    "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
    "files": {
        "wikidata5m_alias.tar.gz": {
            "bytes": 197449751,
            "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8",
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": 168258214,
            "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597",
        },
    },
}


class HashmapBuildError(RuntimeError):
    """Raised when frozen hashmap requirements cannot be satisfied."""


def normalize_display_alias(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("alias must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def first_display_alias(raw_aliases: Sequence[str]) -> str:
    if not raw_aliases:
        raise ValueError("display aliases must be nonempty")
    display = normalize_display_alias(raw_aliases[0])
    if not display:
        raise ValueError("display alias must be nonempty")
    return display


def iter_strict_alias_rows(
    path: str | Path,
    prefix: str,
) -> Iterator[tuple[int, int | str, str]]:
    if prefix not in {"P", "Q"}:
        raise ValueError("alias prefix must be P or Q")
    parse_id = parse_pid if prefix == "P" else parse_qid
    source = Path(path)

    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            row = line.rstrip("\r\n")
            if not row:
                continue
            fields = row.split("\t")
            if len(fields) < 2:
                raise ValueError(
                    f"{source.name}:{line_number}: "
                    "expected a canonical ID and at least one alias"
                )
            canonical_id = fields[0]
            try:
                parsed_id = parse_id(canonical_id)
            except ValueError as exc:
                raise ValueError(
                    f"{source.name}:{line_number}: {exc}"
                ) from exc

            first_seen: dict[str, str] = {}
            for raw_alias in fields[1:]:
                normalized = normalize_alias(raw_alias)
                if not normalized:
                    raise ValueError(
                        f"{source.name}:{line_number}: empty alias"
                    )
                first_seen.setdefault(normalized, raw_alias)
            yield line_number, parsed_id, first_display_alias(tuple(first_seen.values()))


@dataclass(frozen=True)
class SelectedRelation:
    rank: int
    relation_id: str
    label: str
    support: int
    distinct_subjects: int
    distinct_objects: int
    entity_count: int
    quota: int


@dataclass(frozen=True)
class RelationSelection:
    relations: tuple[SelectedRelation, ...]
    filter_counts: dict[str, int]


def canonical_address(subject: int, relation_id: str) -> str:
    if isinstance(subject, bool) or not isinstance(subject, int) or subject < 0:
        raise ValueError("subject must be a nonnegative QID number")
    parse_pid(relation_id)
    return f"Q{subject}\t{relation_id}"


def address_sort_key(subject: int, relation_id: str) -> tuple[bytes, int, int]:
    encoded = canonical_address(subject, relation_id).encode("utf-8")
    return hashlib.sha256(encoded).digest(), subject, int(relation_id[1:])


def relation_quota(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 32:
        raise ValueError("relation rank must be between 1 and 32")
    return 94 if rank <= 24 else 93


def _meets_functionality(item: RelationStats) -> bool:
    return (
        item.distinct_subjects * MIN_FUNCTIONALITY_DENOMINATOR
        >= item.support * MIN_FUNCTIONALITY_NUMERATOR
    )


def _relation_label(raw_aliases: Sequence[str]) -> str:
    return first_display_alias(raw_aliases)


def _filter_counts(
    stats: Sequence[RelationStats],
    selected: Sequence[RelationStats],
) -> dict[str, int]:
    selected_ids = {item.relation_id for item in selected}
    counts = {
        "missing_relation_alias": 0,
        "below_min_support": 0,
        "below_min_functionality": 0,
        "survived_not_selected": 0,
        "selected": len(selected),
    }
    for item in stats:
        if not item.aliases:
            counts["missing_relation_alias"] += 1
        elif item.support < MIN_SUPPORT:
            counts["below_min_support"] += 1
        elif not _meets_functionality(item):
            counts["below_min_functionality"] += 1
        elif item.relation_id not in selected_ids:
            counts["survived_not_selected"] += 1
    return counts


def select_hashmap_relations(
    train_path: str | Path,
    relation_alias_path: str | Path,
    *,
    work_root: str | Path,
) -> RelationSelection:
    relation_aliases = read_aliases(relation_alias_path, "P")
    stats = compute_relation_stats(
        train_path,
        relation_aliases,
        work_root=work_root,
    )
    try:
        selected_stats = select_entity_relations(stats, count=RELATION_COUNT)
    except InstrumentError as exc:
        raise HashmapBuildError(str(exc)) from exc

    relations = tuple(
        SelectedRelation(
            rank=rank,
            relation_id=item.relation_id,
            label=_relation_label(relation_aliases[item.relation_id]),
            support=item.support,
            distinct_subjects=item.distinct_subjects,
            distinct_objects=item.distinct_objects,
            entity_count=item.entity_count,
            quota=relation_quota(rank),
        )
        for rank, item in enumerate(selected_stats, 1)
    )
    return RelationSelection(
        relations=relations,
        filter_counts=_filter_counts(stats, selected_stats),
    )


_EDGE_BATCH_SIZE = 10_000
_ALIAS_BATCH_SIZE = 500


@dataclass(frozen=True)
class HashmapValue:
    id: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True)
class HashmapRecord:
    subject_id: str
    subject_label: str
    relation_id: str
    relation_label: str
    values: tuple[HashmapValue, ...]

    @property
    def canonical_address(self) -> str:
        return f"{self.subject_id}\t{self.relation_id}"

    @property
    def display_key(self) -> str:
        return (
            f"{self.subject_label} [{self.subject_id}], "
            f"{self.relation_label} [{self.relation_id}]"
        )


@dataclass(frozen=True)
class RelationSummary:
    rank: int
    relation_id: str
    label: str
    support: int
    distinct_subjects: int
    distinct_objects: int
    functionality_numerator: int
    functionality_denominator: int
    quota: int
    eligible_keys: int
    eligible_edges: int
    emitted_keys: int
    emitted_edges: int


@dataclass(frozen=True)
class BuildCounts:
    source_triples: int
    selected_relation_triples: int
    unselected_relation_triples: int
    distinct_selected_edges: int
    duplicate_selected_edges: int
    selected_grouped_keys: int
    missing_subject_alias_keys: int
    missing_object_alias_keys: int
    eligible_keys: int
    eligible_edges: int
    unsampled_eligible_keys: int
    emitted_keys: int
    emitted_edges: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class HashmapDataset:
    records: tuple[HashmapRecord, ...]
    relations: tuple[RelationSummary, ...]
    counts: BuildCounts
    relation_filter_counts: dict[str, int]


@dataclass(frozen=True)
class ArchiveReport:
    archive_bytes: int
    archive_sha256: str
    edge_count: int
    key_count: int
    path: str
    relation_count: int


@dataclass(frozen=True)
class _WorstFirst:
    rank: tuple[bytes, int, int]
    subject: int

    def __lt__(self, other: "_WorstFirst") -> bool:
        return (self.rank, self.subject) > (other.rank, other.subject)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HashmapBuildError(message)


def _validate_selection(selection: RelationSelection) -> None:
    relations = selection.relations
    _require(
        len(relations) == RELATION_COUNT,
        f"hashmap builds require exactly {RELATION_COUNT} selected relations",
    )
    seen_relation_ids: set[str] = set()
    for expected_rank, item in enumerate(relations, 1):
        parse_pid(item.relation_id)
        _require(
            item.rank == expected_rank,
            "selected relations must be ordered by consecutive rank",
        )
        _require(
            item.relation_id not in seen_relation_ids,
            f"duplicate selected relation: {item.relation_id}",
        )
        seen_relation_ids.add(item.relation_id)
        _require(
            item.quota == relation_quota(item.rank),
            f"{item.relation_id} quota must match the frozen rank quota",
        )
        _require(
            item.support > 0,
            f"{item.relation_id} support must be positive",
        )
    _require(
        sum(item.quota for item in relations) == KEY_COUNT,
        f"relation quotas must sum to {KEY_COUNT}",
    )


def _configure_disposable_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA cache_size=-262144")


def _create_grouped_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE edges (
          relation INTEGER NOT NULL,
          subject INTEGER NOT NULL,
          object INTEGER NOT NULL,
          PRIMARY KEY (relation, subject, object)
        ) WITHOUT ROWID;

        CREATE TABLE needed_entities (
          entity INTEGER PRIMARY KEY
        ) WITHOUT ROWID;

        CREATE TABLE seen_entity_aliases (
          entity INTEGER PRIMARY KEY
        ) WITHOUT ROWID;

        CREATE TABLE entity_labels (
          entity INTEGER PRIMARY KEY,
          label TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _ingest_edges(
    connection: sqlite3.Connection,
    train_path: str | Path,
    selected_relation_ids: frozenset[str],
) -> tuple[int, int, int]:
    source_triples = 0
    selected_rows = 0
    unselected_rows = 0
    batch: list[tuple[int, int, int]] = []

    def flush() -> None:
        connection.executemany(
            "INSERT OR IGNORE INTO edges (relation, subject, object) "
            "VALUES (?, ?, ?)",
            batch,
        )
        batch.clear()

    with connection:
        for triple in iter_triples(train_path):
            source_triples += 1
            if triple.relation not in selected_relation_ids:
                unselected_rows += 1
                continue
            selected_rows += 1
            batch.append(
                (int(triple.relation[1:]), triple.subject, triple.object)
            )
            if len(batch) >= _EDGE_BATCH_SIZE:
                flush()
        if batch:
            flush()
        connection.execute(
            "INSERT OR IGNORE INTO needed_entities (entity) "
            "SELECT DISTINCT subject FROM edges"
        )
        connection.execute(
            "INSERT OR IGNORE INTO needed_entities (entity) "
            "SELECT DISTINCT object FROM edges"
        )
    return source_triples, selected_rows, unselected_rows


def _ingest_entity_labels(
    connection: sqlite3.Connection,
    entity_alias_path: str | Path,
) -> None:
    source_name = Path(entity_alias_path).name
    batch: list[tuple[int, int, str]] = []

    def flush() -> None:
        lines_by_entity: dict[int, int] = {}
        for line_number, entity, _ in batch:
            if entity in lines_by_entity:
                raise ValueError(
                    f"{source_name}:{line_number}: "
                    f"duplicate canonical ID: Q{entity}"
                )
            lines_by_entity[entity] = line_number
        placeholders = ",".join("?" for _ in batch)
        already_seen = {
            row[0]
            for row in connection.execute(
                "SELECT entity FROM seen_entity_aliases "
                f"WHERE entity IN ({placeholders})",
                [entity for _, entity, _ in batch],
            )
        }
        for line_number, entity, _ in batch:
            if entity in already_seen:
                raise ValueError(
                    f"{source_name}:{line_number}: "
                    f"duplicate canonical ID: Q{entity}"
                )
        connection.executemany(
            "INSERT INTO seen_entity_aliases (entity) VALUES (?)",
            [(entity,) for _, entity, _ in batch],
        )
        connection.executemany(
            "INSERT INTO entity_labels (entity, label) "
            "SELECT needed.entity, ? FROM needed_entities AS needed "
            "WHERE needed.entity = ?",
            [(label, entity) for _, entity, label in batch],
        )
        batch.clear()

    with connection:
        for line_number, entity, label in iter_strict_alias_rows(
            entity_alias_path,
            "Q",
        ):
            batch.append((line_number, int(entity), label))
            if len(batch) >= _ALIAS_BATCH_SIZE:
                flush()
        if batch:
            flush()


@dataclass
class _ScanTotals:
    grouped_keys: int = 0
    grouped_edges: int = 0
    missing_subject_keys: int = 0
    missing_object_keys: int = 0


def _scan_candidates(
    connection: sqlite3.Connection,
    selection: RelationSelection,
) -> tuple[
    _ScanTotals,
    dict[str, list[_WorstFirst]],
    dict[str, int],
    dict[str, int],
]:
    quotas = {item.relation_id: item.quota for item in selection.relations}
    heaps: dict[str, list[_WorstFirst]] = {
        relation_id: [] for relation_id in quotas
    }
    eligible_keys: dict[str, int] = {relation_id: 0 for relation_id in quotas}
    eligible_edges: dict[str, int] = {relation_id: 0 for relation_id in quotas}
    totals = _ScanTotals()

    rows = connection.execute(
        """
        SELECT
          edges.relation,
          edges.subject,
          COUNT(*),
          EXISTS(
            SELECT 1 FROM entity_labels AS subject_labels
            WHERE subject_labels.entity = edges.subject
          ),
          MIN(object_labels.entity IS NOT NULL)
        FROM edges
        LEFT JOIN entity_labels AS object_labels
          ON object_labels.entity = edges.object
        GROUP BY edges.relation, edges.subject
        ORDER BY edges.relation, edges.subject
        """
    )
    for relation_number, subject, value_count, has_subject, has_objects in rows:
        relation_id = f"P{relation_number}"
        totals.grouped_keys += 1
        totals.grouped_edges += value_count
        if not has_subject:
            totals.missing_subject_keys += 1
        elif not has_objects:
            totals.missing_object_keys += 1
        else:
            eligible_keys[relation_id] += 1
            eligible_edges[relation_id] += value_count
            candidate = _WorstFirst(
                rank=address_sort_key(subject, relation_id),
                subject=subject,
            )
            heap = heaps[relation_id]
            if len(heap) < quotas[relation_id]:
                heapq.heappush(heap, candidate)
            elif candidate.rank < heap[0].rank:
                heapq.heapreplace(heap, candidate)
    return totals, heaps, eligible_keys, eligible_edges


def _emit_records(
    connection: sqlite3.Connection,
    selection: RelationSelection,
    heaps: dict[str, list[_WorstFirst]],
) -> tuple[tuple[HashmapRecord, ...], dict[str, int], dict[str, int]]:
    labels_by_relation = {
        item.relation_id: item.label for item in selection.relations
    }
    retained: list[tuple[tuple[bytes, int, int], int, str]] = []
    for relation_id, heap in heaps.items():
        retained.extend(
            (candidate.rank, candidate.subject, relation_id)
            for candidate in heap
        )
    retained.sort()

    records: list[HashmapRecord] = []
    emitted_keys: dict[str, int] = {
        relation_id: 0 for relation_id in heaps
    }
    emitted_edges: dict[str, int] = {
        relation_id: 0 for relation_id in heaps
    }
    addresses: set[str] = set()
    display_keys: set[str] = set()
    for _, subject, relation_id in retained:
        subject_row = connection.execute(
            "SELECT label FROM entity_labels WHERE entity = ?",
            (subject,),
        ).fetchone()
        _require(
            subject_row is not None,
            f"missing subject label for Q{subject}",
        )
        value_rows = connection.execute(
            """
            SELECT edges.object, object_labels.label
            FROM edges
            LEFT JOIN entity_labels AS object_labels
              ON object_labels.entity = edges.object
            WHERE edges.relation = ? AND edges.subject = ?
            ORDER BY edges.object
            """,
            (int(relation_id[1:]), subject),
        ).fetchall()
        _require(
            bool(value_rows),
            f"empty value array for Q{subject} {relation_id}",
        )
        values = []
        for object_number, object_label in value_rows:
            _require(
                object_label is not None,
                f"missing object label for Q{object_number}",
            )
            values.append(
                HashmapValue(id=f"Q{object_number}", label=object_label)
            )
        record = HashmapRecord(
            subject_id=f"Q{subject}",
            subject_label=subject_row[0],
            relation_id=relation_id,
            relation_label=labels_by_relation[relation_id],
            values=tuple(values),
        )
        _require(
            record.canonical_address not in addresses,
            f"duplicate canonical address: {record.canonical_address!r}",
        )
        addresses.add(record.canonical_address)
        _require(
            record.display_key not in display_keys,
            f"duplicate display key: {record.display_key!r}",
        )
        display_keys.add(record.display_key)
        records.append(record)
        emitted_keys[relation_id] += 1
        emitted_edges[relation_id] += len(values)
    return tuple(records), emitted_keys, emitted_edges


def _reconcile_counts(
    counts: BuildCounts,
    selection: RelationSelection,
    *,
    scanned_edges: int,
    grouped_keys_in_database: int,
    emitted_keys_by_relation: dict[str, int],
) -> None:
    _require(
        counts.source_triples
        == counts.selected_relation_triples
        + counts.unselected_relation_triples,
        "source triples must equal selected plus unselected triples",
    )
    _require(
        counts.duplicate_selected_edges >= 0
        and counts.selected_relation_triples
        == counts.distinct_selected_edges + counts.duplicate_selected_edges,
        "selected triples must equal distinct plus duplicate edges",
    )
    _require(
        counts.distinct_selected_edges == scanned_edges,
        "grouped edge scan must cover every distinct selected edge",
    )
    _require(
        counts.selected_grouped_keys == grouped_keys_in_database,
        "grouped key scan must cover every grouped address",
    )
    _require(
        counts.selected_grouped_keys
        == counts.missing_subject_alias_keys
        + counts.missing_object_alias_keys
        + counts.eligible_keys,
        "grouped keys must split into missing-subject, missing-object, "
        "and eligible keys",
    )
    _require(
        counts.unsampled_eligible_keys >= 0
        and counts.eligible_keys
        == counts.unsampled_eligible_keys + counts.emitted_keys,
        "eligible keys must split into unsampled and emitted keys",
    )
    _require(
        counts.emitted_keys == KEY_COUNT,
        f"emitted keys must equal {KEY_COUNT}",
    )
    _require(
        counts.eligible_edges >= counts.eligible_keys,
        "eligible edges must cover every eligible key",
    )
    _require(
        counts.emitted_edges >= counts.emitted_keys,
        "emitted edges must cover every emitted key",
    )
    _require(
        counts.eligible_edges - counts.emitted_edges
        >= counts.unsampled_eligible_keys,
        "unsampled keys must retain at least one eligible edge each",
    )
    for item in selection.relations:
        _require(
            emitted_keys_by_relation[item.relation_id] == item.quota,
            f"{item.relation_id} must emit exactly {item.quota} keys",
        )


def build_hashmap_dataset(
    train_path: str | Path,
    entity_alias_path: str | Path,
    selection: RelationSelection,
    *,
    work_root: str | Path,
) -> HashmapDataset:
    _validate_selection(selection)
    selected_relation_ids = frozenset(
        item.relation_id for item in selection.relations
    )

    with tempfile.TemporaryDirectory(
        prefix="wikidata5m-grouped-",
        dir=work_root,
    ) as directory:
        database = Path(directory) / "grouped.sqlite3"
        connection = sqlite3.connect(database)
        try:
            _configure_disposable_database(connection)
            _create_grouped_tables(connection)
            source_triples, selected_rows, unselected_rows = _ingest_edges(
                connection,
                train_path,
                selected_relation_ids,
            )
            distinct_edges = connection.execute(
                "SELECT COUNT(*) FROM edges"
            ).fetchone()[0]
            grouped_keys_in_database = connection.execute(
                "SELECT COUNT(*) FROM "
                "(SELECT DISTINCT relation, subject FROM edges)"
            ).fetchone()[0]
            _ingest_entity_labels(connection, entity_alias_path)

            totals, heaps, eligible_keys, eligible_edges = _scan_candidates(
                connection,
                selection,
            )
            for item in selection.relations:
                eligible = eligible_keys[item.relation_id]
                if eligible < item.quota:
                    raise HashmapBuildError(
                        f"{item.relation_id} has {eligible} eligible keys "
                        f"but requires {item.quota}"
                    )
            records, emitted_keys, emitted_edges = _emit_records(
                connection,
                selection,
                heaps,
            )

            counts = BuildCounts(
                source_triples=source_triples,
                selected_relation_triples=selected_rows,
                unselected_relation_triples=unselected_rows,
                distinct_selected_edges=distinct_edges,
                duplicate_selected_edges=selected_rows - distinct_edges,
                selected_grouped_keys=totals.grouped_keys,
                missing_subject_alias_keys=totals.missing_subject_keys,
                missing_object_alias_keys=totals.missing_object_keys,
                eligible_keys=sum(eligible_keys.values()),
                eligible_edges=sum(eligible_edges.values()),
                unsampled_eligible_keys=(
                    sum(eligible_keys.values()) - len(records)
                ),
                emitted_keys=len(records),
                emitted_edges=sum(len(record.values) for record in records),
            )
            _reconcile_counts(
                counts,
                selection,
                scanned_edges=totals.grouped_edges,
                grouped_keys_in_database=grouped_keys_in_database,
                emitted_keys_by_relation=emitted_keys,
            )

            relations = tuple(
                RelationSummary(
                    rank=item.rank,
                    relation_id=item.relation_id,
                    label=item.label,
                    support=item.support,
                    distinct_subjects=item.distinct_subjects,
                    distinct_objects=item.distinct_objects,
                    functionality_numerator=Fraction(
                        item.distinct_subjects,
                        item.support,
                    ).numerator,
                    functionality_denominator=Fraction(
                        item.distinct_subjects,
                        item.support,
                    ).denominator,
                    quota=item.quota,
                    eligible_keys=eligible_keys[item.relation_id],
                    eligible_edges=eligible_edges[item.relation_id],
                    emitted_keys=emitted_keys[item.relation_id],
                    emitted_edges=emitted_edges[item.relation_id],
                )
                for item in selection.relations
            )
            dataset = HashmapDataset(
                records=records,
                relations=relations,
                counts=counts,
                relation_filter_counts=dict(selection.filter_counts),
            )
        finally:
            connection.close()

    _require(
        not Path(directory).exists(),
        "temporary grouped-edge state must be removed",
    )
    return dataset


_PACKAGE_MEMBERS = (
    "CITATION.bib",
    "LICENSES/Wikidata-CC0-1.0.txt",
    "README.md",
    "SHA256SUMS",
    "build_manifest.json",
    "hashmap.json",
    "hashmap.jsonl",
    "records.jsonl",
    "relation_summary.json",
    "source/wikidata5m.lock.json",
)
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = (stat.S_IFREG | 0o644) << 16
_FILTER_COUNT_FIELDS = {
    "missing_relation_alias",
    "below_min_support",
    "below_min_functionality",
    "survived_not_selected",
    "selected",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _selector_manifest() -> dict[str, Any]:
    return {
        "minimum_support": MIN_SUPPORT,
        "minimum_functionality_numerator": MIN_FUNCTIONALITY_NUMERATOR,
        "minimum_functionality_denominator": MIN_FUNCTIONALITY_DENOMINATOR,
        "relation_count": RELATION_COUNT,
        "ordering": [
            "support_descending",
            "exact_functionality_descending",
            "numeric_pid_ascending",
        ],
    }


def _values_to_dicts(values: Sequence[HashmapValue]) -> list[dict[str, str]]:
    return [value.to_dict() for value in values]


def _hashmap_rows(dataset: HashmapDataset) -> list[dict[str, Any]]:
    return [
        {
            "key": record.display_key,
            "values": _values_to_dicts(record.values),
        }
        for record in dataset.records
    ]


def _record_rows(dataset: HashmapDataset) -> list[dict[str, Any]]:
    return [
        {
            "display_key": record.display_key,
            "subject": {
                "id": record.subject_id,
                "label": record.subject_label,
            },
            "relation": {
                "id": record.relation_id,
                "label": record.relation_label,
            },
            "values": _values_to_dicts(record.values),
        }
        for record in dataset.records
    ]


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _citation_bytes(lock: WikidataLock) -> bytes:
    return (
        """@article{wang2021kepler,
  title = {KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation},
  author = {Wang, Xiaozhi and Gao, Tianyu and Zhu, Zhaocheng and Zhang, Zhengyan and Liu, Zhiyuan and Li, Juanzi and Tang, Jian},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {9},
  pages = {176--194},
  year = {2021},
  doi = {10.1162/tacl_a_00360}
}

@misc{intfloat2022wikidata5m,
  author = {{intfloat}},
  title = {Wikidata5M Dataset Snapshot},
  year = {2022},
  url = {https://huggingface.co/datasets/intfloat/wikidata5m},
  note = {Revision """
        + lock.revision
        + """}
}
"""
    ).encode("utf-8")


def _readme_bytes(dataset: HashmapDataset, lock: WikidataLock) -> bytes:
    example = _record_rows(dataset)[0]
    example_json = json.dumps(
        example,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    text = f"""# MemorySplit Wikidata5M Real Hashmap (3,000 keys)

This dataset was built from the pinned third-party Wikidata5M derivative hosted by intfloat on Hugging Face. It is not an official Wikimedia Foundation dump.

Wikidata5M aliases do not include language tags and do not designate a canonical label. Labels in this archive are deterministic display text only; QIDs and PIDs are authoritative.

The packaged structured data is distributed under the Wikidata CC0 1.0 public-domain dedication. See https://www.wikidata.org/wiki/Wikidata:Licensing.

## Frozen source and selection

The source revision is `{lock.revision}`. The selector keeps exactly 32 entity relations at support >= 5,000 and exact functionality >= 95/100, ordered by descending support, descending exact functionality, then ascending numeric PID. The highest-ranked 24 relations receive 94 keys each; the remaining 8 relations receive 93 keys each.

`relation_summary.json` retains raw `support` and `distinct_subjects` fields. Its `functionality_numerator` and `functionality_denominator` fields are their reduced exact fraction, not an unreduced copy of those raw counts.

In `build_manifest.json`, `missing_relation_alias` counts relations with no usable unambiguous selector alias after canonicalization; it does not mean only that the raw alias file lacked a row.

## Files

- `CITATION.bib`: citations for KEPLER and the pinned intfloat snapshot.
- `LICENSES/Wikidata-CC0-1.0.txt`: complete CC0 1.0 legal text.
- `README.md`: this archive contract.
- `SHA256SUMS`: SHA-256 for exactly the other nine members.
- `build_manifest.json`: algorithm, environment, source, selector, and count ledger.
- `hashmap.json`: one JSON object from display key to a nonempty value array.
- `hashmap.jsonl`: one object per line with `key` and `values`.
- `records.jsonl`: one structured object per line with `display_key`, `subject`, `relation`, and `values`.
- `relation_summary.json`: 32 ordered relation summaries and quota/count fields.
- `source/wikidata5m.lock.json`: canonical two-archive source lock.

## Schemas

A display key has the form `subject label [QID], relation label [PID]`. Every value is `{{"id":"Q...","label":"..."}}`, and every value container is an array even when it has one item. In `records.jsonl`, `subject` and `relation` are objects with `id` and `label`; QIDs and PIDs are the stable identifiers.

Example array-valued record:

```json
{example_json}
```

## Python loading

Load the JSON mapping:

```python
import json

with open("hashmap.json", encoding="utf-8") as stream:
    mapping = json.load(stream)
```

Stream either JSONL representation:

```python
import json

with open("records.jsonl", encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream]
```

## Reproducibility

`build_manifest.json` records `python`, `zlib_compile`, and `zlib_runtime`. Byte identity is promised only for the exact Python and zlib versions recorded in `build_manifest.json`.
"""
    return text.encode("utf-8")


def _sha256sums_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n"
        for name in sorted(files)
    ).encode("utf-8")


def render_package_files(
    dataset: HashmapDataset,
    selection: RelationSelection,
    lock: WikidataLock,
    cc0_text: str,
) -> dict[str, bytes]:
    if not isinstance(dataset, HashmapDataset):
        raise TypeError("dataset must be a HashmapDataset")
    if not isinstance(selection, RelationSelection):
        raise TypeError("selection must be a RelationSelection")
    if not isinstance(lock, WikidataLock):
        raise TypeError("lock must be a WikidataLock")
    if not isinstance(cc0_text, str):
        raise TypeError("cc0_text must be a string")
    _validate_selection(selection)
    _require(
        dataset.relation_filter_counts == selection.filter_counts,
        "dataset and selection relation filter counts must agree",
    )
    _require(
        len(dataset.records) == KEY_COUNT
        and dataset.counts.emitted_keys == KEY_COUNT,
        f"package requires exactly {KEY_COUNT} records",
    )
    _require(
        len(dataset.relations) == RELATION_COUNT,
        f"package requires exactly {RELATION_COUNT} relation summaries",
    )
    _require(
        tuple(item.relation_id for item in dataset.relations)
        == tuple(item.relation_id for item in selection.relations),
        "dataset and selection relation order must agree",
    )
    _require(
        dataset.counts.emitted_edges
        == sum(len(record.values) for record in dataset.records),
        "dataset emitted edge count must match records",
    )
    cc0_bytes = cc0_text.encode("utf-8")
    _require(
        b"\r" not in cc0_bytes and cc0_bytes.endswith(b"\n"),
        "CC0 text must be LF-only and newline terminated",
    )

    hashmap_rows = _hashmap_rows(dataset)
    mapping = {row["key"]: row["values"] for row in hashmap_rows}
    _require(
        len(mapping) == len(hashmap_rows),
        "display keys must be unique",
    )
    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "build_environment": {
            "python": platform.python_version(),
            "zlib_compile": zlib.ZLIB_VERSION,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        },
        "source": lock.to_dict(),
        "input_split": "wikidata5m_transductive_train.txt",
        "selector": _selector_manifest(),
        "relation_filter_counts": dict(selection.filter_counts),
        "build_counts": dataset.counts.to_dict(),
        "output_counts": {
            "keys": KEY_COUNT,
            "edges": dataset.counts.emitted_edges,
            "relations": RELATION_COUNT,
        },
    }
    files = {
        "CITATION.bib": _citation_bytes(lock),
        "LICENSES/Wikidata-CC0-1.0.txt": cc0_bytes,
        "README.md": _readme_bytes(dataset, lock),
        "build_manifest.json": canonical_json_bytes(manifest),
        "hashmap.json": canonical_json_bytes(mapping),
        "hashmap.jsonl": _canonical_jsonl(hashmap_rows),
        "records.jsonl": _canonical_jsonl(_record_rows(dataset)),
        "relation_summary.json": canonical_json_bytes(
            [asdict(item) for item in dataset.relations]
        ),
        "source/wikidata5m.lock.json": lock.canonical_bytes(),
    }
    files["SHA256SUMS"] = _sha256sums_bytes(files)
    _require(
        set(files) == set(_PACKAGE_MEMBERS),
        "renderer must produce exactly ten package members",
    )
    return files


def write_deterministic_zip(
    path: str | Path,
    files: Mapping[str, bytes],
) -> None:
    if set(files) != set(_PACKAGE_MEMBERS):
        raise ValueError("ZIP input must contain exactly the ten package members")
    for name, content in files.items():
        posix = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or posix.is_absolute()
            or ".." in posix.parts
            or posix.as_posix() != name
        ):
            raise ValueError(f"unsafe package member name: {name!r}")
        if not isinstance(content, bytes):
            raise TypeError(f"package member {name!r} must contain bytes")

    destination = Path(path)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for relative_path in sorted(files):
            info = zipfile.ZipInfo(
                filename=f"{ARCHIVE_ROOT}/{relative_path}",
                date_time=_ZIP_DATE_TIME,
            )
            info.create_system = 3
            info.external_attr = _ZIP_EXTERNAL_ATTR
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                info,
                files[relative_path],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _archive_require(condition: bool, message: str) -> None:
    if not condition:
        raise HashmapBuildError(message)


def _safe_member_relative(name: str) -> str:
    prefix = f"{ARCHIVE_ROOT}/"
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or not name.startswith(prefix)
    ):
        raise HashmapBuildError(f"unsafe ZIP member name: {name!r}")
    relative = name[len(prefix) :]
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or ".." in posix.parts
        or posix.as_posix() != relative
    ):
        raise HashmapBuildError(f"unsafe ZIP member name: {name!r}")
    return relative


def _load_canonical_json(files: Mapping[str, bytes], name: str) -> Any:
    try:
        value = json.loads(files[name].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HashmapBuildError(f"{name} is not valid UTF-8 JSON") from exc
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise HashmapBuildError(f"{name} is not canonical JSON") from exc
    _archive_require(
        files[name] == canonical,
        f"{name} is not canonical JSON",
    )
    return value


def _load_canonical_jsonl(
    files: Mapping[str, bytes],
    name: str,
) -> list[Any]:
    lines = files[name].splitlines(keepends=True)
    _archive_require(bool(lines), f"{name} must contain at least one row")
    rows = []
    for line_number, line in enumerate(lines, 1):
        _archive_require(
            line not in {b"\n", b""},
            f"{name}:{line_number} must not be blank",
        )
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HashmapBuildError(
                f"{name}:{line_number} is not valid UTF-8 JSON"
            ) from exc
        try:
            canonical = canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise HashmapBuildError(
                f"{name}:{line_number} is not canonical JSON"
            ) from exc
        _archive_require(
            line == canonical,
            f"{name}:{line_number} is not canonical JSON",
        )
        rows.append(value)
    return rows


def _validated_nonnegative_int(value: Any, context: str) -> int:
    _archive_require(
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0,
        f"{context} must be a nonnegative integer",
    )
    return value


def _validated_qid(value: Any, context: str) -> int:
    try:
        parsed = parse_qid(value)
    except (TypeError, ValueError) as exc:
        raise HashmapBuildError(f"invalid Q id in {context}: {value!r}") from exc
    _archive_require(
        value == f"Q{parsed}",
        f"invalid Q id in {context}: {value!r}",
    )
    return parsed


def _validated_pid(value: Any, context: str) -> str:
    try:
        parsed = parse_pid(value)
    except (TypeError, ValueError) as exc:
        raise HashmapBuildError(f"invalid P id in {context}: {value!r}") from exc
    _archive_require(
        value == f"P{int(parsed[1:])}",
        f"invalid P id in {context}: {value!r}",
    )
    return parsed


def _validated_label(value: Any, context: str) -> str:
    _archive_require(
        isinstance(value, str)
        and bool(value)
        and normalize_display_alias(value) == value,
        f"{context} must be normalized nonempty display text",
    )
    return value


def _validated_values(value: Any, context: str) -> list[dict[str, str]]:
    _archive_require(
        isinstance(value, list) and bool(value),
        f"{context} must be a nonempty array",
    )
    previous_id = -1
    for index, item in enumerate(value):
        _archive_require(
            isinstance(item, dict) and set(item) == {"id", "label"},
            f"{context}[{index}] must contain only id and label",
        )
        numeric_id = _validated_qid(item["id"], f"{context}[{index}]")
        _validated_label(item["label"], f"{context}[{index}].label")
        _archive_require(
            numeric_id > previous_id,
            f"{context} value QIDs must be unique and numerically ordered",
        )
        previous_id = numeric_id
    return value


def _validate_sha256sums(files: Mapping[str, bytes]) -> None:
    expected_names = sorted(set(_PACKAGE_MEMBERS) - {"SHA256SUMS"})
    lines = files["SHA256SUMS"].decode("utf-8").splitlines()
    _archive_require(
        len(lines) == len(expected_names),
        "SHA256SUMS checksum coverage must contain exactly nine members",
    )
    parsed_names = []
    for line in lines:
        parts = line.split("  ", 1)
        _archive_require(
            len(parts) == 2,
            "SHA256SUMS checksum line must use two spaces",
        )
        digest, name = parts
        _archive_require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "SHA256SUMS checksum digest must be lowercase SHA-256",
        )
        parsed_names.append(name)
        _archive_require(
            name in files and name != "SHA256SUMS",
            f"SHA256SUMS checksum references unknown member: {name!r}",
        )
        _archive_require(
            hashlib.sha256(files[name]).hexdigest() == digest,
            f"checksum drift for {name}",
        )
    _archive_require(
        parsed_names == expected_names,
        "SHA256SUMS checksum coverage must exactly match other members",
    )


def _validate_manifest(
    manifest: Any,
    lock: WikidataLock,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    expected_fields = {
        "algorithm_version",
        "build_environment",
        "source",
        "input_split",
        "selector",
        "relation_filter_counts",
        "build_counts",
        "output_counts",
    }
    _archive_require(
        isinstance(manifest, dict) and set(manifest) == expected_fields,
        "build manifest fields do not match the frozen contract",
    )
    _archive_require(
        manifest["algorithm_version"] == ALGORITHM_VERSION,
        "build manifest algorithm version drift",
    )
    _archive_require(
        manifest["build_environment"]
        == {
            "python": platform.python_version(),
            "zlib_compile": zlib.ZLIB_VERSION,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        },
        "build manifest Python/zlib environment drift",
    )
    _archive_require(
        manifest["source"] == lock.to_dict(),
        "build manifest source lock drift",
    )
    _archive_require(
        manifest["input_split"] == "wikidata5m_transductive_train.txt",
        "build manifest input split drift",
    )
    _archive_require(
        manifest["selector"] == _selector_manifest(),
        "build manifest selector drift",
    )

    filter_counts = manifest["relation_filter_counts"]
    _archive_require(
        isinstance(filter_counts, dict)
        and set(filter_counts) == _FILTER_COUNT_FIELDS,
        "relation filter count fields drift",
    )
    for name, count in filter_counts.items():
        _validated_nonnegative_int(count, f"relation_filter_counts.{name}")
    _archive_require(
        filter_counts["selected"] == RELATION_COUNT,
        "relation filter selected count drift",
    )

    build_counts = manifest["build_counts"]
    build_count_fields = {item.name for item in BuildCounts.__dataclass_fields__.values()}
    _archive_require(
        isinstance(build_counts, dict)
        and set(build_counts) == build_count_fields,
        "build count fields drift",
    )
    for name, count in build_counts.items():
        _validated_nonnegative_int(count, f"build_counts.{name}")
    _archive_require(
        build_counts["source_triples"]
        == build_counts["selected_relation_triples"]
        + build_counts["unselected_relation_triples"],
        "build source triple counts do not reconcile",
    )
    _archive_require(
        build_counts["selected_relation_triples"]
        == build_counts["distinct_selected_edges"]
        + build_counts["duplicate_selected_edges"],
        "build selected edge counts do not reconcile",
    )
    _archive_require(
        build_counts["selected_grouped_keys"]
        == build_counts["missing_subject_alias_keys"]
        + build_counts["missing_object_alias_keys"]
        + build_counts["eligible_keys"],
        "build grouped key counts do not reconcile",
    )
    _archive_require(
        build_counts["eligible_keys"]
        == build_counts["unsampled_eligible_keys"]
        + build_counts["emitted_keys"],
        "build eligible key counts do not reconcile",
    )

    output_counts = manifest["output_counts"]
    _archive_require(
        isinstance(output_counts, dict)
        and set(output_counts) == {"keys", "edges", "relations"},
        "output count fields drift",
    )
    for name, count in output_counts.items():
        _validated_nonnegative_int(count, f"output_counts.{name}")
    _archive_require(
        output_counts["keys"] == KEY_COUNT
        and output_counts["relations"] == RELATION_COUNT,
        "output key or relation count drift",
    )
    _archive_require(
        build_counts["emitted_keys"] == output_counts["keys"]
        and build_counts["emitted_edges"] == output_counts["edges"],
        "build and output counts disagree",
    )
    return filter_counts, build_counts, output_counts


def _validate_relation_summaries(
    value: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _archive_require(
        isinstance(value, list) and len(value) == RELATION_COUNT,
        f"relation_summary.json must contain exactly {RELATION_COUNT} rows",
    )
    expected_fields = set(RelationSummary.__dataclass_fields__)
    by_id: dict[str, dict[str, Any]] = {}
    for expected_rank, item in enumerate(value, 1):
        context = f"relation_summary.json[{expected_rank - 1}]"
        _archive_require(
            isinstance(item, dict) and set(item) == expected_fields,
            f"{context} fields drift",
        )
        rank = _validated_nonnegative_int(item["rank"], f"{context}.rank")
        _archive_require(rank == expected_rank, f"{context} rank drift")
        relation_id = _validated_pid(item["relation_id"], context)
        _archive_require(
            relation_id not in by_id,
            f"duplicate relation summary: {relation_id}",
        )
        _validated_label(item["label"], f"{context}.label")
        support = _validated_nonnegative_int(item["support"], f"{context}.support")
        distinct_subjects = _validated_nonnegative_int(
            item["distinct_subjects"],
            f"{context}.distinct_subjects",
        )
        distinct_objects = _validated_nonnegative_int(
            item["distinct_objects"],
            f"{context}.distinct_objects",
        )
        _archive_require(
            support >= MIN_SUPPORT
            and distinct_subjects <= support
            and distinct_objects <= support,
            f"{context} support/distinct counts violate selector bounds",
        )
        _archive_require(
            distinct_subjects * MIN_FUNCTIONALITY_DENOMINATOR
            >= support * MIN_FUNCTIONALITY_NUMERATOR,
            f"{context} functionality is below the frozen threshold",
        )
        numerator = _validated_nonnegative_int(
            item["functionality_numerator"],
            f"{context}.functionality_numerator",
        )
        denominator = _validated_nonnegative_int(
            item["functionality_denominator"],
            f"{context}.functionality_denominator",
        )
        _archive_require(
            denominator > 0,
            f"{context} functionality denominator must be positive",
        )
        fraction = Fraction(distinct_subjects, support)
        _archive_require(
            (numerator, denominator)
            == (fraction.numerator, fraction.denominator),
            f"{context} functionality must be the reduced exact fraction",
        )
        quota = _validated_nonnegative_int(item["quota"], f"{context}.quota")
        _archive_require(
            quota == relation_quota(expected_rank),
            f"{context} quota drift",
        )
        eligible_keys = _validated_nonnegative_int(
            item["eligible_keys"],
            f"{context}.eligible_keys",
        )
        eligible_edges = _validated_nonnegative_int(
            item["eligible_edges"],
            f"{context}.eligible_edges",
        )
        emitted_keys = _validated_nonnegative_int(
            item["emitted_keys"],
            f"{context}.emitted_keys",
        )
        emitted_edges = _validated_nonnegative_int(
            item["emitted_edges"],
            f"{context}.emitted_edges",
        )
        _archive_require(
            emitted_keys == quota
            and eligible_keys >= emitted_keys
            and eligible_edges >= eligible_keys
            and emitted_edges >= emitted_keys
            and eligible_edges >= emitted_edges,
            f"{context} quota and edge counts do not reconcile",
        )
        by_id[relation_id] = item
    _archive_require(
        sum(item["emitted_keys"] for item in value) == KEY_COUNT,
        "relation summary emitted keys do not sum to 3,000",
    )
    return value, by_id


def _validate_records(
    rows: list[Any],
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, int], dict[str, int]]:
    _archive_require(
        len(rows) == KEY_COUNT,
        f"records.jsonl must contain exactly {KEY_COUNT} rows",
    )
    mapping: dict[str, list[dict[str, str]]] = {}
    addresses: set[str] = set()
    keys_by_relation = {relation_id: 0 for relation_id in summaries}
    edges_by_relation = {relation_id: 0 for relation_id in summaries}
    for index, row in enumerate(rows):
        context = f"records.jsonl:{index + 1}"
        _archive_require(
            isinstance(row, dict)
            and set(row) == {"display_key", "subject", "relation", "values"},
            f"{context} fields drift",
        )
        subject = row["subject"]
        relation = row["relation"]
        _archive_require(
            isinstance(subject, dict) and set(subject) == {"id", "label"},
            f"{context} subject fields drift",
        )
        _archive_require(
            isinstance(relation, dict) and set(relation) == {"id", "label"},
            f"{context} relation fields drift",
        )
        subject_number = _validated_qid(subject["id"], f"{context}.subject")
        subject_label = _validated_label(
            subject["label"],
            f"{context}.subject.label",
        )
        relation_id = _validated_pid(relation["id"], f"{context}.relation")
        relation_label = _validated_label(
            relation["label"],
            f"{context}.relation.label",
        )
        _archive_require(
            relation_id in summaries
            and relation_label == summaries[relation_id]["label"],
            f"{context} relation does not match relation summary",
        )
        expected_key = (
            f"{subject_label} [Q{subject_number}], "
            f"{relation_label} [{relation_id}]"
        )
        _archive_require(
            row["display_key"] == expected_key,
            f"{context} display key does not match structured fields",
        )
        values = _validated_values(row["values"], f"{context}.values")
        address = f"Q{subject_number}\t{relation_id}"
        _archive_require(
            address not in addresses,
            f"duplicate canonical address in records: {address!r}",
        )
        _archive_require(
            expected_key not in mapping,
            f"duplicate display key in records: {expected_key!r}",
        )
        addresses.add(address)
        mapping[expected_key] = values
        keys_by_relation[relation_id] += 1
        edges_by_relation[relation_id] += len(values)
    for relation_id, summary in summaries.items():
        _archive_require(
            keys_by_relation[relation_id] == summary["emitted_keys"]
            and edges_by_relation[relation_id] == summary["emitted_edges"],
            f"{relation_id} records disagree with relation summary",
        )
    return mapping, keys_by_relation, edges_by_relation


def _validate_mapping_representations(
    hashmap: Any,
    hashmap_rows: list[Any],
    records_mapping: dict[str, list[dict[str, str]]],
    record_rows: list[Any],
) -> None:
    _archive_require(
        isinstance(hashmap, dict) and len(hashmap) == KEY_COUNT,
        f"hashmap.json must contain exactly {KEY_COUNT} keys",
    )
    for key, values in hashmap.items():
        _archive_require(
            isinstance(key, str) and bool(key),
            "hashmap.json keys must be nonempty strings",
        )
        _validated_values(values, f"hashmap.json[{key!r}]")

    jsonl_mapping: dict[str, list[dict[str, str]]] = {}
    for index, row in enumerate(hashmap_rows):
        context = f"hashmap.jsonl:{index + 1}"
        _archive_require(
            isinstance(row, dict) and set(row) == {"key", "values"},
            f"{context} fields drift",
        )
        key = row["key"]
        _archive_require(
            isinstance(key, str) and bool(key) and key not in jsonl_mapping,
            f"{context} key must be a unique nonempty string",
        )
        jsonl_mapping[key] = _validated_values(
            row["values"],
            f"{context}.values",
        )
    _archive_require(
        [row["key"] for row in hashmap_rows]
        == [row["display_key"] for row in record_rows],
        "JSONL mapping row order disagrees with records",
    )
    _archive_require(
        hashmap == jsonl_mapping == records_mapping,
        "mapping representations disagree",
    )


def _archive_size_and_sha256(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_archive_structure(
    path: str | Path,
    *,
    expected_dataset: HashmapDataset | None = None,
    lock: WikidataLock | None = None,
    cc0_text: str | None = None,
) -> ArchiveReport:
    archive_path = Path(path)
    _archive_require(
        archive_path.is_file() and not archive_path.is_symlink(),
        f"archive is not a regular file: {archive_path}",
    )
    if lock is None:
        lock = _load_hashmap_lock()
    if cc0_text is None:
        cc0_text = CC0_PATH.read_text(encoding="utf-8")
    expected_names = [
        f"{ARCHIVE_ROOT}/{name}" for name in sorted(_PACKAGE_MEMBERS)
    ]
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            relative_names = [
                _safe_member_relative(item.filename) for item in members
            ]
            _archive_require(
                len(relative_names) == len(set(relative_names)),
                "duplicate ZIP members are forbidden",
            )
            _archive_require(
                len(members) == len(_PACKAGE_MEMBERS),
                "archive must contain exactly ten unique members",
            )
            _archive_require(
                [item.filename for item in members] == expected_names,
                "ZIP members must be in lexical physical order under the required root",
            )
            for item in members:
                _archive_require(
                    not item.is_dir()
                    and item.date_time == _ZIP_DATE_TIME
                    and item.create_system == 3
                    and item.external_attr == _ZIP_EXTERNAL_ATTR
                    and item.compress_type == zipfile.ZIP_DEFLATED,
                    f"ZIP metadata drift for {item.filename}",
                )
            corrupt = archive.testzip()
            _archive_require(
                corrupt is None,
                f"ZIP integrity failure for {corrupt}",
            )
            files = {
                relative: archive.read(item)
                for relative, item in zip(relative_names, members, strict=True)
            }
    except HashmapBuildError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise HashmapBuildError(f"invalid ZIP archive: {archive_path}") from exc

    for name, content in files.items():
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HashmapBuildError(f"{name} is not UTF-8") from exc
        _archive_require(
            b"\r" not in content and content.endswith(b"\n"),
            f"{name} must be LF-only and trailing-newline text",
        )

    _validate_sha256sums(files)
    manifest = _load_canonical_json(files, "build_manifest.json")
    hashmap = _load_canonical_json(files, "hashmap.json")
    hashmap_rows = _load_canonical_jsonl(files, "hashmap.jsonl")
    record_rows = _load_canonical_jsonl(files, "records.jsonl")
    relation_value = _load_canonical_json(files, "relation_summary.json")
    lock_value = _load_canonical_json(files, "source/wikidata5m.lock.json")

    try:
        archived_lock = WikidataLock.from_dict(lock_value)
    except (TypeError, ValueError) as exc:
        raise HashmapBuildError("invalid archived source lock") from exc
    _archive_require(
        archived_lock.to_dict() == lock.to_dict()
        and files["source/wikidata5m.lock.json"] == lock.canonical_bytes(),
        "archived source lock drift",
    )
    _archive_require(
        files["LICENSES/Wikidata-CC0-1.0.txt"] == cc0_text.encode("utf-8"),
        "archived CC0 text drift",
    )
    _archive_require(
        files["CITATION.bib"] == _citation_bytes(lock),
        "citation metadata drift",
    )
    readme = files["README.md"].decode("utf-8")
    for statement in (
        "This dataset was built from the pinned third-party Wikidata5M derivative hosted by intfloat on Hugging Face. It is not an official Wikimedia Foundation dump.",
        "Wikidata5M aliases do not include language tags and do not designate a canonical label. Labels in this archive are deterministic display text only; QIDs and PIDs are authoritative.",
        "The packaged structured data is distributed under the Wikidata CC0 1.0 public-domain dedication. See https://www.wikidata.org/wiki/Wikidata:Licensing.",
        "`missing_relation_alias` counts relations with no usable unambiguous selector alias after canonicalization",
        "reduced exact fraction",
        "Byte identity is promised only for the exact Python and zlib versions recorded in `build_manifest.json`.",
    ):
        _archive_require(statement in readme, "README contract drift")
    for name in _PACKAGE_MEMBERS:
        _archive_require(f"`{name}`" in readme, "README file inventory drift")
    _archive_require(lock.revision in readme, "README source revision drift")

    filter_counts, build_counts, output_counts = _validate_manifest(
        manifest,
        lock,
    )
    summaries, summaries_by_id = _validate_relation_summaries(relation_value)
    records_mapping, _, _ = _validate_records(record_rows, summaries_by_id)
    _validate_mapping_representations(
        hashmap,
        hashmap_rows,
        records_mapping,
        record_rows,
    )
    edge_count = sum(len(values) for values in records_mapping.values())
    _archive_require(
        edge_count == output_counts["edges"]
        and sum(item["emitted_edges"] for item in summaries) == edge_count,
        "archive edge counts do not reconcile",
    )

    if expected_dataset is not None:
        expected_records = _record_rows(expected_dataset)
        expected_summaries = [asdict(item) for item in expected_dataset.relations]
        _archive_require(
            record_rows == expected_records
            and summaries == expected_summaries
            and build_counts == expected_dataset.counts.to_dict()
            and filter_counts == expected_dataset.relation_filter_counts,
            "source-edge disagreement with rebuilt Wikidata5M dataset",
        )

    archive_bytes, archive_sha256 = _archive_size_and_sha256(archive_path)
    return ArchiveReport(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        edge_count=edge_count,
        key_count=len(records_mapping),
        path=str(archive_path),
        relation_count=len(summaries),
    )


def _require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise HashmapBuildError(
            "Wikidata5M hashmap build and verification require Python 3.12"
        )


def _load_hashmap_lock() -> WikidataLock:
    try:
        lock = WikidataLock.from_path(HASHMAP_LOCK_PATH)
    except (OSError, TypeError, ValueError) as exc:
        raise HashmapBuildError("could not load the committed frozen lock") from exc
    if lock.to_dict() != EXPECTED_HASHMAP_LOCK:
        raise HashmapBuildError(
            "committed Wikidata5M hashmap lock differs from the frozen lock"
        )
    return lock


def _find_extracted_file(root: Path, name: str) -> Path:
    matches = sorted(
        path
        for path in root.rglob(name)
        if path.is_file() and not path.is_symlink()
    )
    _require(
        len(matches) == 1,
        f"locked sources must extract exactly one {name}",
    )
    return matches[0]


def _rebuild_from_sources(
    source_root: str | Path,
    lock: WikidataLock,
    work_directory: str | Path,
) -> tuple[HashmapDataset, RelationSelection]:
    work_path = Path(work_directory)
    extracted_root = work_path / "extracted"
    safe_extract_archives(
        source_root,
        extracted_root,
        lock=lock,
    )
    train_path = _find_extracted_file(
        extracted_root,
        "wikidata5m_transductive_train.txt",
    )
    relation_alias_path = _find_extracted_file(
        extracted_root,
        "wikidata5m_relation.txt",
    )
    entity_alias_path = _find_extracted_file(
        extracted_root,
        "wikidata5m_entity.txt",
    )
    selection = select_hashmap_relations(
        train_path,
        relation_alias_path,
        work_root=work_path,
    )
    dataset = build_hashmap_dataset(
        train_path,
        entity_alias_path,
        selection,
        work_root=work_path,
    )
    return dataset, selection


def _prepare_work_root(work_root: str | Path) -> Path:
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    _require(
        root.is_dir() and not root.is_symlink(),
        "work_root must be a regular directory",
    )
    return root


def _flush_and_fsync(path: Path) -> None:
    with path.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _new_sibling_temporary(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def build_wikidata5m_hashmap(
    source_root: str | Path,
    out: str | Path,
    *,
    work_root: str | Path,
) -> ArchiveReport:
    _require_python_312()
    lock = _load_hashmap_lock()
    cc0_text = CC0_PATH.read_text(encoding="utf-8")
    work_path = _prepare_work_root(work_root)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    build_directory: Path | None = None
    staged_report: ArchiveReport | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="wikidata5m-build-",
            dir=work_path,
        ) as directory:
            build_directory = Path(directory)
            dataset, selection = _rebuild_from_sources(
                source_root,
                lock,
                build_directory,
            )
            files = render_package_files(dataset, selection, lock, cc0_text)
            staged = _new_sibling_temporary(destination)
            write_deterministic_zip(staged, files)
            staged_report = _validate_archive_structure(
                staged,
                expected_dataset=dataset,
                lock=lock,
                cc0_text=cc0_text,
            )
            _flush_and_fsync(staged)
        _require(
            build_directory is not None and not build_directory.exists(),
            "temporary build state must be removed before publication",
        )
        assert staged is not None
        assert staged_report is not None
        os.replace(staged, destination)
        staged = None
        return ArchiveReport(
            archive_bytes=staged_report.archive_bytes,
            archive_sha256=staged_report.archive_sha256,
            edge_count=staged_report.edge_count,
            key_count=staged_report.key_count,
            path=str(destination),
            relation_count=staged_report.relation_count,
        )
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def _require_clean_pass_boundary(work_root: Path) -> None:
    stale = sorted(
        path.name
        for path in work_root.iterdir()
        if any(
            path.name.startswith(prefix)
            for prefix in (
                "wikidata5m-build-",
                "wikidata5m-grouped-",
                "wikidata5m-relations-",
            )
        )
    )
    _require(
        not stale,
        "fresh verification requires no prior build/group/stat state: "
        + ", ".join(stale),
    )


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(1024 * 1024)
            right_chunk = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def verify_wikidata5m_hashmap(
    archive: str | Path,
    source_root: str | Path,
    *,
    work_root: str | Path,
) -> ArchiveReport:
    _require_python_312()
    lock = _load_hashmap_lock()
    cc0_text = CC0_PATH.read_text(encoding="utf-8")
    work_path = _prepare_work_root(work_root)
    _require_clean_pass_boundary(work_path)
    candidate = Path(archive)
    verify_directory: Path | None = None
    candidate_report: ArchiveReport | None = None
    with tempfile.TemporaryDirectory(
        prefix="wikidata5m-verify-",
        dir=work_path,
    ) as directory:
        verify_directory = Path(directory)
        dataset, selection = _rebuild_from_sources(
            source_root,
            lock,
            verify_directory,
        )
        files = render_package_files(dataset, selection, lock, cc0_text)
        reference = verify_directory / "reference.zip"
        write_deterministic_zip(reference, files)
        candidate_report = _validate_archive_structure(
            candidate,
            expected_dataset=dataset,
            lock=lock,
            cc0_text=cc0_text,
        )
        _validate_archive_structure(
            reference,
            expected_dataset=dataset,
            lock=lock,
            cc0_text=cc0_text,
        )
        _require(
            _files_equal(candidate, reference),
            "archive bytes disagree with the fresh source-backed rebuild",
        )
    _require(
        verify_directory is not None and not verify_directory.exists(),
        "temporary verification state must be removed",
    )
    assert candidate_report is not None
    return candidate_report
