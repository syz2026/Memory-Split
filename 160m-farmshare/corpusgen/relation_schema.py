from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from corpusgen.relation_codec import RelationCodec
from corpusgen.wikidata5m import (
    AliasCatalog,
    canonicalize_aliases,
    iter_triples,
    parse_pid,
)


MIN_SUPPORT = 5_000
MIN_FUNCTIONALITY_NUMERATOR = 95
MIN_FUNCTIONALITY_DENOMINATOR = 100
DEFAULT_ENTITY_RELATION_COUNT = 32

LITERAL_RELATIONS = (
    ("SYN_L0", ("birth date", "date of birth"), "date"),
    ("SYN_L1", ("founding date", "date founded"), "date"),
    ("SYN_L2", ("population", "number of residents"), "quantity"),
    ("SYN_L3", ("duration", "length of time"), "quantity"),
    ("SYN_L4", ("category alpha", "first category"), "category"),
    ("SYN_L5", ("category beta", "second category"), "category"),
    ("SYN_L6", ("reference code", "identifier code"), "string"),
    ("SYN_L7", ("short label", "display label"), "string"),
)

_LITERAL_ID_RE = re.compile(r"SYN_L[0-7]")
_TARGET_KINDS = {"entity", "date", "quantity", "category", "string"}


class InstrumentError(RuntimeError):
    """Raised when frozen schema-selection requirements cannot be met."""


def _pid_key(relation_id: str) -> tuple[int, str]:
    parse_pid(relation_id)
    return int(relation_id[1:]), relation_id


@dataclass(frozen=True)
class RelationStats:
    relation_id: str
    support: int
    distinct_subjects: int
    distinct_objects: int
    entity_count: int
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        parse_pid(self.relation_id)
        if (
            isinstance(self.support, bool)
            or not isinstance(self.support, int)
            or self.support < 0
        ):
            raise ValueError("support must be a nonnegative integer")
        if (
            isinstance(self.distinct_subjects, bool)
            or not isinstance(self.distinct_subjects, int)
            or self.distinct_subjects < 0
            or self.distinct_subjects > self.support
        ):
            raise ValueError(
                "distinct_subjects must be between zero and support"
            )
        if (
            isinstance(self.distinct_objects, bool)
            or not isinstance(self.distinct_objects, int)
            or self.distinct_objects < 0
            or self.distinct_objects > self.support
        ):
            raise ValueError(
                "distinct_objects must be between zero and support"
            )
        if (
            isinstance(self.entity_count, bool)
            or not isinstance(self.entity_count, int)
            or self.entity_count < self.distinct_subjects
            or self.entity_count < self.distinct_objects
        ):
            raise ValueError(
                "entity_count must cover all distinct subjects and objects"
            )
        object.__setattr__(self, "aliases", tuple(self.aliases))

    @property
    def functionality(self) -> float:
        if self.support == 0:
            return 0.0
        return self.distinct_subjects / self.support

    @property
    def subject_coverage(self) -> float:
        if self.entity_count == 0:
            return 0.0
        return self.distinct_subjects / self.entity_count

    @property
    def target_pool_ratio(self) -> float:
        if self.distinct_subjects == 0:
            return 0.0
        return self.distinct_objects / self.distinct_subjects


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    aliases: tuple[str, ...]
    target_kind: str
    support: int | None = None
    distinct_subjects: int | None = None
    distinct_objects: int | None = None
    entity_count: int | None = None

    def __post_init__(self) -> None:
        if not (
            re.fullmatch(r"P[0-9]+", self.relation_id)
            or _LITERAL_ID_RE.fullmatch(self.relation_id)
        ):
            raise ValueError(f"invalid relation ID: {self.relation_id!r}")
        aliases = tuple(self.aliases)
        if not aliases or any(
            not isinstance(alias, str) or not alias for alias in aliases
        ):
            raise ValueError("relation specs require nonempty aliases")
        object.__setattr__(self, "aliases", aliases)
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError(f"invalid target kind: {self.target_kind!r}")
        statistic_values = (
            self.support,
            self.distinct_subjects,
            self.distinct_objects,
            self.entity_count,
        )
        if _LITERAL_ID_RE.fullmatch(self.relation_id) and any(
            value is not None for value in statistic_values
        ):
            raise ValueError("literal relations must not carry statistics")
        if any(value is None for value in statistic_values) and not all(
            value is None for value in statistic_values
        ):
            raise ValueError(
                "relation statistics must all be present or all be absent"
            )
        if self.support is not None:
            if (
                isinstance(self.support, bool)
                or not isinstance(self.support, int)
                or self.support < 0
            ):
                raise ValueError("support must be a nonnegative integer")
            if (
                isinstance(self.distinct_subjects, bool)
                or not isinstance(self.distinct_subjects, int)
                or self.distinct_subjects < 0
                or self.distinct_subjects > self.support
            ):
                raise ValueError(
                    "distinct_subjects must be between zero and support"
                )
            if (
                isinstance(self.distinct_objects, bool)
                or not isinstance(self.distinct_objects, int)
                or self.distinct_objects < 0
                or self.distinct_objects > self.support
            ):
                raise ValueError(
                    "distinct_objects must be between zero and support"
                )
            if (
                isinstance(self.entity_count, bool)
                or not isinstance(self.entity_count, int)
                or self.entity_count < self.distinct_subjects
                or self.entity_count < self.distinct_objects
            ):
                raise ValueError(
                    "entity_count must cover all distinct subjects and objects"
                )

    @property
    def functionality(self) -> float | None:
        if self.support is None:
            return None
        if self.support == 0:
            return 0.0
        assert self.distinct_subjects is not None
        return self.distinct_subjects / self.support

    @property
    def subject_coverage(self) -> float | None:
        if self.entity_count is None:
            return None
        if self.entity_count == 0:
            return 0.0
        assert self.distinct_subjects is not None
        return self.distinct_subjects / self.entity_count

    @property
    def target_pool_ratio(self) -> float | None:
        if self.distinct_subjects is None:
            return None
        if self.distinct_subjects == 0:
            return 0.0
        assert self.distinct_objects is not None
        return self.distinct_objects / self.distinct_subjects

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "aliases": list(self.aliases),
            "target_kind": self.target_kind,
            "support": self.support,
            "distinct_subjects": self.distinct_subjects,
            "distinct_objects": self.distinct_objects,
            "entity_count": self.entity_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RelationSpec:
        expected = {
            "relation_id",
            "aliases",
            "target_kind",
            "support",
            "distinct_subjects",
            "distinct_objects",
            "entity_count",
        }
        if set(value) != expected:
            raise ValueError("invalid relation spec fields")
        aliases = value["aliases"]
        if not isinstance(aliases, list):
            raise ValueError("relation aliases must be a list")
        return cls(
            relation_id=value["relation_id"],
            aliases=tuple(aliases),
            target_kind=value["target_kind"],
            support=value["support"],
            distinct_subjects=value["distinct_subjects"],
            distinct_objects=value["distinct_objects"],
            entity_count=value["entity_count"],
        )


@dataclass(frozen=True)
class RelationSchema:
    catalog: tuple[RelationSpec, ...]
    path_relation_ids: tuple[str, ...]
    _ambiguous_items: tuple[tuple[str, tuple[str, ...]], ...]
    _entity_count: int | None

    def __init__(
        self,
        catalog: Sequence[RelationSpec],
        path_relation_ids: Sequence[str],
        ambiguous_normalized: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        frozen_catalog = tuple(catalog)
        frozen_path_ids = tuple(path_relation_ids)
        catalog_ids = tuple(spec.relation_id for spec in frozen_catalog)
        if not frozen_catalog or len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("relation schema catalog must be nonempty and unique")
        if len(frozen_path_ids) != len(set(frozen_path_ids)):
            raise ValueError("path relation IDs must be unique")
        unknown = set(frozen_path_ids) - set(catalog_ids)
        if unknown:
            raise ValueError(
                f"path relations missing from catalog: {sorted(unknown)!r}"
            )
        entity_counts = {
            spec.entity_count
            for spec in frozen_catalog
            if spec.target_kind == "entity" and spec.entity_count is not None
        }
        if len(entity_counts) > 1:
            raise ValueError(
                "entity relation specs disagree on global entity count"
            )
        raw_ambiguous = ambiguous_normalized or {}
        ambiguous_items = tuple(
            (
                normalized,
                tuple(sorted(relation_ids, key=_pid_key)),
            )
            for normalized, relation_ids in sorted(raw_ambiguous.items())
        )
        object.__setattr__(self, "catalog", frozen_catalog)
        object.__setattr__(self, "path_relation_ids", frozen_path_ids)
        object.__setattr__(self, "_ambiguous_items", ambiguous_items)
        object.__setattr__(
            self,
            "_entity_count",
            next(iter(entity_counts), None),
        )

    @property
    def codec_catalog(self) -> tuple[str, ...]:
        return tuple(spec.relation_id for spec in self.catalog)

    @property
    def relations(self) -> tuple[RelationSpec, ...]:
        return self.catalog

    @property
    def entity_count(self) -> int | None:
        return self._entity_count

    @property
    def path_relations(self) -> tuple[RelationSpec, ...]:
        by_id = {spec.relation_id: spec for spec in self.catalog}
        return tuple(by_id[relation_id] for relation_id in self.path_relation_ids)

    @property
    def ambiguous_normalized(self) -> dict[str, tuple[str, ...]]:
        return dict(self._ambiguous_items)

    @property
    def codec(self) -> RelationCodec:
        return RelationCodec(self.codec_catalog)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "entity_count": self.entity_count,
            "codec_catalog": list(self.codec_catalog),
            "codec_sha256": self.codec.sha256(),
            "catalog": [spec.to_dict() for spec in self.catalog],
            "path_relation_ids": list(self.path_relation_ids),
            "ambiguous_normalized": {
                key: list(relation_ids)
                for key, relation_ids in self._ambiguous_items
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RelationSchema:
        if "version" not in value:
            raise ValueError("invalid relation schema fields")
        version = value["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != 2
        ):
            raise ValueError(
                f"unsupported relation schema version: {version!r}"
            )
        expected = {
            "version",
            "entity_count",
            "codec_catalog",
            "codec_sha256",
            "catalog",
            "path_relation_ids",
            "ambiguous_normalized",
        }
        if set(value) != expected:
            raise ValueError("invalid relation schema fields")
        raw_catalog = value["catalog"]
        raw_codec_catalog = value["codec_catalog"]
        raw_path_ids = value["path_relation_ids"]
        raw_ambiguous = value["ambiguous_normalized"]
        if not isinstance(raw_catalog, list):
            raise ValueError("relation schema catalog must be a list")
        if not isinstance(raw_codec_catalog, list):
            raise ValueError("codec catalog must be a list")
        if not isinstance(raw_path_ids, list):
            raise ValueError("path relation IDs must be a list")
        if not isinstance(raw_ambiguous, Mapping):
            raise ValueError("ambiguous_normalized must be an object")
        schema = cls(
            tuple(RelationSpec.from_dict(item) for item in raw_catalog),
            tuple(raw_path_ids),
            raw_ambiguous,
        )
        if schema.entity_count != value["entity_count"]:
            raise ValueError(
                "global entity count does not match relation catalog"
            )
        if list(schema.codec_catalog) != raw_codec_catalog:
            raise ValueError("codec catalog does not match relation catalog")
        if schema.codec.sha256() != value["codec_sha256"]:
            raise ValueError("codec catalog SHA-256 mismatch")
        return schema

    @classmethod
    def from_path(cls, path: str | Path) -> RelationSchema:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid relation schema JSON: {source}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError("relation schema must be a JSON object")
        return cls.from_dict(value)

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            + b"\n"
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.canonical_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def select_entity_relations(
    stats: Sequence[RelationStats],
    *,
    count: int = DEFAULT_ENTITY_RELATION_COUNT,
) -> tuple[RelationStats, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("relation count must be a positive integer")
    survivors = [
        item
        for item in stats
        if item.aliases
        and item.support >= MIN_SUPPORT
        and item.distinct_subjects * MIN_FUNCTIONALITY_DENOMINATOR
        >= item.support * MIN_FUNCTIONALITY_NUMERATOR
    ]
    survivors.sort(
        key=lambda item: (
            -item.support,
            -Fraction(item.distinct_subjects, item.support),
            _pid_key(item.relation_id),
        )
    )
    if len(survivors) < count:
        raise InstrumentError(
            f"fewer than {count} entity relations survive frozen thresholds"
        )
    return tuple(survivors[:count])


def _create_statistics_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE relation_counts (
          relation TEXT PRIMARY KEY,
          triples INTEGER NOT NULL
        );
        CREATE TABLE relation_subjects (
          relation TEXT NOT NULL,
          subject INTEGER NOT NULL,
          PRIMARY KEY (relation, subject)
        ) WITHOUT ROWID;
        CREATE TABLE relation_objects (
          relation TEXT NOT NULL,
          object INTEGER NOT NULL,
          PRIMARY KEY (relation, object)
        ) WITHOUT ROWID;
        CREATE TABLE entities (
          entity INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        """
    )


def _populate_statistics(
    connection: sqlite3.Connection,
    transductive_train: Path,
) -> None:
    count_rows: list[tuple[str]] = []
    subject_rows: list[tuple[str, int]] = []
    object_rows: list[tuple[str, int]] = []
    entity_rows: list[tuple[int]] = []

    def flush() -> None:
        connection.executemany(
            """
            INSERT INTO relation_counts (relation, triples) VALUES (?, 1)
            ON CONFLICT(relation) DO UPDATE SET triples = triples + 1
            """,
            count_rows,
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO relation_subjects (relation, subject)
            VALUES (?, ?)
            """,
            subject_rows,
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO relation_objects (relation, object)
            VALUES (?, ?)
            """,
            object_rows,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO entities (entity) VALUES (?)",
            entity_rows,
        )
        count_rows.clear()
        subject_rows.clear()
        object_rows.clear()
        entity_rows.clear()

    with connection:
        for triple in iter_triples(transductive_train):
            count_rows.append((triple.relation,))
            subject_rows.append((triple.relation, triple.subject))
            object_rows.append((triple.relation, triple.object))
            entity_rows.extend(((triple.subject,), (triple.object,)))
            if len(count_rows) >= 10_000:
                flush()
        if count_rows:
            flush()


def _read_statistics(
    connection: sqlite3.Connection,
    aliases: AliasCatalog,
) -> tuple[RelationStats, ...]:
    entity_count = connection.execute(
        "SELECT COUNT(*) FROM entities"
    ).fetchone()[0]
    rows = connection.execute(
        """
        SELECT
          counts.relation,
          counts.triples,
          (
            SELECT COUNT(*)
            FROM relation_subjects AS subjects
            WHERE subjects.relation = counts.relation
          ),
          (
            SELECT COUNT(*)
            FROM relation_objects AS objects
            WHERE objects.relation = counts.relation
          )
        FROM relation_counts AS counts
        """
    )
    stats = [
        RelationStats(
            relation_id=relation_id,
            support=support,
            distinct_subjects=distinct_subjects,
            distinct_objects=distinct_objects,
            entity_count=entity_count,
            aliases=aliases.get(relation_id, ()),
        )
        for relation_id, support, distinct_subjects, distinct_objects in rows
    ]
    stats.sort(key=lambda item: _pid_key(item.relation_id))
    return tuple(stats)


def compute_relation_stats(
    transductive_train: str | Path,
    aliases: Mapping[str, Sequence[str]] | AliasCatalog,
    *,
    work_root: str | Path | None = None,
) -> tuple[RelationStats, ...]:
    training_path = Path(transductive_train)
    if training_path.name != "wikidata5m_transductive_train.txt":
        raise ValueError(
            "relation statistics require wikidata5m transductive training triples"
        )
    alias_catalog = (
        aliases
        if isinstance(aliases, AliasCatalog)
        else canonicalize_aliases(aliases)
    )
    with tempfile.TemporaryDirectory(
        prefix="wikidata5m-relations-",
        dir=None if work_root is None else Path(work_root),
    ) as directory:
        database = Path(directory) / "relation-stats.sqlite3"
        connection = sqlite3.connect(database)
        try:
            _create_statistics_tables(connection)
            _populate_statistics(connection, training_path)
            return _read_statistics(connection, alias_catalog)
        finally:
            connection.close()


def build_relation_schema(
    transductive_train: str | Path,
    aliases: Mapping[str, Sequence[str]] | AliasCatalog,
) -> RelationSchema:
    alias_catalog = (
        aliases
        if isinstance(aliases, AliasCatalog)
        else canonicalize_aliases(aliases)
    )
    stats = compute_relation_stats(
        transductive_train,
        alias_catalog,
    )
    selected = select_entity_relations(
        stats,
        count=DEFAULT_ENTITY_RELATION_COUNT,
    )
    stats_by_id = {item.relation_id: item for item in stats}
    entity_counts = {item.entity_count for item in stats}
    if len(entity_counts) > 1:
        raise ValueError("relation statistics disagree on global entity count")
    entity_count = next(iter(entity_counts), 0)

    entity_specs: list[RelationSpec] = []
    for relation_id in sorted(alias_catalog, key=_pid_key):
        relation_aliases = alias_catalog[relation_id]
        if not relation_aliases:
            continue
        item = stats_by_id.get(relation_id)
        entity_specs.append(
            RelationSpec(
                relation_id=relation_id,
                aliases=relation_aliases,
                target_kind="entity",
                support=0 if item is None else item.support,
                distinct_subjects=(
                    0 if item is None else item.distinct_subjects
                ),
                distinct_objects=(
                    0 if item is None else item.distinct_objects
                ),
                entity_count=entity_count,
            )
        )

    literal_specs = tuple(
        RelationSpec(relation_id, relation_aliases, target_kind)
        for relation_id, relation_aliases, target_kind in LITERAL_RELATIONS
    )
    return RelationSchema(
        catalog=tuple(entity_specs) + literal_specs,
        path_relation_ids=tuple(
            item.relation_id for item in selected
        )
        + tuple(item.relation_id for item in literal_specs),
        ambiguous_normalized=alias_catalog.ambiguous_normalized,
    )
