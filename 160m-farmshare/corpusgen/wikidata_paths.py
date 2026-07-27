"""Frozen, eval-only path artifacts from Wikidata5M inductive triples."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from corpusgen.graph_records import GraphAction, GraphRow, stable_fact_id
from corpusgen.relation_codec import RelationCodec
from corpusgen.relation_schema import RelationSchema, RelationSpec
from corpusgen.wikidata_path_replay import (
    audit_materialized_path_artifacts,
    replay_and_validate_path_twins,
)
from corpusgen.wikidata5m import (
    DEFAULT_WIKIDATA_LOCK_PATH,
    WikidataLock,
    iter_triples,
)
from corpusgen.wikidata_robustness_source import (
    verify_extract_and_bind_schema,
)
from corpusgen.world_splits import composition_hash
from organizer.packed_graph_store import PackedGraphStore


MAX_PATH_ROOTS = 10_000
MAX_HOPS = 6
FROZEN_EXCLUSION_REASONS = (
    "unselected_relation",
    "ambiguous_address",
    "missing_hop",
    "repeated_address",
    "no_counterfactual_alternative",
    "selection_bound",
)
ADDRESS_EXCLUSION_REASONS = (
    "ambiguous_address",
    "unselected_relation",
)
PATH_EXCLUSION_REASONS = (
    "ambiguous_address",
    "missing_hop",
    "repeated_address",
    "no_counterfactual_alternative",
    "selection_bound",
)
_ADDRESS_EXCLUSION_REASONS = ADDRESS_EXCLUSION_REASONS
_PATH_EXCLUSION_REASONS = PATH_EXCLUSION_REASONS
_SOURCE_CAPABILITIES = {
    "literals": False,
    "qualifiers": False,
    "ranks": False,
    "types": False,
}
_UINT63_MAX = (1 << 63) - 1


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _frozen_counts(
    values: Mapping[str, int],
    *,
    allowed: Sequence[str],
) -> Mapping[str, int]:
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown exclusion reasons: {sorted(unknown)!r}")
    output = {}
    for reason in allowed:
        count = values.get(reason, 0)
        output[reason] = _nonnegative_int(count, f"{reason} count")
    return MappingProxyType(output)


@dataclass(frozen=True)
class Survival:
    candidates: int
    surviving: int

    def __post_init__(self) -> None:
        candidates = _nonnegative_int(self.candidates, "candidate count")
        surviving = _nonnegative_int(self.surviving, "surviving count")
        if surviving > candidates:
            raise ValueError("surviving count exceeds candidates")

    @property
    def rate(self) -> float | None:
        return (
            None
            if self.candidates == 0
            else self.surviving / self.candidates
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "candidates": self.candidates,
            "surviving": self.surviving,
            "rate": self.rate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Survival:
        if set(value) != {"candidates", "surviving", "rate"}:
            raise ValueError("invalid survival fields")
        result = cls(value["candidates"], value["surviving"])
        if result.rate != value["rate"]:
            raise ValueError("survival rate does not match counts")
        return result


@dataclass(frozen=True)
class CandidateAccounting:
    candidates: int
    surviving: int
    exclusions: Mapping[str, int]
    allowed_reasons: tuple[str, ...] = field(
        default=FROZEN_EXCLUSION_REASONS,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        candidates = _nonnegative_int(self.candidates, "candidate count")
        surviving = _nonnegative_int(self.surviving, "surviving count")
        frozen = _frozen_counts(
            self.exclusions,
            allowed=self.allowed_reasons,
        )
        if candidates != surviving + sum(frozen.values()):
            raise ValueError(
                "candidate accounting must equal survivors plus exclusions"
            )
        object.__setattr__(self, "exclusions", frozen)

    @property
    def rate(self) -> float | None:
        return (
            None
            if self.candidates == 0
            else self.surviving / self.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "surviving": self.surviving,
            "rate": self.rate,
            "exclusions": dict(self.exclusions),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        allowed_reasons: tuple[str, ...],
    ) -> CandidateAccounting:
        if set(value) != {
            "candidates",
            "surviving",
            "rate",
            "exclusions",
        }:
            raise ValueError("invalid candidate-accounting fields")
        exclusions = value["exclusions"]
        if not isinstance(exclusions, Mapping):
            raise ValueError("candidate exclusions must be an object")
        result = cls(
            value["candidates"],
            value["surviving"],
            exclusions,
            allowed_reasons,
        )
        if result.rate != value["rate"]:
            raise ValueError("candidate survival rate does not match counts")
        return result


@dataclass(frozen=True)
class PathAccounting:
    candidates: int
    surviving: int
    exclusions: Mapping[str, int]
    per_hop: Mapping[str, CandidateAccounting]

    def __post_init__(self) -> None:
        frozen_exclusions = _frozen_counts(
            self.exclusions,
            allowed=_PATH_EXCLUSION_REASONS,
        )
        if self.candidates != self.surviving + sum(
            frozen_exclusions.values()
        ):
            raise ValueError(
                "path candidates must equal survivors plus exclusions"
            )
        expected_hops = tuple(str(value) for value in range(1, MAX_HOPS + 1))
        if tuple(self.per_hop) != expected_hops:
            raise ValueError("path accounting requires hops one through six")
        frozen_hops = {}
        for hop in expected_hops:
            accounting = self.per_hop[hop]
            if not isinstance(accounting, CandidateAccounting):
                raise TypeError("per-hop values must be CandidateAccounting")
            frozen_hops[hop] = accounting
        if self.candidates != sum(item.candidates for item in frozen_hops.values()):
            raise ValueError("per-hop candidates do not sum to path candidates")
        if self.surviving != sum(item.surviving for item in frozen_hops.values()):
            raise ValueError("per-hop survivors do not sum to path survivors")
        for reason in _PATH_EXCLUSION_REASONS:
            if frozen_exclusions[reason] != sum(
                item.exclusions[reason] for item in frozen_hops.values()
            ):
                raise ValueError(
                    f"per-hop {reason} exclusions do not match total"
                )
        object.__setattr__(self, "exclusions", frozen_exclusions)
        object.__setattr__(self, "per_hop", MappingProxyType(frozen_hops))

    @property
    def rate(self) -> float | None:
        return (
            None
            if self.candidates == 0
            else self.surviving / self.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "surviving": self.surviving,
            "rate": self.rate,
            "exclusions": dict(self.exclusions),
            "per_hop": {
                hop: accounting.to_dict()
                for hop, accounting in self.per_hop.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PathAccounting:
        if set(value) != {
            "candidates",
            "surviving",
            "rate",
            "exclusions",
            "per_hop",
        }:
            raise ValueError("invalid path-accounting fields")
        per_hop = value["per_hop"]
        if not isinstance(per_hop, Mapping):
            raise ValueError("per_hop must be an object")
        result = cls(
            value["candidates"],
            value["surviving"],
            value["exclusions"],
            {
                str(hop): CandidateAccounting.from_dict(
                    item,
                    allowed_reasons=_PATH_EXCLUSION_REASONS,
                )
                for hop, item in per_hop.items()
            },
        )
        if result.rate != value["rate"]:
            raise ValueError("path survival rate does not match counts")
        return result


@dataclass(frozen=True)
class CoverageManifest:
    split: str
    artifact_mode: str
    production_evaluation_eligible: bool
    source_file: str
    source_sha256: str
    source_lock_sha256: str
    source_archive_sha256: Mapping[str, str]
    schema_sha256: str
    recomputed_schema_sha256: str
    codec_sha256: str
    entities: Survival
    relations: Survival
    addresses: CandidateAccounting
    paths: PathAccounting
    pair_count: int
    item_count: int
    published_graph_rows: int
    artifacts: Mapping[str, str]
    out_dir: Path = field(compare=False)
    _codec: RelationCodec | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    version: int = 2
    address_policy: str = "unique_only_no_ranking"
    type_metadata: str = "unavailable"
    counterfactual_compatibility: str = "same_relation_domain"
    max_path_roots: int = MAX_PATH_ROOTS
    confirmatory_verdict_eligible: bool = False

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("unsupported coverage manifest version")
        if self.artifact_mode not in {"production", "fixture"}:
            raise ValueError("unsupported Wikidata artifact mode")
        if not isinstance(self.production_evaluation_eligible, bool):
            raise TypeError("production evaluation eligibility must be Boolean")
        expected_eligibility = self.artifact_mode == "production"
        if self.production_evaluation_eligible != expected_eligibility:
            raise ValueError(
                "artifact mode and production evaluation eligibility disagree"
            )
        if self.split not in {"valid", "test"}:
            raise ValueError("Wikidata robustness split must be valid or test")
        if self.address_policy != "unique_only_no_ranking":
            raise ValueError("unsupported Wikidata address policy")
        if self.type_metadata != "unavailable":
            raise ValueError("Wikidata5M type metadata must be unavailable")
        if self.counterfactual_compatibility != "same_relation_domain":
            raise ValueError("unsupported counterfactual compatibility basis")
        if self.confirmatory_verdict_eligible:
            raise ValueError("Wikidata robustness cannot enter the verdict")
        if self.recomputed_schema_sha256 != self.schema_sha256:
            raise ValueError("recomputed schema hash does not match supplied schema")
        for name, value in (
            ("pair_count", self.pair_count),
            ("item_count", self.item_count),
            ("published_graph_rows", self.published_graph_rows),
            ("max_path_roots", self.max_path_roots),
        ):
            _nonnegative_int(value, name)
        if self.item_count != self.pair_count * 2:
            raise ValueError("every robustness pair must contain two items")
        if self.pair_count != self.paths.surviving * 2:
            raise ValueError(
                "every surviving path must emit two task pairs"
            )
        object.__setattr__(self, "out_dir", Path(self.out_dir))
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(sorted(self.artifacts.items()))),
        )
        object.__setattr__(
            self,
            "source_archive_sha256",
            MappingProxyType(dict(sorted(self.source_archive_sha256.items()))),
        )

    @property
    def candidates(self) -> int:
        return self.addresses.candidates

    @property
    def surviving(self) -> int:
        return self.addresses.surviving

    @property
    def exclusions(self) -> Mapping[str, int]:
        return self.addresses.exclusions

    def open_graph(self) -> PackedGraphStore:
        if self._codec is None:
            raise ValueError("relation schema is required to open graph store")
        return PackedGraphStore.load(self.out_dir / "graph.store", self._codec)

    @property
    def graph(self) -> PackedGraphStore:
        return self.open_graph()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "split": self.split,
            "artifact_mode": self.artifact_mode,
            "production_evaluation_eligible": (
                self.production_evaluation_eligible
            ),
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_lock_sha256": self.source_lock_sha256,
            "source_archive_sha256": dict(self.source_archive_sha256),
            "schema_sha256": self.schema_sha256,
            "recomputed_schema_sha256": self.recomputed_schema_sha256,
            "codec_sha256": self.codec_sha256,
            "source_capabilities": dict(_SOURCE_CAPABILITIES),
            "address_policy": self.address_policy,
            "type_metadata": self.type_metadata,
            "counterfactual_compatibility": (
                self.counterfactual_compatibility
            ),
            "selection": {
                "max_path_roots": self.max_path_roots,
                "max_hops": MAX_HOPS,
                "selection_rule": (
                    "sha256_root_then_cyclic_frozen_relation_sequence"
                ),
            },
            "entities": self.entities.to_dict(),
            "relations": self.relations.to_dict(),
            "addresses": self.addresses.to_dict(),
            "paths": self.paths.to_dict(),
            "pair_count": self.pair_count,
            "item_count": self.item_count,
            "published_graph_rows": self.published_graph_rows,
            "artifacts": dict(self.artifacts),
            "analysis_role": "robustness_only",
            "confirmatory_verdict_eligible": (
                self.confirmatory_verdict_eligible
            ),
        }

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        schema: RelationSchema | None = None,
    ) -> CoverageManifest:
        manifest_path = Path(path)
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("coverage manifest must be a JSON object")
        expected = {
            "version",
            "split",
            "artifact_mode",
            "production_evaluation_eligible",
            "source_file",
            "source_sha256",
            "source_lock_sha256",
            "source_archive_sha256",
            "schema_sha256",
            "recomputed_schema_sha256",
            "codec_sha256",
            "source_capabilities",
            "address_policy",
            "type_metadata",
            "counterfactual_compatibility",
            "selection",
            "entities",
            "relations",
            "addresses",
            "paths",
            "pair_count",
            "item_count",
            "published_graph_rows",
            "artifacts",
            "analysis_role",
            "confirmatory_verdict_eligible",
        }
        if set(value) != expected:
            raise ValueError("invalid coverage manifest fields")
        if value["source_capabilities"] != _SOURCE_CAPABILITIES:
            raise ValueError("Wikidata source capabilities do not match")
        selection = value["selection"]
        if not isinstance(selection, Mapping) or set(selection) != {
            "max_path_roots",
            "max_hops",
            "selection_rule",
        }:
            raise ValueError("invalid path selection metadata")
        if (
            selection["max_hops"] != MAX_HOPS
            or selection["selection_rule"]
            != "sha256_root_then_cyclic_frozen_relation_sequence"
        ):
            raise ValueError("unsupported path selection contract")
        if value["analysis_role"] != "robustness_only":
            raise ValueError("Wikidata analysis role must be robustness_only")
        codec = None if schema is None else schema.codec
        result = cls(
            split=value["split"],
            artifact_mode=value["artifact_mode"],
            production_evaluation_eligible=value[
                "production_evaluation_eligible"
            ],
            source_file=value["source_file"],
            source_sha256=value["source_sha256"],
            source_lock_sha256=value["source_lock_sha256"],
            source_archive_sha256=value["source_archive_sha256"],
            schema_sha256=value["schema_sha256"],
            recomputed_schema_sha256=value["recomputed_schema_sha256"],
            codec_sha256=value["codec_sha256"],
            entities=Survival.from_dict(value["entities"]),
            relations=Survival.from_dict(value["relations"]),
            addresses=CandidateAccounting.from_dict(
                value["addresses"],
                allowed_reasons=_ADDRESS_EXCLUSION_REASONS,
            ),
            paths=PathAccounting.from_dict(value["paths"]),
            pair_count=value["pair_count"],
            item_count=value["item_count"],
            published_graph_rows=value["published_graph_rows"],
            artifacts=value["artifacts"],
            out_dir=manifest_path.parent,
            _codec=codec,
            version=value["version"],
            address_policy=value["address_policy"],
            type_metadata=value["type_metadata"],
            counterfactual_compatibility=value[
                "counterfactual_compatibility"
            ],
            max_path_roots=selection["max_path_roots"],
            confirmatory_verdict_eligible=value[
                "confirmatory_verdict_eligible"
            ],
        )
        if schema is not None:
            if result.schema_sha256 != schema.sha256():
                raise ValueError("coverage manifest schema hash mismatch")
            if result.codec_sha256 != schema.codec.sha256():
                raise ValueError("coverage manifest codec hash mismatch")
        return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


@contextmanager
def private_temporary_directory(
    parent: Path,
    prefix: str,
) -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    primary_error: BaseException | None = None
    try:
        yield path
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if os.path.lexists(path):
            try:
                shutil.rmtree(path)
            except BaseException:
                if primary_error is None:
                    raise


def _validate_schema(schema: RelationSchema) -> tuple[RelationSpec, ...]:
    if not isinstance(schema, RelationSchema):
        raise TypeError("schema must be RelationSchema v2")
    selected = tuple(
        spec
        for spec in schema.path_relations
        if spec.target_kind == "entity"
    )
    if not selected:
        raise ValueError("schema requires selected entity path relations")
    if any(not spec.relation_id.startswith("P") for spec in selected):
        raise ValueError("real Wikidata paths require P entity relations")
    return selected


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE entities (
          entity INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE source_relations (
          relation TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE source_addresses (
          subject INTEGER NOT NULL,
          relation TEXT NOT NULL,
          PRIMARY KEY (subject, relation)
        ) WITHOUT ROWID;
        CREATE TABLE selected_triples (
          subject INTEGER NOT NULL,
          relation TEXT NOT NULL,
          object INTEGER NOT NULL,
          PRIMARY KEY (subject, relation, object)
        ) WITHOUT ROWID;
        CREATE TABLE address_stats (
          subject INTEGER NOT NULL,
          relation TEXT NOT NULL,
          object_count INTEGER NOT NULL,
          sole_object INTEGER,
          PRIMARY KEY (subject, relation)
        ) WITHOUT ROWID;
        CREATE TABLE unique_addresses (
          subject INTEGER NOT NULL,
          relation TEXT NOT NULL,
          object INTEGER NOT NULL,
          selection_key BLOB NOT NULL,
          PRIMARY KEY (subject, relation)
        ) WITHOUT ROWID;
        CREATE TABLE ambiguous_addresses (
          subject INTEGER NOT NULL,
          relation TEXT NOT NULL,
          PRIMARY KEY (subject, relation)
        ) WITHOUT ROWID;
        CREATE TABLE relation_domain (
          relation TEXT NOT NULL,
          object INTEGER NOT NULL,
          PRIMARY KEY (relation, object)
        ) WITHOUT ROWID;
        CREATE TABLE published_rows (
          subject INTEGER NOT NULL,
          relation TEXT NOT NULL,
          object INTEGER NOT NULL,
          PRIMARY KEY (subject, relation)
        ) WITHOUT ROWID;
        """
    )


def _flush_source_rows(
    connection: sqlite3.Connection,
    entities: list[tuple[int]],
    relations: list[tuple[str]],
    addresses: list[tuple[int, str]],
    selected: list[tuple[int, str, int]],
) -> None:
    connection.executemany(
        "INSERT OR IGNORE INTO entities (entity) VALUES (?)",
        entities,
    )
    connection.executemany(
        "INSERT OR IGNORE INTO source_relations (relation) VALUES (?)",
        relations,
    )
    connection.executemany(
        """
        INSERT OR IGNORE INTO source_addresses (subject, relation)
        VALUES (?, ?)
        """,
        addresses,
    )
    connection.executemany(
        """
        INSERT OR IGNORE INTO selected_triples (subject, relation, object)
        VALUES (?, ?, ?)
        """,
        selected,
    )
    entities.clear()
    relations.clear()
    addresses.clear()
    selected.clear()


def _populate_source(
    connection: sqlite3.Connection,
    source: Path,
    selected_relations: frozenset[str],
) -> None:
    entities: list[tuple[int]] = []
    relations: list[tuple[str]] = []
    addresses: list[tuple[int, str]] = []
    selected: list[tuple[int, str, int]] = []
    with connection:
        for triple in iter_triples(source):
            if triple.subject > _UINT63_MAX or triple.object > _UINT63_MAX:
                raise ValueError("Wikidata entity ID exceeds SQLite range")
            entities.extend(((triple.subject,), (triple.object,)))
            relations.append((triple.relation,))
            addresses.append((triple.subject, triple.relation))
            if triple.relation in selected_relations:
                selected.append(
                    (triple.subject, triple.relation, triple.object)
                )
            if len(addresses) >= 10_000:
                _flush_source_rows(
                    connection,
                    entities,
                    relations,
                    addresses,
                    selected,
                )
        if addresses:
            _flush_source_rows(
                connection,
                entities,
                relations,
                addresses,
                selected,
            )


def _root_selection_key(split: str, subject: int, relation: str) -> bytes:
    return hashlib.sha256(
        f"{split}\0{subject}\0{relation}".encode()
    ).digest()


def _finalize_addresses(
    connection: sqlite3.Connection,
    split: str,
) -> None:
    with connection:
        connection.execute(
            """
            INSERT INTO address_stats
              (subject, relation, object_count, sole_object)
            SELECT
              subject,
              relation,
              COUNT(*),
              CASE WHEN COUNT(*) = 1 THEN MIN(object) ELSE NULL END
            FROM selected_triples
            GROUP BY subject, relation
            """
        )
        connection.execute(
            """
            INSERT INTO ambiguous_addresses (subject, relation)
            SELECT subject, relation
            FROM address_stats
            WHERE object_count > 1
            """
        )
        batch: list[tuple[int, str, int, bytes]] = []
        for subject, relation, object_id in connection.execute(
            """
            SELECT subject, relation, sole_object
            FROM address_stats
            WHERE object_count = 1
            ORDER BY subject, relation
            """
        ):
            batch.append(
                (
                    subject,
                    relation,
                    object_id,
                    _root_selection_key(split, subject, relation),
                )
            )
            if len(batch) >= 10_000:
                connection.executemany(
                    """
                    INSERT INTO unique_addresses
                      (subject, relation, object, selection_key)
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()
        if batch:
            connection.executemany(
                """
                INSERT INTO unique_addresses
                  (subject, relation, object, selection_key)
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )
        connection.execute(
            """
            INSERT INTO relation_domain (relation, object)
            SELECT DISTINCT relation, object
            FROM unique_addresses
            """
        )
        connection.execute(
            """
            CREATE INDEX unique_addresses_by_subject
            ON unique_addresses (subject)
            """
        )


def _scalar(connection: sqlite3.Connection, query: str, parameters=()) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise AssertionError("aggregate query returned no row")
    return int(row[0])


def _address_accounting(
    connection: sqlite3.Connection,
    selected_relations: frozenset[str],
) -> CandidateAccounting:
    candidates = _scalar(connection, "SELECT COUNT(*) FROM source_addresses")
    unique = _scalar(connection, "SELECT COUNT(*) FROM unique_addresses")
    ambiguous = _scalar(
        connection, "SELECT COUNT(*) FROM ambiguous_addresses"
    )
    selected_placeholders = ",".join("?" for _ in selected_relations)
    unselected = _scalar(
        connection,
        (
            "SELECT COUNT(*) FROM source_addresses "
            f"WHERE relation NOT IN ({selected_placeholders})"
        ),
        tuple(sorted(selected_relations)),
    )
    return CandidateAccounting(
        candidates,
        unique,
        {
            "ambiguous_address": ambiguous,
            "unselected_relation": unselected,
        },
        _ADDRESS_EXCLUSION_REASONS,
    )


def _lookup_next(
    connection: sqlite3.Connection,
    *,
    source_id: int,
    relation_id: str,
) -> tuple[int, str, int] | str:
    row = connection.execute(
        """
        SELECT object_count, sole_object
        FROM address_stats
        WHERE subject = ? AND relation = ?
        """,
        (source_id, relation_id),
    ).fetchone()
    if row is None:
        return "missing_hop"
    if row[0] != 1:
        return "ambiguous_address"
    return (
        source_id,
        relation_id,
        int(row[1]),
    )


def _independently_replay_original_rows(
    connection: sqlite3.Connection,
    rows: Sequence[tuple[int, str, int]],
    split: str,
) -> tuple[GraphRow, ...]:
    endpoint = rows[0][0]
    replayed = []
    for expected in rows:
        source, relation, target = expected
        if source != endpoint:
            raise AssertionError("candidate path is not endpoint-contiguous")
        actual = _lookup_next(
            connection,
            source_id=source,
            relation_id=relation,
        )
        if isinstance(actual, str) or actual != expected:
            raise AssertionError(
                "candidate original row does not match verified source"
            )
        replayed.append(
            GraphRow(
                source,
                relation,
                "out",
                "entity",
                str(target),
                (),
                f"wikidata5m-inductive-{split}",
            )
        )
        endpoint = target
    return tuple(replayed)


def _alternative_target(
    connection: sqlite3.Connection,
    relation: str,
    target: int,
) -> int | None:
    row = connection.execute(
        """
        SELECT object
        FROM relation_domain
        WHERE relation = ? AND object > ?
        ORDER BY object
        LIMIT 1
        """,
        (relation, target),
    ).fetchone()
    if row is None:
        row = connection.execute(
            """
            SELECT object
            FROM relation_domain
            WHERE relation = ? AND object < ?
            ORDER BY object
            LIMIT 1
            """,
            (relation, target),
        ).fetchone()
    return None if row is None else int(row[0])


def _action_plan(relations: Sequence[str]) -> list[dict[str, Any]]:
    if not 1 <= len(relations) <= MAX_HOPS:
        raise ValueError("Wikidata paths require one through six relations")
    actions = [
        GraphAction(0, relation, "out", read=True, halt=False)
        for relation in relations
    ]
    if len(actions) < MAX_HOPS:
        actions.append(
            GraphAction(0, relations[-1], "out", read=False, halt=True)
        )
        while len(actions) < MAX_HOPS:
            actions.append(
                GraphAction(
                    0,
                    relations[-1],
                    "out",
                    read=False,
                    halt=False,
                )
            )
    return [asdict(action) for action in actions]


def _aliases_for_path(
    rows: Sequence[tuple[int, str, int]],
    specs: Mapping[str, RelationSpec],
    candidate_key: str,
) -> tuple[tuple[str, ...], str]:
    aliases = []
    alternate = False
    for index, (_, relation, _) in enumerate(rows):
        relation_aliases = specs[relation].aliases
        choice = int.from_bytes(
            hashlib.sha256(
                f"{candidate_key}\0alias\0{index}".encode()
            ).digest()[:8],
            "big",
        ) % len(relation_aliases)
        aliases.append(relation_aliases[choice])
        alternate = alternate or choice > 0
    return tuple(aliases), (
        "includes_alternate" if alternate else "primary_only"
    )


def _common_meta(
    *,
    pair_id: str,
    variant: str,
    split: str,
    task: str,
    rows: Sequence[tuple[int, str, int]],
    aliases: Sequence[str],
    alias_slice: str,
    answer_choices: Sequence[str],
    changed_row: GraphRow | None,
    endpoint: int,
    original_endpoint: int,
    counterfactual_endpoint: int,
    fact_rows: Sequence[GraphRow],
    comparison_entity: int | None,
) -> dict[str, Any]:
    relations = tuple(row[1] for row in rows)
    return {
        "pair_id": pair_id,
        "variant": variant,
        "wikidata_split": split,
        "entity_slots": [rows[0][0], None, None, None],
        "gold_addresses": [
            [source, relation, "out"] for source, relation, _ in rows
        ],
        "gold_fact_ids": [stable_fact_id(row) for row in fact_rows],
        "gold_actions": _action_plan(relations),
        "answer_choices": list(answer_choices),
        "hop_count": len(rows),
        "relation_ids": list(relations),
        "relation_aliases": list(aliases),
        "relation_path_hash": composition_hash(relations),
        "alias_slice": alias_slice,
        "composition_slice": (
            "single_relation"
            if len(set(relations)) == 1
            else "multi_relation"
        ),
        "changed_row": None if changed_row is None else changed_row.as_json(),
        "counterfactual_changed_rows": int(changed_row is not None),
        "oracle_endpoint": f"Q{endpoint}",
        "original_endpoint": f"Q{original_endpoint}",
        "counterfactual_endpoint": f"Q{counterfactual_endpoint}",
        "comparison_entity": (
            None if comparison_entity is None else f"Q{comparison_entity}"
        ),
        "type_metadata": "unavailable",
        "counterfactual_compatibility": "same_relation_domain",
        "source_capabilities": dict(_SOURCE_CAPABILITIES),
        "analysis_role": "robustness_only",
        "confirmatory_verdict_eligible": False,
        "task": task,
    }


def _emit_path_pairs(
    *,
    split: str,
    candidate_key: str,
    rows: Sequence[tuple[int, str, int]],
    alternative: int,
    specs: Mapping[str, RelationSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provenance = f"wikidata5m-inductive-{split}"
    base_rows = [
        GraphRow(
            source,
            relation,
            "out",
            "entity",
            str(target),
            (),
            provenance,
        )
        for source, relation, target in rows
    ]
    final = base_rows[-1]
    changed = GraphRow(
        final.source_id,
        final.relation_id,
        final.direction,
        "entity",
        str(alternative),
        (),
        f"{provenance}:counterfactual",
    )
    counterfactual_rows = [*base_rows[:-1], changed]
    endpoint = rows[-1][2]
    aliases, alias_slice = _aliases_for_path(rows, specs, candidate_key)
    path_phrase = " then ".join(
        f"{alias} ({relation})"
        for alias, (_, relation, _) in zip(aliases, rows)
    )
    original_items = []
    counterfactual_items = []

    traversal_pair = f"wd-{split}-{candidate_key}-traversal"
    traversal_choices = (f"Q{endpoint}", f"Q{alternative}", "abstain")
    traversal_prompt = (
        f"Slot 0 refers to Q{rows[0][0]}. Follow {path_phrase}. "
        "Return the exact endpoint Q ID."
    )
    for variant, answer, endpoint_value, changed_row, fact_rows, destination in (
        (
            "original",
            f"Q{endpoint}",
            endpoint,
            None,
            base_rows,
            original_items,
        ),
        (
            "counterfactual",
            f"Q{alternative}",
            alternative,
            changed,
            counterfactual_rows,
            counterfactual_items,
        ),
    ):
        destination.append(
            {
                "qid": f"{traversal_pair}-{'o' if variant == 'original' else 'c'}",
                "task": "endpoint_traversal",
                "prompt": traversal_prompt,
                "answer": answer,
                "meta": _common_meta(
                    pair_id=traversal_pair,
                    variant=variant,
                    split=split,
                    task="endpoint_traversal",
                    rows=rows,
                    aliases=aliases,
                    alias_slice=alias_slice,
                    answer_choices=traversal_choices,
                    changed_row=changed_row,
                    endpoint=endpoint_value,
                    original_endpoint=endpoint,
                    counterfactual_endpoint=alternative,
                    fact_rows=fact_rows,
                    comparison_entity=None,
                ),
            }
        )

    equality_pair = f"wd-{split}-{candidate_key}-equality"
    original_equal = int(candidate_key[-1], 16) % 2 == 0
    comparison = endpoint if original_equal else alternative
    equality_prompt = (
        f"Slot 0 refers to Q{rows[0][0]}. Follow {path_phrase}. "
        f"Does the endpoint equal Q{comparison}? Answer yes or no."
    )
    equality_choices = ("yes", "no", "abstain")
    for variant, answer, endpoint_value, changed_row, fact_rows, destination in (
        (
            "original",
            "yes" if original_equal else "no",
            endpoint,
            None,
            base_rows,
            original_items,
        ),
        (
            "counterfactual",
            "no" if original_equal else "yes",
            alternative,
            changed,
            counterfactual_rows,
            counterfactual_items,
        ),
    ):
        destination.append(
            {
                "qid": f"{equality_pair}-{'o' if variant == 'original' else 'c'}",
                "task": "endpoint_equality",
                "prompt": equality_prompt,
                "answer": answer,
                "meta": _common_meta(
                    pair_id=equality_pair,
                    variant=variant,
                    split=split,
                    task="endpoint_equality",
                    rows=rows,
                    aliases=aliases,
                    alias_slice=alias_slice,
                    answer_choices=equality_choices,
                    changed_row=changed_row,
                    endpoint=endpoint_value,
                    original_endpoint=endpoint,
                    counterfactual_endpoint=alternative,
                    fact_rows=fact_rows,
                    comparison_entity=comparison,
                ),
            }
        )
    return original_items, counterfactual_items


def _path_candidate_key(
    split: str,
    root: tuple[int, str, int],
    hop_count: int,
) -> str:
    return hashlib.sha256(
        (
            f"{split}\0{root[0]}\0{root[1]}\0"
            f"{root[2]}\0{hop_count}"
        ).encode()
    ).hexdigest()[:24]


def _write_paths(
    connection: sqlite3.Connection,
    *,
    split: str,
    selected_specs: Sequence[RelationSpec],
    original_path: Path,
    counterfactual_path: Path,
) -> PathAccounting:
    unique_count = _scalar(
        connection, "SELECT COUNT(*) FROM unique_addresses"
    )
    root_limit = min(unique_count, MAX_PATH_ROOTS)
    roots = connection.execute(
        """
        SELECT subject, relation, object
        FROM unique_addresses
        ORDER BY selection_key, subject, relation
        LIMIT ?
        """,
        (root_limit,),
    )
    specs = {spec.relation_id: spec for spec in selected_specs}
    relation_order = tuple(spec.relation_id for spec in selected_specs)
    relation_indices = {
        relation_id: index
        for index, relation_id in enumerate(relation_order)
    }
    per_hop_exclusions = {
        hop: Counter(
            {
                reason: (
                    unique_count - root_limit
                    if reason == "selection_bound"
                    else 0
                )
                for reason in _PATH_EXCLUSION_REASONS
            }
        )
        for hop in range(1, MAX_HOPS + 1)
    }
    per_hop_survivors = Counter()
    pair_count = 0
    with (
        original_path.open("w", encoding="utf-8", newline="") as originals,
        counterfactual_path.open(
            "w", encoding="utf-8", newline=""
        ) as counterfactuals,
        connection,
    ):
        for raw_root in roots:
            root = (int(raw_root[0]), str(raw_root[1]), int(raw_root[2]))
            rows = [root]
            failure_reason: str | None = None
            for hop_count in range(1, MAX_HOPS + 1):
                prefix_key = _path_candidate_key(
                    split, root, hop_count
                )
                if hop_count > 1 and failure_reason is None:
                    expected_relation = relation_order[
                        (
                            relation_indices[root[1]]
                            + hop_count
                            - 1
                        )
                        % len(relation_order)
                    ]
                    next_row = _lookup_next(
                        connection,
                        source_id=rows[-1][2],
                        relation_id=expected_relation,
                    )
                    if isinstance(next_row, str):
                        failure_reason = next_row
                    else:
                        rows.append(next_row)
                        changed_address = next_row[:2]
                        if changed_address in {
                            row[:2] for row in rows[:-1]
                        }:
                            failure_reason = "repeated_address"
                if failure_reason is not None:
                    per_hop_exclusions[hop_count][failure_reason] += 1
                    continue
                alternative = _alternative_target(
                    connection,
                    rows[-1][1],
                    rows[-1][2],
                )
                if alternative is None:
                    per_hop_exclusions[hop_count][
                        "no_counterfactual_alternative"
                    ] += 1
                    continue
                replayed_rows = _independently_replay_original_rows(
                    connection,
                    rows,
                    split,
                )
                original_items, counterfactual_items = _emit_path_pairs(
                    split=split,
                    candidate_key=prefix_key,
                    rows=rows,
                    alternative=alternative,
                    specs=specs,
                )
                replay_and_validate_path_twins(
                    replayed_rows,
                    original_items,
                    counterfactual_items,
                )
                for source, relation, target in rows:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO published_rows
                          (subject, relation, object)
                        VALUES (?, ?, ?)
                        """,
                        (source, relation, target),
                    )
                originals.writelines(
                    _canonical_json_line(item) for item in original_items
                )
                counterfactuals.writelines(
                    _canonical_json_line(item)
                    for item in counterfactual_items
                )
                pair_count += len(original_items)
                per_hop_survivors[hop_count] += 1

    per_hop = {}
    for hop in range(1, MAX_HOPS + 1):
        per_hop[str(hop)] = CandidateAccounting(
            unique_count,
            per_hop_survivors[hop],
            per_hop_exclusions[hop],
            _PATH_EXCLUSION_REASONS,
        )
    total_exclusions = {
        reason: sum(
            per_hop_exclusions[hop][reason]
            for hop in range(1, MAX_HOPS + 1)
        )
        for reason in _PATH_EXCLUSION_REASONS
    }
    path_survivors = sum(per_hop_survivors.values())
    if pair_count != path_survivors * 2:
        raise AssertionError("path pair emission count drift")
    return PathAccounting(
        unique_count * MAX_HOPS,
        path_survivors,
        total_exclusions,
        per_hop,
    )


def _write_exclusion_ledger(
    path: Path,
    *,
    connection: sqlite3.Connection,
    split: str,
    addresses: CandidateAccounting,
    paths: PathAccounting,
) -> None:
    rows = []
    for relation, count in connection.execute(
        """
        SELECT relation, COUNT(*)
        FROM source_addresses
        WHERE relation NOT IN (
          SELECT DISTINCT relation FROM address_stats
        )
        GROUP BY relation
        ORDER BY relation
        """
    ):
        rows.append(
            {
                "version": 1,
                "split": split,
                "stage": "address",
                "reason": "unselected_relation",
                "relation_id": relation,
                "hop_count": None,
                "count": count,
            }
        )
    for relation, count in connection.execute(
        """
        SELECT relation, COUNT(*)
        FROM ambiguous_addresses
        GROUP BY relation
        ORDER BY relation
        """
    ):
        rows.append(
            {
                "version": 1,
                "split": split,
                "stage": "address",
                "reason": "ambiguous_address",
                "relation_id": relation,
                "hop_count": None,
                "count": count,
            }
        )
    for hop, accounting in paths.per_hop.items():
        for reason in _PATH_EXCLUSION_REASONS:
            count = accounting.exclusions[reason]
            if count:
                rows.append(
                    {
                        "version": 1,
                        "split": split,
                        "stage": "path",
                        "reason": reason,
                        "relation_id": None,
                        "hop_count": int(hop),
                        "count": count,
                    }
                )
    address_ledger_total = sum(
        row["count"] for row in rows if row["stage"] == "address"
    )
    path_ledger_total = sum(
        row["count"] for row in rows if row["stage"] == "path"
    )
    if address_ledger_total != sum(addresses.exclusions.values()):
        raise AssertionError("address exclusion ledger count drift")
    if path_ledger_total != sum(paths.exclusions.values()):
        raise AssertionError("path exclusion ledger count drift")
    path.write_text(
        "".join(_canonical_json_line(row) for row in rows),
        encoding="utf-8",
    )


def _published_graph_rows(
    connection: sqlite3.Connection,
    split: str,
) -> Iterator[GraphRow]:
    provenance = f"wikidata5m-inductive-{split}"
    for source, relation, target in connection.execute(
        """
        SELECT subject, relation, object
        FROM published_rows
        ORDER BY subject, relation
        """
    ):
        yield GraphRow(
            source,
            relation,
            "out",
            "entity",
            str(target),
            (),
            provenance,
        )


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "coverage-manifest.json"
    }


def _build_wikidata_paths(
    source: str | Path,
    schema: RelationSchema,
    split: str,
    out_dir: str | Path,
    *,
    fixture_lock: WikidataLock | None,
) -> CoverageManifest:
    """Build one frozen valid/test robustness artifact without training output."""

    if split not in {"valid", "test"}:
        raise ValueError("Wikidata robustness split must be valid or test")
    if fixture_lock is None:
        source_lock = WikidataLock.from_path(DEFAULT_WIKIDATA_LOCK_PATH)
        artifact_mode = "production"
        production_evaluation_eligible = True
    elif isinstance(fixture_lock, WikidataLock):
        source_lock = fixture_lock
        artifact_mode = "fixture"
        production_evaluation_eligible = False
    else:
        raise TypeError("fixture_lock must be a WikidataLock")
    selected_specs = _validate_schema(schema)
    selected_relations = frozenset(
        spec.relation_id for spec in selected_specs
    )
    destination = Path(out_dir)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Wikidata robustness output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with private_temporary_directory(
        destination.parent,
        f".{destination.name}.stage-",
    ) as staging:
        with private_temporary_directory(
            destination.parent,
            f".{destination.name}.source-",
        ) as extraction:
            verified = verify_extract_and_bind_schema(
                source,
                extraction,
                schema,
                split,
                lock=source_lock,
            )
            source_path = verified.split_path
            with private_temporary_directory(
                destination.parent,
                f".{destination.name}.paths-",
            ) as database_dir:
                with closing(
                    sqlite3.connect(database_dir / "paths.sqlite3")
                ) as connection:
                    connection.execute("PRAGMA temp_store=FILE")
                    connection.execute("PRAGMA synchronous=OFF")
                    connection.execute("PRAGMA journal_mode=OFF")
                    _create_tables(connection)
                    _populate_source(
                        connection,
                        source_path,
                        selected_relations,
                    )
                    _finalize_addresses(connection, split)
                    addresses = _address_accounting(
                        connection,
                        selected_relations,
                    )

                    eval_dir = staging / "eval"
                    eval_dir.mkdir()
                    paths = _write_paths(
                        connection,
                        split=split,
                        selected_specs=selected_specs,
                        original_path=eval_dir / "original.jsonl",
                        counterfactual_path=(
                            eval_dir / "counterfactual.jsonl"
                        ),
                    )
                    _write_exclusion_ledger(
                        staging / "exclusion-ledger.jsonl",
                        connection=connection,
                        split=split,
                        addresses=addresses,
                        paths=paths,
                    )

                    with PackedGraphStore.build(
                        staging / "graph.store",
                        _published_graph_rows(connection, split),
                        schema.codec,
                    ) as packed:
                        packed_rows = len(packed)
                        audited_items = audit_materialized_path_artifacts(
                            packed,
                            eval_dir / "original.jsonl",
                            eval_dir / "counterfactual.jsonl",
                        )
                    expected_items = paths.surviving * 4
                    if audited_items != expected_items:
                        raise ValueError(
                            "materialized packed store audit item count mismatch"
                        )
                    source_entities = _scalar(
                        connection,
                        "SELECT COUNT(*) FROM entities",
                    )
                    published_entities = _scalar(
                        connection,
                        """
                        SELECT COUNT(*) FROM (
                          SELECT subject AS entity FROM published_rows
                          UNION
                          SELECT object AS entity FROM published_rows
                        )
                        """,
                    )
                    surviving_relations = _scalar(
                        connection,
                        "SELECT COUNT(DISTINCT relation) FROM published_rows",
                    )
                    pair_count = paths.surviving * 2
                    manifest = CoverageManifest(
                        split=split,
                        artifact_mode=artifact_mode,
                        production_evaluation_eligible=(
                            production_evaluation_eligible
                        ),
                        source_file=source_path.name,
                        source_sha256=verified.source_sha256,
                        source_lock_sha256=(
                            verified.source_lock_sha256
                        ),
                        source_archive_sha256=(
                            verified.source_archive_sha256
                        ),
                        schema_sha256=schema.sha256(),
                        recomputed_schema_sha256=(
                            verified.recomputed_schema.sha256()
                        ),
                        codec_sha256=schema.codec.sha256(),
                        entities=Survival(
                            source_entities,
                            published_entities,
                        ),
                        relations=Survival(
                            len(selected_relations),
                            surviving_relations,
                        ),
                        addresses=addresses,
                        paths=paths,
                        pair_count=pair_count,
                        item_count=pair_count * 2,
                        published_graph_rows=packed_rows,
                        artifacts=_artifact_hashes(staging),
                        out_dir=destination,
                        _codec=schema.codec,
                    )
                    (staging / "coverage-manifest.json").write_text(
                        json.dumps(
                            manifest.to_dict(),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Wikidata robustness output already exists: {destination}"
            )
        os.replace(staging, destination)
        return manifest


def build_wikidata_paths(
    source: str | Path,
    schema: RelationSchema,
    split: str,
    out_dir: str | Path,
) -> CoverageManifest:
    """Build production artifacts bound to the committed Wikidata5M lock."""

    return _build_wikidata_paths(
        source,
        schema,
        split,
        out_dir,
        fixture_lock=None,
    )


def build_fixture_wikidata_paths(
    source: str | Path,
    schema: RelationSchema,
    split: str,
    out_dir: str | Path,
    *,
    fixture_lock: WikidataLock,
) -> CoverageManifest:
    """Build explicitly non-production artifacts from a synthetic fixture."""

    return _build_wikidata_paths(
        source,
        schema,
        split,
        out_dir,
        fixture_lock=fixture_lock,
    )
