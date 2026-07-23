"""Deterministic compiler for the locked seven-lane current dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from corpusgen.current_sources import (
    iter_aliases,
    iter_puzzle_tasks,
    iter_training_triples,
    load_dataset_lock,
    verify_current_sources,
)
from corpusgen.graph_records import (
    GraphAction,
    GraphAddress,
    GraphRow,
    TaggedSegment,
)
from corpusgen.graph_trace import serialize_action
from corpusgen.srgm_worlds import WorldConfig, generate_world, iter_worlds
from train.tokenizer import get_tok


_ROOT = Path(__file__).resolve().parents[1]
_CURRENT_LOCK = _ROOT / "configs" / "current-dataset-lock.json"
_DATASET_LOCK = load_dataset_lock(_CURRENT_LOCK)
_IMPLEMENTATION = _DATASET_LOCK.implementation
TARGET_SHARES = {
    ("fineweb" if lane == "fineweb_edu" else lane): float(share)
    for lane, share in _IMPLEMENTATION["raw_target_lane_shares"].items()
}
LANE_ORDER = tuple(TARGET_SHARES)
POSITION_BINS = int(_IMPLEMENTATION["random_mask_matching"]["packed_position_bins"])
ACTION_SLOTS = int(_IMPLEMENTATION["reasoning_protocol"]["action_slots"])
MAX_READS = int(_IMPLEMENTATION["reasoning_protocol"]["max_reads"])
CONTEXT_TOKENS = int(_IMPLEMENTATION["context_tokens"])
PACK_QUANTUM = 128
_COVERAGE_BY_SCALE = _IMPLEMENTATION["synthetic_fact_load"][
    "wikidata_coverage_by_scale"
]
_TOKEN_FLOORS = {
    scale: int(values["raw_target_tokens"])
    for scale, values in _IMPLEMENTATION["token_floors"].items()
}
_SCALES = set(_TOKEN_FLOORS)
_FACT_LOADS = {
    label: int(count)
    for label, count in _IMPLEMENTATION["synthetic_fact_load"]["labels"].items()
}
_ARC_TRANSFORMS = _IMPLEMENTATION["arc_tasks"]["transforms"]
_SPATIAL_SYMMETRIES = tuple(_ARC_TRANSFORMS["spatial_symmetries"])
_MAX_PUZZLE_TRANSFORMS = int(_ARC_TRANSFORMS["max_unique_excluding_original"])
_WIKIDATA_SAMPLE_SALT = (
    f"{_DATASET_LOCK.dataset_id}/wikidata-29m-balanced-hash-sample"
)
_FORMAT = "memorysplit-current-dataset-v1"
_COMPILER_VERSION = "seven-lane-paged-v2"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _json_line(value: object) -> str:
    return _canonical_bytes(value).decode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


@dataclass(frozen=True)
class CurrentBuildConfig:
    profile: str
    scale: str
    fact_load: str
    data_seed: int
    total_tokens: int

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "full"}:
            raise ValueError("profile must be 'smoke' or 'full'")
        if self.scale not in _SCALES:
            raise ValueError(f"scale must be one of {sorted(_SCALES)}")
        if self.fact_load not in _FACT_LOADS:
            raise ValueError(f"fact_load must be one of {sorted(_FACT_LOADS)}")
        if isinstance(self.data_seed, bool) or not isinstance(self.data_seed, int):
            raise ValueError("data_seed must be a non-negative integer")
        if self.data_seed < 0:
            raise ValueError("data_seed must be a non-negative integer")
        if (
            isinstance(self.total_tokens, bool)
            or not isinstance(self.total_tokens, int)
            or self.total_tokens <= 0
        ):
            raise ValueError("total_tokens must be a positive integer")
        if self.profile == "full" and self.total_tokens < _TOKEN_FLOORS[self.scale]:
            raise ValueError(
                f"full {self.scale} builds require at least "
                f"{_TOKEN_FLOORS[self.scale]} raw tokens"
            )


@dataclass(frozen=True)
class CurrentTriple:
    split: str
    row: int
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class CurrentPuzzle:
    source: str
    source_row: str
    task_id: str
    task: dict[str, Any]


@dataclass(frozen=True)
class CurrentSources:
    source_manifest: dict[str, Any]
    fineweb_rows: tuple[str, ...]
    triples: tuple[CurrentTriple, ...]
    aliases: dict[str, str]
    puzzles: tuple[CurrentPuzzle, ...]

    def iter_fineweb(self) -> Iterator[tuple[str, str]]:
        for index, text in enumerate(self.fineweb_rows, 1):
            yield f"fixture:{index}", text

    def iter_triples(self) -> Iterator[CurrentTriple]:
        yield from self.triples

    def iter_alias_rows(self) -> Iterator[tuple[str, str]]:
        yield from sorted(
            self.aliases.items(),
            key=lambda item: (item[0][0], int(item[0][1:])),
        )

    def iter_puzzles(self) -> Iterator[CurrentPuzzle]:
        yield from self.puzzles

    def manifest_bytes(self) -> bytes:
        return _canonical_bytes(self.source_manifest)


class _VerifiedSources:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.source_manifest = verify_current_sources(_DATASET_LOCK, self.root)

    def manifest_bytes(self) -> bytes:
        return (self.root / "source-manifest.json").read_bytes()

    def iter_fineweb(self) -> Iterator[tuple[str, str]]:
        manifest = self.source_manifest["fineweb_edu"]
        holdout = int(manifest["holdout_records"])
        row_number = 0
        for record in manifest["files"]:
            path = self.root / "fineweb_edu" / record["path"]
            try:
                import pyarrow.parquet as parquet
            except ImportError as error:
                raise RuntimeError(
                    "full FineWeb compilation requires pyarrow"
                ) from error
            source = parquet.ParquetFile(path)
            if "text" not in source.schema.names:
                raise ValueError(f"FineWeb shard lacks text column: {path}")
            for batch in source.iter_batches(columns=["text"], batch_size=1024):
                for text in batch.column(0).to_pylist():
                    row_number += 1
                    if row_number <= holdout:
                        continue
                    if not isinstance(text, str) or not text.strip():
                        continue
                    yield f"{record['path']}:{row_number}", text

    def iter_triples(self) -> Iterator[CurrentTriple]:
        for triple in iter_training_triples(self.root):
            yield CurrentTriple(
                split=triple.split,
                row=triple.row,
                subject=f"Q{triple.subject}",
                relation=triple.relation,
                object=f"Q{triple.object}",
            )

    def iter_alias_rows(self) -> Iterator[tuple[str, str]]:
        for alias in iter_aliases(self.root):
            yield alias.canonical_id, alias.display

    def iter_puzzles(self) -> Iterator[CurrentPuzzle]:
        for puzzle in iter_puzzle_tasks(self.root):
            yield CurrentPuzzle(
                source=puzzle.source,
                source_row=puzzle.path,
                task_id=puzzle.canonical_task_sha256,
                task=puzzle.task,
            )


def fixture_current_sources() -> CurrentSources:
    """Return the tiny, source-archive-free inputs used for packaged smoke."""

    puzzle_one = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
            {"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]},
        ],
        "test": [{"input": [[3, 0], [0, 0]], "output": [[0, 3], [0, 0]]}],
    }
    puzzle_two = {
        "train": [{"input": [[4, 4], [0, 0]], "output": [[0, 0], [4, 4]]}],
        "test": [{"input": [[5, 5], [0, 0]], "output": [[0, 0], [5, 5]]}],
    }
    triples = (
        CurrentTriple("inductive_train", 1, "Q1", "P31", "Q2"),
        CurrentTriple("inductive_train", 2, "Q1", "P31", "Q3"),
        CurrentTriple("inductive_train", 3, "Q1", "P31", "Q4"),
        CurrentTriple("inductive_train", 4, "Q2", "P279", "Q5"),
        CurrentTriple("inductive_train", 5, "Q5", "P279", "Q6"),
        CurrentTriple("transductive_train", 1, "Q6", "P279", "Q7"),
        CurrentTriple("transductive_train", 2, "Q10", "P999999", "Q11"),
        CurrentTriple("transductive_train", 3, "Q20", "P17", "Q21"),
        CurrentTriple("transductive_train", 4, "Q20", "P17", "Q22"),
        # The source spool must collapse exact duplicates.
        CurrentTriple("transductive_train", 5, "Q1", "P31", "Q2"),
    )
    aliases = {
        "Q1": "Alpha",
        "Q2": "Beta",
        "Q3": "Gamma",
        "Q4": "Delta",
        "Q5": "Epsilon",
        "Q6": "Zeta",
        "Q7": "Eta",
        "Q10": "Ten",
        "Q11": "Eleven",
        "Q20": "Twenty",
        "Q21": "Twenty One",
        "Q22": "Twenty Two",
        "P31": "instance of",
        "P279": "subclass of",
        "P17": "country",
        # P999999 intentionally has no alias: canonical identity is the fallback.
    }
    fineweb = (
        "Glaciers carved the valley and left long ridges of gravel behind.",
        "Wind turbines convert moving air into electricity for the local grid.",
        "The observatory records each comet crossing the dark winter sky.",
        "Bees communicate the location of food through patterned movements.",
        "A ceramic glaze changes color as minerals react inside the kiln.",
        "River deltas form when sediment settles where flowing water slows.",
        "A proof proceeds from explicit assumptions through valid deductions.",
        "Libraries preserve local newspapers so later readers can study them.",
    )
    manifest = {
        "format": "memorysplit-current-smoke-sources",
        "dataset_id": "relational-chinchilla",
        "profile": "smoke",
        "scientific_result": False,
        "fineweb_rows": len(fineweb),
        "training_triples": len(triples),
        "puzzle_tasks": 2,
    }
    return CurrentSources(
        source_manifest=manifest,
        fineweb_rows=fineweb,
        triples=triples,
        aliases=aliases,
        puzzles=(
            CurrentPuzzle("arc_agi_1", "training/a.json", "fixture-arc-1", puzzle_one),
            CurrentPuzzle(
                "conceptarc",
                "corpus/group/a.json",
                "fixture-concept-1",
                puzzle_two,
            ),
        ),
    )


def _coerce_sources(sources) -> CurrentSources | _VerifiedSources:
    if isinstance(sources, (CurrentSources, _VerifiedSources)):
        return sources
    if isinstance(sources, (str, os.PathLike, Path)):
        path = Path(sources)
        if path.name == "source-manifest.json" and path.is_file():
            path = path.parent
        return _VerifiedSources(path)
    if isinstance(sources, Mapping):
        required = {"source_manifest", "fineweb_rows", "triples", "aliases", "puzzles"}
        if set(sources) != required:
            raise ValueError("fixture source mapping fields do not match contract")
        triples = tuple(
            item
            if isinstance(item, CurrentTriple)
            else CurrentTriple(**dict(item))
            for item in sources["triples"]
        )
        puzzles = tuple(
            item
            if isinstance(item, CurrentPuzzle)
            else CurrentPuzzle(**dict(item))
            for item in sources["puzzles"]
        )
        return CurrentSources(
            source_manifest=dict(sources["source_manifest"]),
            fineweb_rows=tuple(str(item) for item in sources["fineweb_rows"]),
            triples=triples,
            aliases={str(key): str(value) for key, value in sources["aliases"].items()},
            puzzles=puzzles,
        )
    raise TypeError("sources must be a verified source root or CurrentSources")


@dataclass(frozen=True)
class _FactMeta:
    fact_id: str
    source: str
    source_row: str
    record_type: str


@dataclass(frozen=True)
class _CurrentRecord:
    lane: str
    source: str
    source_row: str
    record_type: str
    record_id: str
    segments: tuple[TaggedSegment, ...]
    facts: tuple[_FactMeta, ...] = ()
    control_fact_ids: tuple[str, ...] = ()
    graph_row: GraphRow | None = None
    triple_provenance: tuple[dict[str, Any], ...] = ()
    replay: bool = False

    def __post_init__(self) -> None:
        if self.lane not in TARGET_SHARES:
            raise ValueError(f"unknown current-dataset lane: {self.lane}")
        controls = sum(segment.role == "random_control" for segment in self.segments)
        if controls != len(self.control_fact_ids):
            raise ValueError("random-control links must align with control segments")
        fact_ids = {fact.fact_id for fact in self.facts}
        if len(fact_ids) != len(self.facts):
            raise ValueError("record fact metadata must be unique")
        if not set(self.control_fact_ids) <= fact_ids:
            raise ValueError("random-control link lacks fact metadata")


@dataclass(frozen=True)
class _EncodedRecord:
    record: _CurrentRecord
    ids: np.ndarray
    roles: tuple[str, ...]
    fact_ids: tuple[str | None, ...]
    control_ids: tuple[str | None, ...]


class _SourceSpool:
    def __init__(self, path: Path, sources: CurrentSources | _VerifiedSources) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-16384")
        self._create()
        self._populate(sources)

    def _create(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE fineweb (
                ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                source_row TEXT NOT NULL,
                text TEXT NOT NULL UNIQUE
            );
            CREATE TABLE triples (
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                split TEXT NOT NULL,
                source_row INTEGER NOT NULL,
                sample_hash BLOB NOT NULL,
                PRIMARY KEY (subject, relation, object)
            ) WITHOUT ROWID;
            CREATE TABLE aliases (
                canonical_id TEXT PRIMARY KEY,
                display TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE puzzles (
                ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_row TEXT NOT NULL,
                task_id TEXT NOT NULL UNIQUE,
                task_json TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _validate_identity(value: str, prefix: str) -> None:
        if (
            not isinstance(value, str)
            or not value.startswith(prefix)
            or not value[1:].isascii()
            or not value[1:].isdigit()
            or int(value[1:]) < 0
        ):
            raise ValueError(f"invalid canonical {prefix} identity: {value!r}")

    def _populate(self, sources: CurrentSources | _VerifiedSources) -> None:
        fineweb_rows = 0
        with self.connection:
            for source_row, text in sources.iter_fineweb():
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("FineWeb source rows must contain non-empty text")
                self.connection.execute(
                    "INSERT OR IGNORE INTO fineweb(source_row, text) VALUES (?, ?)",
                    (source_row, text),
                )
                fineweb_rows += 1
            for triple in sources.iter_triples():
                self._validate_identity(triple.subject, "Q")
                self._validate_identity(triple.relation, "P")
                self._validate_identity(triple.object, "Q")
                if not triple.split or triple.row <= 0:
                    raise ValueError("training triples require split and positive row")
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO triples
                        (
                            subject,
                            relation,
                            object,
                            split,
                            source_row,
                            sample_hash
                        )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        triple.subject,
                        triple.relation,
                        triple.object,
                        triple.split,
                        triple.row,
                        hashlib.sha256(
                            (
                                f"{_WIKIDATA_SAMPLE_SALT}\0{triple.split}\0"
                                f"{triple.relation}\0{triple.subject}\0"
                                f"{triple.object}"
                            ).encode("utf-8")
                        ).digest(),
                    ),
                )
            for canonical_id, display in sources.iter_alias_rows():
                prefix = canonical_id[:1]
                if prefix not in {"P", "Q"}:
                    raise ValueError(f"unexpected alias identity: {canonical_id}")
                self._validate_identity(canonical_id, prefix)
                normalized = " ".join(str(display).split()) or canonical_id
                self.connection.execute(
                    "INSERT OR REPLACE INTO aliases(canonical_id, display) VALUES (?, ?)",
                    (canonical_id, normalized),
                )
            for puzzle in sources.iter_puzzles():
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO puzzles
                        (source, source_row, task_id, task_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        puzzle.source,
                        puzzle.source_row,
                        puzzle.task_id,
                        json.dumps(
                            puzzle.task,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                            allow_nan=False,
                        ),
                    ),
                )
        if fineweb_rows <= 0 or self.count("fineweb") <= 0:
            raise ValueError("current dataset requires FineWeb training rows")
        if self.count("triples") <= 0:
            raise ValueError("current dataset requires training triples")
        if self.count("puzzles") <= 0:
            raise ValueError("current dataset requires puzzle tasks")

    def count(self, table: str) -> int:
        if table not in {"fineweb", "triples", "aliases", "puzzles"}:
            raise ValueError(f"unknown spool table: {table}")
        return int(
            self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        )

    def alias(self, canonical_id: str) -> str:
        row = self.connection.execute(
            "SELECT display FROM aliases WHERE canonical_id = ?",
            (canonical_id,),
        ).fetchone()
        return canonical_id if row is None else str(row[0])

    def select_balanced_training_sample(self, max_triples: int) -> None:
        """Keep a stable, near-equal hash prefix of each split/relation stratum."""

        if (
            isinstance(max_triples, bool)
            or not isinstance(max_triples, int)
            or max_triples <= 0
        ):
            raise ValueError("Wikidata sample size must be a positive integer")
        if self.count("triples") <= max_triples:
            return
        strata = self.connection.execute(
            """
            SELECT split, relation
            FROM triples
            GROUP BY split, relation
            """
        ).fetchall()
        strata = sorted(
            ((str(split), str(relation)) for split, relation in strata),
            key=lambda value: hashlib.sha256(
                (
                    f"{_WIKIDATA_SAMPLE_SALT}\0stratum\0"
                    f"{value[0]}\0{value[1]}"
                ).encode("utf-8")
            ).digest(),
        )
        base, remainder = divmod(max_triples, len(strata))
        with self.connection:
            self.connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS triples_sample_order
                    ON triples(split, relation, sample_hash);
                CREATE TEMP TABLE sampled_triples (
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    PRIMARY KEY(subject, relation, object)
                ) WITHOUT ROWID;
                """
            )
            for index, (split, relation) in enumerate(strata):
                quota = base + (index < remainder)
                if quota <= 0:
                    continue
                selected = self.connection.execute(
                    """
                    SELECT subject, relation, object
                    FROM triples
                    WHERE split = ? AND relation = ?
                    ORDER BY sample_hash, subject, object
                    LIMIT ?
                    """,
                    (split, relation, quota),
                )
                self.connection.executemany(
                    "INSERT INTO sampled_triples VALUES (?, ?, ?)",
                    selected,
                )
            self.connection.execute(
                """
                DELETE FROM triples
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM sampled_triples
                    WHERE sampled_triples.subject = triples.subject
                      AND sampled_triples.relation = triples.relation
                      AND sampled_triples.object = triples.object
                )
                """
            )
            self.connection.execute("DROP TABLE sampled_triples")

    def close(self) -> None:
        self.connection.close()


def _payload_text(identity: str, display: str) -> str:
    return identity if identity == display else f"{identity}|{display}"


def _control_text(tok, payload: str) -> str:
    length = len(tok.encode(payload))
    if length <= 0:
        raise ValueError("factual payload encoded to no tokens")
    text = " the" * length
    if len(tok.encode(text)) != length:
        raise AssertionError("random-control filler lost token alignment")
    return text


def _factual_segments(
    tok,
    payload: str,
    fact_id: str,
) -> tuple[tuple[TaggedSegment, ...], tuple[str, ...]]:
    return (
        (
            TaggedSegment(payload, "payload", fact_id=fact_id),
            TaggedSegment(" | control:", "plain"),
            TaggedSegment(_control_text(tok, payload), "random_control"),
        ),
        (fact_id,),
    )


def _encode_record(tok, record: _CurrentRecord) -> _EncodedRecord:
    ids: list[int] = []
    roles: list[str] = []
    fact_ids: list[str | None] = []
    control_ids: list[str | None] = []
    control_index = 0
    for segment in record.segments:
        encoded = tok.encode(segment.text)
        ids.extend(encoded)
        roles.extend([segment.role] * len(encoded))
        fact_ids.extend([segment.fact_id] * len(encoded))
        if segment.role == "random_control":
            control_id = record.control_fact_ids[control_index]
            control_index += 1
        else:
            control_id = None
        control_ids.extend([control_id] * len(encoded))
    total = len(ids) + 1
    padded = max(PACK_QUANTUM, math.ceil(total / PACK_QUANTUM) * PACK_QUANTUM)
    if padded > CONTEXT_TOKENS:
        raise ValueError(
            f"formatted record exceeds {CONTEXT_TOKENS} tokens: {record.record_id}"
        )
    padding = padded - total
    if padding:
        filler = tok.encode(" the")
        if len(filler) != 1:
            raise AssertionError("packing filler must be one token")
        ids.extend(filler * padding)
        roles.extend(["plain"] * padding)
        fact_ids.extend([None] * padding)
        control_ids.extend([None] * padding)
    ids.append(tok.EOT)
    roles.append("boundary")
    fact_ids.append(None)
    control_ids.append(None)
    if any(token_id < 0 or token_id >= 1 << 16 for token_id in ids):
        raise ValueError("current dataset token id does not fit uint16")
    return _EncodedRecord(
        record=record,
        ids=np.asarray(ids, dtype=np.uint16),
        roles=tuple(roles),
        fact_ids=tuple(fact_ids),
        control_ids=tuple(control_ids),
    )


def _exact_padding(tok, lane: str, length: int, ordinal: int) -> _EncodedRecord:
    if length <= 0:
        raise ValueError("padding record length must be positive")
    filler = tok.encode(" the")
    if len(filler) != 1:
        raise AssertionError("packing filler must be one token")
    ids = [filler[0]] * (length - 1) + [tok.EOT]
    record = _CurrentRecord(
        lane=lane,
        source="packing",
        source_row=f"boundary:{ordinal}",
        record_type="packing_boundary",
        record_id=f"packing-{ordinal}",
        segments=(),
    )
    return _EncodedRecord(
        record=record,
        ids=np.asarray(ids, dtype=np.uint16),
        roles=tuple(["plain"] * (length - 1) + ["boundary"]),
        fact_ids=tuple([None] * length),
        control_ids=tuple([None] * length),
    )


def _packed_position_bin(start: int, end: int, total_tokens: int) -> int:
    if not 0 <= start < end <= total_tokens:
        raise ValueError("packed span must be inside the consumed prefix")
    return min(POSITION_BINS - 1, ((start + end) * POSITION_BINS) // (2 * total_tokens))


def _next_position_boundary(position: int, total_tokens: int) -> int:
    for index in range(1, POSITION_BINS):
        boundary = (index * total_tokens + POSITION_BINS - 1) // POSITION_BINS
        if boundary > position:
            return boundary
    return total_tokens


def _record_crosses_position_boundary(
    start: int,
    length: int,
    total_tokens: int,
) -> bool:
    boundary = _next_position_boundary(start, total_tokens)
    return start < boundary < start + length


def _route_external(fact_id: str, data_seed: int) -> bool:
    digest = hashlib.sha256(f"{data_seed}\0{fact_id}".encode("utf-8")).digest()
    return digest[0] < 128


def _token_chunks(tok, text: str, limit: int = 768) -> Iterator[str]:
    if limit <= 0:
        raise ValueError("token chunk limit must be positive")
    token_ids = tok.encode(text)
    start = 0
    while start < len(token_ids):
        candidate_end = min(len(token_ids), start + limit)
        for end in range(candidate_end, start, -1):
            chunk = tok.decode(token_ids[start:end])
            if tok.encode(chunk) == token_ids[start:end]:
                yield chunk
                start = end
                break
        else:
            raise ValueError("source text cannot be split at a lossless token boundary")


def _iter_fineweb_records(spool: _SourceSpool, tok) -> Iterator[_CurrentRecord]:
    cycle = 0
    while True:
        saw = False
        rows = spool.connection.execute(
            "SELECT ordinal, source_row, text FROM fineweb ORDER BY ordinal"
        )
        for ordinal, source_row, text in rows:
            saw = True
            for chunk_index, chunk in enumerate(_token_chunks(tok, str(text))):
                yield _CurrentRecord(
                    lane="fineweb",
                    source="fineweb_edu",
                    source_row=str(source_row),
                    record_type="fineweb",
                    record_id=f"fineweb-{ordinal}-{chunk_index}",
                    segments=(TaggedSegment(chunk, "plain"),),
                    replay=cycle > 0,
                )
        if not saw:
            raise ValueError("FineWeb spool is empty")
        cycle += 1


def _wikidata_page_record(
    spool: _SourceSpool,
    tok,
    subject: str,
    relation: str,
    page: int,
    page_count: int,
    members: list[tuple[str, str, int]],
    *,
    replay: bool,
) -> _CurrentRecord:
    relation_display = spool.alias(relation)
    segments: list[TaggedSegment] = [
        TaggedSegment(
            (
                f"Wikidata entity {subject}|{spool.alias(subject)} relation "
                f"{relation}|{relation_display} direction out page "
                f"{page}/{page_count} values ["
            ),
            "plain",
        )
    ]
    facts: list[_FactMeta] = []
    controls: list[str] = []
    provenance: list[dict[str, Any]] = []
    targets: list[str] = []
    for index, (object_id, split, source_row) in enumerate(members):
        if index:
            segments.append(TaggedSegment(",", "plain"))
        fact_id = f"wikidata:{subject}:{relation}:{object_id}"
        payload = _payload_text(object_id, spool.alias(object_id))
        paired, links = _factual_segments(tok, payload, fact_id)
        segments.extend(paired)
        controls.extend(links)
        facts.append(
            _FactMeta(
                fact_id=fact_id,
                source="wikidata5m",
                source_row=f"{split}:{source_row}",
                record_type="wikidata_graph",
            )
        )
        provenance.append(
            {
                "fact_id": fact_id,
                "subject": subject,
                "relation": relation,
                "object": object_id,
                "split": split,
                "source_row": source_row,
                "page": page,
            }
        )
        targets.append(object_id)
    segments.append(TaggedSegment("]", "plain"))
    provenance_id = f"wikidata:{subject}:{relation}:out:{page}"
    graph_row = GraphRow(
        source_id=subject,
        relation_id=relation,
        direction="out",
        target_kind="entity",
        target=targets[0],
        provenance_id=provenance_id,
        page=page,
        targets=tuple(targets),
    )
    return _CurrentRecord(
        lane="wikidata_graph",
        source="wikidata5m",
        source_row=",".join(f"{split}:{row}" for _, split, row in members),
        record_type="wikidata_graph",
        record_id=provenance_id,
        segments=tuple(segments),
        facts=tuple(facts),
        control_fact_ids=tuple(controls),
        graph_row=graph_row,
        triple_provenance=tuple(provenance),
        replay=replay,
    )


def _partition_wikidata_group(
    spool: _SourceSpool,
    tok,
    subject: str,
    relation: str,
    members: list[tuple[str, str, int]],
    *,
    page_count_hint: int,
    replay: bool,
) -> list[_CurrentRecord]:
    records: list[_CurrentRecord] = []
    start = 0
    while start < len(members):
        page = len(records)
        best: _CurrentRecord | None = None
        best_end = start
        for end in range(start + 1, len(members) + 1):
            candidate = _wikidata_page_record(
                spool,
                tok,
                subject,
                relation,
                page,
                page_count_hint,
                members[start:end],
                replay=replay,
            )
            try:
                _encode_record(tok, candidate)
            except ValueError as error:
                if "exceeds" not in str(error):
                    raise
                break
            best = candidate
            best_end = end
        if best is None:
            raise ValueError(
                f"one Wikidata list member exceeds {CONTEXT_TOKENS} tokens"
            )
        records.append(best)
        start = best_end
    return records


def _paged_wikidata_group(
    spool: _SourceSpool,
    tok,
    subject: str,
    relation: str,
    members: list[tuple[str, str, int]],
    *,
    replay: bool,
) -> list[_CurrentRecord]:
    hint = 1
    for _ in range(16):
        records = _partition_wikidata_group(
            spool,
            tok,
            subject,
            relation,
            members,
            page_count_hint=hint,
            replay=replay,
        )
        if len(records) == hint:
            return records
        hint = len(records)
    raise ValueError("Wikidata page count did not converge")


def _iter_wikidata_pass(
    spool: _SourceSpool,
    tok,
    *,
    replay: bool,
) -> Iterator[_CurrentRecord]:
    groups = spool.connection.execute(
        """
        SELECT subject, relation
        FROM triples
        GROUP BY subject, relation
        ORDER BY CAST(SUBSTR(subject, 2) AS INTEGER),
                 CAST(SUBSTR(relation, 2) AS INTEGER)
        """
    )
    for subject, relation in groups:
        members = [
            (str(object_id), str(split), int(source_row))
            for object_id, split, source_row in spool.connection.execute(
                """
                SELECT object, split, source_row
                FROM triples
                WHERE subject = ? AND relation = ?
                ORDER BY CAST(SUBSTR(object, 2) AS INTEGER)
                """,
                (subject, relation),
            )
        ]
        yield from _paged_wikidata_group(
            spool,
            tok,
            str(subject),
            str(relation),
            members,
            replay=replay,
        )


def _iter_wikidata_graph_records(
    spool: _SourceSpool,
    tok,
) -> Iterator[_CurrentRecord]:
    pass_index = 0
    while True:
        saw = False
        for record in _iter_wikidata_pass(spool, tok, replay=pass_index > 0):
            saw = True
            yield record
        if not saw:
            raise ValueError("Wikidata training graph is empty")
        pass_index += 1


def _synthetic_fact_record(tok, fact, exposure: int) -> _CurrentRecord:
    payload = json.dumps(
        {
            "target_kind": fact.row.target_kind,
            "target": fact.row.target,
            "qualifiers": list(fact.row.qualifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    paired, controls = _factual_segments(tok, payload, fact.fact_id)
    row = fact.row
    return _CurrentRecord(
        lane="synthetic_graph",
        source="synthetic",
        source_row=f"world:{row.provenance_id}:{exposure}",
        record_type="synthetic_graph",
        record_id=f"synthetic-graph-{exposure}-{fact.fact_id}",
        segments=(
            TaggedSegment(
                f"Synthetic entity {row.source_id} relation {row.relation_id} returns ",
                "plain",
            ),
            *paired,
        ),
        facts=(
            _FactMeta(
                fact.fact_id,
                "synthetic",
                f"{row.provenance_id}:{fact.fact_id}",
                "synthetic_graph",
            ),
        ),
        control_fact_ids=controls,
        graph_row=row,
        replay=exposure > 0,
    )


def _iter_synthetic_graph_records(
    tok,
    config: CurrentBuildConfig,
) -> Iterator[_CurrentRecord]:
    exposure = 0
    while True:
        for world in iter_worlds(
            _FACT_LOADS[config.fact_load],
            64,
            config.data_seed,
        ):
            for fact in world.facts:
                yield _synthetic_fact_record(tok, fact, exposure)
                exposure += 1


def _action_text(tok, action: GraphAction) -> str:
    return tok.decode(serialize_action(action, tok))


def _trace_record(
    tok,
    *,
    lane: str,
    source: str,
    source_row: str,
    record_type: str,
    record_id: str,
    steps: list[tuple[int, GraphRow, str, _FactMeta]],
    answer: str,
    replay: bool,
) -> _CurrentRecord:
    if not 1 <= len(steps) <= 6:
        raise ValueError("training traces require one through six reads")
    segments: list[TaggedSegment] = [
        TaggedSegment(
            f"Query {record_id}: follow {len(steps)} exact graph reads. ",
            "query",
        )
    ]
    facts: dict[str, _FactMeta] = {}
    controls: list[str] = []
    for index in range(ACTION_SLOTS):
        if index < len(steps):
            slot, row, fact_id, fact_meta = steps[index]
            action = GraphAction(
                source_slot=slot,
                relation_id=row.relation_id,
                direction=row.direction,
                read=True,
                halt=False,
                page=row.page,
            )
            segments.append(TaggedSegment(_action_text(tok, action), "action"))
            segments.append(TaggedSegment("<|graph_return|>", "action"))
            payload = json.dumps(
                {
                    "page": row.page,
                    "targets": list(row.values),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            paired, links = _factual_segments(tok, payload, fact_id)
            segments.extend(paired)
            controls.extend(links)
            segments.append(TaggedSegment("<|graph_end|>", "action"))
            facts[fact_id] = fact_meta
            candidate = row.values[0]
        else:
            halt = index == len(steps)
            action = GraphAction(0, "r0", "out", read=False, halt=halt)
            segments.append(TaggedSegment(_action_text(tok, action), "action"))
            segments.extend(
                (
                    TaggedSegment("<|graph_return|>", "action"),
                    TaggedSegment("<|graph_miss|>", "action"),
                    TaggedSegment("<|graph_end|>", "action"),
                )
            )
            candidate = answer
        segments.extend(
            (
                TaggedSegment("<|answer_state|>", "action"),
                TaggedSegment(
                    f" candidate={candidate}",
                    "candidate_state",
                ),
            )
        )
    segments.append(TaggedSegment(f" final={answer}", "final_answer"))
    return _CurrentRecord(
        lane=lane,
        source=source,
        source_row=source_row,
        record_type=record_type,
        record_id=record_id,
        segments=tuple(segments),
        facts=tuple(facts.values()),
        control_fact_ids=tuple(controls),
        replay=replay,
    )


def _iter_synthetic_reasoning_records(
    tok,
    config: CurrentBuildConfig,
) -> Iterator[_CurrentRecord]:
    exposure = 0
    while True:
        world = generate_world(
            exposure,
            WorldConfig(n_entities=64, seed=config.data_seed),
        )
        by_address = {fact.row.address: fact for fact in world.facts}
        entities = sorted({fact.row.source_id for fact in world.facts})
        start = entities[exposure % len(entities)]
        requested = 1 + (exposure % 6)
        current = start
        steps = []
        for hop in range(requested):
            relation = f"r{hop % 4}"
            fact = by_address.get(GraphAddress(current, relation, "out"))
            if fact is None or fact.row.target_kind != "entity":
                break
            meta = _FactMeta(
                fact.fact_id,
                "synthetic",
                f"{fact.row.provenance_id}:{fact.fact_id}",
                "synthetic_reasoning",
            )
            steps.append((0, fact.row, fact.fact_id, meta))
            current = int(fact.row.target)
        if not steps:
            fact = world.facts[0]
            meta = _FactMeta(
                fact.fact_id,
                "synthetic",
                f"{fact.row.provenance_id}:{fact.fact_id}",
                "synthetic_reasoning",
            )
            steps = [(0, fact.row, fact.fact_id, meta)]
            current = fact.row.target
        yield _trace_record(
            tok,
            lane="synthetic_reasoning",
            source="synthetic",
            source_row=f"world:{world.world_id}",
            record_type="synthetic_reasoning",
            record_id=f"synthetic-reasoning-{exposure}",
            steps=steps,
            answer=str(current),
            replay=exposure > 0,
        )
        exposure += 1


def _iter_functional_wikidata(spool: _SourceSpool):
    return spool.connection.execute(
        """
        SELECT subject, relation, MIN(object), MIN(split), MIN(source_row)
        FROM triples
        GROUP BY subject, relation
        HAVING COUNT(*) = 1
        ORDER BY CAST(SUBSTR(subject, 2) AS INTEGER),
                 CAST(SUBSTR(relation, 2) AS INTEGER)
        """
    )


def _iter_wikidata_reasoning_records(
    spool: _SourceSpool,
    tok,
) -> Iterator[_CurrentRecord]:
    exposure = 0
    while True:
        saw = False
        for subject, relation, object_id, split, source_row in _iter_functional_wikidata(
            spool
        ):
            saw = True
            fact_id = f"wikidata:{subject}:{relation}:{object_id}"
            row = GraphRow(
                str(subject),
                str(relation),
                "out",
                "entity",
                str(object_id),
                (),
                f"wikidata:{subject}:{relation}:functional",
                page=0,
            )
            meta = _FactMeta(
                fact_id,
                "wikidata5m",
                f"{split}:{source_row}",
                "wikidata_reasoning",
            )
            yield _trace_record(
                tok,
                lane="wikidata_reasoning",
                source="wikidata5m",
                source_row=f"{split}:{source_row}",
                record_type="wikidata_reasoning",
                record_id=f"wikidata-reasoning-{exposure}-{subject}-{relation}",
                steps=[(0, row, fact_id, meta)],
                answer=str(object_id),
                replay=exposure > 0,
            )
            exposure += 1
        if not saw:
            raise ValueError("Wikidata reasoning requires functional graph addresses")


def _iter_refinement_records(
    tok,
    config: CurrentBuildConfig,
) -> Iterator[_CurrentRecord]:
    exposure = 0
    while True:
        world = generate_world(
            exposure,
            WorldConfig(n_entities=64, seed=config.data_seed ^ 0x5EED),
        )
        for fact in world.facts:
            payload = json.dumps(
                {
                    "target": fact.row.target,
                    "qualifiers": list(fact.row.qualifiers),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            paired, controls = _factual_segments(tok, payload, fact.fact_id)
            corrupted = hashlib.sha256(fact.fact_id.encode()).hexdigest()[:8]
            yield _CurrentRecord(
                lane="relational_refinement",
                source="synthetic",
                source_row=f"{fact.row.provenance_id}:{fact.fact_id}",
                record_type="relational_refinement",
                record_id=f"refinement-{exposure}-{fact.fact_id}",
                segments=(
                    TaggedSegment("Facts: ", "plain"),
                    *paired,
                    TaggedSegment(" Query: correct the candidate. ", "query"),
                    TaggedSegment(f"candidate={corrupted}", "candidate_state"),
                    TaggedSegment(
                        _action_text(
                            tok,
                            GraphAction(
                                0,
                                fact.row.relation_id,
                                fact.row.direction,
                                True,
                                False,
                            ),
                        ),
                        "action",
                    ),
                    TaggedSegment(f" corrected={fact.row.target}", "candidate_state"),
                    TaggedSegment(f" final={fact.row.target}", "final_answer"),
                ),
                facts=(
                    _FactMeta(
                        fact.fact_id,
                        "synthetic",
                        f"{fact.row.provenance_id}:{fact.fact_id}",
                        "relational_refinement",
                    ),
                ),
                control_fact_ids=controls,
                replay=exposure > 0,
            )
            exposure += 1


def _transform_grid(grid: list[list[int]], symmetry: int) -> list[list[int]]:
    value = [list(row) for row in grid]
    if symmetry >= 4:
        value = [list(reversed(row)) for row in value]
        symmetry -= 4
    for _ in range(symmetry):
        value = [list(row) for row in zip(*value[::-1])]
    return value


def _task_grids(task: dict[str, Any]) -> Iterator[list[list[int]]]:
    for split in ("train", "test"):
        for example in task.get(split, []):
            for field in ("input", "output"):
                grid = example.get(field)
                if grid is not None:
                    yield grid


def _color_permutations(task: dict[str, Any]) -> list[tuple[int, ...]]:
    task_digest = hashlib.sha256(_canonical_bytes(task)).hexdigest()
    identity = tuple(range(10))
    permutations = {identity}
    for salt in range(_MAX_PUZZLE_TRANSFORMS + 1):
        foreground = sorted(
            range(1, 10),
            key=lambda color: hashlib.sha256(
                f"{task_digest}\0{salt}\0{color}".encode()
            ).digest(),
        )
        permutations.add((0, *foreground))
        if len(permutations) >= 16:
            break
    return sorted(permutations)


def _common_translations(task: dict[str, Any]) -> list[tuple[int, int]]:
    row_low = column_low = -(1 << 30)
    row_high = column_high = 1 << 30
    constrained = False
    for grid in _task_grids(task):
        if not grid or not grid[0]:
            continue
        occupied = [
            (row, column)
            for row, values in enumerate(grid)
            for column, value in enumerate(values)
            if value != 0
        ]
        if not occupied:
            continue
        constrained = True
        rows = [row for row, _ in occupied]
        columns = [column for _, column in occupied]
        row_low = max(row_low, -min(rows))
        row_high = min(row_high, len(grid) - 1 - max(rows))
        column_low = max(column_low, -min(columns))
        column_high = min(column_high, len(grid[0]) - 1 - max(columns))
    if not constrained:
        return [(0, 0)]
    return [
        (row_shift, column_shift)
        for row_shift in range(row_low, row_high + 1)
        for column_shift in range(column_low, column_high + 1)
    ]


def _translate_grid(
    grid: list[list[int]],
    row_shift: int,
    column_shift: int,
) -> list[list[int]]:
    if not grid:
        return []
    height = len(grid)
    width = len(grid[0])
    translated = [[0 for _ in range(width)] for _ in range(height)]
    for row, values in enumerate(grid):
        if len(values) != width:
            raise ValueError("puzzle grids must be rectangular")
        for column, value in enumerate(values):
            if value == 0:
                continue
            target_row = row + row_shift
            target_column = column + column_shift
            if not (0 <= target_row < height and 0 <= target_column < width):
                raise ValueError("puzzle translation moved a foreground cell out of bounds")
            translated[target_row][target_column] = value
    return translated


def _transform_task(
    task: dict[str, Any],
    symmetry: int,
    colors: tuple[int, ...],
    translation: tuple[int, int],
) -> dict[str, Any]:
    transformed = json.loads(json.dumps(task))
    for split in ("train", "test"):
        for example in transformed.get(split, []):
            for field in ("input", "output"):
                if field in example:
                    grid = _transform_grid(example[field], symmetry)
                    grid = [
                        [colors[value] if 0 <= value <= 9 else value for value in row]
                        for row in grid
                    ]
                    example[field] = _translate_grid(grid, *translation)
    return transformed


def _puzzle_variants(task: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    identity = tuple(range(10))
    parameters: dict[str, tuple[int, tuple[int, ...], tuple[int, int]]] = {}
    for symmetry in range(len(_SPATIAL_SYMMETRIES)):
        symmetric = _transform_task(task, symmetry, identity, (0, 0))
        for colors in _color_permutations(task):
            value = {
                "symmetry": symmetry,
                "colors": list(colors),
                "translation": [0, 0],
            }
            digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
            parameters[digest] = (symmetry, colors, (0, 0))
        for translation in _common_translations(symmetric):
            if translation == (0, 0):
                continue
            value = {
                "symmetry": symmetry,
                "colors": list(identity),
                "translation": list(translation),
            }
            digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
            parameters[digest] = (symmetry, identity, translation)

    original_key = hashlib.sha256(_canonical_bytes(task)).hexdigest()
    variants: list[tuple[str, dict[str, Any]]] = [("0" * 64, task)]
    seen = {original_key}
    for digest in sorted(parameters):
        symmetry, colors, translation = parameters[digest]
        transformed = _transform_task(task, symmetry, colors, translation)
        task_key = hashlib.sha256(_canonical_bytes(transformed)).hexdigest()
        if task_key in seen:
            continue
        seen.add(task_key)
        variants.append((digest, transformed))
        if len(variants) == _MAX_PUZZLE_TRANSFORMS + 1:
            break
    return variants


def _iter_puzzle_records(spool: _SourceSpool, tok) -> Iterator[_CurrentRecord]:
    cycle = 0
    while True:
        saw = False
        rows = spool.connection.execute(
            """
            SELECT ordinal, source, source_row, task_id, task_json
            FROM puzzles
            ORDER BY ordinal
            """
        )
        for ordinal, source, source_row, task_id, task_json in rows:
            saw = True
            task = json.loads(task_json)
            for variant, (parameter_hash, transformed) in enumerate(
                _puzzle_variants(task)
            ):
                text = json.dumps(
                    {
                        "task_start": task_id,
                        "variant": variant,
                        "parameter_sha256": parameter_hash,
                        "demonstrations": transformed.get("train", []),
                        "query": transformed.get("test", []),
                        "task_end": task_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for chunk_index, chunk in enumerate(_token_chunks(tok, text)):
                    yield _CurrentRecord(
                        lane="puzzle_auxiliary",
                        source=str(source),
                        source_row=str(source_row),
                        record_type="puzzle_auxiliary",
                        record_id=(
                            f"puzzle-{cycle}-{ordinal}-{variant}-{chunk_index}"
                        ),
                        segments=(TaggedSegment(chunk, "plain"),),
                        replay=cycle > 0,
                    )
        if not saw:
            raise ValueError("puzzle spool is empty")
        cycle += 1


class _CurrentWriter:
    def __init__(
        self,
        root: Path,
        config: CurrentBuildConfig,
        connection: sqlite3.Connection,
    ) -> None:
        self.root = root
        self.config = config
        self.connection = connection
        self.token_path = root / "train.bin"
        self.sidecars = {
            name: root / f"{name}.weights.bin"
            for name in ("dense", "split", "random")
        }
        self.schedule_path = root / "schedule.jsonl"
        self.factual_path = root / "factual-span-ledger.jsonl"
        self.mask_path = root / "mask-ledger.jsonl"
        self.graph_path = root / "graph.jsonl"
        self.provenance_path = root / "wikidata-provenance.jsonl"
        self.token_file = self.token_path.open("wb")
        self.dense_file = self.sidecars["dense"].open("wb")
        self.split_file = self.sidecars["split"].open("wb")
        self.random_file = self.sidecars["random"].open("w+b", buffering=0)
        self.schedule_file = self.schedule_path.open("w", encoding="utf-8")
        self.factual_file = self.factual_path.open("w", encoding="utf-8")
        self.mask_file = self.mask_path.open("w", encoding="utf-8")
        self.graph_file = self.graph_path.open("w", encoding="utf-8")
        self.provenance_file = self.provenance_path.open("w", encoding="utf-8")
        self.total = 0
        self.records = 0
        self.lane_tokens: Counter[str] = Counter()
        self.lane_records: Counter[str] = Counter()
        self.masked_tokens: Counter[str] = Counter()
        self.graph_rows = 0
        self._closed = False
        self.connection.executescript(
            """
            CREATE TABLE external_spans (
                source TEXT NOT NULL,
                record_type TEXT NOT NULL,
                payload_length INTEGER NOT NULL,
                position_bin INTEGER NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                fact_id TEXT NOT NULL,
                record_index INTEGER NOT NULL
            );
            CREATE TABLE random_candidates (
                source TEXT NOT NULL,
                record_type TEXT NOT NULL,
                payload_length INTEGER NOT NULL,
                position_bin INTEGER NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                fact_id TEXT NOT NULL,
                record_index INTEGER NOT NULL
            );
            CREATE TABLE emitted_graph (
                source_id TEXT NOT NULL,
                relation_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                page INTEGER NOT NULL,
                PRIMARY KEY(source_id, relation_id, direction, page)
            ) WITHOUT ROWID;
            CREATE TABLE emitted_coverage (
                fact_id TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            """
        )

    @property
    def coverage_emitted(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM emitted_coverage"
            ).fetchone()[0]
        )

    @staticmethod
    def _spans(values: tuple[str | None, ...], roles: tuple[str, ...]):
        start = 0
        while start < len(roles):
            role = roles[start]
            value = values[start]
            end = start + 1
            while end < len(roles) and roles[end] == role and values[end] == value:
                end += 1
            yield start, end, role, value
            start = end

    def _write_graph(self, record: _CurrentRecord) -> None:
        row = record.graph_row
        if row is None:
            return
        address = row.address
        inserted = self.connection.execute(
            """
            INSERT OR IGNORE INTO emitted_graph
                (source_id, relation_id, direction, page)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(address.source_id),
                address.relation_id,
                address.direction,
                address.page,
            ),
        ).rowcount
        if inserted:
            self.graph_file.write(_json_line(row.as_json()))
            self.graph_rows += 1
            for provenance in record.triple_provenance:
                self.provenance_file.write(_json_line(provenance))

    def add(self, encoded: _EncodedRecord) -> None:
        record = encoded.record
        start = self.total
        end = start + len(encoded.ids)
        if end > self.config.total_tokens:
            raise ValueError("record exceeds exact consumed-prefix budget")
        ones = bytes([1]) * len(encoded.ids)
        split = bytearray(ones)
        facts = {fact.fact_id: fact for fact in record.facts}

        for local_start, local_end, role, fact_id in self._spans(
            encoded.fact_ids, encoded.roles
        ):
            if role != "payload":
                continue
            if fact_id is None or fact_id not in facts:
                raise ValueError("factual payload lacks source metadata")
            meta = facts[fact_id]
            global_start = start + local_start
            global_end = start + local_end
            length = local_end - local_start
            position_bin = _packed_position_bin(
                global_start,
                global_end,
                self.config.total_tokens,
            )
            external = _route_external(fact_id, self.config.data_seed)
            row = {
                "source": meta.source,
                "source_row": meta.source_row,
                "fact_id": fact_id,
                "route": "external" if external else "internal",
                "record_type": meta.record_type,
                "record_index": self.records,
                "payload_token_length": length,
                "position_bin": position_bin,
                "start": global_start,
                "end": global_end,
                "role": "payload",
            }
            self.factual_file.write(_json_line(row))
            if external:
                split[local_start:local_end] = bytes(length)
                self.masked_tokens["split"] += length
                self.connection.execute(
                    """
                    INSERT INTO external_spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meta.source,
                        meta.record_type,
                        length,
                        position_bin,
                        global_start,
                        global_end,
                        fact_id,
                        self.records,
                    ),
                )
                self.mask_file.write(
                    _json_line(
                        {
                            **row,
                            "condition": "split",
                        }
                    )
                )

        for local_start, local_end, role, fact_id in self._spans(
            encoded.control_ids, encoded.roles
        ):
            if role != "random_control":
                continue
            if fact_id is None or fact_id not in facts:
                raise ValueError("random-control span lacks factual stratum")
            meta = facts[fact_id]
            global_start = start + local_start
            global_end = start + local_end
            length = local_end - local_start
            position_bin = _packed_position_bin(
                global_start,
                global_end,
                self.config.total_tokens,
            )
            self.connection.execute(
                """
                INSERT INTO random_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.source,
                    meta.record_type,
                    length,
                    position_bin,
                    global_start,
                    global_end,
                    fact_id,
                    self.records,
                ),
            )

        self.token_file.write(encoded.ids.tobytes())
        self.dense_file.write(ones)
        self.split_file.write(split)
        self.random_file.write(ones)
        self.schedule_file.write(
            _json_line(
                {
                    "lane": record.lane,
                    "source": record.source,
                    "source_row": record.source_row,
                    "record_type": record.record_type,
                    "record_id": record.record_id,
                    "record_index": self.records,
                    "token_start": start,
                    "token_end": end,
                    "replay": record.replay,
                }
            )
        )
        self._write_graph(record)
        if record.lane == "wikidata_graph" and not record.replay:
            self.connection.executemany(
                "INSERT OR IGNORE INTO emitted_coverage(fact_id) VALUES (?)",
                ((fact.fact_id,) for fact in record.facts),
            )
        self.total = end
        self.records += 1
        self.lane_tokens[record.lane] += len(encoded.ids)
        self.lane_records[record.lane] += 1

    def pair_random_masks(self) -> None:
        keys = self.connection.execute(
            """
            SELECT source, record_type, payload_length, position_bin, COUNT(*)
            FROM external_spans
            GROUP BY source, record_type, payload_length, position_bin
            ORDER BY source, record_type, payload_length, position_bin
            """
        ).fetchall()
        for source, record_type, length, position_bin, required in keys:
            candidates = self.connection.execute(
                """
                SELECT start, end, fact_id, record_index
                FROM random_candidates
                WHERE source = ? AND record_type = ?
                  AND payload_length = ? AND position_bin = ?
                ORDER BY start
                """,
                (source, record_type, length, position_bin),
            ).fetchall()
            if len(candidates) < required:
                raise ValueError(
                    "insufficient exact-stratum random controls for "
                    f"{source}/{record_type}/{length}/{position_bin}"
                )
            ordered = sorted(
                candidates,
                key=lambda row: hashlib.sha256(
                    (
                        f"{self.config.data_seed}\0{source}\0{record_type}\0"
                        f"{length}\0{position_bin}\0{row[0]}"
                    ).encode()
                ).digest(),
            )
            for start, end, fact_id, record_index in ordered[:required]:
                return_position = self.random_file.tell()
                self.random_file.seek(start)
                self.random_file.write(bytes(end - start))
                self.random_file.seek(return_position)
                self.masked_tokens["random"] += end - start
                self.mask_file.write(
                    _json_line(
                        {
                            "source": source,
                            "source_row": None,
                            "fact_id": fact_id,
                            "condition": "random",
                            "record_type": record_type,
                            "record_index": record_index,
                            "payload_token_length": length,
                            "position_bin": position_bin,
                            "start": start,
                            "end": end,
                            "role": "random_control",
                        }
                    )
                )
        if self.masked_tokens["random"] != self.masked_tokens["split"]:
            raise ValueError("Random and Split zero counts differ")

    def close(self) -> None:
        if self._closed:
            return
        for handle in (
            self.token_file,
            self.dense_file,
            self.split_file,
            self.random_file,
            self.schedule_file,
            self.factual_file,
            self.mask_file,
            self.graph_file,
            self.provenance_file,
        ):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._closed = True


def _write_eval_fixture(root: Path) -> None:
    eval_dir = root / "eval"
    eval_dir.mkdir()
    row = GraphRow(0, "r0", "out", "entity", "1", (), "current-smoke-eval")
    (eval_dir / "graph.jsonl").write_text(_json_line(row.as_json()), encoding="utf-8")
    actions = [
        {
            "source_slot": 0,
            "relation_id": "r0",
            "direction": "out",
            "read": True,
            "halt": False,
            "page": 0,
        },
        {
            "source_slot": 0,
            "relation_id": "r0",
            "direction": "out",
            "read": False,
            "halt": True,
            "page": 0,
        },
        *[
            {
                "source_slot": 0,
                "relation_id": "r0",
                "direction": "out",
                "read": False,
                "halt": False,
                "page": 0,
            }
            for _ in range(10)
        ],
    ]
    item = {
        "qid": "current-smoke-eval-0",
        "task": "current_smoke",
        "prompt": "Slot zero names entity zero. Read its r0 page and answer yes or no.",
        "answer": "yes",
        "meta": {
            "pair_id": "current-smoke-eval-0",
            "variant": "original",
            "entity_slots": [0, None, None, None],
            "answer_choices": ["yes", "no"],
            "action_slots": 12,
            "gold_addresses": [[0, "r0", "out", 0]],
            "gold_actions": actions,
        },
    }
    (eval_dir / "items.jsonl").write_text(_json_line(item), encoding="utf-8")


def _complete_pass_tokens(spool: _SourceSpool, tok) -> int:
    return sum(
        len(_encode_record(tok, record).ids)
        for record in _iter_wikidata_pass(spool, tok, replay=False)
    )


def _build_private(
    config: CurrentBuildConfig,
    sources: CurrentSources | _VerifiedSources,
    root: Path,
) -> dict[str, Any]:
    tok = get_tok()
    source_manifest_bytes = sources.manifest_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    spool_path = root / ".current-build.sqlite3"
    spool = _SourceSpool(spool_path, sources)
    writer = _CurrentWriter(root, config, spool.connection)
    source_distinct_triples = spool.count("triples")
    coverage_mode = "complete_training_graph"
    if (
        config.profile == "full"
        and _COVERAGE_BY_SCALE[config.scale]
        == "deterministic_hash_sample_balanced_by_split_and_relation"
    ):
        max_sample = max(
            1,
            int(config.total_tokens * TARGET_SHARES["wikidata_graph"])
            // CONTEXT_TOKENS,
        )
        spool.select_balanced_training_sample(max_sample)
        coverage_mode = _COVERAGE_BY_SCALE[config.scale]
    distinct_triples = spool.count("triples")
    complete_tokens = _complete_pass_tokens(spool, tok)
    if (
        config.profile == "full"
        and complete_tokens > config.total_tokens * TARGET_SHARES["wikidata_graph"]
    ):
        writer.close()
        spool.close()
        raise ValueError(
            "complete accepted Wikidata pass does not fit the graph-lane budget"
        )

    generators = {
        "fineweb": _iter_fineweb_records(spool, tok),
        "wikidata_graph": _iter_wikidata_graph_records(spool, tok),
        "synthetic_graph": _iter_synthetic_graph_records(tok, config),
        "synthetic_reasoning": _iter_synthetic_reasoning_records(tok, config),
        "wikidata_reasoning": _iter_wikidata_reasoning_records(spool, tok),
        "relational_refinement": _iter_refinement_records(tok, config),
        "puzzle_auxiliary": _iter_puzzle_records(spool, tok),
    }
    buffered: dict[str, _EncodedRecord] = {}
    padding_ordinal = 0
    try:
        while writer.total < config.total_tokens:
            remaining = config.total_tokens - writer.total
            deficits = {
                lane: config.total_tokens * TARGET_SHARES[lane]
                - writer.lane_tokens[lane]
                for lane in LANE_ORDER
            }
            ranked = sorted(
                LANE_ORDER,
                key=lambda lane: (-deficits[lane], LANE_ORDER.index(lane)),
            )
            selected: _EncodedRecord | None = None
            for lane in ranked:
                if lane not in buffered:
                    buffered[lane] = _encode_record(tok, next(generators[lane]))
                if len(buffered[lane].ids) <= remaining:
                    selected = buffered.pop(lane)
                    break
            if selected is None:
                lane = ranked[0]
                selected = _exact_padding(tok, lane, remaining, padding_ordinal)
                padding_ordinal += 1
            if selected.record.facts and _record_crosses_position_boundary(
                writer.total,
                len(selected.ids),
                config.total_tokens,
            ):
                buffered[selected.record.lane] = selected
                boundary = _next_position_boundary(
                    writer.total,
                    config.total_tokens,
                )
                pad = boundary - writer.total
                writer.add(
                    _exact_padding(
                        tok,
                        "fineweb",
                        pad,
                        padding_ordinal,
                    )
                )
                padding_ordinal += 1
                continue
            writer.add(selected)
        writer.pair_random_masks()
    finally:
        writer.close()

    shares = {
        lane: writer.lane_tokens[lane] / writer.total for lane in LANE_ORDER
    }
    deviations = {
        lane: abs(shares[lane] - TARGET_SHARES[lane]) for lane in LANE_ORDER
    }
    max_deviation = max(deviations.values())
    complete_once = writer.coverage_emitted == distinct_triples
    if config.profile == "full" and max_deviation > 0.0025:
        spool.close()
        raise ValueError(
            f"full consumed-prefix lane deviation {max_deviation:.6f} exceeds 0.0025"
        )
    if config.profile == "full" and not complete_once:
        spool.close()
        raise ValueError("full build did not emit every accepted triple before replay")
    if not all(writer.lane_records[lane] > 0 for lane in LANE_ORDER):
        spool.close()
        raise ValueError("current dataset must contain every lane")

    _write_eval_fixture(root)
    source_provenance = {
        "format": "memorysplit-current-source-provenance",
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest": json.loads(source_manifest_bytes),
    }
    _write_json(root / "source-provenance.json", source_provenance)
    graph_manifest = {
        "path": "graph.jsonl",
        "sha256": _sha256_file(writer.graph_path),
        "bytes": writer.graph_path.stat().st_size,
        "rows": writer.graph_rows,
        "address_fields": ["entity_qid", "property_pid", "direction", "page"],
        "max_record_tokens": CONTEXT_TOKENS,
    }
    _write_json(root / "graph-manifest.json", graph_manifest)
    _write_json(
        root / "schedule-manifest.json",
        {
            "path": "schedule.jsonl",
            "sha256": _sha256_file(writer.schedule_path),
            "bytes": writer.schedule_path.stat().st_size,
            "records": writer.records,
            "tokens": writer.total,
            "scheduler": "largest-token-deficit/stable-lane-order",
        },
    )
    _write_json(
        root / "mask-manifest.json",
        {
            "factual_ledger": {
                "path": "factual-span-ledger.jsonl",
                "sha256": _sha256_file(writer.factual_path),
                "bytes": writer.factual_path.stat().st_size,
            },
            "mask_ledger": {
                "path": "mask-ledger.jsonl",
                "sha256": _sha256_file(writer.mask_path),
                "bytes": writer.mask_path.stat().st_size,
            },
            "strata": [
                "source",
                "record_type",
                "exact_payload_token_length",
                "packed_position_bin",
            ],
            "packed_position_bins": POSITION_BINS,
            "masked_tokens": dict(sorted(writer.masked_tokens.items())),
        },
    )
    scientific_result = config.profile == "full"
    report = {
        "format": _FORMAT,
        "compiler_version": _COMPILER_VERSION,
        "profile": config.profile,
        "scientific_result": scientific_result,
        "config": asdict(config),
        "target_shares": TARGET_SHARES,
        "tokens": {
            "total": writer.total,
            "lanes": {
                lane: {
                    "tokens": writer.lane_tokens[lane],
                    "records": writer.lane_records[lane],
                    "share": shares[lane],
                    "target_share": TARGET_SHARES[lane],
                    "deviation": deviations[lane],
                }
                for lane in LANE_ORDER
            },
            "max_share_deviation": max_deviation,
        },
        "coverage": {
            "mode": coverage_mode,
            "source_distinct_training_triples": source_distinct_triples,
            "distinct_training_triples": distinct_triples,
            "complete_pass_tokens": complete_tokens,
            "emitted_before_replay": writer.coverage_emitted,
            "complete_once": complete_once,
        },
        "masks": {
            "split_zero_tokens": writer.masked_tokens["split"],
            "random_zero_tokens": writer.masked_tokens["random"],
            "dense_all_one": True,
            "strata_exact": True,
        },
        "source_manifest_sha256": source_manifest_sha256,
        "checks": {
            "all_lanes_present": True,
            "sidecars_aligned": True,
            "dense_all_one": True,
            "split_random_equal_zero_count": (
                writer.masked_tokens["split"] == writer.masked_tokens["random"]
            ),
            "records_at_most_1024": True,
            "manifest_paths_relative": True,
        },
    }
    _write_json(root / "report.json", report)
    spool.close()
    spool_path.unlink()
    return report


def _build_identity(
    config: CurrentBuildConfig,
    sources: CurrentSources | _VerifiedSources,
) -> str:
    value = {
        "format": _FORMAT,
        "compiler_version": _COMPILER_VERSION,
        "config": asdict(config),
        "source_manifest_sha256": hashlib.sha256(
            sources.manifest_bytes()
        ).hexdigest(),
    }
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in [*sorted(directories, reverse=True), root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_manifest(
    root: Path,
    config: CurrentBuildConfig,
    report: dict[str, Any],
    build_id: str,
) -> None:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    artifacts = sorted(
        (_artifact(root, path) for path in paths),
        key=lambda item: item["path"],
    )
    _write_json(
        root / "manifest.json",
        {
            "format": _FORMAT,
            "compiler_version": _COMPILER_VERSION,
            "build_id": build_id,
            "profile": config.profile,
            "scientific_result": report["scientific_result"],
            "dataset_id": "relational-chinchilla",
            "config": asdict(config),
            "report": "report.json",
            "artifacts": artifacts,
        },
    )


def verify_current_dataset(
    out_dir: Path | str,
    expected_profile: str,
) -> dict:
    root = Path(out_dir)
    if expected_profile not in {"smoke", "full"}:
        raise ValueError("expected_profile must be 'smoke' or 'full'")
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"missing regular current dataset directory: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("current dataset manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_bytes())
    required = {
        "format",
        "compiler_version",
        "build_id",
        "profile",
        "scientific_result",
        "dataset_id",
        "config",
        "report",
        "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("current dataset manifest fields do not match contract")
    if (
        manifest["format"] != _FORMAT
        or manifest["compiler_version"] != _COMPILER_VERSION
        or manifest["dataset_id"] != "relational-chinchilla"
        or manifest["profile"] != expected_profile
        or manifest["config"].get("profile") != expected_profile
    ):
        raise ValueError("current dataset manifest identity mismatch")
    if expected_profile == "smoke" and manifest["scientific_result"] is not False:
        raise ValueError("smoke datasets must be explicitly non-scientific")
    if expected_profile == "full" and manifest["scientific_result"] is not True:
        raise ValueError("full datasets must be explicitly scientific")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("current dataset manifest artifacts must be non-empty")
    expected_files = {"manifest.json"}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError("invalid current dataset artifact record")
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("artifact paths must be safe and relative")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe current dataset artifact: {relative}")
        if (
            path.stat().st_size != artifact["bytes"]
            or _sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"current dataset artifact hash drift: {relative}")
        expected_files.add(relative.as_posix())
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"current dataset contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise ValueError("current dataset manifest is not hash-complete")

    config = manifest["config"]
    total = int(config["total_tokens"])
    train_path = root / "train.bin"
    if train_path.stat().st_size != total * np.dtype(np.uint16).itemsize:
        raise ValueError("train.bin does not match the configured uint16 token count")
    zero_counts = {}
    for condition in ("dense", "split", "random"):
        path = root / f"{condition}.weights.bin"
        if path.stat().st_size != total:
            raise ValueError(f"{condition} sidecar is not byte-aligned")
        zero_count = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                if set(chunk) - {0, 1}:
                    raise ValueError(f"{condition} sidecar is not binary")
                zero_count += chunk.count(0)
        zero_counts[condition] = zero_count
    if zero_counts["dense"] != 0:
        raise ValueError("Dense sidecar must be all one")
    if zero_counts["split"] <= 0 or zero_counts["split"] != zero_counts["random"]:
        raise ValueError("Split and Random must have equal nonzero zero counts")

    report_path = root / manifest["report"]
    report = json.loads(report_path.read_bytes())
    if (
        report.get("format") != _FORMAT
        or report.get("compiler_version") != _COMPILER_VERSION
        or report.get("profile") != expected_profile
        or report.get("config") != config
        or report.get("scientific_result") != manifest["scientific_result"]
        or report.get("tokens", {}).get("total") != total
        or report.get("target_shares") != TARGET_SHARES
        or not all(report.get("checks", {}).values())
    ):
        raise ValueError("current dataset report disagrees with manifest")
    if (
        report.get("masks", {}).get("split_zero_tokens") != zero_counts["split"]
        or report.get("masks", {}).get("random_zero_tokens") != zero_counts["random"]
    ):
        raise ValueError("current dataset mask report disagrees with sidecars")
    if expected_profile == "full" and (
        report.get("tokens", {}).get("max_share_deviation", 1.0) > 0.0025
        or report.get("coverage", {}).get("complete_once") is not True
    ):
        raise ValueError("full dataset scientific checks did not pass")
    return report


def build_current_dataset(
    config: CurrentBuildConfig,
    sources,
    out_dir: Path | str,
) -> dict:
    source_set = _coerce_sources(sources)
    if config.profile == "full" and not isinstance(source_set, _VerifiedSources):
        raise ValueError("full builds require a verified Task 1 source root")
    destination = Path(out_dir)
    build_id = _build_identity(config, source_set)
    if (
        destination.is_dir()
        and not destination.is_symlink()
        and not any(destination.iterdir())
    ):
        destination.rmdir()
    if destination.exists() or destination.is_symlink():
        try:
            report = verify_current_dataset(destination, config.profile)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"conflicting current dataset output: {destination}"
            ) from error
        manifest = json.loads((destination / "manifest.json").read_bytes())
        if manifest["build_id"] != build_id or manifest["config"] != asdict(config):
            raise ValueError(f"conflicting current dataset output: {destination}")
        return report

    parent = destination.parent
    if parent.is_symlink():
        raise ValueError(f"current dataset parent is a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    partial = parent / f".{destination.name}.partial-{os.getpid()}"
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"stale current dataset partial output: {partial}")
    partial.mkdir()
    try:
        report = _build_private(config, source_set, partial)
        _write_manifest(partial, config, report, build_id)
        verify_current_dataset(partial, config.profile)
        _fsync_tree(partial)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"current dataset output appeared during build: {destination}"
            )
        partial.rename(destination)
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return verify_current_dataset(destination, config.profile)


def build_fixture_current_dataset(
    out_dir: Path | str,
    *,
    total_tokens: int = 131_072,
) -> dict:
    return build_current_dataset(
        CurrentBuildConfig(
            profile="smoke",
            scale="29m",
            fact_load="n50k",
            data_seed=1,
            total_tokens=total_tokens,
        ),
        fixture_current_sources(),
        out_dir,
    )


def build_reasoning_v2_smoke_fixture(out_dir: Path | str) -> dict:
    """Build the non-scientific intervention-faithfulness smoke artifact."""

    from corpusgen.reasoning.smoke import build_v2_smoke_fixture

    return build_v2_smoke_fixture(out_dir)


def verify_reasoning_v2_smoke_fixture(out_dir: Path | str) -> dict:
    """Verify route dose and semantic closure for a v2 smoke artifact."""

    from corpusgen.reasoning.smoke import verify_v2_smoke_fixture

    return verify_v2_smoke_fixture(out_dir)
