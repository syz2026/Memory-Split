"""Deterministic production renderers for reasoning-v2 exposure lanes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from corpusgen.graph_records import RenderedRecord, ScheduleEntry, TaggedSegment
from corpusgen.reasoning import (
    SemanticFact,
    SupervisedField,
    plan_occurrence_closure,
)
from corpusgen.reasoning_v2.catalog import (
    CatalogDraft,
    CatalogRecord,
    catalog_record_id,
)
from corpusgen.reasoning_v2.contracts import LANE_ORDER, LaneId
from corpusgen.reasoning_v2.semantic import (
    RouteIndex,
    TokenOccurrenceBinding,
    TokenSemanticSpan,
    build_occurrence_closure_ledger,
    derive_sidecar_weights,
)
from corpusgen.reasoning_v2.wikidata_source import (
    TRAINING_SPLITS,
    V2TrainingTriple,
    WikidataDerivedView,
    lookup_alias,
    lookup_training_triple,
)
from corpusgen.srgm_worlds import iter_graph_records, iter_worlds
from train.tokenizer import get_tok


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FINEWEB_PATHS = (
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
    "sample/10BT/002_00000.parquet",
)
_FINEMATH_PATH_RE = re.compile(
    r"finemath-(?:4plus|3plus)/train-[0-9]{5}-of-[0-9]{5}\.parquet\Z"
)
_MAX_TARGET_COUNT = 1_024
_SYNTHETIC_WORLD_ENTITIES = 64
_SYNTHETIC_WORLD_SIZE = 64
_SYNTHETIC_WORLD_ID_OFFSET = 0
_WIKIDATA_LOCATOR_KEYS = frozenset(
    {
        "member",
        "path",
        "row",
        "split",
        "training_edge_key",
        "training_split",
        "wikidata_view_sha256",
    }
)
_SEGMENT_ROLES = {
    "plain": "plain_text",
    "payload": "factual_payload",
    "random_control": "plain_text",
    "rule": "rule",
    "action": "operator",
    "query": "schema",
    "candidate_state": "answer_state",
    "provisional_answer": "answer_state",
    "final_answer": "answer_state",
}


def _byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def _strict_text(
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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON payload is forbidden: {value}")


def _require_finite(value: object, description: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} contains a non-finite value")
    if isinstance(value, list):
        for item in value:
            _require_finite(item, description)
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite(item, description)


def _validate_payload_text(text: str) -> None:
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return
    try:
        value = json.loads(stripped, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError("graph payload is not valid finite JSON") from error
    _require_finite(value, "graph payload")


def _safe_relative_path(value: object, description: str) -> str:
    text = _strict_text(value, description)
    if "\\" in text:
        raise ValueError(f"{description} must be a canonical POSIX path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or relative.as_posix() != text
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{description} is not a safe relative path")
    return text


def _locator(record: CatalogRecord) -> dict[str, str | int]:
    if type(record.source_locator) is not tuple:
        raise ValueError("catalog source locator must be a tuple")
    values: dict[str, str | int] = {}
    keys: list[str] = []
    for item in record.source_locator:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("catalog source locator entry is invalid")
        key, value = item
        key = _strict_text(key, "catalog source locator key")
        if key in values:
            raise ValueError("catalog source locator contains duplicate keys")
        if type(value) is str:
            value = _strict_text(value, "catalog source locator value")
        elif type(value) is int and value >= 0:
            pass
        else:
            raise ValueError("catalog source locator value is invalid")
        keys.append(key)
        values[key] = value
    if tuple(keys) != tuple(sorted(keys, key=_byte_key)):
        raise ValueError("catalog source locator keys are not canonical")
    return values


def _validate_catalog_record(
    record: CatalogRecord,
    lane_id: LaneId,
    allowed_sources: frozenset[str],
) -> dict[str, str | int]:
    if not isinstance(record, CatalogRecord):
        raise TypeError("renderer input must be a CatalogRecord")
    if record.lane_id != lane_id:
        raise ValueError(
            f"renderer lane mismatch: expected={lane_id}, actual={record.lane_id}"
        )
    if record.source_id not in allowed_sources:
        raise ValueError(f"{lane_id} source authority rejects {record.source_id}")
    if (
        type(record.target_count) is not int
        or record.target_count <= 0
        or record.target_count > _MAX_TARGET_COUNT
    ):
        raise ValueError("catalog target count is outside frozen core bounds")
    locator = _locator(record)
    draft = CatalogDraft(
        lane_id=record.lane_id,
        source_id=record.source_id,
        source_key=record.source_key,
        source_byte_sha256=record.source_byte_sha256,
        source_locator=record.source_locator,
        semantic_flags=record.semantic_flags,
        semantic_facts=record.semantic_facts,
    )
    if record.record_id != catalog_record_id(draft, record.target_count):
        raise ValueError("catalog record ID commitment or target-count drift")
    return locator


def _source_relative(locator: dict[str, str | int]) -> str:
    path = locator.get("path")
    source_path = locator.get("source_path")
    if type(path) is str and source_path is None:
        return _safe_relative_path(path, "catalog source path")
    if type(source_path) is str and path is None:
        return _safe_relative_path(source_path, "catalog source path")
    raise ValueError("catalog source locator requires exactly one source path")


def _source_path(
    source_root: Path,
    source_id: str,
    relative: str,
) -> Path:
    parts = PurePosixPath(relative).parts
    path = (
        source_root.joinpath(*parts)
        if parts[0] == source_id
        else source_root.joinpath(source_id, *parts)
    )
    current = source_root
    if not current.is_dir() or current.is_symlink():
        raise ValueError("verified source root is missing or unsafe")
    for part in path.relative_to(source_root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source path crosses a symlink: {relative}")
    return path


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_bound_regular(
    path: Path,
    description: str,
) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        named = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} is missing or unsafe") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(named.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or named.st_nlink != 1
        or opened.st_nlink != 1
        or _file_identity(named) != _file_identity(opened)
    ):
        os.close(descriptor)
        raise ValueError(f"{description} is not an exact regular source file")
    return descriptor, _file_identity(opened)


def _hash_regular_file(path: Path, description: str) -> tuple[str, tuple[int, ...]]:
    descriptor, identity = _open_bound_regular(path, description)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            _file_identity(after) != identity
            or _file_identity(named_after) != identity
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"{description} identity drift while reading")
        return digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _verify_source_hash(
    source_root: Path,
    record: CatalogRecord,
    relative: str,
) -> tuple[Path, tuple[int, ...]]:
    path = _source_path(source_root, record.source_id, relative)
    digest, identity = _hash_regular_file(path, "catalog source file")
    if digest != record.source_byte_sha256:
        raise ValueError("catalog source SHA-256 drift")
    return path, identity


def _reverify_identity(path: Path, expected: tuple[int, ...]) -> None:
    _digest, actual = _hash_regular_file(path, "catalog source file")
    if actual != expected:
        raise ValueError("catalog source file identity drift")


def _check_regular_identity(
    path: Path,
    expected: tuple[int, ...],
    description: str,
) -> None:
    descriptor, actual = _open_bound_regular(path, description)
    try:
        if actual != expected:
            raise ValueError(f"{description} identity drift")
    finally:
        os.close(descriptor)


def _iter_parquet_texts(path: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("exposure rendering requires pyarrow") from error
    with path.open("rb") as handle:
        parquet_file = parquet.ParquetFile(handle)
        if "text" not in parquet_file.schema.names:
            raise ValueError(f"source parquet has no text column: {path}")
        for batch in parquet_file.iter_batches(batch_size=1024, columns=["text"]):
            for value in batch.column(0).to_pylist():
                if type(value) is not str:
                    raise ValueError(f"source parquet text is not a string: {path}")
                yield value


def _row_index(locator: dict[str, str | int]) -> int:
    value = locator.get("row")
    if type(value) is not int:
        raise ValueError("catalog source locator requires an integer row")
    return value


def _read_locked_text(
    path: Path,
    row_index: int,
    *,
    identity: tuple[int, ...],
) -> str:
    selected = None
    for index, value in enumerate(_iter_parquet_texts(path)):
        if index == row_index:
            selected = _strict_text(value, "source text")
            break
    _reverify_identity(path, identity)
    if selected is None:
        raise ValueError("catalog source row is absent from the locked file")
    return selected


def _empty_closure(token_count: int):
    return build_occurrence_closure_ledger(
        token_count=token_count,
        facts=(),
        fields=(),
        bindings=(),
    )


def _render_text(
    record: CatalogRecord,
    routes: RouteIndex,
    text: str,
) -> "ProductionRenderedRecord":
    text = _strict_text(text, "source text")
    tok = get_tok()
    core = tok.encode(text)
    available = record.target_count - 1
    if len(core) < available:
        raise ValueError("locked text row is too short for its target count")
    token_ids = tuple((*core[:available], tok.EOT))
    spans = (TokenSemanticSpan(0, len(token_ids), None, "plain_text"),)
    weights = derive_sidecar_weights(
        len(token_ids),
        spans,
        routes,
        closure=_empty_closure(len(token_ids)),
    )
    if weights.leaks:
        raise ValueError("semantic sidecar leak closure failed")
    return ProductionRenderedRecord(
        record_id=record.record_id,
        lane_id=record.lane_id,
        token_ids=token_ids,
        dense_target_weights=weights.dense,
        split90_target_weights=weights.split90,
        semantic_spans=spans,
        proofs=(),
        flags=record.semantic_flags,
        semantic_leaks=tuple(dict(leak) for leak in weights.leaks),
    )


def _finemath_paths(source_root: Path, subset: str) -> tuple[Path, ...]:
    directory = source_root / "finemath" / subset
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"FineMath {subset} source directory is unsafe")
    paths = tuple(
        sorted(
            directory.glob("train-*.parquet"),
            key=lambda path: path.name.encode("utf-8"),
        )
    )
    for path in paths:
        relative = path.relative_to(source_root / "finemath").as_posix()
        if (
            _FINEMATH_PATH_RE.fullmatch(relative) is None
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError(f"FineMath source path is outside locked authority: {path}")
    return paths


def _nfc_text_sha256(text: str) -> tuple[str, str]:
    normalized = _strict_text(text, "FineMath source text")
    return normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _select_finemath_text(
    source_root: Path,
    relative: str,
    row_index: int,
    target_path: Path,
    target_identity: tuple[int, ...],
) -> str:
    seen: set[str] = set()
    for fineweb_relative in _FINEWEB_PATHS:
        path = _source_path(source_root, "fineweb_edu", fineweb_relative)
        for raw_text in _iter_parquet_texts(path):
            _text, digest = _nfc_text_sha256(raw_text)
            seen.add(digest)

    four_plus = _finemath_paths(source_root, "finemath-4plus")
    three_plus = _finemath_paths(source_root, "finemath-3plus")
    expected_target = target_path.relative_to(source_root / "finemath").as_posix()
    if relative != expected_target:
        raise ValueError("FineMath target path is outside source authority")
    selected = None
    found_target = False
    for path in (*four_plus, *three_plus):
        path_relative = path.relative_to(source_root / "finemath").as_posix()
        for index, raw_text in enumerate(_iter_parquet_texts(path)):
            text, digest = _nfc_text_sha256(raw_text)
            duplicate = digest in seen
            if not duplicate:
                seen.add(digest)
            if path_relative == relative and index == row_index:
                found_target = True
                if duplicate:
                    raise ValueError(
                        "catalog FineMath row was removed by exact cross-deduplication"
                    )
                selected = text
                break
        if selected is not None:
            break
    _reverify_identity(target_path, target_identity)
    if not found_target or selected is None:
        raise ValueError("catalog FineMath row is absent from selected locked shards")
    return selected


def _semantic_facts(record: CatalogRecord) -> tuple[SemanticFact, ...]:
    facts = tuple(
        SemanticFact(fact.fact_id, fact.surfaces)
        for fact in record.semantic_facts
    )
    if tuple(fact.fact_id for fact in facts) != tuple(
        sorted((fact.fact_id for fact in facts), key=_byte_key)
    ):
        raise ValueError("catalog semantic facts are not canonical")
    return facts


def _encoded_graph_core(
    record: CatalogRecord,
    rendered: RenderedRecord,
    facts: tuple[SemanticFact, ...],
) -> tuple[tuple[int, ...], tuple[TokenSemanticSpan, ...], object]:
    if not isinstance(rendered, RenderedRecord):
        raise ValueError("graph source must emit RenderedRecord values")
    if type(rendered.segments) is not tuple or not rendered.segments:
        raise ValueError("graph source record must contain segments")
    fact_by_id = {fact.fact_id: fact for fact in facts}
    fields: list[SupervisedField] = []
    segments: list[TaggedSegment] = []
    for index, segment in enumerate(rendered.segments):
        if not isinstance(segment, TaggedSegment):
            raise ValueError("graph source record contains an invalid segment")
        text = _strict_text(segment.text, "graph segment text")
        if segment.role not in _SEGMENT_ROLES:
            raise ValueError(f"graph segment role is not supported: {segment.role}")
        if segment.role == "payload":
            _validate_payload_text(text)
            if segment.fact_id not in fact_by_id:
                raise ValueError("graph payload fact ID is outside catalog authority")
        fields.append(SupervisedField(f"segment-{index:08d}", text))
        segments.append(segment)
    fields_tuple = tuple(fields)
    plan = plan_occurrence_closure(facts, fields_tuple)
    occurrences_by_field: dict[str, list[object]] = {
        field.field_id: [] for field in fields_tuple
    }
    for occurrence in plan.occurrences:
        occurrences_by_field[occurrence.field_id].append(occurrence)

    tok = get_tok()
    token_ids: list[int] = []
    spans: list[TokenSemanticSpan] = []
    boundary_offsets: dict[str, dict[int, int]] = {}
    field_bases: dict[str, int] = {}
    for segment, field in zip(segments, fields_tuple, strict=True):
        field_bases[field.field_id] = len(token_ids)
        boundaries = {0, len(field.text)}
        for occurrence in occurrences_by_field[field.field_id]:
            boundaries.add(occurrence.start)
            boundaries.add(occurrence.end)
        ordered = sorted(boundaries)
        offsets = {0: 0}
        local_ids: list[int] = []
        for start, end in zip(ordered, ordered[1:]):
            local_ids.extend(tok.encode(field.text[start:end]))
            offsets[end] = len(local_ids)
        boundary_offsets[field.field_id] = offsets
        token_start = len(token_ids)
        token_ids.extend(local_ids)
        token_end = len(token_ids)
        if token_end <= token_start:
            raise ValueError("graph segment encodes to an empty semantic core")
        role = cast(str, _SEGMENT_ROLES[segment.role])
        spans.append(
            TokenSemanticSpan(
                token_start,
                token_end,
                segment.fact_id if segment.role == "payload" else None,
                role,  # type: ignore[arg-type]
            )
        )
    if len(token_ids) > record.target_count - 1:
        raise ValueError("graph semantic core cannot fit the catalog target count")

    bindings = tuple(
        TokenOccurrenceBinding(
            field_id=occurrence.field_id,
            char_start=occurrence.start,
            char_end=occurrence.end,
            fact_id=occurrence.fact_id,
            surface=occurrence.surface,
            token_start=(
                field_bases[occurrence.field_id]
                + boundary_offsets[occurrence.field_id][occurrence.start]
            ),
            token_end=(
                field_bases[occurrence.field_id]
                + boundary_offsets[occurrence.field_id][occurrence.end]
            ),
        )
        for occurrence in plan.occurrences
    )
    closure = build_occurrence_closure_ledger(
        token_count=record.target_count,
        facts=facts,
        fields=fields_tuple,
        bindings=bindings,
    )
    return tuple(token_ids), tuple(spans), closure


def _render_graph(
    record: CatalogRecord,
    routes: RouteIndex,
    rendered: RenderedRecord,
    facts: tuple[SemanticFact, ...],
) -> "ProductionRenderedRecord":
    core, core_spans, closure = _encoded_graph_core(record, rendered, facts)
    available = record.target_count - 1
    if len(core) > available:
        raise ValueError("graph semantic core cannot fit the catalog target count")
    tok = get_tok()
    padding_count = available - len(core)
    padding_tokens = tuple(
        tok.GRAPH_NOOP if index % 2 == 0 else tok.GRAPH_STEP
        for index in range(padding_count)
    )
    token_ids = (*core, *padding_tokens, tok.EOT)
    spans = (
        *core_spans,
        TokenSemanticSpan(len(core), len(token_ids), None, "schema"),
    )
    weights = derive_sidecar_weights(
        len(token_ids),
        spans,
        routes,
        closure=closure,
    )
    if weights.leaks:
        raise ValueError("semantic sidecar leak closure failed")
    return ProductionRenderedRecord(
        record_id=record.record_id,
        lane_id=record.lane_id,
        token_ids=tuple(token_ids),
        dense_target_weights=weights.dense,
        split90_target_weights=weights.split90,
        semantic_spans=tuple(spans),
        proofs=(),
        flags=record.semantic_flags,
        semantic_leaks=tuple(dict(leak) for leak in weights.leaks),
    )


def _wikidata_label(view: WikidataDerivedView, canonical_id: str) -> str:
    alias = lookup_alias(view, canonical_id)
    if alias is None:
        return canonical_id
    return _strict_text(alias.display, "Wikidata display alias")


def _wikidata_payload_text(
    view: WikidataDerivedView,
    triple: V2TrainingTriple,
) -> str:
    subject = f"Q{triple.subject}"
    object_id = f"Q{triple.object}"
    return json.dumps(
        {
            "object": object_id,
            "object_label": _wikidata_label(view, object_id),
            "relation": triple.relation,
            "relation_label": _wikidata_label(view, triple.relation),
            "split": triple.training_split,
            "subject": subject,
            "subject_label": _wikidata_label(view, subject),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _wikidata_edge_key(triple: V2TrainingTriple) -> str:
    return f"Q{triple.subject}\t{triple.relation}\tQ{triple.object}"


def _wikidata_edge_fact_id(triple: V2TrainingTriple) -> str:
    return f"wikidata:Q{triple.subject}:{triple.relation}:Q{triple.object}"


@dataclass(frozen=True)
class ProofEnvelope:
    family: str
    premise_bytes: bytes
    proof_bytes: bytes
    proof_sha256: str
    replay_verified: bool


@dataclass(frozen=True)
class ProductionRenderedRecord:
    record_id: str
    lane_id: LaneId
    token_ids: tuple[int, ...]
    dense_target_weights: bytes
    split90_target_weights: bytes
    semantic_spans: tuple[TokenSemanticSpan, ...]
    proofs: tuple[ProofEnvelope, ...]
    flags: tuple[str, ...]
    semantic_leaks: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or _SHA256_RE.fullmatch(self.record_id) is None:
            raise ValueError("rendered record ID must be a lowercase SHA-256")
        if self.lane_id not in LANE_ORDER:
            raise ValueError("rendered record lane is unknown")
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise ValueError("rendered token IDs must be a nonempty tuple")
        if any(
            type(token) is not int or token < 0 or token >= 65_536
            for token in self.token_ids
        ):
            raise ValueError("rendered token ID is outside uint16 bounds")
        count = len(self.token_ids)
        if (
            type(self.dense_target_weights) is not bytes
            or type(self.split90_target_weights) is not bytes
            or len(self.dense_target_weights) != count
            or len(self.split90_target_weights) != count
        ):
            raise ValueError("rendered sidecar length differs from token count")
        if any(value != 1 for value in self.dense_target_weights):
            raise ValueError("rendered Dense sidecar must contain only one")
        if any(value not in (0, 1) for value in self.split90_target_weights):
            raise ValueError("rendered Split90 sidecar must be binary")
        if type(self.semantic_spans) is not tuple or any(
            not isinstance(span, TokenSemanticSpan) or span.token_end > count
            for span in self.semantic_spans
        ):
            raise ValueError("rendered semantic span exceeds token bounds")
        if type(self.proofs) is not tuple or any(
            not isinstance(proof, ProofEnvelope) for proof in self.proofs
        ):
            raise ValueError("rendered proofs must be ProofEnvelope values")
        if type(self.flags) is not tuple:
            raise ValueError("rendered flags must be a tuple")
        for flag in self.flags:
            _strict_text(flag, "rendered flag")
        if (
            len(self.flags) != len(set(self.flags))
            or self.flags != tuple(sorted(self.flags, key=_byte_key))
        ):
            raise ValueError("rendered flags must be unique and canonical")
        if type(self.semantic_leaks) is not tuple:
            raise ValueError("rendered semantic leaks must be a tuple")
        if self.semantic_leaks:
            raise ValueError("rendered record contains semantic sidecar leaks")


class ProductionLaneRenderer(Protocol):
    lane_id: LaneId
    renderer_version: str

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        raise RuntimeError("production lane renderer must render one catalog record")


class _ExposureRenderer:
    lane_id: LaneId
    renderer_version: str
    allowed_sources: frozenset[str]

    def __init__(self, source_root: Path) -> None:
        if not isinstance(source_root, Path):
            raise TypeError("source_root must be a pathlib.Path")
        self.source_root = source_root

    def _validate(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> dict[str, str | int]:
        if not isinstance(routes, RouteIndex):
            raise TypeError("routes must be a RouteIndex")
        return _validate_catalog_record(record, self.lane_id, self.allowed_sources)


class FineWebEduRenderer(_ExposureRenderer):
    lane_id: LaneId = "fineweb_edu"
    renderer_version = "fineweb-edu-nfc-gpt2-v1"
    allowed_sources = frozenset({"fineweb_edu"})

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        relative = _source_relative(locator)
        if relative not in _FINEWEB_PATHS:
            raise ValueError("FineWeb path is outside the three locked 10BT files")
        path, identity = _verify_source_hash(self.source_root, record, relative)
        text = _read_locked_text(
            path,
            _row_index(locator),
            identity=identity,
        )
        return _render_text(record, routes, text)


class FineMathRenderer(_ExposureRenderer):
    lane_id: LaneId = "finemath"
    renderer_version = "finemath-cross-dedup-gpt2-v1"
    allowed_sources = frozenset({"finemath"})

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        relative = _source_relative(locator)
        if _FINEMATH_PATH_RE.fullmatch(relative) is None:
            raise ValueError("FineMath path is outside locked 4plus/3plus shards")
        path, identity = _verify_source_hash(self.source_root, record, relative)
        text = _select_finemath_text(
            self.source_root,
            relative,
            _row_index(locator),
            path,
            identity,
        )
        return _render_text(record, routes, text)


class WikidataGraphRenderer(_ExposureRenderer):
    lane_id: LaneId = "wikidata_graph"
    renderer_version = "wikidata-training-graph-v1"
    allowed_sources = frozenset({"wikidata5m"})

    def __init__(
        self,
        source_root: Path,
        wikidata_view: WikidataDerivedView,
    ) -> None:
        super().__init__(source_root)
        if not isinstance(wikidata_view, WikidataDerivedView):
            raise TypeError("wikidata_view must be a WikidataDerivedView")
        # An indexed miss is sufficient to prove this exact object was returned by
        # the verified-view builder; caller-constructed dataclasses have no
        # registered authority and are rejected by lookup_alias.
        lookup_alias(wikidata_view, "Q0")
        self._view = wikidata_view
        archives = []
        for artifact in wikidata_view.receipt.archives:
            relative = _safe_relative_path(
                artifact.path,
                "Wikidata receipt archive path",
            )
            path = _source_path(source_root, "wikidata5m", relative)
            digest, identity = _hash_regular_file(
                path,
                "verified Wikidata source archive",
            )
            if digest != artifact.sha256:
                raise ValueError(
                    "Wikidata source archive does not match the verified view"
                )
            archives.append((relative, artifact.sha256, path, identity))
        self._archives = tuple(archives)

    def _archive(
        self,
        relative: str,
    ) -> tuple[str, Path, tuple[int, ...]]:
        for path_name, sha256, path, identity in self._archives:
            if path_name == relative:
                return sha256, path, identity
        raise ValueError("Wikidata archive path is outside the verified view")

    def _reverify_archives(self, boundary: str) -> None:
        for relative, _sha256, path, identity in self._archives:
            _check_regular_identity(
                path,
                identity,
                f"verified Wikidata source archive {relative} {boundary}",
            )

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        if frozenset(locator) != _WIKIDATA_LOCATOR_KEYS:
            raise ValueError("Wikidata locator fields do not match the closed contract")
        if record.semantic_facts:
            raise ValueError(
                "Wikidata graph catalog records must not carry derived semantic facts"
            )
        if record.semantic_flags not in (
            ("graph-training-edge",),
            ("graph-revisit",),
        ):
            raise ValueError(
                "Wikidata graph requires exactly one training-edge or revisit flag"
            )

        split = _strict_text(locator["split"], "Wikidata locator split")
        if split != "train":
            raise ValueError("Wikidata locator split must be the literal train split")
        training_split = _strict_text(
            locator["training_split"],
            "Wikidata training split",
        )
        if training_split not in TRAINING_SPLITS:
            raise ValueError("Wikidata training split is outside the indexed contract")
        member = _safe_relative_path(
            locator["member"],
            "Wikidata decoded member",
        )
        archive_path = _safe_relative_path(
            locator["path"],
            "Wikidata archive path",
        )
        edge_key = _strict_text(
            locator["training_edge_key"],
            "Wikidata training-edge key",
        )
        view_sha256 = _strict_text(
            locator["wikidata_view_sha256"],
            "Wikidata view commitment",
        )
        row = locator["row"]
        if type(row) is not int or row <= 0:
            raise ValueError("Wikidata source row must be a positive integer")

        view = self._view
        if view_sha256 != view.receipt_sha256:
            raise ValueError(
                "Wikidata record does not bind the verified derived view"
            )
        expected_member = f"wikidata5m_{training_split}.txt"
        expected_archive = (
            f"wikidata5m_{training_split.removesuffix('_train')}.tar.gz"
        )
        if member != expected_member:
            raise ValueError(
                "Wikidata locator member does not match its training split"
            )
        if archive_path != expected_archive:
            raise ValueError(
                "Wikidata locator archive path does not match its training split"
            )

        archive_sha256 = self._archive(archive_path)[0]
        if record.source_byte_sha256 != archive_sha256:
            raise ValueError(
                "Wikidata record archive SHA-256 does not match the verified view"
            )
        self._reverify_archives("before render")

        try:
            triple = lookup_training_triple(view, training_split, row)
        except IndexError as error:
            raise ValueError(
                "Wikidata row is outside the verified indexed training split"
            ) from error
        if triple.member != member:
            raise ValueError(
                "Wikidata locator member does not match the indexed training triple"
            )
        if triple.archive_path != archive_path:
            raise ValueError(
                "Wikidata locator archive does not match the indexed training triple"
            )
        if _wikidata_edge_key(triple) != edge_key:
            raise ValueError(
                "Wikidata locator training-edge key does not match the indexed triple"
            )

        text = _wikidata_payload_text(view, triple)
        fact = SemanticFact(_wikidata_edge_fact_id(triple), (text,))
        rendered = _render_graph(
            record,
            routes,
            RenderedRecord(
                segments=(TaggedSegment(text, "payload", fact.fact_id),),
                schedule=ScheduleEntry(
                    component="wikidata_graph",
                    record_id=record.source_key,
                    exposure=record.ordinal,
                    curriculum_band=0,
                ),
            ),
            (fact,),
        )

        # Repeat an O(1) indexed row read after rendering so view authority is
        # checked on both sides of tokenization and sidecar construction.
        if lookup_training_triple(view, training_split, row) != triple:
            raise ValueError("Wikidata indexed training triple changed during render")
        self._reverify_archives("after render")
        return rendered


class SyntheticGraphRenderer(_ExposureRenderer):
    lane_id: LaneId = "synthetic_graph"
    renderer_version = "srgm-seeded-graph-v1"
    allowed_sources = frozenset(
        {"deepmind_mathematics_generator", "reasoning_gym_exact_answer"}
    )

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        relative = _source_relative(locator)
        path, identity = _verify_source_hash(self.source_root, record, relative)
        seed = locator.get("generation_seed")
        exposure = locator.get("graph_exposure")
        if type(seed) is not int or seed < 0 or seed >= 1 << 63:
            raise ValueError("synthetic graph requires the exact catalog seed")
        if type(exposure) is not int or exposure < 0:
            raise ValueError("synthetic graph exposure must be non-negative")

        def worlds_factory() -> Iterable[object]:
            return iter_worlds(
                _SYNTHETIC_WORLD_ENTITIES,
                _SYNTHETIC_WORLD_SIZE,
                seed,
                _SYNTHETIC_WORLD_ID_OFFSET,
            )

        source_records = iter(iter_graph_records(get_tok(), worlds_factory))
        seen_ids: set[str] = set()
        selected = None
        for expected_exposure in range(exposure + 1):
            try:
                candidate = next(source_records)
            except StopIteration as error:
                raise ValueError(
                    "synthetic graph source ended before catalog exposure"
                ) from error
            if not isinstance(candidate, RenderedRecord):
                raise ValueError("synthetic graph source emitted an invalid record")
            schedule = candidate.schedule
            record_id = _strict_text(
                getattr(schedule, "record_id", None),
                "synthetic source record ID",
            )
            if record_id in seen_ids:
                raise ValueError(f"duplicate synthetic source record ID: {record_id}")
            seen_ids.add(record_id)
            if getattr(schedule, "exposure", None) != expected_exposure:
                raise ValueError("synthetic graph source exposure order drift")
            selected = candidate
        if selected is None:
            raise AssertionError("synthetic graph selection is unreachable")
        if selected.schedule.record_id != record.source_key:
            raise ValueError("synthetic source record ID differs from catalog key")
        _reverify_identity(path, identity)
        return _render_graph(record, routes, selected, _semantic_facts(record))


EXPOSURE_RENDERER_TYPES = (
    ("fineweb_edu", FineWebEduRenderer, FineWebEduRenderer.renderer_version),
    ("finemath", FineMathRenderer, FineMathRenderer.renderer_version),
    ("wikidata_graph", WikidataGraphRenderer, WikidataGraphRenderer.renderer_version),
    ("synthetic_graph", SyntheticGraphRenderer, SyntheticGraphRenderer.renderer_version),
)


__all__ = (
    "EXPOSURE_RENDERER_TYPES",
    "FineMathRenderer",
    "FineWebEduRenderer",
    "ProductionLaneRenderer",
    "ProductionRenderedRecord",
    "ProofEnvelope",
    "SyntheticGraphRenderer",
    "WikidataGraphRenderer",
)
