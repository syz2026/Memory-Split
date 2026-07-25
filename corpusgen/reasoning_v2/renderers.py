"""Deterministic production renderers for the four exposure lanes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from corpusgen.graph_records import RenderedRecord, ScheduleEntry, TaggedSegment
from corpusgen.reasoning import (
    AnswerPointer,
    CompositionPremise,
    EqualityPremise,
    ProofObject,
    SemanticFact,
    SupervisedField,
    plan_occurrence_closure,
    serialize_answer_state,
    solve_graph_composition,
    solve_slot_equality,
    verify_proof,
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
    audit_answer_state_surfaces,
    audit_proof_surfaces,
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
from corpusgen.srgm_worlds import (
    CURRICULUM_HOPS,
    iter_graph_records,
    iter_reasoning_records,
    iter_worlds,
)
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
_SEGMENT_ROLES = {
    "plain": "plain_text",
    "payload": "factual_payload",
    "random_control": "plain_text",
    "rule": "rule",
    "action": "operator",
    "query": "schema",
    "proof": "proof",
    "candidate_state": "answer_state",
    "provisional_answer": "answer_state",
    "final_answer": "answer_state",
}
_REASONING_ANSWER_SLOT = 0
_REASONING_MAX_READ_INDEX = 11


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
    keys = []
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


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
    )


def _hash_regular_file(path: Path, description: str) -> tuple[str, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        named = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} is missing or unsafe") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(named.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or named.st_nlink != 1
            or opened.st_nlink != 1
            or _file_identity(named) != _file_identity(opened)
        ):
            raise ValueError(f"{description} is not an exact regular source file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"{description} identity drift while reading")
        return digest.hexdigest(), _file_identity(opened)
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
    record: CatalogRecord,
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
) -> tuple[
    tuple[int, ...],
    tuple[TokenSemanticSpan, ...],
    object,
]:
    if not isinstance(rendered, RenderedRecord):
        raise ValueError("graph source must emit RenderedRecord values")
    if type(rendered.segments) is not tuple or not rendered.segments:
        raise ValueError("graph source record must contain segments")
    fact_by_id = {fact.fact_id: fact for fact in facts}
    fields = []
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


def _view_label(view: WikidataDerivedView, canonical_id: str) -> str:
    record = lookup_alias(view, canonical_id)
    if record is None:
        return canonical_id
    return _strict_text(record.display, "Wikidata display alias")


def _wikidata_payload_text(
    view: WikidataDerivedView,
    triple: V2TrainingTriple,
) -> str:
    subject = f"Q{triple.subject}"
    object_id = f"Q{triple.object}"
    return json.dumps(
        {
            "object": object_id,
            "object_label": _view_label(view, object_id),
            "relation": triple.relation,
            "relation_label": _view_label(view, triple.relation),
            "split": triple.training_split,
            "subject": subject,
            "subject_label": _view_label(view, subject),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


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
            not isinstance(span, TokenSemanticSpan)
            or span.token_end > count
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
            record,
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
            raise TypeError(
                "WikidataGraphRenderer requires a live WikidataDerivedView "
                "session, not a data-only reference"
            )
        wikidata_view._require_open()
        self._view = wikidata_view

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        view = self._view
        view._require_open()

        split = _strict_text(locator.get("split"), "Wikidata locator split")
        if split != "train":
            raise ValueError(
                "Wikidata locator split must be the literal train split"
            )
        training_split = _strict_text(
            locator.get("training_split"),
            "Wikidata training split",
        )
        member = _strict_text(locator.get("member"), "Wikidata decoded member")
        archive_path = _strict_text(locator.get("path"), "Wikidata archive path")
        edge_key = _strict_text(
            locator.get("training_edge_key"),
            "Wikidata training-edge key",
        )
        view_sha256 = _strict_text(
            locator.get("wikidata_view_sha256"),
            "Wikidata view commitment",
        )
        row_number = locator.get("row")
        if type(row_number) is not int or row_number <= 0:
            raise ValueError("Wikidata source row must be a positive integer")

        if view_sha256 != view.receipt_sha256:
            raise ValueError(
                "Wikidata record does not bind the verified derived view"
            )
        archive_sha256 = {
            artifact.path: artifact.sha256 for artifact in view.receipt.archives
        }.get(archive_path)
        if archive_sha256 is None:
            raise ValueError(
                "Wikidata archive path is outside the verified derived view"
            )
        if record.source_byte_sha256 != archive_sha256:
            raise ValueError(
                "Wikidata record archive SHA-256 does not match the locked view"
            )

        graph_flags = {
            flag
            for flag in record.semantic_flags
            if flag in {"graph-training-edge", "graph-revisit"}
        }
        if len(graph_flags) != 1:
            raise ValueError(
                "Wikidata graph requires exactly one training-edge/revisit flag"
            )

        triple = lookup_training_triple(view, training_split, row_number)
        if triple.member != member:
            raise ValueError(
                "Wikidata locator member does not match the verified training triple"
            )
        if triple.archive_path != archive_path:
            raise ValueError(
                "Wikidata locator archive does not match the verified training triple"
            )
        actual_edge_key = (
            f"Q{triple.subject}\t{triple.relation}\tQ{triple.object}"
        )
        if actual_edge_key != edge_key:
            raise ValueError(
                "Wikidata locator training-edge key does not match the verified "
                "triple"
            )

        text = _wikidata_payload_text(view, triple)
        fact = SemanticFact(_wikidata_edge_fact_id(triple), (text,))
        view._require_open()
        rendered = RenderedRecord(
            segments=(TaggedSegment(text, "payload", fact.fact_id),),
            schedule=ScheduleEntry(
                component="wikidata_graph",
                record_id=record.source_key,
                exposure=record.ordinal,
                curriculum_band=0,
            ),
        )
        return _render_graph(record, routes, rendered, (fact,))


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
                raise ValueError("synthetic graph source ended before catalog exposure") from error
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


# ---------------------------------------------------------------------------
# Task 5B: solver-backed reasoning lanes
#
# Each reasoning renderer derives canonical premises, calls the reviewed
# finite-domain solvers, serializes canonical premise and proof bytes into a
# ProofEnvelope, and independently replays both before returning. Proof tokens
# carry the ``proof`` role, only factual returns carry ``factual_payload``, and
# candidate/final answer states carry only value-free ``AnswerPointer`` bytes.
# ---------------------------------------------------------------------------


def _premise_as_dict(premise: object) -> dict[str, object]:
    if isinstance(premise, CompositionPremise):
        return {
            "compose_code": premise.compose_code,
            "fact_id": premise.fact_id,
            "hop": premise.hop,
            "type": "composition",
        }
    if isinstance(premise, EqualityPremise):
        return {
            "fact_id": premise.fact_id,
            "slot": premise.slot,
            "type": "equality",
            "value": premise.value,
        }
    raise TypeError("unsupported reasoning premise")


def _canonical_premise_bytes(premises: tuple[object, ...]) -> bytes:
    return (
        json.dumps(
            [_premise_as_dict(premise) for premise in premises],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _premises_from_bytes(family: str, premise_bytes: bytes) -> tuple[object, ...]:
    value = json.loads(premise_bytes.decode("utf-8"))
    if type(value) is not list or not value:
        raise ValueError("reasoning premise bytes must be a nonempty list")
    premises: list[object] = []
    for item in value:
        if type(item) is not dict:
            raise ValueError("reasoning premise must be a JSON object")
        if family == "graph_composition_mod4":
            if (
                set(item) != {"compose_code", "fact_id", "hop", "type"}
                or item["type"] != "composition"
            ):
                raise ValueError("invalid composition premise bytes")
            premises.append(
                CompositionPremise(
                    fact_id=item["fact_id"],
                    hop=item["hop"],
                    compose_code=item["compose_code"],
                )
            )
        elif family == "slot_equality":
            if (
                set(item) != {"fact_id", "slot", "type", "value"}
                or item["type"] != "equality"
            ):
                raise ValueError("invalid equality premise bytes")
            premises.append(
                EqualityPremise(
                    fact_id=item["fact_id"],
                    slot=item["slot"],
                    value=item["value"],
                )
            )
        else:
            raise ValueError(f"unsupported reasoning proof family: {family}")
    return tuple(premises)


def _solve_reasoning_proof(family: str, premises: tuple[object, ...]) -> ProofObject:
    if family == "graph_composition_mod4":
        return solve_graph_composition(premises)
    if family == "slot_equality":
        return solve_slot_equality(premises)
    raise ValueError(f"unsupported reasoning proof family: {family}")


def _build_proof_envelope(
    family: str,
    premises: tuple[object, ...],
) -> tuple[ProofEnvelope, ProofObject]:
    proof = _solve_reasoning_proof(family, premises)
    if proof.family != family:
        raise ValueError("reasoning solver returned a mismatched proof family")
    proof_bytes = proof.to_bytes()
    premise_bytes = _canonical_premise_bytes(premises)
    replayed_premises = _premises_from_bytes(family, premise_bytes)
    replayed = _solve_reasoning_proof(family, replayed_premises)
    if replayed.to_bytes() != proof_bytes:
        raise ValueError("reasoning premise/proof canonical replay disagreement")
    if not verify_proof(proof, replayed_premises):
        raise ValueError("reasoning solver proof failed independent verification")
    envelope = ProofEnvelope(
        family=family,
        premise_bytes=premise_bytes,
        proof_bytes=proof_bytes,
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        replay_verified=True,
    )
    return envelope, proof


def _answer_pointer_segment(read_index: int, phase: str, role: str) -> TaggedSegment:
    pointer = AnswerPointer(
        slot=_REASONING_ANSWER_SLOT,
        read_index=min(read_index, _REASONING_MAX_READ_INDEX),
        member_index=0,
    )
    return TaggedSegment(serialize_answer_state(pointer, phase=phase), role)


def _render_reasoning(
    record: CatalogRecord,
    routes: RouteIndex,
    *,
    segments: tuple[TaggedSegment, ...],
    facts: tuple[SemanticFact, ...],
    envelopes: tuple[ProofEnvelope, ...],
) -> "ProductionRenderedRecord":
    if not envelopes:
        raise ValueError("reasoning record requires at least one proof envelope")
    rendered = RenderedRecord(
        segments=segments,
        schedule=ScheduleEntry(
            component="reasoning",
            record_id=record.source_key,
            exposure=record.ordinal,
            curriculum_band=0,
        ),
    )
    core, core_spans, closure = _encoded_graph_core(record, rendered, facts)
    available = record.target_count - 1
    if len(core) > available:
        raise ValueError("reasoning semantic core cannot fit the catalog target count")
    for segment in segments:
        if segment.role == "proof":
            audit_proof_surfaces(_strict_text(segment.text, "proof segment"), facts)
        elif segment.role in ("candidate_state", "provisional_answer", "final_answer"):
            audit_answer_state_surfaces(
                _strict_text(segment.text, "answer-state segment"),
                facts,
            )
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
        raise ValueError("reasoning sidecar leak closure failed")
    return ProductionRenderedRecord(
        record_id=record.record_id,
        lane_id=record.lane_id,
        token_ids=tuple(token_ids),
        dense_target_weights=weights.dense,
        split90_target_weights=weights.split90,
        semantic_spans=tuple(spans),
        proofs=tuple(envelopes),
        flags=record.semantic_flags,
        semantic_leaks=tuple(dict(leak) for leak in weights.leaks),
    )


def _reasoning_seed(locator: dict[str, str | int]) -> int:
    seed = locator.get("generation_seed")
    if type(seed) is not int or seed < 0 or seed >= 1 << 63:
        raise ValueError("reasoning lane requires the exact catalog generation seed")
    return seed


def _reasoning_exposure(locator: dict[str, str | int]) -> int:
    exposure = locator.get("graph_exposure")
    if type(exposure) is not int or exposure < 0:
        raise ValueError("reasoning graph exposure must be a non-negative integer")
    return exposure


def _reasoning_hops(locator: dict[str, str | int]) -> int:
    hops = locator.get("reasoning_hops")
    if hops not in CURRICULUM_HOPS:
        raise ValueError("reasoning hops must be a frozen curriculum band")
    return cast(int, hops)


def _reasoning_family_for_answer(answer: str) -> str | None:
    if re.fullmatch(r"r[0-3]", answer) is not None:
        return "graph_composition_mod4"
    if answer in ("yes", "no"):
        return "slot_equality"
    return None


def _reasoning_answer_family(answer: str) -> str:
    family = _reasoning_family_for_answer(answer)
    if family is None:
        raise ValueError("reasoning answer is not a supported solver family")
    return family


def _supported_reasoning_family(rendered: RenderedRecord) -> str | None:
    """Return the frozen solver family for a reasoning record, or None.

    None marks a record whose final answer has no Task 5B solver (notably
    ``date_ordering``, answered with a ``<|slot_N|>`` pointer).
    """

    if not isinstance(rendered, RenderedRecord) or not rendered.segments:
        return None
    answer = rendered.segments[-1].text
    if type(answer) is not str:
        return None
    return _reasoning_family_for_answer(answer)


def iter_supported_reasoning_records(
    tok: object,
    worlds_factory: Callable[[], Iterable[object]],
    *,
    seed: int,
    max_hops: int,
) -> Iterator[tuple[int, RenderedRecord]]:
    """Canonical solver-backed reasoning exposure sequence.

    ``iter_reasoning_records`` interleaves three SRGM tasks, but Task 5B has a
    frozen solver only for ``path_composition`` (answer ``rN``) and
    ``balanced_equality`` (answer ``yes``/``no``). This iterator deterministically
    skips every record whose final answer has no frozen solver (notably
    ``date_ordering``) and numbers the surviving composition/equality records
    contiguously from zero, so an unsolved record never consumes a lane exposure.
    Each yielded record keeps its raw ``ScheduleEntry`` (source ``record_id``, raw
    ``exposure``, ``curriculum_band``) unchanged as provenance.

    Contract: this is the single source of truth for ``verified_synthetic_multihop``
    exposure numbering. ``VerifiedSyntheticMultihopRenderer._select`` consumes it,
    and any future production catalog source that allocates exposures for this lane
    MUST consume this exact sequence so its numbering matches the renderer.
    """

    lane_exposure = 0
    for record in iter_reasoning_records(
        tok, worlds_factory, seed=seed, max_hops=max_hops
    ):
        if not isinstance(record, RenderedRecord):
            raise ValueError("reasoning source emitted an invalid record")
        if _supported_reasoning_family(record) is None:
            continue
        yield lane_exposure, record
        lane_exposure += 1


class VerifiedSyntheticMultihopRenderer(_ExposureRenderer):
    lane_id: LaneId = "verified_synthetic_multihop"
    renderer_version = "srgm-canonical-proof-v1"
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
        seed = _reasoning_seed(locator)
        exposure = _reasoning_exposure(locator)
        hops = _reasoning_hops(locator)

        reasoning = self._select(seed, exposure, hops, record.source_key)
        family, premises, ordered = self._analyze(reasoning, record)
        envelope, proof = _build_proof_envelope(family, premises)
        self._check_oracle(family, proof, reasoning.segments[-1].text)
        segments = self._compose(family, ordered, proof)
        _reverify_identity(path, identity)
        return _render_reasoning(
            record,
            routes,
            segments=segments,
            facts=_semantic_facts(record),
            envelopes=(envelope,),
        )

    def _select(
        self,
        seed: int,
        exposure: int,
        hops: int,
        expected_key: str,
    ) -> RenderedRecord:
        def worlds_factory() -> Iterable[object]:
            return iter_worlds(
                _SYNTHETIC_WORLD_ENTITIES,
                _SYNTHETIC_WORLD_SIZE,
                seed,
                _SYNTHETIC_WORLD_ID_OFFSET,
            )

        supported = iter(
            iter_supported_reasoning_records(
                get_tok(),
                worlds_factory,
                seed=seed,
                max_hops=hops,
            )
        )
        seen_ids: set[str] = set()
        selected: RenderedRecord | None = None
        for expected_exposure in range(exposure + 1):
            try:
                lane_exposure, candidate = next(supported)
            except StopIteration as error:
                raise ValueError(
                    "verified synthetic multihop source ended before catalog exposure"
                ) from error
            schedule = candidate.schedule
            record_id = _strict_text(
                getattr(schedule, "record_id", None),
                "reasoning source record ID",
            )
            if record_id in seen_ids:
                raise ValueError(f"duplicate reasoning source record ID: {record_id}")
            seen_ids.add(record_id)
            if lane_exposure != expected_exposure:
                raise ValueError(
                    "verified synthetic multihop filtered exposure order drift"
                )
            if getattr(schedule, "curriculum_band", None) != hops:
                raise ValueError("verified synthetic multihop curriculum band drift")
            selected = candidate
        if selected is None:
            raise AssertionError("verified synthetic multihop selection is unreachable")
        if selected.schedule.record_id != expected_key:
            raise ValueError(
                "verified synthetic multihop record ID differs from catalog key"
            )
        return selected

    def _analyze(
        self,
        reasoning: RenderedRecord,
        record: CatalogRecord,
    ) -> tuple[str, tuple[object, ...], tuple[tuple[str, str, dict], ...]]:
        ordered: list[tuple[str, str, dict]] = []
        for segment in reasoning.segments:
            if segment.role != "payload":
                continue
            text = _strict_text(segment.text, "reasoning return payload")
            fact_id = _strict_text(segment.fact_id, "reasoning return fact ID")
            try:
                payload = json.loads(text, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as error:
                raise ValueError("reasoning return payload is not valid JSON") from error
            _require_finite(payload, "reasoning return payload")
            ordered.append((fact_id, text, payload))
        if not ordered:
            raise ValueError("reasoning record has no factual returns")
        fact_ids = [fact_id for fact_id, _text, _payload in ordered]
        texts = [text for _fact_id, text, _payload in ordered]
        if len(set(fact_ids)) != len(fact_ids) or len(set(texts)) != len(texts):
            raise ValueError("reasoning returns must be distinct")

        answer = _strict_text(reasoning.segments[-1].text, "reasoning answer")
        family = _reasoning_answer_family(answer)
        if family == "graph_composition_mod4":
            premises: tuple[object, ...] = tuple(
                CompositionPremise(
                    fact_id,
                    hop=index,
                    compose_code=self._compose_code(payload),
                )
                for index, (fact_id, _text, payload) in enumerate(ordered)
            )
        else:
            if len(ordered) != 2:
                raise ValueError("slot equality requires exactly two factual returns")
            premises = tuple(
                EqualityPremise(
                    fact_id,
                    slot=index,
                    value=self._equality_value(payload),
                )
                for index, (fact_id, _text, payload) in enumerate(ordered)
            )

        record_facts = {
            (fact.fact_id, fact.surfaces) for fact in _semantic_facts(record)
        }
        trace_facts = {(fact_id, (text,)) for fact_id, text, _payload in ordered}
        if record_facts != trace_facts:
            raise ValueError(
                "catalog record facts do not match the reasoning trace returns"
            )
        return family, premises, tuple(ordered)

    @staticmethod
    def _compose_code(payload: object) -> int:
        if type(payload) is not dict or "qualifiers" not in payload:
            raise ValueError("composition return lacks a compose qualifier")
        qualifiers = dict(
            (str(key), str(value)) for key, value in payload["qualifiers"]
        )
        if "compose" not in qualifiers:
            raise ValueError("composition return lacks a compose qualifier")
        code = int(qualifiers["compose"])
        if code not in range(4):
            raise ValueError("composition compose code is outside [0, 3]")
        return code

    @staticmethod
    def _equality_value(payload: object) -> str:
        if type(payload) is not dict or "target" not in payload:
            raise ValueError("equality return lacks a target value")
        return _strict_text(payload["target"], "equality return value")

    @staticmethod
    def _check_oracle(family: str, proof: ProofObject, answer: str) -> None:
        conclusion = dict(proof.conclusion)
        if family == "graph_composition_mod4":
            if conclusion.get("relation") != answer:
                raise ValueError(
                    "composition proof disagrees with the reasoning oracle answer"
                )
        elif conclusion.get("equal") is not (answer == "yes"):
            raise ValueError(
                "equality proof disagrees with the reasoning oracle answer"
            )

    @staticmethod
    def _compose(
        family: str,
        ordered: tuple[tuple[str, str, dict], ...],
        proof: ProofObject,
    ) -> tuple[TaggedSegment, ...]:
        segments: list[TaggedSegment] = [
            TaggedSegment(f"<|reasoning|>{family}|reads={len(ordered)}", "query")
        ]
        for index, (fact_id, text, _payload) in enumerate(ordered):
            segments.append(TaggedSegment(f"<|read|>step={index}", "action"))
            segments.append(TaggedSegment(text, "payload", fact_id))
        segments.append(TaggedSegment(proof.to_bytes().decode("utf-8"), "proof"))
        segments.append(
            _answer_pointer_segment(len(ordered), "candidate", "candidate_state")
        )
        segments.append(
            _answer_pointer_segment(len(ordered), "final", "final_answer")
        )
        return tuple(segments)


class RelationalRefinementRenderer(_ExposureRenderer):
    lane_id: LaneId = "relational_refinement"
    renderer_version = "pointer-state-refinement-v1"
    allowed_sources = frozenset(
        {"clrs_text", "prontoqa", "reasoning_gym_exact_answer", "ruletaker"}
    )

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        relative = _source_relative(locator)
        path, identity = _verify_source_hash(self.source_root, record, relative)
        facts = _semantic_facts(record)
        if len(facts) != 2:
            raise ValueError(
                "relational refinement requires exactly two relational facts"
            )
        premises = (
            EqualityPremise(
                facts[0].fact_id,
                slot=0,
                value=_strict_text(facts[0].surfaces[0], "relational value"),
            ),
            EqualityPremise(
                facts[1].fact_id,
                slot=1,
                value=_strict_text(facts[1].surfaces[0], "relational value"),
            ),
        )
        envelope, proof = _build_proof_envelope("slot_equality", premises)
        segments = self._compose(facts, proof)
        _reverify_identity(path, identity)
        return _render_reasoning(
            record,
            routes,
            segments=segments,
            facts=facts,
            envelopes=(envelope,),
        )

    @staticmethod
    def _compose(
        facts: tuple[SemanticFact, ...],
        proof: ProofObject,
    ) -> tuple[TaggedSegment, ...]:
        segments: list[TaggedSegment] = [
            TaggedSegment(f"<|refine|>slots={len(facts)}", "query")
        ]
        for index, fact in enumerate(facts):
            segments.append(TaggedSegment(f"<|slot_read|>slot={index}", "action"))
            segments.append(
                TaggedSegment(fact.surfaces[0], "payload", fact.fact_id)
            )
        for step in range(len(facts)):
            segments.append(
                _answer_pointer_segment(step, "candidate", "candidate_state")
            )
        segments.append(TaggedSegment(proof.to_bytes().decode("utf-8"), "proof"))
        segments.append(
            _answer_pointer_segment(len(facts), "final", "final_answer")
        )
        return tuple(segments)


def _wikidata_path_label(view: WikidataDerivedView, canonical_id: str) -> str:
    record = lookup_alias(view, canonical_id)
    if record is None:
        return canonical_id
    return _strict_text(record.display, "Wikidata display alias")


def _wikidata_path_return_surface(canonical_id: str, label: str) -> str:
    return json.dumps(
        {"entity": canonical_id, "label": label},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _wikidata_path_fact_id(subject: int, relation: str, object_id: int) -> str:
    return f"wikidata-path:Q{subject}:{relation}:Q{object_id}"


def _wikidata_edge_premise_id(triple: V2TrainingTriple) -> str:
    # Exact observed-training-edge provenance: split#row plus the canonical edge.
    return (
        f"{triple.training_split}#{triple.row}:"
        f"Q{triple.subject}:{triple.relation}:Q{triple.object}"
    )


class WikidataPathReasoningRenderer(_ExposureRenderer):
    lane_id: LaneId = "wikidata_path_reasoning"
    renderer_version = "wikidata-training-path-proof-v1"
    allowed_sources = frozenset({"wikidata5m"})

    def __init__(
        self,
        source_root: Path,
        wikidata_view: WikidataDerivedView,
    ) -> None:
        super().__init__(source_root)
        if not isinstance(wikidata_view, WikidataDerivedView):
            raise TypeError(
                "WikidataPathReasoningRenderer requires a live WikidataDerivedView "
                "session, not a data-only reference"
            )
        wikidata_view._require_open()
        self._view = wikidata_view

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        locator = self._validate(record, routes)
        view = self._view
        view._require_open()

        training_split = _strict_text(
            locator.get("training_split"),
            "Wikidata path training split",
        )
        if training_split not in TRAINING_SPLITS:
            raise ValueError(
                "Wikidata path split must be a frozen training split, not a sealed "
                "member"
            )
        member = _strict_text(locator.get("member"), "Wikidata path member")
        archive_path = _strict_text(locator.get("path"), "Wikidata path archive")
        view_sha256 = _strict_text(
            locator.get("wikidata_view_sha256"),
            "Wikidata view commitment",
        )
        rows = self._path_rows(locator.get("path_rows"))

        if view_sha256 != view.receipt_sha256:
            raise ValueError(
                "Wikidata path record does not bind the verified derived view"
            )
        archive_sha256 = {
            artifact.path: artifact.sha256 for artifact in view.receipt.archives
        }.get(archive_path)
        if archive_sha256 is None:
            raise ValueError(
                "Wikidata path archive is outside the verified derived view"
            )
        if record.source_byte_sha256 != archive_sha256:
            raise ValueError(
                "Wikidata path archive SHA-256 does not match the locked view"
            )

        triples: list[V2TrainingTriple] = []
        for row in rows:
            triple = lookup_training_triple(view, training_split, row)
            if triple.training_split != training_split:
                raise ValueError(
                    "Wikidata path triple is not from the requested training split"
                )
            if triple.member != member or triple.archive_path != archive_path:
                raise ValueError(
                    "Wikidata path triple does not match the locked member/archive"
                )
            triples.append(triple)
        adjacency = tuple(zip(triples, triples[1:], strict=False))
        for left, right in adjacency:
            if left.object != right.subject:
                raise ValueError("Wikidata path is not a connected training walk")
        if not adjacency:
            raise ValueError("Wikidata path reasoning requires at least one adjacency")

        # One canonical slot_equality proof per adjacent hop, grounding the walk
        # in the *observed* shared endpoint (Q<left.object> == Q<right.subject>)
        # rather than any hash of the relation PID. Distinct pointer slots and
        # exact edge/provenance premise IDs; every proof must conclude true.
        envelopes: list[ProofEnvelope] = []
        proofs: list[ProofObject] = []
        for left, right in adjacency:
            premises = (
                EqualityPremise(
                    _wikidata_edge_premise_id(left),
                    slot=0,
                    value=f"Q{left.object}",
                ),
                EqualityPremise(
                    _wikidata_edge_premise_id(right),
                    slot=1,
                    value=f"Q{right.subject}",
                ),
            )
            envelope, proof = _build_proof_envelope("slot_equality", premises)
            if dict(proof.conclusion).get("equal") is not True:
                raise ValueError(
                    "Wikidata path adjacency is not grounded in a shared endpoint"
                )
            envelopes.append(envelope)
            proofs.append(proof)

        ordered = self._returns(view, triples)
        record_facts = {
            (fact.fact_id, fact.surfaces) for fact in _semantic_facts(record)
        }
        if record_facts != {(fact_id, (surface,)) for fact_id, surface in ordered}:
            raise ValueError(
                "catalog record facts do not match the verified Wikidata path returns"
            )
        view._require_open()
        segments = self._compose(triples, ordered, tuple(proofs))
        return _render_reasoning(
            record,
            routes,
            segments=segments,
            facts=_semantic_facts(record),
            envelopes=tuple(envelopes),
        )

    @staticmethod
    def _path_rows(value: object) -> tuple[int, ...]:
        text = _strict_text(value, "Wikidata path rows")
        rows: list[int] = []
        for part in text.split(","):
            if not part.isdigit():
                raise ValueError("Wikidata path rows must be canonical integers")
            row = int(part)
            if row <= 0:
                raise ValueError("Wikidata path rows must be positive integers")
            rows.append(row)
        if len(rows) < 2:
            raise ValueError(
                "Wikidata path reasoning requires at least two training hops"
            )
        if text != ",".join(str(row) for row in rows):
            raise ValueError("Wikidata path rows are not canonical")
        return tuple(rows)

    def _returns(
        self,
        view: WikidataDerivedView,
        triples: list[V2TrainingTriple],
    ) -> tuple[tuple[str, str], ...]:
        ordered: list[tuple[str, str]] = []
        for triple in triples:
            canonical = f"Q{triple.object}"
            label = _wikidata_path_label(view, canonical)
            ordered.append(
                (
                    _wikidata_path_fact_id(triple.subject, triple.relation, triple.object),
                    _wikidata_path_return_surface(canonical, label),
                )
            )
        return tuple(ordered)

    @staticmethod
    def _compose(
        triples: list[V2TrainingTriple],
        ordered: tuple[tuple[str, str], ...],
        proofs: tuple[ProofObject, ...],
    ) -> tuple[TaggedSegment, ...]:
        segments: list[TaggedSegment] = [
            TaggedSegment(f"<|path|>hops={len(triples)}", "query")
        ]
        for index, ((fact_id, surface), triple) in enumerate(
            zip(ordered, triples, strict=True)
        ):
            segments.append(
                TaggedSegment(
                    f"<|traverse|>hop={index}|relation={triple.relation}",
                    "action",
                )
            )
            segments.append(TaggedSegment(surface, "payload", fact_id))
        for proof in proofs:
            segments.append(TaggedSegment(proof.to_bytes().decode("utf-8"), "proof"))
        segments.append(
            _answer_pointer_segment(len(triples), "candidate", "candidate_state")
        )
        segments.append(
            _answer_pointer_segment(len(triples), "final", "final_answer")
        )
        return tuple(segments)


EXPOSURE_RENDERER_TYPES = (
    ("fineweb_edu", FineWebEduRenderer, "fineweb-edu-nfc-gpt2-v1"),
    ("finemath", FineMathRenderer, "finemath-cross-dedup-gpt2-v1"),
    ("wikidata_graph", WikidataGraphRenderer, "wikidata-training-graph-v1"),
    ("synthetic_graph", SyntheticGraphRenderer, "srgm-seeded-graph-v1"),
)

REASONING_RENDERER_TYPES = (
    (
        "verified_synthetic_multihop",
        VerifiedSyntheticMultihopRenderer,
        "srgm-canonical-proof-v1",
    ),
    (
        "wikidata_path_reasoning",
        WikidataPathReasoningRenderer,
        "wikidata-training-path-proof-v1",
    ),
    (
        "relational_refinement",
        RelationalRefinementRenderer,
        "pointer-state-refinement-v1",
    ),
)


__all__ = (
    "EXPOSURE_RENDERER_TYPES",
    "REASONING_RENDERER_TYPES",
    "FineMathRenderer",
    "FineWebEduRenderer",
    "ProductionLaneRenderer",
    "ProductionRenderedRecord",
    "ProofEnvelope",
    "RelationalRefinementRenderer",
    "SyntheticGraphRenderer",
    "VerifiedSyntheticMultihopRenderer",
    "WikidataGraphRenderer",
    "WikidataPathReasoningRenderer",
    "iter_supported_reasoning_records",
)
