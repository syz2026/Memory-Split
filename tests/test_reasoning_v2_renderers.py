from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpusgen.graph_records import RenderedRecord, ScheduleEntry, TaggedSegment
from corpusgen.reasoning_v2 import renderers as renderers_module
from corpusgen.reasoning_v2 import wikidata_source as wikidata_source_module
from corpusgen.reasoning_v2.catalog import (
    CatalogDraft,
    CatalogRecord,
    SemanticFactRow,
    WikidataGraphCatalogSource,
    build_input_catalog,
    catalog_record_id,
)
from corpusgen.reasoning_v2.contracts import LANE_ORDER, BuildGeometry, LaneId
from corpusgen.reasoning_v2.renderers import (
    EXPOSURE_RENDERER_TYPES,
    FineMathRenderer,
    FineWebEduRenderer,
    ProductionRenderedRecord,
    ProofEnvelope,
    SyntheticGraphRenderer,
    WikidataGraphRenderer,
)
from corpusgen.reasoning_v2.semantic import RouteIndex, TokenSemanticSpan
from corpusgen.reasoning_v2.source_lock import SourceLock, stage_source_lock
from corpusgen.reasoning_v2.wikidata_source import (
    WikidataDerivedView,
    build_wikidata_derived_view,
    lookup_alias,
    lookup_training_triple,
)
from reasoning_v2_fixtures import (
    FixtureSourceLock,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
)
from test_reasoning_v2_catalog import FixtureLane
from test_reasoning_v2_wikidata_source import (
    EXPECTED_GENERATOR_COMMIT,
    _TarEntry,
    _install_archive_authority,
    _tar_bytes,
)
from train.tokenizer import get_tok


FINEWEB_PATHS = (
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
    "sample/10BT/002_00000.parquet",
)
WIKIDATA_TARGET_COUNT = 256
WIKIDATA_RECORD_COUNT = 3


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_source(
    root: Path,
    source_id: str,
    relative: str,
    payload: bytes,
) -> Path:
    path = root / source_id / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fact(fact_id: str, surface: str) -> SemanticFactRow:
    return SemanticFactRow(
        fact_id=fact_id,
        source="fixture",
        record_type="graph-exposure",
        payload_entropy_bits=Fraction(1),
        scheduled_exposures=1,
        expected_reads=Fraction(0),
        expected_hops=Fraction(0),
        surfaces=(surface,),
    )


def _record(
    *,
    lane_id: LaneId,
    source_id: str,
    source_key: str,
    relative: str,
    source_payload: bytes,
    target_count: int,
    locator: tuple[tuple[str, str | int], ...] = (),
    flags: tuple[str, ...] = (),
    facts: tuple[SemanticFactRow, ...] = (),
) -> CatalogRecord:
    source_locator = tuple(
        sorted(
            (("path", relative), *locator),
            key=lambda item: item[0].encode("utf-8"),
        )
    )
    draft = CatalogDraft(
        lane_id=lane_id,
        source_id=source_id,
        source_key=source_key,
        source_byte_sha256=_sha256(source_payload),
        source_locator=source_locator,
        semantic_flags=flags,
        semantic_facts=facts,
    )
    return CatalogRecord(
        ordinal=0,
        record_id=catalog_record_id(draft, target_count),
        lane_id=lane_id,
        source_id=source_id,
        source_key=source_key,
        source_byte_sha256=draft.source_byte_sha256,
        source_locator=source_locator,
        target_count=target_count,
        semantic_flags=flags,
        semantic_facts=facts,
    )


@pytest.fixture
def route_index(tmp_path: Path):
    opened: list[RouteIndex] = []

    def make(external_fact_ids: set[str]) -> RouteIndex:
        path = (tmp_path / f"routes-{len(opened)}.sqlite3").absolute()
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE selected ("
                "fact_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY"
                ") WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT INTO selected(fact_id) VALUES (?)",
                ((fact_id,) for fact_id in sorted(external_fact_ids)),
            )
            connection.commit()
        finally:
            connection.close()
        os.chmod(path, 0o600)
        index = RouteIndex.open(path)
        opened.append(index)
        return index

    yield make

    for index in opened:
        index.close()


def test_shared_exposure_renderer_contract_and_versions_are_frozen():
    assert EXPOSURE_RENDERER_TYPES == (
        ("fineweb_edu", FineWebEduRenderer, "fineweb-edu-nfc-gpt2-v1"),
        ("finemath", FineMathRenderer, "finemath-cross-dedup-gpt2-v1"),
        ("wikidata_graph", WikidataGraphRenderer, "wikidata-training-graph-v1"),
        ("synthetic_graph", SyntheticGraphRenderer, "srgm-seeded-graph-v1"),
    )
    proof = ProofEnvelope("none", b"premise", b"proof", "a" * 64, True)
    with pytest.raises((AttributeError, TypeError)):
        proof.family = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="sidecar|length|token"):
        ProductionRenderedRecord(
            record_id="b" * 64,
            lane_id="fineweb_edu",
            token_ids=(1,),
            dense_target_weights=b"\x01\x01",
            split90_target_weights=b"\x01",
            semantic_spans=(TokenSemanticSpan(0, 1, None, "plain_text"),),
            proofs=(),
            flags=(),
            semantic_leaks=(),
        )


def test_fineweb_renderer_is_exact_deterministic_and_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    payload = b"locked fineweb bytes"
    relative = FINEWEB_PATHS[0]
    _write_source(source_root, "fineweb_edu", relative, payload)
    monkeypatch.setattr(
        renderers_module,
        "_iter_parquet_texts",
        lambda _path: iter(("FineWeb educational prose. " * 32,)),
    )
    record = _record(
        lane_id="fineweb_edu",
        source_id="fineweb_edu",
        source_key="fineweb:0",
        relative=relative,
        source_payload=payload,
        target_count=16,
        locator=(("row", 0),),
    )
    routes = route_index(set())
    renderer = FineWebEduRenderer(source_root)

    first = renderer.render(record, routes)
    second = renderer.render(record, routes)

    assert first == second
    assert len(first.token_ids) == record.target_count
    assert first.token_ids[-1] == get_tok().EOT
    assert first.dense_target_weights == b"\x01" * record.target_count
    assert first.split90_target_weights == b"\x01" * record.target_count
    assert first.proofs == ()
    assert first.semantic_leaks == ()


def test_finemath_renderer_preserves_cross_deduplicated_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    rows: dict[str, tuple[str, ...]] = {}
    for index, relative in enumerate(FINEWEB_PATHS):
        payload = f"fineweb-{index}".encode()
        _write_source(source_root, "fineweb_edu", relative, payload)
        rows[f"fineweb_edu/{relative}"] = (
            "shared with fineweb" if index == 0 else f"fineweb unique {index}",
        )
    four_plus = "finemath-4plus/train-00000-of-00001.parquet"
    three_plus = "finemath-3plus/train-00000-of-00001.parquet"
    _write_source(source_root, "finemath", four_plus, b"four-plus")
    payload = b"three-plus"
    _write_source(source_root, "finemath", three_plus, payload)
    rows[f"finemath/{four_plus}"] = ("four plus unique",)
    rows[f"finemath/{three_plus}"] = (
        "shared with fineweb",
        "four plus unique",
        "three plus unique",
    )
    calls: list[str] = []

    def texts(path: Path):
        relative = path.relative_to(source_root).as_posix()
        calls.append(relative)
        return iter(rows[relative])

    monkeypatch.setattr(renderers_module, "_iter_parquet_texts", texts)
    expected = "three plus unique"
    record = _record(
        lane_id="finemath",
        source_id="finemath",
        source_key="finemath:three-plus:2",
        relative=three_plus,
        source_payload=payload,
        target_count=len(get_tok().encode(expected)) + 1,
        locator=(("row", 2),),
    )

    rendered = FineMathRenderer(source_root).render(record, route_index(set()))

    assert calls == [
        *(f"fineweb_edu/{path}" for path in FINEWEB_PATHS),
        f"finemath/{four_plus}",
        f"finemath/{three_plus}",
    ]
    assert get_tok().decode(list(rendered.token_ids[:-1])) == expected
    assert rendered.semantic_leaks == ()


def test_synthetic_renderer_replays_catalog_seed_and_closes_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    relative = "src/data.txt"
    payload = b"frozen generator bytes"
    _write_source(
        source_root,
        "deepmind_mathematics_generator",
        relative,
        payload,
    )
    seed = 9_876_543
    world_calls: list[tuple[int, int, int, int]] = []

    def worlds(n_entities, world_size, actual_seed, world_id_offset=0):
        world_calls.append((n_entities, world_size, actual_seed, world_id_offset))
        return iter((SimpleNamespace(world_id=world_id_offset),))

    surface = '{"target":"violet"}'
    emitted = RenderedRecord(
        segments=(TaggedSegment(surface, "payload", "synthetic:violet"),),
        schedule=ScheduleEntry("graph", "synthetic:violet:0", 0, 0),
    )
    monkeypatch.setattr(renderers_module, "iter_worlds", worlds)
    monkeypatch.setattr(
        renderers_module,
        "iter_graph_records",
        lambda _tok, worlds_factory: (
            tuple(worlds_factory()),
            iter((emitted,)),
        )[1],
    )
    record = _record(
        lane_id="synthetic_graph",
        source_id="deepmind_mathematics_generator",
        source_key="synthetic:violet:0",
        relative=relative,
        source_payload=payload,
        target_count=64,
        locator=(("generation_seed", seed), ("graph_exposure", 0)),
        flags=("generated",),
        facts=(_fact("synthetic:violet", surface),),
    )
    renderer = SyntheticGraphRenderer(source_root)
    routes = route_index({"synthetic:violet"})

    first = renderer.render(record, routes)
    second = renderer.render(record, routes)

    assert first == second
    assert world_calls == [(64, 64, seed, 0), (64, 64, seed, 0)]
    assert first.dense_target_weights == b"\x01" * record.target_count
    assert set(first.split90_target_weights) == {0, 1}
    assert first.semantic_leaks == ()


def _wikidata_archive_payloads() -> dict[str, bytes]:
    entries = {
        "wikidata5m_alias.tar.gz": [
            _TarEntry(
                "wikidata5m_entity.txt",
                b"Q1\tAlpha\nQ2\tBeta\nQ3\tGamma\nQ4\tDelta\n",
            ),
            _TarEntry(
                "wikidata5m_relation.txt",
                b"P1\tlinks to\nP2\tpart of\n",
            ),
        ],
        "wikidata5m_inductive.tar.gz": [
            _TarEntry("wikidata5m_inductive_test.txt", b"Q9\tP9\tQ10\n"),
            _TarEntry("wikidata5m_inductive_train.txt", b"Q1\tP1\tQ2\n"),
            _TarEntry("wikidata5m_inductive_valid.txt", b"Q7\tP7\tQ8\n"),
        ],
        "wikidata5m_transductive.tar.gz": [
            _TarEntry("wikidata5m_transductive_test.txt", b"Q11\tP11\tQ12\n"),
            _TarEntry("wikidata5m_transductive_train.txt", b"Q3\tP2\tQ4\n"),
            _TarEntry("wikidata5m_transductive_valid.txt", b"Q13\tP13\tQ14\n"),
        ],
    }
    return {name: _tar_bytes(members) for name, members in entries.items()}


@pytest.fixture
def wikidata_render_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        _wikidata_archive_payloads(),
    )
    lock = SourceLock.from_dict(
        json.loads(authority.source_lock_path.read_bytes())
    )
    source_root = stage_source_lock(
        lock,
        authority.source_root,
        tmp_path / "canonical",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        source_root,
        tmp_path / "views",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    return SimpleNamespace(
        authority=authority,
        lock=lock,
        source_root=source_root,
        view=view,
        tmp_path=tmp_path,
    )


def _wikidata_geometry() -> BuildGeometry:
    lane_quotas: tuple[tuple[LaneId, int], ...] = tuple(
        (
            lane_id,
            WIKIDATA_TARGET_COUNT * WIKIDATA_RECORD_COUNT
            if lane_id == "wikidata_graph"
            else 1,
        )
        for lane_id in LANE_ORDER
    )
    return BuildGeometry(
        profile="canary",
        total_targets=sum(quota for _lane_id, quota in lane_quotas),
        targets_per_update=1,
        context_length=WIKIDATA_TARGET_COUNT,
        shard_count=1,
        allow_fewer_shards=True,
        lane_quotas=lane_quotas,
    )


def _lane_sources(lock: SourceLock, view: WikidataDerivedView):
    sources: dict[LaneId, object] = {
        lane_id: FixtureLane(lane_id=lane_id, lock=lock)
        for lane_id in LANE_ORDER
        if lane_id != "wikidata_graph"
    }
    sources["wikidata_graph"] = WikidataGraphCatalogSource(view)
    return sources


def _wikidata_records(
    authority,
    view: WikidataDerivedView,
    output_name: str,
) -> tuple[CatalogRecord, ...]:
    catalog = build_input_catalog(
        _wikidata_geometry(),
        authority.lock,
        authority.source_root,
        _lane_sources(authority.lock, view),
        authority.tmp_path / output_name,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    return tuple(
        record
        for record in catalog.iter_records()
        if record.lane_id == "wikidata_graph"
    )


def _wikidata_fact_id(subject: int, relation: str, object_id: int) -> str:
    return f"wikidata:Q{subject}:{relation}:Q{object_id}"


def _expected_wikidata_payload(view: WikidataDerivedView, triple) -> str:
    def label(canonical_id: str) -> str:
        alias = lookup_alias(view, canonical_id)
        return alias.display if alias is not None else canonical_id

    subject = f"Q{triple.subject}"
    object_id = f"Q{triple.object}"
    return json.dumps(
        {
            "object": object_id,
            "object_label": label(object_id),
            "relation": triple.relation,
            "relation_label": label(triple.relation),
            "split": triple.training_split,
            "subject": subject,
            "subject_label": label(subject),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _retarget(record: CatalogRecord, **updates: str | int) -> CatalogRecord:
    locator = dict(record.source_locator)
    locator.update(updates)
    source_locator = tuple(
        sorted(locator.items(), key=lambda item: item[0].encode("utf-8"))
    )
    draft = CatalogDraft(
        lane_id=record.lane_id,
        source_id=record.source_id,
        source_key=record.source_key,
        source_byte_sha256=record.source_byte_sha256,
        source_locator=source_locator,
        semantic_flags=record.semantic_flags,
        semantic_facts=record.semantic_facts,
    )
    return replace(
        record,
        source_locator=source_locator,
        record_id=catalog_record_id(draft, record.target_count),
    )


def _rendered_bytes(record: ProductionRenderedRecord) -> bytes:
    token_bytes = struct.pack(f"<{len(record.token_ids)}H", *record.token_ids)
    return token_bytes + record.dense_target_weights + record.split90_target_weights


def test_wikidata_archive_view_catalog_renderer_round_trip(
    wikidata_render_authority,
    route_index,
):
    authority = wikidata_render_authority
    records = _wikidata_records(authority, authority.view, "round-trip")
    routes = route_index(
        {
            _wikidata_fact_id(1, "P1", 2),
            _wikidata_fact_id(3, "P2", 4),
        }
    )
    renderer = WikidataGraphRenderer(authority.source_root, authority.view)

    rendered = tuple(renderer.render(record, routes) for record in records)
    repeated = tuple(renderer.render(record, routes) for record in records)

    assert len(records) == WIKIDATA_RECORD_COUNT
    assert tuple(record.semantic_flags for record in records) == (
        ("graph-training-edge",),
        ("graph-training-edge",),
        ("graph-revisit",),
    )
    triples = (
        lookup_training_triple(authority.view, "inductive_train", 1),
        lookup_training_triple(authority.view, "transductive_train", 1),
        lookup_training_triple(authority.view, "inductive_train", 1),
    )
    expected_payloads = tuple(
        _expected_wikidata_payload(authority.view, triple)
        for triple in triples
    )
    tok = get_tok()
    for item, expected in zip(rendered, expected_payloads, strict=True):
        assert item.lane_id == "wikidata_graph"
        assert len(item.token_ids) == WIKIDATA_TARGET_COUNT
        assert item.token_ids[-1] == tok.EOT
        assert item.dense_target_weights == b"\x01" * WIKIDATA_TARGET_COUNT
        assert item.proofs == ()
        assert item.semantic_leaks == ()
        payload_span = item.semantic_spans[0]
        assert payload_span.role == "factual_payload"
        assert tok.decode(list(item.token_ids[: payload_span.token_end])) == expected
        assert set(item.split90_target_weights[: payload_span.token_end]) == {0}
        assert set(item.split90_target_weights[payload_span.token_end :]) == {1}
    assert rendered == repeated


@pytest.mark.parametrize(
    ("locator_update", "message"),
    (
        ({"wikidata_view_sha256": "0" * 64}, "view"),
        ({"member": "wikidata5m_transductive_train.txt"}, "member"),
        ({"path": "wikidata5m_transductive.tar.gz"}, "archive|path"),
        ({"split": "valid"}, "split|train"),
        ({"training_split": "transductive_train"}, "training split|member|archive"),
        ({"row": 2}, "row|indexed"),
        ({"training_edge_key": "Q1\tP1\tQ999"}, "edge"),
    ),
)
def test_renderer_rejects_wrong_view_receipt_member_split_row_or_edge(
    wikidata_render_authority,
    route_index,
    locator_update: dict[str, str | int],
    message: str,
):
    authority = wikidata_render_authority
    record = _wikidata_records(authority, authority.view, "rejects")[0]
    renderer = WikidataGraphRenderer(authority.source_root, authority.view)

    with pytest.raises(ValueError, match=message):
        renderer.render(_retarget(record, **locator_update), route_index(set()))


def test_renderer_rejects_caller_constructed_view(
    wikidata_render_authority,
):
    authority = wikidata_render_authority
    forged = WikidataDerivedView(
        root=authority.view.root,
        receipt_sha256=authority.view.receipt_sha256,
        receipt=authority.view.receipt,
    )

    with pytest.raises(ValueError, match="verified"):
        WikidataGraphRenderer(authority.source_root, forged)


def test_renderer_uses_indexed_lookups_without_full_stream_or_archive_scan(
    wikidata_render_authority,
    route_index,
    monkeypatch: pytest.MonkeyPatch,
):
    authority = wikidata_render_authority
    records = _wikidata_records(authority, authority.view, "indexed")
    renderer = WikidataGraphRenderer(authority.source_root, authority.view)
    routes = route_index(
        {
            _wikidata_fact_id(1, "P1", 2),
            _wikidata_fact_id(3, "P2", 4),
        }
    )
    triple_calls: list[tuple[str, int]] = []
    alias_calls: list[str] = []
    real_triple_lookup = wikidata_source_module.lookup_training_triple
    real_alias_lookup = wikidata_source_module.lookup_alias

    def triple_lookup(view, training_split, row):
        triple_calls.append((training_split, row))
        return real_triple_lookup(view, training_split, row)

    def alias_lookup(view, canonical_id):
        alias_calls.append(canonical_id)
        return real_alias_lookup(view, canonical_id)

    monkeypatch.setattr(renderers_module, "lookup_training_triple", triple_lookup)
    monkeypatch.setattr(renderers_module, "lookup_alias", alias_lookup)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("per-record full scan is forbidden")

    monkeypatch.setattr(
        wikidata_source_module,
        "iter_v2_training_triples",
        forbidden,
    )
    monkeypatch.setattr(wikidata_source_module, "iter_v2_aliases", forbidden)
    monkeypatch.setattr(
        wikidata_source_module,
        "iter_distinct_training_edges",
        forbidden,
    )
    monkeypatch.setattr(wikidata_source_module.tarfile, "open", forbidden)
    monkeypatch.setattr(renderers_module, "_hash_regular_file", forbidden)

    rendered = tuple(renderer.render(record, routes) for record in records)

    assert len(rendered) == len(records) == WIKIDATA_RECORD_COUNT
    assert len(triple_calls) == 2 * len(records)
    assert len(alias_calls) == 3 * len(records)
    assert not hasattr(renderers_module, "iter_v2_training_triples")
    assert not hasattr(renderers_module, "iter_v2_aliases")
    assert not hasattr(renderers_module, "iter_distinct_training_edges")


def test_end_to_end_rendered_bytes_repeat_across_view_and_catalog_rebuilds(
    wikidata_render_authority,
    route_index,
):
    authority = wikidata_render_authority
    second_view = build_wikidata_derived_view(
        authority.authority.source_lock_path,
        authority.source_root,
        authority.tmp_path / "views-second",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert second_view.receipt_sha256 == authority.view.receipt_sha256
    first_records = _wikidata_records(
        authority,
        authority.view,
        "catalog-first",
    )
    second_records = _wikidata_records(
        authority,
        second_view,
        "catalog-second",
    )
    first_renderer = WikidataGraphRenderer(authority.source_root, authority.view)
    second_renderer = WikidataGraphRenderer(authority.source_root, second_view)
    routes = route_index(
        {
            _wikidata_fact_id(1, "P1", 2),
            _wikidata_fact_id(3, "P2", 4),
        }
    )

    def render_all(renderer, records) -> tuple[bytes, ...]:
        return tuple(
            _rendered_bytes(renderer.render(record, routes))
            for record in records
        )

    first = render_all(first_renderer, first_records)
    second = render_all(second_renderer, second_records)

    assert len(first) == WIKIDATA_RECORD_COUNT
    assert first == second
