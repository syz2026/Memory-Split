from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import replace
from fractions import Fraction
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpusgen.graph_records import RenderedRecord, ScheduleEntry, TaggedSegment
from corpusgen.reasoning import (
    AnswerPointer,
    CompositionPremise,
    EqualityPremise,
    serialize_answer_state,
    solve_graph_composition,
    solve_slot_equality,
    verify_proof,
)
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
    REASONING_RENDERER_TYPES,
    FineMathRenderer,
    FineWebEduRenderer,
    ProductionRenderedRecord,
    ProofEnvelope,
    RelationalRefinementRenderer,
    SyntheticGraphRenderer,
    VerifiedSyntheticMultihopRenderer,
    WikidataGraphRenderer,
    WikidataPathReasoningRenderer,
)
from corpusgen.reasoning_v2.semantic import RouteIndex, TokenSemanticSpan
from corpusgen.reasoning_v2.source_lock import load_source_lock, stage_source_lock
from corpusgen.reasoning_v2.wikidata_source import (
    WikidataDerivedViewRef,
    build_wikidata_derived_view,
    lookup_alias,
    lookup_training_triple,
    open_wikidata_derived_view,
)
from corpusgen.srgm_worlds import iter_reasoning_records, iter_worlds
from train.tokenizer import get_tok
from reasoning_v2_fixtures import (
    FixtureSourceLock,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
)


FINEWEB_PATHS = (
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
    "sample/10BT/002_00000.parquet",
)


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
    lane_id,
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


@pytest.fixture
def renderer_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
    request: pytest.FixtureRequest,
):
    from contextlib import ExitStack

    source_root = tmp_path / "sources"
    source_root.mkdir()
    sessions = ExitStack()

    def make(lane_id: str):
        if lane_id == "verified_synthetic_multihop":
            return _multihop_fixture(source_root, monkeypatch, route_index)
        if lane_id == "relational_refinement":
            return _relational_fixture(source_root, route_index)
        if lane_id == "wikidata_path_reasoning":
            auth = request.getfixturevalue("wikidata_path_authority")
            record, routes = _prepare_wikidata_path(auth, route_index)
            view = sessions.enter_context(_open_render_view(auth))
            return WikidataPathReasoningRenderer(auth.root, view), record, routes

        if lane_id == "fineweb_edu":
            payload = b"fineweb fixture bytes"
            relative = FINEWEB_PATHS[0]
            _write_source(source_root, "fineweb_edu", relative, payload)
            monkeypatch.setattr(
                "corpusgen.reasoning_v2.renderers._iter_parquet_texts",
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
            return FineWebEduRenderer(source_root), record, route_index(set())

        if lane_id == "finemath":
            for index, relative in enumerate(FINEWEB_PATHS):
                _write_source(
                    source_root,
                    "fineweb_edu",
                    relative,
                    f"fineweb-{index}".encode(),
                )
            payload = b"finemath fixture bytes"
            relative = "finemath-4plus/train-00000-of-00001.parquet"
            _write_source(source_root, "finemath", relative, payload)

            def texts(path: Path):
                if "fineweb_edu" in path.parts:
                    return iter((f"FineWeb {path.name}",))
                return iter(("A mathematical derivation. " * 32,))

            monkeypatch.setattr(
                "corpusgen.reasoning_v2.renderers._iter_parquet_texts",
                texts,
            )
            record = _record(
                lane_id="finemath",
                source_id="finemath",
                source_key="finemath:0",
                relative=relative,
                source_payload=payload,
                target_count=16,
                locator=(("row", 0),),
            )
            return FineMathRenderer(source_root), record, route_index(set())

        if lane_id == "synthetic_graph":
            payload = b"synthetic generator bytes"
            relative = "src/data.txt"
            _write_source(
                source_root,
                "deepmind_mathematics_generator",
                relative,
                payload,
            )
            text = '{"target":"cerulean"}'
            rendered = RenderedRecord(
                segments=(TaggedSegment(text, "payload", "synthetic:fact"),),
                schedule=ScheduleEntry("graph", "synthetic:0", 0, 0),
            )
            monkeypatch.setattr(
                "corpusgen.reasoning_v2.renderers.iter_worlds",
                lambda *_args, **_kwargs: iter((SimpleNamespace(world_id=0),)),
            )
            monkeypatch.setattr(
                "corpusgen.reasoning_v2.renderers.iter_graph_records",
                lambda _tok, worlds_factory: (
                    worlds_factory(),
                    iter((rendered,)),
                )[1],
            )
            record = _record(
                lane_id="synthetic_graph",
                source_id="deepmind_mathematics_generator",
                source_key="synthetic:0",
                relative=relative,
                source_payload=payload,
                target_count=48,
                locator=(("generation_seed", 73), ("graph_exposure", 0)),
                flags=("generated",),
                facts=(_fact("synthetic:fact", text),),
            )
            return (
                SyntheticGraphRenderer(source_root),
                record,
                route_index({"synthetic:fact"}),
            )

        raise AssertionError(f"unknown fixture lane: {lane_id}")

    try:
        yield make
    finally:
        sessions.close()


@pytest.mark.parametrize(
    "lane_id",
    ["fineweb_edu", "finemath", "synthetic_graph"],
)
def test_exposure_renderer_is_exact_deterministic_and_closed(
    lane_id,
    renderer_fixture,
):
    renderer, record, routes = renderer_fixture(lane_id)
    first = renderer.render(record, routes)
    second = renderer.render(record, routes)
    assert first == second
    assert len(first.token_ids) == record.target_count
    assert first.dense_target_weights == b"\x01" * record.target_count
    assert len(first.split90_target_weights) == record.target_count
    assert set(first.split90_target_weights) <= {0, 1}
    assert all(0 <= token < 65_536 for token in first.token_ids)
    assert first.token_ids[-1] == get_tok().EOT
    assert first.proofs == ()
    assert first.semantic_leaks == ()


def test_shared_renderer_contract_and_versions_are_frozen():
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


def test_fineweb_reads_only_the_three_locked_10bt_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    calls: list[str] = []

    def texts(path: Path):
        calls.append(path.relative_to(source_root / "fineweb_edu").as_posix())
        return iter(("locked educational text " * 16,))

    monkeypatch.setattr(
        "corpusgen.reasoning_v2.renderers._iter_parquet_texts",
        texts,
    )
    renderer = FineWebEduRenderer(source_root)
    locked_sources = []
    for index, relative in enumerate(FINEWEB_PATHS):
        payload = f"locked-{index}".encode()
        _write_source(source_root, "fineweb_edu", relative, payload)
        locked_sources.append((index, relative, payload))
    unapproved = "sample/10BT/003_00000.parquet"
    unapproved_payload = b"not locked"
    _write_source(
        source_root,
        "fineweb_edu",
        unapproved,
        unapproved_payload,
    )
    with route_index(set()) as routes:
        for index, relative, payload in locked_sources:
            renderer.render(
                _record(
                    lane_id="fineweb_edu",
                    source_id="fineweb_edu",
                    source_key=f"fineweb:{index}",
                    relative=relative,
                    source_payload=payload,
                    target_count=8,
                    locator=(("row", 0),),
                ),
                routes,
            )
        assert calls == list(FINEWEB_PATHS)

        with pytest.raises(ValueError, match="locked|10BT|FineWeb"):
            renderer.render(
                _record(
                    lane_id="fineweb_edu",
                    source_id="fineweb_edu",
                    source_key="fineweb:unapproved",
                    relative=unapproved,
                    source_payload=unapproved_payload,
                    target_count=8,
                    locator=(("row", 0),),
                ),
                routes,
            )
    assert unapproved not in calls


def test_finemath_consumes_4plus_then_cross_deduplicated_3plus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    rows: dict[str, tuple[str, ...]] = {}
    calls: list[str] = []
    for index, relative in enumerate(FINEWEB_PATHS):
        payload = f"fineweb-source-{index}".encode()
        _write_source(source_root, "fineweb_edu", relative, payload)
        rows[f"fineweb_edu/{relative}"] = (
            "shared with fineweb" if index == 0 else f"fineweb unique {index}",
        )
    four_plus = "finemath-4plus/train-00000-of-00001.parquet"
    three_plus = "finemath-3plus/train-00000-of-00001.parquet"
    _write_source(source_root, "finemath", four_plus, b"four-plus-source")
    three_payload = b"three-plus-source"
    _write_source(source_root, "finemath", three_plus, three_payload)
    rows[f"finemath/{four_plus}"] = ("four plus unique",)
    rows[f"finemath/{three_plus}"] = (
        "shared with fineweb",
        "four plus unique",
        "three plus unique",
    )

    def texts(path: Path):
        relative = path.relative_to(source_root).as_posix()
        calls.append(relative)
        return iter(rows[relative])

    monkeypatch.setattr(
        "corpusgen.reasoning_v2.renderers._iter_parquet_texts",
        texts,
    )
    record = _record(
        lane_id="finemath",
        source_id="finemath",
        source_key="finemath:three-plus:2",
        relative=three_plus,
        source_payload=three_payload,
        target_count=len(get_tok().encode("three plus unique")) + 1,
        locator=(("row", 2),),
    )
    with route_index(set()) as routes:
        rendered = FineMathRenderer(source_root).render(record, routes)
    assert calls == [
        *(f"fineweb_edu/{path}" for path in FINEWEB_PATHS),
        f"finemath/{four_plus}",
        f"finemath/{three_plus}",
    ]
    assert get_tok().decode(list(rendered.token_ids[:-1])) == "three plus unique"


def test_synthetic_graph_replays_catalog_seed_and_frozen_world_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    relative = "src/data.txt"
    payload = b"frozen generator"
    _write_source(
        source_root,
        "deepmind_mathematics_generator",
        relative,
        payload,
    )
    calls: list[tuple[int, int, int, int]] = []
    seed = 9_876_543

    def worlds(n_entities, world_size, actual_seed, world_id_offset=0):
        calls.append((n_entities, world_size, actual_seed, world_id_offset))
        return iter((SimpleNamespace(world_id=world_id_offset),))

    text = '{"target":"violet"}'

    def graph_records(_tok, worlds_factory):
        assert tuple(worlds_factory())
        yield RenderedRecord(
            segments=(TaggedSegment(text, "payload", "synthetic:violet"),),
            schedule=ScheduleEntry("graph", "synthetic:violet:0", 0, 0),
        )

    monkeypatch.setattr("corpusgen.reasoning_v2.renderers.iter_worlds", worlds)
    monkeypatch.setattr(
        "corpusgen.reasoning_v2.renderers.iter_graph_records",
        graph_records,
    )
    record = _record(
        lane_id="synthetic_graph",
        source_id="deepmind_mathematics_generator",
        source_key="synthetic:violet:0",
        relative=relative,
        source_payload=payload,
        target_count=48,
        locator=(("generation_seed", seed), ("graph_exposure", 0)),
        flags=("generated",),
        facts=(_fact("synthetic:violet", text),),
    )
    with route_index({"synthetic:violet"}) as routes:
        SyntheticGraphRenderer(source_root).render(record, routes)
    assert calls == [(64, 64, seed, 0)]


def _synthetic_rejection_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rendered_records: tuple[RenderedRecord, ...],
    target_count: int = 48,
    exposure: int = 0,
) -> tuple[SyntheticGraphRenderer, CatalogRecord]:
    source_root = tmp_path / "sources"
    relative = "src/data.txt"
    payload = b"frozen generator"
    _write_source(
        source_root,
        "deepmind_mathematics_generator",
        relative,
        payload,
    )
    monkeypatch.setattr(
        "corpusgen.reasoning_v2.renderers.iter_worlds",
        lambda *_args, **_kwargs: iter((SimpleNamespace(world_id=0),)),
    )
    monkeypatch.setattr(
        "corpusgen.reasoning_v2.renderers.iter_graph_records",
        lambda _tok, worlds_factory: (
            tuple(worlds_factory()),
            iter(rendered_records),
        )[1],
    )
    surface = unicodedata.normalize(
        "NFC",
        rendered_records[-1].segments[0].text,
    )
    fact_id = rendered_records[-1].segments[0].fact_id or "synthetic:fact"
    record = _record(
        lane_id="synthetic_graph",
        source_id="deepmind_mathematics_generator",
        source_key=rendered_records[-1].schedule.record_id,
        relative=relative,
        source_payload=payload,
        target_count=target_count,
        locator=(("generation_seed", 5), ("graph_exposure", exposure)),
        flags=("generated",),
        facts=(_fact(fact_id, surface),),
    )
    return SyntheticGraphRenderer(source_root), record


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("e\u0301", "NFC"),
        ('{"score":NaN}', "finite"),
    ],
)
def test_renderer_rejects_noncanonical_or_nonfinite_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
    text: str,
    message: str,
):
    rendered = RenderedRecord(
        segments=(TaggedSegment(text, "payload", "synthetic:fact"),),
        schedule=ScheduleEntry("graph", "synthetic:record", 0, 0),
    )
    renderer, record = _synthetic_rejection_case(
        tmp_path,
        monkeypatch,
        rendered_records=(rendered,),
    )
    with (
        route_index({"synthetic:fact"}) as routes,
        pytest.raises(ValueError, match=message),
    ):
        renderer.render(record, routes)


def test_renderer_rejects_duplicate_source_record_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    first = RenderedRecord(
        segments=(TaggedSegment('{"target":"first"}', "payload", "fact:first"),),
        schedule=ScheduleEntry("graph", "duplicate", 0, 0),
    )
    second = RenderedRecord(
        segments=(TaggedSegment('{"target":"second"}', "payload", "fact:second"),),
        schedule=ScheduleEntry("graph", "duplicate", 1, 0),
    )
    renderer, record = _synthetic_rejection_case(
        tmp_path,
        monkeypatch,
        rendered_records=(first, second),
        exposure=1,
    )
    with (
        route_index({"fact:second"}) as routes,
        pytest.raises(ValueError, match="duplicate.*record ID"),
    ):
        renderer.render(record, routes)


def test_renderer_rejects_oversized_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    rendered = RenderedRecord(
        segments=(
            TaggedSegment(
                '{"target":"' + ("oversized " * 64) + '"}',
                "payload",
                "synthetic:fact",
            ),
        ),
        schedule=ScheduleEntry("graph", "oversized", 0, 0),
    )
    renderer, record = _synthetic_rejection_case(
        tmp_path,
        monkeypatch,
        rendered_records=(rendered,),
        target_count=4,
    )
    with (
        route_index({"synthetic:fact"}) as routes,
        pytest.raises(ValueError, match="core|target|fit"),
    ):
        renderer.render(record, routes)


def test_renderer_rejects_source_hash_and_target_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_index,
):
    source_root = tmp_path / "sources"
    relative = FINEWEB_PATHS[0]
    payload = b"locked"
    path = _write_source(source_root, "fineweb_edu", relative, payload)
    monkeypatch.setattr(
        "corpusgen.reasoning_v2.renderers._iter_parquet_texts",
        lambda _path: iter(("educational text " * 16,)),
    )
    record = _record(
        lane_id="fineweb_edu",
        source_id="fineweb_edu",
        source_key="fineweb:drift",
        relative=relative,
        source_payload=payload,
        target_count=8,
        locator=(("row", 0),),
    )
    renderer = FineWebEduRenderer(source_root)
    with route_index(set()) as routes:
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="hash|SHA-256|drift"):
            renderer.render(record, routes)

        path.write_bytes(payload)
        with pytest.raises(ValueError, match="record ID|target.*drift|commitment"):
            renderer.render(replace(record, target_count=9), routes)


# ---------------------------------------------------------------------------
# Task 4: indexed Wikidata renderer over a verified derived-view session
#
# These tests replace the earlier monkeypatched Wikidata renderer fixtures with
# a real archive -> staged source authority -> content-addressed derived view ->
# live descriptor session -> production catalog adapter -> renderer round trip.
# ---------------------------------------------------------------------------

_RENDER_TARGET = 256
_RENDER_WIKIDATA_RECORDS = 3


def _render_archive_payloads() -> dict[str, bytes]:
    from test_reasoning_v2_wikidata_source import _TarEntry, _tar_bytes

    entries = {
        "wikidata5m_alias.tar.gz": [
            _TarEntry(
                "wikidata5m_entity.txt",
                b"Q1\tAlpha\nQ2\tBeta\nQ3\tGamma\nQ4\tDelta\n",
            ),
            _TarEntry("wikidata5m_relation.txt", b"P1\tlinks to\nP2\tpart of\n"),
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
    return {name: _tar_bytes(rows) for name, rows in entries.items()}


def _wikidata_fact_id(subject: int, relation: str, object_: int) -> str:
    return f"wikidata:Q{subject}:{relation}:Q{object_}"


def _expected_wikidata_payload(view, triple) -> str:
    def label(canonical_id: str) -> str:
        record = lookup_alias(view, canonical_id)
        return record.display if record is not None else canonical_id

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


def _render_geometry(
    target: int = _RENDER_TARGET,
    wikidata_records: int = _RENDER_WIKIDATA_RECORDS,
) -> BuildGeometry:
    lane_quotas: tuple[tuple[LaneId, int], ...] = tuple(
        (
            lane_id,
            target * wikidata_records if lane_id == "wikidata_graph" else 1,
        )
        for lane_id in LANE_ORDER
    )
    return BuildGeometry(
        profile="canary",
        total_targets=sum(quota for _lane, quota in lane_quotas),
        targets_per_update=1,
        context_length=target,
        shard_count=1,
        allow_fewer_shards=True,
        lane_quotas=lane_quotas,
    )


def _render_lane_sources(lock, view):
    from test_reasoning_v2_catalog import _end_to_end_sources

    return _end_to_end_sources(lock, WikidataGraphCatalogSource(view))


def _retarget(record: CatalogRecord, **locator_updates: str | int) -> CatalogRecord:
    locator = dict(record.source_locator)
    locator.update(locator_updates)
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


@pytest.fixture
def wikidata_render_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    from test_reasoning_v2_wikidata_source import _install_archive_authority

    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        _render_archive_payloads(),
    )
    commit = fixture_source_lock.lock.generator_commit
    lock = load_source_lock(
        authority.source_lock_path,
        expected_generator_commit=commit,
    )
    root = stage_source_lock(
        lock,
        authority.source_root,
        tmp_path / "canonical",
        expected_generator_commit=commit,
    )
    view_ref = build_wikidata_derived_view(
        authority.source_lock_path,
        root,
        tmp_path / "views",
        expected_generator_commit=commit,
    )
    outputs = tmp_path / "catalogs"
    outputs.mkdir()
    return SimpleNamespace(
        lock=lock,
        root=root,
        source_lock_path=authority.source_lock_path,
        view_ref=view_ref,
        commit=commit,
        outputs=outputs,
        tmp_path=tmp_path,
    )


def _open_render_view(auth):
    return open_wikidata_derived_view(
        auth.source_lock_path,
        auth.root,
        auth.view_ref,
        expected_generator_commit=auth.commit,
    )


def _wikidata_records(auth, view, output_name: str) -> list[CatalogRecord]:
    catalog = build_input_catalog(
        _render_geometry(),
        auth.lock,
        auth.root,
        _render_lane_sources(auth.lock, view),
        auth.outputs / output_name,
        expected_generator_commit=auth.commit,
    )
    return [
        record
        for record in catalog.iter_records()
        if record.lane_id == "wikidata_graph"
    ]


def test_wikidata_archive_view_catalog_renderer_round_trip(
    wikidata_render_authority,
    route_index,
):
    auth = wikidata_render_authority
    routes = route_index(
        {_wikidata_fact_id(1, "P1", 2), _wikidata_fact_id(3, "P2", 4)}
    )
    with _open_render_view(auth) as view:
        records = _wikidata_records(auth, view, "round-trip")
        renderer = WikidataGraphRenderer(auth.root, view)
        rendered = [renderer.render(record, routes) for record in records]
        repeat = [renderer.render(record, routes) for record in records]
        edge0 = lookup_training_triple(view, "inductive_train", 1)
        edge1 = lookup_training_triple(view, "transductive_train", 1)
        expected_payloads = [
            _expected_wikidata_payload(view, edge0),
            _expected_wikidata_payload(view, edge1),
            _expected_wikidata_payload(view, edge0),
        ]

    assert [record.source_key for record in records] == [
        "wikidata-edge-000000000000",
        "wikidata-edge-000000000001",
        "wikidata-revisit-000000000000",
    ]
    assert [item.flags for item in rendered] == [
        ("graph-training-edge",),
        ("graph-training-edge",),
        ("graph-revisit",),
    ]
    tok = get_tok()
    for item, expected in zip(rendered, expected_payloads, strict=True):
        assert item.lane_id == "wikidata_graph"
        assert len(item.token_ids) == _RENDER_TARGET
        assert item.token_ids[-1] == tok.EOT
        assert item.dense_target_weights == b"\x01" * _RENDER_TARGET
        assert item.proofs == ()
        assert item.semantic_leaks == ()
        core_end = item.semantic_spans[0].token_end
        assert tok.decode(list(item.token_ids[:core_end])) == expected
        assert set(item.split90_target_weights[:core_end]) == {0}
        assert set(item.split90_target_weights[core_end:]) == {1}
    assert rendered == repeat


def test_renderer_rejects_wrong_view_receipt_member_split_row_or_edge(
    wikidata_render_authority,
    route_index,
):
    auth = wikidata_render_authority
    routes = route_index(set())
    with _open_render_view(auth) as view:
        records = _wikidata_records(auth, view, "rejects")
        edge = records[0]
        renderer = WikidataGraphRenderer(auth.root, view)
        renderer.render(edge, routes)

        with pytest.raises((TypeError, ValueError)):
            WikidataGraphRenderer(auth.root, auth.view_ref)
        assert isinstance(auth.view_ref, WikidataDerivedViewRef)

        with pytest.raises(ValueError, match="view"):
            renderer.render(_retarget(edge, wikidata_view_sha256="0" * 64), routes)
        with pytest.raises(ValueError, match="member"):
            renderer.render(
                _retarget(edge, member="wikidata5m_transductive_train.txt"),
                routes,
            )
        with pytest.raises(ValueError, match="split|train"):
            renderer.render(_retarget(edge, split="valid"), routes)
        with pytest.raises(ValueError, match="row|range"):
            renderer.render(_retarget(edge, row=2), routes)
        with pytest.raises(ValueError, match="edge"):
            renderer.render(
                _retarget(edge, training_edge_key="Q1\tP1\tQ999"),
                routes,
            )

    with pytest.raises(ValueError, match="session|closed|open"):
        renderer.render(edge, routes)


def test_renderer_uses_indexed_lookups_without_full_stream_or_archive_scan(
    wikidata_render_authority,
    route_index,
    monkeypatch: pytest.MonkeyPatch,
):
    auth = wikidata_render_authority
    routes = route_index(
        {_wikidata_fact_id(1, "P1", 2), _wikidata_fact_id(3, "P2", 4)}
    )
    assert not hasattr(renderers_module, "iter_training_triples")
    assert not hasattr(renderers_module, "iter_aliases")
    with _open_render_view(auth) as view:
        records = _wikidata_records(auth, view, "indexed")
        renderer = WikidataGraphRenderer(auth.root, view)

        triple_calls: list[int] = []
        alias_calls: list[int] = []
        real_lookup_triple = wikidata_source_module.lookup_training_triple
        real_lookup_alias = wikidata_source_module.lookup_alias

        def spy_triple(*args, **kwargs):
            triple_calls.append(1)
            return real_lookup_triple(*args, **kwargs)

        def spy_alias(*args, **kwargs):
            alias_calls.append(1)
            return real_lookup_alias(*args, **kwargs)

        monkeypatch.setattr(renderers_module, "lookup_training_triple", spy_triple)
        monkeypatch.setattr(renderers_module, "lookup_alias", spy_alias)

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("per-record full-stream scan is forbidden")

        monkeypatch.setattr(
            wikidata_source_module, "iter_v2_training_triples", _forbidden
        )
        monkeypatch.setattr(wikidata_source_module, "iter_v2_aliases", _forbidden)
        monkeypatch.setattr(
            wikidata_source_module, "iter_distinct_training_edges", _forbidden
        )
        monkeypatch.setattr(wikidata_source_module.tarfile, "open", _forbidden)

        rendered = [renderer.render(record, routes) for record in records]

    assert len(rendered) == len(records) == _RENDER_WIKIDATA_RECORDS
    assert len(triple_calls) == len(records)
    assert len(alias_calls) == 3 * len(records)


def test_end_to_end_rendered_bytes_repeat_across_view_and_catalog_rebuilds(
    wikidata_render_authority,
    route_index,
):
    auth = wikidata_render_authority
    # Rebuild the content-addressed view into two fresh output roots before the
    # route index is opened, so its pinned parent identity stays stable while the
    # catalogs (which only touch their own output parent) are rebuilt below.
    first_ref = build_wikidata_derived_view(
        auth.source_lock_path,
        auth.root,
        auth.tmp_path / "view-first",
        expected_generator_commit=auth.commit,
    )
    second_ref = build_wikidata_derived_view(
        auth.source_lock_path,
        auth.root,
        auth.tmp_path / "view-second",
        expected_generator_commit=auth.commit,
    )
    assert first_ref.receipt_sha256 == second_ref.receipt_sha256
    routes = route_index(
        {_wikidata_fact_id(1, "P1", 2), _wikidata_fact_id(3, "P2", 4)}
    )

    def render_all(view_ref, output_name: str) -> list[ProductionRenderedRecord]:
        with open_wikidata_derived_view(
            auth.source_lock_path,
            auth.root,
            view_ref,
            expected_generator_commit=auth.commit,
        ) as view:
            records = _wikidata_records(auth, view, output_name)
            renderer = WikidataGraphRenderer(auth.root, view)
            return [renderer.render(record, routes) for record in records]

    first = render_all(first_ref, "catalog-first")
    second = render_all(second_ref, "catalog-second")
    assert len(first) == _RENDER_WIKIDATA_RECORDS
    assert first == second


# ---------------------------------------------------------------------------
# Task 5B: solver-backed reasoning lanes
#
# VerifiedSyntheticMultihopRenderer, WikidataPathReasoningRenderer, and
# RelationalRefinementRenderer emit canonical solver-verified proofs and
# value-free pointer answer states over the real proof/solver/reference
# implementations (iter_reasoning_records, solve_graph_composition,
# solve_slot_equality, serialize_answer_state, AnswerPointer, verify_proof).
# ---------------------------------------------------------------------------

_MULTIHOP_SEED = 20_260_724
_MULTIHOP_HOPS = 2
_REASONING_TARGET = 256
_ANSWER_STATE_RE = re.compile(r"<\|answer_state\|>\{[^}]*\}")
_SLOT_RE = re.compile(r"<\|slot_([0-3])\|>")


def decoded_answer_states(decoded: str) -> str:
    return "".join(_ANSWER_STATE_RE.findall(decoded))


def _reasoning_returns(rendered: RenderedRecord) -> tuple[tuple[str, str], ...]:
    return tuple(
        (segment.fact_id, segment.text)
        for segment in rendered.segments
        if segment.role == "payload"
    )


def _select_multihop(seed: int, hops: int, family: str):
    # Number lane exposures through the same canonical supported-family filter the
    # renderer uses, so date-ordering records never shift the exposure a record
    # is built against.
    tok = get_tok()

    def worlds():
        return iter_worlds(64, 64, seed, 0)

    for exposure, rendered in renderers_module.iter_supported_reasoning_records(
        tok, worlds, seed=seed, max_hops=hops
    ):
        if exposure > 400:
            break
        if renderers_module._supported_reasoning_family(rendered) != family:
            continue
        returns = _reasoning_returns(rendered)
        texts = [text for _fid, text in returns]
        fids = [fid for fid, _text in returns]
        if not returns or len(set(texts)) != len(texts) or len(set(fids)) != len(fids):
            continue
        return exposure, rendered, returns
    raise AssertionError(f"no distinct {family} reasoning exposure for seed {seed}")


def _multihop_facts(returns) -> tuple[SemanticFactRow, ...]:
    return tuple(
        sorted(
            (_fact(fid, text) for fid, text in returns),
            key=lambda row: row.fact_id.encode("utf-8"),
        )
    )


def _make_multihop(
    source_root: Path,
    route_index,
    family: str = "graph_composition_mod4",
    target_count: int = _REASONING_TARGET,
):
    seed = _MULTIHOP_SEED
    hops = _MULTIHOP_HOPS
    exposure, rendered, returns = _select_multihop(seed, hops, family)
    relative = "src/data.txt"
    payload = b"deepmind mathematics generator bytes"
    _write_source(source_root, "deepmind_mathematics_generator", relative, payload)
    record = _record(
        lane_id="verified_synthetic_multihop",
        source_id="deepmind_mathematics_generator",
        source_key=rendered.schedule.record_id,
        relative=relative,
        source_payload=payload,
        target_count=target_count,
        locator=(
            ("generation_seed", seed),
            ("graph_exposure", exposure),
            ("reasoning_hops", hops),
        ),
        flags=("generated",),
        facts=_multihop_facts(returns),
    )
    routes = route_index({fid for fid, _ in returns})
    renderer = VerifiedSyntheticMultihopRenderer(source_root)
    return renderer, record, returns, routes, rendered


def _multihop_fixture(source_root: Path, monkeypatch, route_index):
    renderer, record, _returns, routes, _rendered = _make_multihop(
        source_root, route_index, "graph_composition_mod4"
    )
    return renderer, record, routes


def _relational_fixture(source_root: Path, route_index):
    relative = "src/data.txt"
    payload = b"relational refinement generator bytes"
    _write_source(source_root, "ruletaker", relative, payload)
    facts = tuple(
        sorted(
            (
                _fact("relational:slot0", "relation-alpha"),
                _fact("relational:slot1", "relation-beta"),
            ),
            key=lambda row: row.fact_id.encode("utf-8"),
        )
    )
    record = _record(
        lane_id="relational_refinement",
        source_id="ruletaker",
        source_key="relational:0",
        relative=relative,
        source_payload=payload,
        target_count=_REASONING_TARGET,
        locator=(("row", 0),),
        flags=("relational-refinement",),
        facts=facts,
    )
    routes = route_index({"relational:slot0", "relational:slot1"})
    return RelationalRefinementRenderer(source_root), record, routes


def _path_archive_payloads() -> dict[str, bytes]:
    from test_reasoning_v2_wikidata_source import _TarEntry, _tar_bytes

    entries = {
        "wikidata5m_alias.tar.gz": [
            _TarEntry(
                "wikidata5m_entity.txt",
                b"Q1\tAlpha\nQ2\tBeta\nQ3\tGamma\nQ4\tDelta\nQ5\tEpsilon\n",
            ),
            _TarEntry("wikidata5m_relation.txt", b"P1\tlinks to\nP2\tpart of\n"),
        ],
        "wikidata5m_inductive.tar.gz": [
            _TarEntry("wikidata5m_inductive_test.txt", b"Q9\tP9\tQ10\n"),
            _TarEntry(
                "wikidata5m_inductive_train.txt",
                b"Q1\tP1\tQ2\nQ2\tP2\tQ3\n",
            ),
            _TarEntry("wikidata5m_inductive_valid.txt", b"Q7\tP7\tQ8\n"),
        ],
        "wikidata5m_transductive.tar.gz": [
            _TarEntry("wikidata5m_transductive_test.txt", b"Q11\tP11\tQ12\n"),
            _TarEntry("wikidata5m_transductive_train.txt", b"Q4\tP1\tQ5\n"),
            _TarEntry("wikidata5m_transductive_valid.txt", b"Q13\tP13\tQ14\n"),
        ],
    }
    return {name: _tar_bytes(rows) for name, rows in entries.items()}


@pytest.fixture
def wikidata_path_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    from test_reasoning_v2_wikidata_source import _install_archive_authority

    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        _path_archive_payloads(),
    )
    commit = fixture_source_lock.lock.generator_commit
    lock = load_source_lock(
        authority.source_lock_path,
        expected_generator_commit=commit,
    )
    root = stage_source_lock(
        lock,
        authority.source_root,
        tmp_path / "canonical",
        expected_generator_commit=commit,
    )
    view_ref = build_wikidata_derived_view(
        authority.source_lock_path,
        root,
        tmp_path / "views",
        expected_generator_commit=commit,
    )
    return SimpleNamespace(
        lock=lock,
        root=root,
        source_lock_path=authority.source_lock_path,
        view_ref=view_ref,
        commit=commit,
        tmp_path=tmp_path,
    )


def _wikidata_path_record(
    view,
    auth,
    training_split: str,
    rows: tuple[int, ...],
    target_count: int = _REASONING_TARGET,
) -> CatalogRecord:
    triples = [lookup_training_triple(view, training_split, row) for row in rows]
    member = triples[0].member
    archive_path = triples[0].archive_path
    archive_sha = {a.path: a.sha256 for a in view.receipt.archives}[archive_path]
    facts = []
    for triple in triples:
        canonical = f"Q{triple.object}"
        label = renderers_module._wikidata_path_label(view, canonical)
        surface = renderers_module._wikidata_path_return_surface(canonical, label)
        fact_id = renderers_module._wikidata_path_fact_id(
            triple.subject, triple.relation, triple.object
        )
        facts.append(_fact(fact_id, surface))
    facts = tuple(sorted(facts, key=lambda row: row.fact_id.encode("utf-8")))
    locator = tuple(
        sorted(
            (
                ("member", member),
                ("path", archive_path),
                ("path_rows", ",".join(str(row) for row in rows)),
                ("training_split", training_split),
                ("wikidata_view_sha256", view.receipt_sha256),
            ),
            key=lambda item: item[0].encode("utf-8"),
        )
    )
    draft = CatalogDraft(
        lane_id="wikidata_path_reasoning",
        source_id="wikidata5m",
        source_key="wikidata-path-000000000000",
        source_byte_sha256=archive_sha,
        source_locator=locator,
        semantic_flags=("graph-training-path",),
        semantic_facts=facts,
    )
    return CatalogRecord(
        ordinal=0,
        record_id=catalog_record_id(draft, target_count),
        lane_id="wikidata_path_reasoning",
        source_id="wikidata5m",
        source_key="wikidata-path-000000000000",
        source_byte_sha256=archive_sha,
        source_locator=locator,
        target_count=target_count,
        semantic_flags=("graph-training-path",),
        semantic_facts=facts,
    )


def _prepare_wikidata_path(
    auth,
    route_index,
    training_split: str = "inductive_train",
    rows: tuple[int, ...] = (1, 2),
):
    # Build the record in a short-lived session, then open the route index
    # BEFORE the rendering session so the sqlite write never mutates the
    # source-lock parent directory while an authority session is verified open.
    with _open_render_view(auth) as view:
        record = _wikidata_path_record(view, auth, training_split, rows)
    routes = route_index({fact.fact_id for fact in record.semantic_facts})
    return record, routes


@pytest.mark.parametrize(
    "lane_id",
    [
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
    ],
)
def test_reasoning_renderer_has_replayable_proof_and_no_leak(
    lane_id,
    renderer_fixture,
):
    renderer, record, routes = renderer_fixture(lane_id)
    rendered = renderer.render(record, routes)
    assert len(rendered.token_ids) == record.target_count
    assert rendered.proofs
    assert all(proof.replay_verified for proof in rendered.proofs)
    assert all(
        proof.proof_sha256 == hashlib.sha256(proof.proof_bytes).hexdigest()
        for proof in rendered.proofs
    )
    assert rendered.semantic_leaks == ()
    assert rendered.token_ids[-1] == get_tok().EOT
    assert rendered.dense_target_weights == b"\x01" * record.target_count
    # Split90 masks the routed factual returns and supervises everything else.
    assert set(rendered.split90_target_weights) == {0, 1}
    # Rendering is deterministic byte-for-byte.
    assert rendered == renderer.render(record, routes)


def test_refinement_candidate_and_final_states_are_value_free(renderer_fixture):
    renderer, record, routes = renderer_fixture("relational_refinement")
    rendered = renderer.render(record, routes)
    decoded = get_tok().decode(list(rendered.token_ids))
    states = decoded_answer_states(decoded)
    assert states
    for fact in record.semantic_facts:
        assert all(surface not in states for surface in fact.surfaces)


def test_reasoning_renderer_types_are_frozen():
    assert REASONING_RENDERER_TYPES == (
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


@pytest.mark.parametrize(
    "lane_id",
    [
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
    ],
)
def test_reasoning_spans_and_states_use_pointer_roles(lane_id, renderer_fixture):
    renderer, record, routes = renderer_fixture(lane_id)
    rendered = renderer.render(record, routes)
    roles = {span.role for span in rendered.semantic_spans}
    assert {"proof", "factual_payload", "answer_state"} <= roles
    for span in rendered.semantic_spans:
        if span.role == "factual_payload":
            assert span.fact_id is not None
        if span.role in ("proof", "answer_state"):
            assert span.fact_id is None

    decoded = get_tok().decode(list(rendered.token_ids))
    states = _ANSWER_STATE_RE.findall(decoded)
    assert states
    phases = set()
    for state in states:
        payload = json.loads(state[len("<|answer_state|>") :])
        assert set(payload) == {"member_index", "phase", "read_index", "slot"}
        slot_match = _SLOT_RE.fullmatch(payload["slot"])
        assert slot_match is not None
        pointer = AnswerPointer(
            slot=int(slot_match.group(1)),
            read_index=payload["read_index"],
            member_index=payload["member_index"],
        )
        assert serialize_answer_state(pointer, phase=payload["phase"]) == state
        phases.add(payload["phase"])
    assert {"candidate", "final"} <= phases


def test_synthetic_multihop_replays_canonical_composition(
    tmp_path: Path,
    route_index,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    renderer, record, returns, routes, rendered = _make_multihop(
        source_root, route_index, "graph_composition_mod4"
    )
    result = renderer.render(record, routes)
    assert len(result.proofs) == 1
    envelope = result.proofs[0]
    assert envelope.family == "graph_composition_mod4"
    premises = tuple(
        CompositionPremise(
            fid,
            hop=index,
            compose_code=int(dict(json.loads(text)["qualifiers"])["compose"]),
        )
        for index, (fid, text) in enumerate(returns)
    )
    expected = solve_graph_composition(premises)
    assert envelope.proof_bytes == expected.to_bytes()
    assert verify_proof(expected, premises) is True
    assert dict(expected.conclusion)["relation"] == rendered.segments[-1].text


def test_synthetic_multihop_replays_canonical_equality(
    tmp_path: Path,
    route_index,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    renderer, record, returns, routes, rendered = _make_multihop(
        source_root, route_index, "slot_equality"
    )
    result = renderer.render(record, routes)
    envelope = result.proofs[0]
    assert envelope.family == "slot_equality"
    premises = tuple(
        EqualityPremise(fid, slot=index, value=json.loads(text)["target"])
        for index, (fid, text) in enumerate(returns)
    )
    expected = solve_slot_equality(premises)
    assert envelope.proof_bytes == expected.to_bytes()
    assert verify_proof(expected, premises) is True
    assert dict(expected.conclusion)["equal"] == (rendered.segments[-1].text == "yes")


def test_reasoning_renderer_rejects_solver_disagreement(
    tmp_path: Path,
    route_index,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    renderer, record, _returns, routes, _rendered = _make_multihop(
        source_root, route_index, "graph_composition_mod4"
    )
    real = renderers_module.solve_graph_composition

    def tampered(premises):
        proof = real(premises)
        current = dict(proof.conclusion)["relation"]
        return replace(
            proof,
            conclusion=(("relation", "r0" if current != "r0" else "r1"),),
        )

    monkeypatch.setattr(renderers_module, "solve_graph_composition", tampered)
    with pytest.raises(ValueError):
        renderer.render(record, routes)


def test_reasoning_renderer_rejects_premise_mutation(
    tmp_path: Path,
    route_index,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    renderer, record, _returns, routes, _rendered = _make_multihop(
        source_root, route_index, "graph_composition_mod4"
    )
    real = renderers_module._canonical_premise_bytes

    def mutated(premises):
        value = json.loads(real(premises).decode("utf-8"))
        value[0]["compose_code"] = (value[0]["compose_code"] + 1) % 4
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

    monkeypatch.setattr(renderers_module, "_canonical_premise_bytes", mutated)
    with pytest.raises(ValueError):
        renderer.render(record, routes)


def test_wikidata_path_uses_only_training_triples_via_indexed_lookups(
    wikidata_path_authority,
    route_index,
    monkeypatch: pytest.MonkeyPatch,
):
    auth = wikidata_path_authority
    assert not hasattr(renderers_module, "iter_v2_training_triples")
    record, routes = _prepare_wikidata_path(auth, route_index)
    with _open_render_view(auth) as view:
        renderer = WikidataPathReasoningRenderer(auth.root, view)

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("wikidata path used a forbidden scan")

        monkeypatch.setattr(
            wikidata_source_module, "iter_v2_training_triples", _forbidden
        )
        monkeypatch.setattr(wikidata_source_module, "iter_v2_aliases", _forbidden)
        monkeypatch.setattr(
            wikidata_source_module, "iter_distinct_training_edges", _forbidden
        )
        monkeypatch.setattr(wikidata_source_module.tarfile, "open", _forbidden)

        rendered = renderer.render(record, routes)

    assert len(rendered.proofs) == 1
    assert rendered.proofs[0].family == "slot_equality"
    assert rendered.proofs[0].replay_verified is True
    assert rendered.semantic_leaks == ()
    assert len(rendered.token_ids) == record.target_count


def test_wikidata_path_rejects_sealed_split(
    wikidata_path_authority,
    route_index,
):
    auth = wikidata_path_authority
    record, routes = _prepare_wikidata_path(auth, route_index)
    sealed = _retarget(
        record,
        training_split="inductive_test",
        member="wikidata5m_inductive_test.txt",
    )
    with _open_render_view(auth) as view:
        renderer = WikidataPathReasoningRenderer(auth.root, view)
        renderer.render(record, routes)
        # The sealed split is rejected before the route index is consulted.
        with pytest.raises(ValueError, match="train|sealed|split"):
            renderer.render(sealed, routes)


def test_wikidata_path_requires_a_live_session(
    wikidata_path_authority,
    route_index,
):
    auth = wikidata_path_authority
    with pytest.raises((TypeError, ValueError)):
        WikidataPathReasoningRenderer(auth.root, auth.view_ref)
    record, routes = _prepare_wikidata_path(auth, route_index)
    with _open_render_view(auth) as view:
        renderer = WikidataPathReasoningRenderer(auth.root, view)
        renderer.render(record, routes)
    # The retained session is closed; a further render is rejected before the
    # route index is consulted.
    with pytest.raises(ValueError, match="session|closed|open"):
        renderer.render(record, routes)


# --- Fix A: Wikidata path proofs grounded in observed adjacency ------------
#
# Each adjacent hop emits one canonical slot_equality proof comparing
# Q<left.object> with Q<right.subject> (the shared endpoint actually observed in
# the frozen training triples), using distinct pointer slots and exact
# edge/provenance premise IDs. The conclusion is grounded in real endpoints, not
# in a hash of the relation PID.


def test_wikidata_path_proofs_are_grounded_slot_equality_over_endpoints(
    wikidata_path_authority,
    route_index,
):
    auth = wikidata_path_authority
    record, routes = _prepare_wikidata_path(auth, route_index, rows=(1, 2))
    with _open_render_view(auth) as view:
        renderer = WikidataPathReasoningRenderer(auth.root, view)
        rendered = renderer.render(record, routes)
        left = lookup_training_triple(view, "inductive_train", 1)
        right = lookup_training_triple(view, "inductive_train", 2)

    # The two adjacent training edges genuinely share the middle entity.
    assert left.object == right.subject
    assert len(rendered.proofs) == 1
    envelope = rendered.proofs[0]
    assert envelope.family == "slot_equality"
    assert envelope.replay_verified is True

    premises = json.loads(envelope.premise_bytes.decode("utf-8"))
    by_slot = {premise["slot"]: premise for premise in premises}
    assert set(by_slot) == {0, 1}
    # Premise values are the actual adjacent endpoints, and they match because
    # the walk is genuinely connected in the observed training data.
    assert by_slot[0]["value"] == f"Q{left.object}"
    assert by_slot[1]["value"] == f"Q{right.subject}"
    assert by_slot[0]["value"] == by_slot[1]["value"]
    # Exact edge/provenance premise IDs (split#row + canonical edge).
    assert by_slot[0]["fact_id"] == (
        f"inductive_train#1:Q{left.subject}:{left.relation}:Q{left.object}"
    )
    assert by_slot[1]["fact_id"] == (
        f"inductive_train#2:Q{right.subject}:{right.relation}:Q{right.object}"
    )
    # The emitted proof is exactly the independent solver's grounded conclusion.
    expected = solve_slot_equality(
        (
            EqualityPremise(by_slot[0]["fact_id"], slot=0, value=by_slot[0]["value"]),
            EqualityPremise(by_slot[1]["fact_id"], slot=1, value=by_slot[1]["value"]),
        )
    )
    assert envelope.proof_bytes == expected.to_bytes()
    assert dict(expected.conclusion)["equal"] is True
    assert verify_proof(
        expected,
        (
            EqualityPremise(by_slot[0]["fact_id"], slot=0, value=by_slot[0]["value"]),
            EqualityPremise(by_slot[1]["fact_id"], slot=1, value=by_slot[1]["value"]),
        ),
    )


def test_wikidata_path_conclusion_is_not_relation_hash_determined():
    # The discredited PID-hash compose code is gone; the path lane no longer
    # derives any conclusion from hashing a relation name.
    assert not hasattr(renderers_module, "_reasoning_compose_code")
    assert REASONING_RENDERER_TYPES[1][0] == "wikidata_path_reasoning"


def test_wikidata_path_rejects_disconnected_walk(
    wikidata_path_authority,
    route_index,
):
    auth = wikidata_path_authority
    # Rows (2, 1) traverse edge Q2->Q3 then Q1->Q2: the endpoint Q3 does not
    # match the next subject Q1, so the walk is not grounded.
    record, routes = _prepare_wikidata_path(auth, route_index, rows=(2, 1))
    with _open_render_view(auth) as view:
        renderer = WikidataPathReasoningRenderer(auth.root, view)
        with pytest.raises(ValueError, match="connected|walk"):
            renderer.render(record, routes)


def test_wikidata_path_rejects_premise_mutation(
    wikidata_path_authority,
    route_index,
    monkeypatch: pytest.MonkeyPatch,
):
    auth = wikidata_path_authority
    record, routes = _prepare_wikidata_path(auth, route_index)
    real = renderers_module._canonical_premise_bytes

    def mutated(premises):
        value = json.loads(real(premises).decode("utf-8"))
        value[0]["value"] = value[0]["value"] + "-mutated"
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

    monkeypatch.setattr(renderers_module, "_canonical_premise_bytes", mutated)
    with _open_render_view(auth) as view:
        renderer = WikidataPathReasoningRenderer(auth.root, view)
        with pytest.raises(ValueError):
            renderer.render(record, routes)


# --- Fix B: only solver-backed reasoning families consume lane exposures ----
#
# iter_reasoning_records interleaves path_composition, date_ordering, and
# balanced_equality, but Task 5B has frozen solvers only for composition and
# equality. iter_supported_reasoning_records is the canonical shared sequence a
# production catalog source must consume: it skips unsolved families (date
# ordering) and numbers only the supported records, so a date record never
# consumes a lane exposure. Source record IDs / raw schedule provenance survive.


def test_supported_reasoning_iterator_numbers_only_solver_backed_families():
    tok = get_tok()
    seed = _MULTIHOP_SEED
    hops = _MULTIHOP_HOPS

    def worlds():
        return iter_worlds(64, 64, seed, 0)

    raw = list(
        islice(iter_reasoning_records(tok, worlds, seed=seed, max_hops=hops), 60)
    )
    raw_supported = [
        record
        for record in raw
        if renderers_module._supported_reasoning_family(record) is not None
    ]
    # The raw stream really does interleave an unsupported (date-ordering) family.
    assert 0 < len(raw_supported) < len(raw)
    unsupported = [
        record
        for record in raw
        if renderers_module._supported_reasoning_family(record) is None
    ]
    assert unsupported
    assert all(
        record.segments[-1].text.startswith("<|slot_") for record in unsupported
    )

    filtered = list(
        islice(
            renderers_module.iter_supported_reasoning_records(
                tok, worlds, seed=seed, max_hops=hops
            ),
            len(raw_supported),
        )
    )
    # Lane exposures are contiguous from zero and never spent on a date record.
    assert [exposure for exposure, _record in filtered] == list(
        range(len(raw_supported))
    )
    assert all(
        renderers_module._supported_reasoning_family(record) is not None
        for _exposure, record in filtered
    )
    # The filtered lane sequence is exactly the supported subset of the raw
    # stream, in order, preserving source record IDs and raw schedule provenance.
    assert [record.schedule.record_id for _exposure, record in filtered] == [
        record.schedule.record_id for record in raw_supported
    ]
    assert [record.schedule.exposure for _exposure, record in filtered] == [
        record.schedule.exposure for record in raw_supported
    ]
    # Both frozen solver families remain reachable through the filtered sequence.
    assert {
        renderers_module._supported_reasoning_family(record)
        for _exposure, record in filtered
    } == {"graph_composition_mod4", "slot_equality"}


def test_supported_reasoning_iterator_is_the_renderer_lane_sequence(
    tmp_path: Path,
    route_index,
):
    # An equality record only becomes reachable after the interleaved date
    # records are skipped; the renderer selects it by its filtered lane exposure,
    # proving date records never shift or consume a lane exposure.
    source_root = tmp_path / "sources"
    source_root.mkdir()
    renderer, record, returns, routes, rendered = _make_multihop(
        source_root, route_index, "slot_equality"
    )
    result = renderer.render(record, routes)
    assert result.proofs[0].family == "slot_equality"
    assert record.source_key == rendered.schedule.record_id
