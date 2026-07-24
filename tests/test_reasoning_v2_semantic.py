from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

from corpusgen.reasoning import (
    FactMetadata,
    SemanticFact,
    SupervisedField,
    audit_occurrence_closure,
    build_route_manifest,
    plan_occurrence_closure,
)
from corpusgen.reasoning import (
    SemanticLeakageError as ClosureSemanticLeakageError,
)
from corpusgen.reasoning_v2 import (
    RouteArtifacts,
    RouteIndex,
    SemanticLeakageError,
    SidecarWeights,
    TokenSemanticSpan,
    audit_answer_state_surfaces,
    audit_proof_surfaces,
    audit_sidecar_weights,
    build_route_artifacts,
    derive_sidecar_weights,
)
from corpusgen.reasoning_v2 import semantic as semantic_module
from corpusgen.reasoning_v2.catalog import CatalogRecord, SemanticFactRow

_QUARANTINE_DIRECTORY = ".memorysplit-route-quarantine-v1"


@dataclass(frozen=True)
class RoutingCatalog:
    records: tuple[CatalogRecord, ...]

    def iter_records(self):
        return iter(self.records)


def _semantic_fact(
    fact_id: str,
    *,
    entropy: Fraction | int = 10,
    exposures: int = 1,
    expected_reads: Fraction | int = 0,
    expected_hops: Fraction | int = 0,
    source: str = "fixture",
    record_type: str = "graph",
    surfaces: tuple[str, ...] | None = None,
) -> SemanticFactRow:
    return SemanticFactRow(
        fact_id=fact_id,
        source=source,
        record_type=record_type,
        payload_entropy_bits=Fraction(entropy),
        scheduled_exposures=exposures,
        expected_reads=Fraction(expected_reads),
        expected_hops=Fraction(expected_hops),
        surfaces=surfaces or (f"value:{fact_id}",),
    )


def _record(
    index: int,
    *facts: SemanticFactRow,
) -> CatalogRecord:
    return CatalogRecord(
        ordinal=index,
        record_id=f"{index:064x}",
        lane_id="wikidata_graph",
        source_id="wikidata5m",
        source_key=f"edge-{index:08d}",
        source_byte_sha256="a" * 64,
        source_locator=(
            ("path", "wikidata5m_transductive.tar.gz"),
            ("row", index),
            ("split", "train"),
        ),
        target_count=1,
        semantic_flags=("graph-training-edge",),
        semantic_facts=tuple(facts),
    )


def _routing_rows() -> tuple[CatalogRecord, ...]:
    rows = [
        _record(
            index,
            _semantic_fact(
                f"fact-{index:02d}",
                entropy=100 - index,
            ),
        )
        for index in range(9)
    ]
    heavy = _semantic_fact("fact-heavy", entropy=91, exposures=40)
    rows.extend(
        (
            _record(9, heavy),
            _record(10, replace(heavy, scheduled_exposures=60)),
        )
    )
    return tuple(rows)


def _aggregated_facts() -> tuple[FactMetadata, ...]:
    return tuple(
        [
            FactMetadata(
                fact_id=f"fact-{index:02d}",
                source="fixture",
                record_type="graph",
                payload_entropy_bits=100 - index,
                scheduled_exposures=1,
                expected_reads=0,
                expected_hops=0,
                surfaces=(f"value:fact-{index:02d}",),
            )
            for index in range(9)
        ]
        + [
            FactMetadata(
                fact_id="fact-heavy",
                source="fixture",
                record_type="graph",
                payload_entropy_bits=91,
                scheduled_exposures=100,
                expected_reads=0,
                expected_hops=0,
                surfaces=("value:fact-heavy",),
            )
        ]
    )


def _manifest_lines(path: Path) -> tuple[bytes, ...]:
    payload = path.read_bytes()
    lines = tuple(payload.splitlines(keepends=True))
    assert payload == b"".join(lines)
    assert all(line.endswith(b"\n") for line in lines)
    return lines


def _quarantined_directories(parent: Path) -> tuple[Path, ...]:
    root = parent / _QUARANTINE_DIRECTORY
    assert root.is_dir()
    return tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    )


def _write_route_index(path: Path, external_fact_ids: set[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE selected ("
            "fact_id TEXT NOT NULL PRIMARY KEY"
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
    return path


@pytest.fixture
def route_index(tmp_path: Path):
    indexes: list[RouteIndex] = []

    def open_index(external_fact_ids: set[str]) -> RouteIndex:
        path = _write_route_index(
            tmp_path / f"route-index-{len(indexes)}.sqlite3",
            external_fact_ids,
        )
        index = RouteIndex.open(path)
        indexes.append(index)
        return index

    yield open_index

    for index in indexes:
        index.close()


def test_production_route_manifest_matches_reviewed_in_memory_policy_and_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(semantic_module, "_EXTERNAL_SORT_CHUNK_ROWS", 3)
    catalog = RoutingCatalog(_routing_rows())

    artifacts = build_route_artifacts(
        catalog,
        tmp_path / "work",
        tmp_path / "routes",
    )
    expected = build_route_manifest(_aggregated_facts(), "Split90")

    assert isinstance(artifacts, RouteArtifacts)
    with artifacts.open_index() as index:
        assert tuple(index.iter_external_fact_ids()) == tuple(
            sorted(expected.external_fact_ids)
        )
    assert artifacts.external_fact_count == expected.external_count == 9
    assert artifacts.distinct_fact_fraction >= Fraction(9, 10)
    assert artifacts.information_burden_fraction == (
        expected.information_burden_fraction
    )
    assert artifacts.information_burden_fraction >= Fraction(9, 10)
    assert artifacts.dose_report["passed"] is True
    assert "fact-heavy" in expected.external_fact_ids

    lines = _manifest_lines(artifacts.manifest_path)
    header = json.loads(lines[0])
    decisions = tuple(json.loads(line) for line in lines[1:])
    assert header["format"] == "memorysplit-production-route-manifest-v1"
    assert header["policy"] == "train-score-ranked-quota-v1"
    assert header["metadata_scope"] == "training-only"
    assert header["row_count"] == 10
    assert header["decision_stream_sha256"] == hashlib.sha256(
        b"".join(lines[1:])
    ).hexdigest()
    assert decisions == tuple(decision.as_dict() for decision in expected.decisions)
    heavy = next(row for row in decisions if row["fact_id"] == "fact-heavy")
    assert heavy["scheduled_exposures"] == 100

    work_stage = next(
        path
        for path in _quarantined_directories(tmp_path)
        if tuple(path.glob("rank-chunk-*.jsonl"))
    )
    rank_chunks = tuple(work_stage.glob("rank-chunk-*.jsonl"))
    assert len(rank_chunks) >= 4
    assert all(
        1 <= len(path.read_bytes().splitlines()) <= 3 for path in rank_chunks
    )


def test_reverse_catalog_order_is_byte_identical_and_exact_winner_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(semantic_module, "_EXTERNAL_SORT_CHUNK_ROWS", 3)
    rows = _routing_rows()
    output = tmp_path / "routes"
    first = build_route_artifacts(
        RoutingCatalog(rows),
        tmp_path / "work-a",
        output,
    )
    before = _quarantined_directories(tmp_path)
    second = build_route_artifacts(
        RoutingCatalog(tuple(reversed(rows))),
        tmp_path / "work-b",
        output,
    )
    after = _quarantined_directories(tmp_path)

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.dose_report_sha256 == second.dose_report_sha256
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.dose_report_path.read_bytes() == second.dose_report_path.read_bytes()
    with first.open_index() as first_index, second.open_index() as second_index:
        assert tuple(first_index.iter_external_fact_ids()) == tuple(
            second_index.iter_external_fact_ids()
        )
    assert len(after) >= len(before) + 2
    losing_candidate = next(
        path
        for path in after
        if path not in before and (path / "route-index.sqlite3").is_file()
    )
    assert (
        (losing_candidate / "route-index.sqlite3").read_bytes()
        == first.index_path.read_bytes()
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "dose-report.json",
        "route-index.sqlite3",
        "route-manifest.jsonl",
    ]


def test_existing_route_winner_requires_exact_index_bytes(tmp_path: Path):
    output = tmp_path / "routes"
    first = build_route_artifacts(
        RoutingCatalog(_routing_rows()),
        tmp_path / "work-a",
        output,
    )
    before = first.index_path.read_bytes()
    connection = sqlite3.connect(first.index_path)
    try:
        connection.execute("PRAGMA application_id=1297306452")
        connection.commit()
    finally:
        connection.close()
    tampered = first.index_path.read_bytes()
    assert tampered != before

    with pytest.raises(ValueError, match="conflicting route winner"):
        build_route_artifacts(
            RoutingCatalog(tuple(reversed(_routing_rows()))),
            tmp_path / "work-b",
            output,
        )

    assert first.index_path.read_bytes() == tampered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "other-source"),
        ("record_type", "other-record"),
        ("payload_entropy_bits", Fraction(92)),
        ("expected_reads", Fraction(1)),
        ("expected_hops", Fraction(1)),
        ("surfaces", ("other-surface",)),
    ],
)
def test_repeated_fact_requires_byte_identical_nonexposure_metadata(
    tmp_path: Path,
    field: str,
    value: object,
):
    first = _semantic_fact("repeated", exposures=2)
    second = replace(
        first,
        scheduled_exposures=3,
        **{field: value},
    )

    with pytest.raises(ValueError, match="conflicting metadata.*repeated"):
        build_route_artifacts(
            RoutingCatalog((_record(0, first), _record(1, second))),
            tmp_path / "work",
            tmp_path / "routes",
        )

    assert not (tmp_path / "routes").exists()
    retained = _quarantined_directories(tmp_path)
    assert retained


@pytest.mark.parametrize(
    "fact",
    [
        replace(_semantic_fact("fact"), scheduled_exposures=0),
        replace(_semantic_fact("fact"), scheduled_exposures=True),
        replace(
            _semantic_fact("fact"),
            payload_entropy_bits=1,  # type: ignore[arg-type]
        ),
        replace(_semantic_fact("fact"), surfaces=()),
        replace(_semantic_fact("fact"), surfaces=("zeta", "alpha")),
        replace(_semantic_fact("fact"), fact_id="e\u0301"),
    ],
)
def test_route_reducer_rejects_forged_noncanonical_fact_rows(
    tmp_path: Path,
    fact: SemanticFactRow,
):
    with pytest.raises(ValueError, match="semantic|route"):
        build_route_artifacts(
            RoutingCatalog((_record(0, fact),)),
            tmp_path / "work",
            tmp_path / "routes",
        )
    assert not (tmp_path / "routes").exists()


def test_route_reducer_rejects_non_catalog_rows(tmp_path: Path):
    catalog = RoutingCatalog((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CatalogRecord"):
        build_route_artifacts(
            catalog,
            tmp_path / "work",
            tmp_path / "routes",
        )


def test_exact_rational_route_dose_rejects_either_threshold():
    passing = semantic_module.audit_route_dose(
        total_facts=10,
        external_facts=9,
        total_information_burden=Fraction(100),
        external_information_burden=Fraction(90),
    )
    assert passing["passed"] is True
    assert passing["distinct_fact_fraction"] == {
        "denominator": 10,
        "numerator": 9,
    }
    assert passing["information_burden_fraction"] == {
        "denominator": 10,
        "numerator": 9,
    }

    with pytest.raises(ValueError, match="distinct fact"):
        semantic_module.audit_route_dose(
            total_facts=10,
            external_facts=8,
            total_information_burden=Fraction(100),
            external_information_burden=Fraction(100),
        )
    with pytest.raises(ValueError, match="information burden"):
        semantic_module.audit_route_dose(
            total_facts=10,
            external_facts=9,
            total_information_burden=Fraction(100),
            external_information_burden=Fraction(89),
        )


def test_conflicting_existing_route_winner_is_preserved_and_loser_quarantined(
    tmp_path: Path,
):
    output = tmp_path / "routes"
    first = build_route_artifacts(
        RoutingCatalog(_routing_rows()),
        tmp_path / "work-a",
        output,
    )
    manifest_before = first.manifest_path.read_bytes()
    dose_before = first.dose_report_path.read_bytes()
    changed = list(_routing_rows())
    changed[0] = replace(
        changed[0],
        semantic_facts=(
            replace(
                changed[0].semantic_facts[0],
                payload_entropy_bits=Fraction(1),
            ),
        ),
    )
    before = _quarantined_directories(tmp_path)

    with pytest.raises(ValueError, match="conflicting route winner"):
        build_route_artifacts(
            RoutingCatalog(tuple(changed)),
            tmp_path / "work-b",
            output,
        )

    after = _quarantined_directories(tmp_path)
    assert first.manifest_path.read_bytes() == manifest_before
    assert first.dose_report_path.read_bytes() == dose_before
    assert len(after) >= len(before) + 2


def test_failed_route_stages_are_quarantined_without_pathname_deletion(
    tmp_path: Path,
):
    first = _semantic_fact("repeated", exposures=1)
    second = replace(first, source="conflict")

    with pytest.raises(ValueError, match="conflicting metadata"):
        build_route_artifacts(
            RoutingCatalog((_record(0, first), _record(1, second))),
            tmp_path / "work",
            tmp_path / "routes",
        )

    assert not tuple(tmp_path.glob(".work.tmp-*"))
    assert not tuple(tmp_path.glob(".routes.tmp-*"))
    assert _quarantined_directories(tmp_path)
    source = inspect.getsource(semantic_module)
    assert ".resolve(" not in source
    assert "os.unlink(" not in source
    assert "os.rmdir(" not in source
    assert "shutil.rmtree(" not in source
    assert ".unlink(" not in source


def test_route_publication_rejects_symlinked_output_parent(tmp_path: Path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe"):
        build_route_artifacts(
            RoutingCatalog(_routing_rows()),
            tmp_path / "work",
            linked_parent / "routes",
        )

    assert not (real_parent / "routes").exists()


def test_route_index_rejects_symlink_and_hardlink(tmp_path: Path):
    original = _write_route_index(tmp_path / "route-index.sqlite3", {"fact-a"})
    symlink = tmp_path / "symlink.sqlite3"
    symlink.symlink_to(original)
    with pytest.raises(ValueError, match="owner-controlled|unsafe"):
        RouteIndex.open(symlink)

    hardlink = tmp_path / "hardlink.sqlite3"
    os.link(original, hardlink)
    with pytest.raises(ValueError, match="owner-controlled"):
        RouteIndex.open(original)


def test_route_index_open_detects_namespace_swap_and_preserves_both_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write_route_index(tmp_path / "route-index.sqlite3", {"fact-a"})
    swapped = False

    def swap(
        phase: str,
        parent_fd: int,
        name: str,
        pinned_fd: int,
    ) -> None:
        nonlocal swapped
        del pinned_fd
        if phase != "before_sqlite_open" or swapped:
            return
        swapped = True
        os.rename(
            name,
            ".attacker-original-index.sqlite3",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        replacement = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        os.close(replacement)

    monkeypatch.setattr(semantic_module, "_route_index_open_hook", swap)
    with pytest.raises(ValueError, match="identity"):
        RouteIndex.open(path)
    assert path.is_file()
    assert (tmp_path / ".attacker-original-index.sqlite3").is_file()


def test_route_index_open_hook_failure_closes_the_sqlite_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write_route_index(tmp_path / "route-index.sqlite3", {"fact-a"})
    original_connect = semantic_module.sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def fail_after_open(
        phase: str,
        parent_fd: int,
        name: str,
        pinned_fd: int,
    ) -> None:
        del parent_fd, name, pinned_fd
        if phase == "sqlite_opened":
            raise RuntimeError("injected post-open failure")

    monkeypatch.setattr(semantic_module.sqlite3, "connect", tracking_connect)
    monkeypatch.setattr(semantic_module, "_route_index_open_hook", fail_after_open)
    with pytest.raises(RuntimeError, match="injected post-open failure"):
        RouteIndex.open(path)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_route_index_query_replays_namespace_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write_route_index(tmp_path / "route-index.sqlite3", {"fact-a"})
    with RouteIndex.open(path) as index:
        swapped = False

        def swap(
            phase: str,
            parent_fd: int,
            name: str,
            pinned_fd: int,
        ) -> None:
            nonlocal swapped
            del pinned_fd
            if phase != "before_query" or swapped:
                return
            swapped = True
            os.rename(
                name,
                ".attacker-query-original.sqlite3",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            replacement = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            os.close(replacement)

        monkeypatch.setattr(semantic_module, "_route_index_read_hook", swap)
        with pytest.raises(ValueError, match="identity"):
            index.is_external("fact-a")

    assert path.is_file()
    assert (tmp_path / ".attacker-query-original.sqlite3").is_file()


def test_split90_masks_only_every_routed_factual_payload_token(route_index):
    spans = (
        TokenSemanticSpan(0, 2, "fact-a", "factual_payload"),
        TokenSemanticSpan(2, 4, "fact-a", "answer_state"),
        TokenSemanticSpan(4, 6, None, "rule"),
        TokenSemanticSpan(6, 8, "fact-a", "factual_payload"),
    )
    with route_index({"fact-a"}) as index:
        weights = derive_sidecar_weights(8, spans, index)

    assert isinstance(weights, SidecarWeights)
    assert weights.dense == b"\x01" * 8
    assert weights.split90 == b"\x00\x00\x01\x01\x01\x01\x00\x00"
    assert weights.routed_payload_targets == 4
    assert weights.leaks == ()


def test_same_fact_overlapping_payload_spans_mask_the_exact_union(route_index):
    spans = (
        TokenSemanticSpan(0, 4, "fact-a", "factual_payload"),
        TokenSemanticSpan(2, 6, "fact-a", "factual_payload"),
        TokenSemanticSpan(6, 8, None, "operator"),
    )
    with route_index({"fact-a"}) as index:
        weights = derive_sidecar_weights(8, spans, index)
    assert weights.split90 == b"\x00" * 6 + b"\x01" * 2
    assert weights.routed_payload_targets == 6


@pytest.mark.parametrize(
    ("token_count", "spans", "message"),
    [
        (
            4,
            (
                TokenSemanticSpan(2, 4, "fact-a", "factual_payload"),
                TokenSemanticSpan(0, 2, "fact-a", "factual_payload"),
            ),
            "ordered",
        ),
        (
            3,
            (TokenSemanticSpan(0, 4, "fact-a", "factual_payload"),),
            "bounds",
        ),
        (
            4,
            (
                TokenSemanticSpan(0, 3, "fact-a", "factual_payload"),
                TokenSemanticSpan(2, 4, "fact-b", "factual_payload"),
            ),
            "overlap",
        ),
    ],
)
def test_semantic_spans_fail_closed_on_order_bounds_and_cross_fact_overlap(
    route_index,
    token_count: int,
    spans: tuple[TokenSemanticSpan, ...],
    message: str,
):
    with (
        route_index({"fact-a", "fact-b"}) as index,
        pytest.raises(ValueError, match=message),
    ):
        derive_sidecar_weights(token_count, spans, index)


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        (
            (
                TokenSemanticSpan(0, 2, "fact-a", "factual_payload"),
                TokenSemanticSpan(0, 2, "fact-a", "proof"),
            ),
            "proof",
        ),
        (
            (
                TokenSemanticSpan(0, 2, "fact-a", "answer_state"),
                TokenSemanticSpan(0, 2, "fact-a", "factual_payload"),
            ),
            "answer state",
        ),
    ],
)
def test_proof_or_answer_tokens_cannot_also_be_marked_factual(
    route_index,
    spans: tuple[TokenSemanticSpan, ...],
    message: str,
):
    with (
        route_index({"fact-a"}) as index,
        pytest.raises(SemanticLeakageError, match=message),
    ):
        derive_sidecar_weights(2, spans, index)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TokenSemanticSpan(True, 1, "fact-a", "factual_payload"),
        lambda: TokenSemanticSpan(0, 0, "fact-a", "factual_payload"),
        lambda: TokenSemanticSpan(0, 1, None, "factual_payload"),
        lambda: TokenSemanticSpan(0, 1, "fact-a", "unknown"),  # type: ignore[arg-type]
        lambda: TokenSemanticSpan(0, 1, 7, "factual_payload"),  # type: ignore[arg-type]
    ],
)
def test_token_semantic_span_rejects_invalid_scalar_or_role(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_sidecar_audit_rejects_one_unmasked_target_and_nonbinary_value(route_index):
    spans = (
        TokenSemanticSpan(0, 3, "fact-a", "factual_payload"),
        TokenSemanticSpan(3, 4, None, "rule"),
    )
    with route_index({"fact-a"}) as index:
        with pytest.raises(SemanticLeakageError, match="unmasked"):
            audit_sidecar_weights(
                4,
                spans,
                index,
                dense=b"\x01" * 4,
                split90=b"\x00\x01\x00\x01",
            )
        with pytest.raises(SemanticLeakageError, match="binary"):
            audit_sidecar_weights(
                4,
                spans,
                index,
                dense=b"\x01" * 4,
                split90=b"\x00\x00\x02\x01",
            )


def test_sidecar_audit_rejects_masking_nonpayload_or_dense_targets(route_index):
    spans = (
        TokenSemanticSpan(0, 2, "fact-a", "factual_payload"),
        TokenSemanticSpan(2, 4, None, "proof"),
    )
    with route_index({"fact-a"}) as index:
        with pytest.raises(SemanticLeakageError, match="nonpayload"):
            audit_sidecar_weights(
                4,
                spans,
                index,
                dense=b"\x01" * 4,
                split90=b"\x00\x00\x00\x01",
            )
        with pytest.raises(SemanticLeakageError, match="Dense"):
            audit_sidecar_weights(
                4,
                spans,
                index,
                dense=b"\x01\x00\x01\x01",
                split90=b"\x00\x00\x01\x01",
            )


def test_answer_state_surface_copy_fails_closed_on_every_occurrence():
    with pytest.raises(SemanticLeakageError, match="answer state") as caught:
        audit_answer_state_surfaces(
            answer_state=(
                '<|answer_state|>{"candidate":"cerulean",'
                '"final":"cerulean"}'
            ),
            routed_facts=(SemanticFact("fact-a", ("cerulean",)),),
        )
    assert isinstance(caught.value, ClosureSemanticLeakageError)
    assert len(caught.value.report.unmasked_occurrences) == 2


def test_proof_surface_copy_fails_closed():
    with pytest.raises(SemanticLeakageError, match="proof"):
        audit_proof_surfaces(
            proof_text="read fact-a as cerulean",
            routed_facts=(SemanticFact("fact-a", ("cerulean",)),),
        )


def test_answer_and_proof_surface_audits_accept_value_free_text():
    facts = (SemanticFact("fact-a", ("cerulean",)),)
    assert (
        audit_answer_state_surfaces(
            answer_state='<|answer_state|>{"slot":"<|slot_1|>"}',
            routed_facts=facts,
        )
        == ()
    )
    assert (
        audit_proof_surfaces(
            proof_text="read slot 1; apply equality",
            routed_facts=facts,
        )
        == ()
    )


def test_surface_closure_rejects_overlap_missing_fact_unmasked_and_nonbinary():
    fields = (SupervisedField("payload", "cerulean cerulean"),)
    facts = (SemanticFact("fact-a", ("cerulean",)),)
    plan = plan_occurrence_closure(facts, fields)
    masks = {
        field_id: list(values)
        for field_id, values in plan.mask_for_routes({"fact-a"}).items()
    }
    masks["payload"][len("cerulean ")] = 1
    with pytest.raises(ClosureSemanticLeakageError):
        audit_occurrence_closure(facts, fields, {"fact-a"}, masks)

    nonbinary = plan.mask_for_routes({"fact-a"})
    with pytest.raises(ClosureSemanticLeakageError, match="metadata"):
        audit_occurrence_closure(
            facts,
            fields,
            {"fact-a"},
            {"payload": (2,) + nonbinary["payload"][1:]},
        )
    with pytest.raises(ClosureSemanticLeakageError, match="metadata"):
        audit_occurrence_closure(
            facts,
            fields,
            {"missing"},
            nonbinary,
        )
    with pytest.raises(ValueError, match="cross-fact overlapping"):
        plan_occurrence_closure(
            (
                SemanticFact("fact-a", ("cerulean",)),
                SemanticFact("fact-b", ("cerulean blue",)),
            ),
            (SupervisedField("payload", "cerulean blue"),),
        )
