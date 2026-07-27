from __future__ import annotations

import hashlib
import inspect
import io
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

import corpusgen.wikidata_paths as wikidata_paths_module
from corpusgen.graph_records import GraphAddress, GraphRow
from corpusgen.wikidata_path_replay import replay_and_validate_path_twins
from corpusgen.relation_schema import (
    RelationSchema,
    RelationSpec,
    build_relation_schema,
)
from corpusgen.wikidata5m import (
    ArchiveLock,
    SourceDriftError,
    WikidataLock,
    read_aliases,
)
from corpusgen.wikidata_paths import (
    FROZEN_EXCLUSION_REASONS,
    _emit_path_pairs,
    build_wikidata_paths,
)
from organizer.packed_graph_store import PackedGraphStore
from scripts.run_wikidata_evals import run_wikidata_evaluation


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _lock_for_archives(root: Path) -> WikidataLock:
    return WikidataLock(
        repo_id="fixture/wikidata5m",
        repo_type="dataset",
        revision="a" * 40,
        files={
            path.name: ArchiveLock(
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(root.glob("*.tar.gz"))
        },
    )


def _default_valid_bytes() -> bytes:
    rows = [
        "Q1\tP1\tQ2\n",
        "Q2\tP2\tQ3\n",
        "Q3\tP3\tQ4\n",
        "Q4\tP4\tQ5\n",
        "Q5\tP5\tQ6\n",
        "Q6\tP6\tQ7\n",
        "Q50\tP1\tQ51\n",
        "Q50\tP1\tQ52\n",
        "Q60\tP99\tQ61\n",
    ]
    rows.extend(
        f"Q{100 + index}\tP{index}\tQ{200 + index}\n"
        for index in range(1, 7)
    )
    return "".join(rows).encode()


def _make_locked_source(
    root: Path,
    *,
    valid_bytes: bytes | None = None,
) -> tuple[Path, WikidataLock, RelationSchema]:
    root.mkdir(parents=True, exist_ok=True)
    aliases = "".join(
        f"P{index}\trelation {index}\tproperty {index}\n"
        for index in range(1, 33)
    ) + "P99\tunselected relation\n"
    transductive = "".join(
        f"Q{subject}\tP{relation}\tQ{1_000_000 + relation * 10_000 + subject}\n"
        for relation in range(1, 33)
        for subject in range(1, 5_001)
    )
    train_bytes = b"Q900\tP1\tQ901\n"
    _write_tar(
        root / "wikidata5m_alias.tar.gz",
        {
            "wikidata5m_relation.txt": aliases.encode(),
            "wikidata5m_entity.txt": b"Q1\tentity one\n",
        },
    )
    _write_tar(
        root / "wikidata5m_transductive.tar.gz",
        {"wikidata5m_transductive_train.txt": transductive.encode()},
    )
    _write_tar(
        root / "wikidata5m_inductive.tar.gz",
        {
            "wikidata5m_inductive_train.txt": train_bytes,
            "wikidata5m_inductive_valid.txt": (
                _default_valid_bytes()
                if valid_bytes is None
                else valid_bytes
            ),
            "wikidata5m_inductive_test.txt": b"Q700\tP1\tQ701\n",
        },
    )
    schema_input = root.parent / f"{root.name}-schema-input"
    schema_input.mkdir()
    relation_path = schema_input / "wikidata5m_relation.txt"
    relation_path.write_text(aliases, encoding="utf-8")
    transductive_path = (
        schema_input / "wikidata5m_transductive_train.txt"
    )
    transductive_path.write_text(transductive, encoding="utf-8")
    try:
        schema = build_relation_schema(
            transductive_path,
            read_aliases(relation_path, "P"),
        )
    finally:
        shutil.rmtree(schema_input)
    return root, _lock_for_archives(root), schema


@pytest.fixture(scope="module")
def locked_source(tmp_path_factory):
    return _make_locked_source(tmp_path_factory.mktemp("wikidata-archives"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _build(tmp_path: Path, locked_source):
    source, lock, schema = locked_source
    return wikidata_paths_module.build_fixture_wikidata_paths(
        source,
        schema,
        "valid",
        tmp_path / "robustness",
        fixture_lock=lock,
    )


def test_candidate_accounting_is_exhaustive(tmp_path, locked_source):
    report = _build(tmp_path, locked_source)

    assert report.candidates == report.surviving + sum(
        report.exclusions.values()
    )
    assert report.paths.candidates == report.paths.surviving + sum(
        report.paths.exclusions.values()
    )
    assert set(report.exclusions) == {
        "ambiguous_address",
        "unselected_relation",
    }
    assert set(report.paths.exclusions) <= set(FROZEN_EXCLUSION_REASONS)


def test_ambiguous_addresses_are_excluded_not_ranked(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)

    assert report.exclusions["ambiguous_address"] == 1
    with report.open_graph() as graph:
        assert graph.lookup(GraphAddress(50, "P1", "out")) is None


def test_builder_uses_only_requested_inductive_split_and_selected_relations(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)

    with report.open_graph() as graph:
        assert graph.lookup(GraphAddress(900, "P1", "out")) is None
        assert graph.lookup(GraphAddress(700, "P1", "out")) is None
        assert graph.lookup(GraphAddress(60, "P99", "out")) is None
    assert report.split == "valid"
    assert report.address_policy == "unique_only_no_ranking"


def test_production_builder_rejects_arbitrary_extracted_text(tmp_path, locked_source):
    _, _, schema = locked_source
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "wikidata5m_inductive_valid.txt").write_bytes(
        _default_valid_bytes()
    )

    with pytest.raises(SourceDriftError, match="locked archive"):
        build_wikidata_paths(
            extracted,
            schema,
            "valid",
            tmp_path / "out",
        )


def test_production_builder_has_no_custom_lock_escape(tmp_path, locked_source):
    source, lock, schema = locked_source

    assert "fixture_lock" not in inspect.signature(
        build_wikidata_paths
    ).parameters
    with pytest.raises(TypeError, match="fixture_lock"):
        build_wikidata_paths(
            source,
            schema,
            "valid",
            tmp_path / "out",
            fixture_lock=lock,
        )


def test_fixture_builder_marks_artifacts_non_production(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)
    persisted = json.loads(
        (report.out_dir / "coverage-manifest.json").read_text(encoding="utf-8")
    )

    assert report.artifact_mode == "fixture"
    assert report.production_evaluation_eligible is False
    assert persisted["artifact_mode"] == "fixture"
    assert persisted["production_evaluation_eligible"] is False


def test_production_builder_cli_exposes_no_fixture_lock_option():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_wikidata_robustness.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--fixture-lock" not in result.stdout


def test_production_evaluator_has_no_custom_lock_escape():
    assert "source_lock" not in inspect.signature(
        run_wikidata_evaluation
    ).parameters


def test_production_evaluator_cli_exposes_no_source_lock_option():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_wikidata_evals.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--source-lock" not in result.stdout


def test_production_evaluator_rejects_fixture_artifacts_before_checkpoint_load(
    tmp_path, locked_source
):
    _, _, schema = locked_source
    report = _build(tmp_path / "build", locked_source)
    schema_path = tmp_path / "schema.json"
    schema.write(schema_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "ckpt.pt").write_bytes(b"not a checkpoint")
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("frozen", encoding="utf-8")
    output = tmp_path / "results"

    with pytest.raises(ValueError, match="non-production fixture"):
        run_wikidata_evaluation(
            run=run,
            checkpoint="ckpt.pt",
            artifacts=report.out_dir,
            relation_schema=schema_path,
            preregistration=preregistration,
            out_dir=output,
            device="cpu",
        )

    assert not output.exists()


def test_builder_rejects_wrong_archive_byte(tmp_path, locked_source):
    source, lock, schema = locked_source
    corrupt = tmp_path / "corrupt"
    shutil.copytree(source, corrupt)
    archive = corrupt / "wikidata5m_inductive.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"!")

    with pytest.raises(SourceDriftError, match="inductive"):
        wikidata_paths_module.build_fixture_wikidata_paths(
            corrupt,
            schema,
            "valid",
            tmp_path / "out",
            fixture_lock=lock,
        )


def test_builder_rejects_symlinked_locked_archive(tmp_path, locked_source):
    source, lock, schema = locked_source
    linked = tmp_path / "linked"
    linked.mkdir()
    for archive in source.glob("*.tar.gz"):
        destination = linked / archive.name
        if archive.name == "wikidata5m_inductive.tar.gz":
            destination.symlink_to(archive)
        else:
            shutil.copy2(archive, destination)

    with pytest.raises(SourceDriftError, match="inductive"):
        wikidata_paths_module.build_fixture_wikidata_paths(
            linked,
            schema,
            "valid",
            tmp_path / "out",
            fixture_lock=lock,
        )


def test_builder_rejects_train_content_renamed_as_valid(tmp_path):
    source, lock, schema = _make_locked_source(
        tmp_path / "renamed",
        valid_bytes=b"Q900\tP1\tQ901\n",
    )

    with pytest.raises(ValueError, match="split content"):
        wikidata_paths_module.build_fixture_wikidata_paths(
            source,
            schema,
            "valid",
            tmp_path / "out",
            fixture_lock=lock,
        )


def test_builder_rejects_wrong_split_before_extraction(tmp_path, locked_source):
    source, lock, schema = locked_source

    with pytest.raises(ValueError, match="valid or test"):
        wikidata_paths_module.build_fixture_wikidata_paths(
            source,
            schema,
            "train",
            tmp_path / "out",
            fixture_lock=lock,
        )


def test_builder_recomputes_and_rejects_structurally_valid_schema_drift(
    tmp_path, locked_source
):
    source, lock, schema = locked_source
    drifted_catalog = (
        replace(schema.catalog[0], aliases=("drifted alias",)),
        *schema.catalog[1:],
    )
    drifted = RelationSchema(
        drifted_catalog,
        schema.path_relation_ids,
        schema.ambiguous_normalized,
    )

    with pytest.raises(ValueError, match="recomputed relation schema"):
        wikidata_paths_module.build_fixture_wikidata_paths(
            source,
            drifted,
            "valid",
            tmp_path / "out",
            fixture_lock=lock,
        )


def test_manifest_records_verified_archive_and_recomputed_schema_hashes(
    tmp_path, locked_source
):
    _, lock, schema = locked_source
    report = _build(tmp_path, locked_source)

    assert report.recomputed_schema_sha256 == schema.sha256()
    assert report.source_archive_sha256 == {
        name: item.sha256 for name, item in lock.files.items()
    }


def test_complete_one_to_six_hop_paths_emit_both_answer_flipping_tasks(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)
    originals = _jsonl(report.out_dir / "eval" / "original.jsonl")
    counterfactuals = _jsonl(
        report.out_dir / "eval" / "counterfactual.jsonl"
    )

    assert {row["task"] for row in originals} == {
        "endpoint_equality",
        "endpoint_traversal",
    }
    assert {row["meta"]["hop_count"] for row in originals} == set(range(1, 7))
    assert report.paths.per_hop["6"].surviving == 1
    by_pair = {row["meta"]["pair_id"]: row for row in counterfactuals}
    assert set(by_pair) == {row["meta"]["pair_id"] for row in originals}
    for original in originals:
        changed = by_pair[original["meta"]["pair_id"]]
        assert original["prompt"] == changed["prompt"]
        assert original["answer"] != changed["answer"]
        assert len(set(original["meta"]["answer_choices"])) == len(
            original["meta"]["answer_choices"]
        )
        assert original["meta"]["changed_row"] is None
        assert changed["meta"]["changed_row"] is not None
        assert (
            changed["meta"]["changed_row"]["relation_id"]
            == changed["meta"]["gold_addresses"][-1][1]
        )
        assert len(changed["meta"]["gold_addresses"]) == changed["meta"][
            "hop_count"
        ]
        if original["task"] == "endpoint_equality":
            comparison = original["meta"]["comparison_entity"]
            assert (
                original["meta"]["oracle_endpoint"] == comparison
            ) == (original["answer"] == "yes")
            assert (
                changed["meta"]["oracle_endpoint"] == comparison
            ) == (changed["answer"] == "yes")


def test_counterfactuals_disclose_same_relation_domain_not_type_compatibility(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)
    rows = _jsonl(report.out_dir / "eval" / "counterfactual.jsonl")

    assert report.type_metadata == "unavailable"
    assert report.counterfactual_compatibility == "same_relation_domain"
    for row in rows:
        meta = row["meta"]
        assert meta["type_metadata"] == "unavailable"
        assert (
            meta["counterfactual_compatibility"]
            == "same_relation_domain"
        )
        assert "type_compatible" not in json.dumps(row)
        assert meta["source_capabilities"] == {
            "literals": False,
            "qualifiers": False,
            "ranks": False,
            "types": False,
        }


def test_missing_hops_are_counted_and_never_emitted(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)
    emitted = _jsonl(report.out_dir / "eval" / "original.jsonl")

    assert report.paths.exclusions["missing_hop"] > 0
    assert all(
        len(row["meta"]["gold_addresses"]) == row["meta"]["hop_count"]
        for row in emitted
    )


def test_missing_frozen_relation_is_excluded_instead_of_ranking_another_edge(
    tmp_path, locked_source
):
    report = _build(tmp_path, locked_source)
    originals = _jsonl(report.out_dir / "eval" / "original.jsonl")

    assert not any(
        row["meta"]["entity_slots"][0] == 101
        and row["meta"]["hop_count"] == 2
        for row in originals
    )
    assert report.paths.exclusions["missing_hop"] > 0


def test_replay_rejects_cyclic_path_whose_changed_address_repeats():
    rows = (
        (1, "P1", 2),
        (2, "P1", 1),
        (1, "P1", 2),
    )
    originals, counterfactuals = _emit_path_pairs(
        split="valid",
        candidate_key="a" * 24,
        rows=rows,
        alternative=9,
        specs={"P1": RelationSpec("P1", ("parent",), "entity")},
    )
    base_rows = tuple(
        GraphRow(
            source,
            relation,
            "out",
            "entity",
            str(target),
            (),
            "wikidata5m-inductive-valid",
        )
        for source, relation, target in rows
    )

    with pytest.raises(ValueError, match="changed address repeats"):
        replay_and_validate_path_twins(
            base_rows,
            originals,
            counterfactuals,
        )


def test_robustness_builder_is_deterministic_and_emits_no_training_artifact(
    tmp_path, locked_source
):
    first = _build(tmp_path / "first", locked_source)
    second = _build(tmp_path / "second", locked_source)

    assert first.artifacts == second.artifacts
    for root in (first.out_dir, second.out_dir):
        assert not list(root.glob("**/train.bin"))
        assert not list(root.glob("**/*.weights.bin"))
        assert not list(root.glob("**/*optimizer*"))
        assert not list(root.glob("**/*checkpoint*"))


def test_post_build_audit_rejects_bad_materialized_lookup(
    monkeypatch, tmp_path, locked_source
):
    real_lookup = PackedGraphStore.lookup

    def corrupt_lookup(store, address):
        row = real_lookup(store, address)
        if row is None:
            return None
        return replace(row, target=str(int(row.target) + 1))

    monkeypatch.setattr(PackedGraphStore, "lookup", corrupt_lookup)

    with pytest.raises(ValueError, match="materialized packed store"):
        _build(tmp_path, locked_source)

    assert not (tmp_path / "robustness").exists()
    assert not list(tmp_path.glob(".robustness.*"))


def test_cleanup_failure_prevents_publication_and_is_not_suppressed(
    monkeypatch, tmp_path, locked_source
):
    real_rmtree = shutil.rmtree

    def injected_cleanup_failure(path, *args, **kwargs):
        if ".paths-" in str(path):
            raise OSError("injected cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        wikidata_paths_module.shutil,
        "rmtree",
        injected_cleanup_failure,
    )
    destination = tmp_path / "robustness"

    with pytest.raises(OSError, match="injected cleanup failure"):
        _build(tmp_path, locked_source)

    assert not destination.exists()


def test_sqlite_connect_failure_cleans_all_private_staging(
    monkeypatch, tmp_path, locked_source
):
    real_connect = wikidata_paths_module.sqlite3.connect
    calls = 0

    def fail_builder_database(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected sqlite connect failure")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(
        wikidata_paths_module.sqlite3,
        "connect",
        fail_builder_database,
    )

    with pytest.raises(OSError, match="injected sqlite connect failure"):
        _build(tmp_path, locked_source)

    assert not (tmp_path / "robustness").exists()
    assert not list(tmp_path.glob(".robustness.*"))
