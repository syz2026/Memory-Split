from __future__ import annotations

import fcntl
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import corpusgen.v2_sources as v2_sources
from corpusgen.v2_sources import (
    DEFAULT_V2_SOURCE_LOCK,
    InsufficientDiskError,
    V2SourceStageError,
    load_v2_source_lock,
    plan_v2_source_stage,
    stage_v2_sources,
    verify_v2_source_stage,
)
from corpusgen.wikidata5m import SourceDriftError


ROOT = Path(__file__).resolve().parents[1]

_AUXILIARY_IDS = [
    "deepmind_mathematics_generator",
    "clrs_text",
    "ruletaker",
    "prontoqa",
    "reasoning_gym_exact_answer",
    "arc_agi_training",
    "conceptarc_training",
]
_MISSING_SOURCE_LOCKS = [
    "synthetic_graph_generator",
    "verified_synthetic_multihop_generator",
    "wikidata_path_reasoning_generator",
    "relational_refinement_generator",
    "reasoning_solver",
]
_MISSING_MATERIALIZED_LANES = [
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode()
        + b"\n"
    )


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_blob(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _write_parquet(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts}), path)


def _file_record(root: Path, relative: str, *, blob: bool = True) -> dict:
    content = (root / relative).read_bytes()
    record = {
        "bytes": len(content),
        "path": relative,
        "sha256": _sha(content),
    }
    if blob:
        record["git_blob_sha1"] = _git_blob(content)
    return record


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))


def _write_lock(path: Path, value: object) -> str:
    content = _json_bytes(value)
    path.write_bytes(content)
    return _sha(content)


def _fixture_lock(tmp_path: Path):
    inputs = tmp_path / "inputs"
    lock_root = tmp_path / "contract" / "locks"
    lock_root.mkdir(parents=True)

    fineweb_root = inputs / "fineweb"
    fineweb_texts = [["common", "fineweb-zero"]] + [
        [f"fineweb-{index}"] for index in range(1, 14)
    ]
    fineweb_paths = []
    for index, texts in enumerate(fineweb_texts):
        relative = f"sample/10BT/{index:03d}_00000.parquet"
        _write_parquet(fineweb_root / relative, texts)
        fineweb_paths.append(relative)
    fineweb_files = [_file_record(fineweb_root, path) for path in fineweb_paths]
    fineweb = {
        "files": fineweb_files,
        "format": "memorysplit-v2-huggingface-source-lock",
        "license": "ODC-By-1.0",
        "license_evidence": {
            "dataset_card_value": "odc-by",
            "terms_url": "https://commoncrawl.org/terms-of-use",
        },
        "repo_type": "dataset",
        "repository": str(fineweb_root),
        "revision": hashlib.sha1(b"fineweb fixture").hexdigest(),
        "schema_version": 1,
        "selection": {
            "config": "sample-10BT",
            "file_order": "lexicographic_path",
            "scope": "complete_config",
        },
        "source_id": "fineweb_edu",
        "total_bytes": sum(item["bytes"] for item in fineweb_files),
    }

    finemath_root = inputs / "finemath"
    finemath_rows = {
        "finemath-4plus/train-00000-of-00001.parquet": [
            "common",
            "four",
            "duplicate-four",
        ],
        "finemath-3plus/train-00000-of-00001.parquet": [
            "four",
            "three",
            "three",
        ],
    }
    for relative, texts in finemath_rows.items():
        _write_parquet(finemath_root / relative, texts)
    finemath_files = [
        _file_record(finemath_root, relative)
        for relative in sorted(finemath_rows)
    ]
    finemath = {
        "deduplication": {
            "digest": "sha256",
            "encoding": "utf-8",
            "field": "text",
            "identity": "exact_text_bytes",
            "priority": [
                "fineweb_edu",
                "finemath-4plus",
                "finemath-3plus",
            ],
            "sidecar_encoding": (
                "one_unsigned_byte_per_source_row_1_keep_0_drop"
            ),
        },
        "files": finemath_files,
        "format": "memorysplit-v2-huggingface-source-lock",
        "license": "ODC-By-1.0",
        "license_evidence": {
            "dataset_card_value": "odc-by",
            "terms_url": "https://commoncrawl.org/terms-of-use",
        },
        "repo_type": "dataset",
        "repository": str(finemath_root),
        "revision": hashlib.sha1(b"finemath fixture").hexdigest(),
        "schema_version": 1,
        "selection": {
            "file_order": "lexicographic_path_within_subset",
            "subsets_in_order": [
                "finemath-4plus",
                "finemath-3plus-cross-deduplicated-remainder",
            ],
        },
        "source_id": "finemath",
        "subset_metadata": {
            "finemath-3plus": {
                "download_bytes": finemath_files[0]["bytes"],
                "rows": 3,
            },
            "finemath-4plus": {
                "download_bytes": finemath_files[1]["bytes"],
                "rows": 3,
            },
        },
        "total_bytes": sum(item["bytes"] for item in finemath_files),
    }

    wikidata_root = inputs / "wikidata"
    wikidata_archives = {
        "wikidata5m_alias.tar.gz": {
            "wikidata5m_entity.txt": b"Q1\tOne\nQ2\tTwo\nQ3\tThree\nQ4\tFour\nQ5\tFive\n",
            "wikidata5m_relation.txt": b"P1\tfirst\nP2\tsecond\n",
        },
        "wikidata5m_inductive.tar.gz": {
            "wikidata5m_inductive_train.txt": (
                b"Q1\tP1\tQ2\nQ1\tP1\tQ2\n"
            ),
            "wikidata5m_inductive_valid.txt": b"Q3\tP1\tQ4\n",
            "wikidata5m_inductive_test.txt": b"Q4\tP2\tQ5\n",
        },
        "wikidata5m_transductive.tar.gz": {
            "wikidata5m_transductive_train.txt": (
                b"Q1\tP1\tQ2\nQ2\tP2\tQ3\n"
            ),
            "wikidata5m_transductive_valid.txt": b"Q5\tP2\tQ1\n",
            "wikidata5m_transductive_test.txt": b"Q2\tP1\tQ5\n",
        },
    }
    for archive_name, files in wikidata_archives.items():
        _write_tar(
            wikidata_root / archive_name,
            {name: content for name, content in files.items()},
        )
    wikidata_files = []
    for archive_name, members in sorted(wikidata_archives.items()):
        record = _file_record(wikidata_root, archive_name)
        record["members"] = [
            {
                "bytes": len(content),
                "path": name,
                "sha256": _sha(content),
            }
            for name, content in sorted(members.items())
        ]
        wikidata_files.append(record)
    notice = b"fixture Wikidata CC0 notice\n"
    (lock_root / "Wikidata-CC0-1.0.txt").write_bytes(notice)
    wikidata = {
        "complete_once": {
            "sidecar_encoding": (
                "one_unsigned_byte_per_source_row_1_keep_0_drop"
            ),
            "split_order": [
                "wikidata5m_inductive_train.txt",
                "wikidata5m_transductive_train.txt",
            ],
            "triple_identity": "exact_subject_relation_object",
        },
        "files": wikidata_files,
        "format": "memorysplit-v2-huggingface-source-lock",
        "license": "CC0-1.0",
        "license_evidence": {
            "notice_path": "Wikidata-CC0-1.0.txt",
            "notice_sha256": _sha(notice),
        },
        "repo_type": "dataset",
        "repository": str(wikidata_root),
        "revision": hashlib.sha1(b"wikidata fixture").hexdigest(),
        "schema_version": 1,
        "source_id": "wikidata5m",
        "total_bytes": sum(item["bytes"] for item in wikidata_files),
    }

    auxiliary_sources = []
    for index, source_id in enumerate(_AUXILIARY_IDS):
        component_id = f"component-{index}"
        commit = hashlib.sha1(source_id.encode()).hexdigest()
        root_name = f"{component_id}-{commit}"
        archive_path = inputs / "auxiliary" / f"{component_id}.tar.gz"
        archive_files = {
            f"{root_name}/LICENSE": b"fixture Apache-2.0 license\n",
            f"{root_name}/generator.py": f"SOURCE = {source_id!r}\n".encode(),
        }
        _write_tar(archive_path, archive_files)
        selected = []
        for full_path, content in sorted(archive_files.items()):
            relative = full_path.split("/", 1)[1]
            selected.append(
                {
                    "bytes": len(content),
                    "path": relative,
                    "sha256": _sha(content),
                }
            )
        auxiliary_sources.append(
            {
                "components": [
                    {
                        "archive": {
                            "bytes": archive_path.stat().st_size,
                            "root": root_name,
                            "sha256": _sha(archive_path.read_bytes()),
                            "url": archive_path.resolve().as_uri(),
                        },
                        "commit": commit,
                        "files": selected,
                        "id": component_id,
                        "license": "Apache-2.0",
                        "license_paths": ["LICENSE"],
                        "repository": f"https://github.com/fixture/{component_id}",
                    }
                ],
                "id": source_id,
                "training_use": "fixture_objective_answers",
            }
        )
    auxiliary = {
        "format": "memorysplit-v2-objective-auxiliary-lock",
        "schema_version": 1,
        "sources": auxiliary_sources,
    }

    children = {
        "fineweb_edu": ("fineweb-edu.lock.json", fineweb),
        "finemath": ("finemath.lock.json", finemath),
        "wikidata5m": ("wikidata5m-complete-once.lock.json", wikidata),
        "objective_auxiliary": ("objective-auxiliaries.lock.json", auxiliary),
    }
    references = []
    for source_id in (
        "fineweb_edu",
        "finemath",
        "wikidata5m",
        "objective_auxiliary",
    ):
        filename, value = children[source_id]
        digest = _write_lock(lock_root / filename, value)
        references.append(
            {"id": source_id, "path": filename, "sha256": digest}
        )

    scientific_path = lock_root.parent / "reasoning-dataset-v2.json"
    scientific_path.write_bytes(
        (ROOT / "configs" / "reasoning-dataset-v2.json").read_bytes()
    )
    master = {
        "contract_id": "memorysplit-reasoning-dataset-v2",
        "dataset_id": "memorysplit-v2-frozen-upstream-sources",
        "format": "memorysplit-v2-source-set-lock",
        "production_readiness": {
            "missing_materialized_lanes": _MISSING_MATERIALIZED_LANES,
            "missing_source_locks": _MISSING_SOURCE_LOCKS,
            "ready": False,
            "reason": "fixture source stage is not a production corpus",
        },
        "schema_version": 1,
        "scientific_requirements": {
            "path": "../reasoning-dataset-v2.json",
            "sha256": _sha(scientific_path.read_bytes()),
        },
        "source_locks": references,
    }
    master_path = lock_root / "source-set.lock.json"
    _write_lock(master_path, master)
    return load_v2_source_lock(master_path)


def test_committed_source_locks_bind_real_upstream_bytes_and_licenses():
    lock = load_v2_source_lock(DEFAULT_V2_SOURCE_LOCK)

    assert lock.sha256 == (
        "caba0f502afd04d735bbaa16ede6bea308da12f1563cba3168c964717d2f3f3f"
    )
    assert lock.download_bytes == 112_510_223_370
    fineweb = lock.sources["fineweb_edu"]
    assert fineweb["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert len(fineweb["files"]) == 14
    assert fineweb["total_bytes"] == 28_518_193_415
    finemath = lock.sources["finemath"]
    assert finemath["revision"] == "e92b25a616738fe95dc186b64dfb19f9c8525594"
    assert len(finemath["files"]) == 192
    assert finemath["total_bytes"] == 83_404_381_578
    assert sum(
        item["path"].startswith("finemath-4plus/")
        for item in finemath["files"]
    ) == 64
    wikidata = lock.sources["wikidata5m"]
    assert wikidata["revision"] == "6b2b09672129e280c0c9da97ab58154e9d535e6b"
    assert wikidata["total_bytes"] == 532_955_381
    assert all(
        len(member["sha256"]) == 64
        for archive in wikidata["files"]
        for member in archive["members"]
    )

    auxiliary = lock.sources["objective_auxiliary"]["sources"]
    assert [item["id"] for item in auxiliary] == _AUXILIARY_IDS
    assert all(
        component["license_paths"]
        and all(
            any(record["path"] == path for record in component["files"])
            for path in component["license_paths"]
        )
        for source in auxiliary
        for component in source["components"]
    )
    assert all(
        len(set(record["sha256"])) > 1
        for source in auxiliary
        for component in source["components"]
        for record in component["files"]
        if record["bytes"]
    )


def test_objective_lock_excludes_benchmark_evaluation_and_model_outputs():
    lock = load_v2_source_lock(DEFAULT_V2_SOURCE_LOCK)
    paths = {
        record["path"]
        for source in lock.sources["objective_auxiliary"]["sources"]
        for component in source["components"]
        for record in component["files"]
    }

    assert not any("/evaluation/" in f"/{path}" for path in paths)
    assert not any(
        path.startswith("clrs/_src/clrs_text/colabs/") for path in paths
    )
    assert "generated_ood_data.zip" not in paths
    assert "model_outputs_ood.zip" not in paths
    assert "model_outputs_v1.zip" not in paths


def test_production_dry_run_is_read_only_and_current_hf_cli_compatible(tmp_path):
    lock = load_v2_source_lock(DEFAULT_V2_SOURCE_LOCK)
    data_root = tmp_path / "not-created"
    cache = tmp_path / "cache"

    plan = plan_v2_source_stage(
        lock,
        data_root,
        cache_dir=cache,
        hf_command="/opt/bin/hf",
        disk_free_bytes=200 * 1024**3,
    )

    assert plan["execute"] is False
    assert plan["download_bytes"] == 112_510_223_370
    assert plan["disk_preflight"] == "passed"
    assert plan["production_readiness"]["ready"] is False
    hf_operations = [
        operation
        for operation in plan["operations"]
        if operation["kind"] == "huggingface_download"
    ]
    assert [operation["source"] for operation in hf_operations] == [
        "fineweb_edu",
        "finemath",
        "wikidata5m",
    ]
    for operation in hf_operations:
        assert operation["argv"][:2] == ["/opt/bin/hf", "download"]
        assert "--local-dir" in operation["argv"]
        assert "--cache-dir" not in operation["argv"]
        assert operation["env"]["HF_HOME"] == str(
            cache.resolve() / "huggingface"
        )
    assert not data_root.exists()
    assert not cache.exists()


def test_disk_preflight_fails_before_writing(tmp_path):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "no-space"

    with pytest.raises(InsufficientDiskError, match="disk preflight"):
        stage_v2_sources(
            lock,
            data_root,
            execute=True,
            reserve_bytes=0,
            disk_free_bytes=0,
        )

    assert not data_root.exists()


def test_fixture_stage_cross_deduplicates_and_completes_wikidata_once(tmp_path):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "data"

    receipt = stage_v2_sources(
        lock,
        data_root,
        execute=True,
        reserve_bytes=0,
        disk_free_bytes=10 * 1024**3,
    )

    source_root = data_root / lock.dataset_id
    four = (
        source_root
        / "finemath"
        / "selection"
        / "finemath-4plus"
        / "train-00000-of-00001.parquet.keep.u8"
    )
    three = (
        source_root
        / "finemath"
        / "selection"
        / "finemath-3plus"
        / "train-00000-of-00001.parquet.keep.u8"
    )
    assert four.read_bytes() == b"\x00\x01\x01"
    assert three.read_bytes() == b"\x00\x01\x00"
    assert receipt["sources"]["finemath"]["selection_totals"] == {
        "dropped_rows": 3,
        "kept_rows": 3,
        "rows": 6,
    }

    inductive = (
        source_root
        / "wikidata5m"
        / "selection"
        / "wikidata5m_inductive_train.txt.keep.u8"
    )
    transductive = (
        source_root
        / "wikidata5m"
        / "selection"
        / "wikidata5m_transductive_train.txt.keep.u8"
    )
    assert inductive.read_bytes() == b"\x01\x00"
    assert transductive.read_bytes() == b"\x00\x01"
    assert receipt["sources"]["wikidata5m"]["selection_totals"] == {
        "distinct_training_triples": 2,
        "dropped_duplicate_rows": 2,
        "training_rows": 4,
    }
    assert receipt["stage_complete"] is True
    assert receipt["production_corpus_ready"] is False
    assert receipt["missing_source_locks"] == _MISSING_SOURCE_LOCKS
    assert receipt["missing_materialized_lanes"] == _MISSING_MATERIALIZED_LANES
    staged_lock = load_v2_source_lock(
        source_root
        / "source-contract"
        / "sources"
        / "memorysplit-v2"
        / "source-set.lock.json"
    )
    assert staged_lock.sha256 == lock.sha256
    assert (source_root / "source-stage-receipt.json").read_bytes() == _json_bytes(
        receipt
    )
    assert verify_v2_source_stage(lock, source_root) == receipt
    assert (
        stage_v2_sources(
            lock,
            data_root,
            execute=True,
            reserve_bytes=0,
            disk_free_bytes=0,
        )
        == receipt
    )


def test_completed_http_partial_resumes_without_redownload(tmp_path):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "data"
    partial = (
        data_root
        / ".memorysplit-v2-source-stage"
        / lock.sha256
        / "downloads"
        / "objective"
        / ".component-0.tar.gz.partial"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(
        (tmp_path / "inputs" / "auxiliary" / "component-0.tar.gz").read_bytes()
    )

    receipt = stage_v2_sources(
        lock,
        data_root,
        execute=True,
        reserve_bytes=0,
        disk_free_bytes=10 * 1024**3,
    )

    assert receipt["stage_complete"] is True


def test_restart_cannot_remove_a_live_private_stage(tmp_path):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "data"
    staging_parent = data_root / ".memorysplit-v2-source-stage"
    private = staging_parent / lock.sha256
    private.mkdir(parents=True)
    marker = private / "in-progress"
    marker.write_text("owned by first process", encoding="utf-8")
    lock_path = staging_parent / f".{lock.sha256}.lock"

    with lock_path.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(V2SourceStageError, match="another source staging process"):
            stage_v2_sources(
                lock,
                data_root,
                execute=True,
                restart=True,
                reserve_bytes=0,
                disk_free_bytes=10 * 1024**3,
            )

    assert marker.read_text(encoding="utf-8") == "owned by first process"


def test_interrupted_private_stage_resumes_without_publishing(
    tmp_path,
    monkeypatch,
):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "data"
    original = v2_sources._extract_auxiliary_component
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        v2_sources,
        "_extract_auxiliary_component",
        interrupt_once,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        stage_v2_sources(
            lock,
            data_root,
            execute=True,
            reserve_bytes=0,
            disk_free_bytes=10 * 1024**3,
        )
    assert not (data_root / lock.dataset_id).exists()
    private = data_root / ".memorysplit-v2-source-stage" / lock.sha256
    assert private.is_dir()

    monkeypatch.setattr(
        v2_sources,
        "_extract_auxiliary_component",
        original,
    )
    receipt = stage_v2_sources(
        lock,
        data_root,
        execute=True,
        reserve_bytes=0,
        disk_free_bytes=10 * 1024**3,
    )
    assert receipt["stage_complete"] is True
    assert receipt["production_corpus_ready"] is False
    assert (data_root / lock.dataset_id).is_dir()
    assert not private.exists()


def test_published_receipt_detects_source_drift(tmp_path):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "data"
    stage_v2_sources(
        lock,
        data_root,
        execute=True,
        reserve_bytes=0,
        disk_free_bytes=10 * 1024**3,
    )
    source_root = data_root / lock.dataset_id
    task = next(
        (source_root / "objective_auxiliary").glob("*/*/generator.py")
    )
    task.write_bytes(task.read_bytes() + b"# drift\n")

    with pytest.raises(SourceDriftError, match="inventory drift"):
        verify_v2_source_stage(lock, source_root)


def test_forged_receipt_cannot_hide_locked_source_drift(tmp_path):
    lock = _fixture_lock(tmp_path)
    data_root = tmp_path / "data"
    stage_v2_sources(
        lock,
        data_root,
        execute=True,
        reserve_bytes=0,
        disk_free_bytes=10 * 1024**3,
    )
    source_root = data_root / lock.dataset_id
    source_file = next((source_root / "fineweb_edu").glob("**/*.parquet"))
    content = bytearray(source_file.read_bytes())
    content[0] ^= 1
    source_file.write_bytes(content)

    receipt_path = source_root / "source-stage-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    files = v2_sources._inventory(
        source_root,
        exclude={v2_sources.RECEIPT_NAME},
    )
    receipt["files"] = files
    receipt["inventory_sha256"] = v2_sources._tree_digest(files)
    receipt_path.write_bytes(_json_bytes(receipt))

    with pytest.raises(SourceDriftError, match="Hugging Face source file drift"):
        verify_v2_source_stage(lock, source_root)
