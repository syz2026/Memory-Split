from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from corpusgen.current_sources import (
    _audit_wikidata_splits,
    canonical_task_sha256,
    iter_aliases,
    iter_puzzle_tasks,
    iter_training_triples,
    load_dataset_lock,
    stage_current_sources,
    verify_current_sources,
)
from corpusgen.wikidata5m import (
    SourceDriftError,
    UnsafeArchiveError,
    WikidataLock,
    safe_extract_archives,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_LOCK = ROOT / "configs" / "current-dataset-lock.json"
WIKIDATA_NOTICE = ROOT / "sources" / "Wikidata-CC0-1.0.txt"

_ARC_TRAIN = {
    "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
    "test": [{"input": [[1, 1]], "output": [[1, 1]]}],
}
_ARC_TRAIN_2 = {
    "train": [{"input": [[2]], "output": [[3]]}],
    "test": [{"input": [[2, 2]], "output": [[3, 3]]}],
}
_ARC_EVAL = {
    "train": [{"input": [[4]], "output": [[5]]}],
    "test": [{"input": [[4, 4]], "output": [[5, 5]]}],
}
_CONCEPT_TRAIN = {
    "train": [{"input": [[6, 0]], "output": [[0, 6]]}],
    "test": [{"input": [[6, 6]], "output": [[6, 6]]}],
}

_IMPLEMENTATION = {
    "context_tokens": 1024,
    "raw_target_lane_shares": {
        "fineweb_edu": 0.4,
        "wikidata_graph": 0.2,
        "synthetic_graph": 0.1,
        "synthetic_reasoning": 0.15,
        "wikidata_reasoning": 0.075,
        "relational_refinement": 0.025,
        "puzzle_auxiliary": 0.05,
    },
    "token_floors": {
        "29m": {
            "parameters": 28969216,
            "updates": 1106,
            "raw_target_tokens": 579862528,
        },
        "160m": {
            "parameters": 162220800,
            "updates": 6189,
            "raw_target_tokens": 3244818432,
        },
        "360m": {
            "parameters": 356033536,
            "updates": 13582,
            "raw_target_tokens": 7120879616,
        },
    },
    "synthetic_fact_load": {
        "applies_to": "synthetic_entity_count_only",
        "wikidata_training_graph": "constant_complete_for_160m_and_360m",
        "wikidata_varies_by_fact_load": False,
        "wikidata_coverage_by_scale": {
            "29m": "deterministic_hash_sample_balanced_by_split_and_relation",
            "160m": "complete_training_graph",
            "360m": "complete_training_graph",
        },
        "labels": {"n50k": 50000, "n800k": 800000, "n1p8m": 1800000},
    },
    "reasoning_protocol": {
        "action_slots": 12,
        "max_reads": 10,
        "training_reads": {"min": 1, "max": 6},
        "held_out_length_reads": {"min": 7, "max": 10},
        "post_halt_slots": "deterministic_noop",
    },
    "wikidata_pages": {
        "address_fields": ["entity_qid", "property_pid", "direction", "page"],
        "object_order": "distinct_numeric_qid_ascending",
        "page_index_base": 0,
        "functional_row_page": 0,
        "page_rule": "largest_stable_prefix_with_formatted_record_at_most_context",
        "truncate": False,
    },
    "random_mask_matching": {
        "equal_mass_unit": "target_tokens",
        "strata": [
            "source",
            "record_type",
            "exact_payload_token_length",
            "packed_position_bin",
        ],
        "packed_position_bins": 10,
        "candidate_payload_class": "non_factual",
    },
    "composition_partition": {
        "salt": "relational-chinchilla/composition-partition",
        "sequence_lengths": [2, 3],
        "canonical_sequence": "comma_joined_pid_sequence",
        "hash": "sha256",
        "hash_input_parts": [
            "salt_utf8",
            "nul_byte",
            "canonical_sequence_utf8",
        ],
        "digest_integer_byte_order": "big",
        "evaluation_modulus": 5,
        "evaluation_remainders": [0],
    },
    "arc_tasks": {
        "canonical_hash": {
            "algorithm": "sha256",
            "encoding": "utf-8",
            "json": {
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": False,
            },
        },
        "transforms": {
            "include_original": True,
            "max_unique_excluding_original": 63,
            "spatial_symmetries": [
                "identity",
                "rotate_90",
                "rotate_180",
                "rotate_270",
                "reflect_horizontal",
                "reflect_vertical",
                "reflect_main_diagonal",
                "reflect_anti_diagonal",
            ],
            "color_permutations": "global_foreground_1_to_9_with_zero_fixed",
            "translations": "all_common_in_bounds_nonzero_shifts_with_zero_fill",
            "deduplicate_by": "canonical_task_sha256",
            "order_by": "sha256_of_canonical_parameter_json",
        },
    },
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))


def _write_unsafe_tar(path: Path, member: tarfile.TarInfo, data: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(data) if data is not None else None)


def _archive_lock(root: Path, revision: str = "a" * 40) -> dict:
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file():
            content = path.read_bytes()
            files[path.name] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    return {
        "repo_id": str(root),
        "repo_type": "dataset",
        "revision": revision,
        "files": files,
    }


def _init_git_repo(root: Path, files: dict[str, bytes]) -> str:
    _write_files(root, files)
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
    }
    subprocess.run(["git", "init", "-q", str(root)], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, env=env)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
        env=env,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_fixture_contract(
    tmp_path: Path,
    *,
    duplicate_eval_into_train: bool = False,
    duplicate_wikidata_eval_into_train: bool = False,
    malformed_training_triple: bool = False,
) -> tuple[object, Path]:
    inputs = tmp_path / "inputs"
    contract = tmp_path / "contract"
    contract.mkdir()

    fineweb = inputs / "fineweb"
    _write_files(
        fineweb,
        {
            "sample/10BT/000_00000.parquet": b"fixture parquet zero",
            "sample/10BT/001_00000.parquet": b"fixture parquet one",
            "sample/10BT/002_00000.parquet": b"fixture parquet two",
        },
    )

    archives = inputs / "wikidata-archives"
    _write_tar(
        archives / "wikidata5m_alias.tar.gz",
        {
            "wikidata5m_entity.txt": (
                b"Q1\t  Alpha  \talpha\n"
                b"Q2\t\tBeta\n"
                b"Q3\t   \n"
                b"Q4\tDelta\n"
                b"Q5\tEpsilon\n"
            ),
            "wikidata5m_relation.txt": (
                b"P1\t Parent Of \tparent   of\nP2\tlocated in\n"
            ),
        },
    )
    _write_tar(
        archives / "wikidata5m_inductive.tar.gz",
        {
            "wikidata5m_inductive_train.txt": (
                b"Q1\tP1\tQ2\nQ2\tP2\tQ3\n"
            ),
            "wikidata5m_inductive_valid.txt": b"Q3\tP1\tQ4\n",
            "wikidata5m_inductive_test.txt": b"Q4\tP2\tQ5\n",
        },
    )
    _write_tar(
        archives / "wikidata5m_transductive.tar.gz",
        {
            "wikidata5m_transductive_train.txt": (
                b"Q1\tR1\tQ4\n"
                if malformed_training_triple
                else (
                    b"Q3\tP1\tQ4\n"
                    if duplicate_wikidata_eval_into_train
                    else b"Q5\tP1\tQ4\n"
                )
            ),
            "wikidata5m_transductive_valid.txt": b"Q5\tP2\tQ1\n",
            "wikidata5m_transductive_test.txt": b"Q2\tP1\tQ5\n",
        },
    )
    wikidata_lock = _archive_lock(archives)
    (contract / "wikidata5m.lock.json").write_bytes(_json_bytes(wikidata_lock))

    duplicate_task = _ARC_EVAL if duplicate_eval_into_train else _ARC_TRAIN
    git_repos = inputs / "git"
    repositories = {
        "tiny_recursive_models": (
            {"LICENSE": b"fixture MIT\n", "README.md": b"reference\n"},
            "MIT",
        ),
        "arc_agi_1": (
            {
                "LICENSE": b"fixture Apache-2.0\n",
                "data/training/a.json": _json_bytes(duplicate_task),
                "data/evaluation/eval.json": _json_bytes(_ARC_EVAL),
            },
            "Apache-2.0",
        ),
        "arc_agi_2": (
            {
                "LICENSE": b"fixture Apache-2.0\n",
                "data/training/a-duplicate.json": _json_bytes(_ARC_TRAIN),
                "data/training/b.json": _json_bytes(_ARC_TRAIN_2),
                "data/evaluation/eval.json": _json_bytes(
                    {
                        "train": [{"input": [[7]], "output": [[8]]}],
                        "test": [{"input": [[7, 7]], "output": [[8, 8]]}],
                    }
                ),
            },
            "Apache-2.0",
        ),
        "conceptarc": (
            {
                "LICENSE": b"fixture MIT\n",
                "corpus/group/concept.json": _json_bytes(_CONCEPT_TRAIN),
            },
            "MIT",
        ),
    }
    commits = {
        name: _init_git_repo(git_repos / name, files)
        for name, (files, _) in repositories.items()
    }

    license_registry = {
        "format": "memorysplit-current-dataset-licenses",
        "dataset_id": "relational-chinchilla",
        "sources": {
            "fineweb_edu": {"spdx": "ODC-By-1.0"},
            "wikidata5m": {
                "spdx": "CC0-1.0",
                "notice_files": ["Wikidata-CC0-1.0.txt"],
            },
            **{
                name: {"spdx": license_name, "source_license_files": ["LICENSE"]}
                for name, (_, license_name) in repositories.items()
            },
        },
    }
    (contract / "current-dataset-licenses.json").write_bytes(
        _json_bytes(license_registry)
    )
    (contract / "Wikidata-CC0-1.0.txt").write_text("fixture CC0 notice\n")

    sources = {
        "fineweb_edu": {
            "transport": "huggingface",
            "repository": str(fineweb),
            "repo_type": "dataset",
            "revision": "b" * 40,
            "files": [
                "sample/10BT/000_00000.parquet",
                "sample/10BT/001_00000.parquet",
                "sample/10BT/002_00000.parquet",
            ],
            "materialized_jsonl": {
                "path": "fineweb-edu-10BT-2182000.jsonl",
                "rows": 2182000,
                "bytes": 10498726596,
                "sha256": (
                    "f89e844723887daa9714d906bf148bfe6931c2f063e94f791a1e861fefc668ca"
                ),
            },
            "holdout_records": 64,
            "license": "ODC-By-1.0",
        },
        "wikidata5m": {
            "transport": "huggingface",
            "repository": str(archives),
            "repo_type": "dataset",
            "revision": "a" * 40,
            "lock_path": "wikidata5m.lock.json",
            "scope": "wikidata5m-graph-3archive",
            "training_splits": ["inductive_train", "transductive_train"],
            "sealed_splits": [
                "inductive_test",
                "inductive_valid",
                "transductive_test",
                "transductive_valid",
            ],
            "split_files": {
                "inductive_test": "wikidata5m_inductive_test.txt",
                "inductive_train": "wikidata5m_inductive_train.txt",
                "inductive_valid": "wikidata5m_inductive_valid.txt",
                "transductive_test": "wikidata5m_transductive_test.txt",
                "transductive_train": "wikidata5m_transductive_train.txt",
                "transductive_valid": "wikidata5m_transductive_valid.txt",
            },
            "alias_files": {
                "entities": "wikidata5m_entity.txt",
                "relations": "wikidata5m_relation.txt",
            },
            "license": "CC0-1.0",
            "notice_path": "Wikidata-CC0-1.0.txt",
        },
    }
    for name, (_, license_name) in repositories.items():
        source = {
            "transport": "git",
            "repository": str(git_repos / name),
            "commit": commits[name],
            "license": license_name,
            "license_paths": ["LICENSE"],
            "role": (
                "augmentation_reference"
                if name == "tiny_recursive_models"
                else "puzzle"
            ),
        }
        if name == "conceptarc":
            source.update(
                {
                    "training_paths": ["corpus"],
                    "evaluation_paths": [],
                    "include_glob": "**/*.json",
                }
            )
        elif name != "tiny_recursive_models":
            source.update(
                {
                    "training_paths": ["data/training"],
                    "evaluation_paths": ["data/evaluation"],
                    "include_glob": "**/*.json",
                }
            )
        sources[name] = source

    current_lock = {
        "format": "memorysplit-current-dataset-lock",
        "dataset_id": "relational-chinchilla",
        "licenses_path": "current-dataset-licenses.json",
        "sources": sources,
        "implementation": _IMPLEMENTATION,
    }
    lock_path = contract / "current-dataset-lock.json"
    lock_path.write_bytes(_json_bytes(current_lock))
    return load_dataset_lock(lock_path), tmp_path / "data"


def stage_fixture_sources(
    tmp_path: Path,
    *,
    duplicate_eval_into_train: bool = False,
    duplicate_wikidata_eval_into_train: bool = False,
    malformed_training_triple: bool = False,
) -> dict:
    lock, data_root = _make_fixture_contract(
        tmp_path,
        duplicate_eval_into_train=duplicate_eval_into_train,
        duplicate_wikidata_eval_into_train=duplicate_wikidata_eval_into_train,
        malformed_training_triple=malformed_training_triple,
    )
    return stage_current_sources(lock, data_root, execute=True)


def test_committed_lock_freezes_the_approved_implementation_contract():
    lock = load_dataset_lock(CURRENT_LOCK)

    assert lock.dataset_id == "relational-chinchilla"
    assert lock.implementation == _IMPLEMENTATION
    assert lock.sources["fineweb_edu"]["revision"] == (
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    )
    assert lock.sources["fineweb_edu"]["materialized_jsonl"] == {
        "path": "fineweb-edu-10BT-2182000.jsonl",
        "rows": 2182000,
        "bytes": 10498726596,
        "sha256": "f89e844723887daa9714d906bf148bfe6931c2f063e94f791a1e861fefc668ca",
    }
    assert lock.sources["wikidata5m"]["revision"] == (
        "6b2b09672129e280c0c9da97ab58154e9d535e6b"
    )
    assert lock.sources["wikidata5m"]["sealed_splits"] == [
        "inductive_test",
        "inductive_valid",
        "transductive_test",
        "transductive_valid",
    ]
    assert lock.wikidata_lock.to_dict()["files"] == {
        "wikidata5m_alias.tar.gz": {
            "bytes": 197449751,
            "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8",
        },
        "wikidata5m_inductive.tar.gz": {
            "bytes": 167247416,
            "sha256": "955081232cc2de859710bfe3a147f7d8314524010fe5f8c420bb74fdfee4f42a",
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": 168258214,
            "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597",
        },
    }
    assert lock.sources["tiny_recursive_models"]["commit"] == (
        "c01103738605ba39d1430519b1ee0c62f4c707f8"
    )
    assert lock.sources["arc_agi_1"]["commit"] == (
        "399030444e0ab0cc8b4e199870fb20b863846f34"
    )
    assert lock.sources["arc_agi_2"]["commit"] == (
        "f3283f727488ad98fe575ea6a5ac981e4a188e49"
    )
    assert lock.sources["conceptarc"]["commit"] == (
        "0e67da6af879e4bad3d7cd3c196e8d551b445725"
    )


def test_committed_wikidata_notice_matches_the_proven_cc0_asset():
    assert hashlib.sha256(WIKIDATA_NOTICE.read_bytes()).hexdigest() == (
        "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
    )


def test_training_source_seals_evaluation_splits(tmp_path):
    manifest = stage_fixture_sources(tmp_path)
    assert manifest["wikidata"]["training_splits"] == [
        "inductive_train",
        "transductive_train",
    ]
    assert manifest["wikidata"]["sealed_splits"] == [
        "inductive_test",
        "inductive_valid",
        "transductive_test",
        "transductive_valid",
    ]
    assert manifest["wikidata"]["train_sealed_overlap"] == 0


def test_realistic_wikidata_archives_retain_every_official_member(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    manifest = stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"

    expected = {
        "wikidata5m_entity.txt",
        "wikidata5m_relation.txt",
        "wikidata5m_inductive_train.txt",
        "wikidata5m_inductive_valid.txt",
        "wikidata5m_inductive_test.txt",
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive_valid.txt",
        "wikidata5m_transductive_test.txt",
    }
    assert {item["path"] for item in manifest["wikidata"]["files"]} == expected
    assert {
        path.name for path in (source_root / "wikidata5m" / "files").iterdir()
    } == expected


def test_wikidata_evaluation_triple_is_rejected_from_training(tmp_path):
    with pytest.raises(ValueError, match="Wikidata evaluation triple"):
        stage_fixture_sources(
            tmp_path,
            duplicate_wikidata_eval_into_train=True,
        )

    assert not (
        tmp_path / "data" / "relational-chinchilla" / "sources"
    ).exists()


def test_malformed_wikidata_training_row_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid P id"):
        stage_fixture_sources(tmp_path, malformed_training_triple=True)

    assert not (
        tmp_path / "data" / "relational-chinchilla" / "sources"
    ).exists()


def test_wikidata_audit_uses_disk_sqlite_and_counts_duplicates(
    tmp_path,
    monkeypatch,
):
    files_root = tmp_path / "wikidata"
    _write_files(
        files_root,
        {
            "train-a.txt": b"Q1\tP1\tQ2\nQ1\tP1\tQ2\n",
            "train-b.txt": b"Q1\tP1\tQ2\nQ2\tP2\tQ3\n",
            "valid.txt": b"Q3\tP1\tQ5\n",
            "test.txt": b"Q4\tP2\tQ5\n",
        },
    )
    source = {
        "training_splits": ["train_a", "train_b"],
        "sealed_splits": ["test", "valid"],
        "split_files": {
            "train_a": "train-a.txt",
            "train_b": "train-b.txt",
            "test": "test.txt",
            "valid": "valid.txt",
        },
    }
    audit_root = tmp_path / "audit-work"
    audit_root.mkdir()
    opened_databases = []
    real_connect = sqlite3.connect

    def recording_connect(database, *args, **kwargs):
        opened_databases.append(Path(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)

    audit = _audit_wikidata_splits(
        source,
        files_root,
        work_root=audit_root,
    )

    assert len(opened_databases) == 1
    assert opened_databases[0].name == "triples.sqlite3"
    assert opened_databases[0].parent.parent == audit_root
    assert not opened_databases[0].exists()
    assert list(audit_root.iterdir()) == []
    assert audit["audit_backend"] == "sqlite"
    assert audit["split_rows"]["train_a"] == {
        "rows": 2,
        "distinct_triples": 1,
        "duplicate_rows": 1,
    }
    assert audit["training_distinct_triples"] == 2
    assert audit["training_duplicate_rows"] == 2
    assert audit["sealed_distinct_triples"] == 2
    assert audit["train_sealed_overlap"] == 0


def test_arc_evaluation_hash_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="evaluation task"):
        stage_fixture_sources(tmp_path, duplicate_eval_into_train=True)

    assert not (
        tmp_path / "data" / "relational-chinchilla" / "sources"
    ).exists()


def test_puzzle_manifest_locks_every_accepted_file_and_deduplicates(tmp_path):
    manifest = stage_fixture_sources(tmp_path)
    accepted = manifest["puzzles"]["accepted_files"]

    assert [(item["source"], item["path"]) for item in accepted] == [
        ("arc_agi_1", "data/training/a.json"),
        ("arc_agi_2", "data/training/b.json"),
        ("conceptarc", "corpus/group/concept.json"),
    ]
    for item in accepted:
        assert {
            "source",
            "repository",
            "commit",
            "path",
            "bytes",
            "sha256",
            "canonical_task_sha256",
            "license",
        } <= set(item)
        assert item["bytes"] > 0
        assert len(item["sha256"]) == 64
        assert len(item["canonical_task_sha256"]) == 64

    assert manifest["puzzles"]["duplicate_training_files"] == [
        {
            "canonical_task_sha256": canonical_task_sha256(_ARC_TRAIN),
            "duplicate_of": "arc_agi_1:data/training/a.json",
            "path": "data/training/a-duplicate.json",
            "source": "arc_agi_2",
        }
    ]


def test_canonical_task_hash_is_sha256_of_exact_canonical_json():
    expected = hashlib.sha256(
        json.dumps(
            _ARC_TRAIN,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert canonical_task_sha256(_ARC_TRAIN) == expected


def test_strict_training_triple_iterator_reads_only_training_splits(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"

    triples = list(iter_training_triples(source_root))

    assert [
        (row.split, row.row, row.subject, row.relation, row.object)
        for row in triples
    ] == [
        ("inductive_train", 1, 1, "P1", 2),
        ("inductive_train", 2, 2, "P2", 3),
        ("transductive_train", 1, 5, "P1", 4),
    ]


def test_alias_iterator_uses_first_normalized_alias_and_qid_fallback(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"

    aliases = list(iter_aliases(source_root))
    by_id = {row.canonical_id: row for row in aliases}

    assert by_id["Q1"].display == "Alpha"
    assert by_id["Q1"].aliases == ("Alpha",)
    assert by_id["Q2"].display == "Beta"
    assert by_id["Q3"].display == "Q3"
    assert by_id["Q3"].aliases == ()
    assert by_id["P1"].display == "Parent Of"
    assert by_id["P2"].display == "located in"


def test_puzzle_iterator_reads_only_manifest_accepted_tasks(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"

    tasks = list(iter_puzzle_tasks(source_root))

    assert [task.source for task in tasks] == [
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    ]
    assert [task.task for task in tasks] == [
        _ARC_TRAIN,
        _ARC_TRAIN_2,
        _CONCEPT_TRAIN,
    ]


def test_verification_rejects_source_drift_after_publication(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"
    task = source_root / "git" / "arc_agi_1" / "data" / "training" / "a.json"
    task.write_bytes(task.read_bytes() + b" ")

    with pytest.raises(SourceDriftError, match="source manifest drift"):
        verify_current_sources(lock, source_root)


def test_published_source_root_without_manifest_is_never_accepted(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"
    (source_root / "source-manifest.json").unlink()

    with pytest.raises(SourceDriftError, match="missing source manifest"):
        verify_current_sources(lock, source_root)
    with pytest.raises(SourceDriftError, match="missing source manifest"):
        stage_current_sources(lock, data_root, execute=True)

    assert not (source_root / "source-manifest.json").exists()


def test_published_source_root_rejects_structurally_invalid_manifest(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"
    (source_root / "source-manifest.json").write_bytes(
        _json_bytes(
            {
                "format": "memorysplit-current-source-manifest",
                "dataset_id": "relational-chinchilla",
            }
        )
    )

    with pytest.raises(SourceDriftError, match="invalid source manifest"):
        verify_current_sources(lock, source_root)


def test_published_source_namespace_rejects_unknown_file(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"
    (source_root / "wikidata5m" / "unexpected.txt").write_text("unlocked\n")

    with pytest.raises(SourceDriftError, match="unexpected source namespace"):
        verify_current_sources(lock, source_root)


def test_published_source_namespace_rejects_unknown_directory(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"
    (source_root / "git" / "unpinned-source").mkdir()

    with pytest.raises(SourceDriftError, match="unexpected source namespace"):
        verify_current_sources(lock, source_root)


def test_safe_archive_extraction_rejects_traversal_before_publication(tmp_path):
    archive_root = tmp_path / "archives"
    _write_tar(archive_root / "a-good.tar.gz", {"nested/good.txt": b"good"})
    member = tarfile.TarInfo("../escape.txt")
    member.size = 3
    _write_unsafe_tar(archive_root / "z-bad.tar.gz", member, b"bad")
    lock = WikidataLock.from_dict(_archive_lock(archive_root))
    out = tmp_path / "out"

    with pytest.raises(UnsafeArchiveError, match="unsafe archive member"):
        safe_extract_archives(archive_root, out, lock=lock)

    assert not out.exists()
    assert not (tmp_path / "escape.txt").exists()
    assert list(tmp_path.glob(".out.tmp-*")) == []


def test_safe_archive_extraction_rejects_symlinks_transactionally(tmp_path):
    archive_root = tmp_path / "archives"
    member = tarfile.TarInfo("link")
    member.type = tarfile.SYMTYPE
    member.linkname = "target"
    _write_unsafe_tar(archive_root / "unsafe.tar.gz", member, None)
    lock = WikidataLock.from_dict(_archive_lock(archive_root))
    out = tmp_path / "out"

    with pytest.raises(UnsafeArchiveError, match="member type"):
        safe_extract_archives(archive_root, out, lock=lock)

    assert not out.exists()


def test_dry_run_plans_pinned_operations_without_writing(tmp_path):
    lock = load_dataset_lock(CURRENT_LOCK)
    data_root = tmp_path / "data"

    plan = stage_current_sources(lock, data_root, execute=False)

    assert plan["dataset_id"] == "relational-chinchilla"
    assert plan["execute"] is False
    assert not data_root.exists()
    commands = [" ".join(operation["argv"]) for operation in plan["operations"]]
    assert sum(command.startswith("hf download ") for command in commands) == 2
    assert any(
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9" in command
        for command in commands
    )
    for commit in (
        "c01103738605ba39d1430519b1ee0c62f4c707f8",
        "399030444e0ab0cc8b4e199870fb20b863846f34",
        "f3283f727488ad98fe575ea6a5ac981e4a188e49",
        "0e67da6af879e4bad3d7cd3c196e8d551b445725",
    ):
        assert any(commit in command for command in commands)
    assert sum(" archive " in f" {command} " for command in commands) == 4


def test_cli_is_dry_run_by_default_and_honors_cache_dir(tmp_path):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/stage_current_sources.py",
            "--data-root",
            str(data_root),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    assert result.returncode == 0, result.stderr
    assert "hf download HuggingFaceFW/fineweb-edu" in result.stdout
    assert "git" in result.stdout and "archive" in result.stdout
    assert str(cache_dir) in result.stdout
    assert not data_root.exists()
    assert not cache_dir.exists()


def test_dataset_lock_rejects_deterministic_contract_drift(tmp_path):
    lock, _ = _make_fixture_contract(tmp_path)
    raw = json.loads(lock.path.read_text(encoding="utf-8"))
    raw["implementation"]["reasoning_protocol"]["action_slots"] = 11
    lock.path.write_bytes(_json_bytes(raw))

    with pytest.raises(ValueError, match="implementation contract"):
        load_dataset_lock(lock.path)


def test_source_manifest_is_canonical_and_existing_stage_is_idempotent(tmp_path):
    lock, data_root = _make_fixture_contract(tmp_path)
    first = stage_current_sources(lock, data_root, execute=True)
    source_root = data_root / "relational-chinchilla" / "sources"
    manifest_path = source_root / "source-manifest.json"

    assert manifest_path.read_bytes() == _json_bytes(first)
    assert stage_current_sources(lock, data_root, execute=True) == first
