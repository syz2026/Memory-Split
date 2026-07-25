from __future__ import annotations

import gc
import hashlib
import os
import shutil
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest

from corpusgen.current_sources import canonical_task_sha256
from corpusgen.reasoning_v2 import puzzle_source as puzzle_source_module
from corpusgen.reasoning_v2.puzzle_source import (
    PUZZLE_SOURCE_ORDER,
    PuzzleDuplicateRecord,
    PuzzleSourceScan,
    PuzzleTaskLocator,
    V2PuzzleTask,
    iter_v2_puzzle_tasks,
    read_v2_puzzle_task,
    scan_v2_puzzle_sources,
)
from corpusgen.reasoning_v2.source_lock import (
    resolve_source_lock,
    stage_source_lock,
)
from reasoning_v2_fixtures import (
    FakePublicResolver,
    FixtureSourceLock,
    _PUZZLE_EVAL_ONE,
    _PUZZLE_EVAL_TWO,
    _PUZZLE_TASK_ONE,
    _PUZZLE_TASK_ONE_BYTES,
    _PUZZLE_TASK_ONE_REFORMATTED_BYTES,
    _PUZZLE_TASK_THREE,
    _PUZZLE_TASK_TWO,
    _compact_task_bytes,
    _reformatted_task_bytes,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
)


EXPECTED_GENERATOR_COMMIT = "a" * 40

ONE = canonical_task_sha256(_PUZZLE_TASK_ONE)
TWO = canonical_task_sha256(_PUZZLE_TASK_TWO)
THREE = canonical_task_sha256(_PUZZLE_TASK_THREE)
EVAL_ONE = canonical_task_sha256(_PUZZLE_EVAL_ONE)
EVAL_TWO = canonical_task_sha256(_PUZZLE_EVAL_TWO)


def _stage(lock, download_root, canonical_root):
    return stage_source_lock(
        lock,
        download_root,
        canonical_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def _stage_fixture(tmp_path, fixture_source_lock: FixtureSourceLock):
    return _stage(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        tmp_path / "canonical",
    )


def _resolve_and_stage(tmp_path, full_recipe, *, puzzle_files=None):
    resolver = FakePublicResolver(puzzle_files=puzzle_files)
    download_root = tmp_path / "downloads"
    lock = resolve_source_lock(
        full_recipe,
        resolver,
        download_root,
        generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    root = _stage(lock, download_root, tmp_path / "canonical")
    return lock, root


def _scan(lock, root):
    return scan_v2_puzzle_sources(
        lock,
        root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def _entry(lock, source_id):
    return next(entry for entry in lock.sources if entry.source_id == source_id)


def _accepted_locator(scan, source_id, path):
    return next(
        locator
        for locator in scan.accepted
        if locator.source_id == source_id and locator.path == path
    )


# --------------------------------------------------------------------------
# Frozen policy and source/path ordering
# --------------------------------------------------------------------------


def test_frozen_source_order_is_arc1_arc2_conceptarc():
    assert PUZZLE_SOURCE_ORDER == ("arc_agi_1", "arc_agi_2", "conceptarc")


def test_scan_accepts_in_frozen_source_and_path_order(
    tmp_path, fixture_source_lock
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    assert [(l.source_id, l.path) for l in scan.accepted] == [
        ("arc_agi_1", "data/training/a.json"),
        ("arc_agi_2", "data/training/b.json"),
        ("conceptarc", "corpus/concept/c.json"),
    ]
    assert [l.canonical_task_sha256 for l in scan.accepted] == [ONE, TWO, THREE]
    assert [l.test_count for l in scan.accepted] == [1, 2, 1]


def test_scan_uses_direct_content_addressed_layout_without_legacy_manifest(
    tmp_path, fixture_source_lock
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    # Direct <root>/<source_id>/... layout, no legacy manifest or git/ namespace.
    assert root.name == fixture_source_lock.lock.sha256
    assert root.parent.name == "sources"
    assert not (root / "source-manifest.json").exists()
    assert not (root / "git").exists()
    assert (root / "arc_agi_1" / "data/training/a.json").is_file()
    scan = _scan(fixture_source_lock.lock, root)
    assert len(scan.accepted) == 3


def test_scan_rejects_non_content_addressed_root(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    misnamed = tmp_path / "sources" / "not-the-lock-hash"
    misnamed.parent.mkdir(parents=True)
    shutil.copytree(root, misnamed)
    with pytest.raises(ValueError, match="content-addressed"):
        _scan(fixture_source_lock.lock, misnamed)


# --------------------------------------------------------------------------
# Identity, schema, and canonical hashing
# --------------------------------------------------------------------------


def test_read_binds_repository_revision_license_and_identity(
    tmp_path, fixture_source_lock
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_1", "data/training/a.json")
    task = read_v2_puzzle_task(
        fixture_source_lock.lock,
        root,
        locator,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    entry = _entry(fixture_source_lock.lock, "arc_agi_1")
    source_file = next(row for row in entry.files if row.path == locator.path)
    assert isinstance(task, V2PuzzleTask)
    assert task.repository == entry.repository
    assert task.revision == entry.revision
    assert task.license_spdx == entry.license_spdx
    assert task.task == _PUZZLE_TASK_ONE
    assert locator.source_bytes == source_file.bytes
    assert locator.source_sha256 == source_file.sha256
    assert locator.canonical_task_sha256 == ONE


def test_canonical_task_hash_matches_current_sources(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_2", "data/training/b.json")
    assert locator.canonical_task_sha256 == canonical_task_sha256(_PUZZLE_TASK_TWO)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\xfe not json\n", "invalid"),
        (b'{"train":1,"train":2}\n', "duplicate"),
        (b'{"train":[],"test":[],"x":NaN}\n', "non-finite"),
        (b"[1,2,3]\n", "must be a JSON object"),
        (b'{"train":[{"input":1,"output":2}]}\n', "missing the test split"),
        (b'{"train":{},"test":[{"input":1,"output":2}]}\n', "train split must be"),
        (b'{"train":[{"input":1,"output":2}],"test":[]}\n', "test split must be"),
        (
            b'{"train":[{"input":1}],"test":[{"input":1,"output":2}]}\n',
            "must have input and output",
        ),
        (
            b'{"train":[1],"test":[{"input":1,"output":2}]}\n',
            "must be an object",
        ),
    ],
)
def test_scan_rejects_invalid_json_and_schema(
    tmp_path, full_recipe, fixed_contract_environment, payload, message
):
    puzzle_files = {
        "arc_agi_1": {"data/training/a.json": payload},
        "arc_agi_2": {"data/training/b.json": _compact_task_bytes(_PUZZLE_TASK_TWO)},
        "conceptarc": {
            "corpus/concept/c.json": _compact_task_bytes(_PUZZLE_TASK_THREE)
        },
    }
    lock, root = _resolve_and_stage(tmp_path, full_recipe, puzzle_files=puzzle_files)
    with pytest.raises(ValueError, match=message):
        _scan(lock, root)


# --------------------------------------------------------------------------
# Training/evaluation policy, contamination, and deduplication
# --------------------------------------------------------------------------


def test_scan_reads_evaluation_files_across_sources(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    assert scan.evaluation_task_sha256s == tuple(sorted({EVAL_ONE, EVAL_TWO}))
    # Evaluation files never appear in accepted training locators.
    assert all(
        locator.path.split("/")[1] != "evaluation" for locator in scan.accepted
    )


def test_scan_rejects_training_task_present_in_evaluation(
    tmp_path, full_recipe, fixed_contract_environment
):
    overlap = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    puzzle_files = {
        "arc_agi_1": {
            "data/training/a.json": _compact_task_bytes(overlap),
            "data/evaluation/e1.json": _reformatted_task_bytes(overlap),
        },
        "arc_agi_2": {"data/training/b.json": _compact_task_bytes(_PUZZLE_TASK_TWO)},
        "conceptarc": {
            "corpus/concept/c.json": _compact_task_bytes(_PUZZLE_TASK_THREE)
        },
    }
    lock, root = _resolve_and_stage(tmp_path, full_recipe, puzzle_files=puzzle_files)
    with pytest.raises(ValueError, match="overlaps an evaluation task"):
        _scan(lock, root)


def test_scan_records_exact_byte_duplicate(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    raw_dup = next(
        record for record in scan.duplicates if record.reason == "raw_duplicate"
    )
    assert (raw_dup.source_id, raw_dup.path) == (
        "arc_agi_2",
        "data/training/a_rawdup.json",
    )
    assert (raw_dup.duplicate_of_source_id, raw_dup.duplicate_of_path) == (
        "arc_agi_1",
        "data/training/a.json",
    )
    assert raw_dup.canonical_task_sha256 == ONE
    assert raw_dup.raw_sha256 == hashlib.sha256(_PUZZLE_TASK_ONE_BYTES).hexdigest()


def test_scan_records_canonical_duplicate_with_different_raw_bytes(
    tmp_path, fixture_source_lock
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    canonical_dup = next(
        record
        for record in scan.duplicates
        if record.reason == "canonical_duplicate"
    )
    assert (canonical_dup.source_id, canonical_dup.path) == (
        "conceptarc",
        "corpus/concept/a.json",
    )
    assert (
        canonical_dup.duplicate_of_source_id,
        canonical_dup.duplicate_of_path,
    ) == ("arc_agi_1", "data/training/a.json")
    assert canonical_dup.canonical_task_sha256 == ONE
    # Different raw bytes than the winner but the same canonical task.
    assert canonical_dup.raw_sha256 == hashlib.sha256(
        _PUZZLE_TASK_ONE_REFORMATTED_BYTES
    ).hexdigest()
    assert canonical_dup.raw_sha256 != hashlib.sha256(
        _PUZZLE_TASK_ONE_BYTES
    ).hexdigest()


def test_first_winner_is_source_then_path_order(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    winners = {
        locator.canonical_task_sha256: (locator.source_id, locator.path)
        for locator in scan.accepted
    }
    # Task ONE appears in arc_agi_1, arc_agi_2 (raw) and conceptarc (canonical);
    # arc_agi_1 wins by source order and every duplicate points back to it.
    assert winners[ONE] == ("arc_agi_1", "data/training/a.json")
    for record in scan.duplicates:
        assert (record.duplicate_of_source_id, record.duplicate_of_path) == (
            "arc_agi_1",
            "data/training/a.json",
        )


def test_conceptarc_is_training_only(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    concept_accepted = [l for l in scan.accepted if l.source_id == "conceptarc"]
    assert [l.path for l in concept_accepted] == ["corpus/concept/c.json"]
    # ConceptARC has no evaluation policy, so it contributes no evaluation hash.
    concept_entry = _entry(fixture_source_lock.lock, "conceptarc")
    assert all(
        not row.path.startswith("evaluation/") for row in concept_entry.files
    )


# --------------------------------------------------------------------------
# Deterministic, compact, bounded scan evidence
# --------------------------------------------------------------------------


def test_scan_is_deterministic(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    first = _scan(fixture_source_lock.lock, root)
    second = _scan(fixture_source_lock.lock, root)
    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_scan_policy_sha256_is_stable_and_bound_to_locators(
    tmp_path, fixture_source_lock
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    assert len(scan.policy_sha256) == 64
    assert all(l.policy_sha256 == scan.policy_sha256 for l in scan.accepted)


def test_scan_contains_no_raw_task_or_answer_bytes(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    for locator in scan.accepted:
        for value in vars(locator).values():
            assert isinstance(value, (str, int))
    for record in scan.duplicates:
        for value in vars(record).values():
            assert isinstance(value, str)
    for digest in scan.evaluation_task_sha256s:
        assert isinstance(digest, str) and len(digest) == 64


def test_scan_rejects_policy_drift_from_reviewed_lock(
    tmp_path, fixture_source_lock, monkeypatch
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    drifted = tuple(
        replace(policy, training_prefixes=("data/train",))
        if policy.source_id == "arc_agi_1"
        else policy
        for policy in puzzle_source_module._PUZZLE_POLICIES
    )
    monkeypatch.setattr(puzzle_source_module, "_PUZZLE_POLICIES", drifted)
    with pytest.raises(ValueError, match="policy"):
        _scan(fixture_source_lock.lock, root)


def test_scan_peak_memory_is_bounded_with_many_files(
    tmp_path, full_recipe, fixed_contract_environment
):
    def big_task(index):
        return {
            "train": [{"input": "a" * 20000 + str(index), "output": "o"}],
            "test": [{"input": "t", "output": "r"}],
        }

    training = {
        f"data/training/t{index:03d}.json": _compact_task_bytes(big_task(index))
        for index in range(16)
    }
    puzzle_files = {
        "arc_agi_1": training,
        "arc_agi_2": {"data/training/b.json": _compact_task_bytes(_PUZZLE_TASK_TWO)},
        "conceptarc": {
            "corpus/concept/c.json": _compact_task_bytes(_PUZZLE_TASK_THREE)
        },
    }
    total_task_bytes = sum(
        len(payload)
        for source in puzzle_files.values()
        for payload in source.values()
    )
    lock, root = _resolve_and_stage(tmp_path, full_recipe, puzzle_files=puzzle_files)
    gc.collect()
    tracemalloc.start()
    try:
        scan = _scan(lock, root)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(scan.accepted) == 18
    # Payloads are processed one at a time, so peak stays well below the sum of
    # every locked puzzle file's bytes.
    assert peak < total_task_bytes


# --------------------------------------------------------------------------
# Iteration
# --------------------------------------------------------------------------


def test_iter_yields_accepted_tasks_in_order(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    tasks = list(
        iter_v2_puzzle_tasks(
            fixture_source_lock.lock,
            root,
            scan,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    )
    assert [t.locator for t in tasks] == list(scan.accepted)
    assert [t.task for t in tasks] == [
        _PUZZLE_TASK_ONE,
        _PUZZLE_TASK_TWO,
        _PUZZLE_TASK_THREE,
    ]


def test_iter_rejects_scan_commitment_mutation(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    forged = replace(scan, sha256="f" * 64)
    with pytest.raises(ValueError, match="scan commitment"):
        list(
            iter_v2_puzzle_tasks(
                fixture_source_lock.lock,
                root,
                forged,
                expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
            )
        )


# --------------------------------------------------------------------------
# Locator and lookup rejection
# --------------------------------------------------------------------------


def test_read_rejects_unknown_source(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_1", "data/training/a.json")
    forged = replace(locator, source_id="fineweb_edu")
    with pytest.raises(ValueError, match="puzzle source"):
        read_v2_puzzle_task(
            fixture_source_lock.lock,
            root,
            forged,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_read_rejects_unknown_path(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_1", "data/training/a.json")
    forged = replace(locator, path="data/training/missing.json")
    with pytest.raises(ValueError, match="path"):
        read_v2_puzzle_task(
            fixture_source_lock.lock,
            root,
            forged,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_read_rejects_evaluation_locator(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_1", "data/training/a.json")
    entry = _entry(fixture_source_lock.lock, "arc_agi_1")
    evaluation = next(
        row for row in entry.files if row.path == "data/evaluation/e1.json"
    )
    forged = replace(
        locator,
        path="data/evaluation/e1.json",
        source_bytes=evaluation.bytes,
        source_sha256=evaluation.sha256,
    )
    with pytest.raises(ValueError, match="training puzzle path"):
        read_v2_puzzle_task(
            fixture_source_lock.lock,
            root,
            forged,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_sha256", "f" * 64, "source-byte commitment"),
        ("source_bytes", 999999, "source-byte commitment"),
        ("canonical_task_sha256", "f" * 64, "canonical commitment"),
        ("policy_sha256", "f" * 64, "policy commitment"),
        ("test_count", 99, "test-example count"),
    ],
)
def test_read_rejects_locator_commitment_mutation(
    tmp_path, fixture_source_lock, field, value, message
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_1", "data/training/a.json")
    forged = replace(locator, **{field: value})
    with pytest.raises(ValueError, match=message):
        read_v2_puzzle_task(
            fixture_source_lock.lock,
            root,
            forged,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


# --------------------------------------------------------------------------
# Descriptor safety of the selected read
# --------------------------------------------------------------------------


def _good_locator(tmp_path, fixture_source_lock):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    return root, _accepted_locator(scan, "arc_agi_1", "data/training/a.json")


def _read(lock, root, locator):
    return read_v2_puzzle_task(
        lock,
        root,
        locator,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def test_read_rejects_symlinked_file(tmp_path, fixture_source_lock):
    root, locator = _good_locator(tmp_path, fixture_source_lock)
    target = root / "arc_agi_1" / "data/training/a.json"
    payload = target.read_bytes()
    external = tmp_path / "external.json"
    external.write_bytes(payload)
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        _read(fixture_source_lock.lock, root, locator)


def test_read_rejects_hardlinked_file(tmp_path, fixture_source_lock):
    root, locator = _good_locator(tmp_path, fixture_source_lock)
    target = root / "arc_agi_1" / "data/training/a.json"
    external = tmp_path / "external.json"
    external.write_bytes(target.read_bytes())
    target.unlink()
    os.link(external, target)
    with pytest.raises(ValueError, match="hardlink"):
        _read(fixture_source_lock.lock, root, locator)


def test_read_rejects_file_byte_replacement(tmp_path, fixture_source_lock):
    root, locator = _good_locator(tmp_path, fixture_source_lock)
    target = root / "arc_agi_1" / "data/training/a.json"
    target.write_bytes(target.read_bytes() + b"   ")
    with pytest.raises(ValueError, match="raw-byte drift"):
        _read(fixture_source_lock.lock, root, locator)


def test_read_rejects_symlinked_directory(tmp_path, fixture_source_lock):
    root, locator = _good_locator(tmp_path, fixture_source_lock)
    training = root / "arc_agi_1" / "data" / "training"
    relocated = tmp_path / "training-relocated"
    shutil.move(str(training), str(relocated))
    training.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _read(fixture_source_lock.lock, root, locator)


def test_read_detects_post_open_identity_race(
    tmp_path, fixture_source_lock, monkeypatch
):
    root, locator = _good_locator(tmp_path, fixture_source_lock)
    target = root / "arc_agi_1" / "data/training/a.json"
    payload = target.read_bytes()
    state = {"done": False}

    def race(phase, directory_fd, name):
        if phase == "after_open" and name == "a.json" and not state["done"]:
            state["done"] = True
            os.unlink(name, dir_fd=directory_fd)
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            os.write(descriptor, payload)
            os.close(descriptor)

    monkeypatch.setattr(puzzle_source_module, "_selected_read_hook", race)
    with pytest.raises(ValueError, match="identity drift"):
        _read(fixture_source_lock.lock, root, locator)


def test_read_opens_only_selected_file_and_never_scans_siblings(
    tmp_path, fixture_source_lock
):
    root = _stage_fixture(tmp_path, fixture_source_lock)
    scan = _scan(fixture_source_lock.lock, root)
    locator = _accepted_locator(scan, "arc_agi_1", "data/training/a.json")
    # Corrupt a different puzzle source file; a selected read must not touch it.
    sibling = root / "arc_agi_2" / "data/training/b.json"
    sibling.write_bytes(sibling.read_bytes() + b"corruption")
    task = _read(fixture_source_lock.lock, root, locator)
    assert task.task == _PUZZLE_TASK_ONE
    # A full scan re-authenticates the complete root and therefore rejects it.
    with pytest.raises(ValueError):
        _scan(fixture_source_lock.lock, root)


def test_read_does_not_enumerate_directories(
    tmp_path, fixture_source_lock, monkeypatch
):
    root, locator = _good_locator(tmp_path, fixture_source_lock)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("selected read must not enumerate directories")

    monkeypatch.setattr(os, "listdir", forbidden)
    task = _read(fixture_source_lock.lock, root, locator)
    assert task.locator == locator
