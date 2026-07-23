import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import corpusgen.parallel as parallel
from corpusgen.parallel import publication as publication_module
from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel import (
    FixtureRenderer,
    InputCatalog,
    ParallelBuildConfig,
    UnsupportedProductionRenderer,
    UnsupportedSourceError,
    assignments_from_bytes,
    assignments_to_bytes,
    assign_update_aligned_shards,
    build_parallel_corpus,
    largest_deficit_schedule,
    metadata_from_bytes,
    metadata_to_bytes,
    ordered_stream_commitments,
    parallel_build_id,
    publication_staging_path,
    reduce_metadata,
    render_metadata,
    schedule_from_bytes,
    schedule_to_bytes,
    fixture_catalog,
    verify_stream_commitments,
    verify_parallel_corpus,
)


def test_fixture_catalog_is_immutable_canonical_and_round_trips():
    catalog = fixture_catalog(record_count=6)

    assert len(catalog.records) == 6
    assert [record.ordinal for record in catalog.records] == list(range(6))
    assert len({record.record_id for record in catalog.records}) == 6
    with pytest.raises(FrozenInstanceError):
        catalog.records[0].record_id = "changed"

    encoded = catalog.to_bytes()
    assert encoded.endswith(b"\n")
    assert InputCatalog.from_bytes(encoded) == catalog
    assert InputCatalog.from_bytes(encoded).sha256 == catalog.sha256


def test_catalog_rejects_duplicate_source_records_even_with_new_ids():
    first = fixture_catalog(record_count=1).records[0]
    duplicate = replace(first, ordinal=1, record_id="renamed-duplicate")

    with pytest.raises(ValueError, match="duplicate source"):
        InputCatalog((first, duplicate))


def test_metadata_is_canonical_and_independent_of_worker_completion_order():
    catalog = fixture_catalog(record_count=15)
    renderer = FixtureRenderer()

    serial = render_metadata(catalog, renderer, workers=1)
    parallel = render_metadata(catalog, renderer, workers=4)

    assert serial == parallel
    assert [record.ordinal for record in serial] == list(range(15))
    assert all(record.token_length > 0 for record in serial)
    assert all(record.flags == tuple(sorted(record.flags)) for record in serial)
    assert all(len(record.source_sha256) == 64 for record in serial)
    assert all(len(record.render_sha256) == 64 for record in serial)
    assert all(len(record.metadata_sha256) == 64 for record in serial)
    encoded = metadata_to_bytes(serial)
    assert metadata_from_bytes(encoded) == serial


def test_metadata_reducer_fails_closed_on_missing_duplicate_or_tampered_records():
    catalog = fixture_catalog(record_count=6)
    renderer = FixtureRenderer()
    metadata = render_metadata(catalog, renderer)

    with pytest.raises(ValueError, match="missing metadata"):
        reduce_metadata(
            catalog,
            metadata[:-1],
            expected_renderer_id=renderer.renderer_id,
        )
    with pytest.raises(ValueError, match="duplicate metadata"):
        reduce_metadata(
            catalog,
            (*metadata, metadata[0]),
            expected_renderer_id=renderer.renderer_id,
        )
    tampered = replace(metadata[0], renderer_id="untrusted-renderer")
    with pytest.raises(ValueError, match="renderer"):
        reduce_metadata(
            catalog,
            (tampered, *metadata[1:]),
            expected_renderer_id=renderer.renderer_id,
        )


def test_task_ordinal_partitions_are_disjoint_and_complete():
    partitions = [
        parallel.partition_ordinals(
            record_count=17,
            task_index=task_index,
            task_count=4,
        )
        for task_index in range(4)
    ]

    assert partitions == [
        tuple(range(task_index, 17, 4)) for task_index in range(4)
    ]
    assert set().union(*(set(partition) for partition in partitions)) == set(
        range(17)
    )
    assert sum(len(partition) for partition in partitions) == 17


def test_task_results_reduce_in_catalog_order_with_bound_cached_payloads():
    catalog, renderer, config = _fixture_build(record_count=17)
    results = tuple(
        parallel.render_task_result(
            catalog,
            renderer,
            config,
            task_index=task_index,
            task_count=4,
            workers=2,
        )
        for task_index in range(4)
    )

    metadata, payloads = parallel.reduce_task_results(
        catalog,
        renderer.renderer_id,
        config,
        tuple(reversed(results)),
        expected_task_count=4,
    )

    assert metadata == render_metadata(catalog, renderer, workers=3)
    assert tuple(record.ordinal for record in metadata) == tuple(range(17))
    assert set(payloads) == {record.record_id for record in catalog.records}
    assert all(
        hashlib.sha256(payloads[record.record_id]).hexdigest()
        == metadata[record.ordinal].render_sha256
        for record in catalog.records
    )
    assert all(
        result.build_id in parallel.task_result_filename(result)
        for result in results
    )


def test_task_reducer_rejects_missing_duplicate_and_build_id_drift():
    catalog, renderer, config = _fixture_build(record_count=12)
    results = tuple(
        parallel.render_task_result(
            catalog,
            renderer,
            config,
            task_index=task_index,
            task_count=3,
        )
        for task_index in range(3)
    )

    with pytest.raises(ValueError, match="missing task"):
        parallel.reduce_task_results(
            catalog,
            renderer.renderer_id,
            config,
            results[:-1],
            expected_task_count=3,
        )
    with pytest.raises(ValueError, match="duplicate task"):
        parallel.reduce_task_results(
            catalog,
            renderer.renderer_id,
            config,
            (results[0], results[0], results[2]),
            expected_task_count=3,
        )
    with pytest.raises(ValueError, match="build id"):
        parallel.reduce_task_results(
            catalog,
            renderer.renderer_id,
            replace(config, update_tokens=32),
            results,
            expected_task_count=3,
        )


def test_task_result_serialization_rejects_payload_and_result_hash_drift():
    catalog, renderer, config = _fixture_build(record_count=6)
    result = parallel.render_task_result(
        catalog,
        renderer,
        config,
        task_index=0,
        task_count=2,
    )
    encoded = parallel.task_result_to_bytes(result)
    assert parallel.task_result_from_bytes(encoded) == result

    value = json.loads(encoded)
    value["records"][0]["payload_base64"] = base64.b64encode(b"drift").decode(
        "ascii"
    )
    with pytest.raises(ValueError, match="cached payload|task result digest"):
        parallel.task_result_from_bytes(canonical_json_bytes(value))

    value = json.loads(encoded)
    value["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="task result digest"):
        parallel.task_result_from_bytes(canonical_json_bytes(value))


def test_task_result_finalization_matches_serial_publication(tmp_path):
    catalog, renderer, config = _fixture_build(record_count=18)
    results = tuple(
        parallel.render_task_result(
            catalog,
            renderer,
            config,
            task_index=task_index,
            task_count=3,
            workers=2,
        )
        for task_index in range(3)
    )
    serial_root = tmp_path / "serial"
    partitioned_root = tmp_path / "partitioned"

    serial = build_parallel_corpus(
        catalog,
        renderer,
        config,
        serial_root,
        workers=3,
    )
    partitioned = parallel.build_parallel_corpus_from_tasks(
        catalog,
        renderer.renderer_id,
        config,
        partitioned_root,
        results,
        expected_task_count=3,
    )

    assert partitioned == serial
    assert _tree_bytes(partitioned_root) == _tree_bytes(serial_root)


def test_task_workspace_is_build_namespaced_symlink_safe_and_owned(tmp_path):
    catalog, renderer, config = _fixture_build(record_count=9)
    result = parallel.render_task_result(
        catalog,
        renderer,
        config,
        task_index=0,
        task_count=3,
    )
    shared_root = tmp_path / "shared"

    result_path = parallel.publish_task_result(
        shared_root,
        result,
        scheduler_id="job-42",
        nonce="nonce-a",
    )
    workspace = parallel.task_workspace_path(
        shared_root,
        result.build_id,
        scheduler_id="job-42",
        nonce="nonce-a",
    )

    assert result.build_id in result_path.name
    assert result.build_id in result_path.parts
    assert result_path.parent == workspace
    with pytest.raises(ValueError, match="ownership"):
        parallel.cleanup_task_workspace(
            workspace,
            build_id=result.build_id,
            scheduler_id="job-42",
            nonce="wrong",
        )
    assert result_path.is_file()

    victim = tmp_path / "victim-workspace"
    victim.mkdir()
    (victim / "sentinel").write_bytes(b"safe")
    hostile_workspace = parallel.task_workspace_path(
        shared_root,
        result.build_id,
        scheduler_id="job-evil",
        nonce="nonce-a",
    )
    hostile_workspace.symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        parallel.publish_task_result(
            shared_root,
            result,
            scheduler_id="job-evil",
            nonce="nonce-a",
        )
    assert _tree_bytes(victim) == {"sentinel": b"safe"}

    foreign_cleanup_entry = workspace / ".foreign"
    foreign_cleanup_entry.write_bytes(b"keep")
    with pytest.raises(ValueError, match="foreign"):
        parallel.cleanup_task_workspace(
            workspace,
            build_id=result.build_id,
            scheduler_id="job-42",
            nonce="nonce-a",
        )
    assert result_path.is_file()
    assert foreign_cleanup_entry.read_bytes() == b"keep"
    foreign_cleanup_entry.unlink()

    parallel.cleanup_task_workspace(
        workspace,
        build_id=result.build_id,
        scheduler_id="job-42",
        nonce="nonce-a",
    )
    assert not workspace.exists()


def test_task_workspace_loader_requires_exact_complete_foreign_free_results(tmp_path):
    catalog, renderer, config = _fixture_build(record_count=9)
    results = tuple(
        parallel.render_task_result(
            catalog,
            renderer,
            config,
            task_index=task_index,
            task_count=3,
        )
        for task_index in range(3)
    )
    shared_root = tmp_path / "shared"
    for result in results[:2]:
        parallel.publish_task_result(
            shared_root,
            result,
            scheduler_id="job-42",
            nonce="nonce-a",
        )

    with pytest.raises(parallel.IncompleteTaskResults, match="missing task"):
        parallel.load_task_results(
            shared_root,
            results[0].build_id,
            scheduler_id="job-42",
            nonce="nonce-a",
            expected_task_count=3,
        )

    parallel.publish_task_result(
        shared_root,
        results[2],
        scheduler_id="job-42",
        nonce="nonce-a",
    )
    loaded = parallel.load_task_results(
        shared_root,
        results[0].build_id,
        scheduler_id="job-42",
        nonce="nonce-a",
        expected_task_count=3,
    )
    assert loaded == results

    workspace = parallel.task_workspace_path(
        shared_root,
        results[0].build_id,
        scheduler_id="job-42",
        nonce="nonce-a",
    )
    (workspace / "foreign").write_bytes(b"poison")
    with pytest.raises(ValueError, match="foreign"):
        parallel.load_task_results(
            shared_root,
            results[0].build_id,
            scheduler_id="job-42",
            nonce="nonce-a",
            expected_task_count=3,
        )


def test_cli_fixture_tasks_are_locally_orchestratable_and_build_bound(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_parallel_corpus.py"
    shared_root = tmp_path / "shared"
    catalog, renderer, config = _fixture_build(record_count=9)
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    common = [
        "--shared-root",
        str(shared_root),
        "--scheduler-id",
        "local-job",
        "--nonce",
        "nonce-a",
        "--records",
        "9",
        "--task-count",
        "3",
        "--update-tokens",
        "64",
        "--shards",
        "32",
        "--allow-fewer-shards",
    ]

    for task_index in range(3):
        task = subprocess.run(
            [
                sys.executable,
                str(script),
                "render-fixture-task",
                *common,
                "--task-index",
                str(task_index),
                "--workers",
                "2",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert task.returncode == 0, task.stderr
        task_status = json.loads(task.stdout)
        assert task_status["build_id"] == build_id
        assert task_status["task_index"] == task_index
        assert build_id in task_status["result_path"]

    finalize = subprocess.run(
        [
            sys.executable,
            str(script),
            "finalize-fixture-tasks",
            *common,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert finalize.returncode == 0, finalize.stderr
    receipt = json.loads(finalize.stdout)
    output = shared_root / f"fixture-{build_id}"
    assert receipt["build_id"] == build_id
    assert verify_parallel_corpus(output, expected_build_id=build_id) == receipt

    changed_catalog = fixture_catalog(record_count=10)
    changed_build_id = parallel_build_id(
        changed_catalog,
        renderer.renderer_id,
        config,
    )
    assert changed_build_id != build_id
    mismatch = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify",
            "--output",
            str(output),
            "--expected-build-id",
            changed_build_id,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0


def test_cli_finalize_reports_incomplete_without_publishing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_parallel_corpus.py"
    shared_root = tmp_path / "shared"
    common = [
        "--shared-root",
        str(shared_root),
        "--scheduler-id",
        "local-job",
        "--nonce",
        "nonce-a",
        "--records",
        "9",
        "--task-count",
        "3",
        "--update-tokens",
        "64",
        "--shards",
        "32",
        "--allow-fewer-shards",
    ]
    task = subprocess.run(
        [
            sys.executable,
            str(script),
            "render-fixture-task",
            *common,
            "--task-index",
            "0",
            "--workers",
            "1",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert task.returncode == 0, task.stderr
    build_id = json.loads(task.stdout)["build_id"]

    finalize = subprocess.run(
        [
            sys.executable,
            str(script),
            "finalize-fixture-tasks",
            *common,
            "--allow-incomplete",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert finalize.returncode == 75
    assert "missing task" in finalize.stderr
    assert not (shared_root / f"fixture-{build_id}").exists()


def test_largest_deficit_schedule_is_stable_and_covers_each_record_once():
    catalog = fixture_catalog(record_count=8)
    rendered = render_metadata(catalog, FixtureRenderer())
    lanes = ("natural", "natural", "natural", "natural", "facts", "facts", "reasoning", "reasoning")
    metadata = tuple(
        replace(record, lane=lane, token_length=10)
        for record, lane in zip(rendered, lanes, strict=True)
    )
    weights = (("natural", 2), ("facts", 1), ("reasoning", 1))

    schedule = largest_deficit_schedule(metadata, weights)
    reordered = largest_deficit_schedule(tuple(reversed(metadata)), weights)

    assert schedule == reordered
    assert [entry.lane for entry in schedule] == [
        "natural",
        "natural",
        "natural",
        "facts",
        "reasoning",
        "natural",
        "facts",
        "reasoning",
    ]
    assert [entry.record_id for entry in schedule] == [
        metadata[index].record_id for index in (0, 1, 2, 4, 6, 3, 5, 7)
    ]
    assert [(entry.token_start, entry.token_end) for entry in schedule] == [
        (index * 10, (index + 1) * 10) for index in range(8)
    ]
    assert schedule_from_bytes(schedule_to_bytes(schedule)) == schedule


def test_shard_assignments_are_exactly_32_and_update_aligned():
    assignments = assign_update_aligned_shards(
        total_tokens=32 * 5 * 16,
        update_tokens=16,
    )

    assert len(assignments) == 32
    assert all(assignment.shard_count == 32 for assignment in assignments)
    assert all(
        assignment.update_end - assignment.update_start == 5
        for assignment in assignments
    )
    assert assignments[0].token_start == 0
    assert assignments[-1].token_end == 32 * 5 * 16
    assert all(
        assignment.token_start % 16 == assignment.token_end % 16 == 0
        for assignment in assignments
    )
    assert all(
        left.token_end == right.token_start
        for left, right in zip(assignments, assignments[1:])
    )
    encoded = assignments_to_bytes(assignments)
    assert assignments_from_bytes(encoded) == assignments


def test_tiny_shard_assignment_uses_fewer_shards_only_when_explicit():
    with pytest.raises(ValueError, match="fewer updates"):
        assign_update_aligned_shards(total_tokens=35, update_tokens=16)

    assignments = assign_update_aligned_shards(
        total_tokens=35,
        update_tokens=16,
        allow_fewer=True,
    )
    assert len(assignments) == 3
    assert assignments[-1].token_end == 48


def test_ordered_stream_and_merkle_verifier_rejects_wrong_commitments():
    metadata = render_metadata(fixture_catalog(record_count=9), FixtureRenderer())
    schedule = largest_deficit_schedule(
        metadata,
        (("natural", 1), ("facts", 1), ("reasoning", 1)),
    )
    ordered_hash, merkle_root = ordered_stream_commitments(schedule, metadata)

    assert verify_stream_commitments(
        schedule,
        metadata,
        ordered_stream_sha256=ordered_hash,
        merkle_root_sha256=merkle_root,
    )
    with pytest.raises(ValueError, match="ordered stream"):
        verify_stream_commitments(
            schedule,
            metadata,
            ordered_stream_sha256="0" * 64,
            merkle_root_sha256=merkle_root,
        )
    with pytest.raises(ValueError, match="Merkle"):
        verify_stream_commitments(
            schedule,
            metadata,
            ordered_stream_sha256=ordered_hash,
            merkle_root_sha256="0" * 64,
        )


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fixture_build(record_count=12):
    catalog = fixture_catalog(record_count=record_count)
    renderer = FixtureRenderer()
    config = ParallelBuildConfig(
        lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
        update_tokens=64,
        allow_fewer_shards=True,
    )
    return catalog, renderer, config


def _stage_owner_bytes(build_id):
    return canonical_json_bytes(
        {
            "build_id": build_id,
            "format": "memorysplit-parallel-corpus-v1",
            "kind": "publication-staging",
        }
    )


def test_atomic_rename_adapter_never_replaces_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "identity").write_text("source", encoding="utf-8")
    (destination / "identity").write_text("destination", encoding="utf-8")
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(FileExistsError):
            parallel.atomic_rename_noreplace(
                directory_fd,
                source.name,
                directory_fd,
                destination.name,
            )
    finally:
        os.close(directory_fd)

    assert (source / "identity").read_text(encoding="utf-8") == "source"
    assert (destination / "identity").read_text(encoding="utf-8") == "destination"


def test_staging_shards_symlink_receives_no_temporary_bytes(tmp_path):
    catalog, renderer, config = _fixture_build()
    destination = tmp_path / "corpus"
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    staging = publication_staging_path(destination, build_id)
    staging.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (staging / "shards").symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="shards"):
        build_parallel_corpus(catalog, renderer, config, destination)

    assert list(victim.iterdir()) == []


def test_destination_race_is_atomic_noreplace(tmp_path, monkeypatch):
    catalog, renderer, config = _fixture_build()
    destination = tmp_path / "corpus"
    real_rename = publication_module.atomic_rename_noreplace
    raced = False

    def race_once(source_fd, source_name, destination_fd, destination_name):
        nonlocal raced
        if destination_name == destination.name and not raced:
            raced = True
            os.mkdir(destination_name, dir_fd=destination_fd)
            competitor_fd = os.open(
                destination_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=destination_fd,
            )
            try:
                sentinel_fd = os.open(
                    "sentinel",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=competitor_fd,
                )
                os.write(sentinel_fd, b"competitor")
                os.close(sentinel_fd)
            finally:
                os.close(competitor_fd)
        return real_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(publication_module, "atomic_rename_noreplace", race_once)

    with pytest.raises(ValueError, match="conflicting parallel corpus output"):
        build_parallel_corpus(catalog, renderer, config, destination)

    assert (destination / "sentinel").read_bytes() == b"competitor"


def test_crash_resume_cleans_only_owned_stale_regular_temporaries(tmp_path):
    catalog, renderer, config = _fixture_build()
    destination = tmp_path / "corpus"
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    staging = publication_staging_path(destination, build_id)
    staging.mkdir()
    (staging / ".parallel-owner.json").write_bytes(_stage_owner_bytes(build_id))
    stale_root = staging / f".catalog.jsonl.tmp-{build_id}-crash"
    stale_root.write_bytes(b"partial")
    shards = staging / "shards"
    shards.mkdir()
    metadata = render_metadata(catalog, renderer)
    schedule = largest_deficit_schedule(metadata, config.lane_weights)
    assignments = assign_update_aligned_shards(
        total_tokens=schedule[-1].token_end,
        update_tokens=config.update_tokens,
        shard_count=config.shard_count,
        allow_fewer=config.allow_fewer_shards,
    )
    stale_shard = (
        shards / f".{assignments[0].shard_id}.bin.tmp-{build_id}-crash"
    )
    stale_shard.write_bytes(b"partial")

    receipt = build_parallel_corpus(catalog, renderer, config, destination)

    assert verify_parallel_corpus(destination) == receipt
    assert not stale_root.exists()
    assert not stale_shard.exists()
    assert not any(".tmp-" in path.name for path in destination.rglob("*"))


def test_resume_rejects_foreign_or_symlink_temporary_entries(tmp_path):
    catalog, renderer, config = _fixture_build()
    destination = tmp_path / "corpus"
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    staging = publication_staging_path(destination, build_id)
    staging.mkdir()
    (staging / ".parallel-owner.json").write_bytes(_stage_owner_bytes(build_id))
    foreign = staging / ".catalog.jsonl.tmp-foreign"
    foreign.write_bytes(b"do-not-delete")

    with pytest.raises(ValueError, match="foreign"):
        build_parallel_corpus(catalog, renderer, config, destination)
    assert foreign.read_bytes() == b"do-not-delete"

    foreign.unlink()
    victim = tmp_path / "victim-temp"
    victim.write_bytes(b"do-not-touch")
    (staging / f".catalog.jsonl.tmp-{build_id}-crash").symlink_to(victim)
    with pytest.raises(ValueError, match="temporary"):
        build_parallel_corpus(catalog, renderer, config, destination)
    assert victim.read_bytes() == b"do-not-touch"


def test_tiny_publication_is_worker_independent_and_self_verifying(tmp_path):
    catalog = fixture_catalog(record_count=24)
    renderer = FixtureRenderer()
    config = ParallelBuildConfig(
        lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
        update_tokens=64,
        allow_fewer_shards=True,
    )

    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    serial = build_parallel_corpus(
        catalog,
        renderer,
        config,
        serial_root,
        workers=1,
    )
    parallel = build_parallel_corpus(
        catalog,
        renderer,
        config,
        parallel_root,
        workers=4,
    )

    assert serial == parallel
    assert _tree_bytes(serial_root) == _tree_bytes(parallel_root)
    assert verify_parallel_corpus(serial_root) == serial
    assert serial["shard_count"] < 32
    assert len(serial["ordered_stream_sha256"]) == 64
    assert len(serial["merkle_root_sha256"]) == 64
    assert len(serial["packed_stream_sha256"]) == 64
    assert all(len(artifact["sha256"]) == 64 for artifact in serial["artifacts"])


def test_publication_resumes_partial_stage_and_is_idempotent(tmp_path):
    catalog = fixture_catalog(record_count=12)
    renderer = FixtureRenderer()
    config = ParallelBuildConfig(
        lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
        update_tokens=64,
        allow_fewer_shards=True,
    )
    destination = tmp_path / "corpus"
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    staging = publication_staging_path(destination, build_id)
    staging.mkdir()
    (staging / ".parallel-owner.json").write_bytes(_stage_owner_bytes(build_id))
    (staging / "catalog.jsonl").write_bytes(catalog.to_bytes())

    first = build_parallel_corpus(catalog, renderer, config, destination)
    assert destination.is_dir()
    assert not staging.exists()
    before = {
        path.relative_to(destination).as_posix(): (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in destination.rglob("*")
        if path.is_file()
    }

    second = build_parallel_corpus(
        catalog,
        renderer,
        config,
        destination,
        workers=3,
    )
    after = {
        path.relative_to(destination).as_posix(): (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert second == first
    assert after == before


def test_publication_and_rerun_fail_closed_on_tampered_shard(tmp_path):
    catalog = fixture_catalog(record_count=12)
    renderer = FixtureRenderer()
    config = ParallelBuildConfig(
        lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
        update_tokens=64,
        allow_fewer_shards=True,
    )
    destination = tmp_path / "corpus"
    build_parallel_corpus(catalog, renderer, config, destination)
    shard = next((destination / "shards").glob("*.bin"))
    original = shard.read_bytes()
    shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(ValueError, match="digest drift"):
        verify_parallel_corpus(destination)
    with pytest.raises(ValueError, match="conflicting parallel corpus output"):
        build_parallel_corpus(catalog, renderer, config, destination)
    assert shard.read_bytes() != original


def test_production_renderer_adapter_fails_closed_without_publication(tmp_path):
    catalog = fixture_catalog(record_count=3)
    renderer = UnsupportedProductionRenderer("fineweb_edu")
    config = ParallelBuildConfig(
        lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
        update_tokens=64,
        allow_fewer_shards=True,
    )
    destination = tmp_path / "production"

    with pytest.raises(UnsupportedSourceError, match="fineweb_edu"):
        build_parallel_corpus(catalog, renderer, config, destination)
    assert not destination.exists()


def test_cli_runs_tiny_build_and_verification_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_parallel_corpus.py"
    destination = tmp_path / "cli-corpus"
    build = subprocess.run(
        [
            sys.executable,
            str(script),
            "build-fixture",
            "--output",
            str(destination),
            "--records",
            "18",
            "--workers",
            "3",
            "--update-tokens",
            "64",
            "--allow-fewer-shards",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    receipt = json.loads(build.stdout)

    verify = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify",
            "--output",
            str(destination),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout) == receipt


def test_verification_receipt_publication_is_pinned_exact_and_idempotent(tmp_path):
    catalog, renderer, config = _fixture_build()
    corpus = tmp_path / "corpus"
    receipt = build_parallel_corpus(catalog, renderer, config, corpus)
    build_id = receipt["build_id"]
    destination = tmp_path / "verification" / f"receipt-{build_id}.json"

    first = parallel.publish_verification_receipt(
        corpus,
        destination,
        expected_build_id=build_id,
    )
    before = destination.stat()
    second = parallel.publish_verification_receipt(
        corpus,
        destination,
        expected_build_id=build_id,
    )
    after = destination.stat()

    assert first == second == receipt
    assert destination.read_bytes() == canonical_json_bytes(receipt)
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)

    conflict = destination.with_name(f"conflict-{build_id}.json")
    conflict.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="drift"):
        parallel.publish_verification_receipt(
            corpus,
            conflict,
            expected_build_id=build_id,
        )
    assert conflict.read_bytes() == b"foreign"

    victim = tmp_path / "verification-victim"
    victim.write_bytes(b"safe")
    hostile = destination.with_name(f"hostile-{build_id}.json")
    hostile.symlink_to(victim)
    with pytest.raises(ValueError, match="unsafe"):
        parallel.publish_verification_receipt(
            corpus,
            hostile,
            expected_build_id=build_id,
        )
    assert victim.read_bytes() == b"safe"


def test_v2_slurm_templates_encode_real_partition_and_safe_publication():
    root = Path(__file__).resolve().parents[1]
    build = root / "cluster" / "slurm" / "v2_corpus_build.sbatch"
    verify = root / "cluster" / "slurm" / "v2_corpus_verify.sbatch"
    build_text = build.read_text(encoding="utf-8")
    verify_text = verify.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-31" in build_text
    assert "render-fixture-task" in build_text
    assert "finalize-fixture-tasks" in build_text
    assert "--task-index" in build_text
    assert "SLURM_ARRAY_TASK_ID" in build_text
    assert "--task-count" in build_text
    assert "SLURM_ARRAY_TASK_COUNT" in build_text
    assert "--allow-incomplete" in build_text
    assert "SLURM_ARRAY_JOB_ID" in build_text
    assert "MS_RUN_NONCE" in build_text
    assert "build-fixture" not in build_text

    assert "#SBATCH --array" not in verify_text
    assert "MS_BUILD_ID" in verify_text
    assert "fixture-${MS_BUILD_ID}" in verify_text
    assert "publish-verification-receipt" in verify_text
    assert "--expected-build-id" in verify_text

    for template, text in ((build, build_text), (verify, verify_text)):
        text = template.read_text(encoding="utf-8")
        assert "#SBATCH --partition=normal" in text
        assert "#SBATCH --export=NONE" in text
        assert "--export=ALL" not in text
        assert "#SBATCH --gres" not in text
        assert "MS_SHARED_ROOT" in text
        assert "env -i" in text
        assert "rm -rf" not in text
        assert "\nmv " not in text
        assert "\ncp " not in text
        syntax = subprocess.run(
            ["bash", "-n", str(template)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_v2_slurm_scripts_run_locally_as_disjoint_array_and_single_verify(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_script = root / "cluster" / "slurm" / "v2_corpus_build.sbatch"
    verify_script = root / "cluster" / "slurm" / "v2_corpus_verify.sbatch"
    shared_root = tmp_path / "shared"
    local_tmp = tmp_path / "local"
    shared_root.mkdir()
    local_tmp.mkdir()
    base_environment = {
        **os.environ,
        "MS_REPO": str(root),
        "MS_PYTHON": sys.executable,
        "MS_SHARED_ROOT": str(shared_root),
        "MS_RUN_NONCE": "local-nonce",
        "MS_RECORDS": "9",
        "MS_UPDATE_TOKENS": "64",
        "MS_ALLOW_FEWER_SHARDS": "1",
        "SLURM_ARRAY_JOB_ID": "local-array",
        "SLURM_ARRAY_TASK_COUNT": "3",
        "SLURM_CPUS_PER_TASK": "2",
        "SLURM_TMPDIR": str(local_tmp),
    }
    for task_index in range(3):
        task = subprocess.run(
            ["bash", str(build_script)],
            cwd=root,
            env={
                **base_environment,
                "SLURM_ARRAY_TASK_ID": str(task_index),
                "SLURM_JOB_ID": f"local-array-{task_index}",
            },
            check=False,
            capture_output=True,
            text=True,
        )
        assert task.returncode == 0, task.stderr

    catalog, renderer, config = _fixture_build(record_count=9)
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    corpus = shared_root / f"fixture-{build_id}"
    assert verify_parallel_corpus(corpus, expected_build_id=build_id)

    verification = subprocess.run(
        ["bash", str(verify_script)],
        cwd=root,
        env={
            **base_environment,
            "MS_BUILD_ID": build_id,
            "SLURM_JOB_ID": "local-verify",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert verification.returncode == 0, verification.stderr
    receipt_path = (
        shared_root / "verification" / f"receipt-{build_id}.json"
    )
    assert json.loads(receipt_path.read_bytes())["build_id"] == build_id
