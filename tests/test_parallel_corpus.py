import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

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


def test_v2_slurm_templates_are_cpu_arrays_with_isolated_environments():
    root = Path(__file__).resolve().parents[1]
    templates = (
        root / "cluster" / "slurm" / "v2_corpus_build.sbatch",
        root / "cluster" / "slurm" / "v2_corpus_verify.sbatch",
    )

    for template in templates:
        text = template.read_text(encoding="utf-8")
        assert "#SBATCH --array=0-31" in text
        assert "#SBATCH --partition=normal" in text
        assert "#SBATCH --export=NONE" in text
        assert "--export=ALL" not in text
        assert "#SBATCH --gres" not in text
        assert "SLURM_TMPDIR" in text
        assert "MS_SHARED_ROOT" in text
        assert "env -i" in text
        syntax = subprocess.run(
            ["bash", "-n", str(template)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr
