from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import (
    PublicSourceResolver,
    SourceFile,
    SourceLock,
    SourceRequest,
    load_source_lock as _load_source_lock_authority,
    resolve_source_lock,
    stage_source_lock as _stage_source_lock_authority,
    verify_source_tree as _verify_source_tree_authority,
)
from reasoning_v2_fixtures import (
    FakePublicResolver,
    FixtureSourceLock,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
    valid_lock_json,
)


def _write_lock(tmp_path: Path, value: object, *, canonical: bool = True) -> Path:
    path = tmp_path / "lock.json"
    payload = (
        canonical_json_bytes(value)
        if canonical
        else json.dumps(value, indent=2).encode("utf-8")
    )
    path.write_bytes(payload)
    return path


EXPECTED_GENERATOR_COMMIT = "a" * 40


def load_source_lock(
    path: Path,
    *,
    expected_generator_commit: str = EXPECTED_GENERATOR_COMMIT,
):
    return _load_source_lock_authority(
        path,
        expected_generator_commit=expected_generator_commit,
    )


def verify_source_tree(
    lock,
    source_root: Path,
    *,
    expected_generator_commit: str = EXPECTED_GENERATOR_COMMIT,
):
    return _verify_source_tree_authority(
        lock,
        source_root,
        expected_generator_commit=expected_generator_commit,
    )


def stage_source_lock(
    lock,
    download_root: Path,
    canonical_root: Path,
    *,
    expected_generator_commit: str = EXPECTED_GENERATOR_COMMIT,
):
    return _stage_source_lock_authority(
        lock,
        download_root,
        canonical_root,
        expected_generator_commit=expected_generator_commit,
    )


def _source_json(value: dict[str, object], source_id: str) -> dict[str, object]:
    return next(
        source
        for source in value["sources"]
        if source["source_id"] == source_id
    )


def test_resolver_emits_only_immutable_licensed_hash_complete_entries(
    tmp_path,
    fake_public_resolver,
    full_recipe,
):
    lock = resolve_source_lock(
        full_recipe,
        fake_public_resolver,
        tmp_path / "downloads",
        generator_commit="a" * 40,
    )
    by_id = {entry.source_id: entry for entry in lock.sources}
    assert set(by_id) == {
        "fineweb_edu",
        "finemath",
        "wikidata5m",
        "clrs_text",
        "ruletaker",
        "prontoqa",
        "reasoning_gym_exact_answer",
        "deepmind_mathematics_generator",
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    }
    for entry in lock.sources:
        assert entry.revision_kind in {"git_commit", "content_sha256"}
        assert len(entry.revision) in {40, 64}
        assert entry.license_spdx
        assert entry.license_files
        assert entry.files
        assert all(row.bytes >= 0 and len(row.sha256) == 64 for row in entry.files)
        assert not Path(entry.materialized_path).is_absolute()


def test_public_huggingface_resolver_never_uses_ambient_token(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, *, token):
            calls.append(token)

    monkeypatch.setenv("HF_TOKEN", "must-not-be-consumed")
    monkeypatch.setattr(source_lock_module, "HfApi", RecordingApi)
    PublicSourceResolver()
    assert calls == [False]


@pytest.mark.parametrize(
    "revision",
    ["main", "master", "latest", "v2.0.1", "refs/heads/main", ""],
)
def test_source_lock_rejects_mutable_revision_text(
    tmp_path,
    valid_lock_json,
    revision,
):
    valid_lock_json["sources"][0]["revision"] = revision
    with pytest.raises(ValueError, match="immutable revision"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_staging_is_content_addressed_and_rejects_byte_drift(
    tmp_path,
    fixture_source_lock: FixtureSourceLock,
):
    root = stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        tmp_path / "canonical",
    )
    assert root == (
        tmp_path
        / "canonical"
        / "sources"
        / fixture_source_lock.lock.sha256
    )
    assert verify_source_tree(fixture_source_lock.lock, root)["passed"] is True
    assert (
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            tmp_path / "canonical",
        )
        == root
    )
    victim = next(path for path in root.rglob("*") if path.is_file())
    victim.write_bytes(victim.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="source byte drift"):
        verify_source_tree(fixture_source_lock.lock, root)
    with pytest.raises(ValueError, match="source byte drift|conflicting"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            tmp_path / "canonical",
        )


def test_source_lock_rejects_duplicate_source_ids(tmp_path, valid_lock_json):
    valid_lock_json["sources"].append(copy.deepcopy(valid_lock_json["sources"][0]))
    with pytest.raises(ValueError, match="duplicate source"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_source_lock_rejects_duplicate_file_paths(tmp_path, valid_lock_json):
    source = valid_lock_json["sources"][0]
    source["files"].append(copy.deepcopy(source["files"][0]))
    with pytest.raises(ValueError, match="duplicate file"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_source_lock_rejects_file_directory_collisions(
    tmp_path,
    valid_lock_json,
):
    source = valid_lock_json["sources"][0]
    nested = next(row["path"] for row in source["files"] if "/" in row["path"])
    source["files"].append(
        {
            "bytes": 1,
            "path": nested.split("/", 1)[0],
            "sha256": "b" * 64,
        }
    )
    source["files"].sort(key=lambda row: row["path"].encode("utf-8"))
    with pytest.raises(ValueError, match="collides with a directory"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


@pytest.mark.parametrize("materialized_path", ["/absolute", "../escape", "a/../../b"])
def test_source_lock_rejects_unsafe_materialized_paths(
    tmp_path,
    valid_lock_json,
    materialized_path,
):
    valid_lock_json["sources"][0]["materialized_path"] = materialized_path
    with pytest.raises(ValueError, match="materialized path"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


@pytest.mark.parametrize("file_path", ["/absolute", "../escape", "a/../../b"])
def test_source_lock_rejects_unsafe_file_paths(
    tmp_path,
    valid_lock_json,
    file_path,
):
    valid_lock_json["sources"][0]["files"][0]["path"] = file_path
    with pytest.raises(ValueError, match="file path"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_source_lock_rejects_boolean_byte_counts(tmp_path, valid_lock_json):
    valid_lock_json["sources"][0]["files"][0]["bytes"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_source_lock_rejects_unsupported_spdx(tmp_path, valid_lock_json):
    valid_lock_json["sources"][0]["license_spdx"] = "GPL-3.0-only"
    with pytest.raises(ValueError, match="unsupported SPDX"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_source_lock_rejects_missing_license_inventory_file(
    tmp_path,
    valid_lock_json,
):
    source = valid_lock_json["sources"][0]
    license_path = source["license_files"][0]
    source["files"] = [
        row for row in source["files"] if row["path"] != license_path
    ]
    with pytest.raises(ValueError, match="license file"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_source_lock_rejects_noncanonical_json(tmp_path, valid_lock_json):
    with pytest.raises(ValueError, match="canonical"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json, canonical=False))


def test_source_lock_rejects_duplicate_json_keys(tmp_path, valid_lock_json):
    payload = canonical_json_bytes(valid_lock_json)
    duplicate = payload.replace(
        b'{"dataset_contract_sha256":',
        b'{"schema_version":1,"dataset_contract_sha256":',
        1,
    )
    path = tmp_path / "lock.json"
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_source_lock(path)


def test_source_tree_rejects_unlisted_files(
    fixture_source_lock: FixtureSourceLock,
):
    (fixture_source_lock.download_root / "unexpected.txt").write_text("foreign")
    with pytest.raises(ValueError, match="unlisted file"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


def test_source_tree_rejects_symlinks(
    fixture_source_lock: FixtureSourceLock,
):
    victim = next(
        path
        for path in fixture_source_lock.download_root.rglob("*")
        if path.is_file()
    )
    link = fixture_source_lock.download_root / "unsafe-link"
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_source_tree_rejects_special_files(
    fixture_source_lock: FixtureSourceLock,
):
    os.mkfifo(fixture_source_lock.download_root / "unsafe-fifo")
    with pytest.raises(ValueError, match="special"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


def test_source_tree_rejects_hardlinks(
    tmp_path,
    fixture_source_lock: FixtureSourceLock,
):
    victim = next(
        path
        for path in fixture_source_lock.download_root.rglob("*")
        if path.is_file()
    )
    external = tmp_path / "external"
    external.write_bytes(victim.read_bytes())
    victim.unlink()
    os.link(external, victim)
    with pytest.raises(ValueError, match="hardlink"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


@pytest.mark.parametrize("source_id", ["fineweb_edu", "wikidata5m", "arc_agi_1"])
def test_resolver_rejects_fixed_identity_drift(
    tmp_path,
    fixed_contract_environment,
    full_recipe,
    source_id,
):
    resolver = FakePublicResolver(revision_overrides={source_id: "f" * 40})
    with pytest.raises(ValueError, match="fixed source identity drift"):
        resolve_source_lock(
            full_recipe,
            resolver,
            tmp_path / "downloads",
            generator_commit="a" * 40,
        )


def test_resolver_rejects_drift_in_committed_fixed_lock(
    tmp_path,
    fixed_contract_environment,
    full_recipe,
):
    path = source_lock_module.CURRENT_DATASET_LOCK_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["sources"]["fineweb_edu"]["revision"] = "f" * 40
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="fixed FineWeb"):
        resolve_source_lock(
            full_recipe,
            FakePublicResolver(),
            tmp_path / "downloads",
            generator_commit="a" * 40,
        )


def test_ruletaker_archive_url_without_computed_digest_is_unresolved(
    tmp_path,
    fixed_contract_environment,
    full_recipe,
):
    with pytest.raises(ValueError, match="RuleTaker archive.*digest"):
        resolve_source_lock(
            full_recipe,
            FakePublicResolver(omit_ruletaker_archive=True),
            tmp_path / "downloads",
            generator_commit="a" * 40,
        )


def test_unresolved_source_fails_closed(
    tmp_path,
    fixed_contract_environment,
    full_recipe,
):
    with pytest.raises(ValueError, match="unresolved source.*prontoqa"):
        resolve_source_lock(
            full_recipe,
            FakePublicResolver(unresolved_source="prontoqa"),
            tmp_path / "downloads",
            generator_commit="a" * 40,
        )


def test_public_git_resolver_archives_the_exact_resolved_commit(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "LICENSE").write_text("Apache License\nVersion 2.0\n")
    (repository / "data.txt").write_text("immutable bytes\n")
    (repository / "empty-marker").write_bytes(b"")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
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
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    request = next(
        request
        for request in source_lock_module.PUBLIC_REQUESTS
        if request.source_id == "clrs_text"
    )
    real_run_git = source_lock_module._run_git

    def run_local_git(arguments):
        return real_run_git(
            [
                str(repository) if value == request.repository else value
                for value in arguments
            ]
        )

    monkeypatch.setattr(source_lock_module, "_run_git", run_local_git)
    monkeypatch.setattr(source_lock_module, "HfApi", lambda *, token: object())
    entry = PublicSourceResolver().resolve(
        request,
        tmp_path / "downloads",
    )

    assert entry.revision == commit
    assert entry.license_spdx == "Apache-2.0"
    assert {row.path for row in entry.files} == {
        "LICENSE",
        "data.txt",
        "empty-marker",
    }
    empty = next(row for row in entry.files if row.path == "empty-marker")
    assert empty.bytes == 0
    assert empty.sha256 == hashlib.sha256(b"").hexdigest()
    materialized = tmp_path / "downloads" / entry.materialized_path
    assert (materialized / "data.txt").read_text() == "immutable bytes\n"
    assert (materialized / "empty-marker").read_bytes() == b""


def test_public_resolver_rejects_unreviewed_request_before_transport(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(source_lock_module, "HfApi", lambda *, token: object())
    monkeypatch.setattr(
        source_lock_module,
        "_run_git",
        lambda arguments: pytest.fail("unreviewed request reached git"),
    )
    with pytest.raises(ValueError, match="reviewed source catalog"):
        PublicSourceResolver().resolve(
            SourceRequest(
                "local_fixture",
                "git",
                str(tmp_path / "repository"),
                required_license_paths=("LICENSE",),
            ),
            tmp_path / "downloads",
        )


def test_failed_public_resolution_preserves_partial_materialization(
    tmp_path,
    monkeypatch,
):
    request = next(
        row
        for row in source_lock_module.PUBLIC_REQUESTS
        if row.source_id == "clrs_text"
    )
    monkeypatch.setattr(source_lock_module, "HfApi", lambda *, token: object())
    resolver = PublicSourceResolver()

    def fail_after_write(_request, _root, materialized):
        (materialized / "partial").write_bytes(b"partial")
        raise ValueError("fixture failure")

    monkeypatch.setattr(resolver, "_resolve_git", fail_after_write)
    with pytest.raises(ValueError, match="fixture failure"):
        resolver.resolve(request, tmp_path / "downloads")
    assert (
        tmp_path / "downloads" / request.source_id / "partial"
    ).read_bytes() == b"partial"


def test_zero_byte_source_file_verifies_and_stages(
    tmp_path,
    fixture_source_lock,
):
    original = fixture_source_lock.lock
    source = next(
        entry for entry in original.sources if entry.source_id == "clrs_text"
    )
    empty = SourceFile(
        path="src/empty-marker",
        bytes=0,
        sha256=hashlib.sha256(b"").hexdigest(),
    )
    files = tuple(sorted((*source.files, empty), key=lambda row: row.path.encode()))
    updated_source = replace(source, files=files)
    updated = replace(
        original,
        sources=tuple(
            updated_source if entry.source_id == source.source_id else entry
            for entry in original.sources
        ),
    )
    marker = (
        fixture_source_lock.download_root
        / source.materialized_path
        / empty.path
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"")

    assert verify_source_tree(
        updated,
        fixture_source_lock.download_root,
    )["passed"]
    staged = stage_source_lock(
        updated,
        fixture_source_lock.download_root,
        tmp_path / "canonical",
    )
    assert (staged / source.materialized_path / empty.path).read_bytes() == b""


def test_fineweb_reviewed_members_are_exact_paths_not_prefixes():
    request = next(
        request
        for request in source_lock_module.FIXED_REQUESTS
        if request.source_id == "fineweb_edu"
    )
    assert request.required_data_paths == (
        "sample/10BT/000_00000.parquet",
        "sample/10BT/001_00000.parquet",
        "sample/10BT/002_00000.parquet",
    )
    assert request.required_data_prefixes == ()


def test_fineweb_member_child_cannot_satisfy_exact_path(
    tmp_path,
    monkeypatch,
):
    request = next(
        request
        for request in source_lock_module.FIXED_REQUESTS
        if request.source_id == "fineweb_edu"
    )
    entry = FakePublicResolver().resolve(request, tmp_path / "downloads")
    exact = request.required_data_paths[0]
    files = tuple(
        sorted(
            (
                replace(row, path=f"{exact}/child")
                if row.path == exact
                else row
                for row in entry.files
            ),
            key=lambda row: row.path.encode(),
        )
    )
    spoofed = replace(entry, files=files)
    monkeypatch.setattr(source_lock_module, "FIXED_FINEWEB_FILES", {})
    with pytest.raises(ValueError, match="required data file"):
        source_lock_module._validate_resolved_entry(request, spoofed)


def test_reviewed_directory_prefix_requires_a_descendant_file(
    tmp_path,
):
    request = next(
        request
        for request in source_lock_module.FIXED_REQUESTS
        if request.source_id == "arc_agi_1"
    )
    entry = FakePublicResolver().resolve(request, tmp_path / "downloads")
    files = tuple(
        sorted(
            (
                replace(row, path="data/training")
                if row.path == "data/training/a.json"
                else row
                for row in entry.files
            ),
            key=lambda row: row.path.encode(),
        )
    )
    spoofed = replace(entry, files=files)
    with pytest.raises(ValueError, match="required data prefix"):
        source_lock_module._validate_resolved_entry(request, spoofed)


def test_stage_rejects_world_writable_canonical_root(
    tmp_path,
    fixture_source_lock,
):
    canonical = tmp_path / "canonical"
    canonical.mkdir(mode=0o700)
    canonical.chmod(0o777)
    with pytest.raises(ValueError, match="mode"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )


def test_stage_detects_canonical_namespace_replacement_after_pinning(
    tmp_path,
    fixture_source_lock,
    monkeypatch,
):
    canonical = tmp_path / "canonical"
    displaced = tmp_path / "canonical-displaced"
    real_expected_paths = source_lock_module._expected_paths
    swapped = False

    def replace_namespace(lock):
        nonlocal swapped
        if not swapped and canonical.exists():
            swapped = True
            canonical.rename(displaced)
            canonical.mkdir(mode=0o700)
            (canonical / "sources").mkdir(mode=0o700)
        return real_expected_paths(lock)

    monkeypatch.setattr(
        source_lock_module,
        "_expected_paths",
        replace_namespace,
    )
    with pytest.raises(ValueError, match="canonical directory identity drift"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )
    assert list((canonical / "sources").iterdir()) == []
    assert not any(
        path.name.startswith(f".{fixture_source_lock.lock.sha256}.tmp-")
        for path in (displaced / "sources").iterdir()
    )
    quarantines = tuple(
        path
        for path in (displaced / "sources").iterdir()
        if path.name.startswith(".memorysplit-source-cleanup-")
    )
    assert len(quarantines) == 1
    assert quarantines[0].is_dir()
    with pytest.raises(ValueError, match="quarantine.*offline cleanup"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            displaced,
        )


def test_stage_reuse_detects_canonical_namespace_replacement(
    tmp_path,
    fixture_source_lock,
    monkeypatch,
):
    canonical = tmp_path / "canonical"
    displaced = tmp_path / "canonical-displaced"
    stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        canonical,
    )
    real_verify_fd = source_lock_module._verify_source_tree_fd
    calls = 0

    def replace_during_reuse(lock, root_fd, **kwargs):
        nonlocal calls
        if kwargs.get("_run_finemath_proof", True):
            calls += 1
        result = real_verify_fd(lock, root_fd, **kwargs)
        if calls == 2 and kwargs.get("_run_finemath_proof", True):
            canonical.rename(displaced)
            canonical.mkdir(mode=0o700)
            (canonical / "sources").mkdir(mode=0o700)
        return result

    monkeypatch.setattr(
        source_lock_module,
        "_verify_source_tree_fd",
        replace_during_reuse,
    )
    with pytest.raises(ValueError, match="canonical directory identity drift"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )
    assert list((canonical / "sources").iterdir()) == []


def _copy_concurrent_winner(canonical, stage_name, final_name, *, conflict):
    source = canonical / "sources" / stage_name
    winner = canonical / "sources" / final_name
    shutil.copytree(source, winner, copy_function=shutil.copy2)
    if conflict:
        victim = next(path for path in winner.rglob("*") if path.is_file())
        victim.write_bytes(victim.read_bytes() + b"conflict")


def test_matching_concurrent_winner_succeeds_and_preserves_loser(
    tmp_path,
    fixture_source_lock,
    monkeypatch,
):
    canonical = tmp_path / "canonical"
    installed = False

    def install_winner(
        phase,
        _sources_fd,
        stage_name,
        final_name,
        _stage_fd,
    ):
        nonlocal installed
        if phase == "before_publish" and not installed:
            installed = True
            _copy_concurrent_winner(
                canonical,
                stage_name,
                final_name,
                conflict=False,
            )

    monkeypatch.setattr(
        source_lock_module,
        "_stage_publish_hook",
        install_winner,
        raising=False,
    )
    winner = stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        canonical,
    )
    assert winner == canonical / "sources" / fixture_source_lock.lock.sha256
    duplicate_root = canonical / "duplicate-quarantines"
    quarantines = tuple(
        path
        for path in duplicate_root.iterdir()
        if path.is_dir()
    )
    markers = tuple(duplicate_root.glob("*.json"))
    assert len(quarantines) == 1
    assert len(markers) == 1
    assert verify_source_tree(
        fixture_source_lock.lock,
        quarantines[0],
    )["passed"]
    assert not any(
        path.name.startswith(".memorysplit-source-cleanup-")
        for path in (canonical / "sources").iterdir()
    )
    assert (
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )
        == winner
    )


def test_conflicting_concurrent_winner_fails_closed(
    tmp_path,
    fixture_source_lock,
    monkeypatch,
):
    canonical = tmp_path / "canonical"
    installed = False

    def install_winner(
        phase,
        _sources_fd,
        stage_name,
        final_name,
        _stage_fd,
    ):
        nonlocal installed
        if phase == "before_publish" and not installed:
            installed = True
            _copy_concurrent_winner(
                canonical,
                stage_name,
                final_name,
                conflict=True,
            )

    monkeypatch.setattr(
        source_lock_module,
        "_stage_publish_hook",
        install_winner,
        raising=False,
    )
    with pytest.raises(ValueError, match="conflicting source stage"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )
    assert (canonical / "sources" / fixture_source_lock.lock.sha256).is_dir()
    assert any(
        path.name.startswith(".memorysplit-source-cleanup-")
        for path in (canonical / "sources").iterdir()
    )


def _write_forged_duplicate(canonical, lock, *, mutate_final=False):
    winner = canonical / "sources" / lock.sha256
    if mutate_final:
        victim = next(path for path in winner.rglob("*") if path.is_file())
        victim.write_bytes(victim.read_bytes() + b"drift")
    duplicate_root = canonical / "duplicate-quarantines"
    duplicate_root.mkdir(mode=0o700)
    quarantine_name = ".memorysplit-benign-duplicate-forged"
    shutil.copytree(winner, duplicate_root / quarantine_name)
    (duplicate_root / f"{quarantine_name}.json").write_bytes(
        canonical_json_bytes(
            {
                "format": "memorysplit-source-benign-duplicate-v1",
                "quarantine_name": quarantine_name,
                "source_lock_sha256": "f" * 64,
                "winner_name": lock.sha256,
            }
        )
    )


def test_forged_duplicate_quarantine_never_bypasses_final_validation(
    tmp_path,
    fixture_source_lock,
):
    canonical = tmp_path / "canonical"
    stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        canonical,
    )
    _write_forged_duplicate(
        canonical,
        fixture_source_lock.lock,
        mutate_final=True,
    )
    with pytest.raises(ValueError, match="conflicting source stage"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )


def test_forged_duplicate_quarantine_fails_after_valid_final(
    tmp_path,
    fixture_source_lock,
):
    canonical = tmp_path / "canonical"
    stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        canonical,
    )
    _write_forged_duplicate(
        canonical,
        fixture_source_lock.lock,
    )
    with pytest.raises(ValueError, match="duplicate quarantine"):
        stage_source_lock(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
            canonical,
        )


def _offline_finemath_selection(
    tmp_path,
    *,
    fineweb_texts,
    four_plus_texts,
    three_plus_texts,
    quota,
):
    fineweb_paths = []
    text_by_path = {}
    for index, texts in enumerate(fineweb_texts):
        path = tmp_path / f"fineweb-{index}.parquet"
        path.write_bytes(b"fixture")
        fineweb_paths.append(path)
        text_by_path[path] = texts
    materialized = tmp_path / "finemath"
    downloads = []

    def materialize(relative):
        downloads.append(relative)
        path = materialized / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        text_by_path[path] = {
            **four_plus_texts,
            **three_plus_texts,
        }[relative]
        return path

    selection = source_lock_module._select_finemath_files(
        four_plus=tuple(reversed(tuple(four_plus_texts))),
        three_plus=tuple(reversed(tuple(three_plus_texts))),
        fineweb_paths=tuple(fineweb_paths),
        materialize=materialize,
        iter_texts=lambda path: iter(text_by_path[path]),
        encode=lambda text: list(text),
        quota=quota,
        database_path=tmp_path / "finemath-proof.sqlite3",
    )
    return selection, tuple(downloads)


def test_finemath_selection_is_nfc_deduplicated_ordered_and_stops_at_quota(
    tmp_path,
):
    four_a = "finemath-4plus/train-00000-of-00002.parquet"
    four_b = "finemath-4plus/train-00001-of-00002.parquet"
    three = "finemath-3plus/train-00000-of-00001.parquet"
    selection, downloads = _offline_finemath_selection(
        tmp_path,
        fineweb_texts=(("café",), (), ()),
        four_plus_texts={
            four_a: ("cafe\u0301", "x"),
            four_b: ("yy",),
        },
        three_plus_texts={three: ("unused",)},
        quota=4,
    )
    assert selection.selected_paths == (four_a, four_b)
    assert selection.usable_targets == 5
    assert selection.fineweb_duplicate_rows == 1
    assert downloads == (four_a, four_b)


def test_finemath_selection_requires_strictly_more_than_quota(
    tmp_path,
):
    four = "finemath-4plus/train-00000-of-00001.parquet"
    three = "finemath-3plus/train-00000-of-00001.parquet"
    selection, downloads = _offline_finemath_selection(
        tmp_path,
        fineweb_texts=((), (), ()),
        four_plus_texts={four: ("xx",)},
        three_plus_texts={three: ("z",)},
        quota=3,
    )
    assert selection.selected_paths == (four, three)
    assert selection.usable_targets == 5
    assert downloads == (four, three)


@pytest.mark.parametrize(
    ("license_value", "paths", "message"),
    [
        (
            "mit",
            ("README.md", "finemath-4plus/train-00000.parquet"),
            "ODC-By",
        ),
        (
            "odc-by",
            ("finemath-4plus/train-00000.parquet",),
            "inventory",
        ),
        (
            "odc-by",
            ("README.md", "finemath-3plus/train-00000.parquet"),
            "inventory",
        ),
        (
            "odc-by",
            ("README.md", "finemath-4plus/train-nested/evil.parquet"),
            "inventory",
        ),
    ],
)
def test_finemath_metadata_rejects_license_or_inventory_drift(
    license_value,
    paths,
    message,
):
    info = SimpleNamespace(
        card_data={"license": license_value},
        siblings=tuple(SimpleNamespace(rfilename=path) for path in paths),
    )
    with pytest.raises(ValueError, match=message):
        source_lock_module._validated_finemath_paths(info)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__(
                "dataset_contract_sha256",
                "b" * 64,
            ),
            "dataset contract",
        ),
        (
            lambda value: value["sources"][0].__setitem__(
                "repository",
                "https://example.invalid/review-bypass.git",
            ),
            "reviewed source catalog",
        ),
        (
            lambda value: value["sources"][0].__setitem__(
                "transport",
                "huggingface_dataset",
            ),
            "reviewed source catalog",
        ),
        (
            lambda value: value["sources"][0].__setitem__(
                "license_spdx",
                "MIT",
            ),
            "reviewed source catalog",
        ),
        (
            lambda value: value["sources"][0].__setitem__(
                "revision",
                "b" * 40,
            ),
            "fixed source identity",
        ),
    ],
)
def test_load_authority_rejects_canonical_reviewed_catalog_drift(
    tmp_path,
    valid_lock_json,
    mutation,
    message,
):
    mutation(valid_lock_json)
    with pytest.raises(ValueError, match=message):
        load_source_lock(
            _write_lock(tmp_path, valid_lock_json),
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_source_lock_binds_reviewed_catalog_commitment(fixture_source_lock):
    assert fixture_source_lock.lock.source_catalog_sha256 == (
        source_lock_module.reviewed_source_catalog_sha256()
    )


@pytest.mark.parametrize(
    ("source_id", "path"),
    [
        ("fineweb_edu", "sample/10BT/000_00000.parquet"),
        ("wikidata5m", "wikidata5m_alias.tar.gz"),
    ],
)
def test_load_authority_rejects_fixed_file_digest_drift(
    tmp_path,
    valid_lock_json,
    source_id,
    path,
):
    source = _source_json(valid_lock_json, source_id)
    row = next(item for item in source["files"] if item["path"] == path)
    row["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fixed source file identity"):
        load_source_lock(
            _write_lock(tmp_path, valid_lock_json),
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("materialized_path", "alternate/clrs"),
        ("license_files", ["README.md"]),
    ],
)
def test_load_authority_rejects_request_semantics_drift(
    tmp_path,
    valid_lock_json,
    field,
    value,
):
    _source_json(valid_lock_json, "clrs_text")[field] = value
    with pytest.raises(ValueError, match="reviewed source catalog"):
        load_source_lock(
            _write_lock(tmp_path, valid_lock_json),
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_load_authority_rejects_nested_finemath_train_spoof(
    tmp_path,
    valid_lock_json,
):
    source = _source_json(valid_lock_json, "finemath")
    row = next(
        item
        for item in source["files"]
        if item["path"].startswith("finemath-4plus/train-")
    )
    row["path"] = "finemath-4plus/train-nested/evil.parquet"
    source["files"].sort(key=lambda item: item["path"].encode())
    with pytest.raises(ValueError, match="FineMath source"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_finemath_lock_binds_exact_selection_proof(fixture_source_lock):
    entry = next(
        row
        for row in fixture_source_lock.lock.sources
        if row.source_id == "finemath"
    )
    proof = entry.finemath_selection
    assert proof is not None
    assert proof.algorithm == "nfc-fineweb-exact-dedup-gpt2-eot-v1"
    assert proof.quota == source_lock_module._FINEMATH_TARGETS
    assert proof.selected_paths == (
        "finemath-4plus/train-00000-of-00064.parquet",
    )
    assert proof.selected_paths == (
        *proof.four_plus_paths[: len(proof.selected_paths)],
    )
    assert len(proof.three_plus_paths) == 128


def test_finemath_selection_proof_requires_complete_fallback_inventory():
    four_plus = tuple(
        f"finemath-4plus/train-{index:05d}-of-00001.parquet"
        for index in range(1)
    )
    with pytest.raises(ValueError, match="3plus.*sequence"):
        source_lock_module.FineMathSelectionProof(
            algorithm="nfc-fineweb-exact-dedup-gpt2-eot-v1",
            quota=4,
            four_plus_paths=four_plus,
            three_plus_paths=(),
            selected_paths=(four_plus[0],),
            usable_targets=5,
            fineweb_duplicate_rows=0,
        )


@pytest.mark.parametrize(
    "extra_path",
    [
        "finemath-4plus/train-00001-of-00064.parquet",
        "finemath-3plus/train-00000-of-00128.parquet",
        "foreign/unreviewed.bin",
    ],
)
def test_load_rejects_every_extra_finemath_inventory_path(
    tmp_path,
    valid_lock_json,
    extra_path,
):
    source = _source_json(valid_lock_json, "finemath")
    source["files"].append(
        {
            "bytes": 1,
            "path": extra_path,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }
    )
    source["files"].sort(key=lambda item: item["path"].encode())
    with pytest.raises(ValueError, match="FineMath.*inventory"):
        load_source_lock(_write_lock(tmp_path, valid_lock_json))


def test_verify_recomputes_finemath_selection_proof(
    fixture_source_lock,
):
    original = fixture_source_lock.lock
    finemath = next(
        row for row in original.sources if row.source_id == "finemath"
    )
    forged = replace(
        original,
        sources=tuple(
            replace(
                finemath,
                finemath_selection=replace(
                    finemath.finemath_selection,
                    usable_targets=finemath.finemath_selection.usable_targets + 1,
                ),
            )
            if row.source_id == "finemath"
            else row
            for row in original.sources
        ),
    )
    with pytest.raises(ValueError, match="FineMath selection proof"):
        verify_source_tree(forged, fixture_source_lock.download_root)


def _write_file_at(directory_fd, name, payload):
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def test_verify_detects_inserted_then_removed_directory_entry(
    fixture_source_lock,
    monkeypatch,
):
    def mutate(phase, directory_fd, relative, _names):
        if relative:
            return
        if phase == "after_initial_snapshot":
            _write_file_at(directory_fd, "transient-race", b"race")
        elif phase == "before_final_snapshot":
            os.unlink("transient-race", dir_fd=directory_fd)

    monkeypatch.setattr(
        source_lock_module,
        "_directory_verification_hook",
        mutate,
        raising=False,
    )
    with pytest.raises(ValueError, match="directory entry race"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


def test_verify_detects_removed_and_replaced_entry(
    fixture_source_lock,
    monkeypatch,
):
    def mutate(phase, directory_fd, relative, names):
        if (
            phase == "after_initial_snapshot"
            and relative == "clrs_text/src"
            and "data.txt" in names
        ):
            descriptor = os.open("data.txt", os.O_RDONLY, dir_fd=directory_fd)
            try:
                payload = os.read(descriptor, 1 << 20)
            finally:
                os.close(descriptor)
            os.unlink("data.txt", dir_fd=directory_fd)
            _write_file_at(directory_fd, "data.txt", payload)

    monkeypatch.setattr(
        source_lock_module,
        "_directory_verification_hook",
        mutate,
        raising=False,
    )
    with pytest.raises(ValueError, match="directory entry race"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


def test_verify_detects_nested_insert_remove_race(
    fixture_source_lock,
    monkeypatch,
):
    def mutate(phase, directory_fd, relative, _names):
        if relative != "arc_agi_1/data/training":
            return
        if phase == "after_initial_snapshot":
            _write_file_at(directory_fd, "nested-race", b"race")
        elif phase == "before_final_snapshot":
            os.unlink("nested-race", dir_fd=directory_fd)

    monkeypatch.setattr(
        source_lock_module,
        "_directory_verification_hook",
        mutate,
        raising=False,
    )
    with pytest.raises(ValueError, match="directory entry race"):
        verify_source_tree(
            fixture_source_lock.lock,
            fixture_source_lock.download_root,
        )


def test_finemath_proof_detects_file_replacement(
    fixture_source_lock,
    monkeypatch,
):
    root = fixture_source_lock.download_root
    victim = (
        root
        / "finemath"
        / "finemath-4plus"
        / "train-00000-of-00064.parquet"
    )
    mutated = False

    def mutate(phase, _root_fd, _descriptors):
        nonlocal mutated
        if phase == "during_proof" and not mutated:
            mutated = True
            payload = victim.read_bytes()
            victim.unlink()
            victim.write_bytes(payload)

    monkeypatch.setattr(
        source_lock_module,
        "_finemath_proof_hook",
        mutate,
        raising=False,
    )
    with pytest.raises(ValueError, match="changed during FineMath proof"):
        verify_source_tree(
            fixture_source_lock.lock,
            root,
        )


def test_finemath_proof_detects_transient_entry(
    fixture_source_lock,
    monkeypatch,
):
    root = fixture_source_lock.download_root
    transient = root / "proof-transient"
    mutated = False

    def mutate(phase, _root_fd, _descriptors):
        nonlocal mutated
        if phase == "during_proof" and not mutated:
            mutated = True
            transient.write_bytes(b"race")
            transient.unlink()

    monkeypatch.setattr(
        source_lock_module,
        "_finemath_proof_hook",
        mutate,
        raising=False,
    )
    with pytest.raises(ValueError, match="changed during FineMath proof"):
        verify_source_tree(
            fixture_source_lock.lock,
            root,
        )


def test_finemath_proof_detects_directory_replacement(
    tmp_path,
    fixture_source_lock,
    monkeypatch,
):
    root = fixture_source_lock.download_root
    victim = root / "arc_agi_1" / "data" / "training"
    displaced = tmp_path / "training-displaced"
    mutated = False

    def mutate(phase, _root_fd, _descriptors):
        nonlocal mutated
        if phase == "during_proof" and not mutated:
            mutated = True
            victim.rename(displaced)
            victim.mkdir(mode=0o700)
            (victim / "a.json").write_bytes(
                (displaced / "a.json").read_bytes()
            )

    monkeypatch.setattr(
        source_lock_module,
        "_finemath_proof_hook",
        mutate,
        raising=False,
    )
    with pytest.raises(ValueError, match="changed during FineMath proof"):
        verify_source_tree(
            fixture_source_lock.lock,
            root,
        )


def _run_cleanup_with_hook(
    tmp_path,
    monkeypatch,
    setup,
    hook,
    *,
    expect_failure=False,
):
    parent = tmp_path / "cleanup-parent"
    stage = parent / "stage"
    stage.mkdir(parents=True, mode=0o700)
    setup(stage)
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    monkeypatch.setattr(
        source_lock_module,
        "_cleanup_quarantine_hook",
        hook,
        raising=False,
    )
    try:
        if expect_failure:
            with pytest.raises(ValueError, match="cleanup.*identity|restore"):
                source_lock_module._remove_owned_directory_at(
                    parent_fd,
                    "stage",
                    source_lock_module._namespace_identity(stage.lstat()),
                    description="cleanup test stage",
                )
            quarantine_name = None
        else:
            quarantine_name = source_lock_module._remove_owned_directory_at(
                parent_fd,
                "stage",
                source_lock_module._namespace_identity(stage.lstat()),
                description="cleanup test stage",
            )
    finally:
        os.close(parent_fd)
    return parent, quarantine_name


def test_cleanup_preserves_verified_stage_as_quarantine(tmp_path):
    parent = tmp_path / "cleanup-parent"
    stage = parent / "stage"
    stage.mkdir(parents=True, mode=0o700)
    (stage / "payload").write_bytes(b"payload")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        quarantine_name = source_lock_module._remove_owned_directory_at(
            parent_fd,
            "stage",
            source_lock_module._namespace_identity(stage.lstat()),
            description="cleanup test stage",
        )
    finally:
        os.close(parent_fd)
    assert not stage.exists()
    quarantine = parent / quarantine_name
    assert quarantine.is_dir()
    assert (quarantine / "payload").read_bytes() == b"payload"


def test_cleanup_never_unlinks_replacement_file(
    tmp_path,
    monkeypatch,
):
    def setup(stage):
        (stage / "victim").write_bytes(b"verified")

    def hook(phase, parent_fd, name, _quarantine, _descriptor, is_directory):
        if phase == "before_quarantine" and name == "stage" and is_directory:
            stage_fd = _descriptor
            os.rename(
                "victim",
                "original-preserved",
                src_dir_fd=stage_fd,
                dst_dir_fd=stage_fd,
            )
            _write_file_at(stage_fd, "victim", b"replacement")

    parent, quarantine_name = _run_cleanup_with_hook(
        tmp_path,
        monkeypatch,
        setup,
        hook,
    )
    quarantine = parent / quarantine_name
    assert (quarantine / "original-preserved").read_bytes() == b"verified"
    assert (quarantine / "victim").read_bytes() == b"replacement"


def test_cleanup_never_removes_replacement_directory(
    tmp_path,
    monkeypatch,
):
    def setup(stage):
        child = stage / "victim-dir"
        child.mkdir(mode=0o700)
        (child / "verified").write_bytes(b"verified")

    def hook(phase, parent_fd, name, _quarantine, _descriptor, is_directory):
        if phase == "before_quarantine" and name == "stage" and is_directory:
            stage_fd = _descriptor
            os.rename(
                "victim-dir",
                "original-preserved",
                src_dir_fd=stage_fd,
                dst_dir_fd=stage_fd,
            )
            os.mkdir("victim-dir", mode=0o700, dir_fd=stage_fd)
            replacement_fd = os.open(
                "victim-dir",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=stage_fd,
            )
            try:
                _write_file_at(replacement_fd, "replacement", b"replacement")
            finally:
                os.close(replacement_fd)

    parent, quarantine_name = _run_cleanup_with_hook(
        tmp_path,
        monkeypatch,
        setup,
        hook,
    )
    quarantine = parent / quarantine_name
    assert (
        quarantine / "original-preserved" / "verified"
    ).read_bytes() == b"verified"
    assert (
        quarantine / "victim-dir" / "replacement"
    ).read_bytes() == b"replacement"


def test_cleanup_restore_failure_preserves_quarantine_and_blocker(
    tmp_path,
    monkeypatch,
):
    state = {"replaced": False}

    def setup(stage):
        (stage / "victim").write_bytes(b"verified")

    def hook(phase, parent_fd, name, _quarantine, _descriptor, is_directory):
        if (
            phase == "before_quarantine"
            and name == "stage"
            and is_directory
        ):
            os.rename(
                "stage",
                "original-preserved",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir("stage", mode=0o700, dir_fd=parent_fd)
            replacement_fd = os.open(
                "stage",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent_fd,
            )
            try:
                _write_file_at(
                    replacement_fd,
                    "replacement",
                    b"replacement",
                )
            finally:
                os.close(replacement_fd)
            state["replaced"] = True
        elif (
            phase == "after_quarantine"
            and name == "stage"
            and state["replaced"]
        ):
            os.mkdir("stage", mode=0o700, dir_fd=parent_fd)
            blocker_fd = os.open(
                "stage",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent_fd,
            )
            try:
                _write_file_at(blocker_fd, "blocker", b"restore-blocker")
            finally:
                os.close(blocker_fd)

    parent, _quarantine_name = _run_cleanup_with_hook(
        tmp_path,
        monkeypatch,
        setup,
        hook,
        expect_failure=True,
    )
    assert (
        parent / "original-preserved" / "victim"
    ).read_bytes() == b"verified"
    assert (parent / "stage" / "blocker").read_bytes() == b"restore-blocker"
    quarantines = tuple(parent.glob(".memorysplit-source-cleanup-*"))
    assert len(quarantines) == 1
    assert (
        quarantines[0] / "replacement"
    ).read_bytes() == b"replacement"


def test_load_authority_requires_expected_generator_commit(
    tmp_path,
    valid_lock_json,
):
    path = _write_lock(tmp_path, valid_lock_json)
    with pytest.raises(ValueError, match="generator commit"):
        load_source_lock(path, expected_generator_commit="b" * 40)


@pytest.mark.parametrize("authority", ["verify", "stage"])
def test_tree_authorities_reject_caller_authored_catalog_drift(
    tmp_path,
    fixture_source_lock,
    authority,
):
    original = fixture_source_lock.lock
    source = original.sources[0]
    forged = replace(
        original,
        sources=(
            replace(
                source,
                repository="https://example.invalid/review-bypass.git",
            ),
            *original.sources[1:],
        ),
    )
    assert isinstance(forged, SourceLock)
    with pytest.raises(ValueError, match="reviewed source catalog"):
        if authority == "verify":
            verify_source_tree(
                forged,
                fixture_source_lock.download_root,
                expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
            )
        else:
            stage_source_lock(
                forged,
                fixture_source_lock.download_root,
                tmp_path / "canonical",
                expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
            )
