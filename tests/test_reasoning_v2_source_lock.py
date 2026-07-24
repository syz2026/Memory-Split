from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import (
    PublicSourceResolver,
    SourceRequest,
    load_source_lock,
    resolve_source_lock,
    stage_source_lock,
    verify_source_tree,
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
        assert all(row.bytes > 0 and len(row.sha256) == 64 for row in entry.files)
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
    with pytest.raises(ValueError, match="positive integer"):
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

    monkeypatch.setattr(source_lock_module, "HfApi", lambda *, token: object())
    entry = PublicSourceResolver().resolve(
        SourceRequest(
            "local_fixture",
            "git",
            str(repository),
            required_license_paths=("LICENSE",),
        ),
        tmp_path / "downloads",
    )

    assert entry.revision == commit
    assert entry.license_spdx == "Apache-2.0"
    assert {row.path for row in entry.files} == {"LICENSE", "data.txt"}
    materialized = tmp_path / "downloads" / entry.materialized_path
    assert (materialized / "data.txt").read_text() == "immutable bytes\n"
