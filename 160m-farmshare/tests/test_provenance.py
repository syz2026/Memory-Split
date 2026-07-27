from __future__ import annotations

import json
import subprocess

import pytest

from experiment.provenance import (
    SourceProvenance,
    collect_source_provenance,
    tracked_source_tree_sha256,
    verify_source_provenance,
)


def _git(root, *args):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Task Nine Tests")
    (root / "module.py").write_text("VALUE = 1\n")
    (root / "source.lock.json").write_text('{"revision":"pinned"}\n')
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


def test_collect_provenance_binds_clean_revision_tree_and_artifacts(tmp_path):
    root = _clean_repository(tmp_path)
    provenance = collect_source_provenance(
        root,
        artifact_paths={"source_lock": root / "source.lock.json"},
    )

    assert provenance.clean_tree is True
    assert provenance.git_revision == _git(root, "rev-parse", "HEAD")
    assert provenance.source_tree_sha256 == tracked_source_tree_sha256(root)
    assert set(provenance.artifact_sha256) == {"source_lock"}
    assert provenance.python_version
    assert provenance.platform
    assert SourceProvenance.from_dict(provenance.to_dict()) == provenance

    serialized = json.dumps(provenance.to_dict(), sort_keys=True)
    assert str(root) not in serialized


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_real_provenance_rejects_dirty_tree_including_untracked(
    tmp_path,
    dirty_kind,
):
    root = _clean_repository(tmp_path)
    if dirty_kind == "tracked":
        (root / "module.py").write_text("VALUE = 2\n")
    else:
        (root / "untracked.txt").write_text("not committed\n")

    with pytest.raises(ValueError, match="clean|dirty"):
        collect_source_provenance(
            root,
            artifact_paths={"source_lock": root / "source.lock.json"},
        )

    diagnostic = collect_source_provenance(
        root,
        artifact_paths={"source_lock": root / "source.lock.json"},
        require_clean=False,
    )
    assert diagnostic.clean_tree is False


def test_tracked_tree_hash_changes_with_tracked_bytes_not_untracked(tmp_path):
    root = _clean_repository(tmp_path)
    before = tracked_source_tree_sha256(root)
    (root / "untracked.txt").write_text("ignored by tracked digest\n")
    assert tracked_source_tree_sha256(root) == before

    (root / "module.py").write_text("VALUE = 3\n")
    assert tracked_source_tree_sha256(root) != before


def test_verify_source_provenance_rejects_revision_or_tree_drift(tmp_path):
    root = _clean_repository(tmp_path)
    provenance = collect_source_provenance(
        root,
        artifact_paths={"source_lock": root / "source.lock.json"},
    )
    assert verify_source_provenance(root, provenance) == provenance

    (root / "late-untracked.txt").write_text("drift\n")
    with pytest.raises(ValueError, match="clean|untracked|dirty"):
        verify_source_provenance(root, provenance, require_clean=True)
    (root / "late-untracked.txt").unlink()

    (root / "module.py").write_text("VALUE = 9\n")
    with pytest.raises(ValueError, match="source tree|revision|drift"):
        verify_source_provenance(root, provenance)


def test_provenance_rejects_symlinked_or_traversing_artifact_input(tmp_path):
    root = _clean_repository(tmp_path)
    target = root / "source.lock.json"
    link = root / "linked.lock.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|canonical|regular"):
        collect_source_provenance(
            root,
            artifact_paths={"source_lock": link},
            require_clean=False,
        )
    with pytest.raises(ValueError, match="traversal"):
        collect_source_provenance(
            root,
            artifact_paths={
                "source_lock": root / "nested" / ".." / "source.lock.json"
            },
            require_clean=False,
        )


def test_source_provenance_rejects_invalid_hashes_and_unknown_fields(tmp_path):
    root = _clean_repository(tmp_path)
    raw = collect_source_provenance(
        root,
        artifact_paths={"source_lock": root / "source.lock.json"},
    ).to_dict()
    raw["source_tree_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        SourceProvenance.from_dict(raw)

    raw = collect_source_provenance(
        root,
        artifact_paths={"source_lock": root / "source.lock.json"},
    ).to_dict()
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        SourceProvenance.from_dict(raw)
