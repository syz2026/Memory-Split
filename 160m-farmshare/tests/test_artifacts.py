from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from experiment.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_stream,
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
    sha256_file,
)


def test_canonical_json_is_sorted_compact_utf8_and_newline_terminated():
    value = {"z": [3, {"é": True}], "a": 1}

    encoded = canonical_json_bytes(value)

    assert encoded == '{"a":1,"z":[3,{"é":true}]}\n'.encode()
    assert canonical_sha256(value) == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string-key"},
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": Path("not-json")},
    ],
)
def test_canonical_json_rejects_noncanonical_values(value):
    with pytest.raises(ValueError, match="canonical|finite|string"):
        canonical_json_bytes(value)


def test_atomic_writes_use_same_directory_replace_and_fsync_parent(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.json"
    replacements: list[tuple[Path, Path]] = []
    fsync_calls: list[int] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def recording_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)

    def recording_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "replace", recording_replace)
    monkeypatch.setattr(os, "fsync", recording_fsync)

    atomic_write_json(destination, {"value": 1})
    atomic_write_json(destination, {"value": 2})

    assert destination.read_bytes() == b'{"value":2}\n'
    assert len(replacements) == 2
    assert all(source.parent == destination.parent for source, _ in replacements)
    assert all(target == destination for _, target in replacements)
    assert len(fsync_calls) >= 4
    assert not tuple(tmp_path.glob(f".{destination.name}.*"))


def test_atomic_stream_writer_publishes_without_buffering_whole_artifact(
    tmp_path,
):
    destination = tmp_path / "checkpoint.pt"
    writes: list[int] = []

    def write_chunks(stream):
        for chunk in (b"first", b"-", b"second"):
            writes.append(stream.write(chunk))

    atomic_write_stream(destination, write_chunks)

    assert destination.read_bytes() == b"first-second"
    assert writes == [5, 1, 6]
    assert not tuple(tmp_path.glob(f".{destination.name}.*"))


def test_atomic_stream_failure_preserves_previous_artifact(tmp_path):
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"complete")

    def fail_after_partial_write(stream):
        stream.write(b"partial")
        raise RuntimeError("simulated serialization failure")

    with pytest.raises(RuntimeError, match="serialization"):
        atomic_write_stream(destination, fail_after_partial_write)

    assert destination.read_bytes() == b"complete"
    assert not tuple(tmp_path.glob(f".{destination.name}.*"))


def test_canonical_loader_rejects_pretty_json_and_hash_drift(tmp_path):
    path = tmp_path / "value.json"
    atomic_write_json(path, {"a": 1})
    expected_hash = sha256_file(path)
    assert load_canonical_json(path, expected_sha256=expected_hash) == {"a": 1}

    path.write_text(json.dumps({"a": 1}, indent=2) + "\n")
    with pytest.raises(ValueError, match="canonical"):
        load_canonical_json(path)
    with pytest.raises(ValueError, match="SHA-256|hash"):
        load_canonical_json(path, expected_sha256=expected_hash)


def test_artifact_paths_reject_traversal_symlinks_and_nonregular_inputs(
    tmp_path,
):
    target = tmp_path / "target.bin"
    atomic_write_bytes(target, b"payload")
    link = tmp_path / "link.bin"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink|regular"):
        sha256_file(link)
    with pytest.raises(ValueError, match="traversal"):
        atomic_write_bytes(tmp_path / "child" / ".." / "escape.bin", b"x")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|canonical"):
        atomic_write_bytes(linked_parent / "value.bin", b"x")

    with pytest.raises(ValueError, match="symlink|regular"):
        atomic_write_bytes(link, b"replacement")
