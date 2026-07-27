import json
from dataclasses import replace

import pytest

from corpusgen.bed_snapshot import (
    BedSnapshotLock,
    SourceDriftError,
    iter_verified_bed,
    lock_bed_snapshot,
)
from scripts.lock_bed_snapshot import main as lock_bed_main


def _write_bed_jsonl(tmp_path, texts):
    path = tmp_path / "bed.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {"text": text},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for text in texts
        )
    )
    return path


def _lock(path):
    return lock_bed_snapshot(
        path,
        repo_id="HuggingFaceFW/fineweb-edu",
        revision="a" * 40,
        config="sample-10BT",
        split="train",
    )


def test_bed_snapshot_records_source_identity_rows_bytes_and_hash(tmp_path):
    path = _write_bed_jsonl(tmp_path, ["one", "twö"])

    lock = _lock(path)

    assert lock.repo_id == "HuggingFaceFW/fineweb-edu"
    assert lock.revision == "a" * 40
    assert lock.config == "sample-10BT"
    assert lock.split == "train"
    assert lock.rows == 2
    assert lock.bytes == len(path.read_bytes())
    assert len(lock.sha256) == 64
    assert list(iter_verified_bed(path, lock)) == ["one", "twö"]


def test_bed_snapshot_rejects_drift_before_yielding_text(tmp_path):
    path = _write_bed_jsonl(tmp_path, ["one", "two"])
    lock = _lock(path)
    path.write_text(path.read_text() + '{"text":"three"}\n')

    rows = iter_verified_bed(path, lock)

    with pytest.raises(SourceDriftError):
        next(rows)


def test_bed_snapshot_rejects_row_count_drift_before_yielding_text(tmp_path):
    path = _write_bed_jsonl(tmp_path, ["one", "two"])
    lock = replace(_lock(path), rows=3)

    rows = iter_verified_bed(path, lock)

    with pytest.raises(SourceDriftError, match="row count"):
        next(rows)


@pytest.mark.parametrize(
    "line",
    [
        "not-json\n",
        "[]\n",
        "{}\n",
        '{"text":""}\n',
        '{"text":"   "}\n',
        '{"text":1}\n',
    ],
)
def test_bed_snapshot_lock_rejects_malformed_or_empty_text_rows(tmp_path, line):
    path = tmp_path / "bed.jsonl"
    path.write_text(line)

    with pytest.raises(ValueError, match=r"bed.jsonl:1"):
        _lock(path)


def test_bed_snapshot_accepts_blank_lines_but_counts_only_text_rows(tmp_path):
    path = tmp_path / "bed.jsonl"
    path.write_text('\n{"text":"one","source":"fixture"}\n  \n{"text":"two"}\n')

    lock = _lock(path)

    assert lock.rows == 2
    assert list(iter_verified_bed(path, lock)) == ["one", "two"]


@pytest.mark.parametrize("revision", ["", "a" * 39, "g" * 40, "A" * 40])
def test_bed_snapshot_requires_a_lowercase_forty_hex_revision(
    tmp_path, revision
):
    path = _write_bed_jsonl(tmp_path, ["one"])

    with pytest.raises(ValueError, match="40-hex revision"):
        lock_bed_snapshot(
            path,
            repo_id="HuggingFaceFW/fineweb-edu",
            revision=revision,
            config="sample-10BT",
            split="train",
        )


def test_bed_snapshot_manifest_bytes_are_canonical_and_round_trip(tmp_path):
    path = _write_bed_jsonl(tmp_path, ["one"])
    lock = _lock(path)
    expected = (
        json.dumps(
            lock.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    manifest_path = tmp_path / "bed.lock.json"

    assert lock.canonical_bytes() == expected
    lock.write(manifest_path)
    assert manifest_path.read_bytes() == expected
    assert BedSnapshotLock.from_path(manifest_path) == lock


def test_lock_bed_snapshot_script_writes_a_verified_local_manifest(tmp_path):
    path = _write_bed_jsonl(tmp_path, ["one", "two"])
    out = tmp_path / "bed.lock.json"

    status = lock_bed_main(
        [
            str(path),
            "--repo-id",
            "HuggingFaceFW/fineweb-edu",
            "--revision",
            "a" * 40,
            "--config",
            "sample-10BT",
            "--split",
            "train",
            "--out",
            str(out),
        ]
    )

    assert status == 0
    assert BedSnapshotLock.from_path(out) == _lock(path)
