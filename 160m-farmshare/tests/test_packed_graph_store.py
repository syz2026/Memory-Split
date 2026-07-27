from __future__ import annotations

import hashlib
import inspect
import json
import mmap
from pathlib import Path

import pytest

import organizer.packed_graph_store as packed_module
from corpusgen.graph_records import GraphAddress, GraphRow
from corpusgen.relation_codec import RelationCodec
from organizer.graph_store import AtomicGraphStore
from organizer.packed_graph_store import PackedGraphStore


def _codec(*relation_ids: str) -> RelationCodec:
    return RelationCodec(relation_ids or ("P31", "P279", "P1476"))


def _rows() -> tuple[GraphRow, ...]:
    return (
        GraphRow(
            7,
            "P31",
            "out",
            "entity",
            "42",
            (("rank", "preferred"),),
            "wikidata:entity",
        ),
        GraphRow(
            9,
            "P1476",
            "in",
            "literal",
            "Café 東京",
            (("language", "fr"), ("source", "Q1")),
            "wikidata:literal",
        ),
    )


def _rewrite_file_hash(path: Path, name: str) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"][name] = hashlib.sha256(
        (path / name).read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("kind", ["atomic", "packed"])
def test_stores_share_exact_lookup_contract(tmp_path, kind):
    rows = _rows()
    if kind == "atomic":
        store = AtomicGraphStore(rows)
    else:
        store = PackedGraphStore.build(tmp_path / "store", rows, _codec())

    assert store.lookup(rows[0].address) == rows[0]
    assert store.lookup(GraphAddress(7, "P999", "out")) is None
    assert store.lookup(GraphAddress(999, "P31", "out")) is None


@pytest.mark.parametrize("kind", ["atomic", "packed"])
def test_stores_select_provenance_partition_and_bound_entity_ids(
    tmp_path,
    kind,
):
    rows = _rows()
    store = (
        AtomicGraphStore(rows)
        if kind == "atomic"
        else PackedGraphStore.build(tmp_path / "store", rows, _codec())
    )

    assert store.rows_for_provenance("wikidata:entity") == (rows[0],)
    assert store.rows_for_provenance("missing") == ()
    assert store.max_entity_id() == 42


def test_packed_store_round_trips_rows_and_reports_separate_file_bytes(
    tmp_path,
):
    rows = _rows()
    path = tmp_path / "store"
    store = PackedGraphStore.build(path, reversed(rows), _codec())
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    assert len(store) == 2
    assert store.lookup(rows[0].address) == rows[0]
    assert store.lookup(rows[1].address) == rows[1]
    assert store.snapshot_sha256() == AtomicGraphStore(rows).snapshot_sha256()
    assert store.stats().rows == 2
    assert store.stats().index_bytes == (path / "index.bin").stat().st_size
    assert store.stats().row_bytes == (path / "rows.bin").stat().st_size
    assert store.stats().blob_bytes == (path / "blobs.bin").stat().st_size
    assert store.stats().row_bytes > 0
    assert store.stats().blob_bytes > 0
    assert manifest["row_count"] == 2
    assert manifest["codec_sha256"] == _codec().sha256()
    assert manifest["hash_algorithm"] == "blake2b-64-memsplit-v1"
    assert manifest["row_count"] * 10 <= manifest["capacity"] * 7
    assert manifest["capacity"] & (manifest["capacity"] - 1) == 0
    assert set(path.iterdir()) == {
        path / "manifest.json",
        path / "index.bin",
        path / "rows.bin",
        path / "blobs.bin",
    }


def test_packed_store_build_has_no_public_hash_override():
    assert "hash_fn" not in inspect.signature(
        PackedGraphStore.build
    ).parameters


def test_packed_store_compares_full_keys_after_hash_collision(
    tmp_path,
    monkeypatch,
):
    rows = _rows()
    monkeypatch.setattr(packed_module, "_stable_hash", lambda _key: 1)
    store = PackedGraphStore.build(
        tmp_path / "store",
        rows,
        _codec(),
    )

    assert store.lookup(rows[0].address) == rows[0]
    assert store.lookup(rows[1].address) == rows[1]
    assert store.lookup(GraphAddress(999, "P31", "out")) is None


def test_packed_store_build_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    PackedGraphStore.build(first, _rows(), _codec()).close()
    PackedGraphStore.build(second, reversed(_rows()), _codec()).close()

    for name in ("manifest.json", "index.bin", "rows.bin", "blobs.bin"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_packed_store_load_uses_read_only_mmaps(tmp_path, monkeypatch):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()

    calls = []
    real_mmap = mmap.mmap

    def recording_mmap(fileno, length, *, access):
        calls.append((length, access))
        return real_mmap(fileno, length, access=access)

    monkeypatch.setattr(packed_module.mmap, "mmap", recording_mmap)
    store = PackedGraphStore.load(path, _codec())
    try:
        assert store.lookup(_rows()[0].address) == _rows()[0]
        assert calls == [(0, mmap.ACCESS_READ)] * 3
    finally:
        store.close()


def test_packed_store_rejects_duplicate_addresses_before_publication(tmp_path):
    path = tmp_path / "store"
    duplicate = GraphRow(
        7,
        "P31",
        "out",
        "entity",
        "99",
        (),
        "duplicate",
    )

    with pytest.raises(ValueError, match="duplicate graph address"):
        PackedGraphStore.build(path, (*_rows(), duplicate), _codec())
    assert not path.exists()
    assert not list(tmp_path.glob(".store.tmp-*"))


def test_packed_store_rejects_codec_hash_mismatch(tmp_path):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()

    with pytest.raises(ValueError, match="codec hash mismatch"):
        PackedGraphStore.load(path, _codec("P1476", "P31", "P279"))


def test_packed_store_rejects_hash_algorithm_mismatch(tmp_path):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hash_algorithm"] = "sha256"
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash algorithm"):
        PackedGraphStore.load(path, _codec())


def test_failed_post_publication_load_removes_destination_and_can_retry(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "store"
    real_load = PackedGraphStore.load
    calls = 0

    def fail_once(cls, candidate, codec):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected post-publication load failure")
        return real_load(candidate, codec)

    monkeypatch.setattr(PackedGraphStore, "load", classmethod(fail_once))

    with pytest.raises(RuntimeError, match="post-publication"):
        PackedGraphStore.build(path, _rows(), _codec())
    assert not path.exists()

    retried = PackedGraphStore.build(path, _rows(), _codec())
    assert retried.lookup(_rows()[0].address) == _rows()[0]


def test_packed_store_rejects_file_hash_mismatch(tmp_path):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()
    rows_path = path / "rows.bin"
    content = bytearray(rows_path.read_bytes())
    content[0] ^= 1
    rows_path.write_bytes(content)

    with pytest.raises(ValueError, match="file hash mismatch"):
        PackedGraphStore.load(path, _codec())


@pytest.mark.parametrize("name", ["index.bin", "rows.bin", "blobs.bin"])
def test_packed_store_rejects_truncated_files_with_matching_hash(
    tmp_path,
    name,
):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()
    file_path = path / name
    file_path.write_bytes(file_path.read_bytes()[:-1])
    _rewrite_file_hash(path, name)

    with pytest.raises(ValueError, match="invalid|truncated|bounds|size"):
        PackedGraphStore.load(path, _codec())


def test_packed_store_rejects_malformed_manifest(tmp_path):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()
    (path / "manifest.json").write_bytes(b"{")

    with pytest.raises(ValueError, match="manifest"):
        PackedGraphStore.load(path, _codec())


def test_packed_store_rejects_non_integer_schema_version(tmp_path):
    path = tmp_path / "store"
    PackedGraphStore.build(path, _rows(), _codec()).close()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1.0
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema version"):
        PackedGraphStore.load(path, _codec())


def test_packed_store_round_trips_empty_store_deterministically(tmp_path):
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first = PackedGraphStore.build(first_path, (), _codec())
    second = PackedGraphStore.build(second_path, (), _codec())

    assert len(first) == 0
    assert first.lookup(GraphAddress(7, "P31", "out")) is None
    assert first.stats().rows == 0
    assert first.stats().row_bytes == 0
    assert first.stats().blob_bytes == 0
    assert first.stats().index_bytes > 0
    assert first.snapshot_sha256() == AtomicGraphStore().snapshot_sha256()
    for name in ("manifest.json", "index.bin", "rows.bin", "blobs.bin"):
        assert (first_path / name).read_bytes() == (
            second_path / name
        ).read_bytes()
