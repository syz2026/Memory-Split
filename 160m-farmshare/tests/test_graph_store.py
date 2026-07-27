import pytest

from corpusgen.graph_records import GraphAddress, GraphRow
from organizer.graph_store import AtomicGraphStore, GraphStore, StoreStats


def row(source=1, relation="r0", target="2"):
    return GraphRow(
        source_id=source,
        relation_id=relation,
        direction="out",
        target_kind="entity",
        target=target,
        qualifiers=(("compose", "1"),),
        provenance_id="world-0",
    )


def test_atomic_lookup_returns_one_exact_row():
    store = AtomicGraphStore([row()])
    assert store.lookup(GraphAddress(1, "r0", "out")) == row()
    assert store.hits == 1 and store.misses == 0


def test_duplicate_functional_address_is_rejected():
    store = AtomicGraphStore([row()])
    with pytest.raises(ValueError, match="duplicate graph address"):
        store.add(row(target="3"))


def test_missing_address_returns_none():
    store = AtomicGraphStore([row()])
    assert store.lookup(GraphAddress(2, "r0", "out")) is None
    assert store.hits == 0 and store.misses == 1


def test_snapshot_round_trip_is_sorted_and_hash_stable(tmp_path):
    first = row(source=2, relation="r1", target="5")
    second = row(source=1, relation="r0", target="2")
    store = AtomicGraphStore([first, second])
    path = tmp_path / "graph.jsonl"
    store.save(path)
    loaded = AtomicGraphStore.load(path)
    assert loaded.rows() == (second, first)
    assert loaded.snapshot_sha256() == store.snapshot_sha256()


def test_atomic_store_satisfies_public_graph_store_contract():
    store: GraphStore = AtomicGraphStore([row()])

    assert len(store) == 1
    assert store.lookup(row().address) == row()
    assert store.stats() == StoreStats(
        rows=1,
        index_bytes=0,
        row_bytes=len(store.canonical_bytes()),
        blob_bytes=0,
    )
