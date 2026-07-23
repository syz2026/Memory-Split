import pytest

from corpusgen.graph_records import GraphAddress, GraphRow
from organizer.graph_store import AtomicGraphStore


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


def test_page_is_part_of_the_exact_graph_address():
    first = GraphRow(
        "Q1",
        "P31",
        "out",
        "entity",
        "Q2",
        (),
        "wikidata:first",
        page=0,
        targets=("Q2", "Q3"),
    )
    second = GraphRow(
        "Q1",
        "P31",
        "out",
        "entity",
        "Q4",
        (),
        "wikidata:second",
        page=1,
        targets=("Q4", "Q5"),
    )
    store = AtomicGraphStore([second, first])

    assert store.lookup(GraphAddress("Q1", "P31", "out", 0)) == first
    assert store.lookup(GraphAddress("Q1", "P31", "out", 1)) == second
    assert store.pages(GraphAddress("Q1", "P31", "out")) == (first, second)
    assert store.rows() == (first, second)


def test_set_valued_graph_rows_round_trip_without_scalar_api_breakage(tmp_path):
    paged = GraphRow(
        "Q9",
        "P999999",
        "out",
        "entity",
        "Q10",
        (),
        "wikidata:set",
        page=3,
        targets=("Q10", "Q20"),
    )
    scalar = row()
    path = tmp_path / "paged.jsonl"

    AtomicGraphStore([paged, scalar]).save(path)
    loaded = AtomicGraphStore.load(path)

    assert loaded.lookup(paged.address) == paged
    assert loaded.lookup(GraphAddress(1, "r0", "out")) == scalar
    assert scalar.address.page == 0
    assert scalar.values == ("2",)
