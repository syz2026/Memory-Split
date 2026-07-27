from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from corpusgen.graph_records import GraphAddress, GraphRow, stable_fact_id
from corpusgen.relation_codec import RelationCodec
from corpusgen.relation_schema import (
    LITERAL_RELATIONS,
    RelationSchema,
    RelationSpec,
)
from corpusgen.records import QAItem
from corpusgen.srgm_worlds import WorldConfig, generate_eval_pairs, generate_world
from corpusgen.world_splits import build_split_plan, composition_hash
from evals.relational_controls import (
    ControlID,
    EvalMode,
    build_control_view,
    oracle_answer,
)
from organizer.graph_store import AtomicGraphStore
from organizer.packed_graph_store import PackedGraphStore


def test_control_and_mode_enums_are_exact():
    assert {control.value for control in ControlID} == {
        "correct",
        "shuffled_returns",
        "relevant_edge",
        "irrelevant_edge",
        "gold_path",
        "gold_returns",
        "no_query",
        "explicit_miss",
        "handle_swap",
        "entity_rename",
        "graph_isomorphism",
    }
    assert {mode.value for mode in EvalMode} == {
        "memory_off",
        "memory_on",
    }


def _action(
    source_slot: int,
    relation_id: str,
    *,
    read: bool,
    halt: bool = False,
) -> dict:
    return {
        "source_slot": source_slot,
        "relation_id": relation_id,
        "direction": "out",
        "read": read,
        "halt": halt,
    }


def _fixture() -> tuple[QAItem, AtomicGraphStore]:
    rows = [
        GraphRow(1, "r0", "out", "entity", "2", (("compose", "1"),), "world-7"),
        GraphRow(2, "r1", "out", "entity", "3", (("compose", "1"),), "world-7"),
        GraphRow(4, "r0", "out", "entity", "5", (("compose", "2"),), "world-7"),
        GraphRow(5, "r1", "out", "entity", "4", (("compose", "3"),), "world-7"),
        GraphRow(1, "date", "out", "literal", "2001-01-01", (), "world-7"),
        GraphRow(2, "date", "out", "literal", "2002-01-01", (), "world-7"),
    ]
    item = QAItem(
        qid="pair-0-o",
        task="path_composition",
        prompt=(
            "Slot 0 refers to Q1. Start at slot 0 and follow r0 then r1. "
            "Return the composed relation."
        ),
        answer="r2",
        meta={
            "pair_id": "pair-0",
            "variant": "original",
            "template_id": "path_composition:v1",
            "world_id": 7,
            "provenance_id": "world-7",
            "graph_rows": 6,
            "relation_path_hash": composition_hash(("r0", "r1")),
            "composition_split": "seen",
            "hop_count": 2,
            "relations": ["r0", "r1"],
            "entity_slots": [1, None, None, None],
            "gold_addresses": [[1, "r0", "out"], [2, "r1", "out"]],
            "gold_fact_ids": ["f0", "f1"],
            "gold_actions": [
                _action(0, "r0", read=True),
                _action(0, "r1", read=True),
                _action(0, "r0", read=False, halt=True),
                *[_action(0, "r0", read=False) for _ in range(3)],
            ],
            "answer_choices": ["r0", "r1", "r2", "r3"],
            "changed_row": None,
        },
    )
    return item, AtomicGraphStore(rows)


def test_relevant_and_irrelevant_edits_have_opposite_oracle_effects():
    item, store = _fixture()

    relevant = build_control_view(
        item, store, ControlID.RELEVANT_EDGE, seed=19
    )
    irrelevant = build_control_view(
        item, store, ControlID.IRRELEVANT_EDGE, seed=19
    )

    assert oracle_answer(item, relevant.store) != item.answer
    assert relevant.oracle_effect == "changed"
    assert len(relevant.changed_addresses) == 1
    assert relevant.changed_addresses[0] in {
        GraphAddress(1, "r0", "out"),
        GraphAddress(2, "r1", "out"),
    }
    assert oracle_answer(item, irrelevant.store) == item.answer
    assert irrelevant.oracle_effect == "unchanged"
    assert len(irrelevant.changed_addresses) == 1
    assert irrelevant.changed_addresses[0] not in {
        GraphAddress(1, "r0", "out"),
        GraphAddress(2, "r1", "out"),
    }
    assert oracle_answer(item, store) == item.answer


def test_control_view_replays_transformed_dynamic_oracle_trace():
    rows = [
        GraphRow(1, "r0", "out", "entity", "2", (("compose", "0"),), "world-7"),
        GraphRow(2, "r1", "out", "entity", "3", (("compose", "0"),), "world-7"),
        GraphRow(4, "r0", "out", "entity", "5", (("compose", "1"),), "world-7"),
        GraphRow(5, "r1", "out", "entity", "6", (("compose", "2"),), "world-7"),
    ]
    item, _ = _fixture()
    item.answer = "r0"
    item.meta["graph_rows"] = len(rows)
    item.meta["gold_fact_ids"] = [stable_fact_id(row) for row in rows[:2]]
    store = AtomicGraphStore(rows)

    transformed = None
    for seed in range(512):
        candidate = build_control_view(
            item,
            store,
            ControlID.SHUFFLED_RETURNS,
            seed=seed,
        )
        if (
            candidate.oracle_after is not None
            and candidate.oracle_addresses[0] == GraphAddress(1, "r0", "out")
            and candidate.oracle_addresses[1].source_id != 2
        ):
            transformed = candidate
            break

    assert transformed is not None
    assert transformed.oracle_rows[0].target_kind == "entity"
    assert transformed.oracle_addresses[1].source_id == int(
        transformed.oracle_rows[0].target
    )
    assert transformed.item.meta["gold_addresses"] == [
        [
            address.source_id,
            address.relation_id,
            address.direction,
        ]
        for address in transformed.oracle_addresses
    ]
    assert transformed.item.meta["gold_fact_ids"] == [
        stable_fact_id(row) for row in transformed.oracle_rows
    ]


def _schema_fixture() -> RelationSchema:
    entity = tuple(
        RelationSpec(
            relation_id=f"P{index}",
            aliases=(f"relation {index}",),
            target_kind="entity",
            support=84,
            distinct_subjects=80,
            distinct_objects=40,
            entity_count=100,
        )
        for index in range(1, 33)
    )
    literals = tuple(
        RelationSpec(relation_id, aliases, target_kind)
        for relation_id, aliases, target_kind in LITERAL_RELATIONS
    )
    return RelationSchema(
        catalog=entity + literals,
        path_relation_ids=tuple(spec.relation_id for spec in entity + literals),
    )


def test_namespaced_schema_item_resolves_packed_store_partition(tmp_path):
    schema = _schema_fixture()
    plan = build_split_plan(schema, seed=17)
    partition = plan.protected_seen
    world = generate_world(
        0,
        WorldConfig(
            n_entities=100,
            seed=17,
            schema=schema,
            split_plan=plan,
            split_name="protected_seen",
            entity_id_offset=partition.entity_id_range[0],
            world_seed_offset=partition.world_seed_range[0],
        ),
    )
    item = generate_eval_pairs(world, n_pairs_per_task=1, seed=23)[0].original
    expected_provenance = (
        f"{partition.namespace}:world:{partition.world_seed_range[0]}"
    )
    store = PackedGraphStore.build(
        tmp_path / "graph.store",
        (fact.row for fact in world.facts),
        RelationCodec(schema.codec_catalog),
    )
    try:
        view = build_control_view(item, store, ControlID.CORRECT, seed=5)
    finally:
        store.close()

    assert item.meta["provenance_id"] == expected_provenance
    assert view.provenance_id == expected_provenance
    assert len(view.oracle_rows) == item.meta["hop_count"]


def test_shuffled_returns_are_a_seeded_within_kind_derangement():
    item, store = _fixture()
    cache = {}
    first = build_control_view(
        item,
        store,
        ControlID.SHUFFLED_RETURNS,
        seed=91,
        transformation_cache=cache,
    )
    second = build_control_view(
        item,
        store,
        ControlID.SHUFFLED_RETURNS,
        seed=91,
        transformation_cache=cache,
    )
    changed_seed = build_control_view(
        item, store, ControlID.SHUFFLED_RETURNS, seed=92
    )

    assert first.return_sources == second.return_sources
    assert first.store is second.store
    assert first.fingerprint() == second.fingerprint()
    assert all(source != donor for source, donor in first.return_sources)
    for source, donor in first.return_sources:
        assert store.lookup(source).target_kind == store.lookup(donor).target_kind
    assert first.return_sources != changed_seed.return_sources
    assert set(first.changed_addresses) == {
        row.address for row in store.rows()
    }
    record = first.transformation_record()
    assert first.transformation_id == second.transformation_id
    assert record == second.transformation_record()
    assert record["transformation_id"] == first.transformation_id
    assert record["changed_address_count"] == len(store.rows())
    assert "changed_addresses" not in record
    assert "return_sources" not in record


def test_shuffled_returns_fail_closed_on_single_kind_undersupply():
    item, store = _fixture()
    item.meta["graph_rows"] = 5
    undersupplied = AtomicGraphStore(
        [
            row
            for row in store.rows()
            if not (
                row.target_kind == "literal"
                and row.source_id == 2
            )
        ]
    )

    with pytest.raises(ValueError, match="derangement.*literal"):
        build_control_view(
            item,
            undersupplied,
            ControlID.SHUFFLED_RETURNS,
            seed=1,
        )


def test_shuffled_returns_fail_closed_without_payload_derangement():
    item, store = _fixture()
    rows = [
        (
            GraphRow(
                row.source_id,
                row.relation_id,
                row.direction,
                row.target_kind,
                "same",
                (),
                "world-7",
            )
            if row.target_kind == "literal"
            else row
        )
        for row in store.rows()
    ]
    with pytest.raises(ValueError, match="payload derangement.*literal"):
        build_control_view(
            item,
            AtomicGraphStore(rows),
            ControlID.SHUFFLED_RETURNS,
            seed=3,
        )


def _duplicate_heavy_shuffle_fixture(
    first_count: int,
    second_count: int,
) -> tuple[QAItem, list[GraphRow]]:
    item, _ = _fixture()
    rows = [
        GraphRow(
            1 if index == 0 else 10_000 + index,
            "r0" if index == 0 else "ra",
            "out",
            "entity",
            "2",
            (("compose", "1"),),
            "world-7",
        )
        for index in range(first_count)
    ]
    rows.extend(
        GraphRow(
            2 if index == 0 else 20_000 + index,
            "r1" if index == 0 else "rb",
            "out",
            "entity",
            "3",
            (("compose", "1"),),
            "world-7",
        )
        for index in range(second_count)
    )
    item.meta["graph_rows"] = len(rows)
    item.meta["gold_fact_ids"] = [
        stable_fact_id(rows[0]),
        stable_fact_id(rows[first_count]),
    ]
    return item, rows


def test_duplicate_heavy_shuffle_is_iterative_and_linearly_bounded(
    monkeypatch,
):
    import evals.relational_controls as controls

    item, rows = _duplicate_heavy_shuffle_fixture(550, 550)
    calls = 0
    original = controls._row_payload

    def counted(row):
        nonlocal calls
        calls += 1
        if calls > 8 * len(rows):
            pytest.fail("shuffle exceeded its payload-operation budget")
        return original(row)

    monkeypatch.setattr(controls, "_row_payload", counted)
    view = build_control_view(
        item,
        AtomicGraphStore(rows),
        ControlID.SHUFFLED_RETURNS,
        seed=41,
    )

    assert calls <= 8 * len(rows)
    assert len(view.return_sources) == len(rows)
    assert len({donor for _, donor in view.return_sources}) == len(rows)
    by_address = {row.address: row for row in rows}
    assert all(
        original(by_address[recipient]) != original(by_address[donor])
        for recipient, donor in view.return_sources
    )


@pytest.mark.parametrize(
    ("first_count", "second_count", "feasible"),
    [(3, 3, True), (3, 2, False), (2, 2, True)],
)
def test_duplicate_payload_derangement_feasibility_boundary(
    first_count,
    second_count,
    feasible,
):
    item, rows = _duplicate_heavy_shuffle_fixture(
        first_count,
        second_count,
    )
    store = AtomicGraphStore(rows)
    if feasible:
        view = build_control_view(
            item,
            store,
            ControlID.SHUFFLED_RETURNS,
            seed=17,
        )
        assert len(view.return_sources) == len(rows)
    else:
        with pytest.raises(ValueError, match="no true payload derangement"):
            build_control_view(
                item,
                store,
                ControlID.SHUFFLED_RETURNS,
                seed=17,
            )


def test_schema_scale_packed_store_shuffle_completes(tmp_path):
    item, rows = _duplicate_heavy_shuffle_fixture(600, 600)
    store = PackedGraphStore.build(
        tmp_path / "schema-scale.store",
        rows,
        RelationCodec(("r0", "r1", "ra", "rb")),
    )
    try:
        view = build_control_view(
            item,
            store,
            ControlID.SHUFFLED_RETURNS,
            seed=99,
        )
    finally:
        store.close()

    assert len(view.return_sources) == 1_200
    assert view.transformation_record()["changed_address_count"] == 1_200


def test_shuffle_cache_reuses_exact_overlay_without_crossing_counterfactuals():
    import evals.relational_controls as controls

    item, store = _fixture()
    address = GraphAddress(4, "r0", "out")
    original = store.lookup(address)
    assert original is not None
    first_overlay = controls._PatchStore(
        store,
        replacements={
            address: GraphRow(
                original.source_id,
                original.relation_id,
                original.direction,
                original.target_kind,
                "6",
                original.qualifiers,
                original.provenance_id,
            )
        },
    )
    second_overlay = controls._PatchStore(
        store,
        replacements={
            address: GraphRow(
                original.source_id,
                original.relation_id,
                original.direction,
                original.target_kind,
                "7",
                original.qualifiers,
                original.provenance_id,
            )
        },
    )
    cache = {}
    first = build_control_view(
        item,
        first_overlay,
        ControlID.SHUFFLED_RETURNS,
        seed=11,
        transformation_cache=cache,
    )
    repeated = build_control_view(
        item,
        first_overlay,
        ControlID.SHUFFLED_RETURNS,
        seed=11,
        transformation_cache=cache,
    )
    distinct = build_control_view(
        item,
        second_overlay,
        ControlID.SHUFFLED_RETURNS,
        seed=11,
        transformation_cache=cache,
    )

    assert first.store is repeated.store
    assert first.transformation_id == repeated.transformation_id
    assert distinct.store is not first.store
    assert distinct.transformation_id != first.transformation_id


def test_prompt_controls_only_change_the_prompt():
    item, store = _fixture()
    original_meta = json.loads(json.dumps(item.meta))

    no_query = build_control_view(
        item, store, ControlID.NO_QUERY, seed=4
    )
    swapped = build_control_view(
        item, store, ControlID.HANDLE_SWAP, seed=4
    )

    assert "follow" not in no_query.item.prompt
    assert "Return" not in no_query.item.prompt
    assert no_query.item.prompt.startswith("Slot 0 refers to Q1.")
    assert swapped.item.prompt != item.prompt
    assert swapped.item.meta == original_meta
    assert no_query.item.meta == original_meta
    assert swapped.store.snapshot_sha256() == store.snapshot_sha256()
    assert no_query.store.snapshot_sha256() == store.snapshot_sha256()
    assert swapped.changed_addresses == ()
    assert no_query.changed_addresses == ()
    assert item.meta == original_meta


def test_explicit_miss_deletes_one_required_row_and_gold_path_forces_actions():
    item, store = _fixture()

    miss = build_control_view(
        item, store, ControlID.EXPLICIT_MISS, seed=7
    )
    gold = build_control_view(
        item, store, ControlID.GOLD_PATH, seed=7
    )

    assert len(miss.changed_addresses) == 1
    assert miss.store.lookup(miss.changed_addresses[0]) is None
    assert miss.oracle_after is None
    assert miss.oracle_effect == "miss"
    assert len(gold.forced_actions) == 6
    assert gold.forced_returns == (
        store.lookup(GraphAddress(1, "r0", "out")),
        store.lookup(GraphAddress(2, "r1", "out")),
        None,
        None,
        None,
        None,
    )
    assert gold.oracle_effect == "unchanged"


def test_gold_returns_leaves_model_actions_unforced():
    item, store = _fixture()

    view = build_control_view(
        item, store, ControlID.GOLD_RETURNS, seed=7
    )

    assert view.forced_actions is None
    assert view.forced_returns == (
        store.lookup(GraphAddress(1, "r0", "out")),
        store.lookup(GraphAddress(2, "r1", "out")),
        None,
        None,
        None,
        None,
    )
    assert view.oracle_effect == "unchanged"


@pytest.mark.parametrize(
    "control_id",
    [ControlID.ENTITY_RENAME, ControlID.GRAPH_ISOMORPHISM],
)
def test_entity_bijections_preserve_oracle_and_never_collide(control_id):
    item, store = _fixture()

    view = build_control_view(item, store, control_id, seed=123)

    old_entities = set(view.entity_bijection)
    new_entities = set(view.entity_bijection.values())
    assert len(old_entities) == len(new_entities)
    assert all(old != new for old, new in view.entity_bijection.items())
    assert len(view.store.rows()) == len(store.rows())
    assert len({row.address for row in view.store.rows()}) == len(store.rows())
    assert oracle_answer(view.item, view.store) == item.answer
    assert view.oracle_effect == "unchanged"
    for row in store.rows():
        mapped = view.store.lookup(
            GraphAddress(
                view.entity_bijection[row.source_id],
                row.relation_id,
                row.direction,
            )
        )
        assert mapped is not None
        if row.target_kind == "entity":
            assert mapped.target == str(
                view.entity_bijection[int(row.target)]
            )


def test_graph_isomorphism_preserves_cyclic_repeated_paths():
    rows = [
        GraphRow(1, "r0", "out", "entity", "2", (("compose", "1"),), "world-7"),
        GraphRow(2, "r0", "out", "entity", "1", (("compose", "2"),), "world-7"),
    ]
    item, _ = _fixture()
    item.prompt = (
        "Slot 0 refers to Q1. Start at slot 0 and follow "
        "r0 then r0 then r0 then r0. Return the composed relation."
    )
    item.answer = "r2"
    item.meta["relations"] = ["r0"] * 4
    item.meta["relation_path_hash"] = composition_hash(("r0",) * 4)
    item.meta["hop_count"] = 4
    item.meta["graph_rows"] = 2
    item.meta["gold_addresses"] = [
        [1, "r0", "out"],
        [2, "r0", "out"],
        [1, "r0", "out"],
        [2, "r0", "out"],
    ]
    item.meta["gold_actions"] = [
        *[_action(0, "r0", read=True) for _ in range(4)],
        _action(0, "r0", read=False, halt=True),
        _action(0, "r0", read=False),
    ]
    store = AtomicGraphStore(rows)

    view = build_control_view(
        item, store, ControlID.GRAPH_ISOMORPHISM, seed=5
    )

    assert oracle_answer(item, store) == "r2"
    assert oracle_answer(view.item, view.store) == "r2"
    assert len(view.item.meta["gold_addresses"]) == 4
    assert (
        view.item.meta["gold_addresses"][0]
        == view.item.meta["gold_addresses"][2]
    )


def test_controls_reject_bad_seed_unknown_control_and_wrong_oracle():
    item, store = _fixture()
    with pytest.raises(ValueError, match="seed"):
        build_control_view(item, store, ControlID.CORRECT, seed=True)
    with pytest.raises(ValueError, match="control"):
        build_control_view(item, store, "made_up", seed=1)

    item.answer = "wrong"
    with pytest.raises(ValueError, match="oracle"):
        build_control_view(item, store, ControlID.CORRECT, seed=1)


def test_control_fingerprint_is_independent_of_python_hash_seed(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "fingerprint.py"
    script.write_text(
        """
from corpusgen.graph_records import GraphRow
from corpusgen.records import QAItem
from corpusgen.world_splits import composition_hash
from evals.relational_controls import build_control_view
from organizer.graph_store import AtomicGraphStore

rows = [
    GraphRow(1, "r", "out", "entity", "2", (("compose", "1"),), "world-1"),
    GraphRow(2, "r", "out", "entity", "1", (("compose", "2"),), "world-1"),
]
actions = [
    {"source_slot": 0, "relation_id": "r", "direction": "out",
     "read": True, "halt": False},
    {"source_slot": 0, "relation_id": "r", "direction": "out",
     "read": False, "halt": True},
] + [
    {"source_slot": 0, "relation_id": "r", "direction": "out",
     "read": False, "halt": False}
    for _ in range(4)
]
item = QAItem("p-o", "path_composition",
    "Slot 0 refers to Q1. Start at slot 0 and follow r. "
    "Return the composed relation.", "r1", {
        "pair_id": "p", "variant": "original", "template_id": "t",
        "world_id": 1, "provenance_id": "world-1",
        "relation_path_hash": composition_hash(("r",)),
        "graph_rows": 2,
        "composition_split": "seen", "hop_count": 1, "relations": ["r"],
        "entity_slots": [1, None, None, None],
        "gold_addresses": [[1, "r", "out"]], "gold_fact_ids": ["f"],
        "gold_actions": actions, "answer_choices": ["r0", "r1", "r2", "r3"],
        "changed_row": None,
    })
print(build_control_view(
    item, AtomicGraphStore(rows), "graph_isomorphism", 44
).fingerprint())
""".strip()
        + "\n"
    )
    outputs = []
    for hash_seed in ("1", "87654321"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(repo)
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1]
