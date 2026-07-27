from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import islice

import pytest

from corpusgen import srgm_worlds
from corpusgen.graph_records import (
    GraphAddress,
    TaggedSegment,
    relative_position_bin,
    stable_fact_id,
)
from corpusgen.graph_trace import serialize_return
from corpusgen.mask_ledger import RandomMaskUndersupplyError
from corpusgen.relation_schema import (
    LITERAL_RELATIONS,
    RelationSchema,
    RelationSpec,
)
from corpusgen.srgm_worlds import (
    WorldConfig,
    balance_record_random_controls,
    generate_eval_pairs,
    generate_world,
    iter_bed_records,
    iter_graph_records,
    iter_reasoning_records,
    iter_worlds,
    make_factual_recall_item,
)
from corpusgen.world_splits import build_split_plan, composition_hash
from organizer.graph_store import AtomicGraphStore
from train.tokenizer import get_tok


RELATIONS = {f"r{index}" for index in range(6)}


def _wikidata_shaped_schema():
    entity_specs = tuple(
        RelationSpec(
            relation_id=f"P{index}",
            aliases=(f"relation {index}", f"property {index}"),
            target_kind="entity",
            support=84,
            distinct_subjects=80,
            distinct_objects=40,
            entity_count=100,
        )
        for index in range(1, 33)
    )
    literal_specs = tuple(
        RelationSpec(relation_id, aliases, target_kind)
        for relation_id, aliases, target_kind in LITERAL_RELATIONS
    )
    return RelationSchema(
        catalog=entity_specs + literal_specs,
        path_relation_ids=tuple(spec.relation_id for spec in entity_specs)
        + tuple(spec.relation_id for spec in literal_specs),
    )


def _split_world_config(schema, plan, split_name, *, n_entities=100):
    partition = plan.partition(split_name)
    return WorldConfig(
        n_entities=n_entities,
        seed=plan.seed,
        schema=schema,
        split_plan=plan,
        split_name=split_name,
        entity_id_offset=partition.entity_id_range[0],
        world_seed_offset=partition.world_seed_range[0],
    )


def _rows_by_address(world):
    return {fact.row.address: fact.row for fact in world.facts}


def _answer_from_evidence(world, pair, item):
    rows = _rows_by_address(world)
    if item.meta["variant"] == "counterfactual":
        rows[pair.changed_row.address] = pair.changed_row
    used = [
        rows[GraphAddress(int(source), relation, direction)]
        for source, relation, direction in item.meta["gold_addresses"]
    ]
    if item.task == "path_composition":
        compose = sum(int(dict(row.qualifiers)["compose"]) for row in used) % 4
        return f"r{compose}"
    if item.task == "date_ordering":
        return "<|slot_0|>" if used[0].target < used[1].target else "<|slot_1|>"
    if item.task == "balanced_equality":
        return "yes" if used[0].target == used[1].target else "no"
    raise AssertionError(f"unexpected task: {item.task}")


def test_world_has_six_unique_functional_facts_per_entity():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    by_source = defaultdict(list)
    for fact in world.facts:
        by_source[fact.row.source_id].append(fact)

    assert len(world.facts) == 64 * 6
    assert set(by_source) == set(range(64))
    assert all(
        {fact.row.relation_id for fact in facts} == RELATIONS
        for facts in by_source.values()
    )
    assert len({fact.row.address for fact in world.facts}) == len(world.facts)
    assert len({fact.fact_id for fact in world.facts}) == len(world.facts)

    store = AtomicGraphStore(fact.row for fact in world.facts)
    assert all(store.lookup(fact.row.address) == fact.row for fact in world.facts)


def test_world_generation_is_seed_deterministic():
    cfg = WorldConfig(n_entities=64, seed=11, entity_id_offset=512)
    assert generate_world(3, cfg) == generate_world(3, cfg)
    assert generate_world(3, cfg) != generate_world(
        3, WorldConfig(n_entities=64, seed=12, entity_id_offset=512)
    )


def test_public_world_ids_have_disjoint_addresses():
    first = generate_world(3, WorldConfig(n_entities=32, seed=7))
    second = generate_world(4, WorldConfig(n_entities=32, seed=7))
    store = AtomicGraphStore(
        [fact.row for fact in first.facts]
        + [fact.row for fact in second.facts]
    )
    assert len(store) == 2 * 32 * 6


@pytest.mark.parametrize(
    ("n_entities", "expected"),
    [
        (16, [16]),
        (32, [32]),
        (63, [63]),
        (65, [49, 16]),
        (70, [54, 16]),
        (127, [63, 64]),
        (130, [64, 50, 16]),
    ],
)
def test_world_stream_preserves_population_without_undersized_tail(
    n_entities,
    expected,
):
    worlds = list(iter_worlds(n_entities, world_size=64, seed=11))
    source_ids_by_world = [
        {fact.row.source_id for fact in world.facts}
        for world in worlds
    ]

    assert [len(world.entity_names) for world in worlds] == expected
    assert sum(len(world.entity_names) for world in worlds) == n_entities
    assert all(len(world.entity_names) >= 16 for world in worlds)
    assert set().union(*source_ids_by_world) == set(range(n_entities))
    assert all(
        source_ids.isdisjoint(other_ids)
        for index, source_ids in enumerate(source_ids_by_world)
        for other_ids in source_ids_by_world[index + 1 :]
    )


def test_world_stream_rejects_totals_below_reasoning_minimum_immediately():
    with pytest.raises(ValueError, match="n_entities must be at least 16"):
        iter_worlds(
            n_entities=15,
            world_size=64,
            seed=23,
        )


def test_world_iterator_does_not_materialize_the_requested_population():
    worlds = iter_worlds(
        n_entities=10**12,
        world_size=16,
        seed=5,
        world_id_offset=7,
    )
    first = next(worlds)
    assert first.world_id == 7
    assert {fact.row.source_id for fact in first.facts} == set(range(112, 128))


def test_world_configuration_keeps_the_fixed_six_relation_schema():
    with pytest.raises(ValueError, match="relation_count must be 4"):
        generate_world(0, WorldConfig(n_entities=16, relation_count=3))


def test_counterfactual_pairs_change_supporting_evidence_and_replay_to_flip():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    original_rows = _rows_by_address(world)
    pairs = generate_eval_pairs(world, n_pairs_per_task=20, seed=17)

    assert len(pairs) == 60
    assert {pair.task for pair in pairs} == {
        "path_composition",
        "date_ordering",
        "balanced_equality",
    }
    for pair in pairs:
        original = pair.original
        counterfactual = pair.counterfactual
        changed_original = original_rows[pair.changed_row.address]
        gold_addresses = {
            GraphAddress(int(source), relation, direction)
            for source, relation, direction in original.meta["gold_addresses"]
        }

        assert original.answer != counterfactual.answer
        assert _answer_from_evidence(world, pair, original) == original.answer
        assert (
            _answer_from_evidence(world, pair, counterfactual)
            == counterfactual.answer
        )
        assert original.meta["pair_id"] == counterfactual.meta["pair_id"]
        assert original.meta["graph_rows"] == counterfactual.meta["graph_rows"]
        assert original.meta["graph_rows"] == len(world.facts)
        assert original.meta["changed_row"] is None
        assert counterfactual.meta["changed_row"] == pair.changed_row.as_json()
        assert pair.changed_row.address in gold_addresses
        assert pair.changed_row.address == changed_original.address
        assert pair.changed_row != changed_original
        assert pair.changed_row.provenance_id == changed_original.provenance_id


def test_six_hop_counterfactual_replays_independently_through_both_stores():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=11)
    world = generate_world(
        0,
        _split_world_config(schema, plan, "protected_heldout"),
    )
    pair = next(
        pair
        for pair in generate_eval_pairs(
            world,
            n_pairs_per_task=12,
            seed=17,
        )
        if pair.task == "path_composition"
        and pair.original.meta["hop_count"] == 6
    )
    original_store = AtomicGraphStore(fact.row for fact in world.facts)
    counterfactual_store = AtomicGraphStore(
        pair.changed_row
        if fact.row.address == pair.changed_row.address
        else fact.row
        for fact in world.facts
    )
    addresses = tuple(
        GraphAddress(int(source), relation, direction)
        for source, relation, direction in pair.original.meta["gold_addresses"]
    )

    def replay(store):
        rows = tuple(store.lookup(address) for address in addresses)
        assert all(row is not None for row in rows)
        answer = f"r{sum(int(dict(row.qualifiers)['compose']) for row in rows) % 4}"
        return answer, rows

    original_answer, original_rows = replay(original_store)
    counterfactual_answer, counterfactual_rows = replay(counterfactual_store)

    assert len(addresses) == 6
    assert pair.counterfactual.meta["gold_addresses"] == (
        pair.original.meta["gold_addresses"]
    )
    assert original_answer == pair.original.answer
    assert counterfactual_answer == pair.counterfactual.answer
    assert sum(
        original != counterfactual
        for original, counterfactual in zip(
            original_rows,
            counterfactual_rows,
        )
    ) == 1
    assert original_answer != counterfactual_answer


def test_eval_items_persist_exact_six_step_gold_actions():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    pairs = generate_eval_pairs(world, n_pairs_per_task=20, seed=17)

    for pair in pairs:
        for item in (pair.original, pair.counterfactual):
            actions = item.meta["gold_actions"]
            reads = [action for action in actions if action["read"]]

            assert len(actions) == 6
            assert len(reads) == len(item.meta["gold_addresses"])
            assert [
                [action["relation_id"], action["direction"]]
                for action in reads
            ] == [
                [relation, direction]
                for _, relation, direction in item.meta["gold_addresses"]
            ]
            halt = actions[len(reads)]
            assert halt["halt"] and not halt["read"]
            assert all(
                not action["read"] and not action["halt"]
                for action in actions[len(reads) + 1 :]
            )


def test_six_reads_use_implicit_budget_termination():
    from corpusgen.srgm_worlds import make_action_plan

    actions = make_action_plan(("P1",) * 6)

    assert len(actions) == 6
    assert all(action.read and not action.halt for action in actions)


def test_one_to_five_reads_have_halt_then_noops():
    from corpusgen.srgm_worlds import make_action_plan

    actions = make_action_plan(("P1", "P2"))

    assert [action.read for action in actions[:2]] == [True, True]
    assert actions[2].halt
    assert all(not action.read and not action.halt for action in actions[3:])


def test_gold_action_slots_cover_multihop_paths_and_two_branch_reads():
    world = generate_world(0, WorldConfig(n_entities=64, seed=13))
    pairs = generate_eval_pairs(world, n_pairs_per_task=30, seed=19)
    path = next(
        pair.original
        for pair in pairs
        if pair.task == "path_composition"
        and len(pair.original.meta["gold_addresses"]) >= 3
    )
    branch = next(
        pair.original for pair in pairs if pair.task == "date_ordering"
    )

    assert {
        action["source_slot"]
        for action in path.meta["gold_actions"]
        if action["read"]
    } == {0}
    assert [
        action["source_slot"]
        for action in branch.meta["gold_actions"]
        if action["read"]
    ] == [0, 1]


def test_balanced_equality_twins_are_counterbalanced_in_both_orientations():
    world = generate_world(0, WorldConfig(n_entities=64, seed=23))
    equality_pairs = [
        pair
        for pair in generate_eval_pairs(world, n_pairs_per_task=20, seed=29)
        if pair.task == "balanced_equality"
    ]
    assert Counter(pair.original.answer for pair in equality_pairs) == {
        "yes": 10,
        "no": 10,
    }
    assert Counter(pair.counterfactual.answer for pair in equality_pairs) == {
        "yes": 10,
        "no": 10,
    }
    assert all(
        {pair.original.answer, pair.counterfactual.answer} == {"yes", "no"}
        for pair in equality_pairs
    )


def test_protected_answers_are_derived_not_payload_copies():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    payloads = {fact.row.target for fact in world.facts}
    pairs = generate_eval_pairs(world, n_pairs_per_task=10, seed=19)
    for pair in pairs:
        assert pair.original.answer not in payloads
        assert pair.counterfactual.answer not in payloads


def test_factual_recall_answer_is_exact_row_target_not_compose_code():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    fact = next(
        item
        for item in world.facts
        if item.row.target_kind == "entity"
        and "compose" in dict(item.row.qualifiers)
    )

    item = make_factual_recall_item(world, fact, 0, tok)

    assert item.answer == f"Q{fact.row.target}"
    assert item.answer != f"r{dict(fact.row.qualifiers)['compose']}"
    assert item.meta["gold_fact_ids"] == [fact.fact_id]
    assert item.meta["gold_addresses"] == [
        [
            fact.row.source_id,
            fact.row.relation_id,
            fact.row.direction,
        ]
    ]
    assert item.meta["target_kind"] == "entity"


def test_factual_recall_literal_answer_is_exact_stored_utf8():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=11))
    fact = next(
        item for item in world.facts if item.row.target_kind == "literal"
    )

    item = make_factual_recall_item(world, fact, 1, tok)

    assert item.answer == fact.row.target
    assert item.meta["target_kind"] == "literal"


def test_factual_recall_choices_are_same_kind_unique_and_token_prefix_free():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=13))
    facts = (
        next(item for item in world.facts if item.row.target_kind == "entity"),
        next(item for item in world.facts if item.row.target_kind == "literal"),
    )

    for ordinal, fact in enumerate(facts):
        item = make_factual_recall_item(world, fact, ordinal, tok)
        choices = item.meta["answer_choices"]
        encoded = [tuple(tok.encode(choice)) for choice in choices]
        same_kind_answers = {
            (
                f"Q{candidate.row.target}"
                if candidate.row.target_kind == "entity"
                else candidate.row.target
            )
            for candidate in world.facts
            if candidate.row.target_kind == fact.row.target_kind
        }

        assert len(choices) == len(set(choices)) == 4
        assert item.answer in choices
        assert set(choices) <= same_kind_answers
        assert all(encoded)
        assert len(encoded) == len(set(encoded))
        assert all(
            not (
                len(left) < len(right)
                and right[: len(left)] == left
            )
            for index, left in enumerate(encoded)
            for other_index, right in enumerate(encoded)
            if index != other_index
        )


def test_every_schema_alias_is_emitted_once_for_recognition():
    schema = _wikidata_shaped_schema()

    records = tuple(srgm_worlds.iter_relation_alias_records(schema))

    observed = {
        (record.schedule.record_id.split(":", 2)[1], record.segments[0].text)
        for record in records
    }
    expected = {
        (relation.relation_id, alias)
        for relation in schema.catalog
        for alias in relation.aliases
    }
    assert observed == expected
    assert len(records) == len(expected)
    assert all(
        record.segments[0].role == "relation_alias"
        and record.segments[-1].role == "rule"
        and not {
            segment.role for segment in record.segments
        } & {"payload", "random_control"}
        for record in records
    )


def test_bed_and_graph_renderers_are_lazy_over_their_inputs():
    def bed():
        yield "first"
        raise AssertionError("bed input was materialized")

    bed_record = next(iter_bed_records(bed()))
    assert bed_record.segments[0].text == "first"
    assert bed_record.schedule.component == "bed"

    world = generate_world(0, WorldConfig(n_entities=16, seed=31))

    def worlds():
        yield world
        raise AssertionError("world input was materialized")

    tok = get_tok()
    graph_record = next(iter_graph_records(tok, worlds))
    payload = next(
        segment for segment in graph_record.segments if segment.role == "payload"
    )
    controls = [
        segment
        for segment in graph_record.segments
        if segment.role == "random_control"
    ]
    assert graph_record.schedule.component == "graph"
    assert payload.fact_id is not None
    relation_alias = next(
        segment
        for segment in graph_record.segments
        if segment.role == "relation_alias"
    )
    assert relation_alias.fact_id is None
    assert controls
    assert any(
        len(tok.encode(control.text)) == len(tok.encode(payload.text))
        for control in controls
    )


def _record_role_keys(tok, record, role):
    encoded = [
        (segment, tok.encode(segment.text))
        for segment in record.segments
    ]
    document_length = sum(len(ids) for _, ids in encoded) + 1
    keys = Counter()
    start = 0
    for segment, ids in encoded:
        end = start + len(ids)
        if segment.role == role:
            keys[
                (
                    len(ids),
                    relative_position_bin(start, end, document_length),
                )
            ] += 1
        start = end
    return keys


def test_each_payload_record_has_same_record_exact_random_supply():
    tok = get_tok()
    graph_world = generate_world(0, WorldConfig(n_entities=16, seed=31))
    reasoning_world = generate_world(0, WorldConfig(n_entities=64, seed=43))
    records = tuple(
        islice(
            iter_graph_records(tok, lambda: iter((graph_world,))),
            20,
        )
    ) + tuple(
        islice(
            iter_reasoning_records(
                tok,
                lambda: iter((reasoning_world,)),
                seed=47,
                max_hops=4,
            ),
            20,
        )
    )
    for record in records:
        payload_keys = _record_role_keys(tok, record, "payload")
        control_keys = _record_role_keys(tok, record, "random_control")
        for key, count in payload_keys.items():
            assert control_keys[key] >= count
        assert all(
            set(segment.text) <= {"\t"}
            for segment in record.segments
            if segment.role == "random_control"
        )


def test_record_control_balancing_fails_at_fixed_padding_bound():
    tok = get_tok()
    segments = (
        TaggedSegment('"payload"', "payload", "fact"),
        TaggedSegment("\t", "random_control"),
        TaggedSegment(" distant syntax" * 30, "plain"),
    )

    with pytest.raises(
        RandomMaskUndersupplyError,
        match="within 0 padding tokens",
    ):
        balance_record_random_controls(
            tok,
            segments,
            max_padding_tokens=0,
        )


@pytest.mark.parametrize("hop_band", [1, 2, 4])
def test_reasoning_records_have_fixed_hops_and_six_supervised_steps(hop_band):
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=37))
    facts_by_id = {fact.fact_id: fact for fact in world.facts}

    def worlds():
        yield world
        raise AssertionError("world input was materialized")

    record = next(iter_reasoning_records(tok, worlds, seed=41, max_hops=hop_band))
    ids, _, _ = tok.encode_tagged_segments(record.segments)
    payloads = [
        segment for segment in record.segments if segment.role == "payload"
    ]

    assert record.schedule.component == "reasoning"
    assert record.schedule.curriculum_band == hop_band
    assert ids.count(tok.GRAPH_START) == 6
    assert ids.count(tok.ANSWER_STATE) == 6
    assert ids.count(tok.GRAPH_READ) == hop_band
    assert ids.count(tok.GRAPH_HALT) == 1
    assert ids.count(tok.GRAPH_NOOP) == 5 - hop_band
    assert ids.count(tok.GRAPH_MISS) == 6 - hop_band
    assert len({payload.fact_id for payload in payloads}) == hop_band
    assert (
        sum(
            segment.role == "provisional_answer"
            for segment in record.segments
        )
        == 6
    )
    assert sum(segment.role == "final_answer" for segment in record.segments) == 1

    for payload in payloads:
        fact = facts_by_id[payload.fact_id]
        expected_payloads = {
            segment.text
            for segment in serialize_return(fact.row, fact.fact_id)
            if segment.role == "payload"
        }
        assert payload.text in expected_payloads
    final_answer = next(
        segment.text for segment in record.segments if segment.role == "final_answer"
    )
    assert final_answer not in {fact.row.target for fact in world.facts}


def test_reasoning_records_cover_every_task_with_post_halt_noops_and_controls():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=43))
    facts_by_id = {fact.fact_id: fact for fact in world.facts}
    records = list(
        islice(
            iter_reasoning_records(
                tok,
                lambda: iter((world,)),
                seed=47,
                max_hops=2,
            ),
            24,
        )
    )
    records_by_task = {
        task: next(
            record
            for record in records
            if f"-{task}-" in record.schedule.record_id
        )
        for task in (
            "path_composition",
            "date_ordering",
            "balanced_equality",
        )
    }

    for record in records_by_task.values():
        ids, _, _ = tok.encode_tagged_segments(record.segments)
        action_frames = []
        for segment in record.segments:
            if segment.role != "action":
                continue
            segment_ids = tok.encode(segment.text)
            if segment_ids and segment_ids[0] == tok.GRAPH_START:
                action_frames.append(segment_ids)

        assert len(action_frames) == 6
        assert all(len(frame) == 8 for frame in action_frames)
        assert [frame[6] for frame in action_frames] == [
            tok.GRAPH_READ,
            tok.GRAPH_READ,
            tok.GRAPH_HALT,
            tok.GRAPH_NOOP,
            tok.GRAPH_NOOP,
            tok.GRAPH_NOOP,
        ]
        assert ids.count(tok.ANSWER_STATE) == 6
        assert ids.count(tok.GRAPH_MISS) == 4
        assert (
            sum(
                segment.role == "provisional_answer"
                for segment in record.segments
            )
            == 6
        )
        assert (
            sum(
                segment.role == "final_answer"
                for segment in record.segments
            )
            == 1
        )

        payload_indexes = [
            index
            for index, segment in enumerate(record.segments)
            if segment.role == "payload"
        ]
        assert (
            len(
                {
                    record.segments[index].fact_id
                    for index in payload_indexes
                }
            )
            == 2
        )
        for index in payload_indexes:
            payload = record.segments[index]
            fact = facts_by_id[payload.fact_id]
            expected_payloads = {
                segment.text
                for segment in serialize_return(fact.row, fact.fact_id)
                if segment.role == "payload"
            }
            assert payload.text in expected_payloads
            matching_controls = [
                segment
                for segment in record.segments
                if segment.role == "random_control"
                and len(tok.encode(segment.text))
                == len(tok.encode(payload.text))
            ]
            assert matching_controls


def test_reasoning_renderer_rejects_non_curriculum_hop_bands():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=16, seed=43))
    records = iter_reasoning_records(
        tok,
        lambda: iter((world,)),
        seed=47,
        max_hops=3,
    )
    with pytest.raises(ValueError, match="max_hops must be one of"):
        next(records)


def test_schema_shaped_world_preserves_relation_statistics_and_q_handles():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=17)
    partition = plan.partition("train")
    world = generate_world(
        3,
        _split_world_config(schema, plan, "train"),
    )
    entity_ids = tuple(
        range(
            partition.entity_id_range[0],
            partition.entity_id_range[0] + 100,
        )
    )

    assert world.world_id == partition.world_seed_range[0] + 3
    assert world.entity_names == tuple(f"Q{value}" for value in entity_ids)
    assert all(isinstance(fact.row.source_id, int) for fact in world.facts)
    assert world.manifest["type_metadata"] == {
        "available": False,
        "reason": "pinned Wikidata5M source provides no type metadata",
    }
    assert world.manifest["fact_count"] == len(world.facts)

    for spec in schema.path_relations:
        if spec.target_kind != "entity":
            continue
        facts = [
            fact
            for fact in world.facts
            if fact.row.relation_id == spec.relation_id
        ]
        assert len({fact.row.source_id for fact in facts}) == 80
        assert len({fact.row.target for fact in facts}) == 40
        assert abs(len(facts) / 100 - spec.subject_coverage) <= 1 / 100
        assert (
            abs(
                len({fact.row.target for fact in facts}) / len(facts)
                - spec.target_pool_ratio
            )
            <= 1 / len(facts)
        )


def test_sparse_observed_relations_get_a_one_entity_reasoning_backbone():
    schema = _wikidata_shaped_schema()
    sparse_catalog = tuple(
        (
            replace(
                spec,
                support=1,
                distinct_subjects=1,
                distinct_objects=1,
                entity_count=1_000,
            )
            if spec.target_kind == "entity"
            else spec
        )
        for spec in schema.catalog
    )
    sparse_schema = RelationSchema(
        sparse_catalog,
        schema.path_relation_ids,
    )
    plan = build_split_plan(sparse_schema, seed=18)
    world = generate_world(
        0,
        _split_world_config(
            sparse_schema,
            plan,
            "protected_heldout",
            n_entities=64,
        ),
    )
    required_relations = {
        relation_id
        for partition in (plan.protected_seen, plan.protected_heldout)
        for path in partition.compositions
        for relation_id in path
    }
    shapes = {
        shape["relation_id"]: shape
        for shape in world.manifest["relation_shapes"]
    }

    for spec in sparse_schema.path_relations:
        if spec.target_kind != "entity":
            continue
        facts = [
            fact
            for fact in world.facts
            if fact.row.relation_id == spec.relation_id
        ]
        assert len(facts) == (
            1 if spec.relation_id in required_relations else 0
        )
        assert abs(len(facts) / 64 - spec.subject_coverage) <= 1 / 64
        assert shapes[spec.relation_id]["backbone_subjects_added"] == (
            1 if spec.relation_id in required_relations else 0
        )

    pairs = generate_eval_pairs(world, n_pairs_per_task=12, seed=19)
    six_hop = next(
        pair.original
        for pair in pairs
        if pair.task == "path_composition"
        and pair.original.meta["hop_count"] == 6
    )
    assert len(six_hop.meta["gold_addresses"]) == 6


def test_schema_shaped_world_fact_ids_and_bytes_are_stable():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=19)
    cfg = _split_world_config(schema, plan, "development")
    first = generate_world(2, cfg)
    second = generate_world(2, cfg)

    assert first == second
    assert first.canonical_bytes() == second.canonical_bytes()
    assert len({fact.fact_id for fact in first.facts}) == len(first.facts)
    assert all(fact.fact_id == stable_fact_id(fact.row) for fact in first.facts)

    row = first.facts[0].row
    canonical = {
        "provenance_id": row.provenance_id,
        "source_id": row.source_id,
        "relation_id": row.relation_id,
        "direction": row.direction,
        "target_kind": row.target_kind,
        "target": row.target,
        "qualifiers": [list(value) for value in row.qualifiers],
    }
    assert stable_fact_id(row) == __import__("hashlib").sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_world_namespaces_make_entities_and_literal_payloads_disjoint():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=23)
    worlds = {
        split_name: generate_world(
            0,
            _split_world_config(schema, plan, split_name),
        )
        for split_name in ("development", "train", "protected_seen")
    }

    entity_sets = {
        name: {int(value[1:]) for value in world.entity_names}
        for name, world in worlds.items()
    }
    literal_sets = {
        name: {
            fact.row.target
            for fact in world.facts
            if fact.row.target_kind == "literal"
        }
        for name, world in worlds.items()
    }
    paraphrase_sets = {
        name: set(world.manifest["paraphrase_assignment_ids"].values())
        for name, world in worlds.items()
    }
    for values in (entity_sets, literal_sets, paraphrase_sets):
        names = tuple(values)
        assert all(
            values[left].isdisjoint(values[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )


def test_training_and_protected_pairs_enforce_composition_and_hop_contracts():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=29)
    worlds = {
        split_name: generate_world(
            0,
            _split_world_config(schema, plan, split_name),
        )
        for split_name in (
            "development",
            "train",
            "protected_seen",
            "protected_heldout",
        )
    }
    pairs = {
        split_name: generate_eval_pairs(
            world,
            n_pairs_per_task=12,
            seed=31,
        )
        for split_name, world in worlds.items()
    }
    path_items = {
        split_name: [
            pair.original
            for pair in split_pairs
            if pair.task == "path_composition"
        ]
        for split_name, split_pairs in pairs.items()
    }

    assert {
        item.meta["hop_count"] for item in path_items["train"]
    } == {1, 2, 3, 4}
    assert all(
        {item.meta["hop_count"] for item in path_items[split_name]}
        == {1, 2, 3, 4, 5, 6}
        for split_name in ("protected_seen", "protected_heldout")
    )

    train_hashes = {
        item.meta["relation_path_hash"] for item in path_items["train"]
    }
    seen_hashes = {
        item.meta["relation_path_hash"]
        for split_name in ("protected_seen", "protected_heldout")
        for item in path_items[split_name]
        if item.meta["composition_split"] == "seen"
    }
    heldout_hashes = {
        item.meta["relation_path_hash"]
        for split_name in ("protected_seen", "protected_heldout")
        for item in path_items[split_name]
        if item.meta["composition_split"] == "heldout"
    }
    assert seen_hashes <= train_hashes
    assert heldout_hashes.isdisjoint(train_hashes)
    assert seen_hashes == plan.protected_seen.composition_hashes
    assert heldout_hashes == plan.protected_heldout.composition_hashes

    development_hashes = {
        pair.original.meta["relation_path_hash"]
        for pair in pairs["development"]
    }
    protected_hashes = {
        pair.original.meta["relation_path_hash"]
        for split_name in ("protected_seen", "protected_heldout")
        for pair in pairs[split_name]
    }
    assert development_hashes.isdisjoint(protected_hashes)

    all_items = [
        item
        for split_pairs in pairs.values()
        for pair in split_pairs
        for item in (pair.original, pair.counterfactual)
    ]
    required_meta = {
        "world_id",
        "relation_path_hash",
        "template_id",
        "composition_split",
        "hop_count",
    }
    assert all(required_meta <= item.meta.keys() for item in all_items)
    assert all(
        item.meta["relation_path_hash"]
        == composition_hash(tuple(item.meta["relations"]))
        for item in all_items
    )
    assert {"seen", "heldout"} == {
        item.meta["composition_split"] for item in all_items
    }
    assert all(
        slot is None or isinstance(slot, int)
        for item in all_items
        for slot in item.meta["entity_slots"]
    )
    assert all("Q" in item.prompt for item in all_items)

    qids = {
        split_name: {
            item.qid
            for pair in split_pairs
            for item in (pair.original, pair.counterfactual)
        }
        for split_name, split_pairs in pairs.items()
    }
    assert qids["train"].isdisjoint(qids["protected_seen"])
    assert qids["train"].isdisjoint(qids["protected_heldout"])
    assert qids["protected_seen"].isdisjoint(qids["protected_heldout"])

    six_hop = next(
        item
        for item in path_items["protected_heldout"]
        if item.meta["hop_count"] == 6
    )
    assert all(
        action["read"] and not action["halt"]
        for action in six_hop.meta["gold_actions"]
    )
    assert all(
        pair.original.answer != pair.counterfactual.answer
        for split_pairs in pairs.values()
        for pair in split_pairs
    )


def test_same_seed_pair_generation_is_byte_identical():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=37)
    world = generate_world(
        0,
        _split_world_config(schema, plan, "protected_heldout"),
    )

    first = generate_eval_pairs(world, n_pairs_per_task=12, seed=41)
    second = generate_eval_pairs(world, n_pairs_per_task=12, seed=41)
    encode = lambda values: json.dumps(
        [asdict(value) for value in values],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert encode(first) == encode(second)


def test_schema_training_renderer_emits_every_hop_through_curriculum_maximum():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=41)
    world = generate_world(
        0,
        _split_world_config(schema, plan, "train"),
    )
    tok = get_tok()
    records = list(
        islice(
            iter_reasoning_records(
                tok,
                lambda: iter((world,)),
                seed=43,
                max_hops=4,
            ),
            8,
        )
    )

    read_counts = {
        tok.encode_tagged_segments(record.segments)[0].count(tok.GRAPH_READ)
        for record in records
    }
    assert read_counts == {1, 2, 3, 4}
    assert all(record.schedule.curriculum_band == 4 for record in records)
    required_metadata = {
        "world_id",
        "relation_path_hash",
        "template_id",
        "composition_split",
        "hop_count",
    }
    assert all(
        required_metadata <= record.metadata.keys() for record in records
    )
    rendered_hashes = {
        record.metadata["relation_path_hash"] for record in records
    }
    assert plan.protected_seen.composition_hashes <= rendered_hashes
    assert plan.protected_heldout.composition_hashes.isdisjoint(
        rendered_hashes
    )


def test_world_config_rejects_offsets_outside_its_split_namespace():
    schema = _wikidata_shaped_schema()
    plan = build_split_plan(schema, seed=43)

    with pytest.raises(ValueError, match="entity namespace"):
        WorldConfig(
            n_entities=100,
            seed=plan.seed,
            schema=schema,
            split_plan=plan,
            split_name="development",
            entity_id_offset=plan.train.entity_id_range[0],
            world_seed_offset=plan.development.world_seed_range[0],
        )


def test_world_generation_fails_closed_without_disjoint_literal_task_relations():
    schema = _wikidata_shaped_schema()
    reduced_catalog = tuple(
        spec
        for spec in schema.catalog
        if spec.relation_id not in {"SYN_L1", "SYN_L5"}
    )
    reduced_path_ids = tuple(
        relation_id
        for relation_id in schema.path_relation_ids
        if relation_id not in {"SYN_L1", "SYN_L5"}
    )
    reduced_schema = RelationSchema(reduced_catalog, reduced_path_ids)
    plan = build_split_plan(reduced_schema, seed=47)

    with pytest.raises(ValueError, match="two date and two category"):
        generate_world(
            0,
            _split_world_config(
                reduced_schema,
                plan,
                "development",
            ),
        )
