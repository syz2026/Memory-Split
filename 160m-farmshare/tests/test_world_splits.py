from __future__ import annotations

import hashlib
import json
import random
from dataclasses import FrozenInstanceError, replace
from itertools import islice

import pytest

from corpusgen.relation_schema import LITERAL_RELATIONS, RelationSchema, RelationSpec
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
    iter_reasoning_records,
)
from corpusgen.world_splits import (
    ObservedSplitArtifacts,
    ReasoningArtifactSignature,
    SplitArtifactExpectations,
    WorldArtifactSignature,
    audit_disjointness,
    audit_split_plan_disjointness,
    build_split_plan,
    composition_bucket,
    composition_hash,
)
from train.tokenizer import get_tok


TASKS = ("path_composition", "date_ordering", "balanced_equality")


def fixture_schema() -> RelationSchema:
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


def _world_config(schema, plan, split_name):
    partition = plan.partition(split_name)
    return WorldConfig(
        n_entities=100,
        seed=plan.seed,
        schema=schema,
        split_plan=plan,
        split_name=split_name,
        entity_id_offset=partition.entity_id_range[0],
        world_seed_offset=partition.world_seed_range[0],
    )


def _observe_partition(name, bundle):
    return ObservedSplitArtifacts.from_generated(
        name,
        expectations=bundle["expectations"],
        worlds=bundle["worlds"],
        qa_items=bundle["qa_items"],
        rendered_records=bundle["rendered_records"],
    )


def _expected_qa_ids(plan, name, world_id, seed, count):
    namespace = plan.partition(name).question_namespace
    return tuple(
        f"{namespace}:{name}:{world_id}:{seed}:{task}:{index}-{variant}"
        for task in TASKS
        for index in range(count)
        for variant in ("o", "c")
    )


def _expected_rendered_ids(plan, name, world_id, schedule):
    namespace = plan.partition(name).question_namespace
    expected = []
    for seed, max_hops in schedule:
        generated_seed = random.Random(seed).randrange(1 << 30)
        tasks = ("path_composition",) if max_hops == 1 else TASKS
        expected.extend(
            f"{namespace}:{name}:{world_id}:{generated_seed}:"
            f"{task}:{index}-o"
            for task in tasks
            for index in range(8)
        )
    return tuple(expected)


def _render_records(tokenizer, world, schedule, expected_ids):
    records = []
    offset = 0
    for seed, max_hops in schedule:
        count = 8 if max_hops == 1 else 24
        records.extend(
            islice(
                iter_reasoning_records(
                    tokenizer,
                    lambda world=world: iter((world,)),
                    seed=seed,
                    max_hops=max_hops,
                ),
                count,
            )
        )
        offset += count
    assert len(records) == len(expected_ids) == offset
    return tuple(records)


def _snapshot_row_address_sha256(world):
    canonical = [
        {
            "address": {
                "source_id": fact.row.source_id,
                "relation_id": fact.row.relation_id,
                "direction": fact.row.direction,
            },
            "row": fact.row.as_json(),
        }
        for fact in world.facts
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _snapshot_reasoning_signature(artifact_id, metadata):
    return ReasoningArtifactSignature(
        artifact_id=artifact_id,
        world_id=metadata["world_id"],
        relation_path_hash=metadata["relation_path_hash"],
        template_id=metadata["template_id"],
        composition_split=metadata["composition_split"],
        hop_count=metadata["hop_count"],
        relations=tuple(metadata["relations"]),
    )


def _snapshot_expectations(name, snapshot, required_hops):
    return SplitArtifactExpectations(
        name=name,
        world_signatures=tuple(
            WorldArtifactSignature(
                world_id=world.world_id,
                world_seed=world.world_seed,
                fact_ids=tuple(fact.fact_id for fact in world.facts),
                row_address_sha256=_snapshot_row_address_sha256(world),
                fact_count=len(world.facts),
            )
            for world in snapshot["worlds"]
        ),
        qa_signatures=tuple(
            _snapshot_reasoning_signature(item.qid, item.meta)
            for item in snapshot["qa_items"]
        ),
        rendered_signatures=tuple(
            _snapshot_reasoning_signature(
                record.schedule.record_id,
                record.metadata,
            )
            for record in snapshot["rendered_records"]
        ),
        required_hops=required_hops,
    )


@pytest.fixture(scope="module")
def emitted_split_artifacts():
    schema = fixture_schema()
    plan = build_split_plan(schema, seed=17)
    tokenizer = get_tok()
    bundles = {}
    observed = {}
    for ordinal, name in enumerate(
        ("development", "train", "protected_seen", "protected_heldout")
    ):
        partition = plan.partition(name)
        world_id = partition.world_seed_range[0]
        qa_seed = 31
        schedule = (
            tuple((100 + ordinal * 10 + hops, hops) for hops in range(1, 7))
            if name.startswith("protected_")
            else ((41 + ordinal, 4),)
        )
        expected_rendered_ids = _expected_rendered_ids(
            plan,
            name,
            world_id,
            schedule,
        )
        required_hops = (
            frozenset(range(1, 7))
            if name.startswith("protected_")
            else frozenset(range(1, 5))
        )
        snapshot_world = generate_world(
            0,
            _world_config(schema, plan, name),
        )
        snapshot_pairs = generate_eval_pairs(
            snapshot_world,
            n_pairs_per_task=12,
            seed=qa_seed,
        )
        snapshot_records = _render_records(
            tokenizer,
            snapshot_world,
            schedule,
            expected_rendered_ids,
        )
        snapshot = {
            "worlds": (snapshot_world,),
            "qa_items": tuple(
                item
                for pair in snapshot_pairs
                for item in (pair.original, pair.counterfactual)
            ),
            "rendered_records": snapshot_records,
        }
        assert tuple(item.qid for item in snapshot["qa_items"]) == (
            _expected_qa_ids(
                plan,
                name,
                world_id,
                qa_seed,
                12,
            )
        )
        expectations = _snapshot_expectations(
            name,
            snapshot,
            required_hops,
        )
        world = generate_world(0, _world_config(schema, plan, name))
        pairs = generate_eval_pairs(
            world,
            n_pairs_per_task=12,
            seed=qa_seed,
        )
        records = _render_records(
            tokenizer,
            world,
            schedule,
            expected_rendered_ids,
        )
        bundle = {
            "expectations": expectations,
            "snapshot": snapshot,
            "required_hops": required_hops,
            "worlds": (world,),
            "qa_items": tuple(
                item
                for pair in pairs
                for item in (pair.original, pair.counterfactual)
            ),
            "rendered_records": records,
        }
        bundles[name] = bundle
        observed[name] = _observe_partition(name, bundle)
    return plan, bundles, observed


def _replace_observed_partition(observed, name, bundle):
    mutated = dict(observed)
    mutated[name] = _observe_partition(name, bundle)
    return mutated


def _replace_world_fact(world, fact_index, row):
    facts = list(world.facts)
    facts[fact_index] = replace(facts[fact_index], row=row)
    return replace(world, facts=tuple(facts))


def _replace_qa_metadata(items, qid, changes):
    return tuple(
        (
            replace(item, meta={**item.meta, **changes})
            if item.qid == qid
            else item
        )
        for item in items
    )


def _replace_rendered_metadata(records, record_id, changes):
    return tuple(
        (
            replace(record, metadata={**record.metadata, **changes})
            if record.schedule.record_id == record_id
            else record
        )
        for record in records
    )


def _bundle_with_frozen_qa_change(name, bundle, qid, changes):
    snapshot = {
        **bundle["snapshot"],
        "qa_items": _replace_qa_metadata(
            bundle["snapshot"]["qa_items"],
            qid,
            changes,
        ),
    }
    return {
        **bundle,
        "expectations": _snapshot_expectations(
            name,
            snapshot,
            bundle["required_hops"],
        ),
        "snapshot": snapshot,
        "qa_items": _replace_qa_metadata(bundle["qa_items"], qid, changes),
    }


def _bundle_with_frozen_rendered_change(
    name,
    bundle,
    record_id,
    changes,
):
    snapshot = {
        **bundle["snapshot"],
        "rendered_records": _replace_rendered_metadata(
            bundle["snapshot"]["rendered_records"],
            record_id,
            changes,
        ),
    }
    return {
        **bundle,
        "expectations": _snapshot_expectations(
            name,
            snapshot,
            bundle["required_hops"],
        ),
        "snapshot": snapshot,
        "rendered_records": _replace_rendered_metadata(
            bundle["rendered_records"],
            record_id,
            changes,
        ),
    }


def _bundle_with_frozen_world_change(
    name,
    bundle,
    *,
    worlds,
    snapshot_worlds,
):
    snapshot = {**bundle["snapshot"], "worlds": snapshot_worlds}
    return {
        **bundle,
        "expectations": _snapshot_expectations(
            name,
            snapshot,
            bundle["required_hops"],
        ),
        "snapshot": snapshot,
        "worlds": worlds,
    }


def _bundle_with_frozen_rendered_records(
    name,
    bundle,
    *,
    rendered_records,
    snapshot_rendered_records,
):
    snapshot = {
        **bundle["snapshot"],
        "rendered_records": snapshot_rendered_records,
    }
    return {
        **bundle,
        "expectations": _snapshot_expectations(
            name,
            snapshot,
            bundle["required_hops"],
        ),
        "snapshot": snapshot,
        "rendered_records": rendered_records,
    }


def test_composition_hash_and_bucket_follow_the_frozen_canonical_rule():
    relations = ("P31", "P279", "P17")
    expected = hashlib.sha256(
        json.dumps(list(relations), separators=(",", ":")).encode()
    ).hexdigest()

    assert composition_hash(relations) == expected
    assert composition_bucket(relations) == int(expected[:8], 16) % 100


def test_split_plan_is_deterministic_canonical_and_records_missing_types():
    schema = fixture_schema()
    first = build_split_plan(schema, seed=17)
    second = build_split_plan(schema, seed=17)

    assert first == second
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.sha256() == hashlib.sha256(first.canonical_bytes()).hexdigest()
    assert first.schema_sha256 == schema.sha256()
    assert first.to_dict()["type_metadata"] == {
        "available": False,
        "reason": "pinned Wikidata5M source provides no type metadata",
    }


def test_static_split_plan_is_disjoint_on_all_required_axes():
    report = audit_split_plan_disjointness(
        build_split_plan(fixture_schema(), seed=17)
    )

    assert report == {
        "world_seeds": True,
        "entity_ids": True,
        "payload_values": True,
        "paraphrase_assignments": True,
        "question_ids": True,
        "relation_path_hashes": True,
        "heldout_relation_compositions": True,
    }


def test_buckets_and_seen_versus_heldout_contract_are_explicit():
    plan = build_split_plan(fixture_schema(), seed=17)
    train = plan.partition("train")
    development = plan.partition("development")
    seen = plan.partition("protected_seen")
    heldout = plan.partition("protected_heldout")

    assert all(composition_bucket(path) < 80 for path in train.compositions)
    assert all(
        80 <= composition_bucket(path) < 90
        for path in development.compositions
    )
    assert all(
        90 <= composition_bucket(path) < 100
        for path in heldout.compositions
    )
    assert seen.composition_hashes < train.composition_hashes
    assert heldout.composition_hashes.isdisjoint(train.composition_hashes)
    assert {len(path) for path in train.compositions} == {1, 2, 3, 4}
    assert {len(path) for path in seen.compositions} == {1, 2, 3, 4}
    assert {len(path) for path in heldout.compositions} == {
        1,
        2,
        3,
        4,
        5,
        6,
    }


def test_split_partitions_reject_an_empty_composition_schedule():
    plan = build_split_plan(fixture_schema(), seed=17)

    with pytest.raises(ValueError, match="nonempty"):
        replace(plan.train, compositions=())


@pytest.mark.parametrize(
    ("axis", "mutate"),
    [
        (
            "world_seeds",
            lambda plan: replace(
                plan,
                development=replace(
                    plan.development,
                    world_seed_range=plan.train.world_seed_range,
                ),
            ),
        ),
        (
            "entity_ids",
            lambda plan: replace(
                plan,
                development=replace(
                    plan.development,
                    entity_id_range=plan.train.entity_id_range,
                ),
            ),
        ),
        (
            "payload_values",
            lambda plan: replace(
                plan,
                development=replace(
                    plan.development,
                    payload_namespace=plan.train.payload_namespace,
                ),
            ),
        ),
        (
            "paraphrase_assignments",
            lambda plan: replace(
                plan,
                development=replace(
                    plan.development,
                    paraphrase_namespace=plan.train.paraphrase_namespace,
                ),
            ),
        ),
        (
            "question_ids",
            lambda plan: replace(
                plan,
                development=replace(
                    plan.development,
                    question_namespace=plan.train.question_namespace,
                ),
            ),
        ),
        (
            "relation_path_hashes",
            lambda plan: replace(
                plan,
                development=replace(
                    plan.development,
                    compositions=plan.train.compositions,
                ),
            ),
        ),
        (
            "heldout_relation_compositions",
            lambda plan: replace(
                plan,
                protected_heldout=replace(
                    plan.protected_heldout,
                    compositions=plan.train.compositions,
                ),
            ),
        ),
    ],
)
def test_every_static_disjointness_axis_fails_closed(axis, mutate):
    broken = mutate(build_split_plan(fixture_schema(), seed=17))
    report = audit_split_plan_disjointness(broken)

    assert report[axis] is False
    if axis in {
        "world_seeds",
        "entity_ids",
        "payload_values",
        "paraphrase_assignments",
        "question_ids",
    }:
        assert {name for name, passed in report.items() if not passed} == {axis}
    with pytest.raises(ValueError, match=axis):
        broken.require_static_disjointness()


def test_artifact_gate_derives_all_axes_from_emitted_outputs(
    emitted_split_artifacts,
):
    _, bundles, observed = emitted_split_artifacts

    assert audit_disjointness(observed) == {
        "world_seeds": True,
        "entity_ids": True,
        "payload_values": True,
        "paraphrase_assignments": True,
        "question_ids": True,
        "relation_path_hashes": True,
        "heldout_relation_compositions": True,
    }
    development = observed["development"]
    development_world = bundles["development"]["worlds"][0]
    qualifier_payload = development_world.facts[0].row.qualifiers[0][1]
    assert qualifier_payload in development.payload_values
    assert development_world.facts[0].row.target in development.payload_values
    assert development_world.world_seed in development.world_seeds
    assert development_world.facts[0].row.source_id in development.entity_ids
    for name, artifacts in observed.items():
        expected_hops = (
            frozenset(range(1, 7))
            if name.startswith("protected_")
            else frozenset(range(1, 5))
        )
        assert artifacts.qa_hops == expected_hops
        assert artifacts.rendered_hops == expected_hops
        assert artifacts.qa_question_ids.isdisjoint(
            artifacts.rendered_record_ids
        )


def test_frozen_expectation_signatures_require_canonical_digests(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    expectations = bundles["development"]["expectations"]
    world_signature = expectations.world_signatures[0]
    qa_signature = expectations.qa_signatures[0]

    with pytest.raises(FrozenInstanceError):
        world_signature.fact_count = 0
    with pytest.raises(
        ValueError,
        match="row/address digest must be lowercase SHA-256",
    ):
        replace(world_signature, row_address_sha256="not-canonical")
    with pytest.raises(
        ValueError,
        match="canonical relation path hash",
    ):
        replace(qa_signature, relation_path_hash="0" * 64)


def test_split_expectations_are_versioned_canonical_and_round_trip(
    emitted_split_artifacts,
    tmp_path,
):
    _, bundles, _ = emitted_split_artifacts
    expectations = bundles["development"]["expectations"]
    path = tmp_path / "development.expectations.json"

    expectations.write(path)
    restored = SplitArtifactExpectations.from_path(path)

    assert restored == expectations
    assert restored.to_dict()["version"] == 1
    assert path.read_bytes() == expectations.canonical_bytes()
    assert expectations.sha256() == hashlib.sha256(
        expectations.canonical_bytes()
    ).hexdigest()


def test_split_expectations_reject_unknown_serialization_versions(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    value = bundles["development"]["expectations"].to_dict()
    value["version"] = 2

    with pytest.raises(ValueError, match="expectation version"):
        SplitArtifactExpectations.from_dict(value)


def test_world_signature_rejects_synchronized_fact_and_manifest_omission(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    mutated_world = replace(
        world,
        facts=world.facts[:-1],
        manifest={
            **world.manifest,
            "fact_count": len(world.facts) - 1,
        },
    )

    with pytest.raises(
        ValueError,
        match="world artifact signature mismatch",
    ):
        _observe_partition(
            "development",
            {**development, "worlds": (mutated_world,)},
        )


def test_world_signature_rejects_synchronized_seed_substitution(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    mutated_world = replace(
        world,
        world_seed=world.world_seed + 1,
        manifest={
            **world.manifest,
            "world_seed": world.world_seed + 1,
        },
    )

    with pytest.raises(
        ValueError,
        match="world artifact signature mismatch",
    ):
        _observe_partition(
            "development",
            {**development, "worlds": (mutated_world,)},
        )


def test_qa_signature_rejects_same_id_same_hop_content_substitution(
    emitted_split_artifacts,
):
    plan, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    item = next(
        item
        for item in development["qa_items"]
        if item.task == "path_composition"
    )
    relations = next(
        path
        for path in plan.train.compositions
        if len(path) == item.meta["hop_count"]
        and tuple(item.meta["relations"]) != path
    )
    qa_items = _replace_qa_metadata(
        development["qa_items"],
        item.qid,
        {
            "relations": list(relations),
            "relation_path_hash": composition_hash(relations),
        },
    )

    with pytest.raises(ValueError, match="QA artifact signature mismatch"):
        _observe_partition(
            "development",
            {**development, "qa_items": qa_items},
        )


def test_qa_signature_rejects_same_id_world_id_substitution(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    item = development["qa_items"][0]
    qa_items = _replace_qa_metadata(
        development["qa_items"],
        item.qid,
        {"world_id": item.meta["world_id"] + 1},
    )

    with pytest.raises(ValueError, match="QA artifact signature mismatch"):
        _observe_partition(
            "development",
            {**development, "qa_items": qa_items},
        )


def test_qa_signature_rejects_ordered_relation_substitution(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    item = next(
        item
        for item in development["qa_items"]
        if item.task == "path_composition"
        and len(item.meta["relations"]) > 1
        and tuple(item.meta["relations"])
        != tuple(reversed(item.meta["relations"]))
    )
    relations = tuple(reversed(item.meta["relations"]))
    qa_items = _replace_qa_metadata(
        development["qa_items"],
        item.qid,
        {
            "relations": list(relations),
            "relation_path_hash": composition_hash(relations),
        },
    )

    with pytest.raises(ValueError, match="QA artifact signature mismatch"):
        _observe_partition(
            "development",
            {**development, "qa_items": qa_items},
        )


def test_rendered_signature_rejects_same_id_template_substitution(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    record = development["rendered_records"][0]
    records = _replace_rendered_metadata(
        development["rendered_records"],
        record.schedule.record_id,
        {"template_id": f"{record.metadata['template_id']}:substituted"},
    )

    with pytest.raises(
        ValueError,
        match="rendered artifact signature mismatch",
    ):
        _observe_partition(
            "development",
            {**development, "rendered_records": records},
        )


@pytest.mark.parametrize("collection", ["qa_items", "rendered_records"])
def test_artifact_collection_rejects_omitted_expected_ids(
    emitted_split_artifacts,
    collection,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]

    with pytest.raises(ValueError, match="missing"):
        _observe_partition(
            "development",
            {
                **development,
                collection: development[collection][:-1],
            },
        )


@pytest.mark.parametrize("collection", ["qa_items", "rendered_records"])
def test_artifact_collection_rejects_duplicate_ids(
    emitted_split_artifacts,
    collection,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]

    with pytest.raises(ValueError, match="duplicate"):
        _observe_partition(
            "development",
            {
                **development,
                collection: (
                    *development[collection],
                    development[collection][0],
                ),
            },
        )


@pytest.mark.parametrize("collection", ["qa_items", "rendered_records"])
def test_artifact_collection_rejects_extra_ids(
    emitted_split_artifacts,
    collection,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    if collection == "qa_items":
        original = development[collection][0]
        extra = replace(original, qid=f"{original.qid}-extra")
    else:
        original = development[collection][0]
        extra_id = f"{original.schedule.record_id}-extra"
        extra = replace(
            original,
            schedule=replace(original.schedule, record_id=extra_id),
            metadata={**original.metadata, "question_id": extra_id},
        )

    with pytest.raises(ValueError, match="extra"):
        _observe_partition(
            "development",
            {
                **development,
                collection: (*development[collection], extra),
            },
        )


def test_artifact_collection_rejects_duplicate_world_ids(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]

    with pytest.raises(ValueError, match="duplicate world"):
        _observe_partition(
            "development",
            {**development, "worlds": (world, world)},
        )


@pytest.mark.parametrize("mutation", ["omission", "extra"])
def test_artifact_collection_requires_exact_world_ids(
    emitted_split_artifacts,
    mutation,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    if mutation == "omission":
        worlds = ()
        expected_message = "missing"
    else:
        extra_world = replace(
            world,
            world_id=world.world_id + 1,
            world_seed=world.world_seed + 1,
            manifest={
                **world.manifest,
                "world_id": world.world_id + 1,
                "world_seed": world.world_seed + 1,
            },
        )
        worlds = (world, extra_world)
        expected_message = "extra"

    with pytest.raises(ValueError, match=expected_message):
        _observe_partition(
            "development",
            {**development, "worlds": worlds},
        )


def test_artifact_collection_rejects_duplicate_fact_ids(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    mutated_world = replace(
        world,
        facts=(*world.facts, world.facts[0]),
        manifest={
            **world.manifest,
            "fact_count": len(world.facts) + 1,
        },
    )

    with pytest.raises(ValueError, match="duplicate fact"):
        _observe_partition(
            "development",
            {**development, "worlds": (mutated_world,)},
        )


def test_artifact_collection_rejects_duplicate_graph_addresses(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    facts = list(world.facts)
    facts[1] = replace(
        facts[1],
        row=replace(
            facts[1].row,
            source_id=facts[0].row.source_id,
            relation_id=facts[0].row.relation_id,
            direction=facts[0].row.direction,
        ),
    )
    mutated_world = replace(world, facts=tuple(facts))

    with pytest.raises(ValueError, match="duplicate graph address"):
        _observe_partition(
            "development",
            {**development, "worlds": (mutated_world,)},
        )


def test_artifact_collection_rejects_manifest_fact_count_mismatch(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    mutated_world = replace(
        world,
        manifest={
            **world.manifest,
            "fact_count": len(world.facts) - 1,
        },
    )

    with pytest.raises(ValueError, match="manifest fact count"):
        _observe_partition(
            "development",
            {**development, "worlds": (mutated_world,)},
        )


def test_artifact_collection_rejects_rendered_record_id_mismatch(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    records = list(development["rendered_records"])
    records[0] = replace(
        records[0],
        metadata={
            **records[0].metadata,
            "question_id": f"{records[0].schedule.record_id}-mismatch",
        },
    )

    with pytest.raises(ValueError, match="schedule record ID"):
        _observe_partition(
            "development",
            {**development, "rendered_records": tuple(records)},
        )


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_artifact_collection_rejects_incorrect_rendered_hop_coverage(
    emitted_split_artifacts,
    mutation,
):
    plan, bundles, _ = emitted_split_artifacts
    if mutation == "missing":
        name = "protected_heldout"
        bundle = bundles[name]
        original_hop = 6
        replacement_path = next(
            path
            for path in plan.protected_heldout.compositions
            if len(path) == 5
        )
    else:
        name = "development"
        bundle = bundles[name]
        original_hop = 4
        replacement_path = next(
            path
            for path in plan.protected_heldout.compositions
            if len(path) == 5
        )
    def substitute_hop(records):
        mutated = list(records)
        for index, record in enumerate(mutated):
            if record.metadata["hop_count"] != original_hop:
                continue
            mutated[index] = replace(
                record,
                metadata={
                    **record.metadata,
                    "hop_count": len(replacement_path),
                    "relations": list(replacement_path),
                    "relation_path_hash": composition_hash(replacement_path),
                    "composition_split": "heldout",
                },
            )
            if mutation == "unexpected":
                break
        return tuple(mutated)

    mutated_bundle = _bundle_with_frozen_rendered_records(
        name,
        bundle,
        rendered_records=substitute_hop(bundle["rendered_records"]),
        snapshot_rendered_records=substitute_hop(
            bundle["snapshot"]["rendered_records"]
        ),
    )

    with pytest.raises(ValueError, match="rendered hop coverage"):
        _observe_partition(name, mutated_bundle)


def test_artifact_collection_rejects_world_manifest_seed_mismatch(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    mutated_world = replace(
        world,
        manifest={
            **world.manifest,
            "world_seed": world.world_seed + 1,
        },
    )

    with pytest.raises(ValueError, match="world seed.*manifest"):
        _observe_partition(
            "development",
            {**development, "worlds": (mutated_world,)},
        )


def test_artifact_gate_rejects_emitted_entity_id_collision(
    emitted_split_artifacts,
):
    _, bundles, observed = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    fact_index = next(
        index
        for index, fact in enumerate(world.facts)
        if fact.row.target_kind == "literal"
    )
    row = world.facts[fact_index].row
    train_entity = next(iter(observed["train"].entity_ids))
    mutated_world = _replace_world_fact(
        world,
        fact_index,
        replace(row, source_id=train_entity),
    )
    snapshot_world = development["snapshot"]["worlds"][0]
    mutated_snapshot_world = _replace_world_fact(
        snapshot_world,
        fact_index,
        replace(
            snapshot_world.facts[fact_index].row,
            source_id=train_entity,
        ),
    )
    mutated_bundle = _bundle_with_frozen_world_change(
        "development",
        development,
        worlds=(mutated_world,),
        snapshot_worlds=(mutated_snapshot_world,),
    )
    mutated = _replace_observed_partition(
        observed,
        "development",
        mutated_bundle,
    )

    assert audit_disjointness(mutated)["entity_ids"] is False


@pytest.mark.parametrize("payload_source", ["target", "qualifier"])
def test_artifact_gate_rejects_emitted_payload_collision(
    emitted_split_artifacts,
    payload_source,
):
    _, bundles, observed = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    if payload_source == "target":
        fact_index = next(
            index
            for index, fact in enumerate(world.facts)
            if fact.row.target_kind == "literal"
        )
        row = world.facts[fact_index].row
        train_value = next(
            fact.row.target
            for fact in bundles["train"]["worlds"][0].facts
            if fact.row.target_kind == "literal"
        )
        mutated_row = replace(row, target=train_value)
    else:
        fact_index = next(
            index
            for index, fact in enumerate(world.facts)
            if fact.row.qualifiers
        )
        row = world.facts[fact_index].row
        train_value = next(
            fact.row.qualifiers[0][1]
            for fact in bundles["train"]["worlds"][0].facts
            if fact.row.qualifiers
        )
        mutated_row = replace(
            row,
            qualifiers=((row.qualifiers[0][0], train_value),),
        )
    mutated_world = _replace_world_fact(world, fact_index, mutated_row)
    snapshot_world = development["snapshot"]["worlds"][0]
    snapshot_row = snapshot_world.facts[fact_index].row
    if payload_source == "target":
        mutated_snapshot_row = replace(snapshot_row, target=train_value)
    else:
        mutated_snapshot_row = replace(
            snapshot_row,
            qualifiers=((snapshot_row.qualifiers[0][0], train_value),),
        )
    mutated_snapshot_world = _replace_world_fact(
        snapshot_world,
        fact_index,
        mutated_snapshot_row,
    )
    mutated_bundle = _bundle_with_frozen_world_change(
        "development",
        development,
        worlds=(mutated_world,),
        snapshot_worlds=(mutated_snapshot_world,),
    )
    mutated = _replace_observed_partition(
        observed,
        "development",
        mutated_bundle,
    )

    assert audit_disjointness(mutated)["payload_values"] is False


def test_artifact_gate_rejects_emitted_paraphrase_collision(
    emitted_split_artifacts,
):
    _, bundles, observed = emitted_split_artifacts
    development = bundles["development"]
    world = development["worlds"][0]
    assignments = dict(world.manifest["paraphrase_assignment_ids"])
    assignments[next(iter(assignments))] = next(
        iter(observed["train"].paraphrase_assignments)
    )
    mutated_world = replace(
        world,
        manifest={
            **world.manifest,
            "paraphrase_assignment_ids": assignments,
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "development",
        {**development, "worlds": (mutated_world,)},
    )

    assert audit_disjointness(mutated)["paraphrase_assignments"] is False


def test_artifact_collection_rejects_unexpected_question_id(
    emitted_split_artifacts,
):
    _, bundles, _ = emitted_split_artifacts
    development = bundles["development"]
    qa_items = list(development["qa_items"])
    qa_items[0] = replace(
        qa_items[0],
        qid=bundles["train"]["qa_items"][0].qid,
    )

    with pytest.raises(ValueError, match="QA question IDs"):
        _observe_partition(
            "development",
            {**development, "qa_items": tuple(qa_items)},
        )


def test_artifact_gate_rejects_emitted_rendered_path_hash_collision(
    emitted_split_artifacts,
):
    plan, bundles, observed = emitted_split_artifacts
    development = bundles["development"]
    record = next(
        record
        for record in development["rendered_records"]
        if record.metadata["template_id"] == "path_composition:v1"
    )
    relations = next(
        path
        for path in plan.train.compositions
        if len(path) == record.metadata["hop_count"]
    )
    mutated_bundle = _bundle_with_frozen_rendered_change(
        "development",
        development,
        record.schedule.record_id,
        {
            "relations": list(relations),
            "relation_path_hash": composition_hash(relations),
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "development",
        mutated_bundle,
    )

    assert audit_disjointness(mutated)["relation_path_hashes"] is False


def test_artifact_gate_rejects_emitted_heldout_composition_collision(
    emitted_split_artifacts,
):
    _, bundles, observed = emitted_split_artifacts
    heldout = bundles["protected_heldout"]
    train_path = next(
        item
        for item in bundles["train"]["qa_items"]
        if item.task == "path_composition"
    )
    heldout_item = next(
        item
        for item in heldout["qa_items"]
        if item.meta["composition_split"] == "heldout"
    )
    relations = tuple(train_path.meta["relations"])
    mutated_bundle = _bundle_with_frozen_qa_change(
        "protected_heldout",
        heldout,
        heldout_item.qid,
        {
            "relations": list(relations),
            "relation_path_hash": composition_hash(relations),
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "protected_heldout",
        mutated_bundle,
    )

    assert (
        audit_disjointness(mutated)["heldout_relation_compositions"] is False
    )


def test_artifact_gate_reads_heldout_items_from_protected_seen_view(
    emitted_split_artifacts,
):
    plan, bundles, observed = emitted_split_artifacts
    protected_seen = bundles["protected_seen"]
    item = next(
        item
        for item in protected_seen["qa_items"]
        if item.task == "path_composition"
        and item.meta["composition_split"] == "heldout"
        and item.meta["hop_count"] <= 4
    )
    train_path = next(
        path
        for path in plan.train.compositions
        if len(path) == item.meta["hop_count"]
    )
    mutated_bundle = _bundle_with_frozen_qa_change(
        "protected_seen",
        protected_seen,
        item.qid,
        {
            "relations": list(train_path),
            "relation_path_hash": composition_hash(train_path),
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "protected_seen",
        mutated_bundle,
    )
    report = audit_disjointness(mutated)

    assert report["relation_path_hashes"] is False
    assert report["heldout_relation_compositions"] is False


def test_artifact_gate_reads_seen_items_from_protected_heldout_view(
    emitted_split_artifacts,
):
    plan, bundles, observed = emitted_split_artifacts
    protected_heldout = bundles["protected_heldout"]
    item = next(
        item
        for item in protected_heldout["qa_items"]
        if item.task == "path_composition"
        and item.meta["composition_split"] == "seen"
    )
    unknown_path = next(
        path
        for path in plan.protected_heldout.compositions
        if len(path) == item.meta["hop_count"]
    )
    mutated_bundle = _bundle_with_frozen_qa_change(
        "protected_heldout",
        protected_heldout,
        item.qid,
        {
            "relations": list(unknown_path),
            "relation_path_hash": composition_hash(unknown_path),
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "protected_heldout",
        mutated_bundle,
    )
    report = audit_disjointness(mutated)

    assert report["relation_path_hashes"] is False
    assert report["heldout_relation_compositions"] is False


def test_artifact_gate_rejects_heldout_hash_in_mislabeled_training_record(
    emitted_split_artifacts,
):
    plan, bundles, observed = emitted_split_artifacts
    train = bundles["train"]
    heldout_path = plan.protected_heldout.compositions[0]
    record = next(
        record
        for record in train["rendered_records"]
        if record.metadata["template_id"] == "path_composition:v1"
        and record.metadata["hop_count"] == len(heldout_path)
    )
    mutated_bundle = _bundle_with_frozen_rendered_change(
        "train",
        train,
        record.schedule.record_id,
        {
            "relations": list(heldout_path),
            "relation_path_hash": composition_hash(heldout_path),
            "composition_split": "heldout",
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "train",
        mutated_bundle,
    )
    report = audit_disjointness(mutated)

    assert report["relation_path_hashes"] is False
    assert report["heldout_relation_compositions"] is False


def test_artifact_gate_rejects_heldout_hash_in_mislabeled_training_qa(
    emitted_split_artifacts,
):
    plan, bundles, observed = emitted_split_artifacts
    train = bundles["train"]
    heldout_path = plan.protected_heldout.compositions[0]
    item = next(
        item
        for item in train["qa_items"]
        if item.task == "path_composition"
        and item.meta["hop_count"] == len(heldout_path)
    )
    mutated_bundle = _bundle_with_frozen_qa_change(
        "train",
        train,
        item.qid,
        {
            "relations": list(heldout_path),
            "relation_path_hash": composition_hash(heldout_path),
            "composition_split": "heldout",
        },
    )
    mutated = _replace_observed_partition(
        observed,
        "train",
        mutated_bundle,
    )
    report = audit_disjointness(mutated)

    assert report["relation_path_hashes"] is False
    assert report["heldout_relation_compositions"] is False
