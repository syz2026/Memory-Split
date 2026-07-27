from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Iterator

from corpusgen.graph_records import (
    GraphAction,
    GraphAddress,
    GraphRow,
    RenderedRecord,
    ScheduleEntry,
    SelectorFeatures,
    TaggedSegment,
    relative_position_bin,
    stable_fact_id,
)
from corpusgen.graph_trace import serialize_action, serialize_return
from corpusgen.mask_ledger import RandomMaskUndersupplyError
from corpusgen.relation_codec import RelationCodec
from corpusgen.relation_schema import RelationSchema, RelationSpec
from corpusgen.records import QAItem
from corpusgen.world_splits import (
    TYPE_METADATA_REASON,
    SplitName,
    SplitPlan,
    composition_hash,
)


ENTITY_RELATIONS = tuple(f"r{index}" for index in range(4))
DATE_RELATION = "r4"
CATEGORY_RELATION = "r5"
SRGM_RELATION_IDS = (*ENTITY_RELATIONS, DATE_RELATION, CATEGORY_RELATION)
SRGM_RELATION_CODEC = RelationCodec(SRGM_RELATION_IDS)
CURRICULUM_HOPS = (1, 2, 4)
MIN_REASONING_ENTITIES = 16
_DEFAULT_WORLD_ENTITY_STRIDE = 1 << 64


@dataclass(frozen=True)
class WorldConfig:
    n_entities: int = 64
    seed: int = 0
    relation_count: int = 4
    entity_id_offset: int | None = None
    schema: RelationSchema | None = None
    split_plan: SplitPlan | None = None
    split_name: SplitName | None = None
    world_seed_offset: int | None = None

    def __post_init__(self) -> None:
        if self.n_entities <= 0:
            raise ValueError("n_entities must be positive")
        if self.entity_id_offset is not None and self.entity_id_offset < 0:
            raise ValueError("entity_id_offset must be non-negative")
        if (self.schema is None) != (self.split_plan is None):
            raise ValueError("schema and split_plan must be provided together")
        if self.schema is None:
            if self.relation_count != len(ENTITY_RELATIONS):
                raise ValueError("relation_count must be 4")
            if self.split_name is not None or self.world_seed_offset is not None:
                raise ValueError(
                    "legacy worlds cannot use split namespace fields"
                )
            if (
                self.entity_id_offset is None
                and self.n_entities > _DEFAULT_WORLD_ENTITY_STRIDE
            ):
                raise ValueError(
                    "default-offset worlds exceed the entity stride"
                )
            return

        assert self.split_plan is not None
        if self.schema.sha256() != self.split_plan.schema_sha256:
            raise ValueError("split plan does not match relation schema")
        self.split_plan.require_static_disjointness()
        if self.split_name is None:
            raise ValueError("schema-shaped worlds require a split_name")
        partition = self.split_plan.partition(self.split_name)
        if self.entity_id_offset is None:
            raise ValueError(
                "schema-shaped worlds require an explicit entity_id_offset"
            )
        if not (
            partition.entity_id_range[0] <= self.entity_id_offset
            and self.entity_id_offset + self.n_entities
            <= partition.entity_id_range[1]
        ):
            raise ValueError("entity namespace offset is outside its split")
        if self.world_seed_offset is None:
            raise ValueError(
                "schema-shaped worlds require an explicit world_seed_offset"
            )
        if not (
            partition.world_seed_range[0]
            <= self.world_seed_offset
            < partition.world_seed_range[1]
        ):
            raise ValueError("world seed offset is outside its split namespace")


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    row: GraphRow
    features: SelectorFeatures
    audit_class: str


@dataclass(frozen=True)
class GraphWorld:
    world_id: int
    entity_names: tuple[str, ...]
    facts: tuple[GraphFact, ...]
    schema: RelationSchema | None = None
    split_plan: SplitPlan | None = None
    split_name: SplitName | None = None
    world_seed: int | None = None
    reasoning_entity_ids: tuple[int, ...] = ()
    manifest: dict = field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        value = {
            "world_id": self.world_id,
            "world_seed": self.world_seed,
            "split_name": self.split_name,
            "entity_names": list(self.entity_names),
            "reasoning_entity_ids": list(self.reasoning_entity_ids),
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "row": fact.row.as_json(),
                    "features": asdict(fact.features),
                    "audit_class": fact.audit_class,
                }
                for fact in self.facts
            ],
            "manifest": self.manifest,
        }
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )


@dataclass(frozen=True)
class CounterfactualPair:
    task: str
    original: QAItem
    counterfactual: QAItem
    changed_row: GraphRow


def _features(
    exposure: int,
    entropy: float,
    queries: float,
    centrality: float,
) -> SelectorFeatures:
    return SelectorFeatures(
        log_exposure=math.log1p(exposure),
        payload_entropy=entropy,
        payload_tokens=1.0,
        expected_queries=queries,
        path_centrality=centrality,
    )


def _random_date(rng: random.Random) -> str:
    return (
        f"{1930 + rng.randrange(76):04d}-"
        f"{1 + rng.randrange(12):02d}-"
        f"{1 + rng.randrange(28):02d}"
    )


def _category_values(rng: random.Random, n_entities: int) -> list[str]:
    category_count = min(64, max(2, n_entities // 2))
    labels = rng.sample(range(64), category_count)
    values = [
        f"category-{labels[index % category_count]}"
        for index in range(n_entities)
    ]
    rng.shuffle(values)
    return values


def _generate_legacy_world(world_id: int, cfg: WorldConfig) -> GraphWorld:
    if world_id < 0:
        raise ValueError("world_id must be non-negative")

    entity_id_offset = (
        world_id * _DEFAULT_WORLD_ENTITY_STRIDE
        if cfg.entity_id_offset is None
        else cfg.entity_id_offset
    )
    rng = random.Random((cfg.seed << 32) ^ world_id)
    names = tuple(
        f"entity-{entity_id_offset + local_id}-{rng.getrandbits(32):08x}"
        for local_id in range(cfg.n_entities)
    )
    facts: list[GraphFact] = []

    for relation_index, relation_id in enumerate(ENTITY_RELATIONS):
        targets = list(range(cfg.n_entities))
        rng.shuffle(targets)
        for source_id, target_id in enumerate(targets):
            global_source = entity_id_offset + source_id
            global_target = entity_id_offset + target_id
            compose = rng.randrange(4)
            row = GraphRow(
                source_id=global_source,
                relation_id=relation_id,
                direction="out",
                target_kind="entity",
                target=str(global_target),
                qualifiers=(("compose", str(compose)),),
                provenance_id=f"world-{world_id}",
            )
            centrality = (
                1.0
                if source_id < max(1, cfg.n_entities // 4)
                else 0.1
            )
            facts.append(
                GraphFact(
                    fact_id=stable_fact_id(row),
                    row=row,
                    features=_features(
                        exposure=4,
                        entropy=math.log2(cfg.n_entities),
                        queries=1.0,
                        centrality=centrality,
                    ),
                    audit_class=(
                        "central" if centrality == 1.0 else "peripheral"
                    ),
                )
            )

    dates = [_random_date(rng) for _ in range(cfg.n_entities)]
    if cfg.n_entities > 1 and len(set(dates)) == 1:
        dates[-1] = "2099-12-31"
    categories = _category_values(rng, cfg.n_entities)
    for source_id, (date, category) in enumerate(zip(dates, categories)):
        global_source = entity_id_offset + source_id
        for relation_index, relation_id, value, entropy in (
            (4, DATE_RELATION, date, 14.7),
            (5, CATEGORY_RELATION, category, 6.0),
        ):
            row = GraphRow(
                source_id=global_source,
                relation_id=relation_id,
                direction="out",
                target_kind="literal",
                target=value,
                provenance_id=f"world-{world_id}",
            )
            facts.append(
                GraphFact(
                    fact_id=stable_fact_id(row),
                    row=row,
                    features=_features(
                        exposure=2,
                        entropy=entropy,
                        queries=0.25,
                        centrality=0.05,
                    ),
                    audit_class="peripheral",
                )
            )

    return GraphWorld(
        world_id=world_id,
        entity_names=names,
        facts=tuple(facts),
    )


def _rounded_ratio(numerator: int, denominator: int, scale: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * scale + denominator // 2) // denominator


def _schema_relation_specs(
    schema: RelationSchema,
) -> tuple[tuple[RelationSpec, ...], tuple[RelationSpec, ...]]:
    entity_specs = tuple(
        spec for spec in schema.path_relations if spec.target_kind == "entity"
    )
    literal_specs = tuple(
        spec for spec in schema.path_relations if spec.target_kind != "entity"
    )
    if not entity_specs:
        raise ValueError("schema-shaped worlds require entity path relations")
    date_count = sum(spec.target_kind == "date" for spec in literal_specs)
    category_count = sum(
        spec.target_kind == "category" for spec in literal_specs
    )
    if date_count < 2 or category_count < 2:
        raise ValueError(
            "schema-shaped worlds require two date and two category relations"
        )
    return entity_specs, literal_specs


def _literal_value(
    spec: RelationSpec,
    *,
    payload_namespace: str,
    local_id: int,
    relation_index: int,
    n_entities: int,
) -> str:
    prefix = f"{payload_namespace}:{spec.relation_id}"
    if spec.target_kind == "date":
        year = 1000 + (local_id * 37 + relation_index * 101) % 8000
        month = 1 + (local_id * 7 + relation_index) % 12
        day = 1 + (local_id * 11 + relation_index) % 28
        return f"{prefix}:{year:04d}-{month:02d}-{day:02d}"
    if spec.target_kind == "quantity":
        value = local_id * 104_729 + relation_index * 10_007
        return f"{prefix}:{value:020d}"
    if spec.target_kind == "category":
        category_count = min(64, max(2, n_entities // 2))
        return f"{prefix}:{local_id % category_count:04d}"
    if spec.target_kind == "string":
        return f"{prefix}:value-{local_id:012d}"
    raise ValueError(f"unsupported literal target kind: {spec.target_kind}")


def _generate_schema_world(world_id: int, cfg: WorldConfig) -> GraphWorld:
    assert cfg.schema is not None
    assert cfg.split_plan is not None
    assert cfg.split_name is not None
    assert cfg.entity_id_offset is not None
    assert cfg.world_seed_offset is not None
    partition = cfg.split_plan.partition(cfg.split_name)
    global_world_id = cfg.world_seed_offset + world_id
    if global_world_id >= partition.world_seed_range[1]:
        raise ValueError("world id exceeds its split seed namespace")
    provenance = f"{partition.namespace}:world:{global_world_id}"
    rng = random.Random(global_world_id)
    ordered_local_ids = list(range(cfg.n_entities))
    rng.shuffle(ordered_local_ids)
    entity_specs, literal_specs = _schema_relation_specs(cfg.schema)
    required_paths = partition.compositions
    if cfg.split_name in {"protected_seen", "protected_heldout"}:
        required_paths = (
            cfg.split_plan.protected_seen.compositions
            + cfg.split_plan.protected_heldout.compositions
        )
    required_relations = {
        relation_id
        for path in required_paths
        for relation_id in path
    }

    source_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    rounded_source_counts: dict[str, int] = {}
    rounded_target_counts: dict[str, int] = {}
    for spec in entity_specs:
        if (
            spec.distinct_subjects is None
            or spec.distinct_objects is None
            or spec.entity_count is None
        ):
            raise ValueError(
                f"entity relation {spec.relation_id} lacks observed statistics"
            )
        rounded_source_count = _rounded_ratio(
            spec.distinct_subjects,
            spec.entity_count,
            cfg.n_entities,
        )
        rounded_target_count = _rounded_ratio(
            spec.distinct_objects,
            spec.distinct_subjects,
            rounded_source_count,
        )
        source_count = rounded_source_count
        if spec.relation_id in required_relations:
            source_count = max(1, source_count)
        target_count = _rounded_ratio(
            spec.distinct_objects,
            spec.distinct_subjects,
            source_count,
        )
        if source_count and spec.distinct_objects and target_count == 0:
            target_count = 1
        if source_count and target_count == 0:
            raise ValueError(
                f"entity relation {spec.relation_id} has no target pool"
            )
        if target_count > source_count:
            raise ValueError(
                f"entity relation {spec.relation_id} target-pool ratio "
                "cannot be represented by a functional world"
            )
        source_counts[spec.relation_id] = source_count
        target_counts[spec.relation_id] = target_count
        rounded_source_counts[spec.relation_id] = rounded_source_count
        rounded_target_counts[spec.relation_id] = rounded_target_count

    unknown_required = required_relations - set(source_counts)
    if unknown_required:
        raise ValueError(
            f"split plan uses unknown entity relations: "
            f"{sorted(unknown_required)!r}"
        )
    backbone_count = min(
        source_counts[relation_id] for relation_id in required_relations
    )
    if backbone_count <= 0:
        raise ValueError(
            "observed subject coverage cannot supply a reasoning backbone"
        )
    backbone_local_ids = tuple(ordered_local_ids[:backbone_count])
    backbone_set = set(backbone_local_ids)

    facts: list[GraphFact] = []
    relation_shapes: list[dict[str, object]] = []
    for relation_index, spec in enumerate(entity_specs):
        source_count = source_counts[spec.relation_id]
        target_count = target_counts[spec.relation_id]
        source_local_ids = ordered_local_ids[:source_count]
        target_pool = ordered_local_ids[:target_count]
        safe_target_count = min(target_count, backbone_count)
        safe_targets = target_pool[:safe_target_count]
        extra_targets = target_pool[safe_target_count:]
        nonbackbone_ordinal = 0

        for source_ordinal, source_local_id in enumerate(source_local_ids):
            if source_local_id in backbone_set:
                target_local_id = safe_targets[
                    source_ordinal % len(safe_targets)
                ]
            else:
                if nonbackbone_ordinal < len(extra_targets):
                    target_local_id = extra_targets[nonbackbone_ordinal]
                else:
                    target_local_id = target_pool[
                        (nonbackbone_ordinal - len(extra_targets))
                        % len(target_pool)
                    ]
                nonbackbone_ordinal += 1
            global_source = cfg.entity_id_offset + source_local_id
            global_target = cfg.entity_id_offset + target_local_id
            compose_code = (
                global_world_id
                + relation_index
                + source_ordinal
            ) % 4
            compose = partition.entity_id_range[0] * 4 + compose_code
            row = GraphRow(
                source_id=global_source,
                relation_id=spec.relation_id,
                direction="out",
                target_kind="entity",
                target=str(global_target),
                qualifiers=(("compose", str(compose)),),
                provenance_id=provenance,
            )
            central = (
                spec.relation_id in required_relations
                and source_local_id
                in backbone_set
                and backbone_local_ids.index(source_local_id)
                < max(1, backbone_count // 4)
            )
            facts.append(
                GraphFact(
                    fact_id=stable_fact_id(row),
                    row=row,
                    features=_features(
                        exposure=4,
                        entropy=math.log2(max(target_count, 1)),
                        queries=1.0,
                        centrality=1.0 if central else 0.1,
                    ),
                    audit_class="central" if central else "peripheral",
                )
            )
        relation_shapes.append(
            {
                "relation_id": spec.relation_id,
                "observed_subject_coverage": spec.subject_coverage,
                "observed_target_pool_ratio": spec.target_pool_ratio,
                "generated_distinct_subjects": source_count,
                "generated_distinct_objects": target_count,
                "rounded_distinct_subjects": rounded_source_counts[
                    spec.relation_id
                ],
                "rounded_distinct_objects": rounded_target_counts[
                    spec.relation_id
                ],
                "backbone_subjects_added": (
                    source_count
                    - rounded_source_counts[spec.relation_id]
                ),
                "backbone_objects_added": (
                    target_count
                    - rounded_target_counts[spec.relation_id]
                ),
            }
        )

    for relation_index, spec in enumerate(literal_specs):
        for local_id in range(cfg.n_entities):
            row = GraphRow(
                source_id=cfg.entity_id_offset + local_id,
                relation_id=spec.relation_id,
                direction="out",
                target_kind="literal",
                target=_literal_value(
                    spec,
                    payload_namespace=partition.payload_namespace,
                    local_id=local_id,
                    relation_index=relation_index,
                    n_entities=cfg.n_entities,
                ),
                provenance_id=provenance,
            )
            facts.append(
                GraphFact(
                    fact_id=stable_fact_id(row),
                    row=row,
                    features=_features(
                        exposure=2,
                        entropy=math.log2(max(cfg.n_entities, 2)),
                        queries=0.25,
                        centrality=0.05,
                    ),
                    audit_class="peripheral",
                )
            )

    alias_assignments: dict[str, str] = {}
    paraphrase_assignment_ids: dict[str, str] = {}
    for spec in (*entity_specs, *literal_specs):
        alias_index = (
            int(
                composition_hash(
                    (
                        partition.paraphrase_namespace,
                        spec.relation_id,
                    )
                )[:8],
                16,
            )
            % len(spec.aliases)
        )
        alias_assignments[spec.relation_id] = spec.aliases[alias_index]
        paraphrase_assignment_ids[spec.relation_id] = (
            f"{partition.paraphrase_namespace}:"
            f"{spec.relation_id}:{alias_index}"
        )
    manifest = {
        "version": 1,
        "world_id": global_world_id,
        "world_seed": global_world_id,
        "split_name": cfg.split_name,
        "namespace": partition.namespace,
        "schema_sha256": cfg.schema.sha256(),
        "split_plan_sha256": cfg.split_plan.sha256(),
        "entity_id_range": [
            cfg.entity_id_offset,
            cfg.entity_id_offset + cfg.n_entities,
        ],
        "payload_namespace": partition.payload_namespace,
        "paraphrase_namespace": partition.paraphrase_namespace,
        "question_namespace": partition.question_namespace,
        "paraphrase_assignments": alias_assignments,
        "paraphrase_assignment_ids": paraphrase_assignment_ids,
        "compositions": [list(path) for path in partition.compositions],
        "fact_count": len(facts),
        "relation_shapes": relation_shapes,
        "type_metadata": {
            "available": False,
            "reason": TYPE_METADATA_REASON,
        },
    }
    return GraphWorld(
        world_id=global_world_id,
        entity_names=tuple(
            f"Q{cfg.entity_id_offset + local_id}"
            for local_id in range(cfg.n_entities)
        ),
        facts=tuple(facts),
        schema=cfg.schema,
        split_plan=cfg.split_plan,
        split_name=cfg.split_name,
        world_seed=global_world_id,
        reasoning_entity_ids=tuple(
            cfg.entity_id_offset + local_id
            for local_id in backbone_local_ids
        ),
        manifest=manifest,
    )


def generate_world(world_id: int, cfg: WorldConfig) -> GraphWorld:
    if isinstance(world_id, bool) or not isinstance(world_id, int) or world_id < 0:
        raise ValueError("world_id must be non-negative")
    if cfg.schema is None:
        return _generate_legacy_world(world_id, cfg)
    return _generate_schema_world(world_id, cfg)


def _iter_world_sizes(
    n_entities: int,
    world_size: int,
) -> Iterator[int]:
    full_worlds, remainder = divmod(n_entities, world_size)
    if remainder == 0:
        yield from (world_size for _ in range(full_worlds))
        return
    if full_worlds == 0:
        yield remainder
        return
    if remainder >= MIN_REASONING_ENTITIES:
        yield from (world_size for _ in range(full_worlds - 1))
        combined = world_size + remainder
        first = combined // 2
        yield first
        yield combined - first
        return

    deficit = MIN_REASONING_ENTITIES - remainder
    donor_size = world_size - deficit
    yield from (world_size for _ in range(max(0, full_worlds - 1)))
    if donor_size >= MIN_REASONING_ENTITIES:
        yield donor_size
        yield MIN_REASONING_ENTITIES
    else:
        yield world_size + remainder


def iter_worlds(
    n_entities: int,
    world_size: int,
    seed: int,
    world_id_offset: int = 0,
) -> Iterator[GraphWorld]:
    if n_entities < MIN_REASONING_ENTITIES:
        raise ValueError(
            f"n_entities must be at least {MIN_REASONING_ENTITIES}"
        )
    if world_size < MIN_REASONING_ENTITIES:
        raise ValueError(
            f"world_size must be at least {MIN_REASONING_ENTITIES}"
        )
    if world_id_offset < 0:
        raise ValueError("world_id_offset must be non-negative")

    def worlds() -> Iterator[GraphWorld]:
        entity_id_offset = world_id_offset * world_size
        for ordinal, size in enumerate(
            _iter_world_sizes(n_entities, world_size)
        ):
            world_id = world_id_offset + ordinal
            yield generate_world(
                world_id,
                WorldConfig(
                    n_entities=size,
                    seed=seed,
                    entity_id_offset=entity_id_offset,
                ),
            )
            entity_id_offset += size

    return worlds()


def _row_map(world: GraphWorld) -> dict[GraphAddress, GraphFact]:
    rows = {fact.row.address: fact for fact in world.facts}
    if len(rows) != len(world.facts):
        raise ValueError("world contains duplicate functional addresses")
    return rows


def _entity_ids(world: GraphWorld) -> tuple[int, ...]:
    ids = tuple(sorted({fact.row.source_id for fact in world.facts}))
    if len(ids) != len(world.entity_names):
        raise ValueError("entity_names must align one-to-one with source ids")
    return ids


def _relation_for_kind(world: GraphWorld, target_kind: str) -> str:
    if world.schema is None:
        if target_kind == "date":
            return DATE_RELATION
        if target_kind == "category":
            return CATEGORY_RELATION
        raise ValueError(f"legacy worlds lack {target_kind!r} relations")
    matching = tuple(
        spec.relation_id
        for spec in world.schema.path_relations
        if spec.target_kind == target_kind
    )
    if matching:
        if world.split_name == "development" and len(matching) > 1:
            return matching[-1]
        return matching[0]
    raise ValueError(f"world schema lacks a {target_kind!r} relation")


def _relation_surface(world: GraphWorld, relation_id: str) -> str:
    assignments = world.manifest.get("paraphrase_assignments", {})
    if isinstance(assignments, dict):
        value = assignments.get(relation_id)
        if isinstance(value, str) and value:
            return value
    return relation_id


def _path_compositions(
    world: GraphWorld,
    path_hops: int | None,
) -> tuple[tuple[str, ...], ...]:
    if world.split_plan is None or world.split_name is None:
        return ()
    if world.split_name in {"protected_seen", "protected_heldout"}:
        paths = (
            world.split_plan.protected_seen.compositions
            + world.split_plan.protected_heldout.compositions
        )
    else:
        paths = world.split_plan.partition(world.split_name).compositions
    if path_hops is not None:
        if world.split_name in {"development", "train"}:
            paths = tuple(path for path in paths if len(path) <= path_hops)
        else:
            paths = tuple(path for path in paths if len(path) == path_hops)
    if not paths:
        raise ValueError("split plan has no composition for requested hop count")
    return paths


def _follow(
    rows: dict[GraphAddress, GraphFact],
    start: int,
    relations: tuple[str, ...],
) -> tuple[int, int, tuple[GraphFact, ...]]:
    current = start
    used: list[GraphFact] = []
    compose = 0
    for relation in relations:
        fact = rows[GraphAddress(current, relation, "out")]
        if fact.row.target_kind != "entity":
            raise ValueError("path relations must target entities")
        used.append(fact)
        compose = (
            compose + int(dict(fact.row.qualifiers)["compose"])
        ) % 4
        current = int(fact.row.target)
    return current, compose, tuple(used)


def _item(
    *,
    qid: str,
    task: str,
    prompt: str,
    answer: str,
    pair_id: str,
    graph_rows: int,
    world_id: int,
    composition_split: str,
    variant: str,
    entity_slots: tuple[int | None, ...],
    gold_facts: tuple[GraphFact, ...],
    source_slots: tuple[int, ...],
    answer_choices: tuple[str, ...],
    changed_row: GraphRow | None = None,
    relations: tuple[str, ...] = (),
) -> QAItem:
    if len(source_slots) != len(gold_facts):
        raise ValueError("source_slots must align with gold facts")
    provenances = {fact.row.provenance_id for fact in gold_facts}
    if len(provenances) != 1:
        raise ValueError("gold facts must share one exact provenance partition")
    provenance_id = next(iter(provenances))
    if not provenance_id:
        raise ValueError("gold facts require a nonempty provenance_id")
    if (
        changed_row is not None
        and changed_row.provenance_id != provenance_id
    ):
        raise ValueError("counterfactual rows cannot cross provenance partitions")
    read_actions = [
        GraphAction(
            source_slot=source_slot,
            relation_id=fact.row.relation_id,
            direction=fact.row.direction,
            read=True,
            halt=False,
        )
        for source_slot, fact in zip(source_slots, gold_facts)
    ]
    gold_actions = _complete_action_plan(read_actions)

    def action_json(action: GraphAction) -> dict:
        return {
            "source_slot": action.source_slot,
            "relation_id": action.relation_id,
            "direction": action.direction,
            "read": action.read,
            "halt": action.halt,
        }

    return QAItem(
        qid=qid,
        task=task,
        prompt=prompt,
        answer=answer,
        meta={
            "pair_id": pair_id,
            "template": task,
            "template_id": f"{task}:v1",
            "graph_rows": graph_rows,
            "world_id": world_id,
            "provenance_id": provenance_id,
            "relation_path_hash": composition_hash(relations),
            "composition_split": composition_split,
            "hop_count": len(gold_facts),
            "variant": variant,
            "changed_row": (
                None if changed_row is None else changed_row.as_json()
            ),
            "entity_slots": list(entity_slots),
            "gold_addresses": [
                [
                    fact.row.address.source_id,
                    fact.row.address.relation_id,
                    fact.row.address.direction,
                ]
                for fact in gold_facts
            ],
            "gold_fact_ids": [fact.fact_id for fact in gold_facts],
            "gold_actions": [
                action_json(action) for action in gold_actions
            ],
            "answer_choices": list(answer_choices),
            "relations": list(relations),
        },
    )


def _complete_action_plan(
    read_actions: Iterable[GraphAction],
) -> list[GraphAction]:
    reads = list(read_actions)
    if not 1 <= len(reads) <= 6:
        raise ValueError("action plans require one to six reads")
    if any(not action.read or action.halt for action in reads):
        raise ValueError("action plan inputs must all be reads")
    if len(reads) == 6:
        return reads
    filler = reads[0]
    halt = GraphAction(
        filler.source_slot,
        filler.relation_id,
        filler.direction,
        read=False,
        halt=True,
    )
    noop = GraphAction(
        filler.source_slot,
        filler.relation_id,
        filler.direction,
        read=False,
        halt=False,
    )
    return [*reads, halt, *[noop for _ in range(5 - len(reads))]]


def make_action_plan(relation_ids: Iterable[str]) -> list[GraphAction]:
    return _complete_action_plan(
        GraphAction(
            source_slot=0,
            relation_id=relation_id,
            direction="out",
            read=True,
            halt=False,
        )
        for relation_id in relation_ids
    )


def _replace(
    row: GraphRow,
    *,
    target: str | None = None,
    compose: int | None = None,
) -> GraphRow:
    qualifiers = row.qualifiers
    if compose is not None:
        if "compose" not in dict(qualifiers):
            raise ValueError("compose replacement requires a compose qualifier")
        qualifiers = tuple(
            (key, str(compose) if key == "compose" else value)
            for key, value in qualifiers
        )
    return GraphRow(
        source_id=row.source_id,
        relation_id=row.relation_id,
        direction=row.direction,
        target_kind=row.target_kind,
        target=row.target if target is None else target,
        qualifiers=qualifiers,
        provenance_id=row.provenance_id,
    )


def _path_relations(
    rng: random.Random,
    hops: int,
) -> tuple[str, ...]:
    while True:
        relations = tuple(rng.choice(ENTITY_RELATIONS) for _ in range(hops))
        counts = Counter(relations)
        if any(count == 1 for count in counts.values()):
            return relations


def _date_pair(
    rng: random.Random,
    entity_ids: tuple[int, ...],
    rows: dict[GraphAddress, GraphFact],
    relation_id: str,
    earlier_in_slot_zero: bool,
) -> tuple[int, int, GraphFact, GraphFact]:
    for _ in range(1000):
        a, b = rng.sample(entity_ids, 2)
        fact_a = rows[GraphAddress(a, relation_id, "out")]
        fact_b = rows[GraphAddress(b, relation_id, "out")]
        if fact_a.row.target == fact_b.row.target:
            continue
        if (fact_a.row.target < fact_b.row.target) != earlier_in_slot_zero:
            a, b = b, a
            fact_a, fact_b = fact_b, fact_a
        return a, b, fact_a, fact_b
    raise ValueError("world must contain at least two distinct dates")


def _category_groups(
    entity_ids: tuple[int, ...],
    rows: dict[GraphAddress, GraphFact],
    relation_id: str,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for entity_id in entity_ids:
        fact = rows[GraphAddress(entity_id, relation_id, "out")]
        groups[fact.row.target].append(entity_id)
    return dict(groups)


def _category_pair(
    rng: random.Random,
    groups: dict[str, list[int]],
    rows: dict[GraphAddress, GraphFact],
    relation_id: str,
    equal: bool,
) -> tuple[int, int, GraphFact, GraphFact]:
    if equal:
        choices = [value for value, ids in groups.items() if len(ids) >= 2]
        if not choices:
            raise ValueError("world must contain a repeated category")
        category = rng.choice(choices)
        a, b = rng.sample(groups[category], 2)
    else:
        categories = tuple(groups)
        if len(categories) < 2:
            raise ValueError("world must contain at least two categories")
        category_a, category_b = rng.sample(categories, 2)
        a = rng.choice(groups[category_a])
        b = rng.choice(groups[category_b])
    return (
        a,
        b,
        rows[GraphAddress(a, relation_id, "out")],
        rows[GraphAddress(b, relation_id, "out")],
    )


def _different_category(value: str) -> str:
    prefix = "category-"
    if value.startswith(prefix):
        try:
            return f"{prefix}{(int(value[len(prefix):]) + 1) % 64}"
        except ValueError:
            pass
    return f"{value}-counterfactual"


def _composition_answer(rows: Iterable[GraphRow]) -> str:
    compose = sum(
        int(dict(row.qualifiers)["compose"]) for row in rows
    ) % 4
    return f"r{compose}"


def _generate_eval_pairs(
    world: GraphWorld,
    n_pairs_per_task: int,
    seed: int,
    *,
    path_hops: int | None,
    pair_index_offset: int = 0,
) -> list[CounterfactualPair]:
    if n_pairs_per_task < 0:
        raise ValueError("n_pairs_per_task must be non-negative")
    if n_pairs_per_task == 0:
        return []
    if (
        isinstance(pair_index_offset, bool)
        or not isinstance(pair_index_offset, int)
        or pair_index_offset < 0
    ):
        raise ValueError("pair_index_offset must be nonnegative")
    if path_hops is not None:
        if world.schema is None and path_hops not in CURRICULUM_HOPS:
            raise ValueError(f"path_hops must be one of {CURRICULUM_HOPS}")
        if world.schema is not None and path_hops not in range(1, 7):
            raise ValueError("schema-shaped path_hops must be in [1, 6]")

    entity_ids = _entity_ids(world)
    if len(entity_ids) < MIN_REASONING_ENTITIES:
        raise ValueError(
            "reasoning worlds must contain at least sixteen entities"
        )
    names = dict(zip(entity_ids, world.entity_names))
    rows = _row_map(world)
    date_relation = _relation_for_kind(world, "date")
    category_relation = _relation_for_kind(world, "category")
    category_groups = _category_groups(
        entity_ids,
        rows,
        category_relation,
    )
    planned_compositions = _path_compositions(world, path_hops)
    reasoning_entity_ids = world.reasoning_entity_ids or entity_ids
    rng = random.Random((seed << 32) ^ world.world_id)
    date_orientation = rng.randrange(2)
    equality_orientation = rng.randrange(2)
    pairs: list[CounterfactualPair] = []

    for task in (
        "path_composition",
        "date_ordering",
        "balanced_equality",
    ):
        for index in range(
            pair_index_offset,
            pair_index_offset + n_pairs_per_task,
        ):
            if world.split_name is None:
                pair_id = f"{world.world_id}-{seed}-{task}-{index}"
            else:
                question_namespace = world.manifest["question_namespace"]
                pair_id = (
                    f"{question_namespace}:{world.split_name}:"
                    f"{world.world_id}:{seed}:{task}:{index}"
                )
            relations: tuple[str, ...] = ()

            if task == "path_composition":
                a = rng.choice(reasoning_entity_ids)
                if planned_compositions:
                    relations = planned_compositions[
                        index % len(planned_compositions)
                    ]
                else:
                    hops = (
                        path_hops
                        if path_hops is not None
                        else rng.randint(2, 4)
                    )
                    relations = _path_relations(rng, hops)
                _, _, used = _follow(rows, a, relations)
                answer = _composition_answer(fact.row for fact in used)

                relation_counts = Counter(relations)
                change_index = next(
                    position
                    for position, relation in enumerate(relations)
                    if relation_counts[relation] == 1
                )
                changed_fact = used[change_index]
                old_code = int(
                    dict(changed_fact.row.qualifiers)["compose"]
                )
                changed = _replace(
                    changed_fact.row,
                    compose=(
                        old_code
                        - old_code % 4
                        + (old_code + 1) % 4
                    ),
                )
                changed_rows = [
                    changed
                    if fact.row.address == changed.address
                    else fact.row
                    for fact in used
                ]
                flipped = _composition_answer(changed_rows)
                if flipped == answer:
                    raise AssertionError(
                        "path counterfactual failed to change composition"
                    )

                gold_facts = used
                source_slots = (0,) * len(gold_facts)
                entity_slots = (a, None, None, None)
                answer_choices = tuple(f"r{i}" for i in range(4))
                relation_surfaces = tuple(
                    _relation_surface(world, relation_id)
                    for relation_id in relations
                )
                prompt = (
                    f"Slot 0 refers to {names[a]}. Start at slot 0 and follow "
                    f"{' then '.join(relation_surfaces)}. "
                    "Return the composed relation."
                )
            elif task == "date_ordering":
                a, b, fact_a, fact_b = _date_pair(
                    rng,
                    entity_ids,
                    rows,
                    date_relation,
                    earlier_in_slot_zero=(
                        (index + date_orientation) % 2 == 0
                    ),
                )
                answer = (
                    "<|slot_0|>"
                    if fact_a.row.target < fact_b.row.target
                    else "<|slot_1|>"
                )
                changed = _replace(
                    fact_a.row,
                    target=(
                        (
                            f"{fact_a.row.target.rpartition(':')[0]}:"
                            "9999-12-31"
                            if ":" in fact_a.row.target
                            else "9999-12-31"
                        )
                        if answer == "<|slot_0|>"
                        else (
                            f"{fact_a.row.target.rpartition(':')[0]}:"
                            "0001-01-01"
                            if ":" in fact_a.row.target
                            else "0001-01-01"
                        )
                    ),
                )
                flipped = (
                    "<|slot_0|>"
                    if changed.target < fact_b.row.target
                    else "<|slot_1|>"
                )
                gold_facts = (fact_a, fact_b)
                source_slots = (0, 1)
                entity_slots = (a, b, None, None)
                answer_choices = ("<|slot_0|>", "<|slot_1|>")
                relations = (date_relation, date_relation)
                prompt = (
                    f"Slot 0 refers to {names[a]}. "
                    f"Slot 1 refers to {names[b]}. "
                    f"Read {_relation_surface(world, date_relation)} from "
                    "slots 0 and 1. "
                    "Return the earlier slot."
                )
            else:
                want_equal = (index + equality_orientation) % 2 == 0
                a, b, fact_a, fact_b = _category_pair(
                    rng,
                    category_groups,
                    rows,
                    category_relation,
                    equal=want_equal,
                )
                answer = "yes" if want_equal else "no"
                changed = _replace(
                    fact_a.row,
                    target=(
                        _different_category(fact_b.row.target)
                        if want_equal
                        else fact_b.row.target
                    ),
                )
                flipped = (
                    "yes"
                    if changed.target == fact_b.row.target
                    else "no"
                )
                gold_facts = (fact_a, fact_b)
                source_slots = (0, 1)
                entity_slots = (a, b, None, None)
                answer_choices = ("yes", "no")
                relations = (category_relation, category_relation)
                prompt = (
                    f"Slot 0 refers to {names[a]}. "
                    f"Slot 1 refers to {names[b]}. "
                    f"Read {_relation_surface(world, category_relation)} from "
                    "slots 0 and 1. "
                    "Are they equal?"
                )

            if flipped == answer:
                raise AssertionError(
                    f"{task} counterfactual failed to flip the answer"
                )
            is_heldout_composition = (
                task == "path_composition"
                and world.split_plan is not None
                and composition_hash(relations)
                in world.split_plan.protected_heldout.composition_hashes
            )
            composition_split = (
                "heldout" if is_heldout_composition else "seen"
            )
            original = _item(
                qid=f"{pair_id}-o",
                task=task,
                prompt=prompt,
                answer=answer,
                pair_id=pair_id,
                graph_rows=len(world.facts),
                world_id=world.world_id,
                composition_split=composition_split,
                variant="original",
                entity_slots=entity_slots,
                gold_facts=gold_facts,
                source_slots=source_slots,
                answer_choices=answer_choices,
                relations=relations,
            )
            counterfactual = _item(
                qid=f"{pair_id}-c",
                task=task,
                prompt=prompt,
                answer=flipped,
                pair_id=pair_id,
                graph_rows=len(world.facts),
                world_id=world.world_id,
                composition_split=composition_split,
                variant="counterfactual",
                entity_slots=entity_slots,
                gold_facts=gold_facts,
                source_slots=source_slots,
                answer_choices=answer_choices,
                changed_row=changed,
                relations=relations,
            )
            pairs.append(
                CounterfactualPair(
                    task=task,
                    original=original,
                    counterfactual=counterfactual,
                    changed_row=changed,
                )
            )
    return pairs


def generate_eval_pairs(
    world: GraphWorld,
    n_pairs_per_task: int,
    seed: int,
    *,
    pair_index_offset: int = 0,
) -> list[CounterfactualPair]:
    return _generate_eval_pairs(
        world,
        n_pairs_per_task,
        seed,
        path_hops=None,
        pair_index_offset=pair_index_offset,
    )


def make_factual_recall_item(
    world: GraphWorld,
    fact: GraphFact,
    ordinal: int,
    tok,
) -> QAItem:
    if ordinal < 0:
        raise ValueError("factual item ordinal must be non-negative")
    entity_ids = _entity_ids(world)
    names = dict(zip(entity_ids, world.entity_names))
    if fact.row.source_id not in names:
        raise ValueError("factual source is absent from its world")
    answer = _factual_answer(fact.row)
    candidates = sorted(
        {
            _factual_answer(candidate.row)
            for candidate in world.facts
            if candidate.row.target_kind == fact.row.target_kind
            and candidate.fact_id != fact.fact_id
        }
        - {answer}
    )
    if len(candidates) < 3:
        raise ValueError("factual recall requires three same-kind distractors")
    offset = ordinal % len(candidates)
    ordered = candidates[offset:] + candidates[:offset]
    selected = [answer]
    for candidate in ordered:
        proposed = selected + [candidate]
        encoded = [tuple(tok.encode(choice)) for choice in proposed]
        if (
            all(encoded)
            and len(encoded) == len(set(encoded))
            and all(
                not (
                    len(left) < len(right)
                    and right[: len(left)] == left
                )
                for index, left in enumerate(encoded)
                for other_index, right in enumerate(encoded)
                if index != other_index
            )
        ):
            selected.append(candidate)
            if len(selected) == 4:
                break
    if len(selected) != 4:
        raise ValueError(
            "could not construct token-prefix-free factual answer choices"
        )
    choices = selected[1:]
    choices.insert(ordinal % 4, answer)
    qid = f"factual-{world.world_id}-{fact.fact_id}-{ordinal}"
    item = _item(
        qid=qid,
        task="factual_recall",
        prompt=(
            f"Slot 0 refers to {names[fact.row.source_id]}. "
            f"Start at slot 0 and follow "
            f"{_relation_surface(world, fact.row.relation_id)}. "
            "Return the exact stored target."
        ),
        answer=answer,
        pair_id=qid,
        graph_rows=len(world.facts),
        world_id=world.world_id,
        composition_split="seen",
        variant="original",
        entity_slots=(fact.row.source_id, None, None, None),
        gold_facts=(fact,),
        source_slots=(0,),
        answer_choices=tuple(choices),
        relations=(fact.row.relation_id,),
    )
    item.meta["target_kind"] = fact.row.target_kind
    return item


def _factual_answer(row: GraphRow) -> str:
    return f"Q{row.target}" if row.target_kind == "entity" else row.target


def iter_relation_alias_records(
    schema: RelationSchema,
) -> Iterator[RenderedRecord]:
    """Emit every unambiguous catalog alias once, in catalog order."""

    if not isinstance(schema, RelationSchema):
        raise TypeError("relation alias records require a RelationSchema")
    exposure = 0
    for relation in schema.catalog:
        for alias_index, alias in enumerate(relation.aliases):
            yield RenderedRecord(
                segments=(
                    TaggedSegment(alias, "relation_alias"),
                    TaggedSegment(
                        f" identifies relation {relation.relation_id}.",
                        "rule",
                    ),
                ),
                schedule=ScheduleEntry(
                    component="graph",
                    record_id=(
                        f"relation-alias:{relation.relation_id}:"
                        f"{alias_index}"
                    ),
                    exposure=exposure,
                    curriculum_band=0,
                ),
                metadata={
                    "relation_id": relation.relation_id,
                    "alias": alias,
                },
            )
            exposure += 1


def iter_bed_records(bed_iter: Iterable[str]) -> Iterator[RenderedRecord]:
    for index, text in enumerate(bed_iter):
        yield RenderedRecord(
            segments=(TaggedSegment(text, "plain"),),
            schedule=ScheduleEntry(
                component="bed",
                record_id=f"bed-{index}",
                exposure=index,
                curriculum_band=0,
            ),
        )


def _payload_text(row: GraphRow) -> str:
    return json.dumps(
        {
            "target_kind": row.target_kind,
            "target": row.target,
            "qualifiers": list(row.qualifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _cycle_audit_facts(
    facts: tuple[GraphFact, ...],
    audit_class: str,
) -> Iterator[GraphFact]:
    while True:
        found = False
        for fact in facts:
            if fact.audit_class == audit_class:
                found = True
                yield fact
        if not found:
            return


MAX_RECORD_CONTROL_PADDING_TOKENS = 4096


def _exact_role_key_counts(
    encoded: tuple[tuple[TaggedSegment, int], ...],
    role: str,
    document_length: int,
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    start = 0
    for segment, length in encoded:
        end = start + length
        if segment.role == role:
            key = (
                length,
                relative_position_bin(start, end, document_length),
            )
            counts[key] = counts.get(key, 0) + 1
        start = end
    return counts


def balance_record_random_controls(
    tok,
    segments: tuple[TaggedSegment, ...] | list[TaggedSegment],
    *,
    max_padding_tokens: int = MAX_RECORD_CONTROL_PADDING_TOKENS,
) -> tuple[TaggedSegment, ...]:
    """Return a deterministically padded, locally Random-balanced record."""

    if max_padding_tokens < 0:
        raise ValueError("max_padding_tokens must be non-negative")
    segments = tuple(segments)
    encoded = tuple(
        (segment, len(tok.encode(segment.text))) for segment in segments
    )
    base_length = sum(length for _, length in encoded)
    for padding_tokens in range(max_padding_tokens + 1):
        document_length = base_length + padding_tokens + 1
        demands = _exact_role_key_counts(
            encoded,
            "payload",
            document_length,
        )
        candidates = _exact_role_key_counts(
            encoded,
            "random_control",
            document_length,
        )
        if all(
            candidates.get(key, 0) >= count
            for key, count in demands.items()
        ):
            if not padding_tokens:
                return segments
            return (
                *segments,
                TaggedSegment("\t" * padding_tokens, "plain"),
            )
    raise RandomMaskUndersupplyError(
        "record cannot balance payload/control exact keys within "
        f"{max_padding_tokens} padding tokens"
    )


def iter_graph_records(
    tok,
    worlds_factory: Callable[[], Iterable[GraphWorld]],
) -> Iterator[RenderedRecord]:
    exposure = 0
    rule_text = (
        "Composition adds retrieved compose codes modulo four. "
        "Inverse traversal reverses edge direction. Equality is symmetric. "
        "Earlier dates have smaller ISO-8601 strings."
    )
    while True:
        saw_world = False
        for world in worlds_factory():
            saw_world = True
            peripheral_count = sum(
                fact.audit_class == "peripheral" for fact in world.facts
            )
            central_count = sum(
                fact.audit_class == "central" for fact in world.facts
            )
            if not peripheral_count or not central_count:
                raise ValueError(
                    "graph rendering requires peripheral and central facts"
                )

            batches = max(
                math.ceil(peripheral_count / 7),
                math.ceil(central_count / 2),
            )
            peripheral = _cycle_audit_facts(
                world.facts,
                "peripheral",
            )
            central = _cycle_audit_facts(world.facts, "central")
            for batch_index in range(batches):
                batch = tuple(next(peripheral) for _ in range(7)) + tuple(
                    next(central) for _ in range(2)
                )
                for fact in batch:
                    returned = _return_segment_blocks(
                        tok,
                        fact.row,
                        fact.fact_id,
                    )
                    segments = [
                        TaggedSegment(
                            f"Source {fact.row.source_id} relation ",
                            "plain",
                        ),
                        TaggedSegment(
                            _relation_surface(
                                world,
                                fact.row.relation_id,
                            ),
                            "relation_alias",
                        ),
                        TaggedSegment(" returns ", "plain"),
                        *returned,
                    ]
                    segments = list(
                        balance_record_random_controls(
                            tok,
                            segments,
                        )
                    )
                    yield RenderedRecord(
                        segments=tuple(segments),
                        schedule=ScheduleEntry(
                            component="graph",
                            record_id=fact.fact_id,
                            exposure=exposure,
                            curriculum_band=0,
                        ),
                    )
                    exposure += 1

                yield RenderedRecord(
                    segments=(TaggedSegment(rule_text, "rule"),),
                    schedule=ScheduleEntry(
                        component="graph",
                        record_id=(
                            f"rule-{world.world_id}-{batch_index}"
                        ),
                        exposure=exposure,
                        curriculum_band=0,
                    ),
                )
                exposure += 1
        if not saw_world:
            raise ValueError("worlds_factory produced no worlds")


def _answer_segments(answer: str) -> tuple[TaggedSegment, ...]:
    return (
        TaggedSegment("<|answer_state|>", "action"),
        TaggedSegment(answer, "provisional_answer"),
    )


def _return_segment_blocks(
    tok,
    row: GraphRow | None,
    fact_id: str | None,
) -> tuple[TaggedSegment, ...]:
    returned = tuple(serialize_return(row, fact_id))
    if not any(segment.role == "payload" for segment in returned):
        return returned
    balanced: list[TaggedSegment] = []
    for segment in returned:
        if segment.role != "payload":
            balanced.append(segment)
            continue
        length = len(tok.encode(segment.text))
        if not length:
            raise ValueError("payload segments must encode to at least one token")
        control = TaggedSegment("\t" * length, "random_control")
        balanced.extend((control, segment, control))
    return tuple(balanced)


def iter_reasoning_records(
    tok,
    worlds_factory: Callable[[], Iterable[GraphWorld]],
    seed: int,
    max_hops: int,
) -> Iterator[RenderedRecord]:
    if (
        isinstance(max_hops, bool)
        or not isinstance(max_hops, int)
        or max_hops not in range(1, 7)
    ):
        raise ValueError("max_hops must be in [1, 6]")

    rng = random.Random(seed)
    exposure = 0
    while True:
        saw_world = False
        for world in worlds_factory():
            saw_world = True
            if world.schema is None and max_hops not in CURRICULUM_HOPS:
                raise ValueError(f"max_hops must be one of {CURRICULUM_HOPS}")
            fact_map = {fact.fact_id: fact for fact in world.facts}
            relation_codec = (
                world.schema.codec
                if world.schema is not None
                else SRGM_RELATION_CODEC
            )
            filler_relation = relation_codec.relation_ids[0]
            pairs = _generate_eval_pairs(
                world,
                n_pairs_per_task=8,
                seed=rng.randrange(1 << 30),
                path_hops=max_hops,
            )
            pairs_by_task = {
                task: [pair for pair in pairs if pair.task == task]
                for task in (
                    "path_composition",
                    "date_ordering",
                    "balanced_equality",
                )
            }
            path_pairs = pairs_by_task["path_composition"]
            date_pairs = pairs_by_task["date_ordering"]
            equality_pairs = pairs_by_task["balanced_equality"]
            pairs = [
                *path_pairs[:4],
                date_pairs[0],
                equality_pairs[0],
                *path_pairs[4:],
                *date_pairs[1:],
                *equality_pairs[1:],
            ]
            for pair in pairs:
                item = pair.original
                addresses = tuple(
                    GraphAddress(int(source), relation, direction)
                    for source, relation, direction in item.meta[
                        "gold_addresses"
                    ]
                )
                fact_ids = tuple(item.meta["gold_fact_ids"])
                if len(addresses) > max_hops:
                    continue

                segments: list[TaggedSegment] = [
                    TaggedSegment(item.prompt, "plain")
                ]
                for step in range(6):
                    if step < len(addresses):
                        address = addresses[step]
                        fact = fact_map[fact_ids[step]]
                        if fact.row.address != address:
                            raise ValueError(
                                "gold fact id does not match gold address"
                            )
                        source_slot = (
                            0
                            if item.task == "path_composition"
                            else step
                        )
                        action = GraphAction(
                            source_slot=source_slot,
                            relation_id=address.relation_id,
                            direction=address.direction,
                            read=True,
                            halt=False,
                        )
                        segments.append(
                            TaggedSegment(
                                tok.decode(
                                    serialize_action(
                                        action,
                                        tok,
                                        relation_codec,
                                    )
                                ),
                                "action",
                            )
                        )
                        returned = _return_segment_blocks(
                            tok,
                            fact.row,
                            fact.fact_id,
                        )
                        segments.extend(returned)
                    else:
                        is_halt_step = step == len(addresses)
                        action = GraphAction(
                            source_slot=0,
                            relation_id=filler_relation,
                            direction="out",
                            read=False,
                            halt=is_halt_step,
                        )
                        segments.append(
                            TaggedSegment(
                                tok.decode(
                                    serialize_action(
                                        action,
                                        tok,
                                        relation_codec,
                                    )
                                ),
                                "action",
                            )
                        )
                        returned = _return_segment_blocks(tok, None, None)
                        segments.extend(returned)
                    segments.extend(_answer_segments(item.answer))

                segments.append(
                    TaggedSegment(item.answer, "final_answer")
                )
                segments = list(
                    balance_record_random_controls(
                        tok,
                        segments,
                    )
                )
                yield RenderedRecord(
                    segments=tuple(segments),
                    schedule=ScheduleEntry(
                        component="reasoning",
                        record_id=item.qid,
                        exposure=exposure,
                        curriculum_band=max_hops,
                    ),
                    metadata={
                        "question_id": item.qid,
                        "world_id": item.meta["world_id"],
                        "relation_path_hash": item.meta[
                            "relation_path_hash"
                        ],
                        "template_id": item.meta["template_id"],
                        "composition_split": item.meta[
                            "composition_split"
                        ],
                        "hop_count": item.meta["hop_count"],
                        "relations": list(item.meta["relations"]),
                        "gold_fact_ids": list(fact_ids),
                    },
                )
                exposure += 1
        if not saw_world:
            raise ValueError("worlds_factory produced no worlds")
