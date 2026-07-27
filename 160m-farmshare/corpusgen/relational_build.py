from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import islice, zip_longest
from pathlib import Path
from typing import Iterable

import numpy as np

from corpusgen.graph_records import (
    GraphAction,
    GraphRow,
    RANDOM_CONTROL_POSITION_BINS,
    RenderedRecord,
    relative_position_bin,
)
from corpusgen.mask_ledger import (
    PROTECTED_SCHEMA_ROLES,
    WEIGHT_CONDITIONS,
    LeakageError,
    MaskAudit,
    OccurrenceSpool,
    RandomMaskUndersupplyError,
    derive_weight_sidecars,
    should_mask,
    verify_weight_sidecars,
)
from corpusgen.payload_inventory import PayloadInventory
from corpusgen.publication_audit import (
    AUDIT_NAME,
    EXPECTATION_NAME,
    freeze_production_split_expectations,
    freeze_published_artifact_expectations,
    write_published_artifact_audit,
)
from corpusgen.schedule_plan import SchedulePlanSpool
from corpusgen.relation_schema import RelationSchema
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
    iter_bed_records,
    iter_graph_records,
    iter_relation_alias_records,
    iter_reasoning_records,
    iter_worlds,
    make_factual_recall_item,
)
from corpusgen.world_splits import (
    ObservedSplitArtifacts,
    ReasoningArtifactSignature,
    SplitArtifactExpectations,
    SplitPlan,
    WorldArtifactSignature,
    audit_disjointness,
    build_split_plan,
    require_disjointness,
)
from organizer.packed_graph_store import PackedGraphStore


WRITE_COST_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEVELOPMENT_MIXTURES = (
    {"bed": 0.70, "graph": 0.15, "reasoning": 0.15},
    {"bed": 0.65, "graph": 0.15, "reasoning": 0.20},
    {"bed": 0.65, "graph": 0.20, "reasoning": 0.15},
)
COMPONENT_SHARES = DEVELOPMENT_MIXTURES[0]
POSITION_BIN_COUNT = RANDOM_CONTROL_POSITION_BINS
_EVAL_TASKS = (
    "path_composition",
    "date_ordering",
    "balanced_equality",
)
_PROTECTED_TARGET_ROLES = PROTECTED_SCHEMA_ROLES - {"payload"}
_COMPONENT_ORDER = tuple(COMPONENT_SHARES)
_ROUTE_WORLD_ID = 1 << 30
_EVAL_WORLD_ID = 1 << 31
_ROUTE_SEED_XOR = 0x5EED5EED
_ROUTE_STATS_SEED_XOR = 0x13579BDF
_EVAL_SEED_XOR = 0x0E1A15E7


@dataclass(frozen=True)
class FactCost:
    fact_id: str
    entropy: float
    exposures: int
    expected_reads: float
    expected_hops: float


@dataclass(order=True, frozen=True)
class _FactualCandidate:
    priority: int
    fact_id: str
    ordinal: int
    world: object = field(compare=False)
    fact: object = field(compare=False)
    route: str = field(compare=False)


@dataclass(frozen=True)
class EncodedSpan:
    start: int
    end: int
    role: str
    fact_id: str | None = None
    fact_cost: FactCost | None = None
    payload_field: str | None = None
    payload_text: str | None = None


@dataclass(frozen=True)
class ExpectedExternalRange:
    start: int
    end: int
    fact_id: str


@dataclass(frozen=True)
class RoutePolicy:
    write_cost: float
    read_cost: float = 0.25
    hop_cost: float = 0.25

    def is_external(self, fact: FactCost) -> bool:
        predict = fact.entropy / max(fact.exposures, 1)
        external = (
            self.write_cost
            + self.read_cost * fact.expected_reads
            + self.hop_cost * fact.expected_hops
        )
        return predict > external

    def route_rate(self, facts) -> float:
        facts = tuple(facts)
        if not facts:
            raise ValueError("route rate requires at least one fact")
        return sum(self.is_external(fact) for fact in facts) / len(facts)

    def sha256(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


def calibrate_write_cost(facts) -> RoutePolicy:
    facts = tuple(facts)
    if not facts:
        raise ValueError("write-cost calibration requires at least one fact")
    candidates = [RoutePolicy(value) for value in WRITE_COST_GRID]
    valid = [
        policy
        for policy in candidates
        if 0.40 <= policy.route_rate(facts) <= 0.60
    ]
    if not valid:
        raise ValueError("no write cost yields a 40–60% route rate")
    return min(
        valid,
        key=lambda policy: (
            abs(policy.route_rate(facts) - 0.50),
            policy.write_cost,
        ),
    )


@dataclass(frozen=True)
class RelationalBuildConfig:
    n_entities: int
    total_tokens: int
    data_seed: int
    world_size: int = 64
    eval_pairs_per_task: int = 10_000
    eval_pairs_per_world: int = 32
    route_stats_pairs_per_task: int = 64
    guardrail_items: int = 10_000
    shared_text_eval_count: int = 64
    artifact_mode: str = "fixture"
    development_mixture_index: int = 0

    def __post_init__(self) -> None:
        if self.n_entities < 16:
            raise ValueError("n_entities must be at least 16")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.data_seed < 0:
            raise ValueError("data_seed must be non-negative")
        if self.world_size < 16:
            raise ValueError("world_size must be at least 16")
        if self.eval_pairs_per_task <= 0:
            raise ValueError("eval_pairs_per_task must be positive")
        if self.eval_pairs_per_world <= 0:
            raise ValueError("eval_pairs_per_world must be positive")
        if self.route_stats_pairs_per_task <= 0:
            raise ValueError("route_stats_pairs_per_task must be positive")
        if self.guardrail_items <= 0:
            raise ValueError("guardrail_items must be positive")
        if self.shared_text_eval_count <= 0:
            raise ValueError("shared_text_eval_count must be positive")
        if self.artifact_mode not in {"fixture", "development", "protected"}:
            raise ValueError(
                "artifact_mode must be fixture, development, or protected"
            )
        if (
            isinstance(self.development_mixture_index, bool)
            or not isinstance(self.development_mixture_index, int)
            or self.development_mixture_index
            not in range(len(DEVELOPMENT_MIXTURES))
        ):
            raise ValueError("invalid development mixture index")
        if (
            self.artifact_mode == "fixture"
            and self.development_mixture_index != 0
        ):
            raise ValueError(
                "mixture fallbacks require development or a frozen study"
            )

    @property
    def component_shares(self) -> dict[str, float]:
        return dict(DEVELOPMENT_MIXTURES[self.development_mixture_index])


BuildCfg = RelationalBuildConfig
BuildConfig = RelationalBuildConfig


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_line(value) -> str:
    return _canonical_json(value) + "\n"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, path: Path) -> dict:
    relative = path.relative_to(root)
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _position_histogram_json(histogram: Counter) -> dict[str, int]:
    return {
        f"{length}:{relative_bin}": count
        for (length, relative_bin), count in sorted(histogram.items())
    }


def _fact_costs_for_world(
    world,
    *,
    stats_seed: int,
    pairs_per_task: int,
) -> dict[str, FactCost]:
    reads: Counter[str] = Counter()
    hops: Counter[str] = Counter()
    for pair in generate_eval_pairs(world, pairs_per_task, stats_seed):
        fact_ids = tuple(pair.original.meta["gold_fact_ids"])
        for position, fact_id in enumerate(fact_ids):
            reads[fact_id] += 1
            hops[fact_id] += len(fact_ids) - position

    costs = {}
    for fact in world.facts:
        exposures_float = math.expm1(fact.features.log_exposure)
        exposures = int(round(exposures_float))
        if not math.isclose(exposures_float, exposures, abs_tol=1e-9):
            raise ValueError("Task 1 exposure statistic is not integral")
        costs[fact.fact_id] = FactCost(
            fact_id=fact.fact_id,
            entropy=fact.features.payload_entropy,
            exposures=exposures,
            expected_reads=(
                fact.features.expected_queries + reads[fact.fact_id]
            ),
            expected_hops=(
                fact.features.path_centrality + hops[fact.fact_id]
            ),
        )
    return costs


def _cost_digest(costs: Iterable[FactCost]) -> str:
    payload = _canonical_json(
        [asdict(cost) for cost in sorted(costs, key=lambda cost: cost.fact_id)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _iter_world_sizes(
    n_entities: int,
    world_size: int,
) -> Iterable[int]:
    full_worlds, remainder = divmod(n_entities, world_size)
    if remainder == 0:
        yield from (world_size for _ in range(full_worlds))
        return
    if full_worlds == 0:
        yield remainder
        return
    if remainder >= 16:
        yield from (world_size for _ in range(full_worlds - 1))
        combined = world_size + remainder
        yield combined // 2
        yield combined - combined // 2
        return
    deficit = 16 - remainder
    donor_size = world_size - deficit
    yield from (world_size for _ in range(max(0, full_worlds - 1)))
    if donor_size >= 16:
        yield donor_size
        yield 16
    else:
        yield world_size + remainder


def _iter_configured_worlds(
    cfg: RelationalBuildConfig,
    relation_schema: RelationSchema | None,
    split_plan: SplitPlan | None,
    split_name: str,
):
    if relation_schema is None or split_plan is None:
        yield from iter_worlds(
            cfg.n_entities,
            cfg.world_size,
            cfg.data_seed,
        )
        return
    partition = split_plan.partition(split_name)
    entity_offset = partition.entity_id_range[0]
    for ordinal, size in enumerate(
        _iter_world_sizes(cfg.n_entities, cfg.world_size)
    ):
        yield generate_world(
            ordinal,
            WorldConfig(
                n_entities=size,
                seed=cfg.data_seed,
                schema=relation_schema,
                split_plan=split_plan,
                split_name=split_name,
                entity_id_offset=entity_offset,
                world_seed_offset=partition.world_seed_range[0],
            ),
        )
        entity_offset += size


class _CostTrackingWorldFactory:
    def __init__(
        self,
        cfg: RelationalBuildConfig,
        stats_seed: int,
        relation_schema: RelationSchema | None = None,
        split_plan: SplitPlan | None = None,
        split_name: str = "train",
    ):
        self.cfg = cfg
        self.stats_seed = stats_seed
        self.relation_schema = relation_schema
        self.split_plan = split_plan
        self.split_name = split_name
        self.costs: dict[str, FactCost] = {}
        self.audit_classes: dict[str, str] = {}

    def __call__(self):
        for world in _iter_configured_worlds(
            self.cfg,
            self.relation_schema,
            self.split_plan,
            self.split_name,
        ):
            self.costs = _fact_costs_for_world(
                world,
                stats_seed=self.stats_seed,
                pairs_per_task=self.cfg.route_stats_pairs_per_task,
            )
            self.audit_classes = {
                fact.fact_id: fact.audit_class for fact in world.facts
            }
            yield world


def _encode_record(
    tok,
    record: RenderedRecord,
    costs: dict[str, FactCost],
) -> tuple[np.ndarray, list[EncodedSpan]]:
    ids, roles, fact_ids = tok.encode_tagged_segments(record.segments)
    if (
        tok.EOT < 0
        or tok.EOT >= 1 << 16
        or any(token_id < 0 or token_id >= 1 << 16 for token_id in ids)
    ):
        raise ValueError("token id does not fit uint16")
    payload_fields: list[str | None] = []
    payload_texts: list[str | None] = []
    for segment in record.segments:
        segment_length = len(tok.encode(segment.text))
        payload_fields.extend([segment.payload_field] * segment_length)
        payload_texts.extend(
            [segment.text if segment.role == "payload" else None]
            * segment_length
        )
    if (
        len(payload_fields) != len(ids)
        or len(payload_texts) != len(ids)
    ):
        raise ValueError("tagged segment metadata does not align with tokens")
    spans: list[EncodedSpan] = []
    start = 0
    while start < len(ids):
        role = roles[start]
        fact_id = fact_ids[start]
        payload_field = payload_fields[start]
        payload_text = payload_texts[start]
        end = start + 1
        while (
            end < len(ids)
            and roles[end] == role
            and fact_ids[end] == fact_id
            and payload_fields[end] == payload_field
            and payload_texts[end] == payload_text
        ):
            end += 1
        fact_cost = None
        if role == "payload":
            if fact_id not in costs:
                raise ValueError(f"missing route statistics for fact {fact_id}")
            if payload_field is None or payload_text is None:
                raise ValueError(
                    "payload spans require canonical field and text metadata"
                )
            fact_cost = costs[fact_id]
        spans.append(
            EncodedSpan(
                start,
                end,
                role,
                fact_id,
                fact_cost,
                payload_field,
                payload_text,
            )
        )
        start = end

    spans.append(EncodedSpan(len(ids), len(ids) + 1, "boundary"))
    ids.append(tok.EOT)
    return np.asarray(ids, dtype=np.uint16), spans


def derive_weights(
    condition: str,
    spans: list[EncodedSpan],
    policy: RoutePolicy,
    rng: random.Random,
) -> np.ndarray:
    if condition == "random":
        return derive_weight_sidecars(
            spans,
            policy=policy,
            seed=rng.getrandbits(64),
        )[condition]
    length = max((span.end for span in spans), default=0)
    weights = np.ones(length, dtype=np.uint8)
    for span in spans:
        if should_mask(condition, span, policy):
            weights[span.start : span.end] = 0
    return weights


def position_bin(start: int, end: int, document_length: int) -> int:
    return relative_position_bin(start, end, document_length)


def collect_expected_external_ranges(
    spans: list[EncodedSpan],
    policy: RoutePolicy,
) -> tuple[ExpectedExternalRange, ...]:
    expected = []
    for span in spans:
        if (
            span.role != "payload"
            or span.fact_cost is None
            or not policy.is_external(span.fact_cost)
        ):
            continue
        if span.fact_id is None:
            raise ValueError("external payload span requires a fact id")
        expected.append(
            ExpectedExternalRange(
                start=span.start,
                end=span.end,
                fact_id=span.fact_id,
            )
        )
    return tuple(expected)


def validate_split_coverage(
    expected: tuple[ExpectedExternalRange, ...],
    spans: list[EncodedSpan],
    weights: np.ndarray,
    actual_ranges: list[tuple[int, int, EncodedSpan]],
) -> None:
    expected_ranges = [(item.start, item.end) for item in expected]
    actual = [(start, end) for start, end, _ in actual_ranges]
    if actual != expected_ranges:
        raise ValueError(
            "actual Split ranges do not match expected external payload ranges"
        )
    for item in expected:
        if weights[item.start : item.end].any():
            raise ValueError(
                "expected external payload occurrence remained unmasked"
            )
    for span in spans:
        if span.role != "payload" and not weights[
            span.start : span.end
        ].all():
            raise ValueError("protected nonpayload span was Split-masked")

    expected_mask = np.ones(len(weights), dtype=np.uint8)
    for item in expected:
        expected_mask[item.start : item.end] = 0
    if not np.array_equal(weights, expected_mask):
        raise ValueError("Split contains zeros outside expected external ranges")


class CorpusWriter:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.token_path = out_dir / "train.bin"
        self.weight_paths = {
            condition: out_dir / f"{condition}.weights.bin"
            for condition in WEIGHT_CONDITIONS
        }
        self.ledger_path = out_dir / "mask-ledger.jsonl"
        self.occurrence_path = out_dir / "mask-occurrences.sqlite3"
        self.audit_path = out_dir / "mask-audit.json"
        publication_paths = (
            self.token_path,
            *self.weight_paths.values(),
            self.ledger_path,
            self.occurrence_path,
            self.audit_path,
        )
        existing = [path.name for path in publication_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"writer artifacts already exist: {sorted(existing)}"
            )
        self._closed = False
        self._finalized = False
        self.token_file = None
        self.weight_files: dict[str, object] = {}
        self.spool = None
        try:
            self.spool = OccurrenceSpool(self.occurrence_path)
            self.token_file = self.token_path.open("xb")
            self.weight_files = {
                condition: path.open("xb")
                for condition, path in self.weight_paths.items()
            }
        except BaseException:
            self.abort()
            raise
        self.total = 0
        self.records = 0
        self.component_tokens: Counter[str] = Counter()
        self.component_records: Counter[str] = Counter()
        self.protected_target_tokens = 0
        self.all_payload_tokens = 0
        self.selective_payload_tokens = 0
        self.all_payload_ranges = 0
        self.selective_payload_ranges = 0
        self.expected_external_ranges = 0
        self.actual_split_ranges = 0
        self._expected_range_digest = hashlib.sha256()
        self._actual_range_digest = hashlib.sha256()
        self._pending_random: Counter[tuple[int, int]] = Counter()
        self.masked_tokens: Counter[str] = Counter()
        self.span_histograms = {
            condition: Counter() for condition in WEIGHT_CONDITIONS
        }
        self.position_histograms = {
            condition: Counter() for condition in WEIGHT_CONDITIONS
        }
        self.expected_position_histogram: Counter[
            tuple[int, int]
        ] = Counter()
        self.expected_length_histogram: Counter[int] = Counter()
        self.protected_roles_unmasked = True
        self.dense_all_ones = True
        self.audit: MaskAudit | None = None

    def add(
        self,
        component: str,
        token_ids: np.ndarray,
        spans: list[EncodedSpan],
        policy: RoutePolicy,
        rng: random.Random,
    ) -> None:
        del rng
        if self._closed:
            raise ValueError("cannot add records after writer close")
        if token_ids.dtype != np.uint16 or token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional uint16 array")
        demand_counts: Counter[tuple[int, int]] = Counter()
        candidate_counts: Counter[tuple[int, int]] = Counter()
        for span in spans:
            key = (
                span.end - span.start,
                position_bin(span.start, span.end, len(token_ids)),
            )
            if span.role == "payload":
                demand_counts[key] += 1
            elif span.role == "random_control":
                candidate_counts[key] += 1
        for key, count in sorted(demand_counts.items()):
            supplied = candidate_counts.get(key, 0)
            if supplied < count:
                raise RandomMaskUndersupplyError(
                    f"record {self.records} Random-mask exact key {key} "
                    f"requires {count} same-record candidates, found {supplied}"
                )
        weights = {
            condition: np.ones(len(token_ids), dtype=np.uint8)
            for condition in WEIGHT_CONDITIONS
        }
        for span in spans:
            if should_mask("split", span, policy):
                weights["split"][span.start : span.end] = 0
                self.all_payload_tokens += span.end - span.start
                self.all_payload_ranges += 1
                self.expected_external_ranges += 1
                self.actual_split_ranges += 1
                assert span.fact_id is not None
                value = self._range_bytes(
                    self.total + span.start,
                    self.total + span.end,
                    span.fact_id,
                )
                self._expected_range_digest.update(value)
                self._actual_range_digest.update(value)
            if should_mask("selective", span, policy):
                weights["selective"][span.start : span.end] = 0
                self.selective_payload_tokens += span.end - span.start
                self.selective_payload_ranges += 1
            if span.role in _PROTECTED_TARGET_ROLES:
                self.protected_target_tokens += span.end - span.start

        assert self.spool is not None
        self.spool.add_record(
            component=component,
            record_index=self.records,
            global_start=self.total,
            token_ids=token_ids,
            spans=spans,
            policy=policy,
        )
        assert self.token_file is not None
        self.token_file.write(token_ids.tobytes())
        for condition in WEIGHT_CONDITIONS:
            self.weight_files[condition].write(weights[condition].tobytes())

        self.total += len(token_ids)
        self.records += 1
        self.component_tokens[component] += len(token_ids)
        self.component_records[component] += 1

    @staticmethod
    def _range_bytes(start: int, end: int, fact_id: str) -> bytes:
        return _json_line(
            {"start": start, "end": end, "fact_id": fact_id}
        ).encode()

    def close(self) -> None:
        if self._closed:
            return
        if self.token_file is not None:
            self.token_file.close()
        for stream in self.weight_files.values():
            stream.close()
        self._closed = True

    def abort(self) -> None:
        self.close()
        if self.spool is not None:
            self.spool.close()
        for path in (
            self.token_path,
            *self.weight_paths.values(),
            self.ledger_path,
            self.occurrence_path,
            self.audit_path,
        ):
            path.unlink(missing_ok=True)

    def finalize(
        self,
        *,
        seed: int,
        payload_inventory: Path | None = None,
        record_schedule: Path | None = None,
    ) -> MaskAudit:
        if self._finalized:
            assert self.audit is not None
            return self.audit
        self.close()
        assert self.spool is not None
        try:
            try:
                self.spool.finalize_random(
                    self.weight_paths["random"],
                    seed=seed,
                )
                self.spool.export_jsonl(self.ledger_path)
            finally:
                self.spool.close()
            audit = verify_weight_sidecars(
                self.token_path,
                self.weight_paths,
                self.occurrence_path,
                payload_inventory=payload_inventory,
                record_schedule=record_schedule,
            )
            audit.write(self.audit_path)
        except BaseException:
            self.abort()
            raise
        self.audit = audit
        self._finalized = True
        self.masked_tokens.update(audit.masked_tokens)
        for condition, histogram in audit.histograms.items():
            self.position_histograms[condition].update(histogram)
            self.span_histograms[condition].update(
                {
                    length: sum(
                        count
                        for (candidate_length, _), count in histogram.items()
                        if candidate_length == length
                    )
                    for length in {
                        candidate_length
                        for candidate_length, _ in histogram
                    }
                }
            )
        self.expected_position_histogram.update(
            audit.histograms["split"]
        )
        self.expected_length_histogram.update(
            self.span_histograms["split"]
        )
        self.dense_all_ones = audit.dense_all_ones
        self.protected_roles_unmasked = audit.protected_roles_unmasked
        return audit


def _calibrate_policy(
    cfg: RelationalBuildConfig,
    relation_schema: RelationSchema | None = None,
    split_plan: SplitPlan | None = None,
) -> tuple[RoutePolicy, dict]:
    calibration_seed = cfg.data_seed ^ _ROUTE_SEED_XOR
    stats_seed = calibration_seed ^ _ROUTE_STATS_SEED_XOR
    if relation_schema is None or split_plan is None:
        world = generate_world(
            _ROUTE_WORLD_ID,
            WorldConfig(n_entities=cfg.world_size, seed=calibration_seed),
        )
    else:
        partition = split_plan.development
        world = generate_world(
            0,
            WorldConfig(
                n_entities=cfg.world_size,
                seed=calibration_seed,
                schema=relation_schema,
                split_plan=split_plan,
                split_name="development",
                entity_id_offset=partition.entity_id_range[0],
                world_seed_offset=partition.world_seed_range[0],
            ),
        )
    costs_by_id = _fact_costs_for_world(
        world,
        stats_seed=stats_seed,
        pairs_per_task=cfg.route_stats_pairs_per_task,
    )
    costs = tuple(costs_by_id.values())
    policy = calibrate_write_cost(costs)
    calibration = {
        "split_name": (
            "legacy_fixture"
            if relation_schema is None
            else "development"
        ),
        "world_id": world.world_id,
        "entities": len(world.entity_names),
        "facts": len(costs),
        "query_schedule_count_per_family": cfg.route_stats_pairs_per_task,
        "route_rate": policy.route_rate(costs),
        "fact_cost_sha256": _cost_digest(costs),
        "inputs": (
            "payload_entropy,scheduled_exposure_count,"
            "expected_query_count,expected_hop_contribution"
        ),
    }
    return policy, calibration


def _payload_choice_text(row) -> str:
    return _canonical_json(
        {
            "target_kind": row.target_kind,
            "target": row.target,
            "qualifiers": list(row.qualifiers),
        }
    )


def _choices_are_prefix_free(tok, choices: list[str]) -> bool:
    encoded = [tuple(tok.encode(choice)) for choice in choices]
    return (
        all(encoded)
        and len(set(encoded)) == len(encoded)
        and all(
            not (
                len(left) < len(right)
                and right[: len(left)] == left
            )
            for left in encoded
            for right in encoded
            if left != right
        )
    )


def _fact_choice_item(world, fact, ordinal: int, tok, kind: str):
    correct = _payload_choice_text(fact.row)
    candidates = sorted(
        {
            _payload_choice_text(candidate.row)
            for candidate in world.facts
            if candidate.row.relation_id == fact.row.relation_id
            and candidate.row.direction == fact.row.direction
            and candidate.fact_id != fact.fact_id
        }
    )
    candidates = [choice for choice in candidates if choice != correct]
    if len(candidates) < 3:
        return None
    offset = ordinal % len(candidates)
    distractors = [
        candidates[(offset + index) % len(candidates)]
        for index in range(3)
    ]
    answer_index = ordinal % 4
    choices = distractors.copy()
    choices.insert(answer_index, correct)
    if not _choices_are_prefix_free(tok, choices):
        return None
    return {
        "qid": f"{kind}-{world.world_id}-{fact.fact_id}-{ordinal}",
        "kind": kind,
        "prompt": (
            f"Source {fact.row.source_id} relation "
            f"{fact.row.relation_id} returns "
        ),
        "choices": choices,
        "answer_index": answer_index,
        "fact_id": fact.fact_id,
    }


_RULE_CHOICE_ITEMS = (
    (
        "Composition adds retrieved compose codes",
        (
            " modulo four.",
            " by string concatenation.",
            " by taking their maximum.",
            " without preserving order.",
        ),
    ),
    (
        "Inverse traversal",
        (
            " reverses edge direction.",
            " deletes the source entity.",
            " changes every relation id.",
            " returns an arbitrary literal.",
        ),
    ),
    (
        "Equality",
        (
            " is symmetric.",
            " depends on branch order.",
            " is always false.",
            " applies only to dates.",
        ),
    ),
    (
        "Earlier dates",
        (
            " have smaller ISO-8601 strings.",
            " have larger ISO-8601 strings.",
            " cannot be compared lexically.",
            " are selected at random.",
        ),
    ),
)


def _rule_choice_item(ordinal: int, tok) -> dict:
    prompt, raw_choices = _RULE_CHOICE_ITEMS[
        ordinal % len(_RULE_CHOICE_ITEMS)
    ]
    answer_index = ordinal % 4
    choices = list(raw_choices[1:])
    choices.insert(answer_index, raw_choices[0])
    if not _choices_are_prefix_free(tok, choices):
        raise ValueError("rule answer choices must be token-prefix free")
    return {
        "qid": f"internal-rule-{ordinal}",
        "kind": "rule",
        "prompt": prompt,
        "choices": choices,
        "answer_index": answer_index,
    }


def _repeat_fact_items(sources, count: int, tok, kind: str) -> list[dict]:
    if not sources:
        raise ValueError(f"no facts available for {kind} evaluation")
    items = []
    ordinal = 0
    while len(items) < count:
        world, fact = sources[ordinal % len(sources)]
        item = _fact_choice_item(world, fact, ordinal, tok, kind)
        if item is None:
            raise ValueError(f"could not construct prefix-free {kind} choices")
        items.append(item)
        ordinal += 1
    return items


def _reserve_shared_text(bed_iter, tok, count: int):
    iterator = iter(bed_iter)
    heldout_rows = []
    heldout_source = set()
    while len(heldout_rows) < count:
        try:
            raw = next(iterator)
        except StopIteration as error:
            raise ValueError(
                "natural-text stream ended before shared-text holdout"
            ) from error
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("shared-text holdout requires non-empty strings")
        token_ids = tok.encode(raw)
        if not token_ids:
            raise ValueError("shared-text holdout encoded to no tokens")
        heldout_rows.append({"text": tok.decode(token_ids[:512])})
        heldout_source.add(raw)

    def training_stream():
        for text in iterator:
            if text not in heldout_source:
                yield text

    return heldout_rows, training_stream()


def _write_guardrail_eval_files(
    eval_dir: Path,
    guardrail_data: dict,
    shared_text_rows: list[dict],
) -> dict:
    paths = {
        "recognition": eval_dir / "recognition.jsonl",
        "factual": eval_dir / "factual.jsonl",
        "factual_graph": eval_dir / "factual-graph.jsonl",
        "internal": eval_dir / "internal.jsonl",
        "shared_text": eval_dir / "shared_text.jsonl",
        "route_audit": eval_dir / "route-audit.json",
    }
    for name in ("recognition", "internal"):
        paths[name].write_text(
            "".join(_json_line(item) for item in guardrail_data[name])
        )
    paths["factual"].write_text(
        "".join(
            _json_line(asdict(item))
            for item in guardrail_data["factual"]
        )
    )
    paths["factual_graph"].write_text(
        "".join(
            _json_line(row.as_json())
            for row in guardrail_data["factual_rows"]
        )
    )
    paths["shared_text"].write_text(
        "".join(_json_line(item) for item in shared_text_rows)
    )
    _write_json(paths["route_audit"], guardrail_data["route_audit"])
    return {
        "guardrail_items": len(guardrail_data["recognition"]),
        "shared_text_items": len(shared_text_rows),
        "guardrail_paths": {
            name: path.relative_to(eval_dir.parent).as_posix()
            for name, path in paths.items()
        },
        "guardrail_sha256": {
            name: _sha256_file(path) for name, path in paths.items()
        },
    }


def _iter_graph_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield GraphRow.from_json(json.loads(line))


def _packed_store_artifacts(path: Path) -> tuple[Path, ...]:
    return tuple(
        path / name
        for name in ("manifest.json", "index.bin", "rows.bin", "blobs.bin")
    )


def _write_training_graph(
    cfg: RelationalBuildConfig,
    policy: RoutePolicy,
    stats_seed: int,
    path: Path,
    tok,
    relation_schema: RelationSchema | None = None,
    split_plan: SplitPlan | None = None,
) -> tuple[dict, dict]:
    rows = 0
    external = 0
    tail_total = 0
    tail_external = 0
    structure_total = 0
    structure_internal = 0
    recognition_sources = []
    factual_candidates: dict[
        tuple[str, str],
        list[_FactualCandidate],
    ] = {
        (target_kind, route): []
        for target_kind in ("entity", "literal")
        for route in ("internal", "external")
    }
    central_sources = []
    central_target = (cfg.guardrail_items + 1) // 2
    world_plan_path = (
        path.parent / "split-plans" / "production" / "train-worlds.jsonl"
    )
    world_plan_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle, world_plan_path.open("x") as world_plan:
        for world in _iter_configured_worlds(
            cfg,
            relation_schema,
            split_plan,
            "train",
        ):
            provenances = {
                fact.row.provenance_id for fact in world.facts
            }
            if len(provenances) != 1:
                raise ValueError("production world must have one provenance ID")
            world_plan.write(
                _json_line(
                    {
                        "world_id": world.world_id,
                        "world_seed": world.world_seed,
                        "split_name": world.split_name,
                        "provenance_id": next(iter(provenances)),
                        "manifest": world.manifest,
                    }
                )
            )
            costs = _fact_costs_for_world(
                world,
                stats_seed=stats_seed,
                pairs_per_task=cfg.route_stats_pairs_per_task,
            )
            for fact in world.facts:
                handle.write(_json_line(fact.row.as_json()))
                rows += 1
                routed = policy.is_external(costs[fact.fact_id])
                external += routed
                is_tail = (
                    fact.features.payload_entropy >= 6.0
                    and fact.features.expected_queries <= 0.25
                    and fact.features.path_centrality <= 0.05
                )
                if is_tail:
                    tail_total += 1
                    tail_external += routed
                is_central = fact.features.path_centrality >= 1.0
                if is_central:
                    structure_total += 1
                    structure_internal += not routed

                if routed and len(recognition_sources) < cfg.guardrail_items:
                    if _fact_choice_item(
                        world,
                        fact,
                        len(recognition_sources),
                        tok,
                        "external_fact",
                    ) is not None:
                        recognition_sources.append((world, fact))
                route = "external" if routed else "internal"
                stratum = (fact.row.target_kind, route)
                stable_key = int.from_bytes(
                    hashlib.sha256(
                        _json_line(
                            [
                                cfg.data_seed,
                                fact.fact_id,
                                fact.row.target_kind,
                                route,
                            ]
                        ).encode()
                    ).digest(),
                    "big",
                )
                candidate = _FactualCandidate(
                    priority=-stable_key,
                    fact_id=fact.fact_id,
                    ordinal=rows,
                    world=world,
                    fact=fact,
                    route=route,
                )
                heap = factual_candidates[stratum]
                if len(heap) < cfg.guardrail_items:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
                if (
                    not routed
                    and fact.audit_class == "central"
                    and len(central_sources) < central_target
                ):
                    if _fact_choice_item(
                        world,
                        fact,
                        len(central_sources),
                        tok,
                        "central_fact",
                    ) is not None:
                        central_sources.append((world, fact))
            structure_total += 1
            structure_internal += 1

    if tail_total <= 0 or structure_total <= 0:
        raise ValueError("route audit strata must be non-empty")
    route_audit = {
        "route_rate": external / rows,
        "route_total": rows,
        "low_use_high_entropy_external_rate": tail_external / tail_total,
        "low_use_high_entropy_total": tail_total,
        "rules_top_centrality_internal_rate": (
            structure_internal / structure_total
        ),
        "rules_top_centrality_total": structure_total,
    }
    recognition = _repeat_fact_items(
        recognition_sources,
        cfg.guardrail_items,
        tok,
        "external_fact",
    )
    factual_pools = {
        stratum: tuple(
            (candidate.world, candidate.fact, candidate.route)
            for candidate in sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.priority,
                    candidate.fact_id,
                    candidate.ordinal,
                ),
            )
        )
        for stratum, candidates in factual_candidates.items()
    }
    strata = (
        ("entity", "internal"),
        ("entity", "external"),
        ("literal", "internal"),
        ("literal", "external"),
    )
    if cfg.guardrail_items >= len(strata) and any(
        not factual_pools[stratum] for stratum in strata
    ):
        missing = [
            stratum for stratum in strata if not factual_pools[stratum]
        ]
        raise ValueError(f"factual guardrail strata are empty: {missing}")
    available_strata = tuple(
        stratum for stratum in strata if factual_pools[stratum]
    )
    if not available_strata:
        raise ValueError("no train facts for factual evaluation")
    factual_sources = []
    factual = []
    for ordinal in range(cfg.guardrail_items):
        stratum_index = ordinal % len(available_strata)
        stratum = available_strata[stratum_index]
        pool = factual_pools[stratum]
        cycle = ordinal // len(available_strata)
        world, fact, route = pool[cycle % len(pool)]
        factual_sources.append((world, fact, route))
        item = make_factual_recall_item(world, fact, ordinal, tok)
        item.meta["route"] = route
        factual.append(item)
    central_count = (cfg.guardrail_items + 1) // 2
    internal = _repeat_fact_items(
        central_sources,
        central_count,
        tok,
        "central_fact",
    )
    internal.extend(
        _rule_choice_item(index, tok)
        for index in range(cfg.guardrail_items - central_count)
    )
    factual_rows_by_address = {
        fact.row.address: fact.row for _, fact, _ in factual_sources
    }
    manifest = {
        "path": path.name,
        "artifact_class": "jsonl-fixture-compatibility",
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "entities": cfg.n_entities,
        "route_rate": external / rows,
    }
    return manifest, {
        "recognition": recognition,
        "factual": factual,
        "factual_rows": tuple(
            row
            for _, row in sorted(factual_rows_by_address.items())
        ),
        "internal": internal,
        "route_audit": route_audit,
        "factual_fact_ids": [
            item.meta["gold_fact_ids"][0] for item in factual
        ],
        "fact_routes": {
            fact.fact_id: route for _, fact, route in factual_sources
        },
    }


def _write_eval_sets(
    cfg: RelationalBuildConfig,
    eval_dir: Path,
    relation_schema: RelationSchema | None = None,
    split_plan: SplitPlan | None = None,
    *,
    split_name: str | None = None,
    artifact_prefix: str = "eval",
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    graph_path = eval_dir / "graph.jsonl"
    original_path = eval_dir / "original.jsonl"
    counterfactual_path = eval_dir / "counterfactual.jsonl"
    world_plan_path = eval_dir / "worlds.jsonl"
    remaining = cfg.eval_pairs_per_task
    world_index = 0
    graph_rows = 0
    item_count = 0
    eval_seed = cfg.data_seed ^ _EVAL_SEED_XOR

    with (
        graph_path.open("w") as graph_file,
        original_path.open("w") as original_file,
        counterfactual_path.open("w") as counterfactual_file,
        world_plan_path.open("x") as world_plan,
    ):
        while remaining:
            pairs_per_task = min(cfg.eval_pairs_per_world, remaining)
            if relation_schema is None or split_plan is None:
                world = generate_world(
                    _EVAL_WORLD_ID + world_index,
                    WorldConfig(n_entities=cfg.world_size, seed=eval_seed),
                )
            else:
                selected_split = split_name or (
                    "development"
                    if cfg.artifact_mode == "development"
                    else "protected_heldout"
                )
                partition = split_plan.partition(selected_split)
                world = generate_world(
                    world_index,
                    WorldConfig(
                        n_entities=cfg.world_size,
                        seed=eval_seed,
                        schema=relation_schema,
                        split_plan=split_plan,
                        split_name=selected_split,
                        entity_id_offset=(
                            partition.entity_id_range[0]
                            + world_index * cfg.world_size
                        ),
                        world_seed_offset=partition.world_seed_range[0],
                    ),
                )
            provenances = {
                fact.row.provenance_id for fact in world.facts
            }
            if len(provenances) != 1:
                raise ValueError("production eval world needs one provenance ID")
            world_plan.write(
                _json_line(
                    {
                        "world_id": world.world_id,
                        "world_seed": world.world_seed,
                        "split_name": world.split_name,
                        "provenance_id": next(iter(provenances)),
                        "manifest": world.manifest,
                    }
                )
            )
            for fact in world.facts:
                graph_file.write(_json_line(fact.row.as_json()))
                graph_rows += 1
            pairs = generate_eval_pairs(
                world,
                pairs_per_task,
                eval_seed,
                pair_index_offset=(
                    cfg.eval_pairs_per_task - remaining
                ),
            )
            for pair in pairs:
                original_file.write(_json_line(asdict(pair.original)))
                counterfactual_file.write(
                    _json_line(asdict(pair.counterfactual))
                )
                item_count += 1
            remaining -= pairs_per_task
            world_index += 1

    return {
        "graph": f"{artifact_prefix}/graph.jsonl",
        "graph_artifact_class": "jsonl-fixture-compatibility",
        "original": f"{artifact_prefix}/original.jsonl",
        "counterfactual": f"{artifact_prefix}/counterfactual.jsonl",
        "world_plan": f"{artifact_prefix}/worlds.jsonl",
        "graph_rows": graph_rows,
        "worlds": world_index,
        "pairs": item_count,
        "pairs_per_task": cfg.eval_pairs_per_task,
    }


def validate_eval_sets(
    cfg: RelationalBuildConfig,
    training_graph_path: Path,
    eval_graph_path: Path,
    original_path: Path,
    counterfactual_path: Path,
) -> dict[str, bool]:
    def validate_gold_actions(meta: dict) -> list[GraphAction]:
        raw_actions = meta.get("gold_actions")
        if not isinstance(raw_actions, list) or len(raw_actions) != 6:
            raise ValueError("gold actions must contain exactly six steps")
        required = {
            "source_slot",
            "relation_id",
            "direction",
            "read",
            "halt",
        }
        actions = []
        for raw in raw_actions:
            if not isinstance(raw, dict) or set(raw) != required:
                raise ValueError("gold actions have invalid fields")
            actions.append(GraphAction(**raw))
        halts = [
            index for index, action in enumerate(actions) if action.halt
        ]
        if len(halts) > 1:
            raise ValueError("gold actions allow at most one HALT")
        if not halts:
            if not all(action.read for action in actions):
                raise ValueError(
                    "gold actions without HALT must be six READs"
                )
        else:
            halt = halts[0]
            if halt == 0 or not all(
                action.read for action in actions[:halt]
            ):
                raise ValueError(
                    "gold actions before HALT must be one to five reads"
                )
            if any(
                action.read or action.halt for action in actions[halt + 1 :]
            ):
                raise ValueError("gold actions after HALT must be NOOP")
        addresses = meta["gold_addresses"]
        reads = [action for action in actions if action.read]
        if len(reads) != len(addresses) or any(
            action.relation_id != str(address[1])
            or action.direction != str(address[2])
            for action, address in zip(reads, addresses)
        ):
            raise ValueError("gold actions do not match gold addresses")
        return actions

    task_counts = {
        "original": Counter(),
        "counterfactual": Counter(),
    }
    seen_pair_ids = set()
    changed_rows: dict[tuple[int, str, str], set[str]] = {}

    with (
        original_path.open() as original_file,
        counterfactual_path.open() as counterfactual_file,
    ):
        original_lines = (line for line in original_file if line.strip())
        counterfactual_lines = (
            line for line in counterfactual_file if line.strip()
        )
        for line_number, (original_line, counterfactual_line) in enumerate(
            zip_longest(original_lines, counterfactual_lines),
            1,
        ):
            if original_line is None or counterfactual_line is None:
                raise ValueError("every eval pair requires exactly two variants")
            original = json.loads(original_line)
            counterfactual = json.loads(counterfactual_line)
            original_meta = original["meta"]
            counterfactual_meta = counterfactual["meta"]
            original_actions = validate_gold_actions(original_meta)
            counterfactual_actions = validate_gold_actions(
                counterfactual_meta
            )
            if original_actions != counterfactual_actions:
                raise ValueError("eval twins must share exact gold actions")
            pair_id = original_meta["pair_id"]
            if (
                pair_id != counterfactual_meta["pair_id"]
                or pair_id in seen_pair_ids
                or original_meta["variant"] != "original"
                or counterfactual_meta["variant"] != "counterfactual"
            ):
                raise ValueError(
                    f"eval line {line_number} does not contain two variants"
                )
            seen_pair_ids.add(pair_id)

            if original["task"] != counterfactual["task"]:
                raise ValueError("eval twins must have the same task")
            task = original["task"]
            if task not in _EVAL_TASKS:
                raise ValueError(f"unexpected eval task: {task}")
            task_counts["original"][task] += 1
            task_counts["counterfactual"][task] += 1

            if original["answer"] == counterfactual["answer"]:
                raise ValueError("original and counterfactual answers must flip")
            if original_meta.get("changed_row") is not None:
                raise ValueError("original eval item must not contain a changed row")
            changed_row = counterfactual_meta.get("changed_row")
            if not isinstance(changed_row, dict):
                raise ValueError("counterfactual requires a changed supporting row")

            original_gold = {
                (int(source), str(relation), str(direction))
                for source, relation, direction in original_meta[
                    "gold_addresses"
                ]
            }
            counterfactual_gold = {
                (int(source), str(relation), str(direction))
                for source, relation, direction in counterfactual_meta[
                    "gold_addresses"
                ]
            }
            changed_address = (
                int(changed_row["source_id"]),
                str(changed_row["relation_id"]),
                str(changed_row["direction"]),
            )
            if (
                original_gold != counterfactual_gold
                or changed_address not in original_gold
            ):
                raise ValueError(
                    "changed row must be one of the pair's supporting rows"
                )
            changed_rows.setdefault(changed_address, set()).add(
                _canonical_json(changed_row)
            )

    expected_counts = {
        task: cfg.eval_pairs_per_task for task in _EVAL_TASKS
    }
    if (
        dict(task_counts["original"]) != expected_counts
        or dict(task_counts["counterfactual"]) != expected_counts
        or len(seen_pair_ids) != cfg.eval_pairs_per_task * len(_EVAL_TASKS)
    ):
        raise ValueError("eval task counts do not match the frozen contract")

    def source_bounds(path: Path) -> tuple[int, int]:
        minimum = None
        maximum = None
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                source_id = int(json.loads(line)["source_id"])
                minimum = source_id if minimum is None else min(minimum, source_id)
                maximum = source_id if maximum is None else max(maximum, source_id)
        if minimum is None or maximum is None:
            raise ValueError(f"graph is empty: {path.name}")
        return minimum, maximum

    found_changed_addresses = set()
    eval_minimum = None
    eval_maximum = None
    with eval_graph_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = int(row["source_id"])
            eval_minimum = (
                source_id
                if eval_minimum is None
                else min(eval_minimum, source_id)
            )
            eval_maximum = (
                source_id
                if eval_maximum is None
                else max(eval_maximum, source_id)
            )
            address = (
                source_id,
                str(row["relation_id"]),
                str(row["direction"]),
            )
            if address not in changed_rows:
                continue
            found_changed_addresses.add(address)
            base_row = _canonical_json(row)
            if base_row in changed_rows[address]:
                raise ValueError(
                    "counterfactual changed row equals its base supporting row"
                )
    if eval_minimum is None or eval_maximum is None:
        raise ValueError("eval graph is empty")
    if found_changed_addresses != set(changed_rows):
        raise ValueError("changed supporting row is absent from the eval graph")

    train_minimum, train_maximum = source_bounds(training_graph_path)
    if not (
        train_maximum < eval_minimum or eval_maximum < train_minimum
    ):
        raise ValueError(
            "fresh eval graph source ids overlap training source ids"
        )

    return {
        "exact_task_counts": True,
        "two_variants_per_pair": True,
        "answer_flips": True,
        "changed_supporting_row": True,
        "explicit_gold_actions": True,
        "fresh_sources_disjoint": True,
    }


def _world_artifact_signature(world) -> WorldArtifactSignature:
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
    row_address_sha256 = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return WorldArtifactSignature(
        world_id=world.world_id,
        world_seed=world.world_seed,
        fact_ids=tuple(fact.fact_id for fact in world.facts),
        row_address_sha256=row_address_sha256,
        fact_count=len(world.facts),
    )


def _reasoning_artifact_signature(
    artifact_id: str,
    metadata: dict,
) -> ReasoningArtifactSignature:
    return ReasoningArtifactSignature(
        artifact_id=artifact_id,
        world_id=metadata["world_id"],
        relation_path_hash=metadata["relation_path_hash"],
        template_id=metadata["template_id"],
        composition_split=metadata["composition_split"],
        hop_count=metadata["hop_count"],
        relations=tuple(metadata["relations"]),
    )


def _capture_split_expectations(
    split_name: str,
    worlds,
    qa_items,
    rendered_records,
) -> SplitArtifactExpectations:
    required_hops = (
        frozenset(range(1, 7))
        if split_name.startswith("protected_")
        else frozenset(range(1, 5))
    )
    return SplitArtifactExpectations(
        name=split_name,
        world_signatures=tuple(
            _world_artifact_signature(world) for world in worlds
        ),
        qa_signatures=tuple(
            _reasoning_artifact_signature(item.qid, item.meta)
            for item in qa_items
        ),
        rendered_signatures=tuple(
            _reasoning_artifact_signature(
                record.schedule.record_id,
                record.metadata,
            )
            for record in rendered_records
        ),
        required_hops=required_hops,
    )


def _write_split_audit_plan(
    out_dir: Path,
    split_name: str,
    snapshot,
) -> None:
    plan_dir = out_dir / "split-plans" / split_name
    plan_dir.mkdir(parents=True, exist_ok=True)
    worlds, qa_items, rendered_records = snapshot
    with (plan_dir / "worlds.jsonl").open("x") as stream:
        for world in worlds:
            stream.write(
                _json_line(
                    {
                        "world_id": world.world_id,
                        "world_seed": world.world_seed,
                        "split_name": world.split_name,
                        "manifest": world.manifest,
                        "facts": [
                            {
                                "fact_id": fact.fact_id,
                                "row": fact.row.as_json(),
                            }
                            for fact in world.facts
                        ],
                    }
                )
            )
    with (plan_dir / "qa.jsonl").open("x") as stream:
        for item in qa_items:
            stream.write(_json_line(asdict(item)))
    with (plan_dir / "rendered.jsonl").open("x") as stream:
        for record in rendered_records:
            stream.write(
                _json_line(
                    {
                        "record_id": record.schedule.record_id,
                        "metadata": record.metadata,
                    }
                )
            )


def _generate_split_audit_bundle(
    cfg: RelationalBuildConfig,
    tok,
    relation_schema: RelationSchema,
    split_plan: SplitPlan,
    split_name: str,
):
    partition = split_plan.partition(split_name)
    world = generate_world(
        0,
        WorldConfig(
            n_entities=cfg.world_size,
            seed=cfg.data_seed,
            schema=relation_schema,
            split_plan=split_plan,
            split_name=split_name,
            entity_id_offset=partition.entity_id_range[0],
            world_seed_offset=partition.world_seed_range[0],
        ),
    )
    qa_seed = cfg.data_seed ^ 0x51A17
    pairs = generate_eval_pairs(
        world,
        n_pairs_per_task=12,
        seed=qa_seed,
    )
    qa_items = tuple(
        item
        for pair in pairs
        for item in (pair.original, pair.counterfactual)
    )
    rendered = []
    split_ordinal = (
        "development",
        "train",
        "protected_seen",
        "protected_heldout",
    ).index(split_name)
    if split_name.startswith("protected_"):
        schedules = tuple(
            (
                cfg.data_seed ^ (0xA7100 + split_ordinal * 16 + hops),
                hops,
                8,
            )
            for hops in range(1, 7)
        )
    else:
        schedules = (
            (
                cfg.data_seed ^ (0xA7100 + split_ordinal * 16 + 4),
                4,
                24,
            ),
        )
    for seed, max_hops, count in schedules:
        rendered.extend(
            islice(
                iter_reasoning_records(
                    tok,
                    lambda world=world: iter((world,)),
                    seed=seed,
                    max_hops=max_hops,
                ),
                count,
            )
        )
    return (world,), qa_items, tuple(rendered)


def _freeze_and_audit_split_artifacts(
    cfg: RelationalBuildConfig,
    tok,
    relation_schema: RelationSchema,
    split_plan: SplitPlan,
    out_dir: Path,
) -> tuple[dict[str, str], Path, dict[str, bool]]:
    expectation_dir = out_dir / "expectations"
    expectation_dir.mkdir(parents=True, exist_ok=True)
    expectation_hashes: dict[str, str] = {}
    expectation_paths: dict[str, Path] = {}
    split_names = (
        "development",
        "train",
        "protected_seen",
        "protected_heldout",
    )
    for split_name in split_names:
        snapshot = _generate_split_audit_bundle(
            cfg,
            tok,
            relation_schema,
            split_plan,
            split_name,
        )
        _write_split_audit_plan(out_dir, split_name, snapshot)
        expectations = _capture_split_expectations(
            split_name,
            *snapshot,
        )
        path = expectation_dir / f"{split_name}.json"
        expectations.write(path)
        expectation_hashes[split_name] = expectations.sha256()
        expectation_paths[split_name] = path

    observed = {}
    for split_name in split_names:
        regenerated = _generate_split_audit_bundle(
            cfg,
            tok,
            relation_schema,
            split_plan,
            split_name,
        )
        expectations = SplitArtifactExpectations.from_path(
            expectation_paths[split_name]
        )
        if expectations.sha256() != expectation_hashes[split_name]:
            raise ValueError("persisted split expectations changed before audit")
        observed[split_name] = ObservedSplitArtifacts.from_generated(
            split_name,
            expectations=expectations,
            worlds=regenerated[0],
            qa_items=regenerated[1],
            rendered_records=regenerated[2],
        )
    require_disjointness(observed)
    checks = audit_disjointness(observed)
    audit_path = out_dir / "split-audit.json"
    _write_json(
        audit_path,
        {
            "version": 1,
            "expectation_sha256": dict(sorted(expectation_hashes.items())),
            "checks": checks,
        },
    )
    return expectation_hashes, audit_path, checks


_Q_HANDLE_RE = re.compile(r"(?<![A-Za-z0-9_])Q([0-9]+)(?![A-Za-z0-9_])")


def scan_bed_text_for_reserved_values(text: str, split_plan: SplitPlan) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("natural-text records must be nonempty strings")
    protected_partitions = (
        split_plan.protected_seen,
        split_plan.protected_heldout,
    )
    for match in _Q_HANDLE_RE.finditer(text):
        entity_id = int(match.group(1))
        if any(
            partition.entity_id_range[0]
            <= entity_id
            < partition.entity_id_range[1]
            for partition in protected_partitions
        ):
            raise LeakageError(
                f"bed text contains protected entity handle Q{entity_id}"
            )
    for partition in protected_partitions:
        if partition.payload_namespace in text:
            raise LeakageError(
                "bed text contains protected payload prefix "
                f"{partition.payload_namespace!r}"
            )


def _build_relational_corpus_plan(
    cfg: RelationalBuildConfig,
    tok,
    bed_iter,
    out_dir: Path | str,
    *,
    relation_schema: RelationSchema | None = None,
    split_plan: SplitPlan | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    if relation_schema is None:
        if split_plan is not None:
            raise ValueError("split_plan requires a relation schema")
        if cfg.artifact_mode != "fixture":
            raise ValueError(
                "development and protected builds require a relation schema"
            )
        expectation_hashes: dict[str, str] = {}
        split_audit_path: Path | None = None
        split_checks: dict[str, bool] = {}
    else:
        if split_plan is None:
            split_plan = build_split_plan(relation_schema, cfg.data_seed)
        if split_plan.schema_sha256 != relation_schema.sha256():
            raise ValueError("split plan does not match relation schema")
        relation_schema_path = out_dir / "relation-schema.json"
        relation_schema.write(relation_schema_path)
        split_plan_path = out_dir / "split-plan.json"
        split_plan_path.write_bytes(split_plan.canonical_bytes())
        expectation_hashes, split_audit_path, split_checks = (
            _freeze_and_audit_split_artifacts(
                cfg,
                tok,
                relation_schema,
                split_plan,
                out_dir,
            )
        )

        source_bed_iter = bed_iter

        def verified_bed_stream():
            for text in source_bed_iter:
                scan_bed_text_for_reserved_values(text, split_plan)
                yield text

        bed_iter = verified_bed_stream()
    shared_text_rows, bed_iter = _reserve_shared_text(
        bed_iter,
        tok,
        cfg.shared_text_eval_count,
    )

    policy, calibration = _calibrate_policy(
        cfg,
        relation_schema,
        split_plan,
    )
    policy_path = out_dir / "route-policy.json"
    _write_json(
        policy_path,
        {
            "schema_version": 1,
            "policy": asdict(policy),
            "policy_sha256": policy.sha256(),
            "write_cost_grid": list(WRITE_COST_GRID),
            "calibration": calibration,
        },
    )
    policy_manifest_path = out_dir / "policy-manifest.json"
    _write_json(
        policy_manifest_path,
        {
            "path": "route-policy.json",
            "sha256": _sha256_file(policy_path),
            "bytes": policy_path.stat().st_size,
        },
    )

    protected_stats_seed = cfg.data_seed ^ _ROUTE_STATS_SEED_XOR
    graph_path = out_dir / "graph.jsonl"
    graph_manifest, guardrail_data = _write_training_graph(
        cfg,
        policy,
        protected_stats_seed,
        graph_path,
        tok,
        relation_schema,
        split_plan,
    )
    graph_manifest_path = out_dir / "graph-manifest.json"
    _write_json(graph_manifest_path, graph_manifest)

    protected_eval_dirs: dict[str, Path] = {}
    if relation_schema is None:
        eval_report = _write_eval_sets(
            cfg,
            eval_dir,
            relation_schema,
            split_plan,
        )
    else:
        protected_reports = {}
        for protected_name in ("protected_seen", "protected_heldout"):
            protected_dir = eval_dir / protected_name
            protected_eval_dirs[protected_name] = protected_dir
            protected_reports[protected_name] = _write_eval_sets(
                cfg,
                protected_dir,
                relation_schema,
                split_plan,
                split_name=protected_name,
                artifact_prefix=f"eval/{protected_name}",
            )
        heldout_dir = protected_eval_dirs["protected_heldout"]
        for name in (
            "graph.jsonl",
            "original.jsonl",
            "counterfactual.jsonl",
            "worlds.jsonl",
        ):
            shutil.copyfile(heldout_dir / name, eval_dir / name)
        eval_report = {
            **protected_reports["protected_heldout"],
            "graph": "eval/graph.jsonl",
            "original": "eval/original.jsonl",
            "counterfactual": "eval/counterfactual.jsonl",
            "world_plan": "eval/worlds.jsonl",
            "production_splits": protected_reports,
        }
    guardrail_report = _write_guardrail_eval_files(
        eval_dir,
        guardrail_data,
        shared_text_rows,
    )
    eval_report.update(guardrail_report)
    eval_check_sets = [
        validate_eval_sets(
            cfg,
            graph_path,
            target_dir / "graph.jsonl",
            target_dir / "original.jsonl",
            target_dir / "counterfactual.jsonl",
        )
        for target_dir in (
            protected_eval_dirs.values()
            if protected_eval_dirs
            else (eval_dir,)
        )
    ]
    eval_checks = {
        name: all(checks[name] for checks in eval_check_sets)
        for name in eval_check_sets[0]
    }
    eval_report["checks"] = eval_checks
    graph_store_path: Path | None = None
    eval_graph_store_path: Path | None = None
    factual_graph_store_path: Path | None = None
    protected_graph_store_paths: dict[str, Path] = {}
    if relation_schema is not None:
        graph_codec = relation_schema.codec
        graph_store_path = out_dir / "graph.store"
        with PackedGraphStore.build(
            graph_store_path,
            _iter_graph_rows(graph_path),
            graph_codec,
        ) as graph_store:
            graph_manifest["packed_store"] = {
                "path": "graph.store",
                "rows": len(graph_store),
                "snapshot_sha256": graph_store.snapshot_sha256(),
            }
        _write_json(graph_manifest_path, graph_manifest)
        eval_graph_store_path = eval_dir / "graph.store"
        with PackedGraphStore.build(
            eval_graph_store_path,
            _iter_graph_rows(eval_dir / "graph.jsonl"),
            graph_codec,
        ) as eval_graph_store:
            eval_report["packed_graph"] = {
                "path": "eval/graph.store",
                "rows": len(eval_graph_store),
                "snapshot_sha256": eval_graph_store.snapshot_sha256(),
            }
        for protected_name, protected_dir in protected_eval_dirs.items():
            protected_store_path = protected_dir / "graph.store"
            protected_graph_store_paths[protected_name] = protected_store_path
            with PackedGraphStore.build(
                protected_store_path,
                _iter_graph_rows(protected_dir / "graph.jsonl"),
                graph_codec,
            ) as protected_store:
                eval_report["production_splits"][protected_name][
                    "packed_graph"
                ] = {
                    "path": f"eval/{protected_name}/graph.store",
                    "rows": len(protected_store),
                    "snapshot_sha256": protected_store.snapshot_sha256(),
                }
        factual_graph_store_path = eval_dir / "factual-graph.store"
        with PackedGraphStore.build(
            factual_graph_store_path,
            _iter_graph_rows(eval_dir / "factual-graph.jsonl"),
            graph_codec,
        ) as factual_graph_store:
            eval_report["packed_factual_graph"] = {
                "path": "eval/factual-graph.store",
                "rows": len(factual_graph_store),
                "snapshot_sha256": factual_graph_store.snapshot_sha256(),
            }
    eval_manifest_path = out_dir / "eval-manifest.json"
    _write_json(
        eval_manifest_path,
        {
            **eval_report,
            "sha256": {
                "graph": _sha256_file(eval_dir / "graph.jsonl"),
                "original": _sha256_file(eval_dir / "original.jsonl"),
                "counterfactual": _sha256_file(
                    eval_dir / "counterfactual.jsonl"
                ),
                **guardrail_report["guardrail_sha256"],
            },
            "production_split_sha256": {
                name: {
                    artifact: _sha256_file(directory / f"{artifact}.jsonl")
                    for artifact in (
                        "graph",
                        "original",
                        "counterfactual",
                        "worlds",
                    )
                }
                for name, directory in protected_eval_dirs.items()
            },
        },
    )

    rows_by_scope = {"train": _iter_graph_rows(graph_path)}
    if relation_schema is not None:
        for protected_scope in ("protected_seen", "protected_heldout"):
            rows_by_scope[protected_scope] = _iter_graph_rows(
                protected_eval_dirs[protected_scope] / "graph.jsonl"
            )
    payload_inventory = PayloadInventory.from_rows(tok, rows_by_scope)
    payload_inventory_path = out_dir / "payload-inventory.json"
    train_inventory_fact_ids = {
        entry.fact_id
        for entry in payload_inventory.entries
        if entry.scope == "train" and entry.field == "target"
    }
    sampled_factual_fact_ids = set(guardrail_data["factual_fact_ids"])
    if (
        not sampled_factual_fact_ids
        or not sampled_factual_fact_ids <= train_inventory_fact_ids
    ):
        raise ValueError(
            "factual guardrail sample is not bound to train payload inventory"
        )

    graph_factory = _CostTrackingWorldFactory(
        cfg,
        protected_stats_seed,
        relation_schema,
        split_plan,
        "train",
    )
    graph_records = iter_graph_records(tok, graph_factory)
    expected_alias_pairs = (
        []
        if relation_schema is None
        else [
            (relation.relation_id, alias)
            for relation in relation_schema.catalog
            for alias in relation.aliases
        ]
    )
    alias_records = (
        ()
        if relation_schema is None
        else tuple(iter_relation_alias_records(relation_schema))
    )
    observed_alias_pairs: list[tuple[str, str]] = []
    bed_records = iter_bed_records(bed_iter)
    reasoning_factories = {
        band: _CostTrackingWorldFactory(
            cfg,
            protected_stats_seed,
            relation_schema,
            split_plan,
            "train",
        )
        for band in (1, 2, 4)
    }
    reasoning_records = {
        band: iter_reasoning_records(
            tok,
            reasoning_factories[band],
            seed=cfg.data_seed ^ (0xA11CE + band),
            max_hops=band,
        )
        for band in (1, 2, 4)
    }

    schedule_path = out_dir / "schedule.jsonl"
    schedule_plan_path = out_dir / "schedule-plan.sqlite3"
    graph_subcomponents: Counter[str] = Counter()
    curriculum_records = {
        "early": Counter(),
        "middle": Counter(),
        "late": Counter(),
    }
    emitted: Counter[str] = Counter()
    max_record_tokens: Counter[str] = Counter()
    component_targets = cfg.component_shares
    mixture_manifest_path = out_dir / "mixture-manifest.json"
    targets = {
        component: cfg.total_tokens * share
        for component, share in component_targets.items()
    }
    rng = random.Random(cfg.data_seed ^ 0xA5A5A5A5)
    plan = SchedulePlanSpool(schedule_plan_path)
    writer: CorpusWriter | None = None
    try:
        for record in alias_records:
            token_ids, spans = _encode_record(tok, record, {})
            schedule_row = {
                "record_id": record.schedule.record_id,
                "exposure": record.schedule.exposure,
                "curriculum_band": record.schedule.curriculum_band,
                "metadata": record.metadata,
            }
            plan.add_record(
                component="graph",
                token_ids=token_ids,
                spans=spans,
                schedule=schedule_row,
            )
            observed_alias_pairs.append(
                (
                    str(record.metadata["relation_id"]),
                    str(record.metadata["alias"]),
                )
            )
            emitted["graph"] += len(token_ids)
            max_record_tokens["graph"] = max(
                max_record_tokens["graph"],
                len(token_ids),
            )
        while True:
            needed = [
                component
                for component in _COMPONENT_ORDER
                if emitted[component] < targets[component]
            ]
            if needed:
                component = max(
                    needed,
                    key=lambda name: (
                        targets[name] - emitted[name]
                    )
                    / targets[name],
                )
            else:
                break

            records_in_unit = 10 if component == "graph" else 1
            for _ in range(records_in_unit):
                graph_subcomponent = None
                if component == "bed":
                    try:
                        record = next(bed_records)
                    except StopIteration as error:
                        raise ValueError(
                            "natural-text stream ended before its token budget"
                        ) from error
                    costs: dict[str, FactCost] = {}
                elif component == "graph":
                    record = next(graph_records)
                    costs = graph_factory.costs
                    if record.schedule.record_id.startswith("rule-"):
                        graph_subcomponent = "rule"
                    else:
                        graph_subcomponent = graph_factory.audit_classes[
                            record.schedule.record_id
                        ]
                else:
                    relative_position = plan.total_tokens / cfg.total_tokens
                    if relative_position < 0.20:
                        phase = "early"
                        allowed = (1,)
                    elif relative_position < 0.50:
                        phase = "middle"
                        allowed = (1, 2)
                    else:
                        phase = "late"
                        allowed = (1, 2, 4)
                    band = min(
                        allowed,
                        key=lambda value: (
                            curriculum_records[phase][value],
                            allowed.index(value),
                        ),
                    )
                    record = next(reasoning_records[band])
                    costs = reasoning_factories[band].costs
                    curriculum_records[phase][band] += 1

                token_ids, spans = _encode_record(tok, record, costs)
                schedule_row = {
                    "record_id": record.schedule.record_id,
                    "exposure": record.schedule.exposure,
                    "curriculum_band": record.schedule.curriculum_band,
                }
                if record.metadata:
                    schedule_row["metadata"] = record.metadata
                if graph_subcomponent is not None:
                    schedule_row["graph_subcomponent"] = graph_subcomponent
                plan.add_record(
                    component=component,
                    token_ids=token_ids,
                    spans=spans,
                    schedule=schedule_row,
                )
                emitted[component] += len(token_ids)
                max_record_tokens[component] = max(
                    max_record_tokens[component],
                    len(token_ids),
                )
                if graph_subcomponent is not None:
                    graph_subcomponents[graph_subcomponent] += 1

        plan.seal()
        alias_exposure_receipt = {
            "expected_pairs": len(expected_alias_pairs),
            "observed_pairs": len(observed_alias_pairs),
            "pairs_sha256": hashlib.sha256(
                (
                    _canonical_json(sorted(observed_alias_pairs))
                    + "\n"
                ).encode()
            ).hexdigest(),
            "complete": observed_alias_pairs == expected_alias_pairs,
        }
        if not alias_exposure_receipt["complete"]:
            raise ValueError("relation alias exposure is incomplete")
        alias_exposure_sha256 = hashlib.sha256(
            (
                _canonical_json(alias_exposure_receipt)
                + "\n"
            ).encode()
        ).hexdigest()
        payload_inventory = payload_inventory.bind_expected_occurrences(
            plan.payload_occurrence_counts()
        )
        payload_inventory.write(payload_inventory_path)
        plan.write_schedule(schedule_path)
        if relation_schema is not None:
            expectation_hashes = freeze_production_split_expectations(
                out_dir
            )
        generation_expectation_path = (
            out_dir / "generation-expectations.json"
        )
        frozen_inputs = [
            graph_path,
            schedule_path,
            schedule_plan_path,
            payload_inventory_path,
            eval_dir / "graph.jsonl",
            eval_dir / "original.jsonl",
            eval_dir / "counterfactual.jsonl",
            eval_dir / "factual-graph.jsonl",
            out_dir
            / "split-plans"
            / "production"
            / "train-worlds.jsonl",
        ]
        for protected_dir in protected_eval_dirs.values():
            frozen_inputs.extend(
                protected_dir / name
                for name in (
                    "graph.jsonl",
                    "original.jsonl",
                    "counterfactual.jsonl",
                    "worlds.jsonl",
                )
            )
        if relation_schema is not None:
            frozen_inputs.extend(
                out_dir / "expectations" / f"{name}.json"
                for name in (
                    "development",
                    "train",
                    "protected_seen",
                    "protected_heldout",
                )
            )
        for store_path in (
            graph_store_path,
            eval_graph_store_path,
            factual_graph_store_path,
            *protected_graph_store_paths.values(),
        ):
            if store_path is not None:
                frozen_inputs.extend(_packed_store_artifacts(store_path))
        _write_json(
            generation_expectation_path,
            {
                "version": 1,
                "phase": "before_train_replay",
                "artifacts": {
                    path.relative_to(out_dir).as_posix(): {
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for path in sorted(frozen_inputs)
                },
                "payload_inventory_sha256": payload_inventory.sha256(),
                "split_expectation_sha256": dict(
                    sorted(expectation_hashes.items())
                ),
            },
        )
        writer = CorpusWriter(out_dir)
        for planned in plan.iter_records():
            spans = [
                EncodedSpan(
                    start=int(raw["start"]),
                    end=int(raw["end"]),
                    role=str(raw["role"]),
                    fact_id=raw["fact_id"],
                    fact_cost=(
                        None
                        if raw["fact_cost"] is None
                        else FactCost(**raw["fact_cost"])
                    ),
                    payload_field=raw["payload_field"],
                    payload_text=raw["payload_text"],
                )
                for raw in planned.spans
            ]
            writer.add(
                planned.component,
                planned.token_ids,
                spans,
                policy,
                rng,
            )
        if (
            writer.total != plan.total_tokens
            or writer.records != plan.total_records
        ):
            raise ValueError("schedule plan replay changed record totals")
    except BaseException:
        if writer is not None:
            writer.abort()
        plan.close()
        raise
    else:
        plan.close()
        assert writer is not None
        writer.close()
    assert writer is not None
    mask_audit = writer.finalize(
        seed=cfg.data_seed ^ 0xA5A5A5A5,
        payload_inventory=payload_inventory_path,
        record_schedule=schedule_path,
    )

    schedule_sha256 = _sha256_file(schedule_path)
    schedule_manifest_path = out_dir / "schedule-manifest.json"
    _write_json(
        schedule_manifest_path,
        {
            "path": "schedule.jsonl",
            "sha256": schedule_sha256,
            "bytes": schedule_path.stat().st_size,
            "records": writer.records,
            "tokens": writer.total,
            "component_tokens": dict(sorted(writer.component_tokens.items())),
            "component_records": dict(
                sorted(writer.component_records.items())
            ),
            "plan": {
                "path": schedule_plan_path.name,
                "artifact_class": "audit-only-disk-backed-plan",
                "sha256": _sha256_file(schedule_plan_path),
                "bytes": schedule_plan_path.stat().st_size,
            },
        },
    )

    component_shares = {
        component: writer.component_tokens[component] / writer.total
        for component in component_targets
    }
    mixture_deviation = max(
        abs(component_shares[component] - component_targets[component])
        for component in component_targets
    )
    rounding_tolerance = {
        component: max_record_tokens[component]
        * (10 if component == "graph" else 1)
        for component in component_targets
    }
    rounding_deviation = {
        component: writer.component_tokens[component] - targets[component]
        for component in component_targets
    }
    rounding_validated = all(
        0 <= rounding_deviation[component] <= rounding_tolerance[component]
        for component in component_targets
    )
    _write_json(
        mixture_manifest_path,
        {
            "version": 1,
            "artifact_mode": cfg.artifact_mode,
            "selected_index": cfg.development_mixture_index,
            "selected": component_targets,
            "ordered_development_fallbacks": list(DEVELOPMENT_MIXTURES),
            "actual": {
                "total_tokens": writer.total,
                "total_records": writer.records,
                "max_share_deviation": mixture_deviation,
                "component_tokens": dict(
                    sorted(writer.component_tokens.items())
                ),
                "component_records": dict(
                    sorted(writer.component_records.items())
                ),
            },
            "record_rounding": {
                "rule": (
                    "one maximum record per component; graph permits one "
                    "complete 7/2/1 ten-record block"
                ),
                "deviation_tokens": dict(sorted(rounding_deviation.items())),
                "tolerance_tokens": dict(sorted(rounding_tolerance.items())),
                "validated": rounding_validated,
            },
        },
    )

    mask_manifest_path = out_dir / "mask-manifest.json"
    _write_json(
        mask_manifest_path,
        {
            "ledger": {
                "path": "mask-ledger.jsonl",
                "sha256": _sha256_file(writer.ledger_path),
                "bytes": writer.ledger_path.stat().st_size,
            },
            "occurrence_spool": {
                "path": writer.occurrence_path.name,
                "sha256": _sha256_file(writer.occurrence_path),
                "bytes": writer.occurrence_path.stat().st_size,
            },
            "audit": {
                "path": writer.audit_path.name,
                "sha256": _sha256_file(writer.audit_path),
                "bytes": writer.audit_path.stat().st_size,
            },
            "sidecars": {
                condition: {
                    "path": path.name,
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for condition, path in writer.weight_paths.items()
            },
            "masked_tokens": dict(sorted(writer.masked_tokens.items())),
            "span_histograms": {
                condition: {
                    str(length): count
                    for length, count in sorted(histogram.items())
                }
                for condition, histogram in writer.span_histograms.items()
            },
            "position_histograms": {
                "expected": _position_histogram_json(
                    writer.expected_position_histogram
                ),
                "split": _position_histogram_json(
                    writer.position_histograms["split"]
                ),
                "random": _position_histogram_json(
                    writer.position_histograms["random"]
                ),
                "selective": _position_histogram_json(
                    writer.position_histograms["selective"]
                ),
            },
            "all_payload_ranges": {
                "expected_count": writer.expected_external_ranges,
                "actual_split_count": writer.actual_split_ranges,
                "expected_sha256": (
                    writer._expected_range_digest.hexdigest()
                ),
                "actual_split_sha256": (
                    writer._actual_range_digest.hexdigest()
                ),
            },
            "selective_payload_ranges": writer.selective_payload_ranges,
            "payload_inventory": {
                "path": payload_inventory_path.name,
                "sha256": payload_inventory.sha256(),
                "bytes": payload_inventory_path.stat().st_size,
            },
            "protected_target_tokens": writer.protected_target_tokens,
        },
    )

    graph_records_total = sum(graph_subcomponents.values())
    graph_mixture_exact = (
        graph_records_total > 0
        and graph_records_total % 10 == 0
        and graph_subcomponents["peripheral"] * 10
        == graph_records_total * 7
        and graph_subcomponents["central"] * 10
        == graph_records_total * 2
        and graph_subcomponents["rule"] * 10
        == graph_records_total
    )
    split_mass = writer.masked_tokens["split"]
    random_mass = writer.masked_tokens["random"]
    range_digests_match = (
        writer._expected_range_digest.hexdigest()
        == writer._actual_range_digest.hexdigest()
    )
    file_token_count = writer.token_path.stat().st_size // np.dtype(
        np.uint16
    ).itemsize
    sidecars_aligned = all(
        path.stat().st_size == file_token_count
        for path in writer.weight_paths.values()
    )
    checks = {
        "all_payload_coverage": (
            writer.all_payload_tokens == split_mass
            and split_mass > 0
        ),
        "all_payload_ranges_exact": (
            writer.expected_external_ranges == writer.actual_split_ranges
            and range_digests_match
            and writer.expected_length_histogram
            == writer.span_histograms["split"]
            and writer.expected_position_histogram
            == writer.position_histograms["split"]
        ),
        "selective_routes_exact": (
            writer.selective_payload_tokens
            == writer.masked_tokens["selective"]
            and writer.selective_payload_ranges
            == sum(writer.span_histograms["selective"].values())
        ),
        "random_mass_exact": random_mass == split_mass,
        "random_histogram_exact": (
            writer.span_histograms["split"]
            == writer.span_histograms["random"]
            and writer.position_histograms["split"]
            == writer.position_histograms["random"]
        ),
        "independent_mask_audit": (
            mask_audit.pending_random_demands == 0
            and mask_audit.dense_all_ones
            and mask_audit.protected_roles_unmasked
        ),
        "split_disjointness": (
            relation_schema is None or all(split_checks.values())
        ),
        "mixture_within_record_rounding_tolerance": rounding_validated,
        "graph_mixture_exact": graph_mixture_exact,
        "protected_roles_unmasked": writer.protected_roles_unmasked,
        "eval_validity": all(eval_checks.values()),
        "schedule_hash_stable": (
            _sha256_file(schedule_path) == schedule_sha256
        ),
        "sidecars_aligned": (
            sidecars_aligned
            and file_token_count == writer.total
            and writer.dense_all_ones
        ),
        "manifests_relative": True,
    }
    report = {
        "schema_version": 1,
        "config": asdict(cfg),
        "policy": {
            **asdict(policy),
            "sha256": policy.sha256(),
            "calibration_route_rate": calibration["route_rate"],
            "protected_route_rate": graph_manifest["route_rate"],
        },
        "tokens": {
            "total": writer.total,
            "components": {
                component: {
                    "tokens": writer.component_tokens[component],
                    "records": writer.component_records[component],
                    "share": component_shares[component],
                    "target_share": component_targets[component],
                }
                for component in component_targets
            },
            "graph_subcomponent_records": dict(
                sorted(graph_subcomponents.items())
            ),
            "curriculum_records": {
                phase: dict(sorted(counts.items()))
                for phase, counts in curriculum_records.items()
            },
        },
        "masks": {
            "all_payload_tokens": writer.all_payload_tokens,
            "expected_all_payload_ranges": writer.expected_external_ranges,
            "actual_split_ranges": writer.actual_split_ranges,
            "expected_range_sha256": (
                writer._expected_range_digest.hexdigest()
            ),
            "actual_split_range_sha256": (
                writer._actual_range_digest.hexdigest()
            ),
            "split_masked_tokens": split_mass,
            "random_masked_tokens": random_mass,
            "selective_masked_tokens": writer.masked_tokens["selective"],
            "selective_payload_ranges": writer.selective_payload_ranges,
            "protected_target_tokens": writer.protected_target_tokens,
            "split_span_histogram": {
                str(length): count
                for length, count in sorted(
                    writer.span_histograms["split"].items()
                )
            },
            "random_span_histogram": {
                str(length): count
                for length, count in sorted(
                    writer.span_histograms["random"].items()
                )
            },
            "expected_position_histogram": _position_histogram_json(
                writer.expected_position_histogram
            ),
            "split_position_histogram": _position_histogram_json(
                writer.position_histograms["split"]
            ),
            "random_position_histogram": _position_histogram_json(
                writer.position_histograms["random"]
            ),
            "selective_position_histogram": _position_histogram_json(
                writer.position_histograms["selective"]
            ),
            "unmatched_random_keys": {
                f"{length}:{relative_bin}": count
                for (length, relative_bin), count in sorted(
                    writer._pending_random.items()
                )
            },
        },
        "eval": eval_report,
        "guardrail_audit": {
            "factual_fact_ids": list(guardrail_data["factual_fact_ids"]),
        },
        "fact_routes": dict(sorted(guardrail_data["fact_routes"].items())),
        "payload_inventory": payload_inventory_path.name,
        "checks": checks,
    }
    report_path = out_dir / "report.json"
    _write_json(report_path, report)

    artifact_paths = [
        writer.token_path,
        *writer.weight_paths.values(),
        writer.ledger_path,
        writer.occurrence_path,
        writer.audit_path,
        graph_path,
        policy_path,
        schedule_path,
        schedule_plan_path,
        payload_inventory_path,
        generation_expectation_path,
        mixture_manifest_path,
        graph_manifest_path,
        policy_manifest_path,
        schedule_manifest_path,
        mask_manifest_path,
        eval_manifest_path,
        eval_dir / "graph.jsonl",
        eval_dir / "original.jsonl",
        eval_dir / "counterfactual.jsonl",
        eval_dir / "recognition.jsonl",
        eval_dir / "factual.jsonl",
        eval_dir / "factual-graph.jsonl",
        eval_dir / "internal.jsonl",
        eval_dir / "shared_text.jsonl",
        eval_dir / "route-audit.json",
        report_path,
    ]
    if relation_schema is not None:
        assert split_plan is not None
        assert split_audit_path is not None
        artifact_paths.extend(
            (
                out_dir / "relation-schema.json",
                out_dir / "split-plan.json",
                split_audit_path,
                *(
                    out_dir / "expectations" / f"{name}.json"
                    for name in sorted(expectation_hashes)
                ),
            )
        )
    for store_path in (
        graph_store_path,
        eval_graph_store_path,
        factual_graph_store_path,
        *protected_graph_store_paths.values(),
    ):
        if store_path is not None:
            artifact_paths.extend(_packed_store_artifacts(store_path))
    for audit_tree in (
        out_dir / "split-plans",
        *protected_eval_dirs.values(),
    ):
        if audit_tree.is_dir():
            artifact_paths.extend(
                path for path in audit_tree.rglob("*") if path.is_file()
            )
    artifact_paths = list(dict.fromkeys(artifact_paths))
    artifacts = sorted(
        (_artifact(out_dir, path) for path in artifact_paths),
        key=lambda value: value["path"],
    )
    checks["manifests_relative"] = all(
        not Path(artifact["path"]).is_absolute()
        and ".." not in Path(artifact["path"]).parts
        for artifact in artifacts
    )
    if not checks["manifests_relative"]:
        raise ValueError("artifact manifests must contain only relative paths")
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"relational corpus checks failed: {failed}")

    manifest_path = out_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "alias_exposure_receipt": alias_exposure_receipt,
            "alias_exposure_sha256": alias_exposure_sha256,
            "expectation_sha256": dict(sorted(expectation_hashes.items())),
            "generation_expectation_sha256": _sha256_file(
                generation_expectation_path
            ),
            "payload_inventory_sha256": payload_inventory.sha256(),
            "audit_sha256": {
                "mask": _sha256_file(writer.audit_path),
                **(
                    {"split": _sha256_file(split_audit_path)}
                    if split_audit_path is not None
                    else {}
                ),
            },
            "artifacts": artifacts,
        },
    )
    return report


def _validate_protected_freeze(
    cfg: RelationalBuildConfig,
    relation_schema: RelationSchema | None,
    freeze_manifest,
):
    if freeze_manifest is None:
        raise ValueError(
            "a passed frozen study receipt is required for protected generation"
        )
    from scripts.freeze_relational_study import require_launchable_freeze
    from scripts.make_relational_manifest import round_raw_positions

    freeze = require_launchable_freeze(freeze_manifest)
    selected = {
        "bed": freeze.selected_mixture[0],
        "graph": freeze.selected_mixture[1],
        "reasoning": freeze.selected_mixture[2],
    }
    if cfg.component_shares != selected:
        raise ValueError(
            "protected corpus mixture does not match the frozen Gate-2 choice"
        )
    if cfg.data_seed not in freeze.seeds:
        raise ValueError("protected corpus data seed is not frozen")
    allowed = {
        (
            freeze.low_entities,
            round_raw_positions(
                freeze.model_parameters["d160m"]
                * freeze.tokens_per_parameter,
                tokens_per_step=freeze.tokens_per_step,
            ).actual_raw_positions,
        ),
        (
            freeze.high_entities,
            round_raw_positions(
                freeze.model_parameters["d160m"]
                * freeze.tokens_per_parameter,
                tokens_per_step=freeze.tokens_per_step,
            ).actual_raw_positions,
        ),
        (
            freeze.confirmation_entities,
            round_raw_positions(
                freeze.model_parameters["d360m"]
                * freeze.tokens_per_parameter,
                tokens_per_step=freeze.tokens_per_step,
            ).actual_raw_positions,
        ),
    }
    if (cfg.n_entities, cfg.total_tokens) not in allowed:
        raise ValueError(
            "protected corpus entity load or raw-position budget is not frozen"
        )
    protected_settings = {
        "world_size": 64,
        "eval_pairs_per_task": 10_000,
        "eval_pairs_per_world": 32,
        "route_stats_pairs_per_task": 64,
        "guardrail_items": 10_000,
        "shared_text_eval_count": 64,
    }
    drifted_settings = sorted(
        name
        for name, expected in protected_settings.items()
        if getattr(cfg, name) != expected
    )
    if drifted_settings:
        raise ValueError(
            "protected corpus settings are not frozen: "
            f"{drifted_settings}"
        )
    if relation_schema is None:
        raise ValueError("protected corpus requires the frozen relation schema")
    if (
        relation_schema.sha256()
        != freeze.artifact_sha256["relation_schema"]
    ):
        raise ValueError("protected relation schema hash does not match freeze")
    return freeze


def _protected_build_metadata(
    cfg: RelationalBuildConfig,
    freeze_manifest,
) -> dict:
    from scripts.make_relational_manifest import (
        protected_build_metadata,
        round_raw_positions,
    )

    candidates = (
        ("d160m", "low", freeze_manifest.low_entities),
        ("d160m", "high", freeze_manifest.high_entities),
        (
            "d360m",
            "confirmation",
            freeze_manifest.confirmation_entities,
        ),
    )
    matched = [
        (model, load)
        for model, load, entities in candidates
        if cfg.n_entities == entities
        and cfg.total_tokens
        == round_raw_positions(
            freeze_manifest.model_parameters[model]
            * freeze_manifest.tokens_per_parameter,
            tokens_per_step=freeze_manifest.tokens_per_step,
        ).actual_raw_positions
    ]
    if len(matched) != 1:
        raise ValueError("protected corpus build identity is ambiguous")
    model, load = matched[0]
    return protected_build_metadata(
        freeze_manifest,
        model=model,
        load=load,
        entities=cfg.n_entities,
        seed=cfg.data_seed,
    )


def build_relational_corpus(
    cfg: RelationalBuildConfig,
    tok,
    bed_iter,
    out_dir: Path | str,
    *,
    relation_schema: RelationSchema | None = None,
    split_plan: SplitPlan | None = None,
    freeze_manifest=None,
) -> dict:
    """Plan on disk, replay into staging, audit, then publish atomically."""

    protected_freeze = (
        _validate_protected_freeze(cfg, relation_schema, freeze_manifest)
        if cfg.artifact_mode == "protected"
        else None
    )
    protected_build = (
        _protected_build_metadata(cfg, protected_freeze)
        if protected_freeze is not None
        else None
    )
    destination = Path(out_dir)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"relational corpus destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan_dir = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.plan-",
        )
    )
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
        )
    )
    shutil.rmtree(staging_dir)
    try:
        report = _build_relational_corpus_plan(
            cfg,
            tok,
            bed_iter,
            plan_dir,
            relation_schema=relation_schema,
            split_plan=split_plan,
        )
        frozen_split_hashes = (
            {
                name: SplitArtifactExpectations.from_path(
                    plan_dir / "expectations" / f"{name}.json"
                ).sha256()
                for name in (
                    "development",
                    "train",
                    "protected_seen",
                    "protected_heldout",
                )
            }
            if relation_schema is not None
            else {}
        )
        frozen_generation_hash = _sha256_file(
            plan_dir / "generation-expectations.json"
        )
        freeze_published_artifact_expectations(plan_dir)
        shutil.copytree(plan_dir, staging_dir)
        audit_path = write_published_artifact_audit(
            staging_dir,
            None if relation_schema is None else relation_schema.codec,
            frozen_split_expectation_sha256=frozen_split_hashes,
            frozen_generation_expectation_sha256=(
                frozen_generation_hash
            ),
            tok=None if relation_schema is None else tok,
        )
        if frozen_split_hashes:
            published_audit = json.loads(audit_path.read_bytes())
            _write_json(
                staging_dir / "split-audit.json",
                {
                    "version": 2,
                    "phase": "post_publication_reopen",
                    "expectation_sha256": dict(
                        sorted(frozen_split_hashes.items())
                    ),
                    "checks": published_audit["split_checks"],
                },
            )
        manifest_path = staging_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if frozen_split_hashes:
            manifest["expectation_sha256"] = dict(
                sorted(frozen_split_hashes.items())
            )
            manifest["audit_sha256"]["split"] = _sha256_file(
                staging_dir / "split-audit.json"
            )
        manifest["artifacts"] = [
            _artifact(staging_dir, staging_dir / item["path"])
            for item in manifest["artifacts"]
        ]
        manifest["published_expectation_sha256"] = _sha256_file(
            staging_dir / EXPECTATION_NAME
        )
        manifest["audit_sha256"]["published"] = _sha256_file(audit_path)
        if protected_freeze is not None:
            manifest["study_freeze_sha256"] = (
                protected_freeze.freeze_sha256
            )
            manifest["protected_build"] = protected_build
        manifest["artifacts"].extend(
            (
                _artifact(staging_dir, staging_dir / EXPECTATION_NAME),
                _artifact(staging_dir, staging_dir / AUDIT_NAME),
            )
        )
        manifest["artifacts"] = sorted(
            {
                item["path"]: item for item in manifest["artifacts"]
            }.values(),
            key=lambda item: item["path"],
        )
        _write_json(manifest_path, manifest)
        write_published_artifact_audit(
            staging_dir,
            None if relation_schema is None else relation_schema.codec,
            frozen_split_expectation_sha256=frozen_split_hashes,
            frozen_generation_expectation_sha256=(
                frozen_generation_hash
            ),
            tok=None if relation_schema is None else tok,
        )
        if _sha256_file(staging_dir / AUDIT_NAME) != manifest[
            "audit_sha256"
        ]["published"]:
            raise ValueError("published artifact audit hash was not stable")
        if os.path.lexists(destination):
            raise FileExistsError(
                "relational corpus destination appeared during publication: "
                f"{destination}"
            )
        os.replace(staging_dir, destination)
        report["publication_audit"] = {
            "expectation_sha256": manifest[
                "published_expectation_sha256"
            ],
            "audit_sha256": manifest["audit_sha256"]["published"],
        }
        return report
    finally:
        shutil.rmtree(plan_dir, ignore_errors=True)
        shutil.rmtree(staging_dir, ignore_errors=True)
