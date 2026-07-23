from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np

from corpusgen.graph_records import (
    GraphAction,
    RANDOM_CONTROL_POSITION_BINS,
    RenderedRecord,
    relative_position_bin,
)
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
    iter_bed_records,
    iter_graph_records,
    iter_reasoning_records,
    iter_worlds,
    make_factual_recall_item,
)


WRITE_COST_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
COMPONENT_SHARES = {"bed": 0.45, "graph": 0.30, "reasoning": 0.25}
POSITION_BIN_COUNT = RANDOM_CONTROL_POSITION_BINS
MAX_GRAPH_ACTION_SLOTS = 12
MAX_GRAPH_READS = 10
_RANDOM_CANDIDATE_POOL_LIMIT = 64
_EVAL_TASKS = (
    "path_composition",
    "date_ordering",
    "balanced_equality",
)
_PROTECTED_TARGET_ROLES = {
    "rule",
    "action",
    "provisional_answer",
    "final_answer",
}
_COMPONENT_ORDER = tuple(COMPONENT_SHARES)
_EVAL_WORLD_ID = 1 << 31
_ROUTE_STATS_SEED_XOR = 0x13579BDF
_EVAL_SEED_XOR = 0x0E1A15E7


@dataclass(frozen=True)
class FactCost:
    fact_id: str
    entropy: float
    exposures: int
    expected_reads: float
    expected_hops: float


@dataclass(frozen=True)
class EncodedSpan:
    start: int
    end: int
    role: str
    fact_id: str | None = None
    fact_cost: FactCost | None = None


@dataclass(frozen=True)
class ExpectedExternalRange:
    start: int
    end: int
    fact_id: str


@dataclass(frozen=True)
class RandomControlCandidate:
    start: int
    end: int
    record_index: int
    component: str
    position_bin: int


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


def load_route_policy(
    path: Path | str,
    *,
    expected_policy_sha256: str,
) -> tuple[RoutePolicy, dict, bytes]:
    """Load and authenticate one frozen route-policy document."""

    policy_path = Path(path)
    if not policy_path.is_file() or policy_path.is_symlink():
        raise ValueError("route policy must be a regular non-symlink file")
    if (
        not isinstance(expected_policy_sha256, str)
        or len(expected_policy_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_policy_sha256
        )
    ):
        raise ValueError(
            "expected route policy SHA-256 must be 64 lowercase hex characters"
        )

    raw = policy_path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("route policy must contain valid JSON") from error
    required = {
        "schema_version",
        "policy",
        "policy_sha256",
        "calibration",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("route policy fields do not match the frozen schema")
    if document["schema_version"] != 1:
        raise ValueError("route policy schema_version must be 1")
    if not isinstance(document["calibration"], dict):
        raise ValueError("route policy calibration metadata must be a mapping")

    raw_policy = document["policy"]
    policy_fields = {"write_cost", "read_cost", "hop_cost"}
    if not isinstance(raw_policy, dict) or set(raw_policy) != policy_fields:
        raise ValueError("route policy cost fields do not match the schema")
    costs = {}
    for name in policy_fields:
        value = raw_policy[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"route policy {name} must be finite and non-negative")
        costs[name] = float(value)

    policy = RoutePolicy(**costs)
    actual_sha256 = policy.sha256()
    if document["policy_sha256"] != actual_sha256:
        raise ValueError("route policy declared SHA-256 does not match its costs")
    if actual_sha256 != expected_policy_sha256:
        raise ValueError(
            "expected route policy SHA-256 does not match the policy input"
        )
    return policy, document, raw


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


def _histogram_difference(
    expected: Counter,
    actual: Counter,
) -> float:
    keys = set(expected) | set(actual)
    mismatch = sum(abs(expected[key] - actual[key]) for key in keys)
    return mismatch / max(sum(expected.values()), 1)


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


class _CostTrackingWorldFactory:
    def __init__(self, cfg: RelationalBuildConfig, stats_seed: int):
        self.cfg = cfg
        self.stats_seed = stats_seed
        self.costs: dict[str, FactCost] = {}
        self.audit_classes: dict[str, str] = {}

    def __call__(self):
        for world in iter_worlds(
            self.cfg.n_entities,
            self.cfg.world_size,
            self.cfg.data_seed,
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
    spans: list[EncodedSpan] = []
    start = 0
    while start < len(ids):
        role = roles[start]
        fact_id = fact_ids[start]
        end = start + 1
        while (
            end < len(ids)
            and roles[end] == role
            and fact_ids[end] == fact_id
        ):
            end += 1
        fact_cost = None
        if role == "payload":
            if fact_id not in costs:
                raise ValueError(f"missing route statistics for fact {fact_id}")
            fact_cost = costs[fact_id]
        spans.append(
            EncodedSpan(start, end, role, fact_id, fact_cost)
        )
        start = end

    spans.append(EncodedSpan(len(ids), len(ids) + 1, "boundary"))
    ids.append(tok.EOT)
    if any(token_id < 0 or token_id >= 1 << 16 for token_id in ids):
        raise ValueError("token id does not fit uint16")
    return np.asarray(ids, dtype=np.uint16), spans


def derive_weights(
    condition: str,
    spans: list[EncodedSpan],
    policy: RoutePolicy,
    rng: random.Random,
) -> np.ndarray:
    weights, _ = _derive_weight_result(condition, spans, policy, rng)
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


def _derive_weight_result(
    condition: str,
    spans: list[EncodedSpan],
    policy: RoutePolicy,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, EncodedSpan]]]:
    length = max((span.end for span in spans), default=0)
    weights = np.ones(length, dtype=np.uint8)
    external = [
        span
        for span in spans
        if (
            span.role == "payload"
            and span.fact_cost is not None
            and policy.is_external(span.fact_cost)
        )
    ]

    if condition == "dense":
        return weights, []
    if condition == "split":
        masked = []
        for span in external:
            weights[span.start : span.end] = 0
            masked.append((span.start, span.end, span))
        return weights, masked
    if condition != "random":
        raise ValueError(f"unknown target-weight condition: {condition}")

    available = [
        (span.start, span.end, span)
        for span in spans
        if span.role == "random_control" and span.end > span.start
    ]
    masked = []
    for source in external:
        span_length = source.end - source.start
        source_bin = position_bin(
            source.start,
            source.end,
            length,
        )
        candidates = []
        for index, (start, end, plain_span) in enumerate(available):
            if end - start != span_length:
                continue
            if position_bin(start, end, length) != source_bin:
                continue
            candidates.append((index, start, plain_span))
        if not candidates:
            raise ValueError(
                "record lacks a random-control span matching external payload "
                f"key ({span_length}, {source_bin})"
            )
        index, start, plain_span = candidates[rng.randrange(len(candidates))]
        _, old_end, _ = available.pop(index)
        end = start + span_length
        if end != old_end:
            raise AssertionError("random-control candidates must match exactly")
        weights[start:end] = 0
        masked.append((start, end, plain_span))
    return weights, masked


class SharedCorpusWriter:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.token_path = out_dir / "train.bin"
        self.weight_paths = {
            condition: out_dir / f"{condition}.weights.bin"
            for condition in ("dense", "split", "random")
        }
        self.ledger_path = out_dir / "mask-ledger.jsonl"
        self.token_file = self.token_path.open("wb")
        self.weight_files = {
            "dense": self.weight_paths["dense"].open("wb"),
            "split": self.weight_paths["split"].open("wb"),
            "random": self.weight_paths["random"].open(
                "w+b",
                buffering=0,
            ),
        }
        self.ledger_file = self.ledger_path.open("w")
        self.total = 0
        self.records = 0
        self.component_tokens: Counter[str] = Counter()
        self.component_records: Counter[str] = Counter()
        self.external_payload_tokens = 0
        self.protected_target_tokens = 0
        self.masked_tokens: Counter[str] = Counter()
        self.span_histograms = {
            "split": Counter(),
            "random": Counter(),
        }
        self.position_histograms = {
            "split": Counter(),
            "random": Counter(),
        }
        self.expected_position_histogram: Counter[
            tuple[int, int]
        ] = Counter()
        self.expected_length_histogram: Counter[int] = Counter()
        self.expected_external_ranges = 0
        self.actual_split_ranges = 0
        self._expected_range_digest = hashlib.sha256()
        self._actual_range_digest = hashlib.sha256()
        self._pending_random: Counter[tuple[int, int]] = Counter()
        self._candidate_pools: dict[
            tuple[int, int],
            list[RandomControlCandidate],
        ] = {}
        self._candidate_seen: Counter[tuple[int, int]] = Counter()
        self.protected_roles_unmasked = True
        self.dense_all_ones = True
        self._closed = False

    @staticmethod
    def _range_bytes(start: int, end: int, fact_id: str) -> bytes:
        return _json_line(
            {"start": start, "end": end, "fact_id": fact_id}
        ).encode()

    def _select_random_candidate(
        self,
        candidate: RandomControlCandidate,
    ) -> None:
        handle = self.weight_files["random"]
        return_position = handle.tell()
        handle.seek(candidate.start)
        handle.write(bytes(candidate.end - candidate.start))
        handle.seek(return_position)

        key = (
            candidate.end - candidate.start,
            candidate.position_bin,
        )
        self.masked_tokens["random"] += key[0]
        self.span_histograms["random"][key[0]] += 1
        self.position_histograms["random"][key] += 1
        self.ledger_file.write(
            _json_line(
                {
                    "component": candidate.component,
                    "condition": "random",
                    "record_index": candidate.record_index,
                    "start": candidate.start,
                    "end": candidate.end,
                    "length": key[0],
                    "position_bin": key[1],
                    "role": "random_control",
                }
            )
        )

    def _drain_random_pool(
        self,
        key: tuple[int, int],
        rng: random.Random,
    ) -> None:
        pool = self._candidate_pools.get(key, [])
        while self._pending_random[key] and pool:
            candidate = pool.pop(rng.randrange(len(pool)))
            self._pending_random[key] -= 1
            self._select_random_candidate(candidate)
        if not self._pending_random[key]:
            del self._pending_random[key]
        if pool:
            self._candidate_pools[key] = pool
        else:
            self._candidate_pools.pop(key, None)

    def _offer_random_candidate(
        self,
        key: tuple[int, int],
        candidate: RandomControlCandidate,
        rng: random.Random,
    ) -> None:
        self._candidate_seen[key] += 1
        if self._pending_random[key]:
            self._pending_random[key] -= 1
            if not self._pending_random[key]:
                del self._pending_random[key]
            self._select_random_candidate(candidate)
            return

        pool = self._candidate_pools.setdefault(key, [])
        if len(pool) < _RANDOM_CANDIDATE_POOL_LIMIT:
            pool.append(candidate)
            return
        replacement = rng.randrange(self._candidate_seen[key])
        if replacement < _RANDOM_CANDIDATE_POOL_LIMIT:
            pool[replacement] = candidate

    def add(
        self,
        component: str,
        token_ids: np.ndarray,
        spans: list[EncodedSpan],
        policy: RoutePolicy,
        rng: random.Random,
    ) -> None:
        if token_ids.dtype != np.uint16 or token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional uint16 array")
        expected = collect_expected_external_ranges(spans, policy)
        dense, _ = _derive_weight_result("dense", spans, policy, rng)
        split, split_ranges = _derive_weight_result(
            "split",
            spans,
            policy,
            rng,
        )
        validate_split_coverage(expected, spans, split, split_ranges)
        random_control = np.ones(len(token_ids), dtype=np.uint8)
        if len(dense) != len(token_ids) or len(split) != len(token_ids):
            raise ValueError("target weights must align with token ids")

        record_start = self.total
        self.token_file.write(token_ids.tobytes())
        self.weight_files["dense"].write(dense.tobytes())
        self.weight_files["split"].write(split.tobytes())
        self.weight_files["random"].write(random_control.tobytes())
        self.dense_all_ones &= bool(dense.all())

        touched_keys = set()
        for item in expected:
            start = record_start + item.start
            end = record_start + item.end
            length = item.end - item.start
            relative_bin = position_bin(
                item.start,
                item.end,
                len(token_ids),
            )
            key = (length, relative_bin)
            touched_keys.add(key)
            self._pending_random[key] += 1
            self.expected_external_ranges += 1
            self.external_payload_tokens += length
            self.expected_length_histogram[length] += 1
            self.expected_position_histogram[key] += 1
            digest_value = self._range_bytes(start, end, item.fact_id)
            self._expected_range_digest.update(digest_value)
            self.ledger_file.write(
                _json_line(
                    {
                        "component": component,
                        "condition": "expected_split",
                        "record_index": self.records,
                        "start": start,
                        "end": end,
                        "length": length,
                        "position_bin": relative_bin,
                        "role": "payload",
                        "fact_id": item.fact_id,
                    }
                )
            )
        for key in sorted(touched_keys):
            self._drain_random_pool(key, rng)

        for start, end, span in split_ranges:
            length = end - start
            relative_bin = position_bin(start, end, len(token_ids))
            global_start = record_start + start
            global_end = record_start + end
            self.actual_split_ranges += 1
            self.masked_tokens["split"] += length
            self.span_histograms["split"][length] += 1
            self.position_histograms["split"][
                (length, relative_bin)
            ] += 1
            if span.fact_id is None:
                raise ValueError("Split payload range requires a fact id")
            self._actual_range_digest.update(
                self._range_bytes(
                    global_start,
                    global_end,
                    span.fact_id,
                )
            )
            self.ledger_file.write(
                _json_line(
                    {
                        "component": component,
                        "condition": "split",
                        "record_index": self.records,
                        "start": global_start,
                        "end": global_end,
                        "length": length,
                        "position_bin": relative_bin,
                        "role": span.role,
                        "fact_id": span.fact_id,
                    }
                )
            )

        for span in spans:
            if span.role != "random_control":
                continue
            length = span.end - span.start
            relative_bin = position_bin(
                span.start,
                span.end,
                len(token_ids),
            )
            key = (length, relative_bin)
            self._offer_random_candidate(
                key,
                RandomControlCandidate(
                    start=record_start + span.start,
                    end=record_start + span.end,
                    record_index=self.records,
                    component=component,
                    position_bin=relative_bin,
                ),
                rng,
            )

        for span in spans:
            if span.role != "payload" and not split[span.start : span.end].all():
                self.protected_roles_unmasked = False
            if span.role in _PROTECTED_TARGET_ROLES:
                self.protected_target_tokens += span.end - span.start

        self.total += len(token_ids)
        self.records += 1
        self.component_tokens[component] += len(token_ids)
        self.component_records[component] += 1

    def close(self) -> None:
        if self._closed:
            return
        self.token_file.close()
        for handle in self.weight_files.values():
            handle.close()
        self.ledger_file.close()
        self._closed = True


CorpusWriter = SharedCorpusWriter


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


def _write_training_graph(
    cfg: RelationalBuildConfig,
    policy: RoutePolicy,
    stats_seed: int,
    path: Path,
    tok,
) -> tuple[dict, dict]:
    rows = 0
    external = 0
    tail_total = 0
    tail_external = 0
    structure_total = 0
    structure_internal = 0
    recognition_sources = []
    factual_sources = []
    central_sources = []
    central_target = (cfg.guardrail_items + 1) // 2
    with path.open("w") as handle:
        for world in iter_worlds(
            cfg.n_entities,
            cfg.world_size,
            cfg.data_seed,
        ):
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
                if (
                    routed
                    and fact.row.target_kind == "entity"
                    and len(factual_sources) < cfg.guardrail_items
                ):
                    factual_sources.append((world, fact))
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
            # Route-audit rules are one always-internal accounting unit per
            # world, not one unit per emitted rule training record.
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
    if not factual_sources:
        raise ValueError("no external entity facts for factual evaluation")
    factual = []
    for ordinal in range(cfg.guardrail_items):
        world, fact = factual_sources[ordinal % len(factual_sources)]
        factual.append(
            make_factual_recall_item(world, fact, ordinal)
        )
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
        fact.row.address: fact.row for _, fact in factual_sources
    }
    manifest = {
        "path": path.name,
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
    }


def _write_eval_sets(cfg: RelationalBuildConfig, eval_dir: Path) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    graph_path = eval_dir / "graph.jsonl"
    original_path = eval_dir / "original.jsonl"
    counterfactual_path = eval_dir / "counterfactual.jsonl"
    remaining = cfg.eval_pairs_per_task
    world_index = 0
    graph_rows = 0
    item_count = 0
    eval_seed = cfg.data_seed ^ _EVAL_SEED_XOR

    with (
        graph_path.open("w") as graph_file,
        original_path.open("w") as original_file,
        counterfactual_path.open("w") as counterfactual_file,
    ):
        while remaining:
            pairs_per_task = min(cfg.eval_pairs_per_world, remaining)
            world = generate_world(
                _EVAL_WORLD_ID + world_index,
                WorldConfig(n_entities=cfg.world_size, seed=eval_seed),
            )
            for fact in world.facts:
                graph_file.write(_json_line(fact.row.as_json()))
                graph_rows += 1
            pairs = generate_eval_pairs(
                world,
                pairs_per_task,
                eval_seed,
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
        "graph": "eval/graph.jsonl",
        "original": "eval/original.jsonl",
        "counterfactual": "eval/counterfactual.jsonl",
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
        action_slots = int(meta.get("action_slots", 6))
        if (
            not isinstance(raw_actions, list)
            or action_slots not in (6, MAX_GRAPH_ACTION_SLOTS)
            or len(raw_actions) != action_slots
        ):
            raise ValueError("gold actions must contain exactly 6 or 12 steps")
        legacy_fields = {
            "source_slot",
            "relation_id",
            "direction",
            "read",
            "halt",
        }
        actions = []
        for raw in raw_actions:
            if not isinstance(raw, dict) or set(raw) not in (
                legacy_fields,
                legacy_fields | {"page"},
            ):
                raise ValueError("gold actions have invalid fields")
            actions.append(GraphAction(**raw))
        halts = [
            index for index, action in enumerate(actions) if action.halt
        ]
        if len(halts) != 1:
            raise ValueError("gold actions require exactly one HALT")
        halt = halts[0]
        if not all(action.read for action in actions[:halt]):
            raise ValueError("gold actions before HALT must be reads")
        if any(
            action.read or action.halt for action in actions[halt + 1 :]
        ):
            raise ValueError("gold actions after HALT must be NOOP")
        addresses = meta["gold_addresses"]
        reads = [action for action in actions if action.read]
        if len(reads) > MAX_GRAPH_READS:
            raise ValueError("gold actions may contain at most ten reads")
        if len(reads) != len(addresses) or any(
            action.relation_id != str(address[1])
            or action.direction != str(address[2])
            or action.page != (int(address[3]) if len(address) == 4 else 0)
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


def build_relational_corpus(
    cfg: RelationalBuildConfig,
    tok,
    bed_iter,
    out_dir: Path | str,
    *,
    route_policy_path: Path | str,
    expected_policy_sha256: str,
) -> dict:
    policy, policy_document, policy_bytes = load_route_policy(
        route_policy_path,
        expected_policy_sha256=expected_policy_sha256,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    shared_text_rows, bed_iter = _reserve_shared_text(
        bed_iter,
        tok,
        cfg.shared_text_eval_count,
    )

    policy_path = out_dir / "route-policy.json"
    policy_path.write_bytes(policy_bytes)
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
    )
    graph_manifest_path = out_dir / "graph-manifest.json"
    _write_json(graph_manifest_path, graph_manifest)

    eval_report = _write_eval_sets(cfg, eval_dir)
    guardrail_report = _write_guardrail_eval_files(
        eval_dir,
        guardrail_data,
        shared_text_rows,
    )
    eval_report.update(guardrail_report)
    eval_checks = validate_eval_sets(
        cfg,
        graph_path,
        eval_dir / "graph.jsonl",
        eval_dir / "original.jsonl",
        eval_dir / "counterfactual.jsonl",
    )
    eval_report["checks"] = eval_checks
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
        },
    )

    graph_factory = _CostTrackingWorldFactory(cfg, protected_stats_seed)
    graph_records = iter_graph_records(tok, graph_factory)
    bed_records = iter_bed_records(bed_iter)
    reasoning_factories = {
        band: _CostTrackingWorldFactory(cfg, protected_stats_seed)
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

    writer = SharedCorpusWriter(out_dir)
    schedule_path = out_dir / "schedule.jsonl"
    schedule_digest = hashlib.sha256()
    graph_subcomponents: Counter[str] = Counter()
    curriculum_records = {
        "early": Counter(),
        "middle": Counter(),
        "late": Counter(),
    }
    emitted: Counter[str] = Counter()
    targets = {
        component: cfg.total_tokens * share
        for component, share in COMPONENT_SHARES.items()
    }
    rng = random.Random(cfg.data_seed ^ 0xA5A5A5A5)

    try:
        with schedule_path.open("w") as schedule_file:
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
                elif sum(graph_subcomponents.values()) % 10:
                    component = "graph"
                else:
                    break

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
                    relative_position = writer.total / cfg.total_tokens
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
                token_start = writer.total
                writer.add(component, token_ids, spans, policy, rng)
                emitted[component] += len(token_ids)
                if graph_subcomponent is not None:
                    graph_subcomponents[graph_subcomponent] += 1
                schedule_row = {
                    "component": component,
                    "record_id": record.schedule.record_id,
                    "exposure": record.schedule.exposure,
                    "curriculum_band": record.schedule.curriculum_band,
                    "token_start": token_start,
                    "token_end": writer.total,
                }
                if graph_subcomponent is not None:
                    schedule_row["graph_subcomponent"] = graph_subcomponent
                line = _json_line(schedule_row)
                schedule_file.write(line)
                schedule_digest.update(line.encode())
    finally:
        writer.close()

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
            },
            "external_payload_ranges": {
                "expected_count": writer.expected_external_ranges,
                "actual_split_count": writer.actual_split_ranges,
                "expected_sha256": (
                    writer._expected_range_digest.hexdigest()
                ),
                "actual_split_sha256": (
                    writer._actual_range_digest.hexdigest()
                ),
            },
            "protected_target_tokens": writer.protected_target_tokens,
        },
    )

    component_shares = {
        component: writer.component_tokens[component] / writer.total
        for component in COMPONENT_SHARES
    }
    mixture_deviation = max(
        abs(component_shares[component] - COMPONENT_SHARES[component])
        for component in COMPONENT_SHARES
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
    mass_denominator = max(split_mass, 1)
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
        "external_payload_coverage": (
            writer.external_payload_tokens == split_mass
            and split_mass > 0
        ),
        "external_payload_ranges_exact": (
            writer.expected_external_ranges == writer.actual_split_ranges
            and range_digests_match
            and writer.expected_length_histogram
            == writer.span_histograms["split"]
            and writer.expected_position_histogram
            == writer.position_histograms["split"]
        ),
        "random_mass_within_1pct": (
            abs(random_mass - split_mass) / mass_denominator <= 0.01
        ),
        "random_span_histogram_within_1pct": (
            _histogram_difference(
                writer.span_histograms["split"],
                writer.span_histograms["random"],
            )
            <= 0.01
        ),
        "random_position_histogram_within_1pct": (
            _histogram_difference(
                writer.position_histograms["split"],
                writer.position_histograms["random"],
            )
            <= 0.01
        ),
        "mixture_within_1pct": mixture_deviation <= 0.01,
        "graph_mixture_exact": graph_mixture_exact,
        "protected_roles_unmasked": writer.protected_roles_unmasked,
        "eval_validity": all(eval_checks.values()),
        "schedule_hash_stable": (
            schedule_digest.hexdigest() == schedule_sha256
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
            "calibration": policy_document["calibration"],
            "protected_route_rate": graph_manifest["route_rate"],
        },
        "tokens": {
            "total": writer.total,
            "components": {
                component: {
                    "tokens": writer.component_tokens[component],
                    "records": writer.component_records[component],
                    "share": component_shares[component],
                    "target_share": COMPONENT_SHARES[component],
                }
                for component in COMPONENT_SHARES
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
            "external_payload_tokens": writer.external_payload_tokens,
            "expected_external_ranges": writer.expected_external_ranges,
            "actual_split_ranges": writer.actual_split_ranges,
            "expected_range_sha256": (
                writer._expected_range_digest.hexdigest()
            ),
            "actual_split_range_sha256": (
                writer._actual_range_digest.hexdigest()
            ),
            "split_masked_tokens": split_mass,
            "random_masked_tokens": random_mass,
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
            "unmatched_random_keys": {
                f"{length}:{relative_bin}": count
                for (length, relative_bin), count in sorted(
                    writer._pending_random.items()
                )
            },
        },
        "eval": eval_report,
        "checks": checks,
    }
    report_path = out_dir / "report.json"
    _write_json(report_path, report)

    artifact_paths = [
        writer.token_path,
        *writer.weight_paths.values(),
        writer.ledger_path,
        graph_path,
        policy_path,
        schedule_path,
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
            "artifacts": artifacts,
        },
    )
    return report
