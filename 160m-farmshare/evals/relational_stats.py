"""Frozen hierarchical paired inference for the protected relational study.

The inferential unit is an initialization/data seed.  Within each selected
seed, worlds, relation-path hashes, templates, and complete
original/counterfactual pairs are resampled in that order.  A single draw is
shared by every arm and every contrast.

Percentile intervals use the nearest-rank convention.  For ``n`` sorted
replicates, the zero-based index for percentile ``p`` is
``max(0, ceil(p * n) - 1)``.  Consequently, the frozen 10,000-replicate 95%
interval uses indices 249 and 9749.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Any

import numpy as np

from evals.relational_contracts import (
    CheckpointSummary,
    EvalRow,
    GuardrailReport,
    canonical_json_bytes,
    validate_eval_rows,
)
from evals.relational_metrics import EXPECTED_TASKS


CONFIRMATORY_SEEDS = (1001, 1002, 1003, 1004, 1005)
SEEDS = CONFIRMATORY_SEEDS
ALLOWED_VERDICTS = (
    "validated",
    "practical_null",
    "inconclusive",
    "invalid",
)
BOOTSTRAP_VERSION = "hierarchical-paired-v1"
PERCENTILE_CONVENTION = "nearest-rank-zero-based-ceil(p*n)-1"
FROZEN_N_BOOT = 10_000
FROZEN_PERCENTILE_INDICES = (249, 9749)
MAX_BOOTSTRAP_CHUNK = 100
MAX_TRACE_REPLICATES = 100
_PRIMARY_ARMS = frozenset({"split", "dense", "random"})
_PAIR_VARIANTS = frozenset({"original", "counterfactual"})


PairKey = tuple[int, str, str, int, str, str]
JoinKey = tuple[Any, ...]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_bootstrap_config(
    n_boot: object,
    rng_seed: object,
    chunk_size: object,
) -> tuple[int, int, int]:
    n_boot_value = _positive_integer(n_boot, "n boot")
    rng_seed_value = _nonnegative_integer(rng_seed, "rng seed")
    chunk_value = _positive_integer(chunk_size, "chunk size")
    if chunk_value > MAX_BOOTSTRAP_CHUNK:
        raise ValueError(
            f"chunk size must be at most {MAX_BOOTSTRAP_CHUNK}"
        )
    if rng_seed_value > (1 << 128) - 1:
        raise ValueError("rng seed exceeds the supported 128-bit range")
    return n_boot_value, rng_seed_value, chunk_value


def _validate_seed_sequence(
    seeds: Sequence[int],
    *,
    confirmatory: bool,
) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError("seeds must be an ordered sequence")
    if confirmatory and tuple(seeds) != CONFIRMATORY_SEEDS:
        raise ValueError(
            "confirmatory seeds must be exactly 1001 through 1005"
        )
    result = tuple(
        _nonnegative_integer(seed, "seed")
        for seed in seeds
    )
    if len(result) != len(set(result)):
        raise ValueError("seeds must be unique")
    if not result:
        raise ValueError("seeds must not be empty")
    return result


def percentile_indices(n_values: int) -> tuple[int, int]:
    count = _positive_integer(n_values, "percentile sample count")

    def index(percentile: float) -> int:
        return min(count - 1, max(0, math.ceil(percentile * count) - 1))

    return index(0.025), index(0.975)


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("percentile values must be a sequence")
    materialized = [
        _finite_number(value, "percentile values") for value in values
    ]
    if not materialized:
        raise ValueError("percentile values must not be empty")
    materialized.sort()
    low, high = percentile_indices(len(materialized))
    return materialized[low], materialized[high]


@dataclass(frozen=True)
class ContrastEstimate:
    mean: float
    ci_lo: float
    ci_hi: float
    seed_deltas: tuple[float, ...]
    cohen_dz: float | None
    effect_note: str

    def __post_init__(self) -> None:
        mean = _finite_number(self.mean, "contrast mean")
        low = _finite_number(self.ci_lo, "contrast interval lower bound")
        high = _finite_number(self.ci_hi, "contrast interval upper bound")
        if low > high:
            raise ValueError("contrast interval lower bound exceeds upper bound")
        if not low <= mean <= high:
            raise ValueError("contrast interval must contain its point mean")
        deltas = tuple(
            _finite_number(value, "seed deltas")
            for value in self.seed_deltas
        )
        if len(deltas) != len(CONFIRMATORY_SEEDS):
            raise ValueError("contrast estimates require five seed deltas")
        point = fmean(deltas)
        if not math.isclose(point, mean, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                "contrast mean must equal the paired five-seed estimator"
            )
        if self.cohen_dz is not None:
            effect = _finite_number(self.cohen_dz, "paired Cohen dz")
            object.__setattr__(self, "cohen_dz", effect)
        if not isinstance(self.effect_note, str) or not self.effect_note:
            raise ValueError("effect note must be a non-empty string")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "ci_lo", low)
        object.__setattr__(self, "ci_hi", high)
        object.__setattr__(self, "seed_deltas", deltas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "seed_deltas": list(self.seed_deltas),
            "cohen_dz": self.cohen_dz,
            "effect_note": self.effect_note,
        }


@dataclass(frozen=True)
class BootstrapDraw:
    pair_multiplicities: Mapping[PairKey, int]
    cluster_multiplicities: Mapping[tuple[Any, ...], int]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pair_multiplicities",
            MappingProxyType(dict(self.pair_multiplicities)),
        )
        object.__setattr__(
            self,
            "cluster_multiplicities",
            MappingProxyType(dict(self.cluster_multiplicities)),
        )
        object.__setattr__(self, "labels", tuple(self.labels))

    def multiplicity(
        self,
        label: str,
        variant: str,
    ) -> tuple[tuple[PairKey, int], ...]:
        if label not in self.labels:
            raise ValueError(f"unknown trace arm label: {label!r}")
        if variant not in _PAIR_VARIANTS:
            raise ValueError(f"unknown trace variant: {variant!r}")
        return tuple(sorted(self.pair_multiplicities.items()))


@dataclass(frozen=True)
class _PairPanel:
    labels: tuple[str, ...]
    label_arms: Mapping[str, str]
    seeds: tuple[int, ...]
    outcomes: Mapping[str, Mapping[PairKey, float]]
    hierarchy: Mapping[
        int,
        Mapping[
            str,
            Mapping[
                int,
                Mapping[str, Mapping[str, tuple[PairKey, ...]]],
            ],
        ],
    ]


_CELL_IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "model_id",
    "arm",
    "raw_token_count",
    "memory_mode",
    "control_id",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
    "control_seed",
)
_GLOBAL_IDENTITY_FIELDS = (
    "model_id",
    "arm",
    "raw_token_count",
    "memory_mode",
    "control_id",
    "evaluator_sha256",
    "relation_schema_sha256",
    "result_schema_sha256",
    "control_seed",
)
_CROSS_GROUP_ROW_FIELDS = (
    "provenance_id",
    "composition_split",
    "hop",
    "cluster_id",
    "model_id",
    "memory_mode",
    "control_id",
    "evaluator_sha256",
    "relation_schema_sha256",
    "result_schema_sha256",
    "control_seed",
)
_GROUP_IDENTITY_FIELDS = (
    "raw_token_count",
    "data_sha256",
)
_PAIRED_JOIN_FIELD_NAMES = (
    "seed",
    "qid",
    "pair_id",
    "variant",
    "task",
    "world_id",
    "relation_path_hash",
    "template_id",
)


def _pair_key(row: EvalRow) -> PairKey:
    return (
        row.seed,
        row.task,
        row.pair_id,
        row.world_id,
        row.relation_path_hash,
        row.template_id,
    )


def _typed_rows(
    rows: Sequence[EvalRow],
    *,
    label: str,
    seeds: tuple[int, ...],
) -> tuple[EvalRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{label} rows must be a sequence of EvalRow values")
    if not rows:
        raise ValueError(f"{label} rows must not be empty")
    if any(not isinstance(row, EvalRow) for row in rows):
        raise TypeError(
            f"{label} rows must contain semantically validated EvalRow values"
        )
    materialized = tuple(
        EvalRow.from_dict(row.to_dict()) for row in rows
    )
    actual_seeds = {row.seed for row in materialized}
    if actual_seeds != set(seeds):
        raise ValueError(
            f"{label} row seeds do not match the requested seed set"
        )
    arms = {row.arm for row in materialized}
    if len(arms) != 1 or not arms <= _PRIMARY_ARMS:
        raise ValueError(
            f"{label} rows require one approved split/dense/random arm"
        )
    if any(
        row.task not in EXPECTED_TASKS
        or row.variant not in _PAIR_VARIANTS
        for row in materialized
    ):
        raise ValueError(
            f"{label} rows contain an unknown task or variant"
        )
    if any(
        row.memory_mode != "memory_on" or row.control_id != "correct"
        for row in materialized
    ):
        raise ValueError(
            f"{label} primary rows must use memory_on/correct"
        )
    for seed in seeds:
        seed_rows = tuple(row for row in materialized if row.seed == seed)
        if {row.task for row in seed_rows} != set(EXPECTED_TASKS):
            raise ValueError(
                f"{label} seed {seed} task set is not the frozen three tasks"
            )
        for task in EXPECTED_TASKS:
            if not any(row.task == task for row in seed_rows):
                raise ValueError(f"{label} seed {seed} has an empty task")
        for field in _CELL_IDENTITY_FIELDS:
            if len({getattr(row, field) for row in seed_rows}) != 1:
                raise ValueError(
                    f"{label} seed {seed} crossed {field} identity"
                )
    for field in _GLOBAL_IDENTITY_FIELDS:
        if len({getattr(row, field) for row in materialized}) != 1:
            raise ValueError(f"{label} crossed global {field} identity")
    validate_eval_rows(materialized)
    return materialized


def validate_analysis_cell(
    rows: Sequence[EvalRow],
    summary: CheckpointSummary,
    *,
    expected_arm: str,
    expected_seed: int,
) -> tuple[EvalRow, ...]:
    """Revalidate a strict Task-7 summary/row cell before analysis."""

    if not isinstance(summary, CheckpointSummary):
        raise TypeError("analysis cell requires a validated CheckpointSummary")
    validated_summary = CheckpointSummary.from_dict(summary.to_dict())
    seed = _nonnegative_integer(expected_seed, "expected seed")
    if expected_arm not in _PRIMARY_ARMS:
        raise ValueError("analysis cell expected arm is not confirmatory")
    if (
        validated_summary.arm != expected_arm
        or validated_summary.seed != seed
        or validated_summary.memory_mode != "memory_on"
        or validated_summary.control_id != "correct"
    ):
        raise ValueError("analysis cell summary identity mismatch")
    if any(not isinstance(row, EvalRow) for row in rows):
        raise TypeError("analysis cell rows must be validated EvalRow values")
    return validated_summary.validate_rows(rows)


def _validated_identity_groups(
    raw: Mapping[str, Sequence[str]] | None,
    labels: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {"all_labels": labels}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("identity comparison groups must be a non-empty mapping")
    groups: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for group_name in sorted(raw):
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(
                "identity comparison group names must be non-empty strings"
            )
        members_raw = raw[group_name]
        if isinstance(members_raw, (str, bytes)) or not isinstance(
            members_raw, Sequence
        ):
            raise ValueError(
                f"identity comparison group {group_name!r} "
                "must contain an ordered label sequence"
            )
        members = tuple(members_raw)
        if len(members) < 2:
            raise ValueError(
                f"identity comparison group {group_name!r} "
                "requires at least two labels"
            )
        if any(not isinstance(label, str) or not label for label in members):
            raise ValueError(
                f"identity comparison group {group_name!r} "
                "contains an invalid label"
            )
        if len(members) != len(set(members)):
            raise ValueError(
                f"identity comparison group {group_name!r} "
                "contains duplicate labels"
            )
        unknown = sorted(set(members) - set(labels))
        if unknown:
            raise ValueError(
                f"identity comparison group {group_name!r} "
                f"contains unknown labels: {unknown}"
            )
        for label in members:
            if label in owners:
                raise ValueError(
                    f"identity label {label!r} belongs to comparison groups "
                    f"{owners[label]!r} and {group_name!r}"
                )
            owners[label] = group_name
        groups[group_name] = tuple(sorted(members))
    missing = sorted(set(labels) - set(owners))
    if missing:
        raise ValueError(
            "identity comparison groups do not cover labels: "
            f"{missing}"
        )
    return groups


def _build_panel(
    rows_by_label: Mapping[str, Sequence[EvalRow]],
    *,
    seeds: tuple[int, ...],
    identity_groups: Mapping[str, Sequence[str]] | None = None,
) -> _PairPanel:
    if not isinstance(rows_by_label, Mapping) or len(rows_by_label) < 2:
        raise ValueError("paired analysis requires at least two arm labels")
    if any(not isinstance(label, str) or not label for label in rows_by_label):
        raise ValueError("arm labels must be non-empty strings")
    labels = tuple(sorted(rows_by_label))
    groups = _validated_identity_groups(identity_groups, labels)
    materialized = {
        label: _typed_rows(rows_by_label[label], label=label, seeds=seeds)
        for label in labels
    }
    label_arms = {
        label: materialized[label][0].arm for label in labels
    }
    indexed: dict[str, dict[JoinKey, EvalRow]] = {}
    for label in labels:
        index: dict[JoinKey, EvalRow] = {}
        for row in materialized[label]:
            key = row.paired_join_key()
            if key in index:
                raise ValueError(f"{label} has a duplicate paired join key")
            index[key] = row
        indexed[label] = index
    reference_label = labels[0]
    reference_keys = set(indexed[reference_label])
    for label in labels[1:]:
        actual = set(indexed[label])
        if actual != reference_keys:
            raise ValueError(
                "paired identity fields "
                f"{','.join(_PAIRED_JOIN_FIELD_NAMES)} mismatch in "
                "comparison group 'shared_pair_hierarchy'; "
                f"{label} missing={len(reference_keys - actual)} "
                f"extra={len(actual - reference_keys)}"
            )
        for key in sorted(reference_keys):
            reference = indexed[reference_label][key]
            current = indexed[label][key]
            for field in _CROSS_GROUP_ROW_FIELDS:
                if getattr(reference, field) != getattr(current, field):
                    raise ValueError(
                        f"identity field {field} mismatch in comparison "
                        "group 'shared_pair_hierarchy' "
                        f"between labels {reference_label!r} and "
                        f"{label!r} at {key!r}"
                    )
    for group_name, group_labels in groups.items():
        group_reference_label = group_labels[0]
        for label in group_labels[1:]:
            for key in sorted(reference_keys):
                reference = indexed[group_reference_label][key]
                current = indexed[label][key]
                for field in _GROUP_IDENTITY_FIELDS:
                    if getattr(reference, field) != getattr(current, field):
                        raise ValueError(
                            f"identity field {field} mismatch in comparison "
                            f"group {group_name!r} between labels "
                            f"{group_reference_label!r} and {label!r} "
                            f"at {key!r}"
                        )

    grouped: dict[str, dict[PairKey, dict[str, EvalRow]]] = {
        label: defaultdict(dict) for label in labels
    }
    for label in labels:
        for row in materialized[label]:
            pair = grouped[label][_pair_key(row)]
            if row.variant in pair:
                raise ValueError("duplicate variant in paired panel")
            pair[row.variant] = row
    reference_pairs = set(grouped[reference_label])
    outcomes: dict[str, dict[PairKey, float]] = {}
    for label in labels:
        if set(grouped[label]) != reference_pairs:
            raise ValueError("arm pair sets do not match exactly")
        outcomes[label] = {}
        for key, pair in grouped[label].items():
            if set(pair) != _PAIR_VARIANTS:
                raise ValueError("every panel pair requires both variants")
            outcomes[label][key] = float(
                pair["original"].correct
                and pair["counterfactual"].correct
            )

    nested: dict[
        int,
        dict[str, dict[int, dict[str, dict[str, list[PairKey]]]]],
    ] = {
        seed: {
            task: defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
            for task in EXPECTED_TASKS
        }
        for seed in seeds
    }
    for key in sorted(reference_pairs):
        seed, task, _pair_id, world, path_hash, template = key
        nested[seed][task][world][path_hash][template].append(key)

    hierarchy: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        hierarchy[seed] = {}
        for task in EXPECTED_TASKS:
            worlds = nested[seed][task]
            if not worlds:
                raise ValueError(f"seed {seed} task {task} has no pairs")
            hierarchy[seed][task] = {
                world: {
                    path_hash: {
                        template: tuple(sorted(pair_keys))
                        for template, pair_keys in sorted(templates.items())
                    }
                    for path_hash, templates in sorted(paths.items())
                }
                for world, paths in sorted(worlds.items())
            }
    return _PairPanel(
        labels=labels,
        label_arms=MappingProxyType(label_arms),
        seeds=seeds,
        outcomes=MappingProxyType(
            {
                label: MappingProxyType(values)
                for label, values in outcomes.items()
            }
        ),
        hierarchy=MappingProxyType(hierarchy),
    )


def _validated_contrasts(
    contrasts: Mapping[str, Mapping[str, float]],
    labels: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise ValueError("at least one paired contrast is required")
    result: dict[str, dict[str, float]] = {}
    used: set[str] = set()
    for name in sorted(contrasts):
        if not isinstance(name, str) or not name:
            raise ValueError("contrast names must be non-empty strings")
        raw = contrasts[name]
        if not isinstance(raw, Mapping) or len(raw) < 2:
            raise ValueError(f"contrast {name!r} requires at least two arms")
        if not set(raw) <= set(labels):
            raise ValueError(f"contrast {name!r} names an unknown arm label")
        weights = {
            label: _finite_number(weight, f"contrast {name} weight")
            for label, weight in raw.items()
        }
        if any(weight == 0.0 for weight in weights.values()):
            raise ValueError(f"contrast {name!r} contains a zero weight")
        if not math.isclose(
            sum(weights.values()), 0.0, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"contrast {name!r} weights must sum to zero")
        used.update(weights)
        result[name] = weights
    if used != set(labels):
        raise ValueError("every supplied arm label must enter a contrast")
    return result


def _task_value(
    panel: _PairPanel,
    pair_counts: Mapping[PairKey, int],
    weights: Mapping[str, float],
) -> float:
    total = sum(pair_counts.values())
    if total <= 0:
        raise AssertionError("hierarchical task draw selected no pairs")
    value = 0.0
    for label, weight in weights.items():
        successes = sum(
            panel.outcomes[label][key] * count
            for key, count in pair_counts.items()
        )
        value += weight * successes / total
    return value


def _point_seed_deltas(
    panel: _PairPanel,
    weights: Mapping[str, float],
) -> tuple[float, ...]:
    deltas = []
    for seed in panel.seeds:
        task_values = []
        for task in EXPECTED_TASKS:
            counts: Counter[PairKey] = Counter()
            for paths in panel.hierarchy[seed][task].values():
                for templates in paths.values():
                    for pairs in templates.values():
                        counts.update(pairs)
            task_values.append(_task_value(panel, counts, weights))
        deltas.append(fmean(task_values))
    return tuple(deltas)


def paired_task_means(
    split_rows: Sequence[EvalRow],
    dense_rows: Sequence[EvalRow],
    *,
    seeds: Sequence[int] = CONFIRMATORY_SEEDS,
) -> dict[str, float]:
    seed_values = _validate_seed_sequence(seeds, confirmatory=True)
    panel = _build_panel(
        {"split": split_rows, "dense": dense_rows},
        seeds=seed_values,
    )
    if panel.label_arms != {"dense": "dense", "split": "split"}:
        raise ValueError("paired task means require Split and Dense rows")
    output = {}
    for task in EXPECTED_TASKS:
        task_deltas = []
        for seed in seed_values:
            counts: Counter[PairKey] = Counter()
            for paths in panel.hierarchy[seed][task].values():
                for templates in paths.values():
                    for pairs in templates.values():
                        counts.update(pairs)
            task_deltas.append(
                _task_value(panel, counts, {"split": 1.0, "dense": -1.0})
            )
        output[task] = fmean(task_deltas)
    return output


def _sample_values(
    values: tuple[Any, ...],
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    if not values:
        raise AssertionError("cannot resample an empty cluster")
    indices = rng.integers(0, len(values), size=len(values))
    return tuple(values[int(index)] for index in indices)


def _draw_one(
    panel: _PairPanel,
    contrasts: Mapping[str, Mapping[str, float]],
    rng: np.random.Generator,
) -> tuple[dict[str, float], BootstrapDraw]:
    selected_seeds = _sample_values(panel.seeds, rng)
    selected_seed_values: dict[str, list[float]] = {
        name: [] for name in contrasts
    }
    all_pairs: Counter[PairKey] = Counter()
    all_clusters: Counter[tuple[Any, ...]] = Counter()
    for top_position, seed in enumerate(selected_seeds):
        task_values: dict[str, list[float]] = {
            name: [] for name in contrasts
        }
        for task in EXPECTED_TASKS:
            worlds_map = panel.hierarchy[seed][task]
            worlds = tuple(sorted(worlds_map))
            pair_counts: Counter[PairKey] = Counter()
            for world in _sample_values(worlds, rng):
                all_clusters[
                    (top_position, seed, task, "world", world)
                ] += 1
                paths_map = worlds_map[world]
                paths = tuple(sorted(paths_map))
                for path_hash in _sample_values(paths, rng):
                    all_clusters[
                        (
                            top_position,
                            seed,
                            task,
                            "path",
                            world,
                            path_hash,
                        )
                    ] += 1
                    templates_map = paths_map[path_hash]
                    templates = tuple(sorted(templates_map))
                    for template in _sample_values(templates, rng):
                        all_clusters[
                            (
                                top_position,
                                seed,
                                task,
                                "template",
                                world,
                                path_hash,
                                template,
                            )
                        ] += 1
                        pairs = templates_map[template]
                        pair_counts.update(_sample_values(pairs, rng))
            all_pairs.update(pair_counts)
            for name, weights in contrasts.items():
                task_values[name].append(
                    _task_value(panel, pair_counts, weights)
                )
        for name in contrasts:
            selected_seed_values[name].append(fmean(task_values[name]))
    values = {
        name: fmean(seed_values)
        for name, seed_values in selected_seed_values.items()
    }
    return values, BootstrapDraw(
        pair_multiplicities=all_pairs,
        cluster_multiplicities=all_clusters,
        labels=panel.labels,
    )


def _paired_dz(seed_deltas: tuple[float, ...]) -> tuple[float | None, str]:
    mean = fmean(seed_deltas)
    variance = sum((value - mean) ** 2 for value in seed_deltas) / (
        len(seed_deltas) - 1
    )
    standard_deviation = math.sqrt(variance)
    if standard_deviation == 0.0:
        return None, "zero_seed_sd"
    return (
        mean / standard_deviation,
        "paired Cohen's dz over five initialization/data seed deltas",
    )


def hierarchical_paired_contrasts(
    rows_by_label: Mapping[str, Sequence[EvalRow]],
    contrasts: Mapping[str, Mapping[str, float]],
    *,
    seeds: Sequence[int],
    n_boot: int,
    rng_seed: int,
    chunk_size: int = MAX_BOOTSTRAP_CHUNK,
    identity_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, ContrastEstimate]:
    """Estimate paired contrasts with optional load-bound identity groups."""

    seed_values = _validate_seed_sequence(seeds, confirmatory=True)
    n_replicates, random_seed, chunk = _validate_bootstrap_config(
        n_boot, rng_seed, chunk_size
    )
    panel = _build_panel(
        rows_by_label,
        seeds=seed_values,
        identity_groups=identity_groups,
    )
    weights = _validated_contrasts(contrasts, panel.labels)
    point_deltas = {
        name: _point_seed_deltas(panel, contrast)
        for name, contrast in weights.items()
    }
    point_means = {
        name: fmean(values) for name, values in point_deltas.items()
    }
    samples: dict[str, list[float]] = {
        name: [] for name in weights
    }
    rng = np.random.Generator(np.random.PCG64(random_seed))
    generated = 0
    while generated < n_replicates:
        block = min(chunk, n_replicates - generated)
        for _ in range(block):
            values, _trace = _draw_one(panel, weights, rng)
            for name, value in values.items():
                samples[name].append(value)
        generated += block
    estimates = {}
    for name in sorted(weights):
        low, high = percentile_interval(samples[name])
        mean = point_means[name]
        effect, note = _paired_dz(point_deltas[name])
        estimates[name] = ContrastEstimate(
            mean=mean,
            ci_lo=low,
            ci_hi=high,
            seed_deltas=point_deltas[name],
            cohen_dz=effect,
            effect_note=note,
        )
    return estimates


def hierarchical_paired_bootstrap(
    split_rows: Sequence[EvalRow],
    dense_rows: Sequence[EvalRow],
    *,
    seeds: Sequence[int],
    n_boot: int,
    rng_seed: int,
    chunk_size: int = MAX_BOOTSTRAP_CHUNK,
) -> ContrastEstimate:
    estimate = hierarchical_paired_contrasts(
        {"split": split_rows, "dense": dense_rows},
        {"split_dense": {"split": 1.0, "dense": -1.0}},
        seeds=seeds,
        n_boot=n_boot,
        rng_seed=rng_seed,
        chunk_size=chunk_size,
    )["split_dense"]
    panel = _build_panel(
        {"split": split_rows, "dense": dense_rows},
        seeds=tuple(seeds),
    )
    if panel.label_arms != {"dense": "dense", "split": "split"}:
        raise ValueError(
            "hierarchical_paired_bootstrap requires Split and Dense arms"
        )
    return estimate


def hierarchical_paired_bootstrap_reference(
    split_rows: Sequence[EvalRow],
    dense_rows: Sequence[EvalRow],
    *,
    seeds: Sequence[int],
    n_boot: int,
    rng_seed: int,
) -> ContrastEstimate:
    """Simple one-replicate-at-a-time reference for equivalence tests."""

    return hierarchical_paired_bootstrap(
        split_rows,
        dense_rows,
        seeds=seeds,
        n_boot=n_boot,
        rng_seed=rng_seed,
        chunk_size=1,
    )


def bootstrap_trace(
    rows_by_label: Mapping[str, Sequence[EvalRow]],
    *,
    seeds: Sequence[int] = CONFIRMATORY_SEEDS,
    n_boot: int,
    rng_seed: int,
) -> tuple[BootstrapDraw, ...]:
    """Return bounded shared-draw traces for tests and debugging only."""

    seed_values = _validate_seed_sequence(seeds, confirmatory=True)
    n_replicates, random_seed, _ = _validate_bootstrap_config(
        n_boot, rng_seed, min(MAX_BOOTSTRAP_CHUNK, max(1, int(n_boot)))
    )
    if n_replicates > MAX_TRACE_REPLICATES:
        raise ValueError(
            f"bootstrap trace is limited to {MAX_TRACE_REPLICATES} replicates"
        )
    panel = _build_panel(rows_by_label, seeds=seed_values)
    reference = panel.labels[0]
    other = panel.labels[1]
    contrasts = {
        "trace": {reference: 1.0, other: -1.0}
    }
    rng = np.random.Generator(np.random.PCG64(random_seed))
    return tuple(
        _draw_one(panel, contrasts, rng)[1]
        for _ in range(n_replicates)
    )


@dataclass(frozen=True)
class VerdictInputs:
    split_dense_360: ContrastEstimate
    split_dense_160_high: ContrastEstimate
    dose_interaction_160: ContrastEstimate
    split_random_160_high: ContrastEstimate
    task_means_360: Mapping[str, float]
    task_means_160_high: Mapping[str, float]
    guardrail_reports: tuple[GuardrailReport, ...]

    def __post_init__(self) -> None:
        for name in (
            "split_dense_360",
            "split_dense_160_high",
            "dose_interaction_160",
            "split_random_160_high",
        ):
            if not isinstance(getattr(self, name), ContrastEstimate):
                raise TypeError(f"{name} must be a ContrastEstimate")
        for name in ("task_means_360", "task_means_160_high"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} task means must be a mapping")
            frozen = MappingProxyType(
                {
                    task: _finite_number(
                        values[task], f"{name} task mean {task}"
                    )
                    for task in EXPECTED_TASKS
                    if task in values
                }
            )
            object.__setattr__(self, name, frozen)
        reports = tuple(self.guardrail_reports)
        object.__setattr__(self, "guardrail_reports", reports)


def _validated_task_means(
    values: Mapping[str, float],
    name: str,
) -> tuple[float, ...]:
    if not isinstance(values, Mapping) or set(values) != set(EXPECTED_TASKS):
        raise ValueError(f"{name} task means must match the three frozen tasks")
    return tuple(
        _finite_number(values[task], f"{name} task mean {task}")
        for task in EXPECTED_TASKS
    )


def _validated_guardrails(
    reports: Sequence[GuardrailReport],
) -> tuple[GuardrailReport, ...]:
    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise ValueError("guardrail reports must be a sequence")
    if not reports:
        raise ValueError("at least one strict GuardrailReport is required")
    if any(not isinstance(report, GuardrailReport) for report in reports):
        raise TypeError(
            "guardrail reports must contain strict GuardrailReport artifacts"
        )
    validated = tuple(
        GuardrailReport.from_dict(report.to_dict()) for report in reports
    )
    identities = [
        (
            report.model_id,
            report.seed,
            report.raw_token_count,
            report.split_checkpoint_sha256,
            report.dense_checkpoint_sha256,
        )
        for report in validated
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate strict GuardrailReport identity")
    return validated


def allowed_verdicts() -> tuple[str, ...]:
    return ALLOWED_VERDICTS


def verdict_inputs_to_dict(value: VerdictInputs) -> dict[str, Any]:
    reports = _validated_guardrails(value.guardrail_reports)
    task_360 = _validated_task_means(
        value.task_means_360, "360 confirmation"
    )
    task_160 = _validated_task_means(
        value.task_means_160_high, "160 high"
    )
    return {
        "split_dense_360": value.split_dense_360.to_dict(),
        "split_dense_160_high": value.split_dense_160_high.to_dict(),
        "dose_interaction_160": value.dose_interaction_160.to_dict(),
        "split_random_160_high": value.split_random_160_high.to_dict(),
        "task_means_360": {
            task: task_360[index]
            for index, task in enumerate(EXPECTED_TASKS)
        },
        "task_means_160_high": {
            task: task_160[index]
            for index, task in enumerate(EXPECTED_TASKS)
        },
        "guardrail_report_sha256": [
            hashlib.sha256(canonical_json_bytes(report)).hexdigest()
            for report in reports
        ],
        "all_guardrails_passed": all(
            report.confirmatory_passed for report in reports
        ),
    }


def decide_verdict(value: VerdictInputs) -> str:
    if not isinstance(value, VerdictInputs):
        raise TypeError("verdict requires VerdictInputs")
    task_360 = _validated_task_means(
        value.task_means_360, "360 confirmation"
    )
    task_160 = _validated_task_means(
        value.task_means_160_high, "160 high"
    )
    reports = _validated_guardrails(value.guardrail_reports)
    if not all(report.confirmatory_passed for report in reports):
        return "invalid"

    validates = (
        value.split_dense_360.mean >= 0.02
        and value.split_dense_360.ci_lo > 0.0
        and value.split_dense_160_high.ci_lo > 0.0
        and value.dose_interaction_160.ci_lo > 0.0
        and value.split_random_160_high.ci_lo > 0.0
        and all(delta > 0.0 for delta in task_360)
        and all(delta > 0.0 for delta in task_160)
    )
    if validates:
        return "validated"

    practical_null = (
        value.split_dense_160_high.ci_hi < 0.02
        and value.split_dense_360.ci_hi < 0.02
        and value.dose_interaction_160.ci_hi <= 0.0
        and value.split_random_160_high.ci_hi <= 0.01
    )
    return "practical_null" if practical_null else "inconclusive"
