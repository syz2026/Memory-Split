from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, NoReturn, cast


LaneId = Literal[
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
]

LANE_ORDER: tuple[LaneId, ...] = (
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
)

_DATASET_ID = "memorysplit-v2-20x-reasoning-max-cohort"
_CONTRACT_ID = "memorysplit-reasoning-dataset-v2"
_CONTEXT_LENGTH = 1_024
_FULL_SHARD_COUNT = 32
_FULL_TOTAL_TARGETS = 7_120_879_616
_TARGETS_PER_UPDATE = 524_288
_OPTIMIZER_UPDATES = 13_582
_CANARY_TOTAL_TARGETS = 1_048_576

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "status",
        "canonical_hypothesis",
        "sprint_recipe",
        "intervention",
        "evaluation",
        "artifact_bindings",
    }
)
_SPRINT_RECIPE_FIELDS = frozenset(
    {
        "label",
        "reasoning_optimal_claim",
        "model_parameters",
        "nominal_tokens_per_parameter",
        "targets_per_update",
        "optimizer_steps",
        "raw_target_tokens",
        "mixture_unit",
        "share_semantics",
        "lanes",
        "realized_token_allocation",
        "language_floors_percent",
        "source_policy",
        "publication_requirements",
    }
)
_LANE_FIELDS = frozenset(
    {
        "id",
        "share_percent",
        "human_language",
        "broad_general_language",
        "verification",
        "role",
    }
)
_REALIZED_ALLOCATION_FIELDS = frozenset(
    {
        "method",
        "procedure",
        "tie_break",
        "lane_order",
        "total_tokens",
        "token_quotas",
    }
)
_SOURCE_POLICY_FIELDS = frozenset(
    {
        "fineweb_source_lock",
        "wikidata_source_lock",
        "finemath_repository",
        "finemath_subsets_in_order",
        "cross_deduplicate_finemath_against_fineweb",
        "objective_auxiliary_sources",
        "arc_conceptarc_total_max_percent",
        "teacher_generated_cot_total_max_percent",
        "teacher_generated_cot_requires",
        "excluded_from_claim_bearing_core",
    }
)
_PUBLICATION_REQUIREMENT_FIELDS = frozenset(
    {
        "complete_once_wikidata_training_graph",
        "stable_fact_universe_and_exposure_burden",
        "solver_verification_rate",
        "structural_train_evaluation_overlap_max",
        "deterministic_rebuild_required",
        "reasoning_lane_cycle_fill_forbidden",
    }
)
_INTERVENTION_FIELDS = frozenset(
    {
        "primary_arms",
        "published_views",
        "split90_minimum_percent",
        "always_internal",
        "matched_pair_invariants",
        "only_claim_bearing_difference",
        "semantic_mask_closure_required",
        "candidate_and_final_state_fact_surfaces_forbidden",
        "memory",
    }
)
_SPLIT90_DOSE_FIELDS = frozenset(
    {
        "distinct_offloadable_atomic_facts",
        "train_only_information_weighted_burden",
    }
)


@dataclass(frozen=True)
class LaneContract:
    lane_id: LaneId
    share: Fraction
    quota: int
    verification: str


@dataclass(frozen=True)
class BuildGeometry:
    profile: Literal["canary", "full"]
    total_targets: int
    targets_per_update: int
    context_length: int
    shard_count: int
    allow_fewer_shards: bool
    lane_quotas: tuple[tuple[LaneId, int], ...]


@dataclass(frozen=True)
class ReasoningV2Recipe:
    dataset_id: str
    total_targets: int
    targets_per_update: int
    optimizer_updates: int
    context_length: int
    shard_count: int
    lanes: tuple[LaneContract, ...]
    source_policy: dict[str, object]
    intervention: dict[str, object]

    @property
    def lane_shares(self) -> tuple[tuple[LaneId, Fraction], ...]:
        return tuple((lane.lane_id, lane.share) for lane in self.lanes)

    @property
    def lane_quotas(self) -> tuple[tuple[LaneId, int], ...]:
        return tuple((lane.lane_id, lane.quota) for lane in self.lanes)


def hamilton_quotas(
    total_targets: int,
    shares: tuple[tuple[LaneId, Fraction], ...],
) -> tuple[tuple[LaneId, int], ...]:
    if type(total_targets) is not int or total_targets <= 0:
        raise ValueError("total_targets must be a positive integer")
    if type(shares) is not tuple:
        raise TypeError("shares must be a tuple")

    validated: list[tuple[LaneId, Fraction]] = []
    for item in shares:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("share entries must be (lane_id, Fraction) tuples")
        lane, share = item
        if not isinstance(lane, str) or lane not in LANE_ORDER:
            raise ValueError(f"unknown lane in shares: {lane!r}")
        if not isinstance(share, Fraction):
            raise TypeError("lane shares must be fractions")
        if share < 0:
            raise ValueError("lane shares must be non-negative")
        validated.append((cast(LaneId, lane), share))

    frozen_shares = tuple(validated)
    if tuple(lane for lane, _share in frozen_shares) != LANE_ORDER:
        raise ValueError("shares must use the frozen lane order")
    if sum((share for _lane, share in frozen_shares), Fraction()) != 1:
        raise ValueError("lane shares must sum exactly to one")

    floors = {
        lane: (total_targets * share).numerator
        // (total_targets * share).denominator
        for lane, share in frozen_shares
    }
    remainder = total_targets - sum(floors.values())
    ranked = sorted(
        frozen_shares,
        key=lambda item: (
            -((total_targets * item[1]) - floors[item[0]]),
            LANE_ORDER.index(item[0]),
        ),
    )
    for lane, _share in ranked[:remainder]:
        floors[lane] += 1
    return tuple((lane, floors[lane]) for lane in LANE_ORDER)


def balanced_record_lengths(targets: int, context_length: int) -> tuple[int, ...]:
    if type(targets) is not int or targets <= 0:
        raise ValueError("targets must be a positive integer")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    count = (targets + context_length - 1) // context_length
    base, extra = divmod(targets, count)
    return (base + 1,) * extra + (base,) * (count - extra)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _ensure_finite_numbers(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is not allowed")
    if isinstance(value, list):
        for item in value:
            _ensure_finite_numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_finite_numbers(item)


def _load_json(path: Path) -> object:
    if not isinstance(path, Path):
        raise TypeError("recipe path must be a pathlib.Path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"recipe must be a regular file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"recipe must contain valid UTF-8 JSON: {path}") from exc
    _ensure_finite_numbers(value)
    return value


def _require_object(value: object, description: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_list(value: object, description: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{description} must be a JSON array")
    return cast(list[Any], value)


def _require_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    description: str,
) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"{description} has unknown fields: {unknown}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"{description} has missing fields: {missing}")


def _require_positive_integer(value: object, description: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _require_boolean(value: object, description: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{description} must be boolean")
    return value


def _require_string_list(value: object, description: str) -> tuple[str, ...]:
    items = _require_list(value, description)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{description} must contain non-empty strings")
    return tuple(cast(list[str], items))


def _number_as_fraction(value: object, description: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite JSON number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} must be a finite JSON number")
    return Fraction(str(value))


def _percent_as_share(value: object, description: str) -> Fraction:
    return _number_as_fraction(value, description) / 100


def _require_exact(value: object, expected: object, description: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{description} does not match the frozen contract")


def load_recipe(path: Path) -> ReasoningV2Recipe:
    root = _require_object(_load_json(path), "recipe")
    _require_fields(root, _TOP_LEVEL_FIELDS, "top-level recipe")

    sprint = _require_object(root["sprint_recipe"], "sprint recipe")
    _require_fields(sprint, _SPRINT_RECIPE_FIELDS, "sprint recipe")

    raw_lanes = _require_list(sprint["lanes"], "sprint recipe lanes")
    lane_values: list[dict[str, Any]] = []
    for index, raw_lane in enumerate(raw_lanes):
        lane = _require_object(raw_lane, f"lane {index}")
        _require_fields(lane, _LANE_FIELDS, f"lane {index}")
        lane_values.append(lane)

    allocation = _require_object(
        sprint["realized_token_allocation"],
        "realized allocation",
    )
    _require_fields(
        allocation,
        _REALIZED_ALLOCATION_FIELDS,
        "realized allocation",
    )
    source_policy = _require_object(sprint["source_policy"], "source policy")
    _require_fields(source_policy, _SOURCE_POLICY_FIELDS, "source policy")
    publication_requirements = _require_object(
        sprint["publication_requirements"],
        "publication requirements",
    )
    _require_fields(
        publication_requirements,
        _PUBLICATION_REQUIREMENT_FIELDS,
        "publication requirements",
    )
    intervention = _require_object(root["intervention"], "intervention")
    _require_fields(intervention, _INTERVENTION_FIELDS, "intervention")
    split90_dose = _require_object(
        intervention["split90_minimum_percent"],
        "Split90 dose",
    )
    _require_fields(split90_dose, _SPLIT90_DOSE_FIELDS, "Split90 dose")

    _require_exact(root["schema_version"], 2, "recipe schema version")
    _require_exact(root["contract_id"], _CONTRACT_ID, "recipe contract ID")
    _require_object(root["status"], "recipe status")
    _require_string(root["canonical_hypothesis"], "canonical hypothesis")
    _require_object(root["evaluation"], "recipe evaluation")
    _require_object(root["artifact_bindings"], "artifact bindings")

    _require_exact(
        sprint["label"],
        "20tpp_reasoning_maximized_sprint",
        "sprint label",
    )
    _require_exact(
        sprint["reasoning_optimal_claim"],
        False,
        "reasoning-optimal claim",
    )
    model_parameters = _require_positive_integer(
        sprint["model_parameters"],
        "model parameters",
    )
    nominal_tokens_per_parameter = _require_positive_integer(
        sprint["nominal_tokens_per_parameter"],
        "nominal tokens per parameter",
    )
    targets_per_update = _require_positive_integer(
        sprint["targets_per_update"],
        "targets per update",
    )
    optimizer_updates = _require_positive_integer(
        sprint["optimizer_steps"],
        "optimizer updates",
    )
    total_targets = _require_positive_integer(
        sprint["raw_target_tokens"],
        "raw target tokens",
    )
    if total_targets != targets_per_update * optimizer_updates:
        raise ValueError(
            "raw target total must equal the targets-per-update/update-count product"
        )
    if (
        model_parameters != 356_033_536
        or nominal_tokens_per_parameter != 20
        or targets_per_update != _TARGETS_PER_UPDATE
        or optimizer_updates != _OPTIMIZER_UPDATES
        or total_targets != _FULL_TOTAL_TARGETS
    ):
        raise ValueError("sprint geometry does not match the frozen 360M contract")
    _require_exact(
        sprint["mixture_unit"],
        "raw_causal_target_tokens_before_condition_weights",
        "mixture unit",
    )
    _require_exact(
        sprint["share_semantics"],
        "target_percentages",
        "share semantics",
    )
    _require_object(
        sprint["language_floors_percent"],
        "language floors",
    )

    lane_ids = tuple(lane["id"] for lane in lane_values)
    if lane_ids != LANE_ORDER:
        raise ValueError("recipe lanes must use the frozen lane order")
    lanes_without_quotas: list[tuple[LaneId, Fraction, str]] = []
    for expected_id, lane in zip(LANE_ORDER, lane_values, strict=True):
        share = _percent_as_share(
            lane["share_percent"],
            f"{expected_id} share percent",
        )
        _require_boolean(
            lane["human_language"],
            f"{expected_id} human-language flag",
        )
        _require_boolean(
            lane["broad_general_language"],
            f"{expected_id} broad-language flag",
        )
        verification = _require_string(
            lane["verification"],
            f"{expected_id} verification",
        )
        _require_string(lane["role"], f"{expected_id} role")
        lanes_without_quotas.append((expected_id, share, verification))

    _require_exact(
        allocation["method"],
        "hamilton_largest_remainder",
        "allocation method",
    )
    _require_exact(
        allocation["procedure"],
        "floor_each_ideal_then_award_remaining_tokens_by_descending_fractional_remainder",
        "allocation procedure",
    )
    _require_exact(
        allocation["tie_break"],
        "stable_lane_order",
        "allocation tie break",
    )
    raw_lane_order = _require_string_list(
        allocation["lane_order"],
        "allocation lane order",
    )
    if raw_lane_order != LANE_ORDER:
        raise ValueError("allocation lane order must equal the frozen lane order")
    allocation_total = _require_positive_integer(
        allocation["total_tokens"],
        "allocation total tokens",
    )
    if allocation_total != total_targets:
        raise ValueError("allocation total must equal the raw target total")

    quota_values = _require_object(
        allocation["token_quotas"],
        "committed token quotas",
    )
    quota_keys = tuple(quota_values)
    if quota_keys != LANE_ORDER:
        raise ValueError("committed quota keys must use the frozen lane order")
    committed_quotas = tuple(
        (
            lane,
            _require_positive_integer(
                quota_values[lane],
                f"{lane} committed quota",
            ),
        )
        for lane in LANE_ORDER
    )
    if sum(quota for _lane, quota in committed_quotas) != allocation_total:
        raise ValueError("committed quota total does not equal allocation total")
    shares = tuple(
        (lane_id, share)
        for lane_id, share, _verification in lanes_without_quotas
    )
    recomputed_quotas = hamilton_quotas(total_targets, shares)
    if committed_quotas != recomputed_quotas:
        raise ValueError("committed lane quotas do not match Hamilton quotas")
    for field in _SPLIT90_DOSE_FIELDS:
        _number_as_fraction(
            split90_dose[field],
            f"Split90 {field.replace('_', ' ')}",
        )

    quota_by_lane = dict(committed_quotas)
    lanes = tuple(
        LaneContract(
            lane_id=lane_id,
            share=share,
            quota=quota_by_lane[lane_id],
            verification=verification,
        )
        for lane_id, share, verification in lanes_without_quotas
    )
    return ReasoningV2Recipe(
        dataset_id=_DATASET_ID,
        total_targets=total_targets,
        targets_per_update=targets_per_update,
        optimizer_updates=optimizer_updates,
        context_length=_CONTEXT_LENGTH,
        shard_count=_FULL_SHARD_COUNT,
        lanes=lanes,
        source_policy=cast(dict[str, object], source_policy),
        intervention=cast(dict[str, object], intervention),
    )


def geometry_for(
    recipe: ReasoningV2Recipe,
    profile: Literal["canary", "full"],
) -> BuildGeometry:
    if not isinstance(recipe, ReasoningV2Recipe):
        raise TypeError("recipe must be a ReasoningV2Recipe")
    if profile == "full":
        return BuildGeometry(
            profile="full",
            total_targets=recipe.total_targets,
            targets_per_update=recipe.targets_per_update,
            context_length=recipe.context_length,
            shard_count=recipe.shard_count,
            allow_fewer_shards=False,
            lane_quotas=recipe.lane_quotas,
        )
    if profile == "canary":
        return BuildGeometry(
            profile="canary",
            total_targets=_CANARY_TOTAL_TARGETS,
            targets_per_update=recipe.targets_per_update,
            context_length=recipe.context_length,
            shard_count=recipe.shard_count,
            allow_fewer_shards=True,
            lane_quotas=hamilton_quotas(
                _CANARY_TOTAL_TARGETS,
                recipe.lane_shares,
            ),
        )
    raise ValueError("profile must be 'canary' or 'full'")
