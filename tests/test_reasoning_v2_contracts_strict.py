from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from corpusgen.reasoning_v2.contracts import load_recipe


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "configs" / "reasoning-dataset-v2.json"


def _recipe_value() -> dict[str, object]:
    value = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_recipe(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _replace_nested(
    value: dict[str, object],
    path: tuple[str, ...],
    replacement: object,
) -> None:
    current: object = value
    for key in path[:-1]:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict)
    current[path[-1]] = replacement


_DEEP_CONTRACT_MUTATIONS = (
    (
        "source-fineweb-lock",
        ("sprint_recipe", "source_policy", "fineweb_source_lock"),
        "configs/other-lock.json",
    ),
    (
        "source-wikidata-lock-type",
        ("sprint_recipe", "source_policy", "wikidata_source_lock"),
        7,
    ),
    (
        "source-finemath-repository",
        ("sprint_recipe", "source_policy", "finemath_repository"),
        "HuggingFaceTB/other",
    ),
    (
        "source-finemath-order",
        ("sprint_recipe", "source_policy", "finemath_subsets_in_order"),
        [
            "finemath-3plus-cross-deduplicated-remainder",
            "finemath-4plus",
        ],
    ),
    (
        "source-cross-dedup-bool-alias",
        (
            "sprint_recipe",
            "source_policy",
            "cross_deduplicate_finemath_against_fineweb",
        ),
        1,
    ),
    (
        "source-objective-duplicate",
        ("sprint_recipe", "source_policy", "objective_auxiliary_sources"),
        [
            "deepmind_mathematics_generator",
            "deepmind_mathematics_generator",
            "ruletaker",
            "prontoqa",
            "reasoning_gym_exact_answer",
            "arc_agi_training",
            "conceptarc_training",
        ],
    ),
    (
        "source-arc-cap-string-alias",
        ("sprint_recipe", "source_policy", "arc_conceptarc_total_max_percent"),
        "0.25",
    ),
    (
        "source-teacher-cap",
        (
            "sprint_recipe",
            "source_policy",
            "teacher_generated_cot_total_max_percent",
        ),
        0.6,
    ),
    (
        "source-teacher-requirements-order",
        ("sprint_recipe", "source_policy", "teacher_generated_cot_requires"),
        [
            "trace_validation",
            "independent_answer_validation",
            "contamination_review",
        ],
    ),
    (
        "source-exclusions-duplicate",
        ("sprint_recipe", "source_policy", "excluded_from_claim_bearing_core"),
        [
            "proof_pile_variants",
            "proof_pile_variants",
            "benchmark_evaluation_answers",
        ],
    ),
    (
        "publication-complete-graph-bool-alias",
        (
            "sprint_recipe",
            "publication_requirements",
            "complete_once_wikidata_training_graph",
        ),
        1,
    ),
    (
        "publication-stable-facts",
        (
            "sprint_recipe",
            "publication_requirements",
            "stable_fact_universe_and_exposure_burden",
        ),
        False,
    ),
    (
        "publication-verification-rate-int-alias",
        (
            "sprint_recipe",
            "publication_requirements",
            "solver_verification_rate",
        ),
        1,
    ),
    (
        "publication-overlap-int-alias",
        (
            "sprint_recipe",
            "publication_requirements",
            "structural_train_evaluation_overlap_max",
        ),
        0,
    ),
    (
        "publication-rebuild-string-alias",
        (
            "sprint_recipe",
            "publication_requirements",
            "deterministic_rebuild_required",
        ),
        "true",
    ),
    (
        "publication-cycle-fill",
        (
            "sprint_recipe",
            "publication_requirements",
            "reasoning_lane_cycle_fill_forbidden",
        ),
        False,
    ),
    (
        "intervention-primary-arm-order",
        ("intervention", "primary_arms"),
        ["split90", "dense"],
    ),
    (
        "intervention-published-view-duplicate",
        ("intervention", "published_views"),
        [
            "dense",
            "split50",
            "split90",
            "random_fact_90",
            "random_fact_90",
        ],
    ),
    (
        "intervention-distinct-dose-int-alias",
        (
            "intervention",
            "split90_minimum_percent",
            "distinct_offloadable_atomic_facts",
        ),
        90,
    ),
    (
        "intervention-burden-dose",
        (
            "intervention",
            "split90_minimum_percent",
            "train_only_information_weighted_burden",
        ),
        89.0,
    ),
    (
        "intervention-internal-duplicate",
        ("intervention", "always_internal"),
        ["rules", "operators", "schemas", "schemas"],
    ),
    (
        "intervention-invariant-order",
        ("intervention", "matched_pair_invariants"),
        [
            "parameters",
            "model_architecture",
            "initialization",
            "data_seed",
            "raw_target_tokens",
            "token_bytes",
            "token_order",
            "packing_boundaries",
            "optimizer",
            "optimizer_schedule",
            "targets_per_update",
            "training_steps",
            "inference_budget",
            "evaluation_items",
            "exact_graph_memory",
        ],
    ),
    (
        "intervention-claim-difference",
        ("intervention", "only_claim_bearing_difference"),
        "model_architecture",
    ),
    (
        "intervention-closure-bool-alias",
        ("intervention", "semantic_mask_closure_required"),
        1,
    ),
    (
        "intervention-answer-surfaces",
        ("intervention", "candidate_and_final_state_fact_surfaces_forbidden"),
        False,
    ),
    (
        "intervention-memory-kind",
        ("intervention", "memory", "kind"),
        "approximate_graph",
    ),
    (
        "intervention-memory-identical-bool-alias",
        ("intervention", "memory", "identical_between_arms"),
        1,
    ),
    (
        "intervention-memory-byte-identity",
        ("intervention", "memory", "byte_identity_required"),
        False,
    ),
    (
        "intervention-memory-return-order",
        ("intervention", "memory", "may_return"),
        ["MISS", "exact_row"],
    ),
    (
        "intervention-memory-prohibition-duplicate",
        ("intervention", "memory", "may_not"),
        ["rank_candidates", "rank_candidates", "generate_proofs", "generate_answers"],
    ),
)


@pytest.mark.parametrize(
    ("_case", "path", "replacement"),
    _DEEP_CONTRACT_MUTATIONS,
    ids=[case[0] for case in _DEEP_CONTRACT_MUTATIONS],
)
def test_recipe_rejects_deep_policy_semantic_drift(
    tmp_path,
    _case,
    path,
    replacement,
):
    value = _recipe_value()
    _replace_nested(value, path, replacement)

    with pytest.raises(ValueError):
        load_recipe(_write_recipe(tmp_path, value))


def test_validated_recipe_policy_data_is_transitively_immutable():
    recipe = load_recipe(RECIPE_PATH)

    assert isinstance(recipe.source_policy, Mapping)
    assert recipe.source_policy["finemath_subsets_in_order"] == (
        "finemath-4plus",
        "finemath-3plus-cross-deduplicated-remainder",
    )
    assert recipe.intervention["primary_arms"] == ("dense", "split90")
    memory = recipe.intervention["memory"]
    assert isinstance(memory, Mapping)
    assert memory["may_return"] == ("exact_row", "MISS")

    with pytest.raises(TypeError):
        recipe.source_policy["fineweb_source_lock"] = "changed"
    subsets = recipe.source_policy["finemath_subsets_in_order"]
    with pytest.raises(TypeError):
        subsets[0] = "changed"
    with pytest.raises(TypeError):
        memory["kind"] = "changed"
    with pytest.raises(FrozenInstanceError):
        recipe.source_policy = {}

    assert recipe.source_policy["fineweb_source_lock"] == (
        "configs/current-dataset-lock.json"
    )
    assert memory["kind"] == "exact_non_trainable_graph"
