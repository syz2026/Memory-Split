from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "configs" / "reasoning-dataset-v2.json"
PREREG_PATH = ROOT / "configs" / "preregistration-v2.yaml"
AUDIT_PATH = ROOT / "docs" / "MemorySplit-dataset-audit.md"

CANONICAL_HYPOTHESIS = (
    "At matched parameters, initialization, raw tokens, token order, optimizer "
    "schedule and inference budget, removing direct next-token loss from "
    "selected factual payloads improves OOD relational reasoning."
)
EXPECTED_MIXTURE = {
    "fineweb_edu": 25.0,
    "finemath": 15.0,
    "wikidata_graph": 20.0,
    "synthetic_graph": 10.0,
    "verified_synthetic_multihop": 15.0,
    "wikidata_path_reasoning": 7.5,
    "relational_refinement": 2.5,
    "objective_auxiliary": 5.0,
}
EXPECTED_REALIZED_TOKEN_QUOTAS = {
    "fineweb_edu": 1_780_219_904,
    "finemath": 1_068_131_943,
    "wikidata_graph": 1_424_175_923,
    "synthetic_graph": 712_087_962,
    "verified_synthetic_multihop": 1_068_131_942,
    "wikidata_path_reasoning": 534_065_971,
    "relational_refinement": 178_021_990,
    "objective_auxiliary": 356_043_981,
}
EXPECTED_STRATA = {"iid", "composition_ood", "length_ood", "joint_ood"}
PRIMARY_OOD_STRATA = ["composition_ood", "joint_ood"]
PRIMARY_CELL_IDS = [
    "graph__composition_ood",
    "graph__joint_ood",
    "non_path__composition_ood",
    "non_path__joint_ood",
]
PRIMARY_CONTRAST_ID = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)
SECONDARY_CONTRAST_IDS = [
    "secondary_graph_pair_and_proof__composition_joint_ood__split90_minus_dense",
    "secondary_non_path_pair_and_proof__composition_joint_ood__split90_minus_dense",
]
EXPECTED_CONTROLS = {
    "correct_memory",
    "memory_off",
    "shuffled_returns",
    "relevant_edge_swap",
    "irrelevant_edge_swap",
    "gold_path_replay",
    "no_query",
    "entity_rename",
    "graph_isomorphism",
    "page_order_permutation",
}
EXPECTED_GATES = [
    "scientific_contract",
    "route_dose",
    "semantic_closure",
    "proof_verification",
    "ood_seal",
    "corpus_identity",
    "paired_training",
    "checkpoint_resume",
    "evaluation_validity",
    "six_29m_diagnostics",
]
EXPECTED_ARTIFACT_BINDINGS = {
    "expanded_fineweb_snapshot",
    "finemath_snapshot",
    "reasoning_corpus",
    "route_manifest",
    "exact_graph_memory",
    "model_visible_evaluation",
    "sealed_gold_evaluation",
    "environment_lock",
    "run_manifest",
}


def _load_json(path: Path) -> dict:
    assert path.is_file(), f"missing frozen contract: {path.relative_to(ROOT)}"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_yaml(path: Path) -> dict:
    assert path.is_file(), f"missing frozen contract: {path.relative_to(ROOT)}"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _hash_fields(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("sha256"):
                yield key, child
            yield from _hash_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from _hash_fields(child)


def test_dataset_contract_freezes_status_hypothesis_and_sprint_budget():
    contract = _load_json(DATASET_PATH)

    assert contract["schema_version"] == 2
    assert contract["contract_id"] == "memorysplit-reasoning-dataset-v2"
    assert contract["status"] == {
        "scientific_status": "incomplete",
        "interim_evidence_label": "none",
        "protected_launch_allowed": False,
        "reason": "No valid claim-bearing MemorySplit v2 experiment has finished.",
    }
    assert contract["canonical_hypothesis"] == CANONICAL_HYPOTHESIS

    sprint = contract["sprint_recipe"]
    assert sprint["label"] == "20tpp_reasoning_maximized_sprint"
    assert sprint["reasoning_optimal_claim"] is False
    assert sprint["model_parameters"] == 356_033_536
    assert sprint["nominal_tokens_per_parameter"] == 20
    assert sprint["targets_per_update"] == 524_288
    assert sprint["optimizer_steps"] == 13_582
    assert sprint["raw_target_tokens"] == 7_120_879_616
    assert sprint["raw_target_tokens"] == (
        sprint["targets_per_update"] * sprint["optimizer_steps"]
    )
    assert sprint["raw_target_tokens"] >= (
        sprint["model_parameters"] * sprint["nominal_tokens_per_parameter"]
    )
    assert sprint["mixture_unit"] == (
        "raw_causal_target_tokens_before_condition_weights"
    )


def test_target_shares_use_stable_hamilton_realized_token_quotas():
    sprint = _load_json(DATASET_PATH)["sprint_recipe"]
    lanes = sprint["lanes"]
    shares = {lane["id"]: lane["share_percent"] for lane in lanes}

    assert shares == EXPECTED_MIXTURE
    assert sum(shares.values()) == 100.0
    assert sprint.get("share_semantics") == "target_percentages"

    allocation = sprint.get("realized_token_allocation")
    assert allocation, "realized Hamilton token allocation is missing"
    assert allocation["method"] == "hamilton_largest_remainder"
    assert allocation["tie_break"] == "stable_lane_order"
    assert allocation["lane_order"] == list(EXPECTED_MIXTURE)
    assert allocation["total_tokens"] == 7_120_879_616
    assert allocation["token_quotas"] == EXPECTED_REALIZED_TOKEN_QUOTAS
    assert sum(allocation["token_quotas"].values()) == allocation["total_tokens"]

    for lane_id, target_percent in EXPECTED_MIXTURE.items():
        ideal = (
            Decimal(allocation["total_tokens"])
            * Decimal(str(target_percent))
            / Decimal(100)
        )
        error = abs(Decimal(allocation["token_quotas"][lane_id]) - ideal)
        assert error < Decimal(1), f"{lane_id} allocation error is {error}"


def test_language_floors_hold_for_target_mixture():
    sprint = _load_json(DATASET_PATH)["sprint_recipe"]
    lanes = sprint["lanes"]

    floors = sprint["language_floors_percent"]
    assert floors == {
        "human_language": 40.0,
        "broad_general_language": 25.0,
    }
    human_language = sum(
        lane["share_percent"] for lane in lanes if lane["human_language"]
    )
    broad_language = sum(
        lane["share_percent"] for lane in lanes if lane["broad_general_language"]
    )
    assert human_language >= floors["human_language"]
    assert broad_language >= floors["broad_general_language"]


def test_source_caps_and_generated_reasoning_verification_are_frozen():
    sprint = _load_json(DATASET_PATH)["sprint_recipe"]
    policy = sprint["source_policy"]

    assert policy["finemath_subsets_in_order"] == [
        "finemath-4plus",
        "finemath-3plus-cross-deduplicated-remainder",
    ]
    assert policy["cross_deduplicate_finemath_against_fineweb"] is True
    assert policy["arc_conceptarc_total_max_percent"] == 0.25
    assert policy["teacher_generated_cot_total_max_percent"] == 0.5
    assert policy["teacher_generated_cot_requires"] == [
        "independent_answer_validation",
        "trace_validation",
        "contamination_review",
    ]

    lanes = {lane["id"]: lane for lane in sprint["lanes"]}
    for lane_id in (
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
    ):
        assert lanes[lane_id]["verification"] == "solver_required"


def test_intervention_is_matched_and_offloads_most_factual_burden():
    intervention = _load_json(DATASET_PATH)["intervention"]

    assert intervention["primary_arms"] == ["dense", "split90"]
    assert intervention["published_views"] == [
        "dense",
        "split50",
        "split90",
        "random_fact_90",
        "matched_nonfactual_mask",
    ]
    assert intervention["split90_minimum_percent"] == {
        "distinct_offloadable_atomic_facts": 90.0,
        "train_only_information_weighted_burden": 90.0,
    }
    assert set(intervention["always_internal"]) >= {
        "rules",
        "operators",
        "schemas",
        "proof_procedures",
    }
    assert intervention["only_claim_bearing_difference"] == (
        "direct_target_weights_on_routed_factual_payloads"
    )

    invariants = set(intervention["matched_pair_invariants"])
    assert {
        "parameters",
        "initialization",
        "raw_target_tokens",
        "token_order",
        "optimizer",
        "optimizer_schedule",
        "inference_budget",
        "exact_graph_memory",
    } <= invariants

    memory = intervention["memory"]
    assert memory["kind"] == "exact_non_trainable_graph"
    assert memory["identical_between_arms"] is True
    assert memory["byte_identity_required"] is True
    assert memory["may_return"] == ["exact_row", "MISS"]
    assert set(memory["may_not"]) == {
        "rank_candidates",
        "infer_paths",
        "generate_proofs",
        "generate_answers",
    }


def test_primary_endpoint_requires_counterfactual_pair_and_proof():
    evaluation = _load_json(DATASET_PATH)["evaluation"]
    endpoint = evaluation["primary_endpoint"]

    assert endpoint["id"] == "counterfactual_pair_and_proof_accuracy"
    assert endpoint["verifier_required"] is True
    assert endpoint["pair_credit"] == (
        "one_iff_both_twins_have_correct_answers_and_verifier_accepted_proofs"
    )
    assert endpoint["weighting"] == "equal_across_four_primary_cells"
    assert set(endpoint["reasoning_families"]) == {"graph", "non_path"}
    assert endpoint["primary_ood_strata"] == PRIMARY_OOD_STRATA
    assert endpoint["primary_cell_ids"] == PRIMARY_CELL_IDS

    release = evaluation["release_contract"]
    assert release["model_visible_separate_from_sealed_gold"] is True
    assert release["model_visible_contains_answers_or_proofs"] is False
    assert release["sealed_gold_evaluation_only"] is True

    strata = {stratum["id"]: stratum for stratum in evaluation["sealed_strata"]}
    assert set(strata) == EXPECTED_STRATA
    assert all(stratum["sealed"] is True for stratum in strata.values())
    assert all(stratum["training_overlap_allowed"] is False for stratum in strata.values())
    assert {control["id"] for control in evaluation["controls"]} == EXPECTED_CONTROLS


def test_strata_roles_exclude_iid_and_length_from_primary_ood_endpoint():
    dataset_evaluation = _load_json(DATASET_PATH)["evaluation"]
    prereg_evaluation = _load_yaml(PREREG_PATH)["evaluation"]
    expected_classification = {
        "primary_ood_strata": ["composition_ood", "joint_ood"],
        "secondary_strata": ["length_ood"],
        "guardrail_strata": ["iid"],
    }

    assert dataset_evaluation.get("stratum_classification") == expected_classification
    assert prereg_evaluation.get("stratum_classification") == expected_classification
    assert prereg_evaluation["primary_cell_ids"] == PRIMARY_CELL_IDS
    assert "equal_weight_by" not in prereg_evaluation
    assert prereg_evaluation["primary_weighting"] == {
        "method": "equal_across_four_primary_cells",
        "cell_weight": 0.25,
        "cell_ids": PRIMARY_CELL_IDS,
    }
    assert "iid" not in prereg_evaluation["stratum_classification"]["primary_ood_strata"]
    assert "length_ood" not in prereg_evaluation["primary_cell_ids"]

    roles = {
        stratum["id"]: stratum["role"]
        for stratum in dataset_evaluation["sealed_strata"]
    }
    assert roles == {
        "iid": "guardrail",
        "composition_ood": "primary_ood",
        "length_ood": "secondary",
        "joint_ood": "primary_ood",
    }


def test_preregistration_freezes_n5_seed0_and_symmetric_seven_a100_plan():
    prereg = _load_yaml(PREREG_PATH)

    assert prereg["schema_version"] == 2
    assert prereg["preregistration_id"] == "memorysplit-confirmatory-v2"
    assert prereg["frozen"] is True
    assert prereg["current_scientific_status"] == "incomplete"
    assert prereg["current_interim_evidence_label"] == "none"
    assert prereg["protected_launch_allowed"] is False
    assert prereg["canonical_hypothesis"] == CANONICAL_HYPOTHESIS

    cohort = prereg["protected_cohort"]
    assert cohort["condition_pair"] == ["dense", "split90"]
    assert cohort["model_parameters"] == 356_033_536
    assert cohort["terminal_n_pairs"] == 5
    assert cohort["seeds"] == [0, 1, 2, 3, 4]
    assert cohort["precommit_before_seed0_unblinding"] is True
    assert cohort["continue_after_seed0_regardless_of_effect"] is True
    assert cohort["seed0"] == {
        "scientific_status_after_valid_completion": "incomplete",
        "interim_evidence_label": "directional_only",
        "completion_label": "1/5",
        "retained_in_terminal_cohort": True,
        "stop_only_for_validity_or_infrastructure_failure": True,
    }

    training = cohort["training"]
    assert training == {
        "nominal_tokens_per_parameter": 20,
        "targets_per_update": 524_288,
        "optimizer_steps": 13_582,
        "raw_target_tokens": 7_120_879_616,
    }
    assert training["raw_target_tokens"] == (
        training["targets_per_update"] * training["optimizer_steps"]
    )

    allocation = cohort["seed0_allocation"]
    assert allocation["accelerator"] == "A100_80GB"
    assert allocation["total_gpus"] == 7
    assert allocation["dense_training_gpus"] == [0, 1, 2]
    assert allocation["split_training_gpus"] == [3, 4, 5]
    assert allocation["evaluation_verification_gpus"] == [6]
    assert allocation["symmetric_training"] is True
    assert allocation["four_vs_three_training_forbidden"] is True


def test_six_29m_diagnostic_runs_are_exact_and_pending():
    diagnostic = _load_yaml(PREREG_PATH)["development_diagnostics_29m"]

    assert diagnostic["required"] is True
    assert diagnostic["run_count"] == 6
    assert diagnostic["model_parameters"] == 28_969_216
    assert diagnostic["targets_per_update"] == 524_288
    assert diagnostic["optimizer_steps"] == 1_106
    assert diagnostic["raw_target_tokens"] == 579_862_528
    assert diagnostic["protected_launch_requires_all_pass"] is True

    runs = diagnostic["runs"]
    assert [(run["pair_id"], run["arm"]) for run in runs] == [
        ("full_corpus", "dense"),
        ("full_corpus", "split90"),
        ("no_arc_conceptarc", "dense"),
        ("no_arc_conceptarc", "split90"),
        ("no_refinement", "dense"),
        ("no_refinement", "split90"),
    ]
    assert all(run["state"] == "pending" for run in runs)
    replacements = {
        run["pair_id"]: run["token_matched_replacement"] for run in runs
    }
    assert replacements == {
        "full_corpus": "none",
        "no_arc_conceptarc": "fineweb_edu",
        "no_refinement": "verified_standard_relational_records",
    }


def test_primary_inference_is_operationally_frozen_and_narrow_at_n5():
    prereg = _load_yaml(PREREG_PATH)

    assert set(prereg["evaluation"]["required_controls"]) == EXPECTED_CONTROLS
    assert set(prereg["evaluation"]["sealed_strata"]) == EXPECTED_STRATA
    assert (
        prereg["evaluation"]["primary_endpoint"]
        == "counterfactual_pair_and_proof_accuracy"
    )

    analysis = prereg["analysis"]
    assert analysis["independent_unit"] == "paired_training_bundle"
    assert analysis["single_primary_hypothesis"] is True
    primary = analysis.get("primary_hypothesis")
    assert primary, "single primary hypothesis is not operationally specified"
    assert primary["contrast_id"] == PRIMARY_CONTRAST_ID
    assert primary["endpoint"] == "counterfactual_pair_and_proof_accuracy"
    assert primary["estimand"] == "split90_minus_dense"
    assert primary["reasoning_families"] == ["graph", "non_path"]
    assert primary["primary_ood_strata"] == PRIMARY_OOD_STRATA
    assert primary["cell_ids"] == PRIMARY_CELL_IDS
    assert primary["cell_weight"] == 0.25
    assert primary["paired_seed_bundle_delta"] == (
        "split90_equal_weight_omnibus_minus_dense_equal_weight_omnibus"
    )
    assert primary["test"] == {
        "method": "exact_one_sided_exhaustive_sign_flip",
        "alpha": 0.05,
        "statistic": "arithmetic_mean_of_paired_seed_bundle_deltas",
        "sign_assignments": "all_2_to_n",
        "tail_count_rule": "permuted_statistic_greater_than_or_equal_to_observed",
        "equality_counted": True,
        "zero_deltas": "retained",
        "n_pairs": 5,
        "minimum_attainable_p": 0.03125,
        "confirmatory_scope": "primary_omnibus_only",
    }

    assert analysis["hierarchical_bootstrap"] == {
        "draws": 20_000,
        "resampling_levels": ["seed", "world", "counterfactual_pair"],
    }
    secondary = analysis["secondary_family_contrasts"]
    assert secondary["contrast_ids"] == SECONDARY_CONTRAST_IDS
    assert secondary["holm"] == {
        "method": "holm",
        "family_size": 2,
        "tie_break": "stable_lexicographic_contrast_id_order",
    }
    assert secondary["required_for_n5_primary_status"] is False
    assert secondary["broader_both_families_claim"] == {
        "n5_can_establish": False,
        "minimum_pilot_powered_cohort_n": 6,
    }

    assert analysis["practical_equivalence"] == {
        "contrast_id": PRIMARY_CONTRAST_ID,
        "margin_absolute_pair_accuracy": 0.01,
        "method": "two_one_sided_90_percent_confidence_bounds",
        "bound_estimator": "hierarchical_bootstrap_seed_world_pair",
        "lower_rule": "strictly_greater_than_negative_margin",
        "upper_rule": "strictly_less_than_positive_margin",
        "boundary_equality": "does_not_support_equivalence",
    }
    assert analysis["fixed_checkpoint_aulc"] == {
        "enabled": True,
        "optimizer_steps": [1358, 3396, 6791, 10187, 13582],
        "interpolation": "none",
    }
    assert analysis["conclusion_scope"] == {
        "supports_effect_at_n5": "narrow_primary_omnibus_only",
        "requires_all_validity_and_guardrail_gates": True,
        "does_not_establish_both_families_claim": True,
    }


def test_scientific_status_and_interim_evidence_are_separate_axes():
    prereg = _load_yaml(PREREG_PATH)
    policy = prereg["status_policy"]

    assert "precedence" not in policy
    assert policy["scientific_status"]["values"] == [
        "incomplete",
        "invalid",
        "complete",
    ]
    assert policy["scientific_status"]["rules"]["incomplete"] == (
        "missing_required_runs_evaluations_instruments_or_replication"
    )
    assert policy["scientific_status"]["rules"]["invalid"] == (
        "measured_protocol_provenance_endpoint_or_instrument_failure"
    )
    assert policy["interim_evidence_label"]["values"] == [
        "none",
        "directional_only",
        "sign_consistent_only",
    ]
    assert set(policy["scientific_status"]["values"]).isdisjoint(
        {"directional_only", "sign_consistent_only"}
    )
    assert policy["final_inference_conclusion"]["values"] == [
        "not_evaluated",
        "inconclusive",
        "supports_effect",
        "supports_practical_null",
    ]
    assert "failed_to_reject" in policy["forbidden_labels"]

    seed0 = prereg["protected_cohort"]["seed0"]
    assert seed0["scientific_status_after_valid_completion"] == "incomplete"
    assert seed0["interim_evidence_label"] == "directional_only"


def test_protected_run_gates_fail_closed():
    prereg = _load_yaml(PREREG_PATH)
    gates = prereg["protected_run_gates"]
    assert gates["required_order"] == EXPECTED_GATES
    assert gates["all_required"] is True
    assert [gate["id"] for gate in gates["gates"]] == EXPECTED_GATES
    assert all(gate["state"] == "pending" for gate in gates["gates"])
    assert all(gate["evidence_state"] == "unfrozen" for gate in gates["gates"])
    assert all(gate["evidence_sha256"] is None for gate in gates["gates"])


def test_unbuilt_artifact_hashes_are_null_and_never_placeholders():
    dataset = _load_json(DATASET_PATH)
    prereg = _load_yaml(PREREG_PATH)

    for contract in (dataset, prereg):
        bindings = contract["artifact_bindings"]
        assert set(bindings) == EXPECTED_ARTIFACT_BINDINGS
        assert all(binding == {"state": "unfrozen", "sha256": None} for binding in bindings.values())
        assert all(value is None for _, value in _hash_fields(contract))

    for path in (DATASET_PATH, PREREG_PATH, AUDIT_PATH):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"\b[0-9a-fA-F]{64}\b", text) is None


def test_audit_records_separate_status_axes_and_known_gaps():
    assert AUDIT_PATH.is_file(), "missing dataset audit"
    audit = AUDIT_PATH.read_text(encoding="utf-8")

    required_text = (
        "**Current scientific status:** `incomplete`",
        "**Protected launch allowed:** `false`",
        "No valid claim-bearing MemorySplit v2 experiment has finished.",
        "A measured protocol failure sets `scientific_status: invalid`.",
        "Missing terminal evidence sets `scientific_status: incomplete`.",
        "After one valid seed-0 pair, the experiment remains "
        "`scientific_status: incomplete`",
        "`interim_evidence_label: directional_only`",
        "target percentages",
        "Hamilton largest-remainder",
        "six relation programs",
        "Wikidata reasoning is one hop",
        "one refinement template",
        "no production proof verifier",
        "no sealed OOD suite",
        "50% hash route",
        "supervised candidate and final-answer copies",
        "repeated filler",
        "duplicate target positions",
        "six-run 29M diagnostic gate",
        "131,072-token smoke corpus",
        "Do not scale the current compiler beyond 20 tokens per parameter.",
    )
    for text in required_text:
        assert text in audit

    for total, relational in (
        ("579,862,528", "144,965,632"),
        ("3,244,818,432", "811,204,608"),
        ("7,120,879,616", "1,780,219,904"),
    ):
        assert total in audit
        assert relational in audit

    for lane_id, token_quota in EXPECTED_REALIZED_TOKEN_QUOTAS.items():
        assert lane_id in audit
        assert f"{token_quota:,}" in audit

    assert "`invalid` takes precedence over `incomplete`" not in audit
