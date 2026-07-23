from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from msctl.cohort import load_cohort_assignment, load_cohort_assignment_bytes
from msctl.errors import MsctlError


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = REPO_ROOT / "configs" / "cohort-assignment-v3.json"
PREREGISTRATION = REPO_ROOT / "configs" / "preregistration-v3.yaml"
CONFIG_DIR = REPO_ROOT / "configs" / "360m-v3"
V2_ASSIGNMENT = REPO_ROOT / "configs" / "cohort-assignment-v2.json"
V2_PREREGISTRATION = REPO_ROOT / "configs" / "preregistration-v2.yaml"
V2_CONFIG_DIR = REPO_ROOT / "configs" / "360m-v2"

ARMS = ("dense", "split90")
SEEDS = tuple(range(10))
SNAPSHOT_STEPS = [1_358, 3_396, 6_791, 10_187, 13_582]
CONFIG_FIELDS = {
    "schema_version",
    "cohort_id",
    "run_id",
    "condition",
    "seed",
    "model",
    "ctx",
    "train_corpus",
    "sidecar_name",
    "out_dir",
    "micro_batch_size",
    "tokens_per_step",
    "max_steps",
    "total_tokens",
    "lr",
    "warmup_steps",
    "weight_decay",
    "compile",
    "device",
    "log_every",
    "eval_every",
    "snapshot_steps",
    "ckpt_minutes",
}
IDENTITY_FIELDS = {
    "run_id",
    "condition",
    "seed",
    "out_dir",
    "sidecar_name",
}
EXPECTED_TRAINING_FIELDS = {
    "schema_version": 2,
    "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
    "model": "d360m",
    "ctx": 1024,
    "train_corpus": "dataset/receipt.json",
    "micro_batch_size": 8,
    "tokens_per_step": 524_288,
    "max_steps": 13_582,
    "total_tokens": 7_120_879_616,
    "lr": 0.001,
    "warmup_steps": 300,
    "weight_decay": 0.1,
    "compile": True,
    "device": "cuda",
    "log_every": 20,
    "eval_every": 250,
    "snapshot_steps": SNAPSHOT_STEPS,
    "ckpt_minutes": 30,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _yaml_value(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    configs = root / "configs"
    shutil.copytree(CONFIG_DIR, configs / CONFIG_DIR.name)
    shutil.copy2(PREREGISTRATION, configs / PREREGISTRATION.name)
    shutil.copy2(ASSIGNMENT, configs / ASSIGNMENT.name)
    return root, configs / ASSIGNMENT.name


def _snapshot(
    assignment: Path,
) -> tuple[bytes, bytes, dict[str, bytes]]:
    configs_root = assignment.parent
    run_root = configs_root / CONFIG_DIR.name
    config_data = {
        f"configs/{CONFIG_DIR.name}/{path.name}": path.read_bytes()
        for path in sorted(run_root.glob("*.yaml"))
    }
    return (
        assignment.read_bytes(),
        (configs_root / PREREGISTRATION.name).read_bytes(),
        config_data,
    )


def test_v3_assignment_is_canonical_json_with_trailing_newline():
    value = _json_value(ASSIGNMENT)

    assert value == {
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "provider_seeds": {
            "aws-p5.48xlarge": list(SEEDS),
        },
        "raw_target_tokens": 7_120_879_616,
        "schema_version": 3,
        "targets_per_update": 524_288,
    }
    assert ASSIGNMENT.read_bytes() == (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def test_v3_loads_aws_only_complete_paired_snapshot_cohort():
    cohort = load_cohort_assignment(ASSIGNMENT)
    assignment_data, preregistration_data, config_data = _snapshot(ASSIGNMENT)
    bytes_cohort = load_cohort_assignment_bytes(
        assignment_filename=ASSIGNMENT.name,
        assignment_data=assignment_data,
        preregistration_data=preregistration_data,
        config_data=config_data,
    )

    assert cohort == bytes_cohort
    assert cohort.cohort_id == "memorysplit-confirmatory-v3-360m-n10-aws"
    assert cohort.illumina_seeds == ()
    assert cohort.aws_p5_seeds == SEEDS
    assert cohort.model_parameters == 356_033_536
    assert cohort.targets_per_update == 524_288
    assert cohort.optimizer_steps == 13_582
    assert cohort.raw_target_tokens == 7_120_879_616
    assert cohort.targets_per_update * cohort.optimizer_steps == (
        cohort.raw_target_tokens
    )
    assert len(cohort.configs) == 20
    assert {(config.seed, config.condition) for config in cohort.configs} == {
        (seed, arm) for seed in SEEDS for arm in ARMS
    }
    assert sum(len(config.snapshot_steps) for config in cohort.configs) == 100
    assert {config.snapshot_steps for config in cohort.configs} == {
        tuple(SNAPSHOT_STEPS)
    }
    assert cohort.assignment_sha256 == _sha256(ASSIGNMENT)
    assert cohort.preregistration_sha256 == _sha256(PREREGISTRATION)
    assert set(cohort.config_sha256s) == {
        f"configs/360m-v3/{arm}-s{seed}.yaml"
        for seed in SEEDS
        for arm in ARMS
    }
    for relative, digest in cohort.config_sha256s.items():
        assert digest == _sha256(REPO_ROOT / relative)


def test_v3_run_configs_are_exact_matched_pairs():
    cohort = load_cohort_assignment(ASSIGNMENT)
    by_cell = {
        (config.seed, config.condition): _yaml_value(REPO_ROOT / config.path)
        for config in cohort.configs
    }

    for seed in SEEDS:
        dense = by_cell[(seed, "dense")]
        split90 = by_cell[(seed, "split90")]
        assert set(dense) == set(split90) == CONFIG_FIELDS
        assert {
            key: value
            for key, value in dense.items()
            if key not in IDENTITY_FIELDS
        } == {
            key: value
            for key, value in split90.items()
            if key not in IDENTITY_FIELDS
        }

        for arm, value in (("dense", dense), ("split90", split90)):
            assert {
                key: value
                for key, value in value.items()
                if key not in IDENTITY_FIELDS
            } == EXPECTED_TRAINING_FIELDS
            assert value["run_id"] == f"memorysplit-v3-360m-s{seed}-{arm}"
            assert value["condition"] == arm
            assert value["seed"] == seed
            assert value["sidecar_name"] == f"{arm}_target_weights"
            assert value["out_dir"] == f"runs/seed-{seed}/{arm}"


def test_v3_preregistration_freezes_amendment_statistics_and_aws_topology():
    prereg = _yaml_value(PREREGISTRATION)

    assert prereg["schema_version"] == 3
    assert prereg["preregistration_id"] == "memorysplit-confirmatory-v3"
    assert prereg["frozen"] is True
    assert prereg["current_scientific_status"] == "incomplete"
    assert prereg["current_interim_evidence_label"] == "none"
    assert prereg["protected_launch_allowed"] is False

    amendment = prereg["prospective_amendment"]
    assert amendment == {
        "supersedes": "split_provider_n5_execution_plan",
        "execution_plan": "one_aws_p5.48xlarge_n10_cohort",
        "outcome_inspection_before_amendment": {
            "seed_level_confirmatory_result_inspected": False,
            "arm_level_confirmatory_result_inspected": False,
        },
        "continuation": {
            "all_ten_pairs_required": True,
            "effect_direction_may_not_change_continuation": True,
            "stop_only_for": [
                "measured_preregistered_validity_failure",
                "infrastructure_failure",
            ],
        },
        "trained_control_arms_added": False,
        "claim_bearing_condition_pair": ["dense", "split90"],
        "unchanged": [
            "model_architecture",
            "corpus_bytes",
            "corpus_order",
            "target_budget",
            "optimizer_schedule",
            "primary_endpoint",
            "one_sided_direction",
            "alpha",
            "inclusive_tail_rule",
            "validity_gates",
        ],
    }

    cohort = prereg["protected_cohort"]
    assert cohort["condition_pair"] == ["dense", "split90"]
    assert cohort["model_parameters"] == 356_033_536
    assert cohort["terminal_n_pairs"] == 10
    assert cohort["seeds"] == list(SEEDS)
    assert cohort["provider_assignment"] == {
        "aws-p5.48xlarge": list(SEEDS),
    }
    assert cohort["training"] == {
        "nominal_tokens_per_parameter": 20,
        "targets_per_update": 524_288,
        "optimizer_steps": 13_582,
        "raw_target_tokens": 7_120_879_616,
    }
    assert cohort["aws_topology"] == {
        "instance_type": "p5.48xlarge",
        "instances": 1,
        "accelerator": "NVIDIA H100 80GB",
        "total_gpus": 8,
        "dense_training_gpus": [0, 1, 2, 3],
        "split90_training_gpus": [4, 5, 6, 7],
        "train_groups": [4, 4],
        "symmetric_training": True,
        "purchase_model": "on_demand",
    }
    assert cohort["extra_trained_control_arms"] == []
    assert cohort["terminal_evidence"] == {
        "paired_bundles_required": 10,
        "all_validity_and_evaluation_evidence_required": True,
    }

    analysis = prereg["analysis"]
    primary = analysis["primary_hypothesis"]
    assert primary["estimand"] == "split90_minus_dense"
    assert primary["timepoint"] == "final_optimizer_step"
    assert primary["optimizer_step"] == 13_582
    assert primary["test"] == {
        "method": "exact_one_sided_exhaustive_sign_flip",
        "alpha": 0.05,
        "statistic": "arithmetic_mean_of_paired_seed_bundle_deltas",
        "sign_assignments": 1024,
        "tail_count_rule": (
            "permuted_statistic_greater_than_or_equal_to_observed"
        ),
        "equality_counted": True,
        "zero_deltas": "retained",
        "n_pairs": 10,
        "minimum_attainable_p": 0.0009765625,
        "confirmatory_scope": "primary_omnibus_only",
    }
    assert analysis["fixed_checkpoint_aulc"] == {
        "enabled": True,
        "role": "required_secondary_trajectory_evidence",
        "second_primary": False,
        "optimizer_steps": SNAPSHOT_STEPS,
        "all_steps_required_for_every_arm_and_seed": True,
        "interpolation": "none",
        "integral": "right_step",
    }
    assert analysis["hierarchical_bootstrap"] == {
        "bit_generator": "PCG64",
        "rng_seed": 0,
        "draws": 20_000,
        "confidence_interval_percent": 90,
        "resampling_levels": ["seed", "world", "counterfactual_pair"],
    }
    assert analysis["practical_equivalence"] == {
        "contrast_id": prereg["evaluation"]["primary_contrast_id"],
        "margin_absolute_pair_accuracy": 0.01,
        "method": "two_one_sided_90_percent_confidence_bounds",
        "bound_estimator": "hierarchical_bootstrap_seed_world_pair",
        "lower_rule": "strictly_greater_than_negative_margin",
        "upper_rule": "strictly_less_than_positive_margin",
        "boundary_equality": "does_not_support_equivalence",
    }
    assert analysis["terminal_status_requires"] == {
        "paired_bundles": 10,
        "all_validity_and_evaluation_evidence": True,
    }


def test_v3_preregistration_retains_diagnostics_controls_gates_and_null_hashes():
    prereg = _yaml_value(PREREGISTRATION)

    diagnostics = prereg["development_diagnostics_29m"]
    assert diagnostics["required"] is True
    assert diagnostics["run_count"] == 6
    assert diagnostics["model_parameters"] == 28_969_216
    assert diagnostics["targets_per_update"] == 524_288
    assert diagnostics["optimizer_steps"] == 1_106
    assert diagnostics["raw_target_tokens"] == 579_862_528
    assert diagnostics["protected_launch_requires_all_pass"] is True
    assert diagnostics["selection_use_forbidden"] is True
    assert diagnostics["effect_direction_as_pass_criterion_forbidden"] is True
    assert len(diagnostics["runs"]) == 6

    assert set(prereg["evaluation"]["sealed_strata"]) == {
        "iid",
        "composition_ood",
        "length_ood",
        "joint_ood",
    }
    assert set(prereg["evaluation"]["required_controls"]) == {
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
    assert prereg["evaluation"]["verifier_required"] is True
    assert prereg["intervention_gates"] == {
        "minimum_offload_percent": {
            "distinct_offloadable_atomic_facts": 90.0,
            "train_only_information_weighted_burden": 90.0,
        },
        "semantic_mask_closure_required": True,
        "rules_operators_schemas_and_proof_procedures_internal": True,
        "random_fact_and_matched_nonfactual_controls_required": True,
    }
    expected_gate_ids = [
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
    gates = prereg["protected_run_gates"]
    assert gates["all_required"] is True
    assert gates["required_order"] == expected_gate_ids
    assert [gate["id"] for gate in gates["gates"]] == expected_gate_ids
    assert all(gate["state"] == "pending" for gate in gates["gates"])
    assert all(gate["evidence_state"] == "unfrozen" for gate in gates["gates"])
    assert all(gate["evidence_sha256"] is None for gate in gates["gates"])
    assert prereg["protected_launch_allowed"] is False
    assert set(prereg["artifact_bindings"]) == {
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
    assert all(
        binding == {"state": "unfrozen", "sha256": None}
        for binding in prereg["artifact_bindings"].values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "illumina_reintroduced",
        "seed_10",
        "duplicate_seed",
        "wrong_cohort_id",
    ],
)
def test_v3_rejects_malformed_provider_assignment(tmp_path, mutation):
    _, assignment = _fixture_repo(tmp_path)
    value = _json_value(assignment)
    providers = value["provider_seeds"]
    assert isinstance(providers, dict)

    if mutation == "illumina_reintroduced":
        providers["illumina-usfc-prd"] = []
    elif mutation == "seed_10":
        providers["aws-p5.48xlarge"] = [*range(9), 10]
    elif mutation == "duplicate_seed":
        providers["aws-p5.48xlarge"] = [*range(9), 8]
    elif mutation == "wrong_cohort_id":
        value["cohort_id"] = "memorysplit-confirmatory-v2-360m-n5"
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)
    _write_json(assignment, value)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_v3_rejects_missing_or_extra_config_cells(tmp_path, mutation):
    root, assignment = _fixture_repo(tmp_path)
    run_root = root / "configs" / CONFIG_DIR.name
    if mutation == "missing":
        (run_root / "split90-s9.yaml").unlink()
    else:
        shutil.copy2(run_root / "dense-s0.yaml", run_root / "dense-s10.yaml")

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


@pytest.mark.parametrize(
    ("mutation", "filename"),
    [
        ("bad_dataset", "dense-s0.yaml"),
        ("wrong_run_id", "split90-s4.yaml"),
        ("wrong_snapshot_schedule", "dense-s9.yaml"),
        ("seed_10", "split90-s9.yaml"),
    ],
)
def test_v3_rejects_run_config_contract_drift(tmp_path, mutation, filename):
    root, assignment = _fixture_repo(tmp_path)
    config_path = root / "configs" / CONFIG_DIR.name / filename
    value = _yaml_value(config_path)

    if mutation == "bad_dataset":
        value["train_corpus"] = "dataset/corpus-receipt.json"
    elif mutation == "wrong_run_id":
        value["run_id"] = "memorysplit-v2-360m-s4-split90"
    elif mutation == "wrong_snapshot_schedule":
        value["snapshot_steps"] = [1_357, 3_396, 6_791, 10_187, 13_582]
    elif mutation == "seed_10":
        value["seed"] = 10
        value["run_id"] = "memorysplit-v3-360m-s10-split90"
        value["out_dir"] = "runs/seed-10/split90"
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)
    _write_yaml(config_path, value)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


@pytest.mark.parametrize(
    "substitution",
    ["v2_assignment", "v2_preregistration", "v2_run_config"],
)
def test_v3_rejects_v2_file_substitution(tmp_path, substitution):
    root, assignment = _fixture_repo(tmp_path)
    configs = root / "configs"

    if substitution == "v2_assignment":
        shutil.copy2(V2_ASSIGNMENT, assignment)
    elif substitution == "v2_preregistration":
        shutil.copy2(
            V2_PREREGISTRATION,
            configs / PREREGISTRATION.name,
        )
    elif substitution == "v2_run_config":
        shutil.copy2(
            V2_CONFIG_DIR / "dense-s0.yaml",
            configs / CONFIG_DIR.name / "dense-s0.yaml",
        )
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(substitution)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


def test_bytes_loader_requires_explicit_v3_filename_identity():
    v2_config_data = {
        f"configs/360m-v2/{path.name}": path.read_bytes()
        for path in sorted(V2_CONFIG_DIR.glob("*.yaml"))
    }

    with pytest.raises(MsctlError):
        load_cohort_assignment_bytes(
            assignment_filename="cohort-assignment-v3.json",
            assignment_data=V2_ASSIGNMENT.read_bytes(),
            preregistration_data=V2_PREREGISTRATION.read_bytes(),
            config_data=v2_config_data,
        )

    v2 = load_cohort_assignment_bytes(
        assignment_data=V2_ASSIGNMENT.read_bytes(),
        preregistration_data=V2_PREREGISTRATION.read_bytes(),
        config_data=deepcopy(v2_config_data),
    )
    assert v2.cohort_id == "memorysplit-confirmatory-v2-360m-n5"
    assert v2.illumina_seeds == (0,)
    assert v2.aws_p5_seeds == (1, 2, 3, 4)
