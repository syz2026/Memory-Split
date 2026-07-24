from __future__ import annotations

import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from msctl.errors import MsctlError
from msctl.objective_controls_v3 import (
    load_objective_controls_admission,
    load_objective_controls_contract,
    validate_objective_controls_admission,
)
from train.model import PRESETS


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "configs" / "objective-controls-amendment-v3.yaml"
MANIFEST = ROOT / "configs" / "29m-v3" / "manifest.json"
PREREGISTRATION = ROOT / "configs" / "preregistration-v3.yaml"
COHORT = ROOT / "configs" / "cohort-assignment-v3.json"

AMENDMENT_ID = "memorysplit-v3-objective-controls-amendment-1"
MANIFEST_ID = "memorysplit-v3-objective-controls-29m"
PREREGISTRATION_SHA256 = (
    "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
)
COHORT_SHA256 = (
    "47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c"
)
SHARED_INITIALIZATION_ID = "memorysplit-v3-29m-shared-init-s0"
RUN_IDS = (
    "full_corpus_dense",
    "full_corpus_split90",
    "no_arc_conceptarc_dense",
    "no_arc_conceptarc_split90",
    "no_refinement_dense",
    "no_refinement_split90",
    "full_corpus_random_fact90",
    "full_corpus_matched_nonfactual_mask",
)
ORIGINAL_RUN_IDS = RUN_IDS[:6]
CONTROL_RUN_IDS = RUN_IDS[6:]
PRIMARY_CELLS = (
    "graph__composition_ood",
    "graph__joint_ood",
    "non_path__composition_ood",
    "non_path__joint_ood",
)
STATISTICAL_EXCLUSIONS = (
    "primary_exact_test",
    "fixed_checkpoint_aulc",
    "practical_equivalence_test",
    "continuation_decisions",
    "effect_direction_gate",
)
CORPUS_IDS = {
    "full_corpus": "corpus-v3-29m-full-v1",
    "no_arc_conceptarc": (
        "corpus-v3-29m-no-arc-conceptarc-"
        "fineweb-edu-token-matched-v1"
    ),
    "no_refinement": (
        "corpus-v3-29m-no-refinement-"
        "verified-standard-relational-v1"
    ),
}
PROVENANCE_IDS = {
    group: f"provenance-v3-29m-{group.replace('_', '-')}-v1"
    for group in CORPUS_IDS
}
SIDECAR_IDS = {
    run_id: f"sidecar-v3-29m-{run_id.replace('_', '-')}-v1"
    for run_id in RUN_IDS
}
CONFIG_FIELDS = {
    "schema_version",
    "amendment_id",
    "manifest_id",
    "run_id",
    "role",
    "pair_id",
    "condition",
    "seed",
    "initialization_id",
    "model",
    "model_parameters",
    "ctx",
    "corpus_variant_id",
    "train_corpus",
    "sidecar_id",
    "sidecar_name",
    "provenance_id",
    "token_matched_replacement",
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
    "ckpt_minutes",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _copy_contract(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    configs = root / "configs"
    configs.mkdir(parents=True)
    shutil.copy2(AMENDMENT, configs / AMENDMENT.name)
    shutil.copy2(PREREGISTRATION, configs / PREREGISTRATION.name)
    shutil.copy2(COHORT, configs / COHORT.name)
    shutil.copytree(MANIFEST.parent, configs / MANIFEST.parent.name)
    return configs / AMENDMENT.name


def _load_manifest(amendment: Path) -> dict[str, object]:
    path = amendment.parent / "29m-v3" / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite_manifest(amendment: Path, value: object) -> None:
    _write_json(amendment.parent / "29m-v3" / "manifest.json", value)


def _rewrite_config_and_hash(
    amendment: Path,
    run_id: str,
    config: dict[str, object],
    manifest: dict[str, object],
) -> None:
    config_path = amendment.parent / "29m-v3" / f"{run_id}.yaml"
    _write_yaml(config_path, config)
    runs = manifest["runs"]
    assert isinstance(runs, list)
    row = next(row for row in runs if row["run_id"] == run_id)
    row["config_sha256"] = _sha256(config_path)
    _rewrite_manifest(amendment, manifest)


def _valid_admission(contract) -> dict[str, object]:
    corpus_hashes = {
        run.corpus_variant_id: _identity_sha256(
            f"corpus:{run.corpus_variant_id}"
        )
        for run in contract.runs
    }
    provenance_hashes = {
        run.provenance_id: _identity_sha256(
            f"provenance:{run.provenance_id}"
        )
        for run in contract.runs
    }
    return {
        "schema_version": 1,
        "amendment_sha256": contract.amendment_sha256,
        "manifest_sha256": contract.manifest_sha256,
        "runs": [
            {
                "run_id": run.run_id,
                "config_sha256": run.config_sha256,
                "corpus_variant_id": run.corpus_variant_id,
                "corpus_sha256": corpus_hashes[run.corpus_variant_id],
                "sidecar_id": run.sidecar_id,
                "sidecar_sha256": _identity_sha256(
                    f"sidecar:{run.sidecar_id}"
                ),
                "provenance_id": run.provenance_id,
                "provenance_sha256": provenance_hashes[run.provenance_id],
                "seed": run.seed,
                "initialization_id": run.initialization_id,
                "model_parameters": 28_969_216,
                "targets_per_update": 524_288,
                "optimizer_steps": 1_106,
                "raw_target_tokens": 579_862_528,
                "optimization_finite": True,
                "checkpoint_resume_exact": True,
                "checkpoint_resume_max_abs_delta": 0.00001,
                "language_relative_degradation": 0.01,
                "route_audit_passed": True,
                "mask_audit_passed": True,
                "semantic_closure_audit_passed": True,
                "primary_cell_accuracy": (
                    {cell: 0.750001 for cell in PRIMARY_CELLS}
                    if run.run_id in ORIGINAL_RUN_IDS
                    else None
                ),
            }
            for run in contract.runs
        ],
    }


def _admission_run(value: dict[str, object], run_id: str) -> dict[str, object]:
    runs = value["runs"]
    assert isinstance(runs, list)
    return next(row for row in runs if row["run_id"] == run_id)


def test_canonical_amendment_binds_frozen_scope_without_changing_parent_files():
    contract = load_objective_controls_contract(AMENDMENT)

    assert _sha256(PREREGISTRATION) == PREREGISTRATION_SHA256
    assert _sha256(COHORT) == COHORT_SHA256
    assert contract.amendment_id == AMENDMENT_ID
    assert contract.preregistration_sha256 == PREREGISTRATION_SHA256
    assert contract.cohort_assignment_sha256 == COHORT_SHA256
    assert contract.protected_outcomes_inspected is False
    assert contract.protected_cells == tuple(
        f"memorysplit-v3-360m-s{seed}-{arm}"
        for seed in range(10)
        for arm in ("dense", "split90")
    )
    assert len(contract.protected_cells) == 20
    assert contract.added_360m_controls == ()
    assert contract.statistical_exclusions == STATISTICAL_EXCLUSIONS
    assert contract.replicated_360m_selectivity_claim_disclaimed is True


def test_manifest_and_configs_freeze_exact_eight_run_development_matrix():
    contract = load_objective_controls_contract(AMENDMENT)
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))

    assert tuple(run.run_id for run in contract.runs) == RUN_IDS
    assert manifest["run_count"] == 8
    assert MANIFEST.read_bytes() == (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    assert {path.name for path in MANIFEST.parent.iterdir()} == {
        "manifest.json",
        *(f"{run_id}.yaml" for run_id in RUN_IDS),
    }

    for run in contract.runs:
        config_path = ROOT / run.config_path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert set(config) == CONFIG_FIELDS
        assert run.config_sha256 == _sha256(config_path)
        assert config["run_id"] == run.run_id
        assert config["model"] == "toy"
        assert config["model_parameters"] == 28_969_216
        assert config["tokens_per_step"] == 524_288
        assert config["max_steps"] == 1_106
        assert config["total_tokens"] == 579_862_528
        assert config["tokens_per_step"] * config["max_steps"] == config[
            "total_tokens"
        ]
        assert config["seed"] == 0
        assert config["initialization_id"] == SHARED_INITIALIZATION_ID
        assert config["corpus_variant_id"] == run.corpus_variant_id
        assert config["sidecar_id"] == run.sidecar_id
        assert config["provenance_id"] == run.provenance_id

    cfg = PRESETS["toy"]
    hidden = ((int(8 * cfg.d_model / 3) + 63) // 64) * 64
    parameter_count = (
        2 * cfg.vocab_size * cfg.d_model
        + cfg.n_layer
        * (
            4 * cfg.d_model * cfg.d_model
            + 3 * cfg.d_model * hidden
            + 2 * cfg.d_model
        )
        + cfg.d_model
    )
    assert parameter_count == 28_969_216


def test_original_replacements_and_new_control_identities_are_unambiguous():
    contract = load_objective_controls_contract(AMENDMENT)
    by_id = {run.run_id: run for run in contract.runs}

    assert {
        run_id: by_id[run_id].token_matched_replacement
        for run_id in ORIGINAL_RUN_IDS
    } == {
        "full_corpus_dense": "none",
        "full_corpus_split90": "none",
        "no_arc_conceptarc_dense": "fineweb_edu",
        "no_arc_conceptarc_split90": "fineweb_edu",
        "no_refinement_dense": "verified_standard_relational_records",
        "no_refinement_split90": "verified_standard_relational_records",
    }
    assert {
        run_id: by_id[run_id].role for run_id in ORIGINAL_RUN_IDS
    } == {run_id: "learnability" for run_id in ORIGINAL_RUN_IDS}
    assert {
        run_id: by_id[run_id].role for run_id in CONTROL_RUN_IDS
    } == {run_id: "integrity_only" for run_id in CONTROL_RUN_IDS}
    assert {
        run_id: by_id[run_id].corpus_variant_id
        for run_id in (
            "full_corpus_dense",
            "full_corpus_split90",
            *CONTROL_RUN_IDS,
        )
    } == {
        run_id: CORPUS_IDS["full_corpus"]
        for run_id in (
            "full_corpus_dense",
            "full_corpus_split90",
            *CONTROL_RUN_IDS,
        )
    }
    assert by_id["no_arc_conceptarc_dense"].corpus_variant_id == CORPUS_IDS[
        "no_arc_conceptarc"
    ]
    assert by_id["no_refinement_dense"].corpus_variant_id == CORPUS_IDS[
        "no_refinement"
    ]
    assert {
        run.run_id: run.sidecar_id for run in contract.runs
    } == SIDECAR_IDS
    assert {
        group: next(
            run.provenance_id
            for run in contract.runs
            if run.pair_id == group
        )
        for group in CORPUS_IDS
    } == PROVENANCE_IDS


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_contract_rejects_missing_or_extra_runs(tmp_path, mutation):
    amendment = _copy_contract(tmp_path)
    manifest = _load_manifest(amendment)
    runs = manifest["runs"]
    assert isinstance(runs, list)
    if mutation == "missing":
        runs.pop()
    else:
        row = deepcopy(runs[-1])
        row["run_id"] = "unexpected_control"
        row["config"] = "configs/29m-v3/unexpected_control.yaml"
        runs.append(row)
    manifest["run_count"] = len(runs)
    _rewrite_manifest(amendment, manifest)

    with pytest.raises(MsctlError) as caught:
        load_objective_controls_contract(amendment)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("seed", False),
        ("seed", 0.0),
        ("model_parameters", 28_969_216.0),
        ("max_steps", 1_106.0),
        ("compile", 1),
    ],
)
def test_contract_rejects_bool_float_and_integer_aliases(
    tmp_path,
    field,
    alias,
):
    amendment = _copy_contract(tmp_path)
    run_id = "full_corpus_dense"
    config_path = amendment.parent / "29m-v3" / f"{run_id}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = _load_manifest(amendment)
    config[field] = alias
    _rewrite_config_and_hash(amendment, run_id, config, manifest)

    with pytest.raises(MsctlError) as caught:
        load_objective_controls_contract(amendment)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


@pytest.mark.parametrize(
    ("binding", "replacement_run"),
    [
        ("corpus_variant_id", "no_arc_conceptarc_dense"),
        ("sidecar_id", "full_corpus_split90"),
        ("provenance_id", "no_refinement_dense"),
    ],
)
def test_contract_rejects_cross_variant_sidecar_or_provenance_binding(
    tmp_path,
    binding,
    replacement_run,
):
    amendment = _copy_contract(tmp_path)
    manifest = _load_manifest(amendment)
    rows = manifest["runs"]
    assert isinstance(rows, list)
    target = next(row for row in rows if row["run_id"] == "full_corpus_dense")
    replacement = next(row for row in rows if row["run_id"] == replacement_run)
    target[binding] = replacement[binding]
    config_path = amendment.parent / "29m-v3" / "full_corpus_dense.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[binding] = replacement[binding]
    _rewrite_config_and_hash(
        amendment,
        "full_corpus_dense",
        config,
        manifest,
    )

    with pytest.raises(MsctlError) as caught:
        load_objective_controls_contract(amendment)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


@pytest.mark.parametrize("parent", ["preregistration", "cohort"])
def test_contract_rejects_any_frozen_parent_byte_drift(tmp_path, parent):
    amendment = _copy_contract(tmp_path)
    path = amendment.parent / (
        "preregistration-v3.yaml"
        if parent == "preregistration"
        else "cohort-assignment-v3.json"
    )
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(MsctlError) as caught:
        load_objective_controls_contract(amendment)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


def test_admission_accepts_only_boundary_valid_nondirectional_evidence():
    contract = load_objective_controls_contract(AMENDMENT)
    value = _valid_admission(contract)

    admission = validate_objective_controls_admission(contract, value)

    assert admission.run_ids == RUN_IDS
    assert admission.learnability_run_ids == ORIGINAL_RUN_IDS
    assert admission.integrity_only_run_ids == CONTROL_RUN_IDS
    assert admission.directional_control_thresholds_applied == ()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_run",
        "extra_run",
        "config",
        "corpus_variant",
        "sidecar",
        "provenance",
        "corpus_hash_drift",
        "reused_sidecar_hash",
    ],
)
def test_admission_fails_closed_on_run_or_identity_drift(mutation):
    contract = load_objective_controls_contract(AMENDMENT)
    value = _valid_admission(contract)
    runs = value["runs"]
    assert isinstance(runs, list)
    if mutation == "missing_run":
        runs.pop()
    elif mutation == "extra_run":
        row = deepcopy(runs[-1])
        row["run_id"] = "unexpected_control"
        runs.append(row)
    elif mutation == "config":
        runs[0]["config_sha256"] = "0" * 64
    elif mutation == "corpus_variant":
        runs[0]["corpus_variant_id"] = runs[2]["corpus_variant_id"]
    elif mutation == "sidecar":
        runs[0]["sidecar_id"] = runs[1]["sidecar_id"]
    elif mutation == "provenance":
        runs[0]["provenance_id"] = runs[4]["provenance_id"]
    elif mutation == "corpus_hash_drift":
        runs[1]["corpus_sha256"] = "0" * 64
    elif mutation == "reused_sidecar_hash":
        runs[0]["sidecar_sha256"] = runs[1]["sidecar_sha256"]
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)

    with pytest.raises(MsctlError) as caught:
        validate_objective_controls_admission(contract, value)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("optimization_finite", False),
        ("checkpoint_resume_exact", False),
        ("checkpoint_resume_max_abs_delta", 0.0000100001),
        ("checkpoint_resume_max_abs_delta", True),
        ("checkpoint_resume_max_abs_delta", 0),
        ("language_relative_degradation", 0.0100001),
        ("language_relative_degradation", True),
        ("language_relative_degradation", 0),
        ("route_audit_passed", False),
        ("mask_audit_passed", False),
        ("semantic_closure_audit_passed", False),
        ("seed", False),
        ("seed", 0.0),
        ("model_parameters", 28_969_216.0),
        ("optimizer_steps", True),
        ("optimizer_steps", 1_106.0),
    ],
)
def test_admission_rejects_failed_rules_and_numeric_aliases(field, bad_value):
    contract = load_objective_controls_contract(AMENDMENT)
    value = _valid_admission(contract)
    run = _admission_run(value, "full_corpus_dense")
    run[field] = bad_value

    with pytest.raises(MsctlError) as caught:
        validate_objective_controls_admission(contract, value)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


@pytest.mark.parametrize(
    "mutation",
    [
        "at_floor",
        "missing_cell",
        "extra_cell",
        "accuracy_bool",
        "accuracy_integer",
        "control_directional_accuracy",
    ],
)
def test_learnability_floor_applies_only_to_original_six_runs(mutation):
    contract = load_objective_controls_contract(AMENDMENT)
    value = _valid_admission(contract)
    original = _admission_run(value, "full_corpus_dense")
    control = _admission_run(value, "full_corpus_random_fact90")
    if mutation == "at_floor":
        original["primary_cell_accuracy"][PRIMARY_CELLS[0]] = 0.75
    elif mutation == "missing_cell":
        original["primary_cell_accuracy"].pop(PRIMARY_CELLS[0])
    elif mutation == "extra_cell":
        original["primary_cell_accuracy"]["length_ood"] = 0.9
    elif mutation == "accuracy_bool":
        original["primary_cell_accuracy"][PRIMARY_CELLS[0]] = True
    elif mutation == "accuracy_integer":
        original["primary_cell_accuracy"][PRIMARY_CELLS[0]] = 1
    elif mutation == "control_directional_accuracy":
        control["primary_cell_accuracy"] = {
            cell: 0.99 for cell in PRIMARY_CELLS
        }
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)

    with pytest.raises(MsctlError) as caught:
        validate_objective_controls_admission(contract, value)

    assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"


def test_admission_json_loader_rejects_duplicates_and_nonfinite_values(tmp_path):
    contract = load_objective_controls_contract(AMENDMENT)
    value = _valid_admission(contract)
    valid_path = tmp_path / "valid.json"
    _write_json(valid_path, value)
    assert load_objective_controls_admission(valid_path, contract).run_ids == (
        RUN_IDS
    )

    compact = json.dumps(value, separators=(",", ":"), allow_nan=False)
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version":1,' + compact.removeprefix("{"),
        encoding="ascii",
    )
    nonfinite = deepcopy(value)
    _admission_run(
        nonfinite,
        "full_corpus_dense",
    )["checkpoint_resume_max_abs_delta"] = math.nan
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(
        json.dumps(nonfinite, allow_nan=True),
        encoding="ascii",
    )

    for path in (duplicate_path, nonfinite_path):
        with pytest.raises(MsctlError) as caught:
            load_objective_controls_admission(path, contract)
        assert caught.value.code == "OBJECTIVE_CONTROLS_INVALID"
