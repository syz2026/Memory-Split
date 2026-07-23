from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from msctl.cohort import load_cohort_assignment
from msctl.errors import MsctlError


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = REPO_ROOT / "configs" / "cohort-assignment-v2.json"
CONFIG_DIR = REPO_ROOT / "configs" / "360m-v2"
ARMS = ("dense", "split90")
SEEDS = tuple(range(5))
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
    "snap_frac",
    "ckpt_minutes",
}
PAIR_VARIANT_FIELDS = {
    "run_id",
    "condition",
    "seed",
    "out_dir",
    "sidecar_name",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    configs = root / "configs"
    shutil.copytree(CONFIG_DIR, configs / "360m-v2")
    shutil.copy2(
        REPO_ROOT / "configs" / "preregistration-v2.yaml",
        configs / "preregistration-v2.yaml",
    )
    shutil.copy2(ASSIGNMENT, configs / ASSIGNMENT.name)
    return root, configs / ASSIGNMENT.name


def _json_value(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _yaml_value(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text())
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_frozen_cohort_is_disjoint_complete_and_hash_bound():
    cohort = load_cohort_assignment(ASSIGNMENT)

    assert cohort.cohort_id == "memorysplit-confirmatory-v2-360m-n5"
    assert cohort.illumina_seeds == (0,)
    assert cohort.aws_p5_seeds == (1, 2, 3, 4)
    assert set(cohort.illumina_seeds).isdisjoint(cohort.aws_p5_seeds)
    assert set(cohort.illumina_seeds + cohort.aws_p5_seeds) == set(SEEDS)
    assert cohort.model_parameters == 356_033_536
    assert cohort.targets_per_update * cohort.optimizer_steps == 7_120_879_616
    assert cohort.raw_target_tokens == 7_120_879_616
    assert cohort.preregistration_sha256 == _sha256(
        REPO_ROOT / "configs" / "preregistration-v2.yaml"
    )

    assert {(config.seed, config.condition) for config in cohort.configs} == {
        (seed, arm) for seed in SEEDS for arm in ARMS
    }
    assert len(cohort.configs) == 10
    expected_paths = {
        f"configs/360m-v2/{arm}-s{seed}.yaml"
        for seed in SEEDS
        for arm in ARMS
    }
    assert set(cohort.config_sha256s) == expected_paths
    for relative, digest in cohort.config_sha256s.items():
        assert digest == _sha256(REPO_ROOT / relative)


def test_contracts_exports_cohort_assignment_loader():
    from msctl.contracts import load_cohort_assignment as contracts_loader

    assert contracts_loader is load_cohort_assignment
    assert contracts_loader(ASSIGNMENT) == load_cohort_assignment(ASSIGNMENT)


def test_assignment_is_canonical_sorted_json_with_trailing_newline():
    value = _json_value(ASSIGNMENT)

    assert ASSIGNMENT.read_bytes() == (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def test_run_configs_are_exact_matched_pairs_with_logical_paths():
    cohort = load_cohort_assignment(ASSIGNMENT)

    by_cell = {
        (config.seed, config.condition): config for config in cohort.configs
    }
    for seed in SEEDS:
        dense = _yaml_value(REPO_ROOT / by_cell[(seed, "dense")].path)
        split90 = _yaml_value(REPO_ROOT / by_cell[(seed, "split90")].path)
        assert set(dense) == set(split90) == CONFIG_FIELDS
        assert {
            key: value
            for key, value in dense.items()
            if key not in PAIR_VARIANT_FIELDS
        } == {
            key: value
            for key, value in split90.items()
            if key not in PAIR_VARIANT_FIELDS
        }
        for arm, value in (("dense", dense), ("split90", split90)):
            assert value["condition"] == arm
            assert value["seed"] == seed
            assert value["run_id"] == f"memorysplit-v2-360m-s{seed}-{arm}"
            assert value["sidecar_name"] == f"{arm}_target_weights"
            assert value["out_dir"] == f"runs/seed-{seed}/{arm}"
            assert value["train_corpus"] == "dataset/corpus-receipt.json"
            assert value["model"] == "d360m"
            assert value["tokens_per_step"] == 524_288
            assert value["max_steps"] == 13_582
            assert value["total_tokens"] == 7_120_879_616
            assert "$" not in str(value)
            assert ".." not in Path(str(value["out_dir"])).parts


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_seed",
        "missing_seed",
        "seed_zero_in_aws",
        "wrong_model_size",
        "bool_integer",
        "unknown_assignment_field",
    ],
)
def test_assignment_rejects_malformed_cohorts(tmp_path, mutation):
    _, assignment = _fixture_repo(tmp_path)
    value = _json_value(assignment)
    providers = value["provider_seeds"]
    assert isinstance(providers, dict)

    if mutation == "duplicate_seed":
        providers["aws-p5.48xlarge"] = [1, 2, 2, 3, 4]
    elif mutation == "missing_seed":
        providers["aws-p5.48xlarge"] = [1, 2, 3]
    elif mutation == "seed_zero_in_aws":
        providers["aws-p5.48xlarge"] = [0, 1, 2, 3, 4]
        providers["illumina-usfc-prd"] = []
    elif mutation == "wrong_model_size":
        value["model_parameters"] = 356_033_535
    elif mutation == "bool_integer":
        value["optimizer_steps"] = True
    elif mutation == "unknown_assignment_field":
        value["config_dir"] = "configs/360m-v2"
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)
    _write_json(assignment, value)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


@pytest.mark.parametrize(
    "invalid_seed",
    [False, 0.0],
    ids=["bool", "float"],
)
def test_assignment_rejects_non_integer_preregistration_seed(
    tmp_path,
    invalid_seed,
):
    root, assignment = _fixture_repo(tmp_path)
    preregistration = root / "configs" / "preregistration-v2.yaml"
    value = _yaml_value(preregistration)
    protected = value["protected_cohort"]
    assert isinstance(protected, dict)
    seeds = protected["seeds"]
    assert isinstance(seeds, list)
    seeds[0] = invalid_seed
    _write_yaml(preregistration, value)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


@pytest.mark.parametrize(
    ("mutation", "filename"),
    [
        ("generic_split", "split90-s0.yaml"),
        ("wrong_token_count", "dense-s0.yaml"),
        ("mismatched_pair", "split90-s0.yaml"),
        ("bool_integer", "dense-s0.yaml"),
        ("unknown_field", "dense-s0.yaml"),
        ("field_alias", "dense-s0.yaml"),
        ("environment_placeholder", "dense-s0.yaml"),
        ("unsafe_relative_path", "dense-s0.yaml"),
    ],
)
def test_assignment_rejects_malformed_run_configs(tmp_path, mutation, filename):
    root, assignment = _fixture_repo(tmp_path)
    config_path = root / "configs" / "360m-v2" / filename
    value = _yaml_value(config_path)

    if mutation == "generic_split":
        value["condition"] = "split"
    elif mutation == "wrong_token_count":
        value["total_tokens"] = 7_120_879_615
    elif mutation == "mismatched_pair":
        value["lr"] = 0.002
    elif mutation == "bool_integer":
        value["seed"] = True
    elif mutation == "unknown_field":
        value["notes"] = "not part of the contract"
    elif mutation == "field_alias":
        value["arm"] = value.pop("condition")
    elif mutation == "environment_placeholder":
        value["out_dir"] = "${RUN_ROOT}/dense"
    elif mutation == "unsafe_relative_path":
        value["train_corpus"] = "../dataset/corpus-receipt.json"
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)
    _write_yaml(config_path, value)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


def test_assignment_rejects_partial_pair(tmp_path):
    root, assignment = _fixture_repo(tmp_path)
    (root / "configs" / "360m-v2" / "split90-s4.yaml").unlink()

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)


@pytest.mark.parametrize(
    "yaml_text",
    [
        "schema_version: 2\nschema_version: 2\n",
        "schema_version: &version 2\ncohort_id: *version\n",
    ],
)
def test_assignment_rejects_yaml_duplicates_and_aliases(tmp_path, yaml_text):
    root, assignment = _fixture_repo(tmp_path)
    (root / "configs" / "360m-v2" / "dense-s0.yaml").write_text(yaml_text)

    with pytest.raises(MsctlError):
        load_cohort_assignment(assignment)
