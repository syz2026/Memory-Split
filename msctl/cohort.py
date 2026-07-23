"""Strict versioned cohort assignment and run-config contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from .errors import MsctlError


COHORT_ID = "memorysplit-confirmatory-v2-360m-n5"
V3_COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
ILLUMINA_PROVIDER = "illumina-usfc-prd"
AWS_PROVIDER = "aws-p5.48xlarge"
MODEL_PARAMETERS = 356_033_536
TARGETS_PER_UPDATE = 524_288
OPTIMIZER_STEPS = 13_582
RAW_TARGET_TOKENS = 7_120_879_616
SEEDS = tuple(range(5))
V3_SEEDS = tuple(range(10))
ARMS = ("dense", "split90")
SNAPSHOT_STEPS = (1_358, 3_396, 6_791, 10_187, 13_582)
_RUN_CONFIG_PATHS = frozenset(
    f"configs/360m-v2/{arm}-s{seed}.yaml"
    for seed in SEEDS
    for arm in ARMS
)
_ASSIGNMENT_FIELDS = {
    "schema_version",
    "cohort_id",
    "model_parameters",
    "optimizer_steps",
    "provider_seeds",
    "raw_target_tokens",
    "targets_per_update",
}
_CONFIG_FIELDS = {
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
_INTEGER_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "ctx",
    "micro_batch_size",
    "tokens_per_step",
    "max_steps",
    "total_tokens",
    "warmup_steps",
    "log_every",
    "eval_every",
    "ckpt_minutes",
}
_FLOAT_CONFIG_FIELDS = {"lr", "weight_decay"}
_ENVIRONMENT_PLACEHOLDER = re.compile(r"\$|%\{|{{")


@dataclass(frozen=True)
class _CohortContract:
    """Immutable identity and path binding for one frozen cohort version."""

    assignment_filename: str
    assignment_schema_version: int
    preregistration_filename: str
    preregistration_id: str
    preregistration_schema_version: int
    config_dir: str
    cohort_id: str
    seeds: tuple[int, ...]
    illumina_seeds: tuple[int, ...]
    aws_p5_seeds: tuple[int, ...]
    run_id_version: str
    train_corpus: str

    @property
    def run_config_paths(self) -> frozenset[str]:
        return frozenset(
            f"configs/{self.config_dir}/{arm}-s{seed}.yaml"
            for seed in self.seeds
            for arm in ARMS
        )


_V2_CONTRACT = _CohortContract(
    assignment_filename="cohort-assignment-v2.json",
    assignment_schema_version=2,
    preregistration_filename="preregistration-v2.yaml",
    preregistration_id="memorysplit-confirmatory-v2",
    preregistration_schema_version=2,
    config_dir="360m-v2",
    cohort_id=COHORT_ID,
    seeds=SEEDS,
    illumina_seeds=(0,),
    aws_p5_seeds=(1, 2, 3, 4),
    run_id_version="v2",
    train_corpus="dataset/corpus-receipt.json",
)
_V3_CONTRACT = _CohortContract(
    assignment_filename="cohort-assignment-v3.json",
    assignment_schema_version=3,
    preregistration_filename="preregistration-v3.yaml",
    preregistration_id="memorysplit-confirmatory-v3",
    preregistration_schema_version=3,
    config_dir="360m-v3",
    cohort_id=V3_COHORT_ID,
    seeds=V3_SEEDS,
    illumina_seeds=(),
    aws_p5_seeds=V3_SEEDS,
    run_id_version="v3",
    train_corpus="dataset/receipt.json",
)
_COHORT_CONTRACTS = (_V2_CONTRACT, _V3_CONTRACT)


@dataclass(frozen=True)
class CohortRunConfig:
    """One hash-bound Dense or Split90 run configuration."""

    path: str
    sha256: str
    run_id: str
    condition: str
    seed: int
    snapshot_steps: tuple[int, ...]


@dataclass(frozen=True)
class CohortAssignment:
    """One frozen provider assignment and all validated paired run configs."""

    cohort_id: str
    model_parameters: int
    targets_per_update: int
    optimizer_steps: int
    raw_target_tokens: int
    illumina_seeds: tuple[int, ...]
    aws_p5_seeds: tuple[int, ...]
    configs: tuple[CohortRunConfig, ...]
    assignment_sha256: str
    preregistration_sha256: str

    @property
    def config_sha256s(self) -> dict[str, str]:
        return {config.path: config.sha256 for config in self.configs}

    def configs_for_provider(self, provider: str) -> tuple[CohortRunConfig, ...]:
        if provider == ILLUMINA_PROVIDER:
            seeds = set(self.illumina_seeds)
        elif provider == AWS_PROVIDER:
            seeds = set(self.aws_p5_seeds)
        else:
            raise MsctlError(
                "COHORT_INVALID",
                "cohort provider is unsupported",
                details={"provider": provider},
            )
        return tuple(config for config in self.configs if config.seed in seeds)


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise MsctlError(
                "COHORT_INVALID",
                "cohort YAML mapping keys must be scalar",
            ) from error
        if duplicate:
            raise MsctlError(
                "COHORT_INVALID",
                "cohort YAML contains a duplicate field",
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _fail(message: str, *, details: dict[str, object] | None = None) -> None:
    raise MsctlError("COHORT_INVALID", message, details=details)


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise MsctlError("COHORT_INVALID", f"{label} cannot be read") from error


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains a duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda _: _fail(f"{label} contains a non-finite number"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "COHORT_INVALID",
            f"{label} must contain one valid UTF-8 JSON value",
        ) from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object")
    return value


def _yaml_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        text = data.decode("utf-8")
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.events.AliasEvent) or getattr(
                event, "anchor", None
            ):
                _fail(f"{label} must not contain YAML anchors or aliases")
        value = yaml.load(text, Loader=_UniqueSafeLoader)
    except MsctlError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise MsctlError(
            "COHORT_INVALID",
            f"{label} must contain one valid UTF-8 YAML object",
        ) from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object with string fields")
    return value


def _require_exact_fields(
    value: dict[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} has missing or unknown fields",
            details={
                "missing": sorted(expected - actual),
                "unknown": sorted(actual - expected),
            },
        )


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _validate_snapshot_steps(
    value: object,
    *,
    label: str,
    max_steps: int,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        _fail(f"{label} must be a list")
    steps = tuple(
        _require_int(step, label=f"{label}[{index}]")
        for index, step in enumerate(value)
    )
    if (
        not steps
        or tuple(sorted(steps)) != steps
        or len(set(steps)) != len(steps)
        or steps[-1] != max_steps
    ):
        _fail(f"{label} must be sorted, unique, and end at max_steps")
    return steps


def _portable_logical_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("~")
        or _ENVIRONMENT_PLACEHOLDER.search(value)
    ):
        _fail(f"{label} must be a resolved portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        _fail(f"{label} must be a resolved portable relative path")
    return path.as_posix()


def _validate_v3_preregistration(
    value: dict[str, object],
    protected: dict[str, object],
    analysis: dict[str, object],
) -> None:
    amendment = {
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
        "claim_bearing_condition_pair": list(ARMS),
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
    aws_topology = {
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
    primary = analysis.get("primary_hypothesis")
    if not isinstance(primary, dict):
        _fail("v3 preregistration primary hypothesis is missing")
    expected_primary_test = {
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
    expected_aulc = {
        "enabled": True,
        "role": "required_secondary_trajectory_evidence",
        "second_primary": False,
        "optimizer_steps": list(SNAPSHOT_STEPS),
        "all_steps_required_for_every_arm_and_seed": True,
        "interpolation": "none",
        "integral": "right_step",
    }
    expected_bootstrap = {
        "bit_generator": "PCG64",
        "rng_seed": 0,
        "draws": 20_000,
        "confidence_interval_percent": 90,
        "resampling_levels": ["seed", "world", "counterfactual_pair"],
    }
    expected_equivalence = {
        "contrast_id": (
            "primary_omnibus_pair_and_proof__graph_non_path__"
            "composition_joint_ood__split90_minus_dense"
        ),
        "margin_absolute_pair_accuracy": 0.01,
        "method": "two_one_sided_90_percent_confidence_bounds",
        "bound_estimator": "hierarchical_bootstrap_seed_world_pair",
        "lower_rule": "strictly_greater_than_negative_margin",
        "upper_rule": "strictly_less_than_positive_margin",
        "boundary_equality": "does_not_support_equivalence",
    }
    diagnostics = value.get("development_diagnostics_29m")
    if not isinstance(diagnostics, dict):
        _fail("v3 preregistration diagnostics are missing")
    if (
        value.get("prospective_amendment") != amendment
        or protected.get("provider_assignment")
        != {AWS_PROVIDER: list(V3_SEEDS)}
        or protected.get("aws_topology") != aws_topology
        or protected.get("extra_trained_control_arms") != []
        or protected.get("terminal_evidence")
        != {
            "paired_bundles_required": 10,
            "all_validity_and_evaluation_evidence_required": True,
        }
        or primary.get("estimand") != "split90_minus_dense"
        or primary.get("timepoint") != "final_optimizer_step"
        or primary.get("optimizer_step") != OPTIMIZER_STEPS
        or primary.get("test") != expected_primary_test
        or analysis.get("fixed_checkpoint_aulc") != expected_aulc
        or analysis.get("hierarchical_bootstrap") != expected_bootstrap
        or analysis.get("practical_equivalence") != expected_equivalence
        or analysis.get("terminal_status_requires")
        != {
            "paired_bundles": 10,
            "all_validity_and_evaluation_evidence": True,
        }
        or diagnostics.get("effect_direction_as_pass_criterion_forbidden")
        is not True
    ):
        _fail("v3 preregistration conflicts with the prospective amendment")


def _validate_preregistration(
    data: bytes,
    contract: _CohortContract,
) -> tuple[str, tuple[int, ...]]:
    value = _yaml_object(data, label="preregistration")
    protected = value.get("protected_cohort")
    if not isinstance(protected, dict):
        _fail("preregistration protected_cohort is missing")
    training = protected.get("training")
    if not isinstance(training, dict):
        _fail("preregistration protected_cohort.training is missing")
    analysis = value.get("analysis")
    if not isinstance(analysis, dict):
        _fail("preregistration analysis is missing")
    fixed_checkpoint_aulc = analysis.get("fixed_checkpoint_aulc")
    if not isinstance(fixed_checkpoint_aulc, dict):
        _fail("preregistration snapshot_steps are missing")
    snapshot_steps = _validate_snapshot_steps(
        fixed_checkpoint_aulc.get("optimizer_steps"),
        label="preregistration snapshot_steps",
        max_steps=OPTIMIZER_STEPS,
    )
    raw_seeds = protected.get("seeds")
    if not isinstance(raw_seeds, list):
        _fail("preregistration protected_cohort.seeds must be a list")
    preregistered_seeds = tuple(
        _require_int(seed, label=f"preregistration seed[{index}]")
        for index, seed in enumerate(raw_seeds)
    )
    for label, candidate in (
        ("model_parameters", protected.get("model_parameters")),
        ("terminal_n_pairs", protected.get("terminal_n_pairs")),
        ("targets_per_update", training.get("targets_per_update")),
        ("optimizer_steps", training.get("optimizer_steps")),
        ("raw_target_tokens", training.get("raw_target_tokens")),
    ):
        _require_int(candidate, label=f"preregistration {label}")
    if snapshot_steps != SNAPSHOT_STEPS:
        _fail(
            "preregistration snapshot_steps do not match the frozen AULC schedule"
        )
    if (
        value.get("schema_version") != contract.preregistration_schema_version
        or value.get("preregistration_id") != contract.preregistration_id
        or protected.get("condition_pair") != list(ARMS)
        or protected.get("model_parameters") != MODEL_PARAMETERS
        or protected.get("terminal_n_pairs") != len(contract.seeds)
        or preregistered_seeds != contract.seeds
        or training.get("targets_per_update") != TARGETS_PER_UPDATE
        or training.get("optimizer_steps") != OPTIMIZER_STEPS
        or training.get("raw_target_tokens") != RAW_TARGET_TOKENS
    ):
        _fail("cohort assignment conflicts with preregistration")
    if contract is _V3_CONTRACT:
        _validate_v3_preregistration(value, protected, analysis)
    return hashlib.sha256(data).hexdigest(), snapshot_steps


def _validate_assignment(
    value: dict[str, object],
    contract: _CohortContract,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _require_exact_fields(value, _ASSIGNMENT_FIELDS, label="cohort assignment")
    for field in (
        "schema_version",
        "model_parameters",
        "optimizer_steps",
        "raw_target_tokens",
        "targets_per_update",
    ):
        _require_int(value[field], label=f"cohort assignment.{field}")
    providers = value["provider_seeds"]
    expected_providers = (
        {ILLUMINA_PROVIDER, AWS_PROVIDER}
        if contract.illumina_seeds
        else {AWS_PROVIDER}
    )
    if not isinstance(providers, dict) or set(providers) != expected_providers:
        _fail("cohort assignment provider_seeds is invalid")

    parsed: dict[str, tuple[int, ...]] = {}
    for provider in sorted(expected_providers):
        raw_seeds = providers[provider]
        if not isinstance(raw_seeds, list):
            _fail("cohort provider seeds must be lists")
        seeds = tuple(
            _require_int(seed, label=f"cohort provider {provider} seed")
            for seed in raw_seeds
        )
        if len(seeds) != len(set(seeds)):
            _fail("cohort provider assignment contains a duplicate seed")
        parsed[provider] = seeds

    illumina = parsed.get(ILLUMINA_PROVIDER, ())
    aws = parsed[AWS_PROVIDER]
    if (
        value["schema_version"] != contract.assignment_schema_version
        or value["cohort_id"] != contract.cohort_id
        or value["model_parameters"] != MODEL_PARAMETERS
        or value["targets_per_update"] != TARGETS_PER_UPDATE
        or value["optimizer_steps"] != OPTIMIZER_STEPS
        or value["raw_target_tokens"] != RAW_TARGET_TOKENS
        or TARGETS_PER_UPDATE * OPTIMIZER_STEPS != RAW_TARGET_TOKENS
        or illumina != contract.illumina_seeds
        or aws != contract.aws_p5_seeds
        or set(illumina) & set(aws)
        or set(illumina + aws) != set(contract.seeds)
    ):
        _fail("cohort assignment does not match its frozen versioned contract")
    return illumina, aws


def _expected_config(
    contract: _CohortContract,
    seed: int,
    arm: str,
    snapshot_steps: tuple[int, ...],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "cohort_id": contract.cohort_id,
        "run_id": (
            f"memorysplit-{contract.run_id_version}-360m-s{seed}-{arm}"
        ),
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": contract.train_corpus,
        "sidecar_name": f"{arm}_target_weights",
        "out_dir": f"runs/seed-{seed}/{arm}",
        "micro_batch_size": 8,
        "tokens_per_step": TARGETS_PER_UPDATE,
        "max_steps": OPTIMIZER_STEPS,
        "total_tokens": RAW_TARGET_TOKENS,
        "lr": 0.001,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snapshot_steps": list(snapshot_steps),
        "ckpt_minutes": 30,
    }


def _load_run_config(
    data: bytes,
    *,
    contract: _CohortContract,
    relative: str,
    seed: int,
    arm: str,
) -> CohortRunConfig:
    value = _yaml_object(data, label=f"run config {relative}")
    _require_exact_fields(value, _CONFIG_FIELDS, label=f"run config {relative}")
    for field in _INTEGER_CONFIG_FIELDS:
        _require_int(value[field], label=f"run config {relative}.{field}")
    for field in _FLOAT_CONFIG_FIELDS:
        if isinstance(value[field], bool) or not isinstance(value[field], float):
            _fail(f"run config {relative}.{field} must be a float")
    if not isinstance(value["compile"], bool):
        _fail(f"run config {relative}.compile must be boolean")
    snapshot_steps = _validate_snapshot_steps(
        value["snapshot_steps"],
        label=f"run config {relative}.snapshot_steps",
        max_steps=value["max_steps"],
    )
    _portable_logical_path(
        value["train_corpus"],
        label=f"run config {relative}.train_corpus",
    )
    _portable_logical_path(value["out_dir"], label=f"run config {relative}.out_dir")
    if value != _expected_config(contract, seed, arm, snapshot_steps):
        _fail(f"run config {relative} violates frozen Dense/Split90 invariants")
    return CohortRunConfig(
        path=relative,
        sha256=hashlib.sha256(data).hexdigest(),
        run_id=str(value["run_id"]),
        condition=arm,
        seed=seed,
        snapshot_steps=snapshot_steps,
    )


def load_cohort_assignment_bytes(
    *,
    assignment_data: bytes,
    preregistration_data: bytes,
    config_data: Mapping[str, bytes],
    assignment_filename: str = "cohort-assignment-v2.json",
) -> CohortAssignment:
    """Validate one immutable byte snapshot of the complete cohort."""

    contract = next(
        (
            candidate
            for candidate in _COHORT_CONTRACTS
            if candidate.assignment_filename == assignment_filename
        ),
        None,
    )
    if contract is None:
        _fail("cohort assignment filename does not identify a frozen contract")
    if type(assignment_data) is not bytes:
        _fail("cohort assignment snapshot must be immutable bytes")
    if type(preregistration_data) is not bytes:
        _fail("preregistration snapshot must be immutable bytes")
    configs = dict(config_data)
    run_config_paths = contract.run_config_paths
    if set(configs) != run_config_paths or not all(
        isinstance(path, str) and type(data) is bytes
        for path, data in configs.items()
    ):
        _fail(
            "cohort config snapshot must contain exactly the frozen cells",
            details={
                "missing": sorted(run_config_paths - set(configs)),
                "unknown": sorted(set(configs) - run_config_paths),
            },
        )
    value = _json_object(assignment_data, label="cohort assignment")
    illumina, aws = _validate_assignment(value, contract)
    preregistration_sha256, preregistered_snapshot_steps = (
        _validate_preregistration(preregistration_data, contract)
    )

    parsed_configs = []
    for seed in contract.seeds:
        for arm in ARMS:
            relative = f"configs/{contract.config_dir}/{arm}-s{seed}.yaml"
            parsed_configs.append(
                _load_run_config(
                    configs[relative],
                    contract=contract,
                    relative=relative,
                    seed=seed,
                    arm=arm,
                )
            )
    if any(
        config.snapshot_steps != preregistered_snapshot_steps
        for config in parsed_configs
    ):
        _fail(
            "run config snapshot_steps must match the preregistered AULC "
            "schedule across all arms and providers"
        )

    return CohortAssignment(
        cohort_id=contract.cohort_id,
        model_parameters=MODEL_PARAMETERS,
        targets_per_update=TARGETS_PER_UPDATE,
        optimizer_steps=OPTIMIZER_STEPS,
        raw_target_tokens=RAW_TARGET_TOKENS,
        illumina_seeds=illumina,
        aws_p5_seeds=aws,
        configs=tuple(parsed_configs),
        assignment_sha256=hashlib.sha256(assignment_data).hexdigest(),
        preregistration_sha256=preregistration_sha256,
    )


def load_cohort_assignment(path: Path | str) -> CohortAssignment:
    """Load and cross-check the canonical assignment, preregistration, and runs."""

    assignment_path = Path(path)
    if assignment_path.parent.name != "configs":
        _fail("cohort assignment must live directly under configs")
    contract = next(
        (
            candidate
            for candidate in _COHORT_CONTRACTS
            if candidate.assignment_filename == assignment_path.name
        ),
        None,
    )
    if contract is None:
        _fail("cohort assignment filename does not identify a frozen contract")
    configs_root = assignment_path.parent
    run_root = configs_root / contract.config_dir
    if run_root.is_symlink() or not run_root.is_dir():
        _fail("cohort run-config directory must be a regular directory")

    run_config_paths = contract.run_config_paths
    expected_names = {PurePosixPath(path).name for path in run_config_paths}
    try:
        actual_names = {entry.name for entry in run_root.iterdir()}
    except OSError as error:
        raise MsctlError(
            "COHORT_INVALID",
            "cohort run-config directory cannot be read",
        ) from error
    if actual_names != expected_names:
        _fail(
            "cohort run configs must contain exactly the frozen Dense/Split90 cells",
            details={
                "missing": sorted(expected_names - actual_names),
                "unknown": sorted(actual_names - expected_names),
            },
        )

    repo_root = configs_root.parent
    config_data = {}
    for relative in sorted(run_config_paths):
        candidate = repo_root / relative
        try:
            candidate.resolve(strict=True).relative_to(repo_root.resolve())
        except (FileNotFoundError, ValueError) as error:
            raise MsctlError(
                "COHORT_INVALID",
                "cohort run config escapes the repository root",
            ) from error
        config_data[relative] = _read_regular(
            candidate,
            label=f"run config {relative}",
        )

    return load_cohort_assignment_bytes(
        assignment_filename=assignment_path.name,
        assignment_data=_read_regular(
            assignment_path,
            label="cohort assignment",
        ),
        preregistration_data=_read_regular(
            configs_root / contract.preregistration_filename,
            label="preregistration",
        ),
        config_data=config_data,
    )
