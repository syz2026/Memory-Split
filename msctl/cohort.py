"""Strict five-seed cohort assignment and run-config contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from .errors import MsctlError


COHORT_ID = "memorysplit-confirmatory-v2-360m-n5"
ILLUMINA_PROVIDER = "illumina-usfc-prd"
AWS_PROVIDER = "aws-p5.48xlarge"
MODEL_PARAMETERS = 356_033_536
TARGETS_PER_UPDATE = 524_288
OPTIMIZER_STEPS = 13_582
RAW_TARGET_TOKENS = 7_120_879_616
SEEDS = tuple(range(5))
ARMS = ("dense", "split90")
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
    "snap_frac",
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
_FLOAT_CONFIG_FIELDS = {"lr", "weight_decay", "snap_frac"}
_ENVIRONMENT_PLACEHOLDER = re.compile(r"\$|%\{|{{")


@dataclass(frozen=True)
class CohortRunConfig:
    """One hash-bound Dense or Split90 run configuration."""

    path: str
    sha256: str
    run_id: str
    condition: str
    seed: int


@dataclass(frozen=True)
class CohortAssignment:
    """The frozen provider assignment and its ten validated run configs."""

    cohort_id: str
    model_parameters: int
    targets_per_update: int
    optimizer_steps: int
    raw_target_tokens: int
    illumina_seeds: tuple[int, ...]
    aws_p5_seeds: tuple[int, ...]
    configs: tuple[CohortRunConfig, ...]
    assignment_sha256: str

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


def _validate_preregistration(path: Path) -> None:
    value = _yaml_object(
        _read_regular(path, label="preregistration"),
        label="preregistration",
    )
    protected = value.get("protected_cohort")
    if not isinstance(protected, dict):
        _fail("preregistration protected_cohort is missing")
    training = protected.get("training")
    if not isinstance(training, dict):
        _fail("preregistration protected_cohort.training is missing")
    for label, candidate in (
        ("model_parameters", protected.get("model_parameters")),
        ("terminal_n_pairs", protected.get("terminal_n_pairs")),
        ("targets_per_update", training.get("targets_per_update")),
        ("optimizer_steps", training.get("optimizer_steps")),
        ("raw_target_tokens", training.get("raw_target_tokens")),
    ):
        _require_int(candidate, label=f"preregistration {label}")
    if (
        value.get("schema_version") != 2
        or protected.get("condition_pair") != list(ARMS)
        or protected.get("model_parameters") != MODEL_PARAMETERS
        or protected.get("terminal_n_pairs") != len(SEEDS)
        or protected.get("seeds") != list(SEEDS)
        or training.get("targets_per_update") != TARGETS_PER_UPDATE
        or training.get("optimizer_steps") != OPTIMIZER_STEPS
        or training.get("raw_target_tokens") != RAW_TARGET_TOKENS
    ):
        _fail("cohort assignment conflicts with preregistration")


def _validate_assignment(
    value: dict[str, object],
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
    if not isinstance(providers, dict) or set(providers) != {
        ILLUMINA_PROVIDER,
        AWS_PROVIDER,
    }:
        _fail("cohort assignment provider_seeds is invalid")

    parsed: dict[str, tuple[int, ...]] = {}
    for provider in (ILLUMINA_PROVIDER, AWS_PROVIDER):
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

    illumina = parsed[ILLUMINA_PROVIDER]
    aws = parsed[AWS_PROVIDER]
    if (
        value["schema_version"] != 2
        or value["cohort_id"] != COHORT_ID
        or value["model_parameters"] != MODEL_PARAMETERS
        or value["targets_per_update"] != TARGETS_PER_UPDATE
        or value["optimizer_steps"] != OPTIMIZER_STEPS
        or value["raw_target_tokens"] != RAW_TARGET_TOKENS
        or TARGETS_PER_UPDATE * OPTIMIZER_STEPS != RAW_TARGET_TOKENS
        or illumina != (0,)
        or aws != (1, 2, 3, 4)
        or set(illumina) & set(aws)
        or set(illumina + aws) != set(SEEDS)
    ):
        _fail("cohort assignment does not match the frozen five-seed contract")
    return illumina, aws


def _expected_config(seed: int, arm: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "run_id": f"memorysplit-v2-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": "dataset/corpus-receipt.json",
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
        "snap_frac": 0.1,
        "ckpt_minutes": 30,
    }


def _load_run_config(
    path: Path,
    *,
    relative: str,
    seed: int,
    arm: str,
) -> CohortRunConfig:
    data = _read_regular(path, label=f"run config {relative}")
    value = _yaml_object(data, label=f"run config {relative}")
    _require_exact_fields(value, _CONFIG_FIELDS, label=f"run config {relative}")
    for field in _INTEGER_CONFIG_FIELDS:
        _require_int(value[field], label=f"run config {relative}.{field}")
    for field in _FLOAT_CONFIG_FIELDS:
        if isinstance(value[field], bool) or not isinstance(value[field], float):
            _fail(f"run config {relative}.{field} must be a float")
    if not isinstance(value["compile"], bool):
        _fail(f"run config {relative}.compile must be boolean")
    _portable_logical_path(
        value["train_corpus"],
        label=f"run config {relative}.train_corpus",
    )
    _portable_logical_path(value["out_dir"], label=f"run config {relative}.out_dir")
    if value != _expected_config(seed, arm):
        _fail(f"run config {relative} violates frozen Dense/Split90 invariants")
    return CohortRunConfig(
        path=relative,
        sha256=hashlib.sha256(data).hexdigest(),
        run_id=str(value["run_id"]),
        condition=arm,
        seed=seed,
    )


def load_cohort_assignment(path: Path | str) -> CohortAssignment:
    """Load and cross-check the canonical assignment, preregistration, and runs."""

    assignment_path = Path(path)
    assignment_data = _read_regular(
        assignment_path,
        label="cohort assignment",
    )
    if assignment_path.parent.name != "configs":
        _fail("cohort assignment must live directly under configs")
    configs_root = assignment_path.parent
    run_root = configs_root / "360m-v2"
    if run_root.is_symlink() or not run_root.is_dir():
        _fail("cohort run-config directory must be a regular directory")

    value = _json_object(assignment_data, label="cohort assignment")
    illumina, aws = _validate_assignment(value)
    _validate_preregistration(configs_root / "preregistration-v2.yaml")

    expected_names = {
        f"{arm}-s{seed}.yaml" for seed in SEEDS for arm in ARMS
    }
    try:
        actual_names = {entry.name for entry in run_root.iterdir()}
    except OSError as error:
        raise MsctlError(
            "COHORT_INVALID",
            "cohort run-config directory cannot be read",
        ) from error
    if actual_names != expected_names:
        _fail(
            "cohort run configs must contain exactly ten Dense/Split90 cells",
            details={
                "missing": sorted(expected_names - actual_names),
                "unknown": sorted(actual_names - expected_names),
            },
        )

    repo_root = configs_root.parent
    configs = []
    for seed in SEEDS:
        for arm in ARMS:
            filename = f"{arm}-s{seed}.yaml"
            relative = f"configs/360m-v2/{filename}"
            candidate = run_root / filename
            try:
                candidate.resolve(strict=True).relative_to(repo_root.resolve())
            except (FileNotFoundError, ValueError) as error:
                raise MsctlError(
                    "COHORT_INVALID",
                    "cohort run config escapes the repository root",
                ) from error
            configs.append(
                _load_run_config(
                    candidate,
                    relative=relative,
                    seed=seed,
                    arm=arm,
                )
            )

    return CohortAssignment(
        cohort_id=COHORT_ID,
        model_parameters=MODEL_PARAMETERS,
        targets_per_update=TARGETS_PER_UPDATE,
        optimizer_steps=OPTIMIZER_STEPS,
        raw_target_tokens=RAW_TARGET_TOKENS,
        illumina_seeds=illumina,
        aws_p5_seeds=aws,
        configs=tuple(configs),
        assignment_sha256=hashlib.sha256(assignment_data).hexdigest(),
    )
