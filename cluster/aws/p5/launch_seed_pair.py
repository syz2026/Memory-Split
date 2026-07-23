#!/usr/bin/env python3
"""Render and supervise one symmetric Dense/Split90 P5 seed pair."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.interruption_checkpoint import (
    ImdsV2Client,
    InterruptionRequest,
    InterruptionResult,
    S3ObjectStore,
    handle_interruption,
)
from cluster.aws.p5.profile import (
    AwsP5Profile,
    AwsP5Runtime,
    load_aws_p5_profile,
    validate_runtime_environment,
)


PROVIDER = "aws-p5.48xlarge"
COHORT_ID = "memorysplit-confirmatory-v2-360m-n5"
_ARMS = ("dense", "split90")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_H100_RE = re.compile(r"^NVIDIA H100 80GB(?: HBM3)?$")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "provider",
        "cohort_id",
        "seed",
        "profile_sha256",
        "release_sha256",
        "cohort_assignment_sha256",
        "code_commit",
        "bootstrap_receipt",
        "corpus_receipt",
        "runs",
    }
)
_RUN_FIELDS = frozenset(
    {
        "arm",
        "config",
        "config_sha256",
        "master_port",
        "cpu_affinity",
        "data_loader_workers",
        "checkpoint",
        "rank_zero_pid_file",
    }
)
_CONFIG_FIELDS = frozenset(
    {
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
)
_BOOTSTRAP_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "provider",
        "instance_id",
        "instance_type",
        "region",
        "ami_id",
        "container_digest",
        "profile_sha256",
        "release_sha256",
        "cohort_assignment_sha256",
        "corpus_receipt_sha256",
        "code_commit",
        "scratch_root",
        "instance_store",
        "durable_upload_verified",
    }
)


class LaunchError(ValueError):
    """A fail-closed paired-launch validation error."""


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ArmLaunch:
    arm: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path
    config_path: Path
    config_sha256: str
    out_dir: Path
    checkpoint_path: Path
    rank_zero_pid_file: Path
    master_port: int
    cpu_affinity: tuple[int, int]
    data_loader_workers: int


@dataclass(frozen=True)
class LaunchPlan:
    seed: int
    profile: AwsP5Profile
    runtime: AwsP5Runtime
    repo_root: Path
    scratch_root: Path
    manifest_path: Path
    arms: tuple[ArmLaunch, ArmLaunch]
    verified_files: tuple[VerifiedFile, ...]
    release_sha256: str
    corpus_receipt_sha256: str
    code_commit: str


@dataclass(frozen=True)
class SupervisionResult:
    status: str
    returncode: int
    child_pids: Mapping[str, int]
    failed_arm: str | None = None
    peer_terminated: bool = False
    resumable: bool = False
    interruption_receipt: str | None = None


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate_tree(self) -> None: ...


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise LaunchError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LaunchError(f"{label} must be a regular file")
    data = path.read_bytes()
    if len(data) > 16 * 1024 * 1024:
        raise LaunchError(f"{label} is unexpectedly large")
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                LaunchError(f"{label} contains non-finite {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise LaunchError(f"{label} must contain a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    fields: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != fields:
        raise LaunchError(
            f"{label} fields do not match contract; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LaunchError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise LaunchError("code commit must be 40 lowercase hex characters")
    return value


def _hash_regular(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise LaunchError(f"{label} must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_relative(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("~")
        or "$" in value
    ):
        raise LaunchError(f"{label} must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise LaunchError(f"{label} must be a portable relative path")
    return path.as_posix()


def _inside_existing(root: Path, relative: str, *, label: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise LaunchError(f"{label} must not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as error:
        raise LaunchError(f"{label} must remain inside the release root") from error
    return resolved


def _inside_output(root: Path, relative: str, *, label: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise LaunchError(f"{label} must not traverse a symlink")
    try:
        candidate.resolve(strict=False).relative_to(root_resolved)
    except ValueError as error:
        raise LaunchError(f"{label} must remain inside scratch") from error
    return candidate


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise LaunchError("config keys must be strings")
        if key in mapping:
            raise LaunchError(f"config contains duplicate key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _yaml_mapping,
)


def _load_config(path: Path) -> dict[str, object]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_NoDuplicateSafeLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise LaunchError("config must contain valid UTF-8 YAML") from error
    if not isinstance(value, dict):
        raise LaunchError("config must contain a YAML object")
    _exact_fields(value, _CONFIG_FIELDS, label="config")
    return value


def _exact_int(value: object, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise LaunchError(f"{label} must be exactly {expected}")


def _validate_config(
    config: Mapping[str, object],
    *,
    seed: int,
    arm: str,
    corpus_path: str,
) -> str:
    expected = {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "run_id": f"memorysplit-v2-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": corpus_path,
        "sidecar_name": (
            "dense_target_weights"
            if arm == "dense"
            else "split90_target_weights"
        ),
        "out_dir": f"runs/seed-{seed}/{arm}",
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
        "snap_frac": 0.1,
        "ckpt_minutes": 30,
    }
    for name, expected_value in expected.items():
        actual = config[name]
        if isinstance(expected_value, int) and not isinstance(
            expected_value, bool
        ):
            if type(actual) is not int or actual != expected_value:
                raise LaunchError(f"config {name} does not match the cohort")
        elif isinstance(expected_value, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or float(actual) != expected_value
            ):
                raise LaunchError(f"config {name} does not match the cohort")
        elif actual != expected_value or type(actual) is not type(expected_value):
            raise LaunchError(f"config {name} does not match the cohort")
    if config["max_steps"] * config["tokens_per_step"] != config["total_tokens"]:
        raise LaunchError("config token geometry is not integral")
    return _portable_relative(config["out_dir"], label="config out_dir")


def _validate_bootstrap_receipt(
    path: Path,
    *,
    expected_sha256: str,
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
    release_sha256: str,
    cohort_sha256: str,
    corpus_sha256: str,
    code_commit: str,
) -> VerifiedFile:
    digest = _hash_regular(path, label="bootstrap receipt")
    if digest != expected_sha256:
        raise LaunchError("bootstrap receipt SHA-256 mismatch")
    receipt = _load_json(path, label="bootstrap receipt")
    _exact_fields(receipt, _BOOTSTRAP_FIELDS, label="bootstrap receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": "aws-p5-bootstrap",
        "provider": PROVIDER,
        "instance_type": "p5.48xlarge",
        "region": runtime.region,
        "ami_id": runtime.ami_id,
        "container_digest": runtime.container_digest,
        "profile_sha256": profile.sha256,
        "release_sha256": release_sha256,
        "cohort_assignment_sha256": cohort_sha256,
        "corpus_receipt_sha256": corpus_sha256,
        "code_commit": code_commit,
        "scratch_root": profile.scratch_root,
        "durable_upload_verified": True,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise LaunchError(f"bootstrap receipt {key} does not match")
    instance_id = receipt["instance_id"]
    if (
        not isinstance(instance_id, str)
        or re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None
    ):
        raise LaunchError("bootstrap receipt instance ID is invalid")
    store = receipt["instance_store"]
    if store != {
        "device_bytes": profile.instance_store_device_bytes,
        "devices": profile.instance_store_devices,
        "model": profile.instance_store_model,
        "raid_level": profile.raid_level,
    }:
        raise LaunchError("bootstrap receipt instance-store contract does not match")
    return VerifiedFile(path=path, sha256=digest)


def _validate_sidecar_set(
    receipt_root: Path,
    value: object,
    *,
    name: str,
) -> tuple[VerifiedFile, ...]:
    if not isinstance(value, dict):
        raise LaunchError(f"corpus sidecar {name} must be an object")
    _exact_fields(
        value,
        frozenset({"artifacts", "dtype", "items", "ordered_stream_sha256"}),
        label=f"corpus sidecar {name}",
    )
    if value["dtype"] != "uint8":
        raise LaunchError(f"corpus sidecar {name} must use uint8")
    if type(value["items"]) is not int or value["items"] <= 0:
        raise LaunchError(f"corpus sidecar {name} item count is invalid")
    _sha256(
        value["ordered_stream_sha256"],
        label=f"corpus sidecar {name} ordered stream",
    )
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise LaunchError(f"corpus sidecar {name} is missing artifacts")
    verified = []
    total_bytes = 0
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise LaunchError(f"corpus sidecar {name} artifact is invalid")
        _exact_fields(
            artifact,
            frozenset({"path", "bytes", "sha256"}),
            label=f"corpus sidecar {name} artifact",
        )
        relative = _portable_relative(
            artifact["path"], label=f"corpus sidecar {name} artifact path"
        )
        if relative in seen:
            raise LaunchError(f"corpus sidecar {name} repeats an artifact")
        seen.add(relative)
        expected_digest = _sha256(
            artifact["sha256"],
            label=f"corpus sidecar {name} artifact",
        )
        expected_bytes = artifact["bytes"]
        if type(expected_bytes) is not int or expected_bytes <= 0:
            raise LaunchError(f"corpus sidecar {name} artifact size is invalid")
        path = _inside_existing(
            receipt_root,
            relative,
            label=f"corpus sidecar {name} artifact",
        )
        if path.stat().st_size != expected_bytes:
            raise LaunchError(f"corpus sidecar {name} artifact byte mismatch")
        actual_digest = _hash_regular(path, label=f"corpus sidecar {name}")
        if actual_digest != expected_digest:
            raise LaunchError(f"corpus sidecar {name} SHA-256 mismatch")
        total_bytes += expected_bytes
        verified.append(VerifiedFile(path=path, sha256=actual_digest))
    if total_bytes != value["items"]:
        raise LaunchError(f"corpus sidecar {name} is not byte aligned")
    return tuple(verified)


def _validate_corpus_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_ordered_sha256: str,
) -> tuple[VerifiedFile, ...]:
    digest = _hash_regular(path, label="corpus receipt")
    if digest != expected_sha256:
        raise LaunchError("corpus receipt SHA-256 mismatch")
    value = _load_json(path, label="corpus receipt")
    if value.get("format") != "memorysplit-parallel-corpus-v2":
        raise LaunchError("corpus receipt format must be v2")
    if value.get("ordered_stream_sha256") != expected_ordered_sha256:
        raise LaunchError("corpus receipt ordered stream does not match")
    sidecars = value.get("sidecar_sets")
    if not isinstance(sidecars, dict):
        raise LaunchError("corpus receipt is missing sidecar sets")
    required = {"dense_target_weights", "split90_target_weights"}
    if not required <= set(sidecars):
        raise LaunchError("corpus receipt is missing required sidecars")
    files = [VerifiedFile(path=path, sha256=digest)]
    for name in sorted(required):
        files.extend(
            _validate_sidecar_set(path.parent, sidecars[name], name=name)
        )
    return tuple(files)


def _default_instance_type() -> str:
    value = ImdsV2Client().get("meta-data/instance-type")
    if value is None:
        raise LaunchError("IMDSv2 did not return an instance type")
    return value


def _default_gpu_names(environment: Mapping[str, str]) -> tuple[str, ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise LaunchError("nvidia-smi GPU discovery failed")
    return tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )


def _default_port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _arm_by_name(runs: object) -> dict[str, dict[str, object]]:
    if not isinstance(runs, list) or len(runs) != 2:
        raise LaunchError("run manifest must contain one complete pair")
    result = {}
    for run in runs:
        if not isinstance(run, dict):
            raise LaunchError("run manifest pair entries must be objects")
        _exact_fields(run, _RUN_FIELDS, label="run manifest entry")
        arm = run["arm"]
        if arm not in _ARMS or arm in result:
            raise LaunchError("run manifest must contain a unique explicit pair")
        result[arm] = run
    if set(result) != set(_ARMS):
        raise LaunchError("run manifest must contain Dense and Split90 pair")
    return result


def load_launch_plan(
    *,
    seed: int,
    manifest_path: Path | str,
    profile_path: Path | str,
    repo_root: Path | str,
    scratch_root: Path | str,
    environment: Mapping[str, str],
    observed_instance_type: str | None = None,
    gpu_names: Sequence[str] | None = None,
    port_available: Callable[[int], bool] = _default_port_available,
) -> LaunchPlan:
    """Validate all trust roots and return an immutable paired launch plan."""

    profile = load_aws_p5_profile(profile_path)
    try:
        runtime = validate_runtime_environment(profile, environment)
    except ValueError as error:
        raise LaunchError(str(error)) from error
    if type(seed) is not int or seed not in profile.assigned_seeds:
        raise LaunchError("seed must be assigned to AWS: one of 1, 2, 3, 4")
    actual_instance_type = (
        _default_instance_type()
        if observed_instance_type is None
        else observed_instance_type
    )
    if actual_instance_type != "p5.48xlarge":
        raise LaunchError("launch requires an actual p5.48xlarge instance")
    safe_environment = {
        name: environment[name]
        for name in profile.process_env_allowlist
        if name in environment
    }
    actual_gpu_names = (
        _default_gpu_names(safe_environment)
        if gpu_names is None
        else tuple(gpu_names)
    )
    if len(actual_gpu_names) != 8:
        raise LaunchError("launch requires exactly eight H100 devices")
    if any(_H100_RE.fullmatch(name) is None for name in actual_gpu_names):
        raise LaunchError("launch requires eight NVIDIA H100 80GB devices")

    repo = Path(repo_root).resolve(strict=True)
    scratch = Path(scratch_root).resolve(strict=True)
    manifest_file = Path(manifest_path)
    manifest_digest = _hash_regular(manifest_file, label="run manifest")
    manifest = _load_json(manifest_file, label="run manifest")
    _exact_fields(manifest, _MANIFEST_FIELDS, label="run manifest")
    _exact_int(manifest["schema_version"], 1, label="manifest schema version")
    if manifest["provider"] != PROVIDER:
        raise LaunchError("run manifest provider must be aws-p5.48xlarge")
    if manifest["cohort_id"] != COHORT_ID:
        raise LaunchError("run manifest cohort ID does not match")
    if type(manifest["seed"]) is not int or manifest["seed"] != seed:
        raise LaunchError("run manifest seed does not match assigned seed")
    profile_sha256 = _sha256(
        manifest["profile_sha256"], label="manifest profile"
    )
    if profile_sha256 != profile.sha256:
        raise LaunchError("manifest profile SHA-256 does not match")
    release_sha256 = _sha256(
        manifest["release_sha256"], label="manifest release"
    )
    cohort_sha256 = _sha256(
        manifest["cohort_assignment_sha256"],
        label="manifest cohort assignment",
    )
    code_commit = _commit(manifest["code_commit"])

    corpus_binding = manifest["corpus_receipt"]
    if not isinstance(corpus_binding, dict):
        raise LaunchError("manifest corpus receipt binding must be an object")
    _exact_fields(
        corpus_binding,
        frozenset({"path", "sha256", "ordered_stream_sha256"}),
        label="manifest corpus receipt binding",
    )
    corpus_relative = _portable_relative(
        corpus_binding["path"], label="manifest corpus receipt path"
    )
    corpus_sha256 = _sha256(
        corpus_binding["sha256"], label="manifest corpus receipt"
    )
    ordered_sha256 = _sha256(
        corpus_binding["ordered_stream_sha256"],
        label="manifest ordered stream",
    )
    corpus_path = _inside_existing(
        scratch, corpus_relative, label="corpus receipt"
    )
    verified_files = [
        VerifiedFile(path=manifest_file, sha256=manifest_digest),
        *_validate_corpus_receipt(
            corpus_path,
            expected_sha256=corpus_sha256,
            expected_ordered_sha256=ordered_sha256,
        ),
    ]

    bootstrap_binding = manifest["bootstrap_receipt"]
    if not isinstance(bootstrap_binding, dict):
        raise LaunchError("manifest bootstrap receipt binding must be an object")
    _exact_fields(
        bootstrap_binding,
        frozenset({"path", "sha256"}),
        label="manifest bootstrap receipt binding",
    )
    bootstrap_relative = _portable_relative(
        bootstrap_binding["path"], label="bootstrap receipt path"
    )
    bootstrap_path = _inside_existing(
        scratch, bootstrap_relative, label="bootstrap receipt"
    )
    verified_files.append(
        _validate_bootstrap_receipt(
            bootstrap_path,
            expected_sha256=_sha256(
                bootstrap_binding["sha256"], label="manifest bootstrap receipt"
            ),
            profile=profile,
            runtime=runtime,
            release_sha256=release_sha256,
            cohort_sha256=cohort_sha256,
            corpus_sha256=corpus_sha256,
            code_commit=code_commit,
        )
    )

    runs = _arm_by_name(manifest["runs"])
    parsed_runs = []
    ports = []
    worker_budgets = []
    for arm, expected_port_offset, expected_affinity in (
        ("dense", 0, (0, 95)),
        ("split90", 1, (96, 191)),
    ):
        run = runs[arm]
        expected_config = f"configs/360m-v2/{arm}-s{seed}.yaml"
        config_relative = _portable_relative(
            run["config"], label=f"{arm} config path"
        )
        if config_relative != expected_config:
            raise LaunchError(f"{arm} config path does not match assigned seed")
        config_path = _inside_existing(
            repo, config_relative, label=f"{arm} config"
        )
        config_digest = _hash_regular(config_path, label=f"{arm} config")
        if config_digest != _sha256(
            run["config_sha256"], label=f"{arm} config"
        ):
            raise LaunchError(f"{arm} config SHA-256 mismatch")
        config = _load_config(config_path)
        out_relative = _validate_config(
            config,
            seed=seed,
            arm=arm,
            corpus_path=corpus_relative,
        )
        out_dir = _inside_output(
            scratch, out_relative, label=f"{arm} output"
        )
        if out_dir.exists() or out_dir.is_symlink():
            raise LaunchError(f"{arm} output already exists")

        port = run["master_port"]
        if type(port) is not int or port < 1024 or port > 65_535:
            raise LaunchError(f"{arm} master port is invalid")
        expected_port = 29_500 + seed * 2 + expected_port_offset
        if port != expected_port:
            raise LaunchError(f"{arm} master port does not match dense schedule")
        ports.append(port)

        affinity = run["cpu_affinity"]
        if (
            not isinstance(affinity, list)
            or len(affinity) != 2
            or any(type(item) is not int for item in affinity)
            or tuple(affinity) != expected_affinity
        ):
            raise LaunchError(f"{arm} CPU affinity must use one P5 CPU half")
        workers = run["data_loader_workers"]
        if type(workers) is not int or workers <= 0 or workers > 48:
            raise LaunchError(f"{arm} data-loader worker budget is invalid")
        worker_budgets.append(workers)

        checkpoint_relative = _portable_relative(
            run["checkpoint"], label=f"{arm} checkpoint path"
        )
        pid_relative = _portable_relative(
            run["rank_zero_pid_file"],
            label=f"{arm} rank-zero PID path",
        )
        checkpoint_path = _inside_output(
            scratch, checkpoint_relative, label=f"{arm} checkpoint"
        )
        pid_path = _inside_output(
            scratch, pid_relative, label=f"{arm} rank-zero PID file"
        )
        for child, label in (
            (checkpoint_path, "checkpoint"),
            (pid_path, "rank-zero PID file"),
        ):
            try:
                child.relative_to(out_dir)
            except ValueError as error:
                raise LaunchError(
                    f"{arm} {label} must be inside its output directory"
                ) from error

        child_environment = dict(safe_environment)
        child_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": (
                    "0,1,2,3" if arm == "dense" else "4,5,6,7"
                ),
                "MS_DATA_ROOT": str(scratch / "dataset"),
                "MS_DATA_LOADER_WORKERS": str(workers),
                "MS_RANK_ZERO_PID_FILE": str(pid_path),
                "MS_RUN_ROOT": str(scratch),
                "OMP_NUM_THREADS": "1",
                "PYTHONPATH": str(repo),
                "PYTHONUNBUFFERED": "1",
            }
        )
        parsed_runs.append(
            ArmLaunch(
                arm=arm,
                argv=(
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=4",
                    f"--master_port={port}",
                    "scripts/run_train.py",
                    "--config",
                    config_relative,
                ),
                environment=child_environment,
                cwd=repo,
                config_path=config_path,
                config_sha256=config_digest,
                out_dir=out_dir,
                checkpoint_path=checkpoint_path,
                rank_zero_pid_file=pid_path,
                master_port=port,
                cpu_affinity=tuple(affinity),
                data_loader_workers=workers,
            )
        )
        verified_files.append(
            VerifiedFile(path=config_path, sha256=config_digest)
        )

    if ports[0] == ports[1]:
        raise LaunchError("Dense and Split90 master ports must differ")
    for port in ports:
        if not port_available(port):
            raise LaunchError(f"master port {port} is occupied")
    if worker_budgets[0] != worker_budgets[1]:
        raise LaunchError("paired data-loader worker budgets must be equal")

    return LaunchPlan(
        seed=seed,
        profile=profile,
        runtime=runtime,
        repo_root=repo,
        scratch_root=scratch,
        manifest_path=manifest_file,
        arms=(parsed_runs[0], parsed_runs[1]),
        verified_files=tuple(verified_files),
        release_sha256=release_sha256,
        corpus_receipt_sha256=corpus_sha256,
        code_commit=code_commit,
    )


def render_plan(plan: LaunchPlan) -> dict[str, object]:
    """Return the exact dry-run plan without spawning or writing."""

    return {
        "commands": [
            {
                "arm": launch.arm,
                "argv": list(launch.argv),
                "cpu_affinity": list(launch.cpu_affinity),
                "data_loader_workers": launch.data_loader_workers,
                "env": dict(sorted(launch.environment.items())),
                "out_dir": str(launch.out_dir),
            }
            for launch in plan.arms
        ],
        "dry_run": True,
        "ok": True,
        "provider": PROVIDER,
        "schema_version": 1,
        "seed": plan.seed,
    }


class _SubprocessHandle:
    def __init__(self, process: subprocess.Popen, log_handle) -> None:
        self._process = process
        self._log_handle = log_handle
        self.pid = process.pid

    def poll(self) -> int | None:
        result = self._process.poll()
        if result is not None and not self._log_handle.closed:
            self._log_handle.close()
        return result

    def wait(self, timeout: float | None = None) -> int:
        result = self._process.wait(timeout=timeout)
        if not self._log_handle.closed:
            self._log_handle.close()
        return result

    def terminate_tree(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self._process.pid, signal.SIGKILL)
            self._process.wait(timeout=10)
        finally:
            if not self._log_handle.closed:
                self._log_handle.close()


def _spawn_process(launch: ArmLaunch) -> ProcessHandle:
    log_path = launch.out_dir / "launcher.log"
    log_handle = log_path.open("xb")

    def set_affinity() -> None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU affinity is unavailable on this platform")
        start, end = launch.cpu_affinity
        os.sched_setaffinity(0, set(range(start, end + 1)))

    try:
        process = subprocess.Popen(
            list(launch.argv),
            cwd=launch.cwd,
            env=dict(launch.environment),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=set_affinity,
        )
    except BaseException:
        log_handle.close()
        raise
    return _SubprocessHandle(process, log_handle)


def _revalidate_files(plan: LaunchPlan) -> None:
    for item in plan.verified_files:
        if _hash_regular(item.path, label="launch input") != item.sha256:
            raise LaunchError(f"launch input changed after planning: {item.path}")


def _safe_terminate(process: ProcessHandle) -> None:
    try:
        process.terminate_tree()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def _acquire_host_lock(scratch_root: Path):
    lock_path = scratch_root / ".p5-seed-pair.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise LaunchError("P5 seed-pair lock is unsafe or unavailable") from error
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LaunchError("P5 seed-pair lock must be a regular file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise LaunchError(
            "another seed pair is already active on this P5"
        ) from error
    except BaseException:
        handle.close()
        raise
    return handle


def supervise_pair(
    plan: LaunchPlan,
    *,
    spawner: Callable[[ArmLaunch], ProcessHandle] = _spawn_process,
    sleep: Callable[[float], None] = time.sleep,
    notice_source: Callable[[], str | None] | None = None,
    interruption_handler: Callable[
        [LaunchPlan, Mapping[str, int], str], InterruptionResult
    ]
    | None = None,
) -> SupervisionResult:
    """Hold the host-wide lock while supervising exactly one seed pair."""

    lock = _acquire_host_lock(plan.scratch_root)
    try:
        return _supervise_pair_locked(
            plan,
            spawner=spawner,
            sleep=sleep,
            notice_source=notice_source,
            interruption_handler=interruption_handler,
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _supervise_pair_locked(
    plan: LaunchPlan,
    *,
    spawner: Callable[[ArmLaunch], ProcessHandle] = _spawn_process,
    sleep: Callable[[float], None] = time.sleep,
    notice_source: Callable[[], str | None] | None = None,
    interruption_handler: Callable[
        [LaunchPlan, Mapping[str, int], str], InterruptionResult
    ]
    | None = None,
) -> SupervisionResult:
    """Launch both arms, then accept only paired zero exit status."""

    _revalidate_files(plan)
    processes: dict[str, ProcessHandle] = {}
    try:
        for launch in plan.arms:
            try:
                launch.out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            except OSError as error:
                raise LaunchError(
                    f"{launch.arm} output became stale before launch"
                ) from error
            os.chmod(launch.out_dir, 0o700)
        for launch in plan.arms:
            try:
                processes[launch.arm] = spawner(launch)
            except BaseException as error:
                for process in processes.values():
                    _safe_terminate(process)
                raise LaunchError(
                    f"failed to spawn {launch.arm} process group"
                ) from error
        child_pids = {arm: process.pid for arm, process in processes.items()}
        if set(child_pids) != set(_ARMS):
            raise LaunchError("both child PIDs must be recorded before supervision")

        while True:
            if notice_source is not None:
                try:
                    notice = notice_source()
                except Exception as error:
                    for process in processes.values():
                        _safe_terminate(process)
                    raise LaunchError(
                        "interruption notice polling failed; pair was stopped"
                    ) from error
                if notice is not None:
                    if interruption_handler is None:
                        for process in processes.values():
                            _safe_terminate(process)
                        return SupervisionResult(
                            status="interrupted",
                            returncode=74,
                            child_pids=child_pids,
                        )
                    try:
                        interruption = interruption_handler(
                            plan, child_pids, notice
                        )
                    except Exception as error:
                        for process in processes.values():
                            _safe_terminate(process)
                        raise LaunchError(
                            "interruption handling failed; pair was stopped"
                        ) from error
                    for process in processes.values():
                        _safe_terminate(process)
                    return SupervisionResult(
                        status="interrupted",
                        returncode=interruption.exit_code,
                        child_pids=child_pids,
                        resumable=interruption.resumable,
                        interruption_receipt=str(interruption.receipt_path),
                    )

            statuses = {
                arm: process.poll() for arm, process in processes.items()
            }
            failed = next(
                (
                    arm
                    for arm in _ARMS
                    if statuses[arm] is not None and statuses[arm] != 0
                ),
                None,
            )
            if failed is not None:
                peer_terminated = False
                for arm, process in processes.items():
                    if arm != failed and statuses[arm] is None:
                        _safe_terminate(process)
                        peer_terminated = True
                return SupervisionResult(
                    status="failed",
                    returncode=int(statuses[failed]),
                    child_pids=child_pids,
                    failed_arm=failed,
                    peer_terminated=peer_terminated,
                )
            if all(status == 0 for status in statuses.values()):
                return SupervisionResult(
                    status="completed",
                    returncode=0,
                    child_pids=child_pids,
                )
            sleep(0.25)
    except KeyboardInterrupt:
        for process in processes.values():
            _safe_terminate(process)
        return SupervisionResult(
            status="interrupted",
            returncode=130,
            child_pids={
                arm: process.pid for arm, process in processes.items()
            },
        )


def _read_rank_zero_pid(
    path: Path,
    *,
    supervisor_pid: int,
    deadline: float,
) -> int:
    while time.monotonic() <= deadline:
        if not path.is_symlink() and path.is_file():
            text = path.read_text(encoding="ascii").strip()
            if text.isdigit():
                pid = int(text)
                if pid > 1 and _is_descendant(pid, supervisor_pid):
                    return pid
        time.sleep(0.25)
    raise LaunchError("rank-zero PID file was not produced by torchrun")


def _is_descendant(pid: int, ancestor: int) -> bool:
    current = pid
    seen = set()
    while current > 1 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text().split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def _production_interruption_handler(
    plan: LaunchPlan,
    child_pids: Mapping[str, int],
    notice: str,
) -> InterruptionResult:
    deadline = time.monotonic() + 30.0
    rank_zero_pids = {
        launch.arm: _read_rank_zero_pid(
            launch.rank_zero_pid_file,
            supervisor_pid=child_pids[launch.arm],
            deadline=deadline,
        )
        for launch in plan.arms
    }
    store = S3ObjectStore(
        region=plan.runtime.region,
        environment={
            name: plan.arms[0].environment[name]
            for name in ("AWS_REGION", "HOME", "PATH")
            if name in plan.arms[0].environment
        },
    )
    request = InterruptionRequest(
        seed=plan.seed,
        notice=notice,
        rank_zero_pids=rank_zero_pids,
        checkpoint_paths={
            launch.arm: launch.checkpoint_path for launch in plan.arms
        },
        s3_root=plan.runtime.s3_root,
        receipt_path=(
            plan.scratch_root
            / "staging"
            / f"interruption-seed-{plan.seed}.json"
        ),
        release_sha256=plan.release_sha256,
        corpus_receipt_sha256=plan.corpus_receipt_sha256,
        code_commit=plan.code_commit,
        config_sha256={
            launch.arm: launch.config_sha256 for launch in plan.arms
        },
        timeout_seconds=120.0,
    )
    return handle_interruption(request, object_store=store)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=root / "cluster" / "profiles" / "aws-p5.48xlarge.json",
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--scratch-root", type=Path, default=Path("/mnt/memorysplit"))
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        plan = load_launch_plan(
            seed=arguments.seed,
            manifest_path=arguments.manifest,
            profile_path=arguments.profile,
            repo_root=arguments.repo_root,
            scratch_root=arguments.scratch_root,
            environment=os.environ,
        )
        if not arguments.apply:
            report = render_plan(plan)
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 0
        client = ImdsV2Client()
        result = supervise_pair(
            plan,
            notice_source=client.interruption_notice,
            interruption_handler=_production_interruption_handler,
        )
        report = {
            "child_pids": dict(sorted(result.child_pids.items())),
            "dry_run": False,
            "failed_arm": result.failed_arm,
            "interruption_receipt": result.interruption_receipt,
            "ok": result.returncode == 0,
            "peer_terminated": result.peer_terminated,
            "resumable": result.resumable,
            "returncode": result.returncode,
            "schema_version": 1,
            "seed": plan.seed,
            "status": result.status,
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return result.returncode
    except (LaunchError, ValueError, OSError) as error:
        report = {
            "error": str(error),
            "ok": False,
            "schema_version": 1,
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
