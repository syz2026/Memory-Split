#!/usr/bin/env python3
"""Render and supervise one symmetric Dense/Split90 P5 seed pair."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping, Protocol, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.interruption_checkpoint import (
    CommandResult,
    ImdsV2Client,
    InterruptionRequest,
    InterruptionResult,
    S3ObjectStore,
    handle_interruption,
)
from cluster.aws.p5.corpus_contract import (
    CorpusContractError,
    verify_canonical_corpus,
)
from cluster.aws.p5.profile import (
    AwsP5Profile,
    AwsP5Runtime,
    load_aws_p5_profile,
    validate_runtime_environment,
)
from msctl.aws_contracts import (
    ARMS,
    COHORT_ASSIGNMENT_PATH,
    COHORT_ID,
    CONFIG_ROOT,
    DATASET_POINTER_PATH,
    DATASET_RECEIPT_PATH,
    EXPECTED_CONFIG_PATHS,
    PACKAGE_FORMAT_VERSION,
    PROFILE_PATH as PROFILE_MEMBER_PATH,
    PROVIDER,
    SEEDS,
    SNAPSHOT_STEPS,
)


_ARMS = ARMS
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_H100_RE = re.compile(r"^NVIDIA H100 80GB(?: HBM3)?$")
_CONTAINER_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$"
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "provider",
        "cohort_id",
        "seed",
        "profile_sha256",
        "release_sha256",
        "release_members_sha256",
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
        "snapshot_steps",
        "ckpt_minutes",
    }
)
_BOOTSTRAP_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "provider",
        "account_id",
        "instance_id",
        "boot_id",
        "instance_type",
        "region",
        "ami_id",
        "container_digest",
        "container_image",
        "profile_sha256",
        "release_sha256",
        "release_members_sha256",
        "release_root",
        "cohort_assignment_sha256",
        "corpus_receipt_sha256",
        "corpus_build_id",
        "corpus_ordered_stream_sha256",
        "code_commit",
        "scratch_root",
        "runtime_uid",
        "runtime_gid",
        "role_name",
        "role_arn",
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
    runtime_config: Mapping[str, object]
    runtime_config_bytes: bytes
    runtime_config_path: Path
    runtime_config_sha256: str
    scientific_config_sha256: str
    out_dir: Path
    checkpoint_path: Path
    rank_zero_pid_file: Path
    cidfile_path: Path
    container_name: str
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
    release_members_sha256: str
    corpus_receipt_sha256: str
    code_commit: str
    container_image: str
    runtime_uid: int
    runtime_gid: int


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

    def kill_tree(self) -> None: ...

    def pin_container(self, timeout: float) -> None: ...

    def stop_container(self, timeout: float) -> None: ...

    def kill_container(self, timeout: float) -> None: ...

    def container_stopped(self, timeout: float) -> bool: ...


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


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _validate_release_root(
    repo: Path,
    scratch: Path,
    *,
    release_sha256: str,
    release_members_sha256: str,
    profile_sha256: str,
    cohort_sha256: str,
    code_commit: str,
) -> tuple[VerifiedFile, ...]:
    expected_root = scratch / "releases" / release_sha256
    if repo != expected_root.resolve(strict=True):
        raise LaunchError(
            "release root must be the digest-named root under scratch"
        )
    entries = [repo, *repo.rglob("*")]
    for path in entries:
        if path.is_symlink():
            raise LaunchError("verified release must not contain symlinks")
        metadata = path.stat()
        if metadata.st_mode & 0o222:
            raise LaunchError("verified release root must be read-only")
        if path.is_file() and (
            not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        ):
            raise LaunchError(
                "verified release members must be singly linked regular files"
            )
        if not path.is_file() and not path.is_dir():
            raise LaunchError("verified release contains a special entry")

    sums_path = repo / "SHA256SUMS"
    metadata_path = repo / "RELEASE-METADATA.json"
    if (
        sums_path.is_symlink()
        or not sums_path.is_file()
        or metadata_path.is_symlink()
        or not metadata_path.is_file()
    ):
        raise LaunchError("verified release is missing its member manifests")
    sums_bytes = sums_path.read_bytes()
    if hashlib.sha256(sums_bytes).hexdigest() != release_members_sha256:
        raise LaunchError("release member manifest SHA-256 mismatch")
    try:
        sums_text = sums_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise LaunchError("release SHA256SUMS must be ASCII") from error
    if not sums_text or not sums_text.endswith("\n"):
        raise LaunchError("release SHA256SUMS must be newline-terminated")
    checksums: dict[str, str] = {}
    paths: list[str] = []
    for line in sums_text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise LaunchError("release SHA256SUMS line is malformed")
        digest = _sha256(line[:64], label="release member")
        relative = _portable_relative(
            line[66:], label="release member path"
        )
        if relative in checksums:
            raise LaunchError("release SHA256SUMS repeats a member")
        checksums[relative] = digest
        paths.append(relative)
    if paths != sorted(paths):
        raise LaunchError("release SHA256SUMS members must be sorted")
    actual_files = {
        path.relative_to(repo).as_posix()
        for path in repo.rglob("*")
        if path.is_file()
    }
    if actual_files != set(checksums) | {"SHA256SUMS"}:
        raise LaunchError("release member namespace contains extras or omissions")
    verified = [
        VerifiedFile(path=sums_path, sha256=release_members_sha256)
    ]
    for relative, expected_digest in checksums.items():
        path = _inside_existing(repo, relative, label="release member")
        digest = _hash_regular(path, label=f"release member {relative}")
        if digest != expected_digest:
            raise LaunchError(f"release member SHA-256 mismatch: {relative}")
        verified.append(VerifiedFile(path=path, sha256=digest))

    metadata_bytes = metadata_path.read_bytes()
    metadata = _load_json(metadata_path, label="release metadata")
    _exact_fields(
        metadata,
        frozenset(
            {
                "schema_version",
                "package_format_version",
                "provider",
                "source",
                "seed_assignment",
                "cohort_assignment",
                "profile",
                "environment",
                "dataset_pointer",
                "config_sha256",
                "members",
            }
        ),
        label="release metadata",
    )
    source = metadata["source"]
    if not isinstance(source, dict):
        raise LaunchError("release metadata source must be an object")
    _exact_fields(
        source,
        frozenset({"commit", "tree", "dirty"}),
        label="release source",
    )
    if (
        _canonical_pretty(metadata) != metadata_bytes
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or type(metadata.get("package_format_version")) is not int
        or metadata.get("package_format_version") != PACKAGE_FORMAT_VERSION
        or metadata.get("provider") != PROVIDER
        or source["commit"] != code_commit
        or source["dirty"] is not False
        or not isinstance(source["tree"], str)
        or _COMMIT_RE.fullmatch(source["tree"]) is None
        or metadata.get("seed_assignment")
        != {
            "arms": list(ARMS),
            "cohort_id": COHORT_ID,
            "provider": PROVIDER,
            "seeds": list(SEEDS),
        }
    ):
        raise LaunchError("release metadata identity does not match")
    rows = metadata.get("members")
    if not isinstance(rows, list):
        raise LaunchError("release metadata member list is missing")
    row_paths: list[str] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"bytes", "git_blob", "git_mode", "path", "sha256"}
        ):
            raise LaunchError("release metadata member row is invalid")
        relative = _portable_relative(
            row["path"], label="release metadata member"
        )
        row_paths.append(relative)
        path = repo / relative
        if (
            relative not in checksums
            or relative in {"RELEASE-METADATA.json", "SHA256SUMS"}
            or type(row["bytes"]) is not int
            or row["bytes"] != path.stat().st_size
            or row["sha256"] != checksums[relative]
            or row["git_mode"] not in {"100644", "100755"}
            or not isinstance(row["git_blob"], str)
            or re.fullmatch(
                r"(?:[0-9a-f]{40}|[0-9a-f]{64})", row["git_blob"]
            )
            is None
        ):
            raise LaunchError(
                f"release metadata member binding drift: {relative}"
            )
    if (
        row_paths != sorted(row_paths)
        or set(row_paths)
        != set(checksums) - {"RELEASE-METADATA.json"}
    ):
        raise LaunchError("release metadata does not bind every member")

    member_sha256 = {row["path"]: row["sha256"] for row in rows}

    def metadata_binding(
        name: str,
        expected_path: str,
    ) -> tuple[str, str]:
        binding = metadata[name]
        if not isinstance(binding, dict):
            raise LaunchError(f"release {name} binding must be an object")
        _exact_fields(
            binding,
            frozenset({"path", "sha256"}),
            label=f"release {name} binding",
        )
        relative = _portable_relative(
            binding["path"],
            label=f"release {name} path",
        )
        digest = _sha256(
            binding["sha256"],
            label=f"release {name}",
        )
        if (
            relative != expected_path
            or member_sha256.get(relative) != digest
        ):
            raise LaunchError(f"release {name} member binding does not match")
        return relative, digest

    _cohort_path, metadata_cohort_sha256 = metadata_binding(
        "cohort_assignment",
        COHORT_ASSIGNMENT_PATH,
    )
    _profile_path, metadata_profile_sha256 = metadata_binding(
        "profile",
        PROFILE_MEMBER_PATH,
    )
    _pointer_path, metadata_pointer_sha256 = metadata_binding(
        "dataset_pointer",
        DATASET_POINTER_PATH,
    )
    if (
        metadata_cohort_sha256 != cohort_sha256
        or metadata_profile_sha256 != profile_sha256
    ):
        raise LaunchError(
            "release profile or cohort binding does not match the manifest"
        )
    config_sha256 = metadata["config_sha256"]
    if not isinstance(config_sha256, dict) or set(config_sha256) != set(
        EXPECTED_CONFIG_PATHS
    ):
        raise LaunchError("release config namespace does not match exact v3")
    for relative, digest in config_sha256.items():
        expected_digest = _sha256(digest, label=f"release config {relative}")
        if member_sha256.get(relative) != expected_digest:
            raise LaunchError(
                f"release config member binding does not match: {relative}"
            )
    from msctl.contracts import validate_runtime_attested_contract
    from msctl.errors import MsctlError

    try:
        validate_runtime_attested_contract(
            metadata["environment"],
            profile_sha256=metadata_profile_sha256,
        )
    except MsctlError as error:
        raise LaunchError(
            "release runtime-attestation contract is invalid"
        ) from error

    assignment = _load_json(
        _inside_existing(
            repo,
            COHORT_ASSIGNMENT_PATH,
            label="cohort assignment",
        ),
        label="cohort assignment",
    )
    _exact_fields(
        assignment,
        frozenset(
            {
                "schema_version",
                "cohort_id",
                "model_parameters",
                "optimizer_steps",
                "provider_seeds",
                "raw_target_tokens",
                "targets_per_update",
            }
        ),
        label="cohort assignment",
    )
    provider_seeds = assignment["provider_seeds"]
    if (
        type(assignment["schema_version"]) is not int
        or assignment["schema_version"] != 3
        or assignment["cohort_id"] != COHORT_ID
        or type(assignment["model_parameters"]) is not int
        or assignment["model_parameters"] != 356_033_536
        or type(assignment["optimizer_steps"]) is not int
        or assignment["optimizer_steps"] != 13_582
        or type(assignment["raw_target_tokens"]) is not int
        or assignment["raw_target_tokens"] != 7_120_879_616
        or type(assignment["targets_per_update"]) is not int
        or assignment["targets_per_update"] != 524_288
        or not isinstance(provider_seeds, dict)
        or set(provider_seeds) != {PROVIDER}
        or provider_seeds[PROVIDER] != list(SEEDS)
        or any(type(seed) is not int for seed in provider_seeds[PROVIDER])
    ):
        raise LaunchError(
            "cohort assignment must contain AWS-only seeds 0 through 9"
        )
    pointer = _load_json(
        _inside_existing(
            repo,
            DATASET_POINTER_PATH,
            label="dataset pointer",
        ),
        label="dataset pointer",
    )
    if (
        pointer.get("provider") != PROVIDER
        or pointer.get("required_receipt") != DATASET_RECEIPT_PATH
        or pointer.get("full_corpus_in_release") is not False
        or _hash_regular(
            repo / DATASET_POINTER_PATH,
            label="dataset pointer",
        )
        != metadata_pointer_sha256
    ):
        raise LaunchError("release dataset pointer identity does not match")
    return tuple(verified)


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


def _load_config(path: Path) -> dict[str, object]:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise LaunchError("config must contain valid UTF-8 YAML") from error
    value: dict[str, object] = {}
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_index += 1
        if not line or line.isspace():
            continue
        if line.startswith((" ", "\t", "#", "---", "...")) or ":" not in line:
            raise LaunchError("config must use strict flat YAML")
        key, raw_value = line.split(":", 1)
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
            or key in value
        ):
            raise LaunchError("config contains an invalid or duplicate key")
        scalar = raw_value.strip()
        if key == "snapshot_steps":
            if scalar == "[1358, 3396, 6791, 10187, 13582]":
                value[key] = list(SNAPSHOT_STEPS)
                continue
            if scalar:
                raise LaunchError(
                    "config snapshot_steps must use the exact frozen "
                    "integer list"
                )
            for expected_step in SNAPSHOT_STEPS:
                expected_line = f"- {expected_step}"
                if (
                    line_index >= len(lines)
                    or lines[line_index] != expected_line
                ):
                    raise LaunchError(
                        "config snapshot_steps must use the exact frozen "
                        "integer list"
                    )
                line_index += 1
            value[key] = list(SNAPSHOT_STEPS)
            continue
        if (
            not scalar
            or scalar[0] in "[{&*!|>@`"
            or " #" in scalar
            or "\t" in scalar
        ):
            raise LaunchError("config contains an unsupported YAML scalar")
        if scalar == "true":
            parsed: object = True
        elif scalar == "false":
            parsed = False
        elif re.fullmatch(r"-?(?:0|[1-9][0-9]*)", scalar):
            parsed = int(scalar)
        elif re.fullmatch(
            r"-?(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?",
            scalar,
        ):
            parsed = float(scalar)
        elif scalar.startswith('"') and scalar.endswith('"'):
            try:
                parsed = json.loads(scalar)
            except json.JSONDecodeError as error:
                raise LaunchError("config quoted scalar is invalid") from error
            if not isinstance(parsed, str):
                raise LaunchError("config quoted scalar must be a string")
        elif scalar.startswith("'") and scalar.endswith("'"):
            parsed = scalar[1:-1].replace("''", "'")
        elif re.fullmatch(r"[A-Za-z0-9_./-]+", scalar):
            parsed = scalar
        else:
            raise LaunchError("config contains an unsupported YAML scalar")
        value[key] = parsed
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
        "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
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
        "snapshot_steps": list(SNAPSHOT_STEPS),
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
    release_members_sha256: str,
    cohort_sha256: str,
    corpus_sha256: str,
    corpus_build_id: str,
    corpus_ordered_stream_sha256: str,
    code_commit: str,
    observed_instance_id: str,
    observed_boot_id: str,
) -> tuple[VerifiedFile, Mapping[str, object]]:
    digest = _hash_regular(path, label="bootstrap receipt")
    if digest != expected_sha256:
        raise LaunchError("bootstrap receipt SHA-256 mismatch")
    receipt = _load_json(path, label="bootstrap receipt")
    _exact_fields(receipt, _BOOTSTRAP_FIELDS, label="bootstrap receipt")
    expected = {
        "schema_version": 2,
        "receipt_type": "aws-p5-bootstrap",
        "provider": PROVIDER,
        "instance_type": "p5.48xlarge",
        "region": runtime.region,
        "ami_id": runtime.ami_id,
        "container_digest": runtime.container_digest,
        "profile_sha256": profile.sha256,
        "release_sha256": release_sha256,
        "release_members_sha256": release_members_sha256,
        "release_root": f"releases/{release_sha256}",
        "cohort_assignment_sha256": cohort_sha256,
        "corpus_receipt_sha256": corpus_sha256,
        "corpus_build_id": corpus_build_id,
        "corpus_ordered_stream_sha256": corpus_ordered_stream_sha256,
        "code_commit": code_commit,
        "instance_id": observed_instance_id,
        "boot_id": observed_boot_id,
        "scratch_root": profile.scratch_root,
        "runtime_uid": runtime.uid,
        "runtime_gid": runtime.gid,
        "durable_upload_verified": True,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            if key == "release_members_sha256":
                raise LaunchError(
                    "bootstrap receipt release member SHA-256 does not match"
                )
            raise LaunchError(f"bootstrap receipt {key} does not match")
    instance_id = receipt["instance_id"]
    if (
        not isinstance(instance_id, str)
        or re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None
    ):
        raise LaunchError("bootstrap receipt instance ID is invalid")
    if (
        not isinstance(receipt["boot_id"], str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            receipt["boot_id"],
        )
        is None
    ):
        raise LaunchError("bootstrap receipt boot ID is invalid")
    store = receipt["instance_store"]
    if store != {
        "device_bytes": profile.instance_store_device_bytes,
        "devices": profile.instance_store_devices,
        "model": profile.instance_store_model,
        "raid_level": profile.raid_level,
    }:
        raise LaunchError("bootstrap receipt instance-store contract does not match")
    container_image = receipt["container_image"]
    if (
        not isinstance(container_image, str)
        or _CONTAINER_IMAGE_RE.fullmatch(container_image) is None
        or not container_image.endswith("@" + runtime.container_digest)
    ):
        raise LaunchError("bootstrap receipt container image is not digest pinned")
    for name in ("runtime_uid", "runtime_gid"):
        if type(receipt[name]) is not int or receipt[name] <= 0:
            raise LaunchError(
                "bootstrap receipt runtime UID/GID must be explicitly non-root"
            )
    account_id = receipt["account_id"]
    role_name = receipt["role_name"]
    role_arn = receipt["role_arn"]
    if (
        not isinstance(account_id, str)
        or re.fullmatch(r"[0-9]{12}", account_id) is None
        or not isinstance(role_name, str)
        or re.fullmatch(r"[A-Za-z0-9+=,.@_-]{1,64}", role_name) is None
        or not isinstance(role_arn, str)
        or not role_arn.startswith(
            f"arn:aws:sts::{account_id}:assumed-role/{role_name}/"
        )
    ):
        raise LaunchError("bootstrap receipt instance-role identity is invalid")
    return VerifiedFile(path=path, sha256=digest), receipt


def _default_instance_type() -> str:
    value = ImdsV2Client().get("meta-data/instance-type")
    if value is None:
        raise LaunchError("IMDSv2 did not return an instance type")
    return value


def _default_instance_id() -> str:
    value = ImdsV2Client().get("meta-data/instance-id")
    if value is None:
        raise LaunchError("IMDSv2 did not return an instance ID")
    return value


def _default_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeDecodeError) as error:
        raise LaunchError("kernel boot ID is unavailable") from error


def _default_gpu_names(environment: Mapping[str, str]) -> tuple[str, ...]:
    try:
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
            timeout=30.0,
        )
    except subprocess.TimeoutExpired as error:
        raise LaunchError("nvidia-smi GPU discovery timed out") from error
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
    observed_instance_id: str | None = None,
    observed_boot_id: str | None = None,
    gpu_names: Sequence[str] | None = None,
    port_available: Callable[[int], bool] = _default_port_available,
    semantic_corpus_verifier: Callable[
        [Path], Mapping[str, object]
    ]
    | None = None,
    enforce_profile_scratch: bool = True,
) -> LaunchPlan:
    """Validate all trust roots and return an immutable paired launch plan."""

    profile = load_aws_p5_profile(profile_path)
    if (
        profile.profile_id != "aws-p5.48xlarge-v3"
        or profile.assigned_seeds != SEEDS
    ):
        raise LaunchError("launch requires the exact v3 AWS P5 profile")
    try:
        runtime = validate_runtime_environment(profile, environment)
    except ValueError as error:
        raise LaunchError(str(error)) from error
    if type(seed) is not int or seed not in SEEDS:
        raise LaunchError("seed must be assigned to AWS: one of 0 through 9")
    actual_instance_type = (
        _default_instance_type()
        if observed_instance_type is None
        else observed_instance_type
    )
    if actual_instance_type != "p5.48xlarge":
        raise LaunchError("launch requires an actual p5.48xlarge instance")
    actual_instance_id = (
        _default_instance_id()
        if observed_instance_id is None
        else observed_instance_id
    )
    if re.fullmatch(r"i-[0-9a-f]{8,17}", actual_instance_id) is None:
        raise LaunchError("observed instance ID is invalid")
    actual_boot_id = (
        _default_boot_id() if observed_boot_id is None else observed_boot_id
    )
    if (
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            actual_boot_id,
        )
        is None
    ):
        raise LaunchError("observed boot ID is invalid")
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
    if not isinstance(enforce_profile_scratch, bool):
        raise LaunchError("scratch enforcement flag must be boolean")
    if enforce_profile_scratch and scratch != Path(
        os.path.abspath(profile.scratch_root)
    ):
        raise LaunchError(
            "scratch root must be exactly the profile /mnt/memorysplit path"
        )
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
    release_members_sha256 = _sha256(
        manifest["release_members_sha256"],
        label="manifest release members",
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
        frozenset(
            {"path", "sha256", "build_id", "ordered_stream_sha256"}
        ),
        label="manifest corpus receipt binding",
    )
    corpus_relative = _portable_relative(
        corpus_binding["path"], label="manifest corpus receipt path"
    )
    if corpus_relative != DATASET_RECEIPT_PATH:
        raise LaunchError(
            "manifest corpus receipt path must be dataset/receipt.json"
        )
    corpus_sha256 = _sha256(
        corpus_binding["sha256"], label="manifest corpus receipt"
    )
    corpus_build_id = _sha256(
        corpus_binding["build_id"],
        label="manifest corpus build ID",
    )
    ordered_sha256 = _sha256(
        corpus_binding["ordered_stream_sha256"],
        label="manifest ordered stream",
    )
    corpus_path = _inside_existing(
        scratch, corpus_relative, label="corpus receipt"
    )
    try:
        corpus_evidence = verify_canonical_corpus(
            corpus_path,
            expected_sha256=corpus_sha256,
            expected_ordered_sha256=ordered_sha256,
            semantic_verifier=semantic_corpus_verifier,
        )
    except CorpusContractError as error:
        raise LaunchError(str(error)) from error
    if corpus_evidence.receipt.get("build_id") != corpus_build_id:
        raise LaunchError("manifest corpus build ID does not match receipt")
    verified_files = [
        VerifiedFile(path=manifest_file, sha256=manifest_digest),
        *(
            VerifiedFile(path=item.path, sha256=item.sha256)
            for item in corpus_evidence.files
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
    bootstrap_file, bootstrap_receipt = _validate_bootstrap_receipt(
        bootstrap_path,
        expected_sha256=_sha256(
            bootstrap_binding["sha256"], label="manifest bootstrap receipt"
        ),
        profile=profile,
        runtime=runtime,
        release_sha256=release_sha256,
        release_members_sha256=release_members_sha256,
        cohort_sha256=cohort_sha256,
        corpus_sha256=corpus_sha256,
        corpus_build_id=corpus_build_id,
        corpus_ordered_stream_sha256=ordered_sha256,
        code_commit=code_commit,
        observed_instance_id=actual_instance_id,
        observed_boot_id=actual_boot_id,
    )
    verified_files.append(bootstrap_file)
    verified_files.extend(
        _validate_release_root(
            repo,
            scratch,
            release_sha256=release_sha256,
            release_members_sha256=release_members_sha256,
            profile_sha256=profile_sha256,
            cohort_sha256=cohort_sha256,
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
        expected_config = f"{CONFIG_ROOT}/{arm}-s{seed}.yaml"
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
            corpus_path=DATASET_RECEIPT_PATH,
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

        runtime_config = dict(config)
        runtime_config["train_corpus"] = "/dataset"
        runtime_config["out_dir"] = "/output/run"
        runtime_config_bytes = (
            json.dumps(
                runtime_config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        runtime_config_sha256 = hashlib.sha256(
            runtime_config_bytes
        ).hexdigest()
        scientific_config = {
            key: value
            for key, value in config.items()
            if key not in {"train_corpus", "out_dir"}
        }
        scientific_config_sha256 = hashlib.sha256(
            (
                json.dumps(
                    scientific_config,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        runtime_config_path = (
            scratch
            / "staging"
            / "runtime-configs"
            / f"seed-{seed}"
            / f"{arm}.json"
        )
        gpu_ids = "0,1,2,3" if arm == "dense" else "4,5,6,7"
        container_name = f"memorysplit-s{seed}-{arm}"
        cidfile_path = (
            scratch
            / "staging"
            / "container-cids"
            / f"seed-{seed}"
            / f"{arm}.cid"
        )
        container_image = str(bootstrap_receipt["container_image"])
        runtime_uid = int(bootstrap_receipt["runtime_uid"])
        runtime_gid = int(bootstrap_receipt["runtime_gid"])
        mount_values = {
            "release": str(repo),
            "dataset": str(scratch / "dataset"),
            "config": str(runtime_config_path),
            "output": str(out_dir),
        }
        if any(
            any(character in value for character in ",\n\r\x00")
            for value in mount_values.values()
        ):
            raise LaunchError("container bind-mount path contains unsafe characters")
        child_environment = {"PATH": "/usr/bin:/bin"}
        container_argv = (
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--cidfile",
            str(cidfile_path),
            "--read-only",
            "--network=host",
            "--ipc=host",
            "--pid=host",
            "--user",
            f"{runtime_uid}:{runtime_gid}",
            "--workdir",
            "/workspace",
            "--cpuset-cpus",
            f"{affinity[0]}-{affinity[1]}",
            "--gpus",
            f"device={gpu_ids}",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "4096",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=4g",
            "--mount",
            f"type=bind,src={mount_values['release']},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={mount_values['dataset']},dst=/dataset,readonly",
            "--mount",
            (
                f"type=bind,src={mount_values['config']},"
                "dst=/runtime/config.yaml,readonly"
            ),
            "--mount",
            f"type=bind,src={mount_values['output']},dst=/output",
            "--env",
            "HOME=/tmp/home",
            "--env",
            f"MS_DATA_LOADER_WORKERS={workers}",
            "--env",
            "MS_RANK_ZERO_PID_FILE=/output/rank-zero.pid",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            container_image,
            "/opt/conda/bin/python",
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=4",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint=127.0.0.1:{port}",
            "/workspace/scripts/run_train.py",
            "--config",
            "/runtime/config.yaml",
            "--resume",
            "none",
        )
        parsed_runs.append(
            ArmLaunch(
                arm=arm,
                argv=container_argv,
                environment=child_environment,
                cwd=scratch,
                config_path=config_path,
                config_sha256=config_digest,
                runtime_config=runtime_config,
                runtime_config_bytes=runtime_config_bytes,
                runtime_config_path=runtime_config_path,
                runtime_config_sha256=runtime_config_sha256,
                scientific_config_sha256=scientific_config_sha256,
                out_dir=out_dir,
                checkpoint_path=checkpoint_path,
                rank_zero_pid_file=pid_path,
                cidfile_path=cidfile_path,
                container_name=container_name,
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
        release_members_sha256=release_members_sha256,
        corpus_receipt_sha256=corpus_sha256,
        code_commit=code_commit,
        container_image=str(bootstrap_receipt["container_image"]),
        runtime_uid=int(bootstrap_receipt["runtime_uid"]),
        runtime_gid=int(bootstrap_receipt["runtime_gid"]),
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
                "runtime_config": dict(launch.runtime_config),
                "runtime_config_path": str(launch.runtime_config_path),
                "runtime_config_sha256": launch.runtime_config_sha256,
                "scientific_config_sha256": launch.scientific_config_sha256,
            }
            for launch in plan.arms
        ],
        "dry_run": True,
        "ok": True,
        "provider": PROVIDER,
        "schema_version": 1,
        "seed": plan.seed,
    }


_TRAINER_CONTRACT_FIELDS = frozenset(
    {
        "rank_zero_pid_file",
        "receipt_v2",
        "resume_sha256",
        "sidecar_name",
        "sigusr1_checkpoint",
    }
)


def render_trainer_preflight(plan: LaunchPlan) -> tuple[str, ...]:
    """Render the no-GPU contract probe in the same immutable image and release."""

    release = str(plan.repo_root)
    if any(character in release for character in ",\n\r\x00"):
        raise LaunchError("trainer preflight release path is unsafe")
    return (
        "docker",
        "run",
        "--rm",
        "--read-only",
        "--network=none",
        "--ipc=none",
        "--user",
        f"{plan.runtime_uid}:{plan.runtime_gid}",
        "--workdir",
        "/workspace",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",
        "--mount",
        f"type=bind,src={release},dst=/workspace,readonly",
        "--env",
        "HOME=/tmp/home",
        plan.container_image,
        "/opt/conda/bin/python",
        "/workspace/scripts/run_train.py",
        "--capabilities-json",
    )


def _run_trainer_preflight(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise LaunchError("trainer contract preflight timed out") from error
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def preflight_trainer_contract(
    plan: LaunchPlan,
    *,
    runner: Callable[
        [Sequence[str], Mapping[str, str], float], CommandResult
    ] = _run_trainer_preflight,
    timeout_seconds: float = 120.0,
) -> None:
    """Fail before launch unless the integrated trainer exposes every P5 hook."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < float(timeout_seconds) <= 300
    ):
        raise LaunchError("trainer contract timeout must be between 0 and 300 seconds")
    result = runner(
        render_trainer_preflight(plan),
        {"PATH": "/usr/bin:/bin"},
        float(timeout_seconds),
    )
    if result.returncode != 0:
        raise LaunchError("trainer contract preflight failed inside pinned container")
    if result.stderr:
        raise LaunchError("trainer contract preflight wrote unexpected stderr")
    try:
        contract = json.loads(
            result.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite capability value: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise LaunchError("trainer contract preflight returned invalid JSON") from error
    if (
        not isinstance(contract, dict)
        or set(contract) != _TRAINER_CONTRACT_FIELDS
        or any(type(value) is not bool for value in contract.values())
    ):
        raise LaunchError("trainer contract preflight returned invalid evidence")
    canonical = (
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    if result.stdout != canonical:
        raise LaunchError("trainer contract preflight returned noncanonical JSON")
    missing = sorted(name for name, supported in contract.items() if not supported)
    if missing:
        labels = {
            "rank_zero_pid_file": "rank-zero PID file",
            "receipt_v2": "memorysplit-parallel-corpus-v2 receipt",
            "resume_sha256": "explicit resume SHA",
            "sidecar_name": "sidecar_name",
            "sigusr1_checkpoint": "SIGUSR1 checkpoint",
        }
        raise LaunchError(
            "trainer contract is unavailable: "
            + ", ".join(labels[name] for name in missing)
        )


class _SubprocessHandle:
    def __init__(
        self,
        process: subprocess.Popen,
        log_handle,
        container: "_DockerContainerHandle",
    ) -> None:
        self._process = process
        self._log_handle = log_handle
        self._container = container
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
        except ProcessLookupError:
            pass

    def kill_tree(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def stop_container(self, timeout: float) -> None:
        self._container.stop_container(timeout)

    def pin_container(self, timeout: float) -> None:
        self._container.pin_container(timeout)

    def kill_container(self, timeout: float) -> None:
        self._container.kill_container(timeout)

    def container_stopped(self, timeout: float) -> bool:
        return self._container.container_stopped(timeout)


def _run_container_command(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise LaunchError("container cleanup command timed out") from error
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class _DockerContainerHandle:
    def __init__(
        self,
        cidfile_path: Path,
        *,
        runner: Callable[
            [Sequence[str], Mapping[str, str], float], CommandResult
        ] = _run_container_command,
    ) -> None:
        self._cidfile_path = Path(cidfile_path)
        self._runner = runner
        self._container_id: str | None = None

    def _read_id(self, timeout: float) -> str:
        if self._container_id is not None:
            return self._container_id
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise LaunchError("container cleanup timeout must be positive")
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self._cidfile_path, flags)
            except FileNotFoundError:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            except OSError as error:
                raise LaunchError("Docker cidfile is unsafe") from error
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_size > 65
                ):
                    raise LaunchError("Docker cidfile is unsafe")
                raw = os.read(descriptor, 66)
            finally:
                os.close(descriptor)
            try:
                container_id = raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise LaunchError("Docker cidfile is invalid") from error
            if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
                raise LaunchError("Docker cidfile must contain one full container ID")
            self._container_id = container_id
            return container_id
        raise LaunchError("Docker cidfile was not created before shutdown")

    def _call(self, argv: list[str], timeout: float) -> CommandResult:
        return self._runner(
            argv,
            {"PATH": "/usr/bin:/bin"},
            float(timeout),
        )

    def pin_container(self, timeout: float) -> None:
        self._read_id(timeout)

    def stop_container(self, timeout: float) -> None:
        container_id = self._read_id(timeout)
        self._call(
            ["docker", "stop", "--time", "5", container_id],
            timeout,
        )

    def kill_container(self, timeout: float) -> None:
        container_id = self._read_id(timeout)
        self._call(["docker", "kill", container_id], timeout)

    def container_stopped(self, timeout: float) -> bool:
        container_id = self._read_id(timeout)
        result = self._call(
            [
                "docker",
                "inspect",
                "--format={{.State.Status}}",
                container_id,
            ],
            timeout,
        )
        if result.returncode != 0:
            absent = {
                f"Error: No such object: {container_id}",
                f"Error response from daemon: No such container: {container_id}",
            }
            if not result.stdout and result.stderr.strip() in absent:
                return True
            raise LaunchError("Docker inspect could not verify container absence")
        return result.stdout.strip() in {"dead", "exited"}


def _spawn_process(launch: ArmLaunch) -> ProcessHandle:
    log_path = launch.out_dir / "launcher.log"
    log_handle = log_path.open("xb")
    try:
        process = subprocess.Popen(
            list(launch.argv),
            cwd=launch.cwd,
            env=dict(launch.environment),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return _SubprocessHandle(
        process,
        log_handle,
        _DockerContainerHandle(launch.cidfile_path),
    )


def _revalidate_files(plan: LaunchPlan) -> None:
    for item in plan.verified_files:
        if _hash_regular(item.path, label="launch input") != item.sha256:
            raise LaunchError(f"launch input changed after planning: {item.path}")


def _materialize_runtime_config(launch: ArmLaunch) -> None:
    path = launch.runtime_config_path
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise LaunchError("runtime config path must not be a symlink")
    if path.exists():
        if (
            not path.is_file()
            or _hash_regular(path, label="runtime config")
            != launch.runtime_config_sha256
            or path.stat().st_mode & 0o222
        ):
            raise LaunchError("existing runtime config does not match the plan")
        return
    try:
        with path.open("xb") as handle:
            handle.write(launch.runtime_config_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise


def _safe_terminate(process: ProcessHandle) -> None:
    _terminate_all((process,))


def _terminate_all(processes: Sequence[ProcessHandle]) -> None:
    known = tuple(processes)
    active = tuple(
        process for process in known if process.poll() is None
    )
    container_deadline = time.monotonic() + 15.0
    needs_container_kill: set[int] = set()
    for process in known:
        stop_container = getattr(process, "stop_container", None)
        if stop_container is None:
            continue
        try:
            stop_container(max(0.001, container_deadline - time.monotonic()))
        except (OSError, ValueError, subprocess.SubprocessError):
            needs_container_kill.add(id(process))
    for process in active:
        try:
            process.terminate_tree()
        except (OSError, subprocess.SubprocessError):
            pass
    for process in known:
        container_stopped = getattr(process, "container_stopped", None)
        if container_stopped is None:
            continue
        try:
            if not container_stopped(
                max(0.001, container_deadline - time.monotonic())
            ):
                needs_container_kill.add(id(process))
        except (OSError, ValueError, subprocess.SubprocessError):
            needs_container_kill.add(id(process))
    for process in known:
        if id(process) not in needs_container_kill:
            continue
        kill_container = getattr(process, "kill_container", None)
        if kill_container is None:
            continue
        try:
            kill_container(max(0.001, container_deadline - time.monotonic()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    containers_verified = True
    for process in known:
        if id(process) not in needs_container_kill:
            continue
        container_stopped = getattr(process, "container_stopped", None)
        if container_stopped is None:
            containers_verified = False
            continue
        try:
            containers_verified = (
                container_stopped(
                    max(0.001, container_deadline - time.monotonic())
                )
                and containers_verified
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            containers_verified = False
    deadline = time.monotonic() + 10.0
    pending = []
    for process in active:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending.append(process)
            continue
        try:
            process.wait(timeout=remaining)
        except (OSError, subprocess.SubprocessError):
            pending.append(process)
    for process in pending:
        kill_tree = getattr(process, "kill_tree", None)
        if kill_tree is not None:
            try:
                kill_tree()
            except (OSError, subprocess.SubprocessError):
                pass
    kill_deadline = time.monotonic() + 2.0
    for process in pending:
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            process.wait(timeout=remaining)
        except (OSError, subprocess.SubprocessError):
            pass
    if not containers_verified:
        raise LaunchError("Docker containers could not be verified stopped")


@contextmanager
def installed_shutdown_handlers() -> Iterator[Callable[[], int | None]]:
    """Capture termination requests without releasing the supervision lock."""

    requested: dict[str, int | None] = {"signum": None}
    watched = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {signum: signal.getsignal(signum) for signum in watched}

    def capture(signum, _frame) -> None:
        if requested["signum"] is None:
            requested["signum"] = int(signum)

    try:
        for signum in watched:
            signal.signal(signum, capture)
        yield lambda: requested["signum"]
    finally:
        for signum in watched:
            signal.signal(signum, previous[signum])


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
    trainer_preflight: Callable[[LaunchPlan], None] = preflight_trainer_contract,
    rank_zero_resolver: Callable[
        [LaunchPlan, Mapping[str, int]], Mapping[str, int]
    ]
    | None = None,
    shutdown_source: Callable[[], int | None] | None = None,
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
            trainer_preflight=trainer_preflight,
            rank_zero_resolver=rank_zero_resolver,
            shutdown_source=shutdown_source,
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
    trainer_preflight: Callable[[LaunchPlan], None] = preflight_trainer_contract,
    rank_zero_resolver: Callable[
        [LaunchPlan, Mapping[str, int]], Mapping[str, int]
    ]
    | None = None,
    shutdown_source: Callable[[], int | None] | None = None,
) -> SupervisionResult:
    """Launch both arms, then accept only paired zero exit status."""

    _revalidate_files(plan)
    trainer_preflight(plan)
    processes: dict[str, ProcessHandle] = {}
    try:
        for launch in plan.arms:
            _materialize_runtime_config(launch)
        cidfile_roots = {launch.cidfile_path.parent for launch in plan.arms}
        if len(cidfile_roots) != 1:
            raise LaunchError("paired Docker cidfiles must share one private root")
        cidfile_root = cidfile_roots.pop()
        try:
            cidfile_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        except OSError as error:
            raise LaunchError(
                "Docker cidfile root became stale before launch"
            ) from error
        os.chmod(cidfile_root, 0o700)
        for launch in plan.arms:
            try:
                launch.out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            except OSError as error:
                raise LaunchError(
                    f"{launch.arm} output became stale before launch"
                ) from error
            os.chown(launch.out_dir, plan.runtime_uid, plan.runtime_gid)
            os.chmod(launch.out_dir, 0o700)
        for launch in plan.arms:
            try:
                processes[launch.arm] = spawner(launch)
            except BaseException as error:
                _terminate_all(tuple(processes.values()))
                raise LaunchError(
                    f"failed to spawn {launch.arm} process group"
                ) from error
        child_pids = {arm: process.pid for arm, process in processes.items()}
        if set(child_pids) != set(_ARMS):
            raise LaunchError("both child PIDs must be recorded before supervision")
        pin_deadline = time.monotonic() + 30.0
        try:
            for arm in _ARMS:
                processes[arm].pin_container(
                    max(0.001, pin_deadline - time.monotonic())
                )
        except Exception as error:
            _terminate_all(tuple(processes.values()))
            raise LaunchError(
                "both Docker container IDs must be pinned before supervision"
            ) from error
        resolver = rank_zero_resolver or _resolve_rank_zero_pids
        try:
            rank_zero_pids = dict(resolver(plan, child_pids))
        except Exception as error:
            _terminate_all(tuple(processes.values()))
            if isinstance(error, LaunchError):
                raise
            raise LaunchError(
                "rank-zero PID files were not created by both arms"
            ) from error
        if set(rank_zero_pids) != set(_ARMS) or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1
            for pid in rank_zero_pids.values()
        ):
            _terminate_all(tuple(processes.values()))
            raise LaunchError("both pre-existing rank-zero PID files are required")

        while True:
            requested_signal = (
                shutdown_source() if shutdown_source is not None else None
            )
            if requested_signal is not None:
                _terminate_all(tuple(processes.values()))
                return SupervisionResult(
                    status="terminated",
                    returncode=128 + int(requested_signal),
                    child_pids=child_pids,
                    peer_terminated=True,
                )
            if notice_source is not None:
                try:
                    notice = notice_source()
                except Exception as error:
                    _terminate_all(tuple(processes.values()))
                    raise LaunchError(
                        "interruption notice polling failed; pair was stopped"
                    ) from error
                if notice is not None:
                    if interruption_handler is None:
                        _terminate_all(tuple(processes.values()))
                        return SupervisionResult(
                            status="interrupted",
                            returncode=74,
                            child_pids=child_pids,
                        )
                    try:
                        interruption = interruption_handler(
                            plan, rank_zero_pids, notice
                        )
                    except Exception as error:
                        _terminate_all(tuple(processes.values()))
                        raise LaunchError(
                            "interruption handling failed; pair was stopped"
                        ) from error
                    _terminate_all(tuple(processes.values()))
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
                peer_terminated = any(
                    arm != failed and statuses[arm] is None for arm in _ARMS
                )
                _terminate_all(tuple(processes.values()))
                return SupervisionResult(
                    status="failed",
                    returncode=int(statuses[failed]),
                    child_pids=child_pids,
                    failed_arm=failed,
                    peer_terminated=peer_terminated,
                )
            if all(status == 0 for status in statuses.values()):
                _terminate_all(tuple(processes.values()))
                return SupervisionResult(
                    status="completed",
                    returncode=0,
                    child_pids=child_pids,
                )
            sleep(0.25)
    except KeyboardInterrupt:
        _terminate_all(tuple(processes.values()))
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
    runtime_uid: int,
    deadline: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    while monotonic() <= deadline:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            descriptor = None
        except OSError as error:
            raise LaunchError("rank-zero PID file is unsafe") from error
        if descriptor is not None:
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_uid != runtime_uid
                    or metadata.st_size > 32
                ):
                    raise LaunchError("rank-zero PID file is unsafe")
                raw = os.read(descriptor, 33)
            finally:
                os.close(descriptor)
            try:
                text = raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise LaunchError("rank-zero PID file is invalid") from error
            if not text.isdigit():
                raise LaunchError("rank-zero PID file is invalid")
            pid = int(text)
            if pid <= 1:
                raise LaunchError("rank-zero PID file is invalid")
            try:
                os.kill(pid, 0)
            except OSError as error:
                raise LaunchError("rank-zero process is not running") from error
            return pid
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(0.25, remaining))
    raise LaunchError("rank-zero PID file was not produced by torchrun")


def _resolve_rank_zero_pids(
    plan: LaunchPlan,
    _child_pids: Mapping[str, int],
) -> Mapping[str, int]:
    deadline = time.monotonic() + 30.0
    resolved = {
        launch.arm: _read_rank_zero_pid(
            launch.rank_zero_pid_file,
            runtime_uid=plan.runtime_uid,
            deadline=deadline,
        )
        for launch in plan.arms
    }
    if len(set(resolved.values())) != len(_ARMS):
        raise LaunchError("rank-zero PID files must identify distinct processes")
    return resolved


def _production_interruption_handler(
    plan: LaunchPlan,
    rank_zero_pids: Mapping[str, int],
    notice: str,
) -> InterruptionResult:
    staging = plan.scratch_root / "staging"
    with tempfile.TemporaryDirectory(prefix="aws-home-", dir=staging) as home:
        os.chmod(home, 0o700)
        store = S3ObjectStore(
            region=plan.runtime.region,
            environment={
                "AWS_REGION": plan.runtime.region,
                "HOME": home,
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
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
            upload_reserve_seconds=30.0,
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
        default=root / PROFILE_MEMBER_PATH,
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
        with installed_shutdown_handlers() as shutdown_source:
            result = supervise_pair(
                plan,
                notice_source=client.interruption_notice,
                interruption_handler=_production_interruption_handler,
                shutdown_source=shutdown_source,
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
