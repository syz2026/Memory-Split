#!/usr/bin/env python3
"""Executable, fail-closed qualification canary for one AWS P5 host."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import secrets
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.attest_environment import (
    AttestationError,
    MINIMAL_COMMAND_ENVIRONMENT,
    parse_environment_receipt_bytes,
    parse_runtime_lock_bytes,
    read_regular_input,
)
from cluster.aws.p5.corpus_contract import (
    CorpusContractError,
    verify_canonical_corpus,
)
from cluster.aws.p5.profile import load_aws_p5_profile
from msctl.aws_contracts import (
    ARMS,
    COHORT_ID,
    CONFIG_ROOT,
    EXPECTED_CONFIG_PATHS,
    PACKAGE_FORMAT_VERSION,
    PROFILE_PATH,
    PROVIDER,
    SEEDS,
)


RECEIPT_TYPE = "memorysplit-aws-p5-qualification-v1"
OPERATIONAL_RECEIPT_TYPE = "memorysplit-operational-training-v1"
PHASE_ORDER = (
    "hardware",
    "nccl_all_reduce",
    "functional",
    "resume",
    "throughput_4x4",
    "s3_roundtrip",
)
FUNCTIONAL_UPDATES = 1
THROUGHPUT_UPDATES = 100
THROUGHPUT_WARMUP_UPDATES = 10
THROUGHPUT_PORT_BASE = 29_700
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_GPU_NAMES = {
    "NVIDIA H100 80GB",
    "NVIDIA H100 80GB HBM3",
}
_RELEASE_FIELDS = {
    "schema_version",
    "package_format_version",
    "release_id",
    "provider",
    "archive",
    "source",
    "seed_assignment",
    "cohort_assignment",
    "profile",
    "environment",
    "dataset_pointer",
    "cohort_assignment_sha256",
    "profile_sha256",
    "dataset_pointer_sha256",
    "config_sha256",
    "members_sha256",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "provider",
    "cohort_id",
    "seed",
    "release_sha256",
    "release_receipt_sha256",
    "profile_sha256",
    "dataset_pointer_sha256",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "ordered_stream_sha256",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "sealed_evaluation_release_sha256",
    "source_commit",
    "source_tree",
    "runs",
}
_RUN_FIELDS = {
    "run_id",
    "arm",
    "seed",
    "config",
    "config_sha256",
    "estimated_gpu_hours",
}
_CHECKPOINT_EVIDENCE_FIELDS = {
    "checkpoint_sha256",
    "config_fingerprint",
    "config_matches",
    "data",
    "step",
    "world_size",
}
_CHECKPOINT_DATA_FIELDS = {
    "build_id",
    "global_cursor",
    "ordered_stream_sha256",
    "receipt_sha256",
    "sidecar_name",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "provider",
    "instance_id",
    "boot_id",
    "profile_sha256",
    "runtime_lock_sha256",
    "environment_receipt_sha256",
    "release_sha256",
    "release_receipt_sha256",
    "run_manifest_sha256",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "ordered_stream_sha256",
    "source_commit",
    "source_tree",
    "container_image",
    "container_image_digest",
    "seed",
    "phases",
    "hardware",
    "functional",
    "resume",
    "throughput_4x4",
    "s3_roundtrip",
    "passed",
    "started_at",
    "ended_at",
    "total_seconds",
}


class CanaryError(ValueError):
    """One canary input, command, phase, or receipt failed closed."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path
    timeout_seconds: float
    host_work_root: Path | None = None
    gpu_ids: tuple[int, ...] = ()
    cpu_affinity: tuple[int, int] | None = None
    master_port: int | None = None
    concurrent_group: str | None = None


@dataclass(frozen=True)
class ObjectWrite:
    checksum_sha256: str
    byte_count: int
    version_id: str


@dataclass(frozen=True)
class ObjectRead:
    payload: bytes
    checksum_sha256: str
    version_id: str


class CommandReader(Protocol):
    def run(self, spec: CommandSpec) -> CommandResult: ...

    def run_pair(
        self,
        specs: Sequence[CommandSpec],
    ) -> Mapping[str, CommandResult]: ...


class ObjectStore(Protocol):
    def put(
        self,
        uri: str,
        payload: bytes,
        *,
        sha256: str,
    ) -> ObjectWrite: ...

    def get(self, uri: str, *, version_id: str) -> ObjectRead: ...


class TimeReader(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


@dataclass(frozen=True)
class CanaryPlan:
    release_root: Path
    release_receipt_path: Path
    run_manifest_path: Path
    dataset_receipt_path: Path
    environment_receipt_path: Path
    runtime_lock_path: Path
    scratch_root: Path
    output_path: Path
    s3_root: str
    instance_id: str
    boot_id: str
    seed: int
    profile_sha256: str
    runtime_lock_sha256: str
    environment_receipt_sha256: str
    release_sha256: str
    release_receipt_sha256: str
    run_manifest_sha256: str
    dataset_pointer_sha256: str
    dataset_receipt_sha256: str
    dataset_build_id: str
    ordered_stream_sha256: str
    source_commit: str
    source_tree: str
    container_image: str
    container_image_digest: str
    config_sha256: Mapping[str, str]
    config_paths: Mapping[str, str]
    tokens_per_step: int
    tuple_sha256: str
    runtime_facts: Mapping[str, str]


def _canonical(value: object, *, newline: bool) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return encoded + (b"\n" if newline else b"")


def canonical_receipt(receipt: Mapping[str, object]) -> bytes:
    if set(receipt) != _RECEIPT_FIELDS:
        raise CanaryError("qualification receipt fields do not match contract")
    return _canonical(dict(receipt), newline=True)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanaryError(f"JSON contains duplicate field: {key}")
        result[key] = value
    return result


def _read_canonical_object(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> tuple[dict[str, object], bytes]:
    try:
        payload = read_regular_input(
            path,
            label=label,
            maximum_bytes=maximum_bytes,
        )
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CanaryError(f"{label} contains non-finite {constant}")
            ),
        )
    except CanaryError:
        raise
    except (AttestationError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"{label} is not canonical JSON") from error
    if type(value) is not dict or payload != _canonical(value, newline=True):
        raise CanaryError(f"{label} is not one canonical JSON object")
    return value, payload


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise CanaryError(f"{label} fields do not match the closed contract")


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CanaryError(f"{label} must be one lowercase SHA-256")
    return value


def _safe_relative(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        raise CanaryError(f"{label} must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CanaryError(f"{label} must be a portable relative path")
    return path.as_posix()


def _regular_sha256(path: Path, *, label: str) -> str:
    try:
        payload = read_regular_input(
            path,
            label=label,
            maximum_bytes=512 * 1024 * 1024,
        )
    except (AttestationError, OSError) as error:
        raise CanaryError(
            f"{label} must be a singly-linked regular file"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _verify_release_root(
    root: Path,
    release: Mapping[str, object],
) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise CanaryError("release root must be one extracted directory")
    sums_path = root / "SHA256SUMS"
    try:
        sums_bytes = read_regular_input(
            sums_path,
            label="release SHA256SUMS",
            maximum_bytes=16 * 1024 * 1024,
        )
        sums_text = sums_bytes.decode("ascii")
    except (AttestationError, OSError, UnicodeDecodeError) as error:
        raise CanaryError("release SHA256SUMS is invalid") from error
    if (
        not sums_text
        or not sums_text.endswith("\n")
        or hashlib.sha256(sums_bytes).hexdigest()
        != _sha256(release["members_sha256"], label="release members")
    ):
        raise CanaryError("release member manifest hash does not match receipt")
    members: dict[str, str] = {}
    paths: list[str] = []
    for line in sums_text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise CanaryError("release SHA256SUMS line is malformed")
        digest = _sha256(line[:64], label="release member")
        relative = _safe_relative(line[66:], label="release member path")
        if relative == "SHA256SUMS" or relative in members:
            raise CanaryError("release SHA256SUMS has a duplicate member")
        members[relative] = digest
        paths.append(relative)
    if paths != sorted(paths):
        raise CanaryError("release SHA256SUMS members are not sorted")
    actual: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise CanaryError("release root contains a symlink")
        if candidate.is_file():
            actual.add(candidate.relative_to(root).as_posix())
        elif not candidate.is_dir():
            raise CanaryError("release root contains a special entry")
    if actual != set(members) | {"SHA256SUMS"}:
        raise CanaryError("release root has extra or missing members")
    for relative, expected in members.items():
        actual_digest = _regular_sha256(
            root / relative,
            label=f"release member {relative}",
        )
        if actual_digest != expected:
            raise CanaryError(f"release member hash mismatch: {relative}")
    return members


def _load_release(path: Path | str) -> tuple[dict[str, object], bytes]:
    release, payload = _read_canonical_object(path, label="release receipt")
    _exact_fields(release, _RELEASE_FIELDS, label="release receipt")
    archive = release.get("archive")
    source = release.get("source")
    assignment = release.get("seed_assignment")
    profile = release.get("profile")
    pointer = release.get("dataset_pointer")
    cohort = release.get("cohort_assignment")
    config_hashes = release.get("config_sha256")
    if (
        release.get("schema_version") != 1
        or type(release.get("schema_version")) is not int
        or release.get("package_format_version") != PACKAGE_FORMAT_VERSION
        or type(release.get("package_format_version")) is not int
        or release.get("provider") != PROVIDER
        or type(archive) is not dict
        or set(archive) != {"path", "sha256", "bytes"}
        or type(archive["bytes"]) is not int
        or archive["bytes"] <= 0
        or type(source) is not dict
        or set(source) != {"commit", "tree", "dirty"}
        or source["dirty"] is not False
        or not isinstance(source["commit"], str)
        or _COMMIT_RE.fullmatch(source["commit"]) is None
        or not isinstance(source["tree"], str)
        or _COMMIT_RE.fullmatch(source["tree"]) is None
        or type(assignment) is not dict
        or assignment
        != {
            "cohort_id": COHORT_ID,
            "provider": PROVIDER,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
        }
        or type(profile) is not dict
        or set(profile) != {"path", "sha256"}
        or profile["path"] != PROFILE_PATH
        or type(pointer) is not dict
        or set(pointer) != {"path", "sha256"}
        or pointer["path"] != "DATASET-POINTER-AWS.json"
        or type(cohort) is not dict
        or set(cohort) != {"path", "sha256"}
        or cohort["path"] != "configs/cohort-assignment-v3.json"
        or type(config_hashes) is not dict
        or set(config_hashes) != set(EXPECTED_CONFIG_PATHS)
    ):
        raise CanaryError("release receipt is not the exact v3 AWS release")
    for label, value in (
        ("release archive", archive["sha256"]),
        ("release members", release["members_sha256"]),
        ("release profile", profile["sha256"]),
        ("release profile duplicate", release["profile_sha256"]),
        ("release dataset pointer", pointer["sha256"]),
        ("release dataset pointer duplicate", release["dataset_pointer_sha256"]),
        ("release cohort", cohort["sha256"]),
        ("release cohort duplicate", release["cohort_assignment_sha256"]),
    ):
        _sha256(value, label=label)
    if (
        profile["sha256"] != release["profile_sha256"]
        or pointer["sha256"] != release["dataset_pointer_sha256"]
        or cohort["sha256"] != release["cohort_assignment_sha256"]
        or any(
            _sha256(digest, label=f"release config {relative}") != digest
            for relative, digest in config_hashes.items()
        )
    ):
        raise CanaryError("release receipt duplicate identities differ")
    return release, payload


def _load_manifest(
    path: Path | str,
    *,
    release_root: Path,
) -> tuple[dict[str, object], bytes, dict[str, dict[str, object]]]:
    manifest, payload = _read_canonical_object(path, label="run manifest")
    _exact_fields(manifest, _MANIFEST_FIELDS, label="run manifest")
    seed = manifest.get("seed")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 3
        or manifest.get("provider") != PROVIDER
        or manifest.get("cohort_id") != COHORT_ID
        or type(seed) is not int
        or seed not in SEEDS
        or not isinstance(manifest.get("source_commit"), str)
        or _COMMIT_RE.fullmatch(str(manifest["source_commit"])) is None
        or not isinstance(manifest.get("source_tree"), str)
        or _COMMIT_RE.fullmatch(str(manifest["source_tree"])) is None
    ):
        raise CanaryError("run manifest is not exact schema 3")
    for field in (
        "release_sha256",
        "release_receipt_sha256",
        "profile_sha256",
        "dataset_pointer_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "sealed_evaluation_release_sha256",
    ):
        _sha256(manifest[field], label=f"run manifest {field}")
    rows = manifest.get("runs")
    if type(rows) is not list or len(rows) != 2:
        raise CanaryError("run manifest must contain one complete pair")
    by_arm: dict[str, dict[str, object]] = {}
    for row in rows:
        if type(row) is not dict:
            raise CanaryError("run manifest entry must be an object")
        _exact_fields(row, _RUN_FIELDS, label="run manifest entry")
        arm = row.get("arm")
        expected_config = (
            f"{CONFIG_ROOT}/{arm}-s{seed}.yaml"
            if arm in ARMS
            else None
        )
        hours = row.get("estimated_gpu_hours")
        if (
            arm not in ARMS
            or arm in by_arm
            or row.get("seed") != seed
            or type(row.get("seed")) is not int
            or row.get("run_id")
            != f"memorysplit-v3-360m-s{seed}-{arm}"
            or row.get("config") != expected_config
            or isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not math.isfinite(hours)
            or hours <= 0
        ):
            raise CanaryError("run manifest pair identity is invalid")
        config_digest = _sha256(
            row.get("config_sha256"),
            label=f"{arm} config",
        )
        if _regular_sha256(
            release_root / str(expected_config),
            label=f"{arm} config",
        ) != config_digest:
            raise CanaryError(f"{arm} config hash does not match")
        by_arm[str(arm)] = row
    if tuple(by_arm) != ARMS:
        raise CanaryError("run manifest arm order must be Dense then Split90")
    return manifest, payload, by_arm


def _validate_private_root(root: Path, output: Path) -> None:
    try:
        root_meta = root.stat(follow_symlinks=False)
    except OSError as error:
        raise CanaryError("canary scratch root is unavailable") from error
    if (
        root.is_symlink()
        or not stat.S_ISDIR(root_meta.st_mode)
        or root_meta.st_uid != os.geteuid()
        or stat.S_IMODE(root_meta.st_mode) & 0o077
    ):
        raise CanaryError("canary scratch root must be private and owner-controlled")
    try:
        if output.parent.resolve(strict=True) != root.resolve(strict=True):
            raise CanaryError("canary output must be directly inside scratch root")
    except OSError as error:
        raise CanaryError("canary output parent is unavailable") from error
    if output.is_symlink():
        raise CanaryError("canary output must not be a symlink")


def _parse_config_tokens(path: Path) -> int:
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CanaryError("canary config cannot be parsed") from error
    if type(config) is not dict:
        raise CanaryError("canary config must be an object")
    tokens = config.get("tokens_per_step")
    if type(tokens) is not int or tokens <= 0:
        raise CanaryError("canary config tokens_per_step is invalid")
    return tokens


def load_canary_plan(
    *,
    release_root: Path | str,
    release_receipt_path: Path | str,
    run_manifest_path: Path | str,
    dataset_receipt_path: Path | str,
    environment_receipt_path: Path | str,
    runtime_lock_path: Path | str,
    instance_id: str,
    boot_id: str,
    scratch_root: Path | str,
    output_path: Path | str,
    s3_root: str,
    dataset_verifier: Callable[..., object] | None = None,
) -> CanaryPlan:
    """Authenticate every local input before rendering any command."""

    if not isinstance(instance_id, str) or _INSTANCE_RE.fullmatch(instance_id) is None:
        raise CanaryError("selected instance ID is invalid")
    if not isinstance(boot_id, str) or _BOOT_RE.fullmatch(boot_id) is None:
        raise CanaryError("selected boot ID is invalid")
    release_receipt_file = Path(
        os.path.abspath(os.fspath(release_receipt_path))
    )
    run_manifest_file = Path(os.path.abspath(os.fspath(run_manifest_path)))
    dataset_receipt_file = Path(
        os.path.abspath(os.fspath(dataset_receipt_path))
    )
    environment_receipt_file = Path(
        os.path.abspath(os.fspath(environment_receipt_path))
    )
    runtime_lock_file = Path(os.path.abspath(os.fspath(runtime_lock_path)))
    release_root_path = Path(release_root).resolve(strict=True)
    release, release_bytes = _load_release(release_receipt_file)
    members = _verify_release_root(release_root_path, release)
    profile_path = release_root_path / PROFILE_PATH
    try:
        profile = load_aws_p5_profile(profile_path)
    except (OSError, TypeError, ValueError) as error:
        raise CanaryError("v3 P5 profile is invalid") from error
    if (
        profile.profile_id != "aws-p5.48xlarge-v3"
        or profile.provider != PROVIDER
        or profile.assigned_seeds != SEEDS
        or profile.sha256 != release["profile_sha256"]
        or members.get(PROFILE_PATH) != profile.sha256
    ):
        raise CanaryError("release profile identity does not match")

    try:
        runtime_bytes = read_regular_input(
            runtime_lock_file,
            label="runtime lock",
            maximum_bytes=1024 * 1024,
        )
        environment_bytes = read_regular_input(
            environment_receipt_file,
            label="environment receipt",
            maximum_bytes=1024 * 1024,
        )
        runtime_lock = parse_runtime_lock_bytes(runtime_bytes)
        environment = parse_environment_receipt_bytes(environment_bytes)
    except (AttestationError, OSError, TypeError, ValueError) as error:
        raise CanaryError("runtime lock or environment receipt is invalid") from error
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    if (
        runtime_lock["profile_sha256"] != profile.sha256
        or environment["profile_sha256"] != profile.sha256
        or environment["runtime_lock_sha256"] != runtime_sha256
        or environment["control_bundle_sha256"]
        != runtime_lock["control_bundle_sha256"]
        or environment["source_commit"] != runtime_lock["source_commit"]
        or environment["source_tree"] != runtime_lock["source_tree"]
        or environment["container_image"] != runtime_lock["container_image"]
        or environment["container_image_digest"]
        != runtime_lock["container_image_digest"]
        or environment["ami_id"] != runtime_lock["ami_id"]
        or environment["runtime_facts"] != runtime_lock["versions"]
        or environment["instance_id"] != instance_id
        or environment["boot_id"] != boot_id
    ):
        raise CanaryError("environment identity does not match runtime lock")

    manifest, manifest_bytes, runs = _load_manifest(
        run_manifest_file,
        release_root=release_root_path,
    )
    release_receipt_sha256 = hashlib.sha256(release_bytes).hexdigest()
    if (
        manifest["release_sha256"] != release["archive"]["sha256"]
        or manifest["release_receipt_sha256"] != release_receipt_sha256
        or manifest["profile_sha256"] != profile.sha256
        or manifest["dataset_pointer_sha256"]
        != release["dataset_pointer_sha256"]
        or manifest["cohort_assignment_sha256"]
        != release["cohort_assignment_sha256"]
        or manifest["source_commit"] != release["source"]["commit"]
        or manifest["source_tree"] != release["source"]["tree"]
        or manifest["source_commit"] != runtime_lock["source_commit"]
        or manifest["source_tree"] != runtime_lock["source_tree"]
        or any(
            release["config_sha256"][row["config"]]
            != row["config_sha256"]
            for row in runs.values()
        )
    ):
        raise CanaryError("release, manifest, and runtime identities differ")

    dataset_value, dataset_bytes = _read_canonical_object(
        dataset_receipt_file,
        label="dataset receipt",
        maximum_bytes=256 * 1024 * 1024,
    )
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    if (
        dataset_sha256 != manifest["dataset_receipt_sha256"]
        or dataset_value.get("build_id") != manifest["dataset_build_id"]
        or dataset_value.get("ordered_stream_sha256")
        != manifest["ordered_stream_sha256"]
    ):
        raise CanaryError("dataset receipt identity does not match manifest")
    verifier = dataset_verifier or verify_canonical_corpus
    try:
        evidence = verifier(
            dataset_receipt_file,
            expected_sha256=dataset_sha256,
            expected_ordered_sha256=str(manifest["ordered_stream_sha256"]),
        )
    except (CorpusContractError, OSError, TypeError, ValueError) as error:
        raise CanaryError("canonical dataset verification failed") from error
    verified_receipt = getattr(evidence, "receipt", None)
    if (
        not isinstance(verified_receipt, Mapping)
        or verified_receipt.get("build_id") != manifest["dataset_build_id"]
        or verified_receipt.get("ordered_stream_sha256")
        != manifest["ordered_stream_sha256"]
    ):
        raise CanaryError("canonical dataset verification returned wrong identity")

    parsed_s3 = urlsplit(s3_root) if isinstance(s3_root, str) else None
    if (
        parsed_s3 is None
        or parsed_s3.scheme != "s3"
        or not parsed_s3.netloc
        or not parsed_s3.path.strip("/")
        or parsed_s3.query
        or parsed_s3.fragment
        or "\\" in s3_root
    ):
        raise CanaryError("canary S3 root must be one immutable bucket prefix")
    scratch = Path(scratch_root).absolute()
    output = Path(output_path).absolute()
    _validate_private_root(scratch, output)
    config_paths = {
        arm: str(runs[arm]["config"])
        for arm in ARMS
    }
    tokens_per_step = _parse_config_tokens(
        release_root_path / config_paths["dense"]
    )
    if any(
        _parse_config_tokens(release_root_path / config_paths[arm])
        != tokens_per_step
        for arm in ARMS
    ):
        raise CanaryError("paired configs disagree on tokens_per_step")
    manifest_sha256 = hashlib.sha256(
        _canonical(manifest, newline=False)
    ).hexdigest()
    tuple_sha256 = hashlib.sha256(
        _canonical(
            {
                "ami_id": runtime_lock["ami_id"],
                "container_image_digest": runtime_lock["container_image_digest"],
                "dataset_receipt_sha256": dataset_sha256,
                "profile_sha256": profile.sha256,
                "release_sha256": manifest["release_sha256"],
                "runtime_lock_sha256": runtime_sha256,
            },
            newline=False,
        )
    ).hexdigest()
    return CanaryPlan(
        release_root=release_root_path,
        release_receipt_path=release_receipt_file,
        run_manifest_path=run_manifest_file,
        dataset_receipt_path=dataset_receipt_file,
        environment_receipt_path=environment_receipt_file,
        runtime_lock_path=runtime_lock_file,
        scratch_root=scratch,
        output_path=output,
        s3_root=s3_root.rstrip("/"),
        instance_id=instance_id,
        boot_id=boot_id,
        seed=int(manifest["seed"]),
        profile_sha256=profile.sha256,
        runtime_lock_sha256=runtime_sha256,
        environment_receipt_sha256=environment_sha256,
        release_sha256=str(manifest["release_sha256"]),
        release_receipt_sha256=release_receipt_sha256,
        run_manifest_sha256=manifest_sha256,
        dataset_pointer_sha256=str(manifest["dataset_pointer_sha256"]),
        dataset_receipt_sha256=dataset_sha256,
        dataset_build_id=str(manifest["dataset_build_id"]),
        ordered_stream_sha256=str(manifest["ordered_stream_sha256"]),
        source_commit=str(manifest["source_commit"]),
        source_tree=str(manifest["source_tree"]),
        container_image=str(runtime_lock["container_image"]),
        container_image_digest=str(runtime_lock["container_image_digest"]),
        config_sha256={
            arm: str(runs[arm]["config_sha256"])
            for arm in ARMS
        },
        config_paths=config_paths,
        tokens_per_step=tokens_per_step,
        tuple_sha256=tuple_sha256,
        runtime_facts=dict(runtime_lock["versions"]),
    )


def _docker_prefix(
    plan: CanaryPlan,
    *,
    name: str,
    host_work_root: Path,
    gpu_ids: tuple[int, ...],
    cpu_affinity: tuple[int, int],
) -> tuple[str, ...]:
    gpu_value = ",".join(str(value) for value in gpu_ids)
    return (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--name",
        name,
        "--read-only",
        "--network=none",
        "--ipc=host",
        "--workdir",
        "/canary",
        "--cpuset-cpus",
        f"{cpu_affinity[0]}-{cpu_affinity[1]}",
        "--gpus",
        f"device={gpu_value}",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "4096",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=4g",
        "--mount",
        f"type=bind,src={plan.release_root},dst=/release,readonly",
        "--mount",
        (
            f"type=bind,src={plan.dataset_receipt_path.parent},"
            "dst=/canary/dataset,readonly"
        ),
        "--mount",
        f"type=bind,src={host_work_root},dst=/canary",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        plan.container_image,
    )


def _arm_output(plan: CanaryPlan, work_root: Path, arm: str) -> Path:
    return work_root / f"runs/seed-{plan.seed}/{arm}"


def _training_spec(
    plan: CanaryPlan,
    *,
    phase: str,
    arm: str,
    work_root: Path,
    resume_mode: str,
    updates: int,
    ranks: int,
    gpu_ids: tuple[int, ...],
    cpu_affinity: tuple[int, int],
    master_port: int | None = None,
    concurrent_group: str | None = None,
) -> CommandSpec:
    prefix = _docker_prefix(
        plan,
        name=f"memorysplit-canary-{phase}-{arm}",
        host_work_root=work_root,
        gpu_ids=gpu_ids,
        cpu_affinity=cpu_affinity,
    )
    python_argv: tuple[str, ...]
    if ranks == 1:
        python_argv = ("/opt/conda/bin/python",)
    else:
        assert master_port is not None
        python_argv = (
            "/opt/conda/bin/python",
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={ranks}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint=127.0.0.1:{master_port}",
        )
    argv = (
        *prefix,
        *python_argv,
        "/release/scripts/run_train.py",
        "--config",
        f"/release/{plan.config_paths[arm]}",
        "--resume",
        resume_mode,
        "--operational-steps",
        str(updates),
    )
    return CommandSpec(
        name=f"{phase}-{arm}-train",
        argv=argv,
        environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
        cwd=plan.scratch_root,
        timeout_seconds=21_600.0 if ranks == 4 else 3_600.0,
        host_work_root=work_root,
        gpu_ids=gpu_ids,
        cpu_affinity=cpu_affinity,
        master_port=master_port,
        concurrent_group=concurrent_group,
    )


def _inspection_spec(
    plan: CanaryPlan,
    *,
    phase: str,
    arm: str,
    work_root: Path,
    gpu_ids: tuple[int, ...],
    cpu_affinity: tuple[int, int],
) -> CommandSpec:
    prefix = _docker_prefix(
        plan,
        name=f"memorysplit-canary-{phase}-{arm}-inspect",
        host_work_root=work_root,
        gpu_ids=gpu_ids,
        cpu_affinity=cpu_affinity,
    )
    output = PurePosixPath("/canary") / f"runs/seed-{plan.seed}/{arm}"
    return CommandSpec(
        name=f"{phase}-{arm}-inspect",
        argv=(
            *prefix,
            "/opt/conda/bin/python",
            "/release/cluster/aws/p5/canary.py",
            "--inspect-checkpoint",
            str(output / "ckpt.pt"),
            "--config",
            f"/release/{plan.config_paths[arm]}",
        ),
        environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
        cwd=plan.scratch_root,
        timeout_seconds=300.0,
        host_work_root=work_root,
        gpu_ids=gpu_ids,
        cpu_affinity=cpu_affinity,
    )


def _phase_specs(plan: CanaryPlan) -> dict[str, tuple[CommandSpec, ...]]:
    hardware = (
        CommandSpec(
            name="hardware-gpus",
            argv=(
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
            cwd=Path("/usr"),
            timeout_seconds=30.0,
        ),
        CommandSpec(
            name="hardware-fabric-manager",
            argv=(
                "/usr/bin/systemctl",
                "is-active",
                "nvidia-fabricmanager",
            ),
            environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
            cwd=Path("/usr"),
            timeout_seconds=30.0,
        ),
        CommandSpec(
            name="hardware-topology",
            argv=("/usr/bin/nvidia-smi", "topo", "-m"),
            environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
            cwd=Path("/usr"),
            timeout_seconds=30.0,
        ),
        CommandSpec(
            name="hardware-boot-id",
            argv=("/usr/bin/cat", "/proc/sys/kernel/random/boot_id"),
            environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
            cwd=Path("/usr"),
            timeout_seconds=30.0,
        ),
    )
    nccl_root = plan.scratch_root / "nccl"
    nccl_prefix = _docker_prefix(
        plan,
        name="memorysplit-canary-nccl",
        host_work_root=nccl_root,
        gpu_ids=tuple(range(8)),
        cpu_affinity=(0, 191),
    )
    nccl = (
        CommandSpec(
            name="nccl-all-reduce",
            argv=(
                *nccl_prefix,
                "/opt/conda/bin/python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=8",
                "/release/cluster/aws/p5/canary.py",
                "--nccl-worker",
            ),
            environment=dict(MINIMAL_COMMAND_ENVIRONMENT),
            cwd=plan.scratch_root,
            timeout_seconds=600.0,
            host_work_root=nccl_root,
            gpu_ids=tuple(range(8)),
            cpu_affinity=(0, 191),
        ),
    )
    functional_rows: list[CommandSpec] = []
    resume_rows: list[CommandSpec] = []
    for arm, gpu_ids, cpus in (
        ("dense", (0,), (0, 95)),
        ("split90", (4,), (96, 191)),
    ):
        work_root = plan.scratch_root / "functional" / arm
        functional_rows.extend(
            (
                _training_spec(
                    plan,
                    phase="functional",
                    arm=arm,
                    work_root=work_root,
                    resume_mode="none",
                    updates=FUNCTIONAL_UPDATES,
                    ranks=1,
                    gpu_ids=gpu_ids,
                    cpu_affinity=cpus,
                ),
                _inspection_spec(
                    plan,
                    phase="functional",
                    arm=arm,
                    work_root=work_root,
                    gpu_ids=gpu_ids,
                    cpu_affinity=cpus,
                ),
            )
        )
        resume_rows.extend(
            (
                _training_spec(
                    plan,
                    phase="resume",
                    arm=arm,
                    work_root=work_root,
                    resume_mode="auto",
                    updates=FUNCTIONAL_UPDATES,
                    ranks=1,
                    gpu_ids=gpu_ids,
                    cpu_affinity=cpus,
                ),
                _inspection_spec(
                    plan,
                    phase="resume",
                    arm=arm,
                    work_root=work_root,
                    gpu_ids=gpu_ids,
                    cpu_affinity=cpus,
                ),
            )
        )
    throughput = tuple(
        _training_spec(
            plan,
            phase="throughput",
            arm=arm,
            work_root=plan.scratch_root / "throughput" / arm,
            resume_mode="none",
            updates=THROUGHPUT_UPDATES,
            ranks=4,
            gpu_ids=gpus,
            cpu_affinity=cpus,
            master_port=THROUGHPUT_PORT_BASE + plan.seed * 2 + offset,
            concurrent_group="throughput-4x4",
        )
        for arm, gpus, cpus, offset in (
            ("dense", (0, 1, 2, 3), (0, 95), 0),
            ("split90", (4, 5, 6, 7), (96, 191), 1),
        )
    )
    return {
        "hardware": hardware,
        "nccl_all_reduce": nccl,
        "functional": tuple(functional_rows),
        "resume": tuple(resume_rows),
        "throughput_4x4": throughput,
        "s3_roundtrip": (),
    }


def _render_spec(spec: CommandSpec) -> dict[str, object]:
    value: dict[str, object] = {
        "name": spec.name,
        "argv": list(spec.argv),
        "environment": dict(sorted(spec.environment.items())),
        "cwd": str(spec.cwd),
        "timeout_seconds": spec.timeout_seconds,
    }
    if spec.gpu_ids:
        value["gpu_ids"] = list(spec.gpu_ids)
    if spec.cpu_affinity is not None:
        value["cpu_affinity"] = list(spec.cpu_affinity)
    if spec.master_port is not None:
        value["master_port"] = spec.master_port
    if spec.concurrent_group is not None:
        value["concurrent_group"] = spec.concurrent_group
    return value


def render_plan(plan: CanaryPlan) -> dict[str, object]:
    specs = _phase_specs(plan)
    blob = _roundtrip_blob(plan)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    uri = (
        f"{plan.s3_root}/canary-roundtrip/{plan.tuple_sha256}/"
        f"{blob_sha256}.json"
    )
    phases = []
    for name in PHASE_ORDER:
        row: dict[str, object] = {
            "name": name,
            "commands": [_render_spec(spec) for spec in specs[name]],
        }
        if name == "s3_roundtrip":
            row["object_operations"] = [
                {
                    "operation": "put",
                    "uri": uri,
                    "sha256": blob_sha256,
                    "bytes": len(blob),
                },
                {
                    "operation": "get",
                    "uri": uri,
                    "version_required": True,
                },
            ]
        phases.append(row)
    return {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "provider": PROVIDER,
        "instance_id": plan.instance_id,
        "boot_id": plan.boot_id,
        "seed": plan.seed,
        "phase_order": list(PHASE_ORDER),
        "phases": phases,
        "output": str(plan.output_path),
        "qualification_tuple_sha256": plan.tuple_sha256,
    }


def _checked(
    reader: CommandReader,
    spec: CommandSpec,
) -> CommandResult:
    try:
        result = reader.run(spec)
    except Exception as error:
        raise CanaryError(f"{spec.name} command failed") from error
    if (
        not isinstance(result, CommandResult)
        or type(result.returncode) is not int
        or result.returncode != 0
        or not isinstance(result.stdout, str)
        or not isinstance(result.stderr, str)
        or result.stderr != ""
    ):
        raise CanaryError(f"{spec.name} command returned error or stderr")
    return result


def _parse_one_json_line(
    stdout: str,
    *,
    label: str,
    fields: set[str],
) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line]
    parsed: list[dict[str, object]] = []
    for line in lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    CanaryError(f"{label} contains non-finite {constant}")
                ),
            )
        except (json.JSONDecodeError, CanaryError):
            continue
        if type(value) is dict and set(value) == fields:
            parsed.append(value)
    if len(parsed) != 1:
        raise CanaryError(f"{label} did not emit one exact JSON record")
    return parsed[0]


def _parse_complete_json_object(
    stdout: str,
    *,
    label: str,
    fields: set[str],
) -> dict[str, object]:
    try:
        value = json.loads(
            stdout,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CanaryError(f"{label} contains non-finite {constant}")
            ),
        )
    except CanaryError:
        raise
    except json.JSONDecodeError as error:
        raise CanaryError(
            f"{label} did not emit one complete JSON object"
        ) from error
    if type(value) is not dict or set(value) != fields:
        raise CanaryError(f"{label} JSON fields do not match the contract")
    return value


def _hardware_phase(
    plan: CanaryPlan,
    reader: CommandReader,
    specs: Sequence[CommandSpec],
) -> dict[str, object]:
    gpu_result, fabric_result, topology_result, boot_result = (
        _checked(reader, spec) for spec in specs
    )
    gpu_names: list[str] = []
    memory: list[int] = []
    lines = [line for line in gpu_result.stdout.splitlines() if line]
    if len(lines) != 8:
        raise CanaryError("hardware phase requires exactly eight GPUs")
    for expected_index, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise CanaryError("hardware GPU record is malformed")
        try:
            index = int(parts[0])
            memory_mib = int(parts[2])
        except ValueError as error:
            raise CanaryError("hardware GPU record is malformed") from error
        if (
            index != expected_index
            or parts[1] not in _GPU_NAMES
            or not 80_000 <= memory_mib <= 82_000
        ):
            raise CanaryError("hardware is not eight exact H100 80GB devices")
        gpu_names.append(parts[1])
        memory.append(memory_mib)
    if fabric_result.stdout != "active\n":
        raise CanaryError("hardware Fabric Manager is not active")
    topology_lines = [
        line.split()
        for line in topology_result.stdout.splitlines()
        if line.strip()
    ]
    expected_headers = [f"GPU{index}" for index in range(8)]
    if not topology_lines or topology_lines[0][:8] != expected_headers:
        raise CanaryError("hardware NVSwitch topology header is invalid")
    rows = topology_lines[1:9]
    if len(rows) != 8:
        raise CanaryError("hardware NVSwitch topology is incomplete")
    for left, row in enumerate(rows):
        if len(row) < 9 or row[0] != f"GPU{left}":
            raise CanaryError("hardware NVSwitch topology row is invalid")
        for right, link in enumerate(row[1:9]):
            if (left == right and link != "X") or (
                left != right and re.fullmatch(r"NV[1-9][0-9]*", link) is None
            ):
                raise CanaryError("hardware is not fully connected through NVSwitch")
    if (
        boot_result.stdout != plan.boot_id + "\n"
        or _BOOT_RE.fullmatch(boot_result.stdout.strip()) is None
    ):
        raise CanaryError("hardware boot ID differs from the attested boot")
    return {
        "fabric_manager_active": True,
        "gpu_count": 8,
        "gpu_memory_mib": memory,
        "gpu_names": gpu_names,
        "nvswitch_fully_connected": True,
        "runtime_facts": dict(plan.runtime_facts),
    }


def _nccl_phase(reader: CommandReader, spec: CommandSpec) -> dict[str, object]:
    result = _checked(reader, spec)
    record = _parse_one_json_line(
        result.stdout,
        label="NCCL all-reduce",
        fields={"latency_seconds", "reduced_sum", "world_size"},
    )
    latency = record["latency_seconds"]
    if (
        type(record["world_size"]) is not int
        or record["world_size"] != 8
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency <= 0
        or type(record["reduced_sum"]) not in (int, float)
        or not math.isfinite(record["reduced_sum"])
        or float(record["reduced_sum"]) != 36.0
    ):
        raise CanaryError("NCCL all-reduce evidence is invalid")
    return {
        "latency_seconds": float(latency),
        "reduced_sum": 36.0,
        "world_size": 8,
    }


def _read_loss(path: Path, *, expected_step: int) -> float:
    try:
        payload = read_regular_input(
            path,
            label="canary training log",
            maximum_bytes=16 * 1024 * 1024,
        ).decode("utf-8")
    except (AttestationError, OSError, UnicodeDecodeError) as error:
        raise CanaryError("functional training log is unavailable") from error
    rows = []
    for line in payload.splitlines():
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    CanaryError(f"training log contains {constant}")
                ),
            )
        except (json.JSONDecodeError, CanaryError) as error:
            raise CanaryError("functional training log is malformed") from error
        if type(value) is not dict:
            raise CanaryError("functional training log row is not an object")
        rows.append(value)
    if not rows or rows[-1].get("step") != expected_step:
        raise CanaryError("functional training log has the wrong final step")
    loss = rows[-1].get("loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
    ):
        raise CanaryError("functional training loss is not finite")
    return float(loss)


def _revalidate_execution_config(plan: CanaryPlan, arm: str) -> None:
    config_path = plan.release_root / plan.config_paths[arm]
    if _regular_sha256(
        config_path,
        label=f"{arm} execution config",
    ) != plan.config_sha256[arm]:
        raise CanaryError(
            f"{arm} config changed before the execution boundary"
        )


def _checkpoint_evidence(
    plan: CanaryPlan,
    reader: CommandReader,
    spec: CommandSpec,
    *,
    arm: str,
    phase: str,
    expected_step: int,
    prior: Mapping[str, object] | None,
) -> dict[str, object]:
    result = _checked(reader, spec)
    record = _parse_one_json_line(
        result.stdout,
        label=f"{phase} {arm} checkpoint",
        fields=_CHECKPOINT_EVIDENCE_FIELDS,
    )
    data = record.get("data")
    if type(data) is not dict:
        raise CanaryError(f"{phase} {arm} checkpoint data is invalid")
    _exact_fields(data, _CHECKPOINT_DATA_FIELDS, label="checkpoint data")
    expected_sidecar = (
        "dense_target_weights"
        if arm == "dense"
        else "split90_target_weights"
    )
    checkpoint_path = _arm_output(
        plan,
        Path(spec.host_work_root),
        arm,
    ) / "ckpt.pt"
    checkpoint_sha256 = _regular_sha256(
        checkpoint_path,
        label=f"{phase} {arm} checkpoint",
    )
    fingerprint = record["config_fingerprint"]
    if (
        record["config_matches"] is not True
        or record["checkpoint_sha256"] != checkpoint_sha256
        or _SHA256_RE.fullmatch(str(record["checkpoint_sha256"])) is None
        or not isinstance(fingerprint, str)
        or _SHA256_RE.fullmatch(fingerprint) is None
        or type(record["step"]) is not int
        or record["step"] != expected_step
        or type(record["world_size"]) is not int
        or record["world_size"] != 1
        or type(data["global_cursor"]) is not int
        or data["global_cursor"] != expected_step * plan.tokens_per_step
        or data["receipt_sha256"] != plan.dataset_receipt_sha256
        or data["build_id"] != plan.dataset_build_id
        or data["ordered_stream_sha256"] != plan.ordered_stream_sha256
        or data["sidecar_name"] != expected_sidecar
        or (
            prior is not None
            and prior["config_fingerprint"] != fingerprint
        )
    ):
        raise CanaryError(f"{phase} {arm} checkpoint identity is invalid")
    loss = _read_loss(
        checkpoint_path.parent / "log.jsonl",
        expected_step=expected_step,
    )
    value = {
        "checkpoint_sha256": checkpoint_sha256,
        "config_fingerprint": fingerprint,
        "config_sha256": plan.config_sha256[arm],
        "dataset_build_id": plan.dataset_build_id,
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "global_cursor": expected_step * plan.tokens_per_step,
        "loss": loss,
        "ordered_stream_sha256": plan.ordered_stream_sha256,
        "sidecar_name": expected_sidecar,
        "step": expected_step,
        "world_size": 1,
    }
    if prior is not None:
        value["resumed_from_checkpoint_sha256"] = prior["checkpoint_sha256"]
    return value


def _functional_or_resume_phase(
    plan: CanaryPlan,
    reader: CommandReader,
    specs: Sequence[CommandSpec],
    *,
    phase: str,
    prior: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    expected_step = 1 if phase == "functional" else 2
    by_arm: dict[str, dict[str, object]] = {}
    cursor = 0
    for arm in ARMS:
        train_spec = specs[cursor]
        inspect_spec = specs[cursor + 1]
        cursor += 2
        work_root = Path(train_spec.host_work_root)
        if phase == "functional":
            work_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        _revalidate_execution_config(plan, arm)
        _checked(reader, train_spec)
        by_arm[arm] = _checkpoint_evidence(
            plan,
            reader,
            inspect_spec,
            arm=arm,
            phase=phase,
            expected_step=expected_step,
            prior=(prior or {}).get(arm),
        )
    return by_arm


def _operational_metrics(
    stdout: str,
    *,
    arm: str,
) -> list[float]:
    record = _parse_one_json_line(
        stdout,
        label=f"{arm} operational metrics",
        fields={
            "end_step",
            "receipt_type",
            "start_step",
            "step_tok_s",
            "updates",
        },
    )
    values = record["step_tok_s"]
    if (
        record["receipt_type"] != OPERATIONAL_RECEIPT_TYPE
        or type(record["start_step"]) is not int
        or record["start_step"] != 0
        or type(record["end_step"]) is not int
        or record["end_step"] != THROUGHPUT_UPDATES
        or type(record["updates"]) is not int
        or record["updates"] != THROUGHPUT_UPDATES
        or type(values) is not list
    ):
        raise CanaryError(f"{arm} throughput metrics are incomplete")
    return values


def compute_throughput(
    values_by_arm: Mapping[str, Sequence[object]],
    *,
    updates: int,
    warmup_updates: int,
) -> dict[str, object]:
    if (
        type(updates) is not int
        or updates != THROUGHPUT_UPDATES
        or type(warmup_updates) is not int
        or warmup_updates != THROUGHPUT_WARMUP_UPDATES
        or set(values_by_arm) != set(ARMS)
    ):
        raise CanaryError("throughput geometry is invalid")
    validated: dict[str, list[float]] = {}
    for arm in ARMS:
        values = values_by_arm[arm]
        if type(values) not in (list, tuple) or len(values) != updates:
            raise CanaryError(f"{arm} throughput update count is invalid")
        converted = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise CanaryError(f"{arm} throughput contains invalid data")
            converted.append(float(value))
        validated[arm] = converted
    retained = {
        arm: values[warmup_updates:]
        for arm, values in validated.items()
    }
    per_arm = {
        arm: float(statistics.median(retained[arm]))
        for arm in ARMS
    }
    aggregate = [
        retained["dense"][index] + retained["split90"][index]
        for index in range(updates - warmup_updates)
    ]
    result = {
        "aggregate_median_tok_s": float(statistics.median(aggregate)),
        "per_arm_median_tok_s": per_arm,
        "updates": updates,
        "warmup_updates": warmup_updates,
        "world_size_per_arm": 4,
    }
    if any(
        not math.isfinite(value) or value <= 0
        for value in (*per_arm.values(), result["aggregate_median_tok_s"])
    ):
        raise CanaryError("throughput medians are invalid")
    return result


def _throughput_phase(
    plan: CanaryPlan,
    reader: CommandReader,
    specs: Sequence[CommandSpec],
) -> dict[str, object]:
    for spec in specs:
        Path(spec.host_work_root).mkdir(
            parents=True,
            mode=0o700,
            exist_ok=False,
        )
    for arm in ARMS:
        _revalidate_execution_config(plan, arm)
    try:
        results = reader.run_pair(specs)
    except Exception as error:
        raise CanaryError("throughput concurrent pair failed") from error
    if set(results) != {spec.name for spec in specs}:
        raise CanaryError("throughput returned a partial pair")
    values: dict[str, list[float]] = {}
    for spec in specs:
        result = results[spec.name]
        if (
            not isinstance(result, CommandResult)
            or type(result.returncode) is not int
            or result.returncode != 0
            or result.stderr != ""
        ):
            raise CanaryError("throughput peer failed or wrote stderr")
        arm = "dense" if spec.name == "throughput-dense-train" else "split90"
        values[arm] = _operational_metrics(result.stdout, arm=arm)
    return compute_throughput(
        values,
        updates=THROUGHPUT_UPDATES,
        warmup_updates=THROUGHPUT_WARMUP_UPDATES,
    )


def _roundtrip_blob(plan: CanaryPlan) -> bytes:
    return _canonical(
        {
            "qualification_tuple_sha256": plan.tuple_sha256,
            "receipt_type": "memorysplit-aws-p5-canary-blob-v1",
            "seed": plan.seed,
        },
        newline=True,
    )


def qualification_roundtrip_blob(plan: CanaryPlan) -> bytes:
    """Return the exact deterministic S3 canary payload."""

    return _roundtrip_blob(plan)


def parse_qualification_receipt_bytes(
    payload: bytes,
    *,
    plan: CanaryPlan,
) -> dict[str, object]:
    """Independently verify one canonical passing qualification receipt."""

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CanaryError(f"qualification receipt contains {constant}")
            ),
        )
    except CanaryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError("qualification receipt is not valid JSON") from error
    if (
        type(value) is not dict
        or set(value) != _RECEIPT_FIELDS
        or payload != _canonical(value, newline=True)
    ):
        raise CanaryError("qualification receipt is not canonical or exact")
    expected_identity = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "provider": PROVIDER,
        "instance_id": plan.instance_id,
        "boot_id": plan.boot_id,
        "profile_sha256": plan.profile_sha256,
        "runtime_lock_sha256": plan.runtime_lock_sha256,
        "environment_receipt_sha256": plan.environment_receipt_sha256,
        "release_sha256": plan.release_sha256,
        "release_receipt_sha256": plan.release_receipt_sha256,
        "run_manifest_sha256": plan.run_manifest_sha256,
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "dataset_build_id": plan.dataset_build_id,
        "ordered_stream_sha256": plan.ordered_stream_sha256,
        "source_commit": plan.source_commit,
        "source_tree": plan.source_tree,
        "container_image": plan.container_image,
        "container_image_digest": plan.container_image_digest,
        "seed": plan.seed,
        "passed": True,
    }
    if any(type(value.get(key)) is not type(expected) or value.get(key) != expected for key, expected in expected_identity.items()):
        raise CanaryError("qualification receipt identity differs from inputs")
    phases = value["phases"]
    if (
        type(phases) is not list
        or len(phases) != len(PHASE_ORDER)
        or [row.get("name") if type(row) is dict else None for row in phases]
        != list(PHASE_ORDER)
    ):
        raise CanaryError("qualification receipt phase order is invalid")
    for row in phases:
        if (
            set(row) != {"name", "passed", "seconds"}
            or row["passed"] is not True
            or isinstance(row["seconds"], bool)
            or not isinstance(row["seconds"], (int, float))
            or not math.isfinite(row["seconds"])
            or row["seconds"] < 0
        ):
            raise CanaryError("qualification phase record is invalid")
    hardware = value["hardware"]
    if (
        type(hardware) is not dict
        or set(hardware)
        != {
            "fabric_manager_active",
            "gpu_count",
            "gpu_memory_mib",
            "gpu_names",
            "nvswitch_fully_connected",
            "runtime_facts",
        }
        or hardware["fabric_manager_active"] is not True
        or type(hardware["gpu_count"]) is not int
        or hardware["gpu_count"] != 8
        or hardware["nvswitch_fully_connected"] is not True
        or type(hardware["gpu_names"]) is not list
        or len(hardware["gpu_names"]) != 8
        or any(name not in _GPU_NAMES for name in hardware["gpu_names"])
        or type(hardware["gpu_memory_mib"]) is not list
        or len(hardware["gpu_memory_mib"]) != 8
        or any(
            type(memory) is not int or not 80_000 <= memory <= 82_000
            for memory in hardware["gpu_memory_mib"]
        )
        or hardware["runtime_facts"] != dict(plan.runtime_facts)
    ):
        raise CanaryError("qualification hardware evidence is invalid")
    arm_fields = {
        "checkpoint_sha256",
        "config_fingerprint",
        "config_sha256",
        "dataset_build_id",
        "dataset_receipt_sha256",
        "global_cursor",
        "loss",
        "ordered_stream_sha256",
        "sidecar_name",
        "step",
        "world_size",
    }
    for phase, expected_step in (("functional", 1), ("resume", 2)):
        arms = value[phase]
        if type(arms) is not dict or set(arms) != set(ARMS):
            raise CanaryError(f"qualification {phase} pair is incomplete")
        for arm in ARMS:
            record = arms[arm]
            expected_fields = set(arm_fields)
            if phase == "resume":
                expected_fields.add("resumed_from_checkpoint_sha256")
            expected_sidecar = (
                "dense_target_weights"
                if arm == "dense"
                else "split90_target_weights"
            )
            if (
                type(record) is not dict
                or set(record) != expected_fields
                or _SHA256_RE.fullmatch(str(record["checkpoint_sha256"])) is None
                or _SHA256_RE.fullmatch(str(record["config_fingerprint"])) is None
                or record["config_sha256"] != plan.config_sha256[arm]
                or record["dataset_build_id"] != plan.dataset_build_id
                or record["dataset_receipt_sha256"]
                != plan.dataset_receipt_sha256
                or record["ordered_stream_sha256"]
                != plan.ordered_stream_sha256
                or record["sidecar_name"] != expected_sidecar
                or type(record["step"]) is not int
                or record["step"] != expected_step
                or type(record["world_size"]) is not int
                or record["world_size"] != 1
                or type(record["global_cursor"]) is not int
                or record["global_cursor"]
                != expected_step * plan.tokens_per_step
                or isinstance(record["loss"], bool)
                or not isinstance(record["loss"], (int, float))
                or not math.isfinite(record["loss"])
            ):
                raise CanaryError(f"qualification {phase} evidence is invalid")
            if phase == "resume" and (
                record["resumed_from_checkpoint_sha256"]
                != value["functional"][arm]["checkpoint_sha256"]
                or record["config_fingerprint"]
                != value["functional"][arm]["config_fingerprint"]
            ):
                raise CanaryError("qualification resume identity is invalid")
    throughput = value["throughput_4x4"]
    if (
        type(throughput) is not dict
        or set(throughput)
        != {
            "aggregate_median_tok_s",
            "per_arm_median_tok_s",
            "updates",
            "warmup_updates",
            "world_size_per_arm",
        }
        or type(throughput["updates"]) is not int
        or throughput["updates"] != THROUGHPUT_UPDATES
        or type(throughput["warmup_updates"]) is not int
        or throughput["warmup_updates"] != THROUGHPUT_WARMUP_UPDATES
        or type(throughput["world_size_per_arm"]) is not int
        or throughput["world_size_per_arm"] != 4
        or type(throughput["per_arm_median_tok_s"]) is not dict
        or set(throughput["per_arm_median_tok_s"]) != set(ARMS)
    ):
        raise CanaryError("qualification throughput evidence is invalid")
    for metric in (
        throughput["aggregate_median_tok_s"],
        *throughput["per_arm_median_tok_s"].values(),
    ):
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(metric)
            or metric <= 0
        ):
            raise CanaryError("qualification throughput median is invalid")
    roundtrip = value["s3_roundtrip"]
    blob = _roundtrip_blob(plan)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    expected_uri = (
        f"{plan.s3_root}/canary-roundtrip/{plan.tuple_sha256}/"
        f"{blob_sha256}.json"
    )
    if (
        type(roundtrip) is not dict
        or set(roundtrip)
        != {"bytes", "object_uri", "sha256", "version_id"}
        or type(roundtrip["bytes"]) is not int
        or roundtrip["bytes"] != len(blob)
        or roundtrip["object_uri"] != expected_uri
        or roundtrip["sha256"] != blob_sha256
        or not isinstance(roundtrip["version_id"], str)
        or roundtrip["version_id"] in {"", "null"}
    ):
        raise CanaryError("qualification S3 roundtrip evidence is invalid")
    for field in ("started_at", "ended_at"):
        timestamp = value[field]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise CanaryError("qualification timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError as error:
            raise CanaryError("qualification timestamp is invalid") from error
        if parsed.utcoffset() != UTC.utcoffset(parsed):
            raise CanaryError("qualification timestamp is not UTC")
    total_seconds = value["total_seconds"]
    if (
        isinstance(total_seconds, bool)
        or not isinstance(total_seconds, (int, float))
        or not math.isfinite(total_seconds)
        or total_seconds < 0
    ):
        raise CanaryError("qualification total duration is invalid")
    return value


def _s3_phase(
    plan: CanaryPlan,
    store: ObjectStore,
) -> dict[str, object]:
    payload = _roundtrip_blob(plan)
    digest = hashlib.sha256(payload).hexdigest()
    uri = (
        f"{plan.s3_root}/canary-roundtrip/{plan.tuple_sha256}/"
        f"{digest}.json"
    )
    try:
        written = store.put(uri, payload, sha256=digest)
    except Exception as error:
        raise CanaryError("S3 canary object upload failed") from error
    if (
        not isinstance(written, ObjectWrite)
        or written.checksum_sha256 != digest
        or type(written.byte_count) is not int
        or written.byte_count != len(payload)
        or not isinstance(written.version_id, str)
        or written.version_id in {"", "null"}
    ):
        raise CanaryError("S3 object checksum, bytes, or version is invalid")
    try:
        downloaded = store.get(uri, version_id=written.version_id)
    except Exception as error:
        raise CanaryError("S3 canary object download failed") from error
    if (
        not isinstance(downloaded, ObjectRead)
        or downloaded.payload != payload
        or downloaded.checksum_sha256 != digest
        or downloaded.version_id != written.version_id
        or hashlib.sha256(downloaded.payload).hexdigest() != digest
    ):
        raise CanaryError("S3 object byte, checksum, or version mismatch")
    return {
        "bytes": len(payload),
        "object_uri": uri,
        "sha256": digest,
        "version_id": written.version_id,
    }


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CanaryError("time reader returned a naive timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _atomic_no_replace(path: Path, payload: bytes) -> None:
    parent = path.parent
    temporary = parent / (
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    installed = False
    completed = False
    staged_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise CanaryError("short write while staging canary receipt")
            written += count
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        staged_metadata = temporary.stat(follow_symlinks=False)
        staged_identity = (staged_metadata.st_dev, staged_metadata.st_ino)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise CanaryError(
                "qualification receipt exists; no replacement is allowed"
            ) from error
        installed = True
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        metadata = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or path.read_bytes() != payload
        ):
            raise CanaryError("installed qualification receipt is unsafe")
        completed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if installed and not completed and staged_identity is not None:
            try:
                installed_metadata = path.stat(follow_symlinks=False)
                installed_identity = (
                    installed_metadata.st_dev,
                    installed_metadata.st_ino,
                )
                if installed_identity == staged_identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if not installed and path.is_symlink():
            raise CanaryError("qualification output was replaced by a symlink")


class SystemTimeReader:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


def execute_canary(
    plan: CanaryPlan,
    *,
    command_reader: CommandReader,
    object_store: ObjectStore,
    time_reader: TimeReader,
    apply: bool,
) -> dict[str, object]:
    if type(apply) is not bool:
        raise CanaryError("apply must be boolean")
    _validate_private_root(plan.scratch_root, plan.output_path)
    if plan.output_path.exists() or plan.output_path.is_symlink():
        raise CanaryError("qualification receipt exists; no replacement is allowed")
    specs = _phase_specs(plan)
    for spec in specs["nccl_all_reduce"]:
        Path(spec.host_work_root).mkdir(
            parents=True,
            mode=0o700,
            exist_ok=False,
        )
    started_at = _timestamp(time_reader.now())
    started_monotonic = time_reader.monotonic()
    phase_records: list[dict[str, object]] = []
    values: dict[str, object] = {}

    for phase in PHASE_ORDER:
        phase_started = time_reader.monotonic()
        if phase == "hardware":
            value = _hardware_phase(plan, command_reader, specs[phase])
        elif phase == "nccl_all_reduce":
            value = _nccl_phase(command_reader, specs[phase][0])
        elif phase == "functional":
            value = _functional_or_resume_phase(
                plan,
                command_reader,
                specs[phase],
                phase="functional",
            )
        elif phase == "resume":
            value = _functional_or_resume_phase(
                plan,
                command_reader,
                specs[phase],
                phase="resume",
                prior=values["functional"],
            )
        elif phase == "throughput_4x4":
            value = _throughput_phase(plan, command_reader, specs[phase])
        else:
            value = _s3_phase(plan, object_store)
        seconds = time_reader.monotonic() - phase_started
        if not math.isfinite(seconds) or seconds < 0:
            raise CanaryError("phase timer returned invalid elapsed time")
        phase_records.append(
            {
                "name": phase,
                "passed": True,
                "seconds": float(seconds),
            }
        )
        values[phase] = value

    total_seconds = time_reader.monotonic() - started_monotonic
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise CanaryError("canary timer returned invalid total time")
    ended_at = _timestamp(time_reader.now())
    receipt = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "provider": PROVIDER,
        "instance_id": plan.instance_id,
        "boot_id": plan.boot_id,
        "profile_sha256": plan.profile_sha256,
        "runtime_lock_sha256": plan.runtime_lock_sha256,
        "environment_receipt_sha256": plan.environment_receipt_sha256,
        "release_sha256": plan.release_sha256,
        "release_receipt_sha256": plan.release_receipt_sha256,
        "run_manifest_sha256": plan.run_manifest_sha256,
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "dataset_build_id": plan.dataset_build_id,
        "ordered_stream_sha256": plan.ordered_stream_sha256,
        "source_commit": plan.source_commit,
        "source_tree": plan.source_tree,
        "container_image": plan.container_image,
        "container_image_digest": plan.container_image_digest,
        "seed": plan.seed,
        "phases": phase_records,
        "hardware": values["hardware"],
        "functional": values["functional"],
        "resume": values["resume"],
        "throughput_4x4": values["throughput_4x4"],
        "s3_roundtrip": values["s3_roundtrip"],
        "passed": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_seconds": float(total_seconds),
    }
    payload = canonical_receipt(receipt)
    if apply:
        _atomic_no_replace(plan.output_path, payload)
    return receipt


class SubprocessCommandReader:
    def run(self, spec: CommandSpec) -> CommandResult:
        try:
            completed = subprocess.run(
                list(spec.argv),
                cwd=spec.cwd,
                env=dict(spec.environment),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=spec.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CanaryError(f"{spec.name} command is unavailable") from error
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def run_pair(
        self,
        specs: Sequence[CommandSpec],
    ) -> Mapping[str, CommandResult]:
        if len(specs) != 2 or len({spec.name for spec in specs}) != 2:
            raise CanaryError("throughput requires exactly two distinct commands")
        processes: dict[str, tuple[subprocess.Popen, object, object]] = {}
        deadline = time.monotonic() + max(spec.timeout_seconds for spec in specs)
        try:
            for spec in specs:
                stdout = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
                stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
                process = subprocess.Popen(
                    list(spec.argv),
                    cwd=spec.cwd,
                    env=dict(spec.environment),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    shell=False,
                    start_new_session=True,
                )
                processes[spec.name] = (process, stdout, stderr)
            failed = False
            while True:
                statuses = {
                    name: process.poll()
                    for name, (process, _stdout, _stderr) in processes.items()
                }
                if any(status not in (None, 0) for status in statuses.values()):
                    failed = True
                    break
                if all(status == 0 for status in statuses.values()):
                    break
                if time.monotonic() >= deadline:
                    failed = True
                    break
                time.sleep(0.1)
            if failed:
                for process, _stdout, _stderr in processes.values():
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                stop_deadline = time.monotonic() + 10.0
                for process, _stdout, _stderr in processes.values():
                    if process.poll() is None:
                        try:
                            process.wait(
                                timeout=max(0.001, stop_deadline - time.monotonic())
                            )
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
            results = {}
            for name, (process, stdout, stderr) in processes.items():
                if process.poll() is None:
                    process.wait(timeout=2.0)
                stdout.seek(0)
                stderr.seek(0)
                results[name] = CommandResult(
                    returncode=int(process.returncode),
                    stdout=stdout.read(),
                    stderr=stderr.read(),
                )
            return results
        finally:
            for process, stdout, stderr in processes.values():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                stdout.close()
                stderr.close()


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise CanaryError("S3 object URI is invalid")
    return parsed.netloc, parsed.path.lstrip("/")


class AwsCliObjectStore:
    def __init__(
        self,
        *,
        reader: CommandReader,
        region: str,
        scratch_root: Path,
    ) -> None:
        self.reader = reader
        self.region = region
        self.scratch_root = scratch_root
        self.environment = {
            **dict(MINIMAL_COMMAND_ENVIRONMENT),
            "AWS_REGION": region,
            "HOME": str(scratch_root / "aws-home"),
        }
        Path(self.environment["HOME"]).mkdir(mode=0o700, exist_ok=False)

    def put(
        self,
        uri: str,
        payload: bytes,
        *,
        sha256: str,
    ) -> ObjectWrite:
        bucket, key = _s3_parts(uri)
        body = self.scratch_root / f"s3-put-{sha256}.bin"
        body.write_bytes(payload)
        body.chmod(0o600)
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        result = _checked(
            self.reader,
            CommandSpec(
                name="s3-roundtrip-put",
                argv=(
                    "/usr/local/bin/aws",
                    "--no-cli-pager",
                    "--region",
                    self.region,
                    "s3api",
                    "put-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--body",
                    str(body),
                    "--content-length",
                    str(len(payload)),
                    "--checksum-algorithm",
                    "SHA256",
                    "--checksum-sha256",
                    checksum,
                    "--if-none-match",
                    "*",
                    "--output",
                    "json",
                    "--query",
                    "{checksum:ChecksumSHA256,version_id:VersionId}",
                ),
                environment=self.environment,
                cwd=self.scratch_root,
                timeout_seconds=120.0,
            ),
        )
        value = _parse_complete_json_object(
            result.stdout,
            label="S3 put-object",
            fields={"checksum", "version_id"},
        )
        returned_version = value["version_id"]
        if (
            value["checksum"] != checksum
            or not isinstance(returned_version, str)
            or returned_version in {"", "null"}
        ):
            raise CanaryError("S3 put-object checksum or version differs")
        return ObjectWrite(
            checksum_sha256=sha256,
            byte_count=len(payload),
            version_id=returned_version,
        )

    def get(self, uri: str, *, version_id: str) -> ObjectRead:
        bucket, key = _s3_parts(uri)
        destination = self.scratch_root / f"s3-get-{secrets.token_hex(8)}.bin"
        result = _checked(
            self.reader,
            CommandSpec(
                name="s3-roundtrip-get",
                argv=(
                    "/usr/local/bin/aws",
                    "--no-cli-pager",
                    "--region",
                    self.region,
                    "s3api",
                    "get-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--version-id",
                    version_id,
                    "--checksum-mode",
                    "ENABLED",
                    str(destination),
                    "--output",
                    "json",
                    "--query",
                    "{checksum:ChecksumSHA256,version_id:VersionId}",
                ),
                environment=self.environment,
                cwd=self.scratch_root,
                timeout_seconds=120.0,
            ),
        )
        value = _parse_complete_json_object(
            result.stdout,
            label="S3 get-object",
            fields={"checksum", "version_id"},
        )
        payload = destination.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected_checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        returned_version = value["version_id"]
        if (
            value["checksum"] != expected_checksum
            or not isinstance(returned_version, str)
            or returned_version != version_id
            or returned_version in {"", "null"}
        ):
            raise CanaryError("S3 get-object checksum or version differs")
        return ObjectRead(
            payload=payload,
            checksum_sha256=digest,
            version_id=returned_version,
        )


def _inspect_checkpoint(checkpoint: Path, config_path: Path) -> dict[str, object]:
    import torch
    import yaml

    from train.model import GPTConfig, PRESETS
    from train.trainer import resume_config_fingerprint

    checkpoint_bytes = read_regular_input(
        checkpoint,
        label="canary checkpoint",
        maximum_bytes=64 * 1024 * 1024 * 1024,
    )
    config_bytes = read_regular_input(
        config_path,
        label="canary config",
        maximum_bytes=1024 * 1024,
    )
    try:
        state = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=False,
        )
        config = yaml.safe_load(config_bytes.decode("utf-8"))
    except Exception as error:
        raise CanaryError("canary checkpoint cannot be inspected") from error
    required = {
        "checkpoint_version",
        "cfg",
        "config_fingerprint",
        "data",
        "data_provenance",
        "model",
        "opt",
        "rng_by_rank",
        "step",
        "world_size",
    }
    if type(state) is not dict or set(state) != required or type(config) is not dict:
        raise CanaryError("canary checkpoint schema is invalid")
    raw_model = config.get("model")
    if isinstance(raw_model, str):
        if raw_model not in PRESETS:
            raise CanaryError("canary config model is unknown")
        model_config = replace(PRESETS[raw_model])
    elif type(raw_model) is dict:
        model_config = GPTConfig(**raw_model)
    else:
        raise CanaryError("canary config model is invalid")
    if "ctx" in config:
        model_config = replace(model_config, ctx=config["ctx"])
    expected_fingerprint = resume_config_fingerprint(config, model_config)
    provenance = state.get("data_provenance")
    data = state.get("data")
    if type(provenance) is not dict or type(data) is not dict:
        raise CanaryError("canary checkpoint data identity is invalid")
    evidence = {
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "config_fingerprint": state.get("config_fingerprint"),
        "config_matches": (
            state.get("cfg") == config
            and state.get("config_fingerprint") == expected_fingerprint
        ),
        "data": {
            "build_id": provenance.get("build_id"),
            "global_cursor": data.get("global_cursor"),
            "ordered_stream_sha256": provenance.get("ordered_stream_sha256"),
            "receipt_sha256": provenance.get("receipt_sha256"),
            "sidecar_name": provenance.get("sidecar_name"),
        },
        "step": state.get("step"),
        "world_size": state.get("world_size"),
    }
    _exact_fields(evidence, _CHECKPOINT_EVIDENCE_FIELDS, label="checkpoint evidence")
    return evidence


def _nccl_worker() -> int:
    import torch

    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 8:
        raise CanaryError("NCCL worker requires exactly eight ranks")
    torch.cuda.set_device(rank)
    value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
    torch.cuda.synchronize(rank)
    started = time.perf_counter()
    torch.distributed.all_reduce(value)
    torch.cuda.synchronize(rank)
    elapsed = time.perf_counter() - started
    maximum = torch.tensor(elapsed, dtype=torch.float64, device=f"cuda:{rank}")
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    if value.item() != 36.0 or not math.isfinite(maximum.item()):
        raise CanaryError("NCCL all-reduce result is invalid")
    if rank == 0:
        print(
            json.dumps(
                {
                    "latency_seconds": maximum.item(),
                    "reduced_sum": value.item(),
                    "world_size": world_size,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    torch.distributed.destroy_process_group()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root")
    parser.add_argument("--release-receipt")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-receipt")
    parser.add_argument("--environment-receipt")
    parser.add_argument("--runtime-lock")
    parser.add_argument("--instance-id")
    parser.add_argument("--boot-id")
    parser.add_argument("--scratch-root")
    parser.add_argument("--output")
    parser.add_argument("--s3-root")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--nccl-worker", action="store_true")
    parser.add_argument("--inspect-checkpoint")
    parser.add_argument("--config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.nccl_worker:
            return _nccl_worker()
        if arguments.inspect_checkpoint is not None:
            if arguments.config is None:
                raise CanaryError("--inspect-checkpoint requires --config")
            evidence = _inspect_checkpoint(
                Path(arguments.inspect_checkpoint),
                Path(arguments.config),
            )
            print(
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            return 0
        required = (
            "release_root",
            "release_receipt",
            "manifest",
            "dataset_receipt",
            "environment_receipt",
            "runtime_lock",
            "instance_id",
            "boot_id",
            "scratch_root",
            "output",
            "s3_root",
        )
        missing = [name for name in required if getattr(arguments, name) is None]
        if missing:
            raise CanaryError(
                "missing required canary inputs: "
                + ", ".join(name.replace("_", "-") for name in missing)
            )
        plan = load_canary_plan(
            release_root=arguments.release_root,
            release_receipt_path=arguments.release_receipt,
            run_manifest_path=arguments.manifest,
            dataset_receipt_path=arguments.dataset_receipt,
            environment_receipt_path=arguments.environment_receipt,
            runtime_lock_path=arguments.runtime_lock,
            instance_id=arguments.instance_id,
            boot_id=arguments.boot_id,
            scratch_root=arguments.scratch_root,
            output_path=arguments.output,
            s3_root=arguments.s3_root,
        )
        if not arguments.apply:
            print(
                json.dumps(
                    render_plan(plan),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            return 0
        reader = SubprocessCommandReader()
        store = AwsCliObjectStore(
            reader=reader,
            region=str(
                parse_environment_receipt_bytes(
                    read_regular_input(
                        arguments.environment_receipt,
                        label="environment receipt",
                    )
                )["region"]
            ),
            scratch_root=plan.scratch_root,
        )
        receipt = execute_canary(
            plan,
            command_reader=reader,
            object_store=store,
            time_reader=SystemTimeReader(),
            apply=True,
        )
        sys.stdout.buffer.write(canonical_receipt(receipt))
        return 0
    except (CanaryError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "ok": False,
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
