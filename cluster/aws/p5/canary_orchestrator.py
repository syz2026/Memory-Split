#!/usr/bin/env python3
"""Execute one immutable, release-mounted AWS GPU canary plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.canary import (
    QualificationError,
    build_qualification_receipt,
    render_canary_command_plan,
    validate_concurrent_overlap,
)
from cluster.aws.p5.canary_runtime import validate_phase_receipt
from cluster.aws.p5.interruption_checkpoint import ImdsV2Client, S3ObjectStore
from cluster.aws.p5.profile import (
    AwsGpuRuntime,
    load_aws_gpu_profile,
    validate_runtime_environment,
)
from msctl.aws_identity import (
    AwsIdentityError,
    verify_environment_receipt,
    verify_instance_identity_pkcs7,
)
from msctl.jsonutil import canonical_json


ORCHESTRATION_PLAN_TYPE = "memorysplit-aws-gpu-canary-orchestration-v3"
_PLAN_FIELDS = {
    "schema_version",
    "plan_type",
    "instance_id",
    "profile",
    "release_sha256",
    "release_root",
    "provider_selection",
    "environment_receipt",
    "environment_receipt_sha256",
    "command_plan",
    "command_plan_sha256",
    "qualification_receipt_uri",
    "s3_kms_key_id",
}
_PROFILE_FIELDS = {
    "profile_id",
    "provider",
    "profile_sha256",
    "instance_type",
    "gres",
    "runtime_uid",
    "runtime_gid",
}
_SELECTION_FIELDS = {
    "provider_selection_sha256",
    "region",
    "aws_account_id",
    "ami_id",
    "ami_owner_id",
    "container_image",
    "container_digest",
}
_PHASE_ORDER = (
    "prepare",
    "serial",
    "one_step_training",
    "checkpoint_resume",
    "throughput",
)
_CONCURRENT_PHASE_NAMES = {"one_step_training", "throughput"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_MAX_PLAN_BYTES = 4 * 1024 * 1024
_COMMAND_TIMEOUT_SECONDS = 7_200.0


class CanaryOrchestrationError(ValueError):
    """The immutable plan, current runtime, or command evidence is invalid."""


def _canonical_line(value: object) -> bytes:
    return canonical_json(value) + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CanaryOrchestrationError(f"{label} fields do not match")
    return dict(value)


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CanaryOrchestrationError(f"{label} must be lowercase SHA-256")
    return value


def _s3_uri(value: object) -> str:
    if not isinstance(value, str):
        raise CanaryOrchestrationError("qualification URI must be a string")
    parsed = urlsplit(value)
    key = parsed.path.removeprefix("/")
    if (
        parsed.scheme != "s3"
        or parsed.hostname is None
        or parsed.netloc != parsed.hostname
        or parsed.query
        or parsed.fragment
        or not key
        or "\\" in value
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise CanaryOrchestrationError(
            "qualification URI must be one safe S3 object"
        )
    return value


def build_orchestration_plan(
    *,
    profile: object,
    runtime: AwsGpuRuntime,
    release_sha256: str,
    instance_id: str,
    provider_selection: Mapping[str, object],
    environment_receipt_bytes: bytes,
) -> dict[str, object]:
    """Build the sole canonical orchestration envelope used by msctl."""

    release_digest = _digest(release_sha256, label="release")
    if _INSTANCE_RE.fullmatch(instance_id) is None:
        raise CanaryOrchestrationError("canary requires one explicit instance ID")
    selection = _object(
        dict(provider_selection),
        _SELECTION_FIELDS,
        label="provider selection binding",
    )
    for field in ("provider_selection_sha256",):
        _digest(selection[field], label=field)
    if (
        selection["region"] != runtime.region
        or selection["ami_id"] != runtime.ami_id
        or selection["container_image"] != runtime.container_image
        or selection["container_digest"] != runtime.container_digest
    ):
        raise CanaryOrchestrationError(
            "provider selection differs from the selected runtime"
        )
    try:
        verified = verify_environment_receipt(
            environment_receipt_bytes,
            expected_profile_sha256=str(getattr(profile, "sha256", "")),
            expected_container_digest=runtime.container_digest,
            expected_region=runtime.region,
            expected_ami_id=runtime.ami_id,
            expected_account_id=str(selection["aws_account_id"]),
            expected_instance_id=instance_id,
        )
    except AwsIdentityError as error:
        raise CanaryOrchestrationError(
            "canary environment evidence is not authenticated"
        ) from error
    identity = verified.receipt["aws_instance_identity_document"]
    if (
        not isinstance(identity, dict)
        or identity.get("instanceType") != getattr(profile, "instance_type", None)
    ):
        raise CanaryOrchestrationError(
            "canary environment instance type is cross-profile"
        )
    release_root = (
        f"{getattr(profile, 'scratch_root')}/releases/{release_digest}"
    )
    command_plan = render_canary_command_plan(
        profile,
        runtime,
        release_sha256=release_digest,
        release_root=release_root,
    )
    command_plan_sha256 = hashlib.sha256(canonical_json(command_plan)).hexdigest()
    qualification_uri = (
        f"{runtime.s3_root.rstrip('/')}/qualification/"
        f"{getattr(profile, 'profile_id')}/{instance_id}/"
        f"environment-{verified.receipt_sha256}/"
        f"plan-{command_plan_sha256}.json"
    )
    return {
        "schema_version": 3,
        "plan_type": ORCHESTRATION_PLAN_TYPE,
        "instance_id": instance_id,
        "profile": {
            "profile_id": getattr(profile, "profile_id"),
            "provider": getattr(profile, "provider"),
            "profile_sha256": getattr(profile, "sha256"),
            "instance_type": getattr(profile, "instance_type"),
            "gres": getattr(profile, "gres"),
            "runtime_uid": runtime.uid,
            "runtime_gid": runtime.gid,
        },
        "release_sha256": release_digest,
        "release_root": release_root,
        "provider_selection": selection,
        "environment_receipt": dict(verified.receipt),
        "environment_receipt_sha256": verified.receipt_sha256,
        "command_plan": command_plan,
        "command_plan_sha256": command_plan_sha256,
        "qualification_receipt_uri": qualification_uri,
        "s3_kms_key_id": runtime.kms_key_id,
    }


def validate_orchestration_plan(
    value: object,
    *,
    verify_release_mount: bool = False,
) -> tuple[dict[str, object], object, AwsGpuRuntime]:
    """Validate and independently rerender one closed orchestration plan."""

    plan = _object(value, _PLAN_FIELDS, label="canary orchestration plan")
    profile_identity = _object(
        plan["profile"],
        _PROFILE_FIELDS,
        label="canary profile identity",
    )
    selection = _object(
        plan["provider_selection"],
        _SELECTION_FIELDS,
        label="canary provider selection",
    )
    release_sha256 = _digest(plan["release_sha256"], label="canary release")
    _digest(
        plan["environment_receipt_sha256"],
        label="canary environment receipt",
    )
    command_plan_sha256 = _digest(
        plan["command_plan_sha256"],
        label="canary command plan",
    )
    _digest(
        selection["provider_selection_sha256"],
        label="provider selection",
    )
    instance_id = plan["instance_id"]
    release_root = plan["release_root"]
    if (
        plan["schema_version"] != 3
        or plan["plan_type"] != ORCHESTRATION_PLAN_TYPE
        or not isinstance(instance_id, str)
        or _INSTANCE_RE.fullmatch(instance_id) is None
        or not isinstance(release_root, str)
        or release_root
        != f"/mnt/memorysplit/releases/{release_sha256}"
        or profile_identity["profile_id"] != profile_identity["provider"]
        or selection["region"] not in {"us-east-1", "us-west-2"}
        or not isinstance(selection["aws_account_id"], str)
        or _ACCOUNT_RE.fullmatch(str(selection["aws_account_id"])) is None
        or not isinstance(selection["ami_owner_id"], str)
        or _ACCOUNT_RE.fullmatch(str(selection["ami_owner_id"])) is None
        or type(profile_identity["runtime_uid"]) is not int
        or type(profile_identity["runtime_gid"]) is not int
        or int(profile_identity["runtime_uid"]) <= 0
        or int(profile_identity["runtime_gid"]) <= 0
    ):
        raise CanaryOrchestrationError("canary plan identity is invalid")
    environment_bytes = _canonical_line(plan["environment_receipt"])
    if _sha256(environment_bytes) != plan["environment_receipt_sha256"]:
        raise CanaryOrchestrationError(
            "canary environment receipt hash does not match"
        )
    try:
        verified = verify_environment_receipt(
            environment_bytes,
            expected_profile_sha256=str(profile_identity["profile_sha256"]),
            expected_container_digest=str(selection["container_digest"]),
            expected_region=str(selection["region"]),
            expected_ami_id=str(selection["ami_id"]),
            expected_account_id=str(selection["aws_account_id"]),
            expected_instance_id=str(instance_id),
        )
    except AwsIdentityError as error:
        raise CanaryOrchestrationError(
            "canary environment receipt is not authenticated"
        ) from error
    identity = verified.receipt["aws_instance_identity_document"]
    if (
        not isinstance(identity, dict)
        or identity.get("instanceType") != profile_identity["instance_type"]
    ):
        raise CanaryOrchestrationError("canary environment is cross-profile")
    profile_path = (
        Path(str(release_root))
        / "cluster"
        / "profiles"
        / f"{profile_identity['profile_id']}.json"
    )
    if not verify_release_mount:
        profile_path = (
            Path(__file__).resolve().parents[3]
            / "cluster"
            / "profiles"
            / f"{profile_identity['profile_id']}.json"
        )
    try:
        profile = load_aws_gpu_profile(profile_path)
    except (OSError, TypeError, ValueError) as error:
        raise CanaryOrchestrationError(
            "release-mounted canary profile is invalid"
        ) from error
    if (
        profile.sha256 != profile_identity["profile_sha256"]
        or profile.provider != profile_identity["provider"]
        or profile.instance_type != profile_identity["instance_type"]
        or profile.gres != profile_identity["gres"]
    ):
        raise CanaryOrchestrationError(
            "release-mounted canary profile identity differs"
        )
    s3_root = _s3_uri(plan["qualification_receipt_uri"]).rsplit(
        "/qualification/",
        1,
    )[0]
    runtime_environment = {
        "AWS_REGION": str(selection["region"]),
        "MS_S3_ROOT": s3_root,
        "MS_AWS_AMI_ID": str(selection["ami_id"]),
        "MS_CONTAINER_IMAGE": str(selection["container_image"]),
        "MS_CONTAINER_DIGEST": str(selection["container_digest"]),
        "MS_RUNTIME_UID": str(profile_identity["runtime_uid"]),
        "MS_RUNTIME_GID": str(profile_identity["runtime_gid"]),
    }
    if plan["s3_kms_key_id"] is not None:
        runtime_environment["MS_S3_KMS_KEY_ID"] = str(plan["s3_kms_key_id"])
    try:
        runtime = validate_runtime_environment(profile, runtime_environment)
    except ValueError as error:
        raise CanaryOrchestrationError(
            "canary runtime environment is invalid"
        ) from error
    if (
        runtime.kms_key_id is None
        or f":{selection['aws_account_id']}:key/" not in runtime.kms_key_id
    ):
        raise CanaryOrchestrationError(
            "canary KMS key is outside the verified AWS account"
        )
    expected_command_plan = render_canary_command_plan(
        profile,
        runtime,
        release_sha256=release_sha256,
        release_root=str(release_root),
    )
    if (
        plan["command_plan"] != expected_command_plan
        or hashlib.sha256(canonical_json(plan["command_plan"])).hexdigest()
        != command_plan_sha256
    ):
        raise CanaryOrchestrationError(
            "embedded command plan is not the deterministic rendered plan"
        )
    if verify_release_mount:
        try:
            relative = Path(__file__).resolve(strict=True).relative_to(
                Path(str(release_root)).resolve(strict=True)
            )
        except (OSError, ValueError) as error:
            raise CanaryOrchestrationError(
                "orchestrator is not executing from the digest-named release"
            ) from error
        if relative.as_posix() != "cluster/aws/p5/canary_orchestrator.py":
            raise CanaryOrchestrationError(
                "orchestrator release path is not canonical"
            )
    return plan, profile, runtime


def _verify_current_environment(
    plan: Mapping[str, object],
    *,
    metadata_get: Callable[[str], str | None],
    boot_id_get: Callable[[], str],
) -> None:
    receipt = plan["environment_receipt"]
    if not isinstance(receipt, dict):
        raise CanaryOrchestrationError("environment receipt is not an object")
    current_document = metadata_get("dynamic/instance-identity/document")
    current_pkcs7 = metadata_get("dynamic/instance-identity/pkcs7")
    try:
        document = json.loads(str(current_document))
    except json.JSONDecodeError as error:
        raise CanaryOrchestrationError(
            "current IMDS identity document is invalid"
        ) from error
    receipt_pkcs7 = receipt["aws_instance_identity_pkcs7"]
    if (
        not isinstance(document, dict)
        or document != receipt["aws_instance_identity_document"]
        or not isinstance(current_pkcs7, str)
        or "".join(current_pkcs7.split())
        != "".join(str(receipt_pkcs7).split())
        or not verify_instance_identity_pkcs7(
            document,
            current_pkcs7,
            str(document.get("region", "")),
        )
        or boot_id_get().strip() != receipt["boot_id"]
    ):
        raise CanaryOrchestrationError(
            "current instance or boot differs from verified environment evidence"
        )


def _command_receipt_path(
    command: Sequence[str],
    *,
    output_root: Path,
) -> Path:
    try:
        container_path = command[command.index("--output") + 1]
    except (ValueError, IndexError) as error:
        raise CanaryOrchestrationError(
            "canary phase command has no receipt output"
        ) from error
    prefix = "/qualification/"
    if not container_path.startswith(prefix):
        raise CanaryOrchestrationError(
            "canary phase receipt is outside the qualification mount"
        )
    relative = container_path.removeprefix(prefix)
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise CanaryOrchestrationError("canary phase receipt path is unsafe")
    path = output_root.joinpath(*relative.split("/"))
    try:
        path.resolve(strict=False).relative_to(output_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CanaryOrchestrationError(
            "canary phase receipt escapes its output root"
        ) from error
    return path


def _arm(command: Sequence[str]) -> str | None:
    if "--arm" not in command:
        return None
    try:
        value = command[command.index("--arm") + 1]
    except IndexError as error:
        raise CanaryOrchestrationError("canary arm argument is incomplete") from error
    if value not in {"dense", "split90"}:
        raise CanaryOrchestrationError("canary arm argument is invalid")
    return value


def _read_phase_receipt(
    path: Path,
    *,
    stdout: bytes,
) -> tuple[dict[str, object], str]:
    try:
        before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_PLAN_BYTES
        ):
            raise CanaryOrchestrationError(
                "canary phase receipt is not one regular file"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_PLAN_BYTES:
                    raise CanaryOrchestrationError(
                        "canary phase receipt exceeds the size limit"
                    )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise CanaryOrchestrationError(
            "canary phase receipt cannot be read"
        ) from error
    payload = b"".join(chunks)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(payload) != after.st_size:
        raise CanaryOrchestrationError(
            "canary phase receipt changed while read"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
        receipt = validate_phase_receipt(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise CanaryOrchestrationError(
            "canary phase receipt is invalid"
        ) from error
    if payload != _canonical_line(receipt) or stdout != payload:
        raise CanaryOrchestrationError(
            "canary phase stdout and canonical receipt bytes differ"
        )
    return receipt, _sha256(payload)


def _record(
    command: Sequence[str],
    *,
    started: int,
    finished: int,
    stdout: bytes,
    stderr: bytes,
    receipt_sha256: str,
) -> dict[str, object]:
    return {
        "arm": _arm(command),
        "argv_sha256": hashlib.sha256(canonical_json(list(command))).hexdigest(),
        "receipt_sha256": receipt_sha256,
        "started_monotonic_ns": started,
        "finished_monotonic_ns": finished,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
    }


def _run_serial(
    commands: Sequence[Sequence[str]],
    *,
    output_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
    clock_ns: Callable[[], int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    receipts: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for command in commands:
        started = clock_ns()
        try:
            completed = runner(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
                env={
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                },
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CanaryOrchestrationError(
                "serial canary command could not execute"
            ) from error
        finished = clock_ns()
        stdout = bytes(completed.stdout or b"")
        stderr = bytes(completed.stderr or b"")
        if completed.returncode != 0:
            raise CanaryOrchestrationError("serial canary command failed")
        receipt, digest = _read_phase_receipt(
            _command_receipt_path(command, output_root=output_root),
            stdout=stdout,
        )
        receipts.append(receipt)
        records.append(
            _record(
                command,
                started=started,
                finished=finished,
                stdout=stdout,
                stderr=stderr,
                receipt_sha256=digest,
            )
        )
    return receipts, records


def _run_concurrent(
    commands: Sequence[Sequence[str]],
    *,
    output_root: Path,
    popen: Callable[..., subprocess.Popen[bytes]],
    clock_ns: Callable[[], int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    if len(commands) != 2:
        raise CanaryOrchestrationError(
            "concurrent canary phase requires exactly two commands"
        )
    processes: list[tuple[Sequence[str], Any, int]] = []
    for command in commands:
        started = clock_ns()
        try:
            process = popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env={
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                },
            )
        except OSError as error:
            for _, running, _ in processes:
                running.kill()
                running.communicate()
            raise CanaryOrchestrationError(
                "concurrent canary command could not start"
            ) from error
        processes.append((command, process, started))

    # Sampling both children only after both Popen calls proves an observed
    # overlap. If the first child exited while the second was starting, its
    # eventual communicate timestamp would otherwise overstate its lifetime.
    for _, process, _ in processes:
        if process.poll() is None:
            continue
        for _, running, _ in processes:
            if running.poll() is None:
                running.kill()
        for _, running, _ in processes:
            running.communicate()
        raise CanaryOrchestrationError(
            "commands marked concurrent did not actually overlap"
        )

    def communicate(item: tuple[Sequence[str], Any, int]):
        command, process, started = item
        try:
            stdout, stderr = process.communicate(
                timeout=_COMMAND_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise CanaryOrchestrationError(
                "concurrent canary command timed out"
            ) from error
        return (
            command,
            process,
            started,
            clock_ns(),
            bytes(stdout or b""),
            bytes(stderr or b""),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = list(executor.map(communicate, processes))
    receipts: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for command, process, started, finished, stdout, stderr in completed:
        if process.returncode != 0:
            raise CanaryOrchestrationError("concurrent canary command failed")
        receipt, digest = _read_phase_receipt(
            _command_receipt_path(command, output_root=output_root),
            stdout=stdout,
        )
        receipts.append(receipt)
        records.append(
            _record(
                command,
                started=started,
                finished=finished,
                stdout=stdout,
                stderr=stderr,
                receipt_sha256=digest,
            )
        )
    try:
        overlap = validate_concurrent_overlap(records)
    except QualificationError as error:
        raise CanaryOrchestrationError(
            "commands marked concurrent did not actually overlap"
        ) from error
    return receipts, records, overlap


def execute_orchestration_plan(
    value: object,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    metadata_get: Callable[[str], str | None] | None = None,
    boot_id_get: Callable[[], str] = lambda: Path(
        "/proc/sys/kernel/random/boot_id"
    ).read_text(encoding="ascii"),
    verify_release_mount: bool = True,
) -> tuple[dict[str, object], Path]:
    """Execute every rendered phase and return one provenance-closed receipt."""

    plan, profile, runtime = validate_orchestration_plan(
        value,
        verify_release_mount=verify_release_mount,
    )
    if metadata_get is None:
        metadata_get = ImdsV2Client().get
    _verify_current_environment(
        plan,
        metadata_get=metadata_get,
        boot_id_get=boot_id_get,
    )
    command_plan = plan["command_plan"]
    assert isinstance(command_plan, dict)
    output_root = Path(str(command_plan["output_root"]))
    phases = command_plan["phases"]
    if not isinstance(phases, dict) or tuple(phases) != _PHASE_ORDER:
        # Canonical JSON sorts object keys. Execute only the reviewed explicit
        # order, while accepting that decoded object insertion order is irrelevant.
        if not isinstance(phases, dict) or set(phases) != set(_PHASE_ORDER):
            raise CanaryOrchestrationError("canary phase inventory is not exact")
    prepare = phases["prepare"]
    if (
        not isinstance(prepare, dict)
        or prepare.get("concurrent") is not False
        or not isinstance(prepare.get("commands"), list)
        or len(prepare["commands"]) != 1
    ):
        raise CanaryOrchestrationError("canary prepare phase is invalid")
    try:
        completed_prepare = runner(
            list(prepare["commands"][0]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            env={
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CanaryOrchestrationError(
            "canary output preparation could not execute"
        ) from error
    if completed_prepare.returncode != 0:
        raise CanaryOrchestrationError("canary output preparation failed")
    for directory in (
        output_root,
        output_root / "receipts",
        output_root / "checkpoints",
        output_root / "resume",
    ):
        try:
            metadata = directory.stat(follow_symlinks=False)
        except OSError as error:
            raise CanaryOrchestrationError(
                "canary output directory is unavailable"
            ) from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != runtime.uid
            or metadata.st_gid != runtime.gid
        ):
            raise CanaryOrchestrationError(
                "canary output directory ownership or mode is unsafe"
            )
    phase_receipts: list[dict[str, object]] = []
    concurrent_evidence: dict[str, object] = {}
    for phase_name in _PHASE_ORDER[1:]:
        phase = phases[phase_name]
        if (
            not isinstance(phase, dict)
            or set(phase) != {"concurrent", "commands"}
            or not isinstance(phase["commands"], list)
        ):
            raise CanaryOrchestrationError(f"canary phase {phase_name} is invalid")
        if phase_name in _CONCURRENT_PHASE_NAMES:
            if phase["concurrent"] is not True:
                raise CanaryOrchestrationError(
                    f"canary phase {phase_name} must be concurrent"
                )
            receipts, records, overlap = _run_concurrent(
                phase["commands"],
                output_root=output_root,
                popen=popen,
                clock_ns=clock_ns,
            )
            concurrent_evidence[phase_name] = {
                "concurrent": True,
                "commands": records,
                "overlap_ns": overlap,
            }
        else:
            if phase["concurrent"] is not False:
                raise CanaryOrchestrationError(
                    f"canary phase {phase_name} must be serial"
                )
            receipts, _records = _run_serial(
                phase["commands"],
                output_root=output_root,
                runner=runner,
                clock_ns=clock_ns,
            )
        phase_receipts.extend(receipts)
    selection = plan["provider_selection"]
    environment = plan["environment_receipt"]
    assert isinstance(selection, dict) and isinstance(environment, dict)
    identity = environment["aws_instance_identity_document"]
    assert isinstance(identity, dict)
    qualified_at = now().astimezone(UTC).isoformat().replace("+00:00", "Z")
    provenance = {
        "instance_id": identity["instanceId"],
        "boot_id": environment["boot_id"],
        "region": identity["region"],
        "ami_id": identity["imageId"],
        "ami_owner_id": selection["ami_owner_id"],
        "release_sha256": plan["release_sha256"],
        "container_image": selection["container_image"],
        "container_digest": selection["container_digest"],
        "provider_selection_sha256": selection[
            "provider_selection_sha256"
        ],
        "environment_receipt_sha256": plan[
            "environment_receipt_sha256"
        ],
        "qualified_at": qualified_at,
        "command_plan_sha256": plan["command_plan_sha256"],
    }
    receipt = build_qualification_receipt(
        profile,
        phase_receipts,
        provenance=provenance,
        concurrent_phase_evidence=concurrent_evidence,
    )
    destination = output_root / "qualification.json"
    from cluster.aws.p5.canary_runtime import write_receipt

    write_receipt(destination, receipt)
    return receipt, destination


def _download_plan(
    uri: str,
    expected_sha256: str,
    *,
    region: str,
) -> dict[str, object]:
    _digest(expected_sha256, label="orchestration plan")
    parsed = urlsplit(_s3_uri(uri))
    descriptor, temporary_name = tempfile.mkstemp(prefix="canary-plan-")
    os.close(descriptor)
    path = Path(temporary_name)
    try:
        completed = subprocess.run(
            [
                "/usr/bin/env",
                "-i",
                f"AWS_REGION={region}",
                "HOME=/tmp",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "PATH=/usr/bin:/bin",
                "aws",
                "--no-cli-pager",
                "--region",
                region,
                "s3api",
                "get-object",
                "--bucket",
                str(parsed.hostname),
                "--key",
                parsed.path.removeprefix("/"),
                "--checksum-mode",
                "ENABLED",
                str(path),
                "--output",
                "json",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            env={},
            timeout=300,
        )
        payload = path.read_bytes()
    finally:
        path.unlink(missing_ok=True)
    if (
        completed.returncode != 0
        or len(payload) > _MAX_PLAN_BYTES
        or _sha256(payload) != expected_sha256
    ):
        raise CanaryOrchestrationError(
            "immutable orchestration plan download failed verification"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryOrchestrationError(
            "orchestration plan is not valid JSON"
        ) from error
    if payload != _canonical_line(value):
        raise CanaryOrchestrationError(
            "orchestration plan is not canonical JSON"
        )
    return value


def _publish_qualification(
    path: Path,
    uri: str,
    *,
    expected_sha256: str,
    region: str,
    kms_key_id: str | None,
) -> None:
    home = path.parent / ".aws-home"
    home.mkdir(mode=0o700)
    os.chmod(home, 0o700)
    if any(home.iterdir()):
        raise CanaryOrchestrationError(
            "qualification publisher HOME is not empty"
        )
    store = S3ObjectStore(
        region=region,
        kms_key_id=kms_key_id,
        environment={
            "AWS_REGION": region,
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )
    uploaded = store.put_verified(
        path,
        uri,
        expected_sha256=expected_sha256,
        deadline=time.monotonic() + 300.0,
        monotonic=time.monotonic,
    )
    if uploaded is None:
        raise CanaryOrchestrationError(
            "qualification receipt was not durably verified in S3"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-uri", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--region", required=True, choices=("us-east-1", "us-west-2"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        plan = _download_plan(
            arguments.plan_uri,
            arguments.plan_sha256,
            region=arguments.region,
        )
        receipt, path = execute_orchestration_plan(plan)
        payload = _canonical_line(receipt)
        digest = _sha256(payload)
        _publish_qualification(
            path,
            str(plan["qualification_receipt_uri"]),
            expected_sha256=digest,
            region=arguments.region,
            kms_key_id=(
                str(plan["s3_kms_key_id"])
                if plan["s3_kms_key_id"] is not None
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "schema_version": 3,
                    "ok": True,
                    "qualification_receipt_sha256": digest,
                    "qualification_receipt_uri": plan[
                        "qualification_receipt_uri"
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (
        CanaryOrchestrationError,
        QualificationError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_version": 3,
                    "ok": False,
                    "error": type(error).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
