"""Dry-run-first planning primitives for AWS GPU qualification canaries."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

from cluster.aws.p5.canary_orchestrator import (
    ORCHESTRATION_PLAN_TYPE,
    build_orchestration_plan,
    validate_orchestration_plan,
)

from .aws_argv import (
    ARGV_DOCUMENT_NAME,
    ARGV_DOCUMENT_SHA256,
    CANARY_INTENT_TYPE,
)
from .errors import MsctlError
from .fsutil import open_directory, rename_noreplace_at
from .jsonutil import canonical_json, canonical_sha256, require_sha256


_MAX_PLAN_BYTES = 4 * 1024 * 1024


def _fail(message: str, *, code: str = "CANARY_PLAN_INVALID") -> None:
    raise MsctlError(code, message)


def _read_plan(path: Path) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_PLAN_BYTES
        ):
            _fail("canary plan must be one bounded singly linked regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise MsctlError(
            "CANARY_PLAN_INVALID",
            "canary plan cannot be read safely",
        ) from error
    data = b"".join(chunks)
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
    ) or len(data) != after.st_size:
        _fail("canary plan changed while being read")
    return data


def write_canary_plan(path: Path | str, value: Mapping[str, object]) -> Path:
    destination = Path(path)
    directory_fd = open_directory(
        destination.parent,
        label="canary plan output",
        create=True,
    )
    temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        payload = canonical_json(dict(value)) + b"\n"
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            rename_noreplace_at(
                directory_fd,
                temporary,
                directory_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise MsctlError(
                "CANARY_PLAN_EXISTS",
                "refusing to replace an existing canary plan",
            ) from error
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return destination


def create_canary_plan(
    *,
    profile: object,
    runtime: object,
    release: object,
    selection: object,
    environment_receipt: Path | str,
    instance_id: str,
) -> dict[str, object]:
    release_metadata = getattr(release, "metadata", {})
    release_profile = (
        release_metadata.get("profile")
        if isinstance(release_metadata, dict)
        else None
    )
    if (
        getattr(profile, "schema_version", None) != 3
        or getattr(release, "provider", None) != getattr(profile, "provider", None)
        or (
            isinstance(release_profile, dict)
            and release_profile.get("sha256") != getattr(profile, "sha256", None)
        )
        or getattr(selection, "provider", None)
        != getattr(profile, "provider", None)
        or getattr(selection, "profile_sha256", None)
        != getattr(profile, "sha256", None)
        or getattr(selection, "region", None) != getattr(runtime, "region", None)
        or getattr(selection, "ami_id", None) != getattr(runtime, "ami_id", None)
        or getattr(selection, "container_image", None)
        != getattr(runtime, "container_image", None)
        or getattr(selection, "container_digest", None)
        != getattr(runtime, "container_digest", None)
    ):
        _fail("canary inputs are stale or cross-profile")
    environment_path = Path(environment_receipt)
    environment_bytes = _read_plan(environment_path)
    try:
        return build_orchestration_plan(
            profile=profile,
            runtime=runtime,
            release_sha256=str(getattr(release, "archive_sha256", "")),
            instance_id=instance_id,
            provider_selection={
                "provider_selection_sha256": getattr(selection, "sha256"),
                "region": getattr(selection, "region"),
                "aws_account_id": getattr(selection, "aws_account_id"),
                "ami_id": getattr(selection, "ami_id"),
                "ami_owner_id": getattr(selection, "ami_owner_id"),
                "container_image": getattr(selection, "container_image"),
                "container_digest": getattr(selection, "container_digest"),
            },
            environment_receipt_bytes=environment_bytes,
        )
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "CANARY_PLAN_INVALID",
            "canary plan could not be bound to verified environment evidence",
        ) from error


def plan_canary(
    *,
    out: Path | str,
    apply: bool,
    **kwargs: object,
) -> dict[str, object]:
    value = create_canary_plan(**kwargs)
    payload = canonical_json(value) + b"\n"
    result = {
        "schema_version": 3,
        "plan_type": ORCHESTRATION_PLAN_TYPE,
        "plan": value,
        "plan_sha256": hashlib.sha256(payload).hexdigest(),
        "command_plan_sha256": value["command_plan_sha256"],
        "instance_id": value["instance_id"],
        "qualification_receipt_uri": value["qualification_receipt_uri"],
        "out": str(Path(out)),
        "published": False,
    }
    if apply:
        write_canary_plan(out, value)
        result["published"] = True
    return result


def load_canary_plan(
    path: Path | str,
    *,
    profile: object,
    runtime: object,
    expected_instance_id: str,
) -> tuple[dict[str, object], str]:
    candidate = Path(path)
    data = _read_plan(candidate)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "CANARY_PLAN_INVALID",
            "canary plan must contain valid UTF-8 JSON",
        ) from error
    if data != canonical_json(value) + b"\n":
        _fail("canary plan must be canonical JSON plus one newline")
    try:
        plan, loaded_profile, loaded_runtime = validate_orchestration_plan(
            value,
            verify_release_mount=False,
        )
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "CANARY_PLAN_INVALID",
            "canary plan failed closed validation",
        ) from error
    if (
        plan["instance_id"] != expected_instance_id
        or getattr(loaded_profile, "sha256", None)
        != getattr(profile, "sha256", None)
        or loaded_runtime.region != getattr(runtime, "region", None)
        or loaded_runtime.ami_id != getattr(runtime, "ami_id", None)
        or loaded_runtime.container_image
        != getattr(runtime, "container_image", None)
        or loaded_runtime.container_digest
        != getattr(runtime, "container_digest", None)
        or loaded_runtime.uid != getattr(runtime, "uid", None)
        or loaded_runtime.gid != getattr(runtime, "gid", None)
    ):
        _fail("canary plan is bound to another instance or runtime")
    return plan, hashlib.sha256(data).hexdigest()


def canary_plan_uri(s3_root: str, plan_sha256: str) -> str:
    digest = require_sha256(plan_sha256, label="canary plan")
    return (
        f"{s3_root.rstrip('/')}/qualification/plans/sha256/{digest}.json"
    )


def build_canary_intent(
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    plan_uri: str,
    runtime: object,
    control_bundle_sha256: str,
) -> dict[str, object]:
    """Bind one exact release-mounted orchestrator argv to one instance."""

    control_digest = require_sha256(
        control_bundle_sha256,
        label="canary control bundle",
    )
    plan_digest = require_sha256(plan_sha256, label="canary plan")
    if (
        hashlib.sha256(canonical_json(dict(plan)) + b"\n").hexdigest()
        != plan_digest
    ):
        _fail("canary plan bytes changed after validation")
    expected_plan_uri = canary_plan_uri(
        str(getattr(runtime, "s3_root")),
        plan_digest,
    )
    if plan_uri != expected_plan_uri:
        _fail("canary plan URI is not the deterministic content address")
    profile = plan["profile"]
    selection = plan["provider_selection"]
    if not isinstance(profile, dict) or not isinstance(selection, dict):
        _fail("canary plan identities are invalid")
    release_root = str(plan["release_root"])
    step = {
        "name": "execute-canary-orchestration",
        "argv": [
            "/usr/bin/python3",
            f"{release_root}/cluster/aws/p5/canary_orchestrator.py",
            "--plan-uri",
            plan_uri,
            "--plan-sha256",
            plan_digest,
            "--region",
            str(selection["region"]),
        ],
    }
    core = {
        "schema_version": 3,
        "intent_type": CANARY_INTENT_TYPE,
        "operation": "canary",
        "provider": profile["provider"],
        "instance_type": profile["instance_type"],
        "profile_sha256": profile["profile_sha256"],
        "gres": profile["gres"],
        "instance_id": plan["instance_id"],
        "release_sha256": plan["release_sha256"],
        "provider_selection_sha256": selection[
            "provider_selection_sha256"
        ],
        "environment_receipt_sha256": plan[
            "environment_receipt_sha256"
        ],
        "command_plan_sha256": plan["command_plan_sha256"],
        "orchestration_plan_sha256": plan_digest,
        "orchestration_plan_uri": plan_uri,
        "qualification_receipt_uri": plan["qualification_receipt_uri"],
        "control_bundle_sha256": control_digest,
        "environment": {
            "AWS_REGION": getattr(runtime, "region"),
            "MS_AWS_AMI_ID": getattr(runtime, "ami_id"),
            "MS_CONTAINER_DIGEST": getattr(runtime, "container_digest"),
            "MS_RUNTIME_GID": str(getattr(runtime, "gid")),
            "MS_RUNTIME_UID": str(getattr(runtime, "uid")),
            "MS_S3_ROOT": getattr(runtime, "s3_root"),
        },
        "steps": [step],
    }
    operation_id = canonical_sha256(core)
    receipt_root = (
        f"{getattr(runtime, 's3_root').rstrip('/')}/operations/"
        f"{operation_id}/receipts"
    )
    return {
        **core,
        "operation_id": operation_id,
        "ssm_document": {
            "name": ARGV_DOCUMENT_NAME,
            "sha256": ARGV_DOCUMENT_SHA256,
        },
        "started_receipt_uri": f"{receipt_root}/started.json",
        "terminal_receipt_uri": f"{receipt_root}/terminal.json",
    }
