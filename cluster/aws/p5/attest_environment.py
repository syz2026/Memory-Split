#!/usr/bin/env python3
"""Produce one authenticated, content-bound AWS runtime receipt."""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import ipaddress
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.profile import (
    PROFILE_ID_V3,
    parse_aws_p5_profile_bytes,
)
from msctl.aws_contracts import (
    AWS_ENVIRONMENT_RECEIPT_V2_FIELDS,
    AWS_GPU_ATTESTATION_EVIDENCE_V1_FIELDS,
    AWS_RUNTIME_LOCK_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
    validate_digest_pinned_oci_image,
)


PROVIDER = "aws-p5.48xlarge"
PROFILE_ID = PROFILE_ID_V3
RECEIPT_TYPE = "memorysplit-aws-environment-v2"
GPU_EVIDENCE_TYPE = "memorysplit-aws-gpu-attestation-v1"
IMDS_DOCUMENT_PATH = "/latest/dynamic/instance-identity/document"
IMDS_PKCS7_PATH = "/latest/dynamic/instance-identity/pkcs7"
TRUSTED_PYTHON_BINARY = "/usr/bin/python3"
TRUSTED_COMMAND_WORKING_DIRECTORY = "/usr"
MINIMAL_COMMAND_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
}
VERSION_COMMANDS = {
    "python": (
        TRUSTED_PYTHON_BINARY,
        "-I",
        "-P",
        "-c",
        "import platform;print(platform.python_version())",
    ),
    "pytorch": (
        TRUSTED_PYTHON_BINARY,
        "-I",
        "-P",
        "-c",
        "import torch;print(torch.__version__)",
    ),
    "cuda": (
        TRUSTED_PYTHON_BINARY,
        "-I",
        "-P",
        "-c",
        "import torch;print(torch.version.cuda or '')",
    ),
    "cudnn": (
        TRUSTED_PYTHON_BINARY,
        "-I",
        "-P",
        "-c",
        "import torch;print(torch.backends.cudnn.version() or '')",
    ),
    "nccl": (
        TRUSTED_PYTHON_BINARY,
        "-I",
        "-P",
        "-c",
        (
            "import torch;"
            "v=torch.cuda.nccl.version();"
            "print('.'.join(map(str,v)) if isinstance(v,tuple) else (v or ''))"
        ),
    ),
    "nvidia_driver": (
        "/usr/bin/nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ),
    "fabric_manager": (
        "/usr/bin/nv-fabricmanager",
        "--version",
    ),
    "docker": (
        "/usr/bin/docker",
        "version",
        "--format",
        "{{.Server.Version}}",
    ),
    "nvidia_container_runtime": (
        "/usr/bin/nvidia-container-runtime",
        "--version",
    ),
    "aws_cli": (
        "/usr/local/bin/aws",
        "--version",
    ),
}
if tuple(VERSION_COMMANDS) != AWS_RUNTIME_VERSION_FIELDS:
    raise RuntimeError("runtime command fields differ from the shared AWS contract")
SELECTED_HOST_VERSION_COMMANDS = {
    "cuda": ("/usr/local/cuda/bin/nvcc", "--version"),
    "nvidia_driver": VERSION_COMMANDS["nvidia_driver"],
    "fabric_manager": VERSION_COMMANDS["fabric_manager"],
    "docker": VERSION_COMMANDS["docker"],
    "nvidia_container_runtime": VERSION_COMMANDS["nvidia_container_runtime"],
    "aws_cli": VERSION_COMMANDS["aws_cli"],
    "kernel": ("/usr/bin/uname", "--kernel-release"),
    "efa": ("/usr/bin/cat", "/opt/amazon/efa_installed_packages"),
    "ofi_nccl": (
        "/usr/bin/strings",
        "/opt/amazon/ofi-nccl/lib/libnccl-net.so",
    ),
}
_CONTAINER_FACT_FIELDS = {"python", "pytorch", "cuda", "cudnn", "nccl"}
_SELECTED_HOST_FACT_FIELDS = set(SELECTED_HOST_VERSION_COMMANDS)
_P6_PROFILE_ID = "aws-p6-b300.48xlarge-v3"
_P6_PROVIDER = "aws-p6-b300.48xlarge"
_P6_INSTANCE_TYPE = "p6-b300.48xlarge"
_P6_GPU_MODEL = "NVIDIA B300"
_P6_AMI_ID = "ami-0260c4d597dcc8641"
_P6_AMI_OWNER_ID = "898082745236"
_UNSUPPORTED_P6_FRAMEWORK_AMI = "ami-0b39828e6910b0bb8"
_P6_FLOORS = {
    "cuda": "13.0",
    "nvidia_driver": "580.0",
    "kernel": "6.1",
    "efa": "1.44.0",
    "ofi_nccl": "1.17.1",
}
_CONTAINER_FACT_SCRIPT = (
    "import json,platform,torch;"
    "v=torch.cuda.nccl.version();"
    "n='.'.join(map(str,v)) if isinstance(v,tuple) else str(v or '');"
    "print(json.dumps({"
    "'cuda':str(torch.version.cuda or ''),"
    "'cudnn':str(torch.backends.cudnn.version() or ''),"
    "'nccl':n,"
    "'python':platform.python_version(),"
    "'pytorch':str(torch.__version__)"
    "},sort_keys=True,separators=(',',':')))"
)

_LOCK_FIELDS = set(AWS_RUNTIME_LOCK_FIELDS)
_RECEIPT_FIELDS = set(AWS_ENVIRONMENT_RECEIPT_V2_FIELDS)
_GPU_EVIDENCE_FIELDS = set(AWS_GPU_ATTESTATION_EVIDENCE_V1_FIELDS)
_VERSION_FIELDS = set(AWS_RUNTIME_VERSION_FIELDS)
_IDENTITY_REQUIRED_FIELDS = {
    "accountId",
    "instanceId",
    "region",
    "imageId",
    "architecture",
    "privateIp",
}
_IDENTITY_OPTIONAL_FIELDS = {
    "availabilityZone",
    "billingProducts",
    "devpayProductCodes",
    "instanceType",
    "kernelId",
    "marketplaceProductCodes",
    "pendingTime",
    "ramdiskId",
    "version",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_VERSION_RE = re.compile(r"^[0-9]+(?:[._+-][0-9A-Za-z]+)*$")
_FLOATING_VERSION_MARKERS = frozenset(
    {
        "canary",
        "current",
        "dev",
        "edge",
        "head",
        "latest",
        "main",
        "master",
        "nightly",
        "rolling",
        "snapshot",
        "stable",
        "tip",
        "trunk",
        "unknown",
        "unstable",
    }
)


class AttestationError(ValueError):
    """The runtime cannot be bound to the reviewed environment contract."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise AttestationError("attestation value is not canonical JSON") from error


def canonical_receipt(receipt: Mapping[str, object]) -> bytes:
    """Serialize the exact v2 receipt as canonical JSON plus one newline."""

    if set(receipt) != _RECEIPT_FIELDS:
        raise AttestationError("environment receipt fields do not match schema v2")
    return _canonical_json(dict(receipt)) + b"\n"


def canonical_gpu_evidence(evidence: Mapping[str, object]) -> bytes:
    """Serialize closed profile-aware GPU evidence without changing legacy v2."""

    if set(evidence) != _GPU_EVIDENCE_FIELDS:
        raise AttestationError("GPU attestation evidence fields do not match schema v1")
    return _canonical_json(dict(evidence)) + b"\n"


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AttestationError(f"JSON repeats field: {key}")
        value[key] = item
    return value


def _decode_json(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                AttestationError(f"{label} contains non-finite {constant}")
            ),
        )
    except AttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise AttestationError(f"{label} must be a JSON object")
    return value


def _read_regular(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> bytes:
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate == Path("/") or not candidate.name:
        raise AttestationError(f"{label} path is invalid")
    parent_descriptor = _open_directory_nofollow(
        candidate.parent,
        label=f"{label} parent directory",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise AttestationError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise AttestationError(
                f"{label} must be one singly-linked regular file"
            )
        chunks: list[bytes] = []
        if before.st_size:
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise AttestationError(f"{label} changed while being read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise AttestationError(f"{label} changed while being read")
        else:
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, maximum_bytes - total + 1),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum_bytes:
                    raise AttestationError(f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
        try:
            current = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise AttestationError(f"{label} path changed while being read") from error
        identity = lambda details: (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
            details.st_nlink,
        )
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise AttestationError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def read_regular_input(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> bytes:
    """Read one stable, singly linked regular attestation input."""

    return _read_regular(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_nofollow(path: Path | str, *, label: str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current,
                )
            except OSError as error:
                raise AttestationError(
                    f"{label} contains a symlink or non-directory component"
                ) from error
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AttestationError(f"{label} must be lowercase SHA-256")
    return value


def _fixed_version(value: object, *, label: str) -> str:
    immutable_labels = {"alpha", "beta", "cpu", "final", "post", "rc"}
    segments = re.split(r"[._+-]", value.lower()) if isinstance(value, str) else []
    normalized = (
        re.sub(r"[^a-z0-9]", "", value.lower())
        if isinstance(value, str)
        else ""
    )
    if (
        not isinstance(value, str)
        or _VERSION_RE.fullmatch(value) is None
        or any(marker in normalized for marker in _FLOATING_VERSION_MARKERS)
        or any(
            not any(character.isdigit() for character in segment)
            and segment not in immutable_labels
            for segment in segments[1:]
        )
    ):
        raise AttestationError(f"{label} must be one non-floating version")
    return value


def _validate_image(image: object, digest: object) -> tuple[str, str]:
    try:
        return validate_digest_pinned_oci_image(image, digest)
    except ValueError as error:
        raise AttestationError(
            "container image must be a full registry reference pinned by digest"
        ) from error


def _validate_profile(data: bytes) -> str:
    try:
        profile = parse_aws_p5_profile_bytes(data)
    except (TypeError, ValueError) as error:
        raise AttestationError("AWS P5 profile is not the exact v3 profile") from error
    if profile.profile_id != PROFILE_ID:
        raise AttestationError("AWS P5 profile is not the exact v3 profile")
    return profile.sha256


def _validate_runtime_lock(data: bytes) -> dict[str, object]:
    lock = _decode_json(data, label="runtime lock")
    if data != _canonical_json(lock) + b"\n":
        raise AttestationError("runtime lock must use canonical JSON plus one newline")
    if (
        set(lock) != _LOCK_FIELDS
        or type(lock.get("schema_version")) is not int
        or lock["schema_version"] != 1
    ):
        raise AttestationError("runtime lock fields do not match schema version 1")
    if (
        not isinstance(lock["source_commit"], str)
        or _SHA1_RE.fullmatch(lock["source_commit"]) is None
        or not isinstance(lock["source_tree"], str)
        or _SHA1_RE.fullmatch(lock["source_tree"]) is None
        or not isinstance(lock["ami_id"], str)
        or _AMI_RE.fullmatch(lock["ami_id"]) is None
        or not isinstance(lock["ami_owner_id"], str)
        or re.fullmatch(r"[0-9]{12}", lock["ami_owner_id"]) is None
    ):
        raise AttestationError("runtime lock has malformed source or AMI identity")
    _sha256(lock["profile_sha256"], label="runtime lock profile")
    _sha256(lock["control_bundle_sha256"], label="runtime lock control bundle")
    image, digest = _validate_image(
        lock["container_image"],
        lock["container_image_digest"],
    )
    lock["container_image"] = image
    lock["container_image_digest"] = digest
    versions = lock["versions"]
    if not isinstance(versions, dict) or set(versions) != _VERSION_FIELDS:
        raise AttestationError("runtime lock version fields do not match schema")
    for field in VERSION_COMMANDS:
        _fixed_version(versions[field], label=f"runtime lock versions.{field}")
    return lock


def parse_runtime_lock_bytes(data: bytes) -> dict[str, object]:
    """Parse one canonical closed runtime-lock document."""

    if not isinstance(data, bytes):
        raise AttestationError("runtime lock input must be bytes")
    return _validate_runtime_lock(data)


def _validate_identity(
    data: bytes,
    *,
    expected_instance_type: str = "p5.48xlarge",
) -> dict[str, object]:
    identity = _decode_json(data, label="AWS instance identity document")
    fields = set(identity)
    if (
        not _IDENTITY_REQUIRED_FIELDS <= fields
        or not fields <= _IDENTITY_REQUIRED_FIELDS | _IDENTITY_OPTIONAL_FIELDS
        or not all(
        isinstance(identity[field], str) and identity[field]
        for field in _IDENTITY_REQUIRED_FIELDS
        )
    ):
        raise AttestationError("AWS instance identity fields are not exact strings")
    if (
        re.fullmatch(r"[0-9]{12}", str(identity["accountId"])) is None
        or _INSTANCE_RE.fullmatch(str(identity["instanceId"])) is None
        or _REGION_RE.fullmatch(str(identity["region"])) is None
        or _AMI_RE.fullmatch(str(identity["imageId"])) is None
        or identity["architecture"] not in {"x86_64", "arm64"}
    ):
        raise AttestationError("AWS instance identity contains malformed values")
    try:
        address = ipaddress.ip_address(str(identity["privateIp"]))
    except ValueError as error:
        raise AttestationError("AWS instance private IP is invalid") from error
    private_networks = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    if address.version != 4 or not any(
        address in network for network in private_networks
    ):
        raise AttestationError("AWS instance private IP is not private IPv4")
    if (
        "instanceType" in identity
        and identity["instanceType"] != expected_instance_type
    ):
        raise AttestationError("AWS instance identity has the wrong instance type")
    if (
        "availabilityZone" in identity
        and (
            not isinstance(identity["availabilityZone"], str)
            or not str(identity["availabilityZone"]).startswith(
                str(identity["region"])
            )
        )
    ):
        raise AttestationError("AWS instance availability zone is invalid")
    for field in (
        "billingProducts",
        "devpayProductCodes",
        "marketplaceProductCodes",
    ):
        if field in identity and (
            identity[field] is not None
            and (
                not isinstance(identity[field], list)
                or any(not isinstance(item, str) for item in identity[field])
            )
        ):
            raise AttestationError(f"AWS instance identity {field} is invalid")
    for field in ("kernelId", "pendingTime", "ramdiskId", "version"):
        if field in identity and (
            identity[field] is not None
            and (
                not isinstance(identity[field], str)
                or not identity[field]
            )
        ):
            raise AttestationError(f"AWS instance identity {field} is invalid")
    return identity


def _canonical_pkcs7(data: bytes) -> str:
    try:
        compact = b"".join(data.split()).decode("ascii")
        decoded = base64.b64decode(compact, validate=True)
    except (UnicodeDecodeError, ValueError) as error:
        raise AttestationError("AWS identity PKCS7 is not canonical base64") from error
    if not decoded or base64.b64encode(decoded).decode("ascii") != compact:
        raise AttestationError("AWS identity PKCS7 is not canonical base64")
    return compact


def parse_environment_receipt_bytes(data: bytes) -> dict[str, object]:
    """Parse one canonical environment receipt under the closed v2 schema."""

    if not isinstance(data, bytes):
        raise AttestationError("environment receipt input must be bytes")
    receipt = _decode_json(data, label="environment receipt")
    if data != canonical_receipt(receipt):
        raise AttestationError(
            "environment receipt must use canonical JSON plus one newline"
        )
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 2
        or receipt.get("receipt_type") != RECEIPT_TYPE
        or receipt.get("provider") != PROVIDER
    ):
        raise AttestationError("environment receipt identity is invalid")
    for field in (
        "profile_sha256",
        "runtime_lock_sha256",
        "control_bundle_sha256",
    ):
        _sha256(receipt[field], label=f"environment receipt {field}")
    for field in ("source_commit", "source_tree"):
        if (
            not isinstance(receipt[field], str)
            or _SHA1_RE.fullmatch(receipt[field]) is None
        ):
            raise AttestationError(f"environment receipt {field} is invalid")
    _validate_image(
        receipt["container_image"],
        receipt["container_image_digest"],
    )
    identity_value = receipt["aws_instance_identity_document"]
    if not isinstance(identity_value, dict):
        raise AttestationError("AWS instance identity document must be an object")
    identity = _validate_identity(_canonical_json(identity_value))
    pkcs7 = receipt["aws_instance_identity_pkcs7"]
    if (
        not isinstance(pkcs7, str)
        or _canonical_pkcs7(pkcs7.encode("ascii", errors="strict")) != pkcs7
    ):
        raise AttestationError("AWS identity PKCS7 is not canonical base64")
    if (
        receipt.get("account_id") != identity["accountId"]
        or receipt.get("instance_id") != identity["instanceId"]
        or receipt.get("region") != identity["region"]
        or receipt.get("ami_id") != identity["imageId"]
    ):
        raise AttestationError(
            "environment receipt identity duplicates are inconsistent"
        )
    boot_id = receipt["boot_id"]
    if not isinstance(boot_id, str) or _UUID_RE.fullmatch(boot_id) is None:
        raise AttestationError("environment receipt boot ID is invalid")
    facts = receipt["runtime_facts"]
    if not isinstance(facts, dict) or set(facts) != _VERSION_FIELDS:
        raise AttestationError(
            "environment receipt runtime fact fields do not match schema"
        )
    for field in VERSION_COMMANDS:
        _fixed_version(
            facts[field],
            label=f"environment receipt runtime_facts.{field}",
        )
    return receipt


def _command_version(field: str, data: bytes) -> str:
    try:
        lines = [
            line.strip()
            for line in data.decode("utf-8", errors="strict").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as error:
        raise AttestationError(f"{field} version output is not UTF-8") from error
    if not lines:
        raise AttestationError(f"{field} version command returned no value")
    first = lines[0]
    if field == "aws_cli" and first.startswith("aws-cli/"):
        first = first.split("/", 1)[1].split(None, 1)[0]
    elif field in {"fabric_manager", "nvidia_container_runtime"}:
        match = re.search(
            r"\bversion(?:\s+is)?\s*:?\s*([0-9][0-9A-Za-z._+-]*)\b",
            first,
            re.I,
        )
        if match is not None:
            first = match.group(1)
    return _fixed_version(first, label=f"runtime {field}")


def container_inspect_argv(image: str) -> tuple[str, ...]:
    """Return the exact local digest inspection argv."""

    return (
        "/usr/bin/docker",
        "image",
        "inspect",
        "--format",
        "{{json .RepoDigests}}",
        image,
    )


def container_facts_argv(image: str) -> tuple[str, ...]:
    """Measure framework facts inside the exact local image without networking."""

    return (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--gpus",
        "all",
        "--read-only",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--workdir",
        "/",
        "--entrypoint",
        "/usr/bin/python3",
        image,
        "-I",
        "-P",
        "-c",
        _CONTAINER_FACT_SCRIPT,
    )


def gpu_names_argv() -> tuple[str, ...]:
    """Measure every locally visible GPU product name on the host."""

    return (
        "/usr/bin/nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader,nounits",
    )


class UrllibImdsReader:
    """Minimal IMDSv2 reader with no ambient proxy or credential behavior."""

    _ROOT = "http://169.254.169.254"

    def _request(
        self,
        path: str,
        *,
        method: str,
        headers: Mapping[str, str],
    ) -> bytes:
        request = urllib.request.Request(
            self._ROOT + path,
            method=method,
            headers=dict(headers),
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=2) as response:
                return response.read(128 * 1024)
        except OSError as error:
            raise AttestationError("IMDSv2 request failed") from error

    def token(self) -> str:
        data = self._request(
            "/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        try:
            token = data.decode("ascii")
        except UnicodeDecodeError as error:
            raise AttestationError("IMDSv2 token is not ASCII") from error
        if not token or token != token.strip() or len(token) > 4096:
            raise AttestationError("IMDSv2 token is invalid")
        return token

    def read(self, path: str, *, token: str) -> bytes:
        if not token or not path.startswith("/latest/"):
            raise AttestationError("IMDSv2 request identity is invalid")
        return self._request(
            path,
            method="GET",
            headers={"X-aws-ec2-metadata-token": token},
        )


class SubprocessCommandReader:
    """Execute one fixed absolute argv in a minimal environment."""

    def run(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> bytes:
        if (
            not argv
            or not argv[0].startswith("/")
            or any(not isinstance(item, str) or "\x00" in item for item in argv)
        ):
            raise AttestationError("runtime command is not one safe absolute argv")
        try:
            working_directory = os.stat(
                TRUSTED_COMMAND_WORKING_DIRECTORY,
                follow_symlinks=False,
            )
        except OSError as error:
            raise AttestationError(
                "trusted runtime command working directory is unavailable"
            ) from error
        if (
            not stat.S_ISDIR(working_directory.st_mode)
            or working_directory.st_uid != 0
            or stat.S_IMODE(working_directory.st_mode) & 0o022
        ):
            raise AttestationError(
                "runtime command working directory is not trusted"
            )
        try:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                shell=False,
                env=dict(environment),
                cwd=TRUSTED_COMMAND_WORKING_DIRECTORY,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AttestationError("runtime command could not execute") from error
        if completed.returncode != 0:
            raise AttestationError("runtime command failed")
        return completed.stdout if completed.stdout else completed.stderr


def _runtime_facts(command_reader: object, lock: Mapping[str, object]) -> dict[str, str]:
    facts: dict[str, str] = {}
    run = getattr(command_reader, "run", None)
    if not callable(run):
        raise AttestationError("runtime command reader is unavailable")
    for field, argv in VERSION_COMMANDS.items():
        try:
            output = run(argv, environment=MINIMAL_COMMAND_ENVIRONMENT)
        except AttestationError:
            raise
        except Exception as error:
            raise AttestationError(f"runtime {field} command failed") from error
        facts[field] = _command_version(
            field,
            output,
        )
    image = str(lock["container_image"])
    try:
        inspect_output = run(
            container_inspect_argv(image),
            environment=MINIMAL_COMMAND_ENVIRONMENT,
        )
    except AttestationError:
        raise
    except Exception as error:
        raise AttestationError("runtime container inspection failed") from error
    try:
        repo_digests = json.loads(inspect_output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError("local container digest inspection is invalid") from error
    if (
        not isinstance(repo_digests, list)
        or not repo_digests
        or any(not isinstance(item, str) for item in repo_digests)
        or image not in repo_digests
    ):
        raise AttestationError("local container does not carry the locked digest")
    if facts != lock["versions"]:
        raise AttestationError("runtime versions drift from the runtime lock")
    return facts


def _selected_host_version(field: str, data: bytes) -> str:
    if field in {
        "nvidia_driver",
        "fabric_manager",
        "docker",
        "nvidia_container_runtime",
        "aws_cli",
    }:
        return _command_version(field, data)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AttestationError(f"host {field} version is not UTF-8") from error
    if field == "cuda":
        match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)\b", text, re.I)
        if match is None:
            match = re.search(r"\b([0-9]+(?:\.[0-9]+)+)\b", text)
    elif field == "efa":
        match = re.search(
            r"#\s*EFA installer version:\s*([0-9]+(?:\.[0-9]+)+)",
            text,
            re.I,
        )
        if match is None:
            match = re.search(r"\b([0-9]+(?:\.[0-9]+)+)\b", text)
    elif field == "ofi_nccl":
        match = re.search(
            r"aws-ofi-nccl\s+([0-9]+(?:\.[0-9]+)+)",
            text,
            re.I,
        )
        if match is None:
            match = re.search(r"\b([0-9]+(?:\.[0-9]+)+)\b", text)
    else:
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)+(?:[-+._][0-9]+)*)\b", text)
    if match is None:
        raise AttestationError(f"host {field} command returned no fixed version")
    return _fixed_version(match.group(1), label=f"host {field}")


def _version_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise AttestationError(f"{label} is not a measured version")
    match = re.match(r"^([0-9]+(?:\.[0-9]+)*)", value)
    if match is None:
        raise AttestationError(f"{label} is not a numeric measured version")
    return tuple(int(part) for part in match.group(1).split("."))


def _require_floor(actual: object, minimum: str, *, label: str) -> None:
    measured = _version_tuple(actual, label=label)
    floor = _version_tuple(minimum, label=f"{label} floor")
    width = max(len(measured), len(floor))
    if measured + (0,) * (width - len(measured)) < floor + (0,) * (
        width - len(floor)
    ):
        raise AttestationError(f"{label} is below the official P6-B300 floor")


def _selected_profile_values(profile: object) -> dict[str, object]:
    fields = {
        name: getattr(profile, name, None)
        for name in (
            "profile_id",
            "provider",
            "instance_type",
            "gpu_model",
            "allocated_gpus",
            "sha256",
        )
    }
    profile_id = fields["profile_id"]
    expected = {
        "aws-p5.48xlarge-v3": (
            "aws-p5.48xlarge",
            "p5.48xlarge",
            "NVIDIA H100 80GB",
        ),
        _P6_PROFILE_ID: (
            _P6_PROVIDER,
            _P6_INSTANCE_TYPE,
            _P6_GPU_MODEL,
        ),
    }.get(profile_id)
    if (
        expected is None
        or tuple(
            fields[name] for name in ("provider", "instance_type", "gpu_model")
        )
        != expected
        or type(fields["allocated_gpus"]) is not int
        or fields["allocated_gpus"] != 8
        or getattr(profile, "architecture", "x86_64") != "x86_64"
        or not isinstance(fields["sha256"], str)
        or _SHA256_RE.fullmatch(fields["sha256"]) is None
    ):
        raise AttestationError("selected GPU profile identity is not supported")
    if profile_id == _P6_PROFILE_ID:
        supplied_floors = dict(getattr(profile, "software_floors", ()))
        expected_floors = {
            "cuda": "13.0",
            "efa": "1.44.0",
            "kernel": "6.1",
            "nvidia_driver": "R580",
            "ofi_nccl": "1.17.1",
        }
        if any(
            supplied_floors.get(name) != value
            for name, value in expected_floors.items()
        ):
            raise AttestationError(
                "selected P6-B300 profile changes an official software floor"
            )
    return fields


def _run_selected_command(
    run: object,
    argv: Sequence[str],
    *,
    label: str,
) -> bytes:
    try:
        return run(argv, environment=MINIMAL_COMMAND_ENVIRONMENT)
    except AttestationError:
        raise
    except Exception as error:
        raise AttestationError(f"{label} command failed") from error


def _selected_runtime_facts(
    command_reader: object,
    lock: Mapping[str, object],
    profile: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    run = getattr(command_reader, "run", None)
    if not callable(run):
        raise AttestationError("selected GPU command reader is unavailable")
    host_facts: dict[str, str] = {}
    for field, argv in SELECTED_HOST_VERSION_COMMANDS.items():
        output = _run_selected_command(run, argv, label=f"host {field}")
        host_facts[field] = _selected_host_version(field, output)

    names_output = _run_selected_command(
        run,
        gpu_names_argv(),
        label="host GPU identity",
    )
    try:
        gpu_names = [
            line.strip()
            for line in names_output.decode("utf-8", errors="strict").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as error:
        raise AttestationError("GPU identity output is not UTF-8") from error
    if (
        len(gpu_names) != profile["allocated_gpus"]
        or any(name != profile["gpu_model"] for name in gpu_names)
    ):
        raise AttestationError("measured GPU identity differs from selected profile")

    image = str(lock["container_image"])
    inspect_output = _run_selected_command(
        run,
        container_inspect_argv(image),
        label="local digest-pinned container inspection",
    )
    try:
        repo_digests = json.loads(inspect_output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError("local container digest inspection is invalid") from error
    if (
        not isinstance(repo_digests, list)
        or image not in repo_digests
        or any(not isinstance(item, str) for item in repo_digests)
    ):
        raise AttestationError("exact digest-pinned container is not present locally")

    container_output = _run_selected_command(
        run,
        container_facts_argv(image),
        label="digest-pinned container framework measurement",
    )
    container_facts = _decode_json(container_output, label="container facts")
    if (
        container_output != _canonical_json(container_facts) + b"\n"
        or set(container_facts) != _CONTAINER_FACT_FIELDS
    ):
        raise AttestationError("container facts do not match the closed schema")
    normalized_container = {
        field: _fixed_version(
            container_facts[field],
            label=f"container facts.{field}",
        )
        for field in sorted(_CONTAINER_FACT_FIELDS)
    }
    if profile["profile_id"] == _P6_PROFILE_ID:
        for field, minimum in _P6_FLOORS.items():
            _require_floor(
                host_facts[field],
                minimum,
                label=f"P6-B300 host {field}",
            )
    expected_lock_versions = {
        **normalized_container,
        "nvidia_driver": host_facts["nvidia_driver"],
        "fabric_manager": host_facts["fabric_manager"],
        "docker": host_facts["docker"],
        "nvidia_container_runtime": host_facts["nvidia_container_runtime"],
        "aws_cli": host_facts["aws_cli"],
    }
    if expected_lock_versions != lock["versions"]:
        raise AttestationError(
            "measured host/container versions drift from the runtime lock"
        )
    return host_facts, normalized_container


def parse_gpu_evidence_bytes(data: bytes) -> dict[str, object]:
    """Parse canonical profile-aware evidence without accepting it as legacy v2."""

    evidence = _decode_json(data, label="GPU attestation evidence")
    if data != canonical_gpu_evidence(evidence):
        raise AttestationError(
            "GPU attestation evidence must use canonical JSON plus one newline"
        )
    if (
        type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
        or evidence.get("evidence_type") != GPU_EVIDENCE_TYPE
        or evidence.get("profile_id")
        not in {"aws-p5.48xlarge-v3", _P6_PROFILE_ID}
    ):
        raise AttestationError("GPU attestation evidence identity is invalid")
    for field in (
        "profile_sha256",
        "runtime_lock_sha256",
        "control_bundle_sha256",
    ):
        _sha256(evidence[field], label=f"GPU evidence {field}")
    for field in ("source_commit", "source_tree"):
        if (
            not isinstance(evidence[field], str)
            or _SHA1_RE.fullmatch(evidence[field]) is None
        ):
            raise AttestationError(f"GPU evidence {field} is invalid")
    _validate_image(
        evidence["container_image"],
        evidence["container_image_digest"],
    )
    host_facts = evidence["host_facts"]
    container_facts = evidence["container_facts"]
    if (
        not isinstance(host_facts, dict)
        or set(host_facts) != _SELECTED_HOST_FACT_FIELDS
        or not isinstance(container_facts, dict)
        or set(container_facts) != _CONTAINER_FACT_FIELDS
    ):
        raise AttestationError("GPU evidence fact fields do not match schema")
    for field, value in host_facts.items():
        _fixed_version(value, label=f"GPU evidence host_facts.{field}")
    for field, value in container_facts.items():
        _fixed_version(value, label=f"GPU evidence container_facts.{field}")
    expected_instance = (
        _P6_INSTANCE_TYPE
        if evidence["profile_id"] == _P6_PROFILE_ID
        else "p5.48xlarge"
    )
    identity_value = evidence["aws_instance_identity_document"]
    if not isinstance(identity_value, dict):
        raise AttestationError("GPU evidence identity document must be an object")
    identity = _validate_identity(
        _canonical_json(identity_value),
        expected_instance_type=expected_instance,
    )
    if (
        evidence["instance_type"] != expected_instance
        or identity.get("instanceType") != expected_instance
        or evidence["account_id"] != identity["accountId"]
        or evidence["instance_id"] != identity["instanceId"]
        or evidence["region"] != identity["region"]
        or evidence["ami_id"] != identity["imageId"]
        or evidence["gpu_count"] != 8
    ):
        raise AttestationError("GPU evidence duplicated identities are inconsistent")
    expected_profile = {
        "aws-p5.48xlarge-v3": (
            "aws-p5.48xlarge",
            "NVIDIA H100 80GB",
        ),
        _P6_PROFILE_ID: (_P6_PROVIDER, _P6_GPU_MODEL),
    }[evidence["profile_id"]]
    if (
        (evidence["provider"], evidence["gpu_model"]) != expected_profile
        or not isinstance(evidence["ami_owner_id"], str)
        or re.fullmatch(r"[0-9]{12}", evidence["ami_owner_id"]) is None
    ):
        raise AttestationError("GPU evidence profile identity is inconsistent")
    if evidence["profile_id"] == _P6_PROFILE_ID:
        if (
            evidence["ami_id"] != _P6_AMI_ID
            or evidence["ami_owner_id"] != _P6_AMI_OWNER_ID
        ):
            raise AttestationError("GPU evidence is not the supported P6-B300 tuple")
        for field, minimum in _P6_FLOORS.items():
            _require_floor(
                host_facts[field],
                minimum,
                label=f"P6-B300 host {field}",
            )
    boot_id = evidence["boot_id"]
    if not isinstance(boot_id, str) or _UUID_RE.fullmatch(boot_id) is None:
        raise AttestationError("GPU evidence boot ID is invalid")
    pkcs7 = evidence["aws_instance_identity_pkcs7"]
    if not isinstance(pkcs7, str) or _canonical_pkcs7(pkcs7.encode("ascii")) != pkcs7:
        raise AttestationError("GPU evidence PKCS7 is invalid")
    return evidence


def _open_or_create_output_directory(path: Path | str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    child = os.open(
                        component,
                        _directory_flags(),
                        dir_fd=current,
                    )
                except OSError as error:
                    raise AttestationError(
                        "receipt output directory cannot be created safely"
                    ) from error
            except OSError as error:
                raise AttestationError(
                    "receipt output path contains a symlink or non-directory"
                ) from error
            os.close(current)
            current = child
        details = os.fstat(current)
        if (
            details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) & 0o022
        ):
            raise AttestationError(
                "receipt output directory must be owner-controlled"
            )
        return current
    except Exception:
        os.close(current)
        raise


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        count = os.write(descriptor, view[offset:])
        if count <= 0:
            raise AttestationError("short write while staging environment receipt")
        offset += count


def _publish_receipt(path: Path | str, payload: bytes) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    if output == Path("/") or not output.name:
        raise AttestationError("receipt output path is invalid")
    directory = _open_or_create_output_directory(output.parent)
    temporary: str | None = None
    published_identity: tuple[int, int] | None = None
    try:
        for _ in range(128):
            candidate = f".{output.name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                continue
            except OSError as error:
                raise AttestationError(
                    "environment receipt cannot be staged safely"
                ) from error
            temporary = candidate
            break
        else:
            raise AttestationError("cannot allocate receipt staging file")
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            details = os.fstat(descriptor)
            published_identity = (details.st_dev, details.st_ino)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary,
                output.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise AttestationError(
                    "environment receipt output already exists; no replacement allowed"
                ) from error
            raise AttestationError(
                "atomic no-replace receipt publication failed"
            ) from error
        os.unlink(temporary, dir_fd=directory)
        temporary = None
        os.fsync(directory)
        current = os.stat(
            output.name,
            dir_fd=directory,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != published_identity
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) != 0o600
            or current.st_nlink != 1
            or _read_regular(output, label="published environment receipt")
            != payload
        ):
            raise AttestationError(
                "published environment receipt failed identity verification"
            )
    except Exception:
        if published_identity is not None:
            try:
                current = os.stat(
                    output.name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) == published_identity:
                    os.unlink(output.name, dir_fd=directory)
            except OSError:
                pass
        raise
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=directory)
            except OSError:
                pass
        os.close(directory)


def attest_environment(
    *,
    profile_path: Path | str,
    runtime_lock_path: Path | str,
    control_bundle_path: Path | str,
    output_path: Path | str,
    apply: bool,
    imds_reader: object | None = None,
    command_reader: object | None = None,
    boot_id_path: Path | str = "/proc/sys/kernel/random/boot_id",
) -> dict[str, object]:
    """Validate all local and authenticated facts, then optionally write once."""

    profile_data = _read_regular(profile_path, label="AWS P5 profile")
    profile_sha256 = _validate_profile(profile_data)
    lock_data = _read_regular(runtime_lock_path, label="runtime lock")
    lock = parse_runtime_lock_bytes(lock_data)
    control_data = _read_regular(control_bundle_path, label="control bundle")
    control_sha256 = hashlib.sha256(control_data).hexdigest()
    if (
        lock["profile_sha256"] != profile_sha256
        or lock["control_bundle_sha256"] != control_sha256
    ):
        raise AttestationError("profile or control bundle does not match runtime lock")

    imds = imds_reader or UrllibImdsReader()
    token_reader = getattr(imds, "token", None)
    metadata_reader = getattr(imds, "read", None)
    if not callable(token_reader) or not callable(metadata_reader):
        raise AttestationError("IMDSv2 reader is unavailable")
    token = token_reader()
    if not isinstance(token, str) or not token:
        raise AttestationError("IMDSv2 token is invalid")
    identity = _validate_identity(
        metadata_reader(IMDS_DOCUMENT_PATH, token=token)
    )
    pkcs7 = _canonical_pkcs7(
        metadata_reader(IMDS_PKCS7_PATH, token=token)
    )
    if identity["imageId"] != lock["ami_id"]:
        raise AttestationError("instance identity AMI does not match runtime lock")

    boot_data = _read_regular(
        boot_id_path,
        label="kernel boot ID",
        maximum_bytes=128,
    )
    try:
        boot_id = boot_data.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise AttestationError("kernel boot ID is not ASCII") from error
    if _UUID_RE.fullmatch(boot_id) is None:
        raise AttestationError("kernel boot ID is not a lowercase UUID")

    facts = _runtime_facts(command_reader or SubprocessCommandReader(), lock)
    receipt: dict[str, object] = {
        "schema_version": 2,
        "receipt_type": RECEIPT_TYPE,
        "provider": PROVIDER,
        "profile_sha256": profile_sha256,
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "control_bundle_sha256": control_sha256,
        "source_commit": lock["source_commit"],
        "source_tree": lock["source_tree"],
        "container_image": lock["container_image"],
        "container_image_digest": lock["container_image_digest"],
        "aws_instance_identity_document": identity,
        "aws_instance_identity_pkcs7": pkcs7,
        "account_id": identity["accountId"],
        "instance_id": identity["instanceId"],
        "region": identity["region"],
        "ami_id": identity["imageId"],
        "boot_id": boot_id,
        "runtime_facts": facts,
    }
    payload = canonical_receipt(receipt)
    if apply:
        _publish_receipt(output_path, payload)
    return receipt


attest_legacy_p5_environment = attest_environment


def attest_selected_gpu_environment(
    *,
    selected_profile: object,
    runtime_lock_path: Path | str,
    control_bundle_path: Path | str,
    output_path: Path | str,
    apply: bool,
    imds_reader: object | None = None,
    command_reader: object | None = None,
    boot_id_path: Path | str = "/proc/sys/kernel/random/boot_id",
) -> dict[str, object]:
    """Attest a selected P5/P6 tuple while keeping legacy receipt v2 unchanged."""

    profile = _selected_profile_values(selected_profile)
    lock_data = _read_regular(runtime_lock_path, label="runtime lock")
    lock = parse_runtime_lock_bytes(lock_data)
    control_data = _read_regular(control_bundle_path, label="control bundle")
    control_sha256 = hashlib.sha256(control_data).hexdigest()
    if (
        lock["profile_sha256"] != profile["sha256"]
        or lock["control_bundle_sha256"] != control_sha256
    ):
        raise AttestationError(
            "selected profile or control bundle does not match runtime lock"
        )
    if profile["profile_id"] == _P6_PROFILE_ID and (
        lock["ami_id"] == _UNSUPPORTED_P6_FRAMEWORK_AMI
        or lock["ami_id"] != _P6_AMI_ID
        or lock["ami_owner_id"] != _P6_AMI_OWNER_ID
    ):
        raise AttestationError(
            "P6-B300 requires the officially supported immutable Base DLAMI"
        )

    imds = imds_reader or UrllibImdsReader()
    token_reader = getattr(imds, "token", None)
    metadata_reader = getattr(imds, "read", None)
    if not callable(token_reader) or not callable(metadata_reader):
        raise AttestationError("IMDSv2 reader is unavailable")
    token = token_reader()
    if not isinstance(token, str) or not token:
        raise AttestationError("IMDSv2 token is invalid")
    identity = _validate_identity(
        metadata_reader(IMDS_DOCUMENT_PATH, token=token),
        expected_instance_type=str(profile["instance_type"]),
    )
    if identity.get("instanceType") != profile["instance_type"]:
        raise AttestationError(
            "authenticated instance type does not match selected GPU profile"
        )
    pkcs7 = _canonical_pkcs7(metadata_reader(IMDS_PKCS7_PATH, token=token))
    if (
        identity["imageId"] != lock["ami_id"]
        or identity["architecture"] != "x86_64"
    ):
        raise AttestationError(
            "authenticated instance identity differs from selected GPU runtime"
        )

    boot_data = _read_regular(
        boot_id_path,
        label="kernel boot ID",
        maximum_bytes=128,
    )
    try:
        boot_id = boot_data.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise AttestationError("kernel boot ID is not ASCII") from error
    if _UUID_RE.fullmatch(boot_id) is None:
        raise AttestationError("kernel boot ID is not a lowercase UUID")

    host_facts, container_facts = _selected_runtime_facts(
        command_reader or SubprocessCommandReader(),
        lock,
        profile,
    )
    evidence: dict[str, object] = {
        "schema_version": 1,
        "evidence_type": GPU_EVIDENCE_TYPE,
        "profile_id": profile["profile_id"],
        "provider": profile["provider"],
        "profile_sha256": profile["sha256"],
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "control_bundle_sha256": control_sha256,
        "source_commit": lock["source_commit"],
        "source_tree": lock["source_tree"],
        "container_image": lock["container_image"],
        "container_image_digest": lock["container_image_digest"],
        "ami_id": lock["ami_id"],
        "ami_owner_id": lock["ami_owner_id"],
        "instance_type": profile["instance_type"],
        "gpu_model": profile["gpu_model"],
        "gpu_count": profile["allocated_gpus"],
        "host_facts": host_facts,
        "container_facts": container_facts,
        "aws_instance_identity_document": identity,
        "aws_instance_identity_pkcs7": pkcs7,
        "account_id": identity["accountId"],
        "instance_id": identity["instanceId"],
        "region": identity["region"],
        "boot_id": boot_id,
    }
    payload = canonical_gpu_evidence(evidence)
    if parse_gpu_evidence_bytes(payload) != evidence:
        raise AttestationError("GPU evidence did not survive closed-schema parsing")
    if apply:
        _publish_receipt(output_path, payload)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attest one AWS P5 runtime against a canonical lock."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--control-bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        receipt = attest_environment(
            profile_path=arguments.profile,
            runtime_lock_path=arguments.runtime_lock,
            control_bundle_path=arguments.control_bundle,
            output_path=arguments.out,
            apply=arguments.apply,
        )
        sys.stdout.buffer.write(canonical_receipt(receipt))
        return 0
    except (AttestationError, OSError, ValueError) as error:
        sys.stdout.buffer.write(
            _canonical_json(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": str(error),
                }
            )
            + b"\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
