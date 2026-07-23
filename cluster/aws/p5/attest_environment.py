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
    AWS_RUNTIME_LOCK_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
)


PROVIDER = "aws-p5.48xlarge"
PROFILE_ID = PROFILE_ID_V3
RECEIPT_TYPE = "memorysplit-aws-environment-v2"
IMDS_DOCUMENT_PATH = "/latest/dynamic/instance-identity/document"
IMDS_PKCS7_PATH = "/latest/dynamic/instance-identity/pkcs7"
MINIMAL_COMMAND_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
}
VERSION_COMMANDS = {
    "python": (
        "/usr/bin/python3",
        "-c",
        "import platform;print(platform.python_version())",
    ),
    "pytorch": (
        "/usr/bin/python3",
        "-c",
        "import torch;print(torch.__version__)",
    ),
    "cuda": (
        "/usr/bin/python3",
        "-c",
        "import torch;print(torch.version.cuda or '')",
    ),
    "cudnn": (
        "/usr/bin/python3",
        "-c",
        "import torch;print(torch.backends.cudnn.version() or '')",
    ),
    "nccl": (
        "/usr/bin/python3",
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

_LOCK_FIELDS = set(AWS_RUNTIME_LOCK_FIELDS)
_RECEIPT_FIELDS = set(AWS_ENVIRONMENT_RECEIPT_V2_FIELDS)
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
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_VERSION_RE = re.compile(r"^[0-9]+(?:[._+-][0-9A-Za-z]+)*$")


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
    if (
        not isinstance(value, str)
        or _VERSION_RE.fullmatch(value) is None
        or any(
            not any(character.isdigit() for character in segment)
            and segment not in immutable_labels
            for segment in segments[1:]
        )
    ):
        raise AttestationError(f"{label} must be one non-floating version")
    return value


def _validate_image(image: object, digest: object) -> tuple[str, str]:
    validated_digest = str(digest)
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise AttestationError("container image digest is invalid")
    if not isinstance(image, str):
        raise AttestationError("container image must be a string")
    name, separator, image_digest = image.partition("@")
    registry = name.partition("/")[0]
    repository = name.rsplit("/", 1)[-1]
    if (
        separator != "@"
        or image.count("@") != 1
        or image_digest != validated_digest
        or "/" not in name
        or "." not in registry
        or ":" in repository
        or any(character.isspace() for character in image)
    ):
        raise AttestationError(
            "container image must be a full registry reference pinned by digest"
        )
    return image, validated_digest


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


def _validate_identity(data: bytes) -> dict[str, object]:
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
        and identity["instanceType"] != "p5.48xlarge"
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
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                shell=False,
                env=dict(environment),
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
