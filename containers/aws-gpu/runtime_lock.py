#!/usr/bin/env python3
"""Produce deterministic runtime-lock and build-SBOM bytes without networking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cluster.aws.p5.attest_environment import (  # noqa: E402
    AttestationError,
    parse_runtime_lock_bytes,
    read_regular_input,
)
from msctl.aws_contracts import validate_digest_pinned_oci_image  # noqa: E402


BASE_REGISTRY = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training"
)
BASE_DIGEST = (
    "sha256:1414a836532f22b271c03b7ccdbdff3d"
    "aa0591975b3bd9a3cf51601a45b37f4f"
)
BASE_IMAGE = f"{BASE_REGISTRY}@{BASE_DIGEST}"
PLATFORM = "linux/amd64"
HOST_CANDIDATE = {
    "schema_version": 1,
    "ami_id": "ami-0260c4d597dcc8641",
    "ami_owner_id": "898082745236",
    "ami_name": (
        "Deep Learning Base AMI with Single CUDA Ubuntu 24.04 20260523"
    ),
    "architecture": "x86_64",
    "versions": {
        "cuda": "13.2",
        "nvidia_driver": "595.71.05",
        "kernel": "6.17",
        "efa": "1.47.0",
        "ofi_nccl": "1.18.0",
    },
    "minimum_versions": {
        "cuda": "13.0",
        "nvidia_driver": "580.0",
        "kernel": "6.1",
        "efa": "1.44.0",
        "ofi_nccl": "1.17.1",
    },
}
_SPEC_FIELDS = {
    "schema_version",
    "source",
    "control_bundle_sha256",
    "profile_sha256",
    "container",
    "host_runtime_versions",
}
_SOURCE_FIELDS = {"commit", "tree"}
_CONTAINER_FIELDS = {
    "base_image",
    "platform",
    "image_binding",
    "versions",
}
_IMAGE_BINDING_FIELDS = {
    "schema_version",
    "source_commit",
    "repository_uri",
    "container_image",
    "container_image_digest",
}
_CONTAINER_VERSION_FIELDS = {"python", "pytorch", "cuda", "cudnn", "nccl"}
_LOCK_VERSION_FIELDS = (
    "python",
    "pytorch",
    "cuda",
    "cudnn",
    "nccl",
    "nvidia_driver",
    "fabric_manager",
    "docker",
    "nvidia_container_runtime",
    "aws_cli",
)
_HOST_RUNTIME_VERSION_FIELDS = set(_LOCK_VERSION_FIELDS)
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+(?:[._+-][0-9A-Za-z]+)*$")
_SIMPLE_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_LOCK_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9._-]*)=="
    r"(?P<version>[0-9][0-9A-Za-z.!+_-]*)$"
)
_PRIVATE_ECR_RE = re.compile(
    r"^[0-9]{12}\.dkr\.ecr\."
    r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+\.amazonaws\.com/"
    r"[a-z0-9]+(?:(?:[._-]|/)[a-z0-9]+)*$"
)
_FLOATING_MARKERS = frozenset(
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
_SECRET_MARKERS = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "github_token",
    "hf_token",
    "private_key",
    "password",
    "bearer ",
)


class RuntimeArtifactError(ValueError):
    """Runtime inputs cannot be reduced to the current immutable lock schema."""


@dataclass(frozen=True)
class RuntimeArtifacts:
    """Exact bytes for the legacy-compatible lock and separated-facts SBOM."""

    runtime_lock_bytes: bytes
    sbom_bytes: bytes


def canonical_json(value: object) -> bytes:
    """Serialize canonical ASCII JSON with one trailing newline."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise RuntimeArtifactError("runtime artifact is not canonical JSON") from error


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeArtifactError(f"JSON repeats field: {key}")
        value[key] = item
    return value


def _canonical_object(data: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise RuntimeArtifactError(f"{label} input must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                RuntimeArtifactError(f"{label} contains non-finite {constant}")
            ),
        )
    except RuntimeArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeArtifactError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeArtifactError(f"{label} must be a JSON object")
    if data != canonical_json(value):
        raise RuntimeArtifactError(f"{label} must be canonical JSON plus one newline")
    return value


def _reject_unsafe_strings(value: object, *, label: str) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        if (
            "\x00" in value
            or "${" in value
            or "$(" in value
            or any(marker in lowered for marker in _SECRET_MARKERS)
        ):
            raise RuntimeArtifactError(f"{label} contains unsafe secret or expansion text")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_unsafe_strings(key, label=label)
            _reject_unsafe_strings(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _reject_unsafe_strings(item, label=label)


def _sha1(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise RuntimeArtifactError(f"{label} must be lowercase Git SHA-1")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeArtifactError(f"{label} must be lowercase SHA-256")
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
        or any(marker in normalized for marker in _FLOATING_MARKERS)
        or any(
            not any(character.isdigit() for character in segment)
            and segment not in immutable_labels
            for segment in segments[1:]
        )
    ):
        raise RuntimeArtifactError(f"{label} must be one non-floating version")
    return value


def _version_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, str) or _SIMPLE_VERSION_RE.fullmatch(value) is None:
        raise RuntimeArtifactError(f"{label} must be a numeric dotted version")
    return tuple(int(part) for part in value.split("."))


def _at_least(actual: str, minimum: str, *, label: str) -> None:
    actual_parts = _version_tuple(actual, label=label)
    minimum_parts = _version_tuple(minimum, label=f"{label} minimum")
    width = max(len(actual_parts), len(minimum_parts))
    if actual_parts + (0,) * (width - len(actual_parts)) < minimum_parts + (
        0,
    ) * (width - len(minimum_parts)):
        raise RuntimeArtifactError(f"{label} is below the P6-B300 floor")


def _host_candidate(data: bytes) -> dict[str, object]:
    host = _canonical_object(data, label="host candidate")
    _reject_unsafe_strings(host, label="host candidate")
    if host != HOST_CANDIDATE:
        raise RuntimeArtifactError("host candidate differs from the reviewed AMI pin")
    versions = host["versions"]
    minimums = host["minimum_versions"]
    for field in minimums:
        _at_least(
            str(versions[field]),
            str(minimums[field]),
            label=f"host {field}",
        )
    return host


def _version_object(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeArtifactError(f"{label} fields do not match schema")
    return {
        field: _fixed_version(value[field], label=f"{label}.{field}")
        for field in sorted(fields)
    }


def _runtime_spec(data: bytes) -> dict[str, object]:
    spec = _canonical_object(data, label="runtime input")
    _reject_unsafe_strings(spec, label="runtime input")
    if (
        set(spec) != _SPEC_FIELDS
        or type(spec.get("schema_version")) is not int
        or spec["schema_version"] != 1
    ):
        raise RuntimeArtifactError("runtime input fields do not match schema version 1")
    source = spec["source"]
    container = spec["container"]
    host_versions = spec["host_runtime_versions"]
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise RuntimeArtifactError("runtime input source fields do not match schema")
    if not isinstance(container, dict) or set(container) != _CONTAINER_FIELDS:
        raise RuntimeArtifactError("runtime input container fields do not match schema")
    commit = _sha1(source["commit"], label="source commit")
    _sha1(source["tree"], label="source tree")
    _sha256(spec["control_bundle_sha256"], label="control bundle")
    _sha256(spec["profile_sha256"], label="profile")
    if container["base_image"] != BASE_IMAGE or container["platform"] != PLATFORM:
        raise RuntimeArtifactError("container base image or platform is not reviewed")
    binding = container["image_binding"]
    if (
        not isinstance(binding, dict)
        or set(binding) != _IMAGE_BINDING_FIELDS
        or type(binding.get("schema_version")) is not int
        or binding["schema_version"] != 1
        or binding["source_commit"] != commit
        or not isinstance(binding["repository_uri"], str)
        or _PRIVATE_ECR_RE.fullmatch(binding["repository_uri"]) is None
    ):
        raise RuntimeArtifactError("container image binding fields are invalid")
    digest = binding["container_image_digest"]
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise RuntimeArtifactError("container image binding digest is invalid")
    try:
        validate_digest_pinned_oci_image(binding["container_image"], digest)
    except ValueError as error:
        raise RuntimeArtifactError("container image binding is not digest-pinned") from error
    if binding["container_image"] != f"{binding['repository_uri']}@{digest}":
        raise RuntimeArtifactError("container image binding repository is inconsistent")
    container_versions = _version_object(
        container["versions"],
        fields=_CONTAINER_VERSION_FIELDS,
        label="container versions",
    )
    expected_prefixes = {
        "python": "3.12",
        "pytorch": "2.9.0",
        "cuda": "13.0",
    }
    for field, prefix in expected_prefixes.items():
        value = container_versions[field]
        if value != prefix and not value.startswith(prefix + ".") and not value.startswith(
            prefix + "+"
        ):
            raise RuntimeArtifactError(
                f"container {field} does not match the reviewed base release"
            )
    normalized_host_versions = _version_object(
        host_versions,
        fields=_HOST_RUNTIME_VERSION_FIELDS,
        label="host runtime versions",
    )
    for field in ("cuda", "nvidia_driver"):
        if normalized_host_versions[field] != HOST_CANDIDATE["versions"][field]:
            raise RuntimeArtifactError(
                f"host runtime {field} differs from the reviewed AMI fact"
            )
    spec["container"]["versions"] = container_versions
    spec["host_runtime_versions"] = normalized_host_versions
    return spec


def _dependency_packages(data: bytes) -> list[dict[str, object]]:
    if not isinstance(data, bytes):
        raise RuntimeArtifactError("dependency lock input must be bytes")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeArtifactError("dependency lock must be ASCII") from error
    logical_lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    if pending or not logical_lines:
        raise RuntimeArtifactError("dependency lock is empty or truncated")

    packages: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in logical_lines:
        requirement, *options = line.split()
        match = _LOCK_REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            raise RuntimeArtifactError("dependency lock contains a floating requirement")
        name = match.group("name").lower().replace("_", "-")
        if name in seen:
            raise RuntimeArtifactError(f"dependency lock repeats package {name}")
        seen.add(name)
        hashes: list[str] = []
        for option in options:
            prefix = "--hash=sha256:"
            if not option.startswith(prefix):
                raise RuntimeArtifactError("dependency lock contains an unhashed option")
            digest = option.removeprefix(prefix)
            hashes.append(_sha256(digest, label=f"dependency {name} hash"))
        if not hashes or hashes != sorted(set(hashes)):
            raise RuntimeArtifactError(
                f"dependency {name} hashes must be nonempty, unique, and sorted"
            )
        packages.append(
            {
                "name": name,
                "version": match.group("version"),
                "allowed_distribution_sha256": hashes,
            }
        )
    if [item["name"] for item in packages] != sorted(seen):
        raise RuntimeArtifactError("dependency lock packages must be sorted")
    return packages


def produce_runtime_artifacts(
    *,
    spec_bytes: bytes,
    host_candidate_bytes: bytes,
    dependency_lock_bytes: bytes,
) -> RuntimeArtifacts:
    """Produce current-parser-compatible lock bytes and a separated-facts SBOM."""

    spec = _runtime_spec(spec_bytes)
    host = _host_candidate(host_candidate_bytes)
    packages = _dependency_packages(dependency_lock_bytes)
    container = spec["container"]
    container_versions = container["versions"]
    host_runtime_versions = spec["host_runtime_versions"]
    versions = {
        field: host_runtime_versions[field] for field in _LOCK_VERSION_FIELDS
    }
    if tuple(versions) != _LOCK_VERSION_FIELDS:
        raise RuntimeArtifactError("runtime version provenance ordering drifted")
    binding = container["image_binding"]
    lock = {
        "schema_version": 1,
        "source_commit": spec["source"]["commit"],
        "source_tree": spec["source"]["tree"],
        "control_bundle_sha256": spec["control_bundle_sha256"],
        "profile_sha256": spec["profile_sha256"],
        "ami_id": host["ami_id"],
        "ami_owner_id": host["ami_owner_id"],
        "container_image": binding["container_image"],
        "container_image_digest": binding["container_image_digest"],
        "versions": versions,
    }
    lock_bytes = canonical_json(lock)
    try:
        parsed = parse_runtime_lock_bytes(lock_bytes)
    except (AttestationError, TypeError, ValueError) as error:
        raise RuntimeArtifactError(
            "produced runtime lock is incompatible with the current parser"
        ) from error
    if parsed != lock:
        raise RuntimeArtifactError("current runtime-lock parser changed produced values")

    host_versions = {**host["versions"], **host_runtime_versions}
    sbom = {
        "schema_version": 1,
        "document_type": "memorysplit-aws-gpu-sbom-v1",
        "source": dict(spec["source"]),
        "runtime_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(
            dependency_lock_bytes
        ).hexdigest(),
        "host": {
            "ami_id": host["ami_id"],
            "ami_owner_id": host["ami_owner_id"],
            "ami_name": host["ami_name"],
            "architecture": host["architecture"],
            "versions": host_versions,
            "minimum_versions": dict(host["minimum_versions"]),
        },
        "container": {
            "platform": container["platform"],
            "base_image": container["base_image"],
            "image": binding["container_image"],
            "image_digest": binding["container_image_digest"],
            "versions": dict(container_versions),
        },
        "python_packages": packages,
    }
    return RuntimeArtifacts(
        runtime_lock_bytes=lock_bytes,
        sbom_bytes=canonical_json(sbom),
    )


def _write_no_replace(path: Path | str, data: bytes) -> None:
    destination = Path(os.path.abspath(os.fspath(path)))
    parent = destination.parent
    if not parent.is_dir() or destination == Path("/") or not destination.name:
        raise RuntimeArtifactError("output path has no existing parent directory")
    temporary = parent / (
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        details = os.stat(temporary, follow_symlinks=False)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise RuntimeArtifactError("staged output is not one regular file")
        os.link(temporary, destination, follow_symlinks=False)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise RuntimeArtifactError("output already exists") from error
    except OSError as error:
        raise RuntimeArtifactError("output could not be published safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic runtime-lock and SBOM digests. Files are "
            "written no-replace only with --apply."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--host-candidate",
        default=str(root / "host-candidate.json"),
    )
    parser.add_argument(
        "--dependency-lock",
        default=str(root / "requirements.lock"),
    )
    parser.add_argument("--runtime-lock-out", required=True)
    parser.add_argument("--sbom-out", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        runtime_lock_output = Path(arguments.runtime_lock_out).resolve()
        sbom_output = Path(arguments.sbom_out).resolve()
        if runtime_lock_output == sbom_output:
            raise RuntimeArtifactError(
                "runtime-lock and SBOM outputs must be distinct paths"
            )
        artifacts = produce_runtime_artifacts(
            spec_bytes=read_regular_input(arguments.input, label="runtime input"),
            host_candidate_bytes=read_regular_input(
                arguments.host_candidate,
                label="host candidate",
            ),
            dependency_lock_bytes=read_regular_input(
                arguments.dependency_lock,
                label="dependency lock",
            ),
        )
        result = {
            "schema_version": 1,
            "applied": bool(arguments.apply),
            "runtime_lock_output": str(runtime_lock_output),
            "runtime_lock_sha256": hashlib.sha256(
                artifacts.runtime_lock_bytes
            ).hexdigest(),
            "sbom_output": str(sbom_output),
            "sbom_sha256": hashlib.sha256(artifacts.sbom_bytes).hexdigest(),
        }
        if arguments.apply:
            _write_no_replace(
                runtime_lock_output,
                artifacts.runtime_lock_bytes,
            )
            try:
                _write_no_replace(sbom_output, artifacts.sbom_bytes)
            except Exception:
                try:
                    runtime_lock_output.unlink()
                except OSError:
                    pass
                raise
    except (AttestationError, RuntimeArtifactError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
