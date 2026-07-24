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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_image import (  # noqa: E402
    BuildPlanError,
    parse_image_binding_bytes,
)
from cluster.aws.p5.attest_environment import (  # noqa: E402
    AttestationError,
    parse_runtime_lock_bytes,
    read_regular_input,
)
from msctl.aws_contracts import validate_digest_pinned_oci_image  # noqa: E402


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
    "host_runtime_versions",
}
_SOURCE_FIELDS = {"commit", "tree"}
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
_HOST_RUNTIME_VERSION_FIELDS = {
    "nvidia_driver",
    "fabric_manager",
    "docker",
    "nvidia_container_runtime",
    "aws_cli",
}
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+(?:[._+-][0-9A-Za-z]+)*$")
_SIMPLE_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_LOCK_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9._-]*)=="
    r"(?P<version>[0-9][0-9A-Za-z.!+_-]*)$"
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
    host_versions = spec["host_runtime_versions"]
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise RuntimeArtifactError("runtime input source fields do not match schema")
    _sha1(source["commit"], label="source commit")
    _sha1(source["tree"], label="source tree")
    _sha256(spec["control_bundle_sha256"], label="control bundle")
    _sha256(spec["profile_sha256"], label="profile")
    normalized_host_versions = _version_object(
        host_versions,
        fields=_HOST_RUNTIME_VERSION_FIELDS,
        label="host runtime versions",
    )
    if (
        normalized_host_versions["nvidia_driver"]
        != HOST_CANDIDATE["versions"]["nvidia_driver"]
    ):
        raise RuntimeArtifactError(
            "host runtime nvidia_driver differs from the reviewed AMI fact"
        )
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


def _hash_map(value: object, *, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise RuntimeArtifactError(f"{label} must be one nonempty hash map")
    for key, item in value.items():
        if not isinstance(key, str):
            raise RuntimeArtifactError(f"{label} key is not a string")
        if isinstance(item, dict):
            _hash_map(item, label=f"{label}.{key}")
        else:
            _sha256(item, label=f"{label}.{key}")


def _file_commitments(value: object, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise RuntimeArtifactError(f"{label} must be a nonempty file list")
    paths: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "type",
            "commitment_sha256",
        }:
            raise RuntimeArtifactError(f"{label} file fields do not match schema")
        if (
            not isinstance(item["path"], str)
            or not item["path"].startswith("/")
            or item["type"] not in {"regular", "symlink", "directory"}
        ):
            raise RuntimeArtifactError(f"{label} file identity is invalid")
        _sha256(item["commitment_sha256"], label=f"{label} file")
        paths.append(item["path"])
    if paths != sorted(set(paths)):
        raise RuntimeArtifactError(f"{label} files are not unique and sorted")
    return value


def _os_package_inventory(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "manager",
        "database_root",
        "database_files",
        "database_tree_sha256",
        "packages",
        "package_count",
    }:
        raise RuntimeArtifactError("runtime SBOM OS inventory fields do not match")
    if (
        value["manager"] != "dpkg"
        or value["database_root"] != "/var/lib/dpkg"
        or not isinstance(value["packages"], list)
        or not value["packages"]
        or type(value["package_count"]) is not int
        or value["package_count"] != len(value["packages"])
    ):
        raise RuntimeArtifactError("runtime SBOM OS inventory identity is invalid")
    database_files = _file_commitments(
        value["database_files"],
        label="runtime SBOM dpkg database",
    )
    if value["database_tree_sha256"] != hashlib.sha256(
        canonical_json(database_files)
    ).hexdigest():
        raise RuntimeArtifactError("runtime SBOM dpkg database hash is invalid")
    names: list[str] = []
    for package in value["packages"]:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "architecture",
            "status",
            "installed_files",
            "installed_files_sha256",
        }:
            raise RuntimeArtifactError("runtime SBOM OS package fields do not match")
        name = package["name"]
        if (
            not isinstance(name, str)
            or not name
            or package["status"] != "ii"
        ):
            raise RuntimeArtifactError("runtime SBOM OS package name is invalid")
        names.append(name)
        files = _file_commitments(
            package["installed_files"],
            label=f"runtime SBOM OS package {name}",
        )
        if package["installed_files_sha256"] != hashlib.sha256(
            canonical_json(files)
        ).hexdigest():
            raise RuntimeArtifactError("runtime SBOM OS package hash is invalid")
    if names != sorted(set(names)):
        raise RuntimeArtifactError("runtime SBOM OS packages are not sorted")


def parse_runtime_sbom_bytes(data: bytes) -> dict[str, object]:
    """Parse the closed full runtime SBOM emitted beside the legacy lock."""

    sbom = _canonical_object(data, label="runtime SBOM")
    if set(sbom) != {
        "schema_version",
        "document_type",
        "source",
        "runtime_lock_sha256",
        "image_binding_sha256",
        "project_dependency_lock",
        "host",
        "container",
    }:
        raise RuntimeArtifactError("runtime SBOM fields do not match closed schema")
    if (
        type(sbom["schema_version"]) is not int
        or sbom["schema_version"] != 2
        or sbom["document_type"] != "memorysplit-aws-gpu-sbom-v2"
    ):
        raise RuntimeArtifactError("runtime SBOM identity is invalid")
    source = sbom["source"]
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise RuntimeArtifactError("runtime SBOM source fields do not match schema")
    _sha1(source["commit"], label="runtime SBOM source commit")
    _sha1(source["tree"], label="runtime SBOM source tree")
    _sha256(sbom["runtime_lock_sha256"], label="runtime SBOM lock")
    _sha256(sbom["image_binding_sha256"], label="runtime SBOM image binding")

    project = sbom["project_dependency_lock"]
    if not isinstance(project, dict) or set(project) != {"sha256", "packages"}:
        raise RuntimeArtifactError(
            "runtime SBOM project dependency fields do not match schema"
        )
    _sha256(project["sha256"], label="runtime SBOM dependency lock")
    if not isinstance(project["packages"], list):
        raise RuntimeArtifactError("runtime SBOM dependency packages must be a list")
    dependency_names: list[str] = []
    dependency_bindings: dict[str, tuple[str, set[str]]] = {}
    for package in project["packages"]:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "allowed_distribution_sha256",
        }:
            raise RuntimeArtifactError(
                "runtime SBOM dependency package fields do not match schema"
            )
        name = package["name"]
        if not isinstance(name, str) or not name:
            raise RuntimeArtifactError("runtime SBOM dependency name is invalid")
        dependency_names.append(name)
        version = _fixed_version(
            package["version"],
            label=f"runtime SBOM dependency {name}",
        )
        hashes = package["allowed_distribution_sha256"]
        if not isinstance(hashes, list) or not hashes:
            raise RuntimeArtifactError("runtime SBOM dependency hashes are empty")
        for digest in hashes:
            _sha256(digest, label=f"runtime SBOM dependency {name} hash")
        dependency_bindings[name] = (version, set(hashes))
    if dependency_names != sorted(set(dependency_names)):
        raise RuntimeArtifactError("runtime SBOM dependency packages are not sorted")

    host = sbom["host"]
    if not isinstance(host, dict) or set(host) != {
        "ami_id",
        "ami_owner_id",
        "ami_name",
        "architecture",
        "versions",
        "minimum_versions",
    }:
        raise RuntimeArtifactError("runtime SBOM host fields do not match schema")
    if (
        host["ami_id"] != HOST_CANDIDATE["ami_id"]
        or host["ami_owner_id"] != HOST_CANDIDATE["ami_owner_id"]
        or host["ami_name"] != HOST_CANDIDATE["ami_name"]
        or host["architecture"] != HOST_CANDIDATE["architecture"]
        or host["minimum_versions"] != HOST_CANDIDATE["minimum_versions"]
    ):
        raise RuntimeArtifactError("runtime SBOM host differs from reviewed AMI")
    if not isinstance(host["versions"], dict) or not isinstance(
        host["minimum_versions"],
        dict,
    ):
        raise RuntimeArtifactError("runtime SBOM host versions are invalid")
    for name, version in host["versions"].items():
        _fixed_version(version, label=f"runtime SBOM host {name}")
    if any(
        host["versions"].get(name) != version
        for name, version in HOST_CANDIDATE["versions"].items()
    ):
        raise RuntimeArtifactError("runtime SBOM host facts differ from reviewed AMI")

    container = sbom["container"]
    container_fields = {
        "platform",
        "base_image",
        "base_image_digest",
        "image",
        "image_digest",
        "versions",
        "os_release",
        "os_packages",
        "python",
        "installed_python_packages",
        "inventory_method",
        "installed_distribution_count",
        "project_install_report_sha256",
        "build_inputs",
        "repository_transcript_sha256",
        "command_transcript_sha256",
        "inherited_entrypoint",
        "entrypoint_sha256",
        "inspection_artifact_sha256",
    }
    if not isinstance(container, dict) or set(container) != container_fields:
        raise RuntimeArtifactError("runtime SBOM container fields do not match schema")
    if container["platform"] != PLATFORM:
        raise RuntimeArtifactError("runtime SBOM container platform is invalid")
    try:
        validate_digest_pinned_oci_image(
            container["base_image"],
            container["base_image_digest"],
        )
        validate_digest_pinned_oci_image(
            container["image"],
            container["image_digest"],
        )
    except ValueError as error:
        raise RuntimeArtifactError("runtime SBOM image is not digest-pinned") from error
    if (
        not isinstance(container["versions"], dict)
        or set(container["versions"]) != _CONTAINER_VERSION_FIELDS
    ):
        raise RuntimeArtifactError("runtime SBOM container versions are invalid")
    for name, version in container["versions"].items():
        _fixed_version(version, label=f"runtime SBOM container {name}")
    if not isinstance(container["os_release"], dict) or set(
        container["os_release"]
    ) != {"id", "version_id", "pretty_name"}:
        raise RuntimeArtifactError("runtime SBOM OS release is invalid")
    _os_package_inventory(container["os_packages"])
    if not isinstance(container["python"], dict) or set(container["python"]) != {
        "implementation",
        "version",
        "executable",
    }:
        raise RuntimeArtifactError("runtime SBOM Python identity is invalid")
    if container["python"]["executable"] != "/opt/conda/bin/python":
        raise RuntimeArtifactError("runtime SBOM did not use the pinned DLC Python")
    installed = container["installed_python_packages"]
    if (
        not isinstance(installed, list)
        or container["inventory_method"] != "importlib.metadata.distributions"
        or type(container["installed_distribution_count"]) is not int
        or container["installed_distribution_count"] != len(installed)
    ):
        raise RuntimeArtifactError("runtime SBOM installed inventory is invalid")
    installed_names: list[str] = []
    installed_project_names: set[str] = set()
    for package in installed:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "installer",
            "provenance",
            "metadata_file_sha256",
            "record_file_sha256",
            "wheel_file_sha256",
            "installed_files",
            "installed_files_sha256",
        }:
            raise RuntimeArtifactError(
                "runtime SBOM installed package fields do not match schema"
            )
        name = package["name"]
        if not isinstance(name, str) or not name:
            raise RuntimeArtifactError("runtime SBOM installed package is invalid")
        installed_names.append(name)
        _fixed_version(
            package["version"],
            label=f"runtime SBOM installed {name}",
        )
        provenance = package["provenance"]
        if not isinstance(provenance, dict) or provenance.get("kind") not in {
            "project-wheel",
            "inherited-base-image",
        }:
            raise RuntimeArtifactError(
                f"runtime SBOM installed {name} provenance is invalid"
            )
        if provenance["kind"] == "project-wheel":
            if set(provenance) != {"kind", "archive_sha256"}:
                raise RuntimeArtifactError("runtime SBOM wheel provenance is open")
            _sha256(
                provenance["archive_sha256"],
                label=f"runtime SBOM installed {name} archive",
            )
            expected = dependency_bindings.get(name)
            if (
                expected is None
                or expected[0] != package["version"]
                or provenance["archive_sha256"] not in expected[1]
            ):
                raise RuntimeArtifactError(
                    f"runtime SBOM project archive differs from lock: {name}"
                )
            installed_project_names.add(name)
        elif (
            set(provenance) != {"kind", "base_image_digest"}
            or provenance["base_image_digest"] != container["base_image_digest"]
        ):
            raise RuntimeArtifactError("runtime SBOM base provenance is invalid")
        metadata_hashes: list[str] = []
        for field in (
            "metadata_file_sha256",
            "record_file_sha256",
            "wheel_file_sha256",
        ):
            metadata_hashes.append(
                _sha256(
                    package[field],
                    label=f"runtime SBOM installed {name} {field}",
                )
            )
        files = _file_commitments(
            package["installed_files"],
            label=f"runtime SBOM installed {name}",
        )
        if (
            package["installed_files_sha256"]
            != hashlib.sha256(canonical_json(files)).hexdigest()
            or not set(metadata_hashes)
            <= {item["commitment_sha256"] for item in files}
        ):
            raise RuntimeArtifactError(
                f"runtime SBOM installed {name} file hash is invalid"
            )
    if (
        installed_names != sorted(set(installed_names))
        or "torch" not in installed_names
        or installed_project_names != set(dependency_bindings)
    ):
        raise RuntimeArtifactError(
            "runtime SBOM installed inventory is incomplete or unsorted"
        )
    for field in (
        "project_install_report_sha256",
        "entrypoint_sha256",
        "inspection_artifact_sha256",
    ):
        _sha256(container[field], label=f"runtime SBOM container {field}")
    _hash_map(container["build_inputs"], label="runtime SBOM build inputs")
    _hash_map(
        container["repository_transcript_sha256"],
        label="runtime SBOM repository transcripts",
    )
    _hash_map(
        container["command_transcript_sha256"],
        label="runtime SBOM command transcripts",
    )
    inherited = container["inherited_entrypoint"]
    if not isinstance(inherited, dict) or set(inherited) != {
        "entrypoint",
        "command",
    }:
        raise RuntimeArtifactError("runtime SBOM inherited entrypoint is invalid")
    return sbom


def produce_runtime_artifacts(
    *,
    spec_bytes: bytes,
    host_candidate_bytes: bytes,
    dependency_lock_bytes: bytes,
    image_binding_bytes: bytes,
) -> RuntimeArtifacts:
    """Produce a parser-compatible lock and full measured runtime SBOM."""

    spec = _runtime_spec(spec_bytes)
    host = _host_candidate(host_candidate_bytes)
    packages = _dependency_packages(dependency_lock_bytes)
    try:
        binding = parse_image_binding_bytes(
            image_binding_bytes,
            dependency_lock_bytes=dependency_lock_bytes,
        )
    except (BuildPlanError, TypeError, ValueError) as error:
        raise RuntimeArtifactError(
            "image binding is not tool-produced closed build authority"
        ) from error
    if (
        binding["source_commit"] != spec["source"]["commit"]
        or binding["source_tree"] != spec["source"]["tree"]
    ):
        raise RuntimeArtifactError("image binding source differs from runtime input")
    inspection = binding["inspection_artifact"]
    container_versions = _version_object(
        inspection["container_facts"],
        fields=_CONTAINER_VERSION_FIELDS,
        label="measured container versions",
    )
    host_runtime_versions = spec["host_runtime_versions"]
    versions = {
        field: (
            container_versions[field]
            if field in _CONTAINER_VERSION_FIELDS
            else host_runtime_versions[field]
        )
        for field in _LOCK_VERSION_FIELDS
    }
    if tuple(versions) != _LOCK_VERSION_FIELDS:
        raise RuntimeArtifactError("runtime version provenance ordering drifted")
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
        "schema_version": 2,
        "document_type": "memorysplit-aws-gpu-sbom-v2",
        "source": dict(spec["source"]),
        "runtime_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "image_binding_sha256": hashlib.sha256(image_binding_bytes).hexdigest(),
        "project_dependency_lock": {
            "sha256": hashlib.sha256(dependency_lock_bytes).hexdigest(),
            "packages": packages,
        },
        "host": {
            "ami_id": host["ami_id"],
            "ami_owner_id": host["ami_owner_id"],
            "ami_name": host["ami_name"],
            "architecture": host["architecture"],
            "versions": host_versions,
            "minimum_versions": dict(host["minimum_versions"]),
        },
        "container": {
            "platform": PLATFORM,
            "base_image": binding["base_image"],
            "base_image_digest": binding["base_image_digest"],
            "image": binding["container_image"],
            "image_digest": binding["container_image_digest"],
            "versions": dict(container_versions),
            "os_release": dict(inspection["os_release"]),
            "os_packages": dict(inspection["os_packages"]),
            "python": dict(inspection["python"]),
            "installed_python_packages": list(
                inspection["installed_python_packages"]
            ),
            "inventory_method": inspection["inventory_method"],
            "installed_distribution_count": inspection[
                "installed_distribution_count"
            ],
            "project_install_report_sha256": inspection[
                "project_install_report_sha256"
            ],
            "build_inputs": dict(binding["build_inputs"]),
            "repository_transcript_sha256": dict(
                binding["repository_transcript_sha256"]
            ),
            "command_transcript_sha256": dict(
                binding["command_transcript_sha256"]
            ),
            "inherited_entrypoint": dict(binding["inherited_entrypoint"]),
            "entrypoint_sha256": binding["entrypoint_sha256"],
            "inspection_artifact_sha256": binding[
                "inspection_artifact_sha256"
            ],
        },
    }
    sbom_bytes = canonical_json(sbom)
    if parse_runtime_sbom_bytes(sbom_bytes) != sbom:
        raise RuntimeArtifactError("runtime SBOM did not survive closed parsing")
    return RuntimeArtifacts(runtime_lock_bytes=lock_bytes, sbom_bytes=sbom_bytes)


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
    parser.add_argument("--image-binding", required=True)
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
            image_binding_bytes=read_regular_input(
                arguments.image_binding,
                label="image binding",
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
