#!/usr/bin/env python3
"""Emit one canonical, offline inventory of the running container."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


INSPECTION_TYPE = "memorysplit-container-inspection-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_NAME_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_FACT_FIELDS = {"python", "pytorch", "cuda", "cudnn", "nccl"}
_PACKAGE_FIELDS = {
    "name",
    "version",
    "installer",
    "archive_sha256",
    "record_sha256",
    "wheel_metadata_sha256",
}


class InspectionError(ValueError):
    """The running image cannot produce a closed software inventory."""


def canonical_json(value: object) -> bytes:
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
        raise InspectionError("inspection artifact is not canonical JSON") from error


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _optional_text_hash(value: str | None) -> str | None:
    return None if value is None else _sha256_bytes(value.encode("utf-8"))


def _normalized_name(value: object) -> str:
    if not isinstance(value, str):
        raise InspectionError("installed package name must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if _PACKAGE_NAME_RE.fullmatch(normalized) is None:
        raise InspectionError("installed package name is not canonical")
    return normalized


def _fixed_string(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise InspectionError(f"{label} must be one nonempty fixed string")
    return value


def _optional_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise InspectionError(f"{label} must be lowercase SHA-256 or null")
    return value


def _os_release(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise InspectionError("OS release is not UTF-8") from error
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
            try:
                parsed = shlex.split(raw_value, posix=True)
            except ValueError as error:
                raise InspectionError("OS release contains invalid quoting") from error
            if len(parsed) != 1:
                raise InspectionError("OS release value is ambiguous")
            values[key] = parsed[0]
    if set(values) != {"ID", "VERSION_ID", "PRETTY_NAME"}:
        raise InspectionError("OS release lacks required identity fields")
    return {
        "id": _fixed_string(values["ID"], label="OS ID"),
        "version_id": _fixed_string(
            values["VERSION_ID"],
            label="OS version ID",
        ),
        "pretty_name": _fixed_string(
            values["PRETTY_NAME"],
            label="OS pretty name",
        ),
    }


def _install_report(data: bytes) -> tuple[str, dict[tuple[str, str], str]]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectionError("pip install report is invalid JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("install"), list):
        raise InspectionError("pip install report has no closed install list")
    selected: dict[tuple[str, str], str] = {}
    for item in value["install"]:
        if not isinstance(item, dict):
            raise InspectionError("pip install report entry is not an object")
        metadata = item.get("metadata")
        download = item.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise InspectionError("pip install report entry lacks provenance")
        name = _normalized_name(metadata.get("name"))
        version = _fixed_string(
            metadata.get("version"),
            label=f"pip report {name} version",
        )
        archive = download.get("archive_info")
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        digest = hashes.get("sha256") if isinstance(hashes, dict) else None
        digest = _optional_sha256(digest, label=f"pip report {name} archive")
        if digest is None or (name, version) in selected:
            raise InspectionError("pip install report is incomplete or duplicated")
        selected[(name, version)] = digest
    return _sha256_bytes(data), selected


def _installed_packages(
    distributions: Sequence[object],
    selected_archives: Mapping[tuple[str, str], str],
) -> list[dict[str, object]]:
    packages: list[dict[str, object]] = []
    seen: set[str] = set()
    for distribution in distributions:
        metadata = getattr(distribution, "metadata", None)
        if metadata is None:
            raise InspectionError("installed distribution lacks metadata")
        name = _normalized_name(metadata.get("Name"))
        version = _fixed_string(
            getattr(distribution, "version", None),
            label=f"installed {name} version",
        )
        if name in seen:
            raise InspectionError(f"installed package is duplicated: {name}")
        seen.add(name)
        read_text = getattr(distribution, "read_text", None)
        if not callable(read_text):
            raise InspectionError(f"installed package cannot be inventoried: {name}")
        installer = (read_text("INSTALLER") or "").strip() or "unknown"
        packages.append(
            {
                "name": name,
                "version": version,
                "installer": _fixed_string(
                    installer,
                    label=f"installed {name} installer",
                ),
                "archive_sha256": selected_archives.get((name, version)),
                "record_sha256": _optional_text_hash(read_text("RECORD")),
                "wheel_metadata_sha256": _optional_text_hash(read_text("WHEEL")),
            }
        )
    return sorted(packages, key=lambda item: item["name"])


def _container_facts() -> dict[str, str]:
    try:
        import torch
    except ImportError as error:
        raise InspectionError("container does not provide PyTorch") from error
    nccl = torch.cuda.nccl.version()
    nccl_version = (
        ".".join(map(str, nccl))
        if isinstance(nccl, tuple)
        else str(nccl or "")
    )
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version() or ""),
        "nccl": nccl_version,
    }


def build_inspection_artifact(
    *,
    os_release_bytes: bytes,
    install_report_bytes: bytes,
    distributions: Sequence[object],
    container_facts: Mapping[str, object],
    python_version: str,
    python_implementation: str,
) -> dict[str, object]:
    """Build and validate the complete inspection artifact from measured inputs."""

    if set(container_facts) != _FACT_FIELDS:
        raise InspectionError("container fact fields do not match schema")
    normalized_facts = {
        field: _fixed_string(
            container_facts[field],
            label=f"container {field}",
        )
        for field in sorted(_FACT_FIELDS)
    }
    report_sha256, selected = _install_report(install_report_bytes)
    packages = _installed_packages(distributions, selected)
    selected_names = {name for name, _ in selected}
    if not selected_names <= {item["name"] for item in packages}:
        raise InspectionError("pip report package is absent from installed inventory")
    artifact = {
        "schema_version": 1,
        "artifact_type": INSPECTION_TYPE,
        "os_release": _os_release(os_release_bytes),
        "python": {
            "implementation": _fixed_string(
                python_implementation,
                label="Python implementation",
            ),
            "version": _fixed_string(
                python_version,
                label="Python version",
            ),
        },
        "container_facts": normalized_facts,
        "installed_python_packages": packages,
        "inventory_method": "importlib.metadata.distributions",
        "installed_distribution_count": len(packages),
        "project_install_report_sha256": report_sha256,
    }
    parse_inspection_artifact_bytes(canonical_json(artifact))
    return artifact


def parse_inspection_artifact_bytes(data: bytes) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InspectionError("inspection artifact is invalid JSON") from error
    if not isinstance(value, dict) or data != canonical_json(value):
        raise InspectionError("inspection artifact must be canonical JSON")
    if set(value) != {
        "schema_version",
        "artifact_type",
        "os_release",
        "python",
        "container_facts",
        "installed_python_packages",
        "inventory_method",
        "installed_distribution_count",
        "project_install_report_sha256",
    }:
        raise InspectionError("inspection artifact fields do not match schema")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["artifact_type"] != INSPECTION_TYPE
        or not isinstance(value["installed_python_packages"], list)
        or value["inventory_method"] != "importlib.metadata.distributions"
        or type(value["installed_distribution_count"]) is not int
        or value["installed_distribution_count"]
        != len(value["installed_python_packages"])
    ):
        raise InspectionError("inspection artifact identity is invalid")
    packages = value["installed_python_packages"]
    names: list[str] = []
    for package in packages:
        if not isinstance(package, dict) or set(package) != _PACKAGE_FIELDS:
            raise InspectionError("installed package fields do not match schema")
        name = _normalized_name(package["name"])
        names.append(name)
        _fixed_string(package["version"], label=f"installed {name} version")
        _fixed_string(package["installer"], label=f"installed {name} installer")
        for field in (
            "archive_sha256",
            "record_sha256",
            "wheel_metadata_sha256",
        ):
            _optional_sha256(package[field], label=f"installed {name} {field}")
    if names != sorted(set(names)):
        raise InspectionError("installed package inventory is not unique and sorted")
    if not isinstance(value["os_release"], dict) or set(value["os_release"]) != {
        "id",
        "version_id",
        "pretty_name",
    }:
        raise InspectionError("inspection OS release fields do not match schema")
    if not isinstance(value["python"], dict) or set(value["python"]) != {
        "implementation",
        "version",
    }:
        raise InspectionError("inspection Python fields do not match schema")
    if not isinstance(value["container_facts"], dict) or set(
        value["container_facts"]
    ) != _FACT_FIELDS:
        raise InspectionError("inspection container facts do not match schema")
    _optional_sha256(
        value["project_install_report_sha256"],
        label="project install report",
    )
    return value


def main() -> int:
    try:
        report_path = Path("/opt/memorysplit/project-install-report.json")
        artifact = build_inspection_artifact(
            os_release_bytes=Path("/etc/os-release").read_bytes(),
            install_report_bytes=report_path.read_bytes(),
            distributions=tuple(importlib.metadata.distributions()),
            container_facts=_container_facts(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
        )
    except (InspectionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
