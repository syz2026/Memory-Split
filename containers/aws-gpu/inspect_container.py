#!/opt/conda/bin/python
"""Emit one canonical, offline inventory of the running container."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


INSPECTION_TYPE = "memorysplit-container-inspection-v2"
BASE_IMAGE_DIGEST = (
    "sha256:1414a836532f22b271c03b7ccdbdff3d"
    "aa0591975b3bd9a3cf51601a45b37f4f"
)
DLC_PYTHON = "/opt/conda/bin/python"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_NAME_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_FACT_FIELDS = {"python", "pytorch", "cuda", "cudnn", "nccl"}
_PACKAGE_FIELDS = {
    "name",
    "version",
    "installer",
    "provenance",
    "metadata_file_sha256",
    "record_file_sha256",
    "wheel_file_sha256",
    "installed_files",
    "installed_files_sha256",
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


def _path_commitment(path: Path) -> dict[str, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        details = absolute.lstat()
    except OSError as error:
        raise InspectionError(f"installed path is missing: {absolute}") from error
    if stat.S_ISREG(details.st_mode):
        kind = "regular"
        try:
            payload = absolute.read_bytes()
        except OSError as error:
            raise InspectionError(f"installed file cannot be read: {absolute}") from error
    elif stat.S_ISLNK(details.st_mode):
        kind = "symlink"
        payload = os.readlink(absolute).encode("utf-8")
    elif stat.S_ISDIR(details.st_mode):
        kind = "directory"
        payload = b"directory"
    else:
        raise InspectionError(f"installed path type is unsupported: {absolute}")
    return {
        "path": str(absolute),
        "type": kind,
        "commitment_sha256": _sha256_bytes(payload),
    }


def _path_commitments(paths: Sequence[Path]) -> list[dict[str, str]]:
    rendered = [_path_commitment(path) for path in paths]
    rendered.sort(key=lambda item: item["path"])
    names = [item["path"] for item in rendered]
    if not rendered or names != sorted(set(names)):
        raise InspectionError("installed file inventory is empty or duplicated")
    return rendered


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
    *,
    base_image_digest: str,
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
        locate_file = getattr(distribution, "locate_file", None)
        files = getattr(distribution, "files", None)
        if (
            not callable(read_text)
            or not callable(locate_file)
            or not isinstance(files, (list, tuple))
            or not files
        ):
            raise InspectionError(f"installed package cannot be inventoried: {name}")
        installer = (read_text("INSTALLER") or "").strip() or "unknown"
        commitments = _path_commitments(
            tuple(Path(locate_file(item)) for item in files)
        )
        by_basename: dict[str, list[str]] = {}
        for commitment in commitments:
            basename = Path(commitment["path"]).name
            by_basename.setdefault(basename, []).append(
                commitment["commitment_sha256"]
            )
        required_metadata: dict[str, str] = {}
        for filename in ("METADATA", "RECORD", "WHEEL"):
            matches = by_basename.get(filename, [])
            if len(matches) != 1:
                raise InspectionError(
                    f"installed package lacks one exact {filename}: {name}"
                )
            required_metadata[filename] = matches[0]
        selected_archive = selected_archives.get((name, version))
        provenance = (
            {
                "kind": "project-wheel",
                "archive_sha256": selected_archive,
            }
            if selected_archive is not None
            else {
                "kind": "inherited-base-image",
                "base_image_digest": base_image_digest,
            }
        )
        packages.append(
            {
                "name": name,
                "version": version,
                "installer": _fixed_string(
                    installer,
                    label=f"installed {name} installer",
                ),
                "provenance": provenance,
                "metadata_file_sha256": required_metadata["METADATA"],
                "record_file_sha256": required_metadata["RECORD"],
                "wheel_file_sha256": required_metadata["WHEEL"],
                "installed_files": commitments,
                "installed_files_sha256": _sha256_bytes(
                    canonical_json(commitments)
                ),
            }
        )
    normalized = sorted(packages, key=lambda item: item["name"])
    reported = set(selected_archives)
    installed = {(item["name"], item["version"]) for item in normalized}
    if not reported <= installed:
        raise InspectionError("pip report package is absent from installed inventory")
    return normalized


def _validate_path_commitments(
    value: object,
    *,
    label: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise InspectionError(f"{label} must be one nonempty file commitment list")
    paths: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "type",
            "commitment_sha256",
        }:
            raise InspectionError(f"{label} item fields do not match schema")
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or item["type"] not in {"regular", "symlink", "directory"}
            or not isinstance(item["commitment_sha256"], str)
            or _SHA256_RE.fullmatch(item["commitment_sha256"]) is None
        ):
            raise InspectionError(f"{label} item is invalid")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise InspectionError(f"{label} paths are not unique and sorted")
    return value


def _validate_os_package_inventory(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "manager",
        "database_root",
        "database_files",
        "database_tree_sha256",
        "packages",
        "package_count",
    }:
        raise InspectionError("OS package inventory fields do not match schema")
    if (
        value["manager"] != "dpkg"
        or value["database_root"] != "/var/lib/dpkg"
        or not isinstance(value["packages"], list)
        or not value["packages"]
        or type(value["package_count"]) is not int
        or value["package_count"] != len(value["packages"])
    ):
        raise InspectionError("OS package inventory identity is invalid")
    database_files = _validate_path_commitments(
        value["database_files"],
        label="dpkg database files",
    )
    if value["database_tree_sha256"] != _sha256_bytes(
        canonical_json(database_files)
    ):
        raise InspectionError("dpkg database tree commitment is invalid")
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
            raise InspectionError("OS package fields do not match schema")
        name = _fixed_string(package["name"], label="OS package name")
        names.append(name)
        for field in ("version", "architecture", "status"):
            _fixed_string(
                package[field],
                label=f"OS package {name} {field}",
            )
        if package["status"] != "ii":
            raise InspectionError(f"OS package {name} is not fully installed")
        files = _validate_path_commitments(
            package["installed_files"],
            label=f"OS package {name} files",
        )
        if package["installed_files_sha256"] != _sha256_bytes(
            canonical_json(files)
        ):
            raise InspectionError(
                f"OS package {name} file commitment is invalid"
            )
    if names != sorted(set(names)):
        raise InspectionError("OS packages are not unique and sorted")
    return value


def _run_dpkg(argv: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            list(argv),
            cwd="/",
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise InspectionError("dpkg inventory command failed") from error
    if completed.returncode != 0 or completed.stderr:
        raise InspectionError("dpkg inventory command was not clean")
    return bytes(completed.stdout)


def collect_dpkg_inventory(
    *,
    database_root: Path = Path("/var/lib/dpkg"),
) -> dict[str, object]:
    """Measure every dpkg record and every path assigned to each package."""

    query = _run_dpkg(
        (
            "/usr/bin/dpkg-query",
            "--show",
            "--showformat=${binary:Package}\\t${Version}\\t"
            "${Architecture}\\t${db:Status-Abbrev}\\n",
        )
    )
    try:
        lines = query.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise InspectionError("dpkg package database is not UTF-8") from error
    packages: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 4:
            raise InspectionError("dpkg package row is malformed")
        name, version, architecture, status = (
            _fixed_string(part.strip(), label="dpkg package field")
            for part in parts
        )
        if status != "ii":
            continue
        if name in seen:
            raise InspectionError("dpkg package row is duplicated")
        seen.add(name)
        list_output = _run_dpkg(
            ("/usr/bin/dpkg-query", "--listfiles", name)
        )
        try:
            paths = tuple(
                Path(line)
                for line in list_output.decode(
                    "utf-8",
                    errors="strict",
                ).splitlines()
                if line.startswith("/")
            )
        except UnicodeDecodeError as error:
            raise InspectionError("dpkg file list is not UTF-8") from error
        commitments = _path_commitments(paths)
        packages.append(
            {
                "name": name,
                "version": version,
                "architecture": architecture,
                "status": status,
                "installed_files": commitments,
                "installed_files_sha256": _sha256_bytes(
                    canonical_json(commitments)
                ),
            }
        )
    packages.sort(key=lambda item: item["name"])
    database_paths = tuple(
        path
        for path in database_root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    database_files = _path_commitments(database_paths)
    inventory = {
        "manager": "dpkg",
        "database_root": str(database_root),
        "database_files": database_files,
        "database_tree_sha256": _sha256_bytes(
            canonical_json(database_files)
        ),
        "packages": packages,
        "package_count": len(packages),
    }
    return _validate_os_package_inventory(inventory)


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
    os_package_inventory: Mapping[str, object],
    base_image_digest: str,
    container_facts: Mapping[str, object],
    python_version: str,
    python_implementation: str,
    python_executable: str,
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
    if (
        not isinstance(base_image_digest, str)
        or not base_image_digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(base_image_digest.removeprefix("sha256:"))
        is None
    ):
        raise InspectionError("base image digest is invalid")
    packages = _installed_packages(
        distributions,
        selected,
        base_image_digest=base_image_digest,
    )
    artifact = {
        "schema_version": 2,
        "artifact_type": INSPECTION_TYPE,
        "os_release": _os_release(os_release_bytes),
        "os_packages": _validate_os_package_inventory(
            dict(os_package_inventory)
        ),
        "python": {
            "implementation": _fixed_string(
                python_implementation,
                label="Python implementation",
            ),
            "version": _fixed_string(
                python_version,
                label="Python version",
            ),
            "executable": _fixed_string(
                python_executable,
                label="Python executable",
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
        "os_packages",
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
        or value["schema_version"] != 2
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
    inherited_names: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or set(package) != _PACKAGE_FIELDS:
            raise InspectionError("installed package fields do not match schema")
        name = _normalized_name(package["name"])
        names.append(name)
        _fixed_string(package["version"], label=f"installed {name} version")
        _fixed_string(package["installer"], label=f"installed {name} installer")
        provenance = package["provenance"]
        if not isinstance(provenance, dict) or provenance.get("kind") not in {
            "project-wheel",
            "inherited-base-image",
        }:
            raise InspectionError(f"installed {name} provenance is invalid")
        if provenance["kind"] == "project-wheel":
            if set(provenance) != {"kind", "archive_sha256"}:
                raise InspectionError(f"installed {name} wheel provenance is open")
            _optional_sha256(
                provenance["archive_sha256"],
                label=f"installed {name} archive",
            )
            if provenance["archive_sha256"] is None:
                raise InspectionError(f"installed {name} archive hash is missing")
        else:
            if (
                set(provenance) != {"kind", "base_image_digest"}
                or provenance["base_image_digest"] != BASE_IMAGE_DIGEST
            ):
                raise InspectionError(f"installed {name} base provenance is forged")
            inherited_names.add(name)
        metadata_hashes: list[str] = []
        for field in (
            "metadata_file_sha256",
            "record_file_sha256",
            "wheel_file_sha256",
        ):
            digest = _optional_sha256(
                package[field],
                label=f"installed {name} {field}",
            )
            if digest is None:
                raise InspectionError(f"installed {name} metadata hash is missing")
            metadata_hashes.append(digest)
        files = _validate_path_commitments(
            package["installed_files"],
            label=f"installed {name} files",
        )
        if package["installed_files_sha256"] != _sha256_bytes(
            canonical_json(files)
        ):
            raise InspectionError(f"installed {name} file commitment is invalid")
        committed_hashes = {item["commitment_sha256"] for item in files}
        if not set(metadata_hashes) <= committed_hashes:
            raise InspectionError(
                f"installed {name} metadata is absent from file inventory"
            )
    if names != sorted(set(names)):
        raise InspectionError("installed package inventory is not unique and sorted")
    if "torch" not in inherited_names:
        raise InspectionError("inherited DLC Torch is not bound to base image")
    if not isinstance(value["os_release"], dict) or set(value["os_release"]) != {
        "id",
        "version_id",
        "pretty_name",
    }:
        raise InspectionError("inspection OS release fields do not match schema")
    _validate_os_package_inventory(value["os_packages"])
    if not isinstance(value["python"], dict) or set(value["python"]) != {
        "implementation",
        "version",
        "executable",
    }:
        raise InspectionError("inspection Python fields do not match schema")
    if value["python"]["executable"] != DLC_PYTHON:
        raise InspectionError("inspection did not run with the pinned DLC Python")
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
            os_package_inventory=collect_dpkg_inventory(),
            base_image_digest=BASE_IMAGE_DIGEST,
            container_facts=_container_facts(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            python_executable=sys.executable,
        )
    except (InspectionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
