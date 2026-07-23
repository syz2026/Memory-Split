"""Closed-allowlist result evidence collection."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .errors import MsctlError
from .jsonutil import atomic_write, atomic_write_json, canonical_json, sha256_file


_DIRECT_FILES = {
    "SHA256SUMS",
    "config.json",
    "config.yaml",
    "config.yml",
    "log.jsonl",
    "runtime-config.json",
}
_EVIDENCE_SUFFIXES = (".json", ".jsonl", ".md", ".png", ".txt")


def _allowed(relative: Path) -> bool:
    parts = relative.parts
    if len(parts) < 2:
        return False
    nested = parts[1:]
    if len(nested) == 1 and (
        nested[0] in _DIRECT_FILES
        or nested[0].endswith("-evidence.json")
        or nested[0].endswith("-receipt.json")
    ):
        return True
    return (
        nested[0] in {"analysis", "evals", "manifests", "reports"}
        and relative.suffix.lower() in _EVIDENCE_SUFFIXES
    )


def plan_collection(
    *,
    source: Path | str,
    out: Path | str,
) -> dict[str, object]:
    root = Path(source)
    destination = Path(out)
    if root.is_symlink() or not root.is_dir():
        raise MsctlError(
            "COLLECT_SOURCE_INVALID",
            "collection source must be a regular directory",
        )
    root = root.resolve()
    try:
        destination.resolve(strict=False).relative_to(root)
    except ValueError:
        pass
    else:
        raise MsctlError(
            "COLLECT_DESTINATION_INVALID",
            "collection output must be outside the source root",
        )
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MsctlError(
                "COLLECT_SOURCE_UNSAFE",
                "collection source must not contain symlinks",
                details={"path": path.relative_to(root).as_posix()},
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not _allowed(relative):
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "source": str(root),
        "files": rows,
    }


def collect_evidence(
    *,
    source: Path | str,
    out: Path | str,
    apply: bool,
) -> dict[str, object]:
    plan = plan_collection(source=source, out=out)
    if not apply:
        return {
            "plan": plan,
            "collected_files": 0,
            "collected_bytes": 0,
        }
    source_root = Path(str(plan["source"]))
    destination = Path(out)
    if destination.is_symlink():
        raise MsctlError(
            "COLLECT_DESTINATION_INVALID",
            "collection output must not be a symlink",
        )
    if destination.exists():
        receipt = destination / "COLLECTION.json"
        if receipt.is_file() and not receipt.is_symlink():
            existing = plan_collection(source=source, out=out)
            files = existing["files"]
            expected_receipt = (
                canonical_json(
                    {
                        "schema_version": 1,
                        "files": files,
                    }
                )
                + b"\n"
            )
            expected_paths = {
                str(row["path"]) for row in files
            } | {"COLLECTION.json"}
            actual_paths = {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            if receipt.read_bytes() == expected_receipt and actual_paths == expected_paths and all(
                (destination / str(row["path"])).is_file()
                and sha256_file(destination / str(row["path"]))
                == row["sha256"]
                for row in files
            ):
                return {
                    "plan": existing,
                    "collected_files": len(files),
                    "collected_bytes": sum(int(row["bytes"]) for row in files),
                    "created": False,
                }
        raise MsctlError(
            "COLLECT_DESTINATION_EXISTS",
            "refusing to overwrite an existing collection",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        files = plan["files"]
        assert isinstance(files, list)
        for row in files:
            relative = Path(str(row["path"]))
            source_path = source_root / relative
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or source_path.stat().st_size != row["bytes"]
                or sha256_file(source_path) != row["sha256"]
            ):
                raise MsctlError(
                    "COLLECT_RACE",
                    "source evidence changed after planning",
                    details={"path": relative.as_posix()},
                )
        for row in files:
            relative = Path(str(row["path"]))
            atomic_write(
                temporary / relative,
                (source_root / relative).read_bytes(),
            )
        atomic_write_json(
            temporary / "COLLECTION.json",
            {
                "schema_version": 1,
                "files": files,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "plan": plan,
        "collected_files": len(files),
        "collected_bytes": sum(int(row["bytes"]) for row in files),
        "created": True,
    }
