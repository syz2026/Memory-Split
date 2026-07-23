"""Closed-allowlist result evidence collection."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path

from .errors import MsctlError
from .fsutil import (
    atomic_write_at,
    atomic_write_json_at,
    hash_fd,
    load_json_at,
    open_directory,
    open_directory_at,
    open_parent_at,
    open_regular_at,
    read_fd,
    remove_tree_at,
    rename_noreplace_at,
)
from .jsonutil import (
    require_exact_keys,
    require_object,
    require_schema_version,
)


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


def _scan_collection(
    directory_fd: int,
    *,
    prefix: Path = Path(),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(os.listdir(directory_fd)):
        relative = prefix / name
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise MsctlError(
                "COLLECT_SOURCE_UNSAFE",
                "collection source changed while scanning",
                details={"path": relative.as_posix()},
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise MsctlError(
                "COLLECT_SOURCE_UNSAFE",
                "collection source must not contain symlinks",
                details={"path": relative.as_posix()},
            )
        if stat.S_ISDIR(metadata.st_mode):
            try:
                child_fd = open_directory_at(
                    directory_fd,
                    name,
                    label="collection source",
                )
            except MsctlError as error:
                raise MsctlError(
                    "COLLECT_SOURCE_UNSAFE",
                    "collection source contains an unsafe directory",
                    details={"path": relative.as_posix()},
                ) from error
            try:
                rows.extend(_scan_collection(child_fd, prefix=relative))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise MsctlError(
                "COLLECT_SOURCE_UNSAFE",
                "collection source contains a non-regular entry",
                details={"path": relative.as_posix()},
            )
        if not _allowed(relative):
            continue
        try:
            descriptor, parent_fd, _ = open_regular_at(
                directory_fd,
                name,
                label="collection source evidence",
            )
        except MsctlError as error:
            raise MsctlError(
                "COLLECT_SOURCE_UNSAFE",
                "collection source changed before hashing",
                details={"path": relative.as_posix()},
            ) from error
        try:
            size, digest = hash_fd(descriptor)
        except MsctlError as error:
            raise MsctlError(
                "COLLECT_SOURCE_UNSAFE",
                "collection source changed while hashing",
                details={"path": relative.as_posix()},
            ) from error
        finally:
            os.close(descriptor)
            os.close(parent_fd)
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    return rows


def _plan_from_fd(
    source_fd: int,
    *,
    source: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": str(source),
        "files": _scan_collection(source_fd),
    }


def plan_collection(
    *,
    source: Path | str,
    out: Path | str,
) -> dict[str, object]:
    root = Path(source)
    destination = Path(out)
    try:
        source_fd = open_directory(root, label="collection source")
    except MsctlError as error:
        raise MsctlError(
            "COLLECT_SOURCE_INVALID",
            "collection source must be a regular directory",
        ) from error
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
    try:
        return _plan_from_fd(source_fd, source=root)
    finally:
        os.close(source_fd)


def _verify_existing_collection_at(
    parent_fd: int,
    name: str,
    plan: dict[str, object],
) -> bool:
    try:
        destination_fd = open_directory_at(
            parent_fd,
            name,
            label="collection destination",
        )
    except MsctlError:
        return False
    try:
        receipt = require_object(
            load_json_at(
                destination_fd,
                "COLLECTION.json",
                label="collection receipt",
            ),
            label="collection receipt",
        )
        require_exact_keys(
            receipt,
            {"schema_version", "files"},
            label="collection receipt",
        )
        require_schema_version(
            receipt["schema_version"],
            label="collection receipt.schema_version",
        )
        if receipt["files"] != plan["files"]:
            return False
        expected_paths = {
            str(row["path"]) for row in receipt["files"]
        } | {"COLLECTION.json"}

        actual_paths: set[str] = set()

        def walk(fd: int, prefix: Path = Path()) -> None:
            for name in sorted(os.listdir(fd)):
                relative = prefix / name
                metadata = os.stat(
                    name,
                    dir_fd=fd,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(metadata.st_mode):
                    raise MsctlError(
                        "COLLECT_DESTINATION_INVALID",
                        "collection destination contains a symlink",
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    child = open_directory_at(
                        fd,
                        name,
                        label="collection destination",
                    )
                    try:
                        walk(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(metadata.st_mode):
                    actual_paths.add(relative.as_posix())
                else:
                    raise MsctlError(
                        "COLLECT_DESTINATION_INVALID",
                        "collection destination contains a non-regular entry",
                    )

        walk(destination_fd)
        if actual_paths != expected_paths:
            return False
        for row in receipt["files"]:
            descriptor, parent_fd, _ = open_regular_at(
                destination_fd,
                str(row["path"]),
                label="collected evidence",
            )
            try:
                size, digest = hash_fd(descriptor)
            finally:
                os.close(descriptor)
                os.close(parent_fd)
            if size != row["bytes"] or digest != row["sha256"]:
                return False
        return True
    except (MsctlError, TypeError):
        return False
    finally:
        os.close(destination_fd)


def collect_evidence(
    *,
    source: Path | str,
    out: Path | str,
    apply: bool,
) -> dict[str, object]:
    root = Path(source)
    destination = Path(out)
    try:
        source_fd = open_directory(root, label="collection source")
    except MsctlError as error:
        raise MsctlError(
            "COLLECT_SOURCE_INVALID",
            "collection source must be a regular directory",
        ) from error
    root = root.resolve()
    try:
        destination.resolve(strict=False).relative_to(root)
    except ValueError:
        pass
    else:
        os.close(source_fd)
        raise MsctlError(
            "COLLECT_DESTINATION_INVALID",
            "collection output must be outside the source root",
        )
    plan = _plan_from_fd(source_fd, source=root)
    if not apply:
        os.close(source_fd)
        return {
            "plan": plan,
            "collected_files": 0,
            "collected_bytes": 0,
        }
    if destination.name in {"", ".", ".."}:
        os.close(source_fd)
        raise MsctlError(
            "COLLECT_DESTINATION_INVALID",
            "collection output name is invalid",
        )
    try:
        destination_parent_fd = open_directory(
            destination.parent,
            label="collection destination parent",
            create=True,
        )
    except MsctlError as error:
        os.close(source_fd)
        raise MsctlError(
            "COLLECT_DESTINATION_INVALID",
            "collection output parent must not traverse symlinks",
        ) from error
    if _verify_existing_collection_at(
        destination_parent_fd,
        destination.name,
        plan,
    ):
        os.close(destination_parent_fd)
        os.close(source_fd)
        files = plan["files"]
        assert isinstance(files, list)
        return {
            "plan": plan,
            "collected_files": len(files),
            "collected_bytes": sum(int(row["bytes"]) for row in files),
            "created": False,
        }
    try:
        os.stat(
            destination.name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        os.close(destination_parent_fd)
        os.close(source_fd)
        raise MsctlError(
            "COLLECT_DESTINATION_EXISTS",
            "refusing to overwrite an existing collection",
        )
    temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    os.mkdir(temporary, mode=0o700, dir_fd=destination_parent_fd)
    temporary_fd = open_directory_at(
        destination_parent_fd,
        temporary,
        label="private collection staging",
    )
    try:
        files = plan["files"]
        assert isinstance(files, list)
        for row in files:
            relative_text = str(row["path"])
            descriptor, parent_fd, _ = open_regular_at(
                source_fd,
                relative_text,
                label="source evidence",
            )
            try:
                before = os.fstat(descriptor)
                data = read_fd(descriptor)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
                os.close(parent_fd)
            digest = hashlib.sha256(data).hexdigest()
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or len(data) != row["bytes"]
                or digest != row["sha256"]
            ):
                raise MsctlError(
                    "COLLECT_RACE",
                    "source evidence changed after planning",
                    details={"path": relative_text},
                )
            output_parent_fd, output_name = open_parent_at(
                temporary_fd,
                relative_text,
                label="collected evidence output",
                create=True,
            )
            try:
                atomic_write_at(
                    output_parent_fd,
                    output_name,
                    data,
                    label="collected evidence",
                )
            finally:
                os.close(output_parent_fd)
        atomic_write_json_at(
            temporary_fd,
            "COLLECTION.json",
            {
                "schema_version": 1,
                "files": files,
            },
            label="collection receipt",
        )
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        try:
            rename_noreplace_at(
                destination_parent_fd,
                temporary,
                destination_parent_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise MsctlError(
                "COLLECT_DESTINATION_EXISTS",
                "collection destination appeared during publication",
            ) from error
        os.fsync(destination_parent_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        os.close(source_fd)
        try:
            remove_tree_at(
                destination_parent_fd,
                temporary,
                label="private collection staging",
            )
        except (MsctlError, OSError):
            pass
        os.close(destination_parent_fd)
    return {
        "plan": plan,
        "collected_files": len(files),
        "collected_bytes": sum(int(row["bytes"]) for row in files),
        "created": True,
    }
