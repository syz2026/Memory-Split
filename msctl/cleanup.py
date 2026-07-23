"""Hash-bound cleanup planning and race-free application."""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

from .approval import verify_scope_approval
from .contracts import bind_release, load_release, load_run_manifest
from .errors import MsctlError
from .fsutil import (
    hash_fd,
    open_directory,
    open_directory_at,
    open_regular_at,
)
from .jsonutil import (
    canonical_sha256,
    load_json,
    portable_relative,
    require_exact_keys,
    require_nonnegative_int,
    require_object,
    require_schema_version,
    require_sha256,
)
from .profile import IlluminaProfile
from .slurm import resource_request


_DISPOSABLE_COMPONENTS = {
    ".cache",
    "cache",
    "caches",
    "logs",
    "slurm-logs",
}


def _is_disposable(relative: Path) -> bool:
    return bool(set(relative.parts) & _DISPOSABLE_COMPONENTS)


def _scan_cleanup(
    directory_fd: int,
    *,
    prefix: Path = Path(),
) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for name in sorted(os.listdir(directory_fd)):
        relative = prefix / name
        metadata = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(metadata.st_mode):
            raise MsctlError(
                "CLEANUP_UNSAFE",
                "cleanup root must not contain symlinks",
                details={"path": relative.as_posix()},
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = open_directory_at(
                directory_fd,
                name,
                label="cleanup root",
            )
            try:
                files.extend(_scan_cleanup(child_fd, prefix=relative))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise MsctlError(
                "CLEANUP_UNSAFE",
                "cleanup root contains a non-regular entry",
                details={"path": relative.as_posix()},
            )
        if not _is_disposable(relative):
            continue
        descriptor, parent_fd, _ = open_regular_at(
            directory_fd,
            name,
            label="cleanup candidate",
        )
        try:
            size, digest = hash_fd(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    return files


def make_cleanup_plan(root: Path | str) -> dict[str, object]:
    candidate = Path(root)
    try:
        root_fd = open_directory(candidate, label="cleanup root")
    except MsctlError as error:
        raise MsctlError(
            "CLEANUP_ROOT_INVALID",
            "cleanup root must be a regular directory",
        ) from error
    candidate = candidate.resolve()
    try:
        return {
            "schema_version": 1,
            "root": str(candidate),
            "files": _scan_cleanup(root_fd),
        }
    finally:
        os.close(root_fd)


def _load_plan(path: Path | str) -> dict[str, object]:
    plan = require_object(
        load_json(path, label="cleanup plan"),
        label="cleanup plan",
    )
    require_exact_keys(
        plan,
        {"schema_version", "root", "files"},
        label="cleanup plan",
    )
    try:
        require_schema_version(
            plan["schema_version"],
            label="cleanup plan.schema_version",
        )
    except MsctlError as error:
        raise MsctlError(
            "CLEANUP_PLAN_INVALID",
            "cleanup plan schema is invalid",
        ) from error
    if not isinstance(plan["root"], str):
        raise MsctlError("CLEANUP_PLAN_INVALID", "cleanup plan schema is invalid")
    if not isinstance(plan["files"], list):
        raise MsctlError("CLEANUP_PLAN_INVALID", "cleanup files must be a list")
    return plan


def apply_cleanup(
    *,
    profile: IlluminaProfile,
    plan_path: Path | str,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    if not apply:
        raise MsctlError(
            "APPLY_REQUIRED",
            "cleanup apply requires the literal --apply flag",
        )
    plan = _load_plan(plan_path)
    release = load_release(release_path)
    manifest = load_run_manifest(manifest_path, repo_root=repo_root)
    bind_release(release, manifest)
    if (
        manifest.provider != profile.provider
        or manifest.seed != 0
        or len(manifest.runs) != 2
    ):
        raise MsctlError(
            "CLEANUP_MANIFEST_INVALID",
            "Illumina cleanup requires the exact provider-owned seed-0 manifest",
        )
    scope_hash = canonical_sha256(
        {
            "plan_sha256": canonical_sha256(plan),
            "run_manifest_sha256": manifest.sha256,
        }
    )
    files = plan["files"]
    assert isinstance(files, list)
    verify_scope_approval(
        approval_path,
        operation="cleanup",
        release_sha256=release.archive_sha256,
        scope_sha256=scope_hash,
        resources=resource_request(profile, "cleanup"),
        profile=profile,
        environ=(dict(os.environ) if environ is None else environ),
    )
    root = Path(str(plan["root"]))
    try:
        root_fd = open_directory(root, label="cleanup root")
    except MsctlError as error:
        raise MsctlError(
            "CLEANUP_RACE",
            "cleanup root changed after planning",
        ) from error
    checked: list[tuple[int, int, str, str, os.stat_result]] = []
    seen: set[str] = set()
    try:
        for index, item in enumerate(files):
            row = require_object(item, label=f"cleanup files[{index}]")
            require_exact_keys(
                row,
                {"path", "bytes", "sha256"},
                label=f"cleanup files[{index}]",
            )
            relative_text = portable_relative(
                row["path"], label=f"cleanup files[{index}].path"
            )
            if relative_text in seen:
                raise MsctlError(
                    "CLEANUP_PLAN_INVALID",
                    "cleanup plan has duplicate paths",
                )
            seen.add(relative_text)
            relative = Path(relative_text)
            if not _is_disposable(relative):
                raise MsctlError(
                    "CLEANUP_PLAN_INVALID",
                    "cleanup plan includes a protected path",
                    details={"path": relative_text},
                )
            expected_hash = require_sha256(
                row["sha256"], label=f"cleanup files[{index}].sha256"
            )
            expected_bytes = require_nonnegative_int(
                row["bytes"],
                label=f"cleanup files[{index}].bytes",
            )
            descriptor, parent_fd, basename = open_regular_at(
                root_fd,
                relative_text,
                label=f"cleanup files[{index}].path",
            )
            metadata = os.fstat(descriptor)
            try:
                size, digest = hash_fd(descriptor)
            except MsctlError as error:
                os.close(descriptor)
                os.close(parent_fd)
                raise MsctlError(
                    "CLEANUP_RACE",
                    "cleanup candidate changed while being checked",
                    details={"path": relative_text},
                ) from error
            if size != expected_bytes or digest != expected_hash:
                os.close(descriptor)
                os.close(parent_fd)
                raise MsctlError(
                    "CLEANUP_RACE",
                    "cleanup candidate changed after planning",
                    details={"path": relative_text},
                )
            checked.append(
                (descriptor, parent_fd, basename, relative_text, metadata)
            )

        quarantine = f".msctl-cleanup-{secrets.token_hex(12)}"
        os.mkdir(quarantine, mode=0o700, dir_fd=root_fd)
        quarantine_fd = open_directory_at(
            root_fd,
            quarantine,
            label="cleanup quarantine",
        )
        moved: list[tuple[int, str, str]] = []
        try:
            for index, (
                descriptor,
                parent_fd,
                basename,
                relative_text,
                opened,
            ) in enumerate(checked):
                quarantine_name = f"{index:08d}"
                os.rename(
                    basename,
                    quarantine_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=quarantine_fd,
                )
                moved_metadata = os.stat(
                    quarantine_name,
                    dir_fd=quarantine_fd,
                    follow_symlinks=False,
                )
                if (
                    moved_metadata.st_dev != opened.st_dev
                    or moved_metadata.st_ino != opened.st_ino
                ):
                    try:
                        os.rename(
                            quarantine_name,
                            basename,
                            src_dir_fd=quarantine_fd,
                            dst_dir_fd=parent_fd,
                        )
                    except OSError:
                        pass
                    raise MsctlError(
                        "CLEANUP_RACE",
                        "cleanup candidate inode changed before quarantine",
                        details={"path": relative_text},
                    )
                moved.append((parent_fd, basename, quarantine_name))
            for _, _, quarantine_name in moved:
                os.unlink(quarantine_name, dir_fd=quarantine_fd)
            os.fsync(quarantine_fd)
        except Exception:
            for parent_fd, basename, quarantine_name in reversed(moved):
                try:
                    os.rename(
                        quarantine_name,
                        basename,
                        src_dir_fd=quarantine_fd,
                        dst_dir_fd=parent_fd,
                    )
                except OSError:
                    pass
            raise
        finally:
            os.close(quarantine_fd)
            try:
                os.rmdir(quarantine, dir_fd=root_fd)
            except OSError:
                pass
        os.fsync(root_fd)
    finally:
        for descriptor, parent_fd, _, _, _ in checked:
            os.close(descriptor)
            os.close(parent_fd)
        os.close(root_fd)
    return {
        "plan_sha256": scope_hash,
        "run_manifest_sha256": manifest.sha256,
        "deleted_files": len(checked),
        "deleted_bytes": sum(int(row["bytes"]) for row in files),
    }
