"""Hash-bound cleanup planning and race-free application."""

from __future__ import annotations

import os
from pathlib import Path

from .approval import verify_scope_approval
from .contracts import load_release
from .errors import MsctlError
from .jsonutil import (
    canonical_sha256,
    load_json,
    portable_relative,
    require_exact_keys,
    require_object,
    require_sha256,
    resolve_inside,
    sha256_file,
)
from .profile import IlluminaProfile


_DISPOSABLE_COMPONENTS = {
    ".cache",
    "cache",
    "caches",
    "logs",
    "slurm-logs",
}


def _is_disposable(relative: Path) -> bool:
    return bool(set(relative.parts) & _DISPOSABLE_COMPONENTS)


def make_cleanup_plan(root: Path | str) -> dict[str, object]:
    candidate = Path(root)
    if candidate.is_symlink() or not candidate.is_dir():
        raise MsctlError(
            "CLEANUP_ROOT_INVALID",
            "cleanup root must be a regular directory",
        )
    candidate = candidate.resolve()
    files = []
    for path in sorted(candidate.rglob("*")):
        if path.is_symlink():
            raise MsctlError(
                "CLEANUP_UNSAFE",
                "cleanup root must not contain symlinks",
                details={"path": path.relative_to(candidate).as_posix()},
            )
        if not path.is_file():
            continue
        relative = path.relative_to(candidate)
        if not _is_disposable(relative):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "root": str(candidate),
        "files": files,
    }


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
    if plan["schema_version"] != 1 or not isinstance(plan["root"], str):
        raise MsctlError("CLEANUP_PLAN_INVALID", "cleanup plan schema is invalid")
    if not isinstance(plan["files"], list):
        raise MsctlError("CLEANUP_PLAN_INVALID", "cleanup files must be a list")
    return plan


def apply_cleanup(
    *,
    profile: IlluminaProfile,
    plan_path: Path | str,
    release_path: Path | str,
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
    scope_hash = canonical_sha256(plan)
    files = plan["files"]
    assert isinstance(files, list)
    verify_scope_approval(
        approval_path,
        operation="cleanup",
        release_sha256=release.archive_sha256,
        scope_sha256=scope_hash,
        requested_jobs=max(1, len(files)),
        requested_gpu_hours=0.0,
        profile=profile,
        environ=(dict(os.environ) if environ is None else environ),
    )
    root = Path(str(plan["root"]))
    if root.is_symlink() or not root.is_dir():
        raise MsctlError("CLEANUP_RACE", "cleanup root changed after planning")
    root = root.resolve()
    checked: list[Path] = []
    seen: set[str] = set()
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
        path = resolve_inside(
            root,
            relative_text,
            label=f"cleanup files[{index}].path",
        )
        expected_hash = require_sha256(
            row["sha256"], label=f"cleanup files[{index}].sha256"
        )
        expected_bytes = row["bytes"]
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_hash
        ):
            raise MsctlError(
                "CLEANUP_RACE",
                "cleanup candidate changed after planning",
                details={"path": relative_text},
            )
        checked.append(path)
    for path in checked:
        path.unlink()
    directories = sorted(
        {
            parent
            for path in checked
            for parent in path.parents
            if parent != root and root in parent.parents
        },
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    return {
        "plan_sha256": scope_hash,
        "deleted_files": len(checked),
        "deleted_bytes": sum(int(row["bytes"]) for row in files),
    }
