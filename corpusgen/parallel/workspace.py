"""Owned shared namespaces for disjoint parallel task results."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from pathlib import Path

from .canonical import canonical_json_bytes, sha256_hex
from .safeio import (
    atomic_write_or_match,
    clean_owned_temporaries,
    entry_lstat,
    fsync_directory,
    is_owned_temporary,
    list_entries,
    open_directory_at,
    open_directory_path,
    open_parent_directory,
    read_regular_file,
    unlink_regular_if_matches,
)
from .tasks import (
    TaskResult,
    task_result_filename,
    task_result_from_bytes,
    task_result_to_bytes,
)

_WORKSPACE_ROOT = ".memorysplit-v2-builds"
_OWNER_NAME = ".task-workspace-owner.json"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class IncompleteTaskResults(ValueError):
    """Raised when a coordinator runs before every task result is present."""


def _digest(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA-256")
    return value


def _identifier(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} is not a safe scheduler identifier")
    return value


def _workspace_name(scheduler_id: str, nonce: str) -> str:
    return (
        f"{_identifier(scheduler_id, 'scheduler_id')}--"
        f"{_identifier(nonce, 'nonce')}"
    )


def _owner_bytes(
    build_id: str,
    *,
    scheduler_id: str,
    nonce: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "build_id": _digest(build_id, "workspace build_id"),
            "kind": "parallel-task-workspace",
            "nonce": _identifier(nonce, "nonce"),
            "scheduler_id": _identifier(scheduler_id, "scheduler_id"),
        }
    )


def task_workspace_path(
    shared_root: Path | str,
    build_id: str,
    *,
    scheduler_id: str,
    nonce: str,
) -> Path:
    return (
        Path(shared_root)
        / _WORKSPACE_ROOT
        / _digest(build_id, "workspace build_id")
        / _workspace_name(scheduler_id, nonce)
    )


def _open_workspace(
    shared_root: Path | str,
    build_id: str,
    *,
    scheduler_id: str,
    nonce: str,
    create: bool,
) -> tuple[int, Path, bytes, str]:
    workspace = task_workspace_path(
        shared_root,
        build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
    )
    root_fd = open_directory_path(shared_root, create=create)
    namespace_fd = -1
    build_fd = -1
    try:
        namespace_fd, _created = open_directory_at(
            root_fd,
            _WORKSPACE_ROOT,
            create=create,
        )
        build_fd, _created = open_directory_at(
            namespace_fd,
            build_id,
            create=create,
        )
        workspace_fd, created = open_directory_at(
            build_fd,
            workspace.name,
            create=create,
        )
    finally:
        if build_fd >= 0:
            os.close(build_fd)
        if namespace_fd >= 0:
            os.close(namespace_fd)
        os.close(root_fd)
    owner_payload = _owner_bytes(
        build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
    )
    owner_token = sha256_hex(owner_payload)
    try:
        fcntl.flock(workspace_fd, fcntl.LOCK_EX)
        if created:
            if list_entries(workspace_fd):
                raise ValueError("new task workspace is not empty")
            atomic_write_or_match(
                workspace_fd,
                _OWNER_NAME,
                owner_payload,
                owner=owner_token,
            )
        else:
            try:
                actual_owner = read_regular_file(workspace_fd, _OWNER_NAME)
            except FileNotFoundError as error:
                raise ValueError("task workspace ownership marker is missing") from error
            if actual_owner != owner_payload:
                raise ValueError("task workspace ownership marker mismatch")
    except BaseException:
        os.close(workspace_fd)
        raise
    return workspace_fd, workspace, owner_payload, owner_token


def _result_names(build_id: str, task_count: int) -> set[str]:
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
    ):
        raise ValueError("expected_task_count must be a positive integer")
    return {
        f"task-{task_index:05d}-of-{task_count:05d}-{build_id}.json"
        for task_index in range(task_count)
    }


def _validate_workspace_entries(
    workspace_fd: int,
    *,
    result_names: set[str],
    owner_token: str,
) -> None:
    for name in list_entries(workspace_fd):
        metadata = entry_lstat(workspace_fd, name)
        if name == _OWNER_NAME or name in result_names:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"task workspace entry is unsafe: {name}")
        elif is_owned_temporary(name, result_names, owner_token):
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"task result temporary is unsafe: {name}")
        else:
            raise ValueError(f"foreign task workspace entry: {name}")


def publish_task_result(
    shared_root: Path | str,
    result: TaskResult,
    *,
    scheduler_id: str,
    nonce: str,
) -> Path:
    """Atomically install one task result into its owned shared workspace."""

    if not isinstance(result, TaskResult):
        raise TypeError("result must be a TaskResult")
    workspace_fd, workspace, _owner_payload, owner_token = _open_workspace(
        shared_root,
        result.build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
        create=True,
    )
    try:
        result_names = _result_names(result.build_id, result.task_count)
        _validate_workspace_entries(
            workspace_fd,
            result_names=result_names,
            owner_token=owner_token,
        )
        clean_owned_temporaries(
            workspace_fd,
            final_names=result_names,
            owner=owner_token,
        )
        name = task_result_filename(result)
        atomic_write_or_match(
            workspace_fd,
            name,
            task_result_to_bytes(result),
            owner=owner_token,
        )
        return workspace / name
    finally:
        os.close(workspace_fd)


def load_task_results(
    shared_root: Path | str,
    build_id: str,
    *,
    scheduler_id: str,
    nonce: str,
    expected_task_count: int,
) -> tuple[TaskResult, ...]:
    """Load exactly one complete, foreign-free result set."""

    workspace_fd, _workspace, _owner_payload, owner_token = _open_workspace(
        shared_root,
        build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
        create=False,
    )
    try:
        result_names = _result_names(build_id, expected_task_count)
        _validate_workspace_entries(
            workspace_fd,
            result_names=result_names,
            owner_token=owner_token,
        )
        clean_owned_temporaries(
            workspace_fd,
            final_names=result_names,
            owner=owner_token,
        )
        actual_names = set(list_entries(workspace_fd)) - {_OWNER_NAME}
        missing = sorted(result_names - actual_names)
        if missing:
            raise IncompleteTaskResults(
                "missing task results: " + ", ".join(missing)
            )
        if actual_names != result_names:
            raise ValueError("foreign task workspace entries remain")
        results = tuple(
            task_result_from_bytes(read_regular_file(workspace_fd, name))
            for name in sorted(result_names)
        )
        for name, result in zip(sorted(result_names), results, strict=True):
            if task_result_filename(result) != name:
                raise ValueError(f"task result filename binding mismatch: {name}")
        return results
    finally:
        os.close(workspace_fd)


def cleanup_task_workspace(
    workspace: Path | str,
    *,
    build_id: str,
    scheduler_id: str,
    nonce: str,
) -> None:
    """Remove a flat task workspace only after exact ownership proof."""

    workspace_path = Path(workspace)
    expected_path = task_workspace_path(
        workspace_path.parents[2],
        build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
    )
    if workspace_path != expected_path:
        raise ValueError("task workspace ownership path mismatch")
    parent_fd, workspace_name = open_parent_directory(workspace_path)
    workspace_fd = -1
    try:
        workspace_fd, _created = open_directory_at(parent_fd, workspace_name)
        fcntl.flock(workspace_fd, fcntl.LOCK_EX)
        expected_owner = _owner_bytes(
            build_id,
            scheduler_id=scheduler_id,
            nonce=nonce,
        )
        try:
            actual_owner = read_regular_file(workspace_fd, _OWNER_NAME)
        except FileNotFoundError as error:
            raise ValueError("task workspace ownership marker is missing") from error
        if actual_owner != expected_owner:
            raise ValueError("task workspace ownership marker mismatch")
        owner_token = sha256_hex(expected_owner)
        result_pattern = re.compile(
            rf"task-\d{{5}}-of-\d{{5}}-{re.escape(build_id)}\.json\Z"
        )
        temporary_pattern = re.compile(
            rf"\.task-\d{{5}}-of-\d{{5}}-{re.escape(build_id)}"
            rf"\.json\.tmp-{owner_token}-[0-9a-f]{{16}}\Z"
        )
        names = list_entries(workspace_fd)
        for name in names:
            metadata = entry_lstat(workspace_fd, name)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"task workspace cleanup entry is unsafe: {name}")
            if (
                name != _OWNER_NAME
                and result_pattern.fullmatch(name) is None
                and temporary_pattern.fullmatch(name) is None
            ):
                raise ValueError(f"foreign task workspace cleanup entry: {name}")
        for name in names:
            payload = read_regular_file(workspace_fd, name)
            unlink_regular_if_matches(workspace_fd, name, payload)
        os.close(workspace_fd)
        workspace_fd = -1
        os.rmdir(workspace_name, dir_fd=parent_fd)
        fsync_directory(parent_fd)
    finally:
        if workspace_fd >= 0:
            os.close(workspace_fd)
        os.close(parent_fd)
