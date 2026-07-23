"""Owned shared namespaces for disjoint parallel task results."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
from pathlib import Path

from .canonical import canonical_json_bytes, sha256_hex
from .safeio import (
    RetainedTombstoneInventory,
    RetainedTombstoneStore,
    atomic_rename_noreplace,
    atomic_write_or_match,
    clean_owned_temporaries,
    entry_exists,
    entry_lstat,
    fsync_directory,
    is_owned_temporary,
    list_entries,
    open_directory_at,
    open_directory_path,
    open_tombstone_directory,
    read_regular_file,
    retained_tombstone_inventory_fd,
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
_LOCAL_WORKSPACE_ROOT = ".memorysplit-v2-local-tasks"
_LOCAL_OWNER_NAME = ".local-task-owner.json"
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


def _local_identity_bytes(
    build_id: str,
    *,
    scheduler_id: str,
    job_id: str,
    nonce: str,
    task_index: int,
) -> bytes:
    if (
        isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or task_index < 0
    ):
        raise ValueError("task_index must be a non-negative integer")
    return canonical_json_bytes(
        {
            "build_id": _digest(build_id, "local build_id"),
            "job_id": _identifier(job_id, "job_id"),
            "nonce": _identifier(nonce, "nonce"),
            "scheduler_id": _identifier(scheduler_id, "scheduler_id"),
            "task_index": task_index,
        }
    )


def local_task_workspace_path(
    local_root: Path | str,
    build_id: str,
    *,
    scheduler_id: str,
    job_id: str,
    nonce: str,
    task_index: int,
) -> Path:
    identity = _local_identity_bytes(
        build_id,
        scheduler_id=scheduler_id,
        job_id=job_id,
        nonce=nonce,
        task_index=task_index,
    )
    return (
        Path(local_root)
        / _LOCAL_WORKSPACE_ROOT
        / _digest(build_id, "local build_id")
        / f"task-{task_index:05d}-{sha256_hex(identity)[:32]}"
    )


def _local_owner_bytes(
    result: TaskResult,
    *,
    scheduler_id: str,
    job_id: str,
    nonce: str,
) -> bytes:
    identity = _local_identity_bytes(
        result.build_id,
        scheduler_id=scheduler_id,
        job_id=job_id,
        nonce=nonce,
        task_index=result.task_index,
    )
    value = json.loads(identity)
    value.update(
        {
            "kind": "parallel-task-node-local-cache",
            "task_count": result.task_count,
        }
    )
    return canonical_json_bytes(value)


def _open_workspace(
    shared_root: Path | str,
    build_id: str,
    *,
    scheduler_id: str,
    nonce: str,
    create: bool,
) -> tuple[int, RetainedTombstoneStore | None, Path, bytes, str]:
    workspace = task_workspace_path(
        shared_root,
        build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
    )
    root_fd = open_directory_path(shared_root, create=create)
    namespace_fd = -1
    build_fd = -1
    tombstone_fd = None
    try:
        if create:
            tombstone_fd = open_tombstone_directory(root_fd)
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
    except BaseException:
        if tombstone_fd is not None:
            tombstone_fd.close()
        raise
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
            if tombstone_fd is None:
                raise RuntimeError("task workspace creation requires retained objects")
            atomic_write_or_match(
                workspace_fd,
                _OWNER_NAME,
                owner_payload,
                owner=owner_token,
                tombstone_fd=tombstone_fd,
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
        if tombstone_fd is not None:
            tombstone_fd.close()
        raise
    return workspace_fd, tombstone_fd, workspace, owner_payload, owner_token


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
    (
        workspace_fd,
        tombstone_fd,
        workspace,
        _owner_payload,
        owner_token,
    ) = _open_workspace(
        shared_root,
        result.build_id,
        scheduler_id=scheduler_id,
        nonce=nonce,
        create=True,
    )
    if tombstone_fd is None:
        os.close(workspace_fd)
        raise RuntimeError("task publication requires retained tombstones")
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
            tombstone_fd=tombstone_fd,
        )
        name = task_result_filename(result)
        atomic_write_or_match(
            workspace_fd,
            name,
            task_result_to_bytes(result),
            owner=owner_token,
            tombstone_fd=tombstone_fd,
        )
        return workspace / name
    finally:
        os.close(workspace_fd)
        if tombstone_fd is not None:
            tombstone_fd.close()


def _open_local_task_workspace(
    local_root: Path | str,
    result: TaskResult,
    *,
    scheduler_id: str,
    job_id: str,
    nonce: str,
) -> tuple[int, int, RetainedTombstoneStore, Path, bytes, str]:
    workspace = local_task_workspace_path(
        local_root,
        result.build_id,
        scheduler_id=scheduler_id,
        job_id=job_id,
        nonce=nonce,
        task_index=result.task_index,
    )
    root_fd = open_directory_path(local_root)
    namespace_fd = -1
    build_fd = -1
    workspace_fd = -1
    tombstone_fd = None
    try:
        tombstone_fd = open_tombstone_directory(root_fd)
        namespace_fd, _created = open_directory_at(
            root_fd,
            _LOCAL_WORKSPACE_ROOT,
            create=True,
        )
        build_fd, _created = open_directory_at(
            namespace_fd,
            result.build_id,
            create=True,
        )
        workspace_fd, created = open_directory_at(
            build_fd,
            workspace.name,
            create=True,
        )
    except BaseException:
        if workspace_fd >= 0:
            os.close(workspace_fd)
        if build_fd >= 0:
            os.close(build_fd)
        if tombstone_fd is not None:
            tombstone_fd.close()
        raise
    finally:
        if namespace_fd >= 0:
            os.close(namespace_fd)
        os.close(root_fd)
    owner_payload = _local_owner_bytes(
        result,
        scheduler_id=scheduler_id,
        job_id=job_id,
        nonce=nonce,
    )
    owner_token = sha256_hex(owner_payload)
    try:
        fcntl.flock(workspace_fd, fcntl.LOCK_EX)
        if created:
            if list_entries(workspace_fd):
                raise ValueError("new local task workspace is not empty")
            atomic_write_or_match(
                workspace_fd,
                _LOCAL_OWNER_NAME,
                owner_payload,
                owner=owner_token,
                tombstone_fd=tombstone_fd,
            )
        else:
            actual_owner = read_regular_file(
                workspace_fd,
                _LOCAL_OWNER_NAME,
            )
            if actual_owner != owner_payload:
                raise ValueError("local task workspace ownership marker mismatch")
    except BaseException:
        os.close(workspace_fd)
        os.close(build_fd)
        tombstone_fd.close()
        raise
    return (
        workspace_fd,
        build_fd,
        tombstone_fd,
        workspace,
        owner_payload,
        owner_token,
    )


def _validate_local_task_workspace(
    workspace_fd: int,
    *,
    result_name: str,
    owner_token: str,
) -> None:
    final_names = {result_name}
    for name in list_entries(workspace_fd):
        metadata = entry_lstat(workspace_fd, name)
        if name in {_LOCAL_OWNER_NAME, result_name}:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"local task workspace entry is unsafe: {name}")
        elif is_owned_temporary(name, final_names, owner_token):
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"local task temporary is unsafe: {name}")
        else:
            raise ValueError(f"foreign local task workspace entry: {name}")


def _cleanup_local_task_workspace(
    workspace_fd: int,
    *,
    result_name: str,
    owner_token: str,
    tombstone_fd: RetainedTombstoneStore,
) -> None:
    _validate_local_task_workspace(
        workspace_fd,
        result_name=result_name,
        owner_token=owner_token,
    )
    clean_owned_temporaries(
        workspace_fd,
        final_names={result_name},
        owner=owner_token,
        tombstone_fd=tombstone_fd,
    )
    names = sorted(
        list_entries(workspace_fd),
        key=lambda name: name == _LOCAL_OWNER_NAME,
    )
    for name in names:
        payload = read_regular_file(workspace_fd, name)
        unlink_regular_if_matches(
            workspace_fd,
            name,
            payload,
            tombstone_fd=tombstone_fd,
        )


def _restore_quarantined_directory(
    source_parent_fd: int,
    original_name: str,
    tombstone_fd: int,
    tombstone_name: str,
) -> bool:
    try:
        atomic_rename_noreplace(
            tombstone_fd,
            tombstone_name,
            source_parent_fd,
            original_name,
        )
    except FileExistsError:
        fsync_directory(source_parent_fd)
        fsync_directory(tombstone_fd)
        return False
    fsync_directory(source_parent_fd)
    fsync_directory(tombstone_fd)
    return True


def _remove_owned_empty_directory(
    parent_fd: int,
    name: str,
    directory_fd: int,
    *,
    tombstone_fd: RetainedTombstoneStore,
) -> RetainedTombstoneInventory:
    metadata = os.fstat(directory_fd)
    identity = (metadata.st_dev, metadata.st_ino)
    if list_entries(directory_fd):
        raise ValueError(f"owned cleanup directory is not empty: {name}")
    source_parent = os.fstat(parent_fd)
    tombstone_parent = os.fstat(tombstone_fd.directory_fd)
    if (source_parent.st_dev, source_parent.st_ino) == (
        tombstone_parent.st_dev,
        tombstone_parent.st_ino,
    ):
        raise ValueError("retained tombstones require a distinct directory")
    fcntl.flock(tombstone_fd.lock_fd, fcntl.LOCK_EX)
    quarantined_fd = -1
    tombstone_name = ""
    try:
        tombstone_fd.require_admission(
            additional_count=1,
            additional_bytes=0,
        )
        name_digest = sha256_hex(name.encode("utf-8"))[:16]
        for _attempt in range(16):
            candidate = (
                f"directory-{name_digest}-cleanup-{secrets.token_hex(16)}"
            )
            try:
                atomic_rename_noreplace(
                    parent_fd,
                    name,
                    tombstone_fd.directory_fd,
                    candidate,
                )
            except FileExistsError:
                continue
            tombstone_name = candidate
            break
        if not tombstone_name:
            raise FileExistsError(
                "could not allocate a unique retained directory tombstone"
            )
        fsync_directory(parent_fd)
        fsync_directory(tombstone_fd.directory_fd)
        quarantined_fd, _created = open_directory_at(
            tombstone_fd.directory_fd,
            tombstone_name,
        )
        quarantined = os.fstat(quarantined_fd)
        current = entry_lstat(tombstone_fd.directory_fd, tombstone_name)
        if (
            (quarantined.st_dev, quarantined.st_ino) != identity
            or (current.st_dev, current.st_ino) != identity
            or list_entries(quarantined_fd)
        ):
            os.close(quarantined_fd)
            quarantined_fd = -1
            restored = _restore_quarantined_directory(
                parent_fd,
                name,
                tombstone_fd.directory_fd,
                tombstone_name,
            )
            if not restored:
                tombstone_fd.raise_identity_error(name)
            raise ValueError(
                f"owned cleanup directory identity changed: {name}"
            )
    except BaseException:
        if quarantined_fd >= 0:
            os.close(quarantined_fd)
            quarantined_fd = -1
        if tombstone_name and entry_exists(
            tombstone_fd.directory_fd,
            tombstone_name,
        ):
            restored = _restore_quarantined_directory(
                parent_fd,
                name,
                tombstone_fd.directory_fd,
                tombstone_name,
            )
            if not restored:
                tombstone_fd.raise_identity_error(name)
        raise
    finally:
        if quarantined_fd >= 0:
            os.close(quarantined_fd)
        fcntl.flock(tombstone_fd.lock_fd, fcntl.LOCK_UN)
    return tombstone_fd.inventory()


def publish_task_result_via_local_cache(
    local_root: Path | str,
    shared_root: Path | str,
    result: TaskResult,
    *,
    scheduler_id: str,
    job_id: str,
    nonce: str,
) -> Path:
    """Validate a node-local task artifact before shared no-replace install."""

    if not isinstance(result, TaskResult):
        raise TypeError("result must be a TaskResult")
    (
        workspace_fd,
        build_fd,
        tombstone_fd,
        workspace,
        _owner_payload,
        owner_token,
    ) = _open_local_task_workspace(
        local_root,
        result,
        scheduler_id=scheduler_id,
        job_id=job_id,
        nonce=nonce,
    )
    result_name = task_result_filename(result)
    workspace_open = True
    try:
        _validate_local_task_workspace(
            workspace_fd,
            result_name=result_name,
            owner_token=owner_token,
        )
        clean_owned_temporaries(
            workspace_fd,
            final_names={result_name},
            owner=owner_token,
            tombstone_fd=tombstone_fd,
        )
        payload = task_result_to_bytes(result)
        atomic_write_or_match(
            workspace_fd,
            result_name,
            payload,
            owner=owner_token,
            tombstone_fd=tombstone_fd,
        )
        local_payload = read_regular_file(workspace_fd, result_name)
        local_result = task_result_from_bytes(local_payload)
        if local_result != result or task_result_filename(local_result) != result_name:
            raise ValueError("node-local task result binding mismatch")
        published = publish_task_result(
            shared_root,
            local_result,
            scheduler_id=scheduler_id,
            nonce=nonce,
        )
        return published
    finally:
        try:
            _cleanup_local_task_workspace(
                workspace_fd,
                result_name=result_name,
                owner_token=owner_token,
                tombstone_fd=tombstone_fd,
            )
            _remove_owned_empty_directory(
                build_fd,
                workspace.name,
                workspace_fd,
                tombstone_fd=tombstone_fd,
            )
            os.close(workspace_fd)
            workspace_open = False
        finally:
            if workspace_open:
                os.close(workspace_fd)
            os.close(build_fd)
            tombstone_fd.close()


def load_task_results(
    shared_root: Path | str,
    build_id: str,
    *,
    scheduler_id: str,
    nonce: str,
    expected_task_count: int,
) -> tuple[TaskResult, ...]:
    """Load exactly one complete, foreign-free result set."""

    (
        workspace_fd,
        tombstone_fd,
        _workspace,
        _owner_payload,
        owner_token,
    ) = _open_workspace(
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
        if tombstone_fd is not None:
            tombstone_fd.close()


def cleanup_task_workspace(
    workspace: Path | str,
    *,
    build_id: str,
    scheduler_id: str,
    nonce: str,
) -> RetainedTombstoneInventory:
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
    workspace_name = workspace_path.name
    shared_root_fd = open_directory_path(workspace_path.parents[2])
    namespace_fd = -1
    parent_fd = -1
    workspace_fd = -1
    tombstone_fd = None
    inventory = None
    try:
        namespace_fd, _created = open_directory_at(
            shared_root_fd,
            _WORKSPACE_ROOT,
        )
        parent_fd, _created = open_directory_at(namespace_fd, build_id)
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
        tombstone_fd = open_tombstone_directory(shared_root_fd)
        for name in sorted(names, key=lambda name: name == _OWNER_NAME):
            payload = read_regular_file(workspace_fd, name)
            unlink_regular_if_matches(
                workspace_fd,
                name,
                payload,
                tombstone_fd=tombstone_fd,
            )
        inventory = _remove_owned_empty_directory(
            parent_fd,
            workspace_name,
            workspace_fd,
            tombstone_fd=tombstone_fd,
        )
        os.close(workspace_fd)
        workspace_fd = -1
    finally:
        if workspace_fd >= 0:
            os.close(workspace_fd)
        if tombstone_fd is not None:
            tombstone_fd.close()
        if parent_fd >= 0:
            os.close(parent_fd)
        if namespace_fd >= 0:
            os.close(namespace_fd)
        os.close(shared_root_fd)
    if inventory is None:
        raise RuntimeError("task workspace cleanup did not produce an inventory")
    return inventory
