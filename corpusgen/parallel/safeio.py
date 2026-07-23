"""Pinned, no-follow filesystem primitives for corpus publication."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._-]+\Z")
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_TOMBSTONE_DIRECTORY_NAME = ".memorysplit-v2-retained-tombstones"
_TOMBSTONE_DIRECTORY_PREFIX = ".memorysplit-v2-retained-tombstones-"
_TOMBSTONE_OWNER_NAME = ".owner.json"
_TOMBSTONE_LOCK_NAME = ".lock"
_TOMBSTONE_OWNER_BYTES = (
    b'{"format":"memorysplit-v2-retained-tombstones-v1","owner":"memorysplit"}\n'
)
_TOMBSTONE_LOCK_FLAGS = (
    os.O_RDWR
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
RETAINED_TOMBSTONE_ADMISSION_COUNT_THRESHOLD = 4096
RETAINED_TOMBSTONE_ADMISSION_BYTE_THRESHOLD = 64 * 1024**3

if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
    raise RuntimeError("secure corpus publication requires O_DIRECTORY and O_NOFOLLOW")


def retained_tombstone_maintenance_contract() -> dict[str, object]:
    return {
        "coordination_scope": "conforming-builders",
        "hostile_same_uid_mutation_in_scope": False,
        "hard_storage_boundary": "filesystem-or-project-quota",
        "reclamation": "quiescent-operator-maintenance",
        "requires_no_active_build_or_finalizer": True,
        "library_path_deletion": False,
        "procedure": (
            "Stop submissions and confirm no corpus build, task publisher, "
            "or finalizer is active; snapshot the reported inventory; then "
            "remove only the reported entries and their containing tombstone "
            "roots in an operator-controlled maintenance window."
        ),
        "trust_boundary": (
            "Advisory locks and admission thresholds coordinate conforming "
            "builders only. Hostile same-UID mutation is out of scope; "
            "filesystem or project quotas are the hard storage boundary."
        ),
    }


@dataclass(frozen=True)
class RetainedTombstoneInventory:
    count: int
    byte_count: int
    paths: tuple[str, ...]
    roots: tuple[str, ...]

    def summary_dict(self) -> dict[str, object]:
        return {
            "bytes": self.byte_count,
            "count": self.count,
            "paths": list(self.paths),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.summary_dict(),
            "roots": list(self.roots),
            "maintenance": retained_tombstone_maintenance_contract(),
        }


class RetainedTombstoneAdmissionError(ValueError):
    def __init__(
        self,
        inventory: RetainedTombstoneInventory,
        *,
        count_threshold: int,
        byte_threshold: int,
        additional_count: int,
        additional_bytes: int,
    ) -> None:
        self.inventory = inventory
        self.report = {
            "inventory": inventory.summary_dict(),
            "cooperative_thresholds": {
                "bytes": byte_threshold,
                "count": count_threshold,
            },
            "requested_admission": {
                "bytes": additional_bytes,
                "count": additional_count,
            },
        }
        super().__init__(
            "retained tombstone cooperative admission threshold exceeded: "
            + json.dumps(self.report, sort_keys=True, separators=(",", ":"))
        )


class RetainedTombstoneIdentityError(ValueError):
    def __init__(
        self,
        name: str,
        inventory: RetainedTombstoneInventory,
    ) -> None:
        self.inventory = inventory
        self.report = {
            "inventory": inventory.summary_dict(),
            "reason": "uncertain-identity-retained",
        }
        super().__init__(
            f"retained object identity is uncertain after quarantine: {name}; "
            + json.dumps(self.report, sort_keys=True, separators=(",", ":"))
        )


@dataclass
class RetainedTombstoneStore:
    parent_fd: int
    directory_fd: int
    lock_fd: int
    admission_count_threshold: int
    admission_byte_threshold: int

    def duplicate(self) -> "RetainedTombstoneStore":
        self.require_admission(additional_count=0, additional_bytes=0)
        parent_fd = os.dup(self.parent_fd)
        directory_fd = -1
        lock_fd = -1
        try:
            directory_fd = os.dup(self.directory_fd)
            lock_fd = _open_tombstone_lock(directory_fd)
            return RetainedTombstoneStore(
                parent_fd=parent_fd,
                directory_fd=directory_fd,
                lock_fd=lock_fd,
                admission_count_threshold=self.admission_count_threshold,
                admission_byte_threshold=self.admission_byte_threshold,
            )
        except BaseException:
            if lock_fd >= 0:
                os.close(lock_fd)
            if directory_fd >= 0:
                os.close(directory_fd)
            os.close(parent_fd)
            raise

    def close(self) -> None:
        for field_name in ("lock_fd", "directory_fd", "parent_fd"):
            descriptor = getattr(self, field_name)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, field_name, -1)

    def inventory(self) -> RetainedTombstoneInventory:
        return retained_tombstone_inventory_fd(self.parent_fd)

    def require_admission(
        self,
        *,
        additional_count: int,
        additional_bytes: int,
    ) -> RetainedTombstoneInventory:
        """Coordinate cleanup admission among conforming builders.

        This is not a filesystem quota. Same-UID processes that ignore the
        lock can change retained storage; deployments needing a hard storage
        bound must enforce a filesystem or project quota.
        """

        inventory = self.inventory()
        if (
            inventory.count + additional_count
            > self.admission_count_threshold
            or inventory.byte_count + additional_bytes
            > self.admission_byte_threshold
        ):
            raise RetainedTombstoneAdmissionError(
                inventory,
                count_threshold=self.admission_count_threshold,
                byte_threshold=self.admission_byte_threshold,
                additional_count=additional_count,
                additional_bytes=additional_bytes,
            )
        return inventory

    def raise_identity_error(self, name: str) -> None:
        inventory = self.require_admission(
            additional_count=0,
            additional_bytes=0,
        )
        raise RetainedTombstoneIdentityError(name, inventory)


def _entry_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("filesystem entry name must be one safe path component")
    return name


def _owner_token(token: str) -> str:
    if not isinstance(token, str) or not token or _SAFE_TOKEN.fullmatch(token) is None:
        raise ValueError("temporary owner token is unsafe")
    return token


def _raise_rename_error(result: int, source_name: str, destination_name: str) -> None:
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source_name} -> {destination_name}",
    )


def atomic_rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one pinned entry without replacing its destination."""

    source_name = _entry_name(source_name)
    destination_name = _entry_name(destination_name)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    if sys.platform.startswith("linux"):
        try:
            primitive = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace publication requires Linux renameat2"
            ) from error
        primitive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            source_directory_fd,
            source_bytes,
            destination_directory_fd,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        try:
            primitive = libc.renameatx_np
        except AttributeError:
            try:
                primitive = libc.renamex_np
            except AttributeError as error:
                raise RuntimeError(
                    "atomic no-replace publication requires macOS renamex_np"
                ) from error
            primitive.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            primitive.restype = ctypes.c_int
            source_path = os.fsencode(
                f"/dev/fd/{source_directory_fd}/{source_name}"
            )
            destination_path = os.fsencode(
                f"/dev/fd/{destination_directory_fd}/{destination_name}"
            )
            result = primitive(source_path, destination_path, _RENAME_EXCL)
        else:
            primitive.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            primitive.restype = ctypes.c_int
            result = primitive(
                source_directory_fd,
                source_bytes,
                destination_directory_fd,
                destination_bytes,
                _RENAME_EXCL,
            )
    else:
        raise RuntimeError(
            f"no atomic no-replace rename primitive for platform {sys.platform!r}"
        )
    _raise_rename_error(result, source_name, destination_name)


def fsync_directory(directory_fd: int) -> None:
    os.fsync(directory_fd)


def open_directory_at(
    parent_fd: int,
    name: str,
    *,
    create: bool = False,
    mode: int = 0o700,
) -> tuple[int, bool]:
    """Open one no-follow directory component, optionally creating it."""

    name = _entry_name(name)
    created = False
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
            created = True
            fsync_directory(parent_fd)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError as error:
            raise ValueError(f"directory entry is unsafe: {name}") from error
    except OSError as error:
        raise ValueError(f"directory entry is unsafe: {name}") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"directory entry is unsafe: {name}")
    return descriptor, created


def open_directory_path(
    path: Path | str,
    *,
    create: bool = False,
    mode: int = 0o700,
) -> int:
    """Walk and pin a directory path without following any symlink component."""

    target = Path(path)
    raw_parts = target.parts
    if target.is_absolute():
        descriptor = os.open(os.path.sep, _DIRECTORY_FLAGS)
        parts = raw_parts[1:]
    else:
        descriptor = os.open(".", _DIRECTORY_FLAGS)
        parts = raw_parts
    try:
        for part in parts:
            if part == ".":
                continue
            if part == "..":
                raise ValueError("directory traversal is not allowed")
            child, _created = open_directory_at(
                descriptor,
                part,
                create=create,
                mode=mode,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_control_file(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        _WRITE_FLAGS,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_tombstone_lock(
    directory_fd: int,
    *,
    create: bool = False,
    writable: bool = True,
) -> int:
    flags = _TOMBSTONE_LOCK_FLAGS if writable else _READ_FLAGS
    if create:
        if not writable:
            raise ValueError("creating a tombstone lock requires write access")
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(
        _TOMBSTONE_LOCK_NAME,
        flags,
        0o600,
        dir_fd=directory_fd,
    )
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_size != 0
    ):
        os.close(descriptor)
        raise ValueError("retained tombstone lock is unsafe")
    return descriptor


def _validate_tombstone_root(directory_fd: int) -> None:
    owner_metadata = entry_lstat(directory_fd, _TOMBSTONE_OWNER_NAME)
    if (
        not stat.S_ISREG(owner_metadata.st_mode)
        or owner_metadata.st_uid != os.geteuid()
        or owner_metadata.st_nlink != 1
        or read_regular_file(directory_fd, _TOMBSTONE_OWNER_NAME)
        != _TOMBSTONE_OWNER_BYTES
    ):
        raise ValueError("retained tombstone ownership marker is unsafe")


def _is_tombstone_root_name(name: str) -> bool:
    return name == _TOMBSTONE_DIRECTORY_NAME or re.fullmatch(
        re.escape(_TOMBSTONE_DIRECTORY_PREFIX) + r"[0-9a-f]{32}",
        name,
    ) is not None


def retained_tombstone_inventory_fd(
    parent_fd: int,
) -> RetainedTombstoneInventory:
    paths: list[str] = []
    byte_count = 0
    root_names = sorted(
        name for name in list_entries(parent_fd) if _is_tombstone_root_name(name)
    )
    for root_name in root_names:
        root_metadata = entry_lstat(parent_fd, root_name)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError(f"retained tombstone root is unsafe: {root_name}")
        root_fd, _created = open_directory_at(parent_fd, root_name)
        try:
            pinned_root = os.fstat(root_fd)
            if (pinned_root.st_dev, pinned_root.st_ino) != (
                root_metadata.st_dev,
                root_metadata.st_ino,
            ):
                raise ValueError(
                    f"retained tombstone root identity changed: {root_name}"
                )
            if root_name == _TOMBSTONE_DIRECTORY_NAME:
                _validate_tombstone_root(root_fd)
                lock_fd = _open_tombstone_lock(root_fd, writable=False)
                os.close(lock_fd)
                control_names = {
                    _TOMBSTONE_LOCK_NAME,
                    _TOMBSTONE_OWNER_NAME,
                }
            else:
                control_names = set()
            for name in sorted(set(list_entries(root_fd)) - control_names):
                relative = f"{root_name}/{name}"
                metadata = entry_lstat(root_fd, name)
                if stat.S_ISREG(metadata.st_mode):
                    descriptor, pinned_metadata = open_regular_file_at(
                        root_fd,
                        name,
                    )
                    os.close(descriptor)
                    current = entry_lstat(root_fd, name)
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or (
                            current.st_dev,
                            current.st_ino,
                            current.st_size,
                        )
                        != (
                            pinned_metadata.st_dev,
                            pinned_metadata.st_ino,
                            pinned_metadata.st_size,
                        )
                    ):
                        raise ValueError(
                            f"retained tombstone identity changed: {relative}"
                        )
                    byte_count += pinned_metadata.st_size
                elif stat.S_ISDIR(metadata.st_mode):
                    directory_fd, _created = open_directory_at(root_fd, name)
                    try:
                        pinned_directory = os.fstat(directory_fd)
                        if (
                            pinned_directory.st_dev,
                            pinned_directory.st_ino,
                        ) != (metadata.st_dev, metadata.st_ino):
                            raise ValueError(
                                "retained directory tombstone identity changed: "
                                + relative
                            )
                        if list_entries(directory_fd):
                            raise ValueError(
                                "retained directory tombstone is not empty: "
                                + relative
                            )
                        current = entry_lstat(root_fd, name)
                        if (
                            not stat.S_ISDIR(current.st_mode)
                            or (current.st_dev, current.st_ino)
                            != (
                                pinned_directory.st_dev,
                                pinned_directory.st_ino,
                            )
                        ):
                            raise ValueError(
                                "retained directory tombstone identity changed: "
                                + relative
                            )
                    finally:
                        os.close(directory_fd)
                else:
                    raise ValueError(
                        f"retained tombstone entry is unsafe: {relative}"
                    )
                paths.append(relative)
            current_root = entry_lstat(parent_fd, root_name)
            if (
                not stat.S_ISDIR(current_root.st_mode)
                or (current_root.st_dev, current_root.st_ino)
                != (pinned_root.st_dev, pinned_root.st_ino)
            ):
                raise ValueError(
                    f"retained tombstone root identity changed: {root_name}"
                )
        finally:
            os.close(root_fd)
    return RetainedTombstoneInventory(
        count=len(paths),
        byte_count=byte_count,
        paths=tuple(paths),
        roots=tuple(root_names),
    )


def retained_tombstone_inventory(
    parent: Path | str,
) -> RetainedTombstoneInventory:
    parent_fd = open_directory_path(parent)
    try:
        return retained_tombstone_inventory_fd(parent_fd)
    finally:
        os.close(parent_fd)


def open_tombstone_directory(
    parent_fd: int,
    *,
    admission_count_threshold: int | None = None,
    admission_byte_threshold: int | None = None,
) -> RetainedTombstoneStore:
    """Open the pinned retained-object store used instead of path deletion."""

    resolved_count = (
        RETAINED_TOMBSTONE_ADMISSION_COUNT_THRESHOLD
        if admission_count_threshold is None
        else admission_count_threshold
    )
    resolved_bytes = (
        RETAINED_TOMBSTONE_ADMISSION_BYTE_THRESHOLD
        if admission_byte_threshold is None
        else admission_byte_threshold
    )
    if (
        isinstance(resolved_count, bool)
        or not isinstance(resolved_count, int)
        or resolved_count < 0
        or isinstance(resolved_bytes, bool)
        or not isinstance(resolved_bytes, int)
        or resolved_bytes < 0
    ):
        raise ValueError(
            "retained tombstone admission thresholds must be non-negative integers"
        )
    created = False
    try:
        os.mkdir(_TOMBSTONE_DIRECTORY_NAME, mode=0o700, dir_fd=parent_fd)
        created = True
        fsync_directory(parent_fd)
    except FileExistsError:
        pass
    directory_fd, _created = open_directory_at(
        parent_fd,
        _TOMBSTONE_DIRECTORY_NAME,
    )
    lock_fd = -1
    retained_parent_fd = -1
    try:
        directory_metadata = os.fstat(directory_fd)
        if directory_metadata.st_uid != os.geteuid():
            raise ValueError("retained tombstone directory owner is unsafe")
        if created:
            _write_control_file(
                directory_fd,
                _TOMBSTONE_OWNER_NAME,
                _TOMBSTONE_OWNER_BYTES,
            )
            lock_fd = _open_tombstone_lock(directory_fd, create=True)
            fsync_directory(directory_fd)
        else:
            _validate_tombstone_root(directory_fd)
            lock_fd = _open_tombstone_lock(directory_fd)
        retained_parent_fd = os.dup(parent_fd)
        store = RetainedTombstoneStore(
            parent_fd=retained_parent_fd,
            directory_fd=directory_fd,
            lock_fd=lock_fd,
            admission_count_threshold=resolved_count,
            admission_byte_threshold=resolved_bytes,
        )
        fcntl.flock(store.lock_fd, fcntl.LOCK_EX)
        try:
            store.require_admission(
                additional_count=0,
                additional_bytes=0,
            )
        finally:
            fcntl.flock(store.lock_fd, fcntl.LOCK_UN)
        return store
    except BaseException:
        if retained_parent_fd >= 0:
            os.close(retained_parent_fd)
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(directory_fd)
        raise


def open_parent_directory(
    path: Path | str,
    *,
    create: bool = False,
    mode: int = 0o700,
) -> tuple[int, str]:
    target = Path(path)
    if not target.name or target.name in {".", ".."}:
        raise ValueError("path must name a filesystem entry")
    return (
        open_directory_path(target.parent, create=create, mode=mode),
        _entry_name(target.name),
    )


def list_entries(directory_fd: int) -> tuple[str, ...]:
    return tuple(sorted(os.listdir(directory_fd)))


def entry_lstat(directory_fd: int, name: str) -> os.stat_result:
    return os.stat(
        _entry_name(name),
        dir_fd=directory_fd,
        follow_symlinks=False,
    )


def entry_exists(directory_fd: int, name: str) -> bool:
    try:
        entry_lstat(directory_fd, name)
    except FileNotFoundError:
        return False
    return True


def read_regular_file(directory_fd: int, name: str) -> bytes:
    name = _entry_name(name)
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"file entry is unsafe: {name}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"file entry is unsafe: {name}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def open_regular_file_at(
    directory_fd: int,
    name: str,
) -> tuple[int, os.stat_result]:
    """Open one regular file without following links or blocking on specials."""

    name = _entry_name(name)
    try:
        descriptor = os.open(
            name,
            _READ_FLAGS | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"file entry is unsafe: {name}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"file entry is special or unsafe: {name}")
    return descriptor, metadata


def read_file_descriptor(descriptor: int) -> bytes:
    """Read an already pinned descriptor from its current offset to EOF."""

    chunks = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def regular_file_digest(directory_fd: int, name: str) -> tuple[int, str]:
    name = _entry_name(name)
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"file entry is unsafe: {name}") from error
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"file entry is unsafe: {name}")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return metadata.st_size, digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(descriptor)


def _temporary_prefix(final_name: str, owner: str) -> str:
    return f".{_entry_name(final_name)}.tmp-{_owner_token(owner)}-"


def is_owned_temporary(name: str, final_names: set[str], owner: str) -> bool:
    return any(
        name.startswith(_temporary_prefix(final_name, owner))
        for final_name in final_names
    )


def _restore_quarantined_entry(
    source_directory_fd: int,
    original_name: str,
    tombstone_fd: int,
    tombstone_name: str,
) -> bool:
    try:
        atomic_rename_noreplace(
            tombstone_fd,
            tombstone_name,
            source_directory_fd,
            original_name,
        )
    except FileExistsError:
        fsync_directory(source_directory_fd)
        fsync_directory(tombstone_fd)
        return False
    fsync_directory(source_directory_fd)
    fsync_directory(tombstone_fd)
    return True


def _retain_regular_tombstone(
    directory_fd: int,
    name: str,
    *,
    tombstone_fd: RetainedTombstoneStore,
    expected_identity: tuple[int, int] | None = None,
    expected_payload: bytes | None = None,
    missing_ok: bool,
) -> RetainedTombstoneInventory:
    name = _entry_name(name)
    source_directory = os.fstat(directory_fd)
    tombstone_directory = os.fstat(tombstone_fd.directory_fd)
    if (source_directory.st_dev, source_directory.st_ino) == (
        tombstone_directory.st_dev,
        tombstone_directory.st_ino,
    ):
        raise ValueError("retained tombstones require a distinct directory")
    try:
        source_fd, source_metadata = open_regular_file_at(directory_fd, name)
    except FileNotFoundError:
        if missing_ok:
            return tombstone_fd.inventory()
        raise
    locked = False
    try:
        source_identity = (source_metadata.st_dev, source_metadata.st_ino)
        if expected_identity is not None and source_identity != expected_identity:
            raise ValueError(f"owned file identity changed: {name}")
        if (
            expected_payload is not None
            and read_file_descriptor(source_fd) != expected_payload
        ):
            raise ValueError(f"owned file content drift: {name}")

        fcntl.flock(tombstone_fd.lock_fd, fcntl.LOCK_EX)
        locked = True
        tombstone_fd.require_admission(
            additional_count=1,
            additional_bytes=source_metadata.st_size,
        )
        tombstone_name = ""
        name_digest = hashlib.sha256(os.fsencode(name)).hexdigest()[:16]
        for _attempt in range(16):
            candidate = f"file-{name_digest}-{secrets.token_hex(16)}"
            try:
                atomic_rename_noreplace(
                    directory_fd,
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
                "could not allocate a unique retained tombstone"
            )
        fsync_directory(directory_fd)
        fsync_directory(tombstone_fd.directory_fd)

        try:
            quarantined_fd, quarantined_metadata = open_regular_file_at(
                tombstone_fd.directory_fd,
                tombstone_name,
            )
            try:
                quarantined_payload = (
                    read_file_descriptor(quarantined_fd)
                    if expected_payload is not None
                    else None
                )
            finally:
                os.close(quarantined_fd)
        except BaseException:
            restored = _restore_quarantined_entry(
                directory_fd,
                name,
                tombstone_fd.directory_fd,
                tombstone_name,
            )
            if not restored:
                tombstone_fd.raise_identity_error(name)
            raise

        quarantined_identity = (
            quarantined_metadata.st_dev,
            quarantined_metadata.st_ino,
        )
        if (
            quarantined_identity != source_identity
            or quarantined_metadata.st_size != source_metadata.st_size
            or (
                expected_payload is not None
                and quarantined_payload != expected_payload
            )
        ):
            restored = _restore_quarantined_entry(
                directory_fd,
                name,
                tombstone_fd.directory_fd,
                tombstone_name,
            )
            if not restored:
                tombstone_fd.raise_identity_error(name)
            raise ValueError(
                f"owned file identity changed during quarantine: {name}"
            )
        current = entry_lstat(tombstone_fd.directory_fd, tombstone_name)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != source_identity
            or current.st_size != source_metadata.st_size
        ):
            restored = _restore_quarantined_entry(
                directory_fd,
                name,
                tombstone_fd.directory_fd,
                tombstone_name,
            )
            if not restored:
                tombstone_fd.raise_identity_error(name)
            raise ValueError(
                f"owned file identity changed during quarantine: {name}"
            )
        return tombstone_fd.inventory()
    finally:
        if locked:
            fcntl.flock(tombstone_fd.lock_fd, fcntl.LOCK_UN)
        os.close(source_fd)


def _retain_same_regular_tombstone(
    directory_fd: int,
    name: str,
    identity: tuple[int, int],
    *,
    tombstone_fd: RetainedTombstoneStore,
) -> RetainedTombstoneInventory:
    return _retain_regular_tombstone(
        directory_fd,
        name,
        tombstone_fd=tombstone_fd,
        expected_identity=identity,
        missing_ok=True,
    )


def clean_owned_temporaries(
    directory_fd: int,
    *,
    final_names: set[str],
    owner: str,
    tombstone_fd: RetainedTombstoneStore,
) -> tuple[str, ...]:
    """Remove only regular stale files carrying the exact ownership token."""

    removed = []
    for name in list_entries(directory_fd):
        if not is_owned_temporary(name, final_names, owner):
            continue
        metadata = entry_lstat(directory_fd, name)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"owned temporary entry is unsafe: {name}")
        identity = (metadata.st_dev, metadata.st_ino)
        _retain_same_regular_tombstone(
            directory_fd,
            name,
            identity,
            tombstone_fd=tombstone_fd,
        )
        removed.append(name)
    return tuple(removed)


def unlink_regular_if_matches(
    directory_fd: int,
    name: str,
    expected_payload: bytes,
    *,
    tombstone_fd: RetainedTombstoneStore,
) -> RetainedTombstoneInventory:
    """Remove a verified name only by atomically retaining its opened inode."""

    return _retain_regular_tombstone(
        directory_fd,
        name,
        tombstone_fd=tombstone_fd,
        expected_payload=expected_payload,
        missing_ok=False,
    )


class AtomicFileWriter:
    """Stream one file and install it no-replace under a pinned directory fd."""

    def __init__(
        self,
        directory_fd: int,
        final_name: str,
        *,
        owner: str,
        tombstone_fd: RetainedTombstoneStore,
        mode: int = 0o600,
    ) -> None:
        self.directory_fd = os.dup(directory_fd)
        self.tombstone_store: RetainedTombstoneStore | None = None
        self.temporary_name = ""
        self.descriptor = -1
        self._closed = True
        try:
            self.tombstone_store = tombstone_fd.duplicate()
            self.final_name = _entry_name(final_name)
            prefix = _temporary_prefix(self.final_name, owner)
            for _attempt in range(16):
                candidate = f"{prefix}{secrets.token_hex(8)}"
                try:
                    descriptor = os.open(
                        candidate,
                        _WRITE_FLAGS,
                        mode,
                        dir_fd=self.directory_fd,
                    )
                except FileExistsError:
                    continue
                self.temporary_name = candidate
                self.descriptor = descriptor
                break
            if self.descriptor < 0:
                raise FileExistsError(
                    "could not allocate a unique owned temporary file"
                )
            metadata = os.fstat(self.descriptor)
            self._identity = (metadata.st_dev, metadata.st_ino)
            self._closed = False
        except BaseException as construction_error:
            identity = None
            admission_error = None
            if self.descriptor >= 0:
                try:
                    metadata = os.fstat(self.descriptor)
                    identity = (metadata.st_dev, metadata.st_ino)
                except OSError:
                    pass
                os.close(self.descriptor)
                self.descriptor = -1
            if self.temporary_name and identity is not None:
                try:
                    _retain_same_regular_tombstone(
                        self.directory_fd,
                        self.temporary_name,
                        identity,
                        tombstone_fd=self.tombstone_store,
                    )
                except RetainedTombstoneAdmissionError as error:
                    admission_error = error
                except (OSError, ValueError):
                    pass
            os.close(self.directory_fd)
            if self.tombstone_store is not None:
                self.tombstone_store.close()
            if admission_error is not None:
                raise admission_error from construction_error
            raise

    def write(self, payload: bytes) -> None:
        if self._closed:
            raise ValueError("atomic writer is closed")
        view = memoryview(payload)
        while view:
            written = os.write(self.descriptor, view)
            view = view[written:]

    def finish(self, *, expected_bytes: int, expected_sha256: str) -> None:
        if self._closed:
            raise ValueError("atomic writer is closed")
        os.fsync(self.descriptor)
        os.close(self.descriptor)
        self.descriptor = -1
        try:
            atomic_rename_noreplace(
                self.directory_fd,
                self.temporary_name,
                self.directory_fd,
                self.final_name,
            )
        except FileExistsError:
            actual_bytes, actual_sha256 = regular_file_digest(
                self.directory_fd,
                self.final_name,
            )
            if (
                actual_bytes != expected_bytes
                or actual_sha256 != expected_sha256
            ):
                self.abort()
                raise ValueError(f"existing artifact drift: {self.final_name}")
            _retain_same_regular_tombstone(
                self.directory_fd,
                self.temporary_name,
                self._identity,
                tombstone_fd=self.tombstone_store,
            )
        else:
            fsync_directory(self.directory_fd)
        self._closed = True
        os.close(self.directory_fd)
        self.tombstone_store.close()

    def abort(self) -> None:
        if self._closed:
            return
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        try:
            _retain_same_regular_tombstone(
                self.directory_fd,
                self.temporary_name,
                self._identity,
                tombstone_fd=self.tombstone_store,
            )
        finally:
            self._closed = True
            os.close(self.directory_fd)
            self.tombstone_store.close()


def atomic_write_or_match(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    owner: str,
    tombstone_fd: RetainedTombstoneStore,
) -> None:
    """Install bytes no-replace, accepting only an exact existing regular file."""

    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    try:
        existing = read_regular_file(directory_fd, name)
    except FileNotFoundError:
        pass
    else:
        if existing != payload:
            raise ValueError(f"existing artifact drift: {name}")
        return
    writer = AtomicFileWriter(
        directory_fd,
        name,
        owner=owner,
        tombstone_fd=tombstone_fd,
    )
    try:
        writer.write(payload)
        writer.finish(
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    except BaseException:
        writer.abort()
        raise
