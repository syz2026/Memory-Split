"""Descriptor-pinned filesystem primitives with no symlink traversal."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path, PurePosixPath

from .errors import MsctlError
from .jsonutil import canonical_json, portable_relative


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _unsafe(label: str, path: str, error: OSError | None = None) -> MsctlError:
    return MsctlError(
        "UNSAFE_PATH",
        f"{label} must not traverse symlinks or non-directories",
        details={"path": path},
    )


def _directory_parts(path: Path | str, *, label: str) -> tuple[int, list[str]]:
    raw = os.fspath(path)
    candidate = Path(raw)
    parts = list(candidate.parts)
    if candidate.is_absolute():
        descriptor = os.open(os.path.sep, _DIRECTORY_FLAGS)
        parts = parts[1:]
    else:
        descriptor = os.open(".", _DIRECTORY_FLAGS)
    if any(part in {"", ".", "..", os.path.sep} for part in parts):
        os.close(descriptor)
        raise _unsafe(label, raw)
    return descriptor, parts


def open_directory(
    path: Path | str,
    *,
    label: str,
    create: bool = False,
    mode: int = 0o700,
) -> int:
    """Open and pin a directory, rejecting symlinks at every path component."""

    descriptor, parts = _directory_parts(path, label=label)
    raw = os.fspath(path)
    try:
        for part in parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        if error.errno == errno.ENOENT:
            raise MsctlError(
                "FILE_NOT_FOUND",
                f"{label} directory does not exist",
                details={"path": raw},
            ) from error
        raise _unsafe(label, raw, error) from error


def open_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
    create: bool = False,
    mode: int = 0o700,
) -> int:
    if not name or "/" in name or name in {".", ".."}:
        raise _unsafe(label, name)
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise MsctlError(
                "FILE_NOT_FOUND",
                f"{label} directory does not exist",
                details={"path": name},
            )
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
        except FileExistsError:
            pass
        try:
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError as error:
            raise _unsafe(label, name, error) from error
    except OSError as error:
        raise _unsafe(label, name, error) from error


def open_parent_at(
    root_fd: int,
    relative: str,
    *,
    label: str,
    create: bool = False,
    mode: int = 0o700,
) -> tuple[int, str]:
    portable = portable_relative(relative, label=label)
    parts = PurePosixPath(portable).parts
    descriptor = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = open_directory_at(
                descriptor,
                part,
                label=label,
                create=create,
                mode=mode,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except Exception:
        os.close(descriptor)
        raise


def open_regular_at(
    root_fd: int,
    relative: str,
    *,
    label: str,
) -> tuple[int, int, str]:
    """Return an opened regular file and its pinned parent directory."""

    parent_fd, name = open_parent_at(root_fd, relative, label=label)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise _unsafe(label, relative)
        return descriptor, parent_fd, name
    except FileNotFoundError as error:
        os.close(parent_fd)
        raise MsctlError(
            "FILE_NOT_FOUND",
            f"{label} is not a regular file",
            details={"path": relative},
        ) from error
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        raise _unsafe(label, relative, error) from error
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(parent_fd)
        raise


def read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def hash_fd(descriptor: int) -> tuple[int, str]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
    after = os.fstat(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or total != after.st_size:
        raise MsctlError(
            "FILE_CHANGED",
            "file changed while its descriptor was being hashed",
        )
    return total, digest.hexdigest()


def load_json_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> object:
    descriptor, parent_fd, basename = open_regular_at(
        directory_fd,
        name,
        label=label,
    )
    try:
        before = os.fstat(descriptor)
        data = read_fd(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(data) != after.st_size:
        raise MsctlError(
            "FILE_CHANGED",
            f"{label} changed while being read",
            details={"path": basename},
        )
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "INVALID_JSON",
            f"{label} must contain one valid UTF-8 JSON value",
            details={"path": basename},
        ) from error


def atomic_write_at(
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    mode: int = 0o600,
    label: str = "file",
) -> None:
    """Atomically replace one basename relative to a pinned directory."""

    if not name or "/" in name or name in {".", ".."}:
        raise _unsafe(label, name)
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(temporary, flags, mode, dir_fd=directory_fd)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        try:
            existing = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(existing.st_mode):
                raise _unsafe(label, name)
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def atomic_write_json_at(
    directory_fd: int,
    name: str,
    value: object,
    *,
    label: str = "JSON file",
) -> None:
    atomic_write_at(
        directory_fd,
        name,
        canonical_json(value) + b"\n",
        label=label,
    )


def rename_noreplace_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one entry without replacing an existing entry."""

    for name in (source_name, destination_name):
        if not name or "/" in name or name in {".", ".."}:
            raise _unsafe("rename entry", name)
    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_directory_fd,
            encoded_source,
            destination_directory_fd,
            encoded_destination,
            0x00000004,
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_directory_fd,
            encoded_source,
            destination_directory_fd,
            encoded_destination,
            0x00000001,
        )
    else:
        raise MsctlError(
            "ATOMIC_RENAME_UNAVAILABLE",
            "platform lacks atomic no-replace directory publication",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


def remove_tree_at(parent_fd: int, name: str, *, label: str) -> None:
    """Remove a private directory tree relative to a pinned parent."""

    child_fd = open_directory_at(parent_fd, name, label=label)
    try:
        for entry in os.listdir(child_fd):
            metadata = os.stat(
                entry,
                dir_fd=child_fd,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(metadata.st_mode):
                remove_tree_at(child_fd, entry, label=label)
            else:
                os.unlink(entry, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)
