"""Pinned, no-follow filesystem primitives for corpus publication."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import secrets
import stat
import sys
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

if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
    raise RuntimeError("secure corpus publication requires O_DIRECTORY and O_NOFOLLOW")


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


def _unlink_same_regular(
    directory_fd: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    try:
        current = entry_lstat(directory_fd, name)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise ValueError(f"temporary file identity changed: {name}")
    os.unlink(name, dir_fd=directory_fd)
    fsync_directory(directory_fd)


def clean_owned_temporaries(
    directory_fd: int,
    *,
    final_names: set[str],
    owner: str,
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
        _unlink_same_regular(directory_fd, name, identity)
        removed.append(name)
    return tuple(removed)


def unlink_regular_if_matches(
    directory_fd: int,
    name: str,
    expected_payload: bytes,
) -> None:
    """Unlink a pinned regular file only when its bytes and identity match."""

    actual = read_regular_file(directory_fd, name)
    if actual != expected_payload:
        raise ValueError(f"owned file content drift: {name}")
    metadata = entry_lstat(directory_fd, name)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"owned file is unsafe: {name}")
    _unlink_same_regular(
        directory_fd,
        name,
        (metadata.st_dev, metadata.st_ino),
    )


class AtomicFileWriter:
    """Stream one file and install it no-replace under a pinned directory fd."""

    def __init__(
        self,
        directory_fd: int,
        final_name: str,
        *,
        owner: str,
        mode: int = 0o600,
    ) -> None:
        self.directory_fd = os.dup(directory_fd)
        self.final_name = _entry_name(final_name)
        prefix = _temporary_prefix(self.final_name, owner)
        self.temporary_name = ""
        self.descriptor = -1
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
            os.close(self.directory_fd)
            raise FileExistsError("could not allocate a unique owned temporary file")
        metadata = os.fstat(self.descriptor)
        self._identity = (metadata.st_dev, metadata.st_ino)
        self._closed = False

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
            _unlink_same_regular(
                self.directory_fd,
                self.temporary_name,
                self._identity,
            )
        else:
            fsync_directory(self.directory_fd)
        self._closed = True
        os.close(self.directory_fd)

    def abort(self) -> None:
        if self._closed:
            return
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        try:
            _unlink_same_regular(
                self.directory_fd,
                self.temporary_name,
                self._identity,
            )
        finally:
            self._closed = True
            os.close(self.directory_fd)


def atomic_write_or_match(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    owner: str,
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
    writer = AtomicFileWriter(directory_fd, name, owner=owner)
    try:
        writer.write(payload)
        writer.finish(
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
    except BaseException:
        writer.abort()
        raise
