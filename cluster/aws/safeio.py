"""Linux no-replace rename primitive used by AWS corpus publication."""

from __future__ import annotations

import ctypes
import errno
import os
import sys

_RENAME_NOREPLACE = 1


def atomic_rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("AWS corpus publication requires Linux renameat2")
    for name in (source_name, destination_name):
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\x00" in name
        ):
            raise ValueError("rename entry names must be safe path components")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        primitive = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("AWS corpus publication requires Linux renameat2") from error
    primitive.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    primitive.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = primitive(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
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
