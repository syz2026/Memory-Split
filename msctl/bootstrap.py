"""Fixed Slurm stdin bootstrap for authenticated node-local execution."""

from __future__ import annotations

import hashlib


BOOTSTRAP_PAYLOAD = b"""#!/usr/bin/python3
import hashlib
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import PurePosixPath


def abort(message):
    raise SystemExit("memorysplit bootstrap: " + message)


def required(name):
    value = os.environ.get(name)
    if not value or "\\x00" in value or "\\n" in value or "\\r" in value:
        abort("missing or invalid " + name)
    return value


def relative(value, label):
    if (
        not value
        or "\\\\" in value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        abort("unsafe " + label)
    return value


def open_directory_no_follow(path):
    if not os.path.isabs(path):
        abort("runtime directory is not absolute")
    descriptor = os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for part in PurePosixPath(path).parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def copy_authenticated_archive(source, destination, expected_size, expected_hash):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    source_fd = os.open(source, flags)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            abort("release archive is not regular")
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        abort("release archive copy failed")
                    view = view[written:]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
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
    if (
        identity_before != identity_after
        or copied != expected_size
        or digest.hexdigest() != expected_hash
    ):
        abort("release archive authentication failed")


def parse_sums(data, expected_names):
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        abort("member manifest is not ASCII")
    if not text.endswith("\\n"):
        abort("member manifest is not newline terminated")
    rows = {}
    ordered = []
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            abort("member manifest row is invalid")
        digest = line[:64]
        name = relative(line[66:], "member manifest path")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or name in rows
            or name == "SHA256SUMS"
        ):
            abort("member manifest row is invalid")
        rows[name] = digest
        ordered.append(name)
    if ordered != sorted(ordered) or set(rows) != expected_names:
        abort("member manifest is not complete and sorted")
    return rows


def validate_info(info):
    name = relative(info.filename.rstrip("/"), "archive member")
    mode = info.external_attr >> 16
    if info.is_dir():
        if not mode or not stat.S_ISDIR(mode):
            abort("archive directory metadata is invalid")
    elif not mode or not stat.S_ISREG(mode):
        abort("archive contains a non-regular member")
    return name


def extract_authenticated_archive(archive_path, destination, expected_sums_hash):
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile):
        abort("node-local release archive is invalid")
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            abort("release archive contains duplicate members")
        for info in infos:
            validate_info(info)
        file_infos = {info.filename: info for info in infos if not info.is_dir()}
        if "SHA256SUMS" not in file_infos:
            abort("release archive lacks its member manifest")
        sums = archive.read("SHA256SUMS")
        if hashlib.sha256(sums).hexdigest() != expected_sums_hash:
            abort("member manifest authentication failed")
        rows = parse_sums(sums, set(file_infos) - {"SHA256SUMS"})
        rows["SHA256SUMS"] = expected_sums_hash
        os.mkdir(destination, 0o700)
        for name in sorted(file_infos):
            info = file_infos[name]
            target = os.path.join(destination, *PurePosixPath(name).parts)
            parent = os.path.dirname(target)
            os.makedirs(parent, mode=0o700, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            digest = hashlib.sha256()
            size = 0
            try:
                with archive.open(info, "r") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(descriptor, view)
                            if written <= 0:
                                abort("release extraction failed")
                            view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if size != info.file_size or digest.hexdigest() != rows[name]:
                abort("extracted release member authentication failed")
    return rows


def canonical_dataset():
    prefix = os.path.abspath(required("MS_SHARED_ROOT_PREFIX"))
    shared = os.path.abspath(required("MS_SHARED_ROOT"))
    data_relative = relative(
        required("MS_DATA_RELATIVE_PATH"),
        "dataset relative path",
    )
    data = os.path.abspath(required("MS_DATA_ROOT"))
    try:
        if os.path.commonpath((prefix, shared)) != prefix:
            abort("shared root is outside its approved prefix")
    except ValueError:
        abort("shared root prefix is invalid")
    expected = os.path.abspath(os.path.join(shared, *data_relative.split("/")))
    if data != expected:
        abort("dataset root does not match its approved publication")
    for path in (prefix, shared, data):
        descriptor = open_directory_no_follow(path)
        os.close(descriptor)


def main():
    canonical_dataset()
    source_archive = os.path.abspath(required("MS_RELEASE_ARCHIVE"))
    expected_hash = required("MS_RELEASE_ARCHIVE_SHA256")
    expected_sums_hash = required("MS_RELEASE_MEMBERS_SHA256")
    try:
        expected_size = int(required("MS_RELEASE_ARCHIVE_BYTES"))
    except ValueError:
        abort("release archive size is invalid")
    if (
        expected_size <= 0
        or len(expected_hash) != 64
        or len(expected_sums_hash) != 64
    ):
        abort("release hash contract is invalid")
    local_root = os.path.abspath(required("SLURM_TMPDIR"))
    root_fd = open_directory_no_follow(local_root)
    os.close(root_fd)
    work = tempfile.mkdtemp(prefix="memorysplit-bootstrap.", dir=local_root)
    os.chmod(work, 0o700)
    local_archive = os.path.join(work, "release.zip")
    copy_authenticated_archive(
        source_archive,
        local_archive,
        expected_size,
        expected_hash,
    )
    content_parent = os.path.join(local_root, "memorysplit-release")
    os.makedirs(content_parent, mode=0o700, exist_ok=True)
    content_root = os.path.join(content_parent, expected_hash)
    if os.path.lexists(content_root):
        abort("content-addressed release root already exists")
    staging = os.path.join(work, "extract")
    extract_authenticated_archive(
        local_archive,
        staging,
        expected_sums_hash,
    )
    os.rename(staging, content_root)
    job_relative = relative(required("MS_JOB_SCRIPT_REL"), "job script")
    job_script = os.path.join(content_root, *job_relative.split("/"))
    if not os.path.isfile(job_script) or os.path.islink(job_script):
        abort("authenticated job script is unavailable")
    environment = dict(os.environ)
    environment["MS_RELEASE_ROOT"] = content_root
    bindings = (
        ("MS_DENSE_CONFIG_REL", "MS_DENSE_CONFIG"),
        ("MS_SPLIT_CONFIG_REL", "MS_SPLIT_CONFIG"),
        ("MS_TRAIN_ENTRYPOINT_REL", "MS_TRAIN_ENTRYPOINT"),
        ("MS_EVALUATOR_ENTRYPOINT_REL", "MS_EVALUATOR_ENTRYPOINT"),
    )
    for relative_name, absolute_name in bindings:
        value = environment.get(relative_name)
        if value is not None:
            value = relative(value, relative_name)
            environment[absolute_name] = os.path.join(
                content_root,
                *value.split("/"),
            )
    os.chmod(job_script, 0o500)
    os.chdir(content_root)
    os.execve(job_script, [job_script], environment)


if __name__ == "__main__":
    main()
"""

BOOTSTRAP_SHA256 = hashlib.sha256(BOOTSTRAP_PAYLOAD).hexdigest()
