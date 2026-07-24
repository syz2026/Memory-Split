#!/usr/bin/env python3
"""Download and verify one receipt-pinned AWS corpus without mutable S3 reads."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cluster.aws.corpus_builder.contracts import (  # noqa: E402
    CORPUS_BUCKET,
    CORPUS_KEY_PREFIX,
    PhaseReceipt,
    S3ObjectVersion,
    phase_receipt_from_bytes,
)
from cluster.aws.corpus_builder.s3 import (  # noqa: E402
    PublicationError,
    S3Client,
    _download_exact_object_at,
    _resolve_exact_object,
)
from corpusgen.parallel import verify_parallel_corpus  # noqa: E402


_PHASE_RECEIPT_NAME = "phase-receipt.json"


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


@dataclass(frozen=True)
class _DirectoryBinding:
    parent_fd: int
    name: str
    descriptor: int


@dataclass
class _PinnedCleanRoom:
    path: Path
    ancestor_bindings: tuple[_DirectoryBinding, ...]
    ancestor_descriptors: tuple[int, ...]
    parent_fd: int
    root_name: str
    root_fd: int
    control_name: str
    control_fd: int

    def close(self) -> None:
        os.close(self.control_fd)
        os.close(self.root_fd)
        for descriptor in reversed(self.ancestor_descriptors):
            os.close(descriptor)


@dataclass(frozen=True)
class _PinnedReceipt:
    descriptor: int
    initial: os.stat_result
    payload: bytes


def _assert_directory_binding(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    label: str,
    owner_only: bool = False,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise PublicationError(f"{label} authority binding cannot be inspected") from error
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or not _same_inode(opened, named)
    ):
        raise PublicationError(f"{label} authority binding changed")
    if owner_only and stat.S_IMODE(opened.st_mode) != 0o700:
        raise PublicationError(f"{label} must remain owner-only")


def _open_pinned_parent(
    destination: Path,
) -> tuple[Path, tuple[_DirectoryBinding, ...], tuple[int, ...]]:
    path = Path(os.path.abspath(os.fspath(destination)))
    if path.name in {"", ".", ".."}:
        raise PublicationError("clean-room destination must name a directory")
    descriptors: list[int] = []
    bindings: list[_DirectoryBinding] = []
    try:
        current = os.open(os.path.sep, _directory_flags())
        descriptors.append(current)
        for component in path.parent.parts[1:]:
            try:
                named = os.stat(
                    component,
                    dir_fd=current,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise PublicationError(
                    "clean-room destination ancestors must be real directories"
                ) from error
            if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
                raise PublicationError(
                    "clean-room destination ancestors must not be symlinks"
                )
            try:
                child = os.open(component, _directory_flags(), dir_fd=current)
            except OSError as error:
                raise PublicationError(
                    "clean-room destination ancestors must be real directories"
                ) from error
            descriptors.append(child)
            binding = _DirectoryBinding(current, component, child)
            bindings.append(binding)
            _assert_directory_binding(
                current,
                component,
                child,
                label="clean-room ancestor",
            )
            current = child
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    return path, tuple(bindings), tuple(descriptors)


def _create_pinned_directory(parent_fd: int, name: str, *, label: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError as error:
        raise PublicationError(
            f"{label} already exists; an empty owner-only root must be newly created"
        ) from error
    except OSError as error:
        raise PublicationError(f"{label} cannot be created") from error
    descriptor = -1
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
        os.fchmod(descriptor, 0o700)
        _assert_directory_binding(
            parent_fd,
            name,
            descriptor,
            label=label,
            owner_only=True,
        )
        os.fsync(parent_fd)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _prepare_pinned_clean_room(destination: Path) -> _PinnedCleanRoom:
    path, ancestor_bindings, ancestor_descriptors = _open_pinned_parent(destination)
    parent_fd = ancestor_descriptors[-1]
    root_fd = -1
    control_fd = -1
    control_name = f".{path.name}.control"
    try:
        root_fd = _create_pinned_directory(
            parent_fd,
            path.name,
            label="clean-room destination",
        )
        control_fd = _create_pinned_directory(
            parent_fd,
            control_name,
            label="clean-room control directory",
        )
        return _PinnedCleanRoom(
            path=path,
            ancestor_bindings=ancestor_bindings,
            ancestor_descriptors=ancestor_descriptors,
            parent_fd=parent_fd,
            root_name=path.name,
            root_fd=root_fd,
            control_name=control_name,
            control_fd=control_fd,
        )
    except BaseException:
        if control_fd >= 0:
            os.close(control_fd)
        if root_fd >= 0:
            os.close(root_fd)
        for descriptor in reversed(ancestor_descriptors):
            os.close(descriptor)
        raise


def _assert_clean_room_authority(room: _PinnedCleanRoom) -> None:
    for binding in room.ancestor_bindings:
        _assert_directory_binding(
            binding.parent_fd,
            binding.name,
            binding.descriptor,
            label="clean-room ancestor",
        )
    _assert_directory_binding(
        room.parent_fd,
        room.root_name,
        room.root_fd,
        label="clean-room destination",
        owner_only=True,
    )
    _assert_directory_binding(
        room.parent_fd,
        room.control_name,
        room.control_fd,
        label="clean-room control directory",
        owner_only=True,
    )


def _receipt_build_id(reference: S3ObjectVersion) -> str:
    key = urlsplit(reference.uri).path.removeprefix("/")
    prefix_parts = CORPUS_KEY_PREFIX.split("/")
    return key.split("/")[len(prefix_parts)]


def _local_object_paths(
    receipt: PhaseReceipt,
) -> tuple[dict[PurePosixPath, S3ObjectVersion], set[PurePosixPath]]:
    prefix = f"{CORPUS_KEY_PREFIX}/{receipt.build_id}/"
    files: dict[PurePosixPath, S3ObjectVersion] = {}
    directories: set[PurePosixPath] = set()
    for expected in receipt.objects:
        parsed = urlsplit(expected.uri)
        key = parsed.path.removeprefix("/")
        if parsed.hostname != CORPUS_BUCKET or not key.startswith(prefix):
            raise PublicationError("receipt object escapes its corpus build prefix")
        relative_text = key[len(prefix) :]
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or "\\" in relative_text
            or relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative_text == _PHASE_RECEIPT_NAME
            or relative in files
        ):
            raise PublicationError("receipt contains an unsafe local object path")
        files[relative] = expected
        for depth in range(1, len(relative.parts)):
            directories.add(PurePosixPath(*relative.parts[:depth]))
    if set(files) & directories:
        raise PublicationError("receipt object path collides with a directory")
    if PurePosixPath("receipt.json") not in files:
        raise PublicationError("receipt omits the canonical corpus receipt.json")
    return files, directories


def _create_expected_directories(
    root_fd: int,
    directories: set[PurePosixPath],
) -> dict[PurePosixPath, int]:
    descriptors = {PurePosixPath(): root_fd}
    try:
        for relative in sorted(
            directories,
            key=lambda path: (len(path.parts), path.as_posix()),
        ):
            parent = PurePosixPath(*relative.parts[:-1])
            descriptors[relative] = _create_pinned_directory(
                descriptors[parent],
                relative.name,
                label=f"clean-room directory {relative.as_posix()}",
            )
    except BaseException:
        for relative, descriptor in reversed(tuple(descriptors.items())):
            if relative.parts:
                os.close(descriptor)
        raise
    return descriptors


def _close_expected_directories(descriptors: dict[PurePosixPath, int]) -> None:
    for relative, descriptor in reversed(tuple(descriptors.items())):
        if relative.parts:
            os.close(descriptor)


def _assert_exact_namespace(
    descriptors: dict[PurePosixPath, int],
    *,
    file_identities: dict[PurePosixPath, os.stat_result],
    files: set[PurePosixPath],
    directories: set[PurePosixPath],
) -> None:
    actual_files: set[PurePosixPath] = set()
    actual_directories: set[PurePosixPath] = set()
    try:
        for relative in sorted(
            directories,
            key=lambda path: (len(path.parts), path.as_posix()),
        ):
            parent = PurePosixPath(*relative.parts[:-1])
            _assert_directory_binding(
                descriptors[parent],
                relative.name,
                descriptors[relative],
                label=f"clean-room directory {relative.as_posix()}",
                owner_only=True,
            )
        for relative_parent, descriptor in descriptors.items():
            for name in os.listdir(descriptor):
                metadata = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                relative = relative_parent / name
                if stat.S_ISLNK(metadata.st_mode):
                    raise PublicationError(
                        "clean-room namespace contains symlink: "
                        f"{relative.as_posix()}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    actual_directories.add(relative)
                    expected_fd = descriptors.get(relative)
                    if expected_fd is not None and not _same_inode(
                        metadata,
                        os.fstat(expected_fd),
                    ):
                        raise PublicationError(
                            "clean-room directory identity changed: "
                            f"{relative.as_posix()}"
                        )
                elif stat.S_ISREG(metadata.st_mode):
                    actual_files.add(relative)
                    expected_identity = file_identities.get(relative)
                    if expected_identity is not None and (
                        not _same_inode(metadata, expected_identity)
                        or metadata.st_size != expected_identity.st_size
                    ):
                        raise PublicationError(
                            "clean-room file identity changed: "
                            f"{relative.as_posix()}"
                        )
                else:
                    raise PublicationError(
                        "clean-room namespace contains special entry: "
                        f"{relative.as_posix()}"
                    )
    except PublicationError:
        raise
    except OSError as error:
        raise PublicationError("clean-room namespace cannot be inspected") from error
    if actual_files != files or actual_directories != directories:
        raise PublicationError("clean-room namespace is incomplete or foreign")


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as error:
        raise PublicationError("pinned phase receipt cannot be read") from error
    return b"".join(chunks)


def _pin_phase_receipt(
    control_fd: int,
    downloaded: os.stat_result,
    reference: S3ObjectVersion,
) -> _PinnedReceipt:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(_PHASE_RECEIPT_NAME, flags, dir_fd=control_fd)
    except OSError as error:
        raise PublicationError("downloaded phase receipt cannot be pinned") from error
    try:
        initial = os.fstat(descriptor)
        named = os.stat(
            _PHASE_RECEIPT_NAME,
            dir_fd=control_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(initial.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or not _same_inode(downloaded, initial)
            or not _same_inode(initial, named)
            or initial.st_size != reference.bytes
        ):
            raise PublicationError("downloaded phase receipt identity changed")
        payload = _read_descriptor(descriptor)
        if (
            len(payload) != reference.bytes
            or hashlib.sha256(payload).hexdigest() != reference.sha256
        ):
            raise PublicationError("downloaded phase receipt hash or size changed")
        return _PinnedReceipt(
            descriptor=descriptor,
            initial=initial,
            payload=payload,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_pinned_phase_receipt(
    room: _PinnedCleanRoom,
    pinned: _PinnedReceipt,
    reference: S3ObjectVersion,
) -> None:
    try:
        current = os.fstat(pinned.descriptor)
        named = os.stat(
            _PHASE_RECEIPT_NAME,
            dir_fd=room.control_fd,
            follow_symlinks=False,
        )
        entries = set(os.listdir(room.control_fd))
    except OSError as error:
        raise PublicationError("phase receipt authority cannot be inspected") from error
    if (
        entries != {_PHASE_RECEIPT_NAME}
        or not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or not _same_inode(pinned.initial, current)
        or not _same_inode(pinned.initial, named)
        or current.st_size != reference.bytes
        or named.st_size != reference.bytes
    ):
        raise PublicationError("phase receipt authority identity or name binding changed")
    payload = _read_descriptor(pinned.descriptor)
    if (
        payload != pinned.payload
        or len(payload) != reference.bytes
        or hashlib.sha256(payload).hexdigest() != reference.sha256
    ):
        raise PublicationError("phase receipt authority hash or size changed")


def cleanroom_verify(
    s3: S3Client,
    *,
    receipt_uri: str,
    receipt_version_id: str,
    receipt_sha256: str,
    destination: Path,
    expected_build_id: str,
    verifier: Callable[..., object] = verify_parallel_corpus,
) -> object:
    """Download one closed receipt/object set and run the corpus verifier."""

    room = _prepare_pinned_clean_room(destination)
    pinned: _PinnedReceipt | None = None
    directory_fds: dict[PurePosixPath, int] = {}
    try:
        reference = _resolve_exact_object(
            s3,
            uri=receipt_uri,
            version_id=receipt_version_id,
            sha256=receipt_sha256,
        )
        downloaded = _download_exact_object_at(
            s3,
            reference,
            parent_fd=room.control_fd,
            name=_PHASE_RECEIPT_NAME,
        )
        pinned = _pin_phase_receipt(room.control_fd, downloaded, reference)
        try:
            receipt = phase_receipt_from_bytes(pinned.payload)
        except ValueError as error:
            raise PublicationError("downloaded phase receipt is invalid") from error
        if receipt.phase != "final":
            raise PublicationError("clean-room verification requires a final receipt")
        if (
            receipt.build_id != expected_build_id
            or _receipt_build_id(reference) != expected_build_id
        ):
            raise PublicationError("phase receipt build ID does not match expectation")
        object_kms_arns = {item.kms_key_arn for item in receipt.objects}
        if object_kms_arns != {reference.kms_key_arn}:
            raise PublicationError(
                "phase receipt and corpus objects do not share one KMS key"
            )
        files, directories = _local_object_paths(receipt)
        directory_fds = _create_expected_directories(room.root_fd, directories)
        file_identities: dict[PurePosixPath, os.stat_result] = {}
        for relative, expected in files.items():
            parent = PurePosixPath(*relative.parts[:-1])
            file_identities[relative] = _download_exact_object_at(
                s3,
                expected,
                parent_fd=directory_fds[parent],
                name=relative.name,
            )
        _assert_clean_room_authority(room)
        _assert_exact_namespace(
            directory_fds,
            file_identities=file_identities,
            files=set(files),
            directories=directories,
        )
        result = verifier(room.path, expected_build_id=expected_build_id)
        _assert_clean_room_authority(room)
        _assert_exact_namespace(
            directory_fds,
            file_identities=file_identities,
            files=set(files),
            directories=directories,
        )
        _verify_pinned_phase_receipt(room, pinned, reference)
        return result
    finally:
        if pinned is not None:
            os.close(pinned.descriptor)
        if directory_fds:
            _close_expected_directories(directory_fds)
        room.close()


def _live_s3_client(*, profile: str, region: str) -> S3Client:
    try:
        import boto3
    except ImportError as error:
        raise PublicationError("boto3 is required for live clean-room use") from error
    return boto3.Session(
        profile_name=profile,
        region_name=region,
    ).client("s3")


def main(
    argv: Sequence[str] | None = None,
    *,
    s3: S3Client | None = None,
    verifier: Callable[..., object] = verify_parallel_corpus,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-uri", required=True)
    parser.add_argument("--receipt-version-id", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-build-id", required=True)
    parser.add_argument("--profile", choices=("sbsandbox",), default="sbsandbox")
    parser.add_argument("--region", choices=("us-east-1",), default="us-east-1")
    args = parser.parse_args(argv)
    client = s3 or _live_s3_client(profile=args.profile, region=args.region)
    cleanroom_verify(
        client,
        receipt_uri=args.receipt_uri,
        receipt_version_id=args.receipt_version_id,
        receipt_sha256=args.receipt_sha256,
        destination=args.destination,
        expected_build_id=args.expected_build_id,
        verifier=verifier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
