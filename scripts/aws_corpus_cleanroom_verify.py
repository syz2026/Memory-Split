#!/usr/bin/env python3
"""Download and verify one receipt-pinned AWS corpus without mutable S3 reads."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Callable, Sequence
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
    _resolve_exact_object,
    download_exact_object,
)
from corpusgen.parallel import verify_parallel_corpus  # noqa: E402


_PHASE_RECEIPT_NAME = ".phase-receipt.json"


def _prepare_empty_root(destination: Path) -> Path:
    root = Path(destination)
    if root.name in {"", ".", ".."}:
        raise PublicationError("clean-room destination must name a directory")
    try:
        metadata = os.lstat(root)
    except FileNotFoundError:
        try:
            parent = os.lstat(root.parent)
            if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
                raise PublicationError(
                    "clean-room destination parent must be a real directory"
                )
            root.mkdir(mode=0o700)
            metadata = os.lstat(root)
        except PublicationError:
            raise
        except OSError as error:
            raise PublicationError(
                "clean-room destination cannot be created"
            ) from error
    except OSError as error:
        raise PublicationError("clean-room destination cannot be inspected") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise PublicationError("clean-room destination must be a real directory")
    try:
        entries = list(os.scandir(root))
    except OSError as error:
        raise PublicationError("clean-room destination cannot be listed") from error
    if entries:
        first = entries[0]
        kind = "symlink" if first.is_symlink() else "foreign entry"
        raise PublicationError(
            f"clean-room destination must be empty; found {kind}: {first.name}"
        )
    return root


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
    root: Path,
    directories: set[PurePosixPath],
) -> None:
    for relative in sorted(
        directories,
        key=lambda path: (len(path.parts), path.as_posix()),
    ):
        path = root.joinpath(*relative.parts)
        try:
            path.mkdir(mode=0o700)
            metadata = os.lstat(path)
        except OSError as error:
            raise PublicationError(
                f"cannot create clean-room directory: {relative.as_posix()}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise PublicationError(
                f"clean-room directory is unsafe: {relative.as_posix()}"
            )


def _assert_exact_namespace(
    root: Path,
    *,
    files: set[PurePosixPath],
    directories: set[PurePosixPath],
) -> None:
    actual_files: set[PurePosixPath] = set()
    actual_directories: set[PurePosixPath] = set()
    pending = [(root, PurePosixPath())]
    while pending:
        directory, relative_parent = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PublicationError("clean-room namespace cannot be listed") from error
        for entry in entries:
            relative = relative_parent / entry.name
            if entry.is_symlink():
                raise PublicationError(
                    f"clean-room namespace contains symlink: {relative.as_posix()}"
                )
            if entry.is_dir(follow_symlinks=False):
                actual_directories.add(relative)
                pending.append((Path(entry.path), relative))
            elif entry.is_file(follow_symlinks=False):
                actual_files.add(relative)
            else:
                raise PublicationError(
                    f"clean-room namespace contains special entry: "
                    f"{relative.as_posix()}"
                )
    if actual_files != files or actual_directories != directories:
        raise PublicationError("clean-room namespace is incomplete or foreign")


def _remove_phase_receipt(path: Path) -> None:
    try:
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise PublicationError("downloaded phase receipt identity is unsafe")
        path.unlink()
        descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        return
    except PublicationError:
        raise
    except OSError as error:
        raise PublicationError("downloaded phase receipt cannot be removed") from error


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

    root = _prepare_empty_root(destination)
    reference = _resolve_exact_object(
        s3,
        uri=receipt_uri,
        version_id=receipt_version_id,
        sha256=receipt_sha256,
    )
    phase_path = root / _PHASE_RECEIPT_NAME
    try:
        download_exact_object(s3, reference, phase_path)
        try:
            receipt = phase_receipt_from_bytes(phase_path.read_bytes())
        except (OSError, ValueError) as error:
            raise PublicationError("downloaded phase receipt is invalid") from error
    finally:
        _remove_phase_receipt(phase_path)

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
    _create_expected_directories(root, directories)
    for relative, expected in files.items():
        download_exact_object(
            s3,
            expected,
            root.joinpath(*relative.parts),
        )
    _assert_exact_namespace(
        root,
        files=set(files),
        directories=directories,
    )
    return verifier(root, expected_build_id=expected_build_id)


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
