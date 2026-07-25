#!/usr/bin/env python3
"""Run read-only AWS corpus-builder gates and emit one launch intent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cluster.aws.corpus_builder.contracts import (  # noqa: E402
    S3ObjectVersion,
    launch_intent_to_bytes,
    s3_object_version_from_dict,
)
from cluster.aws.corpus_builder.preflight import (  # noqa: E402
    AwsClients,
    PreflightError,
    PreflightRequest,
    run_preflight,
    validate_local_request,
)


_MAX_INPUT_BYTES = 64 * 1024 * 1024


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"JSON input repeats field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> object:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PreflightError(f"{label} must be a regular file")
    payload = source.read_bytes()
    if not payload or len(payload) > _MAX_INPUT_BYTES:
        raise PreflightError(f"{label} has invalid size")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PreflightError(f"{label} contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{label} must contain valid UTF-8 JSON") from error


def _load_object_record(path: Path, *, label: str) -> S3ObjectVersion:
    try:
        return s3_object_version_from_dict(_read_json(path, label=label))
    except ValueError as error:
        raise PreflightError(f"{label} is invalid: {error}") from error


def _load_stack_outputs(path: Path) -> Mapping[str, str]:
    value = _read_json(path, label="stack outputs")
    if not isinstance(value, dict):
        raise PreflightError("stack outputs must be a JSON object")
    outputs: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item:
            raise PreflightError(
                "stack output names and values must be non-empty strings"
            )
        outputs[key] = item
    return outputs


def _live_clients(*, profile: str, region: str) -> AwsClients:
    try:
        import boto3
    except ImportError as error:
        raise PreflightError("boto3 is required for live preflight use") from error
    session = boto3.Session(profile_name=profile, region_name=region)
    return AwsClients(
        sts=session.client("sts", region_name=region),
        ec2=session.client("ec2", region_name=region),
        iam=session.client("iam", region_name=region),
        s3=session.client("s3", region_name=region),
        kms=session.client("kms", region_name=region),
        pricing=session.client("pricing", region_name="us-east-1"),
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise PreflightError("short write while emitting launch intent")
        written += count


def _unlink_owned_name(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int] | None,
) -> None:
    try:
        named = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError:
        named = None
    if (
        identity is not None
        and named is not None
        and (named.st_dev, named.st_ino) != identity
    ):
        raise PreflightError(
            f"launch intent cleanup refused changed inode: {name}"
        )
    os.unlink(name, dir_fd=parent_descriptor)


def _write_intent(path: Path, payload: bytes) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    if output.name in {"", ".", ".."}:
        raise PreflightError("launch intent output must name a file")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise PreflightError("launch intent parent must be a real directory")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    temporary_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = -1
    descriptor = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    temporary_exists = False
    published_exists = False
    success = False
    try:
        parent_descriptor = os.open(output.parent, parent_flags)
        try:
            os.stat(
                output.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise PreflightError("launch intent output already exists")
        temporary_name = (
            f".{output.name}.{os.getpid()}."
            f"{secrets.token_hex(16)}.tmp"
        )
        descriptor = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        opened = os.fstat(descriptor)
        temporary_identity = (opened.st_dev, opened.st_ino)
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        named = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != temporary_identity
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
        ):
            raise PreflightError("launch intent output authority drift")
        descriptor_to_close = descriptor
        descriptor = -1
        os.close(descriptor_to_close)
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise PreflightError(
                "launch intent output already exists"
            ) from error
        published_exists = True
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_exists = False
        final = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(final.st_mode)
            or (final.st_dev, final.st_ino) != temporary_identity
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_size != len(payload)
        ):
            raise PreflightError("launch intent output authority drift")
        os.fsync(parent_descriptor)
        success = True
    except PreflightError:
        raise
    except OSError as error:
        raise PreflightError("cannot emit launch intent securely") from error
    finally:
        primary_error = sys.exception()
        cleanup_errors: list[tuple[str, BaseException]] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as error:
                cleanup_errors.append(("descriptor close", error))
        if parent_descriptor >= 0:
            if not success:
                cleanup_targets = (
                    (output.name, published_exists),
                    (temporary_name, temporary_exists),
                )
                for cleanup_name, exists in cleanup_targets:
                    if cleanup_name is None or not exists:
                        continue
                    try:
                        _unlink_owned_name(
                            parent_descriptor,
                            cleanup_name,
                            temporary_identity,
                        )
                    except BaseException as error:
                        cleanup_errors.append(("unlink", error))
            try:
                os.close(parent_descriptor)
            except OSError as error:
                cleanup_errors.append(("parent close", error))
        if cleanup_errors:
            notes = [
                f"launch intent cleanup {action} failed: {error}"
                for action, error in cleanup_errors
            ]
            if primary_error is not None:
                for note in notes:
                    primary_error.add_note(note)
            else:
                raise PreflightError("; ".join(notes)) from cleanup_errors[0][1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--builder-profile",
        type=Path,
        default=(
            _ROOT
            / "cluster"
            / "profiles"
            / "aws-i4i.16xlarge-corpus-v1.json"
        ),
    )
    parser.add_argument("--package-record", type=Path, required=True)
    parser.add_argument("--source-manifest-record", type=Path, required=True)
    parser.add_argument("--software-gate-receipt", type=Path, required=True)
    parser.add_argument("--stack-outputs", type=Path, required=True)
    parser.add_argument("--ami-id", required=True)
    parser.add_argument("--ami-owner-id", required=True)
    parser.add_argument(
        "--expected-bootstrap-user-data-sha256",
        required=True,
    )
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--profile", choices=("sbsandbox",), default="sbsandbox")
    parser.add_argument("--region", choices=("us-east-1",), default="us-east-1")
    parser.add_argument(
        "--live",
        action="store_true",
        help="construct real read-only AWS clients after local validation",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    aws: AwsClients | None = None,
    now: datetime | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    current_time = now or datetime.now(timezone.utc).replace(microsecond=0)
    request = PreflightRequest(
        profile_path=arguments.builder_profile,
        package=_load_object_record(
            arguments.package_record,
            label="package record",
        ),
        source_manifest=_load_object_record(
            arguments.source_manifest_record,
            label="source-manifest record",
        ),
        software_gate_receipt=arguments.software_gate_receipt,
        stack_outputs=_load_stack_outputs(arguments.stack_outputs),
        ami_id=arguments.ami_id,
        ami_owner_id=arguments.ami_owner_id,
        expected_bootstrap_user_data_sha256=(
            arguments.expected_bootstrap_user_data_sha256
        ),
    )
    validate_local_request(request, now=current_time)
    if aws is not None:
        if arguments.live:
            raise PreflightError(
                "--live cannot be combined with injected AWS clients"
            )
        clients = aws
    elif arguments.live:
        clients = _live_clients(
            profile=arguments.profile,
            region=arguments.region,
        )
    else:
        raise PreflightError(
            "AWS clients must be injected unless --live is explicitly set"
        )
    result = run_preflight(
        request,
        aws=clients,
        now=current_time,
    )
    payload = launch_intent_to_bytes(result.intent)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != result.intent_sha256:
        raise PreflightError("launch intent hash changed after preflight")
    _write_intent(arguments.intent, payload)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
