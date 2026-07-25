#!/usr/bin/env python3
"""Sign one exact msctl dry-run approval scope without rebuilding resources."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msctl.approval import APPROVAL_OPERATIONS, KEYS_ENV  # noqa: E402
from msctl.jsonutil import (  # noqa: E402
    canonical_json,
    require_nonnegative_int,
    require_nonnegative_number,
    require_object,
    require_sha256,
)


class ApprovalSigningError(ValueError):
    """The dry-run report cannot produce one closed approval envelope."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ApprovalSigningError(f"JSON object repeats field: {key}")
        value[key] = item
    return value


def _load_report(path: Path | str) -> dict[str, object]:
    candidate = Path(path)
    try:
        before = candidate.stat(follow_symlinks=False)
        if (
            candidate.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > 2 * 1024 * 1024
        ):
            raise ApprovalSigningError(
                "dry-run report must be one bounded singly linked regular file"
            )
        payload = candidate.read_bytes()
        after = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise ApprovalSigningError("dry-run report cannot be read safely") from error
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
    ):
        raise ApprovalSigningError("dry-run report changed while being read")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ApprovalSigningError(
                    f"dry-run report contains non-finite {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApprovalSigningError(
            "dry-run report must contain one UTF-8 JSON object"
        ) from error
    if not isinstance(value, dict):
        raise ApprovalSigningError("dry-run report root must be an object")
    return value


def _expiry(value: str, *, now: datetime | None) -> str:
    if not value.endswith("Z"):
        raise ApprovalSigningError("approval expiry must be RFC 3339 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ApprovalSigningError(
            "approval expiry must be RFC 3339 UTC"
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed <= (now or datetime.now(UTC))
    ):
        raise ApprovalSigningError("approval expiry must be future UTC")
    return value


def _secret(
    environ: Mapping[str, str],
    *,
    key_id: str,
) -> bytes:
    encoded = environ.get(KEYS_ENV)
    if encoded is None:
        raise ApprovalSigningError(f"{KEYS_ENV} is required")
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ApprovalSigningError(f"{KEYS_ENV} must be a JSON object") from error
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(secret, str)
        for key, secret in value.items()
    ):
        raise ApprovalSigningError(f"{KEYS_ENV} must map key IDs to strings")
    selected = value.get(key_id)
    if not isinstance(selected, str) or len(selected.encode("utf-8")) < 32:
        raise ApprovalSigningError(
            "key ID must select a signing secret of at least 32 bytes"
        )
    return selected.encode("utf-8")


def _scope(
    result: dict[str, object],
    resources: dict[str, object],
    *,
    operation: str,
) -> tuple[str, str]:
    if operation == "fleet-advance":
        release = result.get("release_sha256")
        scope = resources.get("fleet_plan_sha256")
        if result.get("fleet_plan_sha256") != scope:
            raise ApprovalSigningError(
                "fleet advance result and resources bind different plans"
            )
    elif operation == "canary":
        release = resources.get("release_sha256")
        scope = resources.get("orchestration_plan_sha256")
        if result.get("plan_sha256") != scope:
            raise ApprovalSigningError(
                "canary result and resources bind different plans"
            )
    else:
        release = resources.get("release_sha256")
        scope = resources.get("run_manifest_sha256")
        if (
            result.get("run_manifest_sha256") is not None
            and result.get("run_manifest_sha256") != scope
        ):
            raise ApprovalSigningError(
                "lifecycle result and resources bind different manifests"
            )
    try:
        return (
            require_sha256(release, label="approval release"),
            require_sha256(scope, label="approval scope"),
        )
    except Exception as error:
        raise ApprovalSigningError(
            "dry run omits its top-level approval release or scope"
        ) from error


def sign_approval_report(
    report_path: Path | str,
    out_path: Path | str,
    *,
    operation: str,
    key_id: str,
    expires_at: str,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Sign the exact approval_resources from a lifecycle, canary, or advance."""

    if operation not in APPROVAL_OPERATIONS:
        raise ApprovalSigningError("approval operation is unsupported")
    if not key_id:
        raise ApprovalSigningError("approval key ID is required")
    report = _load_report(report_path)
    if report.get("dry_run") is not True:
        raise ApprovalSigningError("approval input must be an msctl dry-run report")
    result = require_object(report.get("result"), label="dry-run result")
    resources = require_object(
        result.get("approval_resources"),
        label="dry-run approval resources",
    )
    if resources.get("operation") != operation:
        raise ApprovalSigningError(
            "dry-run approval resources name a different operation"
        )
    provider = resources.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ApprovalSigningError("approval resources omit the provider")
    release_sha256, scope_sha256 = _scope(
        result,
        resources,
        operation=operation,
    )
    try:
        jobs = require_nonnegative_int(
            resources.get("jobs"),
            label="approval jobs",
        )
        gpu_hours = require_nonnegative_number(
            resources.get("gpu_hours"),
            label="approval GPU hours",
        )
    except Exception as error:
        raise ApprovalSigningError("approval resource limits are invalid") from error
    secret = _secret(
        dict(os.environ if environ is None else environ),
        key_id=key_id,
    )
    unsigned = {
        "schema_version": 1,
        "receipt_id": (
            f"{operation}-{hashlib.sha256(canonical_json(resources)).hexdigest()}"
        ),
        "provider": provider,
        "operation": operation,
        "release_sha256": release_sha256,
        "run_manifest_sha256": scope_sha256,
        "resources": resources,
        "limits": {"gpu_hours": gpu_hours, "jobs": jobs},
        "expires_at": _expiry(expires_at, now=now),
        "key_id": key_id,
    }
    receipt = {
        **unsigned,
        "signature": hmac.new(
            secret,
            canonical_json(unsigned),
            hashlib.sha256,
        ).hexdigest(),
    }
    payload = canonical_json(receipt) + b"\n"
    destination = Path(out_path)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    except FileExistsError as error:
        raise ApprovalSigningError(
            "refusing to replace an existing approval receipt"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return {
        "schema_version": 1,
        "operation": operation,
        "release_sha256": release_sha256,
        "scope_sha256": scope_sha256,
        "approval_sha256": hashlib.sha256(payload).hexdigest(),
        "out": str(destination),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sign one exact msctl dry-run approval_resources object."
    )
    parser.add_argument("--dry-run-report", type=Path, required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report = sign_approval_report(
            arguments.dry_run_report,
            arguments.out,
            operation=arguments.operation,
            key_id=arguments.key_id,
            expires_at=arguments.expires_at,
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except (ApprovalSigningError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
