"""HMAC-signed, scope-bound mutation approvals."""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import UTC, datetime
from pathlib import Path

from .contracts import RunManifest
from .errors import MsctlError
from .jsonutil import (
    canonical_json,
    load_json,
    require_exact_keys,
    require_nonnegative_number,
    require_object,
    require_positive_int,
    require_sha256,
)
from .profile import IlluminaProfile


APPROVAL_OPERATIONS = {
    "submit",
    "resume",
    "cancel",
    "evaluate",
    "cleanup",
}
KEY_ENV = "MSCTL_APPROVAL_KEY"


def _parse_expiry(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MsctlError(
            "APPROVAL_INVALID",
            "approval expiry must be an RFC 3339 UTC timestamp",
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise MsctlError(
            "APPROVAL_INVALID",
            "approval expiry must be an RFC 3339 UTC timestamp",
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise MsctlError(
            "APPROVAL_INVALID",
            "approval expiry must be in UTC",
        )
    return parsed


def verify_approval(
    path: Path | str | None,
    *,
    operation: str,
    release_sha256: str,
    run_manifest: RunManifest,
    profile: IlluminaProfile,
    environ: dict[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    return verify_scope_approval(
        path,
        operation=operation,
        release_sha256=release_sha256,
        scope_sha256=run_manifest.sha256,
        requested_jobs=len(run_manifest.runs),
        requested_gpu_hours=run_manifest.gpu_hours,
        profile=profile,
        environ=environ,
        now=now,
    )


def verify_scope_approval(
    path: Path | str | None,
    *,
    operation: str,
    release_sha256: str,
    scope_sha256: str,
    requested_jobs: int,
    requested_gpu_hours: float,
    profile: IlluminaProfile,
    environ: dict[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    if path is None:
        raise MsctlError(
            "APPROVAL_REQUIRED",
            f"{operation} --apply requires a signed approval receipt",
        )
    receipt = require_object(
        load_json(path, label="approval receipt"),
        label="approval receipt",
    )
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_id",
            "provider",
            "operation",
            "release_sha256",
            "run_manifest_sha256",
            "limits",
            "expires_at",
            "key_id",
            "signature",
        },
        label="approval receipt",
    )
    if receipt["schema_version"] != 1:
        raise MsctlError(
            "APPROVAL_INVALID",
            "unsupported approval receipt schema",
        )
    if (
        not isinstance(receipt["receipt_id"], str)
        or not receipt["receipt_id"]
        or not isinstance(receipt["key_id"], str)
        or not receipt["key_id"]
    ):
        raise MsctlError(
            "APPROVAL_INVALID",
            "approval receipt identifiers are invalid",
        )
    if receipt["operation"] not in APPROVAL_OPERATIONS:
        raise MsctlError(
            "APPROVAL_INVALID",
            "approval operation is unsupported",
        )
    signature = require_sha256(
        receipt["signature"], label="approval receipt.signature"
    )
    env = os.environ if environ is None else environ
    secret = env.get(KEY_ENV)
    if secret is None or len(secret.encode("utf-8")) < 32:
        raise MsctlError(
            "APPROVAL_KEY_UNAVAILABLE",
            f"{KEY_ENV} must contain at least 32 bytes",
        )
    unsigned = {
        key: value for key, value in receipt.items() if key != "signature"
    }
    expected = hmac.new(
        secret.encode("utf-8"),
        canonical_json(unsigned),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise MsctlError(
            "APPROVAL_SIGNATURE_INVALID",
            "approval receipt signature is invalid",
        )
    expiry = _parse_expiry(receipt["expires_at"])
    current = now or datetime.now(UTC)
    if current >= expiry:
        raise MsctlError(
            "APPROVAL_EXPIRED",
            "approval receipt has expired",
            details={"expires_at": receipt["expires_at"]},
        )
    receipt_release = require_sha256(
        receipt["release_sha256"],
        label="approval receipt.release_sha256",
    )
    receipt_runs = require_sha256(
        receipt["run_manifest_sha256"],
        label="approval receipt.run_manifest_sha256",
    )
    if (
        receipt["provider"] != profile.provider
        or receipt["operation"] != operation
        or receipt_release != release_sha256
        or receipt_runs != scope_sha256
    ):
        raise MsctlError(
            "APPROVAL_SCOPE_MISMATCH",
            "approval receipt does not bind this provider and operation",
        )
    limits = require_object(
        receipt["limits"], label="approval receipt.limits"
    )
    require_exact_keys(
        limits,
        {"gpu_hours", "jobs"},
        label="approval receipt.limits",
    )
    gpu_limit = require_nonnegative_number(
        limits["gpu_hours"],
        label="approval receipt.limits.gpu_hours",
    )
    job_limit = require_positive_int(
        limits["jobs"],
        label="approval receipt.limits.jobs",
    )
    if requested_jobs > job_limit or requested_gpu_hours > gpu_limit:
        raise MsctlError(
            "APPROVAL_LIMIT_EXCEEDED",
            "requested jobs or GPU-hours exceed the approval limits",
            details={
                "requested_jobs": requested_jobs,
                "requested_gpu_hours": requested_gpu_hours,
                "limit_jobs": job_limit,
                "limit_gpu_hours": gpu_limit,
            },
        )
    return receipt
