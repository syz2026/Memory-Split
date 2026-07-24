"""Prior-seed finalization admission for safe sequential seed transitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from cluster.aws.p5.run_finalization import (
    FinalizationError,
    parse_run_finalization_receipt_bytes,
)

from .aws_contracts import ARMS, SNAPSHOT_STEPS, validate_sha256
from .aws_lifecycle import ProviderLifecycleBinding


PRIOR_EVIDENCE_OBJECT_COUNT = len(ARMS) * (len(SNAPSHOT_STEPS) + 1) + 2

_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)


class SeedTransitionError(ValueError):
    """A fail-closed sequential seed-transition admission error."""


def _fail(message: str) -> SeedTransitionError:
    return SeedTransitionError(f"seed transition {message}")


def _sha256_text(value: object, *, label: str) -> str:
    try:
        return validate_sha256(value)
    except ValueError as error:
        raise _fail(f"{label} must be a lowercase SHA-256") from error


def _nonnull_version(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value in {"", "null"}:
        raise _fail(f"{label} version ID must be a real non-null version")
    return value


def _s3_uri(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{label} URI must be a string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise _fail(f"{label} URI is invalid") from error
    parts = parsed.path.removeprefix("/").split("/")
    if (
        parsed.scheme != "s3"
        or parsed.hostname is None
        or parsed.netloc != parsed.hostname
        or port is not None
        or parsed.query
        or parsed.fragment
        or _BUCKET_RE.fullmatch(parsed.hostname) is None
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in value
    ):
        raise _fail(f"{label} URI is not one immutable s3:// object")
    return value


@dataclass(frozen=True)
class PriorRunReceiptRef:
    """Exact versioned identity of one prior-seed finalization receipt."""

    uri: str
    sha256: str
    version_id: str

    def __post_init__(self) -> None:
        _s3_uri(self.uri, label="prior run receipt")
        _sha256_text(self.sha256, label="prior run receipt")
        _nonnull_version(self.version_id, label="prior run receipt")


@dataclass(frozen=True)
class PriorEvidenceObject:
    """One HEAD-verifiable durable object bound by the prior finalization."""

    uri: str
    sha256: str
    bytes: int | None
    version_id: str

    def __post_init__(self) -> None:
        _s3_uri(self.uri, label="prior evidence object")
        _sha256_text(self.sha256, label="prior evidence object")
        if self.bytes is not None and (
            type(self.bytes) is not int or self.bytes <= 0
        ):
            raise _fail(
                "prior evidence object bytes must be a positive integer"
            )
        _nonnull_version(self.version_id, label="prior evidence object")


@dataclass(frozen=True)
class AdmittedPriorRun:
    """One admitted prior-seed finalization and its exact evidence set."""

    seed: int
    receipt: PriorRunReceiptRef
    evidence: tuple[PriorEvidenceObject, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 8:
            raise _fail("admitted prior seed must be an integer 0 through 8")
        if not isinstance(self.receipt, PriorRunReceiptRef):
            raise _fail("admitted prior run requires its exact receipt")
        if (
            not isinstance(self.evidence, tuple)
            or len(self.evidence) != PRIOR_EVIDENCE_OBJECT_COUNT
            or any(
                not isinstance(item, PriorEvidenceObject)
                for item in self.evidence
            )
        ):
            raise _fail(
                "admitted prior run must bind exactly 14 evidence objects"
            )


def admit_prior_seed_finalization(
    payload: bytes,
    *,
    ref: PriorRunReceiptRef,
    binding: ProviderLifecycleBinding,
    release_sha256: str,
    release_receipt_sha256: str,
    dataset_receipt_sha256: str,
    dataset_build_id: str,
    ordered_stream_sha256: str,
    source_commit: str,
    source_tree: str,
    instance_id: str,
) -> AdmittedPriorRun:
    """Authenticate one prior-seed finalization receipt before seed N."""

    if not isinstance(ref, PriorRunReceiptRef):
        raise _fail("requires one exact prior receipt reference")
    if not isinstance(binding, ProviderLifecycleBinding):
        raise _fail("requires the authenticated provider lifecycle binding")
    for label, value in (
        ("release", release_sha256),
        ("release receipt", release_receipt_sha256),
        ("dataset receipt", dataset_receipt_sha256),
        ("dataset build", dataset_build_id),
        ("ordered stream", ordered_stream_sha256),
    ):
        _sha256_text(value, label=label)
    for label, value in (
        ("source commit", source_commit),
        ("source tree", source_tree),
    ):
        if not isinstance(value, str) or _GIT_SHA1_RE.fullmatch(value) is None:
            raise _fail(f"{label} must be a lowercase Git SHA-1")
    if (
        not isinstance(instance_id, str)
        or _INSTANCE_RE.fullmatch(instance_id) is None
    ):
        raise _fail("instance ID is invalid")
    if binding.instance_id != instance_id:
        raise _fail("prior run must use the same selected cohort instance")
    if binding.seed < 1:
        raise _fail("seed 0 forbids a prior-seed finalization receipt")
    prior_seed = binding.seed - 1

    try:
        value = parse_run_finalization_receipt_bytes(
            payload,
            receipt_uri=ref.uri,
            receipt_sha256=ref.sha256,
            receipt_version_id=ref.version_id,
        )
    except FinalizationError as error:
        raise _fail(
            f"receipt is not one canonical run finalization: {error}"
        ) from error
    if value["seed"] != prior_seed:
        raise _fail("receipt seed must be exactly the prior sequential seed")
    # Boot ID may legitimately differ after a reboot between seeds; every
    # other lifecycle commitment must match the authenticated binding, so
    # re-parse against the expected binding with only seed and boot rebound.
    try:
        expected_binding = replace(
            binding,
            seed=prior_seed,
            boot_id=str(value["boot_id"]),
        )
    except (TypeError, ValueError) as error:
        raise _fail("receipt boot identity is invalid") from error
    try:
        parse_run_finalization_receipt_bytes(
            payload,
            receipt_uri=ref.uri,
            receipt_sha256=ref.sha256,
            receipt_version_id=ref.version_id,
            expected_binding=expected_binding,
        )
    except FinalizationError as error:
        raise _fail(
            f"receipt lifecycle authority differs: {error}"
        ) from error
    for field, expected_value in (
        ("release_sha256", release_sha256),
        ("release_receipt_sha256", release_receipt_sha256),
        ("dataset_receipt_sha256", dataset_receipt_sha256),
        ("dataset_build_id", dataset_build_id),
        ("ordered_stream_sha256", ordered_stream_sha256),
        ("source_commit", source_commit),
        ("source_tree", source_tree),
    ):
        if value[field] != expected_value:
            raise _fail(
                f"receipt {field} differs from the current cohort identity"
            )

    evidence: list[PriorEvidenceObject] = []
    for row in value["arms"]:
        for item in row["snapshots"]:
            snapshot = item["object"]
            evidence.append(
                PriorEvidenceObject(
                    uri=str(snapshot["uri"]),
                    sha256=str(snapshot["sha256"]),
                    bytes=snapshot["bytes"],
                    version_id=str(snapshot["version_id"]),
                )
            )
        log = row["log"]
        evidence.append(
            PriorEvidenceObject(
                uri=str(log["uri"]),
                sha256=str(log["sha256"]),
                bytes=log["bytes"],
                version_id=str(log["version_id"]),
            )
        )
    checkpoint = value["checkpoint_receipt"]
    evidence.append(
        PriorEvidenceObject(
            uri=str(checkpoint["uri"]),
            sha256=str(checkpoint["sha256"]),
            bytes=None,
            version_id=str(checkpoint["version_id"]),
        )
    )
    evidence.append(
        PriorEvidenceObject(
            uri=ref.uri,
            sha256=ref.sha256,
            bytes=len(payload),
            version_id=ref.version_id,
        )
    )
    return AdmittedPriorRun(
        seed=prior_seed,
        receipt=ref,
        evidence=tuple(evidence),
    )


__all__ = [
    "AdmittedPriorRun",
    "PRIOR_EVIDENCE_OBJECT_COUNT",
    "PriorEvidenceObject",
    "PriorRunReceiptRef",
    "SeedTransitionError",
    "admit_prior_seed_finalization",
]
