"""Durable per-seed training-evidence collection contracts and admission.

Task 3F implements only per-seed training-evidence collection: exactly
sixteen objects (ten snapshots, two logs, two final checkpoints, the Task 3C
checkpoint receipt, and the Task 3D finalization receipt) bound by one
canonical collection receipt. Cohort evaluation/report collection belongs to
the later sealed-evaluation task, so per-seed collection is necessary but
never sufficient for selected cleanup or termination.

Evidence bodies are opaque: no snapshot, log, or checkpoint body is ever
interpreted for outcomes. Only the finalization and checkpoint receipts are
parsed, and only for identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, replace
from typing import Sequence
from urllib.parse import urlsplit

from cluster.aws.p5.checkpoint_mirror import _canonical_json
from cluster.aws.p5.run_finalization import (
    _RECEIPT_BINDING_FIELDS,
    _RECEIPT_FIELDS,
)

from .aws_contracts import (
    ARMS,
    COHORT_ID,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_object_key,
    checkpoint_receipt_key,
    collection_receipt_key,
    log_object_key,
    run_receipt_key,
    snapshot_object_key,
)
from .aws_lifecycle import ProviderLifecycleBinding
from .aws_seed_transition import AdmittedPriorRun
from .errors import MsctlError


COLLECTION_RECEIPT_TYPE = "memorysplit-aws-seed-collection-v3"
SEED_COLLECTION_OBJECT_COUNT = 16
# The collection receipt carries every portable Task 3D provenance and
# lifecycle field; only the arm evidence tree and finalization timestamp are
# replaced by the flat object rows and the collection timestamp.
COLLECTION_RECEIPT_FIELDS = frozenset(
    (_RECEIPT_FIELDS - {"arms", "finalized_at"})
    | {"collected_at", "objects", "run_receipt"}
)

_OBJECT_KINDS = (
    "snapshot",
    "log",
    "checkpoint",
    "checkpoint_receipt",
    "run_receipt",
)
_OBJECT_ROW_FIELDS = frozenset(
    {"arm", "bytes", "kind", "sha256", "step", "uri", "version_id"}
)
_RECEIPT_REF_FIELDS = frozenset({"bytes", "sha256", "uri", "version_id"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_PROFILE_PROVIDERS = {
    "aws-p5.48xlarge-v3": "aws-p5.48xlarge",
    "aws-p6-b300.48xlarge-v3": "aws-p6-b300.48xlarge",
}
_DOWNLOAD_FLOOR_SECONDS = 300.0
_DOWNLOAD_BYTES_PER_SECOND = 8 * 1024 * 1024


def _expected_row_plan() -> tuple[tuple[str, str | None, int | None], ...]:
    plan: list[tuple[str, str | None, int | None]] = []
    for arm in ARMS:
        plan.extend(("snapshot", arm, step) for step in SNAPSHOT_STEPS)
        plan.append(("log", arm, None))
    plan.extend(("checkpoint", arm, None) for arm in ARMS)
    plan.append(("checkpoint_receipt", None, None))
    plan.append(("run_receipt", None, None))
    return tuple(plan)


_ROW_PLAN = _expected_row_plan()


class CollectionError(ValueError):
    """A fail-closed per-seed evidence collection error."""


def _fail(message: str) -> CollectionError:
    return CollectionError(f"seed collection {message}")


def _sha256_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{label} must be a lowercase SHA-256")
    return value


def _nonnull_version(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value in {"", "null"}:
        raise _fail(f"{label} version ID must be a real non-null version")
    return value


def _positive_bytes(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _fail(f"{label} bytes must be a positive exact integer")
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


def _strict_json(payload: bytes, *, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise _fail(f"{label} repeats field {key}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                _fail(f"{label} contains non-finite {constant}")
            ),
        )
    except CollectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise _fail(f"{label} must be one JSON object")
    return parsed


def _object_root(receipt_uri: object, *, suffix: str, label: str) -> str:
    if not isinstance(receipt_uri, str) or not receipt_uri.endswith(suffix):
        raise _fail(f"{label} URI does not use its canonical key")
    root = receipt_uri[: -len(suffix)]
    parsed = urlsplit(root)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or "\\" in root
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise _fail(f"{label} URI root is not a safe s3:// root")
    return root


def _row_key(
    kind: str,
    seed: int,
    arm: str | None,
    step: int | None,
    sha256: str,
) -> str:
    try:
        if kind == "snapshot":
            return snapshot_object_key(seed, arm, step, sha256)
        if kind == "log":
            return log_object_key(seed, arm, sha256)
        if kind == "checkpoint":
            return checkpoint_object_key(seed, arm, sha256)
        if kind == "checkpoint_receipt":
            return checkpoint_receipt_key(seed, sha256)
        return run_receipt_key(seed, sha256)
    except ValueError as error:
        raise _fail(f"{kind} object key is invalid") from error


@dataclass(frozen=True)
class CollectionReceiptRef:
    """Exact versioned identity of one per-seed collection receipt."""

    uri: str
    sha256: str
    version_id: str

    def __post_init__(self) -> None:
        _s3_uri(self.uri, label="collection receipt")
        _sha256_text(self.sha256, label="collection receipt")
        _nonnull_version(self.version_id, label="collection receipt")


@dataclass(frozen=True)
class CollectedObject:
    """One canonical collected evidence object row."""

    kind: str
    arm: str | None
    step: int | None
    uri: str
    sha256: str
    bytes: int
    version_id: str

    def __post_init__(self) -> None:
        if self.kind not in _OBJECT_KINDS:
            raise _fail("collected object kind is not in the closed set")
        if self.kind in {"snapshot", "log", "checkpoint"}:
            if self.arm not in ARMS:
                raise _fail("collected object arm must be dense or split90")
        elif self.arm is not None:
            raise _fail("collected receipt rows must not carry an arm")
        if self.kind == "snapshot":
            if type(self.step) is not int or self.step not in SNAPSHOT_STEPS:
                raise _fail(
                    "collected snapshot step must be in the frozen schedule"
                )
        elif self.step is not None:
            raise _fail("only collected snapshots carry a step")
        _s3_uri(self.uri, label="collected object")
        _sha256_text(self.sha256, label="collected object")
        _positive_bytes(self.bytes, label="collected object")
        _nonnull_version(self.version_id, label="collected object")


@dataclass(frozen=True)
class AdmittedPriorCollection:
    """One admitted prior-seed collection and its two checkpoint objects."""

    seed: int
    receipt: CollectionReceiptRef
    checkpoints: tuple[CollectedObject, CollectedObject]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 8:
            raise _fail(
                "admitted prior collection seed must be an integer 0 "
                "through 8"
            )
        if not isinstance(self.receipt, CollectionReceiptRef):
            raise _fail("admitted prior collection requires its exact receipt")
        if (
            not isinstance(self.checkpoints, tuple)
            or len(self.checkpoints) != 2
            or any(
                not isinstance(item, CollectedObject)
                or item.kind != "checkpoint"
                for item in self.checkpoints
            )
            or tuple(item.arm for item in self.checkpoints) != ARMS
        ):
            raise _fail(
                "admitted prior collection must bind exactly the Dense and "
                "Split90 checkpoints"
            )


def _receipt_reference(
    candidate: object,
    *,
    root: str,
    expected_key_for: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(candidate, dict) or set(candidate) != (
        _RECEIPT_REF_FIELDS
    ):
        raise _fail(f"{label} reference fields do not match")
    digest = _sha256_text(candidate["sha256"], label=label)
    _nonnull_version(candidate["version_id"], label=label)
    _positive_bytes(candidate["bytes"], label=label)
    if candidate["uri"] != f"{root}/" + expected_key_for(digest):
        raise _fail(f"{label} URI does not use its canonical key")
    return candidate


def parse_seed_collection_receipt_bytes(
    payload: bytes,
    *,
    receipt_uri: str,
    receipt_sha256: str,
    receipt_version_id: str,
    expected_binding: ProviderLifecycleBinding | None = None,
) -> dict[str, object]:
    """Parse exact canonical bytes for one per-seed collection receipt."""

    if not isinstance(payload, bytes) or not payload:
        raise _fail("receipt bytes are missing")
    value = _strict_json(payload, label="receipt")
    if _canonical_json(value) != payload:
        raise _fail("receipt bytes are not canonical")
    expected_sha256 = _sha256_text(receipt_sha256, label="receipt")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise _fail("receipt hash does not match its bytes")
    _nonnull_version(receipt_version_id, label="receipt")
    if set(value) != COLLECTION_RECEIPT_FIELDS:
        raise _fail("receipt fields do not match the closed contract")
    seed = value["seed"]
    if type(seed) is not int or seed not in SEEDS:
        raise _fail("receipt seed must be an exact integer from 0 to 9")
    root = _object_root(
        receipt_uri,
        suffix="/" + collection_receipt_key(seed, expected_sha256),
        label="receipt",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 3
        or value["receipt_type"] != COLLECTION_RECEIPT_TYPE
        or value["cohort_id"] != COHORT_ID
        or value["complete"] is not True
    ):
        raise _fail("receipt identity is invalid")
    if (
        not isinstance(value["request_id"], str)
        or _REQUEST_ID_RE.fullmatch(value["request_id"]) is None
    ):
        raise _fail("receipt request ID must be 32 lowercase hex")
    if (
        not isinstance(value["collected_at"], str)
        or _UTC_RE.fullmatch(value["collected_at"]) is None
    ):
        raise _fail("receipt collected_at must be canonical UTC RFC3339")
    if _PROFILE_PROVIDERS.get(value["profile_id"]) != value["provider"]:
        raise _fail("receipt provider does not match its selected profile")
    for field in (
        "profile_sha256",
        "hardware_amendment_sha256",
        "provider_selection_sha256",
        "runtime_lock_sha256",
        "runtime_sbom_sha256",
        "qualification_evidence_sha256",
        "environment_receipt_sha256",
        "canary_receipt_sha256",
        "qualification_approval_receipt_sha256",
        "qualification_approval_public_key_sha256",
        "objective_controls_contract_sha256",
        "release_sha256",
        "release_receipt_sha256",
        "run_manifest_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
    ):
        _sha256_text(value[field], label=f"receipt {field}")
    _nonnull_version(
        value["provider_selection_version_id"],
        label="receipt provider selection",
    )
    for field, pattern in (
        ("source_commit", _GIT_SHA1_RE),
        ("source_tree", _GIT_SHA1_RE),
        ("instance_id", _INSTANCE_RE),
        ("boot_id", _BOOT_RE),
    ):
        if (
            not isinstance(value[field], str)
            or pattern.fullmatch(value[field]) is None
        ):
            raise _fail(f"receipt {field} is invalid")
    if expected_binding is not None:
        if not isinstance(expected_binding, ProviderLifecycleBinding):
            raise _fail("expected binding must be a provider lifecycle binding")
        expected = expected_binding.to_dict()
        if any(
            value[field] != expected[binding_field]
            for field, binding_field in _RECEIPT_BINDING_FIELDS.items()
        ):
            raise _fail(
                "receipt differs from the authenticated provider lifecycle"
            )

    run_reference = _receipt_reference(
        value["run_receipt"],
        root=root,
        expected_key_for=lambda digest: run_receipt_key(seed, digest),
        label="receipt run receipt",
    )
    checkpoint_reference = _receipt_reference(
        value["checkpoint_receipt"],
        root=root,
        expected_key_for=lambda digest: checkpoint_receipt_key(seed, digest),
        label="receipt checkpoint receipt",
    )

    rows = value["objects"]
    if not isinstance(rows, list) or len(rows) != (
        SEED_COLLECTION_OBJECT_COUNT
    ):
        raise _fail(
            "receipt objects must be exactly the sixteen canonical rows"
        )
    for index, (row, (kind, arm, step)) in enumerate(
        zip(rows, _ROW_PLAN, strict=True)
    ):
        label = f"receipt object {index}"
        if not isinstance(row, dict) or set(row) != _OBJECT_ROW_FIELDS:
            raise _fail(f"{label} fields do not match")
        if row["kind"] != kind or row["arm"] != arm or row["step"] != step:
            raise _fail(
                f"{label} does not follow the canonical collection order"
            )
        digest = _sha256_text(row["sha256"], label=label)
        _positive_bytes(row["bytes"], label=label)
        _nonnull_version(row["version_id"], label=label)
        if row["uri"] != f"{root}/" + _row_key(kind, seed, arm, step, digest):
            raise _fail(f"{label} URI does not use its shared key helper")
    for row, reference, label in (
        (rows[14], checkpoint_reference, "checkpoint receipt"),
        (rows[15], run_reference, "run receipt"),
    ):
        if any(
            row[field] != reference[field]
            for field in ("uri", "sha256", "bytes", "version_id")
        ):
            raise _fail(
                f"receipt {label} row does not equal its top-level reference"
            )
    return value


def admit_prior_seed_collection(
    payload: bytes,
    *,
    ref: CollectionReceiptRef,
    admitted: AdmittedPriorRun,
    binding: ProviderLifecycleBinding,
) -> AdmittedPriorCollection:
    """Authenticate one prior-seed collection receipt before seed N."""

    if not isinstance(ref, CollectionReceiptRef):
        raise _fail("requires one exact collection receipt reference")
    if not isinstance(admitted, AdmittedPriorRun):
        raise _fail("requires the admitted prior finalization")
    if not isinstance(binding, ProviderLifecycleBinding):
        raise _fail("requires the authenticated provider lifecycle binding")
    if binding.seed < 1:
        raise _fail("seed 0 forbids a prior-seed collection receipt")
    prior_seed = binding.seed - 1
    if admitted.seed != prior_seed:
        raise _fail(
            "admitted prior finalization seed must be the prior "
            "sequential seed"
        )

    value = parse_seed_collection_receipt_bytes(
        payload,
        receipt_uri=ref.uri,
        receipt_sha256=ref.sha256,
        receipt_version_id=ref.version_id,
    )
    if value["seed"] != prior_seed:
        raise _fail("receipt seed must be exactly the prior sequential seed")
    # Boot ID may legitimately differ after a reboot between seeds; every
    # other lifecycle commitment, including the exact instance, must match
    # the authenticated binding, so re-parse against the expected binding
    # with only seed and boot rebound.
    try:
        expected_binding = replace(
            binding,
            seed=prior_seed,
            boot_id=str(value["boot_id"]),
        )
    except (TypeError, ValueError) as error:
        raise _fail("receipt boot identity is invalid") from error
    parse_seed_collection_receipt_bytes(
        payload,
        receipt_uri=ref.uri,
        receipt_sha256=ref.sha256,
        receipt_version_id=ref.version_id,
        expected_binding=expected_binding,
    )

    run_reference = value["run_receipt"]
    receipt_evidence = admitted.evidence[13]
    if (
        run_reference["uri"] != admitted.receipt.uri
        or run_reference["sha256"] != admitted.receipt.sha256
        or run_reference["version_id"] != admitted.receipt.version_id
        or run_reference["bytes"] != receipt_evidence.bytes
    ):
        raise _fail(
            "run receipt reference differs from the admitted prior "
            "finalization"
        )
    rows = value["objects"]
    shared_rows = (*rows[:12], rows[14], rows[15])
    for row, evidence in zip(shared_rows, admitted.evidence, strict=True):
        if (
            row["uri"] != evidence.uri
            or row["sha256"] != evidence.sha256
            or row["version_id"] != evidence.version_id
            or (evidence.bytes is not None and row["bytes"] != evidence.bytes)
        ):
            raise _fail(
                "objects differ from the admitted prior finalization "
                "evidence"
            )
    checkpoints = tuple(
        CollectedObject(
            kind="checkpoint",
            arm=row["arm"],
            step=None,
            uri=row["uri"],
            sha256=row["sha256"],
            bytes=row["bytes"],
            version_id=row["version_id"],
        )
        for row in rows[12:14]
    )
    return AdmittedPriorCollection(
        seed=prior_seed,
        receipt=ref,
        checkpoints=checkpoints,
    )


def download_timeout_seconds(expected_bytes: object) -> float:
    """Return the bounded body-download timeout for one expected size."""

    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise _fail("download size must be a positive exact integer")
    return max(
        _DOWNLOAD_FLOOR_SECONDS,
        2.0 * expected_bytes / _DOWNLOAD_BYTES_PER_SECOND,
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


class SubprocessAwsDownloadRunner:
    """Run explicit argv with an empty environment and a bounded timeout.

    Multi-gigabyte evidence downloads cannot complete inside the fixed
    60-second controller runner, so every body download declares an explicit
    per-object timeout derived from its expected size.
    """

    def run_json(
        self,
        argv: Sequence[str],
        *,
        operation: str,
        timeout_seconds: float,
    ) -> object:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "download timeout must be a positive finite number",
                details={"operation": operation},
            )
        try:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                env={},
                timeout=timeout_seconds,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
            raise MsctlError(
                "EXTERNAL_UNAVAILABLE",
                "AWS CLI is unavailable",
                details={"operation": operation},
            ) from error
        if completed.returncode != 0:
            raise MsctlError(
                "AWS_COMMAND_FAILED",
                "AWS CLI operation failed",
                details={"operation": operation},
            )
        try:
            return json.loads(
                completed.stdout,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "AWS CLI did not return one valid JSON value",
                details={"operation": operation},
            ) from error


__all__ = [
    "COLLECTION_RECEIPT_FIELDS",
    "COLLECTION_RECEIPT_TYPE",
    "SEED_COLLECTION_OBJECT_COUNT",
    "AdmittedPriorCollection",
    "CollectedObject",
    "CollectionError",
    "CollectionReceiptRef",
    "SubprocessAwsDownloadRunner",
    "admit_prior_seed_collection",
    "download_timeout_seconds",
    "parse_seed_collection_receipt_bytes",
]
