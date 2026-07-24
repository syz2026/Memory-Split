#!/usr/bin/env python3
"""Provider-selected clean-run finalization for one Dense/Split90 seed pair.

This module lives under the legacy ``cluster/aws/p5`` code location; that
path carries no provider authority. Every provider, profile, selection,
runtime, and qualification identity is consumed exclusively from the
authenticated provider lifecycle re-admitted through
``admit_provider_lifecycle(...)`` at gates A/B/C.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from cluster.aws.p5.checkpoint_mirror import (
    CheckpointReceiptRef,
    PublishedCheckpointPair,
    VersionedObjectStore,
    VersionedUploadedObject,
    _canonical_json,
    _publish_exact,
    _read_regular,
)
from cluster.aws.qualification import QualificationTimeReader
from msctl.aws_contracts import (
    ARMS,
    COHORT_ID,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    log_object_key,
    run_receipt_key,
    snapshot_object_key,
)
from msctl.aws_lifecycle import (
    LIFECYCLE_BINDING_FIELDS,
    AuthenticatedProviderLifecycle,
    ProviderLifecycleBinding,
    admit_provider_lifecycle,
    lifecycle_operational_metadata,
)
from msctl.contracts import parse_paired_checkpoint_receipt_v3
from msctl.errors import MsctlError
from train.trainer import parse_model_snapshot_bytes


FINALIZATION_ATTEMPT_SECONDS = 1_200
RUN_FINALIZATION_RECEIPT_TYPE = "memorysplit-aws-paired-run-finalization-v3"
TERMINAL_STEP = 13_582
_CHECKPOINT_RECEIPT_TYPE = "memorysplit-aws-paired-checkpoint-v3"
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
_RUN_ID_RE = re.compile(
    r"^memorysplit-v3-360m-s(?P<seed>[0-9])-(?P<arm>dense|split90)$"
)
_PROFILE_PROVIDERS = {
    "aws-p5.48xlarge-v3": "aws-p5.48xlarge",
    "aws-p6-b300.48xlarge-v3": "aws-p6-b300.48xlarge",
}
_SNAPSHOT_NAMES = tuple(f"step{step:07d}.pt" for step in SNAPSHOT_STEPS)
_LOG_REQUIRED_FIELDS = frozenset(
    {
        "epoch",
        "global_tok_s",
        "global_tokens",
        "loss",
        "loss_ema",
        "lr",
        "step",
        "tok_s",
        "tokens_per_step",
    }
)
_LOG_OPTIONAL_FIELDS = frozenset({"loss_masked_values"})
_RECEIPT_FIELDS = frozenset(
    {
        "arms",
        "boot_id",
        "canary_receipt_sha256",
        "checkpoint_receipt",
        "cohort_id",
        "complete",
        "dataset_build_id",
        "dataset_receipt_sha256",
        "environment_receipt_sha256",
        "finalized_at",
        "hardware_amendment_sha256",
        "instance_id",
        "objective_controls_contract_sha256",
        "ordered_stream_sha256",
        "profile_id",
        "profile_sha256",
        "provider",
        "provider_selection_sha256",
        "provider_selection_version_id",
        "qualification_approval_public_key_sha256",
        "qualification_approval_receipt_sha256",
        "qualification_evidence_sha256",
        "receipt_type",
        "release_receipt_sha256",
        "release_sha256",
        "request_id",
        "run_manifest_sha256",
        "runtime_lock_sha256",
        "runtime_sbom_sha256",
        "schema_version",
        "seed",
        "source_commit",
        "source_tree",
    }
)
_ARM_FIELDS = frozenset(
    {
        "arm",
        "config_fingerprint",
        "config_sha256",
        "final_step",
        "log",
        "run_id",
        "snapshots",
        "world_size",
    }
)
_OBJECT_FIELDS = frozenset({"bytes", "sha256", "uri", "version_id"})
_CHECKPOINT_REF_FIELDS = frozenset({"sha256", "uri", "version_id"})
# The receipt intentionally carries only the portable authority commitments;
# placement fields (account/region/zone/purchase) stay in the run manifest
# and checkpoint receipt, which the receipt binds by hash.
_RECEIPT_BINDING_FIELDS = {
    "boot_id": "boot_id",
    "canary_receipt_sha256": "qualification_canary_receipt_sha256",
    "cohort_id": "cohort_id",
    "environment_receipt_sha256": (
        "qualification_environment_receipt_sha256"
    ),
    "hardware_amendment_sha256": "hardware_amendment_sha256",
    "instance_id": "instance_id",
    "objective_controls_contract_sha256": (
        "objective_controls_contract_sha256"
    ),
    "profile_id": "profile_id",
    "profile_sha256": "profile_sha256",
    "provider": "provider",
    "provider_selection_sha256": "provider_selection_sha256",
    "provider_selection_version_id": "provider_selection_version_id",
    "qualification_approval_public_key_sha256": (
        "qualification_approval_public_key_sha256"
    ),
    "qualification_approval_receipt_sha256": (
        "qualification_approval_receipt_sha256"
    ),
    "qualification_evidence_sha256": "qualification_evidence_sha256",
    "runtime_lock_sha256": "runtime_lock_sha256",
    "runtime_sbom_sha256": "runtime_sbom_sha256",
    "seed": "seed",
}


class FinalizationError(ValueError):
    """A fail-closed paired run finalization error."""


@dataclass(frozen=True)
class PublishedRunReceipt:
    uri: str
    sha256: str
    bytes: int
    version_id: str


@dataclass(frozen=True)
class FinalizationResult:
    receipt: PublishedRunReceipt
    provider_selection_sha256: str
    provider_selection_version_id: str
    evidence: tuple[VersionedUploadedObject, ...]


@dataclass(frozen=True)
class _StagedObject:
    key: str
    sha256: str
    bytes: int
    payload: bytes
    metadata: dict[str, str]


@dataclass(frozen=True)
class _ArmEvidence:
    arm: str
    run_id: str
    config_sha256: str
    config_fingerprint: str
    snapshots: tuple[_StagedObject, ...]
    log: _StagedObject


def _fail(message: str) -> FinalizationError:
    return FinalizationError(f"run finalization {message}")


def _sha256_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{label} must be a lowercase SHA-256")
    return value


def _nonnull_version(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value in {"", "null"}:
        raise _fail(f"{label} version ID must be a real non-null version")
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
    except FinalizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise _fail(f"{label} must be one JSON object")
    return parsed


def _pinned_bytes(
    path: Path | str,
    *,
    label: str,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    expected_mode: int | None = None,
) -> bytes:
    try:
        payload, metadata = _read_regular(Path(path), label=label)
    except (OSError, ValueError) as error:
        raise _fail(f"{label} is missing, linked, or drifting") from error
    if any(value is not None for value in (owner_uid, owner_gid, expected_mode)):
        if (
            type(owner_uid) is not int
            or owner_uid < 0
            or type(owner_gid) is not int
            or owner_gid < 0
            or type(expected_mode) is not int
            or metadata.st_uid != owner_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != expected_mode
        ):
            raise _fail(f"{label} owner or mode is not the runtime identity")
    return payload


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


def _finalized_at_text(time_reader: QualificationTimeReader) -> str:
    now = time_reader.now()
    if (
        not isinstance(now, datetime)
        or now.tzinfo is None
        or now.utcoffset() != timedelta(0)
    ):
        raise _fail("time reader must provide aware UTC time")
    text = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if _UTC_RE.fullmatch(text) is None:
        raise _fail("finalized_at is not canonical UTC RFC3339")
    return text


def parse_run_finalization_receipt_bytes(
    payload: bytes,
    *,
    receipt_uri: str,
    receipt_sha256: str,
    receipt_version_id: str,
    expected_binding: ProviderLifecycleBinding | None = None,
) -> dict[str, object]:
    """Parse exact canonical bytes for one paired run-finalization receipt."""

    if not isinstance(payload, bytes) or not payload:
        raise _fail("receipt bytes are missing")
    value = _strict_json(payload, label="receipt")
    if _canonical_json(value) != payload:
        raise _fail("receipt bytes are not canonical")
    expected_sha256 = _sha256_text(receipt_sha256, label="receipt")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise _fail("receipt hash does not match its bytes")
    _nonnull_version(receipt_version_id, label="receipt")
    if set(value) != _RECEIPT_FIELDS:
        raise _fail("receipt fields do not match the closed contract")
    seed = value["seed"]
    if type(seed) is not int or seed not in SEEDS:
        raise _fail("receipt seed must be an exact integer from 0 to 9")
    root = _object_root(
        receipt_uri,
        suffix="/" + run_receipt_key(seed, expected_sha256),
        label="receipt",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 3
        or value["receipt_type"] != RUN_FINALIZATION_RECEIPT_TYPE
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
        not isinstance(value["finalized_at"], str)
        or _UTC_RE.fullmatch(value["finalized_at"]) is None
    ):
        raise _fail("receipt finalized_at must be canonical UTC RFC3339")
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

    checkpoint = value["checkpoint_receipt"]
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != _CHECKPOINT_REF_FIELDS
    ):
        raise _fail("receipt checkpoint binding fields do not match")
    checkpoint_sha256 = _sha256_text(
        checkpoint["sha256"],
        label="receipt checkpoint",
    )
    _nonnull_version(checkpoint["version_id"], label="receipt checkpoint")
    if checkpoint["uri"] != (
        f"{root}/" + checkpoint_receipt_key(seed, checkpoint_sha256)
    ):
        raise _fail("receipt checkpoint URI does not share the object root")

    def require_published_object(
        candidate: object,
        *,
        expected_key: str,
        label: str,
    ) -> None:
        if not isinstance(candidate, dict) or set(candidate) != (
            _OBJECT_FIELDS
        ):
            raise _fail(f"{label} object fields do not match")
        if type(candidate["bytes"]) is not int or candidate["bytes"] <= 0:
            raise _fail(f"{label} object bytes must be a positive integer")
        _nonnull_version(candidate["version_id"], label=label)
        if candidate["uri"] != f"{root}/{expected_key}":
            raise _fail(f"{label} object URI does not share the root")

    arms = value["arms"]
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        raise _fail("receipt arms must contain Dense then Split90")
    for row, arm in zip(arms, ARMS, strict=True):
        if not isinstance(row, dict) or set(row) != _ARM_FIELDS:
            raise _fail(f"receipt {arm} arm fields do not match")
        if row["arm"] != arm:
            raise _fail("receipt arms must contain Dense then Split90")
        run_match = (
            _RUN_ID_RE.fullmatch(row["run_id"])
            if isinstance(row["run_id"], str)
            else None
        )
        if (
            run_match is None
            or int(run_match.group("seed")) != seed
            or run_match.group("arm") != arm
        ):
            raise _fail(f"receipt {arm} run ID is not seed and arm scoped")
        if (
            type(row["final_step"]) is not int
            or row["final_step"] != TERMINAL_STEP
            or type(row["world_size"]) is not int
            or row["world_size"] != 4
        ):
            raise _fail(f"receipt {arm} arm is not a terminal world-size-4 run")
        _sha256_text(row["config_sha256"], label=f"receipt {arm} config")
        _sha256_text(
            row["config_fingerprint"],
            label=f"receipt {arm} config fingerprint",
        )

        log = row["log"]
        if not isinstance(log, dict) or set(log) != _OBJECT_FIELDS:
            raise _fail(f"receipt {arm} log object fields do not match")
        log_sha256 = _sha256_text(log["sha256"], label=f"receipt {arm} log")
        require_published_object(
            log,
            expected_key=log_object_key(seed, arm, log_sha256),
            label=f"receipt {arm} log",
        )
        snapshots = row["snapshots"]
        if not isinstance(snapshots, list) or len(snapshots) != len(
            SNAPSHOT_STEPS
        ):
            raise _fail(f"receipt {arm} snapshots must be the frozen schedule")
        for item, step in zip(snapshots, SNAPSHOT_STEPS, strict=True):
            if not isinstance(item, dict) or set(item) != {"object", "step"}:
                raise _fail(f"receipt {arm} snapshot fields do not match")
            if type(item["step"]) is not int or item["step"] != step:
                raise _fail(
                    f"receipt {arm} snapshots must follow the frozen schedule"
                )
            snapshot_object = item["object"]
            if not isinstance(snapshot_object, dict) or set(
                snapshot_object
            ) != _OBJECT_FIELDS:
                raise _fail(
                    f"receipt {arm} snapshot object fields do not match"
                )
            snapshot_sha256 = _sha256_text(
                snapshot_object["sha256"],
                label=f"receipt {arm} snapshot step {step}",
            )
            require_published_object(
                snapshot_object,
                expected_key=snapshot_object_key(
                    seed,
                    arm,
                    step,
                    snapshot_sha256,
                ),
                label=f"receipt {arm} snapshot step {step}",
            )
    return value


def _validate_log_bytes(
    payload: bytes,
    *,
    arm: str,
    expected_tokens_per_step: int,
) -> None:
    label = f"{arm} training log"
    if (
        type(expected_tokens_per_step) is not int
        or expected_tokens_per_step <= 0
    ):
        raise _fail(f"{label} expected token geometry is invalid")
    if not isinstance(payload, bytes) or not payload:
        raise _fail(f"{label} is empty")
    if not payload.endswith(b"\n"):
        raise _fail(f"{label} is truncated")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail(f"{label} is not valid UTF-8") from error
    last_step: int | None = None
    for index, line in enumerate(text.split("\n")[:-1]):
        row = _strict_json(
            line.encode("utf-8"),
            label=f"{label} row {index}",
        )
        fields = set(row)
        if not (
            _LOG_REQUIRED_FIELDS
            <= fields
            <= _LOG_REQUIRED_FIELDS | _LOG_OPTIONAL_FIELDS
        ):
            raise _fail(f"{label} row {index} fields are foreign")
        if any(
            isinstance(item, float) and not math.isfinite(item)
            for item in row.values()
        ):
            raise _fail(f"{label} row {index} is not finite")
        step = row["step"]
        if type(step) is not int or step <= 0 or step > TERMINAL_STEP:
            raise _fail(f"{label} row {index} step is invalid or post-terminal")
        for field_name in ("epoch", "global_tokens", "tokens_per_step"):
            value = row[field_name]
            if type(value) is not int or value < 0:
                raise _fail(f"{label} row {index} {field_name} is invalid")
        if (
            row["tokens_per_step"] != expected_tokens_per_step
            or row["global_tokens"] != step * expected_tokens_per_step
        ):
            raise _fail(f"{label} row {index} token counts are inconsistent")
        for field_name in (
            "global_tok_s",
            "loss",
            "loss_ema",
            "lr",
            "tok_s",
            *(
                ("loss_masked_values",)
                if "loss_masked_values" in row
                else ()
            ),
        ):
            value = row[field_name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise _fail(f"{label} row {index} {field_name} is invalid")
        if last_step is not None and step < last_step:
            raise _fail(f"{label} steps are not monotonically nondecreasing")
        last_step = step
    if last_step != TERMINAL_STEP:
        raise _fail(f"{label} is missing its terminal completion record")


def _verified_digest(plan: object, path: Path, *, label: str) -> str:
    for item in getattr(plan, "verified_files", ()):
        if Path(item.path) == path:
            return item.sha256
    raise _fail(f"{label} was not admitted by the reviewed launch plan")


def _require_manifest_binding(
    plan: object,
    binding: ProviderLifecycleBinding,
) -> None:
    manifest_path = Path(plan.manifest_path)
    payload = _pinned_bytes(manifest_path, label="run manifest")
    if hashlib.sha256(payload).hexdigest() != _verified_digest(
        plan,
        manifest_path,
        label="run manifest",
    ):
        raise _fail("run manifest drifted after launch admission")
    value = _strict_json(payload, label="run manifest")
    expected = binding.to_dict()
    if any(
        value.get(field) != expected[field]
        for field in LIFECYCLE_BINDING_FIELDS
    ):
        raise _fail("run manifest differs from the provider lifecycle")


def _require_release_binding(
    plan: object,
    binding: ProviderLifecycleBinding,
) -> None:
    metadata_path = Path(plan.repo_root) / "RELEASE-METADATA.json"
    payload = _pinned_bytes(metadata_path, label="release metadata")
    if hashlib.sha256(payload).hexdigest() != _verified_digest(
        plan,
        metadata_path,
        label="release metadata",
    ):
        raise _fail("release metadata drifted after launch admission")
    value = _strict_json(payload, label="release metadata")
    expected = binding.to_dict()
    if value.get("profile_id") != binding.profile_id or any(
        value.get(field) != expected[field]
        for field in LIFECYCLE_BINDING_FIELDS
    ):
        raise _fail("release metadata differs from the provider lifecycle")


def _require_plan_configs(plan: object) -> None:
    for launch in plan.arms:
        payload = _pinned_bytes(
            launch.config_path,
            label=f"{launch.arm} launch config",
        )
        if hashlib.sha256(payload).hexdigest() != launch.config_sha256:
            raise _fail(
                f"{launch.arm} launch config drifted after plan admission"
            )


def _expected_operational_metadata(
    plan: object,
    launch: object,
    binding: ProviderLifecycleBinding,
) -> dict[str, object]:
    try:
        return lifecycle_operational_metadata(
            binding,
            run_id=str(launch.runtime_config["run_id"]),
            arm=launch.arm,
            config_sha256=launch.config_sha256,
            dataset_receipt_sha256=plan.dataset_receipt_sha256,
            dataset_build_id=plan.dataset_build_id,
            ordered_stream_sha256=plan.ordered_stream_sha256,
            source_commit=plan.code_commit,
            source_tree=str(plan.source_tree),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _fail(
            f"{launch.arm} launch plan cannot derive operational metadata"
        ) from error


def _validated_checkpoint_receipt(
    plan: object,
    binding: ProviderLifecycleBinding,
    checkpoint_pair: PublishedCheckpointPair,
    *,
    object_store: VersionedObjectStore,
    s3_root: str,
    seed: int,
) -> dict[str, str]:
    if not isinstance(checkpoint_pair, PublishedCheckpointPair) or not (
        isinstance(checkpoint_pair.receipt, CheckpointReceiptRef)
    ):
        raise _fail(
            "requires the latest complete paired checkpoint receipt"
        )
    reference = checkpoint_pair.receipt
    payload = _canonical_json(checkpoint_pair.value)
    if (
        hashlib.sha256(payload).hexdigest() != reference.sha256
        or len(payload) != reference.bytes
    ):
        raise _fail("checkpoint receipt bytes do not match its reference")
    if reference.uri != (
        f"{s3_root}/" + checkpoint_receipt_key(seed, reference.sha256)
    ):
        raise _fail("checkpoint receipt URI is outside the durable root")
    try:
        receipt = parse_paired_checkpoint_receipt_v3(
            payload,
            receipt_uri=reference.uri,
            receipt_sha256=reference.sha256,
            receipt_version_id=reference.version_id,
        )
    except MsctlError as error:
        raise _fail("checkpoint receipt is invalid") from error
    if receipt.provider_selection_sha256 is None:
        raise _fail("requires a provider-aware checkpoint receipt")
    expected = binding.to_dict()
    for field in LIFECYCLE_BINDING_FIELDS:
        actual = getattr(receipt, field)
        if field == "arms":
            actual = list(actual)
        if actual != expected[field]:
            raise _fail(
                "checkpoint receipt selection differs from the admitted "
                "provider lifecycle"
            )
    if (
        receipt.seed != plan.seed
        or receipt.release_sha256 != plan.release_sha256
        or receipt.release_receipt_sha256 != plan.release_receipt_sha256
        or receipt.run_manifest_sha256 != plan.run_manifest_sha256
        or receipt.dataset_receipt_sha256 != plan.dataset_receipt_sha256
        or receipt.dataset_build_id != plan.dataset_build_id
        or receipt.ordered_stream_sha256 != plan.ordered_stream_sha256
        or receipt.source_commit != plan.code_commit
        or receipt.source_tree != plan.source_tree
        or receipt.instance_id != plan.instance_id
        or receipt.boot_id != plan.boot_id
        or receipt.environment_receipt_sha256
        != plan.environment_receipt_sha256
    ):
        raise _fail(
            "checkpoint receipt does not bind the reviewed launch plan"
        )
    launches = {launch.arm: launch for launch in plan.arms}
    fingerprints: dict[str, str] = {}
    for checkpoint, arm in zip(receipt.checkpoints, ARMS, strict=True):
        launch = launches[arm]
        if (
            checkpoint.run_id != str(launch.runtime_config["run_id"])
            or checkpoint.config_sha256 != launch.config_sha256
        ):
            raise _fail(
                "checkpoint receipt arm rows do not bind the launch plan"
            )
        fingerprints[arm] = checkpoint.config_fingerprint
    try:
        verified = object_store.head_exact(
            reference.uri,
            sha256=reference.sha256,
            byte_count=reference.bytes,
            metadata={
                "receipt-sha256": reference.sha256,
                "receipt-type": _CHECKPOINT_RECEIPT_TYPE,
                "request-id": receipt.request_id,
                "seed": str(seed),
            },
            version_id=reference.version_id,
        )
    except Exception as error:
        raise _fail("checkpoint receipt object is unverified") from error
    if verified is None or verified.version_id != reference.version_id:
        raise _fail("checkpoint receipt object is unverified")
    return fingerprints


def _admit_arm_evidence(
    plan: object,
    launch: object,
    *,
    seed: int,
    binding: ProviderLifecycleBinding,
    expected_fingerprint: str,
) -> _ArmEvidence:
    arm = launch.arm
    expected_metadata = _expected_operational_metadata(plan, launch, binding)
    run_dir = Path(launch.checkpoint_path).parent
    snapshots_dir = run_dir / "snapshots"
    try:
        directory = os.lstat(snapshots_dir)
    except OSError as error:
        raise _fail(f"{arm} snapshots directory is unavailable") from error
    if not stat.S_ISDIR(directory.st_mode):
        raise _fail(f"{arm} snapshots directory is not a real directory")
    published = sorted(
        name for name in os.listdir(snapshots_dir) if name.endswith(".pt")
    )
    if published != sorted(_SNAPSHOT_NAMES):
        raise _fail(
            f"{arm} snapshots are not exactly the frozen five-step schedule"
        )
    staged: list[_StagedObject] = []
    fingerprint: str | None = None
    for step, name in zip(SNAPSHOT_STEPS, _SNAPSHOT_NAMES, strict=True):
        payload = _pinned_bytes(
            snapshots_dir / name,
            label=f"{arm} snapshot step {step}",
            owner_uid=plan.runtime_uid,
            owner_gid=plan.runtime_gid,
            expected_mode=0o600,
        )
        try:
            state = parse_model_snapshot_bytes(
                payload,
                expected_operational_metadata=expected_metadata,
                require_study_identity=True,
            )
        except ValueError as error:
            raise _fail(f"{arm} snapshot step {step} is invalid") from error
        if state["step"] != step:
            raise _fail(
                f"{arm} snapshot step {step} does not match its schedule slot"
            )
        if state["world_size"] != 4:
            raise _fail(f"{arm} snapshot step {step} world size is not four")
        observed = state["config_fingerprint"]
        fingerprint = observed if fingerprint is None else fingerprint
        if observed != fingerprint or observed != expected_fingerprint:
            raise _fail(
                f"{arm} snapshot config fingerprint differs from the "
                "admitted fingerprint"
            )
        digest = hashlib.sha256(payload).hexdigest()
        staged.append(
            _StagedObject(
                key=snapshot_object_key(seed, arm, step, digest),
                sha256=digest,
                bytes=len(payload),
                payload=payload,
                metadata=_snapshot_metadata(
                    plan,
                    binding,
                    arm=arm,
                    run_id=str(launch.runtime_config["run_id"]),
                    config_sha256=launch.config_sha256,
                    config_fingerprint=expected_fingerprint,
                    step=step,
                    sha256=digest,
                ),
            )
        )
    log_payload = _pinned_bytes(
        run_dir / "log.jsonl",
        label=f"{arm} training log",
        owner_uid=plan.runtime_uid,
        owner_gid=plan.runtime_gid,
        expected_mode=0o600,
    )
    _validate_log_bytes(
        log_payload,
        arm=arm,
        expected_tokens_per_step=launch.runtime_config["tokens_per_step"],
    )
    log_digest = hashlib.sha256(log_payload).hexdigest()
    log = _StagedObject(
        key=log_object_key(seed, arm, log_digest),
        sha256=log_digest,
        bytes=len(log_payload),
        payload=log_payload,
        metadata=_log_metadata(
            plan,
            binding,
            arm=arm,
            run_id=str(launch.runtime_config["run_id"]),
            config_sha256=launch.config_sha256,
            sha256=log_digest,
        ),
    )
    assert fingerprint is not None
    return _ArmEvidence(
        arm=arm,
        run_id=str(launch.runtime_config["run_id"]),
        config_sha256=launch.config_sha256,
        config_fingerprint=fingerprint,
        snapshots=tuple(staged),
        log=log,
    )


def _lifecycle_object_metadata(
    plan: object,
    binding: ProviderLifecycleBinding,
) -> dict[str, str]:
    lifecycle = binding.to_dict()
    if set(lifecycle) != set(LIFECYCLE_BINDING_FIELDS):
        raise _fail("provider lifecycle binding fields are not exact")
    return {
        **{
            field.replace("_", "-"): (
                ",".join(value) if field == "arms" else str(value)
            )
            for field, value in lifecycle.items()
        },
        "data-build-id": plan.dataset_build_id,
        "data-receipt-sha256": plan.dataset_receipt_sha256,
        "ordered-stream-sha256": plan.ordered_stream_sha256,
    }


def _snapshot_metadata(
    plan: object,
    binding: ProviderLifecycleBinding,
    *,
    arm: str,
    run_id: str,
    config_sha256: str,
    config_fingerprint: str,
    step: int,
    sha256: str,
) -> dict[str, str]:
    return {
        **_lifecycle_object_metadata(plan, binding),
        "arm": arm,
        "config-fingerprint": config_fingerprint,
        "config-sha256": config_sha256,
        "run-id": run_id,
        "sha256": sha256,
        "step": str(step),
        "world-size": "4",
    }


def _log_metadata(
    plan: object,
    binding: ProviderLifecycleBinding,
    *,
    arm: str,
    run_id: str,
    config_sha256: str,
    sha256: str,
) -> dict[str, str]:
    return {
        **_lifecycle_object_metadata(plan, binding),
        "arm": arm,
        "config-sha256": config_sha256,
        "final-step": str(TERMINAL_STEP),
        "run-id": run_id,
        "sha256": sha256,
    }


def _stage_payload(staging: Path, name: str, payload: bytes) -> Path:
    path = staging / name
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o400)
    return path


def _publish(
    object_store: VersionedObjectStore,
    path: Path,
    uri: str,
    *,
    sha256: str,
    byte_count: int,
    metadata: Mapping[str, str],
    label: str,
) -> VersionedUploadedObject:
    try:
        return _publish_exact(
            object_store,
            path,
            uri,
            sha256=sha256,
            byte_count=byte_count,
            metadata=metadata,
        )
    except (OSError, ValueError) as error:
        raise _fail(f"{label} publication did not verify") from error


def finalize_paired_run(
    plan: object,
    *,
    checkpoint_pair: PublishedCheckpointPair | None,
    object_store: VersionedObjectStore,
    s3_root: str,
    seed: int,
    request_id: str,
    time_reader: QualificationTimeReader,
    staging_root: Path | str,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    runtime_sbom_path: Path | str,
    objective_controls_amendment_path: Path | str,
    store: object,
    account_id: str,
    instance_id: str,
    boot_id: str,
    expected_selection_version_id: str,
    identity_verifier: object,
    approval_verifier: object,
    trusted_public_key_sha256: str,
) -> FinalizationResult:
    """Publish twelve verified evidence objects and one paired run receipt."""

    deadline = time_reader.monotonic() + FINALIZATION_ATTEMPT_SECONDS

    def guard(stage: str) -> None:
        if time_reader.monotonic() >= deadline:
            raise _fail(f"deadline expired during {stage}")

    binding = getattr(plan, "lifecycle_binding", None)
    if not isinstance(binding, ProviderLifecycleBinding):
        raise _fail("requires an authenticated selected launch plan")
    if (
        type(seed) is not int
        or seed not in SEEDS
        or seed != getattr(plan, "seed", None)
        or seed != binding.seed
    ):
        raise _fail("seed does not match the selected launch plan")
    if (
        not isinstance(request_id, str)
        or _REQUEST_ID_RE.fullmatch(request_id) is None
    ):
        raise _fail("request ID must be 32 lowercase hex")
    runtime = getattr(plan, "runtime", None)
    if (
        not isinstance(s3_root, str)
        or s3_root != getattr(runtime, "s3_root", None)
    ):
        raise _fail("S3 root differs from the reviewed launch plan")
    for field in (
        "release_sha256",
        "release_receipt_sha256",
        "run_manifest_sha256",
        "environment_receipt_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
        "code_commit",
        "source_tree",
        "instance_id",
        "boot_id",
        "manifest_path",
        "repo_root",
    ):
        if getattr(plan, field, None) is None:
            raise _fail("launch plan context is incomplete")
    if (
        binding.instance_id != plan.instance_id
        or binding.boot_id != plan.boot_id
    ):
        raise _fail("launch plan instance identity differs from the lifecycle")
    launches = {launch.arm: launch for launch in plan.arms}
    if set(launches) != set(ARMS):
        raise _fail("launch plan does not contain the paired arms")

    authority = {
        "authority_root": authority_root,
        "repo_root": repo_root,
        "runtime_lock_path": runtime_lock_path,
        "runtime_evidence_path": runtime_evidence_path,
        "runtime_sbom_path": runtime_sbom_path,
        "objective_controls_amendment_path": (
            objective_controls_amendment_path
        ),
        "store": store,
        "account_id": account_id,
        "instance_id": instance_id,
        "boot_id": boot_id,
        "seed": seed,
        "expected_selection_version_id": expected_selection_version_id,
        "identity_verifier": identity_verifier,
        "approval_verifier": approval_verifier,
        "trusted_public_key_sha256": trusted_public_key_sha256,
    }

    def gate(stage: str) -> None:
        guard(stage)
        try:
            admitted = admit_provider_lifecycle(**authority)
        except (TypeError, ValueError, OSError) as error:
            raise _fail(
                f"provider authority admission failed at {stage}"
            ) from error
        if (
            not isinstance(admitted, AuthenticatedProviderLifecycle)
            or admitted.binding != binding
        ):
            raise _fail(f"provider authority drifted at {stage}")
        _require_manifest_binding(plan, binding)
        _require_release_binding(plan, binding)
        _require_plan_configs(plan)

    gate("gate A local evidence admission")
    if checkpoint_pair is None:
        raise _fail("requires the latest complete paired checkpoint receipt")
    fingerprints = _validated_checkpoint_receipt(
        plan,
        binding,
        checkpoint_pair,
        object_store=object_store,
        s3_root=s3_root,
        seed=seed,
    )
    evidence_by_arm = tuple(
        _admit_arm_evidence(
            plan,
            launches[arm],
            seed=seed,
            binding=binding,
            expected_fingerprint=fingerprints[arm],
        )
        for arm in ARMS
    )

    staging_parent = Path(staging_root)
    staging_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(staging_parent, 0o700)
    with tempfile.TemporaryDirectory(
        prefix=f"finalize-{request_id}-",
        dir=staging_parent,
    ) as staging_text:
        staging = Path(staging_text)
        staged_paths: dict[str, Path] = {}
        for evidence in evidence_by_arm:
            for staged in evidence.snapshots:
                staged_paths[staged.key] = _stage_payload(
                    staging,
                    f"{evidence.arm}-{staged.sha256}.pt",
                    staged.payload,
                )
            staged_paths[evidence.log.key] = _stage_payload(
                staging,
                f"{evidence.arm}-{evidence.log.sha256}.jsonl",
                evidence.log.payload,
            )

        gate("gate B evidence publication")
        published: list[VersionedUploadedObject] = []
        published_by_key: dict[str, VersionedUploadedObject] = {}
        for evidence in evidence_by_arm:
            for staged in (*evidence.snapshots, evidence.log):
                guard("evidence publication")
                uploaded = _publish(
                    object_store,
                    staged_paths[staged.key],
                    f"{s3_root}/{staged.key}",
                    sha256=staged.sha256,
                    byte_count=staged.bytes,
                    metadata=staged.metadata,
                    label=f"{evidence.arm} evidence",
                )
                published.append(uploaded)
                published_by_key[staged.key] = uploaded
        if len(published) != 12 or len(published_by_key) != 12:
            raise _fail("publication did not verify all twelve objects")

        gate("gate C receipt publication")
        finalized_at = _finalized_at_text(time_reader)
        arm_rows = []
        for evidence in evidence_by_arm:
            log_object = published_by_key[evidence.log.key]
            arm_rows.append(
                {
                    "arm": evidence.arm,
                    "config_fingerprint": evidence.config_fingerprint,
                    "config_sha256": evidence.config_sha256,
                    "final_step": TERMINAL_STEP,
                    "log": {
                        "bytes": log_object.bytes,
                        "sha256": log_object.sha256,
                        "uri": log_object.uri,
                        "version_id": log_object.version_id,
                    },
                    "run_id": evidence.run_id,
                    "snapshots": [
                        {
                            "object": {
                                "bytes": uploaded.bytes,
                                "sha256": uploaded.sha256,
                                "uri": uploaded.uri,
                                "version_id": uploaded.version_id,
                            },
                            "step": step,
                        }
                        for step, uploaded in zip(
                            SNAPSHOT_STEPS,
                            (
                                published_by_key[staged.key]
                                for staged in evidence.snapshots
                            ),
                            strict=True,
                        )
                    ],
                    "world_size": 4,
                }
            )
        receipt_value = {
            "arms": arm_rows,
            "boot_id": binding.boot_id,
            "canary_receipt_sha256": (
                binding.qualification_canary_receipt_sha256
            ),
            "checkpoint_receipt": {
                "sha256": checkpoint_pair.receipt.sha256,
                "uri": checkpoint_pair.receipt.uri,
                "version_id": checkpoint_pair.receipt.version_id,
            },
            "cohort_id": COHORT_ID,
            "complete": True,
            "dataset_build_id": plan.dataset_build_id,
            "dataset_receipt_sha256": plan.dataset_receipt_sha256,
            "environment_receipt_sha256": (
                binding.qualification_environment_receipt_sha256
            ),
            "finalized_at": finalized_at,
            "hardware_amendment_sha256": (
                binding.hardware_amendment_sha256
            ),
            "instance_id": binding.instance_id,
            "objective_controls_contract_sha256": (
                binding.objective_controls_contract_sha256
            ),
            "ordered_stream_sha256": plan.ordered_stream_sha256,
            "profile_id": binding.profile_id,
            "profile_sha256": binding.profile_sha256,
            "provider": binding.provider,
            "provider_selection_sha256": (
                binding.provider_selection_sha256
            ),
            "provider_selection_version_id": (
                binding.provider_selection_version_id
            ),
            "qualification_approval_public_key_sha256": (
                binding.qualification_approval_public_key_sha256
            ),
            "qualification_approval_receipt_sha256": (
                binding.qualification_approval_receipt_sha256
            ),
            "qualification_evidence_sha256": (
                binding.qualification_evidence_sha256
            ),
            "receipt_type": RUN_FINALIZATION_RECEIPT_TYPE,
            "release_receipt_sha256": plan.release_receipt_sha256,
            "release_sha256": plan.release_sha256,
            "request_id": request_id,
            "run_manifest_sha256": plan.run_manifest_sha256,
            "runtime_lock_sha256": binding.runtime_lock_sha256,
            "runtime_sbom_sha256": binding.runtime_sbom_sha256,
            "schema_version": 3,
            "seed": seed,
            "source_commit": plan.code_commit,
            "source_tree": plan.source_tree,
        }
        receipt_bytes = _canonical_json(receipt_value)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_uri = f"{s3_root}/" + run_receipt_key(seed, receipt_sha256)
        parse_run_finalization_receipt_bytes(
            receipt_bytes,
            receipt_uri=receipt_uri,
            receipt_sha256=receipt_sha256,
            receipt_version_id="unpublished",
            expected_binding=binding,
        )
        receipt_path = _stage_payload(
            staging,
            f"{receipt_sha256}.json",
            receipt_bytes,
        )
        guard("receipt publication")
        receipt_object = _publish(
            object_store,
            receipt_path,
            receipt_uri,
            sha256=receipt_sha256,
            byte_count=len(receipt_bytes),
            metadata={
                "receipt-sha256": receipt_sha256,
                "receipt-type": RUN_FINALIZATION_RECEIPT_TYPE,
                "request-id": request_id,
                "seed": str(seed),
            },
            label="run receipt",
        )
    return FinalizationResult(
        receipt=PublishedRunReceipt(
            uri=receipt_object.uri,
            sha256=receipt_object.sha256,
            bytes=receipt_object.bytes,
            version_id=receipt_object.version_id,
        ),
        provider_selection_sha256=binding.provider_selection_sha256,
        provider_selection_version_id=(
            binding.provider_selection_version_id
        ),
        evidence=tuple(published),
    )


__all__ = [
    "FINALIZATION_ATTEMPT_SECONDS",
    "FinalizationError",
    "FinalizationResult",
    "PublishedRunReceipt",
    "RUN_FINALIZATION_RECEIPT_TYPE",
    "TERMINAL_STEP",
    "finalize_paired_run",
    "parse_run_finalization_receipt_bytes",
]
