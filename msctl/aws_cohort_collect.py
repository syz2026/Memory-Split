"""Durable cohort evaluation-evidence collection contracts (Task 4D).

One cohort collection downloads, independently stream-rehashes, and
receipts exactly 1,012 objects: the 100 sealed-evaluation outputs (ten
per-file members each, in frozen slot order), the ten Task 3F per-seed
collection receipts, the published StudyLockV3, and the published cohort
report. Every object is bound by exact S3 version ID at its slot-scoped
content-addressed key.

Evidence bodies are opaque to collection: outcomes are certified solely by
the full Task 4C report replay, which the controller imports
function-locally. This module must never import ``torch``,
``train.trainer``, or ``evals.confirmatory``; the closed member tuple is
mirror-pinned by tests instead.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

from cluster.aws.p5.checkpoint_mirror import _canonical_json

from .aws_contracts import (
    ARMS,
    COHORT_ID,
    EVALUATION_OUTPUT_MEMBERS,
    SEEDS,
    SNAPSHOT_STEPS,
    cohort_collection_receipt_key,
    cohort_report_object_key,
    collection_receipt_key,
    evaluation_output_member_key,
    study_lock_object_key,
)
from .aws_lifecycle import ProviderLifecycleBinding


COHORT_COLLECTION_RECEIPT_TYPE = "memorysplit-aws-cohort-collection-v3"
COHORT_COLLECTION_OBJECT_COUNT = 1_012
# Exactly the frozen Task 3F COLLECTION_RECEIPT_FIELDS (itself derived
# from the Task 3D finalization fields) minus the per-seed receipt
# references, plus the three cohort anchors and the two sealed-evaluation
# commitments. Enumerated literally because the Task 3F module transitively
# imports torch at module scope; the derivation equality is pinned by a
# mirror test so any upstream drift fails the suite.
COHORT_COLLECTION_RECEIPT_FIELDS = frozenset(
    {
        "boot_id",
        "canary_receipt_sha256",
        "cohort_id",
        "cohort_report",
        "collected_at",
        "complete",
        "dataset_build_id",
        "dataset_receipt_sha256",
        "environment_receipt_sha256",
        "hardware_amendment_sha256",
        "instance_id",
        "objective_controls_contract_sha256",
        "objects",
        "ordered_stream_sha256",
        "preregistration_sha256",
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
        "sealed_evaluation_release_sha256",
        "seed",
        "seed_collections",
        "source_commit",
        "source_tree",
        "study_lock",
    }
)
_TERMINAL_SEED = SEEDS[-1]
# The shape patterns and the closed profile-provider mapping mirror the
# frozen Task 3F module exactly (mirror-pinned by tests); they are
# duplicated here only because importing that module would pull torch in
# through the frozen Task 3D finalization chain.
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
# Mirror-pinned against evals.confirmatory.aggregate._MAX_OUTPUT_FILE_BYTES
# by tests; enforced at index-parse time so an oversize declaration fails
# before any download.
_MAX_EVIDENCE_OBJECT_BYTES = 2 * 1024 * 1024 * 1024
_OBJECT_KINDS = (
    "output_member",
    "seed_collection",
    "study_lock",
    "cohort_report",
)
_OBJECT_ROW_FIELDS = frozenset(
    {
        "arm",
        "bytes",
        "kind",
        "member",
        "seed",
        "sha256",
        "step",
        "uri",
        "version_id",
    }
)
_REFERENCE_FIELDS = frozenset({"bytes", "sha256", "uri", "version_id"})
_SEED_ROW_FIELDS = frozenset({"bytes", "seed", "sha256", "uri", "version_id"})
_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "cohort_id",
        "study_lock",
        "cohort_report",
        "seed_collections",
        "outputs",
    }
)
_INDEX_OUTPUT_FIELDS = frozenset(
    {"slot_index", "seed", "arm", "optimizer_step", "output_id", "members"}
)
_INDEX_MEMBER_FIELDS = frozenset(
    {"member", "uri", "sha256", "version_id", "bytes"}
)
_SHA256_FIELDS = (
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
    "sealed_evaluation_release_sha256",
    "preregistration_sha256",
)
_OUTPUT_ID_RE = re.compile(
    r"^snapshot-evaluation-memorysplit-v3-(?P<profile>.+)-"
    r"s(?P<seed>[0-9]{2})-(?P<arm>dense|split90)-step(?P<step>[0-9]{5})$"
)


def _expected_slot_plan() -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (seed, arm, step)
        for seed in SEEDS
        for arm in ARMS
        for step in SNAPSHOT_STEPS
    )


def _expected_row_plan() -> tuple[
    tuple[str, int | None, str | None, int | None, str | None],
    ...,
]:
    plan: list[
        tuple[str, int | None, str | None, int | None, str | None]
    ] = []
    for seed, arm, step in _SLOT_PLAN:
        plan.extend(
            ("output_member", seed, arm, step, member)
            for member in EVALUATION_OUTPUT_MEMBERS
        )
    plan.extend(("seed_collection", seed, None, None, None) for seed in SEEDS)
    plan.append(("study_lock", None, None, None, None))
    plan.append(("cohort_report", None, None, None, None))
    return tuple(plan)


_SLOT_PLAN = _expected_slot_plan()
_ROW_PLAN = _expected_row_plan()
assert len(_ROW_PLAN) == COHORT_COLLECTION_OBJECT_COUNT


class CohortCollectionError(ValueError):
    """A fail-closed cohort evidence collection error."""


def _fail(message: str) -> CohortCollectionError:
    return CohortCollectionError(f"cohort collection {message}")


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


def _bounded_bytes(value: object, *, label: str) -> int:
    size = _positive_bytes(value, label=label)
    if size > _MAX_EVIDENCE_OBJECT_BYTES:
        raise _fail(f"{label} bytes exceed the sealed member ceiling")
    return size


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


def _safe_s3_root(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{label} root must be a non-empty string")
    root = value.rstrip("/")
    try:
        parsed = urlsplit(root)
        port = parsed.port
    except ValueError as error:
        raise _fail(f"{label} root is invalid") from error
    if (
        parsed.scheme != "s3"
        or parsed.hostname is None
        or parsed.netloc != parsed.hostname
        or port is not None
        or parsed.query
        or parsed.fragment
        or _BUCKET_RE.fullmatch(parsed.hostname) is None
        or "\\" in root
        or any(
            part in {"", ".", ".."}
            for part in parsed.path.removeprefix("/").split("/")
            if parsed.path not in {"", "/"}
        )
    ):
        raise _fail(f"{label} root is not a safe s3:// root")
    return root


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
    except CohortCollectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise _fail(f"{label} must be one JSON object")
    return parsed


def _object_root(receipt_uri: object, *, suffix: str, label: str) -> str:
    if not isinstance(receipt_uri, str) or not receipt_uri.endswith(suffix):
        raise _fail(f"{label} URI does not use its canonical key")
    return _safe_s3_root(receipt_uri[: -len(suffix)], label=label)


def _cohort_row_key(
    kind: str,
    seed: int | None,
    arm: str | None,
    step: int | None,
    member: str | None,
    sha256: str,
) -> str:
    try:
        if kind == "output_member":
            return evaluation_output_member_key(
                seed,
                arm,
                step,
                member,
                sha256,
            )
        if kind == "seed_collection":
            return collection_receipt_key(seed, sha256)
        if kind == "study_lock":
            return study_lock_object_key(sha256)
        return cohort_report_object_key(sha256)
    except ValueError as error:
        raise _fail(f"{kind} object key is invalid") from error


@dataclass(frozen=True)
class CohortEvidenceRef:
    """Exact versioned identity of one durable cohort evidence object."""

    uri: str
    sha256: str
    version_id: str
    bytes: int

    def __post_init__(self) -> None:
        _s3_uri(self.uri, label="cohort evidence reference")
        _sha256_text(self.sha256, label="cohort evidence reference")
        _nonnull_version(
            self.version_id,
            label="cohort evidence reference",
        )
        _positive_bytes(self.bytes, label="cohort evidence reference")


@dataclass(frozen=True)
class CohortCollectedObject:
    """One canonical collected cohort evidence object row."""

    kind: str
    seed: int | None
    arm: str | None
    step: int | None
    member: str | None
    uri: str
    sha256: str
    bytes: int
    version_id: str

    def __post_init__(self) -> None:
        if self.kind not in _OBJECT_KINDS:
            raise _fail("collected object kind is not in the closed set")
        if self.kind == "output_member":
            if type(self.seed) is not int or self.seed not in SEEDS:
                raise _fail("collected member seed must be 0 through 9")
            if self.arm not in ARMS:
                raise _fail("collected member arm must be dense or split90")
            if type(self.step) is not int or self.step not in SNAPSHOT_STEPS:
                raise _fail(
                    "collected member step must be in the frozen schedule"
                )
            if self.member not in EVALUATION_OUTPUT_MEMBERS:
                raise _fail(
                    "collected member must be in the closed ten-name set"
                )
        elif self.kind == "seed_collection":
            if type(self.seed) is not int or self.seed not in SEEDS:
                raise _fail("collected receipt seed must be 0 through 9")
            if (
                self.arm is not None
                or self.step is not None
                or self.member is not None
            ):
                raise _fail(
                    "collected receipt rows carry only their seed identity"
                )
        else:
            if (
                self.seed is not None
                or self.arm is not None
                or self.step is not None
                or self.member is not None
            ):
                raise _fail(
                    "collected anchor rows must not carry slot identity"
                )
        _s3_uri(self.uri, label="collected object")
        _sha256_text(self.sha256, label="collected object")
        _positive_bytes(self.bytes, label="collected object")
        _nonnull_version(self.version_id, label="collected object")


def _evidence_reference(
    candidate: object,
    *,
    root: str,
    key_for,
    label: str,
    bounded: bool = False,
) -> dict[str, object]:
    if not isinstance(candidate, Mapping) or set(candidate) != (
        _REFERENCE_FIELDS
    ):
        raise _fail(f"{label} reference fields do not match")
    digest = _sha256_text(candidate["sha256"], label=label)
    _nonnull_version(candidate["version_id"], label=label)
    if bounded:
        _bounded_bytes(candidate["bytes"], label=label)
    else:
        _positive_bytes(candidate["bytes"], label=label)
    if candidate["uri"] != f"{root}/" + key_for(digest):
        raise _fail(f"{label} URI does not use its canonical key")
    return dict(candidate)


def _seed_collection_rows(
    candidate: object,
    *,
    root: str,
    label: str,
    bounded: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(candidate, list) or len(candidate) != len(SEEDS):
        raise _fail(f"{label} must be exactly ten seed rows")
    rows: list[dict[str, object]] = []
    hashes: list[str] = []
    for seed, raw in zip(SEEDS, candidate, strict=True):
        row_label = f"{label} seed {seed}"
        if not isinstance(raw, Mapping) or set(raw) != _SEED_ROW_FIELDS:
            raise _fail(f"{row_label} fields do not match")
        if raw["seed"] != seed or type(raw["seed"]) is not int:
            raise _fail(f"{label} rows must be seeds 0 through 9 ascending")
        digest = _sha256_text(raw["sha256"], label=row_label)
        _nonnull_version(raw["version_id"], label=row_label)
        if bounded:
            _bounded_bytes(raw["bytes"], label=row_label)
        else:
            _positive_bytes(raw["bytes"], label=row_label)
        if raw["uri"] != f"{root}/" + collection_receipt_key(seed, digest):
            raise _fail(f"{row_label} URI does not use its canonical key")
        hashes.append(digest)
        rows.append(dict(raw))
    if len(set(hashes)) != len(SEEDS):
        raise _fail(f"{label} rows alias one receipt across seeds")
    return rows


def parse_cohort_evidence_index(
    payload: bytes,
    *,
    s3_root: str,
) -> dict[str, object]:
    """Parse and fail-close one untrusted schema-1 cohort evidence index.

    Every URI must sit at its canonical key under the pinned S3 root, every
    byte count is bounded by the sealed member ceiling before any download,
    and every declared identity is subsequently proven by exact-version GET
    plus independent stream rehash in the collection controller.
    """

    if not isinstance(payload, bytes) or not payload:
        raise _fail("evidence index bytes are missing")
    root = _safe_s3_root(s3_root, label="evidence index")
    value = _strict_json(payload, label="evidence index")
    if set(value) != _INDEX_FIELDS:
        raise _fail("evidence index fields do not match the closed schema")
    if type(value["schema_version"]) is not int or value[
        "schema_version"
    ] != 1:
        raise _fail("evidence index schema_version must be exactly 1")
    if value["cohort_id"] != COHORT_ID:
        raise _fail("evidence index cohort identity is invalid")
    _evidence_reference(
        value["study_lock"],
        root=root,
        key_for=study_lock_object_key,
        label="evidence index study lock",
        bounded=True,
    )
    _evidence_reference(
        value["cohort_report"],
        root=root,
        key_for=cohort_report_object_key,
        label="evidence index cohort report",
        bounded=True,
    )
    _seed_collection_rows(
        value["seed_collections"],
        root=root,
        label="evidence index seed collections",
        bounded=True,
    )

    outputs = value["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(_SLOT_PLAN):
        raise _fail("evidence index must enumerate exactly 100 outputs")
    profile_id: str | None = None
    for index, (raw, (seed, arm, step)) in enumerate(
        zip(outputs, _SLOT_PLAN, strict=True)
    ):
        label = f"evidence index output {index}"
        if not isinstance(raw, Mapping) or set(raw) != _INDEX_OUTPUT_FIELDS:
            raise _fail(f"{label} fields do not match")
        if (
            raw["slot_index"] != index
            or type(raw["slot_index"]) is not int
            or raw["seed"] != seed
            or type(raw["seed"]) is not int
            or raw["arm"] != arm
            or raw["optimizer_step"] != step
            or type(raw["optimizer_step"]) is not int
        ):
            raise _fail(
                f"{label} does not follow the frozen 100-slot order"
            )
        output_id = raw["output_id"]
        if not isinstance(output_id, str):
            raise _fail(f"{label} output_id must be a string")
        match = _OUTPUT_ID_RE.fullmatch(output_id)
        if (
            match is None
            or match.group("profile") not in _PROFILE_PROVIDERS
            or int(match.group("seed")) != seed
            or match.group("arm") != arm
            or int(match.group("step")) != step
        ):
            raise _fail(
                f"{label} output_id is not the canonical derivation"
            )
        if profile_id is None:
            profile_id = match.group("profile")
        elif match.group("profile") != profile_id:
            raise _fail(
                "evidence index outputs cross evaluator profiles"
            )
        members = raw["members"]
        if not isinstance(members, list) or len(members) != len(
            EVALUATION_OUTPUT_MEMBERS
        ):
            raise _fail(f"{label} must carry exactly ten member rows")
        for member_name, member_row in zip(
            EVALUATION_OUTPUT_MEMBERS,
            members,
            strict=True,
        ):
            member_label = f"{label} member {member_name}"
            if not isinstance(member_row, Mapping) or set(member_row) != (
                _INDEX_MEMBER_FIELDS
            ):
                raise _fail(f"{member_label} fields do not match")
            if member_row["member"] != member_name:
                raise _fail(
                    f"{label} members are not in sorted member-name order"
                )
            digest = _sha256_text(member_row["sha256"], label=member_label)
            _nonnull_version(member_row["version_id"], label=member_label)
            _bounded_bytes(member_row["bytes"], label=member_label)
            if member_row["uri"] != f"{root}/" + (
                evaluation_output_member_key(
                    seed,
                    arm,
                    step,
                    member_name,
                    digest,
                )
            ):
                raise _fail(
                    f"{member_label} URI does not use its canonical key"
                )
    return value


def parse_cohort_collection_receipt_bytes(
    payload: bytes,
    *,
    receipt_uri: str,
    receipt_sha256: str,
    receipt_version_id: str,
    expected_binding: ProviderLifecycleBinding | None = None,
) -> dict[str, object]:
    """Parse exact canonical bytes for one cohort collection receipt."""

    if not isinstance(payload, bytes) or not payload:
        raise _fail("receipt bytes are missing")
    value = _strict_json(payload, label="receipt")
    if _canonical_json(value) != payload:
        raise _fail("receipt bytes are not canonical")
    expected_sha256 = _sha256_text(receipt_sha256, label="receipt")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise _fail("receipt hash does not match its bytes")
    _nonnull_version(receipt_version_id, label="receipt")
    if set(value) != COHORT_COLLECTION_RECEIPT_FIELDS:
        raise _fail("receipt fields do not match the closed contract")
    if type(value["seed"]) is not int or value["seed"] != _TERMINAL_SEED:
        raise _fail("receipt seed must be the exact terminal seed 9")
    root = _object_root(
        receipt_uri,
        suffix="/" + cohort_collection_receipt_key(expected_sha256),
        label="receipt",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 3
        or value["receipt_type"] != COHORT_COLLECTION_RECEIPT_TYPE
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
    for field in _SHA256_FIELDS:
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
            raise _fail(
                "expected binding must be a provider lifecycle binding"
            )
        # The frozen Task 3D mapping is imported (never duplicated); the
        # import stays function-local because the finalization module
        # transitively imports torch at module scope.
        from cluster.aws.p5.run_finalization import _RECEIPT_BINDING_FIELDS

        expected = expected_binding.to_dict()
        if any(
            value[field] != expected[binding_field]
            for field, binding_field in _RECEIPT_BINDING_FIELDS.items()
        ):
            raise _fail(
                "receipt differs from the authenticated provider lifecycle"
            )

    lock_reference = _evidence_reference(
        value["study_lock"],
        root=root,
        key_for=study_lock_object_key,
        label="receipt study lock",
    )
    report_reference = _evidence_reference(
        value["cohort_report"],
        root=root,
        key_for=cohort_report_object_key,
        label="receipt cohort report",
    )
    seed_references = _seed_collection_rows(
        value["seed_collections"],
        root=root,
        label="receipt seed collections",
    )

    rows = value["objects"]
    if not isinstance(rows, list) or len(rows) != (
        COHORT_COLLECTION_OBJECT_COUNT
    ):
        raise _fail(
            "receipt objects must be exactly the 1012 canonical rows"
        )
    for index, (row, (kind, seed, arm, step, member)) in enumerate(
        zip(rows, _ROW_PLAN, strict=True)
    ):
        label = f"receipt object {index}"
        if not isinstance(row, Mapping) or set(row) != _OBJECT_ROW_FIELDS:
            raise _fail(f"{label} fields do not match")
        if (
            row["kind"] != kind
            or row["seed"] != seed
            or row["arm"] != arm
            or row["step"] != step
            or row["member"] != member
        ):
            raise _fail(
                f"{label} does not follow the canonical collection order"
            )
        digest = _sha256_text(row["sha256"], label=label)
        _positive_bytes(row["bytes"], label=label)
        _nonnull_version(row["version_id"], label=label)
        if row["uri"] != f"{root}/" + _cohort_row_key(
            kind,
            seed,
            arm,
            step,
            member,
            digest,
        ):
            raise _fail(f"{label} URI does not use its shared key helper")
    for offset, reference, label in (
        (1_010, lock_reference, "study lock"),
        (1_011, report_reference, "cohort report"),
    ):
        if any(
            rows[offset][field] != reference[field]
            for field in ("uri", "sha256", "bytes", "version_id")
        ):
            raise _fail(
                f"receipt {label} row does not equal its top-level "
                "reference"
            )
    for seed, reference in zip(SEEDS, seed_references, strict=True):
        row = rows[1_000 + seed]
        if row["seed"] != reference["seed"] or any(
            row[field] != reference[field]
            for field in ("uri", "sha256", "bytes", "version_id")
        ):
            raise _fail(
                f"receipt seed {seed} collection row does not equal its "
                "top-level reference"
            )
    return value


__all__ = [
    "COHORT_COLLECTION_OBJECT_COUNT",
    "COHORT_COLLECTION_RECEIPT_FIELDS",
    "COHORT_COLLECTION_RECEIPT_TYPE",
    "CohortCollectedObject",
    "CohortCollectionError",
    "CohortEvidenceRef",
    "parse_cohort_collection_receipt_bytes",
    "parse_cohort_evidence_index",
]
