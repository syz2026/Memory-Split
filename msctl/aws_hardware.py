"""Prospective AWS hardware amendment and provider-selection contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from cluster.aws.gpu_profile import (
    load_aws_gpu_profile,
    parse_aws_gpu_profile_bytes,
    read_secure_regular_file,
)
from msctl.aws_contracts import (
    AWS_RUNTIME_LOCK_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
    validate_digest_pinned_oci_image,
)
from msctl.errors import MsctlError
from msctl.fsutil import open_directory, rename_noreplace_at


AWS_HARDWARE_AMENDMENT_PATH = "configs/aws-hardware-amendment-v3.json"
P5_PROFILE_PATH = "cluster/profiles/aws-p5.48xlarge-v3.json"
P6_PROFILE_PATH = "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
PREREGISTRATION_PATH = "configs/preregistration-v3.yaml"
COHORT_ASSIGNMENT_PATH = "configs/cohort-assignment-v3.json"
PROVIDER_SELECTION_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json"
)
PROVIDER_SELECTION_S3_KEY = (
    "cohorts/memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json"
)
AWS_HARDWARE_AMENDMENT_SHA256 = (
    "d4cf13b587c751d27756ad7881e538facb7ea79305a098990a568a7b28b6fb14"
)

_MAX_JSON_BYTES = 65_536
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_AVAILABILITY_ZONE_RE = re.compile(
    r"^(?P<region>[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+)[a-z]$"
)
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_P5_PROFILE_SHA256 = (
    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
)
_P6_PROFILE_SHA256 = (
    "6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4"
)
_PREREGISTRATION_SHA256 = (
    "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
)
_COHORT_ASSIGNMENT_SHA256 = (
    "47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c"
)
_EXPECTED_AMENDMENT = {
    "amendment_id": "memorysplit-confirmatory-v3-aws-hardware-selection",
    "amendment_mode": "append_only",
    "cohort_assignment": {
        "path": COHORT_ASSIGNMENT_PATH,
        "sha256": _COHORT_ASSIGNMENT_SHA256,
    },
    "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
    "eligible_profiles": [
        {
            "path": P5_PROFILE_PATH,
            "profile_id": "aws-p5.48xlarge-v3",
            "provider": "aws-p5.48xlarge",
            "sha256": _P5_PROFILE_SHA256,
        },
        {
            "path": P6_PROFILE_PATH,
            "profile_id": "aws-p6-b300.48xlarge-v3",
            "provider": "aws-p6-b300.48xlarge",
            "sha256": _P6_PROFILE_SHA256,
        },
    ],
    "frozen_contract": {
        "arms": ["dense", "split90"],
        "one_profile_for_entire_cohort": True,
        "protected_outcomes_inspected": [],
        "scientific_invariants": "all_non_provider_fields_remain_frozen",
        "seeds": list(range(10)),
        "symmetric_training": True,
        "train_groups": [4, 4],
    },
    "preregistration": {
        "path": PREREGISTRATION_PATH,
        "sha256": _PREREGISTRATION_SHA256,
    },
    "prospective": True,
    "schema_version": 1,
    "supersedes_only": ["provider_assignment", "hardware_topology"],
}


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class HardwareProfileBinding:
    path: str
    profile_id: str
    provider: str
    sha256: str


@dataclass(frozen=True)
class AwsHardwareAmendment:
    schema_version: int
    amendment_id: str
    amendment_mode: str
    prospective: bool
    cohort_id: str
    preregistration: ArtifactBinding
    cohort_assignment: ArtifactBinding
    profiles: tuple[HardwareProfileBinding, ...]
    supersedes_only: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    train_groups: tuple[int, int]
    symmetric_training: bool
    one_profile_for_entire_cohort: bool
    scientific_invariants: str
    protected_outcomes_inspected: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class AwsProviderSelectionReceipt:
    """One immutable hardware/runtime choice for every protected run cell."""

    schema_version: int
    receipt_type: str
    amendment: ArtifactBinding
    authority_local_path: str
    authority_s3_key: str
    cohort_id: str
    preregistration_sha256: str
    cohort_assignment_sha256: str
    profile: HardwareProfileBinding
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    train_groups: tuple[int, int]
    runtime_lock_sha256: str
    runtime_evidence_sha256: str
    ami_id: str
    container_image_digest: str
    account_id: str
    region: str
    availability_zone: str
    purchase_model: str
    selected_at: str
    selection_scope: str
    replacement_policy: str
    protected_outcomes_inspected: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class AwsRuntimeLock:
    schema_version: int
    source_commit: str
    source_tree: str
    control_bundle_sha256: str
    profile_sha256: str
    ami_id: str
    ami_owner_id: str
    container_image: str
    container_image_digest: str
    versions: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True)
class AwsRuntimeEvidence:
    schema_version: int
    receipt_type: str
    profile_sha256: str
    runtime_lock_sha256: str
    account_id: str
    region: str
    availability_zone: str
    versions: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True)
class VersionedSelectionObject:
    key: str
    sha256: str
    bytes: int
    version_id: str


@dataclass(frozen=True)
class PublishedProviderSelection:
    selection: AwsProviderSelectionReceipt
    local_path: Path
    remote: VersionedSelectionObject


class VersionedProviderSelectionStore(Protocol):
    def put_if_none_match(
        self,
        *,
        key: str,
        data: bytes,
        if_none_match: str,
        checksum_sha256: str,
    ) -> VersionedSelectionObject | None: ...

    def head(
        self,
        *,
        key: str,
        version_id: str | None,
    ) -> VersionedSelectionObject | None: ...


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _closed_equal(actual: object, expected: object, *, label: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{label} has the wrong JSON type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} has missing or unknown fields")
        for key, expected_item in expected.items():
            _closed_equal(actual[key], expected_item, label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} has the wrong number of items")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _closed_equal(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
            )
        return
    if actual != expected:
        raise ValueError(f"{label} does not match the frozen contract")


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} bytes must be bytes")
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds 64 KiB")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parse_aws_hardware_amendment_bytes(data: bytes) -> AwsHardwareAmendment:
    """Parse exact amendment bytes under the closed prospective contract."""

    value = _json_object(data, label="hardware amendment")
    _closed_equal(value, _EXPECTED_AMENDMENT, label="hardware amendment")
    digest = hashlib.sha256(data).hexdigest()
    if digest != AWS_HARDWARE_AMENDMENT_SHA256:
        raise ValueError("hardware amendment bytes do not match the frozen hash")
    profiles = tuple(
        HardwareProfileBinding(
            path=profile["path"],
            profile_id=profile["profile_id"],
            provider=profile["provider"],
            sha256=profile["sha256"],
        )
        for profile in value["eligible_profiles"]
    )
    frozen = value["frozen_contract"]
    return AwsHardwareAmendment(
        schema_version=value["schema_version"],
        amendment_id=value["amendment_id"],
        amendment_mode=value["amendment_mode"],
        prospective=value["prospective"],
        cohort_id=value["cohort_id"],
        preregistration=ArtifactBinding(**value["preregistration"]),
        cohort_assignment=ArtifactBinding(**value["cohort_assignment"]),
        profiles=profiles,
        supersedes_only=tuple(value["supersedes_only"]),
        seeds=tuple(frozen["seeds"]),
        arms=tuple(frozen["arms"]),
        train_groups=tuple(frozen["train_groups"]),
        symmetric_training=frozen["symmetric_training"],
        one_profile_for_entire_cohort=frozen[
            "one_profile_for_entire_cohort"
        ],
        scientific_invariants=frozen["scientific_invariants"],
        protected_outcomes_inspected=tuple(
            frozen["protected_outcomes_inspected"]
        ),
        sha256=digest,
    )


def _regular_bytes(
    path: Path,
    *,
    label: str,
    private: bool = False,
) -> bytes:
    return read_secure_regular_file(
        path,
        label=label,
        max_bytes=_MAX_JSON_BYTES,
        private=private,
    )


def _validate_aws_hardware_amendment_files(
    amendment: AwsHardwareAmendment,
    *,
    repo_root: Path | str,
) -> None:
    """Verify every append-only binding against one repository root."""

    root = Path(repo_root)
    bindings = (
        amendment.preregistration,
        amendment.cohort_assignment,
        *amendment.profiles,
    )
    for binding in bindings:
        candidate = root.joinpath(*binding.path.split("/"))
        data = _regular_bytes(candidate, label=binding.path)
        if hashlib.sha256(data).hexdigest() != binding.sha256:
            raise ValueError(f"amendment binding hash mismatch: {binding.path}")
    for binding in amendment.profiles:
        profile = load_aws_gpu_profile(root / binding.path)
        if (
            profile.profile_id != binding.profile_id
            or profile.provider != binding.provider
            or profile.sha256 != binding.sha256
        ):
            raise ValueError(
                f"amendment profile identity mismatch: {binding.profile_id}"
            )


def load_aws_hardware_amendment(
    path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> AwsHardwareAmendment:
    """Load a regular amendment and optionally verify all bound repository files."""

    candidate = Path(path)
    amendment = parse_aws_hardware_amendment_bytes(
        _regular_bytes(candidate, label="hardware amendment")
    )
    if repo_root is not None:
        _validate_aws_hardware_amendment_files(amendment, repo_root=repo_root)
    return amendment


def validate_aws_hardware_amendment_files(
    *,
    repo_root: Path | str,
) -> AwsHardwareAmendment:
    """Load the fixed amendment bytes, then verify every bound file."""

    amendment = parse_aws_hardware_amendment_bytes(
        _regular_bytes(
            Path(repo_root).joinpath(*AWS_HARDWARE_AMENDMENT_PATH.split("/")),
            label="hardware amendment",
        )
    )
    _validate_aws_hardware_amendment_files(amendment, repo_root=repo_root)
    return amendment


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{label} has missing or unknown fields; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _exact_string(value: object, expected: str, *, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("provider selection is not canonical JSON") from error


def _profile_binding(
    value: object,
    *,
    amendment: AwsHardwareAmendment,
) -> HardwareProfileBinding:
    profile = _object(
        value,
        fields=frozenset({"path", "profile_id", "provider", "sha256"}),
        label="provider selection profile",
    )
    for field in ("path", "profile_id", "provider"):
        if not isinstance(profile[field], str):
            raise ValueError(f"provider selection profile {field} must be a string")
    binding = HardwareProfileBinding(
        path=profile["path"],
        profile_id=profile["profile_id"],
        provider=profile["provider"],
        sha256=_sha256(
            profile["sha256"],
            label="provider selection profile SHA-256",
        ),
    )
    if binding not in amendment.profiles:
        raise ValueError(
            "provider selection profile is not one exact eligible profile"
        )
    return binding


def _parse_timestamp(value: object) -> str:
    if (
        not isinstance(value, str)
        or _UTC_TIMESTAMP_RE.fullmatch(value) is None
    ):
        raise ValueError(
            "provider selection timestamp must be UTC to exact seconds"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("provider selection timestamp is not a valid date") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("provider selection timestamp is not canonical")
    return value


def _validate_placement(
    *,
    profile: HardwareProfileBinding,
    account_id: object,
    region: object,
    availability_zone: object,
    purchase_model: object,
) -> tuple[str, str, str, str]:
    if (
        not isinstance(account_id, str)
        or _ACCOUNT_RE.fullmatch(account_id) is None
    ):
        raise ValueError("AWS account ID must be exactly 12 decimal digits")
    if not isinstance(region, str) or _REGION_RE.fullmatch(region) is None:
        raise ValueError("AWS region is invalid")
    if (
        not isinstance(availability_zone, str)
        or (match := _AVAILABILITY_ZONE_RE.fullmatch(availability_zone)) is None
        or match.group("region") != region
    ):
        raise ValueError("AWS availability zone must belong to its region")
    expected_purchase = {
        "aws-p5.48xlarge-v3": "on_demand",
        "aws-p6-b300.48xlarge-v3": "on_demand",
    }[profile.profile_id]
    if purchase_model != expected_purchase:
        raise ValueError("AWS purchase model does not match the selected profile")
    if region != "us-east-1":
        raise ValueError("AWS region is not allowed by the selected profile")
    if (
        profile.profile_id == "aws-p6-b300.48xlarge-v3"
        and availability_zone != "us-east-1d"
    ):
        raise ValueError("P6-B300 selection requires its us-east-1d offering")
    return account_id, region, availability_zone, expected_purchase


def _parse_provider_selection_receipt_bytes(
    data: bytes,
    *,
    amendment: AwsHardwareAmendment,
) -> AwsProviderSelectionReceipt:
    """Parse the sole canonical, cohort-wide provider-selection receipt."""

    if not isinstance(amendment, AwsHardwareAmendment):
        raise TypeError("hardware amendment must be parsed before selection")
    if amendment.sha256 != AWS_HARDWARE_AMENDMENT_SHA256:
        raise ValueError("provider selection requires the frozen amendment")
    value = _json_object(data, label="provider selection receipt")
    if data != _canonical_json(value):
        raise ValueError(
            "provider selection receipt must use its canonical JSON bytes"
        )
    root = _object(
        value,
        fields=frozenset(
            {
                "amendment",
                "authority",
                "aws",
                "cohort",
                "profile",
                "protected_outcomes_inspected",
                "receipt_type",
                "replacement_policy",
                "runtime",
                "schema_version",
                "selected_at",
                "selection_scope",
            }
        ),
        label="provider selection receipt",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("provider selection schema_version must be integer 1")
    receipt_type = _exact_string(
        root["receipt_type"],
        "memorysplit-aws-provider-selection-v1",
        label="provider selection receipt_type",
    )
    selection_scope = _exact_string(
        root["selection_scope"],
        "entire_cohort",
        label="provider selection scope",
    )
    replacement_policy = _exact_string(
        root["replacement_policy"],
        "forbidden",
        label="provider selection replacement policy",
    )
    authority = _object(
        root["authority"],
        fields=frozenset({"local_path", "s3_key"}),
        label="provider selection authority",
    )
    authority_local_path = _exact_string(
        authority["local_path"],
        PROVIDER_SELECTION_LOCAL_PATH,
        label="provider selection local authority path",
    )
    authority_s3_key = _exact_string(
        authority["s3_key"],
        PROVIDER_SELECTION_S3_KEY,
        label="provider selection S3 authority key",
    )

    amendment_value = _object(
        root["amendment"],
        fields=frozenset({"path", "sha256"}),
        label="provider selection amendment",
    )
    amendment_binding = ArtifactBinding(
        path=_exact_string(
            amendment_value["path"],
            AWS_HARDWARE_AMENDMENT_PATH,
            label="provider selection amendment path",
        ),
        sha256=_sha256(
            amendment_value["sha256"],
            label="provider selection amendment SHA-256",
        ),
    )
    if amendment_binding.sha256 != amendment.sha256:
        raise ValueError("provider selection amendment hash does not match")

    cohort = _object(
        root["cohort"],
        fields=frozenset(
            {
                "arms",
                "cohort_assignment_sha256",
                "cohort_id",
                "preregistration_sha256",
                "seeds",
                "train_groups",
            }
        ),
        label="provider selection cohort",
    )
    cohort_id = _exact_string(
        cohort["cohort_id"],
        amendment.cohort_id,
        label="provider selection cohort ID",
    )
    preregistration_sha256 = _sha256(
        cohort["preregistration_sha256"],
        label="provider selection preregistration SHA-256",
    )
    cohort_assignment_sha256 = _sha256(
        cohort["cohort_assignment_sha256"],
        label="provider selection cohort-assignment SHA-256",
    )
    if preregistration_sha256 != amendment.preregistration.sha256:
        raise ValueError("provider selection preregistration hash does not match")
    if cohort_assignment_sha256 != amendment.cohort_assignment.sha256:
        raise ValueError("provider selection cohort-assignment hash does not match")
    for field, expected in (
        ("seeds", list(amendment.seeds)),
        ("arms", list(amendment.arms)),
        ("train_groups", list(amendment.train_groups)),
    ):
        _closed_equal(
            cohort[field],
            expected,
            label=f"provider selection cohort {field}",
        )
    profile = _profile_binding(root["profile"], amendment=amendment)

    runtime = _object(
        root["runtime"],
        fields=frozenset(
            {
                "runtime_lock_sha256",
                "runtime_evidence_sha256",
                "ami_id",
                "container_image_digest",
            }
        ),
        label="provider selection runtime",
    )
    runtime_lock_sha256 = _sha256(
        runtime["runtime_lock_sha256"],
        label="provider selection runtime-lock SHA-256",
    )
    runtime_evidence_sha256 = _sha256(
        runtime["runtime_evidence_sha256"],
        label="provider selection runtime-evidence SHA-256",
    )
    ami_id = runtime["ami_id"]
    if not isinstance(ami_id, str) or _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("provider selection AMI must be one immutable AMI ID")
    container_image_digest = runtime["container_image_digest"]
    if (
        not isinstance(container_image_digest, str)
        or _OCI_DIGEST_RE.fullmatch(container_image_digest) is None
    ):
        raise ValueError(
            "provider selection image digest must be canonical sha256"
        )

    aws = _object(
        root["aws"],
        fields=frozenset(
            {"account_id", "availability_zone", "purchase_model", "region"}
        ),
        label="provider selection AWS placement",
    )
    account_id, region, availability_zone, purchase_model = _validate_placement(
        profile=profile,
        account_id=aws["account_id"],
        region=aws["region"],
        availability_zone=aws["availability_zone"],
        purchase_model=aws["purchase_model"],
    )
    outcomes = root["protected_outcomes_inspected"]
    if not isinstance(outcomes, list) or outcomes:
        raise ValueError(
            "provider selection requires zero protected outcomes inspected"
        )

    return AwsProviderSelectionReceipt(
        schema_version=1,
        receipt_type=receipt_type,
        amendment=amendment_binding,
        authority_local_path=authority_local_path,
        authority_s3_key=authority_s3_key,
        cohort_id=cohort_id,
        preregistration_sha256=preregistration_sha256,
        cohort_assignment_sha256=cohort_assignment_sha256,
        profile=profile,
        seeds=amendment.seeds,
        arms=amendment.arms,
        train_groups=amendment.train_groups,
        runtime_lock_sha256=runtime_lock_sha256,
        runtime_evidence_sha256=runtime_evidence_sha256,
        ami_id=ami_id,
        container_image_digest=container_image_digest,
        account_id=account_id,
        region=region,
        availability_zone=availability_zone,
        purchase_model=purchase_model,
        selected_at=_parse_timestamp(root["selected_at"]),
        selection_scope=selection_scope,
        replacement_policy=replacement_policy,
        protected_outcomes_inspected=(),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def parse_provider_selection_receipt_bytes(
    data: bytes,
    *,
    amendment_data: bytes,
) -> AwsProviderSelectionReceipt:
    """Parse selection bytes against the immutable amendment bytes."""

    amendment = parse_aws_hardware_amendment_bytes(amendment_data)
    return _parse_provider_selection_receipt_bytes(data, amendment=amendment)


def parse_aws_runtime_lock_bytes(data: bytes) -> AwsRuntimeLock:
    """Parse one exact canonical runtime lock from authenticated bytes."""

    value = _json_object(data, label="runtime lock")
    if data != _canonical_json(value):
        raise ValueError("runtime lock must use canonical JSON bytes")
    lock = _object(
        value,
        fields=frozenset(AWS_RUNTIME_LOCK_FIELDS),
        label="runtime lock",
    )
    if type(lock["schema_version"]) is not int or lock["schema_version"] != 1:
        raise ValueError("runtime lock schema_version must be integer 1")
    for field in ("source_commit", "source_tree"):
        if (
            not isinstance(lock[field], str)
            or _COMMIT_RE.fullmatch(lock[field]) is None
        ):
            raise ValueError(f"runtime lock {field} must be a full Git object")
    control_bundle_sha256 = _sha256(
        lock["control_bundle_sha256"],
        label="runtime lock control-bundle SHA-256",
    )
    profile_sha256 = _sha256(
        lock["profile_sha256"],
        label="runtime lock profile SHA-256",
    )
    ami_id = lock["ami_id"]
    if not isinstance(ami_id, str) or _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("runtime lock AMI ID is invalid")
    ami_owner_id = lock["ami_owner_id"]
    if (
        not isinstance(ami_owner_id, str)
        or _ACCOUNT_RE.fullmatch(ami_owner_id) is None
    ):
        raise ValueError("runtime lock AMI owner ID is invalid")
    try:
        container_image, container_image_digest = (
            validate_digest_pinned_oci_image(
                lock["container_image"],
                lock["container_image_digest"],
            )
        )
    except ValueError as error:
        raise ValueError("runtime lock container image is not digest pinned") from error
    versions = _object(
        lock["versions"],
        fields=frozenset(AWS_RUNTIME_VERSION_FIELDS),
        label="runtime lock versions",
    )
    normalized_versions: list[tuple[str, str]] = []
    for field in AWS_RUNTIME_VERSION_FIELDS:
        version = versions[field]
        if (
            not isinstance(version, str)
            or not 1 <= len(version) <= 128
            or any(character in version for character in "\x00\n\r")
        ):
            raise ValueError(f"runtime lock versions.{field} is invalid")
        normalized_versions.append((field, version))
    return AwsRuntimeLock(
        schema_version=1,
        source_commit=lock["source_commit"],
        source_tree=lock["source_tree"],
        control_bundle_sha256=control_bundle_sha256,
        profile_sha256=profile_sha256,
        ami_id=ami_id,
        ami_owner_id=ami_owner_id,
        container_image=container_image,
        container_image_digest=container_image_digest,
        versions=tuple(normalized_versions),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _version_parts(value: str, *, label: str) -> tuple[int, ...]:
    normalized = value[1:] if value.startswith("R") else value
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", normalized) is None:
        raise ValueError(f"{label} must be a numeric version or R branch")
    return tuple(int(part) for part in normalized.split("."))


def _version_at_least(value: str, floor: str, *, label: str) -> None:
    actual = _version_parts(value, label=label)
    minimum = _version_parts(floor, label=f"{label} floor")
    width = max(len(actual), len(minimum))
    if actual + (0,) * (width - len(actual)) < minimum + (0,) * (
        width - len(minimum)
    ):
        raise ValueError(f"{label} is below the selected profile floor")


def _parse_runtime_evidence_bytes(
    data: bytes,
    *,
    profile: object,
) -> AwsRuntimeEvidence:
    value = _json_object(data, label="runtime evidence")
    if data != _canonical_json(value):
        raise ValueError("runtime evidence must use canonical JSON bytes")
    evidence = _object(
        value,
        fields=frozenset(
            {
                "account_id",
                "availability_zone",
                "profile_sha256",
                "receipt_type",
                "region",
                "runtime_lock_sha256",
                "schema_version",
                "versions",
            }
        ),
        label="runtime evidence",
    )
    if (
        type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != 1
    ):
        raise ValueError("runtime evidence schema_version must be integer 1")
    receipt_type = _exact_string(
        evidence["receipt_type"],
        "memorysplit-aws-runtime-evidence-v1",
        label="runtime evidence receipt_type",
    )
    profile_sha256 = _sha256(
        evidence["profile_sha256"],
        label="runtime evidence profile SHA-256",
    )
    runtime_lock_sha256 = _sha256(
        evidence["runtime_lock_sha256"],
        label="runtime evidence runtime-lock SHA-256",
    )
    account_id = evidence["account_id"]
    if (
        not isinstance(account_id, str)
        or _ACCOUNT_RE.fullmatch(account_id) is None
    ):
        raise ValueError("runtime evidence account ID is invalid")
    region = evidence["region"]
    availability_zone = evidence["availability_zone"]
    if not isinstance(region, str) or _REGION_RE.fullmatch(region) is None:
        raise ValueError("runtime evidence region is invalid")
    if (
        not isinstance(availability_zone, str)
        or (match := _AVAILABILITY_ZONE_RE.fullmatch(availability_zone)) is None
        or match.group("region") != region
    ):
        raise ValueError("runtime evidence availability zone is invalid")
    floors = tuple(getattr(profile, "software_floors"))
    versions = _object(
        evidence["versions"],
        fields=frozenset(dict(floors)),
        label="runtime evidence versions",
    )
    normalized_versions: list[tuple[str, str]] = []
    for field, floor in floors:
        version = versions[field]
        if not isinstance(version, str):
            raise ValueError(f"runtime evidence versions.{field} must be a string")
        _version_at_least(
            version,
            floor,
            label=f"runtime evidence versions.{field}",
        )
        normalized_versions.append((field, version))
    return AwsRuntimeEvidence(
        schema_version=1,
        receipt_type=receipt_type,
        profile_sha256=profile_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        account_id=account_id,
        region=region,
        availability_zone=availability_zone,
        versions=tuple(normalized_versions),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def parse_verified_provider_selection_bytes(
    data: bytes,
    *,
    amendment_data: bytes,
    profile_data: bytes,
    runtime_lock_data: bytes,
    runtime_evidence_data: bytes,
) -> AwsProviderSelectionReceipt:
    """Verify selection authority from anchored profile and runtime bytes."""

    amendment = parse_aws_hardware_amendment_bytes(amendment_data)
    profile = parse_aws_gpu_profile_bytes(profile_data)
    eligible = tuple(
        binding
        for binding in amendment.profiles
        if binding.profile_id == profile.profile_id
        and binding.provider == profile.provider
        and binding.sha256 == profile.sha256
    )
    if len(eligible) != 1:
        raise ValueError("selected profile bytes are not amendment eligible")
    receipt = _parse_provider_selection_receipt_bytes(
        data,
        amendment=amendment,
    )
    if receipt.profile != eligible[0]:
        raise ValueError("selection does not match the anchored profile bytes")

    runtime_lock = parse_aws_runtime_lock_bytes(runtime_lock_data)
    if runtime_lock.sha256 != receipt.runtime_lock_sha256:
        raise ValueError("selection runtime-lock hash does not match its bytes")
    if runtime_lock.profile_sha256 != profile.sha256:
        raise ValueError("runtime lock does not bind the selected profile")
    if runtime_lock.ami_id != receipt.ami_id:
        raise ValueError("selection AMI does not match the runtime lock")
    if runtime_lock.container_image_digest != receipt.container_image_digest:
        raise ValueError("selection image digest does not match the runtime lock")

    evidence = _parse_runtime_evidence_bytes(
        runtime_evidence_data,
        profile=profile,
    )
    if evidence.sha256 != receipt.runtime_evidence_sha256:
        raise ValueError("selection runtime-evidence hash does not match its bytes")
    if evidence.profile_sha256 != profile.sha256:
        raise ValueError("runtime evidence does not bind the selected profile")
    if evidence.runtime_lock_sha256 != runtime_lock.sha256:
        raise ValueError("runtime evidence does not bind the runtime lock")
    if (
        evidence.account_id != receipt.account_id
        or evidence.region != receipt.region
        or evidence.availability_zone != receipt.availability_zone
    ):
        raise ValueError("runtime evidence placement differs from selection")
    lock_versions = dict(runtime_lock.versions)
    for field, floor in profile.software_floors:
        if field in lock_versions:
            _version_at_least(
                lock_versions[field],
                floor,
                label=f"runtime lock versions.{field}",
            )
    return receipt


parse_aws_provider_selection_receipt_bytes = (
    parse_provider_selection_receipt_bytes
)


def canonical_provider_selection_receipt_bytes(
    value: object,
    *,
    amendment_data: bytes,
    profile_data: bytes,
    runtime_lock_data: bytes,
    runtime_evidence_data: bytes,
) -> bytes:
    """Encode and verify one provider selection from anchored bytes."""

    data = _canonical_json(value)
    parse_verified_provider_selection_bytes(
        data,
        amendment_data=amendment_data,
        profile_data=profile_data,
        runtime_lock_data=runtime_lock_data,
        runtime_evidence_data=runtime_evidence_data,
    )
    return data


def _load_verified_selection_from_paths(
    selection_data: bytes,
    *,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
) -> AwsProviderSelectionReceipt:
    amendment_data = _regular_bytes(
        Path(repo_root).joinpath(*AWS_HARDWARE_AMENDMENT_PATH.split("/")),
        label="hardware amendment",
    )
    amendment = parse_aws_hardware_amendment_bytes(amendment_data)
    preliminary = _parse_provider_selection_receipt_bytes(
        selection_data,
        amendment=amendment,
    )
    profile_data = _regular_bytes(
        Path(repo_root).joinpath(*preliminary.profile.path.split("/")),
        label="selected hardware profile",
    )
    runtime_lock_data = _regular_bytes(
        Path(runtime_lock_path),
        label="runtime lock",
        private=True,
    )
    runtime_evidence_data = _regular_bytes(
        Path(runtime_evidence_path),
        label="runtime evidence",
        private=True,
    )
    return parse_verified_provider_selection_bytes(
        selection_data,
        amendment_data=amendment_data,
        profile_data=profile_data,
        runtime_lock_data=runtime_lock_data,
        runtime_evidence_data=runtime_evidence_data,
    )


def _write_selection_noreplace(
    path: Path | str,
    data: bytes,
) -> Path:
    destination = Path(path)
    name = destination.name
    if not name or name in {".", ".."}:
        raise ValueError("provider selection destination name is invalid")
    directory_fd = open_directory(
        destination.parent,
        label="provider selection parent",
        create=True,
        mode=0o700,
    )
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        rename_noreplace_at(
            directory_fd,
            temporary,
            directory_fd,
            name,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return destination


def _publish_local_selection(
    *,
    authority_root: Path | str,
    data: bytes,
) -> Path:
    destination = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_LOCAL_PATH.split("/")
    )
    try:
        _write_selection_noreplace(destination, data)
    except FileExistsError as error:
        existing = _regular_bytes(
            destination,
            label="provider selection authority",
            private=True,
        )
        if existing != data:
            raise ValueError(
                "fixed local provider selection conflicts with different bytes"
            ) from error
    installed = _regular_bytes(
        destination,
        label="provider selection authority",
        private=True,
    )
    if installed != data:
        raise ValueError("fixed local provider selection differs after publication")
    return destination


def _preflight_local_selection(
    *,
    authority_root: Path | str,
    data: bytes,
) -> None:
    destination = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_LOCAL_PATH.split("/")
    )
    try:
        existing = _regular_bytes(
            destination,
            label="provider selection authority",
            private=True,
        )
    except ValueError as error:
        cause = error.__cause__
        if isinstance(cause, FileNotFoundError) or (
            isinstance(cause, MsctlError) and cause.code == "FILE_NOT_FOUND"
        ):
            return
        raise
    if existing != data:
        raise ValueError(
            "fixed local provider selection conflicts with different bytes"
        )


def _verified_remote_object(
    value: object,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_version: str | None,
) -> VersionedSelectionObject:
    if not isinstance(value, VersionedSelectionObject):
        raise ValueError("remote selection HEAD did not return an exact object")
    if (
        value.key != PROVIDER_SELECTION_S3_KEY
        or value.sha256 != expected_sha256
        or value.bytes != expected_bytes
        or not isinstance(value.version_id, str)
        or not value.version_id
        or value.version_id == "null"
        or any(character in value.version_id for character in "\x00\n\r")
        or (
            expected_version is not None
            and value.version_id != expected_version
        )
    ):
        raise ValueError("remote selection HEAD checksum, bytes, or version differs")
    return value


def _publish_remote_selection(
    *,
    data: bytes,
    store: VersionedProviderSelectionStore,
) -> VersionedSelectionObject:
    digest = hashlib.sha256(data).hexdigest()
    put = store.put_if_none_match(
        key=PROVIDER_SELECTION_S3_KEY,
        data=data,
        if_none_match="*",
        checksum_sha256=digest,
    )
    if put is None:
        head_version = None
    else:
        put = _verified_remote_object(
            put,
            expected_sha256=digest,
            expected_bytes=len(data),
            expected_version=None,
        )
        head_version = put.version_id
    head = store.head(
        key=PROVIDER_SELECTION_S3_KEY,
        version_id=head_version,
    )
    try:
        return _verified_remote_object(
            head,
            expected_sha256=digest,
            expected_bytes=len(data),
            expected_version=head_version,
        )
    except ValueError as error:
        raise ValueError(
            "remote fixed-key provider selection conflicts or failed exact HEAD"
        ) from error


def publish_provider_selection(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    selection_data: bytes,
    store: VersionedProviderSelectionStore,
) -> PublishedProviderSelection:
    """Publish the sole local and versioned-S3 cohort selection authority."""

    selection = _load_verified_selection_from_paths(
        selection_data,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
    )
    _preflight_local_selection(
        authority_root=authority_root,
        data=selection_data,
    )
    remote = _publish_remote_selection(data=selection_data, store=store)
    local_path = _publish_local_selection(
        authority_root=authority_root,
        data=selection_data,
    )
    return PublishedProviderSelection(
        selection=selection,
        local_path=local_path,
        remote=remote,
    )


def load_local_provider_selection_authority(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
) -> AwsProviderSelectionReceipt:
    """Load and re-verify the sole fixed local selection authority."""

    selection_data = _regular_bytes(
        Path(authority_root).joinpath(*PROVIDER_SELECTION_LOCAL_PATH.split("/")),
        label="provider selection authority",
        private=True,
    )
    return _load_verified_selection_from_paths(
        selection_data,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
    )


def validate_resume_hardware_binding(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    amendment_sha256: str,
    provider_selection_sha256: str,
    profile_sha256: str,
    runtime_lock_sha256: str,
    runtime_evidence_sha256: str,
    seed: int,
    arm: str,
) -> AwsProviderSelectionReceipt:
    """Reload fixed authority bytes before checking one resume binding."""

    receipt = load_local_provider_selection_authority(
        authority_root=authority_root,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
    )
    if amendment_sha256 != receipt.amendment.sha256:
        raise ValueError("hardware amendment binding differs from selection")
    if provider_selection_sha256 != receipt.sha256:
        raise ValueError("provider-selection receipt binding differs")
    if profile_sha256 != receipt.profile.sha256:
        raise ValueError("hardware profile differs from cohort selection")
    if runtime_lock_sha256 != receipt.runtime_lock_sha256:
        raise ValueError("runtime-lock binding differs from cohort selection")
    if runtime_evidence_sha256 != receipt.runtime_evidence_sha256:
        raise ValueError("runtime-evidence binding differs from cohort selection")
    if type(seed) is not int or seed not in receipt.seeds:
        raise ValueError("seed is outside the selected protected cohort")
    if arm not in receipt.arms:
        raise ValueError("arm is outside the selected protected cohort")
    return receipt


validate_provider_selection_binding = validate_resume_hardware_binding
load_provider_selection_receipt = load_local_provider_selection_authority
load_aws_provider_selection_receipt = load_local_provider_selection_authority


__all__ = [
    "AWS_HARDWARE_AMENDMENT_PATH",
    "AWS_HARDWARE_AMENDMENT_SHA256",
    "ArtifactBinding",
    "AwsHardwareAmendment",
    "AwsProviderSelectionReceipt",
    "AwsRuntimeEvidence",
    "AwsRuntimeLock",
    "COHORT_ASSIGNMENT_PATH",
    "HardwareProfileBinding",
    "P5_PROFILE_PATH",
    "P6_PROFILE_PATH",
    "PREREGISTRATION_PATH",
    "PROVIDER_SELECTION_LOCAL_PATH",
    "PROVIDER_SELECTION_S3_KEY",
    "PublishedProviderSelection",
    "VersionedProviderSelectionStore",
    "VersionedSelectionObject",
    "canonical_provider_selection_receipt_bytes",
    "load_aws_hardware_amendment",
    "load_aws_provider_selection_receipt",
    "load_local_provider_selection_authority",
    "load_provider_selection_receipt",
    "parse_aws_provider_selection_receipt_bytes",
    "parse_aws_hardware_amendment_bytes",
    "parse_aws_runtime_lock_bytes",
    "parse_provider_selection_receipt_bytes",
    "parse_verified_provider_selection_bytes",
    "publish_provider_selection",
    "validate_aws_hardware_amendment_files",
    "validate_provider_selection_binding",
    "validate_resume_hardware_binding",
]
