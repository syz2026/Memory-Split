"""Closed, hash-bound AWS v3 provider-selection receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .errors import MsctlError
from .fsutil import open_directory, rename_noreplace_at
from .jsonutil import canonical_json, require_sha256
from .profile import (
    AWS_P5_V3_PROFILE,
    AWS_P6_B300_V3_PROFILE,
    load_profile,
)


SELECTION_RECEIPT_TYPE = "memorysplit-aws-provider-selection-v3"
AMENDMENT_ID = "memorysplit-confirmatory-v3-hardware-amendment"
COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
V3_PROFILE_IDS = (AWS_P5_V3_PROFILE, AWS_P6_B300_V3_PROFILE)
V3_SEEDS = tuple(range(10))
_MAX_JSON_BYTES = 131_072
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REGION_RE = re.compile(r"^us-(?:east-1|west-2)$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_ECR_IMAGE_RE = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>us-(?:east-1|west-2))"
    r"\.amazonaws\.com/(?P<repository>[a-z0-9]+(?:[._/-][a-z0-9]+)*)"
    r"@(?P<digest>sha256:[0-9a-f]{64})$"
)
_CAPACITY_ID_RE = re.compile(r"^[a-z]{2,4}-[A-Za-z0-9-]{8,64}$")
_AMENDMENT_FIELDS = {
    "schema_version",
    "amendment_id",
    "cohort_id",
    "bindings",
    "allowed_profiles",
    "original_scientific_provider_assignment",
    "selection_rule",
    "supersedes_hardware_selection",
    "protected_outcomes_inspected",
}
_BINDING_FIELDS = {"path", "sha256"}
_PROFILE_FIELDS = {
    "profile_id",
    "provider",
    "path",
    "sha256",
    "instance_type",
    "gpu_model",
}
_RULE_FIELDS = {
    "selected_profile_count",
    "same_profile_for_all_pairs",
    "mixed_profiles_forbidden",
    "seeds",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "amendment_sha256",
    "preregistration_sha256",
    "cohort_assignment_sha256",
    "cohort_id",
    "selected_profile_id",
    "provider",
    "profile_sha256",
    "instance_type",
    "gpu_model",
    "region",
    "aws_account_id",
    "ami_id",
    "ami_owner_id",
    "ami_name",
    "ami_describe_sha256",
    "container_image",
    "container_digest",
    "ecr_describe_sha256",
    "image_build_receipt_sha256",
    "image_build_context_sha256",
    "image_dockerfile_sha256",
    "image_lock_sha256",
    "runtime_dependency_lock_sha256",
    "purchase_model",
    "capacity_reservation_id",
    "capacity_block_offering_id",
    "seeds",
    "mixed_profiles",
    "protected_outcomes_inspected",
    "selected_at",
}
_AMI_EVIDENCE_FIELDS = {
    "schema_version",
    "receipt_type",
    "region",
    "image_id",
    "owner_id",
    "name",
}
_ECR_EVIDENCE_FIELDS = {
    "schema_version",
    "receipt_type",
    "region",
    "registry_id",
    "repository_name",
    "image_digest",
    "image_uri",
}
_BUILD_EVIDENCE_FIELDS = {
    "schema_version",
    "receipt_type",
    "aws_account_id",
    "region",
    "container_image",
    "container_digest",
    "build_context_sha256",
    "dockerfile_sha256",
    "image_lock_sha256",
    "runtime_dependency_lock_sha256",
}


@dataclass(frozen=True)
class AllowedProfile:
    profile_id: str
    provider: str
    path: str
    sha256: str
    instance_type: str
    gpu_model: str


@dataclass(frozen=True)
class HardwareAmendment:
    amendment_id: str
    cohort_id: str
    sha256: str
    preregistration_sha256: str
    cohort_assignment_sha256: str
    allowed_profiles: tuple[AllowedProfile, ...]
    path: Path
    value: dict[str, object]

    def profile(self, profile_id: str) -> AllowedProfile:
        matches = [
            candidate
            for candidate in self.allowed_profiles
            if candidate.profile_id == profile_id
        ]
        if len(matches) != 1:
            _fail(
                "selected profile is not allowed by the hardware amendment",
                details={"profile_id": profile_id},
            )
        return matches[0]


@dataclass(frozen=True)
class ProviderSelection:
    amendment_sha256: str
    preregistration_sha256: str
    cohort_assignment_sha256: str
    cohort_id: str
    selected_profile_id: str
    provider: str
    profile_sha256: str
    instance_type: str
    gpu_model: str
    region: str
    aws_account_id: str
    ami_id: str
    ami_owner_id: str
    ami_name: str
    ami_describe_sha256: str
    container_image: str
    container_digest: str
    ecr_describe_sha256: str
    image_build_receipt_sha256: str
    image_build_context_sha256: str
    image_dockerfile_sha256: str
    image_lock_sha256: str
    runtime_dependency_lock_sha256: str
    purchase_model: str
    capacity_reservation_id: str | None
    capacity_block_offering_id: str | None
    seeds: tuple[int, ...]
    selected_at: str
    sha256: str
    path: Path | None
    value: dict[str, object]


def _fail(
    message: str,
    *,
    details: dict[str, object] | None = None,
    code: str = "PROVIDER_SELECTION_INVALID",
) -> None:
    raise MsctlError(code, message, details=details)


def _exact_fields(
    value: dict[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} has missing or unknown fields",
            details={
                "missing": sorted(expected - actual),
                "unknown": sorted(actual - expected),
            },
        )


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if path.is_symlink() or not path.is_file() or before.st_nlink != 1:
            _fail(f"{label} must be one singly linked regular file")
        if before.st_size <= 0 or before.st_size > _MAX_JSON_BYTES:
            _fail(f"{label} exceeds the bounded receipt size")
        data = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise MsctlError(
            "PROVIDER_SELECTION_INVALID",
            f"{label} cannot be read",
        ) from error
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
    ) or len(data) != after.st_size:
        _fail(f"{label} changed while being read")
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains a duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: _fail(
                f"{label} contains non-finite {constant}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "PROVIDER_SELECTION_INVALID",
            f"{label} must contain one valid UTF-8 JSON object",
        ) from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be a JSON object")
    return value


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object")
    return dict(value)


def _resolved_file(repo_root: Path, relative: object, *, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith(("/", "~"))
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        _fail(f"{label} must be a portable repository-relative path")
    candidate = repo_root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise MsctlError(
            "PROVIDER_SELECTION_INVALID",
            f"{label} escapes or is absent from the repository",
        ) from error
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"{label} must resolve to a regular non-symlink file")
    return candidate


def _sha256(value: object, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except MsctlError as error:
        raise MsctlError(
            "PROVIDER_SELECTION_INVALID",
            f"{label} must be lowercase SHA-256",
        ) from error


def _utc_timestamp(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith("Z")
        or len(value) > 40
    ):
        _fail(f"{label} must be an explicit RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise MsctlError(
            "PROVIDER_SELECTION_INVALID",
            f"{label} must be an explicit RFC 3339 UTC timestamp",
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed.year < 2020
        or parsed.year > 9998
    ):
        _fail(f"{label} must be an explicit RFC 3339 UTC timestamp")
    return value


def load_hardware_amendment(path: Path | str) -> HardwareAmendment:
    """Load the canonical amendment and verify every byte-level binding."""

    amendment_path = Path(path)
    data = _read_regular(amendment_path, label="hardware amendment")
    value = _json_object(data, label="hardware amendment")
    _exact_fields(value, _AMENDMENT_FIELDS, label="hardware amendment")
    if (
        value["schema_version"] != 3
        or value["amendment_id"] != AMENDMENT_ID
        or value["cohort_id"] != COHORT_ID
        or value["original_scientific_provider_assignment"]
        != AWS_P5_V3_PROFILE
        or value["supersedes_hardware_selection"] is not True
        or value["protected_outcomes_inspected"] is not False
        or amendment_path.name != "hardware-amendment-v3.json"
        or amendment_path.parent.name != "configs"
    ):
        _fail("hardware amendment identity is invalid")
    repo_root = amendment_path.parent.parent
    bindings = _object(value["bindings"], label="hardware amendment bindings")
    _exact_fields(
        bindings,
        {"cohort_assignment", "preregistration"},
        label="hardware amendment bindings",
    )
    binding_hashes: dict[str, str] = {}
    expected_paths = {
        "cohort_assignment": "configs/cohort-assignment-v3.json",
        "preregistration": "configs/preregistration-v3.yaml",
    }
    for name, expected_path in expected_paths.items():
        binding = _object(
            bindings[name],
            label=f"hardware amendment {name} binding",
        )
        _exact_fields(
            binding,
            _BINDING_FIELDS,
            label=f"hardware amendment {name} binding",
        )
        if binding["path"] != expected_path:
            _fail(f"hardware amendment {name} path is not canonical")
        expected_hash = _sha256(
            binding["sha256"],
            label=f"hardware amendment {name} hash",
        )
        bound_path = _resolved_file(
            repo_root,
            binding["path"],
            label=f"hardware amendment {name}",
        )
        actual_hash = hashlib.sha256(
            _read_regular(bound_path, label=f"bound {name}")
        ).hexdigest()
        if actual_hash != expected_hash:
            _fail(
                f"hardware amendment {name} binding is stale",
                details={"expected": expected_hash, "actual": actual_hash},
            )
        binding_hashes[name] = expected_hash

    rule = _object(value["selection_rule"], label="hardware selection rule")
    _exact_fields(rule, _RULE_FIELDS, label="hardware selection rule")
    if (
        rule["selected_profile_count"] != 1
        or rule["same_profile_for_all_pairs"] is not True
        or rule["mixed_profiles_forbidden"] is not True
        or rule["seeds"] != list(V3_SEEDS)
    ):
        _fail("hardware amendment selection rule is invalid")

    raw_profiles = value["allowed_profiles"]
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 2:
        _fail("hardware amendment must allow exactly two profiles")
    profiles: list[AllowedProfile] = []
    for index, raw in enumerate(raw_profiles):
        row = _object(raw, label=f"allowed profile[{index}]")
        _exact_fields(row, _PROFILE_FIELDS, label=f"allowed profile[{index}]")
        profile_id = row["profile_id"]
        if (
            not isinstance(profile_id, str)
            or row["provider"] != profile_id
            or row["path"] != f"cluster/profiles/{profile_id}.json"
        ):
            _fail("hardware amendment profile identity is invalid")
        profile_path = _resolved_file(
            repo_root,
            row["path"],
            label=f"allowed profile {profile_id}",
        )
        expected_hash = _sha256(
            row["sha256"],
            label=f"allowed profile {profile_id} hash",
        )
        actual_hash = hashlib.sha256(
            _read_regular(profile_path, label=f"allowed profile {profile_id}")
        ).hexdigest()
        if actual_hash != expected_hash:
            _fail(
                "hardware amendment profile binding is stale",
                details={"profile_id": profile_id},
            )
        loaded = load_profile(profile_path)
        if (
            getattr(loaded, "profile_id", None) != profile_id
            or getattr(loaded, "provider", None) != row["provider"]
            or getattr(loaded, "sha256", None) != expected_hash
            or getattr(loaded, "instance_type", None) != row["instance_type"]
            or getattr(loaded, "gpu_model", None) != row["gpu_model"]
        ):
            _fail("hardware amendment profile data conflicts with profile bytes")
        profiles.append(
            AllowedProfile(
                profile_id=profile_id,
                provider=str(row["provider"]),
                path=str(row["path"]),
                sha256=expected_hash,
                instance_type=str(row["instance_type"]),
                gpu_model=str(row["gpu_model"]),
            )
        )
    if tuple(profile.profile_id for profile in profiles) != V3_PROFILE_IDS:
        _fail("hardware amendment allowed profile set or order is invalid")
    return HardwareAmendment(
        amendment_id=AMENDMENT_ID,
        cohort_id=COHORT_ID,
        sha256=hashlib.sha256(data).hexdigest(),
        preregistration_sha256=binding_hashes["preregistration"],
        cohort_assignment_sha256=binding_hashes["cohort_assignment"],
        allowed_profiles=tuple(profiles),
        path=amendment_path,
        value=value,
    )


def _validate_profile_against_amendment(
    profile: object,
    amendment: HardwareAmendment,
) -> AllowedProfile:
    profile_id = getattr(profile, "profile_id", None)
    if not isinstance(profile_id, str) or profile_id not in V3_PROFILE_IDS:
        _fail("provider selection requires one closed v3 AWS GPU profile")
    allowed = amendment.profile(profile_id)
    if (
        getattr(profile, "provider", None) != allowed.provider
        or getattr(profile, "sha256", None) != allowed.sha256
        or getattr(profile, "instance_type", None) != allowed.instance_type
        or getattr(profile, "gpu_model", None) != allowed.gpu_model
        or tuple(getattr(profile, "assigned_seeds", ())) != V3_SEEDS
        or tuple(getattr(profile, "train_groups", ())) != (4, 4)
    ):
        _fail("selected profile conflicts with the hardware amendment")
    return allowed


def _validate_container(
    image: object,
    digest: object,
) -> tuple[str, str]:
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        _fail("container digest must be sha256-pinned")
    if (
        not isinstance(image, str)
        or len(image) > 512
        or image.count("@") != 1
        or image.rpartition("@")[2] != digest
        or "/" not in image.partition("@")[0]
        or "." not in image.partition("/")[0]
        or any(character.isspace() for character in image)
    ):
        _fail("container image must be an exact digest-pinned registry reference")
    return image, digest


def _load_evidence(
    path: Path | str,
    *,
    fields: set[str],
    label: str,
) -> tuple[dict[str, object], str]:
    candidate = Path(path)
    data = _read_regular(candidate, label=label)
    value = _json_object(data, label=label)
    _exact_fields(value, fields, label=label)
    if data != canonical_json(value) + b"\n":
        _fail(f"{label} must use canonical JSON plus one newline")
    return value, hashlib.sha256(data).hexdigest()


def _selection_evidence(
    *,
    amendment: HardwareAmendment,
    region: str,
    ami_id: str,
    container_image: str,
    container_digest: str,
    ami_evidence: Path | str,
    ecr_evidence: Path | str,
    image_build_receipt: Path | str,
) -> dict[str, object]:
    ami, ami_hash = _load_evidence(
        ami_evidence,
        fields=_AMI_EVIDENCE_FIELDS,
        label="AMI describe evidence",
    )
    ecr, ecr_hash = _load_evidence(
        ecr_evidence,
        fields=_ECR_EVIDENCE_FIELDS,
        label="ECR describe evidence",
    )
    build, build_hash = _load_evidence(
        image_build_receipt,
        fields=_BUILD_EVIDENCE_FIELDS,
        label="image build receipt",
    )
    image_match = _ECR_IMAGE_RE.fullmatch(container_image)
    if image_match is None:
        _fail("container image must be a private ECR digest reference")
    account_id = image_match.group("account")
    repository = image_match.group("repository")
    if (
        image_match.group("region") != region
        or image_match.group("digest") != container_digest
        or ami["schema_version"] != 1
        or ami["receipt_type"] != "memorysplit-aws-ami-describe-v1"
        or ami["region"] != region
        or ami["image_id"] != ami_id
        or not isinstance(ami["owner_id"], str)
        or _ACCOUNT_RE.fullmatch(ami["owner_id"]) is None
        or not isinstance(ami["name"], str)
        or not ami["name"]
        or len(ami["name"]) > 255
        or ecr["schema_version"] != 1
        or ecr["receipt_type"] != "memorysplit-aws-ecr-describe-v1"
        or ecr["region"] != region
        or ecr["registry_id"] != account_id
        or ecr["repository_name"] != repository
        or ecr["image_digest"] != container_digest
        or ecr["image_uri"] != container_image
        or build["schema_version"] != 1
        or build["receipt_type"] != "memorysplit-aws-gpu-image-build-v1"
        or build["aws_account_id"] != account_id
        or build["region"] != region
        or build["container_image"] != container_image
        or build["container_digest"] != container_digest
    ):
        _fail("provider evidence is stale, cross-account, or cross-region")
    repo_root = amendment.path.parent.parent
    expected_files = {
        "dockerfile_sha256": repo_root / "containers/aws-gpu/Dockerfile",
        "image_lock_sha256": repo_root / "containers/aws-gpu/image.lock.json",
        "runtime_dependency_lock_sha256": (
            repo_root / "containers/aws-gpu/requirements.lock"
        ),
    }
    for field, local_path in expected_files.items():
        expected = _sha256(build[field], label=f"image build {field}")
        if hashlib.sha256(_read_regular(local_path, label=field)).hexdigest() != expected:
            _fail(f"image build {field} does not bind reviewed repository bytes")
    context_sha256 = _sha256(
        build["build_context_sha256"],
        label="image build context",
    )
    try:
        from scripts.build_aws_gpu_image import build_context_sha256

        actual_context_sha256 = build_context_sha256(
            repo_root / "containers/aws-gpu"
        )
    except (ImportError, OSError, ValueError) as error:
        raise MsctlError(
            "PROVIDER_SELECTION_INVALID",
            "reviewed image build context cannot be validated",
        ) from error
    if actual_context_sha256 != context_sha256:
        _fail("image build context hash does not bind reviewed repository bytes")
    return {
        "aws_account_id": account_id,
        "ami_owner_id": ami["owner_id"],
        "ami_name": ami["name"],
        "ami_describe_sha256": ami_hash,
        "ecr_describe_sha256": ecr_hash,
        "image_build_receipt_sha256": build_hash,
        "image_build_context_sha256": context_sha256,
        "image_dockerfile_sha256": build["dockerfile_sha256"],
        "image_lock_sha256": build["image_lock_sha256"],
        "runtime_dependency_lock_sha256": build[
            "runtime_dependency_lock_sha256"
        ],
    }


def _capacity_identity(
    profile: object,
    *,
    capacity_reservation_id: str | None,
    capacity_block_offering_id: str | None,
) -> dict[str, object]:
    purchase_model = getattr(profile, "purchase_model", None)
    if purchase_model == "on_demand":
        if capacity_reservation_id is not None or capacity_block_offering_id is not None:
            _fail("on-demand selection must not include Capacity Block identity")
    elif purchase_model == "capacity_block":
        if (
            not isinstance(capacity_reservation_id, str)
            or _CAPACITY_ID_RE.fullmatch(capacity_reservation_id) is None
            or not isinstance(capacity_block_offering_id, str)
            or _CAPACITY_ID_RE.fullmatch(capacity_block_offering_id) is None
        ):
            _fail(
                "Capacity Block selection requires exact reservation and offering IDs"
            )
    else:
        _fail("selected profile purchase model is unsupported")
    return {
        "purchase_model": purchase_model,
        "capacity_reservation_id": capacity_reservation_id,
        "capacity_block_offering_id": capacity_block_offering_id,
    }


def create_provider_selection(
    *,
    profile: object,
    amendment: HardwareAmendment,
    region: str,
    ami_id: str,
    container_image: str,
    container_digest: str,
    ami_evidence: Path | str,
    ecr_evidence: Path | str,
    image_build_receipt: Path | str,
    capacity_reservation_id: str | None,
    capacity_block_offering_id: str | None,
    selected_at: str,
) -> dict[str, object]:
    """Render one canonical provider-selection receipt without writing it."""

    allowed = _validate_profile_against_amendment(profile, amendment)
    if not isinstance(region, str) or _REGION_RE.fullmatch(region) is None:
        _fail("provider selection region is invalid")
    if not isinstance(ami_id, str) or _AMI_RE.fullmatch(ami_id) is None:
        _fail("provider selection AMI must be an immutable AMI ID")
    image, digest = _validate_container(container_image, container_digest)
    evidence = _selection_evidence(
        amendment=amendment,
        region=region,
        ami_id=ami_id,
        container_image=image,
        container_digest=digest,
        ami_evidence=ami_evidence,
        ecr_evidence=ecr_evidence,
        image_build_receipt=image_build_receipt,
    )
    capacity = _capacity_identity(
        profile,
        capacity_reservation_id=capacity_reservation_id,
        capacity_block_offering_id=capacity_block_offering_id,
    )
    timestamp = _utc_timestamp(selected_at, label="provider selection selected_at")
    return {
        "schema_version": 3,
        "receipt_type": SELECTION_RECEIPT_TYPE,
        "amendment_sha256": amendment.sha256,
        "preregistration_sha256": amendment.preregistration_sha256,
        "cohort_assignment_sha256": amendment.cohort_assignment_sha256,
        "cohort_id": amendment.cohort_id,
        "selected_profile_id": allowed.profile_id,
        "provider": allowed.provider,
        "profile_sha256": allowed.sha256,
        "instance_type": allowed.instance_type,
        "gpu_model": allowed.gpu_model,
        "region": region,
        "aws_account_id": evidence["aws_account_id"],
        "ami_id": ami_id,
        "ami_owner_id": evidence["ami_owner_id"],
        "ami_name": evidence["ami_name"],
        "ami_describe_sha256": evidence["ami_describe_sha256"],
        "container_image": image,
        "container_digest": digest,
        "ecr_describe_sha256": evidence["ecr_describe_sha256"],
        "image_build_receipt_sha256": evidence[
            "image_build_receipt_sha256"
        ],
        "image_build_context_sha256": evidence["image_build_context_sha256"],
        "image_dockerfile_sha256": evidence["image_dockerfile_sha256"],
        "image_lock_sha256": evidence["image_lock_sha256"],
        "runtime_dependency_lock_sha256": evidence[
            "runtime_dependency_lock_sha256"
        ],
        **capacity,
        "seeds": list(V3_SEEDS),
        "mixed_profiles": False,
        "protected_outcomes_inspected": False,
        "selected_at": timestamp,
    }


def validate_provider_selection(
    value: object,
    *,
    amendment: HardwareAmendment,
    profile: object,
    path: Path | None = None,
    sha256: str | None = None,
) -> ProviderSelection:
    """Validate one decoded receipt against current amendment and profile bytes."""

    receipt = _object(value, label="provider selection receipt")
    _exact_fields(receipt, _RECEIPT_FIELDS, label="provider selection receipt")
    allowed = _validate_profile_against_amendment(profile, amendment)
    image, digest = _validate_container(
        receipt["container_image"],
        receipt["container_digest"],
    )
    timestamp = _utc_timestamp(
        receipt["selected_at"],
        label="provider selection selected_at",
    )
    image_match = _ECR_IMAGE_RE.fullmatch(image)
    capacity = _capacity_identity(
        profile,
        capacity_reservation_id=receipt["capacity_reservation_id"],
        capacity_block_offering_id=receipt["capacity_block_offering_id"],
    )
    for field in (
        "ami_describe_sha256",
        "ecr_describe_sha256",
        "image_build_receipt_sha256",
        "image_build_context_sha256",
        "image_dockerfile_sha256",
        "image_lock_sha256",
        "runtime_dependency_lock_sha256",
    ):
        _sha256(receipt[field], label=f"provider selection {field}")
    if (
        receipt["schema_version"] != 3
        or receipt["receipt_type"] != SELECTION_RECEIPT_TYPE
        or receipt["amendment_sha256"] != amendment.sha256
        or receipt["preregistration_sha256"]
        != amendment.preregistration_sha256
        or receipt["cohort_assignment_sha256"]
        != amendment.cohort_assignment_sha256
        or receipt["cohort_id"] != amendment.cohort_id
        or receipt["selected_profile_id"] != allowed.profile_id
        or receipt["provider"] != allowed.provider
        or receipt["profile_sha256"] != allowed.sha256
        or receipt["instance_type"] != allowed.instance_type
        or receipt["gpu_model"] != allowed.gpu_model
        or not isinstance(receipt["region"], str)
        or _REGION_RE.fullmatch(receipt["region"]) is None
        or not isinstance(receipt["aws_account_id"], str)
        or _ACCOUNT_RE.fullmatch(receipt["aws_account_id"]) is None
        or not isinstance(receipt["ami_id"], str)
        or _AMI_RE.fullmatch(receipt["ami_id"]) is None
        or not isinstance(receipt["ami_owner_id"], str)
        or _ACCOUNT_RE.fullmatch(receipt["ami_owner_id"]) is None
        or not isinstance(receipt["ami_name"], str)
        or not receipt["ami_name"]
        or len(receipt["ami_name"]) > 255
        or image_match is None
        or image_match.group("account") != receipt["aws_account_id"]
        or image_match.group("region") != receipt["region"]
        or image_match.group("digest") != digest
        or receipt["purchase_model"] != capacity["purchase_model"]
        or receipt["seeds"] != list(V3_SEEDS)
        or receipt["mixed_profiles"] is not False
        or receipt["protected_outcomes_inspected"] is not False
    ):
        _fail(
            "provider selection receipt is stale or cross-profile",
            details={
                "selected_profile_id": receipt.get("selected_profile_id"),
                "expected_profile_id": allowed.profile_id,
            },
        )
    receipt_sha256 = sha256 or hashlib.sha256(
        canonical_json(receipt) + b"\n"
    ).hexdigest()
    if _SHA256_RE.fullmatch(receipt_sha256) is None:
        _fail("provider selection receipt hash is invalid")
    return ProviderSelection(
        amendment_sha256=amendment.sha256,
        preregistration_sha256=amendment.preregistration_sha256,
        cohort_assignment_sha256=amendment.cohort_assignment_sha256,
        cohort_id=amendment.cohort_id,
        selected_profile_id=allowed.profile_id,
        provider=allowed.provider,
        profile_sha256=allowed.sha256,
        instance_type=allowed.instance_type,
        gpu_model=allowed.gpu_model,
        region=str(receipt["region"]),
        aws_account_id=str(receipt["aws_account_id"]),
        ami_id=str(receipt["ami_id"]),
        ami_owner_id=str(receipt["ami_owner_id"]),
        ami_name=str(receipt["ami_name"]),
        ami_describe_sha256=str(receipt["ami_describe_sha256"]),
        container_image=image,
        container_digest=digest,
        ecr_describe_sha256=str(receipt["ecr_describe_sha256"]),
        image_build_receipt_sha256=str(
            receipt["image_build_receipt_sha256"]
        ),
        image_build_context_sha256=str(receipt["image_build_context_sha256"]),
        image_dockerfile_sha256=str(receipt["image_dockerfile_sha256"]),
        image_lock_sha256=str(receipt["image_lock_sha256"]),
        runtime_dependency_lock_sha256=str(
            receipt["runtime_dependency_lock_sha256"]
        ),
        purchase_model=str(receipt["purchase_model"]),
        capacity_reservation_id=(
            str(receipt["capacity_reservation_id"])
            if receipt["capacity_reservation_id"] is not None
            else None
        ),
        capacity_block_offering_id=(
            str(receipt["capacity_block_offering_id"])
            if receipt["capacity_block_offering_id"] is not None
            else None
        ),
        seeds=V3_SEEDS,
        selected_at=timestamp,
        sha256=receipt_sha256,
        path=path,
        value=receipt,
    )


def load_provider_selection(
    path: Path | str,
    *,
    amendment: HardwareAmendment | Path | str,
    profile: object,
) -> ProviderSelection:
    """Load one canonical selection receipt and reject stale bindings."""

    selection_path = Path(path)
    data = _read_regular(selection_path, label="provider selection receipt")
    value = _json_object(data, label="provider selection receipt")
    if data != canonical_json(value) + b"\n":
        _fail("provider selection receipt must use canonical JSON plus one newline")
    loaded_amendment = (
        amendment
        if isinstance(amendment, HardwareAmendment)
        else load_hardware_amendment(amendment)
    )
    return validate_provider_selection(
        value,
        amendment=loaded_amendment,
        profile=profile,
        path=selection_path,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def write_provider_selection(
    path: Path | str,
    value: dict[str, object],
) -> Path:
    """Publish canonical receipt bytes with exclusive-create semantics."""

    destination = Path(path)
    directory_fd = open_directory(
        destination.parent,
        label="provider selection output",
        create=True,
    )
    temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
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
        payload = canonical_json(value) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            rename_noreplace_at(
                directory_fd,
                temporary,
                directory_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise MsctlError(
                "PROVIDER_SELECTION_EXISTS",
                "refusing to replace an existing provider selection receipt",
                details={"path": str(destination)},
            ) from error
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


def plan_provider_selection(
    *,
    profile: object,
    amendment_path: Path | str,
    region: str,
    ami_id: str,
    container_image: str,
    container_digest: str,
    ami_evidence: Path | str,
    ecr_evidence: Path | str,
    image_build_receipt: Path | str,
    capacity_reservation_id: str | None,
    capacity_block_offering_id: str | None,
    selected_at: str,
    out: Path | str,
    apply: bool,
) -> dict[str, object]:
    """Render, validate, and optionally publish one selection receipt."""

    amendment = load_hardware_amendment(amendment_path)
    receipt = create_provider_selection(
        profile=profile,
        amendment=amendment,
        region=region,
        ami_id=ami_id,
        container_image=container_image,
        container_digest=container_digest,
        ami_evidence=ami_evidence,
        ecr_evidence=ecr_evidence,
        image_build_receipt=image_build_receipt,
        capacity_reservation_id=capacity_reservation_id,
        capacity_block_offering_id=capacity_block_offering_id,
        selected_at=selected_at,
    )
    validated = validate_provider_selection(
        receipt,
        amendment=amendment,
        profile=profile,
    )
    result = {
        "receipt": receipt,
        "receipt_sha256": validated.sha256,
        "out": str(Path(out)),
        "published": False,
    }
    if apply:
        write_provider_selection(out, receipt)
        result["published"] = True
    return result
