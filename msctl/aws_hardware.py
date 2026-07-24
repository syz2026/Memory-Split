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

from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.fsutil import open_directory, rename_noreplace_at


AWS_HARDWARE_AMENDMENT_PATH = "configs/aws-hardware-amendment-v3.json"
P5_PROFILE_PATH = "cluster/profiles/aws-p5.48xlarge-v3.json"
P6_PROFILE_PATH = "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
PREREGISTRATION_PATH = "configs/preregistration-v3.yaml"
COHORT_ASSIGNMENT_PATH = "configs/cohort-assignment-v3.json"
AWS_HARDWARE_AMENDMENT_SHA256 = (
    "9d6bbaedfe2520bd6ce10957e2c8e923a624144c0feb4b19c3764f71040be755"
)

_MAX_JSON_BYTES = 65_536
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
    "f22ccf259e30b07b7ad9d848723ad10e59092cbf5ea6e3aceca7a556279ce681"
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

    def profile_by_id(self, profile_id: str) -> HardwareProfileBinding:
        matches = tuple(
            profile for profile in self.profiles if profile.profile_id == profile_id
        )
        if len(matches) != 1:
            raise ValueError("profile is not eligible under the hardware amendment")
        return matches[0]


@dataclass(frozen=True)
class AwsProviderSelectionReceipt:
    """One immutable hardware/runtime choice for every protected run cell."""

    schema_version: int
    receipt_type: str
    amendment: ArtifactBinding
    cohort_id: str
    preregistration_sha256: str
    cohort_assignment_sha256: str
    profile: HardwareProfileBinding
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    train_groups: tuple[int, int]
    runtime_lock_sha256: str
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


def _regular_bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def validate_aws_hardware_amendment_files(
    amendment: AwsHardwareAmendment,
    *,
    repo_root: Path | str,
) -> None:
    """Verify every append-only binding against one repository root."""

    root = Path(repo_root).resolve(strict=True)
    bindings = (
        amendment.preregistration,
        amendment.cohort_assignment,
        *amendment.profiles,
    )
    for binding in bindings:
        candidate = root.joinpath(*binding.path.split("/"))
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("amendment binding escapes repository root") from error
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
        validate_aws_hardware_amendment_files(amendment, repo_root=repo_root)
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
        "aws-p6-b300.48xlarge-v3": "capacity_block",
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


def parse_provider_selection_receipt_bytes(
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
            {"runtime_lock_sha256", "ami_id", "container_image_digest"}
        ),
        label="provider selection runtime",
    )
    runtime_lock_sha256 = _sha256(
        runtime["runtime_lock_sha256"],
        label="provider selection runtime-lock SHA-256",
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
        cohort_id=cohort_id,
        preregistration_sha256=preregistration_sha256,
        cohort_assignment_sha256=cohort_assignment_sha256,
        profile=profile,
        seeds=amendment.seeds,
        arms=amendment.arms,
        train_groups=amendment.train_groups,
        runtime_lock_sha256=runtime_lock_sha256,
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


parse_aws_provider_selection_receipt_bytes = (
    parse_provider_selection_receipt_bytes
)


def canonical_provider_selection_receipt_bytes(
    value: object,
    *,
    amendment: AwsHardwareAmendment,
) -> bytes:
    """Encode and validate one provider selection in its sole byte form."""

    data = _canonical_json(value)
    parse_provider_selection_receipt_bytes(data, amendment=amendment)
    return data


def load_provider_selection_receipt(
    path: Path | str,
    *,
    amendment: AwsHardwareAmendment,
) -> AwsProviderSelectionReceipt:
    """Load one canonical provider selection from a regular file."""

    return parse_provider_selection_receipt_bytes(
        _regular_bytes(Path(path), label="provider selection receipt"),
        amendment=amendment,
    )


load_aws_provider_selection_receipt = load_provider_selection_receipt


def write_provider_selection_receipt(
    path: Path | str,
    value: object,
    *,
    amendment: AwsHardwareAmendment,
) -> AwsProviderSelectionReceipt:
    """Atomically publish one canonical selection without replacement."""

    data = canonical_provider_selection_receipt_bytes(
        value,
        amendment=amendment,
    )
    destination = Path(path)
    name = destination.name
    if not name or name in {".", ".."}:
        raise ValueError("provider selection destination name is invalid")
    directory_fd = open_directory(
        destination.parent,
        label="provider selection parent",
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
    return parse_provider_selection_receipt_bytes(data, amendment=amendment)


write_aws_provider_selection_receipt = write_provider_selection_receipt


def validate_resume_hardware_binding(
    receipt: AwsProviderSelectionReceipt,
    *,
    amendment_sha256: str,
    provider_selection_sha256: str,
    profile_sha256: str,
    runtime_lock_sha256: str,
    ami_id: str,
    container_image_digest: str,
    account_id: str,
    region: str,
    availability_zone: str,
    purchase_model: str,
    seed: int,
    arm: str,
) -> None:
    """Reject resume/run context that differs from the cohort-wide selection."""

    if not isinstance(receipt, AwsProviderSelectionReceipt):
        raise TypeError("provider selection receipt must be parsed first")
    if amendment_sha256 != receipt.amendment.sha256:
        raise ValueError("hardware amendment binding differs from selection")
    if provider_selection_sha256 != receipt.sha256:
        raise ValueError("provider-selection receipt binding differs")
    if profile_sha256 != receipt.profile.sha256:
        raise ValueError("hardware profile differs from cohort selection")
    if runtime_lock_sha256 != receipt.runtime_lock_sha256:
        raise ValueError("runtime-lock binding differs from cohort selection")
    if ami_id != receipt.ami_id:
        raise ValueError("AMI binding differs from cohort selection runtime")
    if container_image_digest != receipt.container_image_digest:
        raise ValueError("image digest differs from cohort selection runtime")
    if (
        account_id != receipt.account_id
        or region != receipt.region
        or availability_zone != receipt.availability_zone
        or purchase_model != receipt.purchase_model
    ):
        raise ValueError("AWS placement differs from cohort selection")
    if type(seed) is not int or seed not in receipt.seeds:
        raise ValueError("seed is outside the selected protected cohort")
    if arm not in receipt.arms:
        raise ValueError("arm is outside the selected protected cohort")


validate_provider_selection_binding = validate_resume_hardware_binding


__all__ = [
    "AWS_HARDWARE_AMENDMENT_PATH",
    "AWS_HARDWARE_AMENDMENT_SHA256",
    "ArtifactBinding",
    "AwsHardwareAmendment",
    "AwsProviderSelectionReceipt",
    "COHORT_ASSIGNMENT_PATH",
    "HardwareProfileBinding",
    "P5_PROFILE_PATH",
    "P6_PROFILE_PATH",
    "PREREGISTRATION_PATH",
    "canonical_provider_selection_receipt_bytes",
    "load_aws_hardware_amendment",
    "load_aws_provider_selection_receipt",
    "load_provider_selection_receipt",
    "parse_aws_provider_selection_receipt_bytes",
    "parse_aws_hardware_amendment_bytes",
    "parse_provider_selection_receipt_bytes",
    "validate_aws_hardware_amendment_files",
    "validate_provider_selection_binding",
    "validate_resume_hardware_binding",
    "write_aws_provider_selection_receipt",
    "write_provider_selection_receipt",
]
