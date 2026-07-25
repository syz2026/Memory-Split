"""Read-only AWS corpus-builder preflight and canonical launch intent."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path, PurePosixPath
from typing import TypeVar
from urllib.parse import urlsplit

from .contracts import (
    LAUNCH_INTENT_FORMAT,
    CorpusBuilderProfile,
    LaunchIntent,
    S3ObjectVersion,
    corpus_builder_profile_to_bytes,
    launch_intent_to_bytes,
    load_corpus_builder_profile,
    s3_object_version_from_dict,
)
from .package import (
    ARCHIVE_NAME,
    PACKAGE_FORMAT,
    _REQUIRED_PACKAGE_PATHS,
)
from .s3 import verify_exact_object


PREFLIGHT_CHECKS = (
    "profile-canonical-sha256",
    "exact-s3-versions",
    "production-software-gate",
    "account-and-region",
    "ami-identity",
    "instance-type-availability",
    "launch-template-version",
    "bootstrap-user-data-sha256",
    "private-network-and-security-group",
    "instance-profile-and-builder-role",
    "bucket-and-kms",
    "linux-on-demand-price",
    "maximum-compute-cost",
    "ec2-run-instances-dry-run",
)

_ACCOUNT_ID = "056956104102"
_REGION = "us-east-1"
_INSTANCE_TYPE = "i4i.16xlarge"
_PRICING_LOCATION = "US East (N. Virginia)"
_INTENT_LIFETIME = timedelta(minutes=30)
_MAX_GATE_RECEIPT_BYTES = 64 * 1024 * 1024

_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_LAUNCH_TEMPLATE_RE = re.compile(r"^lt-[0-9a-f]{8,17}$")
_LAUNCH_TEMPLATE_VERSION_RE = re.compile(r"^[1-9][0-9]*$")
_SUBNET_RE = re.compile(r"^subnet-[0-9a-f]{8,17}$")
_SECURITY_GROUP_RE = re.compile(r"^sg-[0-9a-f]{8,17}$")
_VPC_RE = re.compile(r"^vpc-[0-9a-f]{8,17}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_KMS_ARN_RE = re.compile(
    r"^arn:aws:kms:us-east-1:056956104102:key/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_INSTANCE_PROFILE_ARN_RE = re.compile(
    r"^arn:aws:iam::056956104102:instance-profile/"
    r"[A-Za-z0-9+=,.@_-]{1,128}$"
)
_ROLE_ARN_RE = re.compile(
    r"^arn:aws:iam::056956104102:role/[A-Za-z0-9+=,.@_-]{1,64}$"
)

_T = TypeVar("_T")


class PreflightError(ValueError):
    """A fail-closed read-only preflight error."""


@dataclass(frozen=True)
class PreflightRequest:
    profile_path: Path
    package: S3ObjectVersion
    source_manifest: S3ObjectVersion
    software_gate_receipt: Path
    stack_outputs: Mapping[str, str]
    ami_id: str
    ami_owner_id: str
    expected_bootstrap_user_data_sha256: str


@dataclass(frozen=True)
class AwsClients:
    sts: object
    ec2: object
    iam: object
    s3: object
    kms: object
    pricing: object


@dataclass(frozen=True)
class PreflightResult:
    intent: LaunchIntent
    intent_sha256: str
    checks: tuple[str, ...]


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _one_mapping(
    response: object,
    field: str,
    *,
    label: str,
) -> Mapping[str, object]:
    value = _mapping(response, label=f"{label} response").get(field)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{label} must return exactly one {field} entry")
    return _mapping(value[0], label=f"{label} {field} entry")


def _stack_output(
    outputs: Mapping[str, str],
    name: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(outputs, Mapping):
        raise ValueError("stack outputs must be a mapping")
    value = outputs.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"stack output {name} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"stack output {name} has invalid identity syntax")
    return value


def _snapshot_stack_outputs(outputs: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(outputs, Mapping):
        raise ValueError("stack outputs must be a mapping")
    snapshot: dict[str, str] = {}
    for key, value in outputs.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError(
                "stack output names and values must be non-empty strings"
            )
        snapshot[key] = value
    return snapshot


def _record_dict(value: S3ObjectVersion) -> dict[str, object]:
    return {
        "bytes": value.bytes,
        "etag": value.etag,
        "kms_key_arn": value.kms_key_arn,
        "sha256": value.sha256,
        "sse_algorithm": value.sse_algorithm,
        "uri": value.uri,
        "version_id": value.version_id,
    }


def _build_id(value: S3ObjectVersion) -> str:
    parts = urlsplit(value.uri).path.removeprefix("/").split("/")
    if len(parts) < 4:
        raise ValueError("S3 input does not name an object below a build ID")
    return parts[2]


def _load_profile(path: Path) -> tuple[CorpusBuilderProfile, str]:
    profile = load_corpus_builder_profile(path)
    canonical = corpus_builder_profile_to_bytes(profile)
    if Path(path).read_bytes() != canonical:
        raise ValueError("profile bytes changed during canonical validation")
    return profile, hashlib.sha256(canonical).hexdigest()


def _validate_input_records(
    package: S3ObjectVersion,
    source_manifest: S3ObjectVersion,
) -> None:
    for label, value in (
        ("package", package),
        ("source manifest", source_manifest),
    ):
        parsed = s3_object_version_from_dict(_record_dict(value))
        if parsed != value:
            raise ValueError(f"{label} authority changed during validation")
    if _build_id(package) != _build_id(source_manifest):
        raise ValueError("package and source manifest build IDs differ")
    if package.kms_key_arn != source_manifest.kms_key_arn:
        raise ValueError("package and source manifest KMS keys differ")


def _exact_package_revision(s3: object, package: S3ObjectVersion) -> str:
    parsed = urlsplit(package.uri)
    response = _mapping(
        s3.head_object(
            Bucket=parsed.netloc,
            Key=parsed.path.removeprefix("/"),
            VersionId=package.version_id,
        ),
        label="exact package metadata response",
    )
    raw_metadata = response.get("Metadata")
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(
            'exact package object Metadata["revision"] is required '
            "and must be immutable"
        )
    metadata = raw_metadata
    revision = metadata.get("revision")
    if (
        response.get("VersionId") != package.version_id
        or metadata.get("sha256") != package.sha256
    ):
        raise ValueError("exact package object metadata authority drift")
    if (
        not isinstance(revision, str)
        or _OBJECT_ID_RE.fullmatch(revision) is None
    ):
        raise ValueError(
            'exact package object Metadata["revision"] is required '
            "and must be immutable"
        )
    return revision


def _verify_inputs(
    aws: AwsClients,
    package: S3ObjectVersion,
    source_manifest: S3ObjectVersion,
) -> str:
    _validate_input_records(package, source_manifest)
    verify_exact_object(aws.s3, package)
    verify_exact_object(aws.s3, source_manifest)
    return _exact_package_revision(aws.s3, package)


def _read_regular_file(path: Path, *, label: str, maximum: int) -> bytes:
    requested = Path(os.path.abspath(os.fspath(path)))
    try:
        if requested.resolve(strict=True) != requested:
            raise ValueError(f"{label} must not traverse a symlink")
    except OSError as error:
        raise ValueError(f"{label} must be an existing regular file") from error
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file") from error
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_size <= 0
            or initial.st_size > maximum
        ):
            raise ValueError(f"{label} has invalid file type or size")
        chunks: list[bytes] = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was read")
        final = os.fstat(descriptor)
        if (
            (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        ):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"software gate receipt repeats field: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _safe_package_member(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("software gate member path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("software gate member path is invalid")
    return value


def _verify_software_gate(
    path: Path,
    package: S3ObjectVersion,
    *,
    expected_revision: str | None = None,
) -> str:
    payload = _read_regular_file(
        path,
        label="software gate receipt",
        maximum=_MAX_GATE_RECEIPT_BYTES,
    )
    if not payload.endswith(b"\n"):
        raise ValueError("software gate receipt must end with one newline")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(
                    f"software gate receipt contains non-finite value: {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "software gate receipt must contain valid ASCII JSON"
        ) from error
    if _canonical_json_bytes(value) != payload:
        raise ValueError("software gate receipt must be canonical JSON")
    receipt = _mapping(value, label="software gate receipt")
    if set(receipt) != {
        "archive",
        "format",
        "members",
        "revision",
        "schema_version",
    }:
        raise ValueError("software gate receipt fields do not match package schema")
    if receipt["format"] != PACKAGE_FORMAT or receipt["schema_version"] != 1:
        raise ValueError("software gate receipt format or schema version drift")
    revision = receipt["revision"]
    if (
        not isinstance(revision, str)
        or _OBJECT_ID_RE.fullmatch(revision) is None
    ):
        raise ValueError("software gate receipt revision is not immutable")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            "software gate receipt revision does not match the exact package"
        )
    archive = _mapping(receipt["archive"], label="software gate archive")
    if set(archive) != {"bytes", "path", "sha256"}:
        raise ValueError("software gate archive fields do not match package schema")
    if (
        archive["path"] != ARCHIVE_NAME
        or type(archive["bytes"]) is not int
        or archive["bytes"] != package.bytes
        or archive["sha256"] != package.sha256
    ):
        raise ValueError("software gate archive does not bind the exact package")
    members = receipt["members"]
    if not isinstance(members, list) or not members:
        raise ValueError("software gate receipt must list package members")
    paths: list[str] = []
    for index, raw_row in enumerate(members):
        row = _mapping(raw_row, label=f"software gate member {index}")
        if set(row) != {"bytes", "mode", "object_id", "path", "sha256"}:
            raise ValueError("software gate member fields do not match package schema")
        member = _safe_package_member(row["path"])
        if (
            type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or row["mode"] not in {"0644", "0755"}
            or not isinstance(row["object_id"], str)
            or _OBJECT_ID_RE.fullmatch(row["object_id"]) is None
            or not isinstance(row["sha256"], str)
            or _SHA256_RE.fullmatch(row["sha256"]) is None
        ):
            raise ValueError(f"software gate member authority is invalid: {member}")
        paths.append(member)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("software gate package members must be sorted and unique")
    missing = sorted(_REQUIRED_PACKAGE_PATHS - set(paths))
    if missing:
        raise ValueError(
            f"software gate package omits required production member: {missing[0]}"
        )
    return revision


def _verify_account_region(aws: AwsClients) -> None:
    meta = getattr(aws.ec2, "meta", None)
    if getattr(meta, "region_name", None) != _REGION:
        raise ValueError(f"EC2 client region must be exactly {_REGION}")
    response = _mapping(
        aws.sts.get_caller_identity(),
        label="STS caller identity response",
    )
    if response.get("Account") != _ACCOUNT_ID:
        raise ValueError(f"caller account must be exactly {_ACCOUNT_ID}")


def _parse_aws_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is not a valid timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _verify_ami(
    aws: AwsClients,
    *,
    ami_id: str,
    ami_owner_id: str,
    now: datetime,
) -> str:
    if _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("AMI ID must be immutable")
    if _ACCOUNT_RE.fullmatch(ami_owner_id) is None:
        raise ValueError("AMI owner ID must be a 12-digit account")
    image = _one_mapping(
        aws.ec2.describe_images(
            ImageIds=[ami_id],
            Owners=[ami_owner_id],
        ),
        "Images",
        label="AMI lookup",
    )
    if image.get("ImageId") != ami_id or image.get("OwnerId") != ami_owner_id:
        raise ValueError("AMI ID or owner drift")
    if image.get("Architecture") != "x86_64":
        raise ValueError("AMI architecture must be x86_64")
    if image.get("State") != "available":
        raise ValueError("AMI state must be available")
    if image.get("RootDeviceType") != "ebs":
        raise ValueError("AMI root device must be EBS")
    root_name = image.get("RootDeviceName")
    if (
        not isinstance(root_name, str)
        or not root_name.startswith("/dev/")
        or root_name in {"/dev/", "/dev/."}
    ):
        raise ValueError("AMI root device name is invalid")
    created = _parse_aws_timestamp(
        image.get("CreationDate"),
        label="AMI creation date",
    )
    if created > now:
        raise ValueError("AMI creation date is in the future")
    return root_name


def _verify_offering(
    aws: AwsClients,
    *,
    location_type: str,
    location: str,
) -> None:
    response = _mapping(
        aws.ec2.describe_instance_type_offerings(
            Filters=[
                {"Name": "instance-type", "Values": [_INSTANCE_TYPE]},
                {"Name": "location", "Values": [location]},
            ],
            LocationType=location_type,
        ),
        label=f"{location_type} offering response",
    )
    if response.get("NextToken") not in {None, ""}:
        raise ValueError(f"{location_type} offering response was truncated")
    offerings = response.get("InstanceTypeOfferings")
    if not isinstance(offerings, list) or len(offerings) != 1:
        raise ValueError(
            f"{_INSTANCE_TYPE} is not uniquely offered in {location}"
        )
    offering = _mapping(
        offerings[0],
        label=f"{location_type} offering",
    )
    if (
        offering.get("InstanceType") != _INSTANCE_TYPE
        or offering.get("Location") != location
        or offering.get("LocationType") != location_type
    ):
        raise ValueError(f"{location_type} instance offering drift")


def _verify_availability(
    aws: AwsClients,
    outputs: Mapping[str, str],
) -> tuple[str, Mapping[str, object]]:
    subnet_id = _stack_output(outputs, "PrivateSubnetId", pattern=_SUBNET_RE)
    subnet = _one_mapping(
        aws.ec2.describe_subnets(SubnetIds=[subnet_id]),
        "Subnets",
        label="private subnet lookup",
    )
    if subnet.get("SubnetId") != subnet_id or subnet.get("State") != "available":
        raise ValueError("private subnet identity or state drift")
    availability_zone = subnet.get("AvailabilityZone")
    if (
        not isinstance(availability_zone, str)
        or re.fullmatch(r"us-east-1[a-z]", availability_zone) is None
    ):
        raise ValueError("private subnet Availability Zone is invalid")
    _verify_offering(
        aws,
        location_type="region",
        location=_REGION,
    )
    _verify_offering(
        aws,
        location_type="availability-zone",
        location=availability_zone,
    )
    return subnet_id, subnet


def _snapshot_and_verify_availability(
    aws: AwsClients,
    outputs: Mapping[str, str],
) -> tuple[dict[str, str], str, Mapping[str, object]]:
    snapshot = _snapshot_stack_outputs(outputs)
    subnet_id, subnet = _verify_availability(aws, snapshot)
    return snapshot, subnet_id, subnet


def _verify_launch_template(
    aws: AwsClients,
    outputs: Mapping[str, str],
    *,
    ami_id: str,
    ami_root_device_name: str,
    profile: CorpusBuilderProfile,
) -> tuple[str, str, Mapping[str, object]]:
    template_id = _stack_output(
        outputs,
        "LaunchTemplateId",
        pattern=_LAUNCH_TEMPLATE_RE,
    )
    version = _stack_output(
        outputs,
        "LaunchTemplateVersion",
        pattern=_LAUNCH_TEMPLATE_VERSION_RE,
    )
    item = _one_mapping(
        aws.ec2.describe_launch_template_versions(
            LaunchTemplateId=template_id,
            Versions=[version],
        ),
        "LaunchTemplateVersions",
        label="launch template lookup",
    )
    if (
        item.get("LaunchTemplateId") != template_id
        or type(item.get("VersionNumber")) is not int
        or item.get("VersionNumber") != int(version)
    ):
        raise ValueError("launch template ID or version drift")
    data = _mapping(
        item.get("LaunchTemplateData"),
        label="launch template data",
    )
    if data.get("ImageId") != ami_id:
        raise ValueError("launch template AMI drift")
    if data.get("InstanceType") != profile.instance_type:
        raise ValueError("launch template instance type drift")
    if data.get("InstanceInitiatedShutdownBehavior") != "terminate":
        raise ValueError("launch template shutdown behavior must terminate")
    if data.get("DisableApiTermination") is not False:
        raise ValueError("launch template termination protection must be disabled")
    if data.get("Monitoring") != {"Enabled": True}:
        raise ValueError("launch template detailed monitoring must be enabled")
    metadata = _mapping(
        data.get("MetadataOptions"),
        label="launch template metadata options",
    )
    required_metadata = {
        "HttpEndpoint": "enabled",
        "HttpProtocolIpv6": "disabled",
        "HttpPutResponseHopLimit": 1,
        "HttpTokens": "required",
        "InstanceMetadataTags": "disabled",
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError("launch template metadata options drift")
    mappings = data.get("BlockDeviceMappings")
    if not isinstance(mappings, list) or len(mappings) != 1:
        raise ValueError("launch template must define one root block device")
    root = _mapping(mappings[0], label="launch template root block device")
    if root.get("DeviceName") != ami_root_device_name:
        raise ValueError("launch template root device name drift")
    ebs = _mapping(root.get("Ebs"), label="launch template root EBS volume")
    if (
        ebs.get("DeleteOnTermination") is not True
        or ebs.get("Encrypted") is not True
        or type(ebs.get("VolumeSize")) is not int
        or ebs.get("VolumeSize") != profile.root_volume_gib
        or ebs.get("VolumeType") != "gp3"
    ):
        raise ValueError("launch template root EBS contract drift")
    return template_id, version, data


def _reviewed_bootstrap_hash(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            "reviewed expected SHA-256 must be 64 lowercase hexadecimal characters"
        )
    return value


def _verify_bootstrap_user_data(
    outputs: Mapping[str, str],
    *,
    launch_data: Mapping[str, object],
    expected_sha256: str,
) -> None:
    expected = _reviewed_bootstrap_hash(expected_sha256)
    stack_sha256 = _stack_output(
        outputs,
        "BootstrapUserDataSha256",
        pattern=_SHA256_RE,
    )
    if stack_sha256 != expected:
        raise ValueError(
            "BootstrapUserDataSha256 does not match the reviewed expected SHA-256"
        )
    encoded = launch_data.get("UserData")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("launch template UserData is missing")
    try:
        encoded_bytes = encoded.encode("ascii")
        payload = base64.b64decode(encoded_bytes, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ValueError(
            "launch template UserData is not strict base64"
        ) from error
    if not payload or base64.b64encode(payload) != encoded_bytes:
        raise ValueError("launch template UserData is not canonical base64")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != stack_sha256:
        raise ValueError(
            "launch template UserData SHA-256 does not match "
            "BootstrapUserDataSha256"
        )


def _verify_private_network(
    aws: AwsClients,
    outputs: Mapping[str, str],
    *,
    subnet_id: str,
    subnet: Mapping[str, object],
    launch_data: Mapping[str, object],
) -> str:
    security_group_id = _stack_output(
        outputs,
        "SecurityGroupId",
        pattern=_SECURITY_GROUP_RE,
    )
    if (
        subnet.get("SubnetId") != subnet_id
        or subnet.get("MapPublicIpOnLaunch") is not False
    ):
        raise ValueError("private subnet public-IP behavior drift")
    if "SecurityGroupIds" in launch_data or "SecurityGroups" in launch_data:
        raise ValueError("launch template has an alternate security-group path")
    interfaces = launch_data.get("NetworkInterfaces")
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        raise ValueError("launch template must define exactly one interface")
    interface = _mapping(
        interfaces[0],
        label="launch template network interface",
    )
    if (
        type(interface.get("DeviceIndex")) is not int
        or interface.get("DeviceIndex") != 0
        or interface.get("AssociatePublicIpAddress") is not False
        or interface.get("DeleteOnTermination") is not True
        or interface.get("SubnetId") != subnet_id
        or interface.get("Groups") != [security_group_id]
    ):
        raise ValueError("launch template private network binding drift")
    group = _one_mapping(
        aws.ec2.describe_security_groups(GroupIds=[security_group_id]),
        "SecurityGroups",
        label="security-group lookup",
    )
    expected_vpc = subnet.get("VpcId")
    if (
        group.get("GroupId") != security_group_id
        or not isinstance(expected_vpc, str)
        or _VPC_RE.fullmatch(expected_vpc) is None
        or group.get("VpcId") != expected_vpc
        or group.get("IpPermissions") != []
    ):
        raise ValueError("security group identity, VPC, or zero-ingress drift")
    output_vpc = outputs.get("VpcId")
    if output_vpc is not None and output_vpc != expected_vpc:
        raise ValueError("stack VPC output does not match the private subnet")
    return security_group_id


def _verify_iam(
    aws: AwsClients,
    outputs: Mapping[str, str],
    *,
    launch_data: Mapping[str, object],
) -> str:
    profile_arn = _stack_output(
        outputs,
        "BuilderInstanceProfileArn",
        pattern=_INSTANCE_PROFILE_ARN_RE,
    )
    role_arn = _stack_output(
        outputs,
        "BuilderRoleArn",
        pattern=_ROLE_ARN_RE,
    )
    if launch_data.get("IamInstanceProfile") != {"Arn": profile_arn}:
        raise ValueError("launch template instance profile drift")
    profile_name = profile_arn.rsplit("/", 1)[-1]
    role_name = role_arn.rsplit("/", 1)[-1]
    profile = _mapping(
        _mapping(
            aws.iam.get_instance_profile(
                InstanceProfileName=profile_name,
            ),
            label="instance profile response",
        ).get("InstanceProfile"),
        label="instance profile",
    )
    roles = profile.get("Roles")
    if (
        profile.get("Arn") != profile_arn
        or profile.get("InstanceProfileName") != profile_name
        or not isinstance(roles, list)
        or len(roles) != 1
    ):
        raise ValueError("instance profile identity or role count drift")
    attached_role = _mapping(roles[0], label="instance profile role")
    if (
        attached_role.get("Arn") != role_arn
        or attached_role.get("RoleName") != role_name
    ):
        raise ValueError("instance profile builder role drift")
    role = _mapping(
        _mapping(
            aws.iam.get_role(RoleName=role_name),
            label="builder role response",
        ).get("Role"),
        label="builder role",
    )
    if role.get("Arn") != role_arn or role.get("RoleName") != role_name:
        raise ValueError("builder role identity drift")
    return profile_arn


def _verify_bucket_and_kms(
    aws: AwsClients,
    outputs: Mapping[str, str],
    *,
    profile: CorpusBuilderProfile,
    package: S3ObjectVersion,
    source_manifest: S3ObjectVersion,
) -> None:
    bucket = _stack_output(outputs, "ArtifactBucketName")
    key_arn = _stack_output(outputs, "DataKeyArn", pattern=_KMS_ARN_RE)
    if bucket != profile.bucket_name:
        raise ValueError("artifact bucket output does not match the profile")
    if (
        package.kms_key_arn != key_arn
        or source_manifest.kms_key_arn != key_arn
    ):
        raise ValueError("input object KMS key does not match live DataKeyArn")
    versioning = _mapping(
        aws.s3.get_bucket_versioning(Bucket=bucket),
        label="bucket versioning response",
    )
    if versioning.get("Status") != "Enabled":
        raise ValueError("artifact bucket versioning is not enabled")
    public_access = _mapping(
        _mapping(
            aws.s3.get_public_access_block(Bucket=bucket),
            label="bucket public-access response",
        ).get("PublicAccessBlockConfiguration"),
        label="bucket public-access configuration",
    )
    if public_access != {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }:
        raise ValueError("artifact bucket Block Public Access drift")
    ownership = _mapping(
        _mapping(
            aws.s3.get_bucket_ownership_controls(Bucket=bucket),
            label="bucket ownership response",
        ).get("OwnershipControls"),
        label="bucket ownership controls",
    )
    if ownership.get("Rules") != [{"ObjectOwnership": "BucketOwnerEnforced"}]:
        raise ValueError("artifact bucket ownership controls drift")
    encryption = _mapping(
        _mapping(
            aws.s3.get_bucket_encryption(Bucket=bucket),
            label="bucket encryption response",
        ).get("ServerSideEncryptionConfiguration"),
        label="bucket encryption configuration",
    )
    rules = encryption.get("Rules")
    if not isinstance(rules, list) or len(rules) != 1:
        raise ValueError("artifact bucket must have one encryption rule")
    rule = _mapping(rules[0], label="bucket encryption rule")
    default = _mapping(
        rule.get("ApplyServerSideEncryptionByDefault"),
        label="bucket default encryption",
    )
    if (
        default.get("SSEAlgorithm") != "aws:kms"
        or default.get("KMSMasterKeyID") != key_arn
        or rule.get("BucketKeyEnabled") is not True
    ):
        raise ValueError("artifact bucket KMS encryption drift")
    key = _mapping(
        _mapping(
            aws.kms.describe_key(KeyId=key_arn),
            label="KMS describe response",
        ).get("KeyMetadata"),
        label="KMS key metadata",
    )
    if (
        key.get("Arn") != key_arn
        or key.get("Enabled") is not True
        or key.get("KeyState") != "Enabled"
        or key.get("KeyManager") != "CUSTOMER"
        or key.get("KeyUsage") != "ENCRYPT_DECRYPT"
        or key.get("Origin") != "AWS_KMS"
        or key.get("MultiRegion") is not False
    ):
        raise ValueError("dedicated KMS key identity or state drift")


def _price_filters() -> list[dict[str, str]]:
    return [
        {
            "Field": "instanceType",
            "Type": "TERM_MATCH",
            "Value": _INSTANCE_TYPE,
        },
        {
            "Field": "location",
            "Type": "TERM_MATCH",
            "Value": _PRICING_LOCATION,
        },
        {
            "Field": "operatingSystem",
            "Type": "TERM_MATCH",
            "Value": "Linux",
        },
        {
            "Field": "preInstalledSw",
            "Type": "TERM_MATCH",
            "Value": "NA",
        },
        {
            "Field": "tenancy",
            "Type": "TERM_MATCH",
            "Value": "Shared",
        },
        {
            "Field": "capacitystatus",
            "Type": "TERM_MATCH",
            "Value": "Used",
        },
    ]


def _decimal_price(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("Pricing USD amount must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Pricing USD amount is not decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("Pricing USD amount must be finite and non-negative")
    normalized = parsed.normalize()
    if normalized == 0:
        return Decimal("0")
    return Decimal(format(normalized, "f"))


def _current_linux_price(
    aws: AwsClients,
    *,
    now: datetime,
    maximum: Decimal,
) -> Decimal:
    response = _mapping(
        aws.pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=_price_filters(),
            FormatVersion="aws_v1",
            MaxResults=100,
        ),
        label="Pricing response",
    )
    if response.get("NextToken") not in {None, ""}:
        raise ValueError("Pricing response was truncated")
    raw_products = response.get("PriceList")
    if not isinstance(raw_products, list) or not raw_products:
        raise ValueError("Pricing returned no Linux On-Demand product")
    prices: list[Decimal] = []
    for raw_product in raw_products:
        if not isinstance(raw_product, str):
            raise ValueError("Pricing product must be encoded JSON")
        try:
            parsed = json.loads(
                raw_product,
                object_pairs_hook=_unique_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"Pricing product contains {constant}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError("Pricing product is not valid JSON") from error
        product_root = _mapping(parsed, label="Pricing product")
        product = _mapping(
            product_root.get("product"),
            label="Pricing product identity",
        )
        attributes = _mapping(
            product.get("attributes"),
            label="Pricing product attributes",
        )
        expected_attributes = {
            "capacitystatus": "Used",
            "instanceType": _INSTANCE_TYPE,
            "location": _PRICING_LOCATION,
            "operatingSystem": "Linux",
            "preInstalledSw": "NA",
            "tenancy": "Shared",
        }
        if (
            product.get("productFamily") != "Compute Instance"
            or any(
                attributes.get(key) != value
                for key, value in expected_attributes.items()
            )
        ):
            raise ValueError("Pricing product identity drift")
        on_demand = _mapping(
            _mapping(
                product_root.get("terms"),
                label="Pricing terms",
            ).get("OnDemand"),
            label="Pricing On-Demand terms",
        )
        if len(on_demand) != 1:
            raise ValueError("Pricing must return one current On-Demand term")
        term = _mapping(
            next(iter(on_demand.values())),
            label="Pricing On-Demand term",
        )
        if _parse_aws_timestamp(
            term.get("effectiveDate"),
            label="Pricing effective date",
        ) > now:
            raise ValueError("Pricing term is not yet effective")
        dimensions = _mapping(
            term.get("priceDimensions"),
            label="Pricing dimensions",
        )
        if len(dimensions) != 1:
            raise ValueError("Pricing must return one hourly dimension")
        dimension = _mapping(
            next(iter(dimensions.values())),
            label="Pricing hourly dimension",
        )
        if (
            dimension.get("unit") != "Hrs"
            or dimension.get("beginRange") != "0"
            or dimension.get("endRange") != "Inf"
        ):
            raise ValueError("Pricing hourly dimension drift")
        prices.append(
            _decimal_price(
                _mapping(
                    dimension.get("pricePerUnit"),
                    label="Pricing price per unit",
                ).get("USD")
            )
        )
    if len(prices) != 1:
        raise ValueError("Pricing did not resolve one current hourly price")
    price = prices[0]
    if price > maximum:
        raise ValueError("current hourly price exceeds the profile ceiling")
    return price


def _maximum_compute_cost(
    price: Decimal,
    profile: CorpusBuilderProfile,
) -> Decimal:
    if price > profile.max_hourly_usd:
        raise ValueError("current hourly price exceeds the profile ceiling")
    runtime_hours = Decimal(profile.max_runtime_seconds) / Decimal(3600)
    maximum = (price * runtime_hours).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_EVEN,
    )
    if maximum > profile.max_compute_usd:
        raise ValueError("maximum compute cost exceeds the approved ceiling")
    normalized = maximum.normalize()
    if normalized == 0:
        return Decimal("0")
    return Decimal(format(normalized, "f"))


def _aws_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return code if isinstance(code, str) else None


def _verify_dry_run(
    aws: AwsClients,
    *,
    template_id: str,
    template_version: str,
) -> None:
    request = {
        "DryRun": True,
        "LaunchTemplate": {
            "LaunchTemplateId": template_id,
            "Version": template_version,
        },
        "MaxCount": 1,
        "MinCount": 1,
    }
    try:
        response = aws.ec2.run_instances(**request)
    except Exception as error:
        if _aws_error_code(error) == "DryRunOperation":
            return
        raise ValueError("EC2 RunInstances dry-run was denied") from error
    if (
        not isinstance(response, Mapping)
        or response.get("DryRun") is not True
        or "Instances" in response
    ):
        raise ValueError("EC2 did not explicitly confirm a non-mutating dry-run")


def _run_gate(
    checks: list[str],
    name: str,
    action: Callable[[], _T],
) -> _T:
    try:
        result = action()
    except Exception as error:
        raise PreflightError(f"{name}: {error}") from error
    checks.append(name)
    return result


def validate_local_request(
    request: PreflightRequest,
    *,
    now: datetime,
) -> None:
    """Validate every preflight authority that requires no AWS client."""

    _validate_now(now)
    checks: list[str] = []
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[0],
        lambda: _load_profile(request.profile_path),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[1],
        lambda: _validate_input_records(
            request.package,
            request.source_manifest,
        ),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[2],
        lambda: _verify_software_gate(
            request.software_gate_receipt,
            request.package,
        ),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[7],
        lambda: _reviewed_bootstrap_hash(
            request.expected_bootstrap_user_data_sha256
        ),
    )


def _validate_now(now: datetime) -> None:
    if (
        not isinstance(now, datetime)
        or now.tzinfo is None
        or now.utcoffset() != timedelta(0)
        or now.microsecond != 0
    ):
        raise PreflightError("current time must be a canonical UTC second")


def run_preflight(
    request: PreflightRequest,
    *,
    aws: AwsClients,
    now: datetime,
) -> PreflightResult:
    """Run every read-only gate in order and return one short-lived intent."""

    _validate_now(now)
    checks: list[str] = []

    profile, profile_sha256 = _run_gate(
        checks,
        PREFLIGHT_CHECKS[0],
        lambda: _load_profile(request.profile_path),
    )
    package_revision = _run_gate(
        checks,
        PREFLIGHT_CHECKS[1],
        lambda: _verify_inputs(
            aws,
            request.package,
            request.source_manifest,
        ),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[2],
        lambda: _verify_software_gate(
            request.software_gate_receipt,
            request.package,
            expected_revision=package_revision,
        ),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[3],
        lambda: _verify_account_region(aws),
    )
    ami_root_device_name = _run_gate(
        checks,
        PREFLIGHT_CHECKS[4],
        lambda: _verify_ami(
            aws,
            ami_id=request.ami_id,
            ami_owner_id=request.ami_owner_id,
            now=now,
        ),
    )
    stack_outputs, subnet_id, subnet = _run_gate(
        checks,
        PREFLIGHT_CHECKS[5],
        lambda: _snapshot_and_verify_availability(
            aws,
            request.stack_outputs,
        ),
    )
    template_id, template_version, launch_data = _run_gate(
        checks,
        PREFLIGHT_CHECKS[6],
        lambda: _verify_launch_template(
            aws,
            stack_outputs,
            ami_id=request.ami_id,
            ami_root_device_name=ami_root_device_name,
            profile=profile,
        ),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[7],
        lambda: _verify_bootstrap_user_data(
            stack_outputs,
            launch_data=launch_data,
            expected_sha256=request.expected_bootstrap_user_data_sha256,
        ),
    )
    security_group_id = _run_gate(
        checks,
        PREFLIGHT_CHECKS[8],
        lambda: _verify_private_network(
            aws,
            stack_outputs,
            subnet_id=subnet_id,
            subnet=subnet,
            launch_data=launch_data,
        ),
    )
    instance_profile_arn = _run_gate(
        checks,
        PREFLIGHT_CHECKS[9],
        lambda: _verify_iam(
            aws,
            stack_outputs,
            launch_data=launch_data,
        ),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[10],
        lambda: _verify_bucket_and_kms(
            aws,
            stack_outputs,
            profile=profile,
            package=request.package,
            source_manifest=request.source_manifest,
        ),
    )
    hourly_usd = _run_gate(
        checks,
        PREFLIGHT_CHECKS[11],
        lambda: _current_linux_price(
            aws,
            now=now,
            maximum=profile.max_hourly_usd,
        ),
    )
    max_compute_usd = _run_gate(
        checks,
        PREFLIGHT_CHECKS[12],
        lambda: _maximum_compute_cost(hourly_usd, profile),
    )
    _run_gate(
        checks,
        PREFLIGHT_CHECKS[13],
        lambda: _verify_dry_run(
            aws,
            template_id=template_id,
            template_version=template_version,
        ),
    )

    not_after = (now + _INTENT_LIFETIME).astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    intent = LaunchIntent(
        format=LAUNCH_INTENT_FORMAT,
        schema_version=1,
        profile_sha256=profile_sha256,
        package=request.package,
        source_manifest=request.source_manifest,
        ami_id=request.ami_id,
        ami_owner_id=request.ami_owner_id,
        launch_template_id=template_id,
        launch_template_version=template_version,
        subnet_id=subnet_id,
        security_group_id=security_group_id,
        instance_profile_arn=instance_profile_arn,
        hourly_usd=hourly_usd,
        max_compute_usd=max_compute_usd,
        not_after=not_after,
    )
    payload = launch_intent_to_bytes(intent)
    return PreflightResult(
        intent=intent,
        intent_sha256=hashlib.sha256(payload).hexdigest(),
        checks=tuple(checks),
    )
