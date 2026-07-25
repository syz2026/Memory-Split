#!/usr/bin/env python3
"""Validate one closed P5/P6 RunInstances request without calling AWS."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.aws.p5.profile import (  # noqa: E402
    AWS_P5_V3_PROFILE_ID,
    AWS_P6_B300_V3_PROFILE_ID,
    load_aws_gpu_profile,
)


_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_CAPACITY_RESERVATION_RE = re.compile(r"^cr-[0-9a-f]{8,17}$")
_INSTANCE_PROFILE_RE = re.compile(
    r"^arn:aws(?:-[a-z]+)?:iam::[0-9]{12}:instance-profile/"
    r"[A-Za-z0-9+=,.@_/-]{1,128}$"
)
_KMS_KEY_RE = re.compile(
    r"^arn:aws:kms:(?P<region>us-(?:east-1|west-2)):[0-9]{12}:key/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECURITY_GROUP_RE = re.compile(r"^sg-[0-9a-f]{8,17}$")
_SUBNET_RE = re.compile(r"^subnet-[0-9a-f]{8,17}$")
_MAX_REQUEST_BYTES = 256 * 1024
_COMMON_FIELDS = {
    "BlockDeviceMappings",
    "IamInstanceProfile",
    "ImageId",
    "InstanceType",
    "MaxCount",
    "MetadataOptions",
    "MinCount",
    "NetworkInterfaces",
    "TagSpecifications",
}
_CAPACITY_BLOCK_FIELDS = {
    "CapacityReservationSpecification",
    "InstanceMarketOptions",
}
_V3_PROFILE_IDS = {AWS_P5_V3_PROFILE_ID, AWS_P6_B300_V3_PROFILE_ID}


class LaunchRequestError(ValueError):
    """The reviewed RunInstances request is not closed and immutable."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise LaunchRequestError(f"JSON object repeats field: {key}")
        value[key] = item
    return value


def _load_json_object(
    path: Path | str,
    *,
    label: str,
) -> tuple[dict[str, object], bytes]:
    candidate = Path(path)
    try:
        before = candidate.stat(follow_symlinks=False)
        if (
            candidate.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_REQUEST_BYTES
        ):
            raise LaunchRequestError(
                f"{label} must be one bounded singly linked regular file"
            )
        payload = candidate.read_bytes()
        after = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise LaunchRequestError(f"{label} cannot be read safely") from error
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
    ) or len(payload) != after.st_size:
        raise LaunchRequestError(f"{label} changed while being read")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                LaunchRequestError(
                    f"{label} contains non-finite {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchRequestError(
            f"{label} must contain one UTF-8 JSON object"
        ) from error
    if not isinstance(value, dict):
        raise LaunchRequestError(f"{label} root must be an object")
    return value, payload


def _exact(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise LaunchRequestError(
            f"{label} fields do not match the closed contract"
        )
    return value


def extract_capacity_reservation_id(response_path: Path | str) -> str:
    """Extract the purchased CapacityReservationId from its AWS response."""

    response, _payload = _load_json_object(
        response_path,
        label="capacity purchase response",
    )
    reservation = response.get("CapacityReservation")
    if not isinstance(reservation, dict):
        raise LaunchRequestError(
            "capacity purchase response has no CapacityReservation object"
        )
    identifier = reservation.get("CapacityReservationId")
    if (
        not isinstance(identifier, str)
        or _CAPACITY_RESERVATION_RE.fullmatch(identifier) is None
    ):
        raise LaunchRequestError(
            "capacity purchase response has no valid CapacityReservationId"
        )
    return identifier


def extract_run_instance_ids(
    response_path: Path | str,
    *,
    request_path: Path | str,
    expected_request_sha256: str,
) -> tuple[str, ...]:
    """Bind RunInstances response IDs to the validated request bytes."""

    request, request_payload = _load_json_object(
        request_path,
        label="launch request",
    )
    if (
        _SHA256_RE.fullmatch(expected_request_sha256) is None
        or hashlib.sha256(request_payload).hexdigest()
        != expected_request_sha256
    ):
        raise LaunchRequestError(
            "launch request bytes differ from the validated request"
        )
    minimum = request.get("MinCount")
    maximum = request.get("MaxCount")
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or minimum != maximum
        or minimum < 1
    ):
        raise LaunchRequestError("launch request count is not exact")

    response, _payload = _load_json_object(
        response_path,
        label="RunInstances response",
    )
    instances = response.get("Instances")
    if not isinstance(instances, list) or len(instances) != minimum:
        raise LaunchRequestError(
            "RunInstances response count differs from the validated request"
        )
    identifiers: list[str] = []
    for row in instances:
        if not isinstance(row, dict):
            raise LaunchRequestError(
                "RunInstances response contains a non-object instance"
            )
        identifier = row.get("InstanceId")
        if (
            not isinstance(identifier, str)
            or _INSTANCE_ID_RE.fullmatch(identifier) is None
        ):
            raise LaunchRequestError(
                "RunInstances response contains an invalid instance ID"
            )
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise LaunchRequestError("RunInstances response repeats an instance ID")
    return tuple(sorted(identifiers))


def validate_launch_request(
    request_path: Path | str,
    *,
    profile_path: Path | str,
    region: str,
    ami_id: str,
    instance_profile_arn: str,
    subnet_id: str,
    security_group_ids: Sequence[str],
    ebs_kms_key_id: str,
    root_device_name: str,
    root_volume_gib: int,
    cohort_id: str,
    capacity_reservation_id: str | None = None,
) -> dict[str, object]:
    """Validate exact launch fields and return their immutable review identity."""

    profile = load_aws_gpu_profile(profile_path)
    if (
        profile.profile_id not in _V3_PROFILE_IDS
        or profile.schema_version != 3
    ):
        raise LaunchRequestError(
            "launch request validation requires one closed AWS GPU v3 profile"
        )
    request, payload = _load_json_object(
        request_path,
        label="launch request",
    )
    capacity_block = profile.purchase_model == "capacity_block"
    expected_fields = _COMMON_FIELDS | (
        _CAPACITY_BLOCK_FIELDS if capacity_block else set()
    )
    if set(request) != expected_fields:
        raise LaunchRequestError(
            "launch request has missing or unknown top-level fields"
        )
    if region not in {"us-east-1", "us-west-2"}:
        raise LaunchRequestError("launch Region is outside the closed scope")
    if _AMI_RE.fullmatch(ami_id) is None or request["ImageId"] != ami_id:
        raise LaunchRequestError("launch request does not pin the reviewed AMI")
    if (
        _INSTANCE_PROFILE_RE.fullmatch(instance_profile_arn) is None
        or request["IamInstanceProfile"] != {"Arn": instance_profile_arn}
    ):
        raise LaunchRequestError(
            "launch request does not pin the dedicated instance profile ARN"
        )
    if request["InstanceType"] != profile.instance_type:
        raise LaunchRequestError(
            "launch request instance type differs from profile"
        )

    minimum = request["MinCount"]
    maximum = request["MaxCount"]
    max_allowed = 1 if capacity_block else 4
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or minimum != maximum
        or not 1 <= minimum <= max_allowed
    ):
        raise LaunchRequestError(
            "launch count must be one for P6 or an exact one-to-four P5 count"
        )

    groups = list(security_group_ids)
    if (
        _SUBNET_RE.fullmatch(subnet_id) is None
        or not groups
        or len(groups) != len(set(groups))
        or any(_SECURITY_GROUP_RE.fullmatch(group) is None for group in groups)
    ):
        raise LaunchRequestError(
            "private subnet or security-group identity is invalid"
        )
    interfaces = request["NetworkInterfaces"]
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        raise LaunchRequestError("launch request requires one network interface")
    interface = _exact(
        interfaces[0],
        {
            "AssociatePublicIpAddress",
            "DeleteOnTermination",
            "DeviceIndex",
            "Groups",
            "SubnetId",
        },
        label="launch network interface",
    )
    if interface != {
        "AssociatePublicIpAddress": False,
        "DeleteOnTermination": True,
        "DeviceIndex": 0,
        "Groups": groups,
        "SubnetId": subnet_id,
    }:
        raise LaunchRequestError(
            "launch network interface is not the reviewed private interface"
        )

    kms_match = _KMS_KEY_RE.fullmatch(ebs_kms_key_id)
    if kms_match is None or kms_match.group("region") != region:
        raise LaunchRequestError("EBS KMS key must be an immutable ARN in Region")
    if (
        not isinstance(root_device_name, str)
        or re.fullmatch(r"/dev/[A-Za-z0-9._-]+", root_device_name) is None
        or type(root_volume_gib) is not int
        or root_volume_gib < 100
    ):
        raise LaunchRequestError("reviewed root volume identity is invalid")
    mappings = request["BlockDeviceMappings"]
    if not isinstance(mappings, list) or len(mappings) != 1:
        raise LaunchRequestError(
            "launch request requires one encrypted root volume"
        )
    mapping = _exact(
        mappings[0],
        {"DeviceName", "Ebs"},
        label="launch block-device mapping",
    )
    ebs = _exact(
        mapping["Ebs"],
        {
            "DeleteOnTermination",
            "Encrypted",
            "KmsKeyId",
            "VolumeSize",
            "VolumeType",
        },
        label="launch EBS mapping",
    )
    if mapping["DeviceName"] != root_device_name or ebs != {
        "DeleteOnTermination": True,
        "Encrypted": True,
        "KmsKeyId": ebs_kms_key_id,
        "VolumeSize": root_volume_gib,
        "VolumeType": "gp3",
    }:
        raise LaunchRequestError(
            "launch root volume does not match the encrypted reviewed mapping"
        )

    metadata = _exact(
        request["MetadataOptions"],
        {
            "HttpEndpoint",
            "HttpProtocolIpv6",
            "HttpPutResponseHopLimit",
            "HttpTokens",
            "InstanceMetadataTags",
        },
        label="launch metadata options",
    )
    if metadata != {
        "HttpEndpoint": "enabled",
        "HttpProtocolIpv6": "disabled",
        "HttpPutResponseHopLimit": 1,
        "HttpTokens": "required",
        "InstanceMetadataTags": "disabled",
    }:
        raise LaunchRequestError(
            "launch request must require the closed IMDSv2 policy"
        )

    if (
        not isinstance(cohort_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/=+\-@]{0,255}", cohort_id)
        is None
    ):
        raise LaunchRequestError("cohort tag value is invalid")
    tag_specs = request["TagSpecifications"]
    if not isinstance(tag_specs, list) or len(tag_specs) != 2:
        raise LaunchRequestError(
            "launch request must tag exactly the instance and root volume"
        )
    expected_tag = [{"Key": "MemorySplitCohort", "Value": cohort_id}]
    by_resource: dict[str, object] = {}
    for raw in tag_specs:
        row = _exact(
            raw,
            {"ResourceType", "Tags"},
            label="launch tag specification",
        )
        resource_type = row["ResourceType"]
        if not isinstance(resource_type, str) or resource_type in by_resource:
            raise LaunchRequestError(
                "launch tag resources are invalid or duplicated"
            )
        by_resource[resource_type] = row["Tags"]
    if by_resource != {"instance": expected_tag, "volume": expected_tag}:
        raise LaunchRequestError(
            "launch tags do not match the closed cohort identity"
        )

    if capacity_block:
        if (
            capacity_reservation_id is None
            or _CAPACITY_RESERVATION_RE.fullmatch(capacity_reservation_id)
            is None
            or request["InstanceMarketOptions"]
            != {"MarketType": "capacity-block"}
            or request["CapacityReservationSpecification"]
            != {
                "CapacityReservationTarget": {
                    "CapacityReservationId": capacity_reservation_id
                }
            }
        ):
            raise LaunchRequestError(
                "P6 launch does not target the purchased Capacity Block reservation"
            )
    elif capacity_reservation_id is not None:
        raise LaunchRequestError("P5 On-Demand launch must not name a reservation")

    return {
        "schema_version": 1,
        "ok": True,
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "profile_id": profile.profile_id,
        "instance_type": profile.instance_type,
        "purchase_model": profile.purchase_model,
        "region": region,
        "ami_id": ami_id,
        "instance_profile_arn": instance_profile_arn,
        "subnet_id": subnet_id,
        "security_group_ids": groups,
        "ebs_kms_key_id": ebs_kms_key_id,
        "count": minimum,
        "capacity_reservation_id": capacity_reservation_id,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one reviewed P5/P6 RunInstances JSON request locally."
        )
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--ami-id", required=True)
    parser.add_argument("--instance-profile-arn", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--security-group-id", action="append", required=True)
    parser.add_argument("--ebs-kms-key-id", required=True)
    parser.add_argument("--root-device-name", required=True)
    parser.add_argument("--root-volume-gib", type=int, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--capacity-reservation-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report = validate_launch_request(
            arguments.request,
            profile_path=arguments.profile,
            region=arguments.region,
            ami_id=arguments.ami_id,
            instance_profile_arn=arguments.instance_profile_arn,
            subnet_id=arguments.subnet_id,
            security_group_ids=arguments.security_group_id,
            ebs_kms_key_id=arguments.ebs_kms_key_id,
            root_device_name=arguments.root_device_name,
            root_volume_gib=arguments.root_volume_gib,
            cohort_id=arguments.cohort_id,
            capacity_reservation_id=arguments.capacity_reservation_id,
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except (LaunchRequestError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
