"""Explicitly approved, fail-closed EC2 launch control for corpus builders."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Protocol

from .contracts import (
    MAX_HOURLY_USD,
    LaunchIntent,
    launch_intent_from_bytes,
    launch_intent_to_bytes,
)


_INSTANCE_TYPE = "i4i.16xlarge"
_REGION = "us-east-1"
_TARGET_ACCOUNT_ID = "056956104102"
_MAX_RUNTIME = timedelta(hours=24)
_DESCRIBE_ATTEMPTS = 20
_DESCRIBE_DELAY_SECONDS = 0.25
_TERMINATE_ATTEMPTS = 3
_TERMINATION_CONFIRM_ATTEMPTS = 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
_INSTANCE_PROFILE_ACCOUNT_RE = re.compile(
    r"^arn:aws:iam::([0-9]{12}):instance-profile/"
)
_ACTIVE_INSTANCE_STATES = ("pending", "running", "stopping", "stopped")
_TERMINATED_INSTANCE_STATES = {"shutting-down", "terminated"}

_PRICE_FILTERS = (
    ("capacitystatus", "Used"),
    ("instanceType", _INSTANCE_TYPE),
    ("operatingSystem", "Linux"),
    ("preInstalledSw", "NA"),
    ("regionCode", _REGION),
    ("tenancy", "Shared"),
)


class LaunchError(ValueError):
    """A fail-closed launch approval or lifecycle error."""


class Ec2Client(Protocol):
    def get_caller_identity(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_launch_template_versions(
        self,
        **kwargs: object,
    ) -> Mapping[str, object]: ...

    def get_products(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_security_groups(
        self,
        **kwargs: object,
    ) -> Mapping[str, object]: ...

    def run_instances(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_instances(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_instance_attribute(
        self,
        **kwargs: object,
    ) -> Mapping[str, object]: ...

    def terminate_instances(self, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class BuilderLaunch:
    instance_id: str
    launch_intent_sha256: str
    launch_time: str
    terminate_at: str


def _utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise LaunchError("launch time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _intent_expiry(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise LaunchError("launch intent has an invalid expiry") from error


def _system_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _tag_values(intent: LaunchIntent) -> dict[str, str]:
    canonical = launch_intent_to_bytes(intent)
    intent_sha256 = hashlib.sha256(canonical).hexdigest()
    return {
        "MemorySplitCorpusBuilder": "true",
        "Name": f"memorysplit-corpus-builder-{intent_sha256}",
    }


def _verify_target_account(ec2: Ec2Client, intent: LaunchIntent) -> None:
    try:
        response = ec2.get_caller_identity()
    except Exception as error:
        raise LaunchError("target AWS account identity recheck failed") from error
    response = _mapping(response, label="caller identity response")
    profile_match = _INSTANCE_PROFILE_ACCOUNT_RE.match(
        intent.instance_profile_arn
    )
    if (
        response.get("Account") != _TARGET_ACCOUNT_ID
        or intent.ami_owner_id != _TARGET_ACCOUNT_ID
        or profile_match is None
        or profile_match.group(1) != _TARGET_ACCOUNT_ID
    ):
        raise LaunchError(
            "caller, AMI owner, and instance profile must match "
            f"target AWS account {_TARGET_ACCOUNT_ID}"
        )


def _verify_no_security_group_ingress(
    ec2: Ec2Client,
    security_group_id: str,
) -> None:
    try:
        response = ec2.describe_security_groups(
            GroupIds=[security_group_id],
        )
    except Exception as error:
        raise LaunchError("security-group ingress recheck failed") from error
    response = _mapping(response, label="security-group response")
    groups = _list(
        response.get("SecurityGroups"),
        label="security groups",
    )
    if len(groups) != 1:
        raise LaunchError("security group is missing or ambiguous")
    group = _mapping(groups[0], label="security group")
    if group.get("GroupId") != security_group_id:
        raise LaunchError("security group ID drift")
    if group.get("IpPermissions") != []:
        raise LaunchError("security group has inbound rules")


def approved_tag_specifications(
    intent: LaunchIntent,
) -> list[dict[str, object]]:
    """Return the exact instance tags cryptographically bound by the intent."""

    tags = _tag_values(intent)
    return [
        {
            "ResourceType": "instance",
            "Tags": [
                {"Key": key, "Value": value}
                for key, value in sorted(tags.items())
            ],
        }
    ]


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LaunchError(f"{label} must be a mapping")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise LaunchError(f"{label} must be a list")
    return value


def _template_network_matches(
    data: Mapping[str, object],
    intent: LaunchIntent,
) -> bool:
    interfaces = data.get("NetworkInterfaces")
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        return False
    interface = interfaces[0]
    return (
        isinstance(interface, Mapping)
        and interface.get("DeviceIndex") == 0
        and interface.get("AssociatePublicIpAddress") is False
        and interface.get("AssociateCarrierIpAddress") in (None, False)
        and interface.get("NetworkInterfaceId") is None
        and interface.get("Ipv6AddressCount") in (None, 0)
        and interface.get("Ipv6Addresses") in (None, [])
        and interface.get("SubnetId") == intent.subnet_id
        and interface.get("Groups") == [intent.security_group_id]
    )


def _recheck_launch_template(ec2: Ec2Client, intent: LaunchIntent) -> None:
    try:
        response = ec2.describe_launch_template_versions(
            LaunchTemplateId=intent.launch_template_id,
            Versions=[intent.launch_template_version],
        )
    except Exception as error:
        raise LaunchError("launch-template recheck failed") from error
    response = _mapping(response, label="launch-template response")
    versions = _list(
        response.get("LaunchTemplateVersions"),
        label="launch-template versions",
    )
    if len(versions) != 1:
        raise LaunchError("launch-template version is missing or ambiguous")
    version = _mapping(versions[0], label="launch-template version")
    if (
        version.get("LaunchTemplateId") != intent.launch_template_id
        or version.get("VersionNumber")
        != int(intent.launch_template_version)
    ):
        raise LaunchError("launch-template version drift")
    data = _mapping(
        version.get("LaunchTemplateData"),
        label="launch-template data",
    )
    metadata = data.get("MetadataOptions")
    profile = data.get("IamInstanceProfile")
    placement = data.get("Placement")
    if (
        data.get("ImageId") != intent.ami_id
        or data.get("InstanceType") != _INSTANCE_TYPE
        or not isinstance(profile, Mapping)
        or profile.get("Arn") != intent.instance_profile_arn
        or data.get("InstanceInitiatedShutdownBehavior") != "terminate"
        or not isinstance(metadata, Mapping)
        or metadata.get("HttpEndpoint") != "enabled"
        or metadata.get("HttpTokens") != "required"
        or not _template_network_matches(data, intent)
        or data.get("InstanceMarketOptions") not in (None, {})
        or (
            placement is not None
            and (
                not isinstance(placement, Mapping)
                or placement.get("Tenancy", "default") != "default"
            )
        )
    ):
        raise LaunchError("launch-template safety attributes drift")


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise LaunchError(f"pricing product repeats field: {key}")
        value[key] = item
    return value


def _price_document(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise LaunchError("pricing product must be JSON text")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                LaunchError(
                    f"pricing product contains non-finite {constant}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise LaunchError("pricing product must be valid JSON") from error
    return _mapping(parsed, label="pricing product")


def _matching_product_price(value: object) -> list[Decimal]:
    document = _price_document(value)
    product = document.get("product")
    if not isinstance(product, Mapping):
        return []
    attributes = product.get("attributes")
    if (
        product.get("productFamily") != "Compute Instance"
        or not isinstance(attributes, Mapping)
        or any(attributes.get(field) != expected for field, expected in _PRICE_FILTERS)
    ):
        return []
    terms = document.get("terms")
    if not isinstance(terms, Mapping):
        return []
    on_demand = terms.get("OnDemand")
    if not isinstance(on_demand, Mapping):
        return []
    prices: list[Decimal] = []
    for offer in on_demand.values():
        if not isinstance(offer, Mapping):
            continue
        dimensions = offer.get("priceDimensions")
        if not isinstance(dimensions, Mapping):
            continue
        for dimension in dimensions.values():
            if not isinstance(dimension, Mapping):
                continue
            per_unit = dimension.get("pricePerUnit")
            if (
                dimension.get("unit") != "Hrs"
                or dimension.get("beginRange") != "0"
                or dimension.get("endRange") != "Inf"
                or not isinstance(per_unit, Mapping)
                or not isinstance(per_unit.get("USD"), str)
            ):
                continue
            try:
                price = Decimal(per_unit["USD"])
            except InvalidOperation as error:
                raise LaunchError("hourly price is not a decimal") from error
            if not price.is_finite() or price < 0:
                raise LaunchError("hourly price must be finite and non-negative")
            prices.append(price)
    return prices


def _recheck_hourly_price(ec2: Ec2Client, intent: LaunchIntent) -> None:
    try:
        response = ec2.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": field, "Value": value}
                for field, value in _PRICE_FILTERS
            ],
            FormatVersion="aws_v1",
            MaxResults=100,
        )
    except Exception as error:
        raise LaunchError("Linux On-Demand price recheck failed") from error
    response = _mapping(response, label="pricing response")
    if response.get("NextToken") not in (None, ""):
        raise LaunchError("pricing response is unexpectedly paginated")
    products = _list(response.get("PriceList"), label="pricing products")
    prices = [
        price
        for product in products
        for price in _matching_product_price(product)
    ]
    if len(prices) != 1:
        raise LaunchError("Linux On-Demand hourly price is missing or ambiguous")
    if prices[0] > MAX_HOURLY_USD:
        raise LaunchError("current hourly price exceeds the approved ceiling")
    if prices[0] != intent.hourly_usd:
        raise LaunchError("current hourly price differs from the approved intent")


def _instance_id(value: object) -> str | None:
    if not isinstance(value, str) or _INSTANCE_ID_RE.fullmatch(value) is None:
        return None
    return value


def _run_instance_ids(response: object) -> tuple[list[str], int]:
    response = _mapping(response, label="RunInstances response")
    instances = _list(
        response.get("Instances"),
        label="RunInstances instances",
    )
    identifiers = []
    for instance in instances:
        if not isinstance(instance, Mapping):
            continue
        identifier = _instance_id(instance.get("InstanceId"))
        if identifier is not None:
            identifiers.append(identifier)
    return identifiers, len(instances)


def _termination_response_confirms(
    response: object,
    instance_ids: list[str],
) -> bool:
    try:
        response = _mapping(response, label="TerminateInstances response")
        rows = _list(
            response.get("TerminatingInstances"),
            label="terminating instances",
        )
        states: dict[str, str] = {}
        for row_value in rows:
            row = _mapping(row_value, label="terminating instance")
            identifier = _instance_id(row.get("InstanceId"))
            current_state = row.get("CurrentState")
            state_name = (
                current_state.get("Name")
                if isinstance(current_state, Mapping)
                else None
            )
            if (
                identifier is None
                or identifier in states
                or not isinstance(state_name, str)
                or state_name not in _TERMINATED_INSTANCE_STATES
            ):
                return False
            states[identifier] = state_name
    except LaunchError:
        return False
    return set(states) == set(instance_ids)


def _termination_is_confirmed(
    ec2: Ec2Client,
    instance_ids: list[str],
) -> bool:
    for identifier in instance_ids:
        try:
            response = ec2.describe_instances(InstanceIds=[identifier])
        except Exception as error:
            if _not_found(error):
                continue
            return False
        try:
            response = _mapping(
                response,
                label="termination DescribeInstances response",
            )
            if response.get("NextToken") not in (None, ""):
                return False
            instances = _described_instances(response)
        except LaunchError:
            return False
        if len(instances) != 1 or instances[0].get("InstanceId") != identifier:
            return False
        state = instances[0].get("State")
        state_name = state.get("Name") if isinstance(state, Mapping) else None
        if (
            not isinstance(state_name, str)
            or state_name not in _TERMINATED_INSTANCE_STATES
        ):
            return False
    return True


def _terminate_then_raise(
    ec2: Ec2Client,
    instance_ids: list[str],
    message: str,
    *,
    sleep: Callable[[float], None],
    cause: Exception | None = None,
) -> None:
    identifiers = sorted(set(instance_ids))
    if not identifiers:
        raise LaunchError(message) from cause
    cleanup_error: Exception | None = None
    for attempt in range(_TERMINATE_ATTEMPTS):
        try:
            response = ec2.terminate_instances(InstanceIds=identifiers)
        except Exception as error:
            cleanup_error = error
        else:
            if _termination_response_confirms(response, identifiers):
                break
            cleanup_error = LaunchError(
                "TerminateInstances response did not confirm shutdown"
            )
        if attempt + 1 < _TERMINATE_ATTEMPTS:
            sleep(_DESCRIBE_DELAY_SECONDS)

    for attempt in range(_TERMINATION_CONFIRM_ATTEMPTS):
        if _termination_is_confirmed(ec2, identifiers):
            raise LaunchError(message) from cause
        if attempt + 1 < _TERMINATION_CONFIRM_ATTEMPTS:
            sleep(_DESCRIBE_DELAY_SECONDS)

    manual_command = (
        "aws ec2 terminate-instances --profile sbsandbox "
        f"--region {_REGION} --instance-ids {' '.join(identifiers)}"
    )
    raise LaunchError(
        f"{message}; automatic termination could not be confirmed for "
        f"{', '.join(identifiers)}. Manually run: {manual_command}"
    ) from (cleanup_error or cause)


def _not_found(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return False
    detail = response.get("Error")
    return (
        isinstance(detail, Mapping)
        and detail.get("Code") == "InvalidInstanceID.NotFound"
    )


def _described_instances(response: object) -> list[Mapping[str, object]]:
    response = _mapping(response, label="DescribeInstances response")
    reservations = _list(
        response.get("Reservations"),
        label="DescribeInstances reservations",
    )
    instances: list[Mapping[str, object]] = []
    for reservation_value in reservations:
        reservation = _mapping(
            reservation_value,
            label="DescribeInstances reservation",
        )
        rows = _list(
            reservation.get("Instances"),
            label="DescribeInstances instances",
        )
        instances.extend(
            _mapping(row, label="described instance") for row in rows
        )
    return instances


def _refuse_active_approval_replay(
    ec2: Ec2Client,
    intent: LaunchIntent,
) -> None:
    tags = _tag_values(intent)
    try:
        response = ec2.describe_instances(
            Filters=[
                {
                    "Name": "tag:MemorySplitCorpusBuilder",
                    "Values": [tags["MemorySplitCorpusBuilder"]],
                },
                {
                    "Name": "tag:Name",
                    "Values": [tags["Name"]],
                },
                {
                    "Name": "instance-state-name",
                    "Values": list(_ACTIVE_INSTANCE_STATES),
                },
            ]
        )
    except Exception as error:
        raise LaunchError("approval-replay recheck failed") from error
    response = _mapping(response, label="approval-replay response")
    if response.get("NextToken") not in (None, ""):
        raise LaunchError("approval-replay response is unexpectedly paginated")
    for instance in _described_instances(response):
        state = instance.get("State")
        identifier = _instance_id(instance.get("InstanceId"))
        if (
            identifier is None
            or not isinstance(state, Mapping)
            or state.get("Name") not in _ACTIVE_INSTANCE_STATES
            or _tags(instance.get("Tags")) != tags
        ):
            raise LaunchError("approval-replay response is malformed")
        raise LaunchError(
            "approved intent already has an active instance: "
            f"{identifier}"
        )


def _wait_for_instance(
    ec2: Ec2Client,
    instance_id: str,
    *,
    sleep: Callable[[float], None],
) -> Mapping[str, object]:
    for attempt in range(_DESCRIBE_ATTEMPTS):
        try:
            response = ec2.describe_instances(InstanceIds=[instance_id])
        except Exception as error:
            if not _not_found(error):
                raise LaunchError("DescribeInstances failed after launch") from error
            instances: list[Mapping[str, object]] = []
        else:
            instances = _described_instances(response)
        if instances:
            if (
                len(instances) != 1
                or instances[0].get("InstanceId") != instance_id
            ):
                raise LaunchError(
                    "DescribeInstances did not return exactly the launched instance"
                )
            return instances[0]
        if attempt + 1 < _DESCRIBE_ATTEMPTS:
            sleep(_DESCRIBE_DELAY_SECONDS)
    raise LaunchError("launched instance did not become describable")


def _security_group_ids(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    groups: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            return None
        identifier = row.get("GroupId")
        if not isinstance(identifier, str):
            return None
        groups.append(identifier)
    return groups


def _tags(value: object) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None
    tags: dict[str, str] = {}
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"Key", "Value"}:
            return None
        key = row.get("Key")
        tag_value = row.get("Value")
        if (
            not isinstance(key, str)
            or not isinstance(tag_value, str)
            or key in tags
        ):
            return None
        tags[key] = tag_value
    return tags


def _no_public_address(instance: Mapping[str, object]) -> bool:
    if (
        instance.get("PublicIpAddress") not in (None, "")
        or instance.get("Ipv6Address") not in (None, "")
    ):
        return False
    interfaces = instance.get("NetworkInterfaces")
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        return False
    interface = interfaces[0]
    return (
        isinstance(interface, Mapping)
        and interface.get("Association") in (None, {})
        and interface.get("Ipv6Addresses") in (None, [])
    )


def _verify_launched_instance(
    instance: Mapping[str, object],
    intent: LaunchIntent,
) -> None:
    profile = instance.get("IamInstanceProfile")
    metadata = instance.get("MetadataOptions")
    state = instance.get("State")
    interfaces = instance.get("NetworkInterfaces")
    interface = (
        interfaces[0]
        if isinstance(interfaces, list) and len(interfaces) == 1
        else None
    )
    expected_tags = _tag_values(intent)
    if (
        instance.get("InstanceType") != _INSTANCE_TYPE
        or instance.get("ImageId") != intent.ami_id
        or instance.get("SubnetId") != intent.subnet_id
        or _security_group_ids(instance.get("SecurityGroups"))
        != [intent.security_group_id]
        or not isinstance(interface, Mapping)
        or interface.get("SubnetId") != intent.subnet_id
        or _security_group_ids(interface.get("Groups"))
        != [intent.security_group_id]
        or not isinstance(profile, Mapping)
        or profile.get("Arn") != intent.instance_profile_arn
        or not isinstance(metadata, Mapping)
        or metadata.get("HttpEndpoint") != "enabled"
        or metadata.get("HttpTokens") != "required"
        or _tags(instance.get("Tags")) != expected_tags
        or not _no_public_address(instance)
        or instance.get("InstanceLifecycle") not in (None, "")
        or instance.get("Platform") not in (None, "")
        or instance.get("PlatformDetails") != "Linux/UNIX"
        or not isinstance(state, Mapping)
        or state.get("Name") not in {"pending", "running"}
    ):
        raise LaunchError("launched instance safety attributes do not match the intent")


def _verify_shutdown_behavior(
    ec2: Ec2Client,
    instance_id: str,
) -> None:
    try:
        response = ec2.describe_instance_attribute(
            InstanceId=instance_id,
            Attribute="instanceInitiatedShutdownBehavior",
        )
    except Exception as error:
        raise LaunchError(
            "instance shutdown-behavior verification failed"
        ) from error
    response = _mapping(
        response,
        label="instance shutdown-behavior response",
    )
    behavior = response.get("InstanceInitiatedShutdownBehavior")
    if (
        response.get("InstanceId") != instance_id
        or not isinstance(behavior, Mapping)
        or behavior.get("Value") != "terminate"
    ):
        raise LaunchError("instance shutdown behavior is not terminate")


def launch_approved_builder(
    intent_bytes: bytes,
    *,
    approved_intent_sha256: str,
    ec2: Ec2Client,
    now: datetime,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> BuilderLaunch:
    """Launch one exact approved builder and immediately verify its lifecycle."""

    if not isinstance(intent_bytes, bytes):
        raise LaunchError("launch intent must be supplied as bytes")
    if (
        not isinstance(approved_intent_sha256, str)
        or _SHA256_RE.fullmatch(approved_intent_sha256) is None
    ):
        raise LaunchError("approved launch-intent SHA-256 must be lowercase")
    actual_sha256 = hashlib.sha256(intent_bytes).hexdigest()
    if not hmac.compare_digest(actual_sha256, approved_intent_sha256):
        raise LaunchError("approved launch-intent SHA-256 does not match")
    try:
        intent = launch_intent_from_bytes(intent_bytes)
    except ValueError as error:
        raise LaunchError(f"launch intent is invalid: {error}") from error
    current = _utc(now)
    expiry = _intent_expiry(intent.not_after)
    if current >= expiry:
        raise LaunchError("launch intent has expired")

    _verify_target_account(ec2, intent)
    _recheck_launch_template(ec2, intent)
    _recheck_hourly_price(ec2, intent)
    _verify_no_security_group_ingress(ec2, intent.security_group_id)
    _refuse_active_approval_replay(ec2, intent)
    launch_clock = clock or _system_utc_now
    if _utc(launch_clock()) >= expiry:
        raise LaunchError("launch intent has expired")
    pause = sleep or time.sleep
    request = {
        "ClientToken": actual_sha256,
        "LaunchTemplate": {
            "LaunchTemplateId": intent.launch_template_id,
            "Version": intent.launch_template_version,
        },
        "MinCount": 1,
        "MaxCount": 1,
        "TagSpecifications": approved_tag_specifications(intent),
    }
    try:
        response = ec2.run_instances(**request)
    except Exception as error:
        raise LaunchError("RunInstances failed") from error
    try:
        instance_ids, returned_instances = _run_instance_ids(response)
    except LaunchError as error:
        raise LaunchError(str(error)) from error
    if (
        returned_instances != 1
        or len(instance_ids) != 1
        or len(set(instance_ids)) != 1
    ):
        _terminate_then_raise(
            ec2,
            instance_ids,
            "RunInstances did not return exactly one instance",
            sleep=pause,
        )
    instance_id = instance_ids[0]
    launch_time: datetime
    try:
        instance = _wait_for_instance(ec2, instance_id, sleep=pause)
        _verify_launched_instance(instance, intent)
        _verify_shutdown_behavior(ec2, instance_id)
        _verify_no_security_group_ingress(ec2, intent.security_group_id)
        launch_time_value = instance.get("LaunchTime")
        if not isinstance(launch_time_value, datetime):
            raise LaunchError("described instance LaunchTime is missing")
        launch_time = _utc(launch_time_value)
    except Exception as error:
        message = (
            str(error)
            if isinstance(error, LaunchError)
            else "post-launch verification failed"
        )
        _terminate_then_raise(
            ec2,
            [instance_id],
            message,
            sleep=pause,
            cause=error,
        )

    return BuilderLaunch(
        instance_id=instance_id,
        launch_intent_sha256=actual_sha256,
        launch_time=_timestamp(launch_time),
        terminate_at=_timestamp(launch_time + _MAX_RUNTIME),
    )
