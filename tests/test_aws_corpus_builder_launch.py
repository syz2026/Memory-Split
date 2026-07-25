from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from cluster.aws.corpus_builder.contracts import (
    LAUNCH_INTENT_FORMAT,
    LaunchIntent,
    S3ObjectVersion,
    launch_intent_to_bytes,
)
from cluster.aws.corpus_builder.launch import LaunchError, launch_approved_builder
from scripts import aws_corpus_builder_launch


NOW = datetime(2026, 7, 25, 3, 0, 0, tzinfo=timezone.utc)
LAUNCH_TIME = datetime(2026, 7, 25, 3, 5, 0, tzinfo=timezone.utc)
TARGET_ACCOUNT = "056956104102"
AMI_ID = "ami-0123456789abcdef0"
LAUNCH_TEMPLATE_ID = "lt-0123456789abcdef0"
LAUNCH_TEMPLATE_VERSION = "3"
SUBNET_ID = "subnet-0123456789abcdef0"
SECURITY_GROUP_ID = "sg-0123456789abcdef0"
INSTANCE_PROFILE_ARN = (
    "arn:aws:iam::056956104102:instance-profile/memorysplit-corpus-builder"
)
KMS_KEY_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/01234567-89ab-cdef-0123-456789abcdef"
)
BUILD_ID = "b" * 64
PROFILE_SHA256 = "a" * 64
PACKAGE_SHA256 = "c" * 64
SOURCE_MANIFEST_SHA256 = "d" * 64
INSTANCE_ID = "i-0123456789abcdef0"
SECOND_INSTANCE_ID = "i-fedcba98765432100"


def _object_version(name: str, digest: str) -> S3ObjectVersion:
    return S3ObjectVersion(
        uri=(
            "s3://memorysplit-corpus-056956104102-us-east-1/"
            f"v2/builds/{BUILD_ID}/{name}"
        ),
        version_id=f"{name}-version-001",
        bytes=1024,
        sha256=digest,
        etag="0123456789abcdef0123456789abcdef",
        sse_algorithm="aws:kms",
        kms_key_arn=KMS_KEY_ARN,
    )


def _intent_bytes(
    *,
    not_after: str = "2026-07-25T03:30:00Z",
    ami_owner_id: str = TARGET_ACCOUNT,
    instance_profile_arn: str = INSTANCE_PROFILE_ARN,
) -> bytes:
    return launch_intent_to_bytes(
        LaunchIntent(
            format=LAUNCH_INTENT_FORMAT,
            schema_version=1,
            profile_sha256=PROFILE_SHA256,
            package=_object_version("package.tar.gz", PACKAGE_SHA256),
            source_manifest=_object_version(
                "source-manifest.json",
                SOURCE_MANIFEST_SHA256,
            ),
            ami_id=AMI_ID,
            ami_owner_id=ami_owner_id,
            launch_template_id=LAUNCH_TEMPLATE_ID,
            launch_template_version=LAUNCH_TEMPLATE_VERSION,
            subnet_id=SUBNET_ID,
            security_group_id=SECURITY_GROUP_ID,
            instance_profile_arn=instance_profile_arn,
            hourly_usd=Decimal("5.491"),
            max_compute_usd=Decimal("131.78"),
            not_after=not_after,
        )
    )


INTENT_BYTES = _intent_bytes()
INTENT_SHA256 = hashlib.sha256(INTENT_BYTES).hexdigest()


def _expected_tags(payload: bytes = INTENT_BYTES) -> dict[str, str]:
    intent_sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "MemorySplitCorpusBuilder": "true",
        "Name": f"memorysplit-corpus-builder-{intent_sha256}",
    }


def _replay_filters(payload: bytes = INTENT_BYTES) -> list[dict[str, object]]:
    return [
        {
            "Name": "tag:MemorySplitCorpusBuilder",
            "Values": ["true"],
        },
        {
            "Name": "tag:Name",
            "Values": [
                f"memorysplit-corpus-builder-"
                f"{hashlib.sha256(payload).hexdigest()}"
            ],
        },
        {
            "Name": "instance-state-name",
            "Values": ["pending", "running", "stopping", "stopped"],
        },
    ]


def _launch_template_data() -> dict[str, object]:
    return {
        "ImageId": AMI_ID,
        "IamInstanceProfile": {"Arn": INSTANCE_PROFILE_ARN},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "InstanceType": "i4i.16xlarge",
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
        },
        "NetworkInterfaces": [
            {
                "AssociatePublicIpAddress": False,
                "DeviceIndex": 0,
                "Groups": [SECURITY_GROUP_ID],
                "SubnetId": SUBNET_ID,
            }
        ],
    }


def _described_instance(
    *,
    instance_id: str = INSTANCE_ID,
    payload: bytes = INTENT_BYTES,
    state: str = "pending",
    launch_time: datetime = LAUNCH_TIME,
) -> dict[str, object]:
    return {
        "ImageId": AMI_ID,
        "IamInstanceProfile": {
            "Arn": INSTANCE_PROFILE_ARN,
            "Id": "AIPAJUSTAFIXTURE",
        },
        "InstanceId": instance_id,
        "InstanceType": "i4i.16xlarge",
        "LaunchTime": launch_time,
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
            "State": "applied",
        },
        "NetworkInterfaces": [
            {
                "Groups": [
                    {
                        "GroupId": SECURITY_GROUP_ID,
                        "GroupName": "memorysplit-corpus-builder",
                    }
                ],
                "NetworkInterfaceId": "eni-0123456789abcdef0",
                "PrivateIpAddress": "10.0.1.42",
                "SubnetId": SUBNET_ID,
            }
        ],
        "PlatformDetails": "Linux/UNIX",
        "PrivateIpAddress": "10.0.1.42",
        "SecurityGroups": [
            {
                "GroupId": SECURITY_GROUP_ID,
                "GroupName": "memorysplit-corpus-builder",
            }
        ],
        "State": {"Code": 0, "Name": state},
        "SubnetId": SUBNET_ID,
        "Tags": [
            {"Key": key, "Value": value}
            for key, value in sorted(_expected_tags(payload).items())
        ],
    }


def _price_product(hourly_usd: str) -> str:
    return json.dumps(
        {
            "product": {
                "attributes": {
                    "capacitystatus": "Used",
                    "instanceType": "i4i.16xlarge",
                    "operatingSystem": "Linux",
                    "preInstalledSw": "NA",
                    "regionCode": "us-east-1",
                    "tenancy": "Shared",
                },
                "productFamily": "Compute Instance",
            },
            "terms": {
                "OnDemand": {
                    "offer": {
                        "priceDimensions": {
                            "dimension": {
                                "beginRange": "0",
                                "endRange": "Inf",
                                "pricePerUnit": {"USD": hourly_usd},
                                "unit": "Hrs",
                            }
                        }
                    }
                }
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class FakeEc2:
    def __init__(self) -> None:
        self.identity_calls: list[dict[str, object]] = []
        self.template_calls: list[dict[str, object]] = []
        self.price_calls: list[dict[str, object]] = []
        self.security_group_calls: list[dict[str, object]] = []
        self.run_calls: list[dict[str, object]] = []
        self.describe_calls: list[dict[str, object]] = []
        self.attribute_calls: list[dict[str, object]] = []
        self.terminate_calls: list[dict[str, object]] = []
        self.identity_response: dict[str, object] = {
            "Account": TARGET_ACCOUNT,
            "Arn": (
                "arn:aws:sts::056956104102:"
                "assumed-role/memorysplit-controller/test"
            ),
            "UserId": "AROAFIXTURE:test",
        }
        self.template_response: dict[str, object] = {
            "LaunchTemplateVersions": [
                {
                    "LaunchTemplateData": _launch_template_data(),
                    "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
                    "VersionNumber": int(LAUNCH_TEMPLATE_VERSION),
                }
            ]
        }
        self.price_response: dict[str, object] = {
            "FormatVersion": "aws_v1",
            "PriceList": [_price_product("5.491")],
        }
        self.security_group_responses: list[object] = [
            {
                "SecurityGroups": [
                    {
                        "GroupId": SECURITY_GROUP_ID,
                        "IpPermissions": [],
                    }
                ]
            },
            {
                "SecurityGroups": [
                    {
                        "GroupId": SECURITY_GROUP_ID,
                        "IpPermissions": [],
                    }
                ]
            },
        ]
        self.replay_response: dict[str, object] = {"Reservations": []}
        self.run_response: dict[str, object] = {
            "Instances": [{"InstanceId": INSTANCE_ID}]
        }
        self.describe_responses: list[object] = [
            {"Reservations": [{"Instances": [_described_instance()]}]}
        ]
        self.attribute_response: dict[str, object] = {
            "InstanceId": INSTANCE_ID,
            "InstanceInitiatedShutdownBehavior": {"Value": "terminate"},
        }
        self.terminate_error: Exception | None = None

    def get_caller_identity(self, **kwargs: object) -> dict[str, object]:
        self.identity_calls.append(dict(kwargs))
        return deepcopy(self.identity_response)

    def describe_launch_template_versions(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.template_calls.append(dict(kwargs))
        return deepcopy(self.template_response)

    def get_products(self, **kwargs: object) -> dict[str, object]:
        self.price_calls.append(dict(kwargs))
        return deepcopy(self.price_response)

    def describe_security_groups(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.security_group_calls.append(dict(kwargs))
        if not self.security_group_responses:
            return {"SecurityGroups": []}
        response = self.security_group_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, dict)
        return deepcopy(response)

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        self.run_calls.append(deepcopy(dict(kwargs)))
        return deepcopy(self.run_response)

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.describe_calls.append(dict(kwargs))
        if "Filters" in kwargs:
            return deepcopy(self.replay_response)
        if not self.describe_responses:
            return {"Reservations": []}
        response = self.describe_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, dict)
        return deepcopy(response)

    def describe_instance_attribute(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.attribute_calls.append(dict(kwargs))
        return deepcopy(self.attribute_response)

    def terminate_instances(self, **kwargs: object) -> dict[str, object]:
        self.terminate_calls.append(dict(kwargs))
        if self.terminate_error is not None:
            raise self.terminate_error
        return {"TerminatingInstances": []}


def _launch(ec2: FakeEc2, *, payload: bytes = INTENT_BYTES):
    return launch_approved_builder(
        payload,
        approved_intent_sha256=hashlib.sha256(payload).hexdigest(),
        ec2=ec2,
        now=NOW,
        clock=lambda: NOW,
    )


def test_launch_requires_exact_unexpired_intent_hash_and_one_instance():
    ec2 = FakeEc2()

    launch = _launch(ec2)

    assert launch.instance_id == INSTANCE_ID
    assert launch.launch_intent_sha256 == INTENT_SHA256
    assert launch.launch_time == "2026-07-25T03:00:00Z"
    assert launch.terminate_at == "2026-07-26T03:00:00Z"
    assert ec2.identity_calls == [{}]
    assert ec2.template_calls == [
        {
            "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
            "Versions": [LAUNCH_TEMPLATE_VERSION],
        }
    ]
    assert len(ec2.price_calls) == 1
    assert ec2.price_calls[0]["ServiceCode"] == "AmazonEC2"
    filters = ec2.price_calls[0]["Filters"]
    assert isinstance(filters, list)
    assert {
        (item["Field"], item["Value"])
        for item in filters
    } == {
        ("capacitystatus", "Used"),
        ("instanceType", "i4i.16xlarge"),
        ("operatingSystem", "Linux"),
        ("preInstalledSw", "NA"),
        ("regionCode", "us-east-1"),
        ("tenancy", "Shared"),
    }
    assert ec2.run_calls == [
        {
            "ClientToken": INTENT_SHA256,
            "LaunchTemplate": {
                "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
                "Version": LAUNCH_TEMPLATE_VERSION,
            },
            "MaxCount": 1,
            "MinCount": 1,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": key, "Value": value}
                        for key, value in sorted(_expected_tags().items())
                    ],
                }
            ],
        }
    ]
    assert ec2.describe_calls == [
        {"Filters": _replay_filters()},
        {"InstanceIds": [INSTANCE_ID]},
    ]
    assert ec2.attribute_calls == [
        {
            "Attribute": "instanceInitiatedShutdownBehavior",
            "InstanceId": INSTANCE_ID,
        }
    ]
    assert ec2.terminate_calls == []
    assert ec2.security_group_calls == [
        {"GroupIds": [SECURITY_GROUP_ID]},
        {"GroupIds": [SECURITY_GROUP_ID]},
    ]


@pytest.mark.parametrize(
    ("payload", "approval", "now"),
    [
        (INTENT_BYTES, "0" * 64, NOW),
        (_intent_bytes(not_after="2026-07-25T03:00:00Z"), None, NOW),
        (b"not canonical JSON\n", None, NOW),
    ],
    ids=("wrong-hash", "expired", "noncanonical"),
)
def test_launch_rejects_unapproved_expired_or_noncanonical_intent(
    payload: bytes,
    approval: str | None,
    now: datetime,
):
    ec2 = FakeEc2()
    approved = approval or hashlib.sha256(payload).hexdigest()

    with pytest.raises(LaunchError):
        launch_approved_builder(
            payload,
            approved_intent_sha256=approved,
            ec2=ec2,
            now=now,
        )

    assert ec2.template_calls == []
    assert ec2.price_calls == []
    assert ec2.run_calls == []


@pytest.mark.parametrize(
    "failure",
    ("price-drift", "template-drift", "template-public-ip-implicit"),
)
def test_launch_rechecks_price_and_template_immediately_before_launch(failure: str):
    ec2 = FakeEc2()
    if failure == "price-drift":
        ec2.price_response["PriceList"] = [_price_product("5.490")]
    elif failure == "template-drift":
        versions = ec2.template_response["LaunchTemplateVersions"]
        assert isinstance(versions, list)
        versions[0]["VersionNumber"] = 4
    else:
        versions = ec2.template_response["LaunchTemplateVersions"]
        assert isinstance(versions, list)
        data = versions[0]["LaunchTemplateData"]
        assert isinstance(data, dict)
        data.pop("NetworkInterfaces")
        data["SecurityGroupIds"] = [SECURITY_GROUP_ID]
        data["SubnetId"] = SUBNET_ID

    with pytest.raises(LaunchError):
        _launch(ec2)

    assert ec2.run_calls == []
    assert ec2.describe_calls == []
    assert ec2.terminate_calls == []


def test_launch_refuses_an_active_instance_with_the_approved_intent_tag():
    ec2 = FakeEc2()
    ec2.replay_response = {
        "Reservations": [
            {
                "Instances": [
                    _described_instance(state="running"),
                ]
            }
        ]
    }

    with pytest.raises(LaunchError, match="already has an active instance"):
        _launch(ec2)

    assert ec2.run_calls == []
    assert ec2.describe_calls == [
        {"Filters": _replay_filters()}
    ]


def test_run_instances_uses_the_approved_hash_as_its_client_token():
    ec2 = FakeEc2()

    _launch(ec2)

    assert ec2.run_calls[0]["ClientToken"] == INTENT_SHA256


def test_run_instances_rejects_a_non_aws_instance_id_shape():
    ec2 = FakeEc2()
    ec2.run_response = {"Instances": [{"InstanceId": "i-builder"}]}

    with pytest.raises(
        LaunchError,
        match="RunInstances did not return exactly one instance",
    ):
        _launch(ec2)

    assert ec2.terminate_calls == []


def test_launch_rechecks_expiry_after_all_read_only_checks():
    expires = NOW + timedelta(seconds=10)
    payload = _intent_bytes(not_after="2026-07-25T03:00:10Z")
    ec2 = FakeEc2()

    with pytest.raises(LaunchError, match="expired"):
        launch_approved_builder(
            payload,
            approved_intent_sha256=hashlib.sha256(payload).hexdigest(),
            ec2=ec2,
            now=NOW,
            clock=lambda: expires,
        )

    assert ec2.identity_calls == [{}]
    assert len(ec2.template_calls) == 1
    assert len(ec2.price_calls) == 1
    assert ec2.security_group_calls == [
        {"GroupIds": [SECURITY_GROUP_ID]}
    ]
    assert len(ec2.describe_calls) == 1
    assert "Filters" in ec2.describe_calls[0]
    assert ec2.run_calls == []


@pytest.mark.parametrize(
    "mismatch",
    ("caller", "ami-owner", "instance-profile"),
)
def test_launch_requires_every_account_boundary_to_match_target(mismatch: str):
    ec2 = FakeEc2()
    payload = INTENT_BYTES
    if mismatch == "caller":
        ec2.identity_response["Account"] = "999999999999"
    elif mismatch == "ami-owner":
        payload = _intent_bytes(ami_owner_id="999999999999")
    else:
        payload = _intent_bytes(
            instance_profile_arn=(
                "arn:aws:iam::999999999999:"
                "instance-profile/memorysplit-corpus-builder"
            )
        )

    with pytest.raises(LaunchError, match="target AWS account"):
        _launch(ec2, payload=payload)

    assert ec2.identity_calls == [{}]
    assert ec2.template_calls == []
    assert ec2.price_calls == []
    assert ec2.run_calls == []


@pytest.mark.parametrize("stage", ("before-launch", "after-launch"))
def test_launch_rejects_security_group_ingress_drift(stage: str):
    ec2 = FakeEc2()
    no_ingress = {
        "SecurityGroups": [
            {"GroupId": SECURITY_GROUP_ID, "IpPermissions": []}
        ]
    }
    has_ingress = {
        "SecurityGroups": [
            {
                "GroupId": SECURITY_GROUP_ID,
                "IpPermissions": [
                    {
                        "FromPort": 22,
                        "IpProtocol": "tcp",
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                        "ToPort": 22,
                    }
                ],
            }
        ]
    }
    ec2.security_group_responses = (
        [has_ingress] if stage == "before-launch" else [no_ingress, has_ingress]
    )

    with pytest.raises(LaunchError, match="security group has inbound rules"):
        _launch(ec2)

    if stage == "before-launch":
        assert ec2.run_calls == []
        assert ec2.terminate_calls == []
    else:
        assert len(ec2.run_calls) == 1
        assert ec2.terminate_calls == [{"InstanceIds": [INSTANCE_ID]}]


def test_launch_terminates_all_returned_instances_if_run_instances_returns_two():
    ec2 = FakeEc2()
    ec2.run_response = {
        "Instances": [
            {"InstanceId": INSTANCE_ID},
            {"InstanceId": SECOND_INSTANCE_ID},
        ]
    }

    with pytest.raises(LaunchError):
        _launch(ec2)

    assert ec2.describe_calls == [{"Filters": _replay_filters()}]
    assert ec2.terminate_calls == [
        {"InstanceIds": [INSTANCE_ID, SECOND_INSTANCE_ID]}
    ]


def test_launch_rejects_a_malformed_second_run_instance_and_terminates_the_known_one():
    ec2 = FakeEc2()
    ec2.run_response = {
        "Instances": [
            {"InstanceId": INSTANCE_ID},
            {"State": {"Name": "pending"}},
        ]
    }

    with pytest.raises(LaunchError):
        _launch(ec2)

    assert ec2.describe_calls == [{"Filters": _replay_filters()}]
    assert ec2.attribute_calls == []
    assert ec2.terminate_calls == [{"InstanceIds": [INSTANCE_ID]}]


@pytest.mark.parametrize(
    "mismatch",
    (
        "instance-type",
        "subnet",
        "security-group",
        "profile",
        "ami",
        "imds",
        "shutdown",
        "tags",
        "public-ip",
        "ipv6",
        "spot",
        "non-linux",
    ),
)
def test_post_launch_mismatch_terminates_then_raises(mismatch: str):
    ec2 = FakeEc2()
    instance = _described_instance()
    if mismatch == "instance-type":
        instance["InstanceType"] = "i4i.8xlarge"
    elif mismatch == "subnet":
        instance["SubnetId"] = "subnet-fedcba98765432100"
    elif mismatch == "security-group":
        instance["SecurityGroups"] = [
            {"GroupId": "sg-fedcba98765432100", "GroupName": "wrong"}
        ]
    elif mismatch == "profile":
        instance["IamInstanceProfile"] = {
            "Arn": (
                "arn:aws:iam::056956104102:"
                "instance-profile/not-the-builder"
            ),
            "Id": "AIPAJUSTAFIXTURE",
        }
    elif mismatch == "ami":
        instance["ImageId"] = "ami-fedcba98765432100"
    elif mismatch == "imds":
        instance["MetadataOptions"]["HttpTokens"] = "optional"
    elif mismatch == "shutdown":
        ec2.attribute_response["InstanceInitiatedShutdownBehavior"] = {
            "Value": "stop"
        }
    elif mismatch == "tags":
        instance["Tags"][0]["Value"] = "other"
    elif mismatch == "public-ip":
        instance["PublicIpAddress"] = "203.0.113.10"
        instance["NetworkInterfaces"][0]["Association"] = {
            "PublicIp": "203.0.113.10"
        }
    elif mismatch == "ipv6":
        instance["NetworkInterfaces"][0]["Ipv6Addresses"] = [
            {"Ipv6Address": "2001:db8::10"}
        ]
    elif mismatch == "spot":
        instance["InstanceLifecycle"] = "spot"
    else:
        instance["PlatformDetails"] = "Windows"
    ec2.describe_responses = [
        {"Reservations": [{"Instances": [instance]}]}
    ]

    with pytest.raises(LaunchError):
        _launch(ec2)

    assert ec2.terminate_calls == [{"InstanceIds": [INSTANCE_ID]}]


def test_describe_polls_until_the_launched_instance_exists():
    ec2 = FakeEc2()
    ec2.describe_responses = [
        {"Reservations": []},
        {"Reservations": [{"Instances": [_described_instance()]}]},
    ]

    launch = _launch(ec2)

    assert launch.instance_id == INSTANCE_ID
    assert ec2.describe_calls == [
        {"Filters": _replay_filters()},
        {"InstanceIds": [INSTANCE_ID]},
        {"InstanceIds": [INSTANCE_ID]},
    ]
    assert ec2.terminate_calls == []


def test_describe_failure_after_launch_terminates_then_raises():
    ec2 = FakeEc2()
    ec2.describe_responses = [RuntimeError("describe denied")]

    with pytest.raises(LaunchError):
        _launch(ec2)

    assert ec2.terminate_calls == [{"InstanceIds": [INSTANCE_ID]}]


def test_cleanup_failure_is_reported_without_hiding_the_launch_mismatch():
    ec2 = FakeEc2()
    instance = _described_instance()
    instance["ImageId"] = "ami-fedcba98765432100"
    ec2.describe_responses = [
        {"Reservations": [{"Instances": [instance]}]}
    ]
    ec2.terminate_error = RuntimeError("terminate denied")

    with pytest.raises(LaunchError, match="termination failed"):
        _launch(ec2)

    assert ec2.terminate_calls == [{"InstanceIds": [INSTANCE_ID]}]


class FakeSession:
    def __init__(self, client: FakeEc2) -> None:
        self._client = client
        self.calls: list[tuple[str, str | None]] = []

    def client(self, service_name: str, *, region_name: str | None = None):
        self.calls.append((service_name, region_name))
        assert service_name in {"ec2", "pricing", "sts"}
        return self._client


def test_cli_requires_explicit_hash_and_prints_lifecycle_and_cost(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    intent_path = tmp_path / "launch-intent.json"
    intent_path.write_bytes(INTENT_BYTES)
    ec2 = FakeEc2()
    session = FakeSession(ec2)
    monkeypatch.setattr(
        aws_corpus_builder_launch,
        "_new_session",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(
        aws_corpus_builder_launch,
        "_utc_now",
        lambda: NOW,
    )

    result = aws_corpus_builder_launch.main(
        [
            "--intent",
            str(intent_path),
            "--approve-intent-sha256",
            INTENT_SHA256,
            "--profile",
            "sbsandbox",
            "--region",
            "us-east-1",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "instance_id": INSTANCE_ID,
        "launch_intent_sha256": INTENT_SHA256,
        "launch_time": "2026-07-25T03:00:00Z",
        "max_compute_usd": "131.78",
        "terminate_at": "2026-07-26T03:00:00Z",
    }
    assert session.calls == [
        ("ec2", "us-east-1"),
        ("pricing", "us-east-1"),
        ("sts", "us-east-1"),
    ]


@pytest.mark.parametrize(
    "extra",
    (
        ["--yes"],
        ["--approve", INTENT_SHA256],
        ["--profile", "default"],
        ["--region", "us-west-2"],
    ),
    ids=(
        "generic-yes",
        "abbreviated-approval",
        "wrong-profile",
        "wrong-region",
    ),
)
def test_cli_rejects_generic_or_wrong_approval_boundaries(
    tmp_path: Path,
    extra: list[str],
):
    intent_path = tmp_path / "launch-intent.json"
    intent_path.write_bytes(INTENT_BYTES)
    arguments = [
        "--intent",
        str(intent_path),
        "--approve-intent-sha256",
        INTENT_SHA256,
        "--profile",
        "sbsandbox",
        "--region",
        "us-east-1",
        *extra,
    ]

    with pytest.raises(SystemExit):
        aws_corpus_builder_launch._parser().parse_args(arguments)
