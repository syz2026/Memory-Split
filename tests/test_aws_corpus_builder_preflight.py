from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import cluster.aws.corpus_builder.preflight as preflight_module
from cluster.aws.corpus_builder.contracts import (
    CORPUS_BUCKET,
    S3ObjectVersion,
    launch_intent_to_bytes,
)
from cluster.aws.corpus_builder.package import (
    ARCHIVE_NAME,
    PACKAGE_FORMAT,
    _REQUIRED_PACKAGE_PATHS,
)
from cluster.aws.corpus_builder.preflight import (
    PREFLIGHT_CHECKS,
    AwsClients,
    PreflightError,
    PreflightRequest,
    run_preflight,
)
from scripts import aws_corpus_builder_preflight


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "cluster" / "profiles" / "aws-i4i.16xlarge-corpus-v1.json"

NOW = datetime(2026, 7, 25, 4, 0, 0, tzinfo=timezone.utc)
ACCOUNT_ID = "056956104102"
REGION = "us-east-1"
AMI_ID = "ami-0123456789abcdef0"
AMI_OWNER_ID = "137112412989"
LAUNCH_TEMPLATE_ID = "lt-0123456789abcdef0"
LAUNCH_TEMPLATE_VERSION = "7"
SUBNET_ID = "subnet-0123456789abcdef0"
SECURITY_GROUP_ID = "sg-0123456789abcdef0"
VPC_ID = "vpc-0123456789abcdef0"
INSTANCE_PROFILE_ARN = (
    "arn:aws:iam::056956104102:instance-profile/memorysplit-corpus-builder"
)
BUILDER_ROLE_ARN = "arn:aws:iam::056956104102:role/memorysplit-corpus-builder"
KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/01234567-89ab-cdef-0123-456789abcdef"
)
OTHER_KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/fedcba98-7654-3210-fedc-ba9876543210"
)
BUILD_ID = "b" * 64

EXPECTED_CHECKS = (
    "profile-canonical-sha256",
    "exact-s3-versions",
    "production-software-gate",
    "account-and-region",
    "ami-identity",
    "instance-type-availability",
    "launch-template-version",
    "private-network-and-security-group",
    "instance-profile-and-builder-role",
    "bucket-and-kms",
    "linux-on-demand-price",
    "maximum-compute-cost",
    "ec2-run-instances-dry-run",
)


def _canonical_json(value: object) -> bytes:
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


def _object_record(name: str, digest_character: str) -> S3ObjectVersion:
    return S3ObjectVersion(
        uri=f"s3://{CORPUS_BUCKET}/v2/builds/{BUILD_ID}/{name}",
        version_id=f"{name}-version-001",
        bytes=4096 if name == ARCHIVE_NAME else 1024,
        sha256=digest_character * 64,
        etag=digest_character * 32,
        sse_algorithm="aws:kms",
        kms_key_arn=KMS_ARN,
    )


def _package() -> S3ObjectVersion:
    return _object_record(ARCHIVE_NAME, "a")


def _source_manifest() -> S3ObjectVersion:
    return _object_record("source-manifest.json", "c")


def _stack_outputs() -> dict[str, str]:
    return {
        "ArtifactBucketName": CORPUS_BUCKET,
        "BuilderInstanceProfileArn": INSTANCE_PROFILE_ARN,
        "BuilderRoleArn": BUILDER_ROLE_ARN,
        "ControllerRoleArn": (
            "arn:aws:iam::056956104102:role/memorysplit-corpus-controller"
        ),
        "DataKeyArn": KMS_ARN,
        "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
        "LaunchTemplateVersion": LAUNCH_TEMPLATE_VERSION,
        "PrivateSubnetId": SUBNET_ID,
        "SecurityGroupId": SECURITY_GROUP_ID,
        "SsmDocumentName": "memorysplit-corpus-builder",
        "VpcId": VPC_ID,
    }


def _software_gate_receipt(
    path: Path,
    package: S3ObjectVersion,
    *,
    complete: bool = True,
) -> Path:
    rows = []
    for member in sorted(_REQUIRED_PACKAGE_PATHS):
        digest = hashlib.sha256(member.encode("utf-8")).hexdigest()
        rows.append(
            {
                "bytes": len(member.encode("utf-8")),
                "mode": "0644",
                "object_id": "d" * 40,
                "path": member,
                "sha256": digest,
            }
        )
    archive_sha256 = package.sha256 if complete else "f" * 64
    path.write_bytes(
        _canonical_json(
            {
                "archive": {
                    "bytes": package.bytes,
                    "path": ARCHIVE_NAME,
                    "sha256": archive_sha256,
                },
                "format": PACKAGE_FORMAT,
                "members": rows,
                "revision": "e" * 40,
                "schema_version": 1,
            }
        )
    )
    return path


class FakeAwsError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class _Meta:
    def __init__(self, region_name: str) -> None:
        self.region_name = region_name


class FakeSts:
    def __init__(self, owner: "FakeAws") -> None:
        self.owner = owner

    def get_caller_identity(self) -> dict[str, str]:
        self.owner.calls.append("sts.get_caller_identity")
        return {
            "Account": self.owner.account_id,
            "Arn": f"arn:aws:sts::{self.owner.account_id}:assumed-role/controller/session",
            "UserId": "controller:session",
        }


class FakeEc2:
    def __init__(self, owner: "FakeAws") -> None:
        self.owner = owner
        self.meta = _Meta(REGION)
        self.run_instances_calls: list[dict[str, object]] = []

    def describe_images(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("ec2.describe_images")
        assert kwargs == {"ImageIds": [AMI_ID], "Owners": [AMI_OWNER_ID]}
        image = {
            "Architecture": "x86_64",
            "CreationDate": "2026-07-01T00:00:00.000Z",
            "ImageId": AMI_ID,
            "OwnerId": AMI_OWNER_ID,
            "RootDeviceName": "/dev/sda1",
            "RootDeviceType": "ebs",
            "State": "available",
        }
        image.update(self.owner.image_overrides)
        return {"Images": [image]}

    def describe_subnets(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("ec2.describe_subnets")
        assert kwargs == {"SubnetIds": [SUBNET_ID]}
        subnet = {
            "AvailabilityZone": "us-east-1a",
            "AvailabilityZoneId": "use1-az1",
            "MapPublicIpOnLaunch": False,
            "State": "available",
            "SubnetId": SUBNET_ID,
            "VpcId": VPC_ID,
        }
        subnet.update(self.owner.subnet_overrides)
        return {"Subnets": [subnet]}

    def describe_instance_type_offerings(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.owner.calls.append(
            f"ec2.describe_instance_type_offerings:{kwargs['LocationType']}"
        )
        location_type = kwargs["LocationType"]
        assert location_type in {"region", "availability-zone"}
        expected_location = REGION if location_type == "region" else "us-east-1a"
        assert kwargs == {
            "Filters": [
                {"Name": "instance-type", "Values": ["i4i.16xlarge"]},
                {"Name": "location", "Values": [expected_location]},
            ],
            "LocationType": location_type,
        }
        if self.owner.instance_unavailable == location_type:
            return {"InstanceTypeOfferings": []}
        return {
            "InstanceTypeOfferings": [
                {
                    "InstanceType": "i4i.16xlarge",
                    "Location": expected_location,
                    "LocationType": location_type,
                }
            ]
        }

    def describe_launch_template_versions(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.owner.calls.append("ec2.describe_launch_template_versions")
        assert kwargs == {
            "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
            "Versions": [LAUNCH_TEMPLATE_VERSION],
        }
        network_interface = {
            "AssociatePublicIpAddress": False,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [SECURITY_GROUP_ID],
            "SubnetId": SUBNET_ID,
        }
        network_interface.update(self.owner.network_overrides)
        launch_data = {
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "DeleteOnTermination": True,
                        "Encrypted": True,
                        "VolumeSize": 200,
                        "VolumeType": "gp3",
                    },
                }
            ],
            "DisableApiTermination": False,
            "IamInstanceProfile": {"Arn": INSTANCE_PROFILE_ARN},
            "ImageId": AMI_ID,
            "InstanceInitiatedShutdownBehavior": "terminate",
            "InstanceType": "i4i.16xlarge",
            "MetadataOptions": {
                "HttpEndpoint": "enabled",
                "HttpProtocolIpv6": "disabled",
                "HttpPutResponseHopLimit": 1,
                "HttpTokens": "required",
                "InstanceMetadataTags": "disabled",
            },
            "Monitoring": {"Enabled": True},
            "NetworkInterfaces": [network_interface],
        }
        launch_data.update(self.owner.launch_data_overrides)
        version = {
            "LaunchTemplateData": launch_data,
            "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
            "VersionNumber": int(LAUNCH_TEMPLATE_VERSION),
        }
        version.update(self.owner.launch_version_overrides)
        return {"LaunchTemplateVersions": [version]}

    def describe_security_groups(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("ec2.describe_security_groups")
        assert kwargs == {"GroupIds": [SECURITY_GROUP_ID]}
        group = {
            "GroupId": SECURITY_GROUP_ID,
            "IpPermissions": [],
            "IpPermissionsEgress": [
                {
                    "IpProtocol": "-1",
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }
            ],
            "VpcId": VPC_ID,
        }
        group.update(self.owner.security_group_overrides)
        return {"SecurityGroups": [group]}

    def run_instances(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("ec2.run_instances")
        self.run_instances_calls.append(dict(kwargs))
        code = (
            "UnauthorizedOperation"
            if self.owner.dry_run_denied
            else "DryRunOperation"
        )
        raise FakeAwsError(code)


class FakeIam:
    def __init__(self, owner: "FakeAws") -> None:
        self.owner = owner

    def get_instance_profile(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("iam.get_instance_profile")
        assert kwargs == {"InstanceProfileName": "memorysplit-corpus-builder"}
        profile = {
            "Arn": INSTANCE_PROFILE_ARN,
            "InstanceProfileName": "memorysplit-corpus-builder",
            "Roles": [
                {
                    "Arn": BUILDER_ROLE_ARN,
                    "Path": "/",
                    "RoleName": "memorysplit-corpus-builder",
                }
            ],
        }
        profile.update(self.owner.instance_profile_overrides)
        return {"InstanceProfile": profile}

    def get_role(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("iam.get_role")
        assert kwargs == {"RoleName": "memorysplit-corpus-builder"}
        role = {
            "Arn": BUILDER_ROLE_ARN,
            "Path": "/",
            "RoleName": "memorysplit-corpus-builder",
        }
        role.update(self.owner.role_overrides)
        return {"Role": role}


class FakeS3:
    def __init__(self, owner: "FakeAws") -> None:
        self.owner = owner

    def head_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        self.owner.calls.append(f"s3.head_object:{key.rsplit('/', 1)[-1]}")
        records = (self.owner.package, self.owner.source_manifest)
        record = next(
            item
            for item in records
            if urlsplit(item.uri).path.removeprefix("/") == key
        )
        assert kwargs == {
            "Bucket": CORPUS_BUCKET,
            "Key": key,
            "VersionId": record.version_id,
        }
        digest = record.sha256
        if self.owner.source_drift and record == self.owner.source_manifest:
            digest = "f" * 64
        return {
            "ContentLength": record.bytes,
            "ETag": f'"{record.etag}"',
            "Metadata": {"sha256": digest},
            "SSEKMSKeyId": record.kms_key_arn,
            "ServerSideEncryption": record.sse_algorithm,
            "VersionId": record.version_id,
        }

    def get_bucket_versioning(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("s3.get_bucket_versioning")
        assert kwargs == {"Bucket": CORPUS_BUCKET}
        return {"Status": "Suspended" if self.owner.bucket_unversioned else "Enabled"}

    def get_public_access_block(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("s3.get_public_access_block")
        assert kwargs == {"Bucket": CORPUS_BUCKET}
        config = {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        }
        config.update(self.owner.public_access_overrides)
        return {"PublicAccessBlockConfiguration": config}

    def get_bucket_ownership_controls(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("s3.get_bucket_ownership_controls")
        assert kwargs == {"Bucket": CORPUS_BUCKET}
        return {
            "OwnershipControls": {
                "Rules": [
                    {
                        "ObjectOwnership": (
                            "ObjectWriter"
                            if self.owner.wrong_bucket_ownership
                            else "BucketOwnerEnforced"
                        )
                    }
                ]
            }
        }

    def get_bucket_encryption(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("s3.get_bucket_encryption")
        assert kwargs == {"Bucket": CORPUS_BUCKET}
        return {
            "ServerSideEncryptionConfiguration": {
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "KMSMasterKeyID": (
                                OTHER_KMS_ARN
                                if self.owner.wrong_bucket_encryption
                                else KMS_ARN
                            ),
                            "SSEAlgorithm": "aws:kms",
                        },
                        "BucketKeyEnabled": True,
                    }
                ]
            }
        }


class FakeKms:
    def __init__(self, owner: "FakeAws") -> None:
        self.owner = owner

    def describe_key(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("kms.describe_key")
        assert kwargs == {"KeyId": KMS_ARN}
        metadata = {
            "Arn": KMS_ARN,
            "Enabled": True,
            "KeyManager": "CUSTOMER",
            "KeyState": "Enabled",
            "KeyUsage": "ENCRYPT_DECRYPT",
            "MultiRegion": False,
            "Origin": "AWS_KMS",
        }
        metadata.update(self.owner.kms_overrides)
        return {"KeyMetadata": metadata}


class FakePricing:
    def __init__(self, owner: "FakeAws") -> None:
        self.owner = owner
        self.meta = _Meta(REGION)

    def get_products(self, **kwargs: object) -> dict[str, object]:
        self.owner.calls.append("pricing.get_products")
        assert kwargs == {
            "Filters": [
                {
                    "Field": "instanceType",
                    "Type": "TERM_MATCH",
                    "Value": "i4i.16xlarge",
                },
                {
                    "Field": "location",
                    "Type": "TERM_MATCH",
                    "Value": "US East (N. Virginia)",
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
            ],
            "FormatVersion": "aws_v1",
            "MaxResults": 100,
            "ServiceCode": "AmazonEC2",
        }
        sku = "memorysplit-i4i-16xlarge"
        product = {
            "product": {
                "attributes": {
                    "capacitystatus": "Used",
                    "instanceType": "i4i.16xlarge",
                    "location": "US East (N. Virginia)",
                    "operatingSystem": "Linux",
                    "preInstalledSw": "NA",
                    "tenancy": "Shared",
                },
                "productFamily": "Compute Instance",
                "sku": sku,
            },
            "terms": {
                "OnDemand": {
                    f"{sku}.term": {
                        "effectiveDate": "2026-07-01T00:00:00Z",
                        "priceDimensions": {
                            f"{sku}.term.dimension": {
                                "beginRange": "0",
                                "endRange": "Inf",
                                "pricePerUnit": {
                                    "USD": str(self.owner.hourly_price)
                                },
                                "unit": "Hrs",
                            }
                        },
                        "termAttributes": {},
                    }
                }
            },
        }
        return {"PriceList": [json.dumps(product)], "NextToken": ""}


class FakeAws:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.package = _package()
        self.source_manifest = _source_manifest()
        self.account_id = ACCOUNT_ID
        self.image_overrides: dict[str, object] = {}
        self.subnet_overrides: dict[str, object] = {}
        self.instance_unavailable: str | None = None
        self.launch_data_overrides: dict[str, object] = {}
        self.launch_version_overrides: dict[str, object] = {}
        self.network_overrides: dict[str, object] = {}
        self.security_group_overrides: dict[str, object] = {}
        self.instance_profile_overrides: dict[str, object] = {}
        self.role_overrides: dict[str, object] = {}
        self.source_drift = False
        self.bucket_unversioned = False
        self.public_access_overrides: dict[str, object] = {}
        self.wrong_bucket_ownership = False
        self.wrong_bucket_encryption = False
        self.kms_overrides: dict[str, object] = {}
        self.hourly_price = Decimal("5.491")
        self.dry_run_denied = False
        self.sts = FakeSts(self)
        self.ec2 = FakeEc2(self)
        self.iam = FakeIam(self)
        self.s3 = FakeS3(self)
        self.kms = FakeKms(self)
        self.pricing = FakePricing(self)

    def clients(self) -> AwsClients:
        return AwsClients(
            sts=self.sts,
            ec2=self.ec2,
            iam=self.iam,
            s3=self.s3,
            kms=self.kms,
            pricing=self.pricing,
        )


def _request(tmp_path: Path, fake: FakeAws) -> PreflightRequest:
    return PreflightRequest(
        profile_path=PROFILE_PATH,
        package=fake.package,
        source_manifest=fake.source_manifest,
        software_gate_receipt=_software_gate_receipt(
            tmp_path / "software-gate.json",
            fake.package,
        ),
        stack_outputs=_stack_outputs(),
        ami_id=AMI_ID,
        ami_owner_id=AMI_OWNER_ID,
    )


def test_preflight_emits_intent_only_after_every_read_only_gate_and_ec2_dry_run(
    tmp_path,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)

    result = run_preflight(request, aws=fake.clients(), now=NOW)

    assert PREFLIGHT_CHECKS == EXPECTED_CHECKS
    assert result.checks == EXPECTED_CHECKS
    assert result.intent.profile_sha256 == hashlib.sha256(
        PROFILE_PATH.read_bytes()
    ).hexdigest()
    assert result.intent.package == fake.package
    assert result.intent.source_manifest == fake.source_manifest
    assert result.intent.hourly_usd == Decimal("5.491")
    assert result.intent.max_compute_usd == Decimal("131.78")
    assert result.intent.not_after == "2026-07-25T04:30:00Z"
    intent_bytes = launch_intent_to_bytes(result.intent)
    assert result.intent_sha256 == hashlib.sha256(intent_bytes).hexdigest()
    assert fake.ec2.run_instances_calls == [
        {
            "DryRun": True,
            "LaunchTemplate": {
                "LaunchTemplateId": LAUNCH_TEMPLATE_ID,
                "Version": LAUNCH_TEMPLATE_VERSION,
            },
            "MaxCount": 1,
            "MinCount": 1,
        }
    ]
    assert fake.calls == [
        f"s3.head_object:{ARCHIVE_NAME}",
        "s3.head_object:source-manifest.json",
        "sts.get_caller_identity",
        "ec2.describe_images",
        "ec2.describe_subnets",
        "ec2.describe_instance_type_offerings:region",
        "ec2.describe_instance_type_offerings:availability-zone",
        "ec2.describe_launch_template_versions",
        "ec2.describe_security_groups",
        "iam.get_instance_profile",
        "iam.get_role",
        "s3.get_bucket_versioning",
        "s3.get_public_access_block",
        "s3.get_bucket_ownership_controls",
        "s3.get_bucket_encryption",
        "kms.describe_key",
        "pricing.get_products",
        "ec2.run_instances",
    ]


@pytest.mark.parametrize(
    "failure",
    (
        "dirty-package",
        "source-drift",
        "wrong-account",
        "wrong-region",
        "wrong-ami-owner",
        "instance-unavailable",
        "hourly-price-over-cap",
        "bucket-unversioned",
        "wrong-kms-key",
        "wrong-launch-template",
        "dry-run-denied",
    ),
)
def test_preflight_fails_closed_without_launch_intent(
    tmp_path,
    monkeypatch,
    failure,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    if failure == "dirty-package":
        request = replace(
            request,
            software_gate_receipt=_software_gate_receipt(
                tmp_path / "dirty-package.json",
                fake.package,
                complete=False,
            ),
        )
    elif failure == "source-drift":
        fake.source_drift = True
    elif failure == "wrong-account":
        fake.account_id = "999999999999"
    elif failure == "wrong-region":
        fake.ec2.meta.region_name = "us-west-2"
    elif failure == "wrong-ami-owner":
        fake.image_overrides["OwnerId"] = ACCOUNT_ID
    elif failure == "instance-unavailable":
        fake.instance_unavailable = "availability-zone"
    elif failure == "hourly-price-over-cap":
        fake.hourly_price = Decimal("5.492")
    elif failure == "bucket-unversioned":
        fake.bucket_unversioned = True
    elif failure == "wrong-kms-key":
        request = replace(
            request,
            stack_outputs={**request.stack_outputs, "DataKeyArn": OTHER_KMS_ARN},
        )
    elif failure == "wrong-launch-template":
        fake.launch_version_overrides["VersionNumber"] = 8
    elif failure == "dry-run-denied":
        fake.dry_run_denied = True
    else:
        raise AssertionError(failure)

    emitted = []
    real_serializer = launch_intent_to_bytes

    def track_emission(intent):
        emitted.append(intent)
        return real_serializer(intent)

    monkeypatch.setattr(
        preflight_module,
        "launch_intent_to_bytes",
        track_emission,
    )

    with pytest.raises(PreflightError):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert emitted == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("Architecture", "arm64"),
        ("State", "pending"),
        ("RootDeviceType", "instance-store"),
        ("RootDeviceName", ""),
        ("CreationDate", "2027-07-01T00:00:00Z"),
    ),
)
def test_preflight_rejects_each_ami_identity_drift(tmp_path, field, value):
    fake = FakeAws()
    fake.image_overrides[field] = value

    with pytest.raises(PreflightError, match="ami-identity"):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


@pytest.mark.parametrize(
    "failure",
    (
        "public-subnet",
        "public-template-interface",
        "extra-template-security-group",
        "ingress-rule",
        "wrong-vpc",
    ),
)
def test_preflight_rejects_every_public_network_or_ingress_path(
    tmp_path,
    failure,
):
    fake = FakeAws()
    if failure == "public-subnet":
        fake.subnet_overrides["MapPublicIpOnLaunch"] = True
    elif failure == "public-template-interface":
        fake.network_overrides["AssociatePublicIpAddress"] = True
    elif failure == "extra-template-security-group":
        fake.network_overrides["Groups"] = [
            SECURITY_GROUP_ID,
            "sg-fedcba98765432100",
        ]
    elif failure == "ingress-rule":
        fake.security_group_overrides["IpPermissions"] = [
            {
                "FromPort": 22,
                "IpProtocol": "tcp",
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                "ToPort": 22,
            }
        ]
    elif failure == "wrong-vpc":
        fake.security_group_overrides["VpcId"] = "vpc-fedcba98765432100"
    else:
        raise AssertionError(failure)

    with pytest.raises(
        PreflightError,
        match="private-network-and-security-group",
    ):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


@pytest.mark.parametrize(
    "failure",
    (
        "wrong-template-profile",
        "extra-profile-role",
        "wrong-role-arn",
    ),
)
def test_preflight_rejects_instance_profile_or_builder_role_drift(
    tmp_path,
    failure,
):
    fake = FakeAws()
    if failure == "wrong-template-profile":
        fake.launch_data_overrides["IamInstanceProfile"] = {
            "Arn": (
                "arn:aws:iam::056956104102:"
                "instance-profile/other-corpus-builder"
            )
        }
    elif failure == "extra-profile-role":
        fake.instance_profile_overrides["Roles"] = [
            {
                "Arn": BUILDER_ROLE_ARN,
                "Path": "/",
                "RoleName": "memorysplit-corpus-builder",
            },
            {
                "Arn": "arn:aws:iam::056956104102:role/other",
                "Path": "/",
                "RoleName": "other",
            },
        ]
    elif failure == "wrong-role-arn":
        fake.role_overrides["Arn"] = (
            "arn:aws:iam::056956104102:role/other-corpus-builder"
        )
    else:
        raise AssertionError(failure)

    with pytest.raises(
        PreflightError,
        match="instance-profile-and-builder-role",
    ):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


@pytest.mark.parametrize(
    "failure",
    (
        "public-access-not-blocked",
        "wrong-ownership",
        "wrong-bucket-encryption",
        "disabled-kms-key",
    ),
)
def test_preflight_rejects_each_bucket_and_kms_drift(tmp_path, failure):
    fake = FakeAws()
    if failure == "public-access-not-blocked":
        fake.public_access_overrides["RestrictPublicBuckets"] = False
    elif failure == "wrong-ownership":
        fake.wrong_bucket_ownership = True
    elif failure == "wrong-bucket-encryption":
        fake.wrong_bucket_encryption = True
    elif failure == "disabled-kms-key":
        fake.kms_overrides["Enabled"] = False
        fake.kms_overrides["KeyState"] = "Disabled"
    else:
        raise AssertionError(failure)

    with pytest.raises(PreflightError, match="bucket-and-kms"):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


def test_hourly_price_cap_fails_in_the_price_gate(tmp_path):
    fake = FakeAws()
    fake.hourly_price = Decimal("5.492")

    with pytest.raises(PreflightError, match="linux-on-demand-price"):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


def test_noncanonical_profile_fails_before_any_aws_call(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_bytes(PROFILE_PATH.read_bytes() + b" ")
    fake = FakeAws()
    request = replace(_request(tmp_path, fake), profile_path=profile)

    with pytest.raises(PreflightError, match="profile-canonical-sha256"):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert fake.calls == []


@pytest.mark.parametrize(
    "now",
    (
        datetime(2026, 7, 25, 4, 0, 0),
        datetime(2026, 7, 25, 4, 0, 0, 1, tzinfo=timezone.utc),
    ),
)
def test_preflight_requires_canonical_utc_second(tmp_path, now):
    fake = FakeAws()

    with pytest.raises(PreflightError, match="UTC"):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=now)

    assert fake.calls == []


def _write_object_record(path: Path, value: S3ObjectVersion) -> None:
    path.write_bytes(
        _canonical_json(
            {
                "bytes": value.bytes,
                "etag": value.etag,
                "kms_key_arn": value.kms_key_arn,
                "sha256": value.sha256,
                "sse_algorithm": value.sse_algorithm,
                "uri": value.uri,
                "version_id": value.version_id,
            }
        )
    )


def test_preflight_cli_writes_canonical_owner_only_intent_and_prints_hash(
    tmp_path,
    capsys,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    package_path = tmp_path / "package-record.json"
    source_path = tmp_path / "source-record.json"
    outputs_path = tmp_path / "stack-outputs.json"
    intent_path = tmp_path / "launch-intent.json"
    _write_object_record(package_path, fake.package)
    _write_object_record(source_path, fake.source_manifest)
    outputs_path.write_bytes(_canonical_json(dict(request.stack_outputs)))

    result = aws_corpus_builder_preflight.main(
        [
            "--builder-profile",
            str(PROFILE_PATH),
            "--package-record",
            str(package_path),
            "--source-manifest-record",
            str(source_path),
            "--software-gate-receipt",
            str(request.software_gate_receipt),
            "--stack-outputs",
            str(outputs_path),
            "--ami-id",
            AMI_ID,
            "--ami-owner-id",
            AMI_OWNER_ID,
            "--intent",
            str(intent_path),
            "--profile",
            "sbsandbox",
            "--region",
            REGION,
        ],
        aws=fake.clients(),
        now=NOW,
    )

    assert result == 0
    payload = intent_path.read_bytes()
    assert payload == launch_intent_to_bytes(
        run_preflight(request, aws=FakeAws().clients(), now=NOW).intent
    )
    assert stat.S_IMODE(intent_path.stat().st_mode) == 0o600
    assert capsys.readouterr().out == f"{hashlib.sha256(payload).hexdigest()}\n"


def test_preflight_cli_never_creates_intent_when_a_gate_fails(
    tmp_path,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    package_path = tmp_path / "package-record.json"
    source_path = tmp_path / "source-record.json"
    outputs_path = tmp_path / "stack-outputs.json"
    intent_path = tmp_path / "launch-intent.json"
    _write_object_record(package_path, fake.package)
    _write_object_record(source_path, fake.source_manifest)
    outputs_path.write_bytes(_canonical_json(dict(request.stack_outputs)))
    fake.dry_run_denied = True

    with pytest.raises(PreflightError):
        aws_corpus_builder_preflight.main(
            [
                "--builder-profile",
                str(PROFILE_PATH),
                "--package-record",
                str(package_path),
                "--source-manifest-record",
                str(source_path),
                "--software-gate-receipt",
                str(request.software_gate_receipt),
                "--stack-outputs",
                str(outputs_path),
                "--ami-id",
                AMI_ID,
                "--ami-owner-id",
                AMI_OWNER_ID,
                "--intent",
                str(intent_path),
                "--profile",
                "sbsandbox",
                "--region",
                REGION,
            ],
            aws=fake.clients(),
            now=NOW,
        )

    assert not intent_path.exists()


def test_preflight_cli_removes_partial_intent_when_fsync_fails(
    tmp_path,
    monkeypatch,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    package_path = tmp_path / "package-record.json"
    source_path = tmp_path / "source-record.json"
    outputs_path = tmp_path / "stack-outputs.json"
    intent_path = tmp_path / "launch-intent.json"
    _write_object_record(package_path, fake.package)
    _write_object_record(source_path, fake.source_manifest)
    outputs_path.write_bytes(_canonical_json(dict(request.stack_outputs)))

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(aws_corpus_builder_preflight.os, "fsync", fail_fsync)

    with pytest.raises(PreflightError, match="emit"):
        aws_corpus_builder_preflight.main(
            [
                "--builder-profile",
                str(PROFILE_PATH),
                "--package-record",
                str(package_path),
                "--source-manifest-record",
                str(source_path),
                "--software-gate-receipt",
                str(request.software_gate_receipt),
                "--stack-outputs",
                str(outputs_path),
                "--ami-id",
                AMI_ID,
                "--ami-owner-id",
                AMI_OWNER_ID,
                "--intent",
                str(intent_path),
                "--profile",
                "sbsandbox",
                "--region",
                REGION,
            ],
            aws=fake.clients(),
            now=NOW,
        )

    assert not intent_path.exists()
