from __future__ import annotations

import base64
import gzip
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
    load_corpus_builder_profile,
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
OTHER_SUBNET_ID = "subnet-fedcba98765432100"
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
PACKAGE_REVISION = "e" * 40
AUTHORITATIVE_BOOTSTRAP_GZIP_BASE64 = (
    "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"
    "WMZV0bA/S7z/Azkg7aWGjgAA"
)
BOOTSTRAP_USER_DATA = base64.b64decode(
    AUTHORITATIVE_BOOTSTRAP_GZIP_BASE64,
    validate=True,
)
AUTHORITATIVE_BOOTSTRAP_USER_DATA_SHA256 = (
    "11f5fbfd7bee7a01e42956654020d21eeec7455305a2da3c8c7e9fb37a0b139f"
)

EXPECTED_CHECKS = (
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


@pytest.fixture(autouse=True)
def _forbid_real_live_clients(monkeypatch):
    def forbidden_live_clients(**_kwargs):
        raise AssertionError("real boto3 clients are forbidden during tests")

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_live_clients",
        forbidden_live_clients,
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
        "BootstrapUserDataSha256": AUTHORITATIVE_BOOTSTRAP_USER_DATA_SHA256,
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
    revision: str = PACKAGE_REVISION,
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
                "revision": revision,
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
            "UserData": base64.b64encode(
                self.owner.bootstrap_user_data
            ).decode("ascii"),
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
        if self.owner.stack_outputs_to_mutate is not None:
            self.owner.stack_outputs_to_mutate["PrivateSubnetId"] = OTHER_SUBNET_ID
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
        metadata = {"sha256": digest}
        if record == self.owner.package:
            metadata["revision"] = self.owner.package_revision
        return {
            "ContentLength": record.bytes,
            "ETag": f'"{record.etag}"',
            "Metadata": metadata,
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
        self.package_revision = PACKAGE_REVISION
        self.bootstrap_user_data = BOOTSTRAP_USER_DATA
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
        self.stack_outputs_to_mutate: dict[str, str] | None = None
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


def test_request_cannot_override_profile_bootstrap_user_data_hash(tmp_path):
    fake = FakeAws()

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        replace(
            _request(tmp_path, fake),
            expected_bootstrap_user_data_sha256="f" * 64,
        )


def test_profile_bootstrap_hash_matches_authoritative_build_invariant_payload():
    profile = load_corpus_builder_profile(PROFILE_PATH)

    assert len(AUTHORITATIVE_BOOTSTRAP_GZIP_BASE64) == 12_792
    assert hashlib.sha256(
        AUTHORITATIVE_BOOTSTRAP_GZIP_BASE64.encode("ascii")
    ).hexdigest() == (
        "0e93ee325ff77776caf8e5e910e6a7b84850d901d5b596981ad3525fd5fff50d"
    )
    assert len(BOOTSTRAP_USER_DATA) == 9_594
    assert hashlib.sha256(BOOTSTRAP_USER_DATA).hexdigest() == (
        AUTHORITATIVE_BOOTSTRAP_USER_DATA_SHA256
    )
    rendered = gzip.decompress(BOOTSTRAP_USER_DATA)
    assert len(rendered) == 36_486
    assert hashlib.sha256(rendered).hexdigest() == (
        "7a551e1bc5bfc614c3ca699c8359b46143e4c10417a68be1f75b5b73edfdda37"
    )
    assert profile.bootstrap_user_data_sha256 == (
        AUTHORITATIVE_BOOTSTRAP_USER_DATA_SHA256
    )


@pytest.mark.parametrize(
    ("profile_hash", "message"),
    (
        (None, r"missing=.*bootstrap_user_data_sha256"),
        (
            "not-a-sha256",
            (
                r"profile\.bootstrap_user_data_sha256 "
                r"must be a lowercase SHA-256"
            ),
        ),
    ),
)
def test_profile_missing_or_malformed_bootstrap_hash_fails_closed(
    tmp_path,
    profile_hash,
    message,
):
    fake = FakeAws()
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile_hash is None:
        value.pop("bootstrap_user_data_sha256", None)
    else:
        value["bootstrap_user_data_sha256"] = profile_hash
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(_canonical_json(value))
    request = replace(_request(tmp_path, fake), profile_path=profile_path)

    with pytest.raises(
        PreflightError,
        match=rf"profile-canonical-sha256.*{message}",
    ):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert fake.calls == []


def test_alternate_profile_cannot_override_pinned_bootstrap_hash(tmp_path):
    fake = FakeAws()
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    value["bootstrap_user_data_sha256"] = "f" * 64
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(_canonical_json(value))
    request = replace(_request(tmp_path, fake), profile_path=profile_path)

    with pytest.raises(
        PreflightError,
        match=(
            "profile-canonical-sha256.*"
            r"profile\.bootstrap_user_data_sha256 drift"
        ),
    ):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert fake.calls == []


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
        f"s3.head_object:{ARCHIVE_NAME}",
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


def test_bootstrap_user_data_gate_accepts_exact_reviewed_payload(tmp_path):
    fake = FakeAws()

    result = run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert len(result.checks) == 14
    assert result.checks[6:9] == (
        "launch-template-version",
        "bootstrap-user-data-sha256",
        "private-network-and-security-group",
    )


def test_bootstrap_user_data_mismatch_fails_before_intent_or_network_gate(
    tmp_path,
    monkeypatch,
):
    fake = FakeAws()
    fake.bootstrap_user_data = b"#!/bin/bash\nexit 0\n"
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

    with pytest.raises(
        PreflightError,
        match=(
            "bootstrap-user-data-sha256.*"
            r"UserData SHA-256.*profile\.bootstrap_user_data_sha256"
        ),
    ):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert emitted == []
    assert fake.calls[-1] == "ec2.describe_launch_template_versions"
    assert "ec2.describe_security_groups" not in fake.calls
    assert fake.ec2.run_instances_calls == []


@pytest.mark.parametrize(
    "stack_hash",
    (None, "not-a-sha256"),
)
def test_bootstrap_user_data_gate_rejects_missing_or_malformed_stack_hash(
    tmp_path,
    stack_hash,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    outputs = dict(request.stack_outputs)
    if stack_hash is None:
        outputs.pop("BootstrapUserDataSha256")
    else:
        outputs["BootstrapUserDataSha256"] = stack_hash
    request = replace(request, stack_outputs=outputs)

    with pytest.raises(
        PreflightError,
        match="bootstrap-user-data-sha256.*BootstrapUserDataSha256",
    ):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert fake.calls[-1] == "ec2.describe_launch_template_versions"
    assert fake.ec2.run_instances_calls == []


def test_bootstrap_stack_hash_must_match_frozen_profile(tmp_path):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    outputs = dict(request.stack_outputs)
    outputs["BootstrapUserDataSha256"] = "f" * 64
    request = replace(request, stack_outputs=outputs)

    with pytest.raises(
        PreflightError,
        match=(
            "bootstrap-user-data-sha256.*"
            r"BootstrapUserDataSha256.*profile\.bootstrap_user_data_sha256"
        ),
    ):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert fake.calls[-1] == "ec2.describe_launch_template_versions"
    assert fake.ec2.run_instances_calls == []


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


def test_preflight_rejects_manifest_forged_for_an_unrelated_revision(tmp_path):
    fake = FakeAws()
    request = replace(
        _request(tmp_path, fake),
        software_gate_receipt=_software_gate_receipt(
            tmp_path / "forged-software-gate.json",
            fake.package,
            revision="f" * 40,
        ),
    )

    with pytest.raises(PreflightError, match="production-software-gate.*revision"):
        run_preflight(request, aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


def test_missing_package_revision_names_required_s3_metadata_key(tmp_path):
    fake = FakeAws()
    fake.package_revision = ""

    with pytest.raises(
        PreflightError,
        match=r'Metadata\["revision"\]',
    ):
        run_preflight(_request(tmp_path, fake), aws=fake.clients(), now=NOW)

    assert fake.ec2.run_instances_calls == []


def test_intent_uses_the_validated_stack_output_snapshot(tmp_path):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    outputs = request.stack_outputs
    assert isinstance(outputs, dict)
    fake.stack_outputs_to_mutate = outputs

    result = run_preflight(request, aws=fake.clients(), now=NOW)

    assert outputs["PrivateSubnetId"] == OTHER_SUBNET_ID
    assert result.intent.subnet_id == SUBNET_ID


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


def _cli_arguments(
    tmp_path: Path,
    fake: FakeAws,
    request: PreflightRequest,
    *,
    live: bool = False,
) -> tuple[list[str], Path, dict[str, Path]]:
    package_path = tmp_path / "package-record.json"
    source_path = tmp_path / "source-record.json"
    outputs_path = tmp_path / "stack-outputs.json"
    intent_path = tmp_path / "launch-intent.json"
    _write_object_record(package_path, fake.package)
    _write_object_record(source_path, fake.source_manifest)
    outputs_path.write_bytes(_canonical_json(dict(request.stack_outputs)))
    arguments = [
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
    ]
    if live:
        arguments.append("--live")
    return (
        arguments,
        intent_path,
        {
            "package": package_path,
            "source": source_path,
            "stack_outputs": outputs_path,
        },
    )


def test_cli_rejects_caller_bootstrap_hash_override(tmp_path):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    arguments, intent_path, _paths = _cli_arguments(tmp_path, fake, request)
    arguments.extend(
        [
            "--expected-bootstrap-user-data-sha256",
            "f" * 64,
        ]
    )

    with pytest.raises(SystemExit):
        aws_corpus_builder_preflight.main(
            arguments,
            aws=fake.clients(),
            now=NOW,
        )

    assert not intent_path.exists()


@pytest.mark.parametrize(
    "failure",
    ("profile", "package-record", "software-gate"),
)
def test_explicit_live_clients_wait_for_every_purely_local_gate(
    tmp_path,
    monkeypatch,
    failure,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    arguments, intent_path, paths = _cli_arguments(
        tmp_path,
        fake,
        request,
        live=True,
    )
    if failure == "profile":
        profile = tmp_path / "invalid-profile.json"
        profile.write_bytes(PROFILE_PATH.read_bytes() + b" ")
        arguments[arguments.index("--builder-profile") + 1] = str(profile)
    elif failure == "package-record":
        paths["package"].write_text("{}", encoding="utf-8")
    elif failure == "software-gate":
        _software_gate_receipt(
            request.software_gate_receipt,
            fake.package,
            complete=False,
        )
    else:
        raise AssertionError(failure)
    constructed = []

    def fake_live_clients(**kwargs):
        constructed.append(kwargs)
        return fake.clients()

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_live_clients",
        fake_live_clients,
    )

    with pytest.raises(PreflightError):
        aws_corpus_builder_preflight.main(arguments, now=NOW)

    assert constructed == []
    assert not intent_path.exists()


def test_cli_default_requires_injected_clients_without_constructing_live(
    tmp_path,
    monkeypatch,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    arguments, intent_path, _paths = _cli_arguments(tmp_path, fake, request)
    constructed = []

    def fake_live_clients(**kwargs):
        constructed.append(kwargs)
        return fake.clients()

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_live_clients",
        fake_live_clients,
    )

    with pytest.raises(PreflightError, match="injected|--live"):
        aws_corpus_builder_preflight.main(arguments, now=NOW)

    assert constructed == []
    assert not intent_path.exists()


def test_cli_explicit_live_opt_in_uses_only_the_patched_client_factory(
    tmp_path,
    monkeypatch,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    arguments, intent_path, _paths = _cli_arguments(
        tmp_path,
        fake,
        request,
        live=True,
    )
    constructed = []

    def fake_live_clients(**kwargs):
        constructed.append(kwargs)
        return fake.clients()

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_live_clients",
        fake_live_clients,
    )

    assert aws_corpus_builder_preflight.main(arguments, now=NOW) == 0
    assert constructed == [{"profile": "sbsandbox", "region": REGION}]
    assert intent_path.is_file()


def test_cli_invalid_clock_fails_before_live_client_construction(
    tmp_path,
    monkeypatch,
):
    fake = FakeAws()
    request = _request(tmp_path, fake)
    arguments, intent_path, _paths = _cli_arguments(
        tmp_path,
        fake,
        request,
        live=True,
    )
    constructed = []

    def fake_live_clients(**kwargs):
        constructed.append(kwargs)
        return fake.clients()

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_live_clients",
        fake_live_clients,
    )

    with pytest.raises(PreflightError, match="UTC"):
        aws_corpus_builder_preflight.main(
            arguments,
            now=datetime(2026, 7, 25, 4, 0, 0),
        )

    assert constructed == []
    assert not intent_path.exists()


def test_atomic_intent_stays_unpublished_until_temporary_is_fsynced(
    tmp_path,
    monkeypatch,
):
    intent_path = tmp_path / "launch-intent.json"
    visible_during_fsync = []

    def interrupt_fsync(_descriptor):
        visible_during_fsync.append(intent_path.exists())
        raise OSError("simulated interruption before publication")

    monkeypatch.setattr(
        aws_corpus_builder_preflight.os,
        "fsync",
        interrupt_fsync,
    )

    with pytest.raises(PreflightError, match="emit"):
        aws_corpus_builder_preflight._write_intent(
            intent_path,
            b'{"complete":true}\n',
        )

    assert visible_during_fsync == [False]
    assert not intent_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_intent_publish_never_replaces_a_racing_destination(
    tmp_path,
    monkeypatch,
):
    intent_path = tmp_path / "launch-intent.json"
    racing_payload = b'{"racing":"winner"}\n'
    real_link = aws_corpus_builder_preflight.os.link

    def race_before_link(source, destination, **kwargs):
        intent_path.write_bytes(racing_payload)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(
        aws_corpus_builder_preflight.os,
        "link",
        race_before_link,
    )

    with pytest.raises(PreflightError, match="emit|exists"):
        aws_corpus_builder_preflight._write_intent(
            intent_path,
            b'{"complete":true}\n',
        )

    assert intent_path.read_bytes() == racing_payload
    assert list(tmp_path.iterdir()) == [intent_path]


def test_initial_temporary_fstat_failure_still_cleans_created_file(
    tmp_path,
    monkeypatch,
):
    intent_path = tmp_path / "launch-intent.json"
    real_fstat = aws_corpus_builder_preflight.os.fstat
    calls = 0

    def fail_first_fstat(descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated initial fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(
        aws_corpus_builder_preflight.os,
        "fstat",
        fail_first_fstat,
    )

    with pytest.raises(PreflightError, match="emit"):
        aws_corpus_builder_preflight._write_intent(
            intent_path,
            b'{"complete":true}\n',
        )

    assert not intent_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_cleanup_unlink_failure_is_reported_without_hiding_primary_error(
    tmp_path,
    monkeypatch,
):
    intent_path = tmp_path / "launch-intent.json"
    real_unlink = aws_corpus_builder_preflight.os.unlink

    def fail_write(_descriptor, _payload):
        raise PreflightError("simulated primary write failure")

    def fail_unlink(*_args, **_kwargs):
        raise PermissionError("simulated cleanup unlink failure")

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_write_all",
        fail_write,
    )
    monkeypatch.setattr(
        aws_corpus_builder_preflight.os,
        "unlink",
        fail_unlink,
    )

    with pytest.raises(
        PreflightError,
        match="simulated primary write failure",
    ) as caught:
        aws_corpus_builder_preflight._write_intent(
            intent_path,
            b'{"complete":true}\n',
        )

    notes = getattr(caught.value, "__notes__", ())
    assert any("cleanup unlink failed" in note for note in notes)
    monkeypatch.setattr(
        aws_corpus_builder_preflight.os,
        "unlink",
        real_unlink,
    )
    for leftover in tmp_path.iterdir():
        leftover.unlink()


def test_interrupted_write_leaves_no_intent_or_temporary_file(
    tmp_path,
    monkeypatch,
):
    intent_path = tmp_path / "launch-intent.json"

    def interrupt_after_partial_write(descriptor, payload):
        aws_corpus_builder_preflight.os.write(descriptor, payload[:4])
        raise KeyboardInterrupt

    monkeypatch.setattr(
        aws_corpus_builder_preflight,
        "_write_all",
        interrupt_after_partial_write,
    )

    with pytest.raises(KeyboardInterrupt):
        aws_corpus_builder_preflight._write_intent(
            intent_path,
            b'{"complete":true}\n',
        )

    assert not intent_path.exists()
    assert list(tmp_path.iterdir()) == []


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
