from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_aws_gpu_launch_request import (
    LaunchRequestError,
    extract_capacity_reservation_id,
    extract_run_instance_ids,
    main,
    validate_launch_request,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6 = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
LEGACY_P5 = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge.json"
AMI_ID = "ami-0123456789abcdef0"
INSTANCE_PROFILE_ARN = (
    "arn:aws:iam::123456789012:instance-profile/memorysplit-gpu"
)
SUBNET_ID = "subnet-0123456789abcdef0"
SECURITY_GROUP_IDS = ["sg-0123456789abcdef0"]
KMS_KEY_ID = (
    "arn:aws:kms:us-east-1:123456789012:"
    "key/12345678-1234-4234-9234-123456789abc"
)
COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
CAPACITY_RESERVATION_ID = "cr-0123456789abcdef0"


def _request(*, p6: bool, count: int = 1) -> dict[str, object]:
    value: dict[str, object] = {
        "ImageId": AMI_ID,
        "InstanceType": "p6-b300.48xlarge" if p6 else "p5.48xlarge",
        "MinCount": count,
        "MaxCount": count,
        "IamInstanceProfile": {"Arn": INSTANCE_PROFILE_ARN},
        "NetworkInterfaces": [
            {
                "AssociatePublicIpAddress": False,
                "DeleteOnTermination": True,
                "DeviceIndex": 0,
                "Groups": SECURITY_GROUP_IDS,
                "SubnetId": SUBNET_ID,
            }
        ],
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "KmsKeyId": KMS_KEY_ID,
                    "VolumeSize": 500,
                    "VolumeType": "gp3",
                },
            }
        ],
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpProtocolIpv6": "disabled",
            "HttpPutResponseHopLimit": 1,
            "HttpTokens": "required",
            "InstanceMetadataTags": "disabled",
        },
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "MemorySplitCohort", "Value": COHORT_ID}
                ],
            },
            {
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "MemorySplitCohort", "Value": COHORT_ID}
                ],
            },
        ],
    }
    if p6:
        value.update(
            {
                "InstanceMarketOptions": {"MarketType": "capacity-block"},
                "CapacityReservationSpecification": {
                    "CapacityReservationTarget": {
                        "CapacityReservationId": CAPACITY_RESERVATION_ID
                    }
                },
            }
        )
    return value


def _write(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return path


def _validate(path: Path, *, p6: bool, count: int = 1):
    return validate_launch_request(
        path,
        profile_path=P6 if p6 else P5,
        region="us-east-1",
        ami_id=AMI_ID,
        instance_profile_arn=INSTANCE_PROFILE_ARN,
        subnet_id=SUBNET_ID,
        security_group_ids=SECURITY_GROUP_IDS,
        ebs_kms_key_id=KMS_KEY_ID,
        root_device_name="/dev/sda1",
        root_volume_gib=500,
        cohort_id=COHORT_ID,
        capacity_reservation_id=CAPACITY_RESERVATION_ID if p6 else None,
    )


def _cli_arguments(request: Path) -> list[str]:
    return [
        "--request",
        str(request),
        "--profile",
        str(P5),
        "--region",
        "us-east-1",
        "--ami-id",
        AMI_ID,
        "--instance-profile-arn",
        INSTANCE_PROFILE_ARN,
        "--subnet-id",
        SUBNET_ID,
        "--security-group-id",
        SECURITY_GROUP_IDS[0],
        "--ebs-kms-key-id",
        KMS_KEY_ID,
        "--root-device-name",
        "/dev/sda1",
        "--root-volume-gib",
        "500",
        "--cohort-id",
        COHORT_ID,
    ]


@pytest.mark.parametrize(("p6", "count"), [(False, 4), (True, 1)])
def test_validator_accepts_only_closed_p5_or_p6_request(tmp_path, p6, count):
    request = _write(
        tmp_path / "launch-request.json",
        _request(p6=p6, count=count),
    )

    report = _validate(request, p6=p6, count=count)

    assert report["ok"] is True
    assert report["request_sha256"] == hashlib.sha256(
        request.read_bytes()
    ).hexdigest()
    assert report["count"] == count
    assert report["capacity_reservation_id"] == (
        CAPACITY_RESERVATION_ID if p6 else None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(KeyName="forbidden-ssh-key"),
        lambda value: value["NetworkInterfaces"][0].update(
            AssociatePublicIpAddress=True
        ),
        lambda value: value["MetadataOptions"].update(HttpTokens="optional"),
        lambda value: value["BlockDeviceMappings"][0]["Ebs"].update(
            Encrypted=False
        ),
        lambda value: value["IamInstanceProfile"].update(Arn="wrong"),
        lambda value: value.update(MinCount=2),
    ],
)
def test_p5_validator_rejects_open_or_mismatched_request(tmp_path, mutation):
    value = _request(p6=False, count=1)
    mutation(value)
    request = _write(tmp_path / "launch-request.json", value)

    with pytest.raises(LaunchRequestError):
        _validate(request, p6=False)


def test_capacity_block_market_and_reservation_are_both_mandatory(tmp_path):
    for field in (
        "InstanceMarketOptions",
        "CapacityReservationSpecification",
    ):
        value = _request(p6=True)
        del value[field]
        request = _write(tmp_path / f"missing-{field}.json", value)
        with pytest.raises(LaunchRequestError):
            _validate(request, p6=True)

    value = _request(p6=True)
    value["CapacityReservationSpecification"][
        "CapacityReservationTarget"
    ]["CapacityReservationId"] = "cr-fffffffffffffffff"
    request = _write(tmp_path / "wrong-reservation.json", value)
    with pytest.raises(LaunchRequestError, match="purchased"):
        _validate(request, p6=True)


def test_validator_rejects_duplicate_json_fields(tmp_path):
    request = tmp_path / "duplicate.json"
    request.write_text(
        '{"ImageId":"%s","ImageId":"%s"}\n' % (AMI_ID, AMI_ID),
        encoding="ascii",
    )

    with pytest.raises(LaunchRequestError, match="repeats"):
        _validate(request, p6=False)


def test_operator_response_parsers_match_aws_cli_shapes(tmp_path):
    request = _write(
        tmp_path / "launch-request.json",
        _request(p6=False, count=2),
    )
    request_sha256 = hashlib.sha256(request.read_bytes()).hexdigest()
    response = _write(
        tmp_path / "run-instances.json",
        {
            "ReservationId": "r-0123456789abcdef0",
            "Instances": [
                {"InstanceId": "i-00000000000000002"},
                {"InstanceId": "i-00000000000000001"},
            ],
        },
    )
    assert extract_run_instance_ids(
        response,
        request_path=request,
        expected_request_sha256=request_sha256,
    ) == (
        "i-00000000000000001",
        "i-00000000000000002",
    )
    with pytest.raises(LaunchRequestError, match="validated request"):
        extract_run_instance_ids(
            response,
            request_path=request,
            expected_request_sha256="0" * 64,
        )

    purchase = _write(
        tmp_path / "purchase.json",
        {
            "CapacityReservation": {
                "CapacityReservationId": CAPACITY_RESERVATION_ID,
            }
        },
    )
    assert extract_capacity_reservation_id(purchase) == CAPACITY_RESERVATION_ID

    describe_shape = _write(
        tmp_path / "describe-instances.json",
        {
            "Reservations": [
                {
                    "Instances": [
                        {"InstanceId": "i-00000000000000001"},
                        {"InstanceId": "i-00000000000000002"},
                    ]
                }
            ]
        },
    )
    with pytest.raises(LaunchRequestError, match="response count"):
        extract_run_instance_ids(
            describe_shape,
            request_path=request,
            expected_request_sha256=request_sha256,
        )


def test_validator_rejects_legacy_aws_profile(tmp_path):
    request = _write(tmp_path / "launch-request.json", _request(p6=False))

    with pytest.raises(LaunchRequestError, match="v3 profile"):
        validate_launch_request(
            request,
            profile_path=LEGACY_P5,
            region="us-east-1",
            ami_id=AMI_ID,
            instance_profile_arn=INSTANCE_PROFILE_ARN,
            subnet_id=SUBNET_ID,
            security_group_ids=SECURITY_GROUP_IDS,
            ebs_kms_key_id=KMS_KEY_ID,
            root_device_name="/dev/sda1",
            root_volume_gib=500,
            cohort_id=COHORT_ID,
        )


def test_cli_emits_one_machine_readable_validation_report(tmp_path, capsys):
    request = _write(tmp_path / "launch-request.json", _request(p6=False))

    assert main(_cli_arguments(request)) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "ami_id",
        "capacity_reservation_id",
        "count",
        "ebs_kms_key_id",
        "instance_profile_arn",
        "instance_type",
        "ok",
        "profile_id",
        "purchase_model",
        "region",
        "request_sha256",
        "schema_version",
        "security_group_ids",
        "subnet_id",
    }


def test_documented_script_path_invocation_loads_repository_modules(tmp_path):
    request = _write(tmp_path / "launch-request.json", _request(p6=False))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_aws_gpu_launch_request.py",
            *_cli_arguments(request),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["request_sha256"] == hashlib.sha256(
        request.read_bytes()
    ).hexdigest()
