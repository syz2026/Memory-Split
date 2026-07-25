from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT
    / "infra"
    / "aws"
    / "cloudformation"
    / "memorysplit-corpus-builder-foundation.yaml"
)
GUARD_PATH = (
    REPO_ROOT
    / "infra"
    / "aws"
    / "cfn-guard"
    / "memorysplit-corpus-builder.guard"
)
DEV_REQUIREMENTS_PATH = REPO_ROOT / "infra" / "aws" / "requirements-dev.txt"
PROFILE_PATH = (
    REPO_ROOT
    / "cluster"
    / "profiles"
    / "aws-i4i.16xlarge-corpus-v1.json"
)

PAID_CAPACITY_TYPES = {
    "AWS::AutoScaling::AutoScalingGroup",
    "AWS::EC2::CapacityReservation",
    "AWS::EC2::CapacityReservationFleet",
    "AWS::EC2::EC2Fleet",
    "AWS::EC2::Instance",
    "AWS::EC2::SpotFleet",
}
PUBLIC_NETWORK_TYPES = {
    "AWS::EC2::EIP",
    "AWS::EC2::InternetGateway",
    "AWS::EC2::NatGateway",
    "AWS::EC2::VPCGatewayAttachment",
}
FORBIDDEN_BUILDER_RESOURCES = {
    "AWS::EC2::KeyPair",
    "AWS::ECR::Repository",
}
REQUIRED_INTERFACE_ENDPOINTS = {
    "ec2",
    "ec2messages",
    "kms",
    "logs",
    "ssm",
    "ssmmessages",
    "sts",
}
EXPECTED_RESOURCE_TYPES = {
    "ArtifactBucket": "AWS::S3::Bucket",
    "ArtifactBucketPolicy": "AWS::S3::BucketPolicy",
    "BuilderCommandDocument": "AWS::SSM::Document",
    "BuilderInstanceProfile": "AWS::IAM::InstanceProfile",
    "BuilderLaunchTemplate": "AWS::EC2::LaunchTemplate",
    "BuilderLogGroup": "AWS::Logs::LogGroup",
    "BuilderRole": "AWS::IAM::Role",
    "BuilderSecurityGroup": "AWS::EC2::SecurityGroup",
    "BuilderToEndpointHttpsIngress": "AWS::EC2::SecurityGroupIngress",
    "ControllerRole": "AWS::IAM::Role",
    "DataKey": "AWS::KMS::Key",
    "Ec2Endpoint": "AWS::EC2::VPCEndpoint",
    "Ec2MessagesEndpoint": "AWS::EC2::VPCEndpoint",
    "EndpointSecurityGroup": "AWS::EC2::SecurityGroup",
    "KmsEndpoint": "AWS::EC2::VPCEndpoint",
    "LogsEndpoint": "AWS::EC2::VPCEndpoint",
    "MonthlyBudget": "AWS::Budgets::Budget",
    "PrivateRouteTable": "AWS::EC2::RouteTable",
    "PrivateSubnet": "AWS::EC2::Subnet",
    "PrivateSubnetRouteTableAssociation": (
        "AWS::EC2::SubnetRouteTableAssociation"
    ),
    "S3Endpoint": "AWS::EC2::VPCEndpoint",
    "SsmEndpoint": "AWS::EC2::VPCEndpoint",
    "SsmMessagesEndpoint": "AWS::EC2::VPCEndpoint",
    "StsEndpoint": "AWS::EC2::VPCEndpoint",
    "Vpc": "AWS::EC2::VPC",
}
EXPECTED_OUTPUTS = {
    "ArtifactBucketName",
    "BuilderInstanceProfileArn",
    "BuilderRoleArn",
    "ControllerRoleArn",
    "DataKeyArn",
    "LaunchTemplateId",
    "LaunchTemplateVersion",
    "PrivateSubnetId",
    "SecurityGroupId",
    "SsmDocumentName",
    "VpcId",
}
EXPECTED_BUILDER_ACTIONS = {
    "kms:Decrypt",
    "kms:DescribeKey",
    "kms:Encrypt",
    "kms:GenerateDataKey",
    "logs:CreateLogStream",
    "logs:PutLogEvents",
    "s3:AbortMultipartUpload",
    "s3:GetBucketLocation",
    "s3:GetObject",
    "s3:GetObjectAttributes",
    "s3:GetObjectVersion",
    "s3:ListBucket",
    "s3:ListBucketVersions",
    "s3:ListMultipartUploadParts",
    "s3:PutObject",
}
EXPECTED_CONTROLLER_ACTIONS = {
    "ec2:CreateTags",
    "ec2:DescribeImages",
    "ec2:DescribeInstanceAttribute",
    "ec2:DescribeInstances",
    "ec2:DescribeInstanceTypeOfferings",
    "ec2:DescribeLaunchTemplateVersions",
    "ec2:RunInstances",
    "ec2:TerminateInstances",
    "iam:PassRole",
    "kms:CreateGrant",
    "kms:Decrypt",
    "kms:DescribeKey",
    "kms:GenerateDataKeyWithoutPlaintext",
    "kms:ReEncryptFrom",
    "kms:ReEncryptTo",
    "s3:BypassGovernanceRetention",
    "s3:DeleteObjectVersion",
    "s3:ListBucketVersions",
    "ssm:DescribeInstanceInformation",
    "ssm:GetCommandInvocation",
    "ssm:ListCommands",
    "ssm:SendCommand",
    "sts:GetCallerIdentity",
}
EXPECTED_BUILDER_POLICY_SIDS = {
    "DescribeDataKey",
    "ListCorpusNamespaces",
    "LocateCorpusBucket",
    "ReadBuilderInputs",
    "ReadWriteBuildArtifacts",
    "UseDataKey",
    "WriteBuilderLogs",
}
EXPECTED_CONTROLLER_POLICY_SIDS = {
    "CleanupFailedBuildVersions",
    "CreateEbsGrant",
    "DescribeDataKeyForEbs",
    "IdentifyControllerSession",
    "InspectEc2State",
    "InspectSsmState",
    "ListFailedBuildVersions",
    "PassOnlyBuilderRoleToEc2",
    "RunApprovedLaunchTemplate",
    "SendBuilderDocument",
    "SendToTaggedBuilder",
    "TagApprovedLaunchResources",
    "TerminateTaggedBuilder",
    "UseDataKeyForEbs",
}
EXPECTED_ENDPOINT_POLICY_SIDS = {
    "CorpusBucketListing",
    "CorpusBucketLocation",
    "ReadBuilderInputs",
    "ReadWriteBuildArtifacts",
}
EXPECTED_BUCKET_POLICY_SIDS = {
    "DenyInsecureTransport",
    "DenyUnexpectedEncryptionAlgorithm",
    "DenyUnexpectedKmsKey",
}
EXPECTED_SSM_COMMAND = (
    "/usr/local/bin/memorysplit-corpus-builder "
    "--build-id '{{ BuildId }}' "
    "--package-uri '{{ PackageUri }}' "
    "--package-sha256 '{{ PackageSHA256 }}' "
    "--source-manifest-uri '{{ SourceManifestUri }}' "
    "--source-manifest-sha256 '{{ SourceManifestSHA256 }}'"
)
EXPECTED_SSM_PARAMETERS = {
    "BuildId": {
        "allowedPattern": "^[0-9a-f]{64}$",
        "type": "String",
    },
    "PackageSHA256": {
        "allowedPattern": "^[0-9a-f]{64}$",
        "type": "String",
    },
    "PackageUri": {
        "allowedPattern": (
            "^s3://memorysplit-corpus-056956104102-us-east-1/"
            "v2/packages/[A-Za-z0-9._/-]+$"
        ),
        "type": "String",
    },
    "SourceManifestSHA256": {
        "allowedPattern": "^[0-9a-f]{64}$",
        "type": "String",
    },
    "SourceManifestUri": {
        "allowedPattern": (
            "^s3://memorysplit-corpus-056956104102-us-east-1/"
            "v2/sources/[A-Za-z0-9._/-]+$"
        ),
        "type": "String",
    },
}
USER_DATA_PARAMETER_NAMES = tuple(
    f"BuilderUserDataGzipBase64Part{index}" for index in range(1, 7)
)
USER_DATA_PARAMETER_MAX_LENGTHS = (4096, 4096, 4096, 4096, 4096, 1368)


def test_foundation_files_exist_before_invariants_run() -> None:
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in (TEMPLATE_PATH, GUARD_PATH, DEV_REQUIREMENTS_PATH)
        if not path.is_file()
    ]
    assert not missing, f"Task 5 foundation files are missing: {missing}"


@pytest.fixture(scope="module")
def template() -> dict[str, object]:
    if not TEMPLATE_PATH.is_file():
        pytest.skip("Task 5 foundation template is not implemented yet")
    value = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert isinstance(value.get("Resources"), dict)
    return value


def _resources(
    template: dict[str, object], resource_type: str
) -> dict[str, dict[str, object]]:
    resources = template["Resources"]
    assert isinstance(resources, dict)
    return {
        name: resource
        for name, resource in resources.items()
        if isinstance(name, str)
        and isinstance(resource, dict)
        and resource.get("Type") == resource_type
    }


def _walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _action_set(statement: dict[str, object]) -> set[str]:
    actions = statement["Action"]
    if isinstance(actions, str):
        return {actions}
    assert isinstance(actions, list)
    assert all(isinstance(action, str) for action in actions)
    return set(actions)


def _role_statements(
    template: dict[str, object], role_name: str
) -> dict[str, dict[str, object]]:
    resources = template["Resources"]
    assert isinstance(resources, dict)
    role = resources[role_name]
    policies = role["Properties"]["Policies"]
    statements = policies[0]["PolicyDocument"]["Statement"]
    return {statement["Sid"]: statement for statement in statements}


def _role_actions(template: dict[str, object], role_name: str) -> set[str]:
    return {
        action
        for statement in _role_statements(template, role_name).values()
        for action in _action_set(statement)
    }


def _sub_resources(statement: dict[str, object]) -> list[str]:
    resources = statement["Resource"]
    rows = resources if isinstance(resources, list) else [resources]
    return [row["Fn::Sub"] for row in rows if "Fn::Sub" in row]


def _list_prefixes(statement: dict[str, object]) -> list[str]:
    values = statement["Condition"]["ForAnyValue:StringLike"]["s3:prefix"]
    return [
        value["Fn::Sub"] if isinstance(value, dict) else value
        for value in values
    ]


def _guard_count_name(scope: str, sid: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", sid).lower()
    return f"{scope}_{words}_count"


def _statement_list(
    template: dict[str, object],
    resource_name: str,
) -> list[dict[str, object]]:
    resource = template["Resources"][resource_name]
    if resource["Type"] == "AWS::IAM::Role":
        return resource["Properties"]["Policies"][0]["PolicyDocument"][
            "Statement"
        ]
    return resource["Properties"]["PolicyDocument"]["Statement"]


def _assert_guard_selector_model(template: dict[str, object]) -> None:
    resources = template["Resources"]
    launches = [
        resource
        for resource in resources.values()
        if resource["Type"] == "AWS::EC2::LaunchTemplate"
    ]
    assert len(launches) == 1
    data = launches[0]["Properties"]["LaunchTemplateData"]
    assert data["UserData"] == {
        "Fn::Join": [
            "",
            [{"Ref": name} for name in USER_DATA_PARAMETER_NAMES],
        ]
    }

    def tag_count(resource_type: str) -> int:
        return sum(
            1
            for specification in data["TagSpecifications"]
            if specification["ResourceType"] == resource_type
            for tag in specification["Tags"]
            if tag["Key"] == "MemorySplitCorpusBuilder"
            and tag["Value"] == "true"
        )

    assert tag_count("instance") == 1
    assert tag_count("volume") == 1

    expected_policy_sids = {
        "BuilderRole": EXPECTED_BUILDER_POLICY_SIDS,
        "ControllerRole": EXPECTED_CONTROLLER_POLICY_SIDS,
        "S3Endpoint": EXPECTED_ENDPOINT_POLICY_SIDS,
        "ArtifactBucketPolicy": EXPECTED_BUCKET_POLICY_SIDS,
    }
    for resource_name, expected_sids in expected_policy_sids.items():
        statements = _statement_list(template, resource_name)
        actual_sids = [statement["Sid"] for statement in statements]
        assert len(actual_sids) == len(expected_sids)
        assert len(actual_sids) == len(set(actual_sids))
        assert set(actual_sids) == expected_sids

    assert [
        statement["Sid"]
        for statement in resources["BuilderRole"]["Properties"][
            "AssumeRolePolicyDocument"
        ]["Statement"]
    ] == ["Ec2AssumeRole"]
    assert [
        statement["Sid"]
        for statement in resources["ControllerRole"]["Properties"][
            "AssumeRolePolicyDocument"
        ]["Statement"]
    ] == ["AccountAssumeRole"]

    content = resources["BuilderCommandDocument"]["Properties"]["Content"]
    assert content["parameters"] == EXPECTED_SSM_PARAMETERS
    assert len(content["mainSteps"]) == 1
    step = content["mainSteps"][0]
    assert step["action"] == "aws:runShellScript"
    assert step["name"] == "runCorpusBuilder"
    assert step["inputs"] == {
        "runCommand": [EXPECTED_SSM_COMMAND],
        "timeoutSeconds": "84600",
    }


def _cfn_guard_binary() -> str | None:
    return os.environ.get("CFN_GUARD") or shutil.which("cfn-guard")


def _run_cfn_guard(data_path: Path) -> subprocess.CompletedProcess[str] | None:
    binary = _cfn_guard_binary()
    if binary is None:
        return None
    return subprocess.run(
        [
            binary,
            "validate",
            "--rules",
            str(GUARD_PATH),
            "--data",
            str(data_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_template_is_plain_yaml_using_only_long_form_intrinsics(template):
    text = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert template["AWSTemplateFormatVersion"] == "2010-09-09"
    assert isinstance(template.get("Description"), str)
    assert re.search(
        r"(?m)!\s*(?:And|Equals|FindInMap|GetAtt|GetAZs|If|ImportValue|"
        r"Join|Not|Or|Ref|Select|Split|Sub)\b",
        text,
    ) is None
    for node in _walk(template):
        if isinstance(node, dict):
            assert None not in node, "YAML 1.1 coerced a mapping key to null"


def test_template_does_not_wrap_static_strings_in_fn_sub(template):
    for node in _walk(template):
        if not isinstance(node, dict) or "Fn::Sub" not in node:
            continue
        substitution = node["Fn::Sub"]
        if isinstance(substitution, str):
            assert "${" in substitution


def test_parameters_pin_account_region_and_operator_selected_inputs(template):
    assert set(template["Parameters"]) == {
        "BudgetAlertEmail",
        "BuilderAmiId",
        "BuilderAvailabilityZone",
        *USER_DATA_PARAMETER_NAMES,
        "PrivateSubnetCidr",
        "S3GatewayPrefixListId",
        "VpcCidr",
    }
    assert template["Parameters"]["BuilderAmiId"]["Type"] == (
        "AWS::EC2::Image::Id"
    )
    assert template["Parameters"]["BuilderAvailabilityZone"]["Type"] == (
        "AWS::EC2::AvailabilityZone::Name"
    )
    for name in ("VpcCidr", "PrivateSubnetCidr"):
        parameter = template["Parameters"][name]
        assert re.fullmatch(parameter["AllowedPattern"], parameter["Default"])
    prefix_list = template["Parameters"]["S3GatewayPrefixListId"]
    assert prefix_list["Type"] == "String"
    assert prefix_list["AllowedPattern"] == "^pl-[0-9a-f]+$"
    assert "com.amazonaws.us-east-1.s3" in prefix_list["Description"]

    assertions = template["Rules"]["ApprovedAwsDeployment"]["Assertions"]
    assert assertions == [
        {
            "Assert": {
                "Fn::Equals": [
                    {"Ref": "AWS::AccountId"},
                    "056956104102",
                ]
            },
            "AssertDescription": (
                "This foundation is approved only for AWS account 056956104102."
            ),
        },
        {
            "Assert": {
                "Fn::Equals": [
                    {"Ref": "AWS::Region"},
                    "us-east-1",
                ]
            },
            "AssertDescription": (
                "This foundation is approved only for us-east-1."
            ),
        },
    ]


def test_foundation_has_launch_template_but_no_paid_instance_resource(template):
    resources = template["Resources"]
    resource_types = {
        resource["Type"] for resource in resources.values()
    }

    assert resource_types.isdisjoint(PAID_CAPACITY_TYPES)
    launch = resources["BuilderLaunchTemplate"]
    assert launch["Type"] == "AWS::EC2::LaunchTemplate"
    assert launch["Properties"]["LaunchTemplateData"]["InstanceType"] == (
        "i4i.16xlarge"
    )
    assert "KeyName" not in launch["Properties"]["LaunchTemplateData"]


def test_resource_inventory_is_exactly_builder_only(template):
    resources = template["Resources"]
    assert {
        name: resource["Type"] for name, resource in resources.items()
    } == EXPECTED_RESOURCE_TYPES
    assert {
        resource["Type"] for resource in resources.values()
    }.isdisjoint(
        PAID_CAPACITY_TYPES
        | PUBLIC_NETWORK_TYPES
        | FORBIDDEN_BUILDER_RESOURCES
    )
    assert not _resources(template, "AWS::EC2::Route")
    assert not _resources(template, "AWS::IAM::ManagedPolicy")
    assert not _resources(template, "AWS::IAM::Policy")


def test_network_is_single_az_private_endpoint_only_and_zero_ingress(template):
    resources = template["Resources"]
    subnet = resources["PrivateSubnet"]["Properties"]
    assert subnet["AvailabilityZone"] == {"Ref": "BuilderAvailabilityZone"}
    assert subnet["MapPublicIpOnLaunch"] is False
    assert "Ipv6CidrBlock" not in subnet

    builder_group = resources["BuilderSecurityGroup"]["Properties"]
    assert builder_group["SecurityGroupIngress"] == []
    assert builder_group["SecurityGroupEgress"] == [
        {
            "Description": "HTTPS to the interface VPC endpoints",
            "DestinationSecurityGroupId": {
                "Fn::GetAtt": ["EndpointSecurityGroup", "GroupId"]
            },
            "FromPort": 443,
            "IpProtocol": "tcp",
            "ToPort": 443,
        },
        {
            "Description": "HTTPS to the us-east-1 S3 gateway endpoint",
            "DestinationPrefixListId": {"Ref": "S3GatewayPrefixListId"},
            "FromPort": 443,
            "IpProtocol": "tcp",
            "ToPort": 443,
        },
    ]
    endpoint_group = resources["EndpointSecurityGroup"]["Properties"]
    assert endpoint_group["SecurityGroupIngress"] == []
    assert endpoint_group["SecurityGroupEgress"] == [
        {
            "CidrIp": "127.0.0.1/32",
            "Description": "No-op rule suppresses default allow-all egress",
            "IpProtocol": "-1",
        }
    ]
    assert resources["BuilderToEndpointHttpsIngress"] == {
        "Type": "AWS::EC2::SecurityGroupIngress",
        "Properties": {
            "Description": "HTTPS from the corpus builder security group",
            "FromPort": 443,
            "GroupId": {
                "Fn::GetAtt": ["EndpointSecurityGroup", "GroupId"]
            },
            "IpProtocol": "tcp",
            "SourceSecurityGroupId": {
                "Fn::GetAtt": ["BuilderSecurityGroup", "GroupId"]
            },
            "ToPort": 443,
        },
    }

    for resource in resources.values():
        resource_type = resource["Type"]
        properties = resource["Properties"]
        if resource_type == "AWS::EC2::SecurityGroup":
            rules = [
                *properties["SecurityGroupIngress"],
                *properties["SecurityGroupEgress"],
            ]
        elif resource_type in {
            "AWS::EC2::SecurityGroupEgress",
            "AWS::EC2::SecurityGroupIngress",
        }:
            rules = [properties]
        else:
            continue
        for rule in rules:
            assert isinstance(rule.get("Description"), str)
            assert rule["Description"]

    endpoints = _resources(template, "AWS::EC2::VPCEndpoint")
    interface_services: set[str] = set()
    gateway_services: set[str] = set()
    for endpoint in endpoints.values():
        properties = endpoint["Properties"]
        rendered = properties["ServiceName"]["Fn::Sub"]
        prefix = "com.amazonaws.${AWS::Region}."
        assert rendered.startswith(prefix)
        service = rendered.removeprefix(prefix)
        if properties["VpcEndpointType"] == "Interface":
            interface_services.add(service)
            assert properties["PrivateDnsEnabled"] is True
            assert properties["SubnetIds"] == [{"Ref": "PrivateSubnet"}]
            assert properties["SecurityGroupIds"] == [
                {"Fn::GetAtt": ["EndpointSecurityGroup", "GroupId"]}
            ]
        else:
            assert properties["VpcEndpointType"] == "Gateway"
            gateway_services.add(service)
            assert properties["RouteTableIds"] == [
                {"Ref": "PrivateRouteTable"}
            ]
    assert interface_services == REQUIRED_INTERFACE_ENDPOINTS
    assert gateway_services == {"s3"}

    for node in _walk(template):
        if not isinstance(node, dict):
            continue
        assert node.get("AssociatePublicIpAddress") is not True
        assert node.get("MapPublicIpOnLaunch") is not True
        assert node.get("CidrIp") != "0.0.0.0/0"
        assert node.get("CidrIpv6") != "::/0"
        if isinstance(node.get("FromPort"), int) and isinstance(
            node.get("ToPort"), int
        ):
            assert not node["FromPort"] <= 22 <= node["ToPort"]


def test_launch_template_is_private_imdsv2_encrypted_and_terminating(template):
    data = template["Resources"]["BuilderLaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]
    assert data["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert data["DisableApiTermination"] is False
    assert data["EbsOptimized"] is True
    assert data["Monitoring"] == {"Enabled": True}
    assert data["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpProtocolIpv6": "disabled",
        "HttpPutResponseHopLimit": 1,
        "HttpTokens": "required",
        "InstanceMetadataTags": "disabled",
    }
    assert data["NetworkInterfaces"] == [
        {
            "AssociatePublicIpAddress": False,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [
                {"Fn::GetAtt": ["BuilderSecurityGroup", "GroupId"]}
            ],
            "SubnetId": {"Ref": "PrivateSubnet"},
        }
    ]
    assert data["BlockDeviceMappings"] == [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "DeleteOnTermination": True,
                "Encrypted": True,
                "KmsKeyId": {"Fn::GetAtt": ["DataKey", "Arn"]},
                "VolumeSize": 200,
                "VolumeType": "gp3",
            },
        }
    ]
    for resource_type in ("instance", "volume"):
        specification = next(
            item
            for item in data["TagSpecifications"]
            if item["ResourceType"] == resource_type
        )
        assert {
            "Key": "MemorySplitCorpusBuilder",
            "Value": "true",
        } in specification["Tags"]


def test_launch_template_joins_task6_gzip_base64_chunks_without_reencoding(
    template,
):
    parameters = template["Parameters"]
    assert parameters[USER_DATA_PARAMETER_NAMES[0]] == {
        "Type": "String",
        "AllowedPattern": "^H4sI[A-Za-z0-9+/]*={0,2}$",
        "ConstraintDescription": "Must start one deterministic gzip base64 payload.",
        "Description": (
            "First ordered chunk of the Task 6 deterministic gzip/base64 "
            "user data; split at 4096-character boundaries."
        ),
        "MaxLength": 4096,
        "MinLength": 8,
        "NoEcho": True,
    }
    for name, max_length in zip(
        USER_DATA_PARAMETER_NAMES[1:],
        USER_DATA_PARAMETER_MAX_LENGTHS[1:],
        strict=True,
    ):
        assert parameters[name] == {
            "Type": "String",
            "AllowedPattern": "^[A-Za-z0-9+/]*={0,2}$",
            "ConstraintDescription": (
                "Must continue the deterministic gzip base64 payload."
            ),
            "Default": "",
            "Description": (
                "Next ordered chunk of the Task 6 deterministic gzip/base64 "
                "user data; leave empty after the final chunk."
            ),
            "MaxLength": max_length,
            "NoEcho": True,
        }
    assert sum(USER_DATA_PARAMETER_MAX_LENGTHS) == 21848
    assert all(length <= 4096 for length in USER_DATA_PARAMETER_MAX_LENGTHS)
    data = template["Resources"]["BuilderLaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]
    assert data["UserData"] == {
        "Fn::Join": [
            "",
            [{"Ref": name} for name in USER_DATA_PARAMETER_NAMES],
        ]
    }
    assert "Fn::Base64" not in data["UserData"]
    assert data["InstanceInitiatedShutdownBehavior"] == "terminate"


def test_launch_template_matches_frozen_builder_profile(template):
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    data = template["Resources"]["BuilderLaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]
    assert profile["profile_id"] == "aws-i4i.16xlarge-corpus-v1"
    assert profile["region"] == "us-east-1"
    assert data["InstanceType"] == profile["instance_type"]
    assert data["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == profile[
        "root_volume_gib"
    ]
    assert data["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert profile["max_runtime_seconds"] == 24 * 60 * 60
    assert profile["watchdog_shutdown_seconds"] == 23 * 60 * 60 + 30 * 60
    assert profile["watchdog_shutdown_seconds"] < profile["max_runtime_seconds"]
    assert profile["max_hourly_usd"] == "5.491"
    assert profile["max_compute_usd"] == "131.78"
    timeout = template["Resources"]["BuilderCommandDocument"]["Properties"][
        "Content"
    ]["mainSteps"][0]["inputs"]["timeoutSeconds"]
    assert int(timeout) == profile["watchdog_shutdown_seconds"]


def test_retained_state_is_named_versioned_and_kms_encrypted(template):
    resources = template["Resources"]
    data_key = resources["DataKey"]
    assert data_key["DeletionPolicy"] == "Retain"
    assert data_key["UpdateReplacePolicy"] == "Retain"
    assert data_key["Properties"]["EnableKeyRotation"] is True
    assert data_key["Properties"]["KeySpec"] == "SYMMETRIC_DEFAULT"
    assert data_key["Properties"]["KeyUsage"] == "ENCRYPT_DECRYPT"

    bucket = resources["ArtifactBucket"]
    assert bucket["DeletionPolicy"] == "Retain"
    assert bucket["UpdateReplacePolicy"] == "Retain"
    assert bucket["Properties"]["BucketName"] == (
        "memorysplit-corpus-056956104102-us-east-1"
    )
    assert bucket["Properties"]["VersioningConfiguration"] == {
        "Status": "Enabled"
    }
    assert bucket["Properties"]["OwnershipControls"] == {
        "Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]
    }
    assert bucket["Properties"]["PublicAccessBlockConfiguration"] == {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }
    assert bucket["Properties"]["BucketEncryption"] == {
        "ServerSideEncryptionConfiguration": [
            {
                "BucketKeyEnabled": True,
                "ServerSideEncryptionByDefault": {
                    "KMSMasterKeyID": {
                        "Fn::GetAtt": ["DataKey", "Arn"]
                    },
                    "SSEAlgorithm": "aws:kms",
                },
            }
        ]
    }

    log_group = resources["BuilderLogGroup"]
    assert log_group["DeletionPolicy"] == "Retain"
    assert log_group["UpdateReplacePolicy"] == "Retain"
    assert log_group["Properties"]["KmsKeyId"] == {
        "Fn::GetAtt": ["DataKey", "Arn"]
    }
    assert log_group["Properties"]["RetentionInDays"] == 365


def test_claim_bearing_bucket_has_365_day_governance_object_lock(template):
    properties = template["Resources"]["ArtifactBucket"]["Properties"]
    assert properties["ObjectLockEnabled"] is True
    assert properties["ObjectLockConfiguration"] == {
        "ObjectLockEnabled": "Enabled",
        "Rule": {
            "DefaultRetention": {
                "Days": 365,
                "Mode": "GOVERNANCE",
            }
        },
    }


def test_bucket_policy_denies_every_s3_action_over_plaintext_http(template):
    statements = {
        statement["Sid"]: statement
        for statement in template["Resources"]["ArtifactBucketPolicy"][
            "Properties"
        ]["PolicyDocument"]["Statement"]
    }
    assert statements["DenyInsecureTransport"] == {
        "Sid": "DenyInsecureTransport",
        "Effect": "Deny",
        "Principal": "*",
        "Action": "s3:*",
        "Resource": [
            {"Fn::GetAtt": ["ArtifactBucket", "Arn"]},
            {"Fn::Sub": "${ArtifactBucket.Arn}/*"},
        ],
        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
    }


def test_bucket_policy_rejects_wrong_encryption_on_every_corpus_prefix(template):
    statements = {
        statement["Sid"]: statement
        for statement in template["Resources"]["ArtifactBucketPolicy"][
            "Properties"
        ]["PolicyDocument"]["Statement"]
    }
    expected_resources = [
        {"Fn::Sub": "${ArtifactBucket.Arn}/v2/builds/*"},
        {"Fn::Sub": "${ArtifactBucket.Arn}/v2/packages/*"},
        {"Fn::Sub": "${ArtifactBucket.Arn}/v2/sources/*"},
    ]
    for sid in ("DenyUnexpectedEncryptionAlgorithm", "DenyUnexpectedKmsKey"):
        assert statements[sid]["Resource"] == expected_resources


def test_builder_role_has_only_approved_actions_and_ssm_managed_policy(template):
    resources = template["Resources"]
    roles = _resources(template, "AWS::IAM::Role")
    assert set(roles) == {"BuilderRole", "ControllerRole"}
    profiles = _resources(template, "AWS::IAM::InstanceProfile")
    assert set(profiles) == {"BuilderInstanceProfile"}
    assert profiles["BuilderInstanceProfile"]["Properties"] == {
        "Roles": [{"Ref": "BuilderRole"}]
    }

    properties = resources["BuilderRole"]["Properties"]
    assert "RoleName" not in properties
    assert "PermissionsBoundary" not in properties
    assert properties["ManagedPolicyArns"] == [
        {
            "Fn::Sub": (
                "arn:${AWS::Partition}:iam::aws:policy/"
                "AmazonSSMManagedInstanceCore"
            )
        }
    ]
    assert [policy["PolicyName"] for policy in properties["Policies"]] == [
        "BuilderCorpusData"
    ]
    assert _role_actions(template, "BuilderRole") == EXPECTED_BUILDER_ACTIONS
    trust = properties["AssumeRolePolicyDocument"]["Statement"]
    assert trust == [
        {
            "Sid": "Ec2AssumeRole",
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ]


def test_builder_s3_permissions_use_exact_read_and_write_prefixes(template):
    statements = _role_statements(template, "BuilderRole")
    assert _list_prefixes(statements["ListCorpusNamespaces"]) == [
        "v2/builds",
        "v2/builds/*",
        "v2/packages",
        "v2/packages/*",
        "v2/sources",
        "v2/sources/*",
    ]
    assert _sub_resources(statements["ReadBuilderInputs"]) == [
        "${ArtifactBucket.Arn}/v2/packages/*",
        "${ArtifactBucket.Arn}/v2/sources/*",
    ]
    assert _sub_resources(statements["ReadWriteBuildArtifacts"]) == [
        "${ArtifactBucket.Arn}/v2/builds/*"
    ]
    assert _action_set(statements["ReadBuilderInputs"]) == {
        "s3:GetObject",
        "s3:GetObjectAttributes",
        "s3:GetObjectVersion",
    }
    assert _action_set(statements["ReadWriteBuildArtifacts"]) == {
        "s3:AbortMultipartUpload",
        "s3:GetObject",
        "s3:GetObjectAttributes",
        "s3:GetObjectVersion",
        "s3:ListMultipartUploadParts",
        "s3:PutObject",
    }

    for statement in statements.values():
        assert "NotAction" not in statement
        assert "NotResource" not in statement
        assert all("*" not in action for action in _action_set(statement))
        if any(action.startswith("s3:") for action in _action_set(statement)):
            assert statement["Resource"] != "*"
            assert "${ArtifactBucket.Arn}/*" not in _sub_resources(statement)


def test_kms_permissions_require_the_approved_aws_service_and_context(template):
    builder = _role_statements(template, "BuilderRole")
    assert _action_set(builder["UseDataKey"]) == {
        "kms:Decrypt",
        "kms:Encrypt",
        "kms:GenerateDataKey",
    }
    assert builder["UseDataKey"]["Condition"] == {
        "StringEquals": {
            "kms:EncryptionContext:aws:s3:arn": {
                "Fn::GetAtt": ["ArtifactBucket", "Arn"]
            },
            "kms:ViaService": {
                "Fn::Sub": "s3.${AWS::Region}.amazonaws.com"
            },
        }
    }
    assert builder["DescribeDataKey"] == {
        "Sid": "DescribeDataKey",
        "Effect": "Allow",
        "Action": "kms:DescribeKey",
        "Resource": {"Fn::GetAtt": ["DataKey", "Arn"]},
        "Condition": {
            "StringEquals": {
                "kms:ViaService": {
                    "Fn::Sub": "s3.${AWS::Region}.amazonaws.com"
                }
            }
        },
    }

    controller = _role_statements(template, "ControllerRole")
    assert _action_set(controller["UseDataKeyForEbs"]) == {
        "kms:Decrypt",
        "kms:GenerateDataKeyWithoutPlaintext",
        "kms:ReEncryptFrom",
        "kms:ReEncryptTo",
    }
    assert controller["UseDataKeyForEbs"]["Condition"] == {
        "ForAnyValue:StringEquals": {
            "kms:EncryptionContextKeys": ["aws:ebs:id"]
        },
        "StringEquals": {
            "kms:ViaService": {
                "Fn::Sub": "ec2.${AWS::Region}.amazonaws.com"
            }
        },
    }
    assert controller["DescribeDataKeyForEbs"] == {
        "Sid": "DescribeDataKeyForEbs",
        "Effect": "Allow",
        "Action": "kms:DescribeKey",
        "Resource": {"Fn::GetAtt": ["DataKey", "Arn"]},
    }
    assert controller["CreateEbsGrant"]["Condition"] == {
        "Bool": {"kms:GrantIsForAWSResource": "true"},
        "StringEquals": {
            "kms:ViaService": {
                "Fn::Sub": "ec2.${AWS::Region}.amazonaws.com"
            }
        },
    }


def test_controller_lifecycle_and_ssm_permissions_are_exactly_scoped(template):
    resources = template["Resources"]
    properties = resources["ControllerRole"]["Properties"]
    assert "RoleName" not in properties
    assert "ManagedPolicyArns" not in properties
    assert "PermissionsBoundary" not in properties
    assert [policy["PolicyName"] for policy in properties["Policies"]] == [
        "ControllerOperations"
    ]
    assert _role_actions(template, "ControllerRole") == (
        EXPECTED_CONTROLLER_ACTIONS
    )

    statements = _role_statements(template, "ControllerRole")
    run = statements["RunApprovedLaunchTemplate"]
    assert run["Action"] == "ec2:RunInstances"
    assert run["Resource"] == "*"
    assert run["Condition"] == {
        "ArnEquals": {
            "ec2:LaunchTemplate": {
                "Fn::Sub": (
                    "arn:${AWS::Partition}:ec2:${AWS::Region}:"
                    "${AWS::AccountId}:launch-template/"
                    "${BuilderLaunchTemplate}"
                )
            }
        },
        "Bool": {"ec2:IsLaunchTemplateResource": "true"},
        "StringEquals": {
            "aws:RequestTag/MemorySplitCorpusBuilder": "true",
            "ec2:InstanceMarketType": "on-demand",
            "ec2:InstanceType": "i4i.16xlarge",
            "ec2:MetadataHttpTokens": "required",
        },
    }
    instance_arn = {
        "Fn::Sub": (
            "arn:${AWS::Partition}:ec2:${AWS::Region}:"
            "${AWS::AccountId}:instance/*"
        )
    }
    assert "SetBuilderShutdownBehavior" not in statements
    assert "ec2:ModifyInstanceAttribute" not in _role_actions(
        template, "ControllerRole"
    )
    assert statements["TerminateTaggedBuilder"] == {
        "Sid": "TerminateTaggedBuilder",
        "Effect": "Allow",
        "Action": "ec2:TerminateInstances",
        "Resource": instance_arn,
        "Condition": {
            "StringEquals": {
                "ec2:ResourceTag/MemorySplitCorpusBuilder": "true"
            }
        },
    }
    assert statements["PassOnlyBuilderRoleToEc2"] == {
        "Sid": "PassOnlyBuilderRoleToEc2",
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": {"Fn::GetAtt": ["BuilderRole", "Arn"]},
        "Condition": {
            "StringEquals": {
                "iam:PassedToService": "ec2.amazonaws.com"
            }
        },
    }
    document_arn = {
        "Fn::Sub": (
            "arn:${AWS::Partition}:ssm:${AWS::Region}:"
            "${AWS::AccountId}:document/${BuilderCommandDocument}"
        )
    }
    assert statements["SendBuilderDocument"]["Resource"] == document_arn
    assert statements["SendToTaggedBuilder"]["Condition"] == {
        "StringEquals": {
            "ssm:resourceTag/MemorySplitCorpusBuilder": "true"
        }
    }
    assert statements["ListFailedBuildVersions"] == {
        "Sid": "ListFailedBuildVersions",
        "Effect": "Allow",
        "Action": "s3:ListBucketVersions",
        "Resource": {"Fn::GetAtt": ["ArtifactBucket", "Arn"]},
        "Condition": {"StringLike": {"s3:prefix": "v2/builds/*"}},
    }
    assert statements["CleanupFailedBuildVersions"] == {
        "Sid": "CleanupFailedBuildVersions",
        "Effect": "Allow",
        "Action": [
            "s3:BypassGovernanceRetention",
            "s3:DeleteObjectVersion",
        ],
        "Resource": {"Fn::Sub": "${ArtifactBucket.Arn}/v2/builds/*"},
    }

    for statement in statements.values():
        assert "NotAction" not in statement
        assert "NotResource" not in statement
        assert all("*" not in action for action in _action_set(statement))
        if _action_set(statement) & {
            "ec2:CreateTags",
            "ec2:TerminateInstances",
            "s3:BypassGovernanceRetention",
            "s3:DeleteObjectVersion",
            "s3:ListBucketVersions",
        }:
            assert statement["Resource"] != "*"


def test_ssm_document_is_closed_and_controller_targets_only_it(template):
    document = template["Resources"]["BuilderCommandDocument"]
    properties = document["Properties"]
    assert properties["Name"] == "MemorySplit-CorpusBuilderV1"
    assert properties["DocumentType"] == "Command"
    assert properties["DocumentFormat"] == "JSON"
    assert properties["TargetType"] == "/AWS::EC2::Instance"
    assert properties["UpdateMethod"] == "NewVersion"
    content = properties["Content"]
    assert content["schemaVersion"] == "2.2"
    assert content["parameters"] == EXPECTED_SSM_PARAMETERS
    assert len(content["mainSteps"]) == 1
    step = content["mainSteps"][0]
    assert step["action"] == "aws:runShellScript"
    assert step["name"] == "runCorpusBuilder"
    assert step["inputs"]["timeoutSeconds"] == "84600"
    command = "\n".join(step["inputs"]["runCommand"])
    assert command == EXPECTED_SSM_COMMAND
    for parameter in content["parameters"]:
        assert f"{{{{ {parameter} }}}}" in command


def test_budget_is_fixed_at_200_usd_with_three_email_alerts(template):
    budget = template["Resources"]["MonthlyBudget"]["Properties"]
    assert budget["Budget"]["BudgetLimit"] == {
        "Amount": 200,
        "Unit": "USD",
    }
    assert budget["Budget"]["BudgetType"] == "COST"
    assert budget["Budget"]["TimeUnit"] == "MONTHLY"
    notifications = budget["NotificationsWithSubscribers"]
    assert [
        (
            item["Notification"]["NotificationType"],
            item["Notification"]["Threshold"],
        )
        for item in notifications
    ] == [("ACTUAL", 50), ("ACTUAL", 80), ("FORECASTED", 100)]
    for item in notifications:
        assert item["Notification"]["ComparisonOperator"] == "GREATER_THAN"
        assert item["Notification"]["ThresholdType"] == "PERCENTAGE"
        assert item["Subscribers"] == [
            {
                "Address": {"Ref": "BudgetAlertEmail"},
                "SubscriptionType": "EMAIL",
            }
        ]


def test_outputs_are_closed_nonexported_identifiers(template):
    outputs = template["Outputs"]
    assert set(outputs) == EXPECTED_OUTPUTS
    for output in outputs.values():
        assert set(output) == {"Description", "Value"}
        assert isinstance(output["Description"], str) and output["Description"]
        assert "Export" not in output
    assert outputs["LaunchTemplateVersion"]["Value"] == {
        "Fn::GetAtt": ["BuilderLaunchTemplate", "LatestVersionNumber"]
    }
    assert outputs["PrivateSubnetId"]["Value"] == {"Ref": "PrivateSubnet"}
    assert outputs["SecurityGroupId"]["Value"] == {
        "Fn::GetAtt": ["BuilderSecurityGroup", "GroupId"]
    }
    assert outputs["SsmDocumentName"]["Value"] == {
        "Ref": "BuilderCommandDocument"
    }


def test_template_has_no_static_secret_like_material(template):
    del template
    data = TEMPLATE_PATH.read_bytes()
    lowered = data.lower()
    assert re.search(
        rb"-----begin (?:rsa |ec |dsa |openssh )?private key",
        lowered,
    ) is None
    assert re.search(rb"\b(?:akia|asia)[0-9a-z]{16}\b", lowered) is None
    assert re.search(
        rb"(?:access_key|secret_access_key|session_token|password|"
        rb"private_key)\s*[:=]\s*[\"']?[^\s\"']{8,}",
        lowered,
    ) is None
    assert b"{{resolve:secretsmanager:" not in lowered
    assert not any(
        re.search(r"credential|password|private.?key|secret|token", name, re.I)
        for name in template_parameter_names()
    )


def template_parameter_names() -> set[str]:
    value = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return set(value["Parameters"])


def test_cfn_guard_declares_non_vacuous_required_matches(template):
    _assert_guard_selector_model(template)
    validation = _run_cfn_guard(TEMPLATE_PATH)
    if validation is not None:
        assert validation.returncode == 0, validation.stdout + validation.stderr
    assert GUARD_PATH.is_file()
    guard = GUARD_PATH.read_text(encoding="utf-8")
    for count_name in (
        "builder_assume_role_statement_count",
        "builder_instance_tag_count",
        "builder_volume_tag_count",
        "controller_assume_role_statement_count",
        "launch_template_count",
        "ssm_parameter_count",
        "ssm_run_step_count",
        "ssm_step_count",
    ):
        assert f"%{count_name} == 1" in guard or (
            count_name == "ssm_parameter_count"
            and f"%{count_name} == 5" in guard
        )
    expected_scopes = {
        "builder": EXPECTED_BUILDER_POLICY_SIDS,
        "controller": EXPECTED_CONTROLLER_POLICY_SIDS,
        "endpoint": EXPECTED_ENDPOINT_POLICY_SIDS,
        "bucket": EXPECTED_BUCKET_POLICY_SIDS,
    }
    for scope, sids in expected_scopes.items():
        for sid in sids:
            count_name = _guard_count_name(scope, sid)
            assert re.search(
                rf"let {count_name} = count\([^\n]*Sid == '{sid}'[^\n]*\)",
                guard,
            )
            assert f"%{count_name} == 1" in guard
    assert (
        "Resources.BuilderLaunchTemplate.Properties."
        "LaunchTemplateData.UserData[ keys == 'Fn::Join' ][0] == ''"
    ) in guard


@pytest.mark.parametrize(
    "case",
    [
        "bucket-policy-sid",
        "builder-assume-role-sid",
        "builder-policy-sid",
        "controller-assume-role-sid",
        "controller-policy-sid",
        "endpoint-policy-sid",
        "instance-builder-tag",
        "launch-template-type",
        "launch-user-data",
        "ssm-command",
        "ssm-parameter",
        "ssm-step",
        "volume-builder-tag",
    ],
)
def test_guard_selector_model_rejects_missing_or_altered_targets(
    template,
    case,
    tmp_path,
):
    candidate = deepcopy(template)
    resources = candidate["Resources"]

    def statement(resource_name: str, sid: str) -> dict[str, object]:
        return next(
            item
            for item in _statement_list(candidate, resource_name)
            if item["Sid"] == sid
        )

    if case == "bucket-policy-sid":
        statement(
            "ArtifactBucketPolicy", "DenyUnexpectedKmsKey"
        )["Sid"] = "DenyUnexpectedEncryptionAlgorithm"
    elif case == "builder-assume-role-sid":
        resources["BuilderRole"]["Properties"]["AssumeRolePolicyDocument"][
            "Statement"
        ][0]["Sid"] = "WrongBuilderTrust"
    elif case == "builder-policy-sid":
        statement("BuilderRole", "UseDataKey")["Sid"] = "DescribeDataKey"
    elif case == "controller-assume-role-sid":
        resources["ControllerRole"]["Properties"]["AssumeRolePolicyDocument"][
            "Statement"
        ][0]["Sid"] = "WrongControllerTrust"
    elif case == "controller-policy-sid":
        statement("ControllerRole", "RunApprovedLaunchTemplate")[
            "Sid"
        ] = "InspectEc2State"
    elif case == "endpoint-policy-sid":
        statement("S3Endpoint", "CorpusBucketListing")[
            "Sid"
        ] = "CorpusBucketLocation"
    elif case in {"instance-builder-tag", "volume-builder-tag"}:
        target_type = case.removesuffix("-builder-tag")
        specification = next(
            item
            for item in resources["BuilderLaunchTemplate"]["Properties"][
                "LaunchTemplateData"
            ]["TagSpecifications"]
            if item["ResourceType"] == target_type
        )
        next(
            tag
            for tag in specification["Tags"]
            if tag["Key"] == "MemorySplitCorpusBuilder"
        )["Key"] = "WrongBuilderTag"
    elif case == "launch-template-type":
        resources["BuilderLaunchTemplate"]["Type"] = "AWS::EC2::Instance"
    elif case == "launch-user-data":
        del resources["BuilderLaunchTemplate"]["Properties"][
            "LaunchTemplateData"
        ]["UserData"]
    elif case == "ssm-command":
        resources["BuilderCommandDocument"]["Properties"]["Content"][
            "mainSteps"
        ][0]["inputs"]["runCommand"][0] += " --unreviewed"
    elif case == "ssm-parameter":
        del resources["BuilderCommandDocument"]["Properties"]["Content"][
            "parameters"
        ]["PackageUri"]
    elif case == "ssm-step":
        resources["BuilderCommandDocument"]["Properties"]["Content"][
            "mainSteps"
        ][0]["action"] = "aws:runPowerShellScript"
    else:
        raise AssertionError(f"unknown guard mutation: {case}")

    with pytest.raises((AssertionError, KeyError, TypeError)):
        _assert_guard_selector_model(candidate)
    mutated_path = tmp_path / f"{case}.yaml"
    mutated_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False),
        encoding="utf-8",
    )
    validation = _run_cfn_guard(mutated_path)
    if validation is not None:
        assert validation.returncode != 0, (
            f"guard accepted mutation {case}\n"
            f"{validation.stdout}{validation.stderr}"
        )


def test_iac_development_requirements_pin_cfn_lint():
    assert DEV_REQUIREMENTS_PATH.is_file()

    requirements = [
        line.strip()
        for line in DEV_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == ["cfn-lint==1.53.2"]
