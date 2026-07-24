from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from msctl.aws_argv import (
    ARGV_DOCUMENT_CONTENT,
    ARGV_DOCUMENT_NAME,
    ARGV_DOCUMENT_SHA256,
)
from msctl.jsonutil import canonical_json


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT
    / "infra"
    / "aws"
    / "cloudformation"
    / "memorysplit-p5-foundation.yaml"
)
GUARD_PATH = (
    REPO_ROOT
    / "infra"
    / "aws"
    / "cfn-guard"
    / "memorysplit-p5-foundation.guard"
)
DEV_REQUIREMENTS_PATH = REPO_ROOT / "infra" / "aws" / "requirements-dev.txt"

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
REQUIRED_INTERFACE_ENDPOINTS = {
    "ec2",
    "ec2messages",
    "ecr.api",
    "ecr.dkr",
    "kms",
    "logs",
    "ssm",
    "ssmmessages",
    "sts",
}
EXPECTED_OUTPUTS = {
    "ApprovalKeyArn",
    "ArgvDocumentName",
    "ArgvDocumentSha256",
    "ArtifactBucketName",
    "ContainerRepositoryUri",
    "ControllerRoleArn",
    "DataKeyArn",
    "EvaluatorRoleArn",
    "LaunchTemplateId",
    "LaunchTemplateVersion",
    "PrivateSubnetIds",
    "SignerRoleArn",
    "TrainInstanceProfileArn",
    "TrainRoleArn",
    "VpcId",
}


@pytest.fixture(scope="module")
def template() -> dict[str, object]:
    assert TEMPLATE_PATH.is_file(), "Task 5A foundation template is missing"
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


def _role_actions(role: dict[str, object]) -> set[str]:
    properties = role["Properties"]
    assert isinstance(properties, dict)
    policies = properties.get("Policies", [])
    assert isinstance(policies, list)
    actions: set[str] = set()
    for policy in policies:
        assert isinstance(policy, dict)
        document = policy["PolicyDocument"]
        assert isinstance(document, dict)
        statements = document["Statement"]
        assert isinstance(statements, list)
        for statement in statements:
            assert isinstance(statement, dict)
            raw_actions = statement["Action"]
            if isinstance(raw_actions, str):
                actions.add(raw_actions)
            else:
                assert isinstance(raw_actions, list)
                assert all(isinstance(action, str) for action in raw_actions)
                actions.update(raw_actions)
    return actions


def _attached_role_actions(
    template: dict[str, object], role_name: str
) -> set[str]:
    resources = template["Resources"]
    actions = _role_actions(resources[role_name])
    for resource in resources.values():
        if resource.get("Type") != "AWS::IAM::Policy":
            continue
        properties = resource["Properties"]
        if {"Ref": role_name} not in properties["Roles"]:
            continue
        for statement in properties["PolicyDocument"]["Statement"]:
            raw_actions = statement["Action"]
            if isinstance(raw_actions, str):
                actions.add(raw_actions)
            else:
                actions.update(raw_actions)
    return actions


def _attached_role_statements(
    template: dict[str, object], role_name: str
) -> dict[str, dict[str, object]]:
    resources = template["Resources"]
    statements: list[dict[str, object]] = []
    for policy in resources[role_name]["Properties"].get("Policies", []):
        statements.extend(policy["PolicyDocument"]["Statement"])
    for resource in resources.values():
        if resource.get("Type") != "AWS::IAM::Policy":
            continue
        properties = resource["Properties"]
        if {"Ref": role_name} in properties["Roles"]:
            statements.extend(properties["PolicyDocument"]["Statement"])
    return {statement["Sid"]: statement for statement in statements}


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


def test_foundation_defines_launch_template_but_no_paid_capacity(template):
    resources = template["Resources"]
    assert isinstance(resources, dict)
    resource_types = {
        resource["Type"]
        for resource in resources.values()
        if isinstance(resource, dict) and "Type" in resource
    }

    assert resource_types.isdisjoint(PAID_CAPACITY_TYPES)
    launch_templates = _resources(template, "AWS::EC2::LaunchTemplate")
    assert set(launch_templates) == {"P5LaunchTemplate"}
    launch_data = launch_templates["P5LaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]
    assert isinstance(launch_data, dict)
    assert launch_data["InstanceType"] == "p5.48xlarge"
    assert "KeyName" not in launch_data
    assert template["Outputs"]["LaunchTemplateVersion"]["Value"] == {
        "Fn::GetAtt": ["P5LaunchTemplate", "LatestVersionNumber"]
    }


def test_network_parameter_defaults_match_their_allowed_patterns(template):
    parameters = template["Parameters"]
    for name in ("VpcCidr", "PrivateSubnetACidr", "PrivateSubnetBCidr"):
        parameter = parameters[name]
        assert re.fullmatch(
            parameter["AllowedPattern"],
            parameter["Default"],
        )


def test_operator_selected_training_az_binds_the_launch_subnet(template):
    parameters = template["Parameters"]
    assert parameters["TrainingAvailabilityZone"] == {
        "Type": "AWS::EC2::AvailabilityZone::Name",
        "Description": (
            "Availability Zone where p5.48xlarge offering discovery passed."
        ),
    }
    assert parameters["SecondaryAvailabilityZone"] == {
        "Type": "AWS::EC2::AvailabilityZone::Name",
        "Description": "Distinct Availability Zone for private endpoint resilience.",
    }
    assert template["Rules"]["DistinctPrivateAvailabilityZones"] == {
        "Assertions": [
            {
                "Assert": {
                    "Fn::Not": [
                        {
                            "Fn::Equals": [
                                {"Ref": "TrainingAvailabilityZone"},
                                {"Ref": "SecondaryAvailabilityZone"},
                            ]
                        }
                    ]
                },
                "AssertDescription": (
                    "Training and secondary Availability Zones must differ."
                ),
            }
        ]
    }
    resources = template["Resources"]
    assert resources["PrivateSubnetA"]["Properties"]["AvailabilityZone"] == {
        "Ref": "TrainingAvailabilityZone"
    }
    assert resources["PrivateSubnetB"]["Properties"]["AvailabilityZone"] == {
        "Ref": "SecondaryAvailabilityZone"
    }
    network_interface = resources["P5LaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]["NetworkInterfaces"][0]
    assert network_interface["SubnetId"] == {"Ref": "PrivateSubnetA"}
    assert not any(
        isinstance(node, dict) and "Fn::GetAZs" in node
        for node in _walk(template)
    )


def test_network_is_private_endpoint_only_and_has_no_ssh(template):
    resources = template["Resources"]
    assert isinstance(resources, dict)
    resource_types = {
        resource["Type"]
        for resource in resources.values()
        if isinstance(resource, dict) and "Type" in resource
    }
    assert resource_types.isdisjoint(PUBLIC_NETWORK_TYPES)
    assert not _resources(template, "AWS::EC2::Route")

    subnets = _resources(template, "AWS::EC2::Subnet")
    assert set(subnets) == {"PrivateSubnetA", "PrivateSubnetB"}
    for subnet in subnets.values():
        properties = subnet["Properties"]
        assert isinstance(properties, dict)
        assert properties["MapPublicIpOnLaunch"] is False
        assert "Ipv6CidrBlock" not in properties

    endpoints = _resources(template, "AWS::EC2::VPCEndpoint")
    interface_services: set[str] = set()
    gateway_services: set[str] = set()
    for endpoint in endpoints.values():
        properties = endpoint["Properties"]
        assert isinstance(properties, dict)
        service_name = properties["ServiceName"]
        assert isinstance(service_name, dict)
        rendered = service_name["Fn::Sub"]
        assert isinstance(rendered, str)
        prefix = "com.amazonaws.${AWS::Region}."
        assert rendered.startswith(prefix)
        service = rendered.removeprefix(prefix)
        if properties["VpcEndpointType"] == "Interface":
            interface_services.add(service)
            assert properties["PrivateDnsEnabled"] is True
            assert properties["SubnetIds"] == [
                {"Ref": "PrivateSubnetA"},
                {"Ref": "PrivateSubnetB"},
            ]
            assert properties["SecurityGroupIds"] == [
                {"Fn::GetAtt": ["EndpointSecurityGroup", "GroupId"]}
            ]
        elif properties["VpcEndpointType"] == "Gateway":
            gateway_services.add(service)
            assert properties["RouteTableIds"] == [
                {"Ref": "PrivateRouteTableA"},
                {"Ref": "PrivateRouteTableB"},
            ]
        else:
            raise AssertionError("unsupported VPC endpoint type")
    assert interface_services == REQUIRED_INTERFACE_ENDPOINTS
    assert gateway_services == {"s3"}

    train_group = resources["TrainSecurityGroup"]
    assert train_group["Type"] == "AWS::EC2::SecurityGroup"
    assert train_group["Properties"]["SecurityGroupIngress"] == []
    endpoint_ingress = resources["EndpointSecurityGroup"]["Properties"][
        "SecurityGroupIngress"
    ]
    assert endpoint_ingress == [
        {
            "Description": "HTTPS from the zero-ingress train security group",
            "FromPort": 443,
            "IpProtocol": "tcp",
            "SourceSecurityGroupId": {
                "Fn::GetAtt": ["TrainSecurityGroup", "GroupId"]
            },
            "ToPort": 443,
        }
    ]

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


def test_launch_template_requires_imdsv2_and_encrypted_root_volume(template):
    launch_data = template["Resources"]["P5LaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]
    assert launch_data["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpProtocolIpv6": "disabled",
        "HttpPutResponseHopLimit": 1,
        "HttpTokens": "required",
        "InstanceMetadataTags": "disabled",
    }
    assert launch_data["NetworkInterfaces"] == [
        {
            "AssociatePublicIpAddress": False,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [
                {"Fn::GetAtt": ["TrainSecurityGroup", "GroupId"]}
            ],
            "SubnetId": {"Ref": "PrivateSubnetA"},
        }
    ]
    mappings = launch_data["BlockDeviceMappings"]
    assert isinstance(mappings, list) and len(mappings) == 1
    ebs = mappings[0]["Ebs"]
    assert ebs["DeleteOnTermination"] is True
    assert ebs["Encrypted"] is True
    assert ebs["KmsKeyId"] == {"Fn::GetAtt": ["DataKey", "Arn"]}


def test_launch_template_does_not_block_evidence_gated_cleanup(template):
    launch_data = template["Resources"]["P5LaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]

    assert launch_data["DisableApiTermination"] is False
    controller_actions = _attached_role_actions(template, "ControllerRole")
    assert "ec2:ModifyInstanceAttribute" in controller_actions
    assert "ec2:TerminateInstances" in controller_actions


def test_s3_kms_and_ecr_state_is_encrypted_versioned_and_retained(template):
    resources = template["Resources"]

    data_key = resources["DataKey"]
    assert data_key["Type"] == "AWS::KMS::Key"
    assert data_key["DeletionPolicy"] == "Retain"
    assert data_key["UpdateReplacePolicy"] == "Retain"
    assert data_key["Properties"]["EnableKeyRotation"] is True
    assert data_key["Properties"]["KeySpec"] == "SYMMETRIC_DEFAULT"
    assert data_key["Properties"]["KeyUsage"] == "ENCRYPT_DECRYPT"

    approval_key = resources["ApprovalSigningKey"]
    assert approval_key["Type"] == "AWS::KMS::Key"
    assert approval_key["DeletionPolicy"] == "Retain"
    assert approval_key["UpdateReplacePolicy"] == "Retain"
    assert approval_key["Properties"]["KeySpec"] == "ECC_NIST_P384"
    assert approval_key["Properties"]["KeyUsage"] == "SIGN_VERIFY"

    bucket = resources["ArtifactBucket"]
    assert bucket["Type"] == "AWS::S3::Bucket"
    assert bucket["DeletionPolicy"] == "Retain"
    assert bucket["UpdateReplacePolicy"] == "Retain"
    assert bucket["Properties"]["VersioningConfiguration"] == {
        "Status": "Enabled"
    }
    assert bucket["Properties"]["PublicAccessBlockConfiguration"] == {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }
    encryption = bucket["Properties"]["BucketEncryption"][
        "ServerSideEncryptionConfiguration"
    ]
    assert encryption == [
        {
            "BucketKeyEnabled": True,
            "ServerSideEncryptionByDefault": {
                "KMSMasterKeyID": {"Fn::GetAtt": ["DataKey", "Arn"]},
                "SSEAlgorithm": "aws:kms",
            },
        }
    ]

    repository = resources["ContainerRepository"]
    assert repository["Type"] == "AWS::ECR::Repository"
    assert repository["DeletionPolicy"] == "Retain"
    assert repository["UpdateReplacePolicy"] == "Retain"
    assert repository["Properties"]["ImageTagMutability"] == "IMMUTABLE"
    assert repository["Properties"]["ImageScanningConfiguration"] == {
        "ScanOnPush": True
    }
    assert repository["Properties"]["EncryptionConfiguration"] == {
        "EncryptionType": "KMS",
        "KmsKey": {"Fn::GetAtt": ["DataKey", "Arn"]},
    }


def test_roles_are_separate_and_only_signer_can_sign_approvals(template):
    resources = template["Resources"]
    role_names = {"TrainRole", "EvaluatorRole", "ControllerRole", "SignerRole"}
    assert {
        name
        for name, resource in resources.items()
        if resource["Type"] == "AWS::IAM::Role"
    } == role_names
    for name in role_names:
        assert "RoleName" not in resources[name]["Properties"]

    profiles = _resources(template, "AWS::IAM::InstanceProfile")
    assert set(profiles) == {"TrainInstanceProfile"}
    assert profiles["TrainInstanceProfile"]["Properties"]["Roles"] == [
        {"Ref": "TrainRole"}
    ]

    actions = {
        name: _attached_role_actions(template, name) for name in role_names
    }
    assert "kms:Sign" in actions["SignerRole"]
    assert actions["SignerRole"] == {
        "kms:DescribeKey",
        "kms:GetPublicKey",
        "kms:Sign",
    }
    for name in role_names - {"SignerRole"}:
        assert "kms:Sign" not in actions[name]
    assert {"ec2:RunInstances", "iam:PassRole", "ssm:SendCommand"} <= actions[
        "ControllerRole"
    ]
    assert "ec2:RunInstances" not in actions["EvaluatorRole"]
    assert "iam:PassRole" not in actions["EvaluatorRole"]
    assert {"ecr:GetAuthorizationToken", "s3:GetObject"} <= actions["TrainRole"]

    train_trust = resources["TrainRole"]["Properties"][
        "AssumeRolePolicyDocument"
    ]
    assert train_trust["Statement"][0]["Principal"] == {
        "Service": "ec2.amazonaws.com"
    }


def test_controller_covers_rendered_calls_without_unrendered_instance_mutations(
    template,
):
    actions = _attached_role_actions(template, "ControllerRole")
    rendered_actions = {
        "ec2:CreateTags",
        "ec2:DescribeImages",
        "ec2:DescribeInstanceAttribute",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypeOfferings",
        "ec2:ModifyInstanceAttribute",
        "ec2:TerminateInstances",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:PutObject",
        "ssm:CancelCommand",
        "ssm:CreateDocument",
        "ssm:DescribeInstanceInformation",
        "ssm:GetCommandInvocation",
        "ssm:ListCommands",
        "ssm:ListDocuments",
        "ssm:SendCommand",
        "sts:GetCallerIdentity",
    }
    assert rendered_actions <= actions
    assert actions.isdisjoint(
        {
            "ec2:StartInstances",
            "ec2:StopInstances",
            "ssm:ListCommandInvocations",
        }
    )


def test_controller_ebs_kms_permissions_are_exact_and_grant_is_constrained(
    template,
):
    resources = template["Resources"]
    controller_policy = resources["ControllerOperationsPolicy"]
    assert controller_policy["Type"] == "AWS::IAM::Policy"
    assert controller_policy["Properties"]["Roles"] == [
        {"Ref": "ControllerRole"}
    ]
    statements = {
        statement["Sid"]: statement
        for statement in controller_policy["Properties"]["PolicyDocument"][
            "Statement"
        ]
    }
    expected_use_actions = {
        "kms:Decrypt",
        "kms:DescribeKey",
        "kms:GenerateDataKey",
        "kms:GenerateDataKeyWithoutPlaintext",
        "kms:ReEncryptFrom",
        "kms:ReEncryptTo",
    }
    use = statements["UseDataKeyForObjectsAndEbs"]
    assert set(use["Action"]) == expected_use_actions
    assert use["Resource"] == {"Fn::GetAtt": ["DataKey", "Arn"]}
    grant = statements["CreateEbsGrant"]
    assert grant["Action"] == "kms:CreateGrant"
    assert grant["Resource"] == {"Fn::GetAtt": ["DataKey", "Arn"]}
    assert grant["Condition"] == {
        "Bool": {"kms:GrantIsForAWSResource": "true"}
    }

    key_statements = {
        statement["Sid"]: statement
        for statement in resources["DataKey"]["Properties"]["KeyPolicy"][
            "Statement"
        ]
    }
    key_use = key_statements["AllowControllerDataAndEbsUse"]
    assert set(key_use["Action"]) == expected_use_actions
    assert key_use["Principal"] == {
        "AWS": {"Fn::GetAtt": ["ControllerRole", "Arn"]}
    }
    key_grant = key_statements["AllowControllerEbsGrant"]
    assert key_grant["Action"] == "kms:CreateGrant"
    assert key_grant["Principal"] == {
        "AWS": {"Fn::GetAtt": ["ControllerRole", "Arn"]}
    }
    assert key_grant["Condition"] == {
        "Bool": {"kms:GrantIsForAWSResource": "true"}
    }
    assert not {
        "kms:CreateGrant*",
        "kms:GenerateDataKey*",
        "kms:ListGrants",
        "kms:RevokeGrant",
        "kms:*",
    } & _attached_role_actions(template, "ControllerRole")


def test_train_and_evaluator_s3_permissions_are_prefix_scoped(template):
    bucket_arn = {"Fn::GetAtt": ["ArtifactBucket", "Arn"]}
    assert template["Parameters"]["ArtifactRootPrefix"] == {
        "Type": "String",
        "Default": "memorysplit-v3",
        "AllowedPattern": (
            "^[a-z0-9](?:[a-z0-9._-]{0,61}[a-z0-9])?"
            "(?:/[a-z0-9](?:[a-z0-9._-]{0,61}[a-z0-9])?)*$"
        ),
        "MinLength": 1,
        "MaxLength": 127,
        "Description": "Durable S3 key prefix used as MS_S3_ROOT.",
    }
    train = _attached_role_statements(template, "TrainRole")
    assert train["ListTrainPrefixes"] == {
        "Sid": "ListTrainPrefixes",
        "Effect": "Allow",
        "Action": ["s3:ListBucket", "s3:ListBucketVersions"],
        "Resource": bucket_arn,
        "Condition": {
            "ForAnyValue:StringLike": {
                "s3:prefix": [
                    {"Fn::Sub": "${ArtifactRootPrefix}/canary-roundtrip"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/canary-roundtrip/*"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/checkpoints"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/checkpoints/*"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/dataset"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/dataset/*"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/operations"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/operations/*"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/receipts"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/receipts/*"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/releases"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/releases/*"},
                ]
            }
        },
    }
    assert train["ReadBootstrapInputs"]["Resource"] == [
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/dataset/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/releases/*"
            )
        },
    ]
    assert train["ReadWriteTrainingState"]["Resource"] == [
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/"
                "canary-roundtrip/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/checkpoints/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/operations/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/receipts/*"
            )
        },
    ]

    evaluator = _attached_role_statements(template, "EvaluatorRole")
    assert evaluator["ListEvaluatorPrefixes"] == {
        "Sid": "ListEvaluatorPrefixes",
        "Effect": "Allow",
        "Action": ["s3:ListBucket", "s3:ListBucketVersions"],
        "Resource": bucket_arn,
        "Condition": {
            "ForAnyValue:StringLike": {
                "s3:prefix": [
                    {"Fn::Sub": "${ArtifactRootPrefix}/checkpoints"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/checkpoints/*"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/evaluations"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/evaluations/*"},
                    {
                        "Fn::Sub": (
                            "${ArtifactRootPrefix}/receipts/checkpoints"
                        )
                    },
                    {
                        "Fn::Sub": (
                            "${ArtifactRootPrefix}/receipts/checkpoints/*"
                        )
                    },
                    {
                        "Fn::Sub": (
                            "${ArtifactRootPrefix}/receipts/evaluations"
                        )
                    },
                    {
                        "Fn::Sub": (
                            "${ArtifactRootPrefix}/receipts/evaluations/*"
                        )
                    },
                    {"Fn::Sub": "${ArtifactRootPrefix}/sealed"},
                    {"Fn::Sub": "${ArtifactRootPrefix}/sealed/*"},
                ]
            }
        },
    }
    assert evaluator["ReadEvaluationInputs"]["Resource"] == [
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/checkpoints/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/"
                "receipts/checkpoints/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/sealed/*"
            )
        },
    ]
    evaluation_outputs = evaluator["ReadWriteEvaluationEvidence"]["Resource"]
    assert evaluation_outputs == [
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/evaluations/*"
            )
        },
        {
            "Fn::Sub": (
                "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/"
                "receipts/evaluations/*"
            )
        },
    ]
    assert "s3:PutObject" not in evaluator["ReadEvaluationInputs"]["Action"]
    assert all(
        "checkpoints" not in resource["Fn::Sub"]
        for resource in evaluation_outputs
    )
    controller = _attached_role_statements(template, "ControllerRole")
    assert controller["ReadWriteOperationIntents"]["Resource"] == {
        "Fn::Sub": "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/*"
    }
    endpoint_statements = {
        statement["Sid"]: statement
        for statement in template["Resources"]["S3Endpoint"]["Properties"][
            "PolicyDocument"
        ]["Statement"]
    }
    assert endpoint_statements["MemorySplitVersionedObjects"]["Resource"] == {
        "Fn::Sub": "${ArtifactBucket.Arn}/${ArtifactRootPrefix}/*"
    }


def test_controller_mutations_and_ssm_targets_are_strictly_scoped(template):
    statements = _attached_role_statements(template, "ControllerRole")
    run = statements["RunApprovedLaunchTemplate"]
    assert run["Action"] == "ec2:RunInstances"
    assert run["Resource"] == "*"
    assert run["Condition"] == {
        "ArnEquals": {
            "ec2:LaunchTemplate": {
                "Fn::Sub": (
                    "arn:${AWS::Partition}:ec2:${AWS::Region}:"
                    "${AWS::AccountId}:launch-template/"
                    "${P5LaunchTemplate}"
                )
            }
        },
        "Bool": {"ec2:IsLaunchTemplateResource": "true"},
        "StringEquals": {
            "aws:RequestTag/MemorySplitManaged": "true",
            "ec2:InstanceType": "p5.48xlarge",
            "ec2:MetadataHttpTokens": "required",
        },
    }
    launch_data = template["Resources"]["P5LaunchTemplate"]["Properties"][
        "LaunchTemplateData"
    ]
    for specification in launch_data["TagSpecifications"]:
        assert {
            "Key": "MemorySplitManaged",
            "Value": "true",
        } in specification["Tags"]

    instance_arn = {
        "Fn::Sub": (
            "arn:${AWS::Partition}:ec2:${AWS::Region}:"
            "${AWS::AccountId}:instance/*"
        )
    }
    modify = statements["SetApprovedInstanceShutdown"]
    assert modify["Action"] == "ec2:ModifyInstanceAttribute"
    assert modify["Resource"] == instance_arn
    assert modify["Condition"] == {
        "StringEquals": {
            "ec2:ResourceTag/MemorySplitManaged": "true"
        }
    }
    terminate = statements["TerminateBoundInstance"]
    assert terminate["Action"] == "ec2:TerminateInstances"
    assert terminate["Resource"] == instance_arn
    assert terminate["Condition"] == {
        "StringEquals": {
            "ec2:ResourceTag/MemorySplitManaged": "true",
            "ec2:ResourceTag/MemorySplitProvider": "aws-p5.48xlarge",
        }
    }
    bind = statements["BindApprovedInstance"]
    assert bind["Action"] == "ec2:CreateTags"
    assert bind["Resource"] == instance_arn
    assert bind["Condition"]["StringEquals"] == {
        "aws:RequestTag/MemorySplitProvider": "aws-p5.48xlarge",
        "ec2:ResourceTag/MemorySplitManaged": "true",
    }
    assert set(
        bind["Condition"]["ForAllValues:StringEquals"]["aws:TagKeys"]
    ) == {
        "MemorySplitCohortSHA256",
        "MemorySplitContainerDigest",
        "MemorySplitDatasetSHA256",
        "MemorySplitProfileSHA256",
        "MemorySplitProvider",
        "MemorySplitReleaseSHA256",
        "MemorySplitRunManifestSHA256",
        "MemorySplitRuntimeSHA256",
        "MemorySplitSeed",
        "MemorySplitTerminateAt",
    }

    document_arn = {
        "Fn::Sub": (
            "arn:${AWS::Partition}:ssm:${AWS::Region}:"
            "${AWS::AccountId}:document/MemorySplit-ArgvV1"
        )
    }
    assert statements["CreateCanonicalArgvDocument"]["Resource"] == document_arn
    assert statements["SendCanonicalArgvDocument"] == {
        "Sid": "SendCanonicalArgvDocument",
        "Effect": "Allow",
        "Action": "ssm:SendCommand",
        "Resource": document_arn,
    }
    assert statements["SendToTaggedInstances"] == {
        "Sid": "SendToTaggedInstances",
        "Effect": "Allow",
        "Action": "ssm:SendCommand",
        "Resource": instance_arn,
        "Condition": {
            "StringEquals": {
                "ssm:resourceTag/MemorySplitManaged": "true",
                "ssm:resourceTag/MemorySplitProvider": "aws-p5.48xlarge",
            }
        },
    }

    for statement in statements.values():
        actions = statement["Action"]
        action_set = {actions} if isinstance(actions, str) else set(actions)
        if action_set & {
            "ec2:CreateTags",
            "ec2:ModifyInstanceAttribute",
            "ec2:TerminateInstances",
        }:
            assert statement["Resource"] != "*"


def test_ssm_document_content_and_hash_match_runtime_constants(template):
    document = template["Resources"]["ArgvDocument"]
    properties = document["Properties"]

    assert properties["Name"] == ARGV_DOCUMENT_NAME
    assert properties["DocumentType"] == "Command"
    assert properties["DocumentFormat"] == "JSON"
    assert properties["UpdateMethod"] == "NewVersion"
    assert canonical_json(properties["Content"]).decode("ascii") == (
        ARGV_DOCUMENT_CONTENT
    )
    assert template["Outputs"]["ArgvDocumentName"]["Value"] == {
        "Ref": "ArgvDocument"
    }
    assert template["Outputs"]["ArgvDocumentSha256"]["Value"] == (
        ARGV_DOCUMENT_SHA256
    )


def test_budget_has_multiple_email_alerts(template):
    budgets = _resources(template, "AWS::Budgets::Budget")
    assert set(budgets) == {"MonthlyBudget"}
    notifications = budgets["MonthlyBudget"]["Properties"][
        "NotificationsWithSubscribers"
    ]
    assert isinstance(notifications, list) and len(notifications) >= 2
    for item in notifications:
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
    parameters = yaml.safe_load(data)["Parameters"]
    assert not any(
        re.search(r"credential|password|private.?key|secret|token", name, re.I)
        for name in parameters
    )


def test_cfn_guard_rules_and_iac_dev_requirements_cover_task_5a():
    assert GUARD_PATH.is_file(), "Task 5A cfn-guard rules are missing"
    guard = GUARD_PATH.read_text(encoding="utf-8")
    for resource_type in sorted(PAID_CAPACITY_TYPES | PUBLIC_NETWORK_TYPES):
        assert resource_type in guard
    for required_text in (
        "HttpTokens",
        "IMMUTABLE",
        "MapPublicIpOnLaunch",
        "SecurityGroupIngress",
        "SYMMETRIC_DEFAULT",
        "ECC_NIST_P384",
    ):
        assert required_text in guard

    assert DEV_REQUIREMENTS_PATH.is_file()
    requirements = [
        line.strip()
        for line in DEV_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == ["cfn-lint==1.53.2"]


def test_cfn_guard_required_resources_and_role_policies_are_non_vacuous():
    guard = GUARD_PATH.read_text(encoding="utf-8")

    assert "when %launch_templates !empty" not in guard
    assert "when %buckets !empty" not in guard
    assert "when %repositories !empty" not in guard
    for logical_id in (
        "P5LaunchTemplate",
        "ArtifactBucket",
        "ContainerRepository",
        "DataKey",
        "ApprovalSigningKey",
        "TrainRole",
        "EvaluatorRole",
        "ControllerRole",
        "ControllerOperationsPolicy",
        "SignerRole",
        "TrainInstanceProfile",
    ):
        assert f"Resources.{logical_id}.Type ==" in guard
    for policy_invariant in (
        "Sid == 'ListTrainPrefixes'",
        "Sid == 'ListEvaluatorPrefixes'",
        "Sid == 'RunApprovedLaunchTemplate'",
        "Sid == 'CreateEbsGrant'",
        "Sid == 'SendToTaggedInstances'",
        "Sid == 'TerminateBoundInstance'",
        "kms:GrantIsForAWSResource",
        "ec2:IsLaunchTemplateResource",
        "ssm:resourceTag/MemorySplitManaged",
        "some Action[*] == 'kms:Sign'",
    ):
        assert policy_invariant in guard
