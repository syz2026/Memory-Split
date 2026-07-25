from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from msctl.cli import build_parser, main
from msctl.errors import MsctlError


REPO_ROOT = Path(__file__).resolve().parents[1]
START = REPO_ROOT / "AWS-GPU-V3-START.md"
RUNBOOK = REPO_ROOT / "docs" / "AWS-GPU-V3-RUNBOOK.md"
ACCESS = REPO_ROOT / "docs" / "AWS-GPU-ACCESS-REQUEST.md"


def _squash(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())


def test_operator_docs_cover_closed_profiles_and_complete_lifecycle() -> None:
    text = START.read_text(encoding="utf-8") + RUNBOOK.read_text(encoding="utf-8")
    lowered = _squash(text)

    for required in (
        "cluster/profiles/aws-p5.48xlarge-v3.json",
        "cluster/profiles/aws-p6-b300.48xlarge-v3.json",
        "ubuntu 24.04",
        "single cuda",
        "immutable ami id",
        "private ecr",
        "repository@sha256",
        "static credentials",
        "dedicated instance role",
        "private networking",
        "ssm",
        "explicit instance id",
        "six 29m diagnostics",
        "provider-selection receipt",
        "readiness create",
        "protected_launch_allowed",
        "environment",
        "export ms_aws_ami_id",
        "export ms_container_image",
        "export ms_s3_root",
        "export ms_s3_kms_key_id",
        "export ms_aws_instance_profile_arn",
        "bootstrap",
        "control install",
        "aws-runshellscript",
        "canary",
        "100 updates",
        "10 warmup",
        "4+4",
        "one p6",
        "sequential",
        "four p5",
        "3/3/2/2",
        "30 minutes",
        "resume",
        "evaluate",
        "collect",
        "teardown",
        "non-cancellable",
        "exact price",
        "fleet advance",
        "apply-time mutation inventory",
        "/opt/venv/bin/python",
        "ms_s3_kms_key_id",
        "o_nofollow",
        "--if-none-match '*'",
        "credential_process",
        "aws_shared_credentials_file=/dev/null",
        "aws_ec2_metadata_disabled=true",
        "load_operator_credential_process",
        "validate_aws_gpu_launch_request.py",
        "capacityreservation.capacityreservationid",
        "checksum-mode enabled",
        "lexicographic id order",
    ):
        assert required in lowered

    assert "`msctl` never calls `runinstances`" in lowered
    assert "never provisions ec2 capacity" in lowered
    assert "never purchases a capacity block" in lowered
    assert "all ten pairs run sequentially" in lowered
    assert "seeds 0/4/8, 1/5/9, 2/6, and 3/7" in lowered
    assert '$operator_root/receipts/environment-${instance_id}.json' in lowered
    assert "never reuse a receipt from another fleet id" in lowered
    assert "manual or external tag deletion is never progression evidence" in lowered
    assert '"markettype": "capacity-block"' in lowered
    assert '"capacityreservationtarget"' in lowered
    assert "separately controlled evaluator release" in lowered
    assert "actual `study-lock.json` bytes" in lowered
    assert "outside the reviewed source checkout" in lowered
    assert "aws s3 sync /mnt/memorysplit/runs" not in lowered
    assert "--start-date-range replace_with_earliest_start" in lowered
    assert "--end-date-range replace_with_latest_end" in lowered
    assert '["result"]["bundle_sha256"]' in text
    assert '["result"]["command_id"]' in text
    assert '["result"]["intent"]["qualification_receipt_uri"]' in text
    assert '["request_sha256"]' in text
    assert "--bundle-sha256 \"$CONTROL_BUNDLE_SHA256\"" in text
    assert "docker login" in text
    assert "docker pull" in text
    assert text.index("docker login") < text.index("docker pull")
    assert text.count("docker login") >= 2
    image_apply = text.index('  --apply > "$IMAGE_RESULT"')
    assert text.index("docker login") < image_apply
    assert image_apply < text.index("docker logout", image_apply)
    instance_login = text.index("/usr/bin/docker login")
    instance_pull = text.index("/usr/bin/docker pull")
    assert text.index("aws ecr batch-get-image") < instance_login < instance_pull
    assert instance_pull < text.index("/usr/bin/docker image inspect")
    assert (
        "${MS_S3_ROOT}/dataset/${relative}" in text
        and "/mnt/memorysplit/dataset/receipt.json" in text
    )
    assert '--container-image "$MS_CONTAINER_IMAGE"' in text
    for static_key_name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert static_key_name not in text
    assert ":latest" not in text


def test_operator_commands_use_python3_and_documented_cli_shapes_parse(
    tmp_path,
    capsys,
) -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    start = START.read_text(encoding="utf-8")
    bash = "\n".join(
        re.findall(r"```bash\n(.*?)\n```", runbook + start, flags=re.DOTALL)
    )
    assert re.search(r"(?<![/\w])python(?:\s|$)", bash) is None

    bundle = tmp_path / "control-bundle.tar"
    install = build_parser().parse_args(
        [
            "--profile",
            "cluster/profiles/aws-p5.48xlarge-v3.json",
            "control",
            "install",
            "--instance-id",
            "i-0123456789abcdef0",
            "--bundle",
            str(bundle),
            "--bundle-sha256",
            "a" * 64,
        ]
    )
    assert install.bundle == str(bundle)
    assert install.bundle_sha256 == "a" * 64
    with pytest.raises(MsctlError):
        build_parser().parse_args(
            [
                "control",
                "install",
                "--instance-id",
                "i-0123456789abcdef0",
            ]
        )

    assert (
        main(
            [
                "--profile",
                str(
                    REPO_ROOT
                    / "cluster"
                    / "profiles"
                    / "aws-p5.48xlarge-v3.json"
                ),
                "--repo-root",
                str(REPO_ROOT),
                "control",
                "bundle",
                "--out",
                str(bundle),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["command"] == "control bundle"
    assert report["dry_run"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", report["result"]["bundle_sha256"])
    assert not bundle.exists()


def test_documented_bootstrap_paths_and_capacity_response_shapes_match_parsers(
    tmp_path,
) -> None:
    from cluster.aws.p5.bootstrap import (
        BootstrapEvidence,
        _parser as bootstrap_parser,
        render_bootstrap_commands,
    )
    from cluster.aws.p5.profile import (
        load_aws_gpu_profile,
        validate_runtime_environment,
    )
    from scripts.validate_aws_gpu_launch_request import (
        extract_capacity_reservation_id,
    )

    arguments = bootstrap_parser().parse_args(
        [
            "--profile",
            "/opt/memorysplit/control/hash/cluster/profiles/"
            "aws-p6-b300.48xlarge-v3.json",
            "--container-image",
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/repo@sha256:"
            + "a" * 64,
            "--release-archive",
            "/mnt/memorysplit/staging/releases/archive/release.zip",
            "--release-sha256",
            "a" * 64,
            "--release-receipt",
            "/mnt/memorysplit/staging/releases/archive/RELEASE.json",
            "--release-receipt-sha256",
            "b" * 64,
            "--dataset-receipt",
            "/mnt/memorysplit/dataset/receipt.json",
            "--dataset-receipt-sha256",
            "c" * 64,
            "--cohort-assignment",
            "/mnt/memorysplit/staging/releases/archive/"
            "cohort-assignment-v3.json",
            "--cohort-assignment-sha256",
            "d" * 64,
            "--code-commit",
            "e" * 40,
            "--owner-uid",
            "10001",
            "--owner-gid",
            "10001",
            "--aws-private-home",
            "/run/memorysplit-aws",
        ]
    )
    assert arguments.dataset_receipt == Path(
        "/mnt/memorysplit/dataset/receipt.json"
    )
    assert arguments.release_archive == Path(
        "/mnt/memorysplit/staging/releases/archive/release.zip"
    )

    profile = load_aws_gpu_profile(
        REPO_ROOT
        / "cluster"
        / "profiles"
        / "aws-p6-b300.48xlarge-v3.json"
    )
    runtime = validate_runtime_environment(
        profile,
        {
            "AWS_REGION": "us-east-1",
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
            "MS_CONTAINER_IMAGE": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/repo@sha256:"
                + "a" * 64
            ),
            "MS_RUNTIME_UID": "10001",
            "MS_RUNTIME_GID": "10001",
            "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v3",
            "MS_S3_KMS_KEY_ID": (
                "arn:aws:kms:us-east-1:123456789012:"
                "key/12345678-1234-4234-9234-123456789abc"
            ),
        },
    )
    evidence = BootstrapEvidence(
        instance_id="i-0123456789abcdef0",
        instance_type=profile.instance_type,
        ami_id=runtime.ami_id,
        boot_id="12345678-1234-4234-9234-123456789abc",
        account_id="123456789012",
        identity_document={},
        identity_pkcs7="fixture",
        role_name="memorysplit-v3",
        role_arn="arn:aws:iam::123456789012:role/memorysplit-v3",
        gpu_names=("NVIDIA B300",) * 8,
        fabric_manager_active=True,
        instance_store_devices=tuple(
            f"/dev/nvme{index}n1" for index in range(8)
        ),
        container_image=runtime.container_image,
    )
    commands = render_bootstrap_commands(
        profile,
        runtime,
        evidence,
        owner_uid=runtime.uid,
        owner_gid=runtime.gid,
    )
    syncs = [
        command
        for command in commands
        if command[:3] == ["aws", "s3", "sync"]
    ]
    assert [command[3:5] for command in syncs] == [
        [
            "s3://memorysplit-prod/cohort-v3/releases",
            "/mnt/memorysplit/staging/releases",
        ],
        [
            "s3://memorysplit-prod/cohort-v3/dataset",
            "/mnt/memorysplit/dataset",
        ],
    ]
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "${MS_S3_ROOT}/releases/${RELEASE_SHA256}" in runbook
    assert "${MS_S3_ROOT}/dataset/${relative}" in runbook
    assert "extract_capacity_reservation_id" in runbook
    assert "extract_run_instance_ids" in runbook

    purchase = {
        "CapacityReservation": {
            "CapacityReservationId": "cr-0123456789abcdef0"
        }
    }
    purchase_path = tmp_path / "capacity-block-purchase.json"
    purchase_path.write_text(json.dumps(purchase), encoding="utf-8")
    assert (
        extract_capacity_reservation_id(purchase_path)
        == "cr-0123456789abcdef0"
    )


def test_paid_and_mutating_runbook_steps_are_reviewed_before_apply() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    lowered = _squash(text)

    for command in (
        "package_aws_gpu_handoff.py",
        "build_aws_gpu_image.py",
        "provider select",
        "purchase-capacity-block",
        "run-instances",
        "control install",
        "runs instantiate",
        "fleet plan",
        "fleet advance",
        "\n  submit ",
        "\n  resume ",
        "\n  evaluate ",
        "\n  collect ",
        "stop-instances",
        "start-instances",
        "terminate-instances",
    ):
        assert text.lower().count(command.lower()) >= 2, command

    for aws_command in (
        "purchase-capacity-block",
        "run-instances",
        "stop-instances",
        "start-instances",
        "terminate-instances",
    ):
        pattern = re.compile(
            rf"aws ec2 {aws_command}\b.*?(?:--dry-run|--no-dry-run)",
            flags=re.DOTALL,
        )
        matches = pattern.findall(text)
        assert matches, aws_command
    assert text.count("--dry-run") >= 6
    assert text.count("--dryrun") >= 2
    assert text.count("--apply") >= 12
    assert text.count("# DRY RUN") >= 12
    assert "PAID APPLY" in text
    assert "exact-price approval" in text
    assert "do not redirect either packaging report into the repository" in lowered
    assert "never in the source checkout" in lowered
    assert "package-plan.json" not in text
    assert "package-result.json" not in text


def test_runbook_bash_blocks_are_syntactically_valid() -> None:
    blocks = re.findall(
        r"```bash\n(.*?)\n```",
        RUNBOOK.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert blocks
    for index, block in enumerate(blocks):
        completed = subprocess.run(
            ["bash", "-n"],
            input=block,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"bash block {index} is invalid: {completed.stderr}"
        )


def test_access_request_is_temporary_least_privilege_and_complete() -> None:
    text = ACCESS.read_text(encoding="utf-8")
    lowered = _squash(text)

    for action in (
        "sts:GetCallerIdentity",
        "ec2:DescribeImages",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceAttribute",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeInstanceTypeOfferings",
        "ec2:DescribeCapacityBlockOfferings",
        "ec2:PurchaseCapacityBlock",
        "ec2:RunInstances",
        "ec2:CreateTags",
        "ec2:DeleteTags",
        "ec2:ModifyInstanceAttribute",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:TerminateInstances",
        "ssm:SendCommand",
        "ssm:GetParameter",
        "ssm:GetDocument",
        "ssm:ListDocuments",
        "ssm:CreateDocument",
        "ssm:GetCommandInvocation",
        "ssm:ListCommandInvocations",
        "ssm:CancelCommand",
        "ssmmessages:OpenDataChannel",
        "ecr:GetAuthorizationToken",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:BatchGetImage",
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:GenerateDataKey",
        "servicequotas:GetServiceQuota",
        "servicequotas:ListServiceQuotas",
        "pricing:GetProducts",
        "iam:PassRole",
    ):
        assert action in text

    for required in (
        "us-east-1",
        "us-west-2",
        "temporary",
        "time-bounded",
        "static access keys",
        "dedicated instance role",
        "iam:passedtoservice = ec2.amazonaws.com",
        "cohort prefixes",
        "private ecr",
        "baseline operator role",
        "should not include",
        "separate ticket or approval artifact",
        "exact offering id",
        "exact total price",
        "non-cancellable",
        "remove the purchase elevation",
        "internsandboxboundary",
        "identity policy cannot override",
    ):
        assert required in lowered

    assert "iam:PassRole` on `DedicatedInstanceRoleArn" in text
    assert "iam:PassRole` on wildcard" in text
    for forbidden_action in (
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "s3:DeleteObject",
        "s3:PutBucketPolicy",
        "kms:CreateKey",
        "kms:ScheduleKeyDeletion",
        "ecr:DeleteRepository",
    ):
        assert forbidden_action not in text
    assert re.search(r"(?<!\d)\d{12}(?!\d)", text) is None
    assert re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text) is None
