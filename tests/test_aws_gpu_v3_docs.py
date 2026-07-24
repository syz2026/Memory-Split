from __future__ import annotations

import re
import subprocess
from pathlib import Path


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
    for static_key_name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert static_key_name not in text
    assert ":latest" not in text


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
