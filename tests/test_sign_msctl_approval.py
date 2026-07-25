from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cluster.aws.p5.profile import load_aws_gpu_profile
from msctl.approval import KEYS_ENV, verify_scope_approval
from msctl.aws_p5 import aws_resource_request
from msctl.jsonutil import canonical_json
from scripts.sign_msctl_approval import (
    ApprovalSigningError,
    sign_approval_report,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
NOW = datetime(2026, 7, 25, 3, 0, tzinfo=UTC)
EXPIRES_AT = "2026-07-26T03:00:00Z"
KEY_ID = "reviewer-v1"
SECRET = "reviewer-controlled-secret-material-0001"


def _hash(character: str) -> str:
    return character * 64


def _lifecycle_resources(profile) -> dict[str, object]:
    return aws_resource_request(
        "submit",
        profile=profile,
        bindings={
            "ami_id": "ami-0123456789abcdef0",
            "container_image": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                f"memorysplit@sha256:{_hash('a')}"
            ),
            "container_digest": f"sha256:{_hash('a')}",
            "instance_id": "i-0123456789abcdef0",
            "instance_type": profile.instance_type,
            "profile_sha256": profile.sha256,
            "provider": profile.provider,
            "release_sha256": _hash("b"),
            "run_manifest_sha256": _hash("c"),
            "runtime_sha256": _hash("d"),
            "seed": 0,
            "terminate_at": "2026-07-26T00:00:00Z",
            "cohort_assignment_sha256": _hash("e"),
            "preregistration_sha256": _hash("f"),
            "hardware_amendment_sha256": _hash("1"),
            "provider_selection_sha256": _hash("2"),
            "sealed_fixture_sha256": _hash("3"),
            "fleet_plan_sha256": _hash("4"),
            "fleet_wave": 0,
            "control_bundle_sha256": _hash("5"),
            "dataset_pointer_sha256": _hash("6"),
            "dataset_verification_sha256": _hash("7"),
            "environment_receipt_sha256": _hash("8"),
            "launch_readiness_sha256": _hash("9"),
        },
    )


def _canary_resources(profile) -> dict[str, object]:
    return aws_resource_request(
        "canary",
        profile=profile,
        bindings={
            "ami_id": "ami-0123456789abcdef0",
            "container_image": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                f"memorysplit@sha256:{_hash('a')}"
            ),
            "container_digest": f"sha256:{_hash('a')}",
            "instance_id": "i-0123456789abcdef0",
            "instance_type": profile.instance_type,
            "profile_sha256": profile.sha256,
            "provider": profile.provider,
            "release_sha256": _hash("b"),
            "runtime_sha256": _hash("c"),
            "provider_selection_sha256": _hash("d"),
            "environment_receipt_sha256": _hash("e"),
            "command_plan_sha256": _hash("f"),
            "orchestration_plan_sha256": _hash("1"),
            "control_bundle_sha256": _hash("2"),
        },
    )


def _fleet_resources(profile) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "fleet-advance",
        "provider": profile.provider,
        "instance_type": profile.instance_type,
        "profile_sha256": profile.sha256,
        "gres": profile.gres,
        "fleet_plan_sha256": _hash("a"),
        "provider_selection_sha256": _hash("b"),
        "instance_id": "i-0123456789abcdef0",
        "from_seed": 0,
        "from_wave": 0,
        "from_manifest_sha256": _hash("c"),
        "to_seed": 1,
        "to_wave": 1,
        "to_manifest_sha256": _hash("d"),
        "training_state_sha256": _hash("e"),
        "checkpoint_receipt_sha256": _hash("f"),
        "checkpoint_records_sha256": _hash("1"),
        "jobs": 0,
        "allocated_gpus": 0,
        "wall_minutes": 0,
        "gpu_hours": 0,
        "script": "aws-fleet-advance",
    }


@pytest.mark.parametrize("operation", ["submit", "canary", "fleet-advance"])
def test_signer_emits_verifiable_operation_specific_scope(tmp_path, operation):
    profile = load_aws_gpu_profile(PROFILE)
    if operation == "submit":
        resources = _lifecycle_resources(profile)
        release_sha256 = str(resources["release_sha256"])
        scope_sha256 = str(resources["run_manifest_sha256"])
        result = {
            "run_manifest_sha256": scope_sha256,
            "approval_resources": resources,
        }
    elif operation == "canary":
        resources = _canary_resources(profile)
        release_sha256 = str(resources["release_sha256"])
        scope_sha256 = str(resources["orchestration_plan_sha256"])
        result = {
            "plan_sha256": scope_sha256,
            "approval_resources": resources,
        }
    else:
        resources = _fleet_resources(profile)
        release_sha256 = _hash("9")
        scope_sha256 = str(resources["fleet_plan_sha256"])
        result = {
            "release_sha256": release_sha256,
            "fleet_plan_sha256": scope_sha256,
            "approval_resources": resources,
        }
    report = tmp_path / f"{operation}-dry-run.json"
    report.write_bytes(
        canonical_json({"dry_run": True, "result": result}) + b"\n"
    )
    approval = tmp_path / f"{operation}-approval.json"
    environ = {KEYS_ENV: json.dumps({KEY_ID: SECRET})}

    signed = sign_approval_report(
        report,
        approval,
        operation=operation,
        key_id=KEY_ID,
        expires_at=EXPIRES_AT,
        environ=environ,
        now=NOW,
    )

    assert signed["release_sha256"] == release_sha256
    assert signed["scope_sha256"] == scope_sha256
    assert stat.S_IMODE(approval.stat().st_mode) == 0o600
    verified = verify_scope_approval(
        approval,
        operation=operation,
        release_sha256=release_sha256,
        scope_sha256=scope_sha256,
        resources=resources,
        profile=profile,
        environ=environ,
        now=NOW,
    )
    assert verified["resources"] == resources
    with pytest.raises(ApprovalSigningError, match="replace"):
        sign_approval_report(
            report,
            approval,
            operation=operation,
            key_id=KEY_ID,
            expires_at=EXPIRES_AT,
            environ=environ,
            now=NOW,
        )


def test_signer_rejects_fleet_report_without_top_level_release(tmp_path):
    profile = load_aws_gpu_profile(PROFILE)
    resources = _fleet_resources(profile)
    report = tmp_path / "fleet-dry-run.json"
    report.write_bytes(
        canonical_json(
            {
                "dry_run": True,
                "result": {
                    "fleet_plan_sha256": resources["fleet_plan_sha256"],
                    "approval_resources": resources,
                },
            }
        )
        + b"\n"
    )
    approval = tmp_path / "approval.json"

    with pytest.raises(ApprovalSigningError, match="release or scope"):
        sign_approval_report(
            report,
            approval,
            operation="fleet-advance",
            key_id=KEY_ID,
            expires_at=EXPIRES_AT,
            environ={KEYS_ENV: json.dumps({KEY_ID: SECRET})},
            now=NOW,
        )
    assert not approval.exists()
