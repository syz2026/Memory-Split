from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from msctl.aws_readiness import (
    DIAGNOSTIC_IDS,
    DIAGNOSTIC_RECEIPT_TYPE,
    _expected_command_plan_sha256,
    load_launch_readiness,
    plan_launch_readiness,
    validate_launch_readiness,
)
from msctl.errors import MsctlError
from msctl.jsonutil import canonical_json
from tests.test_aws_gpu_canary import _v3_receipt
from tests.test_v3_hardware_amendment import (
    P5,
    _sealed_release,
    _selection,
)


def _write(path: Path, value: object) -> Path:
    path.write_bytes(canonical_json(value) + b"\n")
    return path


def _readiness_inputs(tmp_path: Path) -> dict[str, object]:
    profile, amendment, selection = _selection(P5)
    release = SimpleNamespace(archive_sha256="2" * 64)
    instance_id = "i-0123456789abcdef0"
    boot_id = "12345678-1234-4234-9234-123456789abc"
    environment = {
        "schema_version": 3,
        "profile_sha256": profile.sha256,
        "container_image_digest": selection.container_digest,
        "boot_id": boot_id,
        "aws_instance_identity_document": {
            "accountId": selection.aws_account_id,
            "imageId": selection.ami_id,
            "instanceId": instance_id,
            "instanceType": profile.instance_type,
            "region": selection.region,
        },
        "aws_instance_identity_pkcs7": "reviewed-pkcs7",
    }
    environment_path = _write(tmp_path / "environment.json", environment)
    environment_sha256 = hashlib.sha256(environment_path.read_bytes()).hexdigest()

    qualification = _v3_receipt(profile)
    qualification["provenance"] = {
        "instance_id": instance_id,
        "boot_id": boot_id,
        "region": selection.region,
        "ami_id": selection.ami_id,
        "ami_owner_id": selection.ami_owner_id,
        "release_sha256": release.archive_sha256,
        "container_image": selection.container_image,
        "container_digest": selection.container_digest,
        "provider_selection_sha256": selection.sha256,
        "environment_receipt_sha256": environment_sha256,
        "qualified_at": "2026-07-24T01:00:00Z",
        "command_plan_sha256": _expected_command_plan_sha256(
            profile=profile,
            selection=selection,
            release=release,
        ),
    }
    qualification_path = _write(
        tmp_path / "qualification.json",
        qualification,
    )

    diagnostic_paths: dict[str, Path] = {}
    for index, diagnostic_id in enumerate(DIAGNOSTIC_IDS):
        diagnostic_paths[diagnostic_id] = _write(
            tmp_path / f"{diagnostic_id}.json",
            {
                "schema_version": 3,
                "receipt_type": DIAGNOSTIC_RECEIPT_TYPE,
                "diagnostic_id": diagnostic_id,
                "model_parameters": 28_969_216,
                "targets_per_update": 524_288,
                "optimizer_steps": 1_106,
                "raw_target_tokens": 579_862_528,
                "artifact_sha256": f"{index + 1:x}" * 64,
                "passed": True,
                "reviewer": "gate-reviewer",
                "completed_at": "2026-07-24T00:30:00Z",
            },
        )
    return {
        "profile": profile,
        "amendment": amendment,
        "selection": selection,
        "release": release,
        "environment_receipt": environment_path,
        "qualification_receipt": qualification_path,
        "diagnostic_receipts": diagnostic_paths,
        "sealed_evaluation_release": _sealed_release(tmp_path),
        "reviewer": "launch-reviewer",
        "reviewed_at": "2026-07-24T02:00:00Z",
    }


def test_readiness_cli_contract_binds_every_gate_and_is_exclusive(tmp_path):
    inputs = _readiness_inputs(tmp_path)
    destination = tmp_path / "readiness.json"

    dry_run = plan_launch_readiness(
        out=destination,
        apply=False,
        **inputs,
    )
    assert dry_run["published"] is False
    assert not destination.exists()

    applied = plan_launch_readiness(
        out=destination,
        apply=True,
        **inputs,
    )
    assert applied["published"] is True
    loaded = load_launch_readiness(
        destination,
        profile=inputs["profile"],
        amendment=inputs["amendment"],
        selection=inputs["selection"],
        release=inputs["release"],
        environment_receipt=inputs["environment_receipt"],
        qualification_receipt=inputs["qualification_receipt"],
        diagnostic_receipts=inputs["diagnostic_receipts"],
        sealed_evaluation_release=inputs["sealed_evaluation_release"],
        expected_instance_id="i-0123456789abcdef0",
    )
    assert loaded.sha256 == applied["receipt_sha256"]
    assert loaded.decision["protected_launch_allowed"] is True
    assert set(loaded.bindings["diagnostic_receipt_sha256"]) == set(
        DIAGNOSTIC_IDS
    )
    assert loaded.bindings["study_lock_sha256"] != (
        loaded.bindings["preregistration_sha256"]
    )

    with pytest.raises(MsctlError, match="replace"):
        plan_launch_readiness(out=destination, apply=True, **inputs)


def test_readiness_rejects_false_stale_and_changed_bound_evidence(tmp_path):
    inputs = _readiness_inputs(tmp_path)
    planned = plan_launch_readiness(
        out=tmp_path / "unused.json",
        apply=False,
        **inputs,
    )
    receipt = planned["receipt"]

    denied = copy.deepcopy(receipt)
    denied["decision"]["protected_launch_allowed"] = False
    with pytest.raises(MsctlError, match="false, stale, or cross-profile"):
        validate_launch_readiness(
            denied,
            profile=inputs["profile"],
            amendment=inputs["amendment"],
            selection=inputs["selection"],
            release=inputs["release"],
        )

    with pytest.raises(MsctlError, match="false, stale, or cross-profile"):
        validate_launch_readiness(
            receipt,
            profile=inputs["profile"],
            amendment=inputs["amendment"],
            selection=inputs["selection"],
            release=SimpleNamespace(archive_sha256="f" * 64),
        )

    diagnostic_id = DIAGNOSTIC_IDS[0]
    diagnostic_path = inputs["diagnostic_receipts"][diagnostic_id]
    changed = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    changed["artifact_sha256"] = "f" * 64
    _write(diagnostic_path, changed)
    readiness_path = _write(tmp_path / "readiness.json", receipt)
    with pytest.raises(MsctlError, match="binding is stale"):
        load_launch_readiness(
            readiness_path,
            profile=inputs["profile"],
            amendment=inputs["amendment"],
            selection=inputs["selection"],
            release=inputs["release"],
            diagnostic_receipts=inputs["diagnostic_receipts"],
        )
