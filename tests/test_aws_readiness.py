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
    load_diagnostic_receipt,
    load_launch_readiness,
    plan_launch_readiness,
    sign_diagnostic_receipt,
    sign_launch_readiness,
    validate_launch_readiness,
)
from msctl.errors import MsctlError
from msctl.jsonutil import canonical_json
from tests.test_aws_gpu_canary import _v3_receipt
from tests.test_v3_hardware_amendment import (
    P5,
    _sealed_fixture,
    _selection,
)


def _write(path: Path, value: object) -> Path:
    path.write_bytes(canonical_json(value) + b"\n")
    return path


def _readiness_inputs(tmp_path: Path, monkeypatch) -> dict[str, object]:
    monkeypatch.setenv("MSCTL_APPROVAL_KEY", "k" * 32)
    monkeypatch.setattr(
        "msctl.aws_identity.verify_instance_identity_pkcs7",
        lambda identity, pkcs7, region: (
            bool(identity) and pkcs7 == "YQ==" and bool(region)
        ),
    )
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
        "aws_instance_identity_pkcs7": "YQ==",
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
        artifact = tmp_path / "artifacts" / f"{diagnostic_id}.bin"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(f"diagnostic-{index}\n".encode("ascii"))
        unsigned = {
            "schema_version": 3,
            "receipt_type": DIAGNOSTIC_RECEIPT_TYPE,
            "diagnostic_id": diagnostic_id,
            "model_parameters": 28_969_216,
            "targets_per_update": 524_288,
            "optimizer_steps": 1_106,
            "raw_target_tokens": 579_862_528,
            "artifact_path": f"artifacts/{diagnostic_id}.bin",
            "artifact_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
            "passed": True,
            "reviewer": "gate-reviewer",
            "completed_at": "2026-07-24T00:30:00Z",
        }
        diagnostic_paths[diagnostic_id] = _write(
            tmp_path / f"{diagnostic_id}.json",
            sign_diagnostic_receipt(unsigned, key_id="qualification-review"),
        )
    return {
        "profile": profile,
        "amendment": amendment,
        "selection": selection,
        "release": release,
        "environment_receipt": environment_path,
        "qualification_receipt": qualification_path,
        "diagnostic_receipts": diagnostic_paths,
        "sealed_evaluation_fixture": _sealed_fixture(tmp_path),
        "reviewer": "launch-reviewer",
        "reviewed_at": "2026-07-24T02:00:00Z",
        "key_id": "qualification-review",
    }


def test_readiness_cli_contract_binds_every_gate_and_is_exclusive(
    tmp_path,
    monkeypatch,
):
    inputs = _readiness_inputs(tmp_path, monkeypatch)
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
        sealed_evaluation_fixture=inputs["sealed_evaluation_fixture"],
        expected_instance_id="i-0123456789abcdef0",
    )
    assert loaded.sha256 == applied["receipt_sha256"]
    assert loaded.decision["protected_launch_allowed"] is True
    assert loaded.value["key_id"] == "qualification-review"
    assert len(loaded.value["signature"]) == 64
    assert set(loaded.bindings["diagnostic_receipt_sha256"]) == set(
        DIAGNOSTIC_IDS
    )
    assert "sealed_evaluation_sha256" not in loaded.bindings
    assert "study_lock_sha256" not in loaded.bindings
    assert loaded.bindings["sealed_fixture_sha256"]

    with pytest.raises(MsctlError, match="replace"):
        plan_launch_readiness(out=destination, apply=True, **inputs)


def test_readiness_rejects_false_stale_and_changed_bound_evidence(
    tmp_path,
    monkeypatch,
):
    inputs = _readiness_inputs(tmp_path, monkeypatch)
    planned = plan_launch_readiness(
        out=tmp_path / "unused.json",
        apply=False,
        **inputs,
    )
    receipt = planned["receipt"]

    denied = copy.deepcopy(receipt)
    denied["decision"]["protected_launch_allowed"] = False
    with pytest.raises(MsctlError, match="signature is invalid"):
        validate_launch_readiness(
            denied,
            profile=inputs["profile"],
            amendment=inputs["amendment"],
            selection=inputs["selection"],
            release=inputs["release"],
        )

    denied = sign_launch_readiness(
        denied,
        key_id="qualification-review",
    )
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
    with pytest.raises(MsctlError, match="signature is invalid"):
        load_launch_readiness(
            readiness_path,
            profile=inputs["profile"],
            amendment=inputs["amendment"],
            selection=inputs["selection"],
            release=inputs["release"],
            diagnostic_receipts=inputs["diagnostic_receipts"],
        )


def test_diagnostic_signature_and_artifact_rehash_reject_tampering(
    tmp_path,
    monkeypatch,
):
    inputs = _readiness_inputs(tmp_path, monkeypatch)
    diagnostic_id = DIAGNOSTIC_IDS[0]
    receipt_path = inputs["diagnostic_receipts"][diagnostic_id]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    unsigned = dict(receipt)
    unsigned["signature"] = ""
    _write(receipt_path, unsigned)
    with pytest.raises(MsctlError, match="unsigned"):
        load_diagnostic_receipt(receipt_path, diagnostic_id=diagnostic_id)

    _write(receipt_path, receipt)
    artifact = tmp_path / str(receipt["artifact_path"])
    artifact.write_bytes(b"tampered diagnostic output\n")
    with pytest.raises(MsctlError, match="artifact hash does not match"):
        load_diagnostic_receipt(receipt_path, diagnostic_id=diagnostic_id)


def test_diagnostic_artifact_rehash_rejects_symlinked_parent(
    tmp_path,
    monkeypatch,
):
    inputs = _readiness_inputs(tmp_path, monkeypatch)
    diagnostic_id = DIAGNOSTIC_IDS[0]
    receipt_path = inputs["diagnostic_receipts"][diagnostic_id]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    outside = tmp_path.parent / f"{tmp_path.name}-outside-artifacts"
    outside.mkdir()
    artifact = outside / "diagnostic.bin"
    artifact.write_bytes(b"diagnostic-0\n")
    (tmp_path / "linked-artifacts").symlink_to(outside, target_is_directory=True)
    changed = {
        **receipt,
        "artifact_path": "linked-artifacts/diagnostic.bin",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    _write(
        receipt_path,
        sign_diagnostic_receipt(changed, key_id="qualification-review"),
    )

    with pytest.raises(MsctlError, match="cannot be rehashed safely"):
        load_diagnostic_receipt(receipt_path, diagnostic_id=diagnostic_id)
