from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from msctl.approval import verify_scope_approval
from msctl.aws_argv import RemoteIntentError, _validate_intent
from msctl.aws_fleet import create_fleet_plan, validate_fleet_plan
from msctl.aws_p5 import AwsP5Backend, V3LifecycleContext, aws_resource_request
from msctl.aws_readiness import LaunchReadiness
from msctl.aws_selection import load_hardware_amendment
from msctl.aws_sealed_evaluation import SealedEvaluationFixture
from msctl.contracts import load_run_manifest
from msctl.errors import MsctlError
from msctl.jsonutil import canonical_json, canonical_sha256
from tests.test_v3_hardware_amendment import (
    AMENDMENT,
    CONTAINER_DIGEST,
    CONTAINER_IMAGE,
    P5,
    ROOT,
    _manifests,
    _selection,
)


def _approval_resources(profile: object, operation: str) -> dict[str, object]:
    policy = {
        "submit": (1440, 192.0, "scripts/run_train.py"),
        "resume": (1440, 192.0, "scripts/run_train.py"),
        "evaluate": (360, 48.0, "evals/confirmatory/runner.py"),
    }
    wall_minutes, gpu_hours, script = policy[operation]
    resources: dict[str, object] = {
        "schema_version": 1,
        "operation": operation,
        "jobs": 1,
        "allocated_gpus": 8,
        "wall_minutes": wall_minutes,
        "gpu_hours": gpu_hours,
        "gres": getattr(profile, "gres"),
        "script": script,
        "ami_id": "ami-0123456789abcdef0",
        "container_image": CONTAINER_IMAGE,
        "container_digest": CONTAINER_DIGEST,
        "instance_id": "i-0123456789abcdef0",
        "instance_type": getattr(profile, "instance_type"),
        "profile_sha256": getattr(profile, "sha256"),
        "provider": getattr(profile, "provider"),
        "release_sha256": "1" * 64,
        "run_manifest_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "seed": 0,
        "terminate_at": "2097-12-31T23:00:00Z",
        "cohort_assignment_sha256": "4" * 64,
        "preregistration_sha256": "5" * 64,
        "hardware_amendment_sha256": "6" * 64,
        "provider_selection_sha256": "7" * 64,
        "sealed_fixture_sha256": "8" * 64,
        "fleet_plan_sha256": "9" * 64,
        "fleet_wave": 0,
        "control_bundle_sha256": "a" * 64,
        "launch_readiness_sha256": "b" * 64,
        "dataset_pointer_sha256": "c" * 64,
        "dataset_verification_sha256": "d" * 64,
        "environment_receipt_sha256": "e" * 64,
    }
    if operation == "resume" or (
        operation == "evaluate"
        and getattr(profile, "provider").endswith("-v3")
    ):
        resources["checkpoint_receipt_sha256"] = "f" * 64
    if operation == "evaluate":
        resources["sealed_evaluation_sha256"] = "0" * 64
        resources["study_lock_sha256"] = "f" * 64
    return resources


def _approval_receipt(
    resources: dict[str, object],
    *,
    key: str,
    key_id: str = "reviewer-a",
) -> dict[str, object]:
    unsigned = {
        "schema_version": 1,
        "receipt_id": f"{resources['operation']}-approval",
        "provider": resources["provider"],
        "operation": resources["operation"],
        "release_sha256": resources["release_sha256"],
        "run_manifest_sha256": resources["run_manifest_sha256"],
        "resources": resources,
        "limits": {"gpu_hours": 192.0, "jobs": 1},
        "expires_at": "2098-01-01T00:00:00Z",
        "key_id": key_id,
    }
    return {
        **unsigned,
        "signature": hmac.new(
            key.encode("utf-8"),
            canonical_json(unsigned),
            hashlib.sha256,
        ).hexdigest(),
    }


def _write_receipt(path: Path, value: object) -> Path:
    path.write_bytes(canonical_json(value) + b"\n")
    return path


@pytest.mark.parametrize("operation", ["submit", "resume", "evaluate"])
def test_v3_approval_requires_and_tamper_binds_lifecycle_evidence(
    operation: str,
    tmp_path: Path,
) -> None:
    profile, _amendment, _selection_value = _selection(P5)
    key = "v3-approval-key-" * 3
    resources = _approval_resources(profile, operation)
    receipt = _approval_receipt(resources, key=key)
    path = _write_receipt(tmp_path / f"{operation}.json", receipt)
    kwargs = {
        "path": path,
        "operation": operation,
        "release_sha256": resources["release_sha256"],
        "scope_sha256": resources["run_manifest_sha256"],
        "profile": profile,
        "environ": {"MSCTL_APPROVAL_KEY": key},
        "now": datetime(2026, 7, 25, tzinfo=UTC),
    }

    verified = verify_scope_approval(resources=resources, **kwargs)
    assert verified["resources"] == resources

    for field in (
        "dataset_pointer_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
    ):
        tampered = {**resources, field: "0" * 64}
        with pytest.raises(MsctlError) as changed:
            verify_scope_approval(resources=tampered, **kwargs)
        assert changed.value.code == "APPROVAL_SCOPE_MISMATCH"

        missing = {key: value for key, value in resources.items() if key != field}
        missing_path = _write_receipt(
            tmp_path / f"{operation}-missing-{field}.json",
            _approval_receipt(missing, key=key),
        )
        with pytest.raises(MsctlError):
            verify_scope_approval(
                **{**kwargs, "path": missing_path},
                resources=missing,
            )


@pytest.mark.parametrize("operation", ["submit", "resume", "evaluate"])
def test_legacy_aws_approval_also_requires_lifecycle_evidence(
    operation: str,
    tmp_path: Path,
) -> None:
    profile = SimpleNamespace(
        provider="aws-p5.48xlarge",
        instance_type="p5.48xlarge",
        sha256="a" * 64,
        gres="gpu:h100:8",
        assigned_seeds=(1, 2, 3, 4),
    )
    resources = _approval_resources(profile, operation)
    resources["seed"] = 1
    for field in (
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "hardware_amendment_sha256",
        "provider_selection_sha256",
        "sealed_fixture_sha256",
        "fleet_plan_sha256",
        "fleet_wave",
        "control_bundle_sha256",
        "launch_readiness_sha256",
        "sealed_evaluation_sha256",
        "study_lock_sha256",
    ):
        resources.pop(field, None)
    key = "legacy-lifecycle-approval-key-" * 2
    path = _write_receipt(
        tmp_path / f"legacy-{operation}.json",
        _approval_receipt(resources, key=key),
    )
    kwargs = {
        "path": path,
        "operation": operation,
        "release_sha256": resources["release_sha256"],
        "scope_sha256": resources["run_manifest_sha256"],
        "profile": profile,
        "environ": {"MSCTL_APPROVAL_KEY": key},
        "now": datetime(2026, 7, 25, tzinfo=UTC),
    }

    assert verify_scope_approval(resources=resources, **kwargs)[
        "resources"
    ] == resources
    for field in (
        "dataset_pointer_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
    ):
        missing = {key: value for key, value in resources.items() if key != field}
        missing_path = _write_receipt(
            tmp_path / f"legacy-{operation}-missing-{field}.json",
            _approval_receipt(missing, key=key),
        )
        with pytest.raises(MsctlError):
            verify_scope_approval(
                **{**kwargs, "path": missing_path},
                resources=missing,
            )


def test_approval_key_id_selects_configured_key_and_legacy_key_still_works(
    tmp_path: Path,
) -> None:
    profile, _amendment, _selection_value = _selection(P5)
    resources = _approval_resources(profile, "submit")
    selected_key = "selected-approval-key-" * 2
    legacy_key = "legacy-approval-key-" * 2
    receipt = _approval_receipt(resources, key=selected_key, key_id="reviewer-a")
    path = _write_receipt(tmp_path / "approval.json", receipt)
    common = {
        "path": path,
        "operation": "submit",
        "release_sha256": resources["release_sha256"],
        "scope_sha256": resources["run_manifest_sha256"],
        "resources": resources,
        "profile": profile,
        "now": datetime(2026, 7, 25, tzinfo=UTC),
    }

    assert verify_scope_approval(
        **common,
        environ={
            "MSCTL_APPROVAL_KEY": legacy_key,
            "MSCTL_APPROVAL_KEYS": json.dumps(
                {"reviewer-a": selected_key, "reviewer-b": "b" * 32}
            ),
        },
    )["key_id"] == "reviewer-a"
    with pytest.raises(MsctlError) as wrong_selected_key:
        verify_scope_approval(
            **common,
            environ={
                "MSCTL_APPROVAL_KEY": selected_key,
                "MSCTL_APPROVAL_KEYS": json.dumps({"reviewer-a": "x" * 32}),
            },
        )
    assert wrong_selected_key.value.code == "APPROVAL_SIGNATURE_INVALID"

    legacy_receipt = _approval_receipt(resources, key=legacy_key)
    legacy_path = _write_receipt(tmp_path / "legacy.json", legacy_receipt)
    assert verify_scope_approval(
        **{**common, "path": legacy_path},
        environ={"MSCTL_APPROVAL_KEY": legacy_key},
    )["receipt_id"] == "submit-approval"


def test_canary_approval_binds_plan_instance_runtime_and_environment(
    tmp_path: Path,
) -> None:
    profile, _amendment, selection = _selection(P5)
    plan_sha256 = "1" * 64
    release_sha256 = "2" * 64
    resources = aws_resource_request(
        "canary",
        profile=profile,
        bindings={
            "ami_id": selection.ami_id,
            "container_image": selection.container_image,
            "container_digest": selection.container_digest,
            "instance_id": "i-0123456789abcdef0",
            "instance_type": profile.instance_type,
            "profile_sha256": profile.sha256,
            "provider": profile.provider,
            "release_sha256": release_sha256,
            "runtime_sha256": "3" * 64,
            "provider_selection_sha256": selection.sha256,
            "environment_receipt_sha256": "4" * 64,
            "command_plan_sha256": "5" * 64,
            "orchestration_plan_sha256": plan_sha256,
            "control_bundle_sha256": "6" * 64,
        },
    )
    key = "canary-approval-key-" * 2
    unsigned = {
        "schema_version": 1,
        "receipt_id": "canary-approval",
        "provider": profile.provider,
        "operation": "canary",
        "release_sha256": release_sha256,
        "run_manifest_sha256": plan_sha256,
        "resources": resources,
        "limits": {"gpu_hours": 8.0, "jobs": 1},
        "expires_at": "2098-01-01T00:00:00Z",
        "key_id": "reviewer-a",
    }
    receipt = {
        **unsigned,
        "signature": hmac.new(
            key.encode("utf-8"),
            canonical_json(unsigned),
            hashlib.sha256,
        ).hexdigest(),
    }
    path = _write_receipt(tmp_path / "canary-approval.json", receipt)

    assert verify_scope_approval(
        path,
        operation="canary",
        release_sha256=release_sha256,
        scope_sha256=plan_sha256,
        resources=resources,
        profile=profile,
        environ={"MSCTL_APPROVAL_KEY": key},
        now=datetime(2026, 7, 25, tzinfo=UTC),
    )["resources"] == resources

    with pytest.raises(MsctlError):
        verify_scope_approval(
            path,
            operation="canary",
            release_sha256=release_sha256,
            scope_sha256=plan_sha256,
            resources={
                key: value
                for key, value in resources.items()
                if key != "environment_receipt_sha256"
            },
            profile=profile,
            environ={"MSCTL_APPROVAL_KEY": key},
            now=datetime(2026, 7, 25, tzinfo=UTC),
        )


def _backend_and_environment(tmp_path: Path, monkeypatch):
    profile, _amendment, selection = _selection(P5)
    runtime = SimpleNamespace(
        region=selection.region,
        s3_root="s3://memorysplit-prod/cohort-v3",
        kms_key_id="kms-key",
        ami_id=selection.ami_id,
        container_image=selection.container_image,
        container_digest=selection.container_digest,
        uid=10001,
        gid=10002,
    )
    backend = AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
    )
    instance_id = "i-0123456789abcdef0"
    environment = {
        "schema_version": 3,
        "profile_sha256": profile.sha256,
        "container_image_digest": selection.container_digest,
        "boot_id": "12345678-1234-4234-9234-123456789abc",
        "aws_instance_identity_document": {
            "accountId": selection.aws_account_id,
            "imageId": selection.ami_id,
            "instanceId": instance_id,
            "instanceType": selection.instance_type,
            "region": selection.region,
        },
        "aws_instance_identity_pkcs7": "YQ==",
    }
    monkeypatch.setattr(
        "msctl.aws_identity.verify_instance_identity_pkcs7",
        lambda identity, pkcs7, region: (
            bool(identity) and pkcs7 == "YQ==" and region == selection.region
        ),
    )
    return backend, profile, selection, instance_id, environment


def _dataset_inputs(tmp_path: Path, profile: object, selection: object):
    dataset_sha256 = "3" * 64
    pointer = {
        "schema_version": 1,
        "dataset_id": "memorysplit-current",
        "provider": getattr(profile, "provider"),
        "materialization": "s3",
        "durable_uri_env": "MS_S3_ROOT",
        "scratch_root": "/mnt/memorysplit",
        "relative_path": "dataset",
        "required_receipt": "dataset/receipt.json",
        "required_sidecars": [],
        "source_lock_manifest": "sources/current.lock.json",
        "full_corpus_in_release": False,
    }
    pointer_path = _write_receipt(tmp_path / "dataset-pointer.json", pointer)
    unsigned_verification = {
        "receipt_sha256": dataset_sha256,
        "verified_source": "immutable-s3",
    }
    verification = {
        **unsigned_verification,
        "verification_sha256": canonical_sha256(unsigned_verification),
    }
    verification_path = _write_receipt(
        tmp_path / "dataset-verification.json",
        verification,
    )
    manifest = SimpleNamespace(
        schema_version=3,
        dataset_sha256=dataset_sha256,
        provider_selection_sha256=getattr(selection, "sha256"),
    )
    return manifest, pointer_path, verification_path


def test_v3_lifecycle_environment_uses_shared_identity_verifier_and_safe_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend, profile, selection, instance_id, environment = (
        _backend_and_environment(tmp_path, monkeypatch)
    )
    manifest, pointer, verification = _dataset_inputs(
        tmp_path,
        profile,
        selection,
    )
    environment_path = _write_receipt(tmp_path / "environment.json", environment)
    expected_environment_sha256 = hashlib.sha256(
        environment_path.read_bytes()
    ).hexdigest()

    evidence = backend._load_lifecycle_evidence(
        manifest=manifest,
        dataset_pointer=pointer,
        dataset_root=None,
        dataset_verification=verification,
        environment_receipt=environment_path,
        selection=selection,
        expected_instance_id=instance_id,
    )
    assert evidence["environment_receipt_sha256"] == expected_environment_sha256

    tampered_fields = {
        "accountId": "999999999999",
        "imageId": "ami-11111111111111111",
        "instanceId": "i-11111111111111111",
        "instanceType": "p6-b300.48xlarge",
        "region": "us-west-2",
    }
    for field, value in tampered_fields.items():
        changed = copy.deepcopy(environment)
        changed["aws_instance_identity_document"][field] = value
        changed_path = _write_receipt(
            tmp_path / f"environment-{field}.json",
            changed,
        )
        with pytest.raises(MsctlError) as rejected:
            backend._load_lifecycle_evidence(
                manifest=manifest,
                dataset_pointer=pointer,
                dataset_root=None,
                dataset_verification=verification,
                environment_receipt=changed_path,
                selection=selection,
                expected_instance_id=instance_id,
            )
        assert rejected.value.code == "ENVIRONMENT_RECEIPT_INVALID"

    with pytest.raises(MsctlError) as no_selection:
        backend._load_lifecycle_evidence(
            manifest=manifest,
            dataset_pointer=pointer,
            dataset_root=None,
            dataset_verification=verification,
            environment_receipt=environment_path,
            selection=None,
            expected_instance_id=instance_id,
        )
    assert no_selection.value.code == "ENVIRONMENT_RECEIPT_INVALID"

    with pytest.raises(MsctlError) as no_instance:
        backend._load_lifecycle_evidence(
            manifest=manifest,
            dataset_pointer=pointer,
            dataset_root=None,
            dataset_verification=verification,
            environment_receipt=environment_path,
            selection=selection,
            expected_instance_id=None,
        )
    assert no_instance.value.code == "ENVIRONMENT_RECEIPT_INVALID"

    symlink = tmp_path / "environment-symlink.json"
    symlink.symlink_to(environment_path)
    with pytest.raises(MsctlError) as unsafe_symlink:
        backend._load_lifecycle_evidence(
            manifest=manifest,
            dataset_pointer=pointer,
            dataset_root=None,
            dataset_verification=verification,
            environment_receipt=symlink,
            selection=selection,
            expected_instance_id=instance_id,
        )
    assert unsafe_symlink.value.code == "ENVIRONMENT_RECEIPT_INVALID"

    hardlink = tmp_path / "environment-hardlink.json"
    os.link(environment_path, hardlink)
    with pytest.raises(MsctlError) as unsafe_hardlink:
        backend._load_lifecycle_evidence(
            manifest=manifest,
            dataset_pointer=pointer,
            dataset_root=None,
            dataset_verification=verification,
            environment_receipt=hardlink,
            selection=selection,
            expected_instance_id=instance_id,
        )
    assert unsafe_hardlink.value.code == "ENVIRONMENT_RECEIPT_INVALID"


@pytest.mark.parametrize("operation", ["submit", "resume", "evaluate"])
def test_v3_lifecycle_never_synthesizes_missing_or_mismatched_evidence(
    operation: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend, _profile, _selection_value, _instance_id, _environment = (
        _backend_and_environment(tmp_path, monkeypatch)
    )
    manifest = SimpleNamespace(schema_version=3, dataset_sha256="3" * 64)
    for missing_value, code in (
        (None, "LIFECYCLE_EVIDENCE_REQUIRED"),
        ({}, "LIFECYCLE_EVIDENCE_INVALID"),
    ):
        with pytest.raises(MsctlError) as missing:
            backend._validated_lifecycle_evidence(
                operation=operation,
                manifest=manifest,
                evidence=missing_value,
                context=None,
            )
        assert missing.value.code == code

    readiness = SimpleNamespace(
        bindings={"environment_receipt_sha256": "7" * 64}
    )
    with pytest.raises(MsctlError) as mismatched:
        backend._validated_lifecycle_evidence(
            operation=operation,
            manifest=manifest,
            evidence={
                "dataset_pointer_sha256": "5" * 64,
                "dataset_verification_sha256": "6" * 64,
                "environment_receipt_sha256": "8" * 64,
            },
            context=SimpleNamespace(readiness=readiness),
        )
    assert mismatched.value.code == "ENVIRONMENT_RECEIPT_MISMATCH"


def _v3_submit_intent(tmp_path: Path) -> tuple[dict[str, object], str]:
    profile, selection, manifest_paths = _manifests(tmp_path)
    instance_id = "i-0123456789abcdef0"
    plan_value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        instance_ids=[instance_id],
        repo_root=ROOT,
    )
    plan = validate_fleet_plan(
        plan_value,
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=ROOT,
    )
    manifest = load_run_manifest(manifest_paths[0], repo_root=ROOT)
    amendment = load_hardware_amendment(AMENDMENT)
    environment_sha256 = "7" * 64
    readiness = LaunchReadiness(
        profile_id=profile.profile_id,
        sha256="8" * 64,
        bindings={
            "release_sha256": manifest.release_sha256,
            "provider_selection_sha256": selection.sha256,
            "hardware_amendment_sha256": amendment.sha256,
            "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
            "environment_receipt_sha256": environment_sha256,
        },
        decision={"protected_launch_allowed": True},
        path=None,
        value={},
    )
    context = V3LifecycleContext(
        amendment=amendment,
        selection=selection,
        fleet_plan=plan,
        fleet_binding=plan.binding_for_seed(manifest.seed),
        readiness=readiness,
        sealed_fixture=SealedEvaluationFixture(
            root=tmp_path / "sealed",
            sha256=manifest.sealed_fixture_sha256,
            members={},
        ),
    )
    runtime = SimpleNamespace(
        region=selection.region,
        s3_root="s3://memorysplit-prod/cohort-v3",
        kms_key_id="kms-key",
        ami_id=selection.ami_id,
        container_image=selection.container_image,
        container_digest=selection.container_digest,
        uid=10001,
        gid=10002,
    )
    backend = AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
    )
    release = SimpleNamespace(
        provider=profile.provider,
        archive_sha256=manifest.release_sha256,
        receipt_sha256="9" * 64,
        members_sha256="a" * 64,
        source_commit=manifest.source_commit,
    )
    terminate_at = (
        datetime.now(UTC) + timedelta(hours=1)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=None,
        terminate_at=terminate_at,
        approval_path=None,
        apply=False,
        evidence={
            "dataset_pointer_sha256": "5" * 64,
            "dataset_verification_sha256": "6" * 64,
            "environment_receipt_sha256": environment_sha256,
        },
        context=context,
    )
    return result["operation_intent"], backend.control_bundle.sha256


def _recommit_intent(intent: dict[str, object]) -> bytes:
    identity = {
        key: value
        for key, value in intent.items()
        if key
        not in {
            "operation_id",
            "ssm_document",
            "started_receipt_uri",
            "terminal_receipt_uri",
        }
    }
    operation_id = canonical_sha256(identity)
    intent["operation_id"] = operation_id
    s3_root = intent["environment"]["MS_S3_ROOT"]
    receipt_root = f"{s3_root}/operations/{operation_id}/receipts"
    intent["started_receipt_uri"] = f"{receipt_root}/started.json"
    intent["terminal_receipt_uri"] = f"{receipt_root}/terminal.json"
    return canonical_json(intent)


def test_v3_remote_intent_rejects_unpinned_executables_shells_and_scripts(
    tmp_path: Path,
) -> None:
    intent, control_bundle_sha256 = _v3_submit_intent(tmp_path)
    payload = canonical_json(intent)
    assert _validate_intent(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_control_bundle_sha256=control_bundle_sha256,
    ) == intent

    mutations = []
    arbitrary = copy.deepcopy(intent)
    arbitrary["steps"][0]["argv"][0] = "/usr/local/bin/unreviewed"
    mutations.append(arbitrary)
    shell = copy.deepcopy(intent)
    shell["steps"][-1]["argv"][0] = "/bin/sh"
    mutations.append(shell)
    escaped_script = copy.deepcopy(intent)
    escaped_script["steps"][-1]["argv"][1] = "/tmp/escaped.py"
    mutations.append(escaped_script)
    wrong_control_root = copy.deepcopy(intent)
    bootstrap = next(
        step for step in wrong_control_root["steps"] if step["name"] == "bootstrap"
    )
    bootstrap["argv"][1] = (
        "/opt/memorysplit/control/" + "0" * 64 + "/cluster/aws/p5/bootstrap.py"
    )
    mutations.append(wrong_control_root)

    for changed in mutations:
        changed_payload = _recommit_intent(changed)
        with pytest.raises(RemoteIntentError):
            _validate_intent(
                changed_payload,
                expected_sha256=hashlib.sha256(changed_payload).hexdigest(),
                expected_control_bundle_sha256=control_bundle_sha256,
            )


def test_v3_submit_rejects_every_recommitted_argv_edit(tmp_path: Path) -> None:
    intent, control_bundle_sha256 = _v3_submit_intent(tmp_path)
    python_argv = [
        item
        for step in intent["steps"]
        if step["argv"][0] == "/usr/bin/python3"
        for item in step["argv"]
    ]
    for required in (
        "--release-receipt",
        "--release-receipt-sha256",
        "--code-commit",
        "--release-members-sha256",
        "--run-manifest-sha256",
        "--bootstrap-receipt",
        "--corpus-receipt",
        "--run",
        "--manifest",
        "--profile",
        "--repo-root",
    ):
        assert required in python_argv

    for step_index, step in enumerate(intent["steps"]):
        for argument_index, argument in enumerate(step["argv"]):
            changed = copy.deepcopy(intent)
            changed["steps"][step_index]["argv"][argument_index] = (
                f"{argument}-recommitted-edit"
            )
            changed_payload = _recommit_intent(changed)
            with pytest.raises(RemoteIntentError):
                _validate_intent(
                    changed_payload,
                    expected_sha256=hashlib.sha256(changed_payload).hexdigest(),
                    expected_control_bundle_sha256=control_bundle_sha256,
                )

    renamed = copy.deepcopy(intent)
    renamed["steps"][0]["name"] += "-recommitted-edit"
    renamed_payload = _recommit_intent(renamed)
    with pytest.raises(RemoteIntentError):
        _validate_intent(
            renamed_payload,
            expected_sha256=hashlib.sha256(renamed_payload).hexdigest(),
            expected_control_bundle_sha256=control_bundle_sha256,
        )
