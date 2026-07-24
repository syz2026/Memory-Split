from __future__ import annotations

import inspect
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.gpu_profile import load_aws_gpu_profile
from cluster.aws.qualification import CohortSelectionAuthority
from msctl.aws_hardware import AuthenticatedSelectionBinding
from msctl.aws_lifecycle import (
    LIFECYCLE_BINDING_FIELDS,
    AuthenticatedProviderLifecycle,
    ProviderLifecycleBinding,
    admit_provider_lifecycle,
    lifecycle_operational_metadata,
)
from msctl.contracts import RunManifestV3, load_run_manifest
from msctl.errors import MsctlError


ROOT = Path(__file__).resolve().parents[1]
P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"


def _selection(profile, *, arm: str = "dense") -> AuthenticatedSelectionBinding:
    return AuthenticatedSelectionBinding(
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        amendment_sha256="1" * 64,
        selection_sha256="2" * 64,
        selection_version_id="selection-version-1",
        profile_id=profile.profile_id,
        provider=profile.provider,
        profile_sha256=profile.sha256,
        runtime_lock_sha256="3" * 64,
        qualification_evidence_sha256="4" * 64,
        environment_receipt_sha256="5" * 64,
        canary_receipt_sha256="6" * 64,
        approval_receipt_sha256="7" * 64,
        approval_public_key_sha256="8" * 64,
        account_id="123456789012",
        instance_id="i-0123456789abcdef0",
        boot_id="12345678-1234-4abc-8def-1234567890ab",
        region="us-east-1",
        availability_zone="us-east-1d",
        purchase_model="on_demand",
        seed=0,
        arm=arm,
    )


def _authority(profile) -> CohortSelectionAuthority:
    dense = _selection(profile)
    return CohortSelectionAuthority(
        profile=profile,
        seed=0,
        arms=("dense", "split90"),
        bindings={
            "dense": dense,
            "split90": replace(dense, arm="split90"),
        },
    )


@pytest.mark.parametrize("profile_path", [P5_PROFILE, P6_PROFILE])
def test_lifecycle_authority_uses_only_fixed_paths_and_binds_reviewed_bytes(
    tmp_path,
    monkeypatch,
    profile_path,
):
    import msctl.aws_lifecycle as module

    profile = load_aws_gpu_profile(profile_path)
    authority = _authority(profile)
    calls = []
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_sbom = tmp_path / "runtime-sbom.json"
    objective = tmp_path / "configs" / "objective-controls-amendment-v3.yaml"
    runtime_lock.write_bytes(b"runtime-lock\n")
    runtime_sbom.write_bytes(b"runtime-sbom\n")
    objective.parent.mkdir()
    objective.write_bytes(b"objective\n")

    def admit(**kwargs):
        calls.append(kwargs)
        return authority

    monkeypatch.setattr(module, "admit_cohort_provider_selection", admit)
    monkeypatch.setattr(
        module,
        "_load_runtime_qualification_bundle",
        lambda **kwargs: SimpleNamespace(
            lock_sha256="3" * 64,
            sbom_sha256="9" * 64,
        ),
    )
    monkeypatch.setattr(
        module,
        "load_objective_controls_contract",
        lambda path: SimpleNamespace(
            objective_controls_contract_sha256="a" * 64,
            protected_outcomes_inspected=False,
            added_360m_controls=(),
        ),
    )

    admitted = admit_provider_lifecycle(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=tmp_path / "qualification-evidence.json",
        runtime_sbom_path=runtime_sbom,
        objective_controls_amendment_path=objective,
        store=object(),
        account_id="123456789012",
        instance_id="i-0123456789abcdef0",
        boot_id="12345678-1234-4abc-8def-1234567890ab",
        seed=0,
        expected_selection_version_id="selection-version-1",
        identity_verifier=object(),
        approval_verifier=object(),
        trusted_public_key_sha256="8" * 64,
    )

    assert isinstance(admitted, AuthenticatedProviderLifecycle)
    assert admitted.profile == profile
    assert admitted.arm_bindings == authority.bindings
    assert admitted.binding.to_dict() == {
        "account_id": "123456789012",
        "arms": ["dense", "split90"],
        "availability_zone": "us-east-1d",
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "hardware_amendment_sha256": "1" * 64,
        "instance_id": "i-0123456789abcdef0",
        "objective_controls_contract_sha256": "a" * 64,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "provider": profile.provider,
        "provider_selection_sha256": "2" * 64,
        "provider_selection_version_id": "selection-version-1",
        "purchase_model": "on_demand",
        "qualification_approval_public_key_sha256": "8" * 64,
        "qualification_approval_receipt_sha256": "7" * 64,
        "qualification_canary_receipt_sha256": "6" * 64,
        "qualification_environment_receipt_sha256": "5" * 64,
        "qualification_evidence_sha256": "4" * 64,
        "region": "us-east-1",
        "runtime_lock_sha256": "3" * 64,
        "runtime_sbom_sha256": "9" * 64,
        "seed": 0,
    }
    assert set(admitted.binding.to_dict()) == set(LIFECYCLE_BINDING_FIELDS)
    assert len(calls) == 1
    assert calls[0]["authority_root"] == tmp_path / "authority"
    assert calls[0]["repo_root"] == ROOT
    assert calls[0]["runtime_lock_path"] == runtime_lock
    assert calls[0]["seed"] == 0
    assert calls[0]["expected_selection_version_id"] == "selection-version-1"


def test_public_lifecycle_admission_accepts_no_caller_binding_or_profile():
    parameters = set(inspect.signature(admit_provider_lifecycle).parameters)
    assert "binding" not in parameters
    assert "profile" not in parameters
    assert "provider" not in parameters

    assert "profile_id" not in parameters
    assert {
        "authority_root",
        "repo_root",
        "runtime_lock_path",
        "runtime_evidence_path",
        "runtime_sbom_path",
        "objective_controls_amendment_path",
        "store",
        "account_id",
        "instance_id",
        "boot_id",
        "seed",
        "expected_selection_version_id",
        "identity_verifier",
        "approval_verifier",
        "trusted_public_key_sha256",
    } == parameters


@pytest.mark.parametrize(
    "mutation",
    ["arm-selection", "runtime-lock", "objective-outcomes", "objective-controls"],
)
def test_lifecycle_authority_rejects_cross_binding_and_unfrozen_objective(
    tmp_path,
    monkeypatch,
    mutation,
):
    import msctl.aws_lifecycle as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    authority = _authority(profile)
    if mutation == "arm-selection":
        authority = replace(
            authority,
            bindings={
                **authority.bindings,
                "split90": replace(
                    authority.bindings["split90"],
                    selection_sha256="b" * 64,
                ),
            },
        )
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_sbom = tmp_path / "runtime-sbom.json"
    objective = tmp_path / "configs" / "objective-controls-amendment-v3.yaml"
    runtime_lock.write_bytes(b"runtime-lock\n")
    runtime_sbom.write_bytes(b"runtime-sbom\n")
    objective.parent.mkdir()
    objective.write_bytes(b"objective\n")

    monkeypatch.setattr(
        module,
        "admit_cohort_provider_selection",
        lambda **kwargs: authority,
    )
    monkeypatch.setattr(
        module,
        "_load_runtime_qualification_bundle",
        lambda **kwargs: SimpleNamespace(
            lock_sha256=("c" * 64 if mutation == "runtime-lock" else "3" * 64),
            sbom_sha256="9" * 64,
        ),
    )
    monkeypatch.setattr(
        module,
        "load_objective_controls_contract",
        lambda path: SimpleNamespace(
            objective_controls_contract_sha256=(
                "not-a-hash" if mutation == "objective-controls" else "a" * 64
            ),
            protected_outcomes_inspected=mutation == "objective-outcomes",
            added_360m_controls=(),
        ),
    )

    with pytest.raises(ValueError, match="arm|selection|runtime|objective"):
        admit_provider_lifecycle(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=tmp_path / "qualification-evidence.json",
            runtime_sbom_path=runtime_sbom,
            objective_controls_amendment_path=objective,
            store=object(),
            account_id="123456789012",
            instance_id="i-0123456789abcdef0",
            boot_id="12345678-1234-4abc-8def-1234567890ab",
            seed=0,
            expected_selection_version_id="selection-version-1",
            identity_verifier=object(),
            approval_verifier=object(),
            trusted_public_key_sha256="8" * 64,
        )


def test_operational_metadata_is_strict_and_arm_scoped():
    profile = load_aws_gpu_profile(P5_PROFILE)
    authority = _authority(profile)
    selected = authority.bindings["dense"]
    binding = ProviderLifecycleBinding(
        cohort_id=selected.cohort_id,
        provider=selected.provider,
        profile_id=selected.profile_id,
        profile_sha256=selected.profile_sha256,
        hardware_amendment_sha256=selected.amendment_sha256,
        provider_selection_sha256=selected.selection_sha256,
        provider_selection_version_id=selected.selection_version_id,
        runtime_lock_sha256=selected.runtime_lock_sha256,
        runtime_sbom_sha256="9" * 64,
        qualification_evidence_sha256=selected.qualification_evidence_sha256,
        qualification_environment_receipt_sha256=(
            selected.environment_receipt_sha256
        ),
        qualification_canary_receipt_sha256=selected.canary_receipt_sha256,
        qualification_approval_receipt_sha256=selected.approval_receipt_sha256,
        qualification_approval_public_key_sha256=(
            selected.approval_public_key_sha256
        ),
        objective_controls_contract_sha256="a" * 64,
        account_id=selected.account_id,
        instance_id=selected.instance_id,
        boot_id=selected.boot_id,
        region=selected.region,
        availability_zone=selected.availability_zone,
        purchase_model=selected.purchase_model,
        seed=0,
        arms=("dense", "split90"),
    )
    metadata = lifecycle_operational_metadata(
        binding,
        run_id="memorysplit-v3-360m-s0-dense",
        arm="dense",
        config_sha256="b" * 64,
        dataset_receipt_sha256="c" * 64,
        dataset_build_id="d" * 64,
        ordered_stream_sha256="e" * 64,
        source_commit="f" * 40,
        source_tree="0" * 40,
    )

    assert metadata["provider"] == "aws-p5.48xlarge"
    assert metadata["arm"] == "dense"
    assert metadata["run_id"] == "memorysplit-v3-360m-s0-dense"
    with pytest.raises(ValueError, match="arm"):
        lifecycle_operational_metadata(
            binding,
            run_id="memorysplit-v3-360m-s0-dense",
            arm="split90",
            config_sha256="b" * 64,
            dataset_receipt_sha256="c" * 64,
            dataset_build_id="d" * 64,
            ordered_stream_sha256="e" * 64,
            source_commit="f" * 40,
            source_tree="0" * 40,
        )


def test_run_manifest_v3_closes_over_selected_provider_lifecycle(tmp_path):
    profile = load_aws_gpu_profile(P6_PROFILE)
    selected = _selection(profile)
    binding = ProviderLifecycleBinding(
        cohort_id=selected.cohort_id,
        provider=selected.provider,
        profile_id=selected.profile_id,
        profile_sha256=selected.profile_sha256,
        hardware_amendment_sha256=selected.amendment_sha256,
        provider_selection_sha256=selected.selection_sha256,
        provider_selection_version_id=selected.selection_version_id,
        runtime_lock_sha256=selected.runtime_lock_sha256,
        runtime_sbom_sha256="9" * 64,
        qualification_evidence_sha256=selected.qualification_evidence_sha256,
        qualification_environment_receipt_sha256=(
            selected.environment_receipt_sha256
        ),
        qualification_canary_receipt_sha256=selected.canary_receipt_sha256,
        qualification_approval_receipt_sha256=selected.approval_receipt_sha256,
        qualification_approval_public_key_sha256=(
            selected.approval_public_key_sha256
        ),
        objective_controls_contract_sha256="a" * 64,
        account_id=selected.account_id,
        instance_id=selected.instance_id,
        boot_id=selected.boot_id,
        region=selected.region,
        availability_zone=selected.availability_zone,
        purchase_model=selected.purchase_model,
        seed=0,
        arms=("dense", "split90"),
    )
    runs = []
    for arm in binding.arms:
        relative = f"configs/360m-v3/{arm}-s0.yaml"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        runs.append(
            {
                "run_id": f"memorysplit-v3-360m-s0-{arm}",
                "arm": arm,
                "seed": 0,
                "config": relative,
                "config_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "estimated_gpu_hours": 10.0,
            }
        )
    value = {
        **binding.to_dict(),
        "schema_version": 3,
        "release_sha256": "b" * 64,
        "release_receipt_sha256": "c" * 64,
        "dataset_pointer_sha256": "d" * 64,
        "dataset_receipt_sha256": "e" * 64,
        "dataset_build_id": "f" * 64,
        "ordered_stream_sha256": "0" * 64,
        "cohort_assignment_sha256": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "sealed_evaluation_release_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "source_tree": "5" * 40,
        "runs": runs,
    }
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    parsed = load_run_manifest(path, repo_root=tmp_path)

    assert isinstance(parsed, RunManifestV3)
    assert parsed.provider == profile.provider
    assert parsed.profile_id == profile.profile_id
    assert parsed.provider_selection_sha256 == binding.provider_selection_sha256
    assert (
        parsed.provider_selection_version_id
        == binding.provider_selection_version_id
    )
    assert parsed.runtime_sbom_sha256 == binding.runtime_sbom_sha256
    assert (
        parsed.objective_controls_contract_sha256
        == binding.objective_controls_contract_sha256
    )

    del value["provider_selection_sha256"]
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(MsctlError, match="missing|field|key"):
        load_run_manifest(path, repo_root=tmp_path)


def test_authenticated_launcher_manifest_derives_selected_profile_from_authority(
    tmp_path,
    monkeypatch,
):
    import msctl.aws_launch_manifest as module
    from tests.provider_lifecycle_fixtures import provider_lifecycle

    profile = load_aws_gpu_profile(P6_PROFILE)
    lifecycle = provider_lifecycle(profile)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    bootstrap = scratch / "staging" / "bootstrap.json"
    bootstrap.parent.mkdir()
    bootstrap.write_bytes(b'{"selected":true}\n')
    corpus = scratch / "dataset" / "receipt.json"
    corpus.parent.mkdir()
    corpus.write_text(
        json.dumps(
            {
                "build_id": "b" * 64,
                "ordered_stream_sha256": "c" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    calls = []

    def admit(**kwargs):
        calls.append(kwargs)
        return lifecycle

    monkeypatch.setattr(
        module,
        "admit_provider_lifecycle",
        admit,
        raising=False,
    )
    manifest = module.build_authenticated_launcher_manifest(
        out=scratch / "staging" / "launch.json",
        scratch_root=scratch,
        seed=0,
        release_sha256="d" * 64,
        release_members_sha256="e" * 64,
        release_receipt_sha256="f" * 64,
        run_manifest_sha256="0" * 64,
        cohort_assignment_sha256="1" * 64,
        code_commit="2" * 40,
        source_tree="3" * 40,
        bootstrap_receipt=bootstrap,
        corpus_receipt=corpus,
        runs=[
            {
                "arm": arm,
                "config": f"configs/360m-v3/{arm}-s0.yaml",
                "config_sha256": ("4" if arm == "dense" else "5") * 64,
            }
            for arm in ("dense", "split90")
        ],
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=tmp_path / "runtime-lock.json",
        runtime_evidence_path=tmp_path / "runtime-evidence.json",
        runtime_sbom_path=tmp_path / "runtime-sbom.json",
        objective_controls_amendment_path=(
            ROOT / "configs/objective-controls-amendment-v3.yaml"
        ),
        store=object(),
        account_id=lifecycle.binding.account_id,
        instance_id=lifecycle.binding.instance_id,
        boot_id=lifecycle.binding.boot_id,
        expected_selection_version_id=(
            lifecycle.binding.provider_selection_version_id
        ),
        identity_verifier=object(),
        approval_verifier=object(),
        trusted_public_key_sha256=(
            lifecycle.binding.qualification_approval_public_key_sha256
        ),
    )

    assert manifest["schema_version"] == 3
    assert manifest["provider"] == profile.provider
    assert manifest["profile_id"] == profile.profile_id
    assert all(
        manifest[field] == value
        for field, value in lifecycle.binding.to_dict().items()
    )
    assert len(calls) == 1
    parameters = set(
        inspect.signature(
            module.build_authenticated_launcher_manifest
        ).parameters
    )
    assert "binding" not in parameters
    assert "profile" not in parameters
    assert "provider" not in parameters


def test_controller_constructor_reenters_selected_authority_and_binds_intent(
    tmp_path,
    monkeypatch,
):
    import msctl.aws_p5 as module
    from tests.provider_lifecycle_fixtures import provider_lifecycle

    profile = load_aws_gpu_profile(P6_PROFILE)
    lifecycle = provider_lifecycle(profile)
    calls = []

    def admit(**kwargs):
        calls.append(kwargs)
        return lifecycle

    monkeypatch.setattr(
        module,
        "admit_provider_lifecycle",
        admit,
        raising=False,
    )
    environment = {
        "AWS_REGION": "us-east-1",
        "LANG": "C",
        "LC_ALL": "C",
        "MS_S3_ROOT": "s3://memorysplit-prod/confirmatory-v3",
        "MS_AWS_AMI_ID": "ami-0260c4d597dcc8641",
        "MS_CONTAINER_DIGEST": "sha256:" + "b" * 64,
        "MS_CONTAINER_IMAGE": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "memorysplit/aws-gpu@sha256:"
            + "b" * 64
        ),
        "MS_RUNTIME_GID": "10001",
        "MS_RUNTIME_UID": "10001",
    }
    backend = module.AwsP5Backend.from_authenticated_selection(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=tmp_path / "runtime-lock.json",
        runtime_evidence_path=tmp_path / "runtime-evidence.json",
        runtime_sbom_path=tmp_path / "runtime-sbom.json",
        objective_controls_amendment_path=(
            ROOT / "configs/objective-controls-amendment-v3.yaml"
        ),
        selection_store=object(),
        account_id=lifecycle.binding.account_id,
        instance_id=lifecycle.binding.instance_id,
        boot_id=lifecycle.binding.boot_id,
        seed=0,
        expected_selection_version_id=(
            lifecycle.binding.provider_selection_version_id
        ),
        selection_identity_verifier=object(),
        qualification_approval_verifier=object(),
        trusted_qualification_public_key_sha256=(
            lifecycle.binding.qualification_approval_public_key_sha256
        ),
        runtime_environment=environment,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/"
            "MemorySplitSelected"
        ),
        state_root=tmp_path / "state",
    )

    assert backend.profile == profile
    assert backend.lifecycle_binding == lifecycle.binding
    assert len(calls) == 1
    parameters = set(
        inspect.signature(
            module.AwsP5Backend.from_authenticated_selection
        ).parameters
    )
    assert "binding" not in parameters
    assert "profile" not in parameters
    assert "provider" not in parameters

    runs = tuple(
        SimpleNamespace(
            run_id=f"memorysplit-v3-360m-s0-{arm}",
            arm=arm,
            seed=0,
            config=f"configs/360m-v3/{arm}-s0.yaml",
            config_sha256=("b" if arm == "dense" else "c") * 64,
        )
        for arm in ("dense", "split90")
    )
    manifest = SimpleNamespace(
        **lifecycle.binding.to_dict(),
        schema_version=3,
        release_sha256="d" * 64,
        release_receipt_sha256="e" * 64,
        dataset_pointer_sha256="f" * 64,
        dataset_receipt_sha256="0" * 64,
        dataset_build_id="1" * 64,
        ordered_stream_sha256="2" * 64,
        cohort_assignment_sha256="3" * 64,
        preregistration_sha256="4" * 64,
        sealed_evaluation_release_sha256="5" * 64,
        source_commit="6" * 40,
        source_tree="7" * 40,
        sha256="8" * 64,
        runs=runs,
    )
    release = SimpleNamespace(
        provider=profile.provider,
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="9" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )
    evidence = {
        "dataset_pointer_sha256": manifest.dataset_pointer_sha256,
        "dataset_verification_sha256": manifest.dataset_receipt_sha256,
        "environment_receipt_sha256": (
            manifest.qualification_environment_receipt_sha256
        ),
        "instance_id": lifecycle.binding.instance_id,
        "boot_id": lifecycle.binding.boot_id,
    }
    backend._validate_manifest(manifest)
    assert len(calls) == 2
    core = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at="2026-07-25T00:00:00Z",
        evidence=evidence,
        attempt=1,
        bootstrap_mode="bootstrap",
    )
    assert core["schema_version"] == 3
    assert core["provider_selection_sha256"] == (
        lifecycle.binding.provider_selection_sha256
    )
    assert core["environment"]["MS_PROVIDER"] == profile.provider
    assert core["environment"]["MS_RUNTIME_SBOM_SHA256"] == "9" * 64
    assert core["bootstrap"] == {
        "mode": "bootstrap",
        "receipt_sha256": None,
    }
    assert core["lease_unit"].startswith("memorysplit-auto-terminate-")

    envelope = backend._operation_envelope(
        core,
        instance_id=lifecycle.binding.instance_id,
        terminate_at="2026-07-25T00:00:00Z",
    )
    states = [
        backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id=lifecycle.binding.instance_id,
            terminate_at="2026-07-25T00:00:00Z",
            intent=envelope,
            published={
                "intent_sha256": "a" * 64,
                "intent_uri": (
                    "s3://memorysplit-prod/confirmatory-v3/operations/"
                    "intents/sha256/" + "a" * 64 + ".json"
                ),
            },
            attempt=1,
            bootstrap_mode="bootstrap",
        )
        for run in runs
    ]
    from msctl.state import StateStore

    state_store = StateStore(tmp_path / "state")
    with state_store.locked():
        backend._write_paired_states(state_store, manifest, states)
        stored = [state_store.read_run(run.run_id) for run in runs]
    assert all(
        row["provider_selection_sha256"]
        == lifecycle.binding.provider_selection_sha256
        for row in stored
    )


def test_selected_launcher_consumes_authority_and_passes_operational_metadata(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.p5.launch_seed_pair as module
    from tests.provider_lifecycle_fixtures import provider_lifecycle
    from tests.test_aws_p5_launcher import (
        BOOT_ID,
        H100_NAMES,
        SAFE_ENVIRONMENT,
        _launcher_fixture,
        _rewrite_release_fixture,
        _write_json,
    )

    fixture = _launcher_fixture(tmp_path, seed=0)
    profile = load_aws_gpu_profile(P5_PROFILE)
    lifecycle = provider_lifecycle(profile)
    fixture["manifest"].update(
        {
            **lifecycle.binding.to_dict(),
            "schema_version": 3,
            "dataset_receipt_sha256": fixture["manifest"][
                "corpus_receipt"
            ]["sha256"],
            "environment_receipt_sha256": (
                lifecycle.binding.qualification_environment_receipt_sha256
            ),
            "release_receipt_sha256": "b" * 64,
            "run_manifest_sha256": "c" * 64,
            "source_tree": "5" * 40,
        }
    )

    def add_lifecycle(metadata):
        metadata.update(lifecycle.binding.to_dict())
        metadata["profile_id"] = lifecycle.binding.profile_id

    _rewrite_release_fixture(
        fixture,
        mutate_metadata=add_lifecycle,
    )
    selected_bootstrap = {
        "receipt_type": "memorysplit-aws-gpu-bootstrap-v1",
        "container_image": SAFE_ENVIRONMENT["MS_CONTAINER_IMAGE"],
    }
    _write_json(fixture["bootstrap_path"], selected_bootstrap)
    fixture["manifest"]["bootstrap_receipt"]["sha256"] = hashlib.sha256(
        fixture["bootstrap_path"].read_bytes()
    ).hexdigest()
    _write_json(fixture["manifest_path"], fixture["manifest"])
    monkeypatch.setattr(
        module,
        "admit_provider_lifecycle",
        lambda **kwargs: lifecycle,
    )
    monkeypatch.setattr(
        module,
        "_parse_selected_bootstrap_receipt_bytes",
        lambda *args, **kwargs: selected_bootstrap,
    )
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_sbom = tmp_path / "runtime-sbom.json"
    runtime_lock.write_bytes(b"runtime-lock\n")
    runtime_sbom.write_bytes(b"runtime-sbom\n")

    plan = module.load_authenticated_launch_plan(
        seed=0,
        manifest_path=fixture["manifest_path"],
        repo_root=fixture["repo_root"],
        scratch_root=fixture["scratch_root"],
        environment=SAFE_ENVIRONMENT,
        authority_root=tmp_path / "authority",
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=tmp_path / "runtime-evidence.json",
        runtime_sbom_path=runtime_sbom,
        objective_controls_amendment_path=(
            fixture["repo_root"]
            / "configs/objective-controls-amendment-v3.yaml"
        ),
        store=object(),
        account_id=lifecycle.binding.account_id,
        instance_id=lifecycle.binding.instance_id,
        boot_id=lifecycle.binding.boot_id,
        expected_selection_version_id=(
            lifecycle.binding.provider_selection_version_id
        ),
        identity_verifier=object(),
        approval_verifier=object(),
        trusted_public_key_sha256=(
            lifecycle.binding.qualification_approval_public_key_sha256
        ),
        observed_instance_type="p5.48xlarge",
        observed_instance_id=lifecycle.binding.instance_id,
        observed_boot_id=BOOT_ID,
        gpu_names=H100_NAMES,
        port_available=lambda _port: True,
        semantic_corpus_verifier=lambda _root: fixture["corpus"],
        enforce_profile_scratch=False,
    )

    assert plan.lifecycle_binding == lifecycle.binding
    expected_environment = "MS_PROVIDER_SELECTION_SHA256=" + "2" * 64
    for arm in plan.arms:
        metadata = arm.runtime_config["operational_metadata"]
        assert metadata["provider_selection_sha256"] == "2" * 64
        assert metadata["arm"] == arm.arm
        assert expected_environment in arm.argv
