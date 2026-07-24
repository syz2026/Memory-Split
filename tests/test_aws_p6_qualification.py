from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.aws_hardware import AuthenticatedSelectionBinding


ROOT = Path(__file__).resolve().parents[1]
P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
DENSE_CONFIG = ROOT / "configs" / "360m-v3" / "dense-s0.yaml"
SPLIT90_CONFIG = ROOT / "configs" / "360m-v3" / "split90-s0.yaml"
QUALIFICATION_WORKER = ROOT / "cluster" / "aws" / "qualification_worker.py"
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit/aws-gpu"
    f"@{IMAGE_DIGEST}"
)
AMI_ID = "ami-0260c4d597dcc8641"
AMI_OWNER_ID = "898082745236"
ACCOUNT_ID = "123456789012"
INSTANCE_ID = "i-0123456789abcdef0"
BOOT_ID = "12345678-1234-4abc-8def-1234567890ab"
CONTAINER_FACTS = {
    "python": "3.12.11",
    "pytorch": "2.9.0+cu130",
    "cuda": "13.0",
    "cudnn": "9.10.2",
    "nccl": "2.28.3",
}
HOST_FACTS = {
    "cuda": "13.2",
    "nvidia_driver": "595.71.05",
    "fabric_manager": "595.71.05",
    "docker": "28.5.1",
    "nvidia_container_runtime": "1.18.0",
    "aws_cli": "2.31.7",
    "kernel": "6.17",
    "efa": "1.47.0",
    "ofi_nccl": "1.18.0",
    "nvlsm": "595.71.05",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _runtime_inputs(profile):
    lock = {
        "schema_version": 1,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "control_bundle_sha256": "d" * 64,
        "profile_sha256": profile.sha256,
        "ami_id": AMI_ID,
        "ami_owner_id": AMI_OWNER_ID,
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "versions": {
            **CONTAINER_FACTS,
            "nvidia_driver": HOST_FACTS["nvidia_driver"],
            "fabric_manager": HOST_FACTS["fabric_manager"],
            "docker": HOST_FACTS["docker"],
            "nvidia_container_runtime": HOST_FACTS[
                "nvidia_container_runtime"
            ],
            "aws_cli": HOST_FACTS["aws_cli"],
        },
    }
    lock_data = _canonical(lock)
    sbom = {
        "schema_version": 2,
        "document_type": "memorysplit-aws-gpu-sbom-v2",
        "source": {
            "commit": lock["source_commit"],
            "tree": lock["source_tree"],
        },
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "image_binding_sha256": "e" * 64,
        "project_dependency_lock": {
            "sha256": "f" * 64,
            "packages": [],
        },
        "host": {
            "ami_id": AMI_ID,
            "ami_owner_id": AMI_OWNER_ID,
            "ami_name": (
                "Deep Learning Base AMI with Single CUDA Ubuntu 24.04 "
                "20260523"
            ),
            "architecture": "x86_64",
            "versions": dict(HOST_FACTS),
            "minimum_versions": {
                "cuda": "13.0",
                "nvidia_driver": "580.0",
                "kernel": "6.1",
                "efa": "1.44.0",
                "ofi_nccl": "1.17.1",
            },
        },
        "container": {
            "image": IMAGE,
            "image_digest": IMAGE_DIGEST,
            "versions": dict(CONTAINER_FACTS),
        },
    }
    return lock_data, _canonical(sbom)


def _closed_runtime_inputs(profile):
    path = ROOT / "tests" / "test_aws_gpu_runtime.py"
    spec = importlib.util.spec_from_file_location(
        f"qualification_runtime_fixture_{uuid.uuid4().hex}",
        path,
    )
    assert spec is not None and spec.loader is not None
    fixtures = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixtures
    spec.loader.exec_module(fixtures)
    runtime = fixtures._load_script(fixtures.RUNTIME_LOCK_SCRIPT)
    runtime_spec = fixtures._runtime_spec()
    runtime_spec["profile_sha256"] = profile.sha256
    binding = fixtures._image_binding()
    binding["container_image"] = IMAGE
    binding["container_image_digest"] = IMAGE_DIGEST
    binding["build_inputs"]["qualification_worker_sha256"] = hashlib.sha256(
        QUALIFICATION_WORKER.read_bytes()
    ).hexdigest()
    artifacts = runtime.produce_runtime_artifacts(
        spec_bytes=_canonical(runtime_spec),
        host_candidate_bytes=fixtures.HOST_CANDIDATE.read_bytes(),
        dependency_lock_bytes=fixtures.REQUIREMENTS_LOCK.read_bytes(),
        image_binding_bytes=_canonical(binding),
    )
    return artifacts.runtime_lock_bytes, artifacts.sbom_bytes


def _binding(profile, lock_data):
    return AuthenticatedSelectionBinding(
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        amendment_sha256="1" * 64,
        selection_sha256="2" * 64,
        selection_version_id="selection-version-1",
        profile_id=profile.profile_id,
        provider=profile.provider,
        profile_sha256=profile.sha256,
        runtime_lock_sha256=hashlib.sha256(lock_data).hexdigest(),
        qualification_evidence_sha256="3" * 64,
        environment_receipt_sha256="4" * 64,
        canary_receipt_sha256="5" * 64,
        approval_receipt_sha256="6" * 64,
        approval_public_key_sha256="7" * 64,
        account_id=ACCOUNT_ID,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        region="us-east-1",
        availability_zone="us-east-1d",
        purchase_model="on_demand",
        seed=0,
        arm="dense",
    )


def _cohort_bindings(profile, lock_data):
    dense = _binding(profile, lock_data)
    return dense, replace(dense, arm="split90")


def _cohort_authority(profile, lock_data):
    from cluster.aws.qualification import CohortSelectionAuthority

    dense, split90 = _cohort_bindings(profile, lock_data)
    return CohortSelectionAuthority(
        profile=profile,
        seed=0,
        arms=("dense", "split90"),
        bindings={"dense": dense, "split90": split90},
    )


def test_cohort_authority_reenters_admission_for_both_arms(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, _ = _runtime_inputs(profile)
    dense, split90 = _cohort_bindings(profile, lock_data)
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_evidence = tmp_path / "runtime-evidence.json"
    runtime_lock.write_bytes(lock_data)
    runtime_evidence.write_bytes(_canonical({"evidence": "fixture"}))
    calls = []

    def admit(**kwargs):
        calls.append(kwargs)
        return {"dense": dense, "split90": split90}[kwargs["arm"]]

    monkeypatch.setattr(module, "admit_provider_selection", admit, raising=False)
    authority = module.admit_cohort_provider_selection(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        store=object(),
        account_id=ACCOUNT_ID,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        seed=0,
        expected_selection_version_id="selection-version-1",
        identity_verifier=lambda identity, pkcs7, region: True,
        approval_verifier=object(),
        trusted_public_key_sha256="7" * 64,
    )

    assert [call["arm"] for call in calls] == ["dense", "split90"]
    assert all(call["seed"] == 0 for call in calls)
    assert authority.profile == profile
    assert authority.seed == 0
    assert authority.arms == ("dense", "split90")
    assert authority.bindings == {"dense": dense, "split90": split90}
    fields = module._selection_fields(authority)
    assert fields["seed"] == 0
    assert fields["arms"] == ["dense", "split90"]


def test_cohort_authority_rejects_arm_specific_selection_drift(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, _ = _runtime_inputs(profile)
    dense, split90 = _cohort_bindings(profile, lock_data)
    split90 = replace(split90, selection_sha256="8" * 64)
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_evidence = tmp_path / "runtime-evidence.json"
    runtime_lock.write_bytes(lock_data)
    runtime_evidence.write_bytes(_canonical({"evidence": "fixture"}))

    monkeypatch.setattr(
        module,
        "admit_provider_selection",
        lambda **kwargs: {
            "dense": dense,
            "split90": split90,
        }[kwargs["arm"]],
        raising=False,
    )
    with pytest.raises(ValueError, match="arm|cohort|selection"):
        module.admit_cohort_provider_selection(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=object(),
            account_id=ACCOUNT_ID,
            instance_id=INSTANCE_ID,
            boot_id=BOOT_ID,
            seed=0,
            expected_selection_version_id="selection-version-1",
            identity_verifier=lambda identity, pkcs7, region: True,
            approval_verifier=object(),
            trusted_public_key_sha256="7" * 64,
        )


def test_public_selected_apis_require_authority_paths_not_binding_objects():
    import cluster.aws.p5.attest_environment as attestation
    import cluster.aws.p5.bootstrap as bootstrap
    import cluster.aws.qualification as qualification

    functions = (
        attestation.attest_authenticated_gpu_environment,
        bootstrap.bootstrap_authenticated_gpu_environment,
        qualification.run_authenticated_selected_canary,
    )
    authority_parameters = {
        "authority_root",
        "repo_root",
        "runtime_lock_path",
        "runtime_evidence_path",
        "store",
        "account_id",
        "instance_id",
        "boot_id",
        "seed",
        "expected_selection_version_id",
        "identity_verifier",
        "approval_verifier",
        "trusted_public_key_sha256",
    }
    for function in functions:
        parameters = set(inspect.signature(function).parameters)
        assert "selection_binding" not in parameters
        assert "selected_profile" not in parameters
        assert authority_parameters <= parameters

    unsafe_exports = {
        "build_selected_environment_receipt",
        "build_selected_bootstrap_receipt",
        "validate_authenticated_selection",
        "build_selected_canary_plan",
        "execute_selected_canary",
        "render_selected_canary_plan",
    }
    assert not unsafe_exports & set(qualification.__all__)


def test_forged_binding_builders_are_not_public_module_apis():
    import cluster.aws.p5.attest_environment as attestation
    import cluster.aws.p5.bootstrap as bootstrap
    import cluster.aws.qualification as qualification

    assert not hasattr(attestation, "build_selected_environment_receipt")
    assert not hasattr(bootstrap, "build_selected_bootstrap_receipt")
    assert not hasattr(qualification, "build_selected_environment_receipt")
    assert not hasattr(qualification, "build_selected_bootstrap_receipt")
    assert not hasattr(qualification, "validate_authenticated_selection")


def test_internal_receipts_bind_both_authenticated_arm_scopes():
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    receipt = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    assert receipt["seed"] == 0
    assert receipt["arms"] == ["dense", "split90"]
    assert receipt["provider_selection_sha256"] == (
        authority.bindings["dense"].selection_sha256
    )

    forged = replace(
        authority,
        bindings={
            "dense": authority.bindings["dense"],
            "split90": replace(
                authority.bindings["split90"],
                seed=1,
            ),
        },
    )
    with pytest.raises(ValueError, match="arm|seed|cohort|authority"):
        module._build_selected_environment_receipt(
            selection_authority=forged,
            runtime_lock_data=lock_data,
            runtime_sbom_data=sbom_data,
            attestation_evidence=_attestation(profile, lock_data),
        )


def test_internal_bootstrap_and_canary_keep_cohort_arm_authority(tmp_path):
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    environment = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    bootstrap = module._build_selected_bootstrap_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )
    scratch = tmp_path / "authority-canary"
    scratch.mkdir(mode=0o700)
    plan = module._build_selected_canary_plan(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        bootstrap_receipt=bootstrap,
        qualification_root=ROOT,
        scratch_root=scratch,
        output_path=scratch / "qualification.json",
        s3_root="s3://memorysplit-prod/confirmatory-v3",
    )

    assert bootstrap["seed"] == 0
    assert bootstrap["arms"] == ["dense", "split90"]
    assert plan.authority == authority


@pytest.mark.parametrize(
    "mutation",
    [
        "empty-dependencies",
        "missing-os-inventory",
        "missing-python",
        "missing-nvlsm",
        "missing-worker-hash",
        "wrong-worker-hash",
        "open-container",
        "empty-installed-packages",
    ],
)
def test_runtime_bundle_requires_full_reviewed_sbom_closure(mutation):
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    sbom = json.loads(sbom_data)
    if mutation == "empty-dependencies":
        sbom["project_dependency_lock"]["packages"] = []
    elif mutation == "missing-os-inventory":
        sbom["container"].pop("os_packages")
    elif mutation == "missing-python":
        sbom["container"].pop("python")
    elif mutation == "missing-nvlsm":
        sbom["host"]["versions"].pop("nvlsm", None)
    elif mutation == "missing-worker-hash":
        sbom["container"]["build_inputs"].pop(
            "qualification_worker_sha256",
            None,
        )
    elif mutation == "wrong-worker-hash":
        sbom["container"]["build_inputs"]["qualification_worker_sha256"] = (
            "0" * 64
        )
    elif mutation == "open-container":
        sbom["container"]["unexpected"] = True
    elif mutation == "empty-installed-packages":
        sbom["container"]["installed_python_packages"] = []
        sbom["container"]["installed_distribution_count"] = 0
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="SBOM|provenance|worker|NVLSM|inventory"):
        module._load_runtime_qualification_bundle(
            selection_authority=authority,
            runtime_lock_data=lock_data,
            runtime_sbom_data=_canonical(sbom),
        )


def _attestation(profile, lock_data):
    lock = json.loads(lock_data)
    return {
        "schema_version": 1,
        "evidence_type": "memorysplit-aws-gpu-attestation-v1",
        "profile_id": profile.profile_id,
        "provider": profile.provider,
        "profile_sha256": profile.sha256,
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "control_bundle_sha256": lock["control_bundle_sha256"],
        "source_commit": lock["source_commit"],
        "source_tree": lock["source_tree"],
        "container_image": lock["container_image"],
        "container_image_digest": lock["container_image_digest"],
        "ami_id": lock["ami_id"],
        "ami_owner_id": lock["ami_owner_id"],
        "instance_type": profile.instance_type,
        "gpu_model": profile.gpu_model,
        "gpu_count": profile.allocated_gpus,
        "host_facts": dict(HOST_FACTS),
        "container_facts": {
            field: lock["versions"][field]
            for field in ("python", "pytorch", "cuda", "cudnn", "nccl")
        },
        "aws_instance_identity_document": {
            "accountId": ACCOUNT_ID,
            "architecture": "x86_64",
            "imageId": lock["ami_id"],
            "instanceId": INSTANCE_ID,
            "instanceType": profile.instance_type,
            "privateIp": "10.23.45.67",
            "region": "us-east-1",
        },
        "aws_instance_identity_pkcs7": "c3ludGhldGlj",
        "account_id": ACCOUNT_ID,
        "instance_id": INSTANCE_ID,
        "region": "us-east-1",
        "boot_id": BOOT_ID,
    }


def _hardware(profile):
    return {
        "account_id": ACCOUNT_ID,
        "instance_id": INSTANCE_ID,
        "boot_id": BOOT_ID,
        "ami_id": AMI_ID,
        "instance_type": profile.instance_type,
        "architecture": profile.architecture,
        "vcpus": profile.vcpus,
        "memory_gib": profile.memory_gib,
        "gpu_names": [profile.gpu_model] * profile.allocated_gpus,
        "instance_store": [
            {
                "path": f"/dev/nvme{index}n1",
                "model": profile.instance_store_model,
                "bytes": profile.instance_store_device_bytes,
            }
            for index in range(profile.instance_store_devices)
        ],
    }


def test_selected_environment_receipt_binds_authenticated_p6_authority():
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    binding = authority.bindings["dense"]

    receipt = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    assert receipt["receipt_type"] == (
        "memorysplit-aws-gpu-environment-selection-v1"
    )
    assert receipt["hardware_amendment_sha256"] == binding.amendment_sha256
    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert (
        receipt["provider_selection_version_id"]
        == binding.selection_version_id
    )
    assert receipt["profile_id"] == "aws-p6-b300.48xlarge-v3"
    assert receipt["runtime_lock_sha256"] == binding.runtime_lock_sha256
    assert receipt["runtime_sbom_sha256"] == hashlib.sha256(sbom_data).hexdigest()
    assert receipt["instance_id"] == binding.instance_id
    assert receipt["ami_id"] == AMI_ID
    assert receipt["container_image"] == IMAGE
    assert receipt["host_facts"] == HOST_FACTS
    assert receipt["container_facts"] == CONTAINER_FACTS
    assert receipt["seed"] == 0
    assert receipt["arms"] == ["dense", "split90"]


def test_authenticated_environment_attestation_orchestrates_selected_measurement(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.p5.attest_environment as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    binding = authority.bindings["dense"]
    lock = tmp_path / "runtime-lock.json"
    runtime_evidence = tmp_path / "runtime-evidence.json"
    sbom = tmp_path / "runtime-sbom.json"
    control = tmp_path / "control.zip"
    lock.write_bytes(lock_data)
    runtime_evidence.write_bytes(_canonical({"evidence": "fixture"}))
    sbom.write_bytes(sbom_data)
    control.write_bytes(b"control")
    calls = []

    def attest(**kwargs):
        calls.append(kwargs)
        return _attestation(profile, lock_data)

    monkeypatch.setattr(module, "attest_selected_gpu_environment", attest)
    monkeypatch.setattr(
        module,
        "admit_cohort_provider_selection",
        lambda **kwargs: authority,
    )
    output = tmp_path / "selected-environment.json"
    receipt = module.attest_authenticated_gpu_environment(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=lock,
        runtime_evidence_path=runtime_evidence,
        runtime_sbom_path=sbom,
        store=object(),
        account_id=ACCOUNT_ID,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        seed=0,
        expected_selection_version_id="selection-version-1",
        identity_verifier=object(),
        approval_verifier=object(),
        trusted_public_key_sha256="7" * 64,
        control_bundle_path=control,
        output_path=output,
        apply=False,
        imds_reader=object(),
        command_reader=object(),
        boot_id_path=tmp_path / "boot-id",
    )

    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert len(calls) == 1
    assert calls[0]["selected_profile"] is profile
    assert calls[0]["apply"] is False
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_type", "p5.48xlarge"),
        ("vcpus", 191),
        ("memory_gib", 2048),
        ("gpu_names", ["NVIDIA B200"] * 8),
        ("gpu_names", ["NVIDIA B300"] * 7),
        ("instance_store", []),
    ],
)
def test_selected_bootstrap_requires_exact_profile_hardware(field, value):
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    environment = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    hardware = _hardware(profile)
    hardware[field] = value

    with pytest.raises(ValueError, match="hardware|profile|GPU|memory|vCPU|NVMe"):
        module._build_selected_bootstrap_receipt(
            selection_authority=authority,
            runtime_lock_data=lock_data,
            runtime_sbom_data=sbom_data,
            environment_receipt=environment,
            hardware_evidence=hardware,
        )


def test_selected_bootstrap_receipt_binds_profile_runtime_and_selection():
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    binding = authority.bindings["dense"]
    environment = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    receipt = module._build_selected_bootstrap_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )

    assert receipt["receipt_type"] == "memorysplit-aws-gpu-bootstrap-v1"
    assert receipt["provider"] == profile.provider
    assert receipt["profile_sha256"] == profile.sha256
    assert receipt["runtime_lock_sha256"] == binding.runtime_lock_sha256
    assert receipt["runtime_sbom_sha256"] == hashlib.sha256(sbom_data).hexdigest()
    assert receipt["environment_receipt_sha256"] == hashlib.sha256(
        _canonical(environment)
    ).hexdigest()
    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert receipt["provider_selection_version_id"] == (
        binding.selection_version_id
    )
    assert receipt["hardware"]["vcpus"] == 192
    assert receipt["hardware"]["memory_gib"] == 4096
    assert receipt["hardware"]["gpu_count"] == 8
    assert receipt["hardware"]["instance_store_devices"] == 8
    assert receipt["seed"] == 0
    assert receipt["arms"] == ["dense", "split90"]


def test_selected_environment_and_bootstrap_receipts_are_closed_and_reparsed():
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    environment = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    environment_data = module._canonical_selected_environment_receipt(environment)
    assert module._parse_selected_environment_receipt_bytes(
        environment_data,
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
    ) == environment
    bootstrap = module._build_selected_bootstrap_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )
    bootstrap_data = module._canonical_selected_bootstrap_receipt(bootstrap)
    assert module._parse_selected_bootstrap_receipt_bytes(
        bootstrap_data,
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt_sha256=hashlib.sha256(environment_data).hexdigest(),
    ) == bootstrap

    open_environment = dict(environment)
    open_environment["unexpected"] = True
    with pytest.raises(ValueError, match="field|schema|closed"):
        module._canonical_selected_environment_receipt(open_environment)
    forged_bootstrap = json.loads(bootstrap_data)
    forged_bootstrap["hardware"]["memory_gib"] = 2048
    with pytest.raises(ValueError, match="hardware|profile|memory"):
        module._parse_selected_bootstrap_receipt_bytes(
            _canonical(forged_bootstrap),
            selection_authority=authority,
            runtime_lock_data=lock_data,
            runtime_sbom_data=sbom_data,
            environment_receipt_sha256=hashlib.sha256(
                environment_data
            ).hexdigest(),
        )


def test_authenticated_bootstrap_uses_injected_profile_hardware_reader(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.p5.bootstrap as module
    import cluster.aws.qualification as qualification

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    binding = authority.bindings["dense"]
    environment = qualification._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    class Reader:
        def __init__(self):
            self.calls = []

        def measure(self, *, selected_profile, selection_binding):
            self.calls.append((selected_profile, selection_binding))
            return _hardware(selected_profile)

    reader = Reader()
    lock_path = tmp_path / "runtime-lock.json"
    evidence_path = tmp_path / "runtime-evidence.json"
    sbom_path = tmp_path / "runtime-sbom.json"
    environment_path = tmp_path / "environment.json"
    lock_path.write_bytes(lock_data)
    evidence_path.write_bytes(_canonical({"evidence": "fixture"}))
    sbom_path.write_bytes(sbom_data)
    environment_path.write_bytes(
        qualification._canonical_selected_environment_receipt(environment)
    )
    monkeypatch.setattr(
        module,
        "admit_cohort_provider_selection",
        lambda **kwargs: authority,
    )
    receipt = module.bootstrap_authenticated_gpu_environment(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=lock_path,
        runtime_evidence_path=evidence_path,
        runtime_sbom_path=sbom_path,
        environment_receipt_path=environment_path,
        store=object(),
        account_id=ACCOUNT_ID,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        seed=0,
        expected_selection_version_id="selection-version-1",
        identity_verifier=object(),
        approval_verifier=object(),
        trusted_public_key_sha256="7" * 64,
        hardware_reader=reader,
    )

    assert reader.calls == [(profile, binding)]
    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert receipt["hardware"]["instance_type"] == "p6-b300.48xlarge"


def test_p5_and_p6_authenticated_bindings_cannot_cross_authorize():
    import cluster.aws.qualification as module

    p5 = load_aws_gpu_profile(P5_PROFILE)
    p6 = load_aws_gpu_profile(P6_PROFILE)
    p5_lock, _ = _closed_runtime_inputs(p5)
    p6_lock, _ = _closed_runtime_inputs(p6)

    with pytest.raises(ValueError, match="profile|selection|binding"):
        module._validate_cohort_authority(
            replace(
                _cohort_authority(p5, p5_lock),
                profile=p6,
            )
        )
    with pytest.raises(ValueError, match="profile|selection|binding"):
        module._validate_cohort_authority(
            replace(
                _cohort_authority(p6, p6_lock),
                profile=p5,
            )
        )


def _selected_case(tmp_path):
    import cluster.aws.qualification as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _closed_runtime_inputs(profile)
    authority = _cohort_authority(profile, lock_data)
    binding = authority.bindings["dense"]
    environment = module._build_selected_environment_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    bootstrap = module._build_selected_bootstrap_receipt(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )
    scratch = tmp_path / "selected-canary"
    scratch.mkdir(mode=0o700)
    plan = module._build_selected_canary_plan(
        selection_authority=authority,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        bootstrap_receipt=bootstrap,
        qualification_root=ROOT,
        scratch_root=scratch,
        output_path=scratch / "qualification.json",
        s3_root="s3://memorysplit-prod/confirmatory-v3",
    )
    return {
        "profile": profile,
        "authority": authority,
        "lock_data": lock_data,
        "sbom_data": sbom_data,
        "binding": binding,
        "environment": environment,
        "bootstrap": bootstrap,
        "scratch": scratch,
        "plan": plan,
    }


def _canary_hardware(profile):
    return {
        **_hardware(profile),
        "gpu_devices": [
            {
                "index": index,
                "name": profile.gpu_model,
                "memory_mib": 294_912,
            }
            for index in range(8)
        ],
        "host_facts": dict(HOST_FACTS),
        "topology": {
            "matrix_sha256": "8" * 64,
            "nvlink_active": True,
            "fully_connected": True,
        },
    }


def _container_capabilities():
    return {
        "versions": dict(CONTAINER_FACTS),
        "bf16": True,
        "sdpa_forward": True,
        "sdpa_backward": True,
        "torch_compile": True,
        "fused_adamw": True,
        "one_step_train": True,
        "checkpoint_resume_exact": True,
        "checkpoint_sha256": "9" * 64,
        "resumed_checkpoint_sha256": "9" * 64,
    }


def _group_result(plan, arm):
    group = plan.gpu_groups[arm]
    affinity = plan.cpu_affinities[arm]
    checkpoint = "a" * 64 if arm == "dense" else "b" * 64
    step_seconds = 5.0 if arm == "dense" else 6.0
    checkpoint_step = 100
    tokens_per_step = 524_288
    return {
        "arm": arm,
        "world_size": 4,
        "gpu_ids": list(group),
        "cpu_affinity": list(affinity),
        "master_port": plan.master_ports[arm],
        "all_reduce_sum": 10.0,
        "all_reduce_latency_seconds": 0.004,
        "updates": 100,
        "warmup_updates": 10,
        "step_seconds": [step_seconds] * 100,
        "step_tokens": [tokens_per_step] * 100,
        "peak_memory_bytes": (
            170_000_000_000 if arm == "dense" else 150_000_000_000
        ),
        "geometry": {
            "arm": arm,
            "model": "d360m",
            "ctx": 1024,
            "micro_batch_size": 8,
            "tokens_per_step": tokens_per_step,
            "total_tokens": 7_120_879_616,
            "max_steps": 13_582,
            "sidecar_name": (
                "dense_target_weights"
                if arm == "dense"
                else "split90_target_weights"
            ),
            "compile": True,
        },
        "sidecar": {
            "name": (
                "dense_target_weights"
                if arm == "dense"
                else "split90_target_weights"
            ),
            "stream_sha256": ("c" if arm == "dense" else "d") * 64,
            "items": tokens_per_step,
            "nonzero_targets": (
                tokens_per_step
                if arm == "dense"
                else int(tokens_per_step * 0.9)
            ),
            "target_weight_sum": (
                float(tokens_per_step)
                if arm == "dense"
                else float(int(tokens_per_step * 0.9))
            ),
        },
        "capabilities": {
            "bf16_output_finite": True,
            "sdpa_forward_finite": True,
            "sdpa_backward_finite": True,
            "compiled_model": True,
            "fused_adamw": True,
            "gradient_finite": True,
        },
        "resume": {
            "checkpoint_sha256": checkpoint,
            "checkpoint_step": checkpoint_step,
            "next_step": checkpoint_step + 1,
            "model_equivalent": True,
            "optimizer_equivalent": True,
            "loss_delta": 0.0,
            "cursor_before": checkpoint_step * tokens_per_step,
            "cursor_after": (checkpoint_step + 1) * tokens_per_step,
            "resumed_cursor_after": (checkpoint_step + 1) * tokens_per_step,
            "rng_restored": ["python", "numpy", "torch", "cuda"],
            "tolerance": 1e-6,
        },
    }


class _QualificationRunner:
    def __init__(self, module, plan, *, mutation=None):
        self.module = module
        self.plan = plan
        self.mutation = mutation
        self.calls = []
        self.pairs = []

    def run(self, spec):
        self.calls.append(spec)
        if spec.name == "hardware-inventory":
            value = _canary_hardware(self.plan.profile)
            if self.mutation == "host-floor":
                value["host_facts"]["efa"] = "1.43.9"
            elif self.mutation == "topology":
                value["topology"]["nvlink_active"] = False
            return self.module.QualificationCommandResult(
                0,
                _canonical(value).decode("ascii"),
                "",
            )
        if spec.name == "container-capabilities":
            value = _container_capabilities()
            if self.mutation == "sdpa":
                value["sdpa_backward"] = False
            elif self.mutation == "container-version":
                value["versions"]["cuda"] = "12.9"
            return self.module.QualificationCommandResult(
                0,
                _canonical(value).decode("ascii"),
                "",
            )
        raise AssertionError(f"unexpected command: {spec.name}")

    def run_pair(self, specs):
        self.pairs.append(tuple(specs))
        results = {}
        for spec in specs:
            arm = spec.arm
            value = _group_result(self.plan, arm)
            if self.mutation == "world-size" and arm == "dense":
                value["world_size"] = 8
            elif self.mutation == "gpu-overlap" and arm == "split90":
                value["gpu_ids"] = [0, 1, 2, 3]
            elif self.mutation == "checkpoint" and arm == "split90":
                value["resumed_checkpoint_sha256"] = "0" * 64
            results[spec.name] = self.module.QualificationCommandResult(
                0,
                _canonical(value).decode("ascii"),
                "",
            )
        return results


class _QualificationStore:
    def __init__(self, module):
        self.module = module
        self.objects = {}

    def put(self, uri, payload, *, sha256):
        self.objects[("version-1", uri)] = bytes(payload)
        return self.module.QualificationObjectWrite(
            sha256=sha256,
            bytes=len(payload),
            version_id="version-1",
        )

    def get(self, uri, *, version_id):
        payload = self.objects[(version_id, uri)]
        return self.module.QualificationObjectRead(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            version_id=version_id,
        )


class _QualificationTime:
    def __init__(self):
        self.origin = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
        self.tick = 0

    def now(self):
        value = self.origin + timedelta(seconds=self.tick)
        self.tick += 1
        return value

    def monotonic(self):
        value = float(self.tick)
        self.tick += 1
        return value


class _ProcessClock:
    def __init__(self):
        self.value = 100.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class _FakeQualificationProcess:
    def __init__(self, module, clock, spec, *, pid, duration, returncode=0):
        self.module = module
        self.clock = clock
        self.spec = spec
        self.pid = pid
        self.end = clock.value + duration
        self.returncode = returncode
        self.poll_calls = 0
        self.terminated = False

    def poll(self):
        self.poll_calls += 1
        if self.terminated:
            return -15
        return self.returncode if self.clock.value >= self.end else None

    def result(self):
        arm = self.spec.arm
        return self.module.QualificationCommandResult(
            self.returncode,
            _canonical(_group_result(self.spec.plan, arm)).decode("ascii")
            if hasattr(self.spec, "plan")
            else "",
            "" if self.returncode == 0 else "failed",
        )

    def terminate(self):
        self.terminated = True


class _FakeProcessLauncher:
    def __init__(
        self,
        module,
        plan,
        clock,
        durations,
        returncodes=None,
        mutation=None,
    ):
        self.module = module
        self.plan = plan
        self.clock = clock
        self.durations = durations
        self.returncodes = returncodes or {}
        self.mutation = mutation
        self.starts = []
        self.processes = {}

    def start(self, spec):
        self.starts.append(spec.name)
        process = _FakeQualificationProcess(
            self.module,
            self.clock,
            spec,
            pid=40_000 + len(self.starts),
            duration=self.durations[spec.arm],
            returncode=self.returncodes.get(spec.arm, 0),
        )
        def result(arm=spec.arm, code=process.returncode):
            value = _group_result(self.plan, arm)
            if self.mutation == "world-size" and arm == "dense":
                value["world_size"] = 8
            elif self.mutation == "gpu-overlap" and arm == "split90":
                value["gpu_ids"] = [0, 1, 2, 3]
            elif self.mutation == "checkpoint" and arm == "split90":
                value["resume"]["model_equivalent"] = False
            return self.module.QualificationCommandResult(
                code,
                _canonical(value).decode("ascii"),
                "" if code == 0 else "failed",
            )

        process.result = result
        self.processes[spec.name] = process
        self.clock.value += 0.01
        return process


def test_process_pair_starts_both_before_wait_and_measures_overlap(tmp_path):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    _, _, specs = module._selected_specs(plan)
    clock = _ProcessClock()
    launcher = _FakeProcessLauncher(
        module,
        plan,
        clock,
        {"dense": 2.0, "split90": 3.0},
    )

    results, evidence = module._run_simultaneous_group_processes(
        plan,
        specs,
        process_launcher=launcher,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert launcher.starts == ["dense-4rank", "split90-4rank"]
    assert set(results) == {"dense-4rank", "split90-4rank"}
    assert {row["arm"] for row in evidence} == {"dense", "split90"}
    assert all(row["pid"] > 0 for row in evidence)
    assert all(row["ended_monotonic"] > row["started_monotonic"] for row in evidence)
    assert max(row["started_monotonic"] for row in evidence) < min(
        row["ended_monotonic"] for row in evidence
    )
    assert {row["rendezvous"] for row in evidence} == {
        f"127.0.0.1:{plan.master_ports['dense']}",
        f"127.0.0.1:{plan.master_ports['split90']}",
    }
    assert all(process.poll_calls > 0 for process in launcher.processes.values())


@pytest.mark.parametrize(
    ("durations", "returncodes", "mutation"),
    [
        ({"dense": 0.0, "split90": 3.0}, {}, None),
        ({"dense": 2.0, "split90": 3.0}, {"dense": 17}, None),
        ({"dense": 2.0, "split90": 3.0}, {}, "gpu-overlap"),
        ({"dense": 2.0, "split90": 3.0}, {}, "cpu-overlap"),
        ({"dense": 2.0, "split90": 3.0}, {}, "port-overlap"),
    ],
)
def test_process_pair_rejects_sequential_early_or_overlapping_groups(
    tmp_path,
    durations,
    returncodes,
    mutation,
):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    _, _, specs = module._selected_specs(plan)
    if mutation == "gpu-overlap":
        specs = (specs[0], replace(specs[1], gpu_ids=specs[0].gpu_ids))
    elif mutation == "cpu-overlap":
        specs = (
            specs[0],
            replace(specs[1], cpu_affinity=specs[0].cpu_affinity),
        )
    elif mutation == "port-overlap":
        specs = (specs[0], replace(specs[1], master_port=specs[0].master_port))
    clock = _ProcessClock()
    launcher = _FakeProcessLauncher(
        module,
        plan,
        clock,
        durations,
        returncodes,
    )

    with pytest.raises(
        ValueError,
        match="simultaneous|overlap|early|process|GPU|CPU|port",
    ):
        module._run_simultaneous_group_processes(
            plan,
            specs,
            process_launcher=launcher,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_telemetry_derives_simultaneity_from_process_evidence(tmp_path):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    groups = {
        arm: module._validate_group_result(
            plan,
            _group_result(plan, arm),
            arm=arm,
        )
        for arm in ("dense", "split90")
    }
    evidence = [
        {
            "arm": "dense",
            "pid": 1,
            "started_monotonic": 10.0,
            "ended_monotonic": 20.0,
            "rendezvous": f"127.0.0.1:{plan.master_ports['dense']}",
            "gpu_ids": list(plan.gpu_groups["dense"]),
            "cpu_affinity": list(plan.cpu_affinities["dense"]),
            "returncode": 0,
        },
        {
            "arm": "split90",
            "pid": 2,
            "started_monotonic": 11.0,
            "ended_monotonic": 21.0,
            "rendezvous": f"127.0.0.1:{plan.master_ports['split90']}",
            "gpu_ids": list(plan.gpu_groups["split90"]),
            "cpu_affinity": list(plan.cpu_affinities["split90"]),
            "returncode": 0,
        },
    ]

    telemetry = module._telemetry(plan, groups, evidence)
    assert telemetry["simultaneous_groups"] is True
    sequential = json.loads(json.dumps(evidence))
    sequential[1]["started_monotonic"] = 20.0
    with pytest.raises(ValueError, match="simultaneous|overlap"):
        module._telemetry(plan, groups, sequential)


def test_selected_canary_cli_is_dry_run_by_default_and_apply_is_explicit(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.qualification as module

    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"applied": kwargs["apply"], "profile_id": "fixture"}

    monkeypatch.setattr(module, "run_authenticated_selected_canary", run)
    argv = [
        "--authority-root",
        str(tmp_path / "authority"),
        "--repo-root",
        str(ROOT),
        "--runtime-lock",
        str(tmp_path / "runtime-lock.json"),
        "--runtime-evidence",
        str(tmp_path / "runtime-evidence.json"),
        "--runtime-sbom",
        str(tmp_path / "runtime-sbom.json"),
        "--environment-receipt",
        str(tmp_path / "environment.json"),
        "--bootstrap-receipt",
        str(tmp_path / "bootstrap.json"),
        "--account-id",
        ACCOUNT_ID,
        "--instance-id",
        INSTANCE_ID,
        "--boot-id",
        BOOT_ID,
        "--seed",
        "0",
        "--selection-version-id",
        "selection-version-1",
        "--trusted-public-key-sha256",
        "7" * 64,
        "--selection-bucket",
        "memorysplit-selection-authority",
        "--region",
        "us-east-1",
        "--approval-public-key",
        str(tmp_path / "approval.pem"),
        "--aws-staging-root",
        str(tmp_path / "aws-staging"),
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--output",
        str(tmp_path / "qualification.json"),
        "--s3-root",
        "s3://memorysplit-prod/confirmatory-v3",
    ]
    dependencies = {
        "store": object(),
        "identity_verifier": object(),
        "approval_verifier": object(),
        "command_runner": object(),
        "process_launcher": object(),
        "object_store": object(),
        "time_reader": object(),
        "process_sleep": lambda seconds: None,
    }

    dry_run = module.run_selected_canary_cli(argv, **dependencies)
    applied = module.run_selected_canary_cli(
        [*argv, "--apply"],
        **dependencies,
    )

    assert dry_run == {"applied": False, "profile_id": "fixture"}
    assert applied == {"applied": True, "profile_id": "fixture"}
    assert [call["apply"] for call in calls] == [False, True]
    assert all("selection_binding" not in call for call in calls)
    assert all(call["seed"] == 0 for call in calls)


def test_selected_canary_renders_two_independent_profile_driven_4_rank_groups(
    tmp_path,
):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    rendered = module._render_selected_canary_plan(plan)

    assert plan.gpu_groups == {
        "dense": (0, 1, 2, 3),
        "split90": (4, 5, 6, 7),
    }
    assert plan.cpu_affinities == {
        "dense": (0, 95),
        "split90": (96, 191),
    }
    hardware_argv = rendered["commands"]["hardware"][0]["argv"]
    assert hardware_argv[hardware_argv.index("--region") + 1] == (
        case["binding"].region
    )
    groups = rendered["commands"]["nccl_4x4"]
    assert len(groups) == 2
    assert {row["arm"] for row in groups} == {"dense", "split90"}
    assert all(row["world_size"] == 4 for row in groups)
    assert all(row["concurrent_group"] == "paired-4rank" for row in groups)
    assert groups[0]["master_port"] != groups[1]["master_port"]
    assert not any(
        "--nproc_per_node=8" in argument
        for row in groups
        for argument in row["argv"]
    )
    assert all(
        "--nproc_per_node=4" in row["argv"]
        and IMAGE in row["argv"]
        and "--pull" in row["argv"]
        and "never" in row["argv"]
        for row in groups
    )
    for row in groups:
        arm = row["arm"]
        assert row["argv"][row["argv"].index("--gpu-ids") + 1] == ",".join(
            str(item) for item in plan.gpu_groups[arm]
        )
        assert row["argv"][row["argv"].index("--cpu-affinity") + 1] == (
            f"{plan.cpu_affinities[arm][0]}-{plan.cpu_affinities[arm][1]}"
        )
        assert row["argv"][row["argv"].index("--master-port") + 1] == str(
            plan.master_ports[arm]
        )
    assert rendered["bindings"]["provider_selection_sha256"] == (
        case["binding"].selection_sha256
    )
    assert rendered["bindings"]["provider_selection_version_id"] == (
        "selection-version-1"
    )


def test_selected_canary_executes_measured_p6_qualification_and_eta(tmp_path):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    runner = _QualificationRunner(module, plan)
    store = _QualificationStore(module)
    process_clock = _ProcessClock()
    launcher = _FakeProcessLauncher(
        module,
        plan,
        process_clock,
        {"dense": 2.0, "split90": 3.0},
    )

    receipt = module._execute_selected_canary(
        plan,
        command_runner=runner,
        process_launcher=launcher,
        object_store=store,
        time_reader=_QualificationTime(),
        process_sleep=process_clock.sleep,
        apply=False,
    )

    assert launcher.starts == ["dense-4rank", "split90-4rank"]
    assert receipt["receipt_type"] == "memorysplit-aws-gpu-qualification-v2"
    assert receipt["profile_id"] == "aws-p6-b300.48xlarge-v3"
    assert receipt["provider_selection_sha256"] == (
        case["binding"].selection_sha256
    )
    assert receipt["provider_selection_version_id"] == "selection-version-1"
    assert receipt["runtime_lock_sha256"] == case["binding"].runtime_lock_sha256
    assert receipt["runtime_sbom_sha256"] == hashlib.sha256(
        case["sbom_data"]
    ).hexdigest()
    assert receipt["environment_receipt_sha256"] == hashlib.sha256(
        _canonical(case["environment"])
    ).hexdigest()
    assert receipt["bootstrap_receipt_sha256"] == hashlib.sha256(
        _canonical(case["bootstrap"])
    ).hexdigest()
    assert receipt["hardware"]["vcpus"] == 192
    assert receipt["hardware"]["memory_gib"] == 4096
    assert receipt["hardware"]["topology"]["nvlink_active"] is True
    assert receipt["container"]["versions"] == CONTAINER_FACTS
    assert all(
        receipt["container"][field] is True
        for field in (
            "bf16",
            "sdpa_forward",
            "sdpa_backward",
            "torch_compile",
            "fused_adamw",
            "one_step_train",
            "checkpoint_resume_exact",
        )
    )
    assert {
        row["arm"]: row["world_size"] for row in receipt["nccl_groups"]
    } == {"dense": 4, "split90": 4}
    assert receipt["telemetry"]["per_arm"]["dense"]["median_tok_s"] == (
        pytest.approx(524_288 / 5.0)
    )
    assert receipt["telemetry"]["per_arm"]["split90"]["peak_memory_bytes"] == (
        150_000_000_000
    )
    expected_pair_eta = 7_120_879_616 / (524_288 / 6.0)
    assert receipt["telemetry"]["projected_pair_eta_seconds"] == pytest.approx(
        expected_pair_eta
    )
    assert receipt["telemetry"]["projected_cohort_eta_seconds"] == pytest.approx(
        expected_pair_eta * 10
    )
    assert [row["name"] for row in receipt["phases"]] == list(
        module.SELECTED_PHASE_ORDER
    )
    assert all(
        len(row["evidence_sha256"]) == 64 for row in receipt["phases"]
    )
    assert receipt["telemetry"]["simultaneous_groups"] is True
    assert receipt["telemetry"]["measured_overlap_seconds"] > 0
    assert len(receipt["processes"]) == 2
    payload = module._canonical_selected_qualification_receipt(receipt)
    assert module._parse_selected_qualification_receipt_bytes(
        payload,
        plan=plan,
    ) == receipt
    assert not plan.output_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "host-floor",
        "topology",
        "sdpa",
        "container-version",
        "world-size",
        "gpu-overlap",
        "checkpoint",
    ],
)
def test_selected_canary_fails_closed_on_measured_phase_drift(tmp_path, mutation):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    process_clock = _ProcessClock()
    launcher = _FakeProcessLauncher(
        module,
        plan,
        process_clock,
        {"dense": 2.0, "split90": 3.0},
        mutation=(
            mutation
            if mutation in {"world-size", "gpu-overlap", "checkpoint"}
            else None
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "P6|profile|topology|NVLink|container|capability|NCCL|group|"
            "checkpoint|floor|GPU"
        ),
    ):
        module._execute_selected_canary(
            plan,
            command_runner=_QualificationRunner(
                module,
                plan,
                mutation=mutation,
            ),
            process_launcher=launcher,
            object_store=_QualificationStore(module),
            time_reader=_QualificationTime(),
            process_sleep=process_clock.sleep,
            apply=False,
        )


def test_selected_qualification_receipt_cannot_cross_authorize_profiles(tmp_path):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    p6_plan = case["plan"]
    process_clock = _ProcessClock()
    receipt = module._execute_selected_canary(
        p6_plan,
        command_runner=_QualificationRunner(module, p6_plan),
        process_launcher=_FakeProcessLauncher(
            module,
            p6_plan,
            process_clock,
            {"dense": 2.0, "split90": 3.0},
        ),
        object_store=_QualificationStore(module),
        time_reader=_QualificationTime(),
        process_sleep=process_clock.sleep,
        apply=False,
    )
    payload = module._canonical_selected_qualification_receipt(receipt)
    p5 = load_aws_gpu_profile(P5_PROFILE)
    p5_lock, p5_sbom = _closed_runtime_inputs(p5)
    p5_authority = _cohort_authority(p5, p5_lock)

    with pytest.raises(ValueError, match="profile|provider|selection|binding"):
        module._build_selected_canary_plan(
            selection_authority=p5_authority,
            runtime_lock_data=p5_lock,
            runtime_sbom_data=p5_sbom,
            environment_receipt=case["environment"],
            bootstrap_receipt=case["bootstrap"],
            qualification_root=ROOT,
            scratch_root=case["scratch"],
            output_path=case["scratch"] / "p5.json",
            s3_root="s3://memorysplit-prod/confirmatory-v3",
        )
    with pytest.raises(ValueError, match="profile|provider|selection|binding"):
        module._parse_selected_qualification_receipt_bytes(payload, plan=replace(
            p6_plan,
            authority=p5_authority,
        ))


def test_legacy_p5_canary_contract_stays_byte_and_name_compatible():
    import cluster.aws.p5.canary as module

    assert module.RECEIPT_TYPE == "memorysplit-aws-p5-qualification-v1"
    assert module.PHASE_ORDER == (
        "hardware",
        "nccl_all_reduce",
        "functional",
        "resume",
        "throughput_4x4",
        "s3_roundtrip",
    )
    assert module.FUNCTIONAL_UPDATES == 1
    assert module.THROUGHPUT_UPDATES == 100


def test_digest_pinned_runtime_build_binds_executable_qualification_worker(
    tmp_path,
):
    build_script = ROOT / "containers" / "aws-gpu" / "build_image.py"
    spec = importlib.util.spec_from_file_location(
        f"p6_runtime_build_{uuid.uuid4().hex}",
        build_script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Repository:
        def inspect(self, root):
            return {
                "source_commit": "b" * 40,
                "source_tree": "c" * 40,
                "clean": True,
                "command_transcript_sha256": {
                    "head": "1" * 64,
                    "tree": "2" * 64,
                    "status": "3" * 64,
                },
            }

    plan = module.render_build_plan(
        repository_uri=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "memorysplit/aws-gpu"
        ),
        source_commit="b" * 40,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
        repository_reader=Repository(),
    )

    assert QUALIFICATION_WORKER.is_file()
    assert plan["inputs"]["qualification_worker_sha256"] == hashlib.sha256(
        QUALIFICATION_WORKER.read_bytes()
    ).hexdigest()
    dockerfile = (
        ROOT / "containers" / "aws-gpu" / "Dockerfile"
    ).read_text(encoding="utf-8")
    dockerignore = (
        ROOT / "containers" / "aws-gpu" / "Dockerfile.dockerignore"
    ).read_text(encoding="utf-8")
    assert (
        "COPY cluster/aws/qualification_worker.py "
        "/opt/memorysplit/cluster/aws/qualification_worker.py"
    ) in dockerfile
    assert "!cluster/aws/qualification_worker.py" in dockerignore

    source = QUALIFICATION_WORKER.read_text(encoding="utf-8")
    for required in (
        "torch.bfloat16",
        "scaled_dot_product_attention",
        "torch.compile",
        "fused=True",
        "init_process_group(\"nccl\")",
        "checkpoint_resume_exact",
        "peak_memory_bytes",
    ):
        assert required in source


def test_worker_loads_exact_reviewed_model_batch_token_and_sidecar_geometry():
    spec = importlib.util.spec_from_file_location(
        f"qualification_worker_geometry_{uuid.uuid4().hex}",
        QUALIFICATION_WORKER,
    )
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)

    dense = worker.load_reviewed_geometry(DENSE_CONFIG, arm="dense")
    split90 = worker.load_reviewed_geometry(SPLIT90_CONFIG, arm="split90")
    assert dense == {
        "arm": "dense",
        "model": "d360m",
        "ctx": 1024,
        "micro_batch_size": 8,
        "tokens_per_step": 524_288,
        "total_tokens": 7_120_879_616,
        "max_steps": 13_582,
        "sidecar_name": "dense_target_weights",
        "compile": True,
    }
    assert split90 == {
        **dense,
        "arm": "split90",
        "sidecar_name": "split90_target_weights",
    }


def test_group_result_rejects_toy_or_unmeasured_training_evidence(tmp_path):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    toy = _group_result(plan, "dense")
    toy.pop("geometry")
    toy.pop("sidecar")
    toy.pop("capabilities")
    toy["resume"]["checkpoint_step"] = 0
    with pytest.raises(
        ValueError,
        match="geometry|sidecar|finite|gradient|resume|checkpoint|RNG",
    ):
        module._validate_group_result(plan, toy, arm="dense")


def test_paired_groups_require_distinct_measured_target_weight_sidecars(tmp_path):
    import cluster.aws.qualification as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    dense = module._validate_group_result(
        plan,
        _group_result(plan, "dense"),
        arm="dense",
    )
    split90_value = _group_result(plan, "split90")
    split90_value["sidecar"] = dict(dense["sidecar"])
    split90_value["sidecar"]["name"] = "split90_target_weights"
    split90 = module._validate_group_result(
        plan,
        split90_value,
        arm="split90",
    )

    with pytest.raises(ValueError, match="sidecar|target weight|arms differ"):
        module._validate_paired_group_evidence(
            {"dense": dense, "split90": split90}
        )
