from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.p5.profile import (
    AWS_P5_V3_PROFILE_ID,
    AWS_P6_B300_V3_PROFILE_ID,
    AwsGpuProfile,
    AwsGpuRuntime,
    AwsP5Profile,
    AwsP5Runtime,
    load_aws_gpu_profile,
    load_aws_p5_profile,
    validate_runtime_environment,
)
from msctl.aws_p5 import aws_resource_request, build_aws_backend
from msctl.cli import build_parser, dispatch
from msctl.errors import MsctlError
from msctl.profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "cluster" / "profiles"
P5_V3 = PROFILE_ROOT / f"{AWS_P5_V3_PROFILE_ID}.json"
P6_V3 = PROFILE_ROOT / f"{AWS_P6_B300_V3_PROFILE_ID}.json"


def test_neutral_profile_api_retains_p5_type_and_loader_aliases():
    assert AwsP5Profile is AwsGpuProfile
    assert AwsP5Runtime is AwsGpuRuntime
    assert load_aws_p5_profile is load_aws_gpu_profile


def test_only_legacy_p5_uses_legacy_receipt_protocol_identities():
    legacy = load_aws_gpu_profile(
        PROFILE_ROOT / "aws-p5.48xlarge.json"
    )
    assert legacy.bootstrap_receipt_type == "aws-p5-bootstrap"
    assert (
        legacy.interruption_candidate_receipt_type
        == "aws-p5-interruption-candidate"
    )
    assert legacy.interruption_receipt_type == "aws-p5-paired-interruption"
    assert legacy.resume_commit_protocol == "aws-p5-resume-commit-v1"

    for path in (P5_V3, P6_V3):
        profile = load_aws_gpu_profile(path)
        assert profile.bootstrap_receipt_type == "aws-gpu-bootstrap"
        assert (
            profile.interruption_candidate_receipt_type
            == "aws-gpu-interruption-candidate"
        )
        assert (
            profile.interruption_receipt_type
            == "aws-gpu-paired-interruption"
        )
        assert profile.resume_commit_protocol == "aws-gpu-resume-commit-v1"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            P5_V3,
            {
                "profile_id": "aws-p5.48xlarge-v3",
                "instance_type": "p5.48xlarge",
                "memory_gib": 2048,
                "gpu_model": "NVIDIA H100 80GB",
                "gres": "gpu:h100:8",
                "cuda_minimum": "12.1",
                "driver_minimum": "530",
            },
        ),
        (
            P6_V3,
            {
                "profile_id": "aws-p6-b300.48xlarge-v3",
                "instance_type": "p6-b300.48xlarge",
                "memory_gib": 4096,
                "gpu_model": "NVIDIA B300",
                "gres": "gpu:b300:8",
                "cuda_minimum": "13.0",
                "driver_minimum": "580",
            },
        ),
    ],
)
def test_v3_profiles_are_exact_closed_4_plus_4_contracts(path, expected):
    profile = load_aws_gpu_profile(path)

    assert profile.schema_version == 3
    for field, value in expected.items():
        assert getattr(profile, field) == value
    assert profile.provider == profile.profile_id
    assert profile.vcpus == 192
    assert profile.allocated_gpus == 8
    assert profile.train_groups == (4, 4)
    assert profile.cpu_affinity_halves == ((0, 95), (96, 191))
    assert profile.assigned_seeds == tuple(range(10))
    assert profile.instance_store_devices == 8
    assert profile.instance_store_device_bytes == 3_840_000_000_000
    assert load_profile(path) == profile


def test_p6_profile_freezes_required_runtime_software_floors():
    profile = load_aws_gpu_profile(P6_V3)

    assert profile.software_minimums == {
        "cuda": "13.0",
        "driver": "580",
        "efa": "1.44.0",
        "linux_kernel": "6.1",
        "ofi_nccl": "1.17.1",
    }
    assert profile.matches_gpu_name("NVIDIA B300")
    assert not profile.matches_gpu_name("NVIDIA H100 80GB")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["cpu"].update(memory_gib=8192),
        lambda value: value["gpu"]["name_patterns"].append(".*"),
        lambda value: value["gpu"].update(gres="gpu:any:8"),
        lambda value: value["software"].update(driver_minimum="570"),
        lambda value: value.update(profile_id="aws-custom.48xlarge"),
        lambda value: value.update(extra_hardware={}),
    ],
)
def test_v3_profile_loader_rejects_arbitrary_hardware_json(
    tmp_path,
    mutate,
):
    value = json.loads(P6_V3.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError):
        load_aws_gpu_profile(path)


class _Runner:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def run_json(self, argv, *, operation):
        self.calls.append((list(argv), operation))
        return self.outputs.pop(0)


def _runtime_environment() -> dict[str, str]:
    digest = "sha256:" + "a" * 64
    return {
        "AWS_REGION": "us-east-1",
        "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v3",
        "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
        "MS_CONTAINER_DIGEST": digest,
        "MS_CONTAINER_IMAGE": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            f"memorysplit/aws-gpu@{digest}"
        ),
        "MS_RUNTIME_UID": "10001",
        "MS_RUNTIME_GID": "10001",
        "MS_AWS_INSTANCE_PROFILE_ARN": (
            "arn:aws:iam::123456789012:instance-profile/memorysplit-gpu"
        ),
    }


def test_p6_backend_constructs_and_runs_only_read_only_auth_capacity(tmp_path):
    profile = load_aws_gpu_profile(P6_V3)
    runner = _Runner(
        {
            "account": "123456789012",
            "arn": "arn:aws:iam::123456789012:role/operator",
            "user_id": "operator",
        },
        {
            "offerings": [
                {
                    "instance_type": "p6-b300.48xlarge",
                    "location": "us-east-1",
                    "location_type": "region",
                }
            ]
        },
    )
    backend = build_aws_backend(
        profile=profile,
        state_root=tmp_path / "state",
        environ=_runtime_environment(),
        runner=runner,
    )

    auth = backend.auth_check()
    assert {
        field: auth[field]
        for field in ("provider", "instance_type", "profile_sha256", "gres")
    } == {
        "provider": profile.provider,
        "instance_type": profile.instance_type,
        "profile_sha256": profile.sha256,
        "gres": profile.gres,
    }
    capacity = backend.capacity_check()
    assert capacity["provider"] == profile.provider
    assert capacity["instance_type"] == "p6-b300.48xlarge"
    assert capacity["profile_sha256"] == profile.sha256
    assert capacity["gres"] == profile.gres
    assert capacity["offered"] is True
    rendered = json.dumps(runner.calls)
    assert "run-instances" not in rendered
    assert "p6-b300.48xlarge" in rendered

    resources = aws_resource_request("submit", profile=profile)
    assert resources["allocated_gpus"] == 8
    assert resources["gres"] == "gpu:b300:8"

    with pytest.raises(MsctlError) as unsupported:
        backend.dispatch("env ensure", object())
    assert getattr(unsupported.value, "code", None) == (
        "EXTERNAL_OPERATION_UNSUPPORTED"
    )


@pytest.mark.parametrize("profile_path", [P5_V3, P6_V3])
def test_cli_routes_each_v3_provider_through_existing_backend(profile_path):
    profile = load_profile(profile_path)
    captured = {}

    class Backend:
        def dispatch(self, command, _args):
            return False, {"provider": profile.provider, "command": command}

    def factory(**kwargs):
        captured.update(kwargs)
        return Backend()

    args = build_parser().parse_args(
        ["--profile", str(profile_path), "auth", "check"]
    )
    dry_run, report = dispatch(
        args,
        aws_backend_factory=factory,
        environ={},
    )

    assert dry_run is False
    assert report == {
        "provider": profile.provider,
        "command": "auth check",
    }
    assert captured["profile"] == profile


def test_p6_bootstrap_and_launcher_admission_use_profile_hardware(tmp_path):
    from cluster.aws.p5.bootstrap import inspect_hardware
    from cluster.aws.p5.launch_seed_pair import LaunchError, load_launch_plan
    from tests.test_aws_p5_launcher import (
        BOOT_ID,
        CONTAINER_IMAGE,
        H100_NAMES,
        SAFE_ENVIRONMENT,
        _metadata,
        _ProbeRunner,
    )

    profile = load_aws_gpu_profile(P6_V3)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    b300_names = ("NVIDIA B300",) * 8
    evidence = inspect_hardware(
        profile,
        runtime,
        metadata_get=_metadata("p6-b300.48xlarge"),
        runner=_ProbeRunner(gpu_names=b300_names),
        command_environment={"PATH": "/usr/bin:/bin"},
        container_image=CONTAINER_IMAGE,
        boot_id_get=lambda: BOOT_ID,
    )

    assert evidence.instance_type == "p6-b300.48xlarge"
    assert evidence.gpu_names == b300_names

    common = {
        "seed": 0,
        "manifest_path": tmp_path / "not-reached.json",
        "profile_path": P6_V3,
        "repo_root": tmp_path,
        "scratch_root": tmp_path,
        "environment": SAFE_ENVIRONMENT,
        "observed_instance_id": "i-0123456789abcdef0",
        "observed_boot_id": BOOT_ID,
        "port_available": lambda _port: True,
        "enforce_profile_scratch": False,
    }
    with pytest.raises(LaunchError, match="p6-b300.48xlarge"):
        load_launch_plan(
            **common,
            observed_instance_type="p5.48xlarge",
            gpu_names=b300_names,
        )
    with pytest.raises(LaunchError, match="NVIDIA B300"):
        load_launch_plan(
            **common,
            observed_instance_type="p6-b300.48xlarge",
            gpu_names=H100_NAMES,
        )


def _backend_for_profile(profile, tmp_path, *, runner=None):
    return build_aws_backend(
        profile=profile,
        state_root=tmp_path,
        environ=_runtime_environment(),
        runner=runner or _Runner(),
    )


def _synthetic_pair_manifest(provider: str, *, seed: int = 0):
    runs = tuple(
        SimpleNamespace(
            run_id=f"aws-gpu-v3-s{seed}-{arm}",
            arm=arm,
            seed=seed,
            config=f"configs/{provider}/{arm}.yaml",
            config_sha256=("a" if arm == "dense" else "b") * 64,
        )
        for arm in ("dense", "split90")
    )
    return SimpleNamespace(
        provider=provider,
        seed=seed,
        release_sha256="1" * 64,
        dataset_sha256="2" * 64,
        cohort_assignment_sha256="3" * 64,
        study_lock_sha256="4" * 64,
        source_commit="5" * 40,
        runs=runs,
        sha256="6" * 64,
    )


def test_p5_and_p6_instance_selection_helpers_reject_cross_profile_rows(
    tmp_path,
):
    p5 = load_aws_gpu_profile(P5_V3)
    p6 = load_aws_gpu_profile(P6_V3)
    p5_backend = _backend_for_profile(p5, tmp_path / "p5")
    p6_backend = _backend_for_profile(p6, tmp_path / "p6")
    manifest = _synthetic_pair_manifest(p6.provider)
    row = {
        "instance_id": "i-0123456789abcdef0",
        "instance_type": p6.instance_type,
        "profile_instance_type": p6.instance_type,
        "state": "running",
        "instance_profile_arn": p6_backend.instance_profile_arn,
        "provider": p6.provider,
        "seed": manifest.seed,
        "cohort_sha256": manifest.cohort_assignment_sha256,
        "release_sha256": manifest.release_sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "run_manifest_sha256": manifest.sha256,
        "profile_sha256": p6.sha256,
        "gres": p6.gres,
    }

    assert p6_backend._parse_instances({"instances": [row]}, manifest) == [row]
    with pytest.raises(MsctlError) as cross_profile:
        p5_backend._parse_instances({"instances": [row]}, manifest)
    assert cross_profile.value.code == "INSTANCE_BINDING_MISMATCH"


def test_p5_and_p6_operation_state_cannot_be_reused_across_profiles(tmp_path):
    from msctl.jsonutil import canonical_json
    from msctl.state import StateStore

    p5 = load_aws_gpu_profile(P5_V3)
    p6 = load_aws_gpu_profile(P6_V3)
    p5_backend = _backend_for_profile(p5, tmp_path / "state")
    p6_backend = _backend_for_profile(p6, tmp_path / "state")
    manifest = _synthetic_pair_manifest(p6.provider)
    release = SimpleNamespace(
        provider=p6.provider,
        archive_sha256=manifest.release_sha256,
        receipt_sha256="7" * 64,
        members_sha256="8" * 64,
        source_commit=manifest.source_commit,
    )
    terminate_at = "2099-01-01T00:00:00Z"
    core = p6_backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=terminate_at,
    )
    assert {
        key: core[key]
        for key in ("provider", "instance_type", "profile_sha256", "gres")
    } == {
        "provider": p6.provider,
        "instance_type": p6.instance_type,
        "profile_sha256": p6.sha256,
        "gres": p6.gres,
    }
    intent = p6_backend._operation_envelope(
        core,
        instance_id="i-0123456789abcdef0",
        terminate_at=terminate_at,
    )
    intent_sha256 = hashlib.sha256(canonical_json(intent)).hexdigest()
    published = {
        "intent_sha256": intent_sha256,
        "intent_uri": (
            f"{p6_backend.runtime.s3_root}/operations/intents/"
            f"sha256/{intent_sha256}.json"
        ),
    }
    states = [
        p6_backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id="i-0123456789abcdef0",
            terminate_at=terminate_at,
            intent=intent,
            published=published,
            attempt=1,
        )
        for run in manifest.runs
    ]
    store = StateStore(tmp_path / "state")
    with store.locked():
        p6_backend._write_paired_states(store, manifest, states)
        with pytest.raises(MsctlError) as cross_profile:
            p5_backend._repair_paired_states(store, manifest)
    assert cross_profile.value.code == "STATE_CORRUPT"


@pytest.mark.parametrize("region", ["us-east-1", "us-west-2"])
def test_instance_identity_verifier_selects_a_pinned_region_anchor(
    region,
    monkeypatch,
):
    from msctl import aws_p5

    identity = {
        "instanceId": "i-0123456789abcdef0",
        "instanceType": "p6-b300.48xlarge",
        "region": region,
    }
    observed = {}

    def run(argv, **kwargs):
        certificate = Path(argv[argv.index("-certfile") + 1])
        observed["certificate"] = certificate.read_text(encoding="ascii")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(identity).encode("utf-8"),
        )

    monkeypatch.setattr(aws_p5.subprocess, "run", run)
    assert aws_p5._verify_instance_identity_pkcs7(
        identity,
        "YQ==",
        region,
    )
    assert observed["certificate"] == aws_p5._AWS_DSA_CERTIFICATES[region]
    assert observed["certificate"].startswith("-----BEGIN CERTIFICATE-----\n")
    assert observed["certificate"].endswith("-----END CERTIFICATE-----\n")


def test_instance_identity_verifier_rejects_regions_without_a_pinned_anchor():
    from msctl.aws_p5 import _verify_instance_identity_pkcs7

    with pytest.raises(MsctlError, match="no pinned AWS PKCS7"):
        _verify_instance_identity_pkcs7({}, "YQ==", "us-east-2")
