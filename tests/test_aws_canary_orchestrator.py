from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.p5 import canary_runtime
from cluster.aws.p5.canary_orchestrator import (
    CanaryOrchestrationError,
    _run_concurrent,
)
from cluster.aws.p5.profile import validate_runtime_environment
from msctl.aws_argv import _validate_intent
from msctl.aws_canary import (
    build_canary_intent,
    canary_plan_uri,
    load_canary_plan,
    plan_canary,
)
from msctl.aws_identity import AwsIdentityError, verify_environment_receipt
from msctl.aws_p5 import AwsP5Backend
from msctl.cli import build_parser, dispatch
from msctl.errors import MsctlError
from msctl.jsonutil import canonical_json
from tests.test_v3_hardware_amendment import P5, _selection


def _inputs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "msctl.aws_identity.verify_instance_identity_pkcs7",
        lambda identity, pkcs7, region: (
            bool(identity) and pkcs7 == "YQ==" and bool(region)
        ),
    )
    profile, _amendment, selection = _selection(P5)
    runtime = validate_runtime_environment(
        profile,
        {
            "AWS_REGION": selection.region,
            "MS_S3_ROOT": "s3://memorysplit-prod/qualification-v3",
            "MS_S3_KMS_KEY_ID": (
                f"arn:aws:kms:{selection.region}:123456789012:"
                "key/12345678-1234-4234-9234-123456789abc"
            ),
            "MS_AWS_AMI_ID": selection.ami_id,
            "MS_CONTAINER_IMAGE": selection.container_image,
            "MS_CONTAINER_DIGEST": selection.container_digest,
            "MS_RUNTIME_UID": "10001",
            "MS_RUNTIME_GID": "10002",
        },
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
            "instanceType": profile.instance_type,
            "region": selection.region,
        },
        "aws_instance_identity_pkcs7": "YQ==",
    }
    environment_path = tmp_path / "environment.json"
    environment_path.write_bytes(canonical_json(environment) + b"\n")
    release = SimpleNamespace(
        provider=profile.provider,
        archive_sha256="e" * 64,
        metadata={"profile": {"sha256": profile.sha256}},
    )
    return profile, runtime, release, selection, instance_id, environment_path


def test_canary_plan_and_run_are_dry_run_first_and_content_addressed(
    tmp_path,
    monkeypatch,
):
    profile, runtime, release, selection, instance_id, environment = _inputs(
        tmp_path,
        monkeypatch,
    )
    destination = tmp_path / "canary-plan.json"
    kwargs = {
        "profile": profile,
        "runtime": runtime,
        "release": release,
        "selection": selection,
        "environment_receipt": environment,
        "instance_id": instance_id,
        "out": destination,
    }

    dry = plan_canary(apply=False, **kwargs)
    assert dry["published"] is False
    assert not destination.exists()

    applied = plan_canary(apply=True, **kwargs)
    assert applied["published"] is True
    assert destination.read_bytes() == canonical_json(applied["plan"]) + b"\n"
    plan, plan_sha256 = load_canary_plan(
        destination,
        profile=profile,
        runtime=runtime,
        expected_instance_id=instance_id,
    )
    assert plan_sha256 == applied["plan_sha256"]
    plan_uri = canary_plan_uri(runtime.s3_root, plan_sha256)
    intent = build_canary_intent(
        plan=plan,
        plan_sha256=plan_sha256,
        plan_uri=plan_uri,
        runtime=runtime,
        control_bundle_sha256="c" * 64,
    )
    payload = canonical_json(intent)
    assert _validate_intent(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_control_bundle_sha256="c" * 64,
    ) == intent
    assert intent["steps"][0]["argv"][1] == (
        f"/mnt/memorysplit/releases/{release.archive_sha256}/"
        "cluster/aws/p5/canary_orchestrator.py"
    )
    changed_plan = dict(plan)
    changed_plan["instance_id"] = "i-fedcba98765432100"
    with pytest.raises(MsctlError, match="changed after validation"):
        build_canary_intent(
            plan=changed_plan,
            plan_sha256=plan_sha256,
            plan_uri=plan_uri,
            runtime=runtime,
            control_bundle_sha256="c" * 64,
        )

    backend = AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-gpu"
        ),
        state_root=tmp_path / "state",
    )
    calls = []
    monkeypatch.setattr(
        backend,
        "_require_canary_instance",
        lambda selected: calls.append(("instance", selected)),
    )
    monkeypatch.setattr(
        backend,
        "_require_ssm_online",
        lambda selected: calls.append(("ssm", selected)),
    )
    monkeypatch.setattr(
        backend,
        "_ensure_argv_document",
        lambda: calls.append(("document", None)),
    )
    monkeypatch.setattr(
        backend,
        "_publish_control_bundle",
        lambda: {"version_id": "control-v1"},
    )
    monkeypatch.setattr(
        backend,
        "_publish_canary_plan",
        lambda *args, **kwargs: {"version_id": "plan-v1"},
    )
    monkeypatch.setattr(
        backend,
        "_publish_operation_intent",
        lambda value: {
            "intent_sha256": hashlib.sha256(canonical_json(value)).hexdigest(),
            "intent_uri": (
                f"{runtime.s3_root}/operations/intents/sha256/"
                f"{hashlib.sha256(canonical_json(value)).hexdigest()}.json"
            ),
            "version_id": "intent-v1",
        },
    )
    monkeypatch.setattr(
        backend,
        "_find_operation_command",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        backend,
        "_send_operation_intent",
        lambda **kwargs: "cmd-canary-12345678",
    )

    run_dry = backend.canary_run(
        plan_path=destination,
        instance_id=instance_id,
        apply=False,
    )
    assert run_dry["submitted"] == 0
    assert calls == []
    run = backend.canary_run(
        plan_path=destination,
        instance_id=instance_id,
        apply=True,
    )
    assert run["submitted"] == 1
    assert run["command_id"] == "cmd-canary-12345678"
    assert calls == [
        ("instance", instance_id),
        ("ssm", instance_id),
        ("document", None),
    ]


def test_environment_receipt_requires_the_pkcs7_verifier(tmp_path, monkeypatch):
    profile, runtime, _release, selection, instance_id, environment = _inputs(
        tmp_path,
        monkeypatch,
    )
    data = environment.read_bytes()
    verified = verify_environment_receipt(
        data,
        expected_profile_sha256=profile.sha256,
        expected_container_digest=runtime.container_digest,
        expected_region=runtime.region,
        expected_ami_id=runtime.ami_id,
        expected_account_id=selection.aws_account_id,
        expected_instance_id=instance_id,
    )
    assert verified.instance_id == instance_id

    monkeypatch.setattr(
        "msctl.aws_identity.verify_instance_identity_pkcs7",
        lambda identity, pkcs7, region: False,
    )
    with pytest.raises(AwsIdentityError, match="unsigned"):
        verify_environment_receipt(
            data,
            expected_profile_sha256=profile.sha256,
            expected_container_digest=runtime.container_digest,
            expected_region=runtime.region,
            expected_ami_id=runtime.ami_id,
            expected_account_id=selection.aws_account_id,
            expected_instance_id=instance_id,
        )


def test_canary_cli_routes_dry_run_and_apply_to_the_aws_backend(
    tmp_path,
    monkeypatch,
):
    profile, _runtime, _release, _selection_value, instance_id, _environment = (
        _inputs(tmp_path, monkeypatch)
    )
    calls = []

    class Backend:
        def dispatch(self, command, arguments):
            calls.append((command, arguments.instance_id, arguments.apply))
            return not arguments.apply, {"command": command}

    def backend_factory(**kwargs):
        assert kwargs["profile"] is profile
        assert kwargs["state_root"] == str(tmp_path / "state")
        assert kwargs["environ"] == {"MS_TEST": "closed"}
        return Backend()

    common = [
        "--profile",
        "ignored.json",
        "--state-root",
        str(tmp_path / "state"),
    ]
    plan = [
        *common,
        "canary",
        "plan",
        "--release",
        "release.json",
        "--provider-selection",
        "selection.json",
        "--environment-receipt",
        "environment.json",
        "--instance-id",
        instance_id,
        "--out",
        "canary.json",
    ]
    dry_run, _ = dispatch(
        build_parser().parse_args(plan),
        profile_loader=lambda _path: profile,
        aws_backend_factory=backend_factory,
        environ={"MS_TEST": "closed"},
    )
    assert dry_run is True

    run = [
        *common,
        "canary",
        "run",
        "--canary-plan",
        "canary.json",
        "--instance-id",
        instance_id,
        "--apply",
    ]
    dry_run, _ = dispatch(
        build_parser().parse_args(run),
        profile_loader=lambda _path: profile,
        aws_backend_factory=backend_factory,
        environ={"MS_TEST": "closed"},
    )
    assert dry_run is False
    assert calls == [
        ("canary plan", instance_id, False),
        ("canary run", instance_id, True),
    ]


def _throughput_receipt(profile, runtime, arm):
    evidence = {
        "arm": arm,
        "backend": "nccl",
        "ranks": [0, 1, 2, 3],
        "world_size": 4,
        "updates": 100,
        "warmup_updates": 10,
        "tokens_per_update": 524_288,
        "update_seconds": [1.0] * 100,
        "worker_stdout_sha256": "f" * 64,
    }
    return canary_runtime.build_phase_receipt(
        phase="throughput",
        provider=profile.provider,
        instance_type=profile.instance_type,
        profile_sha256=profile.sha256,
        release_sha256="e" * 64,
        container_image=runtime.container_image,
        container_digest=runtime.container_digest,
        gres=profile.gres,
        gpu_ids=range(4) if arm == "dense" else range(4, 8),
        arm=arm,
        evidence=evidence,
        raw_outputs=[
            {
                "argv": [
                    "/opt/venv/bin/python",
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    "--nproc_per_node=4",
                    "--rdzv_backend=c10d",
                    "/workspace/cluster/aws/p5/canary_runtime.py",
                    "throughput-worker",
                ],
                "returncode": 0,
                "stdout_sha256": "f" * 64,
                "stderr_sha256": "0" * 64,
            }
        ],
    )


def test_orchestrator_rejects_concurrent_commands_without_real_overlap(
    tmp_path,
    monkeypatch,
):
    profile, runtime, _release, _selection_value, _instance, _environment = (
        _inputs(tmp_path, monkeypatch)
    )
    output_root = tmp_path / "qualification"
    receipts = output_root / "receipts"
    receipts.mkdir(parents=True)
    payloads = {}
    commands = []
    for arm in ("dense", "split90"):
        receipt = _throughput_receipt(profile, runtime, arm)
        payload = canonical_json(receipt) + b"\n"
        path = receipts / f"throughput-{arm}.json"
        path.write_bytes(payload)
        payloads[arm] = payload
        commands.append(
            [
                "/usr/bin/docker",
                "--arm",
                arm,
                "--output",
                f"/qualification/receipts/throughput-{arm}.json",
            ]
        )

    class Process:
        returncode = 0

        def __init__(self, payload):
            self.payload = payload

        def communicate(self, timeout=None):
            assert timeout in {None, 7_200.0}
            return self.payload, b""

        def poll(self):
            return self.returncode

        def kill(self):
            pytest.fail("successful fake process must not be killed")

    def popen(argv, **kwargs):
        assert kwargs["shell"] is False
        arm = argv[argv.index("--arm") + 1]
        return Process(payloads[arm])

    ticks = iter((0, 20, 10, 30))
    lock = threading.Lock()

    def clock_ns():
        with lock:
            return next(ticks)

    with pytest.raises(CanaryOrchestrationError, match="did not actually overlap"):
        _run_concurrent(
            commands,
            output_root=output_root,
            popen=popen,
            clock_ns=clock_ns,
        )
