from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import cluster.aws.readiness as readiness
import cluster.aws.reasoning_v3 as corpus
from cluster.aws.readiness import inspect_aws_readiness
from cluster.aws.reasoning_v3 import (
    AwsCorpusError,
    StagedCorpus,
    load_transfer_manifest,
    parse_s3_uri,
    stage_from_s3,
    upload_to_s3,
    verify_staged_corpus,
    verify_upload_sources,
)
from msctl import aws_operations
from msctl.adapters.slurm import load_pair_manifest, plan_sbatch
from msctl.operations import collect, evaluate, resume, status, submit
from msctl.preflight import build_preflight_receipt, validate_preflight
from msctl.profile import load_profile
from msctl.reasoning_cohort import (
    COHORT_ID,
    COMPOSITE_STREAM_SHA256,
    RAW_TARGETS,
    TERMINAL_UPDATES,
    TRANSFER_MANIFEST_SHA256,
    VIRTUAL_RECEIPT_SHA256,
    load_cohort_assignment,
    load_dataset_pointer,
    load_run_config,
    role_config_paths,
)
import scripts.package_aws_reasoning_v3 as package_module
from scripts.evaluate_reasoning_v3_run import evaluate_run
from scripts.generate_aws_reasoning_configs import generate
from scripts.package_aws_reasoning_v3 import build_package, verify_package
from train.model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cluster" / "aws" / "reasoning-v3-corpus-manifest.json"
PROFILE = ROOT / "cluster" / "profiles" / "aws-p5-p6.example.json"
B200_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b200.48xlarge-135m-v1.json"
INEXACT_HARDWARE = (
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
    "NVIDIA B300",
    "NVIDIA GB200 NVL72",
    "not-a-B200 emulator",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tiny_transfer(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict[str, bytes]]:
    remote = tmp_path / "remote"
    sources = tmp_path / "repository"
    token_segments = {
        "base/packed/targets.bin": np.arange(12, dtype=np.uint16).tobytes(),
        "extension/packed/targets.bin": np.arange(12, 20, dtype=np.uint16).tobytes(),
        "base/sidecars/dense_target_weights.bin": bytes([1] * 12),
        "base/sidecars/split90_target_weights.bin": bytes([1, 0] * 6),
        "extension/sidecars/shared_target_weights.bin": bytes([1] * 8),
        "extension/records/manifest.bin": b"tiny-record-manifest",
    }
    base_receipt = {
        "contract_id": "memorysplit-parallel-corpus-v2",
        "raw_target_tokens": 7_120_879_616,
        "task4_publication": {"receipt_sha256": "c" * 64},
    }
    extension_receipt = {
        "base_corpus": {"receipt_sha256": "c" * 64},
        "composite": {
            "raw_target_tokens": 20,
            "stream_sha256": {},
        },
        "contract_id": corpus.CONTRACT_ID,
    }
    packed_sha = hashlib.sha256(
        token_segments["base/packed/targets.bin"]
        + token_segments["extension/packed/targets.bin"]
    ).hexdigest()
    dense_sha = hashlib.sha256(
        token_segments["base/sidecars/dense_target_weights.bin"]
        + token_segments["extension/sidecars/shared_target_weights.bin"]
    ).hexdigest()
    split_sha = hashlib.sha256(
        token_segments["base/sidecars/split90_target_weights.bin"]
        + token_segments["extension/sidecars/shared_target_weights.bin"]
    ).hexdigest()
    composite = {
        "dense_target_weights": dense_sha,
        "packed_targets": packed_sha,
        "split90_target_weights": split_sha,
    }
    extension_receipt["composite"]["stream_sha256"] = composite
    receipt_bytes = (
        json.dumps(extension_receipt, indent=2, sort_keys=True) + "\n"
    ).encode()
    virtual_receipt = hashlib.sha256(receipt_bytes).hexdigest()
    pointer = {
        "expected_composite_stream_sha256": composite,
        "expected_receipt_sha256": virtual_receipt,
        "launch_gate_status": "frozen",
    }
    pointer_bytes = (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode()
    frozen = {
        "composite_stream_sha256": composite,
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "receipt_sha256": virtual_receipt,
    }
    files = {
        **token_segments,
        "base/receipt.json": (
            json.dumps(base_receipt, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "extension/receipt.json": receipt_bytes,
        "locks/reasoning-pointer.json": pointer_bytes,
        "locks/FROZEN.json": (
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    objects = []
    for relative, data in sorted(files.items()):
        source_path = f"sources/{relative}"
        objects.append(
            {
                "bytes": len(data),
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_path": source_path,
            }
        )
        for root, path in (
            (remote, relative),
            (sources, source_path),
        ):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    manifest_value = {
        "composite_stream_sha256": composite,
        "contract_id": corpus.CONTRACT_ID,
        "format": corpus.TRANSFER_FORMAT,
        "objects": objects,
        "raw_target_tokens": 20,
        "schema_version": 1,
        "virtual_receipt_sha256": virtual_receipt,
    }
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, manifest_value)
    monkeypatch.setattr(corpus, "RAW_TARGET_TOKENS", 20)
    monkeypatch.setattr(corpus, "VIRTUAL_RECEIPT_SHA256", virtual_receipt)
    monkeypatch.setattr(corpus, "EXPECTED_COMPOSITE_STREAM_SHA256", composite)
    monkeypatch.setattr(corpus, "EXPECTED_OBJECTS", tuple(objects))
    monkeypatch.setattr(corpus, "TRANSFER_MANIFEST_SHA256", _sha(manifest))
    return manifest, remote, files


def _canary_evidence(path: Path, *, mode: str, updates: int) -> Path:
    arms = {}
    for arm in ("dense", "split90"):
        record = {
            "gpu_supported": True,
            "oom_detected": False,
            "status": "completed",
            "step": updates,
        }
        if mode == "resume":
            record["resume_exact"] = True
        if mode == "throughput":
            record["tokens_per_second"] = 12_345.0
        arms[arm] = record
    _write_json(
        path,
        {
            "arms": arms,
            "mode": mode,
            "status": "completed",
        },
    )
    return path


def test_checked_in_aws_contract_is_frozen_and_all_configs_are_exact():
    manifest = load_transfer_manifest(MANIFEST)
    assert manifest.sha256 == TRANSFER_MANIFEST_SHA256
    assert len(manifest.objects) == 10
    load_dataset_pointer(ROOT / "DATASET-POINTER-AWS-135M-V3.json")
    assignment = load_cohort_assignment(
        ROOT / "configs" / "cohort-assignment-135m-v3-aws-n10.json"
    )
    assert assignment["raw_target_tokens"] == RAW_TARGETS
    assert assignment["terminal_updates"] == TERMINAL_UPDATES
    for seed in range(10):
        for arm in ("dense", "split90"):
            cfg = load_run_config(
                ROOT / "configs" / "135m-v3" / f"{arm}-s{seed}.yaml",
                root=ROOT,
            )
            assert len(cfg["train_bin"]) == len(cfg["train_mask"]) == 2
            assert cfg["dataset"]["scientific_scope"].startswith("successor_")
    assert generate(ROOT, write=False) == []


@pytest.mark.parametrize(
    "value",
    [
        "https://bucket/key",
        "s3://UPPER/key",
        "s3://bucket/../key",
        "s3://bucket/key?versionId=unbound",
        "s3://127.0.0.1/key",
        "s3://bucket/key\n--profile=other",
    ],
)
def test_s3_uri_parser_rejects_ambiguous_or_injectable_values(value):
    with pytest.raises(AwsCorpusError):
        parse_s3_uri(value)
    assert parse_s3_uri("s3://valid-private-bucket/frozen/v3") == (
        "valid-private-bucket",
        "frozen/v3",
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason=(
        "corpus publication deliberately fails closed without Linux renameat2 "
        "RENAME_NOREPLACE; this gate is exercised on the EC2 execution target"
    ),
)
def test_s3_stage_is_hash_gated_atomic_and_idempotent(tmp_path, monkeypatch):
    manifest, remote, _ = _tiny_transfer(tmp_path, monkeypatch)
    destination = tmp_path / "staged"
    calls = []

    def runner(command):
        calls.append(command)
        assert command[:3] == ["aws", "s3", "cp"]
        relative = command[3].split("/frozen/v3/", 1)[1]
        target = Path(command[4])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / relative, target)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    report = stage_from_s3(
        "s3://valid-private-bucket/frozen/v3",
        destination,
        manifest,
        apply=True,
        runner=runner,
    )
    assert report["already_present"] is False
    assert len(calls) == len(corpus.EXPECTED_OBJECTS)
    evidence = verify_staged_corpus(destination, manifest)
    assert evidence.raw_target_tokens == 20
    second = stage_from_s3(
        "s3://valid-private-bucket/frozen/v3",
        destination,
        manifest,
        apply=True,
        runner=lambda command: pytest.fail(f"unexpected download: {command}"),
    )
    assert second["already_present"] is True
    (destination / "extension/packed/targets.bin").write_bytes(b"tampered")
    with pytest.raises(AwsCorpusError, match="differs"):
        stage_from_s3(
            "s3://valid-private-bucket/frozen/v3",
            destination,
            manifest,
            apply=True,
            runner=lambda command: pytest.fail(f"unexpected download: {command}"),
        )


def test_s3_stage_cleans_partial_download_and_upload_sources_reject_symlinks(
    tmp_path,
    monkeypatch,
):
    manifest, _, _ = _tiny_transfer(tmp_path, monkeypatch)
    destination = tmp_path / "staged"
    calls = 0

    def failing_runner(command):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=9, stdout="", stderr="download failed")

    with pytest.raises(AwsCorpusError, match="download failed"):
        stage_from_s3(
            "s3://valid-private-bucket/frozen/v3",
            destination,
            manifest,
            apply=True,
            runner=failing_runner,
        )
    assert calls == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".staged.aws-stage-*"))

    loaded = load_transfer_manifest(manifest)
    source = tmp_path / "repository" / loaded.objects[0].source_path
    original = source.read_bytes()
    symlink_target = tmp_path / "repository" / "symlink-target"
    symlink_target.write_bytes(original)
    source.unlink()
    source.symlink_to(symlink_target)
    with pytest.raises(AwsCorpusError, match="source differs"):
        verify_upload_sources(tmp_path / "repository", manifest)


def test_s3_upload_uses_kms_metadata_and_never_a_shell(tmp_path, monkeypatch):
    manifest, _, _ = _tiny_transfer(tmp_path, monkeypatch)
    repository = tmp_path / "repository"
    uploaded: set[str] = set()
    commands = []

    def runner(command):
        assert isinstance(command, list)
        commands.append(command)
        if command[1:3] == ["s3api", "head-object"]:
            key = command[command.index("--key") + 1]
            relative = key.split("frozen/v3/", 1)[1]
            item = next(value for value in corpus.EXPECTED_OBJECTS if value["path"] == relative)
            if relative in uploaded:
                return SimpleNamespace(returncode=0, stdout=item["sha256"] + "\n", stderr="")
            return SimpleNamespace(returncode=255, stdout="", stderr="404 Not Found")
        relative = command[4].split("/frozen/v3/", 1)[1]
        uploaded.add(relative)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    dry = upload_to_s3(
        repository,
        manifest,
        "s3://valid-private-bucket/frozen/v3",
        kms_key_id="alias/memorysplit",
        runner=lambda command: pytest.fail(f"unexpected dry-run command: {command}"),
    )
    assert dry["applied"] is False
    assert dry["uploaded"] == dry["verified_existing"] == 0

    report = upload_to_s3(
        repository,
        manifest,
        "s3://valid-private-bucket/frozen/v3",
        kms_key_id="alias/memorysplit",
        apply=True,
        runner=runner,
    )
    assert report["uploaded"] == len(corpus.EXPECTED_OBJECTS)
    uploads = [command for command in commands if command[1:3] == ["s3", "cp"]]
    heads = [
        command for command in commands if command[1:3] == ["s3api", "head-object"]
    ]
    assert len(heads) == 2 * len(corpus.EXPECTED_OBJECTS)
    assert all("--sse-kms-key-id" in command for command in uploads)
    assert all("--metadata" in command for command in uploads)
    assert all("--checksum-algorithm" in command for command in uploads)
    assert all(command[command.index("--sse") + 1] == "aws:kms" for command in uploads)

    second = upload_to_s3(
        repository,
        manifest,
        "s3://valid-private-bucket/frozen/v3",
        kms_key_id="alias/memorysplit",
        apply=True,
        runner=runner,
    )
    assert second["uploaded"] == 0
    assert second["verified_existing"] == len(corpus.EXPECTED_OBJECTS)


def test_s3_upload_rejects_wrong_existing_identity(tmp_path, monkeypatch):
    manifest, _, _ = _tiny_transfer(tmp_path, monkeypatch)
    with pytest.raises(AwsCorpusError, match="another identity"):
        upload_to_s3(
            tmp_path / "repository",
            manifest,
            "s3://valid-private-bucket/frozen/v3",
            kms_key_id="alias/memorysplit",
            apply=True,
            runner=lambda command: SimpleNamespace(
                returncode=0,
                stdout="0" * 64,
                stderr="",
            ),
        )


def test_no_broad_gpu_regex_authorizes_protected_b200_work():
    profile = load_profile(PROFILE)
    admitted = [
        name for name in INEXACT_HARDWARE if re.search(profile.gpu_name_regex, name)
    ]
    assert admitted, "the fixture must exercise the shipped substring regex"
    exact_gate = getattr(readiness, "admit_b200_node", None)
    assert exact_gate is not None, (
        "protected B200 work is still authorized by the substring regex "
        f"{profile.gpu_name_regex!r} in {PROFILE.name}, which admits inexact "
        f"hardware {admitted}; cluster.aws.readiness exposes no exact B200 "
        "admission gate"
    )
    shipped = readiness.load_hardware_profile(B200_PROFILE)
    assert shipped.launch_ready is False
    assert shipped.instance_type == "p6-b200.48xlarge"


def test_aws_profile_and_parallelcluster_template_bind_the_exact_b200_node():
    profile = load_profile(PROFILE)
    assert profile.platform == "aws"
    assert profile.gpus_per_pair == 2
    hardware = readiness.load_hardware_profile(B200_PROFILE)
    template = yaml.safe_load(
        (
            ROOT
            / "cluster/aws/parallelcluster/memorysplit-v3-p5.example.yaml"
        ).read_text()
    )
    assert template["Region"] == hardware.region
    queues = template["Scheduling"]["SlurmQueues"]
    resources = [
        resource for queue in queues for resource in queue["ComputeResources"]
    ]
    assert {resource["InstanceType"] for resource in resources} == {
        hardware.instance_type
    }
    assert all(resource["MinCount"] == 0 for resource in resources)
    queue = queues[0]
    assert queue["Name"] == profile.partition
    assert queue["ComputeResources"][0]["MinCount"] == 0
    assert queue["ComputeResources"][0]["MaxCount"] == 2
    assert queue["ComputeResources"][0]["Efa"]["Enabled"] is False
    assert queue["Networking"]["PlacementGroup"]["Enabled"] is False
    assert template["HeadNode"]["Imds"]["Secured"] is True
    assert template["SharedStorage"][0]["EfsSettings"]["DeletionPolicy"] == "Retain"
    s3_access = {
        item["KeyName"]: item["EnableWriteAccess"]
        for item in template["HeadNode"]["Iam"]["S3Access"]
    }
    assert s3_access == {
        "corpus/REPLACE/*": False,
        "evidence/REPLACE/*": True,
        "releases/*": False,
    }


def test_aws_readiness_inspection_is_read_only_and_reports_p_offerings():
    commands = []
    responses = [
        {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/test/session",
            "UserId": "fixture",
        },
        {
            "InstanceTypeOfferings": [
                {"InstanceType": "p5.48xlarge", "Location": "us-east-1a"},
                {"InstanceType": "p5.48xlarge", "Location": "us-east-1f"},
            ]
        },
        [
            {
                "Adjustable": True,
                "QuotaCode": "L-417A185B",
                "QuotaName": "Running On-Demand P instances",
                "Unit": "None",
                "Value": 192.0,
            }
        ],
    ]

    def runner(command):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(responses[len(commands) - 1]),
            stderr="",
        )

    report = inspect_aws_readiness(region="us-east-1", runner=runner)
    assert report["account"] == "123456789012"
    assert report["instance_type_availability_zones"]["p5.48xlarge"] == [
        "us-east-1a",
        "us-east-1f",
    ]
    assert all(command[1] in {"ec2", "service-quotas", "sts"} for command in commands)


def test_aws_instantiate_submit_resume_status_and_collect(tmp_path, monkeypatch):
    evidence = StagedCorpus(
        root=(tmp_path / "dataset").resolve(),
        manifest_sha256=TRANSFER_MANIFEST_SHA256,
        virtual_receipt_sha256=VIRTUAL_RECEIPT_SHA256,
        raw_target_tokens=RAW_TARGETS,
        composite_stream_sha256=COMPOSITE_STREAM_SHA256,
    )
    verification_calls = []

    def verified_dataset(*args, **kwargs):
        verification_calls.append((args, kwargs))
        return evidence

    monkeypatch.setattr(
        aws_operations,
        "verify_staged_corpus",
        verified_dataset,
    )
    result = aws_operations.instantiate_aws(
        dataset_root=tmp_path / "dataset",
        pointer_path=ROOT / "DATASET-POINTER-AWS-135M-V3.json",
        transfer_manifest_path=MANIFEST,
        profile_path=PROFILE,
        runtime_root=tmp_path / "runtime",
        out_root=tmp_path / "outputs",
        repository_root=ROOT,
        seeds=(0,),
    )
    pair_path = result["pair_manifests"][0]
    pair_bytes = pair_path.read_bytes()
    role_manifest = json.loads(result["role_manifest"].read_text())
    assert role_manifest["seeds"] == list(range(10))
    assert len(role_manifest["configs"]) == 20

    expanded = aws_operations.instantiate_aws(
        dataset_root=tmp_path / "dataset",
        pointer_path=ROOT / "DATASET-POINTER-AWS-135M-V3.json",
        transfer_manifest_path=MANIFEST,
        profile_path=PROFILE,
        runtime_root=tmp_path / "runtime",
        out_root=tmp_path / "outputs",
        repository_root=ROOT,
        seeds=(0, 1),
    )
    assert pair_path.read_bytes() == pair_bytes
    assert [path.name for path in expanded["pair_manifests"]] == [
        "pair-s0.json",
        "pair-s1.json",
    ]
    with pytest.raises(FileExistsError, match="runtime config differs"):
        aws_operations.instantiate_aws(
            dataset_root=tmp_path / "dataset",
            pointer_path=ROOT / "DATASET-POINTER-AWS-135M-V3.json",
            transfer_manifest_path=MANIFEST,
            profile_path=PROFILE,
            runtime_root=tmp_path / "runtime",
            out_root=tmp_path / "different-outputs",
            repository_root=ROOT,
            seeds=(0,),
        )
    assert len(verification_calls) == 3

    pair = load_pair_manifest(pair_path)
    assert pair["cohort_id"] == COHORT_ID
    profile = load_profile(PROFILE)
    command = plan_sbatch(
        pair_path,
        profile=profile,
        action="train",
        mode="functional",
        venv_root=tmp_path / "venv",
    )
    assert command[0] == "sbatch"
    assert "--gres=gpu:2" in command
    assert f"--chdir={ROOT}" in command
    assert Path(command[-1]) == ROOT / "cluster/slurm/v2_pair_train.sbatch"
    submitted_commands = []

    def successful_submit(command):
        submitted_commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=f"Submitted batch job {len(submitted_commands)}",
            stderr="",
        )

    submitted = submit(
        [pair_path],
        profile_path=PROFILE,
        mode="functional",
        venv_root=tmp_path / "venv",
        apply=True,
        runner=successful_submit,
    )
    assert submitted["exit_code"] == 0
    for mode in ("resume", "throughput"):
        canary = submit(
            [pair_path],
            profile_path=PROFILE,
            mode=mode,
            venv_root=tmp_path / "venv",
            apply=True,
            runner=successful_submit,
        )
        assert canary["exit_code"] == 0

    for arm in pair["arms"]:
        output = Path(arm["out_dir"])
        output.mkdir(parents=True)
        torch.save(
            {"step": 100, "data": {"cursor": 52_428_800, "epoch": 0}},
            output / "ckpt.pt",
        )
    canary_root = tmp_path / "canaries"
    functional_evidence = _canary_evidence(
        canary_root / "functional.json",
        mode="functional",
        updates=1,
    )
    resume_evidence = _canary_evidence(
        canary_root / "resume.json",
        mode="resume",
        updates=2,
    )
    throughput_evidence = _canary_evidence(
        canary_root / "throughput.json",
        mode="throughput",
        updates=100,
    )
    preflight = build_preflight_receipt(
        profile=profile,
        dataset_receipt_sha256=VIRTUAL_RECEIPT_SHA256,
        functional_evidence=functional_evidence,
        resume_evidence=resume_evidence,
        throughput_evidence=throughput_evidence,
        output=tmp_path / "preflight.json",
        cohort_id=COHORT_ID,
    )
    validate_preflight(
        preflight,
        profile=profile,
        dataset_receipt_sha256=VIRTUAL_RECEIPT_SHA256,
        cohort_id=COHORT_ID,
    )

    protected = submit(
        [pair_path],
        profile_path=PROFILE,
        mode="protected",
        venv_root=tmp_path / "venv",
        preflight_path=preflight,
        apply=True,
        runner=successful_submit,
    )
    assert protected["exit_code"] == 0
    evaluated = evaluate(
        [pair_path],
        profile_path=PROFILE,
        mode="protected",
        venv_root=tmp_path / "venv",
        preflight_path=preflight,
        apply=True,
        runner=successful_submit,
    )
    assert evaluated["exit_code"] == 0
    assert Path(evaluated["commands"][0][-1]) == (
        ROOT / "cluster/slurm/v2_pair_evaluate.sbatch"
    )
    resumed = resume(
        pair_path,
        profile_path=PROFILE,
        venv_root=tmp_path / "venv",
        preflight_path=preflight,
        apply=True,
        runner=successful_submit,
    )
    assert resumed["dry_run"] is False
    assert resumed["exit_code"] == 0
    assert resumed["resume_state"]["step"] == 100
    assert len(submitted_commands) == 6

    evidence_root = tmp_path / "runtime" / "evidence"
    for action in ("train", "evaluate"):
        _write_json(
            evidence_root / f"{pair['pair_id']}-{action}-evidence.json",
            {"status": "completed"},
        )
    state = status([pair_path], evidence_root=evidence_root)
    assert state["pairs"][0]["status"] == "completed"
    collection = collect(
        [pair_path],
        evidence_root=evidence_root,
        output=tmp_path / "collection.json",
    )
    assert collection.is_file()


def test_reasoning_v3_operational_evaluation_crosses_segment_boundary(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ctx = 8
    tokens = [
        np.arange(64, dtype=np.uint16),
        np.arange(64, 128, dtype=np.uint16),
    ]
    masks = [
        np.ones(64, dtype=np.uint8),
        np.ones(64, dtype=np.uint8),
    ]
    token_paths, mask_paths = [], []
    for index, (token, mask) in enumerate(zip(tokens, masks)):
        token_path = tmp_path / f"tokens-{index}.bin"
        mask_path = tmp_path / f"weights-{index}.bin"
        token.tofile(token_path)
        mask.tofile(mask_path)
        token_paths.append(str(token_path))
        mask_paths.append(str(mask_path))
    model_cfg = {
        "n_layer": 1,
        "n_head": 2,
        "d_model": 32,
        "ctx": ctx,
        "vocab_size": 50304,
    }
    cfg = {
        "dataset": {"contract_id": "memorysplit-reasoning-dataset-v3"},
        "dataset_receipt_sha256": VIRTUAL_RECEIPT_SHA256,
        "max_steps": TERMINAL_UPDATES,
        "micro_batch_size": 2,
        "model": model_cfg,
        "run_id": "fixture",
        "seed": 0,
        "tokens_per_step": 524288,
        "total_tokens": RAW_TARGETS,
        "train_bin": token_paths,
        "train_mask": mask_paths,
    }
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = GPT(GPTConfig(**model_cfg))
    torch.save(
        {
            "data": {"cursor": RAW_TARGETS, "epoch": 1},
            "model": model.state_dict(),
            "step": TERMINAL_UPDATES,
        },
        run / "ckpt.pt",
    )
    summary = evaluate_run(run, device="cpu")
    assert summary["evaluation_scope"] == "operational_integrity_only"
    assert summary["base_target_tokens"] == 64
    assert summary["boundary_cursor"] == 56
    assert summary["targets_evaluated"] == 16
    assert summary["target_weight_sum"] == 16


def test_aws_execution_package_is_deterministic_and_contains_no_corpus(tmp_path):
    first = build_package(
        tmp_path / "first.zip",
        source_root=ROOT,
        require_clean=False,
    )
    second = build_package(
        tmp_path / "second.zip",
        source_root=ROOT,
        require_clean=False,
    )
    assert first.read_bytes() == second.read_bytes()
    report = verify_package(first, source_root=ROOT)
    assert report["verified"] is True
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        archive.extractall(unpacked)
    assert not any(name.startswith("corpus-build/") for name in names)
    assert "tests/test_slurm_135m.py" not in names
    assert "docs/AWS-135M-REASONING-V3-RUNBOOK.md" in names

    rebuilt = build_package(
        tmp_path / "rebuilt-from-release.zip",
        source_root=unpacked,
        require_clean=False,
    )
    assert rebuilt.read_bytes() == first.read_bytes()
    assert verify_package(rebuilt, source_root=unpacked)["verified"] is True


def _release_payload(tmp_path):
    archive = build_package(
        tmp_path / "closed.zip",
        source_root=ROOT,
        require_clean=False,
    )
    with zipfile.ZipFile(archive) as handle:
        return {name: handle.read(name) for name in handle.namelist()}


def test_closed_package_carries_every_required_execution_input(tmp_path):
    payload = _release_payload(tmp_path)
    required = {
        package_module.B200_PROFILE_PATH,
        package_module.EVAL_AUTHORITY_PATH,
        package_module.EVAL_BOUNDARY_PATH,
        package_module.EVAL_CONTRACT_PATH,
        package_module.EVAL_RELEASE_POINTER,
        "DATASET-POINTER-AWS-135M-V3.json",
        "cluster/aws/reasoning-v3-corpus-manifest.json",
        "docs/AWS-135M-REASONING-V3-RUNBOOK.md",
        "evals/reasoning_v3/inference.py",
        "evals/reasoning_v3/runner.py",
        "scripts/run_reasoning_v3_evals.py",
        "scripts/run_reasoning_v3_inference.py",
        "tests/test_aws_p6_b200_135m.py",
    }
    assert required <= set(payload)
    configs = role_config_paths("aws-operator")
    assert len(configs) == 20
    assert set(configs) <= set(payload)

    profile = json.loads(payload[package_module.B200_PROFILE_PATH])
    assert profile["hardware"]["instance_type"] == "p6-b200.48xlarge"
    assert profile["pair_geometry"] == {"gpus_per_pair": 2, "pairs_per_node": 4}


def test_closed_package_records_complete_provenance(tmp_path):
    payload = _release_payload(tmp_path)
    receipt = json.loads(payload["release-receipt.json"])
    assert len(receipt["source_revision"]) == 40
    assert len(receipt["source_tree"]) == 40
    assert receipt["corpus_bytes_included"] is False
    assert receipt["sealed_gold_included"] is False
    assert receipt["virtual_corpus_receipt_sha256"] == VIRTUAL_RECEIPT_SHA256
    assert receipt["transfer_manifest_sha256"] == TRANSFER_MANIFEST_SHA256
    assert receipt["member_sha256"] == {
        name: hashlib.sha256(payload[name]).hexdigest()
        for name in payload
        if name not in {"SHA256SUMS", "release-receipt.json"}
    }
    pointer = json.loads(payload[package_module.EVAL_RELEASE_POINTER])
    assert pointer["model_visible"]["bytes_included"] is False
    assert pointer["sealed_gold"]["bytes_included"] is False
    assert receipt["evaluator_code_commitments"] == (
        pointer["evaluator_code_commitments"]
    )
    for entry in receipt["evaluator_code_commitments"]:
        assert (
            hashlib.sha256(payload[entry["path"]]).hexdigest() == entry["sha256"]
        )


def test_closed_package_excludes_protected_bytes(tmp_path):
    payload = _release_payload(tmp_path)
    for name in payload:
        assert not name.startswith(("corpus-build/", "data/", "checkpoints/"))
        assert not name.endswith((".bin", ".npy", ".pt", ".safetensors"))
        assert ".git/" not in f"{name}/"
        assert "__pycache__" not in name
    # The frozen contract and the release pointer only *name* the sealed
    # fields they withhold; no other data member may mention them at all.
    declarations = {
        package_module.EVAL_CONTRACT_PATH,
        package_module.EVAL_RELEASE_POINTER,
    }
    for name, data in payload.items():
        if name in declarations or name.startswith("tests/"):
            continue
        if name.endswith((".json", ".yaml", ".yml")):
            assert b"canonical_answer" not in data


# The fixtures below are assembled at import time so the literal secrets and
# foreign account ids never appear as bytes inside this packaged test module.
_FORBIDDEN_MEMBERS = {
    "corpus_bytes": ("corpus-build/packed/targets.bin", b"corpus"),
    "sealed_gold": ("evaluations/gold.json", b'{"canonical' + b'_answer": "7"}'),
    "secret_key": (".env", b"AWS_SECRET" + b"_ACCESS_KEY=x"),
    "access_key": ("docs/leaked.md", b"AKIA" + b"IOSFODNN7EXAMPLE"),
    "foreign_account": ("docs/acct.md", b"arn:aws:iam::" + b"2109" + b"87654321:role/x"),
    "checkpoint": ("checkpoints/step0015582.pt", b"weights"),
    "cache": ("evals/__pycache__/runner.cpython-312.pyc", b"cached"),
}


@pytest.mark.parametrize("kind", sorted(_FORBIDDEN_MEMBERS))
def test_closed_package_fails_closed_on_unrecognized_members(kind):
    name, data = _FORBIDDEN_MEMBERS[kind]
    with pytest.raises(ValueError):
        package_module._reject_excluded({name: data})


def test_closed_package_rejects_oversized_members():
    oversized = b"0" * (package_module.MAX_MEMBER_BYTES + 1)
    with pytest.raises(ValueError, match="oversized"):
        package_module._reject_excluded({"docs/big.md": oversized})
