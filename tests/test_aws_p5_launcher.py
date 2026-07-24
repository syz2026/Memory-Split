from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
import yaml

import cluster.aws.p5.interruption_checkpoint as interruption_module
import cluster.aws.p5.bootstrap as bootstrap_module
import cluster.aws.p5.launch_seed_pair as launch_module
from cluster.aws.p5.bootstrap import (
    BootstrapError,
    build_bootstrap_receipt,
    inspect_hardware,
    publish_bootstrap_receipt,
    render_bootstrap_commands,
    verify_bootstrap_artifacts,
)
from cluster.aws.p5.interruption_checkpoint import (
    NON_RESUMABLE_EXIT_CODE,
    RESUMABLE_EXIT_CODE,
    CommandResult,
    ImdsV2Client,
    InterruptionRequest,
    S3ObjectStore,
    handle_interruption,
)
from cluster.aws.p5.launch_seed_pair import (
    LaunchError,
    load_launch_plan,
    preflight_trainer_contract,
    render_plan,
    render_trainer_preflight,
    supervise_pair,
)
from cluster.aws.p5.profile import (
    load_aws_p5_profile,
    validate_runtime_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge.json"
BOOTSTRAP_SH = ROOT / "cluster" / "aws" / "p5" / "bootstrap.sh"
HEX = {
    "release": "1" * 64,
    "cohort": "2" * 64,
    "ordered": "3" * 64,
}
CODE_COMMIT = "4" * 40
CONTAINER_IMAGE = (
    "public.ecr.aws/pytorch/pytorch-training:2.4.0-gpu-py311"
    "@sha256:"
    + "a" * 64
)
BOOT_ID = "01234567-89ab-4cde-8f01-23456789abcd"
RUNTIME_UID = os.getuid() or 1000
RUNTIME_GID = os.getgid() or 1000
SAFE_ENVIRONMENT = {
    "AWS_REGION": "us-east-1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
    "MS_S3_KMS_KEY_ID": (
        "arn:aws:kms:us-east-1:123456789012:"
        "key/12345678-1234-4234-9234-123456789abc"
    ),
    "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
    "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
    "MS_CONTAINER_IMAGE": CONTAINER_IMAGE,
    "MS_RUNTIME_GID": str(RUNTIME_GID),
    "MS_RUNTIME_UID": str(RUNTIME_UID),
}
H100_NAMES = ("NVIDIA H100 80GB HBM3",) * 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _write_config(path: Path, *, seed: int, arm: str) -> Path:
    sidecar = (
        "dense_target_weights" if arm == "dense" else "split90_target_weights"
    )
    value = {
        "schema_version": 2,
        "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
        "run_id": f"memorysplit-v2-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": "dataset/corpus-receipt.json",
        "sidecar_name": sidecar,
        "out_dir": f"runs/seed-{seed}/{arm}",
        "micro_batch_size": 8,
        "tokens_per_step": 524288,
        "max_steps": 13582,
        "total_tokens": 7120879616,
        "lr": 0.001,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": 0.1,
        "ckpt_minutes": 30,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _artifact(root: Path, relative: str, content: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "bytes": len(content),
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _sidecar_set(root: Path, name: str, chunks: list[bytes]) -> dict:
    artifacts = [
        _artifact(
            root,
            f"sidecars/{name}/shard-{index:05d}-of-{len(chunks):05d}.bin",
            chunk,
        )
        for index, chunk in enumerate(chunks)
    ]
    content = b"".join(chunks)
    return {
        "artifacts": artifacts,
        "dtype": "uint8",
        "items": len(content),
        "name": name,
        "stream_sha256": hashlib.sha256(content).hexdigest(),
    }


def _launcher_fixture(tmp_path: Path, seed: int = 1) -> dict[str, Path | dict]:
    scratch_root = tmp_path / "scratch"
    repo_root = scratch_root / "releases" / HEX["release"]
    repo_root.mkdir(parents=True)
    scratch_root.mkdir(parents=True, exist_ok=True)

    dataset = scratch_root / "dataset"
    assignments = [
        {
            "shard_count": 2,
            "shard_index": index,
            "token_end": (index + 1) * 4,
            "token_start": index * 4,
            "update_end": index + 1,
            "update_start": index,
        }
        for index in range(2)
    ]
    assignment_bytes = b"".join(
        (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        for value in assignments
    )
    foundation = {
        "assignments.jsonl": assignment_bytes,
        "catalog.jsonl": b'{"fixture":"catalog"}\n',
        "metadata.jsonl": b'{"fixture":"metadata"}\n',
        "schedule.jsonl": b'{"fixture":"schedule"}\n',
    }
    token_values = range(8)
    token_bytes = b"".join(value.to_bytes(2, "little") for value in token_values)
    token_chunks = [token_bytes[:8], token_bytes[8:]]
    artifacts = [
        *(
            _artifact(dataset, relative, content)
            for relative, content in foundation.items()
        ),
        *(
            _artifact(
                dataset,
                f"shards/shard-{index:05d}-of-00002.bin",
                content,
            )
            for index, content in enumerate(token_chunks)
        ),
    ]
    artifacts.sort(key=lambda artifact: artifact["path"])
    dense_chunks = [b"\x01" * 4, b"\x01\x01\x00\x00"]
    split_chunks = [b"\x01\x00\x01\x00", b"\x01\x00\x00\x00"]
    corpus = {
        "artifacts": artifacts,
        "assignments_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "build_id": "8" * 64,
        "catalog_sha256": hashlib.sha256(
            foundation["catalog.jsonl"]
        ).hexdigest(),
        "compiler_version": "metadata-first-foundation-v1",
        "config": {
            "allow_fewer_shards": False,
            "lane_weights": [{"lane": "natural", "weight": 1}],
            "shard_count": 2,
            "update_tokens": 4,
        },
        "format": "memorysplit-parallel-corpus-v2",
        "logical_tokens": 6,
        "merkle_root_sha256": "9" * 64,
        "metadata_sha256": hashlib.sha256(
            foundation["metadata.jsonl"]
        ).hexdigest(),
        "ordered_stream_sha256": HEX["ordered"],
        "packed_stream_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "packed_tokens": 8,
        "padding_tokens": 2,
        "record_count": 1,
        "renderer_id": "fixture-renderer-v1",
        "schedule_sha256": hashlib.sha256(
            foundation["schedule.jsonl"]
        ).hexdigest(),
        "shard_count": 2,
        "sidecar_sets": [
            _sidecar_set(
                dataset,
                "dense_target_weights",
                dense_chunks,
            ),
            _sidecar_set(
                dataset,
                "split90_target_weights",
                split_chunks,
            ),
        ],
    }
    corpus_path = _write_json(dataset / "receipt.json", corpus)

    configs = {}
    for arm in ("dense", "split90"):
        configs[arm] = _write_config(
            repo_root / "configs" / "360m-v2" / f"{arm}-s{seed}.yaml",
            seed=seed,
            arm=arm,
        )
    release_sources = {
        "scripts/run_train.py": (
            "import argparse\n"
            "import json\n"
            "import sys\n"
            "CAPABILITIES = {\n"
            "    'checkpoint_metadata': True,\n"
            "    'rank_zero_pid_file': True,\n"
            "    'receipt_v2': True,\n"
            "    'resume_sha256': True,\n"
            "    'sidecar_name': True,\n"
            "    'sigusr1_checkpoint': True,\n"
            "}\n"
            "if sys.argv[1:] == ['--capabilities-json']:\n"
            "    print(json.dumps(CAPABILITIES, sort_keys=True, separators=(',', ':')))\n"
            "    raise SystemExit(0)\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--resume-path')\n"
            "parser.add_argument('--resume-sha256')\n"
        ),
        "train/data.py": (
            "PARALLEL_SIDECAR_V2_CONTRACT = {'format': "
            "'memorysplit-parallel-corpus-v2'}\n"
            "def load(train_corpus, sidecar_name):\n"
            "    return train_corpus, sidecar_name\n"
        ),
        "train/trainer.py": (
            "import os, signal\n"
            "def train(cfg, *, resume_path=None, resume_sha256=None):\n"
            "    signal.signal(signal.SIGUSR1, lambda *_: None)\n"
            "    return os.environ['MS_RANK_ZERO_PID_FILE']\n"
        ),
    }
    for relative, content in release_sources.items():
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    member_paths = sorted(
        [
            *release_sources,
            *(
                path.relative_to(repo_root).as_posix()
                for path in configs.values()
            ),
        ]
    )
    member_rows = [
        {
            "bytes": (repo_root / relative).stat().st_size,
            "git_blob": "5" * 40,
            "git_mode": "100644",
            "path": relative,
            "sha256": _sha256(repo_root / relative),
        }
        for relative in member_paths
    ]
    release_metadata = {
        "members": member_rows,
        "package_format_version": 1,
        "provider": "aws-p5.48xlarge",
        "schema_version": 1,
        "seed_assignment": {
            "arms": ["dense", "split90"],
            "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
            "provider": "aws-p5.48xlarge",
            "seeds": [1, 2, 3, 4],
        },
        "source": {"commit": CODE_COMMIT, "dirty": False},
    }
    metadata_path = repo_root / "RELEASE-METADATA.json"
    metadata_path.write_text(
        json.dumps(release_metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    checksum_paths = sorted([*member_paths, "RELEASE-METADATA.json"])
    sums_path = repo_root / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{_sha256(repo_root / relative)}  {relative}\n"
            for relative in checksum_paths
        ),
        encoding="ascii",
    )
    release_members_sha256 = _sha256(sums_path)
    for path in repo_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (path for path in repo_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    repo_root.chmod(0o555)

    bootstrap = {
        "account_id": "123456789012",
        "ami_id": SAFE_ENVIRONMENT["MS_AWS_AMI_ID"],
        "boot_id": BOOT_ID,
        "code_commit": CODE_COMMIT,
        "cohort_assignment_sha256": HEX["cohort"],
        "container_image": CONTAINER_IMAGE,
        "container_digest": SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"],
        "corpus_build_id": corpus["build_id"],
        "corpus_receipt_sha256": _sha256(corpus_path),
        "durable_upload_verified": True,
        "instance_id": "i-0123456789abcdef0",
        "instance_store": {
            "device_bytes": 3_840_000_000_000,
            "devices": 8,
            "model": "Amazon EC2 NVMe Instance Storage",
            "raid_level": "0",
        },
        "instance_type": "p5.48xlarge",
        "profile_sha256": _sha256(PROFILE_PATH),
        "provider": "aws-p5.48xlarge",
        "receipt_type": "aws-p5-bootstrap",
        "region": SAFE_ENVIRONMENT["AWS_REGION"],
        "release_members_sha256": release_members_sha256,
        "release_root": f"releases/{HEX['release']}",
        "release_sha256": HEX["release"],
        "role_arn": (
            "arn:aws:sts::123456789012:"
            "assumed-role/MemorySplitP5Role/i-0123456789abcdef0"
        ),
        "role_name": "MemorySplitP5Role",
        "runtime_gid": RUNTIME_GID,
        "runtime_uid": RUNTIME_UID,
        "schema_version": 2,
        "scratch_root": "/mnt/memorysplit",
    }
    bootstrap_path = _write_json(
        scratch_root / "staging" / "bootstrap-receipt.json", bootstrap
    )

    base_port = 29500 + seed * 2
    runs = []
    for arm, port, cpus in (
        ("dense", base_port, [0, 95]),
        ("split90", base_port + 1, [96, 191]),
    ):
        config = configs[arm]
        runs.append(
            {
                "arm": arm,
                "checkpoint": f"runs/seed-{seed}/{arm}/run/ckpt.pt",
                "config": config.relative_to(repo_root).as_posix(),
                "config_sha256": _sha256(config),
                "cpu_affinity": cpus,
                "data_loader_workers": 16,
                "master_port": port,
                "rank_zero_pid_file": f"runs/seed-{seed}/{arm}/rank-zero.pid",
            }
        )
    manifest = {
        "bootstrap_receipt": {
            "path": bootstrap_path.relative_to(scratch_root).as_posix(),
            "sha256": _sha256(bootstrap_path),
        },
        "code_commit": CODE_COMMIT,
        "cohort_assignment_sha256": HEX["cohort"],
        "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
        "corpus_receipt": {
            "ordered_stream_sha256": HEX["ordered"],
            "path": corpus_path.relative_to(scratch_root).as_posix(),
            "sha256": _sha256(corpus_path),
        },
        "profile_sha256": _sha256(PROFILE_PATH),
        "provider": "aws-p5.48xlarge",
        "release_members_sha256": release_members_sha256,
        "release_sha256": HEX["release"],
        "runs": runs,
        "schema_version": 1,
        "seed": seed,
    }
    manifest_path = _write_json(
        scratch_root / "staging" / "run-manifest.json", manifest
    )
    return {
        "repo_root": repo_root,
        "scratch_root": scratch_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "corpus_path": corpus_path,
        "corpus": corpus,
        "bootstrap_path": bootstrap_path,
        "configs": configs,
        "metadata_path": metadata_path,
        "sums_path": sums_path,
        "release_members_sha256": release_members_sha256,
    }


def _load_fixture_plan(fixture: dict, *, seed: int | None = None, **kwargs):
    manifest = fixture["manifest"]
    kwargs.setdefault("observed_instance_id", "i-0123456789abcdef0")
    kwargs.setdefault("observed_boot_id", BOOT_ID)
    kwargs.setdefault("enforce_profile_scratch", False)
    return load_launch_plan(
        seed=manifest["seed"] if seed is None else seed,
        manifest_path=fixture["manifest_path"],
        profile_path=PROFILE_PATH,
        repo_root=fixture["repo_root"],
        scratch_root=fixture["scratch_root"],
        environment=SAFE_ENVIRONMENT,
        observed_instance_type="p5.48xlarge",
        gpu_names=H100_NAMES,
        port_available=lambda _port: True,
        semantic_corpus_verifier=lambda _root: fixture["corpus"],
        **kwargs,
    )


def _refresh_corpus_bindings(fixture: dict) -> None:
    _write_json(fixture["corpus_path"], fixture["corpus"])
    fixture["manifest"]["corpus_receipt"]["sha256"] = _sha256(
        fixture["corpus_path"]
    )
    bootstrap = json.loads(
        fixture["bootstrap_path"].read_text(encoding="utf-8")
    )
    bootstrap["corpus_receipt_sha256"] = _sha256(fixture["corpus_path"])
    _write_json(fixture["bootstrap_path"], bootstrap)
    fixture["manifest"]["bootstrap_receipt"]["sha256"] = _sha256(
        fixture["bootstrap_path"]
    )
    _write_json(fixture["manifest_path"], fixture["manifest"])


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_dry_run_renders_exact_symmetric_four_plus_four_commands(tmp_path, seed):
    fixture = _launcher_fixture(tmp_path, seed)
    report = render_plan(_load_fixture_plan(fixture))

    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["seed"] == seed
    dense, split90 = report["commands"]
    base_port = 29500 + seed * 2
    assert dense["arm"] == "dense"
    assert dense["argv"][0:3] == ["docker", "run", "--rm"]
    assert dense["argv"][-12:] == [
        "/opt/venv/bin/python",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=127.0.0.1:{base_port}",
        "/workspace/scripts/run_train.py",
        "--config",
        "/runtime/config.yaml",
        "--resume",
        "none",
    ]
    assert "--standalone" not in dense["argv"]
    assert "torchrun" not in dense["argv"]
    assert CONTAINER_IMAGE in dense["argv"]
    assert "device=0,1,2,3" in dense["argv"]
    assert "--read-only" in dense["argv"]
    assert "--network=host" in dense["argv"]
    assert "--ipc=host" in dense["argv"]
    assert "--pid=host" in dense["argv"]
    tmpfs = dense["argv"][dense["argv"].index("--tmpfs") + 1]
    assert "exec" in tmpfs.split(",")
    assert "noexec" not in tmpfs.split(",")
    assert ["--name", f"memorysplit-s{seed}-dense"] == dense["argv"][
        dense["argv"].index("--name") : dense["argv"].index("--name") + 2
    ]
    assert ["--cidfile", str(fixture["scratch_root"] / "staging" / "container-cids" / f"seed-{seed}" / "dense.cid")] == dense["argv"][
        dense["argv"].index("--cidfile") : dense["argv"].index("--cidfile") + 2
    ]
    assert str(
        fixture["scratch_root"] / "runs" / f"seed-{seed}" / "dense"
    ) not in dense["argv"][dense["argv"].index("--cidfile") + 1]
    assert ["--user", f"{RUNTIME_UID}:{RUNTIME_GID}"] == dense["argv"][
        dense["argv"].index("--user") : dense["argv"].index("--user") + 2
    ]
    assert (
        f"type=bind,src={fixture['repo_root']},dst=/workspace,readonly"
        in dense["argv"]
    )
    assert (
        f"type=bind,src={fixture['scratch_root'] / 'dataset'},"
        "dst=/dataset,readonly"
        in dense["argv"]
    )
    assert dense["cpu_affinity"] == [0, 95]
    assert split90["arm"] == "split90"
    assert split90["argv"][-6] == (
        f"--rdzv_endpoint=127.0.0.1:{base_port + 1}"
    )
    assert "device=4,5,6,7" in split90["argv"]
    dense_container_env = {
        dense["argv"][index + 1]
        for index, value in enumerate(dense["argv"])
        if value == "--env"
    }
    split_container_env = {
        split90["argv"][index + 1]
        for index, value in enumerate(split90["argv"])
        if value == "--env"
    }
    assert not any(
        value.startswith("CUDA_VISIBLE_DEVICES=")
        for value in dense_container_env | split_container_env
    )
    assert "device=0,1,2,3" in dense["argv"]
    assert "device=4,5,6,7" in split90["argv"]
    assert split90["cpu_affinity"] == [96, 191]
    assert dense["env"] == {"PATH": "/usr/bin:/bin"}
    assert split90["env"] == {"PATH": "/usr/bin:/bin"}
    assert dense["runtime_config"]["train_corpus"] == "/dataset"
    assert dense["runtime_config"]["out_dir"] == "/output/run"
    assert dense["runtime_config"]["sidecar_name"] == "dense_target_weights"
    assert split90["runtime_config"]["train_corpus"] == "/dataset"
    assert split90["runtime_config"]["out_dir"] == "/output/run"
    assert (
        split90["runtime_config"]["sidecar_name"]
        == "split90_target_weights"
    )
    assert len(dense["runtime_config_sha256"]) == 64
    assert len(dense["scientific_config_sha256"]) == 64
    assert not Path(dense["runtime_config_path"]).exists()
    assert not Path(split90["runtime_config_path"]).exists()
    assert not any(
        marker in name
        for command in report["commands"]
        for name in command["env"]
        for marker in ("SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")
    )


@pytest.mark.parametrize("seed", [0, 5])
def test_launcher_rejects_unassigned_seed(tmp_path, seed):
    fixture = _launcher_fixture(tmp_path)

    with pytest.raises(LaunchError, match="seed"):
        _load_fixture_plan(fixture, seed=seed)


def test_launcher_rejects_partial_pair(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    manifest["runs"].pop()
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(LaunchError, match="pair"):
        _load_fixture_plan(fixture)


def test_launcher_rejects_non_utf8_manifest(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    fixture["manifest_path"].write_bytes(
        json.dumps(fixture["manifest"]).encode("utf-16")
    )

    with pytest.raises(LaunchError, match="UTF-8"):
        _load_fixture_plan(fixture)


def test_launcher_rejects_equal_or_occupied_ports(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    manifest["runs"][1]["master_port"] = manifest["runs"][0]["master_port"]
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(LaunchError, match="port"):
        _load_fixture_plan(fixture)

    _write_json(fixture["manifest_path"], fixture["manifest"])
    occupied = fixture["manifest"]["runs"][0]["master_port"]
    with pytest.raises(LaunchError, match="occupied"):
        load_launch_plan(
            seed=1,
            manifest_path=fixture["manifest_path"],
            profile_path=PROFILE_PATH,
            repo_root=fixture["repo_root"],
            scratch_root=fixture["scratch_root"],
            environment=SAFE_ENVIRONMENT,
            observed_instance_type="p5.48xlarge",
            observed_instance_id="i-0123456789abcdef0",
            observed_boot_id=BOOT_ID,
            gpu_names=H100_NAMES,
            port_available=lambda port: port != occupied,
            semantic_corpus_verifier=lambda _root: fixture["corpus"],
            enforce_profile_scratch=False,
        )


@pytest.mark.parametrize(
    ("instance_type", "gpu_names"),
    [
        ("p5.4xlarge", H100_NAMES),
        ("p5.48xlarge", ("NVIDIA A100-SXM4-80GB",) * 8),
        ("p5.48xlarge", H100_NAMES[:7]),
    ],
)
def test_launcher_fails_closed_on_wrong_instance_or_gpus(
    tmp_path, instance_type, gpu_names
):
    fixture = _launcher_fixture(tmp_path)

    with pytest.raises(LaunchError, match="p5.48xlarge|H100|eight"):
        load_launch_plan(
            seed=1,
            manifest_path=fixture["manifest_path"],
            profile_path=PROFILE_PATH,
            repo_root=fixture["repo_root"],
            scratch_root=fixture["scratch_root"],
            environment=SAFE_ENVIRONMENT,
            observed_instance_type=instance_type,
            observed_instance_id="i-0123456789abcdef0",
            observed_boot_id=BOOT_ID,
            gpu_names=gpu_names,
            port_available=lambda _port: True,
            semantic_corpus_verifier=lambda _root: fixture["corpus"],
            enforce_profile_scratch=False,
        )


def test_launcher_rejects_stale_output_before_starting_processes(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    stale = fixture["scratch_root"] / "runs" / "seed-1" / "dense"
    stale.mkdir(parents=True)

    with pytest.raises(LaunchError, match="output"):
        _load_fixture_plan(fixture)


def test_launcher_rejects_wrong_config_or_corpus_hash(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    manifest = deepcopy(fixture["manifest"])
    manifest["runs"][0]["config_sha256"] = "0" * 64
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(LaunchError, match="config"):
        _load_fixture_plan(fixture)

    manifest = deepcopy(fixture["manifest"])
    manifest["corpus_receipt"]["sha256"] = "0" * 64
    _write_json(fixture["manifest_path"], manifest)
    with pytest.raises(LaunchError, match="corpus"):
        _load_fixture_plan(fixture)


def test_launcher_preserves_scientific_config_while_replacing_locations(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    report = render_plan(_load_fixture_plan(fixture))

    for command in report["commands"]:
        source = yaml.safe_load(
            fixture["configs"][command["arm"]].read_text(encoding="utf-8")
        )
        runtime = command["runtime_config"]
        assert {
            key: value
            for key, value in runtime.items()
            if key not in {"train_corpus", "out_dir"}
        } == {
            key: value
            for key, value in source.items()
            if key not in {"train_corpus", "out_dir"}
        }
        assert runtime["train_corpus"] == "/dataset"
        assert runtime["out_dir"] == "/output/run"


def test_trainer_contract_preflight_runs_inside_pinned_container(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    calls = []

    def runner(argv, environment, timeout):
        calls.append((list(argv), dict(environment), timeout))
        completed = subprocess.run(
            [
                sys.executable,
                fixture["repo_root"] / "scripts" / "run_train.py",
                "--capabilities-json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    preflight_trainer_contract(plan, runner=runner, timeout_seconds=90)

    assert len(calls) == 1
    argv, environment, timeout = calls[0]
    assert argv == list(render_trainer_preflight(plan))
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert CONTAINER_IMAGE in argv
    assert argv[-3:] == [
        "/opt/venv/bin/python",
        "/workspace/scripts/run_train.py",
        "--capabilities-json",
    ]
    assert "-c" not in argv
    assert (
        f"type=bind,src={fixture['repo_root']},dst=/workspace,readonly"
        in argv
    )
    assert environment == {"PATH": "/usr/bin:/bin"}
    assert timeout == 90


def test_trainer_contract_preflight_fails_before_any_output_or_spawn(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    spawner = _FakeSpawner(
        {
            "dense": _FakeProcess(101, [0]),
            "split90": _FakeProcess(202, [0]),
        }
    )

    def reject(_plan):
        raise LaunchError("integrated trainer lacks SIGUSR1 checkpoint support")

    with pytest.raises(LaunchError, match="SIGUSR1"):
        supervise_pair(
            plan,
            spawner=spawner,
            sleep=lambda _delay: None,
            trainer_preflight=reject,
        )

    assert spawner.started == []
    assert all(not launch.out_dir.exists() for launch in plan.arms)
    assert all(
        not launch.runtime_config_path.exists() for launch in plan.arms
    )


def test_trainer_contract_preflight_rejects_missing_capability(tmp_path):
    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    contract = {
        "checkpoint_metadata": True,
        "rank_zero_pid_file": True,
        "receipt_v2": True,
        "resume_sha256": True,
        "sidecar_name": True,
        "sigusr1_checkpoint": False,
    }

    with pytest.raises(LaunchError, match="trainer contract|SIGUSR1"):
        preflight_trainer_contract(
            plan,
            runner=lambda _argv, _environment, _timeout: CommandResult(
                0,
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
                + "\n",
                "",
            ),
        )


def test_trainer_preflight_rejects_source_tokens_without_behavior(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    script = fixture["repo_root"] / "scripts" / "run_train.py"
    script.chmod(0o644)
    script.write_text(
        "# --capabilities-json receipt_v2 sidecar_name resume_sha256\n"
        "# rank_zero_pid_file SIGUSR1 checkpoint\n",
        encoding="utf-8",
    )

    def run_fixture(_argv, _environment, timeout):
        completed = subprocess.run(
            [sys.executable, script, "--capabilities-json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    with pytest.raises(LaunchError, match="invalid JSON|capability"):
        preflight_trainer_contract(plan, runner=run_fixture)


def test_trainer_preflight_rejects_duplicate_or_noncanonical_json(tmp_path):
    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    duplicate = (
        '{"checkpoint_metadata":true,"rank_zero_pid_file":true,'
        '"rank_zero_pid_file":true,'
        '"receipt_v2":true,"resume_sha256":true,"sidecar_name":true,'
        '"sigusr1_checkpoint":true}\n'
    )

    with pytest.raises(LaunchError, match="JSON|evidence"):
        preflight_trainer_contract(
            plan,
            runner=lambda _argv, _environment, _timeout: CommandResult(
                0,
                duplicate,
                "",
            ),
        )


def test_launcher_recomputes_every_verified_release_member(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    trainer = fixture["repo_root"] / "train" / "trainer.py"
    trainer.chmod(0o644)
    trainer.write_text("print('tampered')\n", encoding="utf-8")

    with pytest.raises(LaunchError, match="release|member|SHA-256"):
        _load_fixture_plan(fixture)


def test_launcher_rejects_release_root_or_member_binding_drift(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    receipt = json.loads(fixture["bootstrap_path"].read_text(encoding="utf-8"))
    receipt["release_root"] = "releases/" + "0" * 64
    _write_json(fixture["bootstrap_path"], receipt)
    fixture["manifest"]["bootstrap_receipt"]["sha256"] = _sha256(
        fixture["bootstrap_path"]
    )
    _write_json(fixture["manifest_path"], fixture["manifest"])
    with pytest.raises(LaunchError, match="release root|release"):
        _load_fixture_plan(fixture)

    fixture = _launcher_fixture(tmp_path / "members")
    fixture["manifest"]["release_members_sha256"] = "0" * 64
    _write_json(fixture["manifest_path"], fixture["manifest"])
    with pytest.raises(LaunchError, match="release member"):
        _load_fixture_plan(fixture)


def test_launcher_requires_exact_profile_scratch_root(tmp_path):
    fixture = _launcher_fixture(tmp_path)

    with pytest.raises(LaunchError, match="/mnt/memorysplit|scratch"):
        _load_fixture_plan(fixture, enforce_profile_scratch=True)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("observed_instance_id", "i-0fedcba9876543210", "instance"),
        (
            "observed_boot_id",
            "fedcba98-7654-4cba-8012-3456789abcde",
            "boot",
        ),
    ],
)
def test_launcher_rejects_bootstrap_receipt_from_another_boot(
    tmp_path,
    argument,
    value,
    message,
):
    fixture = _launcher_fixture(tmp_path)

    with pytest.raises(LaunchError, match=message):
        _load_fixture_plan(fixture, **{argument: value})


def test_launcher_rejects_missing_or_tampered_sidecar(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    sidecar = (
        fixture["scratch_root"]
        / "dataset"
        / "sidecars"
        / "split90_target_weights"
        / "shard-00000-of-00002.bin"
    )
    sidecar.unlink()

    with pytest.raises(LaunchError, match="sidecar"):
        _load_fixture_plan(fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy-sidecar-mapping",
        "reordered-sidecar-sets",
        "reordered-sidecar-artifacts",
        "wrong-sidecar-stream",
        "reordered-primary-artifacts",
    ],
)
def test_launcher_rejects_noncanonical_task4_receipt_shapes(tmp_path, mutation):
    fixture = _launcher_fixture(tmp_path)
    corpus = fixture["corpus"]
    if mutation == "legacy-sidecar-mapping":
        corpus["sidecar_sets"] = {
            value["name"]: value for value in corpus["sidecar_sets"]
        }
    elif mutation == "reordered-sidecar-sets":
        corpus["sidecar_sets"].reverse()
    elif mutation == "reordered-sidecar-artifacts":
        corpus["sidecar_sets"][1]["artifacts"].reverse()
    elif mutation == "wrong-sidecar-stream":
        corpus["sidecar_sets"][1]["stream_sha256"] = "0" * 64
    else:
        corpus["artifacts"].reverse()
    _refresh_corpus_bindings(fixture)

    with pytest.raises(
        LaunchError,
        match="canonical|corpus|sidecar|artifact|order|stream",
    ):
        _load_fixture_plan(fixture)


@pytest.mark.parametrize(
    "relative",
    [
        "shards/shard-00000-of-00002.bin",
        "sidecars/dense_target_weights/shard-00000-of-00002.bin",
        "sidecars/split90_target_weights/shard-00001-of-00002.bin",
    ],
)
def test_launcher_recomputes_every_task4_artifact_hash(tmp_path, relative):
    fixture = _launcher_fixture(tmp_path)
    artifact = fixture["scratch_root"] / "dataset" / relative
    artifact.write_bytes(artifact.read_bytes() + b"\xff")

    with pytest.raises(LaunchError, match="artifact|corpus|sidecar|digest|bytes"):
        _load_fixture_plan(fixture)


def test_launcher_recomputes_sidecar_ordered_stream_hash(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    sidecar_set = fixture["corpus"]["sidecar_sets"][1]
    artifact_record = sidecar_set["artifacts"][0]
    artifact = fixture["scratch_root"] / "dataset" / artifact_record["path"]
    payload = b"\x00" + artifact.read_bytes()[1:]
    artifact.write_bytes(payload)
    artifact_record["bytes"] = len(payload)
    artifact_record["sha256"] = hashlib.sha256(payload).hexdigest()
    _refresh_corpus_bindings(fixture)

    with pytest.raises(LaunchError, match="stream|sidecar|corpus"):
        _load_fixture_plan(fixture)


def test_launcher_rejects_extra_or_symlinked_task4_artifacts(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    extra = fixture["scratch_root"] / "dataset" / "shards" / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(LaunchError, match="extra|namespace|corpus"):
        _load_fixture_plan(fixture)

    extra.unlink()
    sidecar = (
        fixture["scratch_root"]
        / "dataset"
        / fixture["corpus"]["sidecar_sets"][0]["artifacts"][0]["path"]
    )
    target = tmp_path / "outside.bin"
    sidecar.replace(target)
    sidecar.symlink_to(target)
    with pytest.raises(LaunchError, match="unsafe|symlink|sidecar|corpus"):
        _load_fixture_plan(fixture)


def test_launcher_requires_canonical_task4_semantic_verifier(tmp_path):
    fixture = _launcher_fixture(tmp_path)

    with pytest.raises(LaunchError, match="Task 4|semantic|canonical"):
        load_launch_plan(
            seed=1,
            manifest_path=fixture["manifest_path"],
            profile_path=PROFILE_PATH,
            repo_root=fixture["repo_root"],
            scratch_root=fixture["scratch_root"],
            environment=SAFE_ENVIRONMENT,
            observed_instance_type="p5.48xlarge",
            observed_instance_id="i-0123456789abcdef0",
            observed_boot_id=BOOT_ID,
            gpu_names=H100_NAMES,
            port_available=lambda _port: True,
            enforce_profile_scratch=False,
        )


def test_launcher_rejects_unverified_bootstrap_or_inherited_secret(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    receipt = json.loads(fixture["bootstrap_path"].read_text(encoding="utf-8"))
    receipt["durable_upload_verified"] = False
    _write_json(fixture["bootstrap_path"], receipt)
    manifest = deepcopy(fixture["manifest"])
    manifest["bootstrap_receipt"]["sha256"] = _sha256(fixture["bootstrap_path"])
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(LaunchError, match="bootstrap"):
        _load_fixture_plan(fixture)

    fixture = _launcher_fixture(tmp_path / "secret")
    environment = dict(SAFE_ENVIRONMENT)
    environment["AWS_SECRET_ACCESS_KEY"] = "must-not-leak"
    with pytest.raises(LaunchError, match="secret"):
        load_launch_plan(
            seed=1,
            manifest_path=fixture["manifest_path"],
            profile_path=PROFILE_PATH,
            repo_root=fixture["repo_root"],
            scratch_root=fixture["scratch_root"],
            environment=environment,
            observed_instance_type="p5.48xlarge",
            observed_instance_id="i-0123456789abcdef0",
            observed_boot_id=BOOT_ID,
            gpu_names=H100_NAMES,
            port_available=lambda _port: True,
            semantic_corpus_verifier=lambda _root: fixture["corpus"],
            enforce_profile_scratch=False,
        )


class _FakeProcess:
    def __init__(
        self,
        pid: int,
        polls: list[int | None],
        *,
        waited: int = -15,
        on_terminate=None,
        stop_leaves_container_running: bool = False,
    ):
        self.pid = pid
        self._polls = list(polls)
        self._last = polls[-1]
        self.waited = waited
        self.terminated = False
        self.on_terminate = on_terminate
        self.container_running = True
        self.container_pinned = False
        self.stop_leaves_container_running = stop_leaves_container_running
        self.container_events = []

    def poll(self):
        if self._polls:
            self._last = self._polls.pop(0)
        return self._last

    def wait(self, timeout=None):
        del timeout
        return self.waited if self.terminated else int(self._last or 0)

    def terminate_tree(self):
        if self.on_terminate is not None:
            self.on_terminate()
        self.terminated = True

    def kill_tree(self):
        self.terminated = True

    def stop_container(self, timeout):
        assert timeout > 0
        self.container_events.append("stop")
        if self.on_terminate is not None:
            self.on_terminate()
        if not self.stop_leaves_container_running:
            self.container_running = False

    def pin_container(self, timeout):
        assert timeout > 0
        self.container_pinned = True

    def kill_container(self, timeout):
        assert timeout > 0
        self.container_events.append("kill")
        if self.on_terminate is not None:
            self.on_terminate()
        self.container_running = False

    def container_stopped(self, timeout):
        assert timeout > 0
        self.container_events.append("verify")
        if self.on_terminate is not None:
            self.on_terminate()
        return not self.container_running


class _FakeSpawner:
    def __init__(self, processes: dict[str, _FakeProcess | Exception]):
        self.processes = processes
        self.started = []

    def __call__(self, launch):
        self.started.append(launch)
        process = self.processes[launch.arm]
        if isinstance(process, Exception):
            raise process
        return process


def _pass_trainer_preflight(_plan):
    return None


def _pass_rank_zero_resolver(_plan, child_pids):
    return dict(child_pids)


def test_supervisor_records_both_pids_and_accepts_only_paired_success(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    spawner = _FakeSpawner(
        {
            "dense": _FakeProcess(101, [None, 0]),
            "split90": _FakeProcess(202, [None, 0]),
        }
    )

    result = supervise_pair(
        plan,
        spawner=spawner,
        sleep=lambda _delay: None,
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=_pass_rank_zero_resolver,
    )

    assert result.status == "completed"
    assert result.returncode == 0
    assert result.child_pids == {"dense": 101, "split90": 202}
    assert [launch.arm for launch in spawner.started] == ["dense", "split90"]
    assert all(launch.out_dir.is_dir() for launch in spawner.started)


def test_supervisor_pins_both_container_ids_before_rank_zero_resolution(tmp_path):
    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    dense = _FakeProcess(101, [None, 0])
    split90 = _FakeProcess(202, [None, 0])

    def resolve(_plan, child_pids):
        assert dense.container_pinned is True
        assert split90.container_pinned is True
        return dict(child_pids)

    result = supervise_pair(
        plan,
        spawner=_FakeSpawner({"dense": dense, "split90": split90}),
        sleep=lambda _delay: None,
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=resolve,
    )

    assert result.status == "completed"


def test_supervisor_rejects_a_second_seed_pair_on_the_same_p5(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    lock_path = fixture["scratch_root"] / ".p5-seed-pair.lock"
    lock_path.touch(mode=0o600)

    with lock_path.open("r+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(LaunchError, match="another seed pair"):
            supervise_pair(
                plan,
                spawner=_FakeSpawner(
                    {
                        "dense": _FakeProcess(101, [0]),
                        "split90": _FakeProcess(202, [0]),
                    }
                ),
                sleep=lambda _delay: None,
                trainer_preflight=_pass_trainer_preflight,
                rank_zero_resolver=_pass_rank_zero_resolver,
            )


def test_supervisor_preflights_both_output_directories_before_spawning(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    plan.arms[1].out_dir.mkdir(parents=True)
    dense = _FakeProcess(101, [None])
    spawner = _FakeSpawner(
        {
            "dense": dense,
            "split90": _FakeProcess(202, [0]),
        }
    )

    with pytest.raises(LaunchError, match="output"):
        supervise_pair(
            plan,
            spawner=spawner,
            sleep=lambda _delay: None,
            trainer_preflight=_pass_trainer_preflight,
            rank_zero_resolver=_pass_rank_zero_resolver,
        )

    assert spawner.started == []
    assert dense.terminated is False


def test_supervisor_propagates_failure_and_terminates_peer(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    dense = _FakeProcess(
        101,
        [7],
        stop_leaves_container_running=True,
    )
    split90 = _FakeProcess(
        202,
        [None],
        stop_leaves_container_running=True,
    )
    spawner = _FakeSpawner({"dense": dense, "split90": split90})

    result = supervise_pair(
        plan,
        spawner=spawner,
        sleep=lambda _delay: None,
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=_pass_rank_zero_resolver,
    )

    assert result.status == "failed"
    assert result.returncode == 7
    assert result.failed_arm == "dense"
    assert result.peer_terminated is True
    assert split90.terminated is True
    assert dense.container_events == ["stop", "verify", "kill", "verify"]
    assert split90.container_events == ["stop", "verify", "kill", "verify"]


def test_cleanup_processes_every_known_cid_after_clients_exit():
    dense = _FakeProcess(
        101,
        [7],
        stop_leaves_container_running=True,
    )
    split90 = _FakeProcess(
        202,
        [0],
        stop_leaves_container_running=True,
    )

    launch_module._terminate_all((dense, split90))

    assert dense.container_events == ["stop", "verify", "kill", "verify"]
    assert split90.container_events == ["stop", "verify", "kill", "verify"]


def test_supervisor_terminates_first_arm_when_second_spawn_fails(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    dense = _FakeProcess(101, [None])
    spawner = _FakeSpawner(
        {"dense": dense, "split90": RuntimeError("spawn failed")}
    )

    with pytest.raises(LaunchError, match="spawn"):
        supervise_pair(
            plan,
            spawner=spawner,
            sleep=lambda _delay: None,
            trainer_preflight=_pass_trainer_preflight,
            rank_zero_resolver=_pass_rank_zero_resolver,
        )

    assert dense.terminated is True


def test_supervisor_fail_stops_both_arms_when_interruption_handling_fails(
    tmp_path,
):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    dense = _FakeProcess(101, [None])
    split90 = _FakeProcess(202, [None])
    spawner = _FakeSpawner({"dense": dense, "split90": split90})

    with pytest.raises(LaunchError, match="interruption"):
        supervise_pair(
            plan,
            spawner=spawner,
            sleep=lambda _delay: None,
            notice_source=lambda: "rebalance-recommendation",
            interruption_handler=lambda _plan, _pids, _notice: (_ for _ in ()).throw(
                RuntimeError("receipt failed")
            ),
            trainer_preflight=_pass_trainer_preflight,
            rank_zero_resolver=_pass_rank_zero_resolver,
        )

    assert dense.terminated is True
    assert split90.terminated is True


def test_supervisor_requires_both_preexisting_rank_zero_pid_files(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    dense = _FakeProcess(101, [None])
    split90 = _FakeProcess(202, [None])
    spawner = _FakeSpawner({"dense": dense, "split90": split90})

    with pytest.raises(LaunchError, match="rank-zero PID"):
        supervise_pair(
            plan,
            spawner=spawner,
            sleep=lambda _delay: None,
            trainer_preflight=_pass_trainer_preflight,
            rank_zero_resolver=lambda _plan, _pids: (_ for _ in ()).throw(
                LaunchError("rank-zero PID files were not created")
            ),
        )

    assert dense.terminated is True
    assert split90.terminated is True


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_supervisor_holds_lock_while_signal_stops_both_process_groups(
    tmp_path,
    signum,
):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    lock_path = fixture["scratch_root"] / ".p5-seed-pair.lock"
    lock_observations = []

    def assert_lock_held():
        with lock_path.open("r+b") as contender:
            try:
                fcntl.flock(
                    contender.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                lock_observations.append(True)
            else:
                fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
                lock_observations.append(False)

    dense = _FakeProcess(101, [None], on_terminate=assert_lock_held)
    split90 = _FakeProcess(202, [None], on_terminate=assert_lock_held)
    result = supervise_pair(
        plan,
        spawner=_FakeSpawner({"dense": dense, "split90": split90}),
        sleep=lambda _delay: None,
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=_pass_rank_zero_resolver,
        shutdown_source=lambda: signum,
    )

    assert result.status == "terminated"
    assert result.returncode == 128 + signum
    assert dense.terminated is True
    assert split90.terminated is True
    assert dense.container_events == ["stop", "verify"]
    assert split90.container_events == ["stop", "verify"]
    assert lock_observations
    assert all(lock_observations)


def test_signal_shutdown_kills_running_containers_before_unlock(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    lock_path = fixture["scratch_root"] / ".p5-seed-pair.lock"
    lock_observations = []

    def observe_lock():
        with lock_path.open("r+b") as contender:
            try:
                fcntl.flock(
                    contender.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                lock_observations.append(True)
            else:
                fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
                lock_observations.append(False)

    dense = _FakeProcess(
        101,
        [None],
        on_terminate=observe_lock,
        stop_leaves_container_running=True,
    )
    split90 = _FakeProcess(
        202,
        [None],
        on_terminate=observe_lock,
        stop_leaves_container_running=True,
    )
    result = supervise_pair(
        plan,
        spawner=_FakeSpawner({"dense": dense, "split90": split90}),
        sleep=lambda _delay: None,
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=_pass_rank_zero_resolver,
        shutdown_source=lambda: signal.SIGTERM,
    )

    assert result.status == "terminated"
    assert dense.container_events == ["stop", "verify", "kill", "verify"]
    assert split90.container_events == ["stop", "verify", "kill", "verify"]
    assert lock_observations and all(lock_observations)


def test_docker_cleanup_uses_full_cidfile_id_and_bounded_argv(tmp_path):
    cid = "a" * 64
    cidfile = tmp_path / "docker.cid"
    cidfile.write_text(cid + "\n", encoding="ascii")
    calls = []
    inspections = iter(["running\n", "exited\n"])

    def runner(argv, environment, timeout):
        calls.append((list(argv), dict(environment), timeout))
        if argv[1] == "inspect":
            return CommandResult(0, next(inspections), "")
        return CommandResult(0, "", "")

    container = launch_module._DockerContainerHandle(
        cidfile,
        runner=runner,
    )
    container.stop_container(7.0)
    assert container.container_stopped(7.0) is False
    container.kill_container(7.0)
    assert container.container_stopped(7.0) is True

    assert [call[0][1] for call in calls] == [
        "stop",
        "inspect",
        "kill",
        "inspect",
    ]
    assert all(call[0][-1] == cid for call in calls)
    assert all(call[1] == {"PATH": "/usr/bin:/bin"} for call in calls)
    assert all(0 < call[2] <= 7.0 for call in calls)


def test_docker_cleanup_does_not_treat_daemon_failure_as_absent(tmp_path):
    cidfile = tmp_path / "docker.cid"
    cidfile.write_text("a" * 64 + "\n", encoding="ascii")
    container = launch_module._DockerContainerHandle(
        cidfile,
        runner=lambda _argv, _environment, _timeout: CommandResult(
            1,
            "",
            "Cannot connect to the Docker daemon",
        ),
    )

    with pytest.raises(LaunchError, match="inspect|verify|daemon"):
        container.container_stopped(2.0)


def test_installed_shutdown_handlers_capture_and_restore_all_signals(monkeypatch):
    installed = {}
    restored = []

    monkeypatch.setattr(signal, "getsignal", lambda signum: f"old-{signum}")

    def install(signum, handler):
        if isinstance(handler, str):
            restored.append((signum, handler))
        else:
            installed[signum] = handler

    monkeypatch.setattr(signal, "signal", install)

    with launch_module.installed_shutdown_handlers() as source:
        assert set(installed) == {signal.SIGTERM, signal.SIGHUP, signal.SIGINT}
        installed[signal.SIGHUP](signal.SIGHUP, None)
        assert source() == signal.SIGHUP

    assert set(restored) == {
        (signal.SIGTERM, f"old-{signal.SIGTERM}"),
        (signal.SIGHUP, f"old-{signal.SIGHUP}"),
        (signal.SIGINT, f"old-{signal.SIGINT}"),
    }


class _FakeStore:
    def __init__(self, *, fail_contains: str | None = None, on_put=None):
        self.fail_contains = fail_contains
        self.on_put = on_put
        self.calls = []

    def put_verified(
        self,
        path: Path,
        uri: str,
        *,
        expected_sha256: str,
        deadline=float("inf"),
        monotonic=time.monotonic,
        wall_deadline=float("inf"),
        wall_monotonic=time.monotonic,
    ):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self.calls.append(
            (
                path,
                uri,
                deadline,
                monotonic(),
                expected_sha256,
                payload,
                path.stat().st_mode,
            )
        )
        assert digest == expected_sha256
        if any(
            marker in uri
            for marker in (
                "/checkpoints/",
                "/evidence/sha256/",
                "/resume-commits/",
            )
        ):
            assert path.stat().st_mode & 0o222 == 0
        if self.on_put is not None:
            self.on_put(path, uri)
        if monotonic() >= deadline or wall_monotonic() >= wall_deadline:
            return None
        if self.fail_contains is not None and self.fail_contains in uri:
            return None
        return interruption_module.UploadedObject(
            uri=uri,
            sha256=digest,
            bytes=len(payload),
        )


def _interruption_request(tmp_path: Path) -> InterruptionRequest:
    return InterruptionRequest(
        seed=1,
        notice="spot-instance-action",
        rank_zero_pids={"dense": 101, "split90": 202},
        checkpoint_paths={
            "dense": tmp_path / "dense.pt",
            "split90": tmp_path / "split90.pt",
        },
        s3_root="s3://memorysplit-prod/cohort-v2",
        receipt_path=tmp_path / "interruption-receipt.json",
        release_sha256=HEX["release"],
        corpus_receipt_sha256="5" * 64,
        code_commit=CODE_COMMIT,
        config_sha256={"dense": "6" * 64, "split90": "7" * 64},
        timeout_seconds=5.0,
        upload_reserve_seconds=2.0,
    )


def _atomic_checkpoint(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.next")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


@pytest.mark.parametrize(
    ("timeout_seconds", "upload_reserve_seconds"),
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (5.0, float("nan")),
        (5.0, float("inf")),
    ],
)
def test_interruption_rejects_nonfinite_deadlines(
    tmp_path,
    timeout_seconds,
    upload_reserve_seconds,
):
    request = _interruption_request(tmp_path)
    values = dict(request.__dict__)
    values["timeout_seconds"] = timeout_seconds
    values["upload_reserve_seconds"] = upload_reserve_seconds

    with pytest.raises(ValueError, match="timeout|reserve"):
        InterruptionRequest(**values)


def test_interruption_signals_both_rank_zero_processes_and_emits_paired_receipt(
    tmp_path,
):
    request = _interruption_request(tmp_path)
    pid_to_arm = {101: "dense", 202: "split90"}
    signals = []

    def signal_process(pid, signum):
        signals.append((pid, signum))
        arm = pid_to_arm[pid]
        _atomic_checkpoint(
            request.checkpoint_paths[arm],
            f"{arm}-checkpoint".encode(),
        )

    store = _FakeStore()
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        sleep=lambda _delay: None,
    )

    assert signals == [(101, signal.SIGUSR1), (202, signal.SIGUSR1)]
    assert result.resumable is True
    assert result.exit_code == RESUMABLE_EXIT_CODE
    assert result.receipt_upload_verified is True
    raw = request.receipt_path.read_bytes()
    receipt = json.loads(raw)
    assert raw == (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert receipt["receipt_type"] == "aws-p5-paired-interruption"
    assert receipt["seed"] == 1
    assert "resumable" not in receipt
    assert receipt["protocol"] == "aws-p5-resume-commit-v1"
    candidate_call = next(
        call for call in store.calls if "/evidence/seed-1/sha256/" in call[1]
    )
    candidate = json.loads(candidate_call[5])
    assert [item["arm"] for item in candidate["checkpoints"]] == [
        "dense",
        "split90",
    ]
    assert all(
        item["object_uri"].endswith(f"/sha256/{item['sha256']}.pt")
        for item in candidate["checkpoints"]
    )
    assert len(store.calls) == 4


@pytest.mark.parametrize(
    "profile_name",
    ["aws-p5.48xlarge-v3", "aws-p6-b300.48xlarge-v3"],
)
def test_v3_interruption_uses_neutral_profile_bound_receipt_protocol(
    tmp_path,
    profile_name,
):
    legacy = _interruption_request(tmp_path)
    profile = load_aws_p5_profile(
        ROOT / "cluster" / "profiles" / f"{profile_name}.json"
    )
    values = dict(legacy.__dict__)
    values.update(
        {
            "seed": 0,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
            "instance_type": profile.instance_type,
            "gres": profile.gres,
            "assigned_seeds": profile.assigned_seeds,
            "candidate_receipt_type": (
                profile.interruption_candidate_receipt_type
            ),
            "interruption_receipt_type": profile.interruption_receipt_type,
            "resume_commit_protocol": profile.resume_commit_protocol,
            "checkpoint_metadata_paths": {
                "dense": tmp_path / "dense-checkpoint-meta.json",
                "split90": tmp_path / "split90-checkpoint-meta.json",
            },
            "checkpoint_receipt_path": (
                tmp_path / "v3-checkpoint" / "receipt.json"
            ),
            "run_ids": {
                "dense": "v3-dense-s0",
                "split90": "v3-split90-s0",
            },
            "run_manifest_sha256": "8" * 64,
            "cohort_assignment_sha256": "9" * 64,
            "preregistration_sha256": "a" * 64,
            "hardware_amendment_sha256": "b" * 64,
            "provider_selection_sha256": "c" * 64,
            "sealed_fixture_sha256": "d" * 64,
        }
    )
    request = InterruptionRequest(**values)
    mismatched = dict(values)
    mismatched["candidate_receipt_type"] = "aws-p5-interruption-candidate"
    with pytest.raises(ValueError, match="closed AWS GPU profile"):
        InterruptionRequest(**mismatched)

    def signal_process(pid, _signum):
        arm = {101: "dense", 202: "split90"}[pid]
        payload = f"{profile.profile_id}-{arm}-checkpoint".encode()
        _atomic_checkpoint(
            request.checkpoint_paths[arm],
            payload,
        )
        _atomic_checkpoint(
            request.checkpoint_metadata_paths[arm],
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "receipt_type": "memorysplit-training-checkpoint-v1",
                        "run_id": request.run_ids[arm],
                        "condition": arm,
                        "seed": 0,
                        "step": 42,
                        "max_steps": 1358,
                        "world_size": 4,
                        "config_fingerprint": request.config_sha256[arm],
                        "checkpoint_path": "ckpt.pt",
                        "checkpoint_sha256": hashlib.sha256(
                            payload
                        ).hexdigest(),
                        "checkpoint_bytes": len(payload),
                        "terminal": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("ascii"),
        )

    store = _FakeStore()
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        sleep=lambda _delay: None,
    )

    assert result.resumable is True
    assert result.checkpoint_receipt_path == request.checkpoint_receipt_path
    assert result.checkpoint_receipt_sha256
    assert result.checkpoint_receipt_uri.endswith(
        f"/receipts/{result.checkpoint_receipt_sha256}.json"
    )
    from cluster.aws.p5.terminal_artifacts import (
        verify_checkpoint_receipt_bytes,
    )

    bridged = verify_checkpoint_receipt_bytes(
        request.checkpoint_receipt_path.read_bytes(),
        expected={
            "provider": request.provider,
            "run_manifest_sha256": request.run_manifest_sha256,
            "sealed_fixture_sha256": request.sealed_fixture_sha256,
        },
    )
    assert {row["step"] for row in bridged["checkpoints"]} == {42}
    marker = json.loads(request.receipt_path.read_bytes())
    assert marker["receipt_type"] == "aws-gpu-paired-interruption"
    assert marker["protocol"] == "aws-gpu-resume-commit-v1"
    assert marker["provider"] == request.provider
    assert marker["profile_sha256"] == request.profile_sha256
    assert marker["instance_type"] == request.instance_type
    assert marker["gres"] == request.gres
    candidate_bytes = next(
        call[5] for call in store.calls if "/evidence/seed-0/sha256/" in call[1]
    )
    candidate = json.loads(candidate_bytes)
    assert candidate["receipt_type"] == "aws-gpu-interruption-candidate"
    assert candidate["provider"] == request.provider
    assert candidate["profile_sha256"] == request.profile_sha256
    marker_bytes = next(
        call[5] for call in store.calls if "/resume-commits/" in call[1]
    )
    checkpoint_objects = {
        call[1]: call[5]
        for call in store.calls
        if "/checkpoints/" in call[1] and call[1].endswith(".pt")
    }
    assert interruption_module.verify_resume_commit(
        candidate_bytes=candidate_bytes,
        marker_bytes=marker_bytes,
        checkpoint_objects=checkpoint_objects,
        expected_provider=request.provider,
        expected_profile_sha256=request.profile_sha256,
        expected_instance_type=request.instance_type,
        expected_gres=request.gres,
        expected_candidate_receipt_type=request.candidate_receipt_type,
        expected_interruption_receipt_type=(
            request.interruption_receipt_type
        ),
        expected_resume_commit_protocol=request.resume_commit_protocol,
    )
    with pytest.raises(ValueError):
        interruption_module.verify_resume_commit(
            candidate_bytes=candidate_bytes,
            marker_bytes=marker_bytes,
            checkpoint_objects=checkpoint_objects,
        )


def test_interruption_never_labels_failed_upload_resumable(tmp_path):
    request = _interruption_request(tmp_path)
    store = _FakeStore(fail_contains="/split90/")

    def checkpoint(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(
            request.checkpoint_paths[arm],
            f"{arm}-checkpoint".encode(),
        )

    result = handle_interruption(
        request,
        signal_process=checkpoint,
        object_store=store,
        sleep=lambda _delay: None,
    )

    assert result.resumable is False
    assert result.exit_code == NON_RESUMABLE_EXIT_CODE
    receipt = json.loads(request.receipt_path.read_text(encoding="utf-8"))
    assert "resumable" not in receipt
    assert receipt["receipt_type"] == "aws-p5-interruption-candidate"
    assert not any("/resume-commits/" in call[1] for call in store.calls)


def test_interruption_uploads_immutable_bytes_from_new_atomic_generation(tmp_path):
    request = _interruption_request(tmp_path)
    for path in request.checkpoint_paths.values():
        path.write_bytes(b"old-generation")

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(
            request.checkpoint_paths[arm],
            f"{arm}-generation-one".encode(),
        )

    def mutate_original(_staged, uri):
        if "/dense/" in uri:
            request.checkpoint_paths["dense"].write_bytes(b"dense-generation-two")

    store = _FakeStore(on_put=mutate_original)
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        sleep=lambda _delay: None,
        commit_nonce=lambda: "a" * 32,
    )

    assert result.resumable is True
    dense_upload = next(call for call in store.calls if "/dense/" in call[1])
    assert dense_upload[5] == b"dense-generation-one"
    assert dense_upload[0] != request.checkpoint_paths["dense"]
    assert dense_upload[6] & 0o222 == 0
    candidate_call = next(
        call for call in store.calls if "/evidence/seed-1/sha256/" in call[1]
    )
    candidate = json.loads(candidate_call[5])
    dense = next(item for item in candidate["checkpoints"] if item["arm"] == "dense")
    assert dense["sha256"] == hashlib.sha256(b"dense-generation-one").hexdigest()
    assert set(dense["generation"]) == {
        "ctime_ns",
        "device",
        "gid",
        "inode",
        "mode",
        "mtime_ns",
        "size",
        "uid",
    }


def test_interruption_rejects_in_place_checkpoint_rewrite(tmp_path):
    request = _interruption_request(tmp_path)
    for path in request.checkpoint_paths.values():
        path.write_bytes(b"old-generation")

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        request.checkpoint_paths[arm].write_bytes(b"in-place-rewrite")

    clock = _Clock()
    store = _FakeStore()
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.resumable is False
    assert not any("/checkpoints/" in call[1] for call in store.calls)
    receipt = json.loads(request.receipt_path.read_bytes())
    assert receipt["receipt_type"] == "aws-p5-interruption-candidate"
    assert "resumable" not in receipt


def test_interruption_rejects_preexisting_handoff_before_signaling(tmp_path):
    request = _interruption_request(tmp_path)
    request.receipt_path.write_bytes(b'{"stale":true}\n')
    signals = []
    store = _FakeStore()

    with pytest.raises(ValueError, match="receipt|handoff|exists"):
        handle_interruption(
            request,
            signal_process=lambda pid, signum: signals.append((pid, signum)),
            object_store=store,
            sleep=lambda _delay: None,
        )

    assert signals == []
    assert store.calls == []


def test_uncertain_commit_upload_leaves_only_noncommittal_handoff(tmp_path):
    request = _interruption_request(tmp_path)

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(request.checkpoint_paths[arm], arm.encode())

    store = _FakeStore(fail_contains="/resume-commits/")
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        sleep=lambda _delay: None,
        commit_nonce=lambda: "b" * 32,
    )

    assert result.resumable is False
    local = json.loads(request.receipt_path.read_bytes())
    assert local["receipt_type"] == "aws-p5-interruption-candidate"
    assert "commit_uri" not in local
    for call in store.calls:
        if call[0].suffix == ".json":
            assert "resumable" not in json.loads(call[5])


def test_interruption_rejects_object_store_evidence_for_wrong_uri(tmp_path):
    request = _interruption_request(tmp_path)

    class WrongUriStore(_FakeStore):
        def put_verified(self, *args, **kwargs):
            uploaded = super().put_verified(*args, **kwargs)
            if uploaded is not None and "/dense/" in uploaded.uri:
                return interruption_module.UploadedObject(
                    uri=uploaded.uri + ".wrong",
                    sha256=uploaded.sha256,
                    bytes=uploaded.bytes,
                )
            return uploaded

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(request.checkpoint_paths[arm], arm.encode())

    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=WrongUriStore(),
        sleep=lambda _delay: None,
    )

    assert result.resumable is False
    assert json.loads(request.receipt_path.read_bytes())["receipt_type"] == (
        "aws-p5-interruption-candidate"
    )


def test_resume_commit_is_derived_from_every_fetched_object_hash(tmp_path):
    request = _interruption_request(tmp_path)

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(request.checkpoint_paths[arm], arm.encode())

    store = _FakeStore()
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        sleep=lambda _delay: None,
        commit_nonce=lambda: "c" * 32,
    )
    assert result.resumable is True

    candidate = next(call[5] for call in store.calls if "/evidence/" in call[1])
    marker = next(call[5] for call in store.calls if "/resume-commits/" in call[1])
    fetched = {
        call[1]: call[5]
        for call in store.calls
        if "/checkpoints/" in call[1]
    }
    assert interruption_module.verify_resume_commit(
        candidate_bytes=candidate,
        marker_bytes=marker,
        checkpoint_objects=fetched,
    )
    corrupted = dict(fetched)
    corrupted[next(iter(corrupted))] = b"corrupt"
    with pytest.raises(ValueError, match="hash|digest"):
        interruption_module.verify_resume_commit(
            candidate_bytes=candidate,
            marker_bytes=marker,
            checkpoint_objects=corrupted,
        )
    incomplete_candidate = json.loads(candidate)
    incomplete_candidate["checkpoints"][0]["generation"].pop("uid")
    incomplete_bytes = (
        json.dumps(
            incomplete_candidate,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    rebound_marker = json.loads(marker)
    rebound_marker["candidate"]["bytes"] = len(incomplete_bytes)
    rebound_marker["candidate"]["sha256"] = hashlib.sha256(
        incomplete_bytes
    ).hexdigest()
    rebound_marker["candidate"]["uri"] = (
        "s3://memorysplit-prod/cohort-v2/receipts/interruption/"
        f"evidence/seed-1/sha256/{rebound_marker['candidate']['sha256']}.json"
    )
    rebound_marker_bytes = (
        json.dumps(rebound_marker, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    with pytest.raises(ValueError, match="identity|evidence"):
        interruption_module.verify_resume_commit(
            candidate_bytes=incomplete_bytes,
            marker_bytes=rebound_marker_bytes,
            checkpoint_objects=fetched,
        )
    invalid_binding = json.loads(candidate)
    invalid_binding["release_sha256"] = "not-a-digest"
    invalid_binding_bytes = (
        json.dumps(invalid_binding, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    invalid_marker = json.loads(marker)
    invalid_marker["candidate"]["bytes"] = len(invalid_binding_bytes)
    invalid_marker["candidate"]["sha256"] = hashlib.sha256(
        invalid_binding_bytes
    ).hexdigest()
    invalid_marker["candidate"]["uri"] = (
        "s3://memorysplit-prod/cohort-v2/receipts/interruption/"
        f"evidence/sha256/{invalid_marker['candidate']['sha256']}.json"
    )
    invalid_marker_bytes = (
        json.dumps(invalid_marker, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    with pytest.raises(ValueError, match="binding|digest"):
        interruption_module.verify_resume_commit(
            candidate_bytes=invalid_binding_bytes,
            marker_bytes=invalid_marker_bytes,
            checkpoint_objects=fetched,
        )


class _Clock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.now += delay


def _short_interruption_request(
    request: InterruptionRequest,
    *,
    timeout_seconds: float,
    upload_reserve_seconds: float,
) -> InterruptionRequest:
    values = dict(request.__dict__)
    values["timeout_seconds"] = timeout_seconds
    values["upload_reserve_seconds"] = upload_reserve_seconds
    return InterruptionRequest(**values)


def test_checkpoint_staging_cannot_overrun_wall_deadline(
    tmp_path,
    monkeypatch,
):
    request = _short_interruption_request(
        _interruption_request(tmp_path),
        timeout_seconds=0.3,
        upload_reserve_seconds=0.15,
    )
    original_stage = interruption_module._stage_checkpoint

    def blocked_stage(*args, **kwargs):
        time.sleep(1.0)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(interruption_module, "_stage_checkpoint", blocked_stage)

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(request.checkpoint_paths[arm], arm.encode())

    store = _FakeStore()
    started = time.monotonic()
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        sleep=lambda _delay: None,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.8
    assert result.resumable is False
    assert not any("/resume-commits/" in call[1] for call in store.calls)


def test_checkpoint_staging_gives_each_arm_bounded_opportunity(
    tmp_path,
    monkeypatch,
):
    request = _short_interruption_request(
        _interruption_request(tmp_path),
        timeout_seconds=0.6,
        upload_reserve_seconds=0.3,
    )
    original_stage = interruption_module._stage_checkpoint
    started_at = {
        "test": time.monotonic(),
    }

    def asymmetric_stage(arm, *args, **kwargs):
        marker = tmp_path / f"{arm}.stage-started"
        marker.write_text(str(time.monotonic()), encoding="ascii")
        if arm == "dense":
            time.sleep(1.0)
        return original_stage(arm, *args, **kwargs)

    monkeypatch.setattr(interruption_module, "_stage_checkpoint", asymmetric_stage)

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(request.checkpoint_paths[arm], arm.encode())

    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=_FakeStore(),
        sleep=lambda _delay: None,
    )

    dense_started = float(
        (tmp_path / "dense.stage-started").read_text(encoding="ascii")
    )
    split_started = float(
        (tmp_path / "split90.stage-started").read_text(encoding="ascii")
    )
    assert dense_started - started_at["test"] < 0.5
    assert split_started - started_at["test"] < 0.5
    assert result.resumable is False


def test_checkpoint_stability_observes_both_arms_concurrently(tmp_path):
    request = _interruption_request(tmp_path)
    clock = _Clock()
    dense_written = False

    def signal_process(pid, _signal):
        arm = {101: "dense", 202: "split90"}[pid]
        if arm == "split90":
            _atomic_checkpoint(request.checkpoint_paths[arm], b"split-ready")

    def sleep(delay):
        nonlocal dense_written
        clock.sleep(delay)
        if not dense_written and clock.now >= 12.75:
            _atomic_checkpoint(
                request.checkpoint_paths["dense"],
                b"dense-late",
            )
            dense_written = True

    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=_FakeStore(),
        monotonic=clock.monotonic,
        sleep=sleep,
        commit_nonce=lambda: "d" * 32,
    )

    assert result.resumable is True
    assert clock.now <= 15.0


def test_interruption_reserves_upload_time_inside_one_deadline(tmp_path):
    request = _interruption_request(tmp_path)
    clock = _Clock()

    def signal_process(pid, _signum):
        arm = {101: "dense", 202: "split90"}[pid]
        _atomic_checkpoint(
            request.checkpoint_paths[arm],
            arm.encode("ascii"),
        )

    store = _FakeStore()
    result = handle_interruption(
        request,
        signal_process=signal_process,
        object_store=store,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.resumable is True
    overall_deadline = 10.0 + request.timeout_seconds
    assert all(call[2] == overall_deadline for call in store.calls)
    assert all(call[3] < overall_deadline for call in store.calls)
    assert clock.now <= overall_deadline


def test_interruption_deadline_exhaustion_is_nonresumable_and_stops_uploads(
    tmp_path,
):
    request = _interruption_request(tmp_path)
    clock = _Clock()
    store = _FakeStore()

    result = handle_interruption(
        request,
        signal_process=lambda _pid, _signum: None,
        object_store=store,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.resumable is False
    assert result.exit_code == NON_RESUMABLE_EXIT_CODE
    receipt = json.loads(request.receipt_path.read_text(encoding="utf-8"))
    assert "resumable" not in receipt
    assert receipt["deadline_exhausted"] is True
    assert all(call[3] < call[2] for call in store.calls)


def test_s3_object_store_uses_argv_and_checksum_verification(tmp_path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    artifact.chmod(0o400)
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    expected = base64.b64encode(hashlib.sha256(b"checkpoint").digest()).decode()
    calls = []

    private_home = tmp_path / "aws-home"
    private_home.mkdir(mode=0o700)

    def runner(argv, environment, timeout):
        calls.append((argv, environment, timeout))
        assert isinstance(argv, list)
        if "put-object" in argv:
            return CommandResult(0, json.dumps({"ChecksumSHA256": expected}), "")
        return CommandResult(
            0,
            json.dumps(
                {
                    "ChecksumSHA256": expected,
                    "ContentLength": len(b"checkpoint"),
                    "Metadata": {"sha256": digest},
                }
            ),
            "",
        )

    store = S3ObjectStore(
        region="us-east-1",
        runner=runner,
        environment={
            "AWS_REGION": "us-east-1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(private_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    )

    uploaded = store.put_verified(
        artifact,
        "s3://memorysplit-prod/cohort-v2/checkpoints/seed-1/dense.pt",
        expected_sha256=digest,
        deadline=20.0,
        monotonic=lambda: 10.0,
    )
    assert uploaded == interruption_module.UploadedObject(
        uri="s3://memorysplit-prod/cohort-v2/checkpoints/seed-1/dense.pt",
        sha256=digest,
        bytes=len(b"checkpoint"),
    )
    assert len(calls) == 2
    assert calls[0][0][:3] == ["aws", "s3api", "put-object"]
    assert ["--metadata", f"sha256={digest}"] == calls[0][0][
        calls[0][0].index("--metadata") : calls[0][0].index("--metadata") + 2
    ]
    assert ["--if-none-match", "*"] == calls[0][0][
        calls[0][0].index("--if-none-match") :
        calls[0][0].index("--if-none-match") + 2
    ]
    assert calls[1][0][:3] == ["aws", "s3api", "head-object"]
    assert all(0 < timeout <= 10.0 for _, _, timeout in calls)
    assert all(
        "AWS_SECRET_ACCESS_KEY" not in environment
        for _, environment, _timeout in calls
    )


def test_s3_prehash_is_cancelled_at_wall_deadline(tmp_path, monkeypatch):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    artifact.chmod(0o400)
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    private_home = tmp_path / "aws-home"
    private_home.mkdir(mode=0o700)
    calls = []

    def blocked_hash(*_args, **_kwargs):
        time.sleep(1.0)
        return None

    monkeypatch.setattr(
        interruption_module,
        "_hash_file_sync",
        blocked_hash,
        raising=False,
    )
    store = S3ObjectStore(
        region="us-east-1",
        runner=lambda argv, environment, timeout: (
            calls.append((argv, environment, timeout))
            or CommandResult(1, "", "must not run")
        ),
        environment={
            "AWS_REGION": "us-east-1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(private_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
    )

    started = time.monotonic()
    uploaded = store.put_verified(
        artifact,
        "s3://memorysplit-prod/cohort-v2/checkpoints/dense.pt",
        expected_sha256=digest,
        deadline=started + 0.2,
        monotonic=time.monotonic,
    )

    assert time.monotonic() - started < 0.7
    assert uploaded is None
    assert calls == []


def test_imdsv2_client_refreshes_an_expired_token(monkeypatch):
    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.value

    responses = iter(
        [
            Response(b"old-token"),
            HTTPError("http://imds", 401, "expired", {}, None),
            Response(b"new-token"),
            Response(b"p5.48xlarge"),
        ]
    )
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(interruption_module, "urlopen", fake_urlopen)

    assert ImdsV2Client().get("meta-data/instance-type") == "p5.48xlarge"
    assert [request.get_method() for request, _ in requests] == [
        "PUT",
        "GET",
        "PUT",
        "GET",
    ]
    assert requests[-1][0].get_header("X-aws-ec2-metadata-token") == "new-token"


class _ProbeRunner:
    def __init__(
        self,
        *,
        gpu_names=H100_NAMES,
        devices=8,
        fabric_active=True,
        block_override=None,
        holders="",
        swap="",
        md_member=False,
        wipe_signatures=None,
        caller_account="123456789012",
        caller_arn=(
            "arn:aws:sts::123456789012:"
            "assumed-role/MemorySplitP5Role/i-0123456789abcdef0"
        ),
        timeout_prefix=None,
    ):
        self.gpu_names = gpu_names
        self.devices = devices
        self.fabric_active = fabric_active
        self.block_override = block_override or {}
        self.holders = holders
        self.swap = swap
        self.md_member = md_member
        self.wipe_signatures = (
            [] if wipe_signatures is None else wipe_signatures
        )
        self.caller_account = caller_account
        self.caller_arn = caller_arn
        self.timeout_prefix = timeout_prefix
        self.calls = []

    def __call__(self, argv, environment, timeout):
        self.calls.append((argv, environment, timeout))
        if (
            self.timeout_prefix is not None
            and argv[: len(self.timeout_prefix)] == list(self.timeout_prefix)
        ):
            raise subprocess.TimeoutExpired(argv, timeout)
        if argv[0] == "nvidia-smi":
            return CommandResult(0, "\n".join(self.gpu_names) + "\n", "")
        if argv[:2] == ["systemctl", "is-active"]:
            return CommandResult(
                0 if self.fabric_active else 3,
                "active\n" if self.fabric_active else "inactive\n",
                "",
            )
        if argv[0] == "lsblk":
            blockdevices = [
                {
                    "model": "Amazon EC2 NVMe Instance Storage",
                    "mountpoints": [None],
                    "path": f"/dev/test-instance-store-{index}",
                    "size": 3_840_000_000_000,
                    "type": "disk",
                    "fstype": None,
                    "fsver": None,
                    "label": None,
                    "uuid": None,
                    "pttype": None,
                    "parttype": None,
                }
                for index in range(self.devices)
            ]
            if blockdevices:
                blockdevices[0].update(self.block_override)
            return CommandResult(
                0, json.dumps({"blockdevices": blockdevices}), ""
            )
        if argv[0] == "swapon":
            return CommandResult(0, self.swap, "")
        if argv[0] == "ls" and argv[-1].endswith("/holders"):
            return CommandResult(0, self.holders, "")
        if argv[0] == "mdadm" and argv[1:3] == ["--examine", "--brief"]:
            return CommandResult(0 if self.md_member else 1, "", "")
        if argv[0] == "wipefs":
            return CommandResult(
                0,
                json.dumps({"signatures": self.wipe_signatures}),
                "",
            )
        if argv[:3] == ["aws", "sts", "get-caller-identity"]:
            return CommandResult(
                0,
                json.dumps(
                    {
                        "Account": self.caller_account,
                        "Arn": self.caller_arn,
                        "UserId": "AROATEST:i-0123456789abcdef0",
                    }
                ),
                "",
            )
        if argv[:3] == ["docker", "image", "inspect"]:
            digest_ref = argv[-1]
            return CommandResult(0, json.dumps([digest_ref]) + "\n", "")
        raise AssertionError(f"unexpected probe command: {argv}")


def _metadata(instance_type="p5.48xlarge"):
    values = {
        "meta-data/instance-id": "i-0123456789abcdef0",
        "meta-data/instance-type": instance_type,
        "meta-data/ami-id": SAFE_ENVIRONMENT["MS_AWS_AMI_ID"],
        "meta-data/iam/security-credentials/": "MemorySplitP5Role",
        "dynamic/instance-identity/document": json.dumps(
            {
                "accountId": "123456789012",
                "imageId": SAFE_ENVIRONMENT["MS_AWS_AMI_ID"],
                "instanceId": "i-0123456789abcdef0",
                "instanceType": instance_type,
                "region": SAFE_ENVIRONMENT["AWS_REGION"],
            }
        ),
        "dynamic/instance-identity/pkcs7": "c2lnbmF0dXJl",
    }
    return values.__getitem__


def test_bootstrap_inspects_p5_hardware_by_nvme_model_and_renders_argv_commands():
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    runner = _ProbeRunner()
    image = (
        "public.ecr.aws/example/memorysplit@"
        + SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"]
    )

    evidence = inspect_hardware(
        profile,
        runtime,
        metadata_get=_metadata(),
        runner=runner,
        command_environment={
            "AWS_REGION": runtime.region,
            "PATH": "/usr/bin:/bin",
            "HOME": "/private/empty",
        },
        container_image=image,
        boot_id_get=lambda: BOOT_ID,
    )
    commands = render_bootstrap_commands(
        profile,
        runtime,
        evidence,
        owner_uid=1000,
        owner_gid=1000,
    )

    assert evidence.instance_type == "p5.48xlarge"
    assert evidence.ami_id == SAFE_ENVIRONMENT["MS_AWS_AMI_ID"]
    assert len(evidence.instance_store_devices) == 8
    assert all(isinstance(command, list) for command in commands)
    mdadm = next(command for command in commands if command[0] == "mdadm")
    assert mdadm[-8:] == list(evidence.instance_store_devices)
    assert any(command[:3] == ["aws", "s3", "sync"] for command in commands)
    assert all("AWS_SECRET_ACCESS_KEY" not in command for command in commands)
    lsblk = next(argv for argv, _environment, _timeout in runner.calls if argv[0] == "lsblk")
    assert "--tree" in lsblk
    columns = lsblk[lsblk.index("--output") + 1].split(",")
    assert "NAME" in columns
    assert "PATH" in columns


def test_bootstrap_command_timeout_becomes_fail_closed_bootstrap_error():
    def timeout(_argv, _environment, timeout_seconds):
        raise subprocess.TimeoutExpired(["nvidia-smi"], timeout_seconds)

    with pytest.raises(BootstrapError, match="GPU discovery timed out"):
        bootstrap_module._checked(
            timeout,
            ["nvidia-smi"],
            {"PATH": "/usr/bin:/bin"},
            operation="GPU discovery",
            timeout_seconds=12.0,
        )


@pytest.mark.parametrize(
    "prefix",
    [
        ("systemctl", "is-active"),
        ("mdadm", "--examine"),
    ],
)
def test_bootstrap_all_probe_timeouts_fail_closed(prefix):
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)

    with pytest.raises(BootstrapError, match="timed out"):
        inspect_hardware(
            profile,
            runtime,
            metadata_get=_metadata(),
            runner=_ProbeRunner(timeout_prefix=prefix),
            command_environment={
                "AWS_REGION": runtime.region,
                "PATH": "/usr/bin:/bin",
                "HOME": "/private/empty",
            },
            container_image=CONTAINER_IMAGE,
            boot_id_get=lambda: BOOT_ID,
        )


def test_bootstrap_main_emits_canonical_json_for_command_timeout(
    tmp_path,
    monkeypatch,
    capsys,
):
    private_home = tmp_path / "aws-home"
    private_home.mkdir(mode=0o700)
    calls = []

    def timeout(argv, _environment, timeout_seconds):
        calls.append((list(argv), timeout_seconds))
        raise subprocess.TimeoutExpired(argv, timeout_seconds)

    class MetadataClient:
        get = staticmethod(_metadata())

    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    monkeypatch.setattr(bootstrap_module.os, "environ", dict(SAFE_ENVIRONMENT))
    monkeypatch.setattr(
        bootstrap_module,
        "load_aws_p5_profile",
        lambda _path: profile,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "validate_runtime_environment",
        lambda _profile, _environment: runtime,
    )
    monkeypatch.setattr(bootstrap_module, "_run_command", timeout)
    monkeypatch.setattr(bootstrap_module, "_default_boot_id", lambda: BOOT_ID)
    monkeypatch.setattr(bootstrap_module, "ImdsV2Client", MetadataClient)
    monkeypatch.setattr(
        bootstrap_module,
        "build_aws_command_environment",
        lambda _profile, runtime, private_home: {
            "AWS_REGION": runtime.region,
            "HOME": str(private_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )
    missing = tmp_path / "not-needed-before-probe"
    exit_code = bootstrap_module.main(
        [
            "--profile",
            str(PROFILE_PATH),
            "--container-image",
            CONTAINER_IMAGE,
            "--release-archive",
            str(missing),
            "--release-sha256",
            "1" * 64,
            "--release-receipt",
            str(missing),
            "--release-receipt-sha256",
            "2" * 64,
            "--dataset-receipt",
            str(missing),
            "--dataset-receipt-sha256",
            "3" * 64,
            "--cohort-assignment",
            str(missing),
            "--cohort-assignment-sha256",
            "4" * 64,
            "--code-commit",
            CODE_COMMIT,
            "--owner-uid",
            str(RUNTIME_UID),
            "--owner-gid",
            str(RUNTIME_GID),
            "--aws-private-home",
            str(private_home),
        ]
    )

    assert calls and calls[0][0][:3] == ["aws", "sts", "get-caller-identity"]
    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "BootstrapError",
        "ok": False,
        "schema_version": 1,
    }


def test_bootstrap_binds_imdsv2_role_sts_identity_and_boot_id():
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    runner = _ProbeRunner()

    evidence = inspect_hardware(
        profile,
        runtime,
        metadata_get=_metadata(),
        runner=runner,
        command_environment={
            "AWS_REGION": runtime.region,
            "PATH": "/usr/bin:/bin",
            "HOME": "/private/empty",
        },
        container_image=CONTAINER_IMAGE,
        boot_id_get=lambda: BOOT_ID,
    )

    assert evidence.boot_id == BOOT_ID
    assert evidence.role_name == "MemorySplitP5Role"
    assert evidence.account_id == "123456789012"
    assert evidence.role_arn.endswith(
        "assumed-role/MemorySplitP5Role/i-0123456789abcdef0"
    )
    assert any(
        argv[:3] == ["aws", "sts", "get-caller-identity"]
        for argv, _environment, _timeout in runner.calls
    )
    assert all(0 < timeout <= 30 for _argv, _environment, timeout in runner.calls)


def test_bootstrap_rejects_sts_identity_not_bound_to_imdsv2_role():
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)

    with pytest.raises(BootstrapError, match="role|identity|account"):
        inspect_hardware(
            profile,
            runtime,
            metadata_get=_metadata(),
            runner=_ProbeRunner(caller_account="999999999999"),
            command_environment={
                "AWS_REGION": runtime.region,
                "PATH": "/usr/bin:/bin",
                "HOME": "/private/empty",
            },
            container_image=CONTAINER_IMAGE,
            boot_id_get=lambda: BOOT_ID,
        )


def test_aws_commands_use_only_private_empty_home_and_fixed_allowlist(tmp_path):
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    private_home = tmp_path / "aws-home"
    private_home.mkdir(mode=0o700)

    environment = bootstrap_module.build_aws_command_environment(
        profile,
        runtime,
        private_home=private_home,
    )

    assert environment == {
        "AWS_REGION": "us-east-1",
        "HOME": str(private_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    assert set(environment).isdisjoint(
        {
            "AWS_PROFILE",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
        }
    )

    (private_home / "credentials").write_text("not empty", encoding="utf-8")
    with pytest.raises(BootstrapError, match="empty"):
        bootstrap_module.build_aws_command_environment(
            profile,
            runtime,
            private_home=private_home,
        )


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (
            _ProbeRunner(
                block_override={
                    "children": [
                        {
                            "path": "/dev/test-instance-store-0p1",
                            "type": "part",
                        }
                    ]
                }
            ),
            "children|partition",
        ),
        (_ProbeRunner(block_override={"fstype": "xfs"}), "filesystem|signature"),
        (_ProbeRunner(block_override={"pttype": "gpt"}), "partition|signature"),
        (_ProbeRunner(holders="md127\n"), "holder"),
        (_ProbeRunner(swap="/dev/test-instance-store-0\n"), "swap"),
        (_ProbeRunner(md_member=True), "RAID|md"),
        (
            _ProbeRunner(
                wipe_signatures=[
                    {"offset": "0x0", "type": "ext4", "uuid": "fixture"}
                ]
            ),
            "signature|filesystem",
        ),
    ],
)
def test_bootstrap_recursively_rejects_in_use_or_signed_nvme(runner, message):
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)

    with pytest.raises(BootstrapError, match=message):
        inspect_hardware(
            profile,
            runtime,
            metadata_get=_metadata(),
            runner=runner,
            command_environment={
                "AWS_REGION": runtime.region,
                "PATH": "/usr/bin:/bin",
                "HOME": "/private/empty",
            },
            container_image=CONTAINER_IMAGE,
            boot_id_get=lambda: BOOT_ID,
        )


def test_destructive_nvme_render_requires_apply_authorization_and_nonroot_owner():
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    evidence = inspect_hardware(
        profile,
        runtime,
        metadata_get=_metadata(),
        runner=_ProbeRunner(),
        command_environment={
            "AWS_REGION": runtime.region,
            "PATH": "/usr/bin:/bin",
            "HOME": "/private/empty",
        },
        container_image=CONTAINER_IMAGE,
        boot_id_get=lambda: BOOT_ID,
    )

    with pytest.raises(BootstrapError, match="authorization"):
        render_bootstrap_commands(
            profile,
            runtime,
            evidence,
            owner_uid=1000,
            owner_gid=1000,
            apply=True,
            destructive_authorized=False,
        )
    with pytest.raises(BootstrapError, match="non-root"):
        render_bootstrap_commands(
            profile,
            runtime,
            evidence,
            owner_uid=0,
            owner_gid=0,
        )


@pytest.mark.parametrize(
    ("runner", "metadata_get", "message"),
    [
        (_ProbeRunner(), _metadata("p5.4xlarge"), "p5.48xlarge"),
        (_ProbeRunner(gpu_names=H100_NAMES[:7]), _metadata(), "eight"),
        (_ProbeRunner(gpu_names=("NVIDIA A100-SXM4-80GB",) * 8), _metadata(), "H100"),
        (_ProbeRunner(devices=7), _metadata(), "instance-store"),
        (_ProbeRunner(fabric_active=False), _metadata(), "Fabric Manager"),
    ],
)
def test_bootstrap_fails_closed_on_hardware_drift(
    runner, metadata_get, message
):
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)

    with pytest.raises(BootstrapError, match=message):
        inspect_hardware(
            profile,
            runtime,
            metadata_get=metadata_get,
            runner=runner,
            command_environment={
                "AWS_REGION": runtime.region,
                "PATH": "/usr/bin:/bin",
                "HOME": "/private/empty",
            },
            container_image=(
                "public.ecr.aws/example/memorysplit@"
                + SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"]
            ),
            boot_id_get=lambda: BOOT_ID,
        )


def _task7_release_fixture(tmp_path: Path, *, corrupt_sum: bool = False):
    member_payload = {
        "scripts/run_train.py": b"print('verified release')\n",
        "train/trainer.py": b"def train():\n    return None\n",
    }
    metadata = {
        "members": [
            {
                "bytes": len(payload),
                "git_blob": "5" * 40,
                "git_mode": "100644",
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for relative, payload in sorted(member_payload.items())
        ],
        "package_format_version": 1,
        "provider": "aws-p5.48xlarge",
        "schema_version": 1,
        "seed_assignment": {
            "arms": ["dense", "split90"],
            "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
            "provider": "aws-p5.48xlarge",
            "seeds": [1, 2, 3, 4],
        },
        "source": {"commit": CODE_COMMIT, "dirty": False},
    }
    metadata_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    payload = {**member_payload, "RELEASE-METADATA.json": metadata_bytes}
    sums = "".join(
        (
            ("0" * 64 if corrupt_sum and index == 0 else hashlib.sha256(data).hexdigest())
            + f"  {relative}\n"
        )
        for index, (relative, data) in enumerate(sorted(payload.items()))
    ).encode("ascii")
    payload["SHA256SUMS"] = sums
    archive = tmp_path / "ms-aws-p5-r1-fixture.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for relative, data in sorted(payload.items()):
            info = zipfile.ZipInfo(relative)
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            output.writestr(info, data)
    dataset_receipt = tmp_path / "receipt.json"
    cohort = tmp_path / "cohort.json"
    dataset_receipt.write_bytes(
        ('{"build_id":"' + "b" * 64 + '"}\n').encode("ascii")
    )
    cohort.write_bytes(b'{"cohort":true}\n')
    release_receipt = _write_json(
        tmp_path / "RELEASE-AWS-P5.json",
        {
            "archive": {
                "bytes": archive.stat().st_size,
                "path": archive.name,
                "sha256": _sha256(archive),
            },
            "cohort_assignment_sha256": _sha256(cohort),
            "dataset_receipt_sha256": _sha256(dataset_receipt),
            "members_sha256": hashlib.sha256(sums).hexdigest(),
            "source": {"commit": CODE_COMMIT, "dirty": False},
        },
    )
    return archive, release_receipt, dataset_receipt, cohort


def test_bootstrap_extracts_only_verified_task7_release_read_only(tmp_path):
    archive, release_receipt, dataset_receipt, cohort = _task7_release_fixture(
        tmp_path
    )
    artifacts = verify_bootstrap_artifacts(
        release_archive=archive,
        release_sha256=_sha256(archive),
        release_receipt=release_receipt,
        release_receipt_sha256=_sha256(release_receipt),
        dataset_receipt=dataset_receipt,
        dataset_receipt_sha256=_sha256(dataset_receipt),
        cohort_assignment=cohort,
        cohort_assignment_sha256=_sha256(cohort),
        code_commit=CODE_COMMIT,
    )

    prepared = bootstrap_module.extract_verified_release(
        release_archive=archive,
        artifacts=artifacts,
        scratch_root=tmp_path / "scratch",
        owner_uid=os.getuid(),
        owner_gid=os.getgid(),
    )

    assert prepared.root == (
        tmp_path / "scratch" / "releases" / _sha256(archive)
    )
    assert prepared.members_sha256 == hashlib.sha256(
        (prepared.root / "SHA256SUMS").read_bytes()
    ).hexdigest()
    assert (prepared.root / "scripts" / "run_train.py").read_bytes() == (
        b"print('verified release')\n"
    )
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in prepared.root.rglob("*")
    )
    reused = bootstrap_module.extract_verified_release(
        release_archive=archive,
        artifacts=artifacts,
        scratch_root=tmp_path / "scratch",
        owner_uid=os.getuid(),
        owner_gid=os.getgid(),
    )
    assert reused == prepared

    changed = prepared.root / "scripts" / "run_train.py"
    changed.chmod(0o644)
    changed.write_bytes(b"print('drifted release')\n")
    changed.chmod(0o444)
    with pytest.raises(BootstrapError, match="identity drifted"):
        bootstrap_module.extract_verified_release(
            release_archive=archive,
            artifacts=artifacts,
            scratch_root=tmp_path / "scratch",
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )


def test_existing_bootstrap_storage_is_reused_only_with_exact_root_identity(
    tmp_path,
):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    profile = SimpleNamespace(
        scratch_root=str(scratch),
        instance_store_devices=2,
    )

    def runner(argv, _environment, _timeout):
        if argv[0] == "findmnt":
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "filesystems": [
                            {
                                "target": str(scratch),
                                "source": "/dev/md/memorysplit",
                                "fstype": "xfs",
                                "options": "rw,noatime,nodiratime",
                            }
                        ]
                    }
                ),
                stderr="",
            )
        if argv[0] == "mdadm":
            return CommandResult(
                returncode=0,
                stdout=(
                    "MD_LEVEL=raid0\n"
                    "MD_DEVICES=2\n"
                    "MD_DEVICE_0_DEV=/dev/nvme1n1\n"
                    "MD_DEVICE_1_DEV=/dev/nvme2n1\n"
                ),
                stderr="",
            )
        pytest.fail(f"unexpected storage probe: {argv}")

    expected = bootstrap_module._inspect_existing_storage(
        profile,
        owner_uid=os.getuid(),
        owner_gid=os.getgid(),
        runner=runner,
        command_environment={},
    )
    assert expected == ("/dev/nvme1n1", "/dev/nvme2n1")

    scratch.chmod(0o755)
    with pytest.raises(BootstrapError, match="identity has drifted"):
        bootstrap_module._inspect_existing_storage(
            profile,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
            runner=runner,
            command_environment={},
        )


def test_bootstrap_rejects_hash_consistent_outer_archive_with_bad_inner_sum(
    tmp_path,
):
    archive, release_receipt, dataset_receipt, cohort = _task7_release_fixture(
        tmp_path,
        corrupt_sum=True,
    )

    with pytest.raises(BootstrapError, match="SHA256SUMS|member"):
        verify_bootstrap_artifacts(
            release_archive=archive,
            release_sha256=_sha256(archive),
            release_receipt=release_receipt,
            release_receipt_sha256=_sha256(release_receipt),
            dataset_receipt=dataset_receipt,
            dataset_receipt_sha256=_sha256(dataset_receipt),
            cohort_assignment=cohort,
            cohort_assignment_sha256=_sha256(cohort),
            code_commit=CODE_COMMIT,
        )


def test_bootstrap_verifies_all_hashes_and_publishes_canonical_receipt(tmp_path):
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    (
        release_archive,
        release_receipt,
        dataset_receipt,
        cohort,
    ) = _task7_release_fixture(
        tmp_path,
    )
    artifacts = verify_bootstrap_artifacts(
        release_archive=release_archive,
        release_sha256=_sha256(release_archive),
        release_receipt=release_receipt,
        release_receipt_sha256=_sha256(release_receipt),
        dataset_receipt=dataset_receipt,
        dataset_receipt_sha256=_sha256(dataset_receipt),
        cohort_assignment=cohort,
        cohort_assignment_sha256=_sha256(cohort),
        code_commit=CODE_COMMIT,
    )
    evidence = inspect_hardware(
        profile,
        runtime,
        metadata_get=_metadata(),
        runner=_ProbeRunner(),
        command_environment={
            "AWS_REGION": runtime.region,
            "PATH": "/usr/bin:/bin",
            "HOME": "/private/empty",
        },
        container_image=(
            "public.ecr.aws/example/memorysplit@"
            + SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"]
        ),
        boot_id_get=lambda: BOOT_ID,
    )
    prepared = bootstrap_module.PreparedRelease(
        root=(
            Path(profile.scratch_root)
            / "releases"
            / artifacts.release_sha256
        ),
        members_sha256=artifacts.release_members_sha256,
    )
    receipt = build_bootstrap_receipt(
        profile=profile,
        runtime=runtime,
        evidence=evidence,
        artifacts=artifacts,
        prepared_release=prepared,
        durable_upload_verified=True,
    )
    store = _FakeStore()
    receipt_path = tmp_path / "bootstrap-receipt.json"

    assert publish_bootstrap_receipt(
        receipt,
        receipt_path=receipt_path,
        receipt_uri=(
            "s3://memorysplit-prod/cohort-v2/receipts/bootstrap/"
            "i-0123456789abcdef0.json"
        ),
        object_store=store,
    )
    raw = receipt_path.read_bytes()
    assert raw == (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert receipt["durable_upload_verified"] is True
    assert receipt["release_sha256"] == _sha256(release_archive)
    assert receipt["release_members_sha256"] == artifacts.release_members_sha256
    assert receipt["corpus_receipt_sha256"] == _sha256(dataset_receipt)
    assert receipt["corpus_build_id"] == "b" * 64
    assert receipt["runtime_uid"] == RUNTIME_UID
    assert receipt["runtime_gid"] == RUNTIME_GID
    assert receipt["boot_id"] == BOOT_ID
    assert publish_bootstrap_receipt(
        receipt,
        receipt_path=receipt_path,
        receipt_uri=(
            "s3://memorysplit-prod/cohort-v2/receipts/bootstrap/"
            "i-0123456789abcdef0.json"
        ),
        object_store=store,
    )
    changed = {**receipt, "release_sha256": "0" * 64}
    with pytest.raises(BootstrapError, match="identity has drifted"):
        publish_bootstrap_receipt(
            changed,
            receipt_path=receipt_path,
            receipt_uri=(
                "s3://memorysplit-prod/cohort-v2/receipts/bootstrap/"
                "i-0123456789abcdef0.json"
            ),
            object_store=store,
        )


def test_bootstrap_rejects_non_utf8_release_receipt(tmp_path):
    release_archive = tmp_path / "release.zip"
    dataset_receipt = tmp_path / "corpus-receipt.json"
    cohort = tmp_path / "cohort.json"
    release_archive.write_bytes(b"release")
    dataset_receipt.write_bytes(
        ('{"build_id":"' + "b" * 64 + '"}\n').encode("ascii")
    )
    cohort.write_bytes(b'{"cohort":true}\n')
    release_receipt = tmp_path / "RELEASE-AWS-P5.json"
    release_receipt.write_bytes(
        json.dumps(
            {
                "archive": {"sha256": _sha256(release_archive)},
                "cohort_assignment_sha256": _sha256(cohort),
                "dataset_receipt_sha256": _sha256(dataset_receipt),
                "source": {"commit": CODE_COMMIT},
            }
        ).encode("utf-16")
    )

    with pytest.raises(BootstrapError, match="JSON"):
        verify_bootstrap_artifacts(
            release_archive=release_archive,
            release_sha256=_sha256(release_archive),
            release_receipt=release_receipt,
            release_receipt_sha256=_sha256(release_receipt),
            dataset_receipt=dataset_receipt,
            dataset_receipt_sha256=_sha256(dataset_receipt),
            cohort_assignment=cohort,
            cohort_assignment_sha256=_sha256(cohort),
            code_commit=CODE_COMMIT,
        )


def test_bootstrap_shell_is_strict_thin_wrapper_without_aws_shell_commands():
    text = BOOTSTRAP_SH.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert 'exec "${PYTHON:-python3}"' in text
    assert "aws " not in text


@pytest.mark.parametrize(
    "script",
    [
        ROOT / "cluster" / "aws" / "p5" / "bootstrap.py",
        ROOT / "cluster" / "aws" / "p5" / "launch_seed_pair.py",
    ],
)
def test_p5_commands_start_from_outside_repository_without_pythonpath(
    tmp_path, script
):
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage:")
