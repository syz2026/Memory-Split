from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import signal
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

import cluster.aws.p5.interruption_checkpoint as interruption_module
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
    render_plan,
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
SAFE_ENVIRONMENT = {
    "AWS_REGION": "us-east-1",
    "HOME": "/home/operator",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONPATH": ".",
    "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
    "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
    "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
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


def _sidecar_set(path: str, content: bytes) -> dict:
    return {
        "artifacts": [
            {
                "bytes": len(content),
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
        "dtype": "uint8",
        "items": len(content),
        "ordered_stream_sha256": hashlib.sha256(content).hexdigest(),
    }


def _launcher_fixture(tmp_path: Path, seed: int = 1) -> dict[str, Path | dict]:
    repo_root = tmp_path / "release"
    scratch_root = tmp_path / "scratch"
    repo_root.mkdir(parents=True)
    scratch_root.mkdir(parents=True)

    dataset = scratch_root / "dataset"
    dense_bytes = b"\x01\x01\x01\x01"
    split_bytes = b"\x01\x00\x01\x00"
    dense_sidecar = dataset / "sidecars" / "dense" / "000.bin"
    split_sidecar = dataset / "sidecars" / "split90" / "000.bin"
    dense_sidecar.parent.mkdir(parents=True)
    split_sidecar.parent.mkdir(parents=True)
    dense_sidecar.write_bytes(dense_bytes)
    split_sidecar.write_bytes(split_bytes)
    corpus = {
        "format": "memorysplit-parallel-corpus-v2",
        "ordered_stream_sha256": HEX["ordered"],
        "sidecar_sets": {
            "dense_target_weights": _sidecar_set(
                "sidecars/dense/000.bin", dense_bytes
            ),
            "split90_target_weights": _sidecar_set(
                "sidecars/split90/000.bin", split_bytes
            ),
        },
    }
    corpus_path = _write_json(dataset / "corpus-receipt.json", corpus)

    configs = {}
    for arm in ("dense", "split90"):
        configs[arm] = _write_config(
            repo_root / "configs" / "360m-v2" / f"{arm}-s{seed}.yaml",
            seed=seed,
            arm=arm,
        )

    bootstrap = {
        "ami_id": SAFE_ENVIRONMENT["MS_AWS_AMI_ID"],
        "code_commit": CODE_COMMIT,
        "cohort_assignment_sha256": HEX["cohort"],
        "container_digest": SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"],
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
        "release_sha256": HEX["release"],
        "schema_version": 1,
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
                "checkpoint": f"runs/seed-{seed}/{arm}/checkpoint.pt",
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
        "release_sha256": HEX["release"],
        "runs": runs,
        "schema_version": 1,
        "seed": seed,
    }
    manifest_path = _write_json(repo_root / "run-manifest.json", manifest)
    return {
        "repo_root": repo_root,
        "scratch_root": scratch_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "corpus_path": corpus_path,
        "bootstrap_path": bootstrap_path,
        "configs": configs,
    }


def _load_fixture_plan(fixture: dict, *, seed: int | None = None, **kwargs):
    manifest = fixture["manifest"]
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
        **kwargs,
    )


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
    assert dense["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert dense["argv"] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        f"--master_port={base_port}",
        "scripts/run_train.py",
        "--config",
        f"configs/360m-v2/dense-s{seed}.yaml",
    ]
    assert dense["cpu_affinity"] == [0, 95]
    assert split90["arm"] == "split90"
    assert split90["env"]["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert split90["argv"] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        f"--master_port={base_port + 1}",
        "scripts/run_train.py",
        "--config",
        f"configs/360m-v2/split90-s{seed}.yaml",
    ]
    assert split90["cpu_affinity"] == [96, 191]
    assert dense["env"]["MS_DATA_LOADER_WORKERS"] == "16"
    assert split90["env"]["MS_DATA_LOADER_WORKERS"] == "16"
    assert dense["env"]["MS_DATA_ROOT"] == str(fixture["scratch_root"] / "dataset")
    assert split90["env"]["MS_DATA_ROOT"] == str(
        fixture["scratch_root"] / "dataset"
    )
    assert dense["env"]["MS_RUN_ROOT"] == str(fixture["scratch_root"])
    assert split90["env"]["MS_RUN_ROOT"] == str(fixture["scratch_root"])
    assert dense["env"]["PYTHONPATH"] == str(fixture["repo_root"])
    assert split90["env"]["PYTHONPATH"] == str(fixture["repo_root"])
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
            gpu_names=H100_NAMES,
            port_available=lambda port: port != occupied,
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
            gpu_names=gpu_names,
            port_available=lambda _port: True,
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


def test_launcher_rejects_missing_or_tampered_sidecar(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    sidecar = (
        fixture["scratch_root"]
        / "dataset"
        / "sidecars"
        / "split90"
        / "000.bin"
    )
    sidecar.unlink()

    with pytest.raises(LaunchError, match="sidecar"):
        _load_fixture_plan(fixture)


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
            gpu_names=H100_NAMES,
            port_available=lambda _port: True,
        )


class _FakeProcess:
    def __init__(self, pid: int, polls: list[int | None], *, waited: int = -15):
        self.pid = pid
        self._polls = list(polls)
        self._last = polls[-1]
        self.waited = waited
        self.terminated = False

    def poll(self):
        if self._polls:
            self._last = self._polls.pop(0)
        return self._last

    def wait(self, timeout=None):
        del timeout
        return self.waited if self.terminated else int(self._last or 0)

    def terminate_tree(self):
        self.terminated = True


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


def test_supervisor_records_both_pids_and_accepts_only_paired_success(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    spawner = _FakeSpawner(
        {
            "dense": _FakeProcess(101, [None, 0]),
            "split90": _FakeProcess(202, [None, 0]),
        }
    )

    result = supervise_pair(plan, spawner=spawner, sleep=lambda _delay: None)

    assert result.status == "completed"
    assert result.returncode == 0
    assert result.child_pids == {"dense": 101, "split90": 202}
    assert [launch.arm for launch in spawner.started] == ["dense", "split90"]
    assert all(launch.out_dir.is_dir() for launch in spawner.started)


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
        supervise_pair(plan, spawner=spawner, sleep=lambda _delay: None)

    assert spawner.started == []
    assert dense.terminated is False


def test_supervisor_propagates_failure_and_terminates_peer(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    dense = _FakeProcess(101, [7])
    split90 = _FakeProcess(202, [None])
    spawner = _FakeSpawner({"dense": dense, "split90": split90})

    result = supervise_pair(plan, spawner=spawner, sleep=lambda _delay: None)

    assert result.status == "failed"
    assert result.returncode == 7
    assert result.failed_arm == "dense"
    assert result.peer_terminated is True
    assert split90.terminated is True


def test_supervisor_terminates_first_arm_when_second_spawn_fails(tmp_path):
    fixture = _launcher_fixture(tmp_path)
    plan = _load_fixture_plan(fixture)
    dense = _FakeProcess(101, [None])
    spawner = _FakeSpawner(
        {"dense": dense, "split90": RuntimeError("spawn failed")}
    )

    with pytest.raises(LaunchError, match="spawn"):
        supervise_pair(plan, spawner=spawner, sleep=lambda _delay: None)

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
        )

    assert dense.terminated is True
    assert split90.terminated is True


class _FakeStore:
    def __init__(self, *, fail_suffix: str | None = None):
        self.fail_suffix = fail_suffix
        self.calls: list[tuple[Path, str]] = []

    def put_verified(self, path: Path, uri: str) -> bool:
        self.calls.append((path, uri))
        return self.fail_suffix is None or not uri.endswith(self.fail_suffix)


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
    )


def test_interruption_signals_both_rank_zero_processes_and_emits_paired_receipt(
    tmp_path,
):
    request = _interruption_request(tmp_path)
    pid_to_arm = {101: "dense", 202: "split90"}
    signals = []

    def signal_process(pid, signum):
        signals.append((pid, signum))
        arm = pid_to_arm[pid]
        request.checkpoint_paths[arm].write_bytes(f"{arm}-checkpoint".encode())

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
    assert receipt["resumable"] is True
    assert [item["arm"] for item in receipt["checkpoints"]] == [
        "dense",
        "split90",
    ]
    assert all(item["upload_verified"] for item in receipt["checkpoints"])
    assert len(store.calls) == 3


def test_interruption_never_labels_failed_upload_resumable(tmp_path):
    request = _interruption_request(tmp_path)
    for arm, path in request.checkpoint_paths.items():
        path.write_bytes(f"{arm}-checkpoint".encode())
    store = _FakeStore(fail_suffix="/split90/checkpoint.pt")

    result = handle_interruption(
        request,
        signal_process=lambda _pid, _signal: None,
        object_store=store,
        sleep=lambda _delay: None,
    )

    assert result.resumable is False
    assert result.exit_code == NON_RESUMABLE_EXIT_CODE
    receipt = json.loads(request.receipt_path.read_text(encoding="utf-8"))
    assert receipt["resumable"] is False
    assert any(not item["upload_verified"] for item in receipt["checkpoints"])


def test_s3_object_store_uses_argv_and_checksum_verification(tmp_path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    expected = base64.b64encode(hashlib.sha256(b"checkpoint").digest()).decode()
    calls = []

    def runner(argv, environment):
        calls.append((argv, environment))
        assert isinstance(argv, list)
        if "put-object" in argv:
            return CommandResult(0, json.dumps({"ChecksumSHA256": expected}), "")
        return CommandResult(0, json.dumps({"ChecksumSHA256": expected}), "")

    store = S3ObjectStore(
        region="us-east-1",
        runner=runner,
        environment={
            "AWS_REGION": "us-east-1",
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/operator",
        },
    )

    assert store.put_verified(
        artifact,
        "s3://memorysplit-prod/cohort-v2/checkpoints/seed-1/dense.pt",
    )
    assert len(calls) == 2
    assert calls[0][0][:3] == ["aws", "s3api", "put-object"]
    assert calls[1][0][:3] == ["aws", "s3api", "head-object"]
    assert all(
        "AWS_SECRET_ACCESS_KEY" not in environment
        for _, environment in calls
    )


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
    def __init__(self, *, gpu_names=H100_NAMES, devices=8, fabric_active=True):
        self.gpu_names = gpu_names
        self.devices = devices
        self.fabric_active = fabric_active
        self.calls = []

    def __call__(self, argv, environment):
        self.calls.append((argv, environment))
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
                }
                for index in range(self.devices)
            ]
            return CommandResult(
                0, json.dumps({"blockdevices": blockdevices}), ""
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
            "PATH": SAFE_ENVIRONMENT["PATH"],
            "HOME": SAFE_ENVIRONMENT["HOME"],
        },
        container_image=image,
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
                "PATH": SAFE_ENVIRONMENT["PATH"],
                "HOME": SAFE_ENVIRONMENT["HOME"],
            },
            container_image=(
                "public.ecr.aws/example/memorysplit@"
                + SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"]
            ),
        )


def test_bootstrap_verifies_all_hashes_and_publishes_canonical_receipt(tmp_path):
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    release_archive = tmp_path / "release.zip"
    dataset_receipt = tmp_path / "corpus-receipt.json"
    cohort = tmp_path / "cohort.json"
    release_archive.write_bytes(b"release")
    dataset_receipt.write_bytes(b'{"dataset":true}\n')
    cohort.write_bytes(b'{"cohort":true}\n')
    release_receipt = _write_json(
        tmp_path / "RELEASE-AWS-P5.json",
        {
            "archive": {"sha256": _sha256(release_archive)},
            "cohort_assignment_sha256": _sha256(cohort),
            "dataset_receipt_sha256": _sha256(dataset_receipt),
            "source": {"commit": CODE_COMMIT},
        },
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
            "PATH": SAFE_ENVIRONMENT["PATH"],
            "HOME": SAFE_ENVIRONMENT["HOME"],
        },
        container_image=(
            "public.ecr.aws/example/memorysplit@"
            + SAFE_ENVIRONMENT["MS_CONTAINER_DIGEST"]
        ),
    )
    receipt = build_bootstrap_receipt(
        profile=profile,
        runtime=runtime,
        evidence=evidence,
        artifacts=artifacts,
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
    assert receipt["corpus_receipt_sha256"] == _sha256(dataset_receipt)


def test_bootstrap_rejects_non_utf8_release_receipt(tmp_path):
    release_archive = tmp_path / "release.zip"
    dataset_receipt = tmp_path / "corpus-receipt.json"
    cohort = tmp_path / "cohort.json"
    release_archive.write_bytes(b"release")
    dataset_receipt.write_bytes(b'{"dataset":true}\n')
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
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage:")
