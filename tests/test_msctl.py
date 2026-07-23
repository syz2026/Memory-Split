from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "cluster" / "profiles" / "illumina-usfc-prd.json"
HEX64 = "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _release(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "RELEASE.json",
        {
            "schema_version": 1,
            "release_id": "r1-test",
            "provider": "illumina-usfc-prd",
            "archive": {
                "path": "ms-illumina-r1-test.zip",
                "sha256": "1" * 64,
                "bytes": 123,
            },
            "source": {"commit": "2" * 40, "dirty": False},
            "members_sha256": "3" * 64,
        },
    )


def _runs(tmp_path: Path) -> tuple[Path, dict]:
    configs = tmp_path / "configs" / "v2"
    dense = configs / "dense-s0.json"
    split = configs / "split90-s0.json"
    _write_json(dense, {"arm": "dense", "seed": 0})
    _write_json(split, {"arm": "split90", "seed": 0})
    value = {
        "schema_version": 1,
        "provider": "illumina-usfc-prd",
        "release_sha256": "1" * 64,
        "dataset_sha256": "4" * 64,
        "runs": [
            {
                "run_id": "v2-dense-s0",
                "arm": "dense",
                "seed": 0,
                "config": "configs/v2/dense-s0.json",
                "config_sha256": _sha256(dense),
                "estimated_gpu_hours": 100.0,
            },
            {
                "run_id": "v2-split90-s0",
                "arm": "split90",
                "seed": 0,
                "config": "configs/v2/split90-s0.json",
                "config_sha256": _sha256(split),
                "estimated_gpu_hours": 100.0,
            },
        ],
    }
    return _write_json(tmp_path / "runs.json", value), value


def _approval(
    tmp_path: Path,
    *,
    operation: str,
    runs: dict,
    key: str,
    release_sha256: str = "1" * 64,
    expires_at: str = "2999-01-01T00:00:00Z",
) -> Path:
    unsigned = {
        "schema_version": 1,
        "receipt_id": f"approve-{operation}",
        "provider": "illumina-usfc-prd",
        "operation": operation,
        "release_sha256": release_sha256,
        "run_manifest_sha256": hashlib.sha256(_canonical(runs)).hexdigest(),
        "limits": {"gpu_hours": 250.0, "jobs": 2},
        "expires_at": expires_at,
        "key_id": "operator-test",
    }
    unsigned["signature"] = hmac.new(
        key.encode(), _canonical(unsigned), hashlib.sha256
    ).hexdigest()
    return _write_json(tmp_path / f"{operation}-approval.json", unsigned)


def _run_msctl(
    *arguments: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    merged["PYTHONPATH"] = str(REPO_ROOT)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-m", "msctl", *arguments],
        cwd=cwd,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


def _single_report(completed: subprocess.CompletedProcess[str]) -> dict:
    assert completed.stderr == ""
    assert completed.stdout.endswith("\n")
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert isinstance(report, dict)
    assert report["schema_version"] == 1
    assert isinstance(report["ok"], bool)
    return report


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--profile",
        str(PROFILE),
        "--repo-root",
        str(tmp_path),
        "--state-root",
        str(tmp_path / "state"),
    ]


def test_profile_freezes_illumina_seven_a100_layout():
    profile = json.loads(PROFILE.read_text())

    assert profile["schema_version"] == 1
    assert profile["profile_id"] == "illumina-usfc-prd"
    assert profile["provider"] == "illumina-usfc-prd"
    assert profile["cluster"] == "usfc-prd"
    assert profile["login"]["max_vcpus"] == 4
    assert profile["login"]["heavy_compute_allowed"] is False
    assert profile["cpu"] == {
        "nodes": 35,
        "cpus_per_node": 56,
        "local_ssd_required": True,
    }
    assert profile["gpu"]["model"] == "NVIDIA A100 80GB"
    assert profile["gpu"]["allocated"] == 7
    assert profile["gpu"]["seed0_train_groups"] == [3, 3]
    assert profile["gpu"]["evaluation_gpus"] == 1
    assert profile["storage"]["shared_root_env"] == "MS_SHARED_ROOT"
    assert profile["storage"]["shared_root_prefix"] == "/illumina"
    assert "MSCTL_APPROVAL_KEY" not in profile["job_env_allowlist"]


def test_runs_render_is_deterministic_dry_run_with_explicit_environment(tmp_path):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    args = [
        *_base_args(tmp_path),
        "runs",
        "render",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
    ]

    first = _run_msctl(*args)
    second = _run_msctl(*args)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    report = _single_report(first)
    assert report["ok"] is True
    assert report["command"] == "runs render"
    assert report["dry_run"] is True
    assert report["result"]["layout"] == {
        "allocated_gpus": 7,
        "train_groups": [3, 3],
        "evaluation_gpus": 1,
    }
    commands = report["result"]["commands"]
    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == ["sbatch", "--parsable"]
    assert "--gres=gpu:a100:7" in command
    exports = next(item for item in command if item.startswith("--export="))
    assert exports.startswith("--export=NONE,")
    assert "ALL" not in exports
    assert "MSCTL_APPROVAL_KEY" not in exports
    assert command[-1] == "cluster/slurm/v2_seed0.sbatch"


@pytest.mark.parametrize(
    "arguments",
    [
        ("not-a-command",),
        ("submit",),
        ("cleanup", "apply"),
        ("--help",),
    ],
)
def test_every_cli_path_emits_exactly_one_json_object(arguments):
    completed = _run_msctl(*arguments)

    report = _single_report(completed)
    if arguments == ("--help",):
        assert completed.returncode == 0
        assert report["ok"] is True
        assert "help" in report["result"]
    else:
        assert completed.returncode != 0
        assert report["ok"] is False
        assert report["error"]["code"] == "CLI_USAGE"


def test_submit_is_dry_run_by_default_and_does_not_require_approval(tmp_path):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
    )

    assert completed.returncode == 0
    report = _single_report(completed)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["result"]["submitted"] == 0
    assert not (tmp_path / "state").exists()


def test_submit_apply_requires_approval_before_any_external_command(tmp_path):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    marker = tmp_path / "sbatch-called"
    binary = tmp_path / "bin" / "sbatch"
    binary.parent.mkdir()
    binary.write_text(
        "#!/bin/sh\n"
        f"touch {marker}\n"
        "printf '777;usfc-prd\\n'\n"
    )
    binary.chmod(0o755)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--apply",
        env={"PATH": str(binary.parent)},
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == "APPROVAL_REQUIRED"
    assert not marker.exists()


def test_submit_uses_parsable_job_id_and_is_idempotent_for_active_runs(tmp_path):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "k" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    count = tmp_path / "sbatch-count"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        f"n=0; test ! -f '{count}' || n=$(cat '{count}')\n"
        f"n=$((n + 1)); printf '%s' \"$n\" > '{count}'\n"
        "printf '777;usfc-prd\\n'\n"
    )
    sbatch.chmod(0o755)
    squeue = bin_dir / "squeue"
    squeue.write_text("#!/bin/sh\nprintf '777|RUNNING\\n'\n")
    squeue.chmod(0o755)
    sacct = bin_dir / "sacct"
    sacct.write_text("#!/bin/sh\nexit 0\n")
    sacct.chmod(0o755)
    args = [
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(approval),
        "--apply",
    ]
    env = {
        "PATH": str(bin_dir),
        "MSCTL_APPROVAL_KEY": key,
    }

    first = _run_msctl(*args, env=env)
    second = _run_msctl(*args, env=env)

    assert first.returncode == second.returncode == 0
    first_report = _single_report(first)
    second_report = _single_report(second)
    assert first_report["result"]["job_id"] == "777"
    assert first_report["result"]["submitted"] == 1
    assert second_report["result"]["job_id"] == "777"
    assert second_report["result"]["submitted"] == 0
    assert second_report["result"]["idempotent"] is True
    assert count.read_text() == "1"
    for run_id in ("v2-dense-s0", "v2-split90-s0"):
        state = json.loads(
            (tmp_path / "state" / "runs" / f"{run_id}.json").read_text()
        )
        assert state["job_id"] == "777"
        assert state["status"] in {"SUBMITTED", "RUNNING"}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("expired", "APPROVAL_EXPIRED"),
        ("wrong_release", "APPROVAL_SCOPE_MISMATCH"),
        ("bad_signature", "APPROVAL_SIGNATURE_INVALID"),
        ("job_limit", "APPROVAL_LIMIT_EXCEEDED"),
    ],
)
def test_submit_fails_closed_on_invalid_approval(
    tmp_path,
    mutation,
    expected_code,
):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "s" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
        release_sha256=(
            "9" * 64 if mutation == "wrong_release" else "1" * 64
        ),
        expires_at=(
            "2000-01-01T00:00:00Z"
            if mutation == "expired"
            else "2999-01-01T00:00:00Z"
        ),
    )
    value = json.loads(approval.read_text())
    if mutation == "bad_signature":
        value["signature"] = "0" * 64
        _write_json(approval, value)
    elif mutation == "job_limit":
        value["limits"]["jobs"] = 1
        unsigned = {key_: item for key_, item in value.items() if key_ != "signature"}
        value["signature"] = hmac.new(
            key.encode(), _canonical(unsigned), hashlib.sha256
        ).hexdigest()
        _write_json(approval, value)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(approval),
        "--apply",
        env={"MSCTL_APPROVAL_KEY": key, "PATH": ""},
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == expected_code
    assert not (tmp_path / "state" / "runs").exists()


def test_capacity_check_fails_closed_when_slurm_is_unavailable():
    completed = _run_msctl(
        "--profile",
        str(PROFILE),
        "capacity",
        "check",
        env={"PATH": ""},
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == "EXTERNAL_UNAVAILABLE"
    assert report["error"]["details"]["operation"] == "capacity check"


def test_profile_and_slurm_assets_forbid_ambient_environment_exports():
    scripts = sorted((REPO_ROOT / "cluster" / "slurm").glob("v2_*.sbatch"))
    assert scripts
    assert any(path.name == "v2_seed0.sbatch" for path in scripts)
    assert any(path.name == "v2_evaluate.sbatch" for path in scripts)
    for script in scripts:
        text = script.read_text()
        assert "--export=ALL" not in text
        assert "MSCTL_APPROVAL_KEY" not in text
        assert "env -i" in text or "job_env_allowlist" in text


def test_seed0_slurm_script_is_symmetric_and_reserves_gpu_seven():
    text = (REPO_ROOT / "cluster" / "slurm" / "v2_seed0.sbatch").read_text()

    assert "#SBATCH --gres=gpu:a100:7" in text
    assert "0,1,2" in text
    assert "3,4,5" in text
    assert "GPU 6" in text or "gpu 6" in text
    assert text.count("--nproc_per_node=3") == 2
    assert "--resume auto" in text


def test_agent_start_and_project_skill_enforce_plan_then_apply():
    start = (REPO_ROOT / "AGENT-START.md").read_text()
    skill = (
        REPO_ROOT / ".cursor" / "skills" / "memorysplit-cluster" / "SKILL.md"
    ).read_text()

    for text in (start, skill):
        lowered = text.lower()
        assert "msctl" in lowered
        assert "dry-run" in lowered
        assert "--apply" in lowered
        assert "approval" in lowered
        assert "stop" in lowered
        assert "secret" in lowered
        assert "duplicate" in lowered or "resubmit" in lowered
    assert skill.startswith("---\nname: memorysplit-cluster\n")
    assert "Use when" in skill.split("---", 2)[1]


def _write_executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(0o755)
    return path


def _submitted_pair(tmp_path: Path) -> tuple[Path, Path, dict, str, Path]:
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "q" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    bin_dir = tmp_path / "bin"
    _write_executable(bin_dir / "sbatch", "printf '777;usfc-prd\\n'\n")
    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )
    assert completed.returncode == 0, completed.stdout
    return release, manifest, runs, key, bin_dir


def _scope_approval(
    tmp_path: Path,
    *,
    operation: str,
    scope_sha256: str,
    key: str,
    jobs: int,
    gpu_hours: float = 0.0,
) -> Path:
    unsigned = {
        "schema_version": 1,
        "receipt_id": f"approve-{operation}-scope",
        "provider": "illumina-usfc-prd",
        "operation": operation,
        "release_sha256": "1" * 64,
        "run_manifest_sha256": scope_sha256,
        "limits": {"gpu_hours": gpu_hours, "jobs": jobs},
        "expires_at": "2999-01-01T00:00:00Z",
        "key_id": "operator-test",
    }
    unsigned["signature"] = hmac.new(
        key.encode(), _canonical(unsigned), hashlib.sha256
    ).hexdigest()
    return _write_json(tmp_path / f"{operation}-scope-approval.json", unsigned)


def test_env_ensure_apply_builds_hash_bound_environment_and_is_idempotent(
    tmp_path,
):
    lock = tmp_path / "requirements-illumina.lock"
    lock.write_text("# empty test lock; production lock contains hashes\n")
    environment = tmp_path / "environment"
    args = [
        *_base_args(tmp_path),
        "env",
        "ensure",
        "--root",
        str(environment),
        "--lock",
        str(lock),
        "--apply",
    ]

    first = _run_msctl(*args)
    second = _run_msctl(*args)

    assert first.returncode == second.returncode == 0
    first_report = _single_report(first)
    second_report = _single_report(second)
    assert first_report["result"]["created"] is True
    assert second_report["result"]["created"] is False
    assert (environment / "bin" / "python").is_file()
    receipt = json.loads((environment / "msctl-env-receipt.json").read_text())
    assert receipt["lock_sha256"] == _sha256(lock)
    assert receipt["provider"] == "illumina-usfc-prd"


def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    shard = root / "shards" / "000.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"deterministic shard")
    files = [
        {
            "path": "shards/000.bin",
            "bytes": shard.stat().st_size,
            "sha256": _sha256(shard),
        }
    ]
    receipt = {
        "schema_version": 1,
        "dataset_id": "memorysplit-v2-20x-seed0",
        "provider": "illumina-usfc-prd",
        "release_sha256": "1" * 64,
        "dataset_sha256": "4" * 64,
        "ordered_stream_sha256": "5" * 64,
        "merkle_root": "6" * 64,
        "files": files,
    }
    _write_json(root / "dataset-receipt.json", receipt)
    pointer = _write_json(
        tmp_path / "DATASET-POINTER.json",
        {
            "schema_version": 1,
            "dataset_id": "memorysplit-v2-20x-seed0",
            "provider": "illumina-usfc-prd",
            "shared_root_env": "MS_SHARED_ROOT",
            "shared_root_prefix": "/illumina",
            "relative_path": "memorysplit/datasets/v2-20x-seed0",
            "materialization": "slurm",
            "full_corpus_in_release": False,
            "source_lock_manifest": "configs/reasoning-dataset-v2.json",
            "required_receipt": "dataset-receipt.json",
        },
    )
    return pointer, root


def test_dataset_verify_hashes_every_receipted_file_and_rejects_tampering(
    tmp_path,
):
    pointer, root = _dataset_fixture(tmp_path)
    args = [
        *_base_args(tmp_path),
        "dataset",
        "verify",
        "--pointer",
        str(pointer),
        "--dataset-root",
        str(root),
    ]

    valid = _run_msctl(*args)
    (root / "shards" / "000.bin").write_bytes(b"tampered")
    invalid = _run_msctl(*args)

    assert valid.returncode == 0
    valid_report = _single_report(valid)
    assert valid_report["result"]["verified_files"] == 1
    assert valid_report["result"]["dataset_sha256"] == "4" * 64
    assert invalid.returncode != 0
    assert _single_report(invalid)["error"]["code"] == "DATASET_HASH_MISMATCH"


def test_dataset_ensure_apply_fails_closed_without_cluster_builder(tmp_path):
    pointer, _ = _dataset_fixture(tmp_path)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "dataset",
        "ensure",
        "--pointer",
        str(pointer),
        "--apply",
        env={"PATH": ""},
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == "EXTERNAL_UNAVAILABLE"
    assert report["error"]["details"]["operation"] == "dataset ensure"


def test_status_reconciles_squeue_and_cached_mode_never_calls_slurm(tmp_path):
    _, manifest, _, _, bin_dir = _submitted_pair(tmp_path)
    _write_executable(bin_dir / "squeue", "printf '777|RUNNING\\n'\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")
    live = _run_msctl(
        *_base_args(tmp_path),
        "status",
        "--manifest",
        str(manifest),
        env={"PATH": str(bin_dir)},
    )
    for path in bin_dir.iterdir():
        path.unlink()
    cached = _run_msctl(
        *_base_args(tmp_path),
        "status",
        "--manifest",
        str(manifest),
        "--cached",
        env={"PATH": ""},
    )

    assert live.returncode == cached.returncode == 0
    assert _single_report(live)["result"]["status"] == "RUNNING"
    cached_report = _single_report(cached)
    assert cached_report["result"]["status"] == "RUNNING"
    assert cached_report["result"]["authoritative"] is False


def _checkpoint_receipt(
    tmp_path: Path,
    *,
    runs: dict,
    mismatch: bool = False,
) -> Path:
    checkpoints = []
    for row in runs["runs"]:
        path = tmp_path / "checkpoints" / f"{row['run_id']}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"checkpoint:{row['run_id']}".encode())
        checkpoints.append(
            {
                "run_id": row["run_id"],
                "path": f"checkpoints/{path.name}",
                "sha256": _sha256(path),
                "config_sha256": (
                    "0" * 64 if mismatch else row["config_sha256"]
                ),
                "step": 200,
                "world_size": 3,
            }
        )
    return _write_json(
        tmp_path / "checkpoint-receipt.json",
        {
            "schema_version": 1,
            "provider": "illumina-usfc-prd",
            "release_sha256": "1" * 64,
            "run_manifest_sha256": hashlib.sha256(
                _canonical(runs)
            ).hexdigest(),
            "dataset_sha256": "4" * 64,
            "checkpoints": checkpoints,
        },
    )


def test_resume_reconciles_terminal_job_and_requires_matching_checkpoints(tmp_path):
    release, manifest, runs, key, bin_dir = _submitted_pair(tmp_path)
    _write_executable(bin_dir / "squeue", "exit 0\n")
    _write_executable(bin_dir / "sacct", "printf '777|FAILED\\n'\n")
    _write_executable(bin_dir / "sbatch", "printf '888;usfc-prd\\n'\n")
    receipt = _checkpoint_receipt(tmp_path, runs=runs)
    approval = _approval(
        tmp_path,
        operation="resume",
        runs=runs,
        key=key,
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        "resume",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--checkpoint-receipt",
        str(receipt),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )

    assert completed.returncode == 0
    report = _single_report(completed)
    assert report["result"]["job_id"] == "888"
    assert report["result"]["attempt"] == 2
    for row in runs["runs"]:
        state = json.loads(
            (tmp_path / "state" / "runs" / f"{row['run_id']}.json").read_text()
        )
        assert state["attempt"] == 2
        assert state["job_id"] == "888"


def test_resume_rejects_checkpoint_provenance_before_sbatch(tmp_path):
    release, manifest, runs, key, bin_dir = _submitted_pair(tmp_path)
    marker = tmp_path / "resubmitted"
    _write_executable(bin_dir / "squeue", "exit 0\n")
    _write_executable(bin_dir / "sacct", "printf '777|FAILED\\n'\n")
    _write_executable(bin_dir / "sbatch", f"touch '{marker}'\n")
    receipt = _checkpoint_receipt(tmp_path, runs=runs, mismatch=True)
    approval = _approval(
        tmp_path,
        operation="resume",
        runs=runs,
        key=key,
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        "resume",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--checkpoint-receipt",
        str(receipt),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )

    assert completed.returncode != 0
    assert (
        _single_report(completed)["error"]["code"]
        == "CHECKPOINT_PROVENANCE_MISMATCH"
    )
    assert not marker.exists()


def test_cancel_requires_approval_reconciles_then_calls_scancel(tmp_path):
    release, manifest, runs, key, bin_dir = _submitted_pair(tmp_path)
    marker = tmp_path / "cancelled"
    _write_executable(bin_dir / "squeue", "printf '777|RUNNING\\n'\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")
    _write_executable(bin_dir / "scancel", f"printf 'yes' > '{marker}'\n")
    approval = _approval(
        tmp_path,
        operation="cancel",
        runs=runs,
        key=key,
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        "cancel",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )

    assert completed.returncode == 0
    assert marker.read_text() == "yes"
    assert _single_report(completed)["result"]["status"] == "CANCEL_REQUESTED"


def test_evaluate_approval_submits_one_reserved_gpu_job_idempotently(tmp_path):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "e" * 32
    approval = _approval(
        tmp_path,
        operation="evaluate",
        runs=runs,
        key=key,
    )
    count = tmp_path / "eval-count"
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "sbatch",
        f"printf x >> '{count}'\nprintf '999;usfc-prd\\n'\n",
    )
    _write_executable(bin_dir / "squeue", "printf '999|PENDING\\n'\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")
    args = [
        *_base_args(tmp_path),
        "evaluate",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(approval),
        "--apply",
    ]
    environment = {"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key}

    first = _run_msctl(*args, env=environment)
    second = _run_msctl(*args, env=environment)

    assert first.returncode == second.returncode == 0
    first_report = _single_report(first)
    second_report = _single_report(second)
    command = first_report["result"]["command"]
    assert "--gres=gpu:a100:1" in command
    assert first_report["result"]["job_id"] == "999"
    assert second_report["result"]["idempotent"] is True
    assert count.read_text() == "x"


def test_collect_apply_copies_only_closed_evidence_allowlist(tmp_path):
    source = tmp_path / "source-runs"
    run = source / "v2-dense-s0"
    (run / "evals").mkdir(parents=True)
    (run / "log.jsonl").write_text('{"step":1}\n')
    (run / "config.json").write_text('{"run":"dense"}\n')
    (run / "evals" / "summary.json").write_text('{"ok":true}\n')
    (run / "run-receipt.json").write_text('{"verified":true}\n')
    (run / "SHA256SUMS").write_text("abc  summary.json\n")
    (run / "checkpoint.pt").write_bytes(b"must not collect")
    out = tmp_path / "collected"

    completed = _run_msctl(
        *_base_args(tmp_path),
        "collect",
        "--source",
        str(source),
        "--out",
        str(out),
        "--apply",
    )

    assert completed.returncode == 0
    report = _single_report(completed)
    assert report["result"]["collected_files"] == 5
    assert (out / "v2-dense-s0" / "evals" / "summary.json").is_file()
    assert (out / "v2-dense-s0" / "run-receipt.json").is_file()
    assert not (out / "v2-dense-s0" / "checkpoint.pt").exists()
    collection = json.loads((out / "COLLECTION.json").read_text())
    assert len(collection["files"]) == 5


def test_cleanup_apply_is_hash_bound_and_preflights_races(tmp_path):
    release = _release(tmp_path)
    root = tmp_path / "runs"
    first = root / "v2-dense-s0" / "logs" / "worker.log"
    second = root / "v2-dense-s0" / "cache" / "compile.bin"
    checkpoint = root / "v2-dense-s0" / "checkpoint.pt"
    for path, data in (
        (first, b"log"),
        (second, b"cache"),
        (checkpoint, b"keep"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    planned = _run_msctl(
        *_base_args(tmp_path),
        "cleanup",
        "plan",
        "--root",
        str(root),
    )
    assert planned.returncode == 0
    plan = _single_report(planned)["result"]["plan"]
    assert {row["path"] for row in plan["files"]} == {
        "v2-dense-s0/cache/compile.bin",
        "v2-dense-s0/logs/worker.log",
    }
    plan_path = _write_json(tmp_path / "cleanup-plan.json", plan)
    key = "c" * 32
    approval = _scope_approval(
        tmp_path,
        operation="cleanup",
        scope_sha256=hashlib.sha256(_canonical(plan)).hexdigest(),
        key=key,
        jobs=2,
    )
    second.write_bytes(b"raced")

    raced = _run_msctl(
        *_base_args(tmp_path),
        "cleanup",
        "apply",
        "--plan",
        str(plan_path),
        "--release",
        str(release),
        "--approval",
        str(approval),
        "--apply",
        env={"MSCTL_APPROVAL_KEY": key},
    )

    assert raced.returncode != 0
    assert _single_report(raced)["error"]["code"] == "CLEANUP_RACE"
    assert first.exists() and second.exists() and checkpoint.exists()

    second.write_bytes(b"cache")
    applied = _run_msctl(
        *_base_args(tmp_path),
        "cleanup",
        "apply",
        "--plan",
        str(plan_path),
        "--release",
        str(release),
        "--approval",
        str(approval),
        "--apply",
        env={"MSCTL_APPROVAL_KEY": key},
    )
    assert applied.returncode == 0
    assert _single_report(applied)["result"]["deleted_files"] == 2
    assert not first.exists() and not second.exists()
    assert checkpoint.exists()


def test_external_errors_redact_approval_key_and_token(tmp_path):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "super-secret-approval-key-value!!"
    fake_token = "hf_" + "abcdefgh12345678"
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "sbatch",
        f"printf 'token={fake_token}\\n' >&2\nexit 1\n",
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    serialized = json.dumps(report)
    assert key not in serialized
    assert fake_token not in serialized
    assert "[REDACTED]" in serialized


def test_capacity_check_validates_known_cpu_and_a100_floor(tmp_path):
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "sinfo",
        "printf 'cpu|(null)|35|56\\ngpu|gpu:a100:8|1|56\\n'\n",
    )
    _write_executable(bin_dir / "squeue", "exit 0\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")

    completed = _run_msctl(
        "--profile",
        str(PROFILE),
        "capacity",
        "check",
        env={"PATH": str(bin_dir), "USER": "operator"},
    )

    assert completed.returncode == 0
    report = _single_report(completed)
    assert report["result"]["observed"]["cpu_nodes"] == 35
    assert report["result"]["observed"]["a100_gpus"] == 8
    assert report["result"]["sufficient"] is True


def test_capacity_check_fails_closed_on_insufficient_a100s(tmp_path):
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "sinfo",
        "printf 'cpu|(null)|35|56\\ngpu|gpu:a100:4|1|56\\n'\n",
    )
    _write_executable(bin_dir / "squeue", "exit 0\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")

    completed = _run_msctl(
        "--profile",
        str(PROFILE),
        "capacity",
        "check",
        env={"PATH": str(bin_dir), "USER": "operator"},
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == "CAPACITY_INSUFFICIENT"


def test_dataset_verify_rejects_unreceipted_symlinks(tmp_path):
    pointer, root = _dataset_fixture(tmp_path)
    os.symlink("shards/000.bin", root / "latest.bin")

    completed = _run_msctl(
        *_base_args(tmp_path),
        "dataset",
        "verify",
        "--pointer",
        str(pointer),
        "--dataset-root",
        str(root),
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == "DATASET_FILE_UNSAFE"


def test_collection_idempotence_authenticates_existing_receipt(tmp_path):
    source = tmp_path / "source-runs"
    run = source / "v2-dense-s0"
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text('{"step":1}\n')
    out = tmp_path / "collected"
    args = [
        *_base_args(tmp_path),
        "collect",
        "--source",
        str(source),
        "--out",
        str(out),
        "--apply",
    ]
    first = _run_msctl(*args)
    assert first.returncode == 0
    (out / "COLLECTION.json").write_text('{"tampered":true}\n')

    second = _run_msctl(*args)

    assert second.returncode != 0
    assert (
        _single_report(second)["error"]["code"]
        == "COLLECT_DESTINATION_EXISTS"
    )


def test_render_exports_runtime_roots_by_name_and_seed_script_hashes_configs(
    tmp_path,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "runs",
        "render",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
    )

    command = _single_report(completed)["result"]["commands"][0]
    exports = next(item for item in command if item.startswith("--export="))
    for name in ("MS_SHARED_ROOT", "MS_ENV_ROOT", "MS_DATA_ROOT", "MS_OUT_ROOT"):
        assert name in exports
    script = (REPO_ROOT / "cluster" / "slurm" / "v2_seed0.sbatch").read_text()
    assert "MS_DENSE_CONFIG_SHA256" in script
    assert "MS_SPLIT_CONFIG_SHA256" in script
    assert "hashlib.sha256" in script
