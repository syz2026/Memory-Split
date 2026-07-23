from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "cluster" / "profiles" / "illumina-usfc-prd.json"
HEX64 = "a" * 64
TRAIN_ENTRYPOINT = b"""\
MSCTL_DDP_CONTRACT = "memorysplit-ddp-v1"
FLAGS = ("--config", "--resume-path")
"""
EVALUATOR_ENTRYPOINT = b"""\
MSCTL_EVALUATOR_CONTRACT = "memorysplit-confirmatory-evaluator-v1"
FLAGS = ("evaluate", "--run", "--sealed-release", "--device")
"""


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


def _test_profile(tmp_path: Path) -> Path:
    path = tmp_path / "cluster" / "profiles" / PROFILE.name
    if path.is_file():
        return path
    value = json.loads(PROFILE.read_text())
    value["environment"] = {
        "status": "pinned",
        "lock_path": "requirements-illumina.lock",
        "contract": {
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "platform": "linux_x86_64",
            "cuda_version": "12.4",
        },
    }
    for name in (
        "MS_RELEASE_ROOT",
        "MS_TRAIN_ENTRYPOINT",
        "MS_EVALUATOR_ENTRYPOINT",
    ):
        if name not in value["job_env_allowlist"]:
            value["job_env_allowlist"].append(name)
    return _write_json(path, value)


def _release(
    tmp_path: Path,
    *,
    train_entrypoint_bytes: bytes = TRAIN_ENTRYPOINT,
    evaluator_entrypoint_bytes: bytes = EVALUATOR_ENTRYPOINT,
) -> Path:
    release_path = tmp_path / "RELEASE.json"
    if release_path.is_file():
        return release_path
    pointer, _ = _dataset_fixture(tmp_path)
    source_lock = tmp_path / "configs" / "reasoning-dataset-v2.json"
    source_lock.parent.mkdir(parents=True, exist_ok=True)
    source_lock.write_text('{"schema_version":2,"fixture":true}\n')
    profile_copy = _test_profile(tmp_path)
    dense = _write_json(
        tmp_path / "configs" / "v2" / "dense-s0.json",
        {"arm": "dense", "seed": 0},
    )
    split = _write_json(
        tmp_path / "configs" / "v2" / "split90-s0.json",
        {"arm": "split90", "seed": 0},
    )
    seed_script = tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch"
    evaluate_script = tmp_path / "cluster" / "slurm" / "v2_evaluate.sbatch"
    seed_script.parent.mkdir(parents=True, exist_ok=True)
    seed_script.write_bytes(
        (REPO_ROOT / "cluster" / "slurm" / seed_script.name).read_bytes()
    )
    evaluate_script.write_bytes(
        (REPO_ROOT / "cluster" / "slurm" / evaluate_script.name).read_bytes()
    )
    train_entrypoint = tmp_path / "scripts" / "run_train.py"
    train_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    train_entrypoint.write_bytes(train_entrypoint_bytes)
    evaluator_entrypoint = tmp_path / "evals" / "confirmatory" / "runner.py"
    evaluator_entrypoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator_entrypoint.write_bytes(evaluator_entrypoint_bytes)
    environment_lock = tmp_path / "requirements-illumina.lock"
    environment_lock.write_text(
        "# memorysplit-illumina-lock-v1\n"
        "# platform: linux_x86_64\n"
        "# python-implementation: CPython\n"
        "# python-version: 3.12.0\n"
        "# cuda-version: 12.4\n"
        "fixture==1 --hash=sha256:" + "1" * 64 + "\n"
    )
    source_members = {
        "DATASET-POINTER.json": pointer.read_bytes(),
        "cluster/profiles/illumina-usfc-prd.json": profile_copy.read_bytes(),
        "cluster/slurm/v2_evaluate.sbatch": evaluate_script.read_bytes(),
        "cluster/slurm/v2_seed0.sbatch": seed_script.read_bytes(),
        "configs/reasoning-dataset-v2.json": source_lock.read_bytes(),
        "configs/v2/dense-s0.json": dense.read_bytes(),
        "configs/v2/split90-s0.json": split.read_bytes(),
        "evals/confirmatory/runner.py": evaluator_entrypoint.read_bytes(),
        "requirements-illumina.lock": environment_lock.read_bytes(),
        "scripts/run_train.py": train_entrypoint.read_bytes(),
    }
    member_rows = [
        {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob": "2" * 40,
        }
        for relative, data in sorted(source_members.items())
    ]
    metadata = (
        json.dumps(
            {
                "schema_version": 1,
                "provider": "illumina-usfc-prd",
                "source": {"commit": "2" * 40, "dirty": False},
                "profile_sha256": hashlib.sha256(
                    source_members[
                        "cluster/profiles/illumina-usfc-prd.json"
                    ]
                ).hexdigest(),
                "environment_hashes": {
                    "requirements-illumina.lock": hashlib.sha256(
                        environment_lock.read_bytes()
                    ).hexdigest()
                },
                "members": member_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    payload = {**source_members, "RELEASE-METADATA.json": metadata}
    sums = "".join(
        f"{hashlib.sha256(payload[name]).hexdigest()}  {name}\n"
        for name in sorted(payload)
    ).encode("ascii")
    payload["SHA256SUMS"] = sums
    (tmp_path / "RELEASE-METADATA.json").write_bytes(metadata)
    (tmp_path / "SHA256SUMS").write_bytes(sums)
    archive = tmp_path / "ms-illumina-r1-test.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            handle.writestr(info, data)
    archive_hash = _sha256(archive)
    (tmp_path / f"{archive.name}.sha256").write_text(
        f"{archive_hash}  {archive.name}\n"
    )
    return _write_json(
        release_path,
        {
            "schema_version": 1,
            "release_id": "r1-test",
            "provider": "illumina-usfc-prd",
            "archive": {
                "path": archive.name,
                "sha256": archive_hash,
                "bytes": archive.stat().st_size,
            },
            "source": {"commit": "2" * 40, "dirty": False},
            "members_sha256": hashlib.sha256(sums).hexdigest(),
        },
    )


def _runs(tmp_path: Path) -> tuple[Path, dict]:
    release = _release(tmp_path)
    configs = tmp_path / "configs" / "v2"
    dense = configs / "dense-s0.json"
    split = configs / "split90-s0.json"
    _write_json(dense, {"arm": "dense", "seed": 0})
    _write_json(split, {"arm": "split90", "seed": 0})
    pointer, dataset_root = _dataset_fixture(tmp_path)
    parallel_receipt = json.loads(
        (dataset_root / "receipt.json").read_text()
    )
    source_lock = tmp_path / "configs" / "reasoning-dataset-v2.json"
    from msctl.dataset import dataset_identity

    identity = dataset_identity(
        pointer=json.loads(pointer.read_text()),
        parallel_receipt=parallel_receipt,
        receipt_sha256=_sha256(dataset_root / "receipt.json"),
        source_lock_sha256=_sha256(source_lock),
    )
    release_value = json.loads(release.read_text())
    value = {
        "schema_version": 1,
        "provider": "illumina-usfc-prd",
        "release_sha256": release_value["archive"]["sha256"],
        "dataset_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
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
    release_sha256: str | None = None,
    expires_at: str = "2999-01-01T00:00:00Z",
    gpu_hours: float | None = None,
) -> Path:
    from msctl.profile import load_profile
    from msctl.slurm import resource_request

    resources = resource_request(load_profile(_test_profile(tmp_path)), operation)
    if release_sha256 is None:
        release_sha256 = str(runs["release_sha256"])
    if gpu_hours is None:
        gpu_hours = float(resources["gpu_hours"])
    unsigned = {
        "schema_version": 1,
        "receipt_id": f"approve-{operation}",
        "provider": "illumina-usfc-prd",
        "operation": operation,
        "release_sha256": release_sha256,
        "run_manifest_sha256": hashlib.sha256(_canonical(runs)).hexdigest(),
        "resources": resources,
        "limits": {"gpu_hours": gpu_hours, "jobs": int(resources["jobs"])},
        "expires_at": expires_at,
        "key_id": "operator-test",
    }
    unsigned["signature"] = hmac.new(
        key.encode(), _canonical(unsigned), hashlib.sha256
    ).hexdigest()
    return _write_json(tmp_path / f"{operation}-approval.json", unsigned)


def _environment_receipt(tmp_path: Path) -> Path:
    from msctl.profile import load_profile

    _release(tmp_path)
    profile = load_profile(_test_profile(tmp_path))
    root = tmp_path / "environment"
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return _write_json(
        root / "msctl-env-receipt.json",
        {
            "schema_version": 1,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
            "lock_sha256": _sha256(tmp_path / "requirements-illumina.lock"),
            "environment_root": str(root.resolve()),
            "python": profile.python_version,
            "platform": profile.platform,
            "cuda_version": profile.cuda_version,
            "created_at": "2026-07-23T00:00:00Z",
        },
    )


def _run_msctl(
    *arguments: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    bind_dataset: bool = True,
    bind_environment: bool = True,
) -> subprocess.CompletedProcess[str]:
    rendered = list(arguments)
    if "--repo-root" in rendered:
        root = Path(rendered[rendered.index("--repo-root") + 1])
        if bind_dataset:
            if any(
                command in rendered
                for command in ("submit", "resume", "evaluate")
            ) or ("runs" in rendered and "render" in rendered):
                if "--dataset-pointer" not in rendered:
                    rendered.extend(
                        [
                            "--dataset-pointer",
                            str(root / "DATASET-POINTER.json"),
                            "--dataset-root",
                            str(_dataset_fixture(root)[1]),
                        ]
                    )
            if "dataset" in rendered and "verify" in rendered:
                release = _release(root)
                manifest, _ = _runs(root)
                if "--release" not in rendered:
                    rendered.extend(
                        [
                            "--release",
                            str(release),
                            "--manifest",
                            str(manifest),
                        ]
                    )
        if bind_environment and (
            any(command in rendered for command in ("submit", "resume", "evaluate"))
            or ("runs" in rendered and "render" in rendered)
        ):
            if "--environment-receipt" not in rendered:
                rendered.extend(
                    [
                        "--environment-receipt",
                        str(_environment_receipt(root)),
                    ]
                )
    merged = dict(os.environ)
    merged["PYTHONPATH"] = str(REPO_ROOT)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-m", "msctl", *rendered],
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
        str(_test_profile(tmp_path)),
        "--repo-root",
        str(tmp_path),
        "--state-root",
        str(tmp_path / "state"),
    ]


def _production_base_args(tmp_path: Path) -> list[str]:
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
    assert not exports.startswith("--export=NONE,")
    assert "ALL" not in exports
    assert "NONE" not in exports
    assert "MSCTL_APPROVAL_KEY" not in exports
    assert command[-1] == str(
        (tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch").resolve()
    )


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
    ("discovery", "expected_code"),
    [
        ("one", None),
        ("none", "SUBMISSION_UNCERTAIN"),
        ("multiple", "SUBMISSION_MULTIPLE_MATCHES"),
    ],
)
def test_submit_recovers_exact_intent_or_refuses_uncertain_resubmission(
    tmp_path,
    discovery,
    expected_code,
):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "r" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "sbatch", "printf 'not-a-job-id\\n'\n")
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

    interrupted = _run_msctl(*args, env=env)
    assert interrupted.returncode != 0
    state = json.loads(
        (tmp_path / "state" / "runs" / "v2-dense-s0.json").read_text()
    )
    submission_key = state["submission_key"]
    marker = tmp_path / "unexpected-resubmit"
    _write_executable(bin_dir / "sbatch", f"touch '{marker}'\n")
    submission_comment = f"msctl:{submission_key}"
    if discovery == "one":
        queue_output = f"printf '777|{submission_comment}|RUNNING\\n'\n"
    elif discovery == "multiple":
        queue_output = (
            f"printf '777|{submission_comment}|RUNNING\\n"
            f"778|{submission_comment}|PENDING\\n'\n"
        )
    else:
        queue_output = "exit 0\n"
    _write_executable(bin_dir / "squeue", queue_output)
    _write_executable(bin_dir / "sacct", "exit 0\n")

    recovered = _run_msctl(*args, env=env)

    assert not marker.exists()
    report = _single_report(recovered)
    if expected_code is None:
        assert recovered.returncode == 0
        assert report["result"]["job_id"] == "777"
        assert report["result"]["submitted"] == 0
        assert report["result"]["idempotent"] is True
    else:
        assert recovered.returncode != 0
        assert report["error"]["code"] == expected_code
        if expected_code == "SUBMISSION_UNCERTAIN":
            assert report["error"]["details"]["recoverable"] is True


def test_submit_recovery_repairs_a_partial_post_sbatch_state_update(tmp_path):
    release, manifest, _, key, bin_dir = _submitted_pair(tmp_path)
    dense_path = tmp_path / "state" / "runs" / "v2-dense-s0.json"
    split_path = tmp_path / "state" / "runs" / "v2-split90-s0.json"
    split = json.loads(split_path.read_text())
    split["job_id"] = None
    split["status"] = "SUBMITTING"
    _write_json(split_path, split)
    _write_executable(
        bin_dir / "squeue",
        "printf '777|RUNNING\\n'\n",
    )
    _write_executable(bin_dir / "sacct", "exit 0\n")
    marker = tmp_path / "unexpected-resubmit"
    _write_executable(bin_dir / "sbatch", f"touch '{marker}'\n")

    recovered = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--approval",
        str(tmp_path / "submit-approval.json"),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )

    assert recovered.returncode == 0
    assert not marker.exists()
    assert _single_report(recovered)["result"]["job_id"] == "777"
    assert json.loads(dense_path.read_text())["job_id"] == "777"
    assert json.loads(split_path.read_text())["job_id"] == "777"


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
            "9" * 64
            if mutation == "wrong_release"
            else str(runs["release_sha256"])
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
        value["limits"]["jobs"] = 0
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
    assert "--resume auto" not in text
    assert "memorysplit-ddp-v1" in text


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
    from msctl.profile import load_profile
    from msctl.slurm import resource_request

    release_sha256 = json.loads(_release(tmp_path).read_text())["archive"][
        "sha256"
    ]
    unsigned = {
        "schema_version": 1,
        "receipt_id": f"approve-{operation}-scope",
        "provider": "illumina-usfc-prd",
        "operation": operation,
        "release_sha256": release_sha256,
        "run_manifest_sha256": scope_sha256,
        "resources": resource_request(load_profile(PROFILE), operation),
        "limits": {"gpu_hours": gpu_hours, "jobs": jobs},
        "expires_at": "2999-01-01T00:00:00Z",
        "key_id": "operator-test",
    }
    unsigned["signature"] = hmac.new(
        key.encode(), _canonical(unsigned), hashlib.sha256
    ).hexdigest()
    return _write_json(tmp_path / f"{operation}-scope-approval.json", unsigned)


def test_env_ensure_dry_run_reports_missing_site_contract(tmp_path):
    lock = tmp_path / "requirements-illumina.lock"
    lock.write_text("# empty test lock; production lock contains hashes\n")
    environment = tmp_path / "environment"
    args = [
        *_production_base_args(tmp_path),
        "env",
        "ensure",
        "--root",
        str(environment),
        "--lock",
        str(lock),
    ]

    completed = _run_msctl(*args)

    assert completed.returncode == 0
    report = _single_report(completed)
    assert report["dry_run"] is True
    assert report["result"]["created"] is False
    assert report["result"]["missing_operator_inputs"] == [
        "python_version",
        "cuda_version",
    ]
    assert not environment.exists()


def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    from corpusgen.parallel import (
        FixtureRenderer,
        ParallelBuildConfig,
        build_parallel_corpus,
        fixture_catalog,
    )

    root = (
        tmp_path
        / "publication"
        / "memorysplit"
        / "datasets"
        / "v2-20x-seed0"
    )
    if not root.exists():
        build_parallel_corpus(
            fixture_catalog(record_count=6),
            FixtureRenderer(),
            ParallelBuildConfig(
                lane_weights=(
                    ("natural", 1),
                    ("facts", 1),
                    ("reasoning", 1),
                ),
                update_tokens=64,
                allow_fewer_shards=True,
            ),
            root,
        )
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
            "required_receipt": "receipt.json",
            "receipt_format": "memorysplit-parallel-corpus-v1",
            "identity_scheme": "memorysplit-dataset-binding-v1",
            "verification_receipt_format": (
                "memorysplit-dataset-verification-v1"
            ),
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
    shard = next((root / "shards").glob("*.bin"))
    shard.write_bytes(b"tampered")
    invalid = _run_msctl(*args)

    assert valid.returncode == 0
    valid_report = _single_report(valid)
    assert valid_report["result"]["verified_files"] >= 5
    manifest = json.loads((tmp_path / "runs.json").read_text())
    assert (
        valid_report["result"]["dataset_sha256"]
        == manifest["dataset_sha256"]
    )
    assert invalid.returncode != 0
    assert (
        _single_report(invalid)["error"]["code"]
        == "DATASET_RECEIPT_INVALID"
    )


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
    release, manifest, _, _, bin_dir = _submitted_pair(tmp_path)
    _write_executable(bin_dir / "squeue", "printf '777|RUNNING\\n'\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")
    live = _run_msctl(
        *_base_args(tmp_path),
        "status",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        env={"PATH": str(bin_dir)},
    )
    for path in bin_dir.iterdir():
        path.unlink()
    cached = _run_msctl(
        *_base_args(tmp_path),
        "status",
        "--release",
        str(release),
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
            "release_sha256": runs["release_sha256"],
            "run_manifest_sha256": hashlib.sha256(
                _canonical(runs)
            ).hexdigest(),
            "dataset_sha256": runs["dataset_sha256"],
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


def test_resume_never_resubmits_when_slurm_cannot_prove_terminal_state(tmp_path):
    release, manifest, runs, key, bin_dir = _submitted_pair(tmp_path)
    _write_executable(bin_dir / "squeue", "exit 0\n")
    _write_executable(bin_dir / "sacct", "exit 0\n")
    marker = tmp_path / "unexpected-resume"
    _write_executable(bin_dir / "sbatch", f"touch '{marker}'\n")
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

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == "STATUS_UNKNOWN"
    assert not marker.exists()


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


def test_cleanup_quarantines_and_checks_the_exact_opened_inode(
    tmp_path,
    monkeypatch,
):
    import msctl.cleanup as cleanup_module
    from msctl.errors import MsctlError
    from msctl.profile import load_profile

    release = _release(tmp_path)
    root = tmp_path / "runs"
    target = root / "v2-dense-s0" / "logs" / "worker.log"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"planned inode")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement inode")
    plan = cleanup_module.make_cleanup_plan(root)
    plan_path = _write_json(tmp_path / "cleanup-plan.json", plan)
    key = "q" * 32
    approval = _scope_approval(
        tmp_path,
        operation="cleanup",
        scope_sha256=hashlib.sha256(_canonical(plan)).hexdigest(),
        key=key,
        jobs=1,
    )
    real_rename = os.rename
    raced = False

    def racing_rename(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        nonlocal raced
        if source == target.name and src_dir_fd is not None and not raced:
            raced = True
            real_rename(
                source,
                "displaced.log",
                src_dir_fd=src_dir_fd,
                dst_dir_fd=src_dir_fd,
            )
            os.replace(replacement, target)
        return real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(cleanup_module.os, "rename", racing_rename)

    with pytest.raises(MsctlError) as caught:
        cleanup_module.apply_cleanup(
            profile=load_profile(PROFILE),
            plan_path=plan_path,
            release_path=release,
            approval_path=approval,
            apply=True,
            environ={"MSCTL_APPROVAL_KEY": key},
        )

    assert caught.value.code == "CLEANUP_RACE"
    assert raced is True
    assert target.read_bytes() == b"replacement inode"
    assert (target.parent / "displaced.log").read_bytes() == b"planned inode"


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
        (
            f"printf 'token={fake_token}\\n' >&2\n"
            f"printf '{fake_token}\\n' >&2\n"
            "exit 1\n"
        ),
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
    assert (
        _single_report(completed)["error"]["code"]
        == "DATASET_RECEIPT_INVALID"
    )


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
    receipt = json.loads((out / "COLLECTION.json").read_text())
    receipt["schema_version"] = True
    _write_json(out / "COLLECTION.json", receipt)

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


def test_slurm_export_is_an_explicit_supported_allowlist_without_all_or_none(
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

    assert completed.returncode == 0
    command = _single_report(completed)["result"]["commands"][0]
    export = next(item for item in command if item.startswith("--export="))
    exported_names = {
        item.split("=", 1)[0]
        for item in export.removeprefix("--export=").split(",")
    }
    assert "ALL" not in exported_names
    assert "NONE" not in exported_names
    assert {
        "MS_SHARED_ROOT",
        "MS_ENV_ROOT",
        "MS_DATA_ROOT",
        "MS_OUT_ROOT",
        "MS_PROVIDER",
    } <= exported_names


def test_load_release_rejects_missing_archive_before_returning(tmp_path):
    from msctl.contracts import load_release
    from msctl.errors import MsctlError

    release = _release(tmp_path)
    release_value = json.loads(release.read_text())
    (tmp_path / release_value["archive"]["path"]).unlink()

    with pytest.raises(MsctlError) as caught:
        load_release(release)

    assert caught.value.code == "RELEASE_ARCHIVE_INVALID"


@pytest.mark.parametrize(
    "mutation",
    ["archive_symlink", "archive_bytes", "external_symlink", "missing_member"],
)
def test_load_release_authenticates_all_external_and_internal_bytes(
    tmp_path,
    mutation,
):
    from msctl.contracts import load_release
    from msctl.errors import MsctlError

    release = _release(tmp_path)
    release_value = json.loads(release.read_text())
    archive = tmp_path / release_value["archive"]["path"]
    checksum = archive.with_name(archive.name + ".sha256")
    if mutation == "archive_symlink":
        outside = tmp_path / "outside.zip"
        outside.write_bytes(archive.read_bytes())
        archive.unlink()
        archive.symlink_to(outside)
    elif mutation == "archive_bytes":
        archive.write_bytes(archive.read_bytes() + b"tamper")
    elif mutation == "external_symlink":
        outside = tmp_path / "outside.sha256"
        outside.write_bytes(checksum.read_bytes())
        checksum.unlink()
        checksum.symlink_to(outside)
    else:
        rewritten = tmp_path / "rewritten.zip"
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(
            rewritten,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as destination:
            for info in source.infolist():
                if info.filename != "DATASET-POINTER.json":
                    destination.writestr(info, source.read(info.filename))
        os.replace(rewritten, archive)
        archive_hash = _sha256(archive)
        release_value["archive"]["bytes"] = archive.stat().st_size
        release_value["archive"]["sha256"] = archive_hash
        _write_json(release, release_value)
        checksum.write_text(f"{archive_hash}  {archive.name}\n")

    with pytest.raises(MsctlError) as caught:
        load_release(release)

    assert caught.value.code in {
        "RELEASE_ARCHIVE_INVALID",
        "RELEASE_INTERNAL_INVALID",
    }


@pytest.mark.parametrize(
    ("runtime_member", "expected_code"),
    [
        ("profile", "PROFILE_RELEASE_MISMATCH"),
        ("slurm", "RELEASE_MEMBER_MISMATCH"),
        ("config", "RELEASE_MEMBER_MISMATCH"),
    ],
)
def test_run_operations_execute_only_local_bytes_bound_to_release(
    tmp_path,
    runtime_member,
    expected_code,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    profile = PROFILE
    if runtime_member == "profile":
        profile = tmp_path / "alternate-profile.json"
        profile.write_text(
            json.dumps(
                json.loads(PROFILE.read_text()),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif runtime_member == "slurm":
        with (tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch").open("a") as file:
            file.write("# changed after release\n")
    else:
        config = tmp_path / "configs" / "v2" / "dense-s0.json"
        with config.open("a") as file:
            file.write(" ")
        manifest_value = json.loads(manifest.read_text())
        dense_run = next(
            row for row in manifest_value["runs"] if row["arm"] == "dense"
        )
        dense_run["config_sha256"] = _sha256(config)
        _write_json(manifest, manifest_value)

    completed = _run_msctl(
        "--profile",
        str(profile),
        "--repo-root",
        str(tmp_path),
        "runs",
        "render",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--dataset-pointer",
        str(tmp_path / "DATASET-POINTER.json"),
        "--dataset-root",
        str(_dataset_fixture(tmp_path)[1]),
        bind_dataset=False,
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == expected_code


def test_run_submission_requires_explicit_dataset_verification_inputs(tmp_path):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        bind_dataset=False,
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == "CLI_USAGE"


def test_submit_accepts_exact_prior_dataset_verification_and_rejects_staleness(
    tmp_path,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    pointer = tmp_path / "DATASET-POINTER.json"
    root = _dataset_fixture(tmp_path)[1]
    verified = _run_msctl(
        *_base_args(tmp_path),
        "dataset",
        "verify",
        "--pointer",
        str(pointer),
        "--dataset-root",
        str(root),
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        bind_dataset=False,
    )
    assert verified.returncode == 0
    verification = _write_json(
        tmp_path / "dataset-verification.json",
        _single_report(verified)["result"],
    )
    args = [
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--dataset-pointer",
        str(pointer),
        "--dataset-verification",
        str(verification),
    ]

    accepted = _run_msctl(*args, bind_dataset=False)
    assert accepted.returncode == 0
    assert (
        _single_report(accepted)["result"]["dataset_verification"][
            "verification_sha256"
        ]
        == json.loads(verification.read_text())["verification_sha256"]
    )

    shard = next((root / "shards").glob("*.bin"))
    shard.write_bytes(shard.read_bytes() + b"stale")
    rejected = _run_msctl(*args, bind_dataset=False)
    assert rejected.returncode != 0
    assert (
        _single_report(rejected)["error"]["code"]
        == "DATASET_VERIFICATION_STALE"
    )


def test_dataset_verification_receipt_write_is_dry_run_first_and_strict(
    tmp_path,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    dataset_root = _dataset_fixture(tmp_path)[1]
    verification = tmp_path / "dataset-verification.json"
    args = [
        *_base_args(tmp_path),
        "dataset",
        "verify",
        "--pointer",
        str(tmp_path / "DATASET-POINTER.json"),
        "--dataset-root",
        str(dataset_root),
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--verification-out",
        str(verification),
    ]

    planned = _run_msctl(*args, bind_dataset=False)
    assert planned.returncode == 0
    assert _single_report(planned)["dry_run"] is True
    assert not verification.exists()

    written = _run_msctl(*args, "--apply", bind_dataset=False)
    assert written.returncode == 0
    assert _single_report(written)["dry_run"] is False
    value = json.loads(verification.read_text())
    assert value == _single_report(written)["result"]
    value["schema_version"] = True
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "verification_sha256"
    }
    value["verification_sha256"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    _write_json(verification, value)

    rejected = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--dataset-pointer",
        str(tmp_path / "DATASET-POINTER.json"),
        "--dataset-verification",
        str(verification),
        bind_dataset=False,
    )
    assert rejected.returncode != 0
    assert (
        _single_report(rejected)["error"]["code"]
        == "DATASET_VERIFICATION_INVALID"
    )


def test_legacy_dataset_receipt_scalar_claims_are_not_trusted(tmp_path):
    pointer, root = _dataset_fixture(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["ordered_stream_sha256"] = "5" * 64
    (root / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    )

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
    assert (
        _single_report(completed)["error"]["code"]
        == "DATASET_RECEIPT_INVALID"
    )


@pytest.mark.parametrize("binding", ["pointer", "source_lock"])
def test_dataset_verification_binds_release_pointer_and_source_lock(
    tmp_path,
    binding,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    pointer = tmp_path / "DATASET-POINTER.json"
    dataset_root = _dataset_fixture(tmp_path)[1]
    if binding == "pointer":
        value = json.loads(pointer.read_text())
        value["relative_path"] = value["relative_path"] + "-different"
        _write_json(pointer, value)
    else:
        (tmp_path / "configs" / "reasoning-dataset-v2.json").write_text(
            '{"schema_version":2,"fixture":"changed"}\n'
        )

    completed = _run_msctl(
        *_base_args(tmp_path),
        "dataset",
        "verify",
        "--pointer",
        str(pointer),
        "--dataset-root",
        str(dataset_root),
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        bind_dataset=False,
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] in {
        "DATASET_RELEASE_MISMATCH",
        "DATASET_SOURCE_LOCK_MISMATCH",
        "RELEASE_MEMBER_MISMATCH",
    }


def test_submit_approval_uses_rendered_252_gpu_hour_allocation(tmp_path):
    from msctl.approval import verify_approval
    from msctl.contracts import load_run_manifest
    from msctl.errors import MsctlError
    from msctl.profile import load_profile

    _, runs = _runs(tmp_path)
    manifest = load_run_manifest(tmp_path / "runs.json", repo_root=tmp_path)
    key = "r" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
        gpu_hours=250.0,
    )

    with pytest.raises(MsctlError) as caught:
        verify_approval(
            approval,
            operation="submit",
            release_sha256=str(runs["release_sha256"]),
            run_manifest=manifest,
            profile=load_profile(PROFILE),
            environ={"MSCTL_APPROVAL_KEY": key},
        )

    assert caught.value.code == "APPROVAL_LIMIT_EXCEEDED"
    assert caught.value.details["requested_gpu_hours"] == 252.0


@pytest.mark.parametrize(
    "loader_name",
    ["profile", "release", "run_manifest", "dataset_pointer"],
)
@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_schema_versions_reject_non_integer_values(
    tmp_path,
    loader_name,
    invalid_version,
):
    from msctl.contracts import load_release, load_run_manifest
    from msctl.dataset import load_pointer
    from msctl.errors import MsctlError
    from msctl.profile import load_profile

    if loader_name == "profile":
        value = json.loads(PROFILE.read_text())
        value["schema_version"] = invalid_version
        path = _write_json(tmp_path / "profile.json", value)

        def invoke():
            return load_profile(path)

    elif loader_name == "release":
        path = _release(tmp_path)
        value = json.loads(path.read_text())
        value["schema_version"] = invalid_version
        _write_json(path, value)

        def invoke():
            return load_release(path)

    elif loader_name == "run_manifest":
        path, _ = _runs(tmp_path)
        value = json.loads(path.read_text())
        value["schema_version"] = invalid_version
        _write_json(path, value)

        def invoke():
            return load_run_manifest(path, repo_root=tmp_path)

    else:
        path, _ = _dataset_fixture(tmp_path)
        value = json.loads(path.read_text())
        value["schema_version"] = invalid_version
        _write_json(path, value)

        def invoke():
            return load_pointer(path, load_profile(PROFILE))

    with pytest.raises(MsctlError) as caught:
        invoke()

    assert caught.value.code in {
        "PROFILE_INVALID",
        "RELEASE_INVALID",
        "RUN_MANIFEST_INVALID",
        "DATASET_POINTER_INVALID",
        "SCHEMA_INVALID",
    }


def test_approval_and_checkpoint_schema_versions_reject_json_booleans(tmp_path):
    from msctl.approval import verify_approval
    from msctl.contracts import (
        load_release,
        load_run_manifest,
        verify_checkpoint_receipt,
    )
    from msctl.errors import MsctlError
    from msctl.profile import load_profile

    release_path = _release(tmp_path)
    manifest_path, runs = _runs(tmp_path)
    release = load_release(release_path)
    manifest = load_run_manifest(manifest_path, repo_root=tmp_path)
    key = "v" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    approval_value = json.loads(approval.read_text())
    approval_value["schema_version"] = True
    _write_json(approval, approval_value)
    with pytest.raises(MsctlError) as approval_error:
        verify_approval(
            approval,
            operation="submit",
            release_sha256=release.archive_sha256,
            run_manifest=manifest,
            profile=load_profile(PROFILE),
            environ={"MSCTL_APPROVAL_KEY": key},
        )
    assert approval_error.value.code == "APPROVAL_INVALID"

    checkpoint = _checkpoint_receipt(tmp_path, runs=runs)
    checkpoint_value = json.loads(checkpoint.read_text())
    checkpoint_value["schema_version"] = True
    _write_json(checkpoint, checkpoint_value)
    with pytest.raises(MsctlError) as checkpoint_error:
        verify_checkpoint_receipt(
            checkpoint,
            release=release,
            manifest=manifest,
        )
    assert checkpoint_error.value.code == "CHECKPOINT_PROVENANCE_MISMATCH"


def test_environment_and_cleanup_schema_versions_reject_json_booleans(tmp_path):
    import msctl.cleanup as cleanup_module
    import msctl.environment as environment_module
    from msctl.errors import MsctlError
    from msctl.profile import load_profile

    environment_receipt = _write_json(
        tmp_path / "ENVIRONMENT.json",
        {
            "schema_version": True,
            "provider": "illumina-usfc-prd",
            "profile_sha256": "1" * 64,
            "lock_sha256": "2" * 64,
            "python": "3.11.9",
            "platform": "linux_x86_64",
            "cuda_version": "12.4",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    with pytest.raises(MsctlError) as environment_error:
        environment_module._read_receipt(environment_receipt)
    assert environment_error.value.code == "ENV_RECEIPT_INVALID"

    root = tmp_path / "runs"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "worker.log").write_text("log\n")
    plan = cleanup_module.make_cleanup_plan(root)
    plan["schema_version"] = True
    plan_path = _write_json(tmp_path / "cleanup-plan-bool.json", plan)
    with pytest.raises(MsctlError) as cleanup_error:
        cleanup_module.apply_cleanup(
            profile=load_profile(PROFILE),
            plan_path=plan_path,
            release_path=tmp_path / "unused-release.json",
            approval_path=None,
            apply=True,
            environ={},
        )
    assert cleanup_error.value.code == "CLEANUP_PLAN_INVALID"


def test_state_store_rejects_symlinked_child_directories(tmp_path):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    state_root = tmp_path / "state"
    attacker = tmp_path / "attacker"
    state_root.mkdir()
    attacker.mkdir()
    os.symlink(attacker, state_root / "runs")
    store = StateStore(state_root)

    with pytest.raises(MsctlError) as caught:
        with store.locked():
            store.write_run(
                "v2-dense-s0",
                {
                    "schema_version": 1,
                    "run_id": "v2-dense-s0",
                },
            )

    assert caught.value.code == "UNSAFE_STATE"
    assert not (attacker / "v2-dense-s0.json").exists()


def test_collection_copies_bytes_from_the_descriptor_it_hashed(
    tmp_path,
    monkeypatch,
):
    import msctl.collect as collect_module

    source = tmp_path / "source"
    evidence = source / "v2-dense-s0" / "log.jsonl"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"original\n")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"attacker\n")
    original_inode = evidence.stat().st_ino
    original_read = collect_module.read_fd
    raced = False

    def racing_read(descriptor):
        nonlocal raced
        data = original_read(descriptor)
        if os.fstat(descriptor).st_ino == original_inode and not raced:
            raced = True
            displaced = tmp_path / "displaced"
            os.replace(evidence, displaced)
            os.replace(replacement, evidence)
        return data

    monkeypatch.setattr(collect_module, "read_fd", racing_read)

    result = collect_module.collect_evidence(
        source=source,
        out=tmp_path / "collected",
        apply=True,
    )

    assert result["collected_files"] == 1
    assert raced is True
    assert (
        tmp_path / "collected" / "v2-dense-s0" / "log.jsonl"
    ).read_bytes() == b"original\n"


def test_collection_publication_never_replaces_a_racing_destination(
    tmp_path,
    monkeypatch,
):
    import msctl.collect as collect_module
    from msctl.errors import MsctlError

    source = tmp_path / "source"
    evidence = source / "v2-dense-s0" / "log.jsonl"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"step":1}\n')
    destination = tmp_path / "collected"
    original_rename = collect_module.rename_noreplace_at

    def racing_rename(source_fd, source_name, destination_fd, destination_name):
        os.mkdir(destination_name, dir_fd=destination_fd)
        attacker_fd = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=destination_fd,
        )
        try:
            marker_fd = os.open(
                "attacker-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=attacker_fd,
            )
            os.close(marker_fd)
        finally:
            os.close(attacker_fd)
        original_rename(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )

    monkeypatch.setattr(
        collect_module,
        "rename_noreplace_at",
        racing_rename,
    )

    with pytest.raises(MsctlError) as caught:
        collect_module.collect_evidence(
            source=source,
            out=destination,
            apply=True,
        )

    assert caught.value.code == "COLLECT_DESTINATION_EXISTS"
    assert (destination / "attacker-marker").is_file()
    assert not (destination / "v2-dense-s0" / "log.jsonl").exists()


def test_production_environment_apply_fails_closed_until_site_contract_is_pinned(
    tmp_path,
):
    lock = tmp_path / "requirements-illumina.lock"
    lock.write_text("# deliberately empty test lock\n")

    completed = _run_msctl(
        *_production_base_args(tmp_path),
        "env",
        "ensure",
        "--root",
        str(tmp_path / "environment"),
        "--lock",
        str(lock),
        "--apply",
    )

    assert completed.returncode != 0
    assert (
        _single_report(completed)["error"]["code"]
        == "ENV_CONTRACT_INCOMPLETE"
    )


def test_environment_rejects_a_lock_without_exact_platform_headers_and_hashes(
    tmp_path,
):
    profile_value = json.loads(PROFILE.read_text())
    profile_value["environment"] = {
        "status": "pinned",
        "lock_path": "requirements-illumina.lock",
        "contract": {
            "python_implementation": "CPython",
            "python_version": "3.11.9",
            "platform": "linux_x86_64",
            "cuda_version": "12.4",
        },
    }
    profile = _write_json(tmp_path / "profile.json", profile_value)
    lock = tmp_path / "requirements-illumina.lock"
    lock.write_text("example==1.0 --hash=sha256:" + "0" * 64 + "\n")

    completed = _run_msctl(
        "--profile",
        str(profile),
        "--repo-root",
        str(tmp_path),
        "env",
        "ensure",
        "--root",
        str(tmp_path / "environment"),
        "--lock",
        str(lock),
        "--apply",
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == "ENV_LOCK_INVALID"


def test_resume_exports_verified_checkpoint_paths_and_never_uses_auto(tmp_path):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    receipt = _checkpoint_receipt(tmp_path, runs=runs)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "resume",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--checkpoint-receipt",
        str(receipt),
    )

    assert completed.returncode == 0
    command = _single_report(completed)["result"]["command"]
    export = next(item for item in command if item.startswith("--export="))
    assert "MS_DENSE_RESUME_PATH=" in export
    assert "MS_SPLIT_RESUME_PATH=" in export
    script = (REPO_ROOT / "cluster" / "slurm" / "v2_seed0.sbatch").read_text()
    assert "--resume auto" not in script
    assert "--resume-path" in script
    assert "memorysplit-ddp-v1" in script


def test_evaluation_script_uses_one_preflighted_runner_contract():
    script = (REPO_ROOT / "cluster" / "slurm" / "v2_evaluate.sbatch").read_text()

    assert "python -m evals.confirmatory" not in script
    assert "evals/confirmatory/runner.py" in script
    assert "memorysplit-confirmatory-evaluator-v1" in script


def test_rendered_submission_has_manifest_bound_job_name_and_comment(tmp_path):
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
    job_name = next(item for item in command if item.startswith("--job-name="))
    comment = next(item for item in command if item.startswith("--comment="))
    assert job_name != "--job-name=ms-v2-seed0"
    assert comment.startswith("--comment=msctl:")
    assert len(comment.removeprefix("--comment=msctl:")) == 64


def test_render_binds_verified_runtime_roots_independent_of_cwd_and_ambient_data(
    tmp_path,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    alternate_cwd = tmp_path / "untrusted-cwd"
    alternate_cwd.mkdir()

    completed = _run_msctl(
        *_base_args(tmp_path),
        "runs",
        "render",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        cwd=alternate_cwd,
        env={
            "MS_SHARED_ROOT": "/attacker/shared",
            "MS_DATA_ROOT": "/attacker/data",
            "MS_ENV_ROOT": "/attacker/environment",
        },
    )

    assert completed.returncode == 0
    command = _single_report(completed)["result"]["commands"][0]
    export = next(item for item in command if item.startswith("--export="))
    assert f"MS_DATA_ROOT={_dataset_fixture(tmp_path)[1].resolve()}" in export
    assert f"MS_RELEASE_ROOT={tmp_path.resolve()}" in export
    assert f"MS_ENV_ROOT={(tmp_path / 'environment').resolve()}" in export
    assert "/attacker" not in export
    assert f"--chdir={tmp_path.resolve()}" in command
    assert command[-1] == str(
        (tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch").resolve()
    )


def test_submit_requires_environment_receipt_before_any_sbatch(tmp_path):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "e" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    marker = tmp_path / "sbatch-called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "sbatch",
        f"touch '{marker}'\nprintf '777;usfc-prd\\n'\n",
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
        bind_environment=False,
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == "ENV_RECEIPT_REQUIRED"
    assert not marker.exists()


@pytest.mark.parametrize(
    ("operation", "member"),
    [
        ("submit", "scripts/run_train.py"),
        ("evaluate", "evals/confirmatory/runner.py"),
    ],
)
def test_paid_submission_rejects_replaced_entrypoint_before_sbatch(
    tmp_path,
    operation,
    member,
):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "x" * 32
    approval = _approval(
        tmp_path,
        operation=operation,
        runs=runs,
        key=key,
    )
    (tmp_path / member).write_text(
        'MSCTL_DDP_CONTRACT = "memorysplit-ddp-v1"\n'
        'MSCTL_EVALUATOR_CONTRACT = "memorysplit-confirmatory-evaluator-v1"\n'
    )
    marker = tmp_path / "sbatch-called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "sbatch",
        f"touch '{marker}'\nprintf '777;usfc-prd\\n'\n",
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        operation,
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
    assert _single_report(completed)["error"]["code"] == (
        "RELEASE_MEMBER_MISMATCH"
    )
    assert not marker.exists()


@pytest.mark.parametrize("operation", ["submit", "evaluate"])
def test_authenticated_but_incomplete_runtime_contract_fails_before_sbatch(
    tmp_path,
    operation,
):
    incomplete = b'FLAGS = ("--config", "--run")\n'
    release = _release(
        tmp_path,
        train_entrypoint_bytes=(
            incomplete if operation == "submit" else TRAIN_ENTRYPOINT
        ),
        evaluator_entrypoint_bytes=(
            incomplete if operation == "evaluate" else EVALUATOR_ENTRYPOINT
        ),
    )
    manifest, runs = _runs(tmp_path)
    key = "p" * 32
    approval = _approval(
        tmp_path,
        operation=operation,
        runs=runs,
        key=key,
    )
    marker = tmp_path / "sbatch-called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "sbatch",
        f"touch '{marker}'\nprintf '777;usfc-prd\\n'\n",
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        operation,
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
    assert _single_report(completed)["error"]["code"] == (
        "RUNTIME_PREFLIGHT_FAILED"
    )
    assert not marker.exists()


def test_partial_prepared_pair_intent_repairs_first_state_and_submits_once(
    tmp_path,
):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "j" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "sbatch", "printf 'invalid\\n'\n")
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
    env = {"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key}

    interrupted = _run_msctl(*args, env=env)
    assert interrupted.returncode != 0
    intents = list((tmp_path / "state" / "intents").glob("*.json"))
    assert len(intents) == 1
    intent = json.loads(intents[0].read_text())
    intent["phase"] = "PREPARED"
    intent["job_id"] = None
    _write_json(intents[0], intent)
    (tmp_path / "state" / "runs" / "v2-split90-s0.json").unlink()
    count = tmp_path / "submit-count"
    _write_executable(
        bin_dir / "sbatch",
        f"printf '1' > '{count}'\nprintf '777;usfc-prd\\n'\n",
    )

    recovered = _run_msctl(*args, env=env)

    assert recovered.returncode == 0
    assert count.read_text() == "1"
    assert _single_report(recovered)["result"]["job_id"] == "777"
    for run_id in ("v2-dense-s0", "v2-split90-s0"):
        state = json.loads(
            (tmp_path / "state" / "runs" / f"{run_id}.json").read_text()
        )
        assert state["job_id"] == "777"


@pytest.mark.parametrize("mutation", ["unknown", "missing", "wrong_type"])
def test_run_state_reader_rejects_nonexact_or_wrongly_typed_schema(
    tmp_path,
    mutation,
):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    run_id = "v2-dense-s0"
    value = {
        "schema_version": 1,
        "run_id": run_id,
        "arm": "dense",
        "seed": 0,
        "provider": "illumina-usfc-prd",
        "release_sha256": "1" * 64,
        "run_manifest_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "dataset_sha256": "4" * 64,
        "dataset_verification_sha256": "5" * 64,
        "environment_receipt_sha256": "6" * 64,
        "operation": "submit",
        "submission_key": "7" * 64,
        "resource_request": {
            "schema_version": 1,
            "operation": "submit",
            "jobs": 1,
            "allocated_gpus": 7,
            "wall_minutes": 2160,
            "gpu_hours": 252.0,
            "gres": "gpu:a100:7",
            "script": "cluster/slurm/v2_seed0.sbatch",
        },
        "job_id": None,
        "status": "SUBMITTING",
        "attempt": 1,
        "created_at": "2026-07-23T00:00:00Z",
        "updated_at": "2026-07-23T00:00:00Z",
    }
    if mutation == "unknown":
        value["unexpected"] = True
    elif mutation == "missing":
        del value["arm"]
    else:
        value["attempt"] = True
    store = StateStore(tmp_path / "state")
    with store.locked():
        pass
    _write_json(tmp_path / "state" / "runs" / f"{run_id}.json", value)

    with store.locked(), pytest.raises(MsctlError) as caught:
        store.read_run(run_id)

    assert caught.value.code == "STATE_CORRUPT"


@pytest.mark.parametrize("reader", ["evaluation", "intent"])
@pytest.mark.parametrize("mutation", ["unknown", "missing", "wrong_type"])
def test_other_lifecycle_readers_are_exact_and_strict(
    tmp_path,
    reader,
    mutation,
):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    resource = {
        "schema_version": 1,
        "operation": "evaluate" if reader == "evaluation" else "submit",
        "jobs": 1,
        "allocated_gpus": 1 if reader == "evaluation" else 7,
        "wall_minutes": 360 if reader == "evaluation" else 2160,
        "gpu_hours": 6.0 if reader == "evaluation" else 252.0,
        "gres": "gpu:a100:1" if reader == "evaluation" else "gpu:a100:7",
        "script": (
            "cluster/slurm/v2_evaluate.sbatch"
            if reader == "evaluation"
            else "cluster/slurm/v2_seed0.sbatch"
        ),
    }
    if reader == "evaluation":
        key = "2" * 64
        value = {
            "schema_version": 1,
            "provider": "illumina-usfc-prd",
            "release_sha256": "1" * 64,
            "run_manifest_sha256": key,
            "dataset_sha256": "3" * 64,
            "dataset_verification_sha256": "4" * 64,
            "environment_receipt_sha256": "5" * 64,
            "operation": "evaluate",
            "submission_key": "6" * 64,
            "resource_request": resource,
            "job_id": None,
            "status": "SUBMITTING",
            "created_at": "2026-07-23T00:00:00Z",
            "updated_at": "2026-07-23T00:00:00Z",
        }
        path = tmp_path / "state" / "evaluations" / f"{key}.json"
    else:
        key = "6" * 64
        value = {
            "schema_version": 1,
            "submission_key": key,
            "provider": "illumina-usfc-prd",
            "release_sha256": "1" * 64,
            "run_manifest_sha256": "2" * 64,
            "dataset_sha256": "3" * 64,
            "dataset_verification_sha256": "4" * 64,
            "environment_receipt_sha256": "5" * 64,
            "operation": "submit",
            "resource_request": resource,
            "run_ids": ["v2-dense-s0", "v2-split90-s0"],
            "attempt": 1,
            "checkpoint_receipt_sha256": None,
            "phase": "PREPARED",
            "job_id": None,
            "created_at": "2026-07-23T00:00:00Z",
            "updated_at": "2026-07-23T00:00:00Z",
        }
        path = tmp_path / "state" / "intents" / f"{key}.json"
    if mutation == "unknown":
        value["unexpected"] = True
    elif mutation == "missing":
        del value["provider"]
    else:
        value["schema_version"] = True
    store = StateStore(tmp_path / "state")
    with store.locked():
        pass
    _write_json(path, value)

    with store.locked(), pytest.raises(MsctlError) as caught:
        if reader == "evaluation":
            store.read_evaluation(key)
        else:
            store.read_intent(key)

    assert caught.value.code == "STATE_CORRUPT"


def test_dataset_verification_caches_device_identity_and_rejects_device_drift(
    tmp_path,
):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)
    verified = _run_msctl(
        *_base_args(tmp_path),
        "dataset",
        "verify",
        "--pointer",
        str(tmp_path / "DATASET-POINTER.json"),
        "--dataset-root",
        str(_dataset_fixture(tmp_path)[1]),
        "--release",
        str(release),
        "--manifest",
        str(manifest),
    )
    verification = _single_report(verified)["result"]
    assert all("device" in row for row in verification["file_identities"])
    for row in verification["file_identities"]:
        row["device"] += 1
    unsigned = {
        key: value
        for key, value in verification.items()
        if key != "verification_sha256"
    }
    verification["verification_sha256"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    verification_path = _write_json(
        tmp_path / "dataset-verification-device-drift.json",
        verification,
    )

    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--dataset-pointer",
        str(tmp_path / "DATASET-POINTER.json"),
        "--dataset-verification",
        str(verification_path),
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == (
        "DATASET_VERIFICATION_STALE"
    )
