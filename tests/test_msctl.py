from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "cluster" / "profiles" / "illumina-usfc-prd.json"
HEX64 = "a" * 64
_AWS_TERMINATE_AT = (
    datetime.now(UTC) + timedelta(hours=4)
).replace(microsecond=0).isoformat().replace("+00:00", "Z")
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


def _test_shared_root(tmp_path: Path) -> Path:
    return (tmp_path / "publication").resolve()


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
    from msctl.environment import _runtime_receipt_fields
    from msctl.profile import load_profile

    release_path = _release(tmp_path)
    profile = load_profile(_test_profile(tmp_path))
    root = tmp_path / "environment"
    receipt_path = root / "msctl-env-receipt.json"
    if receipt_path.is_file():
        return receipt_path
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.executable, python)
    runtime = _runtime_receipt_fields(root, profile=profile)
    release = json.loads(release_path.read_text())
    return _write_json(
        receipt_path,
        {
            "schema_version": 2,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
            "release_sha256": release["archive"]["sha256"],
            "lock_sha256": _sha256(tmp_path / "requirements-illumina.lock"),
            "environment_root": str(root.resolve()),
            "python": profile.python_version,
            "platform": profile.platform,
            "cuda_version": profile.cuda_version,
            "cuda_driver_version": "550.54",
            **runtime,
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
    test_prefix: str | None = None
    if "--repo-root" in rendered:
        root = Path(rendered[rendered.index("--repo-root") + 1])
        test_prefix = str(root.resolve())
        requires_dataset = (
            any(
                command in rendered
                for command in ("submit", "resume", "evaluate")
            )
            or ("runs" in rendered and "render" in rendered)
            or ("dataset" in rendered and "verify" in rendered)
        )
        if requires_dataset and "--shared-root" not in rendered:
            rendered.extend(["--shared-root", str(_test_shared_root(root))])
        if (
            "env" in rendered
            and "ensure" in rendered
            and "--release" not in rendered
            and Path(rendered[rendered.index("--profile") + 1]).resolve()
            == (
                root / "cluster" / "profiles" / PROFILE.name
            ).resolve()
        ):
            rendered.extend(["--release", str(_release(root))])
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
    if test_prefix is not None:
        merged["MSCTL_TEST_SHARED_ROOT_PREFIX"] = test_prefix
    if env:
        merged.update(env)
    if (
        "--apply" in rendered
        and (
            any(
                command in rendered
                for command in ("submit", "resume", "evaluate")
            )
            or ("env" in rendered and "ensure" in rendered)
        )
        and "--repo-root" in rendered
    ):
        root = Path(rendered[rendered.index("--repo-root") + 1])
        probes = root / ".test-runtime-probes"
        probes.mkdir(exist_ok=True)
        _write_executable(
            probes / "nvcc",
            "printf 'Cuda compilation tools, release 12.4, V12.4.0\\n'\n",
        )
        _write_executable(
            probes / "nvidia-smi",
            "printf '550.54\\n'\n",
        )
        merged["PATH"] = str(probes) + os.pathsep + merged.get("PATH", "")
    runner = (
        "import dataclasses,os,sys;"
        "import msctl.cli as c;"
        "_load=c.load_profile;"
        "_profile=lambda path:_load(path);"
        "c.load_profile=lambda path:("
        "dataclasses.replace(_profile(path),shared_root_prefix="
        "os.environ.get('MSCTL_TEST_SHARED_ROOT_PREFIX') or "
        "_profile(path).shared_root_prefix)"
        " if hasattr(_profile(path),'shared_root_prefix') else _profile(path)"
        ");"
        "raise SystemExit(c.main(sys.argv[1:]))"
    )
    return subprocess.run(
        [sys.executable, "-c", runner, *rendered],
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
    assert report["result"]["bootstrap_sha256"]
    assert "--chdir=/" in command
    assert str(
        (tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch").resolve()
    ) not in command


def test_relative_release_exports_authenticated_absolute_archive(tmp_path):
    release = _release(tmp_path)
    manifest, _ = _runs(tmp_path)

    completed = _run_msctl(
        *_base_args(tmp_path),
        "runs",
        "render",
        "--release",
        release.name,
        "--manifest",
        str(manifest),
        cwd=tmp_path,
    )

    assert completed.returncode == 0
    command = _single_report(completed)["result"]["commands"][0]
    export = next(item for item in command if item.startswith("--export="))
    release_value = json.loads(release.read_text())
    archive = (release.parent / release_value["archive"]["path"]).resolve()
    assert f"MS_RELEASE_ARCHIVE={archive}" in export
    assert (
        f"MS_RELEASE_ARCHIVE={release_value['archive']['path']}" not in export
    )
    assert "--chdir=/" in command


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
    assert not list((tmp_path / "state" / "runs").glob("*.json"))


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
            "shared_root_prefix": str(tmp_path.resolve()),
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
    manifest, manifest_value = _runs(tmp_path)
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
        scope_sha256=hashlib.sha256(
            _canonical(
                {
                    "plan_sha256": hashlib.sha256(
                        _canonical(plan)
                    ).hexdigest(),
                    "run_manifest_sha256": hashlib.sha256(
                        _canonical(manifest_value)
                    ).hexdigest(),
                }
            )
        ).hexdigest(),
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
        "--manifest",
        str(manifest),
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
        "--manifest",
        str(manifest),
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
    manifest, manifest_value = _runs(tmp_path)
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
        scope_sha256=hashlib.sha256(
            _canonical(
                {
                    "plan_sha256": hashlib.sha256(
                        _canonical(plan)
                    ).hexdigest(),
                    "run_manifest_sha256": hashlib.sha256(
                        _canonical(manifest_value)
                    ).hexdigest(),
                }
            )
        ).hexdigest(),
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
            manifest_path=manifest,
            repo_root=tmp_path,
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
            manifest_path=tmp_path / "unused-manifest.json",
            repo_root=tmp_path,
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


def _aws_profile_value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "aws-p5.48xlarge",
        "provider": "aws-p5.48xlarge",
        "instance_type": "p5.48xlarge",
        "purchase_model": "on_demand",
        "cpu": {"vcpus": 192, "memory_gib": 2048},
        "gpu": {
            "model": "NVIDIA H100 80GB",
            "allocated": 8,
            "seed_train_groups": [4, 4],
        },
        "storage": {
            "scratch_root": "/mnt/memorysplit",
            "durable_uri_env": "MS_S3_ROOT",
            "instance_store": {
                "model": "Amazon EC2 NVMe Instance Storage",
                "devices": 8,
                "device_bytes": 3_840_000_000_000,
                "raid_level": "0",
            },
        },
        "runtime": {
            "region_env": "AWS_REGION",
            "ami_id_env": "MS_AWS_AMI_ID",
            "container_digest_env": "MS_CONTAINER_DIGEST",
            "runtime_uid_env": "MS_RUNTIME_UID",
            "runtime_gid_env": "MS_RUNTIME_GID",
        },
        "assigned_seeds": [1, 2, 3, 4],
        "process_env_allowlist": [
            "AWS_REGION",
            "LANG",
            "LC_ALL",
        ],
    }


def test_aws_profile_adapter_rejects_incomplete_runtime_with_stable_json_error(
    tmp_path,
):
    profile = _write_json(tmp_path / "aws-profile.json", _aws_profile_value())

    completed = _run_msctl(
        "--profile",
        str(profile),
        "auth",
        "check",
        env={
            "AWS_REGION": "us-east-1",
            "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
            "MS_CONTAINER_IMAGE": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
                "@sha256:" + "a" * 64
            ),
        },
    )

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == "AWS_RUNTIME_INVALID"


def test_cli_dispatch_routes_aws_provider_to_aws_backend(tmp_path):
    from msctl.cli import build_parser, dispatch

    profile = _aws_profile_object()
    captured = {}

    class Backend:
        def dispatch(self, command, args):
            captured["command"] = command
            captured["args"] = args
            return False, {"provider": profile.provider, "routed": command}

    def factory(**kwargs):
        captured["factory"] = kwargs
        return Backend()

    args = build_parser().parse_args(
        ["--profile", str(tmp_path / "aws.json"), "auth", "check"]
    )
    dry_run, result = dispatch(
        args,
        profile_loader=lambda _: profile,
        aws_backend_factory=factory,
        environ={"AWS_REGION": "us-east-1"},
    )

    assert dry_run is False
    assert result == {
        "provider": "aws-p5.48xlarge",
        "routed": "auth check",
    }
    assert captured["command"] == "auth check"
    assert captured["factory"]["profile"] is profile
    assert captured["factory"]["state_root"] == ".msctl-state"


def test_aws_cli_parser_requires_aws_lifecycle_evidence(tmp_path):
    from msctl.cli import build_parser, dispatch

    profile = _aws_profile_object()
    captured = {}

    class Backend:
        def dispatch(self, command, args):
            captured["command"] = command
            return True, {"provider": profile.provider, "submitted": 0}

    args = build_parser().parse_args(
        [
            "--profile",
            str(tmp_path / "aws.json"),
            "submit",
            "--release",
            str(tmp_path / "RELEASE.json"),
            "--manifest",
            str(tmp_path / "runs-s1.json"),
            "--instance-id",
            "i-0123456789abcdef0",
            "--terminate-at",
            _AWS_TERMINATE_AT,
            "--dataset-pointer",
            str(tmp_path / "DATASET-POINTER.json"),
            "--dataset-verification",
            str(tmp_path / "dataset-verification.json"),
            "--environment-receipt",
            str(tmp_path / "AWS-ENVIRONMENT.json"),
        ]
    )
    dry_run, result = dispatch(
        args,
        profile_loader=lambda _: profile,
        aws_backend_factory=lambda **_: Backend(),
        environ={},
    )

    assert dry_run is True
    assert result["submitted"] == 0
    assert captured["command"] == "submit"


def test_aws_cli_submit_requires_instance_selection_and_deadline(tmp_path):
    from msctl.cli import build_parser, dispatch

    args = build_parser().parse_args(
        [
            "--profile",
            str(tmp_path / "aws.json"),
            "submit",
            "--release",
            str(tmp_path / "RELEASE.json"),
            "--manifest",
            str(tmp_path / "runs-s1.json"),
        ]
    )

    with pytest.raises(Exception) as caught:
        dispatch(
            args,
            profile_loader=lambda _: _aws_profile_object(),
            aws_backend_factory=lambda **_: pytest.fail(
                "backend must not be built without exact selection"
            ),
            environ={},
        )

    assert getattr(caught.value, "code", None) == "CLI_USAGE"
    assert caught.value.details["missing"] == [
        "--instance-id",
        "--terminate-at",
    ]


def test_unknown_provider_is_rejected_before_provider_specific_parsing(tmp_path):
    profile_value = _aws_profile_value()
    profile_value["profile_id"] = "unknown-provider"
    profile_value["provider"] = "unknown-provider"
    profile = _write_json(tmp_path / "unknown-profile.json", profile_value)

    completed = _run_msctl("--profile", str(profile), "auth", "check")

    assert completed.returncode != 0
    report = _single_report(completed)
    assert report["error"]["code"] == "PROVIDER_UNSUPPORTED"


class _FakeCohort:
    def __init__(self, root: Path) -> None:
        assignment = root / "configs" / "cohort-assignment-v2.json"
        self.cohort_id = "memorysplit-confirmatory-v2-360m-n5"
        self.assignment_sha256 = _sha256(assignment)
        self.preregistration_sha256 = _sha256(
            root / "configs" / "preregistration-v2.yaml"
        )
        self.illumina_seeds = (0,)
        self.aws_p5_seeds = (1, 2, 3, 4)
        self.configs = tuple(
            SimpleNamespace(
                path=f"configs/360m-v2/{arm}-s{seed}.yaml",
                sha256=_sha256(
                    root / "configs" / "360m-v2" / f"{arm}-s{seed}.yaml"
                ),
                run_id=f"memorysplit-v2-360m-s{seed}-{arm}",
                condition=arm,
                seed=seed,
            )
            for seed in range(5)
            for arm in ("dense", "split90")
        )

    def configs_for_provider(self, provider: str):
        seeds = (
            set(self.illumina_seeds)
            if provider == "illumina-usfc-prd"
            else set(self.aws_p5_seeds)
        )
        return tuple(config for config in self.configs if config.seed in seeds)


def _cohort_release(tmp_path: Path, provider: str) -> Path:
    configs = tmp_path / "configs" / "360m-v2"
    configs.mkdir(parents=True)
    assignment = _write_json(
        tmp_path / "configs" / "cohort-assignment-v2.json",
        {
            "schema_version": 2,
            "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
        },
    )
    study_lock = tmp_path / "configs" / "preregistration-v2.yaml"
    study_lock.write_text("schema_version: 2\nstudy: frozen\n")
    for seed in range(5):
        for arm in ("dense", "split90"):
            (configs / f"{arm}-s{seed}.yaml").write_text(
                f"schema_version: 2\ncondition: {arm}\nseed: {seed}\n"
            )
    selected = (0,) if provider == "illumina-usfc-prd" else (1, 2, 3, 4)
    profile_name = (
        "illumina-usfc-prd.json"
        if provider == "illumina-usfc-prd"
        else "aws-p5.48xlarge.json"
    )
    profile_bytes = (
        PROFILE.read_bytes()
        if provider == "illumina-usfc-prd"
        else json.dumps(_aws_profile_value(), sort_keys=True).encode() + b"\n"
    )
    source_members = {
        f"cluster/profiles/{profile_name}": profile_bytes,
        "configs/cohort-assignment-v2.json": assignment.read_bytes(),
        "configs/preregistration-v2.yaml": study_lock.read_bytes(),
        **{
            f"configs/360m-v2/{arm}-s{seed}.yaml": (
                configs / f"{arm}-s{seed}.yaml"
            ).read_bytes()
            for seed in selected
            for arm in ("dense", "split90")
        },
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
                "provider": provider,
                "source": {"commit": "2" * 40, "dirty": False},
                "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                "environment_hashes": {},
                "seed_assignment": {
                    "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
                    "provider": provider,
                    "seeds": list(selected),
                    "arms": ["dense", "split90"],
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
    archive = tmp_path / f"{provider}-release.zip"
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
    archive_sha256 = _sha256(archive)
    (tmp_path / f"{archive.name}.sha256").write_text(
        f"{archive_sha256}  {archive.name}\n"
    )
    return _write_json(
        tmp_path / f"{provider}-RELEASE.json",
        {
            "schema_version": 1,
            "release_id": "r1-cohort-test",
            "provider": provider,
            "archive": {
                "path": archive.name,
                "sha256": archive_sha256,
                "bytes": archive.stat().st_size,
            },
            "source": {"commit": "2" * 40, "dirty": False},
            "members_sha256": hashlib.sha256(sums).hexdigest(),
        },
    )


def test_runs_instantiate_dry_run_is_canonical_and_apply_is_no_replace(tmp_path):
    from msctl.operations import instantiate_run_manifest

    release = _cohort_release(tmp_path, "illumina-usfc-prd")
    dataset_receipt = _write_json(
        tmp_path / "dataset-receipt.json",
        {"schema_version": 1, "dataset_id": "fixture"},
    )
    out = tmp_path / "runs-s0.json"
    cohort = _FakeCohort(tmp_path)
    arguments = {
        "profile": SimpleNamespace(
            provider="illumina-usfc-prd",
            source_sha256=_sha256(PROFILE),
        ),
        "release_path": release,
        "dataset_receipt": dataset_receipt,
        "seed": 0,
        "out": out,
        "repo_root": tmp_path,
        "cohort_loader": lambda _: cohort,
    }

    planned = instantiate_run_manifest(**arguments, apply=False)

    release_value = json.loads(release.read_text())
    manifest = planned["manifest"]
    assert manifest == {
        "schema_version": 2,
        "provider": "illumina-usfc-prd",
        "seed": 0,
        "release_sha256": release_value["archive"]["sha256"],
        "dataset_sha256": _sha256(dataset_receipt),
        "cohort_assignment_sha256": cohort.assignment_sha256,
        "study_lock_sha256": _sha256(
            tmp_path / "configs" / "preregistration-v2.yaml"
        ),
        "source_commit": "2" * 40,
        "runs": [
            {
                "run_id": "memorysplit-v2-360m-s0-dense",
                "arm": "dense",
                "seed": 0,
                "config": "configs/360m-v2/dense-s0.yaml",
                "config_sha256": _sha256(
                    tmp_path / "configs" / "360m-v2" / "dense-s0.yaml"
                ),
            },
            {
                "run_id": "memorysplit-v2-360m-s0-split90",
                "arm": "split90",
                "seed": 0,
                "config": "configs/360m-v2/split90-s0.yaml",
                "config_sha256": _sha256(
                    tmp_path / "configs" / "360m-v2" / "split90-s0.yaml"
                ),
            },
        ],
    }
    assert planned["manifest_sha256"] == hashlib.sha256(
        _canonical(manifest)
    ).hexdigest()
    assert planned["published"] is False
    assert not out.exists()

    with pytest.raises(Exception) as caught:
        instantiate_run_manifest(
            **{
                **arguments,
                "profile": SimpleNamespace(
                    provider="illumina-usfc-prd",
                    source_sha256="0" * 64,
                ),
            },
            apply=False,
        )
    assert getattr(caught.value, "code", None) == "PROFILE_RELEASE_MISMATCH"

    mismatched_cohort = _FakeCohort(tmp_path)
    mismatched_cohort.assignment_sha256 = "0" * 64
    with pytest.raises(Exception) as caught:
        instantiate_run_manifest(
            **{**arguments, "cohort_loader": lambda _: mismatched_cohort},
            apply=False,
        )
    assert getattr(caught.value, "code", None) == "RELEASE_COHORT_MISMATCH"

    applied = instantiate_run_manifest(**arguments, apply=True)

    assert applied["published"] is True
    assert out.read_bytes() == _canonical(manifest) + b"\n"
    from msctl.contracts import load_run_manifest

    loaded = load_run_manifest(out, repo_root=tmp_path)
    assert loaded.schema_version == 2
    assert loaded.seed == 0
    assert loaded.cohort_assignment_sha256 == cohort.assignment_sha256
    assert loaded.study_lock_sha256 == manifest["study_lock_sha256"]
    assert loaded.source_commit == "2" * 40
    with pytest.raises(Exception) as caught:
        instantiate_run_manifest(**arguments, apply=True)
    assert getattr(caught.value, "code", None) == "RUN_MANIFEST_EXISTS"

    manifest["unexpected"] = True
    _write_json(out, manifest)
    with pytest.raises(Exception):
        load_run_manifest(out, repo_root=tmp_path)

    manifest.pop("unexpected")
    manifest["source_commit"] = "not-a-commit"
    _write_json(out, manifest)
    with pytest.raises(Exception) as caught:
        load_run_manifest(out, repo_root=tmp_path)
    assert getattr(caught.value, "code", None) == "RUN_MANIFEST_INVALID"


def test_runs_instantiate_cli_dispatch_supports_injected_cohort_adapter(tmp_path):
    from msctl.cli import build_parser, dispatch

    release = _cohort_release(tmp_path, "illumina-usfc-prd")
    dataset_receipt = _write_json(
        tmp_path / "dataset-receipt.json",
        {"schema_version": 1, "dataset_id": "fixture"},
    )
    out = tmp_path / "runs-s0.json"
    args = build_parser().parse_args(
        [
            "--profile",
            str(PROFILE),
            "--repo-root",
            str(tmp_path),
            "runs",
            "instantiate",
            "--release",
            str(release),
            "--dataset-receipt",
            str(dataset_receipt),
            "--seed",
            "0",
            "--out",
            str(out),
        ]
    )

    dry_run, result = dispatch(
        args,
        profile_loader=lambda _: SimpleNamespace(
            provider="illumina-usfc-prd",
            source_sha256=_sha256(PROFILE),
        ),
        cohort_loader=lambda _: _FakeCohort(tmp_path),
    )

    assert dry_run is True
    assert result["manifest"]["seed"] == 0
    assert result["published"] is False
    assert not out.exists()


def test_runs_instantiate_accepts_each_aws_owned_seed_pair(tmp_path):
    from cluster.aws.p5.corpus_contract import verify_canonical_corpus
    from msctl.operations import instantiate_run_manifest
    from tests.test_aws_p5_launcher import _launcher_fixture

    release = _cohort_release(tmp_path, "aws-p5.48xlarge")
    fixture = _launcher_fixture(tmp_path / "task4")
    profile_bytes = (
        json.dumps(_aws_profile_value(), sort_keys=True).encode() + b"\n"
    )

    def verify(receipt_path, *, expected_sha256, expected_ordered_sha256):
        return verify_canonical_corpus(
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
            semantic_verifier=lambda _root: fixture["corpus"],
        )

    for seed in (1, 2, 3, 4):
        result = instantiate_run_manifest(
            profile=SimpleNamespace(
                provider="aws-p5.48xlarge",
                sha256=hashlib.sha256(profile_bytes).hexdigest(),
            ),
            release_path=release,
            dataset_receipt=fixture["corpus_path"],
            seed=seed,
            out=tmp_path / f"runs-s{seed}.json",
            repo_root=tmp_path,
            cohort_loader=lambda _: _FakeCohort(tmp_path),
            dataset_verifier=verify,
            apply=False,
        )

        assert result["manifest"]["seed"] == seed
        assert {run["seed"] for run in result["manifest"]["runs"]} == {seed}
        assert result["published"] is False


@pytest.mark.parametrize(
    ("provider", "seed"),
    [
        ("illumina-usfc-prd", 1),
        ("aws-p5.48xlarge", 0),
        ("aws-p5.48xlarge", 5),
    ],
)
def test_runs_instantiate_enforces_exact_provider_seed_ownership(
    tmp_path,
    provider,
    seed,
):
    from msctl.operations import instantiate_run_manifest

    release = _cohort_release(tmp_path, provider)
    dataset_receipt = _write_json(
        tmp_path / "dataset-receipt.json",
        {"schema_version": 1, "dataset_id": "fixture"},
    )
    cohort = _FakeCohort(tmp_path)

    with pytest.raises(Exception) as caught:
        instantiate_run_manifest(
            profile=SimpleNamespace(provider=provider),
            release_path=release,
            dataset_receipt=dataset_receipt,
            seed=seed,
            out=tmp_path / f"runs-s{seed}.json",
            repo_root=tmp_path,
            cohort_loader=lambda _: cohort,
            apply=False,
        )

    assert getattr(caught.value, "code", None) == "SEED_OWNERSHIP_VIOLATION"


def test_aws_runs_instantiate_rejects_arbitrary_dataset_json(tmp_path):
    from msctl.operations import instantiate_run_manifest

    release = _cohort_release(tmp_path, "aws-p5.48xlarge")
    arbitrary = _write_json(
        tmp_path / "dataset-receipt.json",
        {"schema_version": 2, "build_id": "b" * 64},
    )

    with pytest.raises(Exception) as caught:
        instantiate_run_manifest(
            profile=SimpleNamespace(
                provider="aws-p5.48xlarge",
                sha256=hashlib.sha256(
                    json.dumps(
                        _aws_profile_value(),
                        sort_keys=True,
                    ).encode()
                    + b"\n"
                ).hexdigest(),
            ),
            release_path=release,
            dataset_receipt=arbitrary,
            seed=1,
            out=tmp_path / "runs-s1.json",
            repo_root=tmp_path,
            cohort_loader=lambda _: _FakeCohort(tmp_path),
            apply=False,
        )
    assert getattr(caught.value, "code", None) == "DATASET_RECEIPT_INVALID"


def test_aws_runs_instantiate_verifies_real_task4_files_and_device_ids(
    tmp_path,
):
    from cluster.aws.p5.corpus_contract import verify_canonical_corpus
    from msctl.operations import instantiate_run_manifest
    from tests.test_aws_p5_launcher import _launcher_fixture

    release = _cohort_release(tmp_path, "aws-p5.48xlarge")
    fixture = _launcher_fixture(tmp_path / "task4")

    def verify(receipt_path, *, expected_sha256, expected_ordered_sha256):
        return verify_canonical_corpus(
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
            semantic_verifier=lambda _root: fixture["corpus"],
        )

    result = instantiate_run_manifest(
        profile=SimpleNamespace(
            provider="aws-p5.48xlarge",
            sha256=hashlib.sha256(
                json.dumps(_aws_profile_value(), sort_keys=True).encode() + b"\n"
            ).hexdigest(),
        ),
        release_path=release,
        dataset_receipt=fixture["corpus_path"],
        seed=1,
        out=tmp_path / "runs-s1.json",
        repo_root=tmp_path,
        cohort_loader=lambda _: _FakeCohort(tmp_path),
        dataset_verifier=verify,
        apply=False,
    )

    verification = result["dataset_verification"]
    assert verification["receipt_sha256"] == _sha256(fixture["corpus_path"])
    assert len(verification["file_identities"]) > 5
    assert all(
        {"path", "sha256", "bytes", "device", "inode"} <= set(identity)
        for identity in verification["file_identities"]
    )


class _FakeAwsRunner:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[str], str]] = []

    def run_json(self, argv, *, operation: str):
        self.calls.append((list(argv), operation))
        if not self.outputs:
            raise AssertionError(f"unexpected AWS call: {argv}")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _aws_profile_object():
    return SimpleNamespace(
        schema_version=1,
        profile_id="aws-p5.48xlarge",
        provider="aws-p5.48xlarge",
        instance_type="p5.48xlarge",
        gres="gpu:h100:8",
        purchase_model="on_demand",
        allocated_gpus=8,
        train_groups=(4, 4),
        assigned_seeds=(1, 2, 3, 4),
        process_env_allowlist=(
            "AWS_REGION",
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PYTHONPATH",
        ),
        sha256="7" * 64,
    )


def _aws_runtime_object():
    return SimpleNamespace(
        region="us-east-1",
        s3_root="s3://memorysplit-prod/cohort-v2",
        ami_id="ami-0123456789abcdef0",
        container_digest="sha256:" + "8" * 64,
        container_image=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
            "@sha256:" + "8" * 64
        ),
    )


def _aws_manifest_object(seed: int = 1):
    runs = tuple(
        SimpleNamespace(
            run_id=f"memorysplit-v2-360m-s{seed}-{arm}",
            arm=arm,
            seed=seed,
            config=f"configs/360m-v2/{arm}-s{seed}.yaml",
            config_sha256=("a" if arm == "dense" else "b") * 64,
        )
        for arm in ("dense", "split90")
    )
    value = {
        "schema_version": 2,
        "provider": "aws-p5.48xlarge",
        "seed": seed,
        "release_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "cohort_assignment_sha256": "3" * 64,
        "study_lock_sha256": "4" * 64,
        "source_commit": "5" * 40,
        "runs": [
            {
                "run_id": run.run_id,
                "arm": run.arm,
                "seed": run.seed,
                "config": run.config,
                "config_sha256": run.config_sha256,
            }
            for run in runs
        ],
    }
    return SimpleNamespace(
        schema_version=2,
        provider="aws-p5.48xlarge",
        seed=seed,
        release_sha256="1" * 64,
        dataset_sha256="2" * 64,
        cohort_assignment_sha256="3" * 64,
        study_lock_sha256="4" * 64,
        source_commit="5" * 40,
        runs=runs,
        value=value,
        sha256=hashlib.sha256(_canonical(value)).hexdigest(),
    )


def _aws_release_object():
    return SimpleNamespace(
        provider="aws-p5.48xlarge",
        archive_sha256="1" * 64,
        receipt_sha256="9" * 64,
        members_sha256="6" * 64,
        source_commit="5" * 40,
    )


def test_aws_launch_boundary_uses_pinned_image_private_home_and_bounded_deadline(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    release = _aws_release_object()
    manifest = _aws_manifest_object()
    intent = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=(
            __import__("datetime").datetime.now(__import__("datetime").UTC)
            + __import__("datetime").timedelta(hours=1)
        ).isoformat().replace("+00:00", "Z"),
    )
    by_name = {step["name"]: step["argv"] for step in intent["steps"]}

    private_home = by_name["prepare-aws-private-home"]
    assert private_home[-1] == "/var/lib/memorysplit/aws-private-home"
    assert private_home[private_home.index("-m") + 1] == "0700"
    bootstrap = by_name["bootstrap"]
    assert (
        bootstrap[bootstrap.index("--container-image") + 1]
        == _aws_runtime_object().container_image
    )
    assert (
        bootstrap[bootstrap.index("--aws-private-home") + 1]
        == "/var/lib/memorysplit/aws-private-home"
    )

    with pytest.raises(Exception) as caught:
        backend._validate_submit_selection(
            "i-0123456789abcdef0",
            "2099-01-01T00:00:00Z",
        )
    assert getattr(caught.value, "code", None) == "TERMINATION_DEADLINE_INVALID"


def test_aws_evaluation_runs_task8_module_in_pinned_container(tmp_path):
    from msctl.aws_p5 import AwsP5Backend

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    commands = backend._evaluation_argv(
        _aws_release_object(),
        _aws_manifest_object(),
    )

    assert len(commands) == 2
    for argv in commands:
        image_index = argv.index(_aws_runtime_object().container_image)
        assert argv[:2] == ["/usr/bin/docker", "run"]
        assert argv[image_index + 1 : image_index + 5] == [
            "/opt/venv/bin/python",
            "-m",
            "evals.confirmatory",
            "evaluate",
        ]
        assert "runner.py" not in " ".join(argv)


def test_aws_pair_journal_repairs_crash_after_first_arm_write(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    manifest = _aws_manifest_object()
    release = _aws_release_object()
    deadline = (
        __import__("datetime").datetime.now(__import__("datetime").UTC)
        + __import__("datetime").timedelta(hours=1)
    ).isoformat().replace("+00:00", "Z")
    intent = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=deadline,
        ),
        instance_id="i-0123456789abcdef0",
        terminate_at=deadline,
    )
    published = {
        "intent_sha256": hashlib.sha256(_canonical(intent)).hexdigest(),
        "intent_uri": (
            f"{_aws_runtime_object().s3_root}/operations/intents/sha256/"
            f"{hashlib.sha256(_canonical(intent)).hexdigest()}.json"
        ),
    }
    states = [
        backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id="i-0123456789abcdef0",
            terminate_at=deadline,
            intent=intent,
            published=published,
            attempt=1,
        )
        for run in manifest.runs
    ]
    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_aws_pair(
            manifest.sha256,
            {
                "schema_version": 1,
                "provider": "aws-p5.48xlarge",
                "instance_type": "p5.48xlarge",
                "profile_sha256": backend.profile.sha256,
                "gres": "gpu:h100:8",
                "run_manifest_sha256": manifest.sha256,
                "operation_id": intent["operation_id"],
                "states": states,
            },
        )
        store.write_run(str(states[0]["run_id"]), states[0])
        backend._repair_paired_states(store, manifest)
        assert store.read_run(str(states[1]["run_id"])) == states[1]


def test_aws_pair_journal_restores_both_stale_arm_files(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    manifest = _aws_manifest_object()
    release = _aws_release_object()
    intent = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=_AWS_TERMINATE_AT,
        ),
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    digest = hashlib.sha256(_canonical(intent)).hexdigest()
    states = [
        backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id="i-0123456789abcdef0",
            terminate_at=_AWS_TERMINATE_AT,
            intent=intent,
            published={
                "intent_sha256": digest,
                "intent_uri": (
                    f"{_aws_runtime_object().s3_root}/operations/intents/"
                    f"sha256/{digest}.json"
                ),
            },
            attempt=1,
        )
        for run in manifest.runs
    ]
    stale = [dict(state) for state in states]
    durable = [dict(state) for state in states]
    for state in stale:
        state["updated_at"] = "2026-07-25T00:00:00Z"
    for state in durable:
        state["status"] = "SENDING"
        state["send_attempted"] = True
        state["updated_at"] = "2026-07-25T00:00:01Z"

    store = StateStore(backend.state_root)
    with store.locked():
        for state in stale:
            store.write_run(str(state["run_id"]), state)
        store.write_aws_pair(
            manifest.sha256,
            {
                "schema_version": 1,
                "provider": backend.profile.provider,
                "instance_type": backend.profile.instance_type,
                "profile_sha256": backend.profile.sha256,
                "gres": backend.profile.gres,
                "run_manifest_sha256": manifest.sha256,
                "operation_id": intent["operation_id"],
                "states": durable,
            },
        )
        backend._repair_paired_states(store, manifest)
        assert [
            store.read_run(str(state["run_id"])) for state in durable
        ] == durable


def _bound_instance(
    manifest,
    *,
    instance_id: str = "i-0123456789abcdef0",
    release_sha256: str | None = None,
):
    return {
        "instance_id": instance_id,
        "instance_type": "p5.48xlarge",
        "profile_instance_type": "p5.48xlarge",
        "state": "running",
        "instance_profile_arn": (
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        "provider": "aws-p5.48xlarge",
        "seed": manifest.seed,
        "cohort_sha256": manifest.cohort_assignment_sha256,
        "release_sha256": release_sha256 or manifest.release_sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "run_manifest_sha256": manifest.sha256,
        "profile_sha256": _aws_profile_object().sha256,
        "gres": "gpu:h100:8",
    }


def _selected_instance(
    manifest,
    *,
    instance_id: str = "i-0123456789abcdef0",
    bound: bool,
):
    runtime = _aws_runtime_object()
    profile = _aws_profile_object()
    tags = {
        "provider": "aws-p5.48xlarge",
        "profile_instance_type": "p5.48xlarge",
        "seed": manifest.seed,
        "cohort_sha256": manifest.cohort_assignment_sha256,
        "release_sha256": manifest.release_sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "run_manifest_sha256": manifest.sha256,
        "profile_sha256": profile.sha256,
        "runtime_sha256": hashlib.sha256(
            _canonical(
                {
                    "ami_id": runtime.ami_id,
                    "container_image": runtime.container_image,
                    "container_digest": runtime.container_digest,
                    "gid": getattr(runtime, "gid", 1000),
                    "region": runtime.region,
                    "s3_root": runtime.s3_root,
                    "uid": getattr(runtime, "uid", 1000),
                }
            )
        ).hexdigest(),
        "container_digest": runtime.container_digest,
        "gres": "gpu:h100:8",
        "terminate_at": _AWS_TERMINATE_AT,
    }
    if not bound:
        tags = {key: None for key in tags}
    return {
        "instance_id": instance_id,
        "instance_type": "p5.48xlarge",
        "state": "running",
        "instance_profile_arn": (
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        "ami_id": runtime.ami_id,
        **tags,
    }


def _aws_submit_outputs(backend, release, manifest, *, ssm_online=True):
    from msctl.aws_p5 import (
        ARGV_DOCUMENT_NAME,
        ARGV_DOCUMENT_SHA256,
    )
    from msctl.jsonutil import canonical_json

    instance_id = "i-0123456789abcdef0"
    terminate_at = _AWS_TERMINATE_AT
    core = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=terminate_at,
    )
    intent = backend._operation_envelope(
        core,
        instance_id=instance_id,
        terminate_at=terminate_at,
    )
    payload = canonical_json(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = __import__("base64").b64encode(
        bytes.fromhex(digest)
    ).decode("ascii")
    outputs = [
        {"instances": []},
        {"instances": [_selected_instance(manifest, bound=False)]},
        {
            "managed_instances": (
                [{"instance_id": instance_id, "ping_status": "Online"}]
                if ssm_online
                else []
            )
        },
    ]
    if not ssm_online:
        return outputs
    outputs.extend(
        [
            {
                "documents": [
                    {
                        "name": ARGV_DOCUMENT_NAME,
                        "hash": ARGV_DOCUMENT_SHA256,
                        "status": "Active",
                    }
                ]
            },
            {"instances": [_selected_instance(manifest, bound=False)]},
            {},
            {},
            {"instances": [_selected_instance(manifest, bound=True)]},
            {
                "attribute": {
                    "instance_id": instance_id,
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "version_id": "version-1",
                }
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "content_length": len(payload),
                    "metadata": {
                        "operation-id": intent["operation_id"],
                        "sha256": digest,
                    },
                    "version_id": "version-1",
                }
            },
            {"command": {"command_id": "cmd-0123456789abcdef0"}},
        ]
    )
    return outputs


def _aws_run_state(
    backend,
    release,
    manifest,
    run,
    *,
    status,
    command_id,
):
    from msctl.jsonutil import canonical_json

    instance_id = "i-0123456789abcdef0"
    terminate_at = _AWS_TERMINATE_AT
    core = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=terminate_at,
    )
    intent = backend._operation_envelope(
        core,
        instance_id=instance_id,
        terminate_at=terminate_at,
    )
    digest = hashlib.sha256(canonical_json(intent)).hexdigest()
    state = backend._new_aws_run_state(
        run=run,
        manifest=manifest,
        operation="submit",
        instance_id=instance_id,
        terminate_at=terminate_at,
        intent=intent,
        published={
            "intent_sha256": digest,
            "intent_uri": (
                "s3://memorysplit-prod/cohort-v2/operations/intents/"
                f"sha256/{digest}.json"
            ),
        },
        attempt=1,
    )
    state["command_id"] = command_id
    state["send_attempted"] = True
    state["status"] = status
    return state


def test_aws_cli_runner_is_argv_only_sanitized_and_strict(tmp_path):
    from msctl.aws_p5 import AwsP5Backend

    runner = _FakeAwsRunner(
        {
            "account": "123456789012",
            "arn": "arn:aws:sts::123456789012:assumed-role/msctl/operator",
            "user_id": "AROATEST:operator",
            "unexpected": True,
        }
    )
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )

    with pytest.raises(Exception) as caught:
        backend.auth_check()

    assert getattr(caught.value, "code", None) == "AWS_OUTPUT_INVALID"
    argv, operation = runner.calls[0]
    assert argv[:2] == ["env", "-i"]
    assert "aws" in argv
    assert operation == "auth check"
    rendered = json.dumps(argv)
    for secret in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "MSCTL_APPROVAL_KEY",
    ):
        assert secret not in rendered

    credentialed_runtime = _aws_runtime_object()
    credentialed_runtime.s3_root = (
        "s3://access-key:secret-key@memorysplit-prod/cohort-v2"
    )
    with pytest.raises(Exception) as caught:
        AwsP5Backend(
            profile=_aws_profile_object(),
            runtime=credentialed_runtime,
            instance_profile_arn=(
                "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
            ),
            state_root=tmp_path / "unsafe-state",
            runner=_FakeAwsRunner(),
        )
    assert getattr(caught.value, "code", None) == "AWS_RUNTIME_INVALID"


def test_aws_cli_runner_maps_nonzero_exit_without_leaking_output(monkeypatch):
    from msctl.aws_p5 import SubprocessAwsJsonRunner

    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=17,
            stdout='{"secret":"must-not-leak"}',
            stderr="credential=must-not-leak",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(Exception) as caught:
        SubprocessAwsJsonRunner().run_json(
            ["env", "-i", "aws", "sts", "get-caller-identity"],
            operation="auth check",
        )

    assert getattr(caught.value, "code", None) == "AWS_COMMAND_FAILED"
    assert "must-not-leak" not in str(caught.value)
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["env"] == {}

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(
            returncode=0,
            stdout='{"account":"first","account":"second"}',
            stderr="",
        ),
    )
    with pytest.raises(Exception) as caught:
        SubprocessAwsJsonRunner().run_json(
            ["env", "-i", "aws", "sts", "get-caller-identity"],
            operation="auth check",
        )
    assert getattr(caught.value, "code", None) == "AWS_OUTPUT_INVALID"


def test_aws_read_only_lifecycle_is_dry_run_first_and_argv_only(tmp_path):
    from cluster.aws.p5.corpus_contract import verify_canonical_corpus
    from msctl.aws_p5 import AwsP5Backend
    from tests.test_aws_p5_launcher import _launcher_fixture

    fixture = _launcher_fixture(tmp_path / "task4")

    def verify(receipt_path, *, expected_sha256, expected_ordered_sha256):
        return verify_canonical_corpus(
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
            semantic_verifier=lambda _root: fixture["corpus"],
        )

    runner = _FakeAwsRunner(
        {
            "offerings": [
                {
                    "instance_type": "p5.48xlarge",
                    "location": "us-east-1",
                    "location_type": "region",
                }
            ]
        }
    )
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        corpus_verifier=verify,
    )

    capacity = backend.capacity_check()
    assert capacity["offered"] is True
    assert runner.calls[0][0][:2] == ["env", "-i"]

    lock = tmp_path / "requirements-aws-p5.lock"
    lock.write_text("fixture==1.0 --hash=sha256:" + "a" * 64 + "\n")
    with pytest.raises(Exception) as static_environment:
        backend.dispatch(
            "env ensure",
            SimpleNamespace(
                apply=False,
                lock=lock,
                root=tmp_path / "environment",
            ),
        )
    assert (
        getattr(static_environment.value, "code", None)
        == "STATIC_ENVIRONMENT_FORBIDDEN"
    )

    dry_run, dataset = backend.dispatch(
        "dataset ensure",
        SimpleNamespace(apply=False, receipt=fixture["corpus_path"]),
    )
    assert dry_run is True
    assert dataset["verified"] is False
    assert any("put-object" in argv for argv in dataset["commands"])
    assert any("head-object" in argv for argv in dataset["commands"])

    dry_run, collection = backend.dispatch(
        "collect",
        SimpleNamespace(
            apply=False,
            source="results/seed-1.json",
            out=tmp_path / "seed-1.json",
        ),
    )
    assert dry_run is True
    assert collection["collected"] == 0
    assert "get-object" in collection["commands"][0]
    assert len(runner.calls) == 1


def test_aws_dataset_ensure_plans_every_verified_task4_object_not_head_only(
    tmp_path,
):
    from cluster.aws.p5.corpus_contract import verify_canonical_corpus
    from msctl.aws_p5 import AwsP5Backend
    from tests.test_aws_p5_launcher import _launcher_fixture

    fixture = _launcher_fixture(tmp_path / "task4")

    def verify(receipt_path, *, expected_sha256, expected_ordered_sha256):
        return verify_canonical_corpus(
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
            semantic_verifier=lambda _root: fixture["corpus"],
        )

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
        corpus_verifier=verify,
    )

    planned = backend.dataset_ensure(
        receipt_path=fixture["corpus_path"],
        apply=False,
    )

    identities = planned["file_identities"]
    puts = [
        argv
        for argv in planned["commands"]
        if "put-object" in argv
    ]
    heads = [
        argv
        for argv in planned["commands"]
        if "head-object" in argv
    ]
    assert len(puts) == len(identities)
    assert len(heads) == len(identities)
    assert all("--if-none-match" in argv for argv in puts)
    assert all("--checksum-sha256" in argv for argv in puts)
    assert all("device" in identity for identity in identities)


def test_aws_environment_ensure_rejects_removed_static_lock_api(tmp_path):
    from msctl.aws_p5 import AwsP5Backend

    lock = tmp_path / "requirements-aws-p5.lock"
    lock.write_bytes(b"torch==2.7.1 --hash=sha256:" + b"a" * 64 + b"\n")
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )

    with pytest.raises(Exception) as caught:
        backend.env_ensure(
            root=tmp_path / "environment",
            lock=lock,
            apply=False,
        )
    assert getattr(caught.value, "code", None) == "STATIC_ENVIRONMENT_FORBIDDEN"


def test_aws_render_uses_reviewed_bootstrap_and_paired_launcher_only(tmp_path):
    from msctl.aws_p5 import AwsP5Backend

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )

    rendered = backend.render(
        release=_aws_release_object(),
        manifest=_aws_manifest_object(),
    )
    steps = rendered["operation_intent"]["steps"]

    names = [step["name"] for step in steps]
    assert names == [
        "prepare-aws-private-home",
        "prepare-staging",
        "materialize-release-archive",
        "materialize-release-receipt",
        "materialize-cohort-assignment",
        "materialize-dataset",
        "bootstrap",
        "build-launcher-manifest",
        "paired-launch",
    ]
    bootstrap = steps[names.index("bootstrap")]["argv"]
    launcher = steps[names.index("paired-launch")]["argv"]
    assert bootstrap[1].endswith("/cluster/aws/p5/bootstrap.py")
    assert launcher[1].endswith(
        "/cluster/aws/p5/launch_seed_pair.py"
    )
    assert "torchrun" not in json.dumps(rendered)
    assert "torch.distributed.run" not in json.dumps(rendered)


def test_aws_post_bootstrap_builder_emits_exact_task3_launcher_manifest(
    tmp_path,
):
    from msctl.aws_launch_manifest import build_launcher_manifest
    from tests.test_aws_p5_launcher import _launcher_fixture

    fixture = _launcher_fixture(tmp_path / "task3")
    expected = fixture["manifest"]
    runs = [
        {
            "arm": row["arm"],
            "config": row["config"],
            "config_sha256": row["config_sha256"],
        }
        for row in expected["runs"]
    ]

    actual = build_launcher_manifest(
        out=fixture["scratch_root"] / "staging" / "msctl-manifest.json",
        scratch_root=fixture["scratch_root"],
        seed=expected["seed"],
        profile_sha256=expected["profile_sha256"],
        release_sha256=expected["release_sha256"],
        release_members_sha256=expected["release_members_sha256"],
        cohort_assignment_sha256=expected["cohort_assignment_sha256"],
        code_commit=expected["code_commit"],
        bootstrap_receipt=fixture["bootstrap_path"],
        corpus_receipt=fixture["corpus_path"],
        runs=runs,
    )

    assert actual == expected


def test_aws_submit_is_dry_run_by_default_and_approval_precedes_aws_calls(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend

    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )
    manifest = _aws_manifest_object()
    instance_id = "i-0123456789abcdef0"
    terminate_at = _AWS_TERMINATE_AT

    planned = backend.submit(
        release=_aws_release_object(),
        manifest=manifest,
        instance_id=instance_id,
        terminate_at=terminate_at,
        approval_path=None,
        apply=False,
    )

    assert planned["submitted"] == 0
    assert planned["instance_id"] == instance_id
    assert planned["terminate_at"] == terminate_at
    assert all("run-instances" not in command for command in planned["commands"])
    assert any(
        "--instance-ids" in command
        and command[command.index("--instance-ids") + 1] == instance_id
        for command in planned["commands"]
    )
    steps = planned["operation_intent"]["steps"]
    names = [step["name"] for step in steps]
    assert names[0] == "auto-termination"
    assert "materialize-release-archive" in names
    assert "materialize-dataset" in names
    assert names[-3:] == [
        "bootstrap",
        "build-launcher-manifest",
        "paired-launch",
    ]
    bootstrap = steps[names.index("bootstrap")]["argv"]
    launcher = steps[names.index("paired-launch")]["argv"]
    assert bootstrap[1].endswith("/cluster/aws/p5/bootstrap.py")
    assert "--release-archive" in bootstrap
    assert "--dataset-receipt" in bootstrap
    assert "--cohort-assignment" in bootstrap
    assert "--apply" in bootstrap
    assert launcher[1].endswith("/cluster/aws/p5/launch_seed_pair.py")
    assert launcher[launcher.index("--seed") + 1] == str(manifest.seed)
    assert "--manifest" in launcher
    assert "--apply" in launcher
    assert "torchrun" not in json.dumps(planned)
    assert "torch.distributed.run" not in json.dumps(planned)
    assert runner.calls == []

    with pytest.raises(Exception) as caught:
        backend.submit(
            release=_aws_release_object(),
            manifest=manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            approval_path=None,
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "APPROVAL_REQUIRED"
    assert runner.calls == []


def test_aws_submit_binds_selected_instance_runtime_and_deadline_to_approval(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend

    manifest = _aws_manifest_object()
    selected = "i-0123456789abcdef0"
    discovered = _bound_instance(
        manifest,
        instance_id="i-11111111111111111",
    )
    runner = _FakeAwsRunner({"instances": [discovered]})
    captured = {}

    def approve(**kwargs):
        captured.update(kwargs)
        return {}

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=discovered["instance_profile_arn"],
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=approve,
    )

    with pytest.raises(Exception) as caught:
        backend.submit(
            release=_aws_release_object(),
            manifest=manifest,
            instance_id=selected,
            terminate_at=_AWS_TERMINATE_AT,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )

    assert getattr(caught.value, "code", None) == "INSTANCE_SELECTION_CONFLICT"
    resources = captured["resources"]
    assert resources["instance_id"] == selected
    assert resources["terminate_at"] == _AWS_TERMINATE_AT
    assert resources["release_sha256"] == manifest.release_sha256
    assert resources["run_manifest_sha256"] == manifest.sha256
    assert resources["seed"] == manifest.seed
    assert resources["ami_id"] == _aws_runtime_object().ami_id
    assert resources["container_digest"] == _aws_runtime_object().container_digest
    assert resources["profile_sha256"] == _aws_profile_object().sha256
    assert len(resources["runtime_sha256"]) == 64
    assert all("run-instances" not in argv for argv, _ in runner.calls)
    assert all("send-command" not in argv for argv, _ in runner.calls)


def test_aws_approval_binds_every_extended_execution_resource(tmp_path):
    from datetime import UTC, datetime

    from msctl.approval import verify_scope_approval
    from msctl.aws_p5 import AwsP5Backend

    key = "approval-key-" * 4
    manifest = _aws_manifest_object()
    release = _aws_release_object()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    resources = backend._submit_resources(
        release=release,
        manifest=manifest,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    resources.update(
        {
            "dataset_pointer_sha256": "5" * 64,
            "dataset_verification_sha256": "6" * 64,
            "environment_receipt_sha256": "7" * 64,
        }
    )
    unsigned = {
        "schema_version": 1,
        "receipt_id": "aws-submit-fixture",
        "provider": "aws-p5.48xlarge",
        "operation": "submit",
        "release_sha256": release.archive_sha256,
        "run_manifest_sha256": manifest.sha256,
        "resources": resources,
        "limits": {"gpu_hours": 192.0, "jobs": 1},
        "expires_at": "2098-01-01T00:00:00Z",
        "key_id": "fixture",
    }
    receipt = {
        **unsigned,
        "signature": hmac.new(
            key.encode(),
            _canonical(unsigned),
            hashlib.sha256,
        ).hexdigest(),
    }
    path = _write_json(tmp_path / "approval.json", receipt)

    verified = verify_scope_approval(
        path,
        operation="submit",
        release_sha256=release.archive_sha256,
        scope_sha256=manifest.sha256,
        resources=resources,
        profile=_aws_profile_object(),
        environ={"MSCTL_APPROVAL_KEY": key},
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    assert verified["resources"]["instance_id"] == "i-0123456789abcdef0"

    with pytest.raises(Exception) as caught:
        verify_scope_approval(
            path,
            operation="submit",
            release_sha256=release.archive_sha256,
            scope_sha256=manifest.sha256,
            resources={**resources, "instance_id": "i-11111111111111111"},
            profile=_aws_profile_object(),
            environ={"MSCTL_APPROVAL_KEY": key},
            now=datetime(2026, 7, 23, tzinfo=UTC),
        )
    assert getattr(caught.value, "code", None) == "APPROVAL_SCOPE_MISMATCH"


def test_aws_selected_instance_is_validated_then_tagged_with_exact_binding(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend

    manifest = _aws_manifest_object()
    selected = _selected_instance(manifest, bound=False)
    runner = _FakeAwsRunner(
        {"instances": [selected]},
        {},
        {},
        {"instances": [_selected_instance(manifest, bound=True)]},
        {
            "attribute": {
                "instance_id": selected["instance_id"],
                "shutdown_behavior": "terminate",
            }
        },
    )
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=selected["instance_profile_arn"],
        state_root=tmp_path / "state",
        runner=runner,
    )

    result = backend._bind_selected_instance(
        manifest,
        instance_id=selected["instance_id"],
        terminate_at=_AWS_TERMINATE_AT,
    )

    assert result["instance_id"] == selected["instance_id"]
    assert result["terminate_at"] == _AWS_TERMINATE_AT
    assert all("run-instances" not in argv for argv, _ in runner.calls)
    tag_call = next(argv for argv, _ in runner.calls if "create-tags" in argv)
    rendered = tag_call[tag_call.index("--tags") + 1]
    for value in (
        manifest.sha256,
        manifest.release_sha256,
        _aws_profile_object().sha256,
        _aws_runtime_object().container_digest,
        _AWS_TERMINATE_AT,
    ):
        assert value in rendered
    assert any("modify-instance-attribute" in argv for argv, _ in runner.calls)

    mismatched = _selected_instance(manifest, bound=False)
    mismatched["ami_id"] = "ami-11111111111111111"
    rejected = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=selected["instance_profile_arn"],
        state_root=tmp_path / "other-state",
        runner=_FakeAwsRunner({"instances": [mismatched]}),
    )
    with pytest.raises(Exception) as caught:
        rejected._bind_selected_instance(
            manifest,
            instance_id=selected["instance_id"],
            terminate_at=_AWS_TERMINATE_AT,
        )
    assert getattr(caught.value, "code", None) == "INSTANCE_BINDING_MISMATCH"


def test_aws_operation_intent_uses_conditional_content_addressed_s3_put(
    tmp_path,
):
    import base64

    from msctl.aws_p5 import AwsP5Backend

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    core = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=_AWS_TERMINATE_AT,
    )
    intent = backend._operation_envelope(
        core,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    payload = _canonical(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
    runner = _FakeAwsRunner(
        {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "intent-version-1",
            }
        },
        {
            "object": {
                "checksum_sha256": checksum,
                "content_length": len(payload),
                "metadata": {
                    "operation-id": intent["operation_id"],
                    "sha256": digest,
                },
                "version_id": "intent-version-1",
            }
        },
    )
    backend.runner = runner

    published = backend._publish_operation_intent(intent)

    assert published["intent_sha256"] == digest
    assert published["intent_uri"].endswith(
        f"/operations/intents/sha256/{digest}.json"
    )
    put, head = [argv for argv, _operation in runner.calls]
    assert "put-object" in put
    assert put[put.index("--if-none-match") + 1] == "*"
    assert put[put.index("--checksum-sha256") + 1] == checksum
    assert put[put.index("--metadata") + 1] == (
        f"operation-id={intent['operation_id']},sha256={digest}"
    )
    assert "head-object" in head


def test_aws_operation_intent_recovers_a_lost_conditional_put_response(
    tmp_path,
):
    import base64

    from msctl.aws_p5 import AwsP5Backend
    from msctl.errors import MsctlError

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    intent = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=_AWS_TERMINATE_AT,
        ),
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    payload = _canonical(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
    backend.runner = _FakeAwsRunner(
        MsctlError("AWS_COMMAND_FAILED", "put response was lost"),
        {
            "object": {
                "checksum_sha256": checksum,
                "content_length": len(payload),
                "metadata": {
                    "operation-id": intent["operation_id"],
                    "sha256": digest,
                },
                "version_id": "intent-version-1",
            }
        },
    )

    published = backend._publish_operation_intent(intent)

    assert published["intent_sha256"] == digest
    assert len(backend.runner.calls) == 2
    assert "put-object" in backend.runner.calls[0][0]
    assert "head-object" in backend.runner.calls[1][0]


def test_aws_provisions_only_the_fixed_argv_document_hash(tmp_path):
    from msctl.aws_p5 import (
        ARGV_DOCUMENT_NAME,
        ARGV_DOCUMENT_SHA256,
        AwsP5Backend,
    )

    runner = _FakeAwsRunner(
        {"documents": []},
        {
            "document": {
                "hash": ARGV_DOCUMENT_SHA256,
                "name": ARGV_DOCUMENT_NAME,
                "status": "Active",
            }
        },
    )
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )

    backend._ensure_argv_document()

    create = next(argv for argv, _ in runner.calls if "create-document" in argv)
    assert create[create.index("--name") + 1] == ARGV_DOCUMENT_NAME
    assert hashlib.sha256(
        create[create.index("--content") + 1].encode("ascii")
    ).hexdigest() == ARGV_DOCUMENT_SHA256
    assert "msctl/aws_argv.py" in create[create.index("--content") + 1]

    rejected = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "other-state",
        runner=_FakeAwsRunner(
            {
                "documents": [
                    {
                        "hash": "0" * 64,
                        "name": ARGV_DOCUMENT_NAME,
                        "status": "Active",
                    }
                ]
            }
        ),
    )
    with pytest.raises(Exception) as caught:
        rejected._ensure_argv_document()
    assert getattr(caught.value, "code", None) == "SSM_DOCUMENT_MISMATCH"


def test_aws_remote_wrapper_acquires_once_and_writes_terminal_receipt(tmp_path):
    from msctl.aws_argv import RemoteIntentError, _receipt, execute_intent
    from msctl.aws_p5 import AwsP5Backend
    from msctl.jsonutil import canonical_json

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    manifest = _aws_manifest_object()
    release = _aws_release_object()
    core = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=_AWS_TERMINATE_AT,
    )
    intent = backend._operation_envelope(
        core,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    payload = canonical_json(intent)
    digest = hashlib.sha256(payload).hexdigest()
    intent_uri = (
        "s3://memorysplit-prod/cohort-v2/operations/intents/"
        f"sha256/{digest}.json"
    )

    class Store:
        def __init__(self):
            self.objects = {intent_uri: payload}
            self.metadata = {}

        def read(self, uri, *, expected_sha256=None):
            value = self.objects[uri]
            if expected_sha256 is not None:
                assert hashlib.sha256(value).hexdigest() == expected_sha256
            return value

        def read_if_exists(self, uri):
            return self.objects.get(uri)

        def put_if_absent(self, uri, value, *, metadata):
            if uri in self.objects:
                return self.objects[uri] == value
            self.objects[uri] = value
            self.metadata[uri] = dict(metadata)
            return True

    class Executor:
        def __init__(self):
            self.calls = []

        def run(self, argv, *, environment):
            self.calls.append((list(argv), dict(environment)))
            return 0

    store = Store()
    executor = Executor()
    first = execute_intent(
        intent_uri=intent_uri,
        intent_sha256=digest,
        store=store,
        executor=executor,
    )
    second = execute_intent(
        intent_uri=intent_uri,
        intent_sha256=digest,
        store=store,
        executor=executor,
    )

    assert first["executed"] is True
    assert second["executed"] is False
    assert second["idempotent"] is True
    assert second["terminal"] is True
    assert second["status"] == "success"
    assert second["recovery_required"] is False
    assert second["returncode"] == 0
    assert len(executor.calls) == len(intent["steps"])
    assert store.metadata[intent["started_receipt_uri"]] == {
        "operation-id": intent["operation_id"],
        "intent-sha256": digest,
        "receipt-kind": "started",
    }
    assert store.metadata[intent["terminal_receipt_uri"]][
        "receipt-kind"
    ] == "terminal"
    assert all(
        not any(name in environment for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ))
        for _, environment in executor.calls
    )
    stuck_store = Store()
    stuck_store.objects[intent["started_receipt_uri"]] = _receipt(
        intent,
        intent_sha256=digest,
        kind="started",
        nonce="a" * 32,
    )
    stuck_executor = Executor()
    stuck = execute_intent(
        intent_uri=intent_uri,
        intent_sha256=digest,
        store=stuck_store,
        executor=stuck_executor,
    )
    assert stuck["executed"] is False
    assert stuck["terminal"] is False
    assert stuck["status"] == "recovery-required"
    assert stuck["recovery_required"] is True
    assert stuck["returncode"] == 75
    assert stuck_executor.calls == []
    corrupt_store = Store()
    corrupt_store.objects[intent["started_receipt_uri"]] = b"{}\n"
    with pytest.raises(RemoteIntentError, match="started receipt"):
        execute_intent(
            intent_uri=intent_uri,
            intent_sha256=digest,
            store=corrupt_store,
            executor=Executor(),
        )


def test_aws_remote_wrapper_main_returns_nonzero_for_recovery(monkeypatch):
    import msctl.aws_argv as aws_argv

    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr(aws_argv, "AwsCliObjectStore", lambda **_kwargs: object())
    monkeypatch.setattr(aws_argv, "SubprocessArgvExecutor", lambda: object())
    monkeypatch.setattr(
        aws_argv,
        "execute_intent",
        lambda **_kwargs: {
            "schema_version": 1,
            "operation_id": "1" * 64,
            "executed": False,
            "idempotent": True,
            "terminal": False,
            "status": "recovery-required",
            "recovery_required": True,
            "returncode": 75,
        },
    )

    assert aws_argv.main(
        [
            "--intent-uri",
            "s3://memorysplit-prod/intents/intent.json",
            "--intent-sha256",
            "2" * 64,
        ]
    ) == 75


def test_aws_remote_wrapper_rejects_unknown_checkpoint_receipt_fields(tmp_path):
    from msctl.aws_argv import RemoteIntentError, _validate_intent
    from msctl.aws_p5 import AwsP5Backend
    from msctl.contracts import verify_checkpoint_receipt
    from msctl.jsonutil import canonical_json

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    receipt = verify_checkpoint_receipt(
        _aws_checkpoint_receipt(tmp_path, manifest),
        release=release,
        manifest=manifest,
    )
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    core = backend._training_operation_intent(
        operation="resume",
        release=release,
        manifest=manifest,
        checkpoints=backend._checkpoint_map(manifest, receipt),
        checkpoint_receipt_sha256=receipt.sha256,
    )
    core["checkpoint_receipt"]["checkpoints"][0]["unexpected"] = True
    intent = backend._operation_envelope(
        core,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    payload = canonical_json(intent)

    with pytest.raises(RemoteIntentError, match="checkpoint"):
        _validate_intent(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_aws_remote_wrapper_rejects_a_substituted_argv_document_hash(tmp_path):
    from msctl.aws_argv import RemoteIntentError, _validate_intent
    from msctl.aws_p5 import AwsP5Backend
    from msctl.jsonutil import canonical_json

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    intent = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=_aws_release_object(),
            manifest=_aws_manifest_object(),
            terminate_at=_AWS_TERMINATE_AT,
        ),
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    intent["ssm_document"]["sha256"] = "0" * 64
    payload = canonical_json(intent)

    with pytest.raises(RemoteIntentError, match="SSM document"):
        _validate_intent(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_aws_remote_wrapper_rejects_an_expired_operation_intent(tmp_path):
    from msctl.aws_argv import RemoteIntentError, _validate_intent
    from msctl.aws_p5 import AwsP5Backend
    from msctl.jsonutil import canonical_json, canonical_sha256

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    intent = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=_aws_release_object(),
            manifest=_aws_manifest_object(),
            terminate_at=_AWS_TERMINATE_AT,
        ),
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    intent["terminate_at"] = "2000-01-01T00:00:00Z"
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
    intent["operation_id"] = canonical_sha256(identity)
    receipt_root = (
        "s3://memorysplit-prod/cohort-v2/operations/"
        f"{intent['operation_id']}/receipts"
    )
    intent["started_receipt_uri"] = f"{receipt_root}/started.json"
    intent["terminal_receipt_uri"] = f"{receipt_root}/terminal.json"
    payload = canonical_json(intent)

    with pytest.raises(RemoteIntentError, match="termination deadline"):
        _validate_intent(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_aws_remote_wrapper_rejects_redirected_operation_receipts(tmp_path):
    from msctl.aws_argv import RemoteIntentError, _validate_intent
    from msctl.aws_p5 import AwsP5Backend
    from msctl.jsonutil import canonical_json

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    intent = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=_aws_release_object(),
            manifest=_aws_manifest_object(),
            terminate_at=_AWS_TERMINATE_AT,
        ),
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    intent["started_receipt_uri"] = (
        "s3://memorysplit-prod/cohort-v2/operations/redirected/started.json"
    )
    payload = canonical_json(intent)

    with pytest.raises(RemoteIntentError, match="receipt"):
        _validate_intent(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_aws_submit_response_loss_never_resends_the_same_operation(tmp_path):
    import base64

    from msctl.aws_argv import _receipt
    from msctl.aws_p5 import (
        ARGV_DOCUMENT_NAME,
        ARGV_DOCUMENT_SHA256,
        AwsP5Backend,
    )
    from msctl.errors import MsctlError

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    selected = _selected_instance(manifest, bound=False)
    common = {
        "profile": _aws_profile_object(),
        "runtime": _aws_runtime_object(),
        "instance_profile_arn": selected["instance_profile_arn"],
        "state_root": tmp_path / "state",
        "approval_verifier": lambda **_: {},
    }
    preparer = AwsP5Backend(runner=_FakeAwsRunner(), **common)
    core = preparer._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at=_AWS_TERMINATE_AT,
    )
    intent = preparer._operation_envelope(
        core,
        instance_id=selected["instance_id"],
        terminate_at=_AWS_TERMINATE_AT,
    )
    payload = _canonical(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
    first_runner = _FakeAwsRunner(
        {"instances": []},
        {"instances": [selected]},
        {
            "managed_instances": [
                {
                    "instance_id": selected["instance_id"],
                    "ping_status": "Online",
                }
            ]
        },
        {
            "documents": [
                {
                    "hash": ARGV_DOCUMENT_SHA256,
                    "name": ARGV_DOCUMENT_NAME,
                    "status": "Active",
                }
            ]
        },
        {"instances": [selected]},
        {},
        {},
        {"instances": [_selected_instance(manifest, bound=True)]},
        {
            "attribute": {
                "instance_id": selected["instance_id"],
                "shutdown_behavior": "terminate",
            }
        },
        {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "intent-version-1",
            }
        },
        {
            "object": {
                "checksum_sha256": checksum,
                "content_length": len(payload),
                "metadata": {
                    "operation-id": intent["operation_id"],
                    "sha256": digest,
                },
                "version_id": "intent-version-1",
            }
        },
        MsctlError("AWS_COMMAND_FAILED", "simulated response loss"),
    )
    first = AwsP5Backend(runner=first_runner, **common)

    with pytest.raises(Exception) as caught:
        first.submit(
            release=release,
            manifest=manifest,
            instance_id=selected["instance_id"],
            terminate_at=_AWS_TERMINATE_AT,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "AWS_COMMAND_FAILED"
    assert sum("send-command" in argv for argv, _ in first_runner.calls) == 1
    states = [
        json.loads(
            (
                tmp_path
                / "state"
                / "runs"
                / f"{run.run_id}.json"
            ).read_text()
        )
        for run in manifest.runs
    ]
    assert {state["status"] for state in states} == {"SENDING"}
    assert {state["send_attempted"] for state in states} == {True}
    assert {state["intent_sha256"] for state in states} == {digest}

    second_runner = _FakeAwsRunner(
        {"commands": []},
        MsctlError("AWS_COMMAND_FAILED", "started receipt absent"),
        MsctlError("AWS_COMMAND_FAILED", "terminal receipt absent"),
    )
    second = AwsP5Backend(runner=second_runner, **common)
    with pytest.raises(Exception) as caught:
        second.submit(
            release=release,
            manifest=manifest,
            instance_id=selected["instance_id"],
            terminate_at=_AWS_TERMINATE_AT,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "SUBMISSION_UNCERTAIN"
    assert all("send-command" not in argv for argv, _ in second_runner.calls)

    started_payload = _receipt(
        intent,
        intent_sha256=digest,
        kind="started",
        nonce="b" * 32,
    )
    started_checksum = base64.b64encode(
        hashlib.sha256(started_payload).digest()
    ).decode("ascii")
    receipt_metadata = {
        "operation-id": intent["operation_id"],
        "intent-sha256": digest,
    }
    started_object = {
        "receipt": {
            "checksum_sha256": started_checksum,
            "content_length": len(started_payload),
            "metadata": {**receipt_metadata, "receipt-kind": "started"},
            "version_id": "started-version-2",
        }
    }

    class OperationReceiptRunner(_FakeAwsRunner):
        def __init__(self, *outputs, payloads):
            super().__init__(*outputs)
            self.payloads = payloads

        def run_json(self, argv, *, operation):
            if "get-object" in argv:
                destination = Path(argv[argv.index("--checksum-mode") + 2])
                key = argv[argv.index("--key") + 1]
                kind = Path(key).stem
                destination.write_bytes(self.payloads[kind])
            return super().run_json(argv, operation=operation)

    recovery_runner = OperationReceiptRunner(
        {"commands": []},
        started_object,
        started_object,
        MsctlError("AWS_COMMAND_FAILED", "terminal receipt absent"),
        payloads={"started": started_payload},
    )
    recovery = AwsP5Backend(runner=recovery_runner, **common)
    with pytest.raises(Exception) as recovery_required:
        recovery.submit(
            release=release,
            manifest=manifest,
            instance_id=selected["instance_id"],
            terminate_at=_AWS_TERMINATE_AT,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )
    assert getattr(recovery_required.value, "code", None) == (
        "REMOTE_RECOVERY_REQUIRED"
    )
    recovered_states = [
        json.loads(
            (
                tmp_path
                / "state"
                / "runs"
                / f"{run.run_id}.json"
            ).read_text()
        )
        for run in manifest.runs
    ]
    assert {state["status"] for state in recovered_states} == {
        "RECOVERY_REQUIRED"
    }
    assert all(
        "send-command" not in argv for argv, _ in recovery_runner.calls
    )

    terminal_payload = _receipt(
        intent,
        intent_sha256=digest,
        kind="terminal",
        nonce="b" * 32,
        returncode=0,
    )
    terminal_checksum = base64.b64encode(
        hashlib.sha256(terminal_payload).digest()
    ).decode("ascii")
    terminal_object = {
        "receipt": {
            "checksum_sha256": terminal_checksum,
            "content_length": len(terminal_payload),
            "metadata": {**receipt_metadata, "receipt-kind": "terminal"},
            "version_id": "terminal-version-1",
        }
    }

    terminal_runner = OperationReceiptRunner(
        {"commands": []},
        started_object,
        started_object,
        terminal_object,
        terminal_object,
        payloads={
            "started": started_payload,
            "terminal": terminal_payload,
        },
    )
    reconciler = AwsP5Backend(runner=terminal_runner, **common)
    terminal_result = reconciler.status(
        release=release,
        manifest=manifest,
        cached=False,
    )
    assert terminal_result["status"] == "REMOTE_TERMINAL_SUCCESS"
    assert terminal_result["returncode"] == 0
    assert terminal_result["terminal_receipt_sha256"] == hashlib.sha256(
        terminal_payload
    ).hexdigest()
    assert all(
        "send-command" not in argv for argv, _ in terminal_runner.calls
    )
    terminal_states = [
        json.loads(
            (
                tmp_path
                / "state"
                / "runs"
                / f"{run.run_id}.json"
            ).read_text()
        )
        for run in manifest.runs
    ]
    assert {state["status"] for state in terminal_states} == {
        "REMOTE_TERMINAL_SUCCESS"
    }

    mismatched_payload = _receipt(
        intent,
        intent_sha256=digest,
        kind="terminal",
        nonce="c" * 32,
        returncode=0,
    )
    mismatched_checksum = base64.b64encode(
        hashlib.sha256(mismatched_payload).digest()
    ).decode("ascii")
    mismatched_object = {
        "receipt": {
            "checksum_sha256": mismatched_checksum,
            "content_length": len(mismatched_payload),
            "metadata": {**receipt_metadata, "receipt-kind": "terminal"},
            "version_id": "terminal-version-2",
        }
    }
    mismatch_runner = OperationReceiptRunner(
        {"commands": []},
        started_object,
        started_object,
        mismatched_object,
        mismatched_object,
        payloads={
            "started": started_payload,
            "terminal": mismatched_payload,
        },
    )
    mismatch = AwsP5Backend(runner=mismatch_runner, **common)
    with pytest.raises(MsctlError) as mismatched:
        mismatch.status(
            release=release,
            manifest=manifest,
            cached=False,
        )
    assert mismatched.value.code == "REMOTE_RECEIPT_INVALID"


def test_aws_submit_rejects_state_bound_to_a_different_operation_intent(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=lambda **_: {},
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for run in manifest.runs:
            state = _aws_run_state(
                backend,
                release,
                manifest,
                run,
                status="SENDING",
                command_id=None,
            )
            state["operation_id"] = "f" * 64
            store.write_run(run.run_id, state)

    with pytest.raises(Exception) as caught:
        backend.submit(
            release=release,
            manifest=manifest,
            instance_id="i-0123456789abcdef0",
            terminate_at=_AWS_TERMINATE_AT,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )

    assert getattr(caught.value, "code", None) == "STATE_CORRUPT"
    assert runner.calls == []


def test_aws_lifecycle_rejects_state_with_a_different_runtime_binding(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for run in manifest.runs:
            state = _aws_run_state(
                backend,
                release,
                manifest,
                run,
                status="InProgress",
                command_id="cmd-0123456789abcdef0",
            )
            state["runtime_sha256"] = "e" * 64
            store.write_run(run.run_id, state)

    with pytest.raises(Exception) as caught:
        backend.status(
            release=release,
            manifest=manifest,
            cached=True,
        )

    assert getattr(caught.value, "code", None) == "RUN_ID_CONFLICT"


def test_aws_lifecycle_rejects_divergent_paired_operation_state(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for index, run in enumerate(manifest.runs):
            state = _aws_run_state(
                backend,
                release,
                manifest,
                run,
                status="InProgress",
                command_id="cmd-0123456789abcdef0",
            )
            if index:
                state["operation_id"] = "d" * 64
            store.write_run(run.run_id, state)

    with pytest.raises(Exception) as caught:
        backend.status(
            release=release,
            manifest=manifest,
            cached=True,
        )

    assert getattr(caught.value, "code", None) == "STATE_INCOMPLETE"


def test_aws_paid_continuation_rejects_an_expired_instance_deadline(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=lambda **_: {},
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for run in manifest.runs:
            state = _aws_run_state(
                backend,
                release,
                manifest,
                run,
                status="Success",
                command_id="cmd-0123456789abcdef0",
            )
            state["terminate_at"] = "2000-01-01T00:00:00Z"
            store.write_run(run.run_id, state)

    with pytest.raises(Exception) as caught:
        backend.evaluate(
            release=release,
            manifest=manifest,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )

    assert getattr(caught.value, "code", None) == "TERMINATION_DEADLINE_INVALID"
    assert runner.calls == []


def test_aws_instance_discovery_rejects_unknown_fields_wrong_tags_and_duplicates(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend

    manifest = _aws_manifest_object()
    exact = _bound_instance(manifest)
    outputs = (
        {"instances": [{**exact, "unexpected": True}]},
        {
            "instances": [
                _bound_instance(manifest, release_sha256="9" * 64)
            ]
        },
        {
            "instances": [
                exact,
                _bound_instance(
                    manifest,
                    instance_id="i-11111111111111111",
                ),
            ]
        },
    )
    runner = _FakeAwsRunner(*outputs)
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )

    expected_codes = (
        "AWS_OUTPUT_INVALID",
        "INSTANCE_BINDING_MISMATCH",
        "DUPLICATE_ACTIVE_SEED",
    )
    for expected in expected_codes:
        with pytest.raises(Exception) as caught:
            backend.discover_instances(manifest)
        assert getattr(caught.value, "code", None) == expected


def test_aws_submit_persists_one_paired_intent_and_is_idempotent(tmp_path):
    from msctl.aws_p5 import AwsP5Backend

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=lambda **_: {},
    )
    runner.outputs.extend(_aws_submit_outputs(backend, release, manifest))
    runner.outputs.append(
        {
            "command": {
                "command_id": "cmd-0123456789abcdef0",
                "status": "InProgress",
            }
        }
    )

    first = backend.submit(
        release=release,
        manifest=manifest,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )
    second = backend.submit(
        release=release,
        manifest=manifest,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )

    assert first["submitted"] == 1
    assert first["command_id"] == "cmd-0123456789abcdef0"
    assert second["submitted"] == 0
    assert second["idempotent"] is True
    assert sum("run-instances" in argv for argv, _ in runner.calls) == 0
    assert sum("send-command" in argv for argv, _ in runner.calls) == 1
    states = [
        json.loads(
            (
                tmp_path
                / "state"
                / "runs"
                / f"{run.run_id}.json"
            ).read_text()
        )
        for run in manifest.runs
    ]
    assert {state["instance_id"] for state in states} == {
        "i-0123456789abcdef0"
    }
    assert {state["command_id"] for state in states} == {
        "cmd-0123456789abcdef0"
    }
    assert {state["ami_id"] for state in states} == {
        backend.runtime.ami_id
    }
    assert {state["container_digest"] for state in states} == {
        backend.runtime.container_digest
    }
    assert {state["runtime_sha256"] for state in states} == {
        backend._runtime_sha256()
    }


def test_aws_submit_ssm_preflight_failure_never_records_or_sends_intent(tmp_path):
    from msctl.aws_p5 import AwsP5Backend

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    first_runner = _FakeAwsRunner()
    arguments = {
        "profile": _aws_profile_object(),
        "runtime": _aws_runtime_object(),
        "instance_profile_arn": (
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        "state_root": tmp_path / "state",
        "approval_verifier": lambda **_: {},
    }
    first = AwsP5Backend(runner=first_runner, **arguments)
    first_runner.outputs.extend(
        _aws_submit_outputs(
            first,
            release,
            manifest,
            ssm_online=False,
        )
    )

    with pytest.raises(Exception) as caught:
        first.submit(
            release=release,
            manifest=manifest,
            instance_id="i-0123456789abcdef0",
            terminate_at=_AWS_TERMINATE_AT,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "SSM_UNAVAILABLE"
    assert not list((tmp_path / "state" / "runs").glob("*.json"))
    assert all(
        "send-command" not in argv and "run-instances" not in argv
        for argv, _ in first_runner.calls
    )
    assert all(
        "modify-instance-attribute" not in argv and "create-tags" not in argv
        for argv, _ in first_runner.calls
    )


def test_aws_evaluate_and_cleanup_refuse_active_training(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    command_id = "cmd-0123456789abcdef0"

    for operation in ("evaluate", "cleanup"):
        state_root = tmp_path / f"{operation}-state"
        outputs = [
            {
                "instances": [
                    _selected_instance(manifest, bound=True)
                ]
            },
            {
                "attribute": {
                    "instance_id": "i-0123456789abcdef0",
                    "shutdown_behavior": "terminate",
                }
            },
        ]
        outputs.append(
            {
                "command": {
                    "command_id": command_id,
                    "status": "InProgress",
                }
            }
        )
        runner = _FakeAwsRunner(*outputs)
        backend = AwsP5Backend(
            profile=_aws_profile_object(),
            runtime=_aws_runtime_object(),
            instance_profile_arn=(
                "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
            ),
            state_root=state_root,
            runner=runner,
            approval_verifier=lambda **_: {},
        )
        store = StateStore(state_root)
        with store.locked():
            for run in manifest.runs:
                store.write_run(
                    run.run_id,
                    _aws_run_state(
                        backend,
                        release,
                        manifest,
                        run,
                        status="InProgress",
                        command_id=command_id,
                    ),
                )

        with pytest.raises(Exception) as caught:
            getattr(backend, operation)(
                release=release,
                manifest=manifest,
                approval_path=tmp_path / "approval.json",
                apply=True,
            )

        assert getattr(caught.value, "code", None) == "RUN_ALREADY_ACTIVE"
        assert not any("send-command" in argv for argv, _ in runner.calls)
        assert not any(
            "terminate-instances" in argv for argv, _ in runner.calls
        )


def test_aws_cancel_evaluate_and_cleanup_mutate_one_paired_state(tmp_path):
    import base64

    from msctl.aws_p5 import (
        ARGV_DOCUMENT_NAME,
        ARGV_DOCUMENT_SHA256,
        AwsP5Backend,
    )
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    instance = _bound_instance(manifest)
    command_id = "cmd-0123456789abcdef0"

    def seed_state(root: Path, backend: AwsP5Backend) -> None:
        store = StateStore(root)
        with store.locked():
            for run in manifest.runs:
                store.write_run(
                    run.run_id,
                    _aws_run_state(
                        backend,
                        release,
                        manifest,
                        run,
                        status="Success",
                        command_id=command_id,
                    ),
                )

    common = {
        "profile": _aws_profile_object(),
        "runtime": _aws_runtime_object(),
        "instance_profile_arn": instance["instance_profile_arn"],
        "approval_verifier": lambda **_: {},
    }

    cancel_root = tmp_path / "cancel-state"
    cancel_runner = _FakeAwsRunner(
        {
            "command": {
                "command_id": command_id,
                "status": "InProgress",
            }
        },
        {},
    )
    cancel_backend = AwsP5Backend(
        state_root=cancel_root,
        runner=cancel_runner,
        **common,
    )
    seed_state(cancel_root, cancel_backend)
    cancelled = cancel_backend.cancel(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )
    cancelled_again = cancel_backend.cancel(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )
    assert cancelled["cancelled"] == 1
    assert cancelled_again["cancelled"] == 0
    assert cancelled_again["idempotent"] is True
    assert sum(
        "cancel-command" in argv for argv, _ in cancel_runner.calls
    ) == 1

    evaluate_root = tmp_path / "evaluate-state"
    evaluate_runner = _FakeAwsRunner()
    evaluate_backend = AwsP5Backend(
        state_root=evaluate_root,
        runner=evaluate_runner,
        **common,
    )
    seed_state(evaluate_root, evaluate_backend)
    evaluation_intent = evaluate_backend._operation_envelope(
        evaluate_backend._evaluation_operation_intent(release, manifest),
        instance_id=instance["instance_id"],
        terminate_at=_AWS_TERMINATE_AT,
    )
    evaluation_payload = _canonical(evaluation_intent)
    evaluation_digest = hashlib.sha256(evaluation_payload).hexdigest()
    evaluation_checksum = base64.b64encode(
        bytes.fromhex(evaluation_digest)
    ).decode("ascii")
    evaluate_runner.outputs.extend(
        [
            {"instances": [_selected_instance(manifest, bound=True)]},
            {
                "attribute": {
                    "instance_id": instance["instance_id"],
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "command": {
                    "command_id": command_id,
                    "status": "Success",
                }
            },
            {
                "managed_instances": [
                    {
                        "instance_id": instance["instance_id"],
                        "ping_status": "Online",
                    }
                ]
            },
            {
                "documents": [
                    {
                        "name": ARGV_DOCUMENT_NAME,
                        "hash": ARGV_DOCUMENT_SHA256,
                        "status": "Active",
                    }
                ]
            },
            {
                "object": {
                    "checksum_sha256": evaluation_checksum,
                    "version_id": "evaluation-version",
                }
            },
            {
                "object": {
                    "checksum_sha256": evaluation_checksum,
                    "content_length": len(evaluation_payload),
                    "metadata": {
                        "operation-id": evaluation_intent["operation_id"],
                        "sha256": evaluation_digest,
                    },
                    "version_id": "evaluation-version",
                }
            },
            {"command": {"command_id": "cmd-evaluate-12345678"}},
        ]
    )
    evaluated = evaluate_backend.evaluate(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )
    assert evaluated["submitted"] == 1
    assert sum(
        "send-command" in argv for argv, _ in evaluate_runner.calls
    ) == 1

    cleanup_root = tmp_path / "cleanup-state"
    cleanup_runner = _FakeAwsRunner(
        {"instances": [_selected_instance(manifest, bound=True)]},
        {
            "attribute": {
                "instance_id": instance["instance_id"],
                "shutdown_behavior": "terminate",
            }
        },
        {
            "command": {
                "command_id": command_id,
                "status": "Cancelled",
            }
        },
        {
            "terminating_instances": [
                {
                    "instance_id": instance["instance_id"],
                    "current_state": "shutting-down",
                    "previous_state": "running",
                }
            ]
        },
    )
    cleanup_backend = AwsP5Backend(
        state_root=cleanup_root,
        runner=cleanup_runner,
        **common,
    )
    seed_state(cleanup_root, cleanup_backend)
    cleaned = cleanup_backend.cleanup(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )
    assert cleaned["terminated"] == 1
    assert sum(
        "terminate-instances" in argv for argv, _ in cleanup_runner.calls
    ) == 1


def test_aws_cleanup_binds_approval_and_revalidates_exact_selected_instance(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    command_id = "cmd-0123456789abcdef0"
    instance_id = "i-0123456789abcdef0"
    approvals = []
    runner = _FakeAwsRunner(
        {"instances": [_selected_instance(manifest, bound=True)]},
        {
            "attribute": {
                "instance_id": instance_id,
                "shutdown_behavior": "terminate",
            }
        },
        {
            "command": {
                "command_id": command_id,
                "status": "Cancelled",
            }
        },
        {
            "terminating_instances": [
                {
                    "instance_id": instance_id,
                    "current_state": "shutting-down",
                    "previous_state": "running",
                }
            ]
        },
    )

    def approve(**kwargs):
        approvals.append(kwargs)
        return {}

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=approve,
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for run in manifest.runs:
            store.write_run(
                run.run_id,
                _aws_run_state(
                    backend,
                    release,
                    manifest,
                    run,
                    status="Success",
                    command_id=command_id,
                ),
            )

    result = backend.cleanup(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "cleanup-approval.json",
        apply=True,
    )
    repeated = backend.cleanup(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "cleanup-approval.json",
        apply=True,
    )

    assert result["terminated"] == 1
    assert repeated["terminated"] == 0
    assert repeated["idempotent"] is True
    resources = approvals[0]["resources"]
    assert resources["instance_id"] == instance_id
    assert resources["run_manifest_sha256"] == manifest.sha256
    assert resources["release_sha256"] == release.archive_sha256
    assert resources["seed"] == manifest.seed
    assert resources["runtime_sha256"] == backend._runtime_sha256()
    assert next(
        index
        for index, (argv, _) in enumerate(runner.calls)
        if "terminate-instances" in argv
    ) > next(
        index
        for index, (_, operation) in enumerate(runner.calls)
        if operation == "verify cleanup instance"
    )


def _aws_checkpoint_receipt(tmp_path: Path, manifest, *, world_size: int = 4):
    rows = []
    for run in manifest.runs:
        checkpoint = tmp_path / "checkpoints" / f"{run.run_id}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{run.run_id}".encode())
        rows.append(
            {
                "run_id": run.run_id,
                "arm": run.arm,
                "seed": run.seed,
                "path": f"checkpoints/{checkpoint.name}",
                "sha256": _sha256(checkpoint),
                "config_sha256": run.config_sha256,
                "dataset_sha256": manifest.dataset_sha256,
                "source_commit": manifest.source_commit,
                "step": 400,
                "world_size": world_size,
            }
        )
    return _write_json(
        tmp_path / "aws-checkpoint-receipt.json",
        {
            "schema_version": 2,
            "provider": "aws-p5.48xlarge",
            "release_sha256": manifest.release_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_sha256": manifest.dataset_sha256,
            "source_commit": manifest.source_commit,
            "checkpoints": rows,
        },
    )


def test_aws_resume_requires_world_size_four_and_forwards_both_receipts(
    tmp_path,
):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.contracts import verify_checkpoint_receipt
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    receipt_path = _aws_checkpoint_receipt(tmp_path, manifest)
    receipt = verify_checkpoint_receipt(
        receipt_path,
        release=release,
        manifest=manifest,
    )
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    states = [
        _aws_run_state(
            backend,
            release,
            manifest,
            run,
            status="Success",
            command_id="training-command-12345678",
        )
        for run in manifest.runs
    ]
    store = StateStore(backend.state_root)
    with store.locked():
        backend._write_paired_states(store, manifest, states)

    planned = backend.resume(
        release=release,
        manifest=manifest,
        checkpoint_receipt=receipt,
        approval_path=None,
        apply=False,
    )

    intent = planned["operation_intent"]
    assert intent["operation"] == "resume"
    assert intent["checkpoint_receipt"]["sha256"] == receipt.sha256
    assert planned["approval_resources"]["instance_id"] == (
        "i-0123456789abcdef0"
    )
    assert planned["approval_resources"]["terminate_at"] == _AWS_TERMINATE_AT
    assert planned["approval_resources"]["checkpoint_receipt_sha256"] == (
        receipt.sha256
    )
    by_arm = {
        row["arm"]: row for row in intent["checkpoint_receipt"]["checkpoints"]
    }
    for checkpoint in receipt.checkpoints:
        assert by_arm[checkpoint.arm]["resume_path"].startswith(
            f"/mnt/memorysplit/staging/resume/{receipt.sha256}/"
        )
        assert by_arm[checkpoint.arm]["resume_sha256"] == checkpoint.sha256
        assert by_arm[checkpoint.arm]["world_size"] == 4
    names = [step["name"] for step in intent["steps"]]
    assert names == [
        "prepare-resume-staging",
        "materialize-resume-receipt",
        "materialize-resume-dense",
        "materialize-resume-split90",
        "paired-launch",
    ]
    launcher = next(
        step["argv"]
        for step in intent["steps"]
        if step["name"] == "paired-launch"
    )
    assert launcher[1].endswith("/msctl/aws_resume_launch.py")
    reviewed_launcher = (
        "/mnt/memorysplit/releases/"
        f"{release.archive_sha256}/cluster/aws/p5/launch_seed_pair.py"
    )
    assert launcher[launcher.index("--launcher") + 1] == reviewed_launcher
    checkpoint_arguments = [
        json.loads(launcher[index + 1])
        for index, value in enumerate(launcher)
        if value == "--checkpoint"
    ]
    assert checkpoint_arguments == [
        by_arm["dense"],
        by_arm["split90"],
    ]
    assert len(
        [argv for argv in planned["checkpoint_commands"] if "put-object" in argv]
    ) == 3
    assert len(
        [argv for argv in planned["checkpoint_commands"] if "head-object" in argv]
    ) == 3
    assert "torchrun" not in json.dumps(intent)

    invalid = _aws_checkpoint_receipt(tmp_path, manifest, world_size=3)
    with pytest.raises(Exception) as caught:
        verify_checkpoint_receipt(
            invalid,
            release=release,
            manifest=manifest,
        )
    assert getattr(caught.value, "code", None) == "CHECKPOINT_PROVENANCE_MISMATCH"

    receipt_path = _aws_checkpoint_receipt(tmp_path, manifest)
    mismatched_steps = json.loads(receipt_path.read_text())
    mismatched_steps["checkpoints"][1]["step"] += 1
    _write_json(receipt_path, mismatched_steps)
    with pytest.raises(Exception) as caught:
        verify_checkpoint_receipt(
            receipt_path,
            release=release,
            manifest=manifest,
        )
    assert getattr(caught.value, "code", None) == "CHECKPOINT_PROVENANCE_MISMATCH"


def test_aws_resume_adapter_injects_receipt_bound_checkpoints_into_real_launcher(
    tmp_path,
):
    from cluster.aws.p5 import launch_seed_pair
    from msctl.aws_resume_launch import bind_resume_checkpoints
    from tests.test_aws_p5_launcher import _launcher_fixture, _load_fixture_plan

    fixture = _launcher_fixture(tmp_path / "launcher", seed=2)
    plan = _load_fixture_plan(fixture)
    checkpoint_root = fixture["scratch_root"] / "staging" / "resume" / ("a" * 64)
    checkpoint_root.mkdir(parents=True)
    bindings = []
    for arm in ("dense", "split90"):
        checkpoint = checkpoint_root / f"{arm}.pt"
        checkpoint.write_bytes(f"checkpoint:{arm}".encode())
        bindings.append(
            {
                "arm": arm,
                "resume_path": str(checkpoint),
                "resume_sha256": _sha256(checkpoint),
                "world_size": 4,
            }
        )

    resumed = bind_resume_checkpoints(
        plan,
        checkpoint_receipt_sha256="a" * 64,
        checkpoints=bindings,
        launcher_module=launch_seed_pair,
    )

    assert {item.path for item in resumed.verified_files} >= {
        Path(row["resume_path"]) for row in bindings
    }
    by_arm = {launch.arm: launch for launch in resumed.arms}
    for binding in bindings:
        argv = by_arm[binding["arm"]].argv
        assert "--resume" not in argv
        assert argv[argv.index("--resume-path") + 1] == "/resume/checkpoint.pt"
        assert (
            argv[argv.index("--resume-sha256") + 1]
            == binding["resume_sha256"]
        )
        mount = (
            "type=bind,"
            f"src={binding['resume_path']},"
            "dst=/resume/checkpoint.pt,readonly"
        )
        assert mount in argv
        assert argv.index(mount) < argv.index(resumed.container_image)
        assert "--nproc_per_node=4" in argv


def test_aws_resume_adapter_archives_the_previous_pair_as_one_recoverable_root(
    tmp_path,
):
    from msctl.aws_resume_launch import prepare_resume_output_roots
    from tests.test_aws_p5_launcher import _launcher_fixture, _load_fixture_plan

    fixture = _launcher_fixture(tmp_path / "launcher", seed=3)
    seed_root = fixture["scratch_root"] / "runs" / "seed-3"
    for arm in ("dense", "split90"):
        output = seed_root / arm
        output.mkdir(parents=True)
        (output / "partial.log").write_text(f"{arm}\n")
    cid_root = (
        fixture["scratch_root"]
        / "staging"
        / "container-cids"
        / "seed-3"
    )
    cid_root.mkdir(parents=True)
    (cid_root / "dense.cid").write_text("a" * 64)

    archive = prepare_resume_output_roots(
        fixture["scratch_root"],
        seed=3,
        checkpoint_receipt_sha256="b" * 64,
    )
    repeated = prepare_resume_output_roots(
        fixture["scratch_root"],
        seed=3,
        checkpoint_receipt_sha256="b" * 64,
    )

    assert repeated == archive
    assert not seed_root.exists()
    assert (archive / "runs" / "dense" / "partial.log").is_file()
    assert (archive / "runs" / "split90" / "partial.log").is_file()
    assert (archive / "container-cids" / "dense.cid").is_file()
    assert _load_fixture_plan(fixture).seed == 3


def test_aws_resume_adapter_rejects_a_symlinked_archive_parent(tmp_path):
    from msctl.aws_resume_launch import (
        ResumeLaunchError,
        prepare_resume_output_roots,
    )

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (scratch / "resume-history").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ResumeLaunchError, match="archive"):
        prepare_resume_output_roots(
            scratch,
            seed=4,
            checkpoint_receipt_sha256="c" * 64,
        )

    assert list(outside.iterdir()) == []


def test_aws_resume_adapter_verifies_before_archiving_outputs(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import msctl.aws_resume_launch as adapter

    scratch = tmp_path / "scratch"
    source = scratch / "runs" / "seed-2"
    for arm in ("dense", "split90"):
        output = source / arm
        output.mkdir(parents=True)
        (output / "partial.log").write_text(f"{arm}\n")
    receipt = tmp_path / "checkpoint-receipt.json"
    receipt.write_text("{}")
    launcher = tmp_path / "launch_seed_pair.py"
    launcher.write_text("# fixture\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    repo_root = tmp_path / "release"
    repo_root.mkdir()
    monkeypatch.setattr(adapter, "_verify_launcher_path", lambda *_: None)
    monkeypatch.setattr(
        adapter.reviewed_launcher,
        "load_launch_plan",
        lambda **_: SimpleNamespace(),
    )

    result = adapter.main(
        [
            "--seed",
            "2",
            "--manifest",
            str(manifest),
            "--profile",
            str(profile),
            "--repo-root",
            str(repo_root),
            "--scratch-root",
            str(scratch),
            "--launcher",
            str(launcher),
            "--checkpoint-receipt",
            str(receipt),
            "--checkpoint-receipt-sha256",
            _sha256(receipt),
            "--run-manifest-sha256",
            "a" * 64,
            "--checkpoint",
            "{}",
            "--checkpoint",
            "{}",
            "--apply",
        ]
    )

    assert result == 2
    assert source.is_dir()
    assert not (scratch / "resume-history").exists()


def test_aws_resume_response_loss_reconciles_without_resending(tmp_path):
    import base64

    from msctl.aws_p5 import (
        ARGV_DOCUMENT_NAME,
        ARGV_DOCUMENT_SHA256,
        AwsP5Backend,
    )
    from msctl.contracts import verify_checkpoint_receipt
    from msctl.errors import MsctlError
    from msctl.jsonutil import canonical_json
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    checkpoint_receipt = verify_checkpoint_receipt(
        _aws_checkpoint_receipt(tmp_path, manifest),
        release=release,
        manifest=manifest,
    )
    command_id = "cmd-0123456789abcdef0"
    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=lambda **_: {},
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for run in manifest.runs:
            store.write_run(
                run.run_id,
                _aws_run_state(
                    backend,
                    release,
                    manifest,
                    run,
                    status="Failed",
                    command_id=command_id,
                ),
            )
    core = backend._training_operation_intent(
        operation="resume",
        release=release,
        manifest=manifest,
        checkpoints=backend._checkpoint_map(
            manifest,
            checkpoint_receipt,
        ),
        checkpoint_receipt_sha256=checkpoint_receipt.sha256,
    )
    intent = backend._operation_envelope(
        core,
        instance_id="i-0123456789abcdef0",
        terminate_at=_AWS_TERMINATE_AT,
    )
    payload = canonical_json(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    checkpoint_outputs = []
    checkpoint_plan = backend._checkpoint_publication(
        checkpoint_receipt=checkpoint_receipt,
        checkpoints=backend._checkpoint_map(manifest, checkpoint_receipt),
        apply=False,
    )
    for item in checkpoint_plan["objects"]:
        checkpoint_outputs.extend(
            [
                {},
                {
                    "object": {
                        "bytes": item["bytes"],
                        "checksum_sha256": base64.b64encode(
                            bytes.fromhex(item["sha256"])
                        ).decode("ascii"),
                        "metadata": {"sha256": item["sha256"]},
                        "version_id": "checkpoint-version-1",
                    }
                },
            ]
        )
    runner.outputs.extend(
        [
            {"instances": [_selected_instance(manifest, bound=True)]},
            {
                "attribute": {
                    "instance_id": "i-0123456789abcdef0",
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "command": {
                    "command_id": command_id,
                    "status": "Failed",
                }
            },
            {
                "managed_instances": [
                    {
                        "instance_id": "i-0123456789abcdef0",
                        "ping_status": "Online",
                    }
                ]
            },
                {
                    "documents": [
                        {
                            "name": ARGV_DOCUMENT_NAME,
                            "hash": ARGV_DOCUMENT_SHA256,
                            "status": "Active",
                        }
                    ]
                },
            *checkpoint_outputs,
            {
                "object": {
                    "checksum_sha256": checksum,
                    "version_id": "version-1",
                }
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "content_length": len(payload),
                    "metadata": {
                        "operation-id": intent["operation_id"],
                        "sha256": digest,
                    },
                    "version_id": "version-1",
                }
            },
            MsctlError("AWS_COMMAND_FAILED", "response lost"),
        ]
    )

    with pytest.raises(Exception) as caught:
        backend.resume(
            release=release,
            manifest=manifest,
            checkpoint_receipt=checkpoint_receipt,
            approval_path=tmp_path / "approval.json",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "AWS_COMMAND_FAILED"
    assert next(
        index
        for index, (_, operation) in enumerate(runner.calls)
        if operation == "verify SSM document"
    ) < next(
        index
        for index, (_, operation) in enumerate(runner.calls)
        if operation == "materialize checkpoint object"
    )

    retry_runner = _FakeAwsRunner(
        {"instances": [_selected_instance(manifest, bound=True)]},
        {
            "attribute": {
                "instance_id": "i-0123456789abcdef0",
                "shutdown_behavior": "terminate",
            }
        },
        {
            "commands": [
                {
                    "command_id": "cmd-resume-12345678",
                    "status": "InProgress",
                    "comment": intent["operation_id"],
                    "instance_ids": ["i-0123456789abcdef0"],
                }
            ]
        },
    )
    retry = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=retry_runner,
        approval_verifier=lambda **_: {},
    )
    result = retry.resume(
        release=release,
        manifest=manifest,
        checkpoint_receipt=checkpoint_receipt,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )

    assert result["submitted"] == 0
    assert result["idempotent"] is True
    assert result["command_id"] == "cmd-resume-12345678"
    assert all(
        "send-command" not in argv for argv, _ in retry_runner.calls
    )


def test_aws_evaluation_uses_canonical_confirmatory_runner_interface(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    states = [
        _aws_run_state(
            backend,
            release,
            manifest,
            run,
            status="Success",
            command_id="training-command-12345678",
        )
        for run in manifest.runs
    ]
    store = StateStore(backend.state_root)
    with store.locked():
        backend._write_paired_states(store, manifest, states)

    planned = backend.evaluate(
        release=release,
        manifest=manifest,
        approval_path=None,
        apply=False,
    )
    assert planned["approval_resources"]["instance_id"] == (
        "i-0123456789abcdef0"
    )
    assert planned["approval_resources"]["terminate_at"] == _AWS_TERMINATE_AT

    commands = planned["operation_intent"]["steps"]
    assert len(commands) == 2
    for step in commands:
        argv = step["argv"]
        image_index = argv.index(_aws_runtime_object().container_image)
        assert argv[:2] == ["/usr/bin/docker", "run"]
        assert argv[image_index + 1 : image_index + 5] == [
            "/opt/venv/bin/python",
            "-m",
            "evals.confirmatory",
            "evaluate",
        ]
        assert "--run" in argv
        assert "--sealed-release" in argv
        assert "--expected-study-lock-sha256" in argv
        assert "--device" in argv
        assert "--output-dir" in argv
        assert "--run-id" not in argv
        assert "--run-manifest-sha256" not in argv


def test_aws_evaluation_uses_one_content_addressed_fixed_document_intent(
    tmp_path,
):
    import base64

    from msctl.aws_p5 import (
        ARGV_DOCUMENT_NAME,
        ARGV_DOCUMENT_SHA256,
        AwsP5Backend,
    )
    from msctl.jsonutil import canonical_json
    from msctl.state import StateStore

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    instance_id = "i-0123456789abcdef0"
    terminate_at = _AWS_TERMINATE_AT
    training_command = "cmd-0123456789abcdef0"
    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
        approval_verifier=lambda **_: {},
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        for run in manifest.runs:
            store.write_run(
                run.run_id,
                _aws_run_state(
                    backend,
                    release,
                    manifest,
                    run,
                    status="Success",
                    command_id=training_command,
                ),
            )
    core = backend._evaluation_operation_intent(release, manifest)
    intent = backend._operation_envelope(
        core,
        instance_id=instance_id,
        terminate_at=terminate_at,
    )
    payload = canonical_json(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    runner.outputs.extend(
        [
            {"instances": [_selected_instance(manifest, bound=True)]},
            {
                "attribute": {
                    "instance_id": instance_id,
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "command": {
                    "command_id": training_command,
                    "status": "Success",
                }
            },
            {
                "managed_instances": [
                    {
                        "instance_id": instance_id,
                        "ping_status": "Online",
                    }
                ]
            },
            {
                "documents": [
                    {
                        "name": ARGV_DOCUMENT_NAME,
                        "hash": ARGV_DOCUMENT_SHA256,
                        "status": "Active",
                    }
                ]
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "version_id": "evaluation-intent-version",
                }
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "content_length": len(payload),
                    "metadata": {
                        "operation-id": intent["operation_id"],
                        "sha256": digest,
                    },
                    "version_id": "evaluation-intent-version",
                }
            },
            {"command": {"command_id": "cmd-evaluate-12345678"}},
            {"instances": [_selected_instance(manifest, bound=True)]},
            {
                "attribute": {
                    "instance_id": instance_id,
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "command": {
                    "command_id": training_command,
                    "status": "Success",
                }
            },
            {
                "command": {
                    "command_id": "cmd-evaluate-12345678",
                    "status": "Success",
                }
            },
        ]
    )

    first = backend.evaluate(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )
    second = backend.evaluate(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )

    assert first["submitted"] == 1
    assert second["submitted"] == 0
    assert second["idempotent"] is True
    send_calls = [
        argv for argv, _ in runner.calls if "send-command" in argv
    ]
    assert len(send_calls) == 1
    send = send_calls[0]
    assert send[send.index("--document-name") + 1] == ARGV_DOCUMENT_NAME
    assert send[send.index("--document-hash") + 1] == ARGV_DOCUMENT_SHA256
    assert send[send.index("--comment") + 1] == intent["operation_id"]
    assert "IntentUri" in send[send.index("--parameters") + 1]
    assert "argv" not in send[send.index("--parameters") + 1]


def test_every_paid_aws_mutation_requires_approval_before_runner(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.contracts import verify_checkpoint_receipt

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    receipt = verify_checkpoint_receipt(
        _aws_checkpoint_receipt(tmp_path, manifest),
        release=release,
        manifest=manifest,
    )
    runner = _FakeAwsRunner()
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )

    mutations = (
        lambda: backend.resume(
            release=release,
            manifest=manifest,
            checkpoint_receipt=receipt,
            approval_path=None,
            apply=True,
        ),
        lambda: backend.cancel(
            release=release,
            manifest=manifest,
            approval_path=None,
            apply=True,
        ),
        lambda: backend.evaluate(
            release=release,
            manifest=manifest,
            approval_path=None,
            apply=True,
        ),
        lambda: backend.cleanup(
            release=release,
            manifest=manifest,
            approval_path=None,
            apply=True,
        ),
    )
    for mutate in mutations:
        with pytest.raises(Exception) as caught:
            mutate()
        assert getattr(caught.value, "code", None) == "APPROVAL_REQUIRED"
    assert runner.calls == []
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
    release_value = json.loads(release.read_text())
    assert (
        f"MS_RELEASE_ARCHIVE={tmp_path / release_value['archive']['path']}"
        in export
    )
    assert "MS_JOB_SCRIPT_REL=cluster/slurm/v2_seed0.sbatch" in export
    assert "MS_RELEASE_ROOT=" not in export
    assert f"MS_ENV_ROOT={(tmp_path / 'environment').resolve()}" in export
    assert "/attacker" not in export
    assert "--chdir=/" in command


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


def test_paid_submission_sends_only_fixed_bootstrap_on_stdin(tmp_path):
    from msctl.slurm import BOOTSTRAP_SHA256

    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "b" * 32
    approval = _approval(
        tmp_path,
        operation="submit",
        runs=runs,
        key=key,
    )
    captured = tmp_path / "captured-bootstrap.py"
    argv = tmp_path / "captured-argv"
    marker = tmp_path / "untrusted-script-executed"
    shared_script = tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "sbatch",
        f"printf '#!/bin/sh\\ntouch {marker}\\n' > '{shared_script}'\n"
        f"cat > '{captured}'\nprintf '%s\\n' \"$@\" > '{argv}'\n"
        "last=''\nfor item in \"$@\"; do last=\"$item\"; done\n"
        "if [ -f \"$last\" ]; then \"$last\"; fi\n"
        "printf '777;usfc-prd\\n'\n",
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

    assert completed.returncode == 0
    assert hashlib.sha256(captured.read_bytes()).hexdigest() == BOOTSTRAP_SHA256
    submitted_argv = argv.read_text()
    assert "MS_RELEASE_ARCHIVE_SHA256=" in submitted_argv
    assert "MS_RELEASE_MEMBERS_SHA256=" in submitted_argv
    assert str(tmp_path / "cluster" / "slurm" / "v2_seed0.sbatch") not in (
        submitted_argv
    )
    assert "--chdir=/" in submitted_argv
    assert f"--chdir={tmp_path}" not in submitted_argv
    assert not marker.exists()


def test_bootstrap_rejects_archive_replacement_without_executing_payload(
    tmp_path,
):
    from msctl.slurm import BOOTSTRAP_PAYLOAD

    release_path = _release(tmp_path)
    release = json.loads(release_path.read_text())
    archive = tmp_path / release["archive"]["path"]
    marker = tmp_path / "unauthenticated-code-executed"
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as handle:
        handle.writestr(
            "cluster/slurm/v2_seed0.sbatch",
            f"#!/bin/sh\ntouch '{marker}'\n",
        )
    archive.write_bytes(malicious.read_bytes())
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_bytes(BOOTSTRAP_PAYLOAD)
    (tmp_path / "node-local").mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_TMPDIR": str(tmp_path / "node-local"),
        "MS_RELEASE_ARCHIVE": str(archive),
        "MS_RELEASE_ARCHIVE_SHA256": release["archive"]["sha256"],
        "MS_RELEASE_ARCHIVE_BYTES": str(release["archive"]["bytes"]),
        "MS_RELEASE_MEMBERS_SHA256": release["members_sha256"],
        "MS_JOB_SCRIPT_REL": "cluster/slurm/v2_seed0.sbatch",
        "MS_SHARED_ROOT_PREFIX": str(tmp_path / "publication"),
        "MS_SHARED_ROOT": str(tmp_path / "publication"),
        "MS_DATA_RELATIVE_PATH": "memorysplit/datasets/v2-20x-seed0",
        "MS_DATA_ROOT": str(_dataset_fixture(tmp_path)[1]),
    }

    completed = subprocess.run(
        [sys.executable, str(bootstrap)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not marker.exists()


def test_bootstrap_executes_only_authenticated_node_local_member(tmp_path):
    from msctl.slurm import BOOTSTRAP_PAYLOAD

    marker = tmp_path / "authenticated-code-executed"
    job_name = "cluster/slurm/job.sbatch"
    job = (
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(os.environ['MS_RELEASE_ROOT'])\n"
    ).encode()
    sums = f"{hashlib.sha256(job).hexdigest()}  {job_name}\n".encode()
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, data in ((job_name, job), ("SHA256SUMS", sums)):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            handle.writestr(info, data)
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_bytes(BOOTSTRAP_PAYLOAD)
    node_local = tmp_path / "node-local"
    node_local.mkdir()
    _, dataset_root = _dataset_fixture(tmp_path)
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_TMPDIR": str(node_local),
        "MS_RELEASE_ARCHIVE": str(archive),
        "MS_RELEASE_ARCHIVE_SHA256": _sha256(archive),
        "MS_RELEASE_ARCHIVE_BYTES": str(archive.stat().st_size),
        "MS_RELEASE_MEMBERS_SHA256": hashlib.sha256(sums).hexdigest(),
        "MS_JOB_SCRIPT_REL": job_name,
        "MS_SHARED_ROOT_PREFIX": str(tmp_path),
        "MS_SHARED_ROOT": str(_test_shared_root(tmp_path)),
        "MS_DATA_RELATIVE_PATH": "memorysplit/datasets/v2-20x-seed0",
        "MS_DATA_ROOT": str(dataset_root),
    }

    completed = subprocess.run(
        [sys.executable, str(bootstrap)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    executed_root = Path(marker.read_text())
    assert executed_root == (
        node_local / "memorysplit-release" / _sha256(archive)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "shell_stub",
        "shell_stub_recommitted",
        "changed_runtime",
        "changed_receipt",
    ],
)
def test_environment_runtime_mutation_fails_before_sbatch(tmp_path, mutation):
    from msctl.environment import measure_environment

    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    receipt = _environment_receipt(tmp_path)
    python = receipt.parent / "bin" / "python"
    if mutation in {"shell_stub", "shell_stub_recommitted"}:
        python.write_text("#!/bin/sh\nprintf '{}\\n'\n")
        python.chmod(0o755)
        if mutation == "shell_stub_recommitted":
            value = json.loads(receipt.read_text())
            value.update(measure_environment(receipt.parent))
            _write_json(receipt, value)
    else:
        if mutation == "changed_runtime":
            with python.open("ab") as handle:
                handle.write(b"runtime replacement")
        else:
            value = json.loads(receipt.read_text())
            value["environment_tree_merkle_sha256"] = "0" * 64
            _write_json(receipt, value)
    key = "m" * 32
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
        "--environment-receipt",
        str(receipt),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
        bind_environment=False,
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == (
        "ENV_RUNTIME_MISMATCH"
    )
    assert not marker.exists()


@pytest.mark.parametrize("shared_root_kind", ["alternate", "outside_prefix"])
def test_dataset_rejects_alternate_approved_root_before_sbatch(
    tmp_path,
    shared_root_kind,
):
    release = _release(tmp_path)
    manifest, runs = _runs(tmp_path)
    key = "d" * 32
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

    shared_root = (
        tmp_path / "alternate-publication"
        if shared_root_kind == "alternate"
        else Path("/alternate-publication")
    )
    completed = _run_msctl(
        *_base_args(tmp_path),
        "submit",
        "--release",
        str(release),
        "--manifest",
        str(manifest),
        "--shared-root",
        str(shared_root),
        "--approval",
        str(approval),
        "--apply",
        env={"PATH": str(bin_dir), "MSCTL_APPROVAL_KEY": key},
    )

    assert completed.returncode != 0
    assert _single_report(completed)["error"]["code"] == "DATASET_ROOT_MISMATCH"
    assert not marker.exists()


def test_dataset_pointer_rejects_relative_path_traversal(tmp_path):
    from dataclasses import replace

    from msctl.dataset import load_pointer
    from msctl.errors import MsctlError
    from msctl.profile import load_profile

    pointer, _ = _dataset_fixture(tmp_path)
    value = json.loads(pointer.read_text())
    value["relative_path"] = "../outside"
    _write_json(pointer, value)
    profile = replace(
        load_profile(_test_profile(tmp_path)),
        shared_root_prefix=str(tmp_path.resolve()),
    )

    with pytest.raises(MsctlError):
        load_pointer(pointer, profile)
