from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiment.ledger import RunLedger
from scripts.freeze_relational_study import make_fixture_freeze
from scripts.make_relational_manifest import build_manifest, publish_run_configs
from scripts.run_train import resolve_relative_config
from tests.relational_asset_fixtures import stage_launchable_relational_assets
from train.trainer import validate_run_start

import cluster.aws.run_relational_manifest as aws_launcher


REPO_ROOT = Path(__file__).resolve().parents[1]
FARMSHARE = REPO_ROOT / "cluster" / "farmshare" / "submit_relational_manifest.sh"


def _published_manifest(tmp_path: Path, *, launchable: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if launchable:
        _freeze, _receipt, manifest, data_root = (
            stage_launchable_relational_assets(tmp_path / "assets")
        )
    else:
        manifest = build_manifest(make_fixture_freeze())
        data_root = None
    published = publish_run_configs(tmp_path / "published", manifest)
    return manifest, published / "run-manifest.json", data_root


def test_farmshare_launcher_defaults_to_side_effect_free_dry_run(tmp_path):
    manifest, path, _ = _published_manifest(tmp_path)
    marker = tmp_path / "sbatch-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        f"printf called > {marker}\n"
        "exit 99\n"
    )
    fake_sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON": sys.executable,
        "PYTHONPATH": str(REPO_ROOT),
    }

    result = subprocess.run(
        [str(FARMSHARE), str(path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.count("sbatch ") == 35
    assert "--gres=gpu:L40S:1" in result.stdout
    assert not marker.exists()

    refused = subprocess.run(
        [
            str(FARMSHARE),
            "--submit",
            "--ack-freeze",
            manifest.freeze_sha256,
            str(path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "nonlaunchable fixture" in refused.stderr
    assert not marker.exists()


class _FakeRunner:
    def __init__(self, failed_index: int | None = None):
        self.failed_index = failed_index
        self.commands: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, command, *, env, cwd):
        index = len(self.commands)
        self.commands.append((list(command), dict(env)))
        return subprocess.CompletedProcess(
            command,
            7 if index == self.failed_index else 0,
        )


def _mutate(path: Path) -> None:
    changed = bytearray(path.read_bytes())
    changed[0] ^= 1
    path.write_bytes(changed)

@pytest.mark.parametrize(
    ("asset", "match"),
    [
        ("stream", "stream SHA-256"),
        ("weights", "weights SHA-256"),
    ],
)
def test_farmshare_submit_rechecks_assets_before_sbatch_or_ledger(
    tmp_path,
    asset,
    match,
):
    manifest, path, data_root = _published_manifest(
        tmp_path / "publication",
        launchable=True,
    )
    out_root = tmp_path / "out"
    out_root.mkdir()
    run = manifest.runs[0]
    token_path = data_root / run.data_rel
    weights_path = data_root / run.weights_rel
    resolved = resolve_relative_config(
        run.to_dict(),
        run_manifest=manifest,
        environ={"DATA_ROOT": str(data_root), "OUT_ROOT": str(out_root)},
    )
    validate_run_start(resolved, resume="none")
    _mutate(token_path if asset == "stream" else weights_path)
    marker = tmp_path / "sbatch-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        f"printf called >> {marker}\n"
        "exit 0\n"
    )
    fake_sbatch.chmod(0o755)
    env = {
        **os.environ,
        "DATA_ROOT": str(data_root),
        "OUT_ROOT": str(out_root),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON": sys.executable,
        "PYTHONPATH": str(REPO_ROOT),
    }

    result = subprocess.run(
        [
            str(FARMSHARE),
            "--submit",
            "--ack-freeze",
            manifest.freeze_sha256,
            str(path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert match in result.stderr
    assert not marker.exists()
    assert list(out_root.iterdir()) == []


@pytest.mark.parametrize(
    ("asset", "match"),
    [
        ("stream", "stream SHA-256"),
        ("weights", "weights SHA-256"),
    ],
)
def test_aws_execute_rechecks_assets_after_preflight_before_runner_or_ledger(
    tmp_path,
    capsys,
    asset,
    match,
):
    manifest, path, data_root = _published_manifest(
        tmp_path / "publication",
        launchable=True,
    )
    out_root = tmp_path / "out"
    out_root.mkdir()
    run = manifest.runs[0]
    token_path = data_root / run.data_rel
    weights_path = data_root / run.weights_rel
    resolved = resolve_relative_config(
        run.to_dict(),
        run_manifest=manifest,
        environ={"DATA_ROOT": str(data_root), "OUT_ROOT": str(out_root)},
    )
    validate_run_start(resolved, resume="none")
    selected = token_path if asset == "stream" else weights_path
    runner = _FakeRunner()

    def passing_preflight_then_mutate(**_kwargs):
        _mutate(selected)
        return {"passed": True, "gpu_count": 1}

    status = aws_launcher.main(
        [
            "--execute",
            "--ack-freeze",
            manifest.freeze_sha256,
            "--gpu-count",
            "1",
            "--capacity-mode",
            "on-demand",
            "--data-root",
            str(data_root),
            "--out-root",
            str(out_root),
            str(path),
        ],
        runner=runner,
        preflight_runner=passing_preflight_then_mutate,
    )
    captured = capsys.readouterr()

    assert status != 0
    assert match in captured.err
    assert runner.commands == []
    assert list(out_root.iterdir()) == []


def test_receipt_finalized_manifest_passes_both_fake_launchers(tmp_path):
    manifest, path, data_root = _published_manifest(
        tmp_path / "publication",
        launchable=True,
    )
    farmshare_out = tmp_path / "farmshare-out"
    farmshare_out.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "sbatch-called"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        f"printf 'called\\n' >> {marker}\n"
        "exit 0\n"
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "DATA_ROOT": str(data_root),
        "OUT_ROOT": str(farmshare_out),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON": sys.executable,
        "PYTHONPATH": str(REPO_ROOT),
    }

    farmshare = subprocess.run(
        [
            str(FARMSHARE),
            "--submit",
            "--ack-freeze",
            manifest.freeze_sha256,
            str(path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert farmshare.returncode == 0, farmshare.stderr
    assert marker.read_text().splitlines() == ["called"] * 35

    aws_out = tmp_path / "aws-out"
    aws_out.mkdir()
    runner = _FakeRunner()
    status = aws_launcher.main(
        [
            "--execute",
            "--ack-freeze",
            manifest.freeze_sha256,
            "--gpu-count",
            "1",
            "--capacity-mode",
            "on-demand",
            "--data-root",
            str(data_root),
            "--out-root",
            str(aws_out),
            str(path),
        ],
        runner=runner,
        preflight_runner=lambda **_kwargs: {"passed": True, "gpu_count": 1},
    )
    assert status == 0
    assert len(runner.commands) == 35


def test_aws_gpu_count_is_an_explicit_sequential_assignment(
    tmp_path,
    capsys,
):
    _, path, _ = _published_manifest(tmp_path)

    status = aws_launcher.main(
        ["--dry-run", "--gpu-count", "2", str(path)]
    )
    output = capsys.readouterr().out.splitlines()
    help_text = aws_launcher._parser().format_help()

    assert status == 0
    assert [line.split("=", 1)[1].split(" ", 1)[0] for line in output[:4]] == [
        "0",
        "1",
        "0",
        "1",
    ]
    assert "--gpu-count" in help_text
    assert "--workers" not in help_text
    assert "sequential" in help_text.lower()


def test_aws_launcher_defaults_to_dry_run_and_refuses_fixtures(
    tmp_path,
    capsys,
):
    manifest, path, _ = _published_manifest(tmp_path)
    runner = _FakeRunner()

    status = aws_launcher.main(["--dry-run", str(path)], runner=runner)
    output = capsys.readouterr().out

    assert status == 0
    assert output.count("CUDA_VISIBLE_DEVICES=") == 35
    assert runner.commands == []

    status = aws_launcher.main(
        [
            "--execute",
            "--ack-freeze",
            manifest.freeze_sha256,
            str(path),
        ],
        runner=runner,
        preflight_runner=lambda **kwargs: {"passed": True, "gpu_count": 1},
    )
    assert status != 0
    assert runner.commands == []


def test_aws_execution_requires_data_root_with_injected_preflight(tmp_path):
    manifest, path, _ = _published_manifest(tmp_path, launchable=True)
    out_root = tmp_path / "out"
    out_root.mkdir()
    runner = _FakeRunner()

    status = aws_launcher.main(
        [
            "--execute",
            "--ack-freeze",
            manifest.freeze_sha256,
            "--capacity-mode",
            "on-demand",
            "--out-root",
            str(out_root),
            str(path),
        ],
        runner=runner,
        preflight_runner=lambda **kwargs: {"passed": True, "gpu_count": 1},
    )

    assert status != 0
    assert runner.commands == []
    assert list(out_root.iterdir()) == []


def test_aws_execution_requires_exact_ack_retains_failures_and_continues(
    tmp_path,
):
    manifest, path, data_root = _published_manifest(
        tmp_path,
        launchable=True,
    )
    out_root = tmp_path / "out"
    out_root.mkdir()
    wrong_runner = _FakeRunner()

    wrong = aws_launcher.main(
        [
            "--execute",
            "--ack-freeze",
            "0" * 64,
            "--out-root",
            str(out_root),
            str(path),
        ],
        runner=wrong_runner,
        preflight_runner=lambda **kwargs: {"passed": True, "gpu_count": 1},
    )
    assert wrong != 0
    assert wrong_runner.commands == []

    runner = _FakeRunner(failed_index=3)
    status = aws_launcher.main(
        [
            "--execute",
            "--ack-freeze",
            manifest.freeze_sha256,
            "--gpu-count",
            "2",
            "--capacity-mode",
            "capacity-block",
            "--data-root",
            str(data_root),
            "--out-root",
            str(out_root),
            str(path),
        ],
        runner=runner,
        preflight_runner=lambda **kwargs: {"passed": True, "gpu_count": 2},
    )

    assert status != 0
    assert len(runner.commands) == 35
    assert {entry[1]["CUDA_VISIBLE_DEVICES"] for entry in runner.commands} == {
        "0",
        "1",
    }
    failed_run = manifest.runs[3]
    events = RunLedger(out_root, failed_run.run_id).events()
    assert [event.event_type for event in events] == [
        "planned",
        "preflight_passed",
        "launch_requested",
        "failed",
    ]
