from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

import scripts.platform_preflight as preflight


REPO_ROOT = Path(__file__).resolve().parents[1]


def _passing_smoke_report():
    return {
        "bundle_byte_deterministic": True,
        "bundle_verified": True,
        "controls": [
            "correct",
            "shuffled_returns",
            "relevant_edge",
            "irrelevant_edge",
            "gold_path",
            "gold_returns",
            "no_query",
            "explicit_miss",
            "handle_swap",
            "entity_rename",
            "graph_isomorphism",
        ],
        "corpus_builds": 2,
        "corpus_byte_deterministic": True,
        "corpus_sha256": "1" * 64,
        "dense_steps": 2,
        "eval_cells": 22,
        "extracted_bundle_verified": True,
        "matrix_runs": 35,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
        "resume_compared_next_update": True,
        "resume_exact": True,
        "schemas_validated": [
            "freeze-v1.schema.json",
            "relational-asset-receipt-v1.schema.json",
            "relational-result-v1.schema.json",
            "run-config-v1.schema.json",
            "run-manifest-v1.schema.json",
        ],
        "sidecar_sha256": {
            label: str(index) * 64
            for index, label in enumerate(
                ("dense", "random", "selective", "split"),
                start=2,
            )
        },
        "sidecars": ["dense", "random", "selective", "split"],
        "shared_stream": True,
        "split_steps": 2,
        "synthetic_run_count": 35,
        "verdict_branches": [
            "validated",
            "practical_null",
            "inconclusive",
            "invalid",
        ],
    }


def test_local_preflight_checks_lock_imports_bundle_roots_and_smoke(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "relational-run.tar.gz"
    bundle.write_bytes(b"fixture")
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    asset = data_root / "fixture" / "train.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"hash-verified external corpus")
    build_manifest = asset.parent / "manifest.json"
    build_manifest.write_bytes(b'{"build":"frozen"}\n')
    asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
    monkeypatch.setattr(
        preflight,
        "_verify_bundle",
        lambda *args, **kwargs: {
            "verified": True,
            "external_assets": [
                {
                    "kind": "corpus",
                    "path": "fixture/train.bin",
                    "commitment_sha256": "1" * 64,
                    "sha256": asset_sha256,
                    "bytes": asset.stat().st_size,
                    "build_manifest_sha256": hashlib.sha256(
                        build_manifest.read_bytes()
                    ).hexdigest(),
                }
            ],
        },
    )
    monkeypatch.setattr(
        preflight,
        "_required_import_versions",
        lambda: {
            "datasets": "5.0.0",
            "huggingface-hub": "1.24.0",
            "matplotlib": "3.11.0",
            "numpy": "2.5.1",
            "pytest": "9.1.1",
            "pyyaml": "6.0.3",
            "tiktoken": "0.13.0",
            "torch": "2.13.0",
            "tqdm": "4.69.0",
        },
    )
    monkeypatch.setattr(
        preflight,
        "_run_smoke",
        lambda *args, **kwargs: _passing_smoke_report(),
    )

    report = preflight.run_preflight(
        "local",
        bundle=bundle,
        data_root=data_root,
        out_root=out_root,
        requirements_root=REPO_ROOT / "requirements",
        python_version=(3, 12, 0),
        system="Darwin",
        machine="arm64",
        run_smoke_check=True,
    )

    assert report["passed"] is True
    assert report["platform"] == "local"
    assert report["lock"].endswith("macos-arm64-py312.lock")
    assert report["checks"]["imports"]["torch"] == "2.13.0"
    assert report["checks"]["bundle"]["verified"] is True
    assert report["checks"]["external_assets"] == 1
    assert report["checks"]["smoke"]["resume_exact"] is True
    assert report["checks"]["roots"]["data_writable"] is True
    assert report["checks"]["roots"]["out_writable"] is True


def test_external_asset_preflight_rechecks_bound_build_manifest(tmp_path):
    data_root = tmp_path / "data"
    asset = data_root / "relational" / "build" / "train.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"stream")
    build_manifest = asset.parent / "manifest.json"
    build_manifest.write_bytes(b'{"build":"frozen"}\n')
    inventory = [
        {
            "kind": "corpus",
            "path": "relational/build/train.bin",
            "commitment_sha256": "1" * 64,
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "bytes": asset.stat().st_size,
            "build_manifest_sha256": hashlib.sha256(
                build_manifest.read_bytes()
            ).hexdigest(),
        }
    ]
    assert preflight._validate_external_assets(data_root, inventory) == 1
    build_manifest.write_bytes(b'{"build":"mutated"}\n')

    with pytest.raises(preflight.PreflightError, match="build manifest"):
        preflight._validate_external_assets(data_root, inventory)


def test_farmshare_batch_requests_exactly_one_l40s_and_resume(tmp_path):
    report = preflight.validate_farmshare_batch_script(
        REPO_ROOT / "cluster" / "slurm" / "relational_train.sbatch"
    )

    assert report == {
        "gpu_type": "L40S",
        "gpu_count": 1,
        "nodes": 1,
        "tasks": 1,
        "resume": "auto",
    }

    bad = tmp_path / "bad.sbatch"
    bad.write_text(
        "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --gres=gpu:L40S:2\n"
    )
    with pytest.raises(preflight.PreflightError, match="one L40S|resume|node"):
        preflight.validate_farmshare_batch_script(bad)


def test_farmshare_preflight_requires_slurm_commands_and_writable_scratch(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"fixture")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(
        preflight,
        "_verify_bundle",
        lambda *args, **kwargs: {"verified": True},
    )
    commands = {"sbatch", "sinfo", "squeue", "srun"}

    report = preflight.run_preflight(
        "farmshare",
        bundle=bundle,
        scratch=scratch,
        requirements_root=REPO_ROOT / "requirements",
        python_version=(3, 12, 0),
        command_lookup=lambda name: f"/usr/bin/{name}" if name in commands else None,
        smoke_report=_passing_smoke_report(),
    )
    assert report["passed"] is True
    assert report["checks"]["slurm"]["gpu_count"] == 1

    with pytest.raises(preflight.PreflightError, match="squeue"):
        preflight.run_preflight(
            "farmshare",
            bundle=bundle,
            scratch=scratch,
            requirements_root=REPO_ROOT / "requirements",
            python_version=(3, 12, 0),
            command_lookup=lambda name: (
                f"/usr/bin/{name}" if name != "squeue" else None
            ),
            smoke_report=_passing_smoke_report(),
        )


@pytest.mark.parametrize(
    "inventory",
    [
        "NVIDIA H100 80GB HBM3, 81559\nNVIDIA L40S, 46068\n",
        "NVIDIA A100-SXM4-80GB, 81920\n",
        "NVIDIA H100 80GB HBM3, 70000\n",
    ],
)
def test_aws_gpu_inventory_rejects_mixed_non_h100_or_too_small(inventory):
    with pytest.raises(preflight.PreflightError, match="H100|homogeneous|80"):
        preflight.parse_h100_inventory(inventory)


def test_aws_preflight_checks_capacity_h100_memory_disk_and_gpu_count(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"fixture")
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    monkeypatch.setattr(
        preflight,
        "_verify_bundle",
        lambda *args, **kwargs: {"verified": True},
    )
    disk = shutil.disk_usage(tmp_path)

    report = preflight.run_preflight(
        "aws",
        bundle=bundle,
        data_root=data_root,
        out_root=out_root,
        requirements_root=REPO_ROOT / "requirements",
        python_version=(3, 12, 0),
        capacity_mode="on-demand",
        expected_gpus=2,
        command_output=lambda command: (
            "NVIDIA H100 80GB HBM3, 81559\n"
            "NVIDIA H100 80GB HBM3, 81559\n"
        ),
        total_memory_bytes=256 * 1024**3,
        disk_usage=lambda path: disk._replace(free=2 * 1024**4),
        smoke_report=_passing_smoke_report(),
    )

    assert report["passed"] is True
    assert report["capacity_mode"] == "on-demand"
    assert report["checks"]["gpus"]["count"] == 2
    assert report["checks"]["expected_gpus"] == 2

    with pytest.raises(preflight.PreflightError, match="capacity"):
        preflight.run_preflight(
            "aws",
            bundle=bundle,
            data_root=data_root,
            out_root=out_root,
            requirements_root=REPO_ROOT / "requirements",
            python_version=(3, 12, 0),
            capacity_mode=None,
            expected_gpus=2,
            command_output=lambda command: (
                "NVIDIA H100 80GB HBM3, 81559\n" * 2
            ),
            total_memory_bytes=256 * 1024**3,
            disk_usage=lambda path: disk,
            smoke_report=_passing_smoke_report(),
        )


def test_preflight_rejects_partial_smoke_evidence():
    with pytest.raises(preflight.PreflightError, match="smoke"):
        preflight._validate_smoke_report(
            {"resume_exact": True, "bundle_verified": True}
        )
