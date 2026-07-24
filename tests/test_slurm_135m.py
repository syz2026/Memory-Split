from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import msctl.operations as operations
from msctl.adapters.slurm import plan_sbatch
from msctl.cohort import COHORT_ID
from msctl.operations import inspect_paired_resume, submit
from msctl.profile import load_profile
from scripts.run_135m_pair import _write_pair_checkpoint_receipt


ROOT = Path(__file__).resolve().parents[1]
FARM_PROFILE = ROOT / "cluster" / "profiles" / "farmshare-l40s.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_manifest(tmp_path: Path, profile_sha: str, seed: int = 0) -> Path:
    manifest = {
        "arms": [
            {
                "arm": "dense",
                "config_path": f"configs/135m-v2/dense-s{seed}.yaml",
                "config_sha256": "1" * 64,
                "out_dir": str(tmp_path / "dense"),
                "run_id": f"d135m_dense_full_s{seed}",
                "runtime_config": str(tmp_path / "dense.yaml"),
                "runtime_config_sha256": "2" * 64,
            },
            {
                "arm": "split90",
                "config_path": f"configs/135m-v2/split90-s{seed}.yaml",
                "config_sha256": "3" * 64,
                "out_dir": str(tmp_path / "split90"),
                "run_id": f"d135m_split90_full_s{seed}",
                "runtime_config": str(tmp_path / "split90.yaml"),
                "runtime_config_sha256": "4" * 64,
            },
        ],
        "cohort_id": COHORT_ID,
        "dataset": {
            "ordered_token_stream_sha256": "5" * 64,
            "receipt_sha256": "6" * 64,
        },
        "operator": "farmshare-lead",
        "pair_id": f"d135m_full_s{seed}",
        "profile_sha256": profile_sha,
        "provider": "farmshare-l40s",
        "schema_version": 1,
        "seed": seed,
    }
    path = tmp_path / f"pair-s{seed}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _preflight(tmp_path: Path, profile_sha: str) -> Path:
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "canaries": {
                    "exact_paired_resume": {"passed": True},
                    "one_hundred_update_throughput_oom": {
                        "oom_detected": False,
                        "passed": True,
                    },
                    "one_update_functional": {"passed": True},
                },
                "cohort_id": COHORT_ID,
                "dataset_receipt_sha256": "6" * 64,
                "profile_sha256": profile_sha,
                "schema_version": 1,
                "site_id": "farmshare-l40s",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_profiles_require_exactly_two_gpus_and_safe_fields():
    profile = load_profile(FARM_PROFILE)
    assert profile.platform == "farmshare"
    assert profile.gpus_per_pair == 2
    assert profile.gres == "gpu:L40S:2"
    assert profile.sha256 == _sha(FARM_PROFILE)
    for name in (
        "mit-collaborator-a.example.json",
        "mit-collaborator-b.example.json",
    ):
        mit = load_profile(ROOT / "cluster" / "profiles" / name)
        assert mit.platform == "mit"
        assert mit.gpus_per_pair == 2
        assert mit.gres.endswith(":2")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("partition", "gpu;touch-pwned"),
        ("account", "../../account"),
        ("qos", "normal && false"),
        ("gres", "gpu:h100:1"),
        ("gres", "gpu:h100:2 --wrap=x"),
        ("gpu_name_regex", "$(whoami)"),
        ("python", "/opt/venv/bin/python"),
        ("gpus_per_pair", 1),
    ],
)
def test_profile_rejects_injection_paths_and_non_pair_resources(
    tmp_path, field, value
):
    raw = json.loads(FARM_PROFILE.read_text())
    raw[field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_profile(path)


def test_pair_sbatch_argv_is_deterministic_and_resource_injected(tmp_path):
    profile = load_profile(FARM_PROFILE)
    pair = _pair_manifest(tmp_path, profile.sha256)
    first = plan_sbatch(
        pair,
        profile=profile,
        action="train",
        mode="functional",
        venv_root=tmp_path / "venv",
    )
    second = plan_sbatch(
        pair,
        profile=profile,
        action="train",
        mode="functional",
        venv_root=tmp_path / "venv",
    )
    assert first == second
    assert "--gres=gpu:L40S:2" in first
    assert "--partition=gpu" in first
    assert "--qos=gpu" in first
    assert f"--chdir={ROOT}" in first
    assert any(arg.startswith("--export=NONE,") for arg in first)
    assert Path(first[-1]) == ROOT / "cluster/slurm/v2_pair_train.sbatch"


def test_protected_submit_requires_all_bound_canaries(tmp_path):
    profile = load_profile(FARM_PROFILE)
    pair = _pair_manifest(tmp_path, profile.sha256)
    with pytest.raises(ValueError, match="preflight"):
        plan_sbatch(
            pair,
            profile=profile,
            action="train",
            mode="protected",
            venv_root=tmp_path / "venv",
        )
    preflight = _preflight(tmp_path, profile.sha256)
    command = plan_sbatch(
        pair,
        profile=profile,
        action="train",
        mode="protected",
        venv_root=tmp_path / "venv",
        preflight_path=preflight,
    )
    assert any(f"PREFLIGHT_PATH={preflight.resolve()}" in arg for arg in command)


def test_submit_is_dry_run_by_default_and_apply_aggregates_failures(tmp_path):
    profile = load_profile(FARM_PROFILE)
    pairs = [
        _pair_manifest(tmp_path, profile.sha256, seed=seed)
        for seed in (0, 5)
    ]
    calls = []
    dry = submit(
        pairs,
        profile_path=FARM_PROFILE,
        mode="functional",
        venv_root=tmp_path / "venv",
        runner=lambda command: calls.append(command),
    )
    assert calls == []
    assert dry["dry_run"] is True
    assert dry["planned"] == 2

    def runner(command):
        calls.append(command)
        return SimpleNamespace(
            returncode=7 if len(calls) == 2 else 0,
            stdout="submitted",
            stderr="rejected" if len(calls) == 2 else "",
        )

    report = submit(
        pairs,
        profile_path=FARM_PROFILE,
        mode="functional",
        venv_root=tmp_path / "venv",
        apply=True,
        runner=runner,
    )
    assert report["attempted"] == 2
    assert report["submitted"] == 1
    assert report["exit_code"] == 1


def test_paired_resume_requires_both_arms_at_same_step_and_cursor(tmp_path):
    dense = tmp_path / "dense.pt"
    split = tmp_path / "split.pt"
    dense.write_bytes(b"dense")
    split.write_bytes(b"split")
    states = {
        dense: {"step": 100, "data": {"cursor": 52428800, "epoch": 0}},
        split: {"step": 100, "data": {"cursor": 52428800, "epoch": 0}},
    }
    result = inspect_paired_resume(
        dense,
        split,
        loader=lambda path: states[path],
    )
    assert result == {"cursor": 52428800, "epoch": 0, "mode": "resume", "step": 100}
    states[split]["step"] = 99
    with pytest.raises(ValueError, match="step"):
        inspect_paired_resume(dense, split, loader=lambda path: states[path])
    split.unlink()
    with pytest.raises(ValueError, match="both"):
        inspect_paired_resume(dense, split, loader=lambda path: states[path])


def test_pair_scripts_are_generic_supervised_and_atomic():
    train = (ROOT / "cluster" / "slurm" / "v2_pair_train.sbatch").read_text()
    evaluate = (
        ROOT / "cluster" / "slurm" / "v2_pair_evaluate.sbatch"
    ).read_text()
    for text in (train, evaluate):
        assert "#SBATCH --partition=" not in text
        assert "#SBATCH --gres=" not in text
        assert "env -i" in text
        assert "PAIR_MANIFEST" in text
        assert "PROFILE_PATH" in text
    for required in (
        'allocation_gpus="${CUDA_VISIBLE_DEVICES:-}"',
        '"CUDA_VISIBLE_DEVICES=$dense_gpu"',
        '"CUDA_VISIBLE_DEVICES=$split90_gpu"',
        "expected exactly two Slurm-assigned GPUs",
        'wait "$dense_pid"',
        'wait "$split_pid"',
        "kill",
        ".partial",
        "os.replace",
    ):
        assert required in train
    assert "CUDA_VISIBLE_DEVICES=0" not in train
    assert "CUDA_VISIBLE_DEVICES=1" not in train


def test_instantiate_writes_two_hash_bound_pairs_without_replacement(
    tmp_path, monkeypatch
):
    fake_dataset = {
        "cohort_id": COHORT_ID,
        "dataset": {
            "lane_ids": [f"lane-{index}" for index in range(8)],
            "ordered_token_stream_sha256": "5" * 64,
            "raw_target_tokens": 7_120_879_616,
            "receipt_sha256": "6" * 64,
            "semantic_verification_sha256": "7" * 64,
            "stream_sha256": {},
        },
    }
    monkeypatch.setattr(
        operations,
        "build_role_manifest",
        lambda *args, **kwargs: fake_dataset,
    )
    result = operations.instantiate(
        "farmshare-lead",
        dataset_root=tmp_path / "dataset",
        pointer_path=tmp_path / "pointer.json",
        source_lock_path=tmp_path / "source-lock.json",
        profile_path=FARM_PROFILE,
        runtime_root=tmp_path / "runtime",
        out_root=tmp_path / "outputs",
        repository_root=ROOT,
    )
    assert len(result["pair_manifests"]) == 2
    for path in result["pair_manifests"]:
        pair = json.loads(path.read_text())
        assert [record["arm"] for record in pair["arms"]] == ["dense", "split90"]
        assert pair["dataset"]["receipt_sha256"] == "6" * 64
    with pytest.raises(FileExistsError):
        operations.instantiate(
            "farmshare-lead",
            dataset_root=tmp_path / "dataset",
            pointer_path=tmp_path / "pointer.json",
            source_lock_path=tmp_path / "source-lock.json",
            profile_path=FARM_PROFILE,
            runtime_root=tmp_path / "runtime",
            out_root=tmp_path / "outputs",
            repository_root=ROOT,
        )


def test_terminal_pair_checkpoint_receipt_is_hash_bound_and_no_replace(tmp_path):
    profile = load_profile(FARM_PROFILE)
    pair_path = _pair_manifest(tmp_path, profile.sha256)
    pair = json.loads(pair_path.read_text())
    terminal_step = 13_582
    terminal_cursor = 7_120_879_616
    for record in pair["arms"]:
        cfg = {
            "arm": record["arm"],
            "max_steps": terminal_step,
            "run_id": record["run_id"],
        }
        runtime_path = Path(record["runtime_config"])
        runtime_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        record["runtime_config_sha256"] = _sha(runtime_path)
        output = Path(record["out_dir"])
        output.mkdir()
        torch.save(
            {
                "cfg": cfg,
                "data": {"cursor": terminal_cursor, "epoch": 0},
                "step": terminal_step,
            },
            output / "ckpt.pt",
        )

    receipt_path = _write_pair_checkpoint_receipt(
        pair,
        evidence_root=tmp_path / "evidence",
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["pair_id"] == "d135m_full_s0"
    assert receipt["terminal_step"] == terminal_step
    assert receipt["terminal_cursor"] == terminal_cursor
    assert set(receipt["checkpoints"]) == {"dense", "split90"}
    for checkpoint in receipt["checkpoints"].values():
        path = Path(checkpoint["checkpoint_path"])
        assert checkpoint["checkpoint_sha256"] == _sha(path)
        assert checkpoint["bytes"] == path.stat().st_size
    with pytest.raises(FileExistsError):
        _write_pair_checkpoint_receipt(
            pair,
            evidence_root=tmp_path / "evidence",
        )


def test_terminal_pair_checkpoint_receipt_rejects_cursor_drift(tmp_path):
    profile = load_profile(FARM_PROFILE)
    pair = json.loads(_pair_manifest(tmp_path, profile.sha256).read_text())
    for index, record in enumerate(pair["arms"]):
        cfg = {
            "arm": record["arm"],
            "max_steps": 13_582,
            "run_id": record["run_id"],
        }
        runtime_path = Path(record["runtime_config"])
        runtime_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        record["runtime_config_sha256"] = _sha(runtime_path)
        output = Path(record["out_dir"])
        output.mkdir()
        torch.save(
            {
                "cfg": cfg,
                "data": {"cursor": 7_120_879_616 + index, "epoch": 0},
                "step": 13_582,
            },
            output / "ckpt.pt",
        )
    with pytest.raises(ValueError, match="cursor"):
        _write_pair_checkpoint_receipt(
            pair,
            evidence_root=tmp_path / "evidence",
        )
