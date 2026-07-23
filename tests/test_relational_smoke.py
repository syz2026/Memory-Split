from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from scripts.relational_smoke_test import run_smoke


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SMOKE = ROOT / "fixtures" / "current-smoke"


def _fixture_hashes():
    return {
        path.relative_to(CURRENT_SMOKE).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in CURRENT_SMOKE.rglob("*")
        if path.is_file()
    }


def test_local_pipeline_uses_real_paired_training_and_evaluation(tmp_path):
    before = _fixture_hashes()
    report = run_smoke(
        tmp_path,
        fixture=CURRENT_SMOKE,
        steps=2,
        device="cpu",
    )

    assert report == {
        "shared_stream": True,
        "dense_steps": 2,
        "split_steps": 2,
        "resume_exact": True,
        "resume_step": 3,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
        "fixture_unchanged": True,
        "profile": "smoke",
        "scientific_result": False,
    }
    assert json.loads((tmp_path / "smoke-report.json").read_text()) == report
    assert _fixture_hashes() == before

    for arm in ("dense", "split"):
        checkpoint = torch.load(
            tmp_path / "runs" / arm / "ckpt.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["step"] == 2
        config = checkpoint["cfg"]
        assert config["train_bin"] == str(CURRENT_SMOKE / "train.bin")
        assert config["train_weights"] == str(
            CURRENT_SMOKE / f"{arm}.weights.bin"
        )

    for mode in ("off", "on"):
        summary = json.loads(
            (tmp_path / "evals" / f"memory_{mode}" / "summary.json").read_text()
        )
        assert summary["memory"] == mode
        assert summary["n_items"] > 0
        assert summary["action_slots"] == 12
