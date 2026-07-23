import json
import hashlib

import pytest
import torch
import yaml

from scripts.make_kqa_configs import (
    make_continuation_config,
    validate_source_pair,
)


def test_make_continuation_config_repoints_data_and_resets_schedule(tmp_path):
    source = tmp_path / "source_dense"
    source.mkdir()
    torch.save({"model": {}, "step": 17}, source / "ckpt.pt")
    original = {
        "run_id": "old",
        "model": "toy",
        "ctx": 64,
        "arm": "dense",
        "train_bin": "/old/train.bin",
        "train_mask": None,
        "data_dir": "/old/data",
        "out_dir": "/old/run",
        "total_tokens": 1_000_000,
        "tokens_per_step": 1_000,
        "micro_batch_size": 4,
        "lr": 1e-3,
        "warmup_steps": 300,
        "max_steps": 999,
        "seed": 7,
    }
    (source / "config.yaml").write_text(yaml.safe_dump(original))

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "continuation_report.json").write_text(
        json.dumps(
            {
                "data_fingerprint": "abcdef1234567890",
                "arms": {
                    "dense": {
                        "total_tokens": 10_010,
                        "artifact_sha256": {
                            "train_bin": "dense-bin-sha",
                            "train_mask": "dense-mask-sha",
                        },
                    },
                    "split": {
                        "total_tokens": 10_020,
                        "artifact_sha256": {
                            "train_bin": "split-bin-sha",
                            "train_mask": "split-mask-sha",
                        },
                    },
                },
            }
        )
    )
    outputs = tmp_path / "outputs"
    config = make_continuation_config(
        str(source),
        "dense",
        corpus,
        outputs,
        continuation_tokens=10_000,
        lr=1e-4,
        micro_batch_size=1,
        precision="auto",
    )

    assert config["init_from"] == str((source / "ckpt.pt").resolve())
    assert config["train_bin"] == str((corpus / "dense/train.bin").resolve())
    assert config["train_mask"] is None
    source_sha = hashlib.sha256((source / "ckpt.pt").read_bytes()).hexdigest()
    assert config["out_dir"].startswith(
        str(
            (
                outputs
                / f"source_dense_kqa_dense_s{source_sha[:8]}_abcdef12_t10000_c"
            ).resolve()
        )
    )
    assert config["total_tokens"] == 10_000
    assert config["lr"] == 1e-4
    assert config["micro_batch_size"] == 1
    assert config["warmup_steps"] == 1
    assert config["max_steps"] == 11
    assert config["data_fingerprint"] == "abcdef1234567890"
    assert config["source_checkpoint_sha256"] == source_sha
    assert config["source_checkpoint_step"] == 17
    assert config["train_sha256"] == "dense-bin-sha"
    assert config["precision"] == "fp32"
    assert len(config["continuation_id"]) == 64


def test_validate_source_pair_rejects_mismatched_seed(tmp_path):
    base = {
        "model": "toy",
        "ctx": 64,
        "seed": 7,
        "total_tokens": 1_000,
        "tokens_per_step": 256,
        "load": "mini",
        "n_entities": 10,
    }
    paths = {}
    for arm in ("dense", "split"):
        run = tmp_path / arm
        run.mkdir()
        torch.save({"model": {}, "step": 12}, run / "ckpt.pt")
        (run / "config.yaml").write_text(yaml.safe_dump({**base, "arm": arm}))
        paths[arm] = run
    identity = validate_source_pair(str(paths["dense"]), str(paths["split"]))
    assert identity["arms"]["dense"]["step"] == 12
    assert identity["pair_id"]

    mismatch = {**base, "arm": "split", "seed": 8}
    (paths["split"] / "config.yaml").write_text(yaml.safe_dump(mismatch))
    with pytest.raises(ValueError, match="seed"):
        validate_source_pair(str(paths["dense"]), str(paths["split"]))


def test_validate_source_pair_rejects_mismatched_checkpoint_step(tmp_path):
    base = {
        "model": "toy",
        "ctx": 64,
        "seed": 7,
        "total_tokens": 1_000,
        "tokens_per_step": 256,
        "micro_batch_size": 2,
        "warmup_steps": 3,
        "arm": None,
    }
    paths = {}
    for arm, step in (("dense", 12), ("split", 13)):
        run = tmp_path / arm
        run.mkdir()
        torch.save({"model": {}, "step": step}, run / "ckpt.pt")
        (run / "config.yaml").write_text(
            yaml.safe_dump({**base, "arm": arm})
        )
        paths[arm] = run

    with pytest.raises(ValueError, match="checkpoint steps differ"):
        validate_source_pair(str(paths["dense"]), str(paths["split"]))
