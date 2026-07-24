import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys

import numpy as np
import pytest
import torch

import train.safeio as safeio
import train.trainer as trainer_module
from train.model import GPT, GPTConfig
from train.trainer import Trainer, cosine_lr
from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.aws_lifecycle import lifecycle_operational_metadata
from tests.provider_lifecycle_fixtures import provider_lifecycle


def write_corpus(tmp_path, n=40192, mask_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    # learnable structure: short repeating motifs + noise
    motif = rng.integers(0, 200, size=50)
    toks = np.tile(motif, n // 50 + 1)[:n].astype(np.uint16)
    noise_at = rng.random(n) < 0.05
    toks[noise_at] = rng.integers(0, 200, size=noise_at.sum())
    mask = (rng.random(n) > mask_frac).astype(np.uint8)
    bp, mp = tmp_path / "train.bin", tmp_path / "train.mask.bin"
    toks.tofile(bp)
    mask.tofile(mp)
    return bp, mp


def base_cfg(tmp_path, bp, mp):
    return {
        "model": {"n_layer": 2, "n_head": 2, "d_model": 64, "ctx": 64, "vocab_size": 50304},
        "train_bin": str(bp),
        "train_mask": str(mp) if mp is not None else None,
        "micro_batch_size": 4,
        "tokens_per_step": 4 * 64,
        "max_steps": 80,
        "lr": 3e-3,
        "warmup_steps": 5,
        "seed": 7,
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
        "log_every": 5,
        "eval_every": 10,
        "snap_frac": 0.5,
        "ckpt_minutes": 999,
    }


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_checkpoint_request_parent_group(path):
    os.chown(path, -1, os.getegid())


def tiny_cfg(tmp_path, bp, mp, *, out_name="out", max_steps=2):
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": max_steps,
            "out_dir": str(tmp_path / out_name),
            "log_every": 1,
            "eval_every": 100,
            "snap_frac": 0.5,
        }
    )
    return cfg


def _operational_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    lifecycle = provider_lifecycle(
        load_aws_gpu_profile(
            root / "cluster/profiles/aws-p5.48xlarge-v3.json"
        )
    )
    return lifecycle_operational_metadata(
        lifecycle.binding,
        run_id="memorysplit-v3-360m-s0-dense",
        arm="dense",
        config_sha256="b" * 64,
        dataset_receipt_sha256="c" * 64,
        dataset_build_id="d" * 64,
        ordered_stream_sha256="e" * 64,
        source_commit="f" * 40,
        source_tree="0" * 40,
    )


def test_production_model_snapshot_is_self_authenticating_and_legacy_is_explicit(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=256)
    selected_cfg = tiny_cfg(tmp_path, bp, mp, out_name="selected")
    selected_cfg["seed"] = 0
    selected_cfg["operational_metadata"] = _operational_metadata()
    selected = Trainer(selected_cfg)
    selected.step = 1
    selected.save_snapshot()
    selected.save_ckpt()
    selected.close()
    snapshot_path = tmp_path / "selected" / "snapshots" / "step0000001.pt"
    payload = snapshot_path.read_bytes()

    parsed = trainer_module.parse_model_snapshot_bytes(
        payload,
        expected_operational_metadata=selected_cfg["operational_metadata"],
    )

    assert parsed["run_id"] == "memorysplit-v3-360m-s0-dense"
    assert parsed["arm"] == "dense"
    assert parsed["config_sha256"] == "b" * 64
    assert parsed["provider_selection_sha256"] == "2" * 64
    assert parsed["runtime_sbom_sha256"] == "9" * 64
    assert parsed["objective_controls_contract_sha256"] == "a" * 64
    assert parsed["config_fingerprint"] == selected.config_fingerprint
    checkpoint_metadata = json.loads(
        (tmp_path / "selected" / "ckpt.meta.json").read_text(
            encoding="ascii"
        )
    )
    assert all(
        checkpoint_metadata[field] == value
        for field, value in selected_cfg["operational_metadata"].items()
    )

    mutated = copy.deepcopy(parsed)
    mutated["provider_selection_sha256"] = "1" * 64
    buffer = trainer_module.io.BytesIO()
    torch.save(mutated, buffer)
    with pytest.raises(ValueError, match="operational|selection|snapshot"):
        trainer_module.parse_model_snapshot_bytes(
            buffer.getvalue(),
            expected_operational_metadata=selected_cfg[
                "operational_metadata"
            ],
        )

    legacy_cfg = tiny_cfg(tmp_path, bp, mp, out_name="legacy")
    legacy = Trainer(legacy_cfg)
    legacy.step = 1
    legacy.save_snapshot()
    legacy.close()
    legacy_payload = (
        tmp_path / "legacy" / "snapshots" / "step0000001.pt"
    ).read_bytes()
    with pytest.raises(ValueError, match="legacy|operational"):
        trainer_module.parse_model_snapshot_bytes(legacy_payload)
    assert trainer_module.parse_model_snapshot_bytes(
        legacy_payload,
        allow_legacy=True,
    )["step"] == 1


def test_run_train_capabilities_json_is_strict_and_does_not_require_config():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    for name in (
        "LOCAL_RANK",
        "MS_RANK_ZERO_PID_FILE",
        "RANK",
        "WORLD_SIZE",
    ):
        environment.pop(name, None)

    completed = subprocess.run(
        [
            sys.executable,
            root / "scripts" / "run_train.py",
            "--capabilities-json",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    expected = {
        "rank_zero_pid_file": True,
        "receipt_v2": True,
        "resume_sha256": True,
        "sidecar_name": True,
        "sigusr1_checkpoint": True,
        "sigusr1_request_token": True,
    }

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == (
        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_checkpoint_request_token_is_canonical_and_arm_bound() -> None:
    request_id = "a" * 32

    payload = safeio.checkpoint_request_token_bytes(
        request_id,
        arm="dense",
    )

    assert payload == (
        b'{"arm":"dense","request_id":"'
        + request_id.encode("ascii")
        + b'","schema_version":1}\n'
    )
    assert (
        safeio.parse_checkpoint_request_token(
            payload,
            expected_arm="dense",
        )
        == request_id
    )
    with pytest.raises(ValueError, match="arm"):
        safeio.parse_checkpoint_request_token(
            payload,
            expected_arm="split90",
        )
    with pytest.raises(ValueError, match="canonical|fields"):
        safeio.parse_checkpoint_request_token(
            payload[:-1],
            expected_arm="dense",
        )


def test_checkpoint_request_token_missing_ok_is_explicit_and_strict_by_default(
    tmp_path,
) -> None:
    parent = tmp_path / "missing"
    parent.mkdir(mode=0o700)
    set_checkpoint_request_parent_group(parent)
    path = parent / "checkpoint-request.json"

    with pytest.raises(FileNotFoundError, match="request token|missing"):
        safeio.consume_checkpoint_request_token(
            path,
            expected_arm="dense",
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )

    assert (
        safeio.consume_checkpoint_request_token(
            path,
            expected_arm="dense",
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            missing_ok=True,
        )
        is None
    )


def test_checkpoint_request_token_publish_is_exclusive_and_substitution_safe(
    tmp_path,
) -> None:
    parent = tmp_path / "run"
    parent.mkdir(mode=0o700)
    set_checkpoint_request_parent_group(parent)
    path = parent / "checkpoint-request.json"
    first = safeio.publish_checkpoint_request_token(
        path,
        request_id="a" * 32,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )

    with pytest.raises(FileExistsError, match="exists|pending"):
        safeio.publish_checkpoint_request_token(
            path,
            request_id="b" * 32,
            arm="dense",
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
        )
    assert path.read_bytes() == safeio.checkpoint_request_token_bytes(
        "a" * 32,
        arm="dense",
    )

    path.unlink()
    replacement = safeio.checkpoint_request_token_bytes(
        "b" * 32,
        arm="dense",
    )
    path.write_bytes(replacement)
    path.chmod(0o400)
    with pytest.raises(ValueError, match="identity|changed"):
        safeio.cleanup_checkpoint_request_token(first)
    assert path.read_bytes() == replacement


@pytest.mark.parametrize("mutation", ["hardlink", "mode", "owner"])
def test_checkpoint_request_token_consume_rejects_unsafe_files(
    tmp_path,
    mutation,
) -> None:
    parent = tmp_path / mutation
    parent.mkdir(mode=0o700)
    set_checkpoint_request_parent_group(parent)
    path = parent / "checkpoint-request.json"
    published = safeio.publish_checkpoint_request_token(
        path,
        request_id="c" * 32,
        arm="split90",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )
    expected_uid = os.geteuid()
    if mutation == "hardlink":
        os.link(path, parent / "linked-token")
    elif mutation == "mode":
        path.chmod(0o600)
    elif mutation == "owner":
        expected_uid += 1
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="hard.link|mode|owner|unsafe"):
        safeio.consume_checkpoint_request_token(
            path,
            expected_arm="split90",
            expected_uid=expected_uid,
            expected_gid=os.getegid(),
        )

    if mutation == "hardlink":
        (parent / "linked-token").unlink()
    if path.exists():
        path.chmod(0o400)
        safeio.cleanup_checkpoint_request_token_path(
            path,
            request_id=published.request_id,
            arm=published.arm,
            owner_uid=published.owner_uid,
            owner_gid=published.owner_gid,
        )


def test_checkpoint_request_token_rejects_wrong_parent_gid_before_claim(
    tmp_path,
    monkeypatch,
) -> None:
    parent = tmp_path / "wrong-parent-gid"
    parent.mkdir(mode=0o700)
    set_checkpoint_request_parent_group(parent)
    path = parent / "checkpoint-request.json"
    safeio.publish_checkpoint_request_token(
        path,
        request_id="d" * 32,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )
    rename_calls = []
    real_rename = safeio.os.rename

    def recording_rename(*args, **kwargs):
        rename_calls.append((args, kwargs))
        return real_rename(*args, **kwargs)

    monkeypatch.setattr(safeio.os, "rename", recording_rename)

    with pytest.raises(ValueError, match="parent|foreign|unsafe"):
        safeio.consume_checkpoint_request_token(
            path,
            expected_arm="dense",
            expected_uid=os.geteuid(),
            expected_gid=os.getegid() + 1,
        )

    assert rename_calls == []
    assert path.exists()


def test_checkpoint_request_token_rejects_symlinked_path_components(
    tmp_path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    set_checkpoint_request_parent_group(real_parent)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    path = linked_parent / "checkpoint-request.json"

    with pytest.raises(ValueError, match="symlink|unsafe"):
        safeio.publish_checkpoint_request_token(
            path,
            request_id="d" * 32,
            arm="dense",
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
        )
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    (real_parent / "checkpoint-request.json").symlink_to(target)
    with pytest.raises(FileExistsError, match="exists|pending"):
        safeio.publish_checkpoint_request_token(
            real_parent / "checkpoint-request.json",
            request_id="d" * 32,
            arm="dense",
            owner_uid=os.geteuid(),
            owner_gid=os.getegid(),
        )
    assert target.read_bytes() == b"unchanged"


def test_operational_steps_preserve_config_fingerprint_and_resume_one_to_two(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=4)
    original = copy.deepcopy(cfg)

    first = trainer_module.train(cfg, operational_steps=1)
    first_checkpoint = torch.load(
        first.ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    config_bytes = (first.out_dir / "config.yaml").read_bytes()
    fingerprint = first.config_fingerprint

    assert cfg == original
    assert first.step == 1
    assert first.data.global_cursor == cfg["tokens_per_step"]
    assert first_checkpoint["cfg"] == original
    assert first_checkpoint["config_fingerprint"] == fingerprint
    assert first_checkpoint["step"] == 1
    assert first_checkpoint["data"]["global_cursor"] == cfg["tokens_per_step"]
    assert first.max_steps == 4
    first.close()

    resumed = trainer_module.train(
        cfg,
        resume="auto",
        operational_steps=1,
    )
    resumed_checkpoint = torch.load(
        resumed.ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    assert cfg == original
    assert resumed.step == 2
    assert resumed.data.global_cursor == 2 * cfg["tokens_per_step"]
    assert resumed.config_fingerprint == fingerprint
    assert resumed_checkpoint["cfg"] == original
    assert resumed_checkpoint["config_fingerprint"] == fingerprint
    assert resumed_checkpoint["step"] == 2
    assert resumed_checkpoint["data"]["global_cursor"] == (
        2 * cfg["tokens_per_step"]
    )
    assert resumed.max_steps == 4
    assert (resumed.out_dir / "config.yaml").read_bytes() == config_bytes
    resumed.close()


def test_operational_steps_are_positive_exact_integers_and_cap_at_max_steps(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)

    for invalid in (True, 1.0, 0, -1, sys.maxsize + 1):
        with pytest.raises(ValueError, match="operational_steps"):
            trainer_module.train(cfg, operational_steps=invalid)
        assert not Path(cfg["out_dir"]).exists()

    first = trainer_module.train(cfg, operational_steps=1)
    first.close()
    resumed = trainer_module.train(
        cfg,
        resume="auto",
        operational_steps=2,
    )

    assert resumed.step == 2
    assert resumed.data.global_cursor == 2 * cfg["tokens_per_step"]
    resumed.close()


def test_operational_training_exposes_one_finite_metric_per_completed_update(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=3)

    trainer = trainer_module.train(cfg, operational_steps=2)

    assert trainer.operational_start_step == 0
    assert trainer.step == 2
    assert len(trainer.operational_step_tok_s) == 2
    assert all(
        isinstance(value, float) and np.isfinite(value) and value > 0
        for value in trainer.operational_step_tok_s
    )
    trainer.close()


def test_run_train_emits_canonical_operational_metrics_without_config_mutation(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    before = config_path.read_bytes()
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            root / "scripts" / "run_train.py",
            "--config",
            config_path,
            "--operational-steps",
            "1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    lines = completed.stdout.splitlines()
    metrics = json.loads(lines[0])
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert metrics["receipt_type"] == "memorysplit-operational-training-v1"
    assert metrics["start_step"] == 0
    assert metrics["end_step"] == 1
    assert metrics["updates"] == 1
    assert len(metrics["step_tok_s"]) == 1
    assert np.isfinite(metrics["step_tok_s"][0])
    assert metrics["step_tok_s"][0] > 0
    assert lines[1].startswith("done: step=1 out=")
    assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    "arguments",
    [
        ["--operational-steps", "0"],
        ["--operational-steps", "-1"],
        ["--operational-steps", "1.0"],
        ["--operational-steps", "true"],
        ["--operational-steps", "+1"],
        ["--operational-steps", "01"],
        ["--operational-steps", str(sys.maxsize + 1)],
        ["--operational-steps", "1", "--operational-steps", "2"],
        ["--operational-step", "1"],
        ["--operational-step=1"],
        ["--operational-s", "1"],
        ["--oper=1"],
        ["--o", "1"],
    ],
)
def test_run_train_operational_steps_cli_fails_closed_before_config_read(
    tmp_path,
    arguments,
):
    root = Path(__file__).resolve().parents[1]
    missing_config = tmp_path / "must-not-be-read.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            root / "scripts" / "run_train.py",
            "--config",
            missing_config,
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "operational-steps" in completed.stderr
    assert "No such file" not in completed.stderr


def test_preregistered_v2_snapshot_steps_are_preserved_exactly():
    expected = (1358, 3396, 6791, 10187, 13582)

    actual = trainer_module.resolve_snapshot_steps(
        {
            "schema_version": 2,
            "snapshot_steps": list(expected),
        },
        max_steps=13582,
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("config", "max_steps"),
    [
        ({"schema_version": 2, "snap_frac": 0.1}, 10),
        (
            {
                "schema_version": 2,
                "snap_frac": 0.1,
                "snapshot_steps": [5, 10],
            },
            10,
        ),
        ({"schema_version": 2, "snapshot_steps": [5, True]}, 10),
        ({"schema_version": 2, "snapshot_steps": [5.0, 10]}, 10),
        ({"schema_version": 2, "snapshot_steps": [5, 5, 10]}, 10),
        ({"schema_version": 2, "snapshot_steps": [6, 5, 10]}, 10),
        ({"schema_version": 2, "snapshot_steps": [5, 11]}, 10),
        ({"schema_version": 2, "snapshot_steps": [5, 9]}, 10),
    ],
)
def test_v2_snapshot_steps_reject_legacy_or_inexact_schedules(config, max_steps):
    with pytest.raises(ValueError, match="snapshot_steps|snap_frac"):
        trainer_module.resolve_snapshot_steps(config, max_steps=max_steps)


def test_explicit_snapshot_steps_save_only_the_declared_steps(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=3)
    cfg.pop("snap_frac")
    cfg.update(
        {
            "schema_version": 2,
            "snapshot_steps": [1, 3],
        }
    )

    trainer = Trainer(cfg)
    trainer.train_steps()

    assert sorted(path.name for path in (trainer.out_dir / "snapshots").iterdir()) == [
        "step0000001.pt",
        "step0000003.pt",
    ]
    trainer.close()


def test_fresh_launch_refuses_even_an_empty_existing_output_before_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileExistsError, match="fresh.*output"):
        Trainer(cfg)

    assert marker.read_text() == "unchanged"
    assert set(output.iterdir()) == {marker}


def test_auto_resume_requires_default_checkpoint_before_output_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        Trainer(cfg, resume="auto")

    assert marker.read_text() == "unchanged"
    assert set(output.iterdir()) == {marker}


def test_auto_resume_never_starts_fresh_when_output_is_absent(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)

    with pytest.raises(FileNotFoundError, match="checkpoint|missing"):
        Trainer(cfg, resume="auto")

    assert not Path(cfg["out_dir"]).exists()


def test_default_auto_resume_loads_existing_checkpoint_without_rewriting_config(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 2
    first = Trainer(cfg)
    first.train_steps(1)
    config_before = (first.out_dir / "config.yaml").read_bytes()
    first.close()

    resumed = Trainer(cfg, resume="auto")

    assert resumed.step == 1
    assert (resumed.out_dir / "config.yaml").read_bytes() == config_before
    resumed.close()


def test_auto_resume_removes_stale_step_two_artifacts_before_exact_replay(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    step_one_checkpoint = first.ckpt_path.read_bytes()
    first.train_steps(1)
    first.close()
    Path(cfg["out_dir"], "ckpt.pt").write_bytes(step_one_checkpoint)

    resumed = Trainer(cfg, resume="auto")

    assert resumed.step == 1
    assert [
        json.loads(line)["step"]
        for line in resumed.log_path.read_text().splitlines()
    ] == [1]
    assert sorted(
        path.name
        for path in (resumed.out_dir / "snapshots").glob("step*.pt")
    ) == ["step0000001.pt"]

    resumed.train_steps(1)

    replayed_steps = [
        json.loads(line)["step"]
        for line in resumed.log_path.read_text().splitlines()
    ]
    assert replayed_steps == [1, 2]
    assert len(replayed_steps) == len(set(replayed_steps))
    assert sorted(
        path.name
        for path in (resumed.out_dir / "snapshots").glob("step*.pt")
    ) == ["step0000001.pt", "step0000002.pt"]
    resumed.close()


def test_auto_resume_truncates_only_partial_final_log_record_before_replay(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    log_path = Path(cfg["out_dir"], "log.jsonl")
    durable_log = log_path.read_bytes()
    assert durable_log.endswith(b"\n")
    log_path.write_bytes(durable_log + b'{"step":2,"loss":')

    resumed = Trainer(cfg, resume="auto")

    assert resumed.step == 1
    assert log_path.read_bytes() == durable_log
    resumed.train_steps(1)
    assert [
        json.loads(line)["step"]
        for line in log_path.read_text().splitlines()
    ] == [1, 2]
    resumed.close()


def test_auto_resume_cleans_owned_orphan_snapshot_atomic_temporary(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    temporary = (
        Path(cfg["out_dir"])
        / "snapshots"
        / ".step0000002.pt.tmp-4242-0123456789abcdef"
    )
    temporary.write_bytes(b"interrupted snapshot write")
    temporary.chmod(0o600)

    resumed = Trainer(cfg, resume="auto")

    assert not temporary.exists()
    resumed.train_steps(1)
    assert (
        Path(cfg["out_dir"]) / "snapshots" / "step0000002.pt"
    ).is_file()
    resumed.close()


def test_auto_resume_cleans_owned_orphan_log_atomic_temporary(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    temporary = (
        Path(cfg["out_dir"])
        / ".log.jsonl.tmp-4242-0011223344556677"
    )
    temporary.write_bytes(b"interrupted log rewrite")
    temporary.chmod(0o600)

    resumed = Trainer(cfg, resume="auto")

    assert not temporary.exists()
    resumed.close()


def test_auto_resume_cleans_owned_snapshot_hardlink_install_state(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    snapshots = Path(cfg["out_dir"]) / "snapshots"
    state = torch.load(snapshots / "step0000001.pt", weights_only=False)
    state["step"] = 2
    temporary = snapshots / ".step0000002.pt.tmp-4242-fedcba9876543210"
    torch.save(state, temporary)
    temporary.chmod(0o600)
    interrupted_final = snapshots / "step0000002.pt"
    os.link(temporary, interrupted_final)
    assert temporary.stat().st_ino == interrupted_final.stat().st_ino
    assert temporary.stat().st_nlink == 2

    resumed = Trainer(cfg, resume="auto")

    assert not temporary.exists()
    assert not interrupted_final.exists()
    resumed.train_steps(1)
    assert interrupted_final.is_file()
    resumed.close()


def test_auto_resume_rejects_foreign_atomic_temporary_near_miss(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    foreign = (
        Path(cfg["out_dir"])
        / "snapshots"
        / ".step0000002.pt.tmp-foreign"
    )
    foreign.write_bytes(b"do not delete")

    with pytest.raises(ValueError, match="foreign|temporary"):
        Trainer(cfg, resume="auto")

    assert foreign.read_bytes() == b"do not delete"


def test_auto_resume_rejects_foreign_root_atomic_temporary_near_miss(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    foreign = Path(cfg["out_dir"]) / ".log.jsonl.tmp-foreign"
    foreign.write_bytes(b"do not delete")

    with pytest.raises(ValueError, match="foreign|temporary"):
        Trainer(cfg, resume="auto")

    assert foreign.read_bytes() == b"do not delete"


@pytest.mark.parametrize("corruption", ["malformed-log", "foreign-snapshot"])
def test_auto_resume_rejects_malformed_or_foreign_owned_artifacts(
    tmp_path,
    corruption,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=2)
    first = Trainer(cfg)
    first.train_steps(1)
    first.close()
    output = Path(cfg["out_dir"])
    if corruption == "malformed-log":
        artifact = output / "log.jsonl"
        artifact.write_bytes(artifact.read_bytes() + b"not-json\n")
    else:
        artifact = output / "snapshots" / "foreign.pt"
        artifact.write_bytes(b"foreign")
    before = artifact.read_bytes()

    with pytest.raises(ValueError, match="log|snapshot|foreign"):
        Trainer(cfg, resume="auto")

    assert artifact.read_bytes() == before


def test_auto_resume_rejects_snapshot_step_not_declared_by_exact_schedule(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp, max_steps=3)
    cfg.pop("snap_frac")
    cfg.update(
        {
            "schema_version": 2,
            "snapshot_steps": [1, 3],
        }
    )
    first = Trainer(cfg)
    first.train_steps()
    first.close()
    snapshots = Path(cfg["out_dir"]) / "snapshots"
    undeclared_state = torch.load(
        snapshots / "step0000001.pt",
        weights_only=False,
    )
    undeclared_state["step"] = 2
    undeclared = snapshots / "step0000002.pt"
    torch.save(undeclared_state, undeclared)

    with pytest.raises(ValueError, match="snapshot.*not configured"):
        Trainer(cfg, resume="auto")

    assert undeclared.is_file()


def test_external_resume_requires_matching_sha_before_creating_output(tmp_path):
    bp, mp = write_corpus(tmp_path)
    source_cfg = base_cfg(tmp_path, bp, mp)
    source_cfg["max_steps"] = 2
    source = Trainer(source_cfg)
    source.train_steps(1)
    checkpoint = source.ckpt_path
    source.close()
    resumed_cfg = dict(source_cfg, out_dir=str(tmp_path / "resumed"))

    with pytest.raises(ValueError, match="resume_sha256"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=checkpoint,
        )
    assert not (tmp_path / "resumed").exists()

    with pytest.raises(ValueError, match="SHA-256"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=checkpoint,
            resume_sha256="0" * 64,
        )
    assert not (tmp_path / "resumed").exists()

    resumed = Trainer(
        resumed_cfg,
        resume="auto",
        resume_path=checkpoint,
        resume_sha256=file_sha256(checkpoint),
    )
    assert resumed.step == 1
    assert resumed.data.global_cursor == source_cfg["tokens_per_step"]
    resumed.close()


def test_external_resume_rejects_symlinked_checkpoint_without_output_mutation(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path)
    source_cfg = base_cfg(tmp_path, bp, mp)
    source_cfg["max_steps"] = 1
    source = Trainer(source_cfg)
    source.save_ckpt()
    linked = tmp_path / "linked-checkpoint.pt"
    linked.symlink_to(source.ckpt_path)
    resumed_cfg = dict(source_cfg, out_dir=str(tmp_path / "resumed"))

    with pytest.raises(ValueError, match="symlink|unsafe"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=linked,
            resume_sha256=file_sha256(source.ckpt_path),
        )

    assert not (tmp_path / "resumed").exists()
    source.close()


def test_external_resume_hashes_and_loads_exactly_the_same_descriptor(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    source_cfg = base_cfg(tmp_path, bp, mp)
    source_cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 2,
        }
    )
    source = Trainer(source_cfg)
    source.train_steps(1)
    checkpoint = source.ckpt_path
    expected_sha256 = file_sha256(checkpoint)
    source.close()
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"not a checkpoint")
    real_read = safeio._read_and_hash_fd
    swapped = False

    def replace_during_read(fd):
        nonlocal swapped
        if not swapped:
            os.replace(replacement, checkpoint)
            swapped = True
        return real_read(fd)

    monkeypatch.setattr(safeio, "_read_and_hash_fd", replace_during_read)
    resumed_cfg = dict(source_cfg, out_dir=str(tmp_path / "resumed-pinned"))
    resumed = Trainer(
        resumed_cfg,
        resume="auto",
        resume_path=checkpoint,
        resume_sha256=expected_sha256,
    )

    assert resumed.step == 1
    assert checkpoint.read_bytes() == b"not a checkpoint"
    resumed.close()


def test_output_rejects_symlinked_parent_component_before_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["out_dir"] = str(linked_parent / "out")

    with pytest.raises(ValueError, match="symlink|unsafe"):
        Trainer(cfg)

    assert not (real_parent / "out").exists()


def test_durable_config_checkpoint_and_snapshot_fsync_files_and_directories(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    real_fsync = safeio.os.fsync
    fsync_kinds = []

    def recording_fsync(fd):
        fsync_kinds.append(
            "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        )
        return real_fsync(fd)

    monkeypatch.setattr(safeio.os, "fsync", recording_fsync)
    trainer = Trainer(cfg)
    trainer.save_snapshot()
    trainer.save_ckpt()

    assert "file" in fsync_kinds
    assert "directory" in fsync_kinds
    assert not list(trainer.out_dir.rglob("*.tmp-*"))
    trainer.close()


def test_checkpoint_metadata_binds_installed_generation(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    trainer = Trainer(cfg)

    trainer.save_ckpt()

    metadata_path = trainer.out_dir / "ckpt.meta.json"
    payload = metadata_path.read_bytes()
    metadata = json.loads(payload)
    checkpoint_stat = trainer.ckpt_path.stat(follow_symlinks=False)
    assert payload == (
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    assert metadata == {
        "checkpoint_version": 3,
        "config_fingerprint": trainer.config_fingerprint,
        "data": {
            "build_id": None,
            "global_cursor": 0,
            "ordered_stream_sha256": None,
            "receipt_sha256": None,
            "sidecar_name": None,
        },
        "installed": {
            "bytes": checkpoint_stat.st_size,
            "ctime_ns": checkpoint_stat.st_ctime_ns,
            "device": checkpoint_stat.st_dev,
            "gid": checkpoint_stat.st_gid,
            "inode": checkpoint_stat.st_ino,
            "links": checkpoint_stat.st_nlink,
            "mode": checkpoint_stat.st_mode,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
            "uid": checkpoint_stat.st_uid,
        },
        "request_token": None,
        "receipt_type": "memorysplit-trainer-checkpoint-v1",
        "schema_version": 1,
        "step": 0,
        "world_size": 1,
    }
    trainer.close()


def test_checkpoint_and_snapshot_writes_reject_symlink_swaps(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    trainer = Trainer(cfg)
    checkpoint_target = tmp_path / "checkpoint-target"
    checkpoint_target.write_bytes(b"unchanged")
    trainer.ckpt_path.symlink_to(checkpoint_target)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        trainer.save_ckpt()
    assert checkpoint_target.read_bytes() == b"unchanged"
    trainer.ckpt_path.unlink()

    snapshots = trainer.out_dir / "snapshots"
    moved = trainer.out_dir / "moved-snapshots"
    snapshots.rename(moved)
    snapshots.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="changed|unsafe"):
        trainer.save_snapshot()
    trainer.close()


def test_sigusr1_handler_only_requests_checkpoint_until_safe_boundary(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    pid_path = tmp_path / "rank-zero.pid"
    installed = {}
    monkeypatch.delenv("MS_CHECKPOINT_REQUEST_TOKEN_FILE", raising=False)
    monkeypatch.delenv("MS_CHECKPOINT_REQUEST_ARM", raising=False)
    monkeypatch.setenv("MS_RANK_ZERO_PID_FILE", str(pid_path))
    monkeypatch.setattr(signal, "getsignal", lambda signum: f"previous-{signum}")

    def install(signum, handler):
        installed[signum] = handler

    monkeypatch.setattr(signal, "signal", install)
    trainer = Trainer(cfg)

    assert pid_path.read_text(encoding="ascii") == f"{os.getpid()}\n"
    installed[signal.SIGUSR1](signal.SIGUSR1, None)
    assert not trainer.ckpt_path.exists()

    assert trainer._service_checkpoint_request() is True
    assert torch.load(trainer.ckpt_path, weights_only=False)["step"] == 0
    metadata = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert metadata["request_token"] is None
    assert trainer.checkpoint_request_token_path is None
    trainer.close()


def test_sigusr1_service_consumes_and_binds_exact_request_token(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    cfg["condition"] = "dense"
    token_path = Path(cfg["out_dir"]) / "checkpoint-request.json"
    installed = {}
    monkeypatch.setenv(
        "MS_CHECKPOINT_REQUEST_TOKEN_FILE",
        str(token_path),
    )
    monkeypatch.setenv("MS_CHECKPOINT_REQUEST_ARM", "dense")
    monkeypatch.setattr(signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )
    trainer = Trainer(cfg)
    set_checkpoint_request_parent_group(trainer.out_dir)
    request_id = "a" * 32
    safeio.publish_checkpoint_request_token(
        token_path,
        request_id=request_id,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )

    installed[signal.SIGUSR1](signal.SIGUSR1, None)

    assert token_path.exists()
    assert not trainer.ckpt_path.exists()
    assert trainer._service_checkpoint_request() is True
    metadata = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert metadata["request_token"] == request_id
    assert not token_path.exists()

    trainer.save_ckpt()
    periodic = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert periodic["request_token"] is None
    trainer.close()


def test_sigusr1_service_abandons_cleaned_token_and_services_next_signal(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    cfg["condition"] = "dense"
    token_path = Path(cfg["out_dir"]) / "checkpoint-request.json"
    installed = {}
    monkeypatch.setenv(
        "MS_CHECKPOINT_REQUEST_TOKEN_FILE",
        str(token_path),
    )
    monkeypatch.setenv("MS_CHECKPOINT_REQUEST_ARM", "dense")
    monkeypatch.setattr(signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )
    trainer = Trainer(cfg)
    set_checkpoint_request_parent_group(trainer.out_dir)
    abandoned = safeio.publish_checkpoint_request_token(
        token_path,
        request_id="a" * 32,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )
    installed[signal.SIGUSR1](signal.SIGUSR1, None)
    assert safeio.cleanup_checkpoint_request_token(abandoned) is True

    assert trainer._service_checkpoint_request() is False
    assert not trainer.ckpt_path.exists()

    safeio.publish_checkpoint_request_token(
        token_path,
        request_id="b" * 32,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )
    assert trainer._service_checkpoint_request() is False
    assert token_path.exists()

    installed[signal.SIGUSR1](signal.SIGUSR1, None)
    assert trainer._service_checkpoint_request() is True
    metadata = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert metadata["request_token"] == "b" * 32
    assert not token_path.exists()
    trainer.close()


def test_missing_signal_token_preserves_due_periodic_checkpoint(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    cfg["condition"] = "dense"
    token_path = Path(cfg["out_dir"]) / "checkpoint-request.json"
    installed = {}
    monkeypatch.setenv(
        "MS_CHECKPOINT_REQUEST_TOKEN_FILE",
        str(token_path),
    )
    monkeypatch.setenv("MS_CHECKPOINT_REQUEST_ARM", "dense")
    monkeypatch.setattr(signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )
    trainer = Trainer(cfg)
    set_checkpoint_request_parent_group(trainer.out_dir)
    abandoned = safeio.publish_checkpoint_request_token(
        token_path,
        request_id="c" * 32,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )
    installed[signal.SIGUSR1](signal.SIGUSR1, None)
    assert safeio.cleanup_checkpoint_request_token(abandoned) is True

    assert trainer._service_checkpoint_request(checkpoint_due=True) is False
    metadata = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert trainer.ckpt_path.exists()
    assert metadata["request_token"] is None
    trainer.close()


def test_periodic_checkpoint_does_not_consume_pending_signal_token(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    cfg["condition"] = "dense"
    token_path = Path(cfg["out_dir"]) / "checkpoint-request.json"
    installed = {}
    monkeypatch.setenv(
        "MS_CHECKPOINT_REQUEST_TOKEN_FILE",
        str(token_path),
    )
    monkeypatch.setenv("MS_CHECKPOINT_REQUEST_ARM", "dense")
    monkeypatch.setattr(signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )
    trainer = Trainer(cfg)
    set_checkpoint_request_parent_group(trainer.out_dir)
    safeio.publish_checkpoint_request_token(
        token_path,
        request_id="c" * 32,
        arm="dense",
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )

    assert trainer._service_checkpoint_request(checkpoint_due=True) is False
    periodic = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert periodic["request_token"] is None
    assert token_path.exists()

    installed[signal.SIGUSR1](signal.SIGUSR1, None)
    assert trainer._service_checkpoint_request() is True
    acknowledged = json.loads(
        (trainer.out_dir / "ckpt.meta.json").read_text(encoding="ascii")
    )
    assert acknowledged["request_token"] == "c" * 32
    assert not token_path.exists()
    trainer.close()


def test_log_append_rejects_hard_link_before_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = tiny_cfg(tmp_path, bp, mp)
    trainer = Trainer(cfg)
    trainer.train_steps(1)
    linked_log = tmp_path / "linked-log.jsonl"
    os.link(trainer.log_path, linked_log)
    before = trainer.log_path.read_bytes()

    with pytest.raises(ValueError, match="hard.link|owned"):
        trainer._output.root.append_bytes("log.jsonl", b"foreign\n")

    assert trainer.log_path.read_bytes() == before
    assert linked_log.read_bytes() == before
    trainer.close()


def test_loss_decreases_and_logs(tmp_path):
    bp, mp = write_corpus(tmp_path)
    tr = Trainer(base_cfg(tmp_path, bp, mp))
    tr.train_steps()
    rows = [json.loads(line) for line in open(tr.log_path)]
    first, last = rows[0]["loss"], rows[-1]["loss_ema"]
    assert last < first * 0.8, (first, last)
    assert any("loss_masked_values" in r for r in rows)
    assert (tr.out_dir / "ckpt.pt").exists()
    snaps = list((tr.out_dir / "snapshots").glob("*.pt"))
    assert snaps


def test_checkpoint_resume_exact_batches(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 10
    a = Trainer(cfg)
    a.train_steps(6)
    a.save_ckpt()
    cursor_after_6 = a.data.state_dict()["cursor"]

    cfg2 = dict(cfg, out_dir=str(tmp_path / "out2"))
    b = Trainer(
        cfg2,
        resume="auto",
        resume_path=a.ckpt_path,
        resume_sha256=file_sha256(a.ckpt_path),
    )
    assert b.step == 6
    assert b.data.state_dict()["cursor"] == cursor_after_6
    xb, _ = b.data.next_batch()
    xa, _ = a.data.next_batch()
    assert (xa == xb).all()


def test_checkpoint_rejects_world_size_mismatch(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    state["world_size"] = 2
    mismatched = tmp_path / "world-size-mismatch.pt"
    torch.save(state, mismatched)

    resumed_cfg = dict(cfg, out_dir=str(tmp_path / "world-size-resume"))
    with pytest.raises(ValueError, match="world size"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=mismatched,
            resume_sha256=file_sha256(mismatched),
        )


def test_checkpoint_rejects_data_provenance_mismatch(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    state["data_provenance"] = {"format_version": 2, "tokens": "different"}
    mismatched = tmp_path / "provenance-mismatch.pt"
    torch.save(state, mismatched)

    resumed_cfg = dict(cfg, out_dir=str(tmp_path / "provenance-resume"))
    with pytest.raises(ValueError, match="provenance"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=mismatched,
            resume_sha256=file_sha256(mismatched),
        )


def test_checkpoint_rejects_nested_data_provenance_numeric_type_drift(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    state["data_provenance"] = copy.deepcopy(state["data_provenance"])
    byte_count = state["data_provenance"]["tokens"]["bytes"]
    state["data_provenance"]["tokens"]["bytes"] = float(byte_count)
    drifted = tmp_path / "provenance-type-drift.pt"
    torch.save(state, drifted)

    with pytest.raises(ValueError, match="provenance"):
        trainer.load_ckpt(drifted, sha256=file_sha256(drifted))
    trainer.close()


def test_checkpoint_rejects_adamw_parameter_step_drift(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 2,
        }
    )
    trainer = Trainer(cfg)
    trainer.train_steps(1)
    state = torch.load(trainer.ckpt_path, weights_only=False)
    assert state["step"] == 1
    parameter_state = next(iter(state["opt"]["state"].values()))
    original_step = parameter_state["step"]
    parameter_state["step"] = original_step + 1
    assert parameter_state["step"].dtype == original_step.dtype
    assert parameter_state["step"].device == original_step.device
    drifted = tmp_path / "adamw-step-drift.pt"
    torch.save(state, drifted)

    with pytest.raises(ValueError, match="AdamW step"):
        trainer.load_ckpt(drifted, sha256=file_sha256(drifted))
    trainer.close()


def test_checkpoint_rejects_training_config_mismatch(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    changed_cfg = dict(
        cfg,
        lr=cfg["lr"] * 2,
        out_dir=str(tmp_path / "changed-out"),
    )
    with pytest.raises(ValueError, match="config"):
        Trainer(
            changed_cfg,
            resume="auto",
            resume_path=trainer.ckpt_path,
            resume_sha256=file_sha256(trainer.ckpt_path),
        )


def test_checkpoint_rejects_changed_sidecar_content(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    changed_mask = tmp_path / "changed.mask.bin"
    mask = np.fromfile(mp, dtype=np.uint8)
    mask[0] ^= 1
    mask.tofile(changed_mask)
    changed_cfg = dict(
        cfg,
        train_mask=str(changed_mask),
        out_dir=str(tmp_path / "changed-sidecar-out"),
    )
    with pytest.raises(ValueError, match="provenance"):
        Trainer(
            changed_cfg,
            resume="auto",
            resume_path=trainer.ckpt_path,
            resume_sha256=file_sha256(trainer.ckpt_path),
        )


def test_checkpoint_records_rng_state_by_rank(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()

    state = torch.load(trainer.ckpt_path, weights_only=False)
    assert len(state["rng_by_rank"]) == 1
    assert set(state["rng_by_rank"][0]) == {
        "python",
        "numpy",
        "torch",
        "cuda",
    }
    assert torch.equal(state["rng_by_rank"][0]["torch"], torch.get_rng_state())


@pytest.mark.parametrize(
    "case",
    [
        "top_extra",
        "version_bool",
        "world_bool",
        "step_bool",
        "cfg_extra",
        "model_extra",
        "optimizer_extra",
        "data_extra",
        "data_version_bool",
        "cursor_bool",
        "epoch_bool",
        "rng_extra",
        "cpu_cuda_rng",
    ],
)
def test_checkpoint_schema_is_exact_and_rejects_bool_integers(tmp_path, case):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    assert state["checkpoint_version"] == 3

    if case == "top_extra":
        state["unexpected"] = None
    elif case == "version_bool":
        state["checkpoint_version"] = True
    elif case == "world_bool":
        state["world_size"] = True
    elif case == "step_bool":
        state["step"] = False
    elif case == "cfg_extra":
        state["cfg"]["unexpected"] = None
    elif case == "model_extra":
        state["model"]["unexpected"] = torch.tensor(0)
    elif case == "optimizer_extra":
        state["opt"]["unexpected"] = None
    elif case == "data_extra":
        state["data"]["unexpected"] = None
    elif case == "data_version_bool":
        state["data"]["format_version"] = True
    elif case == "cursor_bool":
        state["data"]["cursor"] = False
    elif case == "epoch_bool":
        state["data"]["epoch"] = False
    elif case == "rng_extra":
        state["rng_by_rank"][0]["unexpected"] = None
    elif case == "cpu_cuda_rng":
        state["rng_by_rank"][0]["cuda"] = torch.zeros(8, dtype=torch.uint8)
    else:
        raise AssertionError(case)

    malformed = tmp_path / f"{case}.pt"
    torch.save(state, malformed)
    model_before = {
        name: value.detach().clone()
        for name, value in trainer._raw_model().state_dict().items()
    }
    with pytest.raises(ValueError):
        trainer.load_ckpt(malformed, sha256=file_sha256(malformed))

    assert trainer.step == 0
    assert trainer.data.global_cursor == 0
    assert all(
        torch.equal(value, model_before[name])
        for name, value in trainer._raw_model().state_dict().items()
    )
    trainer.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tokens_per_step", 257, "divisible"),
        ("micro_batch_size", 0, "micro_batch_size"),
    ],
)
def test_invalid_update_configs_are_rejected(tmp_path, field, value, message):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg[field] = value

    with pytest.raises(ValueError, match=message):
        Trainer(cfg)


def test_total_tokens_must_be_whole_optimizer_updates(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.pop("max_steps")
    cfg["total_tokens"] = cfg["tokens_per_step"] * 2 + 1

    with pytest.raises(ValueError, match="total_tokens"):
        Trainer(cfg)


def test_max_steps_must_exactly_match_total_tokens(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 2
    cfg["total_tokens"] = cfg["tokens_per_step"] * 3

    with pytest.raises(ValueError, match="max_steps.*total_tokens"):
        Trainer(cfg)


def test_micro_batch_size_is_a_cap_not_a_required_local_row_count(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "micro_batch_size": 32,
            "tokens_per_step": 8,
            "max_steps": 1,
        }
    )

    trainer = Trainer(cfg)

    assert trainer.local_sequences == 2
    assert trainer.accum == 1


def test_non_master_rank_does_not_write_snapshot_artifact(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.is_master = False

    trainer.save_snapshot()

    assert not (trainer.out_dir / "snapshots" / "step0000000.pt").exists()


def test_weighted_training_uses_existing_gpt_and_sidecar(tmp_path):
    bp, _ = write_corpus(tmp_path, n=1024)
    weights_path = tmp_path / "train.weights.bin"
    np.zeros(1024, dtype=np.uint8).tofile(weights_path)
    cfg = base_cfg(tmp_path, bp, None)
    cfg.update(
        {
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "d_model": 64,
                "ctx": 64,
                "vocab_size": 256,
            },
            "micro_batch_size": 2,
            "tokens_per_step": 2 * 64,
            "max_steps": 1,
            "train_weights": str(weights_path),
        }
    )
    trainer = Trainer(cfg)
    assert type(trainer.model) is GPT
    assert trainer.data.target_weights is not None
    assert trainer.train_steps(1) == 0.0
    checkpoint = torch.load(trainer.ckpt_path, weights_only=False)
    assert set(checkpoint) == {
        "model",
        "opt",
        "data",
        "step",
        "rng_by_rank",
        "cfg",
        "checkpoint_version",
        "config_fingerprint",
        "world_size",
        "data_provenance",
    }


def test_gpt_can_return_weighted_local_loss_sum_for_global_normalization():
    torch.manual_seed(3)
    model = GPT(
        GPTConfig(
            n_layer=1,
            n_head=1,
            d_model=16,
            ctx=4,
            vocab_size=32,
        )
    )
    x = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2, 4))
    weights = torch.linspace(0.25, 1.0, 8).reshape(2, 4)

    _, mean_loss = model(x, targets, target_weights=weights)
    _, loss_sum = model(
        x,
        targets,
        target_weights=weights,
        loss_reduction="sum",
    )

    assert torch.allclose(loss_sum, mean_loss * targets.numel())


def test_single_process_update_uses_final_partial_microbatch(tmp_path):
    bp, _ = write_corpus(tmp_path, n=1000)
    weights_path = tmp_path / "train.weights.bin"
    np.ones(1000, dtype=np.uint8).tofile(weights_path)
    cfg = base_cfg(tmp_path, bp, None)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 16,
                "ctx": 8,
                "vocab_size": 256,
            },
            "micro_batch_size": 3,
            "tokens_per_step": 5 * 8,
            "max_steps": 1,
            "train_weights": str(weights_path),
        }
    )

    trainer = Trainer(cfg)
    trainer.train_steps(1)

    assert trainer.accum == 2
    assert trainer.data.global_cursor == 40


def test_cosine_schedule():
    assert cosine_lr(0, 1.0, 10, 100) < 0.2
    assert abs(cosine_lr(10, 1.0, 10, 100) - 1.0) < 0.01
    assert cosine_lr(99, 1.0, 10, 100) < 0.2
    assert cosine_lr(150, 1.0, 10, 100) == 0.1
