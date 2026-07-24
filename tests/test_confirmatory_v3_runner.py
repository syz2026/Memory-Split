from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

import cluster.aws.gpu_profile as gpu_profile
from evals.confirmatory import sealing
from evals.confirmatory import __main__ as runner_cli
import evals.confirmatory.runner as runner_module
from evals.confirmatory.actions import ActionSlot
from evals.confirmatory.aggregate import plan_snapshot_evaluations
from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
    store_content_sha256,
)
from evals.confirmatory.fixtures import positive_fixture
from evals.confirmatory.runner import (
    DeterministicFixtureAdapter,
    Submission,
    evaluate,
    preflight,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
)
from msctl.aws_contracts import (
    ARMS,
    AWS_RUNTIME_VERSION_FIELDS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    snapshot_object_key,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
)
from train.tokenizer import get_tok
from train.trainer import Trainer


P5_PROFILE_SHA256 = (
    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
)


@dataclass(frozen=True)
class _V3Fixture:
    run: Path
    release: Path
    output: Path
    lock: StudyLockV3
    lock_sha256: str
    run_binding: dict
    item_ids: tuple[str, ...]
    submissions: dict[str, Submission]


def _jsonl(records: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _v3_source_bytes() -> tuple[bytes, bytes, bytes]:
    fixture = positive_fixture()
    items = [
        json.loads(line)
        for line in fixture.artifacts["items.jsonl"].splitlines()
    ]
    stores = [
        json.loads(line)
        for line in fixture.artifacts["stores.jsonl"].splitlines()
    ]
    gold = [
        json.loads(line)
        for line in fixture.artifacts["sealed-gold.jsonl"].splitlines()
    ]
    store_hashes = {}
    for store in stores:
        for row in store["rows"]:
            if row["relation_id"] == "P1":
                row["relation_id"] = "r0"
        store["content_sha256"] = store_content_sha256(
            store["store_id"],
            store["world_id"],
            store["rows"],
        )
        store_hashes[store["store_id"]] = store["content_sha256"]
    store_by_item = {item["item_id"]: item["store_id"] for item in items}
    for record in gold:
        for action in record["proof"]:
            if action["relation_id"] == "P1":
                action["relation_id"] = "r0"
        record["store_sha256"] = store_hashes[store_by_item[record["item_id"]]]
    return _jsonl(items), _jsonl(stores), _jsonl(gold)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_execution_inputs(
    run: Path,
    repository_root: Path,
) -> dict[str, str]:
    profile_data = repository_root.joinpath(
        "cluster/profiles/aws-p5.48xlarge-v3.json"
    ).read_bytes()
    run.joinpath("evaluator-profile.json").write_bytes(profile_data)
    profile_sha256 = hashlib.sha256(profile_data).hexdigest()
    runtime_facts = {
        field: f"test-{field}-1"
        for field in AWS_RUNTIME_VERSION_FIELDS
    }
    runtime_facts["pytorch"] = str(torch.__version__)
    runtime_facts["cuda"] = "12.8"
    runtime_lock = {
        "schema_version": 1,
        "source_commit": "2" * 40,
        "source_tree": "3" * 40,
        "control_bundle_sha256": "4" * 64,
        "profile_sha256": profile_sha256,
        "ami_id": "ami-0123456789abcdef0",
        "ami_owner_id": "123456789012",
        "container_image": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "memorysplit@sha256:" + "5" * 64
        ),
        "container_image_digest": "sha256:" + "5" * 64,
        "versions": runtime_facts,
    }
    runtime_lock_bytes = canonical_json_bytes(runtime_lock)
    run.joinpath("evaluator-runtime-lock.json").write_bytes(
        runtime_lock_bytes
    )
    runtime_lock_sha256 = hashlib.sha256(runtime_lock_bytes).hexdigest()
    identity = {
        "accountId": "123456789012",
        "architecture": "x86_64",
        "imageId": runtime_lock["ami_id"],
        "instanceId": "i-0123456789abcdef0",
        "privateIp": "10.23.45.67",
        "region": "us-east-1",
    }
    environment = {
        "schema_version": 2,
        "receipt_type": "memorysplit-aws-environment-v2",
        "provider": "aws-p5.48xlarge",
        "profile_sha256": profile_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "control_bundle_sha256": runtime_lock["control_bundle_sha256"],
        "source_commit": runtime_lock["source_commit"],
        "source_tree": runtime_lock["source_tree"],
        "container_image": runtime_lock["container_image"],
        "container_image_digest": runtime_lock[
            "container_image_digest"
        ],
        "aws_instance_identity_document": identity,
        "aws_instance_identity_pkcs7": base64.b64encode(
            b"signed-evaluator-instance"
        ).decode("ascii"),
        "account_id": identity["accountId"],
        "instance_id": identity["instanceId"],
        "region": identity["region"],
        "ami_id": identity["imageId"],
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "runtime_facts": runtime_facts,
    }
    environment_bytes = canonical_json_bytes(environment)
    run.joinpath("evaluator-environment-receipt.json").write_bytes(
        environment_bytes
    )
    return {
        "runtime_lock_sha256": runtime_lock_sha256,
        "environment_receipt_sha256": hashlib.sha256(
            environment_bytes
        ).hexdigest(),
    }


def _write_trainer_snapshot(
    root: Path,
    destination: Path,
) -> tuple[bytes, dict]:
    root.mkdir()
    tokens = np.zeros(524_288, dtype=np.uint16)
    mask = np.ones(524_288, dtype=np.uint8)
    token_path = root / "train.bin"
    mask_path = root / "train.mask.bin"
    tokens.tofile(token_path)
    mask.tofile(mask_path)
    tok = get_tok()
    response_ids = tuple(tok.encode(" candidate=done"))
    transitions: dict[int, int] = {}
    for source_id, target_id in zip(
        (tok.ANSWER_STATE, *response_ids),
        (*response_ids, tok.GRAPH_START),
        strict=True,
    ):
        previous = transitions.setdefault(source_id, target_id)
        assert previous == target_id
    model_config = {
        "n_layer": 0,
        "n_head": 1,
        "d_model": len(transitions),
        "vocab_size": 50304,
        "ctx": 512,
    }
    run_config = {
        "schema_version": 2,
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "run_id": "memorysplit-v3-360m-s0-dense",
        "condition": "dense",
        "model": model_config,
        "seed": 0,
        "train_bin": str(token_path),
        "train_mask": str(mask_path),
        "out_dir": str(root / "trainer-output"),
        "micro_batch_size": 2,
        "tokens_per_step": 524_288,
        "max_steps": 1_358,
        "lr": 1e-3,
        "warmup_steps": 5,
        "weight_decay": 0.1,
        "compile": False,
        "device": "cpu",
        "log_every": 20,
        "eval_every": 250,
        "snapshot_steps": [1_358],
        "ckpt_minutes": 999,
    }
    trainer = Trainer(run_config)
    trainer.data.provenance.update(
        {
            "receipt_sha256": _digest("data-receipt"),
            "build_id": _digest("data-build"),
            "ordered_stream_sha256": _digest("ordered-stream"),
        }
    )
    model = trainer._raw_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ln_f.weight.fill_(1.0)
        for dimension, (source_id, target_id) in enumerate(transitions.items()):
            model.wte.weight[source_id, dimension] = 1.0
            model.lm_head.weight[target_id, dimension] = 1.0
    trainer.step = 1_358
    trainer.world_size = 4
    trainer.save_snapshot()
    source = (
        trainer.out_dir / "snapshots" / "step0001358.pt"
    )
    content = source.read_bytes()
    state = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    trainer.close()
    destination.parent.mkdir()
    destination.write_bytes(content)
    return content, state


def _build_v3_fixture(tmp_path: Path) -> _V3Fixture:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    source.chmod(0o700)
    items_bytes, stores_bytes, gold_bytes = _v3_source_bytes()
    for name, content in (
        ("items.jsonl", items_bytes),
        ("stores.jsonl", stores_bytes),
        ("sealed-gold.jsonl", gold_bytes),
    ):
        source.joinpath(name).write_bytes(content)

    release_root = tmp_path / "releases"
    release_root.mkdir(mode=0o700)
    release_root.chmod(0o700)
    repository_root = Path(__file__).resolve().parents[1]
    sealed = sealing.seal_release(
        source_dir=source,
        preregistration_path=repository_root / "configs" / "preregistration-v3.yaml",
        output_root=release_root,
        apply=True,
    )

    run = tmp_path / "run"
    run.mkdir()
    execution = _write_execution_inputs(run, repository_root)
    snapshot_path = run / "snapshots" / "step0001358.pt"
    snapshot_bytes, snapshot_state = _write_trainer_snapshot(
        tmp_path / "snapshot-source",
        snapshot_path,
    )
    selected_snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    selection_sha256 = "a" * 64
    selection_version = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                selected = (seed, arm, step) == (0, "dense", 1_358)
                same_training_run = seed == 0 and arm == "dense"
                snapshot_sha256 = (
                    selected_snapshot_sha256
                    if selected
                    else _digest(f"snapshot:{seed}:{arm}:{step}")
                )
                receipt_sha256 = _digest(f"receipt:{seed}:{step}")
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": snapshot_sha256,
                        "s3_object_key": snapshot_object_key(
                            seed,
                            arm,
                            step,
                            snapshot_sha256,
                        ),
                        "s3_version_id": f"checkpoint-version-{seed}-{arm}-{step}",
                        "checkpoint_receipt_sha256": receipt_sha256,
                        "checkpoint_receipt_s3_object_key": (
                            checkpoint_receipt_key(seed, receipt_sha256)
                        ),
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}-{step}"
                        ),
                        "provider_selection_sha256": selection_sha256,
                        "provider_selection_s3_version_id": selection_version,
                        "snapshot_version": 2,
                        "training_run_id": (
                            f"memorysplit-v3-360m-s{seed}-{arm}"
                        ),
                        "config_fingerprint": (
                            snapshot_state["config_fingerprint"]
                            if same_training_run
                            else _digest(f"config:{seed}:{arm}")
                        ),
                        "training_config_sha256": (
                            snapshot_state["study_identity"][
                                "config_sha256"
                            ]
                            if same_training_run
                            else _digest(f"config-bytes:{seed}:{arm}")
                        ),
                        "model_config_sha256": snapshot_state[
                            "study_identity"
                        ]["model_cfg_sha256"],
                        "model_identity": snapshot_state["study_identity"][
                            "model_identity"
                        ],
                        "data_provenance_sha256": (
                            snapshot_state["study_identity"][
                                "data_provenance_sha256"
                            ]
                            if same_training_run
                            else _digest(f"data:{seed}:{arm}")
                        ),
                        "data_receipt_sha256": _digest("data-receipt"),
                        "data_build_id": _digest("data-build"),
                        "ordered_stream_sha256": _digest(
                            "ordered-stream"
                        ),
                        "world_size": 4,
                        "tokens_per_step": 524_288,
                    }
                )
    lock = StudyLockV3.from_dict(
        {
            "record_type": STUDY_LOCK_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
            "sealed_evaluation_release_sha256": sealed.release_sha256,
            "provider_selection": {
                "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
                "provider_selection_s3_key": PROVIDER_SELECTION_S3_KEY,
                "provider_selection_sha256": selection_sha256,
                "provider_selection_s3_version_id": selection_version,
                "hardware_amendment_sha256": AWS_HARDWARE_AMENDMENT_SHA256,
                "selected_provider": "aws-p5.48xlarge",
                "profile_id": "aws-p5.48xlarge-v3",
                "profile_sha256": P5_PROFILE_SHA256,
                "runtime_lock_sha256": execution[
                    "runtime_lock_sha256"
                ],
                "qualification_evidence_sha256": "c" * 64,
                "environment_receipt_sha256": execution[
                    "environment_receipt_sha256"
                ],
                "canary_receipt_sha256": "e" * 64,
                "approval_receipt_sha256": "f" * 64,
                "approval_public_key_sha256": "1" * 64,
            },
            "snapshots": snapshots,
        }
    )
    lock_sha256 = canonical_sha256(lock.to_dict())
    run.joinpath("study-lock.json").write_bytes(
        canonical_json_bytes(lock.to_dict())
    )
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
    )[0]
    run.joinpath("run.json").write_bytes(plan.run_json_bytes)

    gold_records = [json.loads(line) for line in gold_bytes.splitlines()]
    submissions = {
        record["item_id"]: Submission(
            item_id=record["item_id"],
            answer=record["answer"],
            actions=tuple(
                ActionSlot.from_dict(action) for action in record["proof"]
            ),
        )
        for record in gold_records
    }
    item_ids = tuple(
        record["item_id"]
        for record in (
            json.loads(line) for line in items_bytes.splitlines()
        )
    )
    return _V3Fixture(
        run=run,
        release=sealed.release_dir,
        output=tmp_path / plan.output_name,
        lock=lock,
        lock_sha256=lock_sha256,
        run_binding=plan.binding.to_dict(),
        item_ids=item_ids,
        submissions=submissions,
    )


def _rewrite_run(fixture: _V3Fixture, **changes) -> None:
    value = dict(fixture.run_binding)
    value.update(changes)
    fixture.run.joinpath("run.json").write_bytes(canonical_json_bytes(value))


def _rebind_snapshot(fixture: _V3Fixture, state: dict) -> _V3Fixture:
    path = fixture.run.joinpath(fixture.run_binding["snapshot_path"])
    torch.save(state, path)
    snapshot_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    raw_lock = fixture.lock.to_dict()
    selected = raw_lock["snapshots"][0]
    assert (selected["seed"], selected["arm"], selected["optimizer_step"]) == (
        0,
        "dense",
        1_358,
    )
    selected["checkpoint_sha256"] = snapshot_sha256
    selected["s3_object_key"] = snapshot_object_key(
        0,
        "dense",
        1_358,
        snapshot_sha256,
    )
    lock = StudyLockV3.from_dict(raw_lock)
    lock_sha256 = canonical_sha256(lock.to_dict())
    fixture.run.joinpath("study-lock.json").write_bytes(
        canonical_json_bytes(lock.to_dict())
    )
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
    )[0]
    fixture.run.joinpath("run.json").write_bytes(plan.run_json_bytes)
    return replace(
        fixture,
        lock=lock,
        lock_sha256=lock_sha256,
        run_binding=plan.binding.to_dict(),
        output=fixture.output.parent / plan.output_name,
    )


def _rebind_environment_receipt(
    fixture: _V3Fixture,
    environment: dict,
) -> _V3Fixture:
    path = fixture.run / "evaluator-environment-receipt.json"
    content = canonical_json_bytes(environment)
    path.write_bytes(content)
    environment_sha256 = hashlib.sha256(content).hexdigest()
    raw_lock = fixture.lock.to_dict()
    raw_lock["provider_selection"][
        "environment_receipt_sha256"
    ] = environment_sha256
    lock = StudyLockV3.from_dict(raw_lock)
    lock_sha256 = canonical_sha256(lock.to_dict())
    fixture.run.joinpath("study-lock.json").write_bytes(
        canonical_json_bytes(lock.to_dict())
    )
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
    )[0]
    fixture.run.joinpath("run.json").write_bytes(plan.run_json_bytes)
    return replace(
        fixture,
        lock=lock,
        lock_sha256=lock_sha256,
        run_binding=plan.binding.to_dict(),
        output=fixture.output.parent / plan.output_name,
    )


def test_v3_preflight_uses_only_one_snapshot_and_the_sealed_release(tmp_path):
    fixture = _build_v3_fixture(tmp_path)

    result = preflight(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.lock_sha256,
    )

    assert {path.name for path in fixture.run.iterdir()} == {
        "evaluator-environment-receipt.json",
        "evaluator-profile.json",
        "evaluator-runtime-lock.json",
        "run.json",
        "snapshots",
        "study-lock.json",
    }
    assert {
        path.name for path in fixture.run.joinpath("snapshots").iterdir()
    } == {"step0001358.pt"}
    assert {path.name for path in fixture.release.iterdir()} == {
        "items.jsonl",
        "stores.jsonl",
        "sealed-gold.jsonl",
        "sealed-release.json",
    }
    assert result.seed == 0
    assert result.condition_id == "dense"
    assert result.optimizer_step == 1_358
    assert result.snapshot_sha256 == fixture.run_binding["snapshot_sha256"]
    assert result.output_id == fixture.run_binding["output_id"]
    assert result.selected_provider == "aws-p5.48xlarge"
    assert result.item_count == len(fixture.item_ids)


def test_v3_preflight_rejects_optimizer_rng_full_checkpoints(tmp_path):
    fixture = _build_v3_fixture(tmp_path)
    path = fixture.run.joinpath(fixture.run_binding["snapshot_path"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    state.update(
        {
            "cfg": {"condition": "dense", "seed": 0},
            "data": {},
            "opt": {},
            "rng_by_rank": [],
        }
    )
    fixture = _rebind_snapshot(fixture, state)

    with pytest.raises(ValueError, match="model.only|snapshot.*fields|full"):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("step", "step"),
        ("arm", "arm"),
        ("config", "config"),
        ("model_cfg", "model.*config"),
        ("data", "data.*provenance"),
        ("world_size", "world.size"),
    ],
)
def test_v3_preflight_rejects_crossed_snapshot_provenance(
    tmp_path,
    mutation,
    message,
):
    fixture = _build_v3_fixture(tmp_path)
    path = fixture.run.joinpath(fixture.run_binding["snapshot_path"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "step":
        state["step"] = 3_396
    elif mutation == "arm":
        state["study_identity"]["arm"] = "split90"
    elif mutation == "config":
        state["config_fingerprint"] = "0" * 64
    elif mutation == "model_cfg":
        state["model_cfg"]["ctx"] += 1
    elif mutation == "data":
        state["data_provenance"]["token_count"] += 1
    else:
        state["world_size"] = 8
    fixture = _rebind_snapshot(fixture, state)

    with pytest.raises(ValueError, match=message):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )


@pytest.mark.parametrize(
    ("member", "attack"),
    [
        ("run.json", "symlink"),
        ("run.json", "hardlink"),
        ("run.json", "unsafe_mode"),
        ("study-lock.json", "symlink"),
        ("study-lock.json", "hardlink"),
        ("study-lock.json", "unsafe_mode"),
        ("snapshot", "symlink"),
        ("snapshot", "hardlink"),
        ("snapshot", "unsafe_mode"),
    ],
)
def test_v3_preflight_rejects_unsafe_bound_input_files(
    tmp_path,
    member,
    attack,
):
    fixture = _build_v3_fixture(tmp_path)
    path = (
        fixture.run.joinpath(fixture.run_binding["snapshot_path"])
        if member == "snapshot"
        else fixture.run / member
    )
    if attack == "symlink":
        displaced = path.with_name(path.name + ".displaced")
        path.rename(displaced)
        path.symlink_to(displaced)
    elif attack == "hardlink":
        os.link(path, path.with_name(path.name + ".alias"))
    else:
        path.chmod(0o666)

    with pytest.raises(ValueError, match="symlink|link|mode|writable|unsafe"):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )


def test_v3_preflight_rejects_same_bytes_path_replacement_during_read(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    target = fixture.run / "run.json"
    target_inode = target.stat().st_ino
    content = target.read_bytes()
    displaced = fixture.run / "run.displaced"
    original_read = gpu_profile.os.read
    replaced = False

    def replace_after_read(descriptor, count):
        nonlocal replaced
        chunk = original_read(descriptor, count)
        if (
            not replaced
            and os.fstat(descriptor).st_ino == target_inode
            and chunk
        ):
            replaced = True
            target.rename(displaced)
            target.write_bytes(content)
        return chunk

    monkeypatch.setattr(gpu_profile.os, "read", replace_after_read)

    with pytest.raises(ValueError, match="changed|replaced|identity"):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )
    assert replaced
    assert displaced.read_bytes() == content
    assert target.read_bytes() == content


def test_v3_preflight_bounds_run_binding_bytes(tmp_path):
    fixture = _build_v3_fixture(tmp_path)
    fixture.run.joinpath("run.json").write_bytes(b"x" * 262_145)

    with pytest.raises(ValueError, match="byte limit|exceeds|too large"):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )


def test_v3_run_identity_dispatches_without_filename_markers(tmp_path):
    fixture = _build_v3_fixture(tmp_path)
    manifest = fixture.release / "sealed-release.json"
    manifest.rename(fixture.release / "renamed-manifest.json")

    with pytest.raises(ValueError) as caught:
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )
    assert "sealed release" in str(caught.value)
    assert "study-lock.json is missing" not in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    ["wrong-schema", "duplicate-record-type", "unknown-field"],
)
def test_v3_dispatch_rejects_forged_or_ambiguous_run_identity(
    tmp_path,
    mutation,
):
    fixture = _build_v3_fixture(tmp_path)
    path = fixture.run / "run.json"
    raw = dict(fixture.run_binding)
    if mutation == "wrong-schema":
        raw["schema_version"] = 2
        content = canonical_json_bytes(raw)
    elif mutation == "unknown-field":
        raw["v2_record_type"] = "memorysplit.confirmatory.run-binding.v2"
        content = canonical_json_bytes(raw)
    else:
        canonical = canonical_json_bytes(raw).decode("utf-8")
        content = canonical.replace(
            '"record_type":',
            (
                '"record_type":"memorysplit.confirmatory.run-binding.v2",'
                '"record_type":'
            ),
            1,
        ).encode("utf-8")
    path.write_bytes(content)

    with pytest.raises(ValueError, match="run|schema|fields|canonical|exact"):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )


def test_v3_cli_dry_run_reports_step_profile_and_output_identity(
    tmp_path,
    capsys,
):
    fixture = _build_v3_fixture(tmp_path)

    return_code = runner_cli.main(
        [
            "evaluate",
            "--run",
            str(fixture.run),
            "--sealed-release",
            str(fixture.release),
            "--expected-study-lock-sha256",
            fixture.lock_sha256,
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    payload = json.loads(captured.out)
    assert payload["optimizer_step"] == 1_358
    assert payload["output_id"] == fixture.run_binding["output_id"]
    assert payload["selected_provider"] == "aws-p5.48xlarge"
    assert payload["condition_id"] == "dense"


def test_v3_cli_provider_qualified_flag_enforces_publication_gate(
    tmp_path,
    capsys,
):
    fixture = _build_v3_fixture(tmp_path)

    return_code = runner_cli.main(
        [
            "evaluate",
            "--run",
            str(fixture.run),
            "--sealed-release",
            str(fixture.release),
            "--expected-study-lock-sha256",
            fixture.lock_sha256,
            "--device",
            "cuda",
            "--output-dir",
            str(fixture.output),
            "--provider-qualified",
        ],
        model_adapter=DeterministicFixtureAdapter(fixture.submissions),
    )

    payload = json.loads(capsys.readouterr().out)
    assert return_code != 0
    assert payload["status"] == "error"
    assert "adapter" in payload["error"] or "fixture" in payload["error"]
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "snapshot_s3_version_id",
            "other-checkpoint-version",
            "checkpoint|snapshot|version",
        ),
        (
            "checkpoint_receipt_s3_version_id",
            "other-receipt-version",
            "receipt|snapshot|version",
        ),
        ("config_fingerprint", "0" * 64, "config"),
        ("model_config_sha256", "0" * 64, "model.*config"),
        (
            "data_provenance_sha256",
            "0" * 64,
            "data.*provenance",
        ),
        (
            "sealed_evaluation_release_sha256",
            "0" * 64,
            "sealed.*release|commitment",
        ),
        ("study_lock_sha256", "0" * 64, "study.lock"),
        ("provider_selection_sha256", "0" * 64, "provider.*selection"),
        (
            "evaluator_runtime_lock_sha256",
            "0" * 64,
            "runtime|provider.*selection",
        ),
        (
            "evaluator_qualification_evidence_sha256",
            "0" * 64,
            "qualification|provider.*selection",
        ),
        ("output_id", "wrong-output", "output"),
    ],
)
def test_v3_preflight_rejects_every_crossed_run_binding(
    tmp_path,
    field,
    value,
    message,
):
    fixture = _build_v3_fixture(tmp_path)
    _rewrite_run(fixture, **{field: value})

    with pytest.raises(ValueError, match=message):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )


def test_v3_evaluate_keeps_gold_unopened_until_all_submissions_complete(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    delegate = DeterministicFixtureAdapter(fixture.submissions)
    submitted = []
    gold_opened = False
    original = sealing._open_pinned_file

    def record_gold(parent_fd, name, label):
        nonlocal gold_opened
        if name == "sealed-gold.jsonl":
            gold_opened = True
        return original(parent_fd, name, label)

    class GuardedAdapter:
        def generate(self, item, store):
            assert not gold_opened
            submitted.append(item.item_id)
            return delegate.generate(item, store)

    monkeypatch.setattr(sealing, "_open_pinned_file", record_gold)

    result = evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.lock_sha256,
        output_dir=fixture.output,
        model_adapter=GuardedAdapter(),
    )

    assert tuple(submitted) == fixture.item_ids
    assert gold_opened
    assert result.optimizer_step == 1_358
    assert result.output_id == fixture.run_binding["output_id"]


def test_v3_evaluate_rechecks_snapshot_before_opening_gold(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    delegate = DeterministicFixtureAdapter(fixture.submissions)
    submitted = 0
    gold_opened = False
    original = sealing._open_pinned_file

    def record_gold(parent_fd, name, label):
        nonlocal gold_opened
        if name == "sealed-gold.jsonl":
            gold_opened = True
        return original(parent_fd, name, label)

    class MutatingAdapter:
        def generate(self, item, store):
            nonlocal submitted
            submitted += 1
            result = delegate.generate(item, store)
            if submitted == len(fixture.item_ids):
                fixture.run.joinpath(
                    fixture.run_binding["snapshot_path"]
                ).write_bytes(b"drift")
            return result

    monkeypatch.setattr(sealing, "_open_pinned_file", record_gold)

    with pytest.raises(
        ValueError,
        match="model snapshot.*changed|snapshot.*hash",
    ):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            model_adapter=MutatingAdapter(),
        )
    assert not gold_opened
    assert not fixture.output.exists()


@pytest.mark.parametrize("name", ["run.json", "study-lock.json"])
def test_v3_evaluate_rechecks_control_bindings_before_opening_gold(
    tmp_path,
    monkeypatch,
    name,
):
    fixture = _build_v3_fixture(tmp_path)
    delegate = DeterministicFixtureAdapter(fixture.submissions)
    submitted = 0
    gold_opened = False
    original = sealing._open_pinned_file

    def record_gold(parent_fd, member, label):
        nonlocal gold_opened
        if member == "sealed-gold.jsonl":
            gold_opened = True
        return original(parent_fd, member, label)

    class MutatingAdapter:
        def generate(self, item, store):
            nonlocal submitted
            submitted += 1
            result = delegate.generate(item, store)
            if submitted == len(fixture.item_ids):
                fixture.run.joinpath(name).write_bytes(b"drift")
            return result

    monkeypatch.setattr(sealing, "_open_pinned_file", record_gold)

    with pytest.raises(ValueError, match="run.json|study-lock.json"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            model_adapter=MutatingAdapter(),
        )
    assert not gold_opened
    assert not fixture.output.exists()


def test_v3_evaluate_publishes_bound_snapshot_evidence_without_conclusion(
    tmp_path,
):
    fixture = _build_v3_fixture(tmp_path)

    result = evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.lock_sha256,
        output_dir=fixture.output,
        model_adapter=DeterministicFixtureAdapter(fixture.submissions),
    )

    assert result.output_dir == fixture.output
    assert {path.name for path in fixture.output.iterdir()} == {
        "inference.json",
        "items.jsonl",
        "metrics.json",
        "outcomes.jsonl",
        "output.json",
        "run.json",
        "sealed-gold.jsonl",
        "sealed-release.json",
        "stores.jsonl",
        "study-lock.json",
    }
    assert fixture.output.joinpath("run.json").read_bytes() == (
        fixture.run.joinpath("run.json").read_bytes()
    )
    outcomes = [
        json.loads(line)
        for line in fixture.output.joinpath("outcomes.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert tuple(row["item_id"] for row in outcomes) == fixture.item_ids
    assert all(row["schema_version"] == 3 for row in outcomes)
    assert all(row["optimizer_step"] == 1_358 for row in outcomes)
    assert all(row["arm"] == "dense" for row in outcomes)

    inference = json.loads(
        fixture.output.joinpath("inference.json").read_bytes()
    )
    assert inference["scope"] == "cohort"
    assert inference["final_conclusion"] is None
    assert inference["cohort_aggregation_status"] == "not_implemented"
    assert "supports_effect" not in inference
    assert "supports_practical_null" not in inference

    manifest = json.loads(fixture.output.joinpath("output.json").read_bytes())
    assert manifest["output_id"] == fixture.run_binding["output_id"]
    assert manifest["study_lock_sha256"] == fixture.lock_sha256
    assert (
        manifest["sealed_evaluation_release_sha256"]
        == fixture.lock.sealed_evaluation_release_sha256
    )
    assert manifest["provider_selection_sha256"] == "a" * 64
    assert manifest["provider_selection_s3_version_id"] == (
        "provider-selection-version-p5"
    )
    assert manifest["final_conclusion"] is None
    assert manifest["production_qualified"] is False
    assert manifest["publication_class"] == "test_only"
    assert manifest["execution_identity_before"]["device_type"] == "cpu"
    assert manifest["execution_identity_after"]["device_type"] == "cpu"
    assert manifest["execution_identity_before"]["adapter_kind"] == (
        "fixture"
    )
    assert result.production_qualified is False
    assert result.authoritative_commitment == hashlib.sha256(
        fixture.output.joinpath("output.json").read_bytes()
    ).hexdigest()
    assert result.path_authority == "informational_reopen_required"
    assert [artifact["path"] for artifact in manifest["artifacts"]] == sorted(
        artifact["path"] for artifact in manifest["artifacts"]
    )
    for artifact in manifest["artifacts"]:
        content = fixture.output.joinpath(artifact["path"]).read_bytes()
        assert artifact["bytes"] == len(content)
        assert artifact["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    ("device", "adapter", "message"),
    [
        ("cpu", None, "cuda"),
        ("mps", None, "cuda"),
        ("cuda", "fixture", "fixture|adapter"),
    ],
)
def test_provider_qualified_publication_rejects_nonproduction_execution(
    tmp_path,
    device,
    adapter,
    message,
):
    fixture = _build_v3_fixture(tmp_path)
    model_adapter = (
        DeterministicFixtureAdapter(fixture.submissions)
        if adapter == "fixture"
        else None
    )

    with pytest.raises(ValueError, match=message):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            model_adapter=model_adapter,
            device=device,
            provider_qualified=True,
            execution_identity_verifier=lambda *_args: True,
            execution_probe=lambda _device: {
                "device_type": device,
                "cuda_available": device == "cuda",
                "torch_version": str(torch.__version__),
                "cuda_version": "12.8",
                "device_count": 8,
                "device_name": "NVIDIA H100 80GB",
                "device_capability": [9, 0],
            },
        )
    assert not fixture.output.exists()


def _patch_production_repository_adapter(monkeypatch, fixture):
    delegate = DeterministicFixtureAdapter(fixture.submissions)
    adapter = object.__new__(runner_module.RepositoryGPTAdapter)
    adapter.generate = delegate.generate
    monkeypatch.setattr(
        runner_module.RepositoryGPTAdapter,
        "from_bound_run",
        classmethod(lambda _cls, _run, _binding, _device: adapter),
    )


def _qualified_cuda_probe(_device):
    return {
        "device_type": "cuda",
        "cuda_available": True,
        "torch_version": str(torch.__version__),
        "cuda_version": "12.8",
        "device_count": 8,
        "device_name": "NVIDIA H100 80GB",
        "device_capability": [9, 0],
    }


def test_provider_qualified_output_reauthenticates_and_binds_execution(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    _patch_production_repository_adapter(monkeypatch, fixture)
    verification_calls = 0

    def verify_identity(_identity, _pkcs7, _region):
        nonlocal verification_calls
        verification_calls += 1
        return True

    result = evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.lock_sha256,
        output_dir=fixture.output,
        device="cuda",
        provider_qualified=True,
        execution_identity_verifier=verify_identity,
        execution_probe=_qualified_cuda_probe,
    )

    manifest = json.loads(fixture.output.joinpath("output.json").read_bytes())
    assert verification_calls == 2
    assert result.production_qualified is True
    assert manifest["production_qualified"] is True
    assert manifest["publication_class"] == "provider_qualified"
    assert manifest["execution_identity_before"] == (
        manifest["execution_identity_after"]
    )
    execution = manifest["execution_identity_before"]
    assert execution["device_type"] == "cuda"
    assert execution["torch_version"] == str(torch.__version__)
    assert execution["cuda_version"] == "12.8"
    assert execution["environment_receipt_sha256"] == (
        fixture.run_binding["evaluator_environment_receipt_sha256"]
    )
    assert execution["profile_sha256"] == P5_PROFILE_SHA256


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "account", "boot"],
)
def test_provider_qualified_output_rejects_invalid_aws_identity_fields(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture = _build_v3_fixture(tmp_path)
    path = fixture.run / "evaluator-environment-receipt.json"
    environment = json.loads(path.read_bytes())
    if mutation == "unknown":
        environment["aws_instance_identity_document"]["unexpected"] = "forged"
    elif mutation == "account":
        environment["aws_instance_identity_document"]["accountId"] = "bad"
        environment["account_id"] = "bad"
    else:
        environment["boot_id"] = "not-a-boot-id"
    fixture = _rebind_environment_receipt(fixture, environment)
    _patch_production_repository_adapter(monkeypatch, fixture)

    with pytest.raises(ValueError, match="AWS identity|identity.*fields"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            device="cuda",
            provider_qualified=True,
            execution_identity_verifier=lambda *_args: True,
            execution_probe=_qualified_cuda_probe,
        )
    assert not fixture.output.exists()


def test_provider_qualified_output_rejects_cross_profile_cuda_device(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    _patch_production_repository_adapter(monkeypatch, fixture)
    wrong_device = dict(_qualified_cuda_probe("cuda"))
    wrong_device["device_name"] = "NVIDIA A100"

    with pytest.raises(ValueError, match="cuda|device|profile"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            device="cuda",
            provider_qualified=True,
            execution_identity_verifier=lambda *_args: True,
            execution_probe=lambda _device: wrong_device,
        )
    assert not fixture.output.exists()


def test_provider_qualified_output_fails_if_reauthentication_changes(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    _patch_production_repository_adapter(monkeypatch, fixture)
    verification_calls = 0
    gold_opened = False
    original = sealing._open_pinned_file

    def verify_identity(_identity, _pkcs7, _region):
        nonlocal verification_calls
        verification_calls += 1
        return verification_calls == 1

    def record_gold(parent_fd, name, label):
        nonlocal gold_opened
        if name == "sealed-gold.jsonl":
            gold_opened = True
        return original(parent_fd, name, label)

    monkeypatch.setattr(sealing, "_open_pinned_file", record_gold)

    with pytest.raises(ValueError, match="execution|identity|signature"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            device="cuda",
            provider_qualified=True,
            execution_identity_verifier=verify_identity,
            execution_probe=_qualified_cuda_probe,
        )
    assert verification_calls == 2
    assert not gold_opened
    assert not fixture.output.exists()


def test_v3_evaluate_loads_a_real_trainer_snapshot_without_config_file(
    tmp_path,
):
    fixture = _build_v3_fixture(tmp_path)

    result = evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.lock_sha256,
        output_dir=fixture.output,
        device="cpu",
    )

    assert result.optimizer_step == 1_358
    outcomes = [
        json.loads(line)
        for line in fixture.output.joinpath("outcomes.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert outcomes
    assert all(row["submitted_answer"] == "done" for row in outcomes)


def test_v3_evaluate_rejects_noncanonical_output_name_before_inference(tmp_path):
    fixture = _build_v3_fixture(tmp_path)

    class ForbiddenAdapter:
        def generate(self, item, store):
            raise AssertionError("invalid output identity reached inference")

    with pytest.raises(ValueError, match="output.*identity|output.*name"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=tmp_path / "other-output",
            model_adapter=ForbiddenAdapter(),
        )


def test_v3_output_rejects_parent_replacement_during_publication(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir(mode=0o700)
    output_parent.chmod(0o700)
    output = output_parent / fixture.run_binding["output_id"]
    displaced = tmp_path / "output-parent-displaced"
    original_mkdtemp = tempfile.mkdtemp
    swapped = False

    def swap_parent(*args, **kwargs):
        nonlocal swapped
        if not swapped and Path(kwargs["dir"]) == output_parent:
            swapped = True
            output_parent.rename(displaced)
            output_parent.mkdir(mode=0o700)
            output_parent.chmod(0o700)
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(runner_module.tempfile, "mkdtemp", swap_parent)

    with pytest.raises(ValueError, match="output parent.*changed|replaced|identity"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=output,
            model_adapter=DeterministicFixtureAdapter(fixture.submissions),
        )
    assert swapped
    assert not output.exists()
    assert displaced.is_dir()


def test_v3_output_quarantines_installed_tree_on_final_identity_failure(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    original = runner_module._assert_output_parent
    injected = False

    def fail_after_install(path, descriptor, expected):
        nonlocal injected
        original(path, descriptor, expected)
        if not injected and fixture.output.exists():
            injected = True
            raise ValueError("injected final output identity failure")

    monkeypatch.setattr(
        runner_module,
        "_assert_output_parent",
        fail_after_install,
    )

    with pytest.raises(ValueError, match="injected final output identity"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            model_adapter=DeterministicFixtureAdapter(fixture.submissions),
        )
    quarantines = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(
            f".{fixture.output.name}.confirmatory-v3-quarantine-"
        )
    ]
    assert injected
    assert not fixture.output.exists()
    assert len(quarantines) == 1
    assert quarantines[0].joinpath("output.json").is_file()


def test_v3_output_final_authority_rejects_installed_name_replacement(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    replacement_payload = b"replacement owner"
    displaced_name = "displaced-installed-output"

    def replace_after_prior_checks(event, **context):
        if event != "before_final_output_authority":
            return
        parent_fd = context["parent_fd"]
        output_name = context["output_name"]
        os.rename(
            output_name,
            displaced_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(output_name, 0o700, dir_fd=parent_fd)
        replacement_fd = os.open(
            output_name,
            os.O_RDONLY | os.O_DIRECTORY,
            dir_fd=parent_fd,
        )
        try:
            descriptor = os.open(
                "sentinel",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_fd,
            )
            try:
                os.write(descriptor, replacement_payload)
            finally:
                os.close(descriptor)
        finally:
            os.close(replacement_fd)

    monkeypatch.setattr(
        runner_module,
        "_run_output_mutation_hook",
        replace_after_prior_checks,
        raising=False,
    )

    with pytest.raises(ValueError, match="output.*replaced|final.*authority"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            model_adapter=DeterministicFixtureAdapter(fixture.submissions),
        )
    assert fixture.output.joinpath("sentinel").read_bytes() == (
        replacement_payload
    )
    quarantines = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(
            f".{fixture.output.name}.confirmatory-v3-quarantine-"
        )
    ]
    assert len(quarantines) == 1
    assert quarantines[0].joinpath("output.json").is_file()


def test_v3_output_failure_quarantines_without_pathname_deletion(
    tmp_path,
    monkeypatch,
):
    fixture = _build_v3_fixture(tmp_path)
    original = runner_module._publish_file_at
    writes = 0

    def fail_third_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected v3 output write failure")
        return original(*args, **kwargs)

    def forbid_deletion(*_args, **_kwargs):
        raise AssertionError("v3 output authority must not delete pathnames")

    monkeypatch.setattr(runner_module, "_publish_file_at", fail_third_write)
    monkeypatch.setattr(runner_module.os, "unlink", forbid_deletion)
    monkeypatch.setattr(runner_module.os, "rmdir", forbid_deletion)

    with pytest.raises(OSError, match="injected v3 output write failure"):
        evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
            output_dir=fixture.output,
            model_adapter=DeterministicFixtureAdapter(fixture.submissions),
        )
    quarantines = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(
            f".{fixture.output.name}.confirmatory-v3-quarantine-"
        )
    ]
    assert not fixture.output.exists()
    assert len(quarantines) == 1
    assert {path.name for path in quarantines[0].iterdir()} == {
        "inference.json",
        "items.jsonl",
    }


def test_v3_preflight_rejects_model_visible_release_drift_without_gold(tmp_path):
    fixture = _build_v3_fixture(tmp_path)
    items = fixture.release / "items.jsonl"
    items.write_bytes(items.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="items|hash|commitment|canonical"):
        preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.lock_sha256,
        )
