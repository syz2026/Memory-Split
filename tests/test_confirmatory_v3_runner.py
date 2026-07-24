from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest
import torch

from evals.confirmatory import sealing
from evals.confirmatory import __main__ as runner_cli
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
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_object_key,
    checkpoint_receipt_key,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
)
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok


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


def _write_tiny_checkpoint(path: Path) -> bytes:
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
        "condition": "dense",
        "model": model_config,
        "seed": 0,
    }
    model = GPT(GPTConfig(**model_config))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ln_f.weight.fill_(1.0)
        for dimension, (source_id, target_id) in enumerate(transitions.items()):
            model.wte.weight[source_id, dimension] = 1.0
            model.lm_head.weight[target_id, dimension] = 1.0
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": run_config,
            "step": 1_358,
        },
        path,
    )
    return path.read_bytes()


def _build_v3_fixture(
    tmp_path: Path,
    *,
    real_checkpoint: bool = False,
) -> _V3Fixture:
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
    checkpoint_path = run / "checkpoint.pt"
    checkpoint_bytes = (
        _write_tiny_checkpoint(checkpoint_path)
        if real_checkpoint
        else b"selected isolated checkpoint\n"
    )
    if not real_checkpoint:
        checkpoint_path.write_bytes(checkpoint_bytes)
    selected_checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    selection_sha256 = "a" * 64
    selection_version = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                selected = (seed, arm, step) == (0, "dense", 1_358)
                checkpoint_sha256 = (
                    selected_checkpoint_sha256
                    if selected
                    else _digest(f"checkpoint:{seed}:{arm}:{step}")
                )
                receipt_sha256 = _digest(f"receipt:{seed}:{step}")
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": checkpoint_sha256,
                        "s3_object_key": checkpoint_object_key(
                            seed,
                            arm,
                            checkpoint_sha256,
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
                "runtime_lock_sha256": "b" * 64,
                "qualification_evidence_sha256": "c" * 64,
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


def test_v3_preflight_uses_only_one_checkpoint_and_the_sealed_release(tmp_path):
    fixture = _build_v3_fixture(tmp_path)

    result = preflight(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.lock_sha256,
    )

    assert {path.name for path in fixture.run.iterdir()} == {
        "checkpoint.pt",
        "run.json",
        "study-lock.json",
    }
    assert {path.name for path in fixture.release.iterdir()} == {
        "items.jsonl",
        "stores.jsonl",
        "sealed-gold.jsonl",
        "sealed-release.json",
    }
    assert result.seed == 0
    assert result.condition_id == "dense"
    assert result.optimizer_step == 1_358
    assert result.output_id == fixture.run_binding["output_id"]
    assert result.selected_provider == "aws-p5.48xlarge"
    assert result.item_count == len(fixture.item_ids)


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "checkpoint_s3_version_id",
            "other-checkpoint-version",
            "checkpoint|snapshot|version",
        ),
        (
            "checkpoint_receipt_s3_version_id",
            "other-receipt-version",
            "receipt|snapshot|version",
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


def test_v3_evaluate_rechecks_checkpoint_before_opening_gold(
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
                fixture.run.joinpath("checkpoint.pt").write_bytes(b"drift")
            return result

    monkeypatch.setattr(sealing, "_open_pinned_file", record_gold)

    with pytest.raises(ValueError, match="checkpoint.*changed|checkpoint.*hash"):
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
    assert [artifact["path"] for artifact in manifest["artifacts"]] == sorted(
        artifact["path"] for artifact in manifest["artifacts"]
    )
    for artifact in manifest["artifacts"]:
        content = fixture.output.joinpath(artifact["path"]).read_bytes()
        assert artifact["bytes"] == len(content)
        assert artifact["sha256"] == hashlib.sha256(content).hexdigest()


def test_v3_evaluate_loads_a_real_checkpoint_without_a_config_file(tmp_path):
    fixture = _build_v3_fixture(tmp_path, real_checkpoint=True)

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
