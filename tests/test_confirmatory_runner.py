from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    SEALED_GOLD_SCHEMA,
    STORE_SCHEMA,
    ItemRecord,
    canonical_json_bytes,
    store_content_sha256,
)


def _runner():
    return importlib.import_module("evals.confirmatory.runner")


def _noop(op: str = "noop") -> dict[str, object]:
    return {
        "source_slot": None,
        "relation_id": None,
        "direction": None,
        "op": op,
    }


def _proof() -> list[dict[str, object]]:
    return [
        {
            "source_slot": 0,
            "relation_id": "P1",
            "direction": "out",
            "op": "read",
        },
        _noop("halt"),
        *[_noop() for _ in range(10)],
    ]


def _jsonl(records) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


@dataclass(frozen=True)
class _Fixture:
    run: Path
    release: Path
    output: Path
    expected_study_lock_sha256: str
    item_ids: tuple[str, ...]
    submissions: dict[str, object]


def _sealed_fixture(tmp_path: Path) -> _Fixture:
    runner = _runner()
    run = tmp_path / "run"
    release = tmp_path / "sealed"
    run.mkdir()
    release.mkdir()

    checkpoint_path = run / "ckpt.pt"
    checkpoint_path.write_bytes(b"fixture checkpoint bytes\n")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    config_path = run / "config.json"
    config_path.write_bytes(
        canonical_json_bytes(
            {
                "condition": "split90",
                "model": "fixture-model",
                "seed": 0,
            }
        )
    )
    configuration_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    route_dose_sha256 = "e" * 64
    corpus_sha256 = "c" * 64
    code_sha256 = "d" * 64

    items = []
    gold = []
    stores = []
    submissions = {}
    shapes = (
        ("iid", 2, "seen"),
        ("composition_ood", 2, "heldout"),
        ("length_ood", 7, "seen"),
        ("joint_ood", 7, "heldout"),
    )
    controls = (
        ("correct_memory", "memory_on", "correct"),
        ("memory_off", "memory_off", "correct"),
        ("shuffled_returns", "memory_on", "shuffled_returns"),
        ("relevant_edge_swap", "memory_on", "relevant_edge"),
        ("irrelevant_edge_swap", "memory_on", "irrelevant_edge"),
        ("gold_path_replay", "memory_on", "gold_path"),
        ("no_query", "memory_on", "no_query"),
        ("entity_rename", "memory_on", "entity_rename"),
        ("graph_isomorphism", "memory_on", "graph_isomorphism"),
        ("page_order_permutation", "memory_on", "page_order_permutation"),
    )
    for family in ("graph", "non_path"):
        for stratum, path_length, composition_split in shapes:
            for control_id, memory_mode, control in controls:
                pair_id = f"{family}-{stratum}-{control_id}"
                world_id = f"world-{pair_id}"
                store_id = f"store-{pair_id}"
                rows = [
                    {
                        "source_id": "Q1",
                        "relation_id": "P1",
                        "direction": "out",
                        "target_kind": "literal",
                        "target": "done",
                        "qualifiers": {},
                    }
                ]
                store_sha256 = store_content_sha256(store_id, world_id, rows)
                stores.append(
                    {
                        "record_type": STORE_SCHEMA,
                        "schema_version": CONTRACT_VERSION,
                        "store_id": store_id,
                        "world_id": world_id,
                        "rows": rows,
                        "content_sha256": store_sha256,
                    }
                )
                for twin in ("original", "counterfactual"):
                    item_id = f"{pair_id}-{twin}"
                    items.append(
                        {
                            "record_type": ITEM_SCHEMA,
                            "schema_version": CONTRACT_VERSION,
                            "item_id": item_id,
                            "pair_id": pair_id,
                            "twin": twin,
                            "stratum": stratum,
                            "family": family,
                            "world_id": world_id,
                            "task": "fixture_task",
                            "path_length": path_length,
                            "composition_split": composition_split,
                            "composition_id": f"composition-{pair_id}",
                            "prompt": "Follow P1 from Q1.",
                            "initial_slots": ["Q1", None, None, None],
                            "store_id": store_id,
                            "memory_mode": memory_mode,
                            "control": control,
                        }
                    )
                    gold.append(
                        {
                            "record_type": SEALED_GOLD_SCHEMA,
                            "schema_version": CONTRACT_VERSION,
                            "item_id": item_id,
                            "pair_id": pair_id,
                            "twin": twin,
                            "answer": "done",
                            "proof": _proof(),
                            "solver_id": "lookup-chain-v1",
                            "store_sha256": store_sha256,
                        }
                    )
                    submissions[item_id] = runner.Submission(
                        item_id=item_id,
                        answer="done",
                        actions=_proof(),
                    )

    items.sort(key=lambda row: row["item_id"])
    gold.sort(key=lambda row: row["item_id"])
    stores.sort(key=lambda row: row["store_id"])
    item_ids = tuple(row["item_id"] for row in items)
    pair_ids = sorted({row["pair_id"] for row in items})
    world_ids = sorted({row["world_id"] for row in items})
    checkpoints = []
    for seed in range(5):
        for condition_id, arm in (("dense", "dense"), ("split90", "split")):
            selected = seed == 0 and condition_id == "split90"
            checkpoints.append(
                {
                    "record_type": CHECKPOINT_SCHEMA,
                    "schema_version": CONTRACT_VERSION,
                    "checkpoint_sha256": (
                        checkpoint_sha256
                        if selected
                        else hashlib.sha256(
                            f"checkpoint:{seed}:{condition_id}".encode()
                        ).hexdigest()
                    ),
                    "model_id": "fixture-model",
                    "arm": arm,
                    "condition_id": condition_id,
                    "seed": seed,
                    "raw_token_count": 1,
                    "configuration_sha256": (
                        configuration_sha256
                        if selected
                        else hashlib.sha256(
                            f"configuration:{seed}:{condition_id}".encode()
                        ).hexdigest()
                    ),
                    "route_dose_sha256": (
                        route_dose_sha256
                        if selected
                        else hashlib.sha256(
                            f"route-dose:{seed}:{condition_id}".encode()
                        ).hexdigest()
                    ),
                    "corpus_sha256": corpus_sha256,
                    "code_sha256": code_sha256,
                }
            )
    artifact_bytes = {
        "items.jsonl": _jsonl(items),
        "sealed-gold.jsonl": _jsonl(gold),
        "stores.jsonl": _jsonl(stores),
        "checkpoints.jsonl": _jsonl(checkpoints),
    }
    preregistration_sha256 = (
        "fee38e363298d3def46b741320c9d7df4523d0ff3cd249187cf52d54046cbbf0"
    )
    receipt_ids = (
        "gate:scientific_contract",
        "gate:route_dose",
        "gate:semantic_closure",
        "gate:proof_verification",
        "gate:ood_seal",
        "gate:corpus_identity",
        "gate:paired_training",
        "gate:checkpoint_resume",
        "gate:evaluation_validity",
        "gate:six_29m_diagnostics",
        "guardrail:iid",
        "control:correct_memory",
        "control:memory_off",
        "control:shuffled_returns",
        "control:relevant_edge_swap",
        "control:irrelevant_edge_swap",
        "control:gold_path_replay",
        "control:no_query",
        "control:entity_rename",
        "control:graph_isomorphism",
        "control:page_order_permutation",
    )
    validity_receipts = [
        {
            "receipt_id": receipt_id,
            "kind": receipt_id.split(":", 1)[0],
            "state": "passed",
            "evidence_sha256": hashlib.sha256(
                f"evidence:{receipt_id}".encode()
            ).hexdigest(),
        }
        for receipt_id in receipt_ids
    ]
    study_lock = {
        "record_type": "memorysplit.confirmatory.study-lock.v2",
        "schema_version": CONTRACT_VERSION,
        "preregistration_sha256": preregistration_sha256,
        "release": {
            "items_sha256": hashlib.sha256(artifact_bytes["items.jsonl"]).hexdigest(),
            "sealed_gold_sha256": hashlib.sha256(
                artifact_bytes["sealed-gold.jsonl"]
            ).hexdigest(),
            "stores_sha256": hashlib.sha256(artifact_bytes["stores.jsonl"]).hexdigest(),
            "checkpoints_sha256": hashlib.sha256(
                artifact_bytes["checkpoints.jsonl"]
            ).hexdigest(),
            "item_ids": list(item_ids),
            "pair_ids": pair_ids,
            "world_ids": world_ids,
            "evaluation_cells": [
                {
                    "item_id": item_id,
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "seed": checkpoint["seed"],
                    "condition_id": checkpoint["condition_id"],
                }
                for checkpoint in checkpoints
                for item_id in item_ids
            ],
            "item_count": len(item_ids),
            "pair_count": len(pair_ids),
            "world_count": len(world_ids),
            "evaluation_cell_count": len(item_ids) * len(checkpoints),
            "required_families": ["graph", "non_path"],
            "required_strata": [
                "iid",
                "composition_ood",
                "length_ood",
                "joint_ood",
            ],
            "required_memory_modes": ["memory_off", "memory_on"],
            "required_controls": [
                "correct_memory",
                "memory_off",
                "shuffled_returns",
                "relevant_edge_swap",
                "irrelevant_edge_swap",
                "gold_path_replay",
                "no_query",
                "entity_rename",
                "graph_isomorphism",
                "page_order_permutation",
            ],
        },
        "checkpoints": [
            {
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "seed": checkpoint["seed"],
                "condition_id": checkpoint["condition_id"],
                "configuration_sha256": checkpoint["configuration_sha256"],
                "route_dose_sha256": checkpoint["route_dose_sha256"],
            }
            for checkpoint in checkpoints
        ],
        "validity_receipts": [
            {
                "receipt_id": receipt["receipt_id"],
                "kind": receipt["kind"],
                "state": receipt["state"],
                "receipt_sha256": hashlib.sha256(
                    canonical_json_bytes(receipt)
                ).hexdigest(),
            }
            for receipt in validity_receipts
        ],
    }
    study_lock_bytes = canonical_json_bytes(study_lock)
    expected_study_lock_sha256 = hashlib.sha256(study_lock_bytes).hexdigest()
    artifact_bytes["study-lock.json"] = study_lock_bytes
    artifact_bytes["validity.json"] = canonical_json_bytes(
        {
            "record_type": "memorysplit.confirmatory.validity-evidence.v2",
            "schema_version": CONTRACT_VERSION,
            "study_lock_sha256": expected_study_lock_sha256,
            "preregistration_sha256": preregistration_sha256,
            "receipts": validity_receipts,
        }
    )
    for name, content in artifact_bytes.items():
        release.joinpath(name).write_bytes(content)

    run.joinpath("run.json").write_bytes(
        canonical_json_bytes(
            {
                "record_type": "memorysplit.confirmatory.run-binding.v2",
                "schema_version": CONTRACT_VERSION,
                "run_id": "fixture-s0-split90",
                "checkpoint_path": "ckpt.pt",
                "checkpoint_sha256": checkpoint_sha256,
                "configuration_path": "config.json",
                "configuration_sha256": configuration_sha256,
                "route_dose_sha256": route_dose_sha256,
                "corpus_sha256": corpus_sha256,
                "code_sha256": code_sha256,
                "seed": 0,
                "condition_id": "split90",
            }
        )
    )
    return _Fixture(
        run=run,
        release=release,
        output=tmp_path / "evidence",
        expected_study_lock_sha256=expected_study_lock_sha256,
        item_ids=item_ids,
        submissions=submissions,
    )


def _reseal_study_lock(fixture: _Fixture, **release_changes) -> str:
    path = fixture.release / "study-lock.json"
    lock = json.loads(path.read_bytes())
    lock["release"].update(release_changes)
    content = canonical_json_bytes(lock)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _install_trusted_reporting(monkeypatch, runner) -> None:
    required = (
        "checkpoints.jsonl",
        "inference.json",
        "items.jsonl",
        "metrics.json",
        "outcomes.jsonl",
        "sealed-gold.jsonl",
        "study-lock.json",
        "stores.jsonl",
        "validity.json",
    )

    @dataclass(frozen=True)
    class Report:
        study_lock_sha256: str
        artifacts_sha256: str

        def to_dict(self):
            return {
                "record_type": "memorysplit.confirmatory.artifact-report.v2",
                "schema_version": CONTRACT_VERSION,
                "study_lock_sha256": self.study_lock_sha256,
                "artifacts_sha256": self.artifacts_sha256,
            }

    def build_artifact_report(*, artifacts, expected_study_lock_sha256):
        assert set(artifacts) == set(required)
        assert hashlib.sha256(artifacts["study-lock.json"]).hexdigest() == (
            expected_study_lock_sha256
        )
        digest = hashlib.sha256(
            b"".join(artifacts[name] for name in sorted(artifacts))
        ).hexdigest()
        return Report(expected_study_lock_sha256, digest)

    def publish_artifact_report(
        path,
        report,
        artifacts,
        *,
        expected_study_lock_sha256,
    ):
        assert (
            build_artifact_report(
                artifacts=artifacts,
                expected_study_lock_sha256=expected_study_lock_sha256,
            )
            == report
        )
        Path(path).write_bytes(canonical_json_bytes(report))
        return Path(path)

    monkeypatch.setattr(runner.reporting, "REQUIRED_ARTIFACTS", required)
    monkeypatch.setattr(
        runner.reporting,
        "build_artifact_report",
        build_artifact_report,
    )
    monkeypatch.setattr(
        runner.reporting,
        "publish_artifact_report",
        publish_artifact_report,
    )


def test_runner_exposes_injected_model_adapter_contract():
    assert importlib.util.find_spec("evals.confirmatory.runner") is not None
    runner = _runner()

    submission = runner.Submission(
        item_id="item-1",
        answer="done",
        actions=_proof(),
    )
    adapter = runner.DeterministicFixtureAdapter({"item-1": submission})

    assert adapter.generate(type("VisibleItem", (), {"item_id": "item-1"})()) == (
        submission
    )
    with pytest.raises(ValueError, match="missing"):
        adapter.generate(type("VisibleItem", (), {"item_id": "item-2"})())


def test_submission_requires_exactly_twelve_valid_action_slots():
    runner = _runner()

    with pytest.raises(ValueError, match="12"):
        runner.Submission(
            item_id="item-1",
            answer="done",
            actions=_proof()[:-1],
        )
    with pytest.raises(ValueError, match="item_id"):
        runner.Submission(item_id="", answer="done", actions=_proof())
    with pytest.raises(ValueError, match="answer"):
        runner.Submission(item_id="item-1", answer=None, actions=_proof())


def test_external_study_lock_is_checked_before_model_visible_items(tmp_path):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    fixture.release.joinpath("items.jsonl").write_bytes(b"not-json\n")

    class RecordingAdapter:
        calls = []

        def generate(self, item):
            self.calls.append(item)
            raise AssertionError("adapter must not be called")

    adapter = RecordingAdapter()
    with pytest.raises(ValueError, match="study lock.*external"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256="0" * 64,
            model_adapter=adapter,
            output_dir=fixture.output,
        )

    assert adapter.calls == []
    assert not fixture.output.exists()


def test_runner_keeps_gold_sealed_replays_solver_and_publishes_canonical_evidence(
    tmp_path,
    monkeypatch,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    _install_trusted_reporting(monkeypatch, runner)
    gold_path = fixture.release / "sealed-gold.jsonl"
    gold_path.chmod(0)
    delegate = runner.DeterministicFixtureAdapter(fixture.submissions)

    class GoldOpeningGuardAdapter:
        def __init__(self):
            self.item_ids = []

        def generate(self, item):
            assert isinstance(item, ItemRecord)
            assert not hasattr(item, "answer")
            assert not hasattr(item, "proof")
            self.item_ids.append(item.item_id)
            if len(self.item_ids) == 1:
                gold_path.chmod(0o600)
            return delegate.generate(item)

    adapter = GoldOpeningGuardAdapter()
    result = runner.evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        model_adapter=adapter,
        output_dir=fixture.output,
    )

    assert result.item_count == len(fixture.item_ids)
    assert tuple(adapter.item_ids) == fixture.item_ids
    assert result.study_lock_sha256 == fixture.expected_study_lock_sha256
    assert result.output_dir == fixture.output
    assert fixture.output.joinpath("artifact-report.json").is_file()
    outcomes = [
        json.loads(line)
        for line in fixture.output.joinpath("outcomes.jsonl").read_bytes().splitlines()
    ]
    assert {row["item_id"] for row in outcomes} == set(fixture.item_ids)
    assert all(len(row["submitted_proof"]) == 12 for row in outcomes)
    assert all(row["submitted_answer"] == "done" for row in outcomes)
    assert all(
        not {"proof_valid", "answer_valid", "valid", "complete"} & set(row)
        for row in outcomes
    )
    metrics = json.loads(fixture.output.joinpath("metrics.json").read_bytes())
    assert metrics["summaries"][0]["primary_accuracy"] == 1.0
    inference = json.loads(fixture.output.joinpath("inference.json").read_bytes())
    assert "supports_effect" not in inference
    assert "supports_practical_null" not in inference
    assert all(
        line.endswith(b"\n")
        for name in runner.reporting.REQUIRED_ARTIFACTS
        for line in fixture.output.joinpath(name).read_bytes().splitlines(keepends=True)
    )

    with pytest.raises(FileExistsError, match="output"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=adapter,
            output_dir=fixture.output,
        )
    assert tuple(adapter.item_ids) == fixture.item_ids


def test_solver_replay_not_adapter_assertions_determines_metrics(
    tmp_path,
    monkeypatch,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    _install_trusted_reporting(monkeypatch, runner)
    wrong = {
        item_id: runner.Submission(
            item_id=item_id,
            answer="forged",
            actions=_proof(),
        )
        for item_id in fixture.item_ids
    }

    runner.evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        model_adapter=runner.DeterministicFixtureAdapter(wrong),
        output_dir=fixture.output,
    )

    metrics = json.loads(fixture.output.joinpath("metrics.json").read_bytes())
    assert metrics["summaries"][0]["primary_accuracy"] == 0.0
    outcomes = fixture.output.joinpath("outcomes.jsonl").read_text()
    assert "proof_valid" not in outcomes
    assert "answer_valid" not in outcomes


def test_runner_rejects_malformed_submission_without_publishing(tmp_path):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)

    class MalformedAdapter:
        def generate(self, item):
            return {
                "item_id": item.item_id,
                "answer": "done",
                "actions": _proof()[:-1],
            }

    with pytest.raises(ValueError, match="Submission"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=MalformedAdapter(),
            output_dir=fixture.output,
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("condition_id", "split", "split90|condition"),
        ("seed", 1, "seed|binding"),
        ("checkpoint_sha256", "0" * 64, "checkpoint|hash"),
        ("configuration_sha256", "0" * 64, "configuration|hash"),
        ("corpus_sha256", "0" * 64, "corpus|binding"),
        ("code_sha256", "0" * 64, "code|binding"),
    ],
)
def test_runner_rejects_invalid_run_checkpoint_bindings(
    tmp_path,
    field,
    value,
    message,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    run_path = fixture.run / "run.json"
    raw = json.loads(run_path.read_bytes())
    raw[field] = value
    run_path.write_bytes(canonical_json_bytes(raw))

    with pytest.raises(ValueError, match=message):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=runner.DeterministicFixtureAdapter(fixture.submissions),
            output_dir=fixture.output,
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_runner_rejects_missing_or_duplicate_model_visible_items(
    tmp_path,
    mutation,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    path = fixture.release / "items.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if mutation == "missing":
        changed = b"".join(lines[:-1])
    else:
        changed = b"".join([*lines, lines[0]])
    path.write_bytes(changed)
    expected = _reseal_study_lock(
        fixture,
        items_sha256=hashlib.sha256(changed).hexdigest(),
    )

    with pytest.raises(ValueError, match="missing|duplicate|registry"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=expected,
            model_adapter=runner.DeterministicFixtureAdapter(fixture.submissions),
            output_dir=fixture.output,
        )
    assert not fixture.output.exists()


def test_unhardened_reporting_api_fails_closed_at_report_boundary(
    tmp_path,
    monkeypatch,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setattr(
        runner.reporting,
        "REQUIRED_ARTIFACTS",
        (
            "checkpoints.jsonl",
            "inference.json",
            "items.jsonl",
            "metrics.json",
            "outcomes.jsonl",
            "sealed-gold.jsonl",
            "stores.jsonl",
        ),
    )

    with pytest.raises(
        runner.ReportingInterfaceUnavailable,
        match=(
            r"build_artifact_report\(\*, artifacts, "
            r"expected_study_lock_sha256\)"
        ),
    ) as caught:
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=runner.DeterministicFixtureAdapter(fixture.submissions),
            output_dir=fixture.output,
        )

    assert "publish_artifact_report" in str(caught.value)
    assert "study-lock.json" in str(caught.value)
    assert "validity.json" in str(caught.value)
    assert not fixture.output.exists()


def test_cli_dry_run_emits_one_json_object_and_only_logs_to_stderr(
    tmp_path,
    capsys,
):
    fixture = _sealed_fixture(tmp_path)
    entry = importlib.import_module("evals.confirmatory.__main__")

    return_code = entry.main(
        [
            "evaluate",
            "--run",
            str(fixture.run),
            "--sealed-release",
            str(fixture.release),
            "--expected-study-lock-sha256",
            fixture.expected_study_lock_sha256,
            "--device",
            "cuda",
        ]
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert return_code == 0
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result == {
        "checkpoint_sha256": hashlib.sha256(
            fixture.run.joinpath("ckpt.pt").read_bytes()
        ).hexdigest(),
        "condition_id": "split90",
        "device": "cuda",
        "item_count": len(fixture.item_ids),
        "seed": 0,
        "status": "dry_run",
        "study_lock_sha256": fixture.expected_study_lock_sha256,
    }
    assert "dry-run" in captured.err
    assert not fixture.output.exists()


def test_cli_evaluate_uses_injected_adapter_and_publishes_once(
    tmp_path,
    monkeypatch,
    capsys,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    _install_trusted_reporting(monkeypatch, runner)
    entry = importlib.import_module("evals.confirmatory.__main__")

    return_code = entry.main(
        [
            "evaluate",
            "--run",
            str(fixture.run),
            "--sealed-release",
            str(fixture.release),
            "--expected-study-lock-sha256",
            fixture.expected_study_lock_sha256,
            "--device",
            "cpu",
            "--output-dir",
            str(fixture.output),
        ],
        model_adapter=runner.DeterministicFixtureAdapter(fixture.submissions),
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert return_code == 0
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["status"] == "published"
    assert result["output_dir"] == str(fixture.output)
    assert result["item_count"] == len(fixture.item_ids)
    assert "published" in captured.err


@pytest.mark.parametrize(
    "arguments",
    [
        ["evaluate", "--unknown"],
        ["evaluate"],
    ],
)
def test_cli_errors_still_emit_one_json_object(arguments, capsys):
    entry = importlib.import_module("evals.confirmatory.__main__")

    return_code = entry.main(arguments)

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert return_code != 0
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["status"] == "error"
    assert isinstance(result["error"], str) and result["error"]
    assert captured.err


def test_cli_help_preserves_json_only_stdout(capsys):
    entry = importlib.import_module("evals.confirmatory.__main__")

    return_code = entry.main(["--help"])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert return_code == 0
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"status": "help"}
    assert "usage:" in captured.err
