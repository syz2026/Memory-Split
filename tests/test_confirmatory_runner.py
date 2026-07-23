from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

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
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok


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
            "relation_id": "r0",
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
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = tmp_path / "run"
    release = tmp_path / "sealed"
    run.mkdir()
    release.mkdir()

    model_config = {
        "n_layer": 0,
        "n_head": 1,
        "d_model": 2,
        "vocab_size": 50304,
        "ctx": 512,
    }
    run_config = {
        "condition": "split90",
        "model": model_config,
        "seed": 0,
    }
    config_path = run / "config.json"
    config_path.write_bytes(canonical_json_bytes(run_config))
    configuration_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    tok = get_tok()
    answer_ids = tok.encode("done")
    assert len(answer_ids) == 1
    answer_id = answer_ids[0]
    model = GPT(GPTConfig(**model_config))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ln_f.weight.fill_(1.0)
        model.wte.weight[tok.ANSWER_STATE] = torch.tensor([1.0, 0.0])
        model.lm_head.weight[answer_id] = torch.tensor([1.0, 0.0])
        model.wte.weight[answer_id] = torch.tensor([0.0, 1.0])
        model.lm_head.weight[tok.GRAPH_START] = torch.tensor([0.0, 1.0])
    checkpoint_path = run / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": run_config,
            "step": 0,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
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
                        "relation_id": "r0",
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
                            "prompt": "Follow r0 from Q1.",
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


def _mutate_study_lock(fixture: _Fixture, mutation) -> str:
    path = fixture.release / "study-lock.json"
    lock = json.loads(path.read_bytes())
    mutation(lock)
    content = canonical_json_bytes(lock)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _staging_paths(fixture: _Fixture) -> list[Path]:
    return list(fixture.output.parent.glob(f".{fixture.output.name}.confirmatory-*"))


def test_runner_exposes_injected_model_adapter_contract():
    assert importlib.util.find_spec("evals.confirmatory.runner") is not None
    runner = _runner()

    submission = runner.Submission(
        item_id="item-1",
        answer="done",
        actions=_proof(),
    )
    adapter = runner.DeterministicFixtureAdapter({"item-1": submission})

    visible = type("VisibleItem", (), {"item_id": "item-1"})()
    assert adapter.generate(visible, None) == submission
    with pytest.raises(ValueError, match="missing"):
        adapter.generate(type("VisibleItem", (), {"item_id": "item-2"})(), None)


def test_repository_adapter_rejects_checkpoint_hash_architecture_and_state_drift(
    tmp_path,
):
    runner = _runner()

    hash_fixture = _sealed_fixture(tmp_path / "hash")
    binding = runner._load_run_binding(hash_fixture.run)
    hash_fixture.run.joinpath("ckpt.pt").write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint.*hash"):
        runner.RepositoryGPTAdapter.from_bound_run(
            hash_fixture.run,
            binding,
            "cpu",
        )

    architecture_fixture = _sealed_fixture(tmp_path / "architecture")
    binding = runner._load_run_binding(architecture_fixture.run)
    checkpoint_path = architecture_fixture.run / "ckpt.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state["cfg"] = copy.deepcopy(state["cfg"])
    state["cfg"]["model"]["d_model"] = 4
    torch.save(state, checkpoint_path)
    binding = replace(
        binding,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="architecture|config"):
        runner.RepositoryGPTAdapter.from_bound_run(
            architecture_fixture.run,
            binding,
            "cpu",
        )

    state_fixture = _sealed_fixture(tmp_path / "state")
    binding = runner._load_run_binding(state_fixture.run)
    checkpoint_path = state_fixture.run / "ckpt.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    del state["model"]["ln_f.weight"]
    torch.save(state, checkpoint_path)
    binding = replace(
        binding,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="state"):
        runner.RepositoryGPTAdapter.from_bound_run(
            state_fixture.run,
            binding,
            "cpu",
        )


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

        def generate(self, item, store):
            self.calls.append((item, store))
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("preregistration", "preregistration"),
        ("controls", "control|required"),
        ("receipts", "receipt"),
    ],
)
def test_real_hardened_study_lock_rejects_malformed_trust_roots(
    tmp_path,
    mutation,
    message,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)

    def change(lock):
        if mutation == "preregistration":
            lock["preregistration_sha256"] = "0" * 64
        elif mutation == "controls":
            lock["release"]["required_controls"] = []
        else:
            lock["validity_receipts"] = []

    expected = _mutate_study_lock(fixture, change)
    with pytest.raises(ValueError, match=message):
        runner.preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=expected,
        )


def test_missing_hardened_study_lock_api_fails_closed(tmp_path, monkeypatch):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)

    def unavailable():
        raise runner.ValidationInterfaceUnavailable(
            "hardened study-lock validation API unavailable"
        )

    monkeypatch.setattr(runner, "_hardened_study_lock_api", unavailable)
    with pytest.raises(
        runner.ValidationInterfaceUnavailable,
        match="hardened study-lock validation API unavailable",
    ):
        runner.preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )


def test_pre_f934_readiness_semantics_fail_closed(tmp_path, monkeypatch):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    hardening = runner._hardened_study_lock_api()
    monkeypatch.setattr(
        hardening,
        "evaluate_readiness",
        lambda _lock, _evidence: object(),
    )

    with pytest.raises(
        runner.ValidationInterfaceUnavailable,
        match="externally rooted readiness.*f934124",
    ):
        runner.preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )


def test_runner_rejects_store_relation_outside_frozen_model_vocabulary(tmp_path):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    stores_path = fixture.release / "stores.jsonl"
    changed = []
    for line in stores_path.read_bytes().splitlines():
        store = json.loads(line)
        store["rows"][0]["relation_id"] = "P1"
        store["content_sha256"] = store_content_sha256(
            store["store_id"],
            store["world_id"],
            store["rows"],
        )
        changed.append(canonical_json_bytes(store))
    stores_content = b"".join(changed)
    stores_path.write_bytes(stores_content)
    expected = _reseal_study_lock(
        fixture,
        stores_sha256=hashlib.sha256(stores_content).hexdigest(),
    )

    with pytest.raises(ValueError, match="unsupported.*relation"):
        runner.preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=expected,
        )


def test_runner_rejects_drifted_frozen_graph_protocol(tmp_path, monkeypatch):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    tok = get_tok()
    monkeypatch.setattr(tok, "GRAPH_START", tok.GRAPH_START + 1)

    with pytest.raises(
        runner.ValidationInterfaceUnavailable,
        match="graph action protocol",
    ):
        runner.preflight(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )


def test_runner_keeps_gold_sealed_replays_solver_and_publishes_canonical_evidence(
    tmp_path,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    gold_path = fixture.release / "sealed-gold.jsonl"
    gold_path.chmod(0)
    delegate = runner.DeterministicFixtureAdapter(fixture.submissions)

    class GoldOpeningGuardAdapter:
        def __init__(self):
            self.item_ids = []

        def generate(self, item, store):
            assert isinstance(item, ItemRecord)
            assert not hasattr(item, "answer")
            assert not hasattr(item, "proof")
            if item.memory_mode.value == "memory_off":
                assert store is None
            else:
                assert store is not None
                assert store.store_id == item.store_id
            self.item_ids.append(item.item_id)
            if len(self.item_ids) == 1:
                gold_path.chmod(0o600)
            return delegate.generate(item, store)

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


def test_transactional_publication_cleans_mid_write_failure_and_allows_retry(
    tmp_path,
    monkeypatch,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    adapter = runner.DeterministicFixtureAdapter(fixture.submissions)
    original = runner._publish_file
    writes = 0

    def fail_mid_write(path, content):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected artifact write failure")
        return original(path, content)

    monkeypatch.setattr(runner, "_publish_file", fail_mid_write)
    with pytest.raises(OSError, match="injected artifact write failure"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=adapter,
            output_dir=fixture.output,
        )
    assert not fixture.output.exists()
    assert _staging_paths(fixture) == []

    monkeypatch.setattr(runner, "_publish_file", original)
    result = runner.evaluate(
        run=fixture.run,
        sealed_release=fixture.release,
        expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        model_adapter=adapter,
        output_dir=fixture.output,
    )
    assert result.output_dir == fixture.output


def test_transactional_publication_cleans_report_failure(tmp_path, monkeypatch):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    original = runner.reporting.publish_artifact_report

    def fail_after_report(
        path,
        report,
        artifacts,
        *,
        expected_study_lock_sha256,
    ):
        original(
            path,
            report,
            artifacts,
            expected_study_lock_sha256=expected_study_lock_sha256,
        )
        raise OSError("injected report failure")

    monkeypatch.setattr(
        runner.reporting,
        "publish_artifact_report",
        fail_after_report,
    )
    with pytest.raises(OSError, match="injected report failure"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=runner.DeterministicFixtureAdapter(fixture.submissions),
            output_dir=fixture.output,
        )
    assert not fixture.output.exists()
    assert _staging_paths(fixture) == []


def test_transactional_publication_never_replaces_collision(
    tmp_path,
    monkeypatch,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    original = runner._atomic_publish_directory

    def collide(staging, output):
        output.mkdir()
        output.joinpath("other-owner").write_text("untouched")
        return original(staging, output)

    monkeypatch.setattr(runner, "_atomic_publish_directory", collide)
    with pytest.raises(FileExistsError, match="output|exists"):
        runner.evaluate(
            run=fixture.run,
            sealed_release=fixture.release,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
            model_adapter=runner.DeterministicFixtureAdapter(fixture.submissions),
            output_dir=fixture.output,
        )
    assert [path.name for path in fixture.output.iterdir()] == ["other-owner"]
    assert _staging_paths(fixture) == []


def test_solver_replay_not_adapter_assertions_determines_metrics(tmp_path):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
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
        def generate(self, item, store):
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
    capsys,
):
    runner = _runner()
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


def test_cli_builds_real_repository_adapter_with_memory_boundaries(
    tmp_path,
    monkeypatch,
    capsys,
):
    runner = _runner()
    fixture = _sealed_fixture(tmp_path)
    entry = importlib.import_module("evals.confirmatory.__main__")
    original = runner.RepositoryGPTAdapter.generate
    observed = []

    def record_and_generate(self, item, store):
        observed.append(
            (
                item.memory_mode.value,
                store is None,
                type(self.model),
            )
        )
        return original(self, item, store)

    monkeypatch.setattr(
        runner.RepositoryGPTAdapter,
        "generate",
        record_and_generate,
    )
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
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["status"] == "published"
    assert observed
    assert all(model_type is GPT for _, _, model_type in observed)
    assert {
        (memory_mode, store_is_none) for memory_mode, store_is_none, _ in observed
    } == {("memory_off", True), ("memory_on", False)}
    outcomes = [
        json.loads(line)
        for line in fixture.output.joinpath("outcomes.jsonl").read_bytes().splitlines()
    ]
    assert all(row["submitted_answer"] == "done" for row in outcomes)
    assert all(
        [action["op"] for action in row["submitted_proof"]]
        == [*(["read"] * 10), "halt", "noop"]
        for row in outcomes
    )
    assert fixture.output.joinpath("artifact-report.json").is_file()


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
    runner = _runner()
    entry = importlib.import_module("evals.confirmatory.__main__")

    return_code = entry.main(["--help"])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert return_code == 0
    assert len(lines) == 1
    assert runner.MSCTL_EVALUATOR_CONTRACT == ("memorysplit-confirmatory-evaluator-v1")
    assert json.loads(lines[0]) == {
        "canonical_flags": [
            "--run",
            "--sealed-release",
            "--expected-study-lock-sha256",
            "--device",
            "--output-dir",
        ],
        "canonical_invocation": (
            "evaluate --run RUN --sealed-release RELEASE "
            "--expected-study-lock-sha256 HASH --device DEVICE "
            "--output-dir OUTPUT"
        ),
        "command": "evaluate",
        "contract": runner.MSCTL_EVALUATOR_CONTRACT,
        "status": "help",
    }
    assert "usage:" in captured.err


@pytest.mark.parametrize("mode", ["module", "direct"])
def test_module_and_direct_cli_help_match_machine_contract(mode):
    runner = _runner()
    root = Path(__file__).resolve().parents[1]
    command = (
        [sys.executable, "-m", "evals.confirmatory", "--help"]
        if mode == "module"
        else [sys.executable, str(Path(runner.__file__).resolve()), "--help"]
    )

    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 1
    payload = json.loads(completed.stdout)
    assert payload["contract"] == "memorysplit-confirmatory-evaluator-v1"
    assert payload["canonical_flags"] == [
        "--run",
        "--sealed-release",
        "--expected-study-lock-sha256",
        "--device",
        "--output-dir",
    ]
    assert "usage:" in completed.stderr
