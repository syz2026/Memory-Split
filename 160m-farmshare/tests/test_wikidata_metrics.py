from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from evals.wikidata_metrics import compute_wikidata_metrics, selective_accuracy
from scripts.run_wikidata_evals import (
    build_eval_provenance,
    checkpoint_sha256,
    default_evaluator_files,
    require_claim_bearing_checkpoint,
    resolve_run_checkpoint,
    verify_checkpoint_config,
    verify_checkpoint_unchanged,
)


def _action(relation: str = "P1") -> list:
    return [0, relation, "out", True, False]


def _row(
    *,
    memory: str,
    pair: str,
    variant: str,
    task: str,
    correct: bool,
    abstained: bool = False,
    hop_count: int = 1,
    alias_slice: str = "primary_only",
    composition_slice: str = "single_relation",
) -> dict:
    gold = [_action("P1" if index == 0 else "P2") for index in range(hop_count)]
    return {
        "qid": f"{pair}-{variant}",
        "pair_id": pair,
        "variant": variant,
        "task": task,
        "memory": memory,
        "correct": correct,
        "pred": "abstain" if abstained else "answer",
        "answer": "answer",
        "abstained": abstained,
        "actions": list(gold),
        "all_actions": list(gold),
        "gold_actions": list(gold),
        "gold_all_actions": list(gold),
        "correct_referents": [True] * hop_count,
        "misses": int(memory == "off"),
        "meta": {
            "hop_count": hop_count,
            "alias_slice": alias_slice,
            "composition_slice": composition_slice,
            "relation_path_hash": f"path-{pair}",
        },
    }


def _rows() -> list[dict]:
    rows = []
    for memory in ("on", "off"):
        rows.extend(
            (
                _row(
                    memory=memory,
                    pair="traversal-1",
                    variant="original",
                    task="endpoint_traversal",
                    correct=memory == "on",
                ),
                _row(
                    memory=memory,
                    pair="traversal-1",
                    variant="counterfactual",
                    task="endpoint_traversal",
                    correct=memory == "on",
                ),
                _row(
                    memory=memory,
                    pair="equality-2",
                    variant="original",
                    task="endpoint_equality",
                    correct=True,
                    hop_count=2,
                    alias_slice="includes_alternate",
                    composition_slice="multi_relation",
                ),
                _row(
                    memory=memory,
                    pair="equality-2",
                    variant="counterfactual",
                    task="endpoint_equality",
                    correct=False,
                    abstained=True,
                    hop_count=2,
                    alias_slice="includes_alternate",
                    composition_slice="multi_relation",
                ),
            )
        )
    return rows


def _coverage() -> dict:
    return {
        "entities": {"candidates": 12, "surviving": 8},
        "relations": {"candidates": 2, "surviving": 2},
        "addresses": {
            "candidates": 10,
            "surviving": 7,
            "exclusions": {
                "ambiguous_address": 2,
                "unselected_relation": 1,
            },
        },
        "paths": {
            "candidates": 20,
            "surviving": 10,
            "exclusions": {
                "ambiguous_address": 0,
                "missing_hop": 10,
                "repeated_address": 0,
                "no_counterfactual_alternative": 0,
                "selection_bound": 0,
            },
        },
    }


def _checkpoint_cfg(condition: str = "dense") -> dict:
    return {
        "condition": condition,
        "model": "micro",
        "seed": 17,
        "train_bin": "corpus/train.bin",
        "train_mask": (
            None if condition == "dense" else "corpus/train.mask.bin"
        ),
        "train_weights": None,
        "out_dir": "runs/arm-17",
        "ctx": 256,
    }


def test_reports_store_accuracy_coverage_abstention_and_explicit_selective_denominator():
    report = compute_wikidata_metrics(_rows(), _coverage())

    assert report["store"]["on"]["accuracy"] == {
        "value": 0.75,
        "numerator": 3,
        "denominator": 4,
    }
    assert report["store"]["on"]["coverage"] == {
        "value": 0.75,
        "numerator": 3,
        "denominator": 4,
    }
    assert report["store"]["on"]["abstention"] == {
        "value": 0.25,
        "numerator": 1,
        "denominator": 4,
    }
    assert report["store"]["on"]["selective_accuracy"] == {
        "value": 1.0,
        "numerator": 3,
        "denominator": 3,
        "population_denominator": 4,
    }
    assert report["store"]["on"]["pair_accuracy"]["denominator"] == 2
    assert report["confirmatory_verdict_eligible"] is False


def test_reports_action_path_per_hop_and_required_slices():
    report = compute_wikidata_metrics(_rows(), _coverage())
    transfer = report["store"]["on"]["transfer"]

    assert transfer["exact_action_path"]["value"] == 1.0
    assert transfer["all_action_slots"]["value"] == 1.0
    assert transfer["per_hop"]["1"]["relation"]["value"] == 1.0
    assert transfer["per_hop"]["2"]["referent"]["value"] == 1.0
    assert set(report["slices"]) == {
        "alias",
        "composition",
        "composition_path",
        "hop",
        "task",
    }
    assert set(report["slices"]["alias"]["on"]) == {
        "includes_alternate",
        "primary_only",
    }
    assert set(report["slices"]["composition"]["on"]) == {
        "multi_relation",
        "single_relation",
    }
    assert set(report["slices"]["hop"]["on"]) == {"1", "2"}
    assert set(report["slices"]["composition_path"]["on"]) == {
        "path-equality-2",
        "path-traversal-1",
    }


def test_miss_rate_uses_all_actual_read_attempts_as_its_denominator():
    rows = _rows()
    for row in rows:
        if row["memory"] == "off":
            row["actions"].append(_action("P2"))
            row["all_actions"].append(_action("P2"))
            row["misses"] = 1

    report = compute_wikidata_metrics(rows, _coverage())

    assert report["store"]["off"]["transfer"]["miss_rate"] == {
        "value": 0.4,
        "numerator": 4,
        "denominator": 10,
    }


def test_artifact_survival_denominators_are_carried_into_results():
    report = compute_wikidata_metrics(_rows(), _coverage())

    assert report["artifact_survival"]["entities"]["value"] == 8 / 12
    assert report["artifact_survival"]["relations"]["value"] == 1.0
    assert report["artifact_survival"]["paths"]["value"] == 0.5
    assert report["artifact_survival"]["paths"]["denominator"] == 20


def test_final_results_include_every_frozen_exclusion_with_denominators():
    report = compute_wikidata_metrics(_rows(), _coverage())
    accounting = report["exclusion_accounting"]

    assert accounting["address"]["invariant"] == {
        "candidates": 10,
        "surviving": 7,
        "excluded": 3,
        "candidate_equals_survivor_plus_exclusions": True,
    }
    assert accounting["address"]["exclusions"] == {
        "ambiguous_address": {
            "value": 0.2,
            "numerator": 2,
            "denominator": 10,
        },
        "unselected_relation": {
            "value": 0.1,
            "numerator": 1,
            "denominator": 10,
        },
    }
    assert set(accounting["path"]["exclusions"]) == {
        "ambiguous_address",
        "missing_hop",
        "repeated_address",
        "no_counterfactual_alternative",
        "selection_bound",
    }
    assert accounting["path"]["exclusions"]["missing_hop"] == {
        "value": 0.5,
        "numerator": 10,
        "denominator": 20,
    }
    assert accounting["path"]["invariant"][
        "candidate_equals_survivor_plus_exclusions"
    ] is True


def test_selective_accuracy_uses_null_value_for_zero_answered_denominator():
    measurement = selective_accuracy(
        [{"correct": False, "abstained": True} for _ in range(3)]
    )

    assert measurement == {
        "value": None,
        "numerator": 0,
        "denominator": 0,
        "population_denominator": 3,
    }


def test_runner_requires_claim_bearing_checkpoint_and_preserves_its_hash(
    tmp_path,
):
    checkpoint = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "cfg": _checkpoint_cfg(),
        },
        checkpoint,
    )

    state = require_claim_bearing_checkpoint(checkpoint)
    before = checkpoint_sha256(checkpoint)
    integrity = verify_checkpoint_unchanged(checkpoint, before)

    assert "model" in state
    assert integrity == {
        "checkpoint_sha256": before,
        "checkpoint_sha256_after": before,
        "checkpoint_mutated": False,
        "fine_tuning_performed": False,
    }

    checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="checkpoint changed during evaluation"):
        verify_checkpoint_unchanged(checkpoint, before)


def test_checkpoint_requires_embedded_config_mapping(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint)

    with pytest.raises(ValueError, match="embedded cfg"):
        require_claim_bearing_checkpoint(checkpoint)


def test_checkpoint_path_must_be_relative_regular_and_contained(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "ckpt.pt"
    torch.save(
        {"model": {"weight": torch.tensor([1.0])}, "cfg": _checkpoint_cfg()},
        checkpoint,
    )
    outside = tmp_path / "outside.pt"
    outside.write_bytes(checkpoint.read_bytes())

    assert resolve_run_checkpoint(run, "ckpt.pt") == checkpoint.resolve()
    assert resolve_run_checkpoint(run, checkpoint.resolve()) == checkpoint.resolve()
    with pytest.raises(ValueError, match="contained"):
        resolve_run_checkpoint(run, outside)
    with pytest.raises(ValueError, match="parent traversal"):
        resolve_run_checkpoint(run, "../outside.pt")
    link = run / "link.pt"
    link.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="regular non-symlink"):
        resolve_run_checkpoint(run, "link.pt")


def test_checkpoint_embedded_config_rejects_compatible_shape_wrong_arm(
    tmp_path,
):
    run = tmp_path / "run"
    run.mkdir()
    run_cfg = _checkpoint_cfg("dense")
    (run / "config.yaml").write_text(
        yaml.safe_dump(run_cfg, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_cfg = {**run_cfg, "condition": "split"}

    with pytest.raises(ValueError, match="condition identity mismatch"):
        verify_checkpoint_config(run, {"cfg": checkpoint_cfg})


def test_checkpoint_config_identity_is_derived_from_verified_embedded_cfg(
    tmp_path,
):
    run = tmp_path / "run"
    run.mkdir()
    cfg = _checkpoint_cfg("random")
    (run / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    embedded, identities = verify_checkpoint_config(
        run,
        {"cfg": dict(reversed(tuple(cfg.items())))},
    )

    assert embedded["condition"] == "random"
    assert embedded["model"] == "micro"
    assert identities["condition"] == "random"
    assert identities["model"] == "micro"
    assert identities["seed"] == 17
    assert len(identities["configuration_sha256"]) == 64


def test_eval_provenance_records_all_frozen_hashes_and_robustness_boundary(
    tmp_path,
):
    paths = {}
    for name in (
        "checkpoint",
        "source_lock",
        "schema",
        "evaluator",
        "preregistration",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path

    provenance = build_eval_provenance(
        checkpoint=paths["checkpoint"],
        source_lock=paths["source_lock"],
        schema=paths["schema"],
        evaluator_files=(paths["evaluator"],),
        preregistration=paths["preregistration"],
    )

    for name in (
        "checkpoint",
        "source_lock",
        "schema",
        "evaluator",
        "preregistration",
    ):
        assert provenance[f"{name}_sha256"] == hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
    assert provenance["evaluation_only"] is True
    assert provenance["fine_tuning_performed"] is False
    assert provenance["confirmatory_verdict_eligible"] is False
    assert json.dumps(provenance, sort_keys=True)


def test_default_evaluator_hash_covers_frozen_dependency_closure():
    root = default_evaluator_files()[0].parents[1]
    assert {
        path.relative_to(root).as_posix()
        for path in default_evaluator_files()
    } >= {
        "corpusgen/graph_records.py",
        "corpusgen/graph_trace.py",
        "corpusgen/mask_ledger.py",
        "corpusgen/payload_inventory.py",
        "train/model.py",
        "train/data.py",
        "train/tokenizer.py",
        "organizer/packed_graph_store.py",
        "organizer/graph_store.py",
        "corpusgen/relation_codec.py",
        "corpusgen/relation_schema.py",
        "corpusgen/srgm_worlds.py",
        "corpusgen/world_splits.py",
        "evals/generate.py",
        "evals/relational_generate.py",
        "evals/relational_metrics.py",
        "evals/relational_pairing.py",
        "evals/scorers.py",
        "evals/wikidata_metrics.py",
        "scripts/run_wikidata_evals.py",
        "scripts/run_relational_evals.py",
    }


def _local_import_paths(path: Path, root: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    paths = set()
    for module in modules:
        parts = module.split(".")
        candidates = (
            root.joinpath(*parts).with_suffix(".py"),
            root.joinpath(*parts, "__init__.py"),
        )
        local = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if local is not None:
            paths.add(local.resolve())
    return paths


def test_default_evaluator_files_are_closed_over_local_imports():
    evaluator_files = {path.resolve() for path in default_evaluator_files()}
    root = default_evaluator_files()[0].parents[1].resolve()
    missing = {
        dependency.relative_to(root).as_posix()
        for path in evaluator_files
        for dependency in _local_import_paths(path, root)
        if dependency not in evaluator_files
    }

    assert missing == set()


def test_world_splits_drift_changes_default_evaluator_hash(tmp_path):
    root = default_evaluator_files()[0].parents[1]
    copied_files = []
    for source in default_evaluator_files():
        destination = tmp_path / "closure" / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied_files.append(destination)
    world_splits = tmp_path / "closure" / "corpusgen" / "world_splits.py"
    checkpoint = tmp_path / "checkpoint"
    source_lock = tmp_path / "source-lock"
    schema = tmp_path / "schema"
    preregistration = tmp_path / "preregistration"
    for path in (checkpoint, source_lock, schema, preregistration):
        path.write_text(path.name, encoding="utf-8")

    before = build_eval_provenance(
        checkpoint=checkpoint,
        source_lock=source_lock,
        schema=schema,
        evaluator_files=copied_files,
        preregistration=preregistration,
    )
    world_splits.write_bytes(world_splits.read_bytes() + b"\n# drift\n")
    after = build_eval_provenance(
        checkpoint=checkpoint,
        source_lock=source_lock,
        schema=schema,
        evaluator_files=copied_files,
        preregistration=preregistration,
    )

    assert before["evaluator_sha256"] != after["evaluator_sha256"]


def test_evaluator_hash_is_independent_of_input_file_order(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    source_lock = tmp_path / "source-lock"
    schema = tmp_path / "schema"
    preregistration = tmp_path / "preregistration"
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    for path in (
        checkpoint,
        source_lock,
        schema,
        preregistration,
        first,
        second,
    ):
        path.write_text(path.name, encoding="utf-8")

    forward = build_eval_provenance(
        checkpoint=checkpoint,
        source_lock=source_lock,
        schema=schema,
        evaluator_files=(first, second),
        preregistration=preregistration,
    )
    reverse = build_eval_provenance(
        checkpoint=checkpoint,
        source_lock=source_lock,
        schema=schema,
        evaluator_files=(second, first),
        preregistration=preregistration,
    )

    assert forward["evaluator_sha256"] == reverse["evaluator_sha256"]


def test_one_byte_evaluator_dependency_drift_changes_closure_hash(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    source_lock = tmp_path / "source-lock"
    schema = tmp_path / "schema"
    preregistration = tmp_path / "preregistration"
    first = tmp_path / "train" / "model.py"
    second = tmp_path / "evals" / "scorers.py"
    first.parent.mkdir()
    second.parent.mkdir()
    for path in (checkpoint, source_lock, schema, preregistration):
        path.write_text(path.name, encoding="utf-8")
    first.write_bytes(b"model")
    second.write_bytes(b"scorer")

    before = build_eval_provenance(
        checkpoint=checkpoint,
        source_lock=source_lock,
        schema=schema,
        evaluator_files=(first, second),
        preregistration=preregistration,
    )
    second.write_bytes(b"scoreS")
    after = build_eval_provenance(
        checkpoint=checkpoint,
        source_lock=source_lock,
        schema=schema,
        evaluator_files=(first, second),
        preregistration=preregistration,
    )

    assert before["evaluator_sha256"] != after["evaluator_sha256"]
