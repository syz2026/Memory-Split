#!/usr/bin/env python
"""Run the complete deterministic Task-11 relational CPU smoke."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from corpusgen.relational_build import (
    RelationalBuildConfig,
    build_relational_corpus,
)
from corpusgen.records import QAItem
from evals.relational_contracts import (
    GuardrailReport,
    load_result_schema,
    validate_result_payload,
)
from evals.relational_controls import (
    ControlID,
    EvalMode,
    build_control_view,
)
from evals.relational_generate import decode_items
from evals.relational_metrics import EXPECTED_TASKS, wilson_interval
from evals.relational_stats import (
    ALLOWED_VERDICTS,
    ContrastEstimate,
    VerdictInputs,
    decide_verdict,
)
from experiment.artifacts import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from organizer.graph_store import AtomicGraphStore
from scripts.analyze_relational import _require_expected_run_matrix
from scripts.freeze_relational_study import (
    make_fixture_freeze,
    write_freeze_manifest,
)
from scripts.make_relational_manifest import (
    RunConfig,
    build_manifest,
    write_run_manifest,
)
from scripts.package_relational_run import package_run
from scripts.run_relational_evals import (
    _states_to_rows,
    _summary,
    _write_jsonl,
    store_for_item,
)
from scripts.verify_relational_bundle import (
    verify_bundle,
    verify_extracted_bundle,
)
from train.tokenizer import get_tok
from train.trainer import Trainer, validate_run_start


SMOKE_FIXTURE = {
    "n_entities": 32,
    "total_tokens": 40_000,
    "data_seed": 1,
    "world_size": 32,
    "eval_pairs_per_task": 4,
    "eval_pairs_per_world": 4,
    "route_stats_pairs_per_task": 64,
    "guardrail_items": 4,
    "shared_text_eval_count": 4,
}
SMOKE_STEPS = 2
_MODEL = {
    "n_layer": 1,
    "n_head": 1,
    "d_model": 8,
    "ctx": 320,
    "vocab_size": 50_304,
}
_BED = (
    "Glaciers carved the valley and left long ridges of gravel behind.",
    "Wind turbines convert moving air into electricity for the local grid.",
    "The old observatory records each comet crossing the night sky.",
    "Bees communicate the location of food through patterned movements.",
)
_SIDECARS = {
    "dense": "dense.weights.bin",
    "random": "random.weights.bin",
    "selective": "selective.weights.bin",
    "split": "split.weights.bin",
}
_SCHEMAS = (
    "freeze-v1.schema.json",
    "relational-asset-receipt-v1.schema.json",
    "relational-result-v1.schema.json",
    "run-config-v1.schema.json",
    "run-manifest-v1.schema.json",
)


def _bed_stream():
    for index in itertools.count():
        yield f"{_BED[index % len(_BED)]} Deterministic passage {index}."


def _trainer_config(
    root: Path,
    corpus: Path,
    arm: str,
    *,
    steps: int,
    device: str,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    return {
        "condition": arm,
        "model": dict(_MODEL),
        "train_bin": str(corpus / "train.bin"),
        "train_weights": str(corpus / f"{arm}.weights.bin"),
        "micro_batch_size": 1,
        "tokens_per_step": _MODEL["ctx"],
        "max_steps": steps,
        "lr": 1e-3,
        "warmup_steps": 1,
        "seed": 19,
        "device": device,
        "out_dir": str(
            root / "runs" / arm if out_dir is None else out_dir
        ),
        "log_every": steps,
        "eval_every": steps,
        "snap_frac": 1.0,
        "ckpt_minutes": 999,
    }


def _same_state(left, right) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _resume_is_exact(root: Path, corpus: Path, device: str) -> bool:
    uninterrupted_cfg = _trainer_config(
        root,
        corpus,
        "dense",
        steps=3,
        device=device,
        out_dir=root / "resume" / "uninterrupted",
    )
    uninterrupted = Trainer(uninterrupted_cfg)
    uninterrupted.train_steps(3)

    interrupted_cfg = _trainer_config(
        root,
        corpus,
        "dense",
        steps=3,
        device=device,
        out_dir=root / "resume" / "interrupted",
    )
    interrupted = Trainer(interrupted_cfg)
    interrupted.train_steps(2)
    start = validate_run_start(interrupted_cfg, resume="auto")
    resumed = Trainer(start.cfg, run_start=start)
    resumed.load_validated_start(start)
    if resumed.step != 2:
        return False
    resumed.train_steps(1)

    uninterrupted_state = torch.load(
        uninterrupted.ckpt_path,
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        resumed.ckpt_path,
        map_location="cpu",
        weights_only=True,
    )
    comparable = (
        "model",
        "opt",
        "data",
        "step",
        "scheduler",
        "rng_python",
        "rng_numpy",
        "rng_torch",
        "rng_cuda",
        "rng_mps",
        "running_loss",
        "last_step_loss",
    )
    return all(
        _nested_equal(uninterrupted_state[field], resumed_state[field])
        for field in comparable
    )


def _load_smoke_items(corpus: Path) -> list[QAItem]:
    items: list[QAItem] = []
    for name in ("original.jsonl", "counterfactual.jsonl"):
        items.extend(
            QAItem(**json.loads(line))
            for line in (corpus / "eval" / name).read_text().splitlines()
            if line.strip()
        )
    expected_pairs = SMOKE_FIXTURE["eval_pairs_per_task"]
    for task in EXPECTED_TASKS:
        task_items = [item for item in items if item.task == task]
        pairs: dict[str, set[str]] = {}
        for item in task_items:
            pair_id = str(item.meta["pair_id"])
            pairs.setdefault(pair_id, set()).add(str(item.meta["variant"]))
        if (
            len(task_items) != 2 * expected_pairs
            or len(pairs) != expected_pairs
            or any(
                variants != {"original", "counterfactual"}
                for variants in pairs.values()
            )
        ):
            raise ValueError(f"incomplete smoke eval pairs for {task}")
    return items


def _evaluate_modes(
    root: Path,
    corpus: Path,
    trainer: Trainer,
    tok: Any,
) -> list[str]:
    items = _load_smoke_items(corpus)
    base_store = AtomicGraphStore.load(corpus / "eval" / "graph.jsonl")
    modes: list[str] = []
    trainer.model.eval()
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            trainer.model,
            tok,
            items,
            lambda item, enabled=memory_on: store_for_item(
                base_store,
                item,
                memory_on=enabled,
            ),
            device="cpu",
            batch_size=8,
        )
        rows = _states_to_rows(items, states)
        mode_dir = root / "evals" / f"memory_{memory}"
        _write_jsonl(mode_dir / "rows.jsonl", rows)
        summary = _summary(
            rows,
            SMOKE_FIXTURE["eval_pairs_per_task"],
            memory,
        )
        atomic_write_json(mode_dir / "summary.json", summary)
        modes.append(memory)
    return modes


def _control_matrix(
    root: Path,
    corpus: Path,
    trainer: Trainer,
    tok: Any,
) -> list[dict[str, Any]]:
    items = _load_smoke_items(corpus)
    item = next(
        value
        for value in items
        if value.task == EXPECTED_TASKS[0]
        and value.meta["variant"] == "original"
    )
    base_store = AtomicGraphStore.load(corpus / "eval" / "graph.jsonl")
    source_store = store_for_item(base_store, item, memory_on=True)
    assert source_store is not None
    cache: dict[tuple, tuple] = {}
    cells: list[dict[str, Any]] = []
    trainer.model.eval()
    for control in ControlID:
        view = build_control_view(
            item,
            source_store,
            control,
            seed=20260723,
            transformation_cache=cache,
        )
        replay_actions = (
            None
            if view.forced_actions is None
            else {str(view.item.qid): view.forced_actions}
        )
        replay_returns = (
            None
            if view.forced_returns is None
            else {str(view.item.qid): view.forced_returns}
        )
        replay_store = (
            None
            if view.forced_returns is None
            else {str(view.item.qid): view.store}
        )
        for memory in ("off", "on"):
            states = decode_items(
                trainer.model,
                tok,
                [view.item],
                store=view.store if memory == "on" else None,
                device="cpu",
                batch_size=1,
                forced_actions=replay_actions,
                forced_returns=replay_returns,
                forced_return_store=replay_store,
            )
            state = states[0]
            if (
                len(state.actions) != 6
                or len(state.rows) != 6
                or len(state.provisional_answers) != 6
                or len(state.answer_logits) != 6
            ):
                raise AssertionError("control cell did not complete six steps")
            cells.append(
                {
                    "control": control.value,
                    "memory": memory,
                    "qid": str(view.item.qid),
                    "oracle_effect": view.oracle_effect,
                    "prediction": state.provisional_answers[-1],
                    "misses": state.misses,
                    "fingerprint": view.fingerprint(),
                }
            )
    if len(cells) != len(ControlID) * len(EvalMode):
        raise AssertionError("control matrix does not contain exactly 22 cells")
    atomic_write_json(
        root / "evals" / "control-matrix.json",
        {
            "record_type": "relational_smoke_control_matrix",
            "schema_version": 1,
            "cells": cells,
        },
    )
    return cells


def _tree_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssertionError("smoke corpus contains a symlink")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    return inventory


def _check(check_id: str, **overrides: Any) -> dict[str, Any]:
    value = {
        "check_id": check_id,
        "value": 1.0,
        "reference_value": None,
        "threshold": 1.0,
        "comparison": ">=",
        "passed": True,
        "numerator": 1,
        "denominator": 1,
    }
    value.update(overrides)
    return value


def _synthetic_guardrail(*, passed: bool) -> GuardrailReport:
    guards = {
        "factual_job": {
            "passed": True,
            "checks": [
                _check("split_on_recall_floor", threshold=0.95),
                _check(
                    "split_on_recall_noninferiority",
                    reference_value=0.0,
                    threshold=-0.02,
                ),
            ],
        },
        "split_off_leakage": {
            "passed": passed,
            "checks": [
                _check(
                    "split_off_recall",
                    value=0.0 if passed else 0.05,
                    threshold=0.05,
                    comparison="<",
                    passed=passed,
                    numerator=0 if passed else 1,
                    denominator=20,
                ),
                _check(
                    "split_off_recognition_wilson_hi",
                    value=wilson_interval(0, 20)[1],
                    threshold=0.30,
                    comparison="<",
                    numerator=0,
                    denominator=20,
                ),
            ],
        },
        "retrieval_procedure": {
            "passed": True,
            "checks": [
                _check("split_off_first_hop", threshold=0.75),
                _check(
                    "gold_return_path_noninferiority",
                    reference_value=0.0,
                    threshold=-0.05,
                ),
            ],
        },
        "relation_rule": {
            "passed": True,
            "checks": [
                _check(
                    "rule_noninferiority",
                    reference_value=0.0,
                    threshold=-0.02,
                )
            ],
        },
        "natural_text": {
            "passed": True,
            "checks": [
                _check(
                    "bpb_noninferiority",
                    value=1.0,
                    reference_value=1.0,
                    threshold=1.01,
                    comparison="<=",
                    numerator=None,
                    denominator=None,
                )
            ],
        },
        "instrument_integrity": {
            "passed": True,
            "checks": [
                _check(
                    "integrity_conjunction",
                    value=True,
                    threshold=True,
                    comparison="==",
                    numerator=None,
                    denominator=None,
                )
            ],
        },
    }
    return GuardrailReport.from_dict(
        {
            "record_type": "guardrail_report",
            "schema_version": 1,
            "split_checkpoint_sha256": "1" * 64,
            "dense_checkpoint_sha256": "6" * 64,
            "model_id": "d160m",
            "seed": 1001,
            "raw_token_count": 811_104_000,
            "evaluator_sha256": "2" * 64,
            "data_sha256": "3" * 64,
            "relation_schema_sha256": "4" * 64,
            "split_configuration_sha256": "c" * 64,
            "dense_configuration_sha256": "e" * 64,
            "result_schema_sha256": "d" * 64,
            "split_result_provenance_sha256": "5" * 64,
            "dense_result_provenance_sha256": "f" * 64,
            "study_provenance_sha256": "0" * 64,
            "pairing_receipt_sha256": "7" * 64,
            "split_guardrail_source_sha256": "8" * 64,
            "dense_guardrail_source_sha256": "9" * 64,
            "split_matrix_manifest_sha256": "a" * 64,
            "dense_matrix_manifest_sha256": "b" * 64,
            "guards": guards,
            "confirmatory_passed": passed,
        }
    )


def _estimate(mean: float, low: float, high: float) -> ContrastEstimate:
    return ContrastEstimate(
        mean=mean,
        ci_lo=low,
        ci_hi=high,
        seed_deltas=(mean,) * 5,
        cohen_dz=None,
        effect_note="Task-11 synthetic branch fixture",
    )


def _verdict_inputs(branch: str) -> VerdictInputs:
    task_positive = {task: 0.001 for task in EXPECTED_TASKS}
    task_zero = {task: 0.0 for task in EXPECTED_TASKS}
    if branch == "validated":
        return VerdictInputs(
            split_dense_360=_estimate(0.02, 0.001, 0.04),
            split_dense_160_high=_estimate(0.02, 0.001, 0.04),
            dose_interaction_160=_estimate(0.01, 0.001, 0.03),
            split_random_160_high=_estimate(0.01, 0.001, 0.03),
            task_means_360=task_positive,
            task_means_160_high=task_positive,
            guardrail_reports=(_synthetic_guardrail(passed=True),),
        )
    if branch == "practical_null":
        return VerdictInputs(
            split_dense_360=_estimate(0.0, -0.01, 0.019),
            split_dense_160_high=_estimate(0.0, -0.01, 0.019),
            dose_interaction_160=_estimate(-0.01, -0.02, 0.0),
            split_random_160_high=_estimate(0.0, -0.01, 0.01),
            task_means_360=task_zero,
            task_means_160_high=task_zero,
            guardrail_reports=(_synthetic_guardrail(passed=True),),
        )
    if branch == "inconclusive":
        return VerdictInputs(
            split_dense_360=_estimate(0.01, -0.01, 0.04),
            split_dense_160_high=_estimate(0.01, -0.01, 0.04),
            dose_interaction_160=_estimate(0.0, -0.01, 0.02),
            split_random_160_high=_estimate(0.0, -0.01, 0.02),
            task_means_360=task_zero,
            task_means_160_high=task_zero,
            guardrail_reports=(_synthetic_guardrail(passed=True),),
        )
    if branch == "invalid":
        value = _verdict_inputs("validated")
        return VerdictInputs(
            split_dense_360=value.split_dense_360,
            split_dense_160_high=value.split_dense_160_high,
            dose_interaction_160=value.dose_interaction_160,
            split_random_160_high=value.split_random_160_high,
            task_means_360=value.task_means_360,
            task_means_160_high=value.task_means_160_high,
            guardrail_reports=(_synthetic_guardrail(passed=False),),
        )
    raise ValueError(f"unknown synthetic verdict branch: {branch}")


def _synthetic_analysis(root: Path, manifest: Any) -> list[str]:
    runs = {
        run.key: {
            "run_id": run.run_id,
            "config_sha256": run.config_sha256,
            "synthetic_score": index / 1000,
        }
        for index, run in enumerate(manifest.runs)
    }
    _require_expected_run_matrix(runs, manifest)
    verdicts = {
        branch: decide_verdict(_verdict_inputs(branch))
        for branch in ALLOWED_VERDICTS
    }
    if verdicts != {branch: branch for branch in ALLOWED_VERDICTS}:
        raise AssertionError("synthetic analysis missed a frozen verdict branch")
    atomic_write_json(
        root / "synthetic-analysis.json",
        {
            "record_type": "relational_smoke_analysis",
            "schema_version": 1,
            "run_manifest_sha256": manifest.manifest_sha256,
            "runs": [
                {"key": list(key), **runs[key]} for key in sorted(runs)
            ],
            "verdicts": verdicts,
        },
    )
    return list(ALLOWED_VERDICTS)


def _validate_schemas(
    source_root: Path,
    freeze: Any,
    manifest: Any,
) -> list[str]:
    for name in _SCHEMAS:
        schema = json.loads((source_root / "schemas" / name).read_bytes())
        if (
            not isinstance(schema, dict)
            or schema.get("$schema")
            != "https://json-schema.org/draft/2020-12/schema"
        ):
            raise AssertionError(f"invalid Task-11 schema: {name}")
    if manifest.freeze.to_dict() != freeze.to_dict():
        raise AssertionError("manifest is not bound to fixture freeze")
    for run in manifest.runs:
        if RunConfig.from_dict(run.to_dict()) != run:
            raise AssertionError("run config failed semantic schema round-trip")
    load_result_schema(source_root / "schemas" / "relational-result-v1.schema.json")
    validate_result_payload(_synthetic_guardrail(passed=True).to_dict())
    return sorted(_SCHEMAS)


def _build_contracts(root: Path) -> tuple[Any, Any, Path, Path]:
    directory = root / "contracts"
    directory.mkdir()
    freeze = make_fixture_freeze()
    manifest = build_manifest(freeze)
    freeze_path = write_freeze_manifest(directory / "freeze.json", freeze)
    manifest_path = write_run_manifest(
        directory / "run-manifest.json",
        manifest,
    )
    return freeze, manifest, freeze_path, manifest_path


def _package_and_verify(
    root: Path,
    source_root: Path,
    freeze_path: Path,
    manifest_path: Path,
) -> tuple[bool, bool, bool]:
    bundles = root / "bundles"
    bundles.mkdir()
    first = package_run(
        bundles / "first.tar.gz",
        source_root=source_root,
        freeze_path=freeze_path,
        run_manifest_path=manifest_path,
        require_clean=False,
    )
    second = package_run(
        bundles / "second.tar.gz",
        source_root=source_root,
        freeze_path=freeze_path,
        run_manifest_path=manifest_path,
        require_clean=False,
    )
    deterministic = first.read_bytes() == second.read_bytes()
    extraction = root / "extracted-bundle"
    archive_report = verify_bundle(first, extract_to=extraction)
    extracted_report = verify_extracted_bundle(extraction)
    return (
        deterministic,
        archive_report["verified"] is True
        and archive_report["offline_tests_passed"] is True,
        extracted_report["verified"] is True
        and extracted_report["offline_tests_passed"] is True,
    )


def run_smoke(
    out_dir: Path | str,
    *,
    steps: int = SMOKE_STEPS,
    device: str = "cpu",
    source_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build, train, resume, analyze, package, and independently verify."""

    if steps != SMOKE_STEPS:
        raise ValueError("the local smoke contract requires exactly two steps")
    if device != "cpu":
        raise ValueError("the local smoke contract requires device='cpu'")
    source = (
        Path(__file__).resolve().parents[1]
        if source_root is None
        else Path(source_root)
    )
    source = source.resolve(strict=True)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"smoke output directory must be empty: {root}")

    fixture_path = source / "fixtures" / "relational-smoke.json"
    fixture = json.loads(fixture_path.read_bytes())
    if fixture != {**SMOKE_FIXTURE, "steps": SMOKE_STEPS}:
        raise AssertionError("packaged smoke fixture drifted from executable smoke")

    tok = get_tok()
    corpus = root / "corpus"
    repeated_corpus = root / "corpus-repeat"
    build_reports = [
        build_relational_corpus(
            RelationalBuildConfig(**SMOKE_FIXTURE),
            tok,
            _bed_stream(),
            destination,
        )
        for destination in (corpus, repeated_corpus)
    ]
    if not all(
        all(report["checks"].values()) for report in build_reports
    ):
        raise AssertionError("smoke corpus failed relational build checks")
    first_inventory = _tree_inventory(corpus)
    second_inventory = _tree_inventory(repeated_corpus)
    corpus_deterministic = first_inventory == second_inventory
    sidecars = sorted(
        label
        for label, name in _SIDECARS.items()
        if (corpus / name).is_file()
    )
    if sidecars != sorted(_SIDECARS):
        raise AssertionError("smoke corpus does not contain four exact sidecars")

    dense = Trainer(
        _trainer_config(root, corpus, "dense", steps=steps, device=device)
    )
    split = Trainer(
        _trainer_config(root, corpus, "split", steps=steps, device=device)
    )
    initial_state = {
        name: value.detach().clone()
        for name, value in dense.model.state_dict().items()
    }
    dense.model.load_state_dict(initial_state)
    split.model.load_state_dict(initial_state)
    if not _same_state(dense.model, split.model):
        raise AssertionError("Dense and Split initial states differ")

    dense.train_steps(steps)
    split.train_steps(steps)
    resume_exact = _resume_is_exact(root, corpus, device)
    modes = _evaluate_modes(root, corpus, split, tok)
    control_cells = _control_matrix(root, corpus, split, tok)
    pair_count = SMOKE_FIXTURE["eval_pairs_per_task"]
    pairs_complete = all(
        json.loads(
            (root / "evals" / f"memory_{mode}" / "summary.json").read_bytes()
        )["n_pairs_per_task"]
        == pair_count
        for mode in modes
    )

    freeze, manifest, freeze_path, manifest_path = _build_contracts(root)
    verdict_branches = _synthetic_analysis(root, manifest)
    schemas = _validate_schemas(source, freeze, manifest)
    bundle_deterministic, bundle_verified, extracted_verified = (
        _package_and_verify(
            root,
            source,
            freeze_path,
            manifest_path,
        )
    )

    report = {
        "shared_stream": (
            dense.cfg["train_bin"] == split.cfg["train_bin"]
            and dense.data.n_tokens == split.data.n_tokens
        ),
        "dense_steps": dense.step,
        "split_steps": split.step,
        "resume_exact": resume_exact,
        "resume_compared_next_update": True,
        "memory_modes": modes,
        "pairs_complete": pairs_complete,
        "corpus_builds": 2,
        "corpus_byte_deterministic": corpus_deterministic,
        "corpus_sha256": canonical_sha256(first_inventory),
        "sidecars": sidecars,
        "sidecar_sha256": {
            label: sha256_file(corpus / _SIDECARS[label])
            for label in sidecars
        },
        "controls": [control.value for control in ControlID],
        "eval_cells": len(control_cells),
        "synthetic_run_count": len(manifest.runs),
        "matrix_runs": len(manifest.runs),
        "verdict_branches": verdict_branches,
        "schemas_validated": schemas,
        "bundle_byte_deterministic": bundle_deterministic,
        "bundle_verified": bundle_verified,
        "extracted_bundle_verified": extracted_verified,
    }
    required_true = (
        "shared_stream",
        "resume_exact",
        "pairs_complete",
        "corpus_byte_deterministic",
        "bundle_byte_deterministic",
        "bundle_verified",
        "extracted_bundle_verified",
    )
    if (
        not all(report[name] for name in required_true)
        or report["dense_steps"] != steps
        or report["split_steps"] != steps
        or report["memory_modes"] != ["off", "on"]
        or report["eval_cells"] != 22
        or report["synthetic_run_count"] != 35
    ):
        raise AssertionError(f"local relational smoke failed: {report}")
    atomic_write_json(root / "smoke-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        help="new or explicitly empty output directory",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)
    report = run_smoke(args.out, device=args.device)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
