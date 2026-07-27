#!/usr/bin/env python
"""Aggregate the exact 35-run relational matrix and publish the frozen verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.checkpoint_binding import load_run_configuration
from evals.relational_contracts import (
    CheckpointSummary,
    EvalRow,
    GuardrailReport,
    canonical_json_bytes,
    validate_published_evaluation,
)
from evals.relational_controls import ControlID, EvalMode
from evals.relational_metrics import EXPECTED_TASKS
from evals.relational_pairing import (
    PairingReceipt,
    build_pairing_receipt,
    validate_pairing_receipt,
)
from evals.relational_reporting import (
    REPORT_SECTIONS,
    AnalysisSection,
    build_analysis_document,
    publish_analysis_bundle,
    validate_analysis_bundle,
)
from evals.relational_stats import (
    BOOTSTRAP_VERSION,
    CONFIRMATORY_SEEDS as SEEDS,
    FROZEN_N_BOOT,
    FROZEN_PERCENTILE_INDICES,
    MAX_BOOTSTRAP_CHUNK,
    VerdictInputs,
    hierarchical_paired_bootstrap,
    hierarchical_paired_contrasts,
    paired_task_means,
    validate_analysis_cell,
)
from experiment.provenance import verify_source_provenance
from scripts.make_relational_manifest import (
    RunManifest,
    load_run_manifest,
    require_launchable,
)
from scripts.run_relational_evals import (
    _ValidatedCheckpointMatrix,
    _validate_exact_checkpoint_matrix,
)

def _validated_run_manifest(
    manifest: RunManifest | Mapping[str, object],
) -> RunManifest:
    if isinstance(manifest, RunManifest):
        return RunManifest.from_dict(manifest.to_dict())
    return RunManifest.from_dict(manifest)


def expected_run_keys(
    run_manifest: RunManifest | Mapping[str, object],
) -> set[tuple[str, str, str, int]]:
    manifest = _validated_run_manifest(run_manifest)
    return {run.key for run in manifest.runs}


def _validate_run_config_against_manifest(
    cfg: Mapping[str, object],
    run_manifest: RunManifest | Mapping[str, object],
) -> tuple[str, str, str, int]:
    if not isinstance(cfg, Mapping):
        raise ValueError("loaded run config must be an object")
    manifest = _validated_run_manifest(run_manifest)
    if (
        not isinstance(cfg.get("model"), str)
        or not isinstance(cfg.get("condition"), str)
        or not isinstance(cfg.get("load"), str)
        or isinstance(cfg.get("seed"), bool)
        or not isinstance(cfg.get("seed"), int)
    ):
        raise ValueError("loaded run config key is invalid")
    key = (
        cfg.get("model"),
        cfg.get("condition"),
        cfg.get("load"),
        cfg.get("seed"),
    )
    expected = {run.key: run for run in manifest.runs}.get(key)
    if expected is None:
        raise ValueError("loaded run config key is absent from run manifest")
    for field, value in expected.to_dict().items():
        if cfg.get(field) != value:
            raise ValueError(
                f"loaded run config {field} does not match run manifest"
            )
    runtime_fields = {
        "micro_batch_size": 8,
        "tokens_per_step": expected.tokens_per_step,
        "max_steps": expected.steps,
        "total_tokens": expected.actual_raw_positions,
        "lr": expected.optimizer["lr"],
        "weight_decay": expected.optimizer["weight_decay"],
        "warmup_steps": expected.scheduler["warmup_steps"],
        "ctx": expected.architecture["ctx"],
    }
    for field, value in runtime_fields.items():
        if cfg.get(field) != value:
            raise ValueError(
                f"loaded runtime field {field} does not match run manifest"
            )
    if cfg.get("train_mask") is not None:
        raise ValueError("Task-9 runtime config cannot add a train mask")
    if cfg.get("compile", False) is not False:
        raise ValueError("Task-9 runtime compile mode is not frozen")

    def runtime_root(value: object, relative: str, name: str) -> Path:
        if not isinstance(value, str):
            raise ValueError(f"loaded runtime {name} must be a path")
        candidate = Path(value)
        suffix = Path(relative)
        if (
            not candidate.is_absolute()
            or ".." in candidate.parts
            or len(candidate.parts) <= len(suffix.parts)
            or candidate.parts[-len(suffix.parts) :] != suffix.parts
            or candidate.is_symlink()
            or candidate.resolve(strict=False) != candidate
        ):
            raise ValueError(
                f"loaded runtime {name} does not match its relative manifest path"
            )
        root = candidate
        for _ in suffix.parts:
            root = root.parent
        return root

    data_root = runtime_root(cfg.get("train_bin"), expected.data_rel, "train_bin")
    weights_root = runtime_root(
        cfg.get("train_weights"),
        expected.weights_rel,
        "train_weights",
    )
    runtime_root(cfg.get("out_dir"), expected.out_rel, "out_dir")
    if data_root != weights_root:
        raise ValueError("loaded runtime data roots do not match")
    return expected.key


def _select_frozen_checkpoint(matrices, expected_raw_positions: int):
    matching = [
        matrix
        for matrix in matrices
        if matrix[
            (EvalMode.MEMORY_ON, ControlID.CORRECT)
        ].raw_token_count
        == expected_raw_positions
    ]
    if not matching:
        raise ValueError("run is missing the frozen raw-token checkpoint")
    if len(matching) > 1:
        raise ValueError("run has duplicate frozen raw-token checkpoints")
    return matching[0]


def _validate_checkpoint_freeze_identity(
    anchor,
    run_manifest: RunManifest | Mapping[str, object],
) -> None:
    manifest = _validated_run_manifest(run_manifest)
    if (
        getattr(anchor, "relation_schema_sha256", None)
        != manifest.freeze.artifact_sha256["relation_schema"]
    ):
        raise ValueError(
            "checkpoint relation schema does not match the study freeze"
        )


def _require_expected_run_matrix(
    runs,
    run_manifest: RunManifest | Mapping[str, object],
) -> None:
    actual = set(runs)
    expected = expected_run_keys(run_manifest)
    if actual != expected:
        raise ValueError(
            "run matrix mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _load_complete_runs(
    root: Path,
    run_manifest: RunManifest | Mapping[str, object],
) -> dict:
    manifest = _validated_run_manifest(run_manifest)
    if root.is_symlink():
        raise ValueError("runs root cannot be a symlink")
    if not root.is_dir():
        raise FileNotFoundError(root)
    if root.resolve(strict=True) != root.absolute():
        raise ValueError("runs root path is not canonical")
    runs = {}
    for directory in sorted(root.iterdir()):
        if directory.is_symlink():
            raise ValueError("runs root cannot contain symlinked entries")
        if not directory.is_dir():
            continue
        config_path = directory / "config.yaml"
        if not os.path.lexists(config_path):
            continue
        cfg, config_identities = load_run_configuration(directory)
        if (
            not isinstance(cfg.get("model"), str)
            or not cfg["model"]
            or not isinstance(cfg.get("condition"), str)
            or not cfg["condition"]
            or not isinstance(cfg.get("load"), str)
            or not cfg["load"]
            or isinstance(cfg.get("seed"), bool)
            or not isinstance(cfg.get("seed"), int)
        ):
            raise ValueError(f"invalid run config for {directory.name}")
        key = _validate_run_config_against_manifest(cfg, manifest)
        expected_run = next(run for run in manifest.runs if run.key == key)
        if Path(str(cfg["out_dir"])).absolute() != directory.absolute():
            raise ValueError("run directory does not match frozen out_rel")
        evals = directory / "evals"
        if evals.is_symlink() or not evals.is_dir():
            raise ValueError(
                f"incomplete relational evaluation for {directory.name}"
            )
        obsolete = {
            "memory_on",
            "memory_off",
            "guardrails.json",
        } & {entry.name for entry in evals.iterdir()}
        if obsolete:
            raise ValueError(
                "obsolete relational evaluation layout is retired"
            )
        checkpoint_dirs = [
            entry
            for entry in evals.iterdir()
            if (
                entry.is_dir()
                and not entry.is_symlink()
                and len(entry.name) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in entry.name
                )
            )
        ]
        if not checkpoint_dirs:
            raise ValueError(
                f"no exact checkpoint matrix for {directory.name}"
            )
        allowed_entries = {checkpoint.name for checkpoint in checkpoint_dirs}
        if cfg["condition"] == "split":
            allowed_entries.update(
                {"guardrail-report.json", "pairing-receipt.json"}
            )
        if {entry.name for entry in evals.iterdir()} != allowed_entries:
            raise ValueError(
                "relational evaluation tree contains legacy or ad-hoc "
                "analysis artifacts"
            )
        matrix_entries = [
            (checkpoint, _validate_exact_checkpoint_matrix(checkpoint))
            for checkpoint in checkpoint_dirs
        ]
        for checkpoint, matrix in matrix_entries:
            anchor = matrix[(EvalMode.MEMORY_ON, ControlID.CORRECT)]
            if (
                matrix.checkpoint_dir.absolute() != checkpoint.absolute()
                or anchor.checkpoint_sha256 != checkpoint.name
                or anchor.model_id != cfg["model"]
                or anchor.arm != cfg["condition"]
                or anchor.seed != cfg["seed"]
                or anchor.configuration_sha256
                != config_identities["configuration_sha256"]
            ):
                raise ValueError(
                    "checkpoint matrix configuration identity mismatch"
                )
        matrices = [matrix for _, matrix in matrix_entries]
        selected = _select_frozen_checkpoint(
            matrices,
            expected_run.actual_raw_positions,
        )
        selected_anchor = selected[
            (EvalMode.MEMORY_ON, ControlID.CORRECT)
        ]
        _validate_checkpoint_freeze_identity(selected_anchor, manifest)
        if (
            selected_anchor.model_id != str(cfg["model"])
            or selected_anchor.arm != str(cfg["condition"])
            or selected_anchor.seed != int(cfg["seed"])
        ):
            raise ValueError("run config and checkpoint matrix identity mismatch")
        primary_rows: list[EvalRow] = []
        revalidated_anchor = validate_published_evaluation(
            selected.checkpoint_dir
            / EvalMode.MEMORY_ON.value
            / ControlID.CORRECT.value,
            row_consumer=primary_rows.append,
        )
        if revalidated_anchor != selected_anchor:
            raise ValueError("selected primary summary changed during row load")
        primary_rows = list(
            validate_analysis_cell(
                primary_rows,
                selected_anchor,
                expected_arm=str(cfg["condition"]),
                expected_seed=int(cfg["seed"]),
            )
        )
        guardrail_path = evals / "guardrail-report.json"
        receipt_path = evals / "pairing-receipt.json"
        report = None
        receipt = None
        if cfg["condition"] == "split":
            if (
                guardrail_path.is_symlink()
                or not guardrail_path.is_file()
                or receipt_path.is_symlink()
                or not receipt_path.is_file()
            ):
                raise ValueError(
                    "Split run requires strict guardrail report and "
                    "pairing receipt"
                )
            report_content = guardrail_path.read_bytes()
            report = GuardrailReport.from_dict(json.loads(report_content))
            if report_content != canonical_json_bytes(report):
                raise ValueError("guardrail report is not canonical")
            receipt_content = receipt_path.read_bytes()
            receipt = validate_pairing_receipt(json.loads(receipt_content))
        elif guardrail_path.exists() or receipt_path.exists():
            raise ValueError(
                "non-Split run cannot publish confirmatory guardrail artifacts"
            )
        if key in runs:
            raise ValueError(f"duplicate run key: {key}")
        runs[key] = {
            "cfg": cfg,
            "configuration_sha256": config_identities[
                "configuration_sha256"
            ],
            "on": selected[(EvalMode.MEMORY_ON, ControlID.CORRECT)],
            "off": selected[(EvalMode.MEMORY_OFF, ControlID.CORRECT)],
            "rows": tuple(primary_rows),
            "matrix": selected,
            "guardrail_report": report,
            "guardrail_report_path": guardrail_path if report is not None else None,
            "guardrail_report_content": (
                report_content if report is not None else None
            ),
            "pairing_receipt": receipt,
            "pairing_receipt_path": receipt_path if receipt is not None else None,
            "pairing_receipt_content": (
                receipt_content if receipt is not None else None
            ),
            "directory": str(directory),
        }
    _require_expected_run_matrix(runs, manifest)
    _collect_guardrails(runs)
    return runs


def _condition_rows(
    runs: dict,
    model: str,
    condition: str,
    load: str,
) -> tuple[EvalRow, ...]:
    combined: list[EvalRow] = []
    for seed in SEEDS:
        run = runs[(model, condition, load, seed)]
        summary = run.get("on")
        rows = run.get("rows")
        if not isinstance(summary, CheckpointSummary):
            raise ValueError(
                "analysis requires validated CheckpointSummary values"
            )
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("analysis requires typed primary EvalRow sequences")
        combined.extend(
            validate_analysis_cell(
                rows,
                summary,
                expected_arm=condition,
                expected_seed=seed,
            )
        )
    return tuple(combined)


def _validate_bound_guardrail_report(
    split_run: dict,
    dense_run: dict,
) -> GuardrailReport:
    raw = split_run.get("guardrail_report")
    if raw is None:
        raise ValueError("Task 8 requires strict GuardrailReport artifacts")
    report = GuardrailReport.from_dict(
        raw.to_dict() if isinstance(raw, GuardrailReport) else raw
    )
    split_matrix = split_run.get("matrix")
    dense_matrix = dense_run.get("matrix")
    split_anchor = split_run.get("on")
    dense_anchor = dense_run.get("on")
    if (
        not isinstance(split_matrix, _ValidatedCheckpointMatrix)
        or not isinstance(dense_matrix, _ValidatedCheckpointMatrix)
        or not isinstance(split_anchor, CheckpointSummary)
        or not isinstance(dense_anchor, CheckpointSummary)
    ):
        raise ValueError(
            "Task 8 requires provenance-bound strict GuardrailReport artifacts"
        )
    split_directory = Path(split_run.get("directory", ""))
    dense_directory = Path(dense_run.get("directory", ""))
    report_path = split_run.get("guardrail_report_path")
    receipt_path = split_run.get("pairing_receipt_path")
    expected_report_path = (
        split_directory / "evals" / "guardrail-report.json"
    )
    expected_receipt_path = (
        split_directory / "evals" / "pairing-receipt.json"
    )
    report_content = split_run.get("guardrail_report_content")
    if (
        not isinstance(report_path, Path)
        or report_path.absolute() != expected_report_path.absolute()
        or report_path.is_symlink()
        or not report_path.is_file()
        or not isinstance(report_content, bytes)
        or report_content != canonical_json_bytes(report)
        or report_path.read_bytes() != report_content
    ):
        raise ValueError(
            "guardrail report is not at the canonical report location"
        )
    if (
        not isinstance(receipt_path, Path)
        or receipt_path.absolute() != expected_receipt_path.absolute()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        raise ValueError(
            "pairing receipt is not at the canonical Split run location"
        )
    if (
        split_matrix.checkpoint_dir.absolute()
        != (
            split_directory
            / "evals"
            / split_anchor.checkpoint_sha256
        ).absolute()
        or dense_matrix.checkpoint_dir.absolute()
        != (
            dense_directory
            / "evals"
            / dense_anchor.checkpoint_sha256
        ).absolute()
    ):
        raise ValueError("guardrail report checkpoint matrix location mismatch")
    raw_receipt = split_run.get("pairing_receipt")
    receipt = validate_pairing_receipt(
        raw_receipt.to_dict()
        if isinstance(raw_receipt, PairingReceipt)
        else raw_receipt
    )
    expected_receipt = build_pairing_receipt(
        split_anchor,
        dense_anchor,
        split_run.get("cfg"),
        dense_run.get("cfg"),
    )
    if receipt != expected_receipt:
        raise ValueError("pairing receipt input binding mismatch")
    receipt_content = split_run.get("pairing_receipt_content")
    if (
        not isinstance(receipt_content, bytes)
        or receipt_content != canonical_json_bytes(receipt)
        or receipt_path.read_bytes() != receipt_content
    ):
        raise ValueError("pairing receipt bytes are not canonically bound")
    expected = {
        "split_checkpoint_sha256": split_anchor.checkpoint_sha256,
        "dense_checkpoint_sha256": dense_anchor.checkpoint_sha256,
        "model_id": split_anchor.model_id,
        "seed": split_anchor.seed,
        "raw_token_count": split_anchor.raw_token_count,
        "evaluator_sha256": split_anchor.evaluator_sha256,
        "data_sha256": split_anchor.data_sha256,
        "relation_schema_sha256": split_anchor.relation_schema_sha256,
        "split_configuration_sha256": (
            split_anchor.configuration_sha256
        ),
        "dense_configuration_sha256": (
            dense_anchor.configuration_sha256
        ),
        "result_schema_sha256": split_anchor.result_schema_sha256,
        "split_result_provenance_sha256": (
            split_anchor.provenance_sha256
        ),
        "dense_result_provenance_sha256": (
            dense_anchor.provenance_sha256
        ),
        "study_provenance_sha256": receipt.study_provenance_sha256,
        "pairing_receipt_sha256": receipt.receipt_sha256,
        "split_guardrail_source_sha256": (
            split_matrix.guardrail_artifact_sha256
        ),
        "dense_guardrail_source_sha256": (
            dense_matrix.guardrail_artifact_sha256
        ),
        "split_matrix_manifest_sha256": (
            split_matrix.matrix_manifest_sha256
        ),
        "dense_matrix_manifest_sha256": (
            dense_matrix.matrix_manifest_sha256
        ),
    }
    for field, expected_value in expected.items():
        if getattr(report, field) != expected_value:
            raise ValueError(f"guardrail report {field} binding mismatch")
    for field in (
        "model_id",
        "seed",
        "raw_token_count",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "result_schema_sha256",
    ):
        if getattr(dense_anchor, field) != getattr(split_anchor, field):
            raise ValueError(
                f"Split/Dense checkpoint {field} binding mismatch"
            )
    return report


def _collect_guardrails(runs: dict) -> tuple[GuardrailReport, ...]:
    reports: list[GuardrailReport] = []
    for key, run in sorted(runs.items()):
        if key[1] != "split":
            continue
        if run.get("guardrail_report") is None:
            raise ValueError(
                "Task 8 requires strict GuardrailReport artifacts"
            )
        dense_key = (key[0], "dense", key[2], key[3])
        dense_run = runs.get(dense_key)
        if dense_run is None:
            raise ValueError(
                "Task 8 requires the corresponding Dense checkpoint"
            )
        report = _validate_bound_guardrail_report(
            run,
            dense_run,
        )
        reports.append(report)
    if not reports:
        raise ValueError("no strict guardrail reports were collected")
    return tuple(reports)


def _run_matrix_records(runs: Mapping) -> tuple[dict, ...]:
    records = []
    for (model, arm, load, seed), run in sorted(runs.items()):
        summary = run.get("on")
        if not isinstance(summary, CheckpointSummary):
            raise ValueError("run matrix requires strict primary summaries")
        records.append(
            {
                "model": model,
                "arm": arm,
                "load": load,
                "seed": seed,
                "checkpoint_sha256": summary.checkpoint_sha256,
                "raw_token_count": summary.raw_token_count,
                "configuration_sha256": summary.configuration_sha256,
                "result_provenance_sha256": summary.provenance_sha256,
                "rows_sha256": summary.rows_sha256,
            }
        )
    return tuple(records)


def analyze_runs(
    runs: dict,
    *,
    run_manifest: RunManifest | Mapping[str, object],
    rng_seed: int | None = None,
    input_bindings: Mapping[str, object] | None = None,
    secondary_analyses: Mapping[str, object] | None = None,
) -> dict:
    """Run only the frozen row-level confirmatory estimator and decision."""

    manifest = _validated_run_manifest(run_manifest)
    _require_expected_run_matrix(runs, manifest)
    low_load = manifest.load_labels["low"]
    high_load = manifest.load_labels["high"]
    confirmation_load = manifest.load_labels["confirmation"]
    guardrails = _collect_guardrails(runs)
    if rng_seed is None:
        raise ValueError("the preregistered bootstrap rng seed is required")
    if input_bindings is None:
        raise ValueError("analysis input bindings are required")
    secondary = {} if secondary_analyses is None else secondary_analyses

    rows_160 = {
        "dense_low": _condition_rows(
            runs, "d160m", "dense", low_load
        ),
        "split_low": _condition_rows(
            runs, "d160m", "split", low_load
        ),
        "dense_high": _condition_rows(
            runs, "d160m", "dense", high_load
        ),
        "split_high": _condition_rows(
            runs, "d160m", "split", high_load
        ),
        "random_high": _condition_rows(
            runs, "d160m", "random", high_load
        ),
    }
    estimates_160 = hierarchical_paired_contrasts(
        rows_160,
        {
            "split_dense_low": {
                "split_low": 1.0,
                "dense_low": -1.0,
            },
            "split_dense_high": {
                "split_high": 1.0,
                "dense_high": -1.0,
            },
            "dose_interaction": {
                "split_high": 1.0,
                "dense_high": -1.0,
                "split_low": -1.0,
                "dense_low": 1.0,
            },
            "split_random_high": {
                "split_high": 1.0,
                "random_high": -1.0,
            },
        },
        seeds=SEEDS,
        n_boot=FROZEN_N_BOOT,
        rng_seed=rng_seed,
        identity_groups={
            "d160m_low": ("dense_low", "split_low"),
            "d160m_high": (
                "dense_high",
                "random_high",
                "split_high",
            ),
        },
    )
    split_360 = _condition_rows(
        runs, "d360m", "split", confirmation_load
    )
    dense_360 = _condition_rows(
        runs, "d360m", "dense", confirmation_load
    )
    estimate_360 = hierarchical_paired_bootstrap(
        split_360,
        dense_360,
        seeds=SEEDS,
        n_boot=FROZEN_N_BOOT,
        rng_seed=rng_seed,
    )
    inputs = VerdictInputs(
        split_dense_360=estimate_360,
        split_dense_160_high=estimates_160["split_dense_high"],
        dose_interaction_160=estimates_160["dose_interaction"],
        split_random_160_high=estimates_160["split_random_high"],
        task_means_360=paired_task_means(
            split_360,
            dense_360,
            seeds=SEEDS,
        ),
        task_means_160_high=paired_task_means(
            rows_160["split_high"],
            rows_160["dense_high"],
            seeds=SEEDS,
        ),
        guardrail_reports=guardrails,
    )
    expected_receipts = sorted(
        report.pairing_receipt_sha256 for report in guardrails
    )
    supplied_receipts = (
        input_bindings.get("guardrail_receipt_sha256")
        if isinstance(input_bindings, Mapping)
        else None
    )
    if (
        isinstance(supplied_receipts, (str, bytes))
        or not isinstance(supplied_receipts, Sequence)
        or sorted(supplied_receipts) != expected_receipts
    ):
        raise ValueError(
            "analysis input bindings do not match strict guardrail receipts"
        )
    return build_analysis_document(
        verdict_inputs=inputs,
        input_bindings=input_bindings,
        run_matrix=_run_matrix_records(runs),
        bootstrap_config={
            "version": BOOTSTRAP_VERSION,
            "n_boot": FROZEN_N_BOOT,
            "rng_seed": rng_seed,
            "chunk_size": MAX_BOOTSTRAP_CHUNK,
            "percentile_indices": list(FROZEN_PERCENTILE_INDICES),
        },
        secondary_analyses=secondary,
    )


def _task_pair_accuracy(
    summary: CheckpointSummary,
    task: str,
) -> float:
    tasks = summary.metrics["by_task"]
    if set(tasks) != set(EXPECTED_TASKS):
        raise ValueError("summary task set does not match frozen strata")
    value = tasks[task]["pair_accuracy"]["value"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("task pair accuracy must be finite and in [0, 1]")
    return float(value)


def _pair_composite(summary: CheckpointSummary) -> float:
    return sum(
        _task_pair_accuracy(summary, task) for task in EXPECTED_TASKS
    ) / len(EXPECTED_TASKS)


def _numeric_leaves(
    value: object,
    *,
    prefix: str = "",
):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield prefix or "value", float(value)
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(value[key], prefix=child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    ):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from _numeric_leaves(item, prefix=child)


def build_report_sections(
    runs: dict,
    analysis: Mapping[str, object],
    *,
    run_manifest: RunManifest | Mapping[str, object],
    guardrails: Sequence[GuardrailReport] | None = None,
    secondary_analyses: Mapping[str, object] | None = None,
) -> dict[str, AnalysisSection]:
    """Derive report-only rows without feeding them back into the verdict."""

    manifest = _validated_run_manifest(run_manifest)
    _require_expected_run_matrix(runs, manifest)
    high_load = manifest.load_labels["high"]
    confirmation_load = manifest.load_labels["confirmation"]
    reports = (
        _collect_guardrails(runs)
        if guardrails is None
        else tuple(guardrails)
    )
    inputs = analysis["verdict_inputs"]
    if not isinstance(inputs, Mapping):
        raise ValueError("analysis verdict inputs are missing")
    paired_names = (
        "split_dense_360",
        "split_dense_160_high",
        "split_random_160_high",
    )
    paired_rows = []
    for name in paired_names:
        estimate = inputs[name]
        for seed, delta in zip(
            SEEDS, estimate["seed_deltas"], strict=True
        ):
            paired_rows.append(
                {
                    "contrast": name,
                    "seed": seed,
                    "delta": delta,
                    "mean": estimate["mean"],
                    "ci_lo": estimate["ci_lo"],
                    "ci_hi": estimate["ci_hi"],
                    "cohen_dz": estimate["cohen_dz"],
                }
            )
    for scale, model, load in (
        ("d160m_high", "d160m", high_load),
        ("d360m_confirmation", "d360m", confirmation_load),
    ):
        for seed in SEEDS:
            split_summary = runs[(model, "split", load, seed)]["on"]
            dense_summary = runs[(model, "dense", load, seed)]["on"]
            if not isinstance(
                split_summary, CheckpointSummary
            ) or not isinstance(dense_summary, CheckpointSummary):
                raise ValueError("task report requires strict summaries")
            for task in EXPECTED_TASKS:
                paired_rows.append(
                    {
                        "contrast": "split_dense_task",
                        "scale": scale,
                        "task": task,
                        "seed": seed,
                        "delta": (
                            _task_pair_accuracy(split_summary, task)
                            - _task_pair_accuracy(dense_summary, task)
                        ),
                    }
                )
    high = inputs["split_dense_160_high"]["seed_deltas"]
    interaction = inputs["dose_interaction_160"]["seed_deltas"]
    dose_rows = tuple(
        {
            "seed": seed,
            "low_delta": high_delta - interaction_delta,
            "high_delta": high_delta,
            "interaction": interaction_delta,
        }
        for seed, high_delta, interaction_delta in zip(
            SEEDS, high, interaction, strict=True
        )
    )

    memory_rows = []
    control_rows = []
    for (model, arm, load, seed), run in sorted(runs.items()):
        for label in ("on", "off"):
            summary = run.get(label)
            if isinstance(summary, CheckpointSummary):
                memory_rows.append(
                    {
                        "model": model,
                        "arm": arm,
                        "load": load,
                        "seed": seed,
                        "memory_mode": summary.memory_mode,
                        "accuracy": _pair_composite(summary),
                    }
                )
        matrix = run.get("matrix")
        summaries = (
            matrix.items()
            if isinstance(matrix, _ValidatedCheckpointMatrix)
            else (
                (
                    (EvalMode.MEMORY_ON, ControlID.CORRECT),
                    run["on"],
                ),
            )
        )
        for (mode, control), summary in summaries:
            if not isinstance(summary, CheckpointSummary):
                raise ValueError("control report requires strict summaries")
            for family, slices in (
                ("hop", summary.metrics["by_hop"]),
                ("composition", summary.metrics["by_composition"]),
            ):
                for slice_name, metrics in sorted(slices.items()):
                    control_rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "load": load,
                            "seed": seed,
                            "memory_mode": mode.value,
                            "control": control.value,
                            "slice_kind": family,
                            "slice": slice_name,
                            "pair_accuracy": metrics[
                                "pair_accuracy"
                            ]["value"],
                            "numerator": metrics[
                                "pair_accuracy"
                            ]["numerator"],
                            "denominator": metrics[
                                "pair_accuracy"
                            ]["denominator"],
                        }
                    )

    guardrail_rows = []
    for report in reports:
        for guard_name, guard in sorted(report.guards.items()):
            for check in guard["checks"]:
                guardrail_rows.append(
                    {
                        "model": report.model_id,
                        "seed": report.seed,
                        "guard": guard_name,
                        "check": check["check_id"],
                        "value": check["value"],
                        "reference_value": check["reference_value"],
                        "threshold": check["threshold"],
                        "passed": check["passed"],
                    }
                )

    secondary = (
        analysis.get("secondary_analyses", {})
        if secondary_analyses is None
        else secondary_analyses
    )
    wikidata_rows = []
    if isinstance(secondary, Mapping):
        for analysis_name, value in sorted(secondary.items()):
            leaves = tuple(_numeric_leaves(value))
            if not leaves:
                wikidata_rows.append(
                    {
                        "dataset": analysis_name,
                        "metric": "status",
                        "value": None,
                    }
                )
            for metric, measured in leaves:
                wikidata_rows.append(
                    {
                        "dataset": analysis_name,
                        "metric": metric,
                        "value": measured,
                    }
                )
    sections = {
        "paired_deltas": AnalysisSection(
            analysis_role="confirmatory",
            rows=tuple(paired_rows),
        ),
        "dose_interaction": AnalysisSection(
            analysis_role="confirmatory",
            rows=dose_rows,
        ),
        "memory_factorial": AnalysisSection(
            analysis_role="supporting_only",
            rows=tuple(memory_rows),
        ),
        "controls_by_hop_composition": AnalysisSection(
            analysis_role="supporting_only",
            rows=tuple(control_rows),
        ),
        "guardrails": AnalysisSection(
            analysis_role="instrument_only",
            rows=tuple(guardrail_rows),
        ),
        "wikidata_robustness": AnalysisSection(
            analysis_role="robustness_only",
            rows=tuple(wikidata_rows),
        ),
    }
    if set(sections) != set(REPORT_SECTIONS):
        raise AssertionError("report section set drifted")
    return sections


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"analysis input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    files = (
        root / "evals" / "figures.py",
        root / "evals" / "relational_reporting.py",
        root / "evals" / "relational_stats.py",
        root / "scripts" / "analyze_relational.py",
    )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                path.relative_to(root).as_posix(): _sha256_file(path)
                for path in files
            }
        )
    ).hexdigest()


def _runs_root_sha256(
    runs: dict,
    reports: Sequence[GuardrailReport],
    run_manifest: RunManifest,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "run_manifest_sha256": run_manifest.manifest_sha256,
                "freeze_sha256": run_manifest.freeze_sha256,
                "run_matrix": list(_run_matrix_records(runs)),
                "guardrail_receipts": sorted(
                    report.pairing_receipt_sha256
                    for report in reports
                ),
            }
        )
    ).hexdigest()


def _load_secondary_analyses(
    paths: Sequence[str],
) -> dict[str, object]:
    analyses = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"secondary analysis must be a regular file: {path}"
            )
        content = path.read_bytes()
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"secondary analysis JSON is invalid: {path}"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("analysis_role") != "robustness_only"
            or value.get("confirmatory_verdict_eligible") is not False
        ):
            raise ValueError(
                "Wikidata inputs must be verdict-ineligible robustness results"
            )
        identity = ":".join(
            str(value.get(field, path.stem))
            for field in ("condition", "model", "seed")
        )
        if identity in analyses:
            raise ValueError("duplicate Wikidata robustness identity")
        analyses[identity] = {
            "analysis_role": "robustness_only",
            "confirmatory_verdict_eligible": False,
            "artifact_sha256": hashlib.sha256(content).hexdigest(),
            "result": value,
        }
    return analyses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict hierarchical confirmatory relational analysis."
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--rng-seed", required=True, type=int)
    parser.add_argument(
        "--wikidata-results",
        action="append",
        default=[],
        help=(
            "verdict-ineligible Wikidata result JSON; repeat for each result"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    preregistration = Path(args.preregistration)
    run_manifest = require_launchable(load_run_manifest(args.run_manifest))
    verify_source_provenance(
        Path(__file__).resolve().parents[1],
        run_manifest.freeze.source_provenance,
    )
    preregistration_sha256 = _sha256_file(preregistration)
    if (
        preregistration_sha256
        != run_manifest.freeze.artifact_sha256["preregistration"]
    ):
        raise ValueError(
            "preregistration hash does not match the frozen run manifest"
        )
    runs = _load_complete_runs(runs_root, run_manifest)
    reports = _collect_guardrails(runs)
    input_bindings = {
        "runs_root_sha256": _runs_root_sha256(
            runs,
            reports,
            run_manifest,
        ),
        "preregistration_sha256": preregistration_sha256,
        "analysis_code_sha256": _analysis_code_sha256(),
        "guardrail_receipt_sha256": [
            report.pairing_receipt_sha256 for report in reports
        ],
    }
    result = analyze_runs(
        runs,
        run_manifest=run_manifest,
        rng_seed=args.rng_seed,
        input_bindings=input_bindings,
        secondary_analyses={},
    )
    # Robustness artifacts are deliberately read only after the confirmatory
    # decision and its hash have been fixed.
    secondary = _load_secondary_analyses(args.wikidata_results)
    result["secondary_analyses"] = secondary
    sections = build_report_sections(
        runs,
        result,
        run_manifest=run_manifest,
        guardrails=reports,
        secondary_analyses=secondary,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_analysis_bundle(
        output,
        analysis=result,
        sections=sections,
    )
    published = validate_analysis_bundle(output)
    print(json.dumps(published, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
