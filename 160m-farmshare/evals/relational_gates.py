"""Fail-closed development-gate receipts for the relational study."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.relational_design import (
    DesignPowerReceipt,
    validate_design_receipt,
)
from evals.relational_contracts import EvalRow, rows_sha256, validate_eval_rows
from experiment.artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
    validate_sha256,
)


CORE_TASKS = (
    "path_composition",
    "date_ordering",
    "balanced_equality",
)
SMOKE_REPORT_FIELDS = frozenset(
    {
        "bundle_byte_deterministic",
        "bundle_verified",
        "controls",
        "corpus_builds",
        "corpus_byte_deterministic",
        "corpus_sha256",
        "dense_steps",
        "eval_cells",
        "extracted_bundle_verified",
        "matrix_runs",
        "memory_modes",
        "pairs_complete",
        "resume_compared_next_update",
        "resume_exact",
        "schemas_validated",
        "shared_stream",
        "sidecar_sha256",
        "sidecars",
        "split_steps",
        "synthetic_run_count",
        "verdict_branches",
    }
)
ORDERED_MIXTURES = (
    (0.70, 0.15, 0.15),
    (0.65, 0.15, 0.20),
    (0.65, 0.20, 0.15),
)
LOAD_CANDIDATES = (50_000, 200_000, 800_000)
CONFIRMATION_CANDIDATES = (800_000, 1_800_000, 3_200_000)
D160_PARAMETERS = 162_220_800
D360_PARAMETERS = 356_033_536
GATE_1_PARAMETERS = 28_969_216
_PROTECTED_SEEDS = {1001, 1002, 1003, 1004, 1005}
COMMON_INPUT_HASH_FIELDS = (
    "source_lock_sha256",
    "relation_schema_sha256",
    "preregistration_sha256",
    "evaluator_sha256",
    "analysis_sha256",
    "source_tree_sha256",
)
_RULE_VERSIONS = {
    0: "relational-gate0-smoke-v1",
    1: "relational-gate1-learnability-v1",
    2: "relational-gate2-ordered-mixture-v1",
    3: "relational-gate3-dense-load-v1",
    4: "relational-gate4-token-budget-v1",
    5: "relational-gate5-prospective-power-v1",
}
_INPUT_FIELDS = {
    "record_type",
    "schema_version",
    "scope",
    "input_hashes",
    "mixtures",
    "loads",
    "budget",
}
_COMMON_RECEIPT_FIELDS = {
    "record_type",
    "schema_version",
    "gate",
    "rule_version",
    "input_hashes",
    "measurements",
    "passed",
    "decision",
    "decision_sha256",
    "receipt_sha256",
}
_DEVELOPMENT_RECEIPT_FIELDS = {"development_input_sha256"}
_GATE_FIELDS = {
    0: _COMMON_RECEIPT_FIELDS
    | {"smoke_report_sha256"},
    1: _COMMON_RECEIPT_FIELDS
    | _DEVELOPMENT_RECEIPT_FIELDS
    | {"selected_mixture", "selected_mixture_index"},
    2: _COMMON_RECEIPT_FIELDS
    | _DEVELOPMENT_RECEIPT_FIELDS
    | {
        "selected_mixture",
        "selected_mixture_index",
        "gate_1_receipt_sha256",
    },
    3: _COMMON_RECEIPT_FIELDS
    | _DEVELOPMENT_RECEIPT_FIELDS
    | {
        "low_entities",
        "high_entities",
        "confirmation_entities",
        "gate_2_receipt_sha256",
    },
    4: _COMMON_RECEIPT_FIELDS
    | _DEVELOPMENT_RECEIPT_FIELDS
    | {"tokens_per_parameter", "gate_3_receipt_sha256"},
    5: _COMMON_RECEIPT_FIELDS
    | {
        "design_receipt",
        "design_receipt_sha256",
        "development_binding",
        "development_binding_sha256",
        "high_entities",
        "tokens_per_parameter",
        "gate_3_receipt_sha256",
        "gate_4_receipt_sha256",
    },
}
_ROW_EVIDENCE_FIELDS = {
    "rows_sha256",
    "model_id",
    "development_seed",
    "checkpoint_sha256",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "provenance_sha256",
    "pair_counts",
}


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean")
    return value


def _input_hashes(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != set(
        COMMON_INPUT_HASH_FIELDS
    ):
        raise ValueError("common input hash fields are not exact")
    return {
        name: validate_sha256(raw[name], name)
        for name in COMMON_INPUT_HASH_FIELDS
    }


def _task_accuracies(raw: object, name: str) -> dict[str, float]:
    if not isinstance(raw, Mapping) or set(raw) != set(CORE_TASKS):
        raise ValueError(f"{name} task strata are not exact")
    return {
        task: _number(
            raw[task],
            f"{name}.{task}",
            minimum=0.0,
            maximum=1.0,
        )
        for task in CORE_TASKS
    }


def _recognition(raw: object, name: str) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != {"successes", "total"}:
        raise ValueError(f"{name} recognition counts are not exact")
    successes = _integer(raw["successes"], f"{name}.successes")
    total = _integer(raw["total"], f"{name}.total", minimum=1)
    if successes > total:
        raise ValueError(f"{name} successes exceed total")
    return {"successes": successes, "total": total}


def _normalized_row_evidence(
    raw: object,
    name: str,
    *,
    arm: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _ROW_EVIDENCE_FIELDS:
        raise ValueError(f"{name} row evidence fields are not exact")
    if raw["model_id"] != "d29m":
        raise ValueError("Gate 1-2 rows must use the 29M development model")
    seed = _integer(raw["development_seed"], f"{name}.development_seed")
    if seed in _PROTECTED_SEEDS:
        raise ValueError("development rows cannot use a protected seed")
    pair_counts = raw["pair_counts"]
    if not isinstance(pair_counts, Mapping) or set(pair_counts) != set(
        CORE_TASKS
    ):
        raise ValueError(f"{name} pair counts are not exact")
    return {
        "rows_sha256": validate_sha256(
            raw["rows_sha256"],
            f"{name} rows SHA-256",
        ),
        "model_id": "d29m",
        "development_seed": seed,
        "checkpoint_sha256": validate_sha256(
            raw["checkpoint_sha256"],
            f"{name} checkpoint SHA-256",
        ),
        "raw_token_count": _integer(
            raw["raw_token_count"],
            f"{name}.raw_token_count",
            minimum=1,
        ),
        "evaluator_sha256": validate_sha256(
            raw["evaluator_sha256"],
            f"{name} evaluator SHA-256",
        ),
        "data_sha256": validate_sha256(
            raw["data_sha256"],
            f"{name} data SHA-256",
        ),
        "relation_schema_sha256": validate_sha256(
            raw["relation_schema_sha256"],
            f"{name} relation schema SHA-256",
        ),
        "provenance_sha256": validate_sha256(
            raw["provenance_sha256"],
            f"{name} provenance SHA-256",
        ),
        "pair_counts": {
            task: _integer(
                pair_counts[task],
                f"{name}.{task}.pair_count",
                minimum=1,
            )
            for task in CORE_TASKS
        },
    }


def _validated_development_rows(
    raw: object,
    name: str,
    *,
    arm: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{name} rows must be an array")
    rows = validate_eval_rows(
        EvalRow.from_dict(item) if isinstance(item, Mapping) else item
        for item in raw
    )
    if (
        {row.task for row in rows} != set(CORE_TASKS)
        or {row.arm for row in rows} != {arm}
        or {row.model_id for row in rows} != {"d29m"}
        or {row.memory_mode for row in rows} != {"memory_on"}
        or {row.control_id for row in rows} != {"correct"}
    ):
        raise ValueError(
            f"{name} rows do not match the Gate 1-2 development protocol"
        )
    seeds = {row.seed for row in rows}
    if len(seeds) != 1 or seeds & _PROTECTED_SEEDS:
        raise ValueError(
            f"{name} rows require one disjoint development seed"
        )
    identity_fields = (
        "checkpoint_sha256",
        "raw_token_count",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "provenance_sha256",
    )
    identities = {
        field: {getattr(row, field) for row in rows}
        for field in identity_fields
    }
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError(f"{name} rows contain inconsistent provenance")
    grouped: dict[str, dict[str, list[EvalRow]]] = {
        task: defaultdict(list) for task in CORE_TASKS
    }
    for row in rows:
        grouped[row.task][row.pair_id].append(row)
    pair_counts = {
        task: len(grouped[task]) for task in CORE_TASKS
    }
    accuracies = {
        task: sum(
            all(row.correct for row in pair)
            for pair in grouped[task].values()
        )
        / pair_counts[task]
        for task in CORE_TASKS
    }
    evidence = {
        "rows_sha256": rows_sha256(rows),
        "model_id": "d29m",
        "development_seed": next(iter(seeds)),
        **{
            field: next(iter(values))
            for field, values in identities.items()
        },
        "pair_counts": pair_counts,
    }
    return accuracies, _normalized_row_evidence(
        evidence,
        name,
        arm=arm,
    )


def _mixture_arm(
    raw: object,
    name: str,
    *,
    arm: str,
    require_recognition: bool,
    require_rows: bool,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be an object")
    allowed = {
        "task_pair_accuracy",
        "natural_text_bpb",
        "fact_recognition",
        "rows" if require_rows else "row_evidence",
    }
    required = {
        "task_pair_accuracy",
        "natural_text_bpb",
        "rows" if require_rows else "row_evidence",
    }
    if set(raw) - allowed or not required <= set(raw):
        raise ValueError(f"{name} fields are not exact")
    if require_recognition and "fact_recognition" not in raw:
        raise ValueError("Dense mixture requires fact recognition counts")
    reported = _task_accuracies(
            raw["task_pair_accuracy"],
            f"{name}.task_pair_accuracy",
        )
    if require_rows:
        computed, row_evidence = _validated_development_rows(
            raw["rows"],
            name,
            arm=arm,
        )
        if reported != computed:
            raise ValueError(
                f"{name} task accuracy does not match validated rows"
            )
    else:
        computed = reported
        row_evidence = _normalized_row_evidence(
            raw["row_evidence"],
            name,
            arm=arm,
        )
    result: dict[str, Any] = {
        "task_pair_accuracy": computed,
        "natural_text_bpb": _number(
            raw["natural_text_bpb"],
            f"{name}.natural_text_bpb",
            minimum=0.0,
        ),
        "row_evidence": row_evidence,
    }
    if "fact_recognition" in raw:
        result["fact_recognition"] = _recognition(
            raw["fact_recognition"],
            f"{name}.fact_recognition",
        )
    return result


def _normalize_mixtures(
    raw: object,
    *,
    require_rows: bool = False,
) -> list[dict[str, Any]]:
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or len(raw) != len(ORDERED_MIXTURES)
    ):
        raise ValueError("development mixtures must contain the ordered grid")
    result = []
    for index, (item, expected) in enumerate(
        zip(raw, ORDERED_MIXTURES, strict=True)
    ):
        if not isinstance(item, Mapping) or set(item) != {
            "mixture",
            "parameter_count",
            "manifest_sha256",
            "arms",
        }:
            raise ValueError("mixture measurement fields are not exact")
        mixture = item["mixture"]
        if (
            isinstance(mixture, (str, bytes))
            or not isinstance(mixture, Sequence)
            or len(mixture) != 3
        ):
            raise ValueError("mixture must contain bed, graph, and reasoning")
        values = tuple(
            _number(value, f"mixtures[{index}]", minimum=0.0, maximum=1.0)
            for value in mixture
        )
        if values != expected:
            raise ValueError("development mixture order or values drifted")
        if item["parameter_count"] != GATE_1_PARAMETERS:
            raise ValueError("Gate 1-2 require exactly 28,969,216 parameters")
        arms = item["arms"]
        if not isinstance(arms, Mapping) or set(arms) != {"dense", "split"}:
            raise ValueError("mixture arms must be exactly Dense and Split")
        normalized_arms = {
            "dense": _mixture_arm(
                arms["dense"],
                f"mixtures[{index}].dense",
                arm="dense",
                require_recognition=True,
                require_rows=require_rows,
            ),
            "split": _mixture_arm(
                arms["split"],
                f"mixtures[{index}].split",
                arm="split",
                require_recognition=False,
                require_rows=require_rows,
            ),
        }
        dense_evidence = normalized_arms["dense"]["row_evidence"]
        split_evidence = normalized_arms["split"]["row_evidence"]
        for field in (
            "model_id",
            "development_seed",
            "raw_token_count",
            "evaluator_sha256",
            "data_sha256",
            "relation_schema_sha256",
        ):
            if dense_evidence[field] != split_evidence[field]:
                raise ValueError(
                    f"mixture {index} paired row evidence disagrees on {field}"
                )
        result.append(
            {
                "mixture": list(values),
                "parameter_count": GATE_1_PARAMETERS,
                "manifest_sha256": validate_sha256(
                    item["manifest_sha256"],
                    f"mixture {index} manifest SHA-256",
                ),
                "arms": normalized_arms,
            }
        )
    return result


def _normalize_loads(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("load calibration must be an array")
    loads = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {
            "entities",
            "manifest_sha256",
            "dense_fact_recall",
            "dense_reasoning_composite",
        }:
            raise ValueError("load measurement fields are not exact")
        loads.append(
            {
                "entities": _integer(
                    item["entities"],
                    f"loads[{index}].entities",
                    minimum=1,
                ),
                "manifest_sha256": validate_sha256(
                    item["manifest_sha256"],
                    f"load {index} manifest SHA-256",
                ),
                "dense_fact_recall": _number(
                    item["dense_fact_recall"],
                    f"loads[{index}].dense_fact_recall",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "dense_reasoning_composite": _number(
                    item["dense_reasoning_composite"],
                    f"loads[{index}].dense_reasoning_composite",
                    minimum=0.0,
                    maximum=1.0,
                ),
            }
        )
    if (
        len(loads) != len(LOAD_CANDIDATES)
        or {item["entities"] for item in loads} != set(LOAD_CANDIDATES)
    ):
        raise ValueError("load calibration candidates are not exact")
    return sorted(loads, key=lambda item: item["entities"])


def _normalize_budget_arm(raw: object, name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "task_pair_accuracy",
        "natural_text_bpb",
    }:
        raise ValueError(f"{name} fields are not exact")
    return {
        "task_pair_accuracy": _task_accuracies(
            raw["task_pair_accuracy"],
            f"{name}.task_pair_accuracy",
        ),
        "natural_text_bpb": _number(
            raw["natural_text_bpb"],
            f"{name}.natural_text_bpb",
            minimum=0.0,
        ),
    }


def _normalize_budget(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {"at_10x", "at_20x"}:
        raise ValueError("budget checkpoints must be exactly 10x and 20x")
    at_10 = raw["at_10x"]
    if not isinstance(at_10, Mapping) or set(at_10) != {
        "manifest_sha256",
        "arms",
        "fact_recognition_passed",
        "memory_guardrails_passed",
        "natural_text_noninferiority_passed",
        "paired_change_ci_to_20x",
    }:
        raise ValueError("10x budget fields are not exact")
    arms = at_10["arms"]
    if not isinstance(arms, Mapping) or set(arms) != {"dense", "split"}:
        raise ValueError("10x budget arms must be Dense and Split")
    expected_intervals = {
        f"{arm}.{metric}"
        for arm in ("dense", "split")
        for metric in (*CORE_TASKS, "fact_recall")
    }
    intervals = at_10["paired_change_ci_to_20x"]
    if not isinstance(intervals, Mapping) or set(intervals) != expected_intervals:
        raise ValueError("10x-to-20x paired interval set is not exact")
    normalized_intervals = {}
    for name in sorted(intervals):
        interval = intervals[name]
        if not isinstance(interval, Mapping) or set(interval) != {
            "ci_lo",
            "ci_hi",
        }:
            raise ValueError(f"{name} interval fields are not exact")
        low = _number(interval["ci_lo"], f"{name}.ci_lo")
        high = _number(interval["ci_hi"], f"{name}.ci_hi")
        if low > high:
            raise ValueError(f"{name} confidence interval is reversed")
        normalized_intervals[name] = {"ci_lo": low, "ci_hi": high}
    at_20 = raw["at_20x"]
    if not isinstance(at_20, Mapping) or set(at_20) != {
        "manifest_sha256",
        "available",
    }:
        raise ValueError("20x budget fields are not exact")
    return {
        "at_10x": {
            "manifest_sha256": validate_sha256(
                at_10["manifest_sha256"],
                "10x manifest SHA-256",
            ),
            "arms": {
                arm: _normalize_budget_arm(
                    arms[arm],
                    f"budget.at_10x.{arm}",
                )
                for arm in ("dense", "split")
            },
            "fact_recognition_passed": _boolean(
                at_10["fact_recognition_passed"],
                "fact_recognition_passed",
            ),
            "memory_guardrails_passed": _boolean(
                at_10["memory_guardrails_passed"],
                "memory_guardrails_passed",
            ),
            "natural_text_noninferiority_passed": _boolean(
                at_10["natural_text_noninferiority_passed"],
                "natural_text_noninferiority_passed",
            ),
            "paired_change_ci_to_20x": normalized_intervals,
        },
        "at_20x": {
            "manifest_sha256": validate_sha256(
                at_20["manifest_sha256"],
                "20x manifest SHA-256",
            ),
            "available": _boolean(
                at_20["available"],
                "20x availability",
            ),
        },
    }


def _normalize_inputs(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _INPUT_FIELDS:
        raise ValueError("development gate input fields are not exact")
    if (
        raw["record_type"] != "relational_development_gate_inputs"
        or raw["schema_version"] != 1
        or raw["scope"] != "development"
    ):
        raise ValueError("gate inputs must use the development-only protocol")
    return {
        "record_type": raw["record_type"],
        "schema_version": 1,
        "scope": "development",
        "input_hashes": _input_hashes(raw["input_hashes"]),
        "mixtures": _normalize_mixtures(
            raw["mixtures"],
            require_rows=True,
        ),
        "loads": _normalize_loads(raw["loads"]),
        "budget": _normalize_budget(raw["budget"]),
    }


def _learnable(mixture: Mapping[str, Any]) -> bool:
    return all(
        value > 0.75
        for arm in ("dense", "split")
        for value in mixture["arms"][arm]["task_pair_accuracy"].values()
    )


def _wilson_lower(successes: int, total: int) -> float:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - radius


def _mixture_passes(mixture: Mapping[str, Any]) -> bool:
    recognition = mixture["arms"]["dense"]["fact_recognition"]
    return (
        _learnable(mixture)
        and _wilson_lower(
            recognition["successes"],
            recognition["total"],
        )
        > 0.30
        and all(
            math.isfinite(mixture["arms"][arm]["natural_text_bpb"])
            for arm in ("dense", "split")
        )
    )


def _select_loads(
    loads: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int | None, int | None]:
    low_candidates = [
        item
        for item in loads
        if item["dense_fact_recall"] >= 0.80
        and item["dense_reasoning_composite"] >= 0.75
    ]
    if not low_candidates:
        return None, None, None
    low = min(low_candidates, key=lambda item: item["entities"])
    high_candidates = [
        item
        for item in loads
        if item["entities"] != low["entities"]
        and item["dense_fact_recall"] >= 0.40
        and item["dense_reasoning_composite"] >= 0.40
        and (
            item["dense_fact_recall"]
            <= low["dense_fact_recall"] - 0.10
            or item["dense_reasoning_composite"]
            <= low["dense_reasoning_composite"] - 0.05
        )
    ]
    if not high_candidates:
        return None, None, None
    high = max(high_candidates, key=lambda item: item["entities"])
    target = high["entities"] * D360_PARAMETERS / D160_PARAMETERS
    confirmation = min(
        CONFIRMATION_CANDIDATES,
        key=lambda entities: (abs(entities - target), entities),
    )
    return low["entities"], high["entities"], confirmation


def _select_budget(budget: Mapping[str, Any]) -> int | None:
    if not budget["at_20x"]["available"]:
        return None
    at_10 = budget["at_10x"]
    ten_passes = (
        all(
            value > 0.75
            for arm in ("dense", "split")
            for value in at_10["arms"][arm][
                "task_pair_accuracy"
            ].values()
        )
        and at_10["fact_recognition_passed"]
        and at_10["memory_guardrails_passed"]
        and at_10["natural_text_noninferiority_passed"]
        and all(
            interval["ci_lo"] >= -0.01
            and interval["ci_hi"] <= 0.01
            for interval in at_10["paired_change_ci_to_20x"].values()
        )
    )
    return 10 if ten_passes else 20


def _decision_for(receipt: Mapping[str, Any]) -> dict[str, Any]:
    gate = receipt["gate"]
    decision: dict[str, Any] = {
        "record_type": "relational_gate_decision",
        "schema_version": 1,
        "gate": gate,
        "rule_version": receipt["rule_version"],
        "input_hashes": copy.deepcopy(dict(receipt["input_hashes"])),
        "passed": receipt["passed"],
    }
    if gate in {1, 2, 3, 4}:
        decision["development_input_sha256"] = receipt[
            "development_input_sha256"
        ]
    if gate in {1, 2}:
        decision.update(
            selected_mixture=copy.deepcopy(receipt["selected_mixture"]),
            selected_mixture_index=receipt["selected_mixture_index"],
        )
        if gate == 2:
            decision["gate_1_receipt_sha256"] = receipt[
                "gate_1_receipt_sha256"
            ]
    elif gate == 3:
        decision.update(
            low_entities=receipt["low_entities"],
            high_entities=receipt["high_entities"],
            confirmation_entities=receipt["confirmation_entities"],
            gate_2_receipt_sha256=receipt["gate_2_receipt_sha256"],
        )
    elif gate == 4:
        decision["tokens_per_parameter"] = receipt["tokens_per_parameter"]
        decision["gate_3_receipt_sha256"] = receipt[
            "gate_3_receipt_sha256"
        ]
    elif gate == 0:
        decision["smoke_report_sha256"] = receipt["smoke_report_sha256"]
    elif gate == 5:
        decision["design_receipt_sha256"] = receipt[
            "design_receipt_sha256"
        ]
        decision["development_binding_sha256"] = receipt[
            "development_binding_sha256"
        ]
        decision["high_entities"] = receipt["high_entities"]
        decision["tokens_per_parameter"] = receipt[
            "tokens_per_parameter"
        ]
        decision["gate_3_receipt_sha256"] = receipt[
            "gate_3_receipt_sha256"
        ]
        decision["gate_4_receipt_sha256"] = receipt[
            "gate_4_receipt_sha256"
        ]
    return decision


def rehash_gate_receipt(raw: Mapping[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(dict(raw))
    receipt["decision"] = _decision_for(receipt)
    receipt["decision_sha256"] = canonical_sha256(receipt["decision"])
    without_hash = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = canonical_sha256(without_hash)
    return receipt


def _receipt(
    gate: int,
    input_hashes: Mapping[str, str],
    measurements: Any,
    passed: bool,
    **selection: Any,
) -> dict[str, Any]:
    base = {
        "record_type": "relational_gate_receipt",
        "schema_version": 1,
        "gate": gate,
        "rule_version": _RULE_VERSIONS[gate],
        "input_hashes": _input_hashes(input_hashes),
        "measurements": copy.deepcopy(measurements),
        "passed": bool(passed),
        **selection,
        "decision": {},
        "decision_sha256": "0" * 64,
        "receipt_sha256": "0" * 64,
    }
    return rehash_gate_receipt(base)


def evaluate_development_gates(
    development: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Apply Gates 1--4 without opening any protected artifact."""

    normalized = _normalize_inputs(development)
    input_hashes = normalized["input_hashes"]
    development_input_sha256 = canonical_sha256(normalized)
    mixtures = normalized["mixtures"]
    learnable_indices = [
        index for index, mixture in enumerate(mixtures) if _learnable(mixture)
    ]
    gate_1_index = learnable_indices[0] if learnable_indices else None
    gate_1 = _receipt(
        1,
        input_hashes,
        mixtures,
        gate_1_index is not None,
        selected_mixture=(
            mixtures[gate_1_index]["mixture"]
            if gate_1_index is not None
            else None
        ),
        selected_mixture_index=gate_1_index,
        development_input_sha256=development_input_sha256,
    )
    passing_indices = [
        index
        for index, mixture in enumerate(mixtures)
        if _mixture_passes(mixture)
    ]
    gate_2_index = passing_indices[0] if passing_indices else None
    gate_2 = _receipt(
        2,
        input_hashes,
        mixtures,
        gate_2_index is not None,
        selected_mixture=(
            mixtures[gate_2_index]["mixture"]
            if gate_2_index is not None
            else None
        ),
        selected_mixture_index=gate_2_index,
        gate_1_receipt_sha256=gate_1["receipt_sha256"],
        development_input_sha256=development_input_sha256,
    )
    low, high, confirmation = _select_loads(normalized["loads"])
    gate_3 = _receipt(
        3,
        input_hashes,
        normalized["loads"],
        low is not None,
        low_entities=low,
        high_entities=high,
        confirmation_entities=confirmation,
        gate_2_receipt_sha256=gate_2["receipt_sha256"],
        development_input_sha256=development_input_sha256,
    )
    token_budget = _select_budget(normalized["budget"])
    gate_4 = _receipt(
        4,
        input_hashes,
        normalized["budget"],
        token_budget is not None,
        tokens_per_parameter=token_budget,
        gate_3_receipt_sha256=gate_3["receipt_sha256"],
        development_input_sha256=development_input_sha256,
    )
    return {
        "gate_1": gate_1,
        "gate_2": gate_2,
        "gate_3": gate_3,
        "gate_4": gate_4,
    }


def smoke_report_passes(report: object) -> bool:
    if (
        not isinstance(report, Mapping)
        or set(report) != SMOKE_REPORT_FIELDS
    ):
        raise ValueError("Gate 0 smoke report fields are not exact")
    required_true = (
        "bundle_byte_deterministic",
        "bundle_verified",
        "corpus_byte_deterministic",
        "extracted_bundle_verified",
        "pairs_complete",
        "resume_compared_next_update",
        "resume_exact",
        "shared_stream",
    )
    if any(not isinstance(report[field], bool) for field in required_true):
        raise ValueError("Gate 0 deterministic fields must be booleans")
    counts = (
        "corpus_builds",
        "dense_steps",
        "eval_cells",
        "matrix_runs",
        "split_steps",
        "synthetic_run_count",
    )
    if any(
        isinstance(report[field], bool) or not isinstance(report[field], int)
        for field in counts
    ):
        raise ValueError("Gate 0 count fields must be integers")
    validate_sha256(report["corpus_sha256"], "Gate 0 corpus SHA-256")
    sidecars = ("dense", "random", "selective", "split")
    sidecar_hashes = report["sidecar_sha256"]
    if not isinstance(sidecar_hashes, Mapping) or set(sidecar_hashes) != set(
        sidecars
    ):
        raise ValueError("Gate 0 sidecar hashes are not exact")
    for label in sidecars:
        validate_sha256(
            sidecar_hashes[label],
            f"Gate 0 {label} sidecar SHA-256",
        )
    verdicts = report["verdict_branches"]
    if (
        not isinstance(verdicts, list)
        or any(not isinstance(verdict, str) for verdict in verdicts)
    ):
        raise ValueError("Gate 0 verdict branches are invalid")
    from evals.relational_controls import ControlID
    from evals.relational_stats import ALLOWED_VERDICTS

    return (
        all(report[field] is True for field in required_true)
        and report["corpus_builds"] == 2
        and report["dense_steps"] == 2
        and report["eval_cells"] == 22
        and report["matrix_runs"] == 35
        and report["split_steps"] == 2
        and report["synthetic_run_count"] == 35
        and report["memory_modes"] == ["off", "on"]
        and report["sidecars"] == list(sidecars)
        and report["controls"] == [control.value for control in ControlID]
        and len(verdicts) == len(ALLOWED_VERDICTS)
        and set(verdicts) == set(ALLOWED_VERDICTS)
        and report["schemas_validated"]
        == [
            "freeze-v1.schema.json",
            "relational-asset-receipt-v1.schema.json",
            "relational-result-v1.schema.json",
            "run-config-v1.schema.json",
            "run-manifest-v1.schema.json",
        ]
    )


def _smoke_passes(report: object) -> bool:
    return smoke_report_passes(report)


def build_gate_0_receipt(
    smoke_report: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    passed = _smoke_passes(smoke_report)
    return _receipt(
        0,
        input_hashes,
        copy.deepcopy(dict(smoke_report)),
        passed,
        smoke_report_sha256=canonical_sha256(smoke_report),
    )


def _normalize_gate_5_development_binding(
    raw: object,
    design_receipt: DesignPowerReceipt,
    *,
    expected_high_entities: int | None = None,
    expected_tokens_per_parameter: int | None = None,
    expected_data_sha256: str | None = None,
) -> dict[str, Any]:
    fields = {
        "record_type",
        "schema_version",
        "model_id",
        "parameter_count",
        "high_entities",
        "tokens_per_parameter",
        "raw_token_count",
        "inputs",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("Gate 5 development binding fields are not exact")
    if (
        raw["record_type"] != "gate5_development_binding"
        or raw["schema_version"] != 1
        or raw["model_id"] != "d160m"
        or raw["parameter_count"] != D160_PARAMETERS
    ):
        raise ValueError("Gate 5 development binding protocol drifted")
    high_entities = _integer(
        raw["high_entities"],
        "Gate 5 high entity load",
        minimum=1,
    )
    tokens_per_parameter = _integer(
        raw["tokens_per_parameter"],
        "Gate 5 tokens per parameter",
        minimum=1,
    )
    if tokens_per_parameter not in {10, 20}:
        raise ValueError("Gate 5 token budget must be 10x or 20x")
    if (
        expected_high_entities is not None
        and high_entities != expected_high_entities
    ):
        raise ValueError("Gate 5 rows do not use the Gate 3 high load")
    if (
        expected_tokens_per_parameter is not None
        and tokens_per_parameter != expected_tokens_per_parameter
    ):
        raise ValueError("Gate 5 rows do not use the Gate 4 token budget")
    raw_token_count = _integer(
        raw["raw_token_count"],
        "Gate 5 raw token count",
        minimum=1,
    )
    target = D160_PARAMETERS * tokens_per_parameter
    if abs(raw_token_count - target) / target >= 0.0002:
        raise ValueError(
            "Gate 5 raw token count violates the strict rounding tolerance"
        )
    inputs = raw["inputs"]
    labels = set(design_receipt.blinded_input_hashes)
    if not isinstance(inputs, Mapping) or set(inputs) != labels:
        raise ValueError("Gate 5 development input labels mismatch")
    normalized_inputs: dict[str, dict[str, str]] = {}
    for label in sorted(labels):
        item = inputs[label]
        if not isinstance(item, Mapping) or set(item) != {
            "rows_sha256",
            "checkpoint_sha256",
            "data_sha256",
        }:
            raise ValueError("Gate 5 development input fields are not exact")
        rows_hash = validate_sha256(
            item["rows_sha256"],
            f"Gate 5 {label} rows SHA-256",
        )
        if rows_hash != design_receipt.blinded_input_hashes[label]:
            raise ValueError("Gate 5 design row hash binding mismatch")
        data_hash = validate_sha256(
            item["data_sha256"],
            f"Gate 5 {label} data SHA-256",
        )
        if (
            expected_data_sha256 is not None
            and data_hash != expected_data_sha256
        ):
            raise ValueError("Gate 5 data does not match the Gate 3 high load")
        normalized_inputs[label] = {
            "rows_sha256": rows_hash,
            "checkpoint_sha256": validate_sha256(
                item["checkpoint_sha256"],
                f"Gate 5 {label} checkpoint SHA-256",
            ),
            "data_sha256": data_hash,
        }
    if len(
        {item["checkpoint_sha256"] for item in normalized_inputs.values()}
    ) != len(normalized_inputs):
        raise ValueError("Gate 5 arm checkpoints must be distinct")
    return {
        "record_type": "gate5_development_binding",
        "schema_version": 1,
        "model_id": "d160m",
        "parameter_count": D160_PARAMETERS,
        "high_entities": high_entities,
        "tokens_per_parameter": tokens_per_parameter,
        "raw_token_count": raw_token_count,
        "inputs": normalized_inputs,
    }


def _selected_high_manifest_sha256(
    gate_3_receipt: Mapping[str, Any],
) -> str:
    selected = gate_3_receipt["high_entities"]
    matches = [
        item["manifest_sha256"]
        for item in gate_3_receipt["measurements"]
        if item["entities"] == selected
    ]
    if len(matches) != 1:
        raise ValueError("Gate 3 high-load manifest binding is not unique")
    return validate_sha256(matches[0], "Gate 3 high-load manifest SHA-256")


def build_gate_5_receipt(
    design_receipt: DesignPowerReceipt | Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str],
    gate_3_receipt: Mapping[str, Any],
    gate_4_receipt: Mapping[str, Any],
    development_binding: Mapping[str, Any],
) -> dict[str, Any]:
    raw = (
        design_receipt.to_dict()
        if isinstance(design_receipt, DesignPowerReceipt)
        else design_receipt
    )
    validated = validate_design_receipt(raw)
    serialized = validated.to_dict()
    validated_gate_3 = validate_gate_receipt(
        gate_3_receipt,
        expected_gate=3,
    )
    validated_gate_4 = validate_gate_receipt(
        gate_4_receipt,
        expected_gate=4,
    )
    if not validated_gate_3["passed"] or not validated_gate_4["passed"]:
        raise ValueError("Gate 5 requires passing Gate 3 and Gate 4 receipts")
    binding = _normalize_gate_5_development_binding(
        development_binding,
        validated,
        expected_high_entities=validated_gate_3["high_entities"],
        expected_tokens_per_parameter=validated_gate_4[
            "tokens_per_parameter"
        ],
        expected_data_sha256=_selected_high_manifest_sha256(
            validated_gate_3
        ),
    )
    return _receipt(
        5,
        input_hashes,
        {
            "power": validated.power,
            "power_ci_lo": validated.power_ci_lo,
            "power_ci_hi": validated.power_ci_hi,
            "studies": validated.studies,
            "pairs": validated.pairs,
        },
        validated.passed,
        design_receipt=serialized,
        design_receipt_sha256=canonical_sha256(serialized),
        development_binding=binding,
        development_binding_sha256=canonical_sha256(binding),
        high_entities=binding["high_entities"],
        tokens_per_parameter=binding["tokens_per_parameter"],
        gate_3_receipt_sha256=validated_gate_3["receipt_sha256"],
        gate_4_receipt_sha256=validated_gate_4["receipt_sha256"],
    )


def _validate_semantics(receipt: Mapping[str, Any]) -> None:
    gate = receipt["gate"]
    passed = _boolean(receipt["passed"], "gate passed")
    measurements = receipt["measurements"]
    if gate in {1, 2, 3, 4}:
        validate_sha256(
            receipt["development_input_sha256"],
            "development input SHA-256",
        )
    if gate == 0:
        expected_passed = _smoke_passes(measurements)
        if receipt["smoke_report_sha256"] != canonical_sha256(measurements):
            raise ValueError("Gate 0 smoke report hash mismatch")
    elif gate in {1, 2}:
        if gate == 2:
            validate_sha256(
                receipt["gate_1_receipt_sha256"],
                "Gate 1 receipt SHA-256",
            )
        normalized = _normalize_mixtures(measurements)
        predicate = _learnable if gate == 1 else _mixture_passes
        indices = [
            index
            for index, mixture in enumerate(normalized)
            if predicate(mixture)
        ]
        selected = indices[0] if indices else None
        expected_passed = selected is not None
        expected_mixture = (
            normalized[selected]["mixture"] if selected is not None else None
        )
        if (
            receipt["selected_mixture_index"] != selected
            or receipt["selected_mixture"] != expected_mixture
        ):
            raise ValueError(f"Gate {gate} selected mixture violates its rule")
    elif gate == 3:
        validate_sha256(
            receipt["gate_2_receipt_sha256"],
            "Gate 2 receipt SHA-256",
        )
        normalized = _normalize_loads(measurements)
        low, high, confirmation = _select_loads(normalized)
        expected_passed = low is not None
        if (
            receipt["low_entities"],
            receipt["high_entities"],
            receipt["confirmation_entities"],
        ) != (low, high, confirmation):
            raise ValueError("Gate 3 selected loads violate the frozen rule")
    elif gate == 4:
        validate_sha256(
            receipt["gate_3_receipt_sha256"],
            "Gate 3 receipt SHA-256",
        )
        normalized = _normalize_budget(measurements)
        selected = _select_budget(normalized)
        expected_passed = selected is not None
        if receipt["tokens_per_parameter"] != selected:
            raise ValueError("Gate 4 token budget violates the frozen rule")
    elif gate == 5:
        validate_sha256(
            receipt["gate_3_receipt_sha256"],
            "Gate 3 receipt SHA-256",
        )
        validate_sha256(
            receipt["gate_4_receipt_sha256"],
            "Gate 4 receipt SHA-256",
        )
        validated = validate_design_receipt(receipt["design_receipt"])
        expected_passed = validated.passed
        if receipt["design_receipt_sha256"] != canonical_sha256(
            validated.to_dict()
        ):
            raise ValueError("Gate 5 design receipt hash mismatch")
        high_entities = _integer(
            receipt["high_entities"],
            "Gate 5 high entity load",
            minimum=1,
        )
        tokens_per_parameter = _integer(
            receipt["tokens_per_parameter"],
            "Gate 5 tokens per parameter",
            minimum=1,
        )
        binding = _normalize_gate_5_development_binding(
            receipt["development_binding"],
            validated,
            expected_high_entities=high_entities,
            expected_tokens_per_parameter=tokens_per_parameter,
        )
        if receipt["development_binding_sha256"] != canonical_sha256(
            binding
        ):
            raise ValueError("Gate 5 development binding hash mismatch")
        expected_measurements = {
            "power": validated.power,
            "power_ci_lo": validated.power_ci_lo,
            "power_ci_hi": validated.power_ci_hi,
            "studies": validated.studies,
            "pairs": validated.pairs,
        }
        if measurements != expected_measurements:
            raise ValueError("Gate 5 measured values mismatch")
    else:
        raise ValueError("gate number must be in 0 through 5")
    if passed != expected_passed:
        raise ValueError(f"Gate {gate} passed decision is inconsistent")


def validate_gate_receipt(
    raw: Mapping[str, Any],
    *,
    expected_gate: int | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("gate receipt must be an object")
    gate = raw.get("gate")
    if (
        isinstance(gate, bool)
        or not isinstance(gate, int)
        or gate not in _GATE_FIELDS
    ):
        raise ValueError("gate receipt number must be in 0 through 5")
    if expected_gate is not None and gate != expected_gate:
        raise ValueError("gate receipt number mismatch")
    if set(raw) != _GATE_FIELDS[gate]:
        raise ValueError(f"Gate {gate} receipt fields are not exact")
    if (
        raw["record_type"] != "relational_gate_receipt"
        or raw["schema_version"] != 1
        or raw["rule_version"] != _RULE_VERSIONS[gate]
    ):
        raise ValueError(f"Gate {gate} receipt protocol mismatch")
    normalized = json.loads(canonical_json_bytes(raw))
    normalized["input_hashes"] = _input_hashes(normalized["input_hashes"])
    expected_decision = _decision_for(normalized)
    if normalized["decision"] != expected_decision:
        raise ValueError(f"Gate {gate} decision fields mismatch")
    if normalized["decision_sha256"] != canonical_sha256(expected_decision):
        raise ValueError(f"Gate {gate} decision hash mismatch")
    validate_sha256(normalized["receipt_sha256"], "gate receipt SHA-256")
    without_hash = {
        key: value
        for key, value in normalized.items()
        if key != "receipt_sha256"
    }
    if normalized["receipt_sha256"] != canonical_sha256(without_hash):
        raise ValueError(f"Gate {gate} receipt hash mismatch")
    _validate_semantics(normalized)
    return normalized


def load_development_gate_inputs(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if any(part.lower().startswith("protected") for part in source.parts):
        raise ValueError("development gate loader never opens a protected path")
    raw = load_canonical_json(source)
    return _normalize_inputs(raw)
