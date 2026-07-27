"""Exact relational endpoints and frozen guardrail measurements."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import torch

from evals.relational_contracts import (
    EvalRow,
    GuardrailReport,
    validate_eval_rows,
)

EXPECTED_TASKS = (
    "path_composition",
    "date_ordering",
    "balanced_equality",
)
EXPECTED_PAIRS_PER_TASK = 10_000
PREREGISTERED_REASONING_MILESTONES = {
    f"{task}_pair_accuracy_0.75": 0.75
    for task in EXPECTED_TASKS
}
_Z_975 = 1.959963984540054
_TOLERANCE = 1e-12

CONFIRMATORY_STUDY_DEFINITION = {
    "record_type": "confirmatory_study_definition",
    "schema_version": 1,
    "result_contract": "relational-result-v1",
    "paired_arms": ["split", "dense"],
    "guard_checks": {
        "split_on_recall_floor": {
            "rule": "absolute_rate",
            "comparison": ">=",
            "threshold": 0.95,
        },
        "split_on_recall_noninferiority": {
            "rule": "reference_additive_margin",
            "comparison": ">=",
            "margin": -0.02,
        },
        "split_off_recall": {
            "rule": "absolute_rate",
            "comparison": "<",
            "threshold": 0.05,
        },
        "split_off_recognition_wilson_hi": {
            "rule": "wilson_upper_bound_95",
            "comparison": "<",
            "threshold": 0.30,
        },
        "split_off_first_hop": {
            "rule": "absolute_rate",
            "comparison": ">=",
            "threshold": 0.75,
        },
        "gold_return_path_noninferiority": {
            "rule": "reference_additive_margin",
            "comparison": ">=",
            "margin": -0.05,
        },
        "rule_noninferiority": {
            "rule": "reference_additive_margin",
            "comparison": ">=",
            "margin": -0.02,
        },
        "bpb_noninferiority": {
            "rule": "reference_multiplier",
            "comparison": "<=",
            "multiplier": 1.01,
        },
        "integrity_conjunction": {
            "rule": "boolean_conjunction",
            "comparison": "==",
            "threshold": True,
        },
    },
    "integrity_inputs": [
        "mask_ledger",
        "corpus_pairing",
        "provenance",
        "exact_matrix",
    ],
    "excluded_analysis_roles": ["exploratory_only"],
}


def confirmatory_study_definition_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            CONFIRMATORY_STUDY_DEFINITION,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _field(value, name: str):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _row_pair_id(row) -> str:
    if isinstance(row, Mapping) and "pair_id" in row:
        return str(row["pair_id"])
    meta = _field(row, "meta")
    return str(meta["pair_id"])


def _row_variant(row) -> str:
    if isinstance(row, Mapping) and "variant" in row:
        variant = str(row["variant"])
    else:
        meta = _field(row, "meta")
        if "variant" in meta:
            variant = str(meta["variant"])
        else:
            qid = str(_field(row, "qid"))
            if qid.endswith(("-o", "_o", "-original")):
                variant = "original"
            elif qid.endswith(("-c", "_c", "-counterfactual")):
                variant = "counterfactual"
            else:
                raise ValueError(
                    "every pair requires original and counterfactual variants"
                )
    if variant not in ("original", "counterfactual"):
        raise ValueError(f"unexpected pair variant: {variant}")
    return variant


def counterfactual_pair_accuracy(
    rows,
    *,
    expected_pairs: int | None = None,
) -> float:
    materialized = list(rows)
    if not materialized:
        raise ValueError("counterfactual accuracy requires at least one pair")
    if expected_pairs is not None and expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")

    grouped: dict[str, dict[str, Mapping]] = defaultdict(dict)
    seen_qids = set()
    for row in materialized:
        pair_id = _row_pair_id(row)
        variant = _row_variant(row)
        qid = str(_field(row, "qid"))
        if variant in grouped[pair_id]:
            raise ValueError(
                "every pair requires distinct original and counterfactual variants"
            )
        if qid in seen_qids:
            raise ValueError(f"duplicate eval qid: {qid}")
        seen_qids.add(qid)
        grouped[pair_id][variant] = row

    if expected_pairs is not None and len(grouped) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} pairs, got {len(grouped)}"
        )
    required = {"original", "counterfactual"}
    if any(set(pair) != required for pair in grouped.values()):
        raise ValueError(
            "every pair requires original and counterfactual variants"
        )
    successes = 0
    for pair in grouped.values():
        original_correct = _field(pair["original"], "correct")
        counterfactual_correct = _field(pair["counterfactual"], "correct")
        if not isinstance(original_correct, bool) or not isinstance(
            counterfactual_correct, bool
        ):
            raise ValueError("pair correct fields must be Boolean")
        successes += original_correct and counterfactual_correct
    return successes / len(grouped)


def assert_expected_counts(
    rows_by_task: Mapping[str, Sequence],
    n_pairs: int = EXPECTED_PAIRS_PER_TASK,
) -> None:
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    actual_tasks = set(rows_by_task)
    expected_tasks = set(EXPECTED_TASKS)
    if actual_tasks != expected_tasks:
        raise ValueError(
            "task set mismatch; "
            f"missing={sorted(expected_tasks - actual_tasks)}, "
            f"extra={sorted(actual_tasks - expected_tasks)}"
        )
    expected_rows = 2 * n_pairs
    for task in EXPECTED_TASKS:
        rows = list(rows_by_task[task])
        if len(rows) != expected_rows:
            raise ValueError(
                f"{task}: expected {expected_rows} rows, got {len(rows)}"
            )
        if any(str(_field(row, "task")) != task for row in rows):
            raise ValueError(f"{task}: row task labels do not match stratum")
        counterfactual_pair_accuracy(rows, expected_pairs=n_pairs)


def _nonnegative_int(row, name: str) -> int:
    value = _field(row, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def path_metrics(rows) -> dict:
    materialized = list(rows)
    if not materialized:
        raise ValueError("path metrics require at least one row")

    hop_correct = 0
    hop_total = 0
    exact = 0
    misses = 0
    for row in materialized:
        actions = list(_field(row, "actions"))
        gold = list(_field(row, "gold_actions"))
        if not gold:
            raise ValueError("every path row requires at least one gold action")
        exact += actions == gold
        hop_total += len(gold)
        hop_correct += sum(
            predicted == expected
            for predicted, expected in zip(actions, gold)
        )
        misses += _nonnegative_int(row, "misses")

    n_rows = len(materialized)
    return {
        "full_path_exact": exact / n_rows,
        "per_hop_accuracy": hop_correct / hop_total,
        "miss_rate": misses / hop_total,
    }


def path_diagnostics(rows) -> dict:
    materialized = list(rows)
    core = path_metrics(materialized)
    referent_correct = 0
    hop_total = 0
    excess_reads = 0
    malformed = 0
    action_steps = 0
    halt_steps = []
    halted = 0
    for row in materialized:
        gold = list(_field(row, "gold_actions"))
        referents = list(_field(row, "correct_referents"))
        if len(referents) != len(gold):
            raise ValueError(
                "correct_referents must contain one value per gold action"
            )
        n_steps = _nonnegative_int(row, "n_steps")
        if n_steps != 6:
            raise ValueError("every relational decode must contain six steps")
        halt_step = _field(row, "halt_step")
        if halt_step is not None:
            if (
                isinstance(halt_step, bool)
                or not isinstance(halt_step, int)
                or not 1 <= halt_step <= 6
            ):
                raise ValueError("halt_step must be in [1, 6] or None")
            halted += 1
            halt_steps.append(halt_step)
        else:
            halt_steps.append(7)
        hop_total += len(gold)
        referent_correct += sum(bool(value) for value in referents)
        excess_reads += _nonnegative_int(row, "excess_reads")
        malformed += _nonnegative_int(row, "malformed")
        action_steps += n_steps

    n_rows = len(materialized)
    return {
        **core,
        "correct_referent_rate": referent_correct / hop_total,
        "excess_read_rate": excess_reads / hop_total,
        "malformed_rate": malformed / action_steps,
        "mean_excess_reads": excess_reads / n_rows,
        "mean_halt_step": sum(halt_steps) / n_rows,
        "halt_rate": halted / n_rows,
        "n_rows": n_rows,
        "n_gold_hops": hop_total,
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = _Z_975,
) -> tuple[float, float]:
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
    ):
        raise ValueError("total must be positive")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or not 0 <= successes <= total
    ):
        raise ValueError("successes must be between zero and total")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * (
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return center - radius, center + radius


def exact_accuracy(rows, *, expected_count: int | None = None) -> dict:
    materialized = list(rows)
    if not materialized:
        raise ValueError("accuracy requires at least one row")
    if expected_count is not None and len(materialized) != expected_count:
        raise ValueError(
            f"expected {expected_count} rows, got {len(materialized)}"
        )
    correct = sum(bool(_field(row, "correct")) for row in materialized)
    low, high = wilson_interval(correct, len(materialized))
    return {
        "accuracy": correct / len(materialized),
        "ci_lo": low,
        "ci_hi": high,
        "n": len(materialized),
        "correct": correct,
    }


def _score_value(value) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("choice score tuples must not be empty")
        value = value[0]
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("choice scores must be finite")
    return score


def recognition_accuracy(
    score_choices,
    items,
    *,
    expected_count: int | None = None,
) -> dict:
    materialized = list(items)
    if not materialized:
        raise ValueError("recognition requires at least one item")
    if expected_count is not None and len(materialized) != expected_count:
        raise ValueError(
            f"expected {expected_count} recognition items, "
            f"got {len(materialized)}"
        )
    correct = 0
    for item in materialized:
        prompt = str(_field(item, "prompt"))
        choices = list(_field(item, "choices"))
        if len(choices) != 4:
            raise ValueError("recognition items require exactly four choices")
        answer_index = _field(item, "answer_index")
        if (
            isinstance(answer_index, bool)
            or not isinstance(answer_index, int)
            or not 0 <= answer_index < 4
        ):
            raise ValueError("answer_index must select one of four choices")
        raw_scores = list(score_choices(prompt, choices))
        if len(raw_scores) != 4:
            raise ValueError("score_choices must return one score per choice")
        scores = [_score_value(value) for value in raw_scores]
        correct += (
            max(range(len(scores)), key=scores.__getitem__) == answer_index
        )
    low, high = wilson_interval(correct, len(materialized))
    return {
        "accuracy": correct / len(materialized),
        "ci_lo": low,
        "ci_hi": high,
        "n": len(materialized),
        "correct": correct,
    }


def score_choice_loglikelihoods(
    model,
    tok,
    prompt: str,
    choices,
    device=None,
) -> list[float]:
    """Score choices with ``GPT.forward_step`` and no padded batches."""

    resolved_device = torch.device(
        device if device is not None else getattr(model, "device", "cpu")
    )
    context_ids = tok.encode(prompt) or [tok.EOT]
    scores = []
    for choice in choices:
        choice_ids = tok.encode(str(choice))
        if not choice_ids:
            raise ValueError("every choice must encode to at least one token")
        input_ids = context_ids + choice_ids[:-1]
        value = torch.tensor(
            [input_ids], dtype=torch.long, device=resolved_device
        )
        with torch.no_grad():
            logits, _ = model.forward_step(value, None)
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        positions = torch.arange(
            len(context_ids) - 1,
            len(context_ids) - 1 + len(choice_ids),
            device=resolved_device,
        )
        targets = torch.tensor(
            choice_ids, dtype=torch.long, device=resolved_device
        )
        scores.append(float(log_probs[positions, targets].sum().item()))
    return scores


def shared_text_bpb(
    total_nll_nats: float,
    total_utf8_bytes: int,
) -> float:
    if (
        isinstance(total_utf8_bytes, bool)
        or not isinstance(total_utf8_bytes, int)
        or total_utf8_bytes <= 0
    ):
        raise ValueError("held-out text must contain bytes")
    if not math.isfinite(total_nll_nats) or total_nll_nats < 0:
        raise ValueError("total_nll_nats must be finite and non-negative")
    return total_nll_nats / (total_utf8_bytes * math.log(2))


def measure_shared_text_bpb(model, tok, texts, device=None) -> dict:
    """Measure UTF-8 BPB using only unpadded ``forward_step`` calls."""

    materialized = [str(text) for text in texts]
    if not materialized:
        raise ValueError("shared-text measurement requires at least one text")
    resolved_device = torch.device(
        device if device is not None else getattr(model, "device", "cpu")
    )
    total_nll = 0.0
    total_bytes = 0
    for text in materialized:
        token_ids = tok.encode(text)
        if not token_ids:
            if text.encode("utf-8"):
                raise ValueError("non-empty text encoded to no tokens")
            continue
        inputs = [tok.EOT] + token_ids[:-1]
        value = torch.tensor(
            [inputs], dtype=torch.long, device=resolved_device
        )
        targets = torch.tensor(
            token_ids, dtype=torch.long, device=resolved_device
        )
        with torch.no_grad():
            logits, _ = model.forward_step(value, None)
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        positions = torch.arange(len(token_ids), device=resolved_device)
        total_nll -= float(log_probs[positions, targets].sum().item())
        total_bytes += len(text.encode("utf-8"))
    return {
        "bpb": shared_text_bpb(total_nll, total_bytes),
        "total_nll_nats": total_nll,
        "total_utf8_bytes": total_bytes,
        "n": len(materialized),
    }


def _validated_accuracy(measurement: Mapping, name: str) -> tuple[float, int]:
    accuracy = float(measurement["accuracy"])
    n = measurement["n"]
    if not 0 <= accuracy <= 1 or not math.isfinite(accuracy):
        raise ValueError(f"{name} accuracy must be in [0, 1]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"{name} n must be positive")
    return accuracy, n


def recognition_guardrails(
    dense_recognition: Mapping,
    split_store_off_recognition: Mapping,
) -> dict[str, dict]:
    dense_accuracy, dense_n = _validated_accuracy(
        dense_recognition, "dense recognition"
    )
    split_accuracy, split_n = _validated_accuracy(
        split_store_off_recognition, "split recognition"
    )
    dense_low = float(dense_recognition["ci_lo"])
    split_high = float(split_store_off_recognition["ci_hi"])
    if dense_n != split_n:
        raise ValueError("recognition arms must use the same item count")
    if (
        not math.isfinite(dense_low)
        or not math.isfinite(split_high)
        or not 0 <= dense_low <= 1
        or not 0 <= split_high <= 1
    ):
        raise ValueError("recognition confidence bounds must be in [0, 1]")
    return {
        "burden": {
            "value": dense_low,
            "estimate": dense_accuracy,
            "threshold": 0.30,
            "comparison": ">",
            "passed": dense_low > 0.30,
            "n": dense_n,
        },
        "leakage": {
            "value": split_high,
            "estimate": split_accuracy,
            "threshold": 0.30,
            "comparison": "<",
            "passed": split_high < 0.30,
            "n": split_n,
        },
    }


def _paired_accuracy_guardrail(
    split: Mapping,
    dense: Mapping,
    name: str,
) -> dict:
    split_accuracy, split_n = _validated_accuracy(split, f"split {name}")
    dense_accuracy, dense_n = _validated_accuracy(dense, f"dense {name}")
    if split_n != dense_n:
        raise ValueError(f"{name} arms must use the same item count")
    delta = split_accuracy - dense_accuracy
    return {
        "value": delta,
        "split": split_accuracy,
        "dense": dense_accuracy,
        "threshold": -0.02,
        "comparison": ">=",
        "test": "one-sided noninferiority",
        "rule": "split >= dense - 0.02",
        "passed": delta >= -0.02 - _TOLERANCE,
        "n": min(split_n, dense_n),
        "n_split": split_n,
        "n_dense": dense_n,
    }


def internal_knowledge_guardrail(
    split_internal: Mapping,
    dense_internal: Mapping,
) -> dict:
    """One-sided noninferiority: Split may trail Dense by at most .02."""

    return _paired_accuracy_guardrail(
        split_internal, dense_internal, "internal knowledge"
    )


def language_bpb_guardrail(
    split_language: Mapping,
    dense_language: Mapping,
) -> dict:
    split_bpb = float(split_language["bpb"])
    dense_bpb = float(dense_language["bpb"])
    split_bytes = split_language["total_utf8_bytes"]
    dense_bytes = dense_language["total_utf8_bytes"]
    if (
        not math.isfinite(split_bpb)
        or not math.isfinite(dense_bpb)
        or split_bpb < 0
        or dense_bpb <= 0
    ):
        raise ValueError("language BPB values must be finite and positive")
    if (
        isinstance(split_bytes, bool)
        or not isinstance(split_bytes, int)
        or split_bytes <= 0
        or isinstance(dense_bytes, bool)
        or not isinstance(dense_bytes, int)
        or dense_bytes <= 0
    ):
        raise ValueError("language byte counts must be positive")
    if split_bytes != dense_bytes:
        raise ValueError("language arms must score the same shared text")
    ratio = split_bpb / dense_bpb
    return {
        "value": ratio,
        "split_bpb": split_bpb,
        "dense_bpb": dense_bpb,
        "threshold": 1.01,
        "comparison": "<=",
        "passed": ratio <= 1.01 + _TOLERANCE,
        "n": min(split_bytes, dense_bytes),
        "split_utf8_bytes": split_bytes,
        "dense_utf8_bytes": dense_bytes,
    }


def _rate_measurement(
    value: float,
    *,
    threshold,
    comparison: str,
    passed: bool,
    n: int,
) -> dict:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("guardrail rates must be in [0, 1]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("guardrail counts must be positive")
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
        "n": n,
    }


def route_guardrails(audit: Mapping) -> dict[str, dict]:
    route = float(audit["route_rate"])
    tail = float(audit["low_use_high_entropy_external_rate"])
    structure = float(audit["rules_top_centrality_internal_rate"])
    return {
        "route_rate": _rate_measurement(
            route,
            threshold=[0.40, 0.60],
            comparison="inside",
            passed=0.40 <= route <= 0.60,
            n=audit["route_total"],
        ),
        "tail_external": _rate_measurement(
            tail,
            threshold=0.80,
            comparison=">=",
            passed=tail >= 0.80,
            n=audit["low_use_high_entropy_total"],
        ),
        "structure_internal": _rate_measurement(
            structure,
            threshold=0.80,
            comparison=">=",
            passed=structure >= 0.80,
            n=audit["rules_top_centrality_total"],
        ),
    }


def mask_ledger_guardrail(audit: Mapping, *, condition: str) -> dict:
    if condition not in ("dense", "split", "random"):
        raise ValueError(f"unexpected training condition: {condition}")
    unmasked = audit["unmasked_external_payloads"]
    external_total = audit["external_payload_occurrences"]
    masked_protected = audit["masked_rule_action_answer_targets"]
    protected_total = audit["rule_action_answer_targets"]
    for name, value in (
        ("unmasked_external_payloads", unmasked),
        ("external_payload_occurrences", external_total),
        ("masked_rule_action_answer_targets", masked_protected),
        ("rule_action_answer_targets", protected_total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if external_total <= 0 or protected_total <= 0:
        raise ValueError("mask ledger denominators must be positive")
    if unmasked > external_total or masked_protected > protected_total:
        raise ValueError("mask ledger violations exceed audited occurrences")
    external_mask_applicable = condition == "split"
    violations = masked_protected + (
        unmasked if external_mask_applicable else 0
    )
    return {
        "value": violations,
        "condition": condition,
        "external_mask_applicable": external_mask_applicable,
        "unmasked_external_payloads": unmasked,
        "masked_rule_action_answer_targets": masked_protected,
        "threshold": 0,
        "comparison": "==",
        "passed": violations == 0,
        "n": external_total + protected_total,
    }


def _count_rate(
    numerator: int,
    denominator: int,
) -> dict[str, int | float | None]:
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
        or numerator > denominator
    ):
        raise ValueError(
            "rate counts must satisfy 0 <= numerator <= denominator"
        )
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _strict_pair_rate(rows: Sequence[EvalRow]) -> dict:
    if not rows:
        return _count_rate(0, 0)
    grouped: dict[str, dict[str, EvalRow]] = defaultdict(dict)
    for row in rows:
        if row.variant in grouped[row.pair_id]:
            raise ValueError("duplicate variant in metric pair")
        grouped[row.pair_id][row.variant] = row
    if any(
        set(pair) != {"original", "counterfactual"}
        for pair in grouped.values()
    ):
        raise ValueError("metric slices require both variants in every pair")
    successes = sum(
        pair["original"].correct and pair["counterfactual"].correct
        for pair in grouped.values()
    )
    return _count_rate(successes, len(grouped))


def _strict_action_path(row: EvalRow) -> bool:
    predicted = [action for action in row.all_actions if action[3]]
    gold = [action for action in row.gold_all_actions if action[3]]
    return predicted == gold


def _slice_checkpoint_metrics(rows: Sequence[EvalRow]) -> dict:
    if not rows:
        return {
            "item_accuracy": _count_rate(0, 0),
            "pair_accuracy": _count_rate(0, 0),
            "exact_action_path": _count_rate(0, 0),
        }
    return {
        "item_accuracy": _count_rate(
            sum(row.correct for row in rows),
            len(rows),
        ),
        "pair_accuracy": _strict_pair_rate(rows),
        "exact_action_path": _count_rate(
            sum(_strict_action_path(row) for row in rows),
            len(rows),
        ),
    }


class CheckpointMetricAccumulator:
    """Incrementally compute strict metrics from complete two-row pairs."""

    _IDENTITY_FIELDS = (
        "checkpoint_sha256",
        "model_id",
        "arm",
        "seed",
        "raw_token_count",
        "memory_mode",
        "control_id",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "configuration_sha256",
        "result_schema_sha256",
        "provenance_sha256",
        "control_seed",
    )

    def __init__(
        self,
        *,
        milestone_crossings: Mapping[str, int | None] | None = None,
    ) -> None:
        self._identity: tuple | None = None
        self._n_rows = 0
        self._n_pairs = 0
        self._item_correct = 0
        self._pair_correct = 0
        self._all_six_exact = 0
        self._path_exact = 0
        self._retrieval_correct = 0
        self._answer_given_retrieval = 0
        self._malformed = 0
        self._misses = 0
        self._abstained = 0
        self._excess_reads = 0
        self._read_attempts = 0
        self._lookup_count = 0
        self._lookup_latency_ns = 0
        self._locality_correct = 0
        self._locality_total = 0
        self._store_rows: int | None = None
        self._store_bytes: int | None = None
        self._per_hop = [
            {
                "denominator": 0,
                "relation": 0,
                "direction": 0,
                "action": 0,
                "referent": 0,
            }
            for _ in range(6)
        ]
        self._slices = {
            "by_hop": {
                str(hop): self._empty_slice() for hop in range(1, 7)
            },
            "by_composition": {
                split: self._empty_slice() for split in ("seen", "heldout")
            },
            "by_task": {
                task: self._empty_slice() for task in EXPECTED_TASKS
            },
        }
        self._crossings = (
            {
                name: None
                for name in PREREGISTERED_REASONING_MILESTONES
            }
            if milestone_crossings is None
            else dict(milestone_crossings)
        )
        self._validate_crossings()

    @staticmethod
    def _empty_slice() -> dict[str, int]:
        return {
            "item_correct": 0,
            "item_total": 0,
            "pair_correct": 0,
            "pair_total": 0,
            "path_exact": 0,
        }

    def _validate_crossings(self) -> None:
        if set(self._crossings) != set(PREREGISTERED_REASONING_MILESTONES):
            raise ValueError(
                "milestone crossing set must match preregistered milestones"
            )
        for name, crossing in self._crossings.items():
            if crossing is not None and (
                isinstance(crossing, bool)
                or not isinstance(crossing, int)
                or crossing < 0
            ):
                raise ValueError(
                    f"milestone crossing {name} must be nonnegative or null"
                )

    def _add_row(self, row: EvalRow) -> bool:
        identity = tuple(
            getattr(row, field) for field in self._IDENTITY_FIELDS
        )
        if self._identity is None:
            self._identity = identity
            self._store_rows = row.store_rows
            self._store_bytes = row.store_bytes
        elif identity != self._identity:
            raise ValueError(
                "checkpoint metrics cannot cross evaluation identities"
            )
        if (
            row.store_rows != self._store_rows
            or row.store_bytes != self._store_bytes
        ):
            raise ValueError(
                "store rows/bytes must match within a checkpoint"
            )

        predicted = [action for action in row.all_actions if action[3]]
        gold = [action for action in row.gold_all_actions if action[3]]
        exact_path = predicted == gold
        retrieved = (
            exact_path
            and len(row.correct_referents) == row.hop
            and all(row.correct_referents)
        )
        self._n_rows += 1
        self._item_correct += row.correct
        self._all_six_exact += row.all_actions == row.gold_all_actions
        self._path_exact += exact_path
        self._retrieval_correct += retrieved
        self._answer_given_retrieval += row.correct and retrieved
        self._malformed += row.malformed
        self._misses += row.misses
        self._abstained += row.abstained
        self._excess_reads += row.excess_reads
        self._read_attempts += len(predicted)
        self._lookup_count += row.lookup_count
        self._lookup_latency_ns += row.lookup_latency_ns
        if row.edit_locality_correct is not None:
            self._locality_total += 1
            self._locality_correct += row.edit_locality_correct

        for hop_index, expected in enumerate(gold):
            metrics = self._per_hop[hop_index]
            metrics["denominator"] += 1
            actual = (
                predicted[hop_index]
                if hop_index < len(predicted)
                else None
            )
            metrics["action"] += actual == expected
            metrics["relation"] += (
                actual is not None and actual[1] == expected[1]
            )
            metrics["direction"] += (
                actual is not None and actual[2] == expected[2]
            )
            metrics["referent"] += bool(
                row.correct_referents[hop_index]
            )

        for family, key in (
            ("by_hop", str(row.hop)),
            ("by_composition", row.composition_split),
            ("by_task", row.task),
        ):
            values = self._slices[family][key]
            values["item_total"] += 1
            values["item_correct"] += row.correct
            values["path_exact"] += exact_path
        return exact_path

    def add_pair(
        self,
        rows: Iterable[EvalRow | Mapping],
    ) -> None:
        pair = validate_eval_rows(rows)
        if len(pair) != 2 or len({row.pair_id for row in pair}) != 1:
            raise ValueError(
                "metric accumulator requires one complete pair at a time"
            )
        by_variant = {row.variant: row for row in pair}
        if set(by_variant) != {"original", "counterfactual"}:
            raise ValueError("metric accumulator requires both variants")
        exact_paths = [self._add_row(row) for row in pair]
        pair_correct = (
            by_variant["original"].correct
            and by_variant["counterfactual"].correct
        )
        self._n_pairs += 1
        self._pair_correct += pair_correct
        first = pair[0]
        for family, key in (
            ("by_hop", str(first.hop)),
            ("by_composition", first.composition_split),
            ("by_task", first.task),
        ):
            values = self._slices[family][key]
            values["pair_total"] += 1
            values["pair_correct"] += pair_correct
        if len(exact_paths) != 2:
            raise AssertionError("complete pairs must contain two rows")

    @staticmethod
    def _finished_slice(values: Mapping[str, int]) -> dict:
        return {
            "item_accuracy": _count_rate(
                values["item_correct"],
                values["item_total"],
            ),
            "pair_accuracy": _count_rate(
                values["pair_correct"],
                values["pair_total"],
            ),
            "exact_action_path": _count_rate(
                values["path_exact"],
                values["item_total"],
            ),
        }

    def finalize(self) -> dict:
        if self._identity is None or self._n_rows == 0 or self._n_pairs == 0:
            raise ValueError("checkpoint metrics require evaluation pairs")
        control_id = self._identity[
            self._IDENTITY_FIELDS.index("control_id")
        ]
        per_hop = {}
        for hop_index, values in enumerate(self._per_hop, start=1):
            denominator = values["denominator"]
            per_hop[str(hop_index)] = {
                name: _count_rate(values[name], denominator)
                for name in ("relation", "direction", "action", "referent")
            }
        return {
            "item_accuracy": _count_rate(
                self._item_correct,
                self._n_rows,
            ),
            "pair_accuracy": _count_rate(
                self._pair_correct,
                self._n_pairs,
            ),
            "all_six_action_exact": _count_rate(
                self._all_six_exact,
                self._n_rows,
            ),
            "exact_action_path": _count_rate(
                self._path_exact,
                self._n_rows,
            ),
            "answer_given_correct_retrieval": _count_rate(
                self._answer_given_retrieval,
                self._retrieval_correct,
            ),
            "gold_path_answer_accuracy": (
                _count_rate(self._item_correct, self._n_rows)
                if control_id == "gold_path"
                else _count_rate(0, 0)
            ),
            "malformed_rate": _count_rate(
                self._malformed,
                6 * self._n_rows,
            ),
            "miss_rate": _count_rate(
                self._misses,
                self._read_attempts,
            ),
            "abstention_rate": _count_rate(
                self._abstained,
                self._n_rows,
            ),
            "excess_read_rate": _count_rate(
                self._excess_reads,
                6 * self._n_rows,
            ),
            "per_hop": per_hop,
            **{
                family: {
                    key: self._finished_slice(values)
                    for key, values in slices.items()
                }
                for family, slices in self._slices.items()
            },
            "store": {
                "rows": self._store_rows,
                "bytes": self._store_bytes,
                "lookup_latency_ns": (
                    None
                    if self._lookup_count == 0
                    else self._lookup_latency_ns / self._lookup_count
                ),
                "lookup_count": self._lookup_count,
            },
            "edit_locality": _count_rate(
                self._locality_correct,
                self._locality_total,
            ),
            "milestone_crossings": {
                name: self._crossings[name]
                for name in sorted(self._crossings)
            },
        }


def compute_checkpoint_metrics(
    rows: Iterable[EvalRow | Mapping],
    *,
    milestone_crossings: Mapping[str, int | None] | None = None,
) -> dict:
    """Compute every preregistered row/checkpoint metric with counts."""

    materialized = validate_eval_rows(rows)
    identity_fields = (
        "checkpoint_sha256",
        "model_id",
        "arm",
        "seed",
        "raw_token_count",
        "memory_mode",
        "control_id",
        "evaluator_sha256",
        "data_sha256",
        "relation_schema_sha256",
        "configuration_sha256",
        "result_schema_sha256",
        "provenance_sha256",
        "control_seed",
    )
    first = materialized[0]
    for row in materialized[1:]:
        if any(
            getattr(row, field) != getattr(first, field)
            for field in identity_fields
        ):
            raise ValueError(
                "checkpoint metrics cannot cross evaluation identities"
            )
    n_rows = len(materialized)
    pair_rate = _strict_pair_rate(materialized)
    predicted_reads = [
        [action for action in row.all_actions if action[3]]
        for row in materialized
    ]
    gold_reads = [
        [action for action in row.gold_all_actions if action[3]]
        for row in materialized
    ]
    exact_paths = [
        predicted == gold
        for predicted, gold in zip(predicted_reads, gold_reads)
    ]
    retrieval_correct = [
        exact
        and len(row.correct_referents) == row.hop
        and all(row.correct_referents)
        for row, exact in zip(materialized, exact_paths)
    ]

    per_hop: dict[str, dict[str, dict]] = {}
    for hop_index in range(6):
        totals = {
            "relation": 0,
            "direction": 0,
            "action": 0,
            "referent": 0,
        }
        denominator = 0
        for row, predicted, gold in zip(
            materialized,
            predicted_reads,
            gold_reads,
        ):
            if hop_index >= len(gold):
                continue
            denominator += 1
            actual = (
                predicted[hop_index]
                if hop_index < len(predicted)
                else None
            )
            expected = gold[hop_index]
            totals["action"] += actual == expected
            totals["relation"] += (
                actual is not None and actual[1] == expected[1]
            )
            totals["direction"] += (
                actual is not None and actual[2] == expected[2]
            )
            totals["referent"] += bool(
                row.correct_referents[hop_index]
            )
        per_hop[str(hop_index + 1)] = {
            name: _count_rate(total, denominator)
            for name, total in totals.items()
        }

    by_hop = {
        str(hop): _slice_checkpoint_metrics(
            [row for row in materialized if row.hop == hop]
        )
        for hop in range(1, 7)
    }
    by_composition = {
        split: _slice_checkpoint_metrics(
            [
                row
                for row in materialized
                if row.composition_split == split
            ]
        )
        for split in ("seen", "heldout")
    }
    by_task = {
        task: _slice_checkpoint_metrics(
            [row for row in materialized if row.task == task]
        )
        for task in EXPECTED_TASKS
    }
    store_rows = {row.store_rows for row in materialized}
    store_bytes = {row.store_bytes for row in materialized}
    if len(store_rows) != 1 or len(store_bytes) != 1:
        raise ValueError("store rows/bytes must match within a checkpoint")
    read_attempts = sum(len(actions) for actions in predicted_reads)
    lookup_count = sum(row.lookup_count for row in materialized)
    lookup_latency = (
        None
        if lookup_count == 0
        else sum(row.lookup_latency_ns for row in materialized)
        / lookup_count
    )
    locality_values = [
        row.edit_locality_correct
        for row in materialized
        if row.edit_locality_correct is not None
    ]
    crossings = (
        {
            name: None
            for name in PREREGISTERED_REASONING_MILESTONES
        }
        if milestone_crossings is None
        else dict(milestone_crossings)
    )
    if not crossings:
        raise ValueError("milestone crossings must not be empty")
    for name, crossing in crossings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("milestone names must be nonempty strings")
        if crossing is not None and (
            isinstance(crossing, bool)
            or not isinstance(crossing, int)
            or crossing < 0
        ):
            raise ValueError(
                "milestone crossings must be nonnegative integers or null"
            )
    return {
        "item_accuracy": _count_rate(
            sum(row.correct for row in materialized),
            n_rows,
        ),
        "pair_accuracy": pair_rate,
        "all_six_action_exact": _count_rate(
            sum(
                row.all_actions == row.gold_all_actions
                for row in materialized
            ),
            n_rows,
        ),
        "exact_action_path": _count_rate(
            sum(exact_paths),
            n_rows,
        ),
        "answer_given_correct_retrieval": _count_rate(
            sum(
                row.correct and retrieved
                for row, retrieved in zip(
                    materialized,
                    retrieval_correct,
                )
            ),
            sum(retrieval_correct),
        ),
        "gold_path_answer_accuracy": (
            _count_rate(
                sum(row.correct for row in materialized),
                n_rows,
            )
            if first.control_id == "gold_path"
            else _count_rate(0, 0)
        ),
        "malformed_rate": _count_rate(
            sum(row.malformed for row in materialized),
            n_rows * 6,
        ),
        "miss_rate": _count_rate(
            sum(row.misses for row in materialized),
            read_attempts,
        ),
        "abstention_rate": _count_rate(
            sum(row.abstained for row in materialized),
            n_rows,
        ),
        "excess_read_rate": _count_rate(
            sum(row.excess_reads for row in materialized),
            n_rows * 6,
        ),
        "per_hop": per_hop,
        "by_hop": by_hop,
        "by_composition": by_composition,
        "by_task": by_task,
        "store": {
            "rows": next(iter(store_rows)),
            "bytes": next(iter(store_bytes)),
            "lookup_latency_ns": lookup_latency,
            "lookup_count": lookup_count,
        },
        "edit_locality": _count_rate(
            sum(value is True for value in locality_values),
            len(locality_values),
        ),
        "milestone_crossings": {
            name: crossings[name] for name in sorted(crossings)
        },
    }


_FROZEN_CHECKPOINT_FIELDS = {
    "tokens_per_parameter",
    "raw_token_count",
    "metrics",
}


def first_frozen_milestone_crossings(
    history: Iterable[Mapping],
    milestones: Mapping[str, float],
) -> dict[str, int | None]:
    """Return the first observed 5x/10x/20x raw-token crossing."""

    if not isinstance(milestones, Mapping) or not milestones:
        raise ValueError("milestones must be a nonempty mapping")
    thresholds: dict[str, float] = {}
    for name, value in milestones.items():
        if not isinstance(name, str) or not name:
            raise ValueError("milestone names must be nonempty strings")
        thresholds[name] = _guard_number(
            value,
            f"milestone {name}",
            minimum=0.0,
            maximum=1.0,
        )
    checkpoints: dict[int, tuple[int, dict[str, float]]] = {}
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping) or set(raw) != _FROZEN_CHECKPOINT_FIELDS:
            raise ValueError(
                "frozen checkpoint fields must be exact"
            )
        multiple = raw["tokens_per_parameter"]
        if (
            isinstance(multiple, bool)
            or not isinstance(multiple, int)
            or multiple not in {5, 10, 20}
        ):
            raise ValueError(
                "crossings require exactly the 5, 10, and 20 checkpoints"
            )
        if multiple in checkpoints:
            raise ValueError("duplicate frozen checkpoint multiple")
        raw_tokens = raw["raw_token_count"]
        if (
            isinstance(raw_tokens, bool)
            or not isinstance(raw_tokens, int)
            or raw_tokens < 0
        ):
            raise ValueError("raw_token_count must be nonnegative")
        values = raw["metrics"]
        if not isinstance(values, Mapping) or set(values) != set(thresholds):
            raise ValueError("checkpoint milestone metric set mismatch")
        checkpoints[multiple] = (
            raw_tokens,
            {
                name: _guard_number(
                    values[name],
                    f"checkpoint {index} metric {name}",
                    minimum=0.0,
                    maximum=1.0,
                )
                for name in thresholds
            },
        )
    if set(checkpoints) != {5, 10, 20}:
        raise ValueError(
            "crossings require exactly the 5, 10, and 20 checkpoints"
        )
    ordered = [checkpoints[multiple] for multiple in (5, 10, 20)]
    if any(
        current[0] <= previous[0]
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError("raw token counts must increase across checkpoints")
    return {
        name: next(
            (
                raw_tokens
                for raw_tokens, values in ordered
                if values[name] >= threshold
            ),
            None,
        )
        for name, threshold in sorted(thresholds.items())
    }


_GUARDRAIL_INPUT_FIELDS = {
    "split_checkpoint_sha256",
    "dense_checkpoint_sha256",
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "split_configuration_sha256",
    "dense_configuration_sha256",
    "result_schema_sha256",
    "split_result_provenance_sha256",
    "dense_result_provenance_sha256",
    "study_provenance_sha256",
    "pairing_receipt_sha256",
    "split_guardrail_source_sha256",
    "dense_guardrail_source_sha256",
    "split_matrix_manifest_sha256",
    "dense_matrix_manifest_sha256",
    "measurements",
    "integrity",
}
_GUARDRAIL_MEASUREMENTS = {
    "split_on_exact_recall",
    "dense_on_exact_recall",
    "split_off_exact_recall",
    "split_off_recognition",
    "split_off_first_hop_accuracy",
    "split_gold_return_path_accuracy",
    "split_on_path_accuracy",
    "split_rule_accuracy",
    "dense_rule_accuracy",
    "split_bpb",
    "dense_bpb",
}
_BOUND_RATE_FIELDS = {
    "numerator",
    "denominator",
    "arm",
    "memory_mode",
    "control_id",
    "checkpoint_sha256",
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
}
_BOUND_SCALAR_FIELDS = {
    "value",
    "denominator",
    "arm",
    "memory_mode",
    "control_id",
    "checkpoint_sha256",
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
}
_SHA256_PATTERN = __import__("re").compile(r"[0-9a-f]{64}")


def _guard_exact_fields(value, expected: set[str], name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return value


def _guard_hash(value, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _guard_int(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} is out of range")
    return value


def _guard_number(
    value,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} is below its allowed range")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} exceeds its allowed range")
    return result


def _bound_rate(
    raw,
    name: str,
    *,
    arm: str,
    memory_mode: str,
    control_id: str,
    checkpoint_sha256: str,
    identity: Mapping,
) -> dict:
    value = _guard_exact_fields(raw, _BOUND_RATE_FIELDS, name)
    if value["arm"] != arm:
        raise ValueError(f"{name} arm mismatch")
    if value["memory_mode"] != memory_mode:
        raise ValueError(f"{name} memory_mode mismatch")
    if value["control_id"] != control_id:
        raise ValueError(f"{name} control_id mismatch")
    if _guard_hash(
        value["checkpoint_sha256"],
        f"{name} checkpoint_sha256",
    ) != checkpoint_sha256:
        raise ValueError(f"{name} checkpoint mismatch")
    for field, expected_value in identity.items():
        if value[field] != expected_value:
            raise ValueError(f"{name} {field} mismatch")
    numerator = _guard_int(value["numerator"], f"{name} numerator")
    denominator = _guard_int(
        value["denominator"],
        f"{name} denominator",
        positive=True,
    )
    if numerator > denominator:
        raise ValueError(f"{name} numerator exceeds denominator")
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _bound_scalar(
    raw,
    name: str,
    *,
    arm: str,
    memory_mode: str,
    checkpoint_sha256: str,
    identity: Mapping,
) -> dict:
    value = _guard_exact_fields(raw, _BOUND_SCALAR_FIELDS, name)
    if value["arm"] != arm:
        raise ValueError(f"{name} arm mismatch")
    if value["memory_mode"] != memory_mode:
        raise ValueError(f"{name} memory_mode mismatch")
    if value["control_id"] != "correct":
        raise ValueError(f"{name} control_id mismatch")
    if _guard_hash(
        value["checkpoint_sha256"],
        f"{name} checkpoint_sha256",
    ) != checkpoint_sha256:
        raise ValueError(f"{name} checkpoint mismatch")
    for field, expected_value in identity.items():
        if value[field] != expected_value:
            raise ValueError(f"{name} {field} mismatch")
    scalar = _guard_number(value["value"], f"{name} value", minimum=0.0)
    if scalar <= 0.0:
        raise ValueError(f"{name} value must be positive")
    return {
        "value": scalar,
        "denominator": _guard_int(
            value["denominator"],
            f"{name} denominator",
            positive=True,
        ),
    }


def _guard_check(
    check_id: str,
    value: float | bool,
    threshold: float | bool,
    comparison: str,
    *,
    reference_value: float | None = None,
    numerator: int | None = None,
    denominator: int | None = None,
) -> dict:
    if comparison == ">=":
        passed = value >= threshold
    elif comparison == "<":
        passed = value < threshold
    elif comparison == "<=":
        passed = value <= threshold
    elif comparison == "==":
        passed = value == threshold
    else:
        raise ValueError("unsupported guardrail comparison")
    return {
        "check_id": check_id,
        "value": value,
        "reference_value": reference_value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
        "numerator": numerator,
        "denominator": denominator,
    }


def evaluate_confirmatory_guardrails(inputs: Mapping) -> GuardrailReport:
    """Apply the six frozen confirmatory guards and no route diagnostics."""

    value = _guard_exact_fields(
        inputs,
        _GUARDRAIL_INPUT_FIELDS,
        "confirmatory guardrail inputs",
    )
    split_checkpoint = _guard_hash(
        value["split_checkpoint_sha256"],
        "split_checkpoint_sha256",
    )
    dense_checkpoint = _guard_hash(
        value["dense_checkpoint_sha256"],
        "dense_checkpoint_sha256",
    )
    if split_checkpoint == dense_checkpoint:
        raise ValueError("split and dense checkpoints must be distinct")
    model_id = value["model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    shared_identity = {
        "model_id": model_id,
        "seed": _guard_int(value["seed"], "seed"),
        "raw_token_count": _guard_int(
            value["raw_token_count"],
            "raw_token_count",
        ),
        "evaluator_sha256": _guard_hash(
            value["evaluator_sha256"],
            "evaluator_sha256",
        ),
        "data_sha256": _guard_hash(
            value["data_sha256"],
            "data_sha256",
        ),
        "relation_schema_sha256": _guard_hash(
            value["relation_schema_sha256"],
            "relation_schema_sha256",
        ),
        "result_schema_sha256": _guard_hash(
            value["result_schema_sha256"],
            "result_schema_sha256",
        ),
    }
    configuration_bindings = {
        name: _guard_hash(value[name], name)
        for name in (
            "split_configuration_sha256",
            "dense_configuration_sha256",
        )
    }
    provenance_bindings = {
        name: _guard_hash(value[name], name)
        for name in (
            "split_result_provenance_sha256",
            "dense_result_provenance_sha256",
            "study_provenance_sha256",
        )
    }
    if (
        provenance_bindings["split_result_provenance_sha256"]
        == provenance_bindings["dense_result_provenance_sha256"]
    ):
        raise ValueError(
            "Split and Dense result provenance hashes must be distinct"
        )
    evidence_bindings = {
        name: _guard_hash(value[name], name)
        for name in (
            "pairing_receipt_sha256",
            "split_guardrail_source_sha256",
            "dense_guardrail_source_sha256",
            "split_matrix_manifest_sha256",
            "dense_matrix_manifest_sha256",
        )
    }
    measurements = _guard_exact_fields(
        value["measurements"],
        _GUARDRAIL_MEASUREMENTS,
        "guardrail measurements",
    )
    expected = {
        "split_on_exact_recall": (
            "split", "memory_on", "correct", split_checkpoint
        ),
        "dense_on_exact_recall": (
            "dense", "memory_on", "correct", dense_checkpoint
        ),
        "split_off_exact_recall": (
            "split", "memory_off", "correct", split_checkpoint
        ),
        "split_off_recognition": (
            "split", "memory_off", "correct", split_checkpoint
        ),
        "split_off_first_hop_accuracy": (
            "split", "memory_off", "correct", split_checkpoint
        ),
        "split_gold_return_path_accuracy": (
            "split", "memory_on", "gold_returns", split_checkpoint
        ),
        "split_on_path_accuracy": (
            "split", "memory_on", "correct", split_checkpoint
        ),
        "split_rule_accuracy": (
            "split", "memory_on", "correct", split_checkpoint
        ),
        "dense_rule_accuracy": (
            "dense", "memory_on", "correct", dense_checkpoint
        ),
    }
    rates = {
        name: _bound_rate(
            measurements[name],
            name,
            arm=arm,
            memory_mode=mode,
            control_id=control,
            checkpoint_sha256=checkpoint,
            identity={
                **shared_identity,
                "configuration_sha256": configuration_bindings[
                    f"{arm}_configuration_sha256"
                ],
                "provenance_sha256": provenance_bindings[
                    f"{arm}_result_provenance_sha256"
                ],
            },
        )
        for name, (arm, mode, control, checkpoint) in expected.items()
    }
    split_bpb = _bound_scalar(
        measurements["split_bpb"],
        "split_bpb",
        arm="split",
        memory_mode="memory_off",
        checkpoint_sha256=split_checkpoint,
        identity={
            **shared_identity,
            "configuration_sha256": configuration_bindings[
                "split_configuration_sha256"
            ],
            "provenance_sha256": provenance_bindings[
                "split_result_provenance_sha256"
            ],
        },
    )
    dense_bpb = _bound_scalar(
        measurements["dense_bpb"],
        "dense_bpb",
        arm="dense",
        memory_mode="memory_off",
        checkpoint_sha256=dense_checkpoint,
        identity={
            **shared_identity,
            "configuration_sha256": configuration_bindings[
                "dense_configuration_sha256"
            ],
            "provenance_sha256": provenance_bindings[
                "dense_result_provenance_sha256"
            ],
        },
    )
    paired_denominators = (
        ("split_on_exact_recall", "dense_on_exact_recall"),
        (
            "split_gold_return_path_accuracy",
            "split_on_path_accuracy",
        ),
        ("split_rule_accuracy", "dense_rule_accuracy"),
    )
    for left, right in paired_denominators:
        if rates[left]["denominator"] != rates[right]["denominator"]:
            raise ValueError(
                f"{left}/{right} denominator mismatch"
            )
    if split_bpb["denominator"] != dense_bpb["denominator"]:
        raise ValueError("split_bpb/dense_bpb denominator mismatch")

    split_on = rates["split_on_exact_recall"]
    dense_on = rates["dense_on_exact_recall"]
    split_off = rates["split_off_exact_recall"]
    recognition = rates["split_off_recognition"]
    first_hop = rates["split_off_first_hop_accuracy"]
    gold_path = rates["split_gold_return_path_accuracy"]
    split_path = rates["split_on_path_accuracy"]
    split_rule = rates["split_rule_accuracy"]
    dense_rule = rates["dense_rule_accuracy"]
    _, recognition_hi = wilson_interval(
        recognition["numerator"],
        recognition["denominator"],
    )

    integrity = _guard_exact_fields(
        value["integrity"],
        {
            "mask_ledger",
            "corpus_pairing",
            "provenance",
            "exact_matrix",
        },
        "integrity",
    )
    for name, flag in integrity.items():
        if not isinstance(flag, bool):
            raise ValueError(f"integrity {name} must be Boolean")
    integrity_passed = all(integrity.values())

    guards = {
        "factual_job": {
            "checks": [
                _guard_check(
                    "split_on_recall_floor",
                    split_on["value"],
                    0.95,
                    ">=",
                    numerator=split_on["numerator"],
                    denominator=split_on["denominator"],
                ),
                _guard_check(
                    "split_on_recall_noninferiority",
                    split_on["value"],
                    dense_on["value"] - 0.02,
                    ">=",
                    reference_value=dense_on["value"],
                    numerator=split_on["numerator"],
                    denominator=split_on["denominator"],
                ),
            ],
        },
        "split_off_leakage": {
            "checks": [
                _guard_check(
                    "split_off_recall",
                    split_off["value"],
                    0.05,
                    "<",
                    numerator=split_off["numerator"],
                    denominator=split_off["denominator"],
                ),
                _guard_check(
                    "split_off_recognition_wilson_hi",
                    recognition_hi,
                    0.30,
                    "<",
                    numerator=recognition["numerator"],
                    denominator=recognition["denominator"],
                ),
            ],
        },
        "retrieval_procedure": {
            "checks": [
                _guard_check(
                    "split_off_first_hop",
                    first_hop["value"],
                    0.75,
                    ">=",
                    numerator=first_hop["numerator"],
                    denominator=first_hop["denominator"],
                ),
                _guard_check(
                    "gold_return_path_noninferiority",
                    gold_path["value"],
                    split_path["value"] - 0.05,
                    ">=",
                    reference_value=split_path["value"],
                    numerator=gold_path["numerator"],
                    denominator=gold_path["denominator"],
                ),
            ],
        },
        "relation_rule": {
            "checks": [
                _guard_check(
                    "rule_noninferiority",
                    split_rule["value"],
                    dense_rule["value"] - 0.02,
                    ">=",
                    reference_value=dense_rule["value"],
                    numerator=split_rule["numerator"],
                    denominator=split_rule["denominator"],
                )
            ],
        },
        "natural_text": {
            "checks": [
                _guard_check(
                    "bpb_noninferiority",
                    split_bpb["value"],
                    dense_bpb["value"] * 1.01,
                    "<=",
                    reference_value=dense_bpb["value"],
                )
            ],
        },
        "instrument_integrity": {
            "checks": [
                _guard_check(
                    "integrity_conjunction",
                    integrity_passed,
                    True,
                    "==",
                )
            ],
        },
    }
    for guard in guards.values():
        guard["passed"] = all(
            check["passed"] for check in guard["checks"]
        )
    report = {
        "record_type": "guardrail_report",
        "schema_version": 1,
        "split_checkpoint_sha256": split_checkpoint,
        "dense_checkpoint_sha256": dense_checkpoint,
        **shared_identity,
        **configuration_bindings,
        **provenance_bindings,
        **evidence_bindings,
        "guards": guards,
        "confirmatory_passed": all(
            guard["passed"] for guard in guards.values()
        ),
    }
    if not isinstance(value["model_id"], str) or not value["model_id"]:
        raise ValueError("model_id must be a nonempty string")
    return GuardrailReport.from_dict(report)
