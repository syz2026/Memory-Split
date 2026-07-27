from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence

from evals.relational_contracts import (
    CheckpointSummary,
    EvalRow,
    cluster_id_for,
    rows_sha256,
)
from evals.relational_metrics import EXPECTED_TASKS, compute_checkpoint_metrics


PROTECTED_SEEDS = (1001, 1002, 1003, 1004, 1005)
DEVELOPMENT_SEEDS = (201, 202, 203)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_rows(
    label: str,
    arm: str,
    *,
    seeds: Sequence[int] = PROTECTED_SEEDS,
    model_id: str = "d160m",
    namespace: str = "protected",
    pair_counts: Mapping[str, int] | None = None,
    success: Callable[[str, int, str, int], bool] | None = None,
    world_count: int = 2,
    path_count: int = 2,
    template_count: int = 2,
) -> tuple[EvalRow, ...]:
    counts = (
        {task: 2 for task in EXPECTED_TASKS}
        if pair_counts is None
        else dict(pair_counts)
    )
    if set(counts) != set(EXPECTED_TASKS):
        raise ValueError("fixture pair counts must cover every task")
    scorer = success or (lambda _label, _seed, _task, _index: True)
    rows: list[EvalRow] = []
    for seed in seeds:
        for task_index, task in enumerate(EXPECTED_TASKS):
            for index in range(counts[task]):
                world_slot = index % world_count
                path_slot = (index // world_count) % path_count
                template_slot = (
                    index // max(1, world_count * path_count)
                ) % template_count
                world_id = (
                    (10_000 if namespace.startswith("protected") else 20_000)
                    + task_index * 1_000
                    + world_slot
                )
                relation_path_hash = _digest(
                    f"{namespace}:{task}:path:{path_slot}"
                )
                template_id = (
                    f"{namespace}:{task}:template:{template_slot}"
                )
                pair_id = (
                    f"{namespace}:{seed}:{task}:w{world_slot}:"
                    f"p{path_slot}:t{template_slot}:i{index}"
                )
                pair_correct = bool(scorer(label, seed, task, index))
                for variant in ("original", "counterfactual"):
                    suffix = "o" if variant == "original" else "c"
                    answer = "r1"
                    prediction = answer if pair_correct else "r0"
                    action = [0, "r0", "out", True, False]
                    halt = [0, "r0", "out", False, True]
                    noop = [0, "r0", "out", False, False]
                    value = {
                        "record_type": "eval_row",
                        "schema_version": 1,
                        "qid": f"{pair_id}-{suffix}",
                        "pair_id": pair_id,
                        "variant": variant,
                        "task": task,
                        "world_id": world_id,
                        "provenance_id": f"{namespace}:world:{world_id}",
                        "relation_path_hash": relation_path_hash,
                        "template_id": template_id,
                        "composition_split": "seen",
                        "hop": 1,
                        "seed": seed,
                        "model_id": model_id,
                        "arm": arm,
                        "checkpoint_sha256": _digest(
                            f"checkpoint:{label}:{seed}"
                        ),
                        "raw_token_count": 1_000_000,
                        "memory_mode": "memory_on",
                        "control_id": "correct",
                        "evaluator_sha256": _digest("evaluator"),
                        "data_sha256": _digest("protected-eval-data"),
                        "relation_schema_sha256": _digest("relation-schema"),
                        "configuration_sha256": _digest(
                            f"configuration:{label}:{seed}"
                        ),
                        "result_schema_sha256": _digest("result-schema"),
                        "provenance_sha256": _digest(
                            f"result-provenance:{label}:{seed}"
                        ),
                        "cluster_id": cluster_id_for(
                            seed=seed,
                            world_id=world_id,
                            relation_path_hash=relation_path_hash,
                            template_id=template_id,
                        ),
                        "prediction": prediction,
                        "answer": answer,
                        "correct": pair_correct,
                        "prediction_source": "model",
                        "all_actions": [action, halt, noop, noop, noop, noop],
                        "gold_all_actions": [
                            action,
                            halt,
                            noop,
                            noop,
                            noop,
                            noop,
                        ],
                        "returned_addresses": [
                            [world_id, "r0", "out"],
                            None,
                            None,
                            None,
                            None,
                            None,
                        ],
                        "gold_addresses": [[world_id, "r0", "out"]],
                        "correct_referents": [True],
                        "misses": 0,
                        "malformed": 0,
                        "abstained": False,
                        "excess_reads": 0,
                        "halt_step": 2,
                        "answer_logits": [[-0.1] for _ in range(6)],
                        "lookup_latency_ns": 11,
                        "lookup_count": 1,
                        "store_rows": 100,
                        "store_bytes": 1_000,
                        "control_seed": 77,
                        "transformation_id": None,
                        "source_store_sha256": None,
                        "transformed_store_sha256": None,
                        "transformation_metadata_sha256": None,
                        "changed_addresses": [],
                        "oracle_before": answer,
                        "oracle_after": answer,
                        "oracle_effect": "unchanged",
                        "edit_locality_correct": None,
                    }
                    rows.append(EvalRow.from_dict(value))
    return tuple(rows)


def replace_row(row: EvalRow, **changes) -> EvalRow:
    value = row.to_dict()
    value.update(changes)
    if {
        "seed",
        "world_id",
        "relation_path_hash",
        "template_id",
    } & changes.keys():
        value["cluster_id"] = cluster_id_for(
            seed=value["seed"],
            world_id=value["world_id"],
            relation_path_hash=value["relation_path_hash"],
            template_id=value["template_id"],
        )
    return EvalRow.from_dict(value)


def make_summary(rows: Sequence[EvalRow]) -> CheckpointSummary:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("summary fixture requires rows")
    first = materialized[0]
    return CheckpointSummary.from_dict(
        {
            "record_type": "checkpoint_summary",
            "schema_version": 1,
            "checkpoint_sha256": first.checkpoint_sha256,
            "model_id": first.model_id,
            "arm": first.arm,
            "seed": first.seed,
            "raw_token_count": first.raw_token_count,
            "memory_mode": first.memory_mode,
            "control_id": first.control_id,
            "evaluator_sha256": first.evaluator_sha256,
            "data_sha256": first.data_sha256,
            "relation_schema_sha256": first.relation_schema_sha256,
            "configuration_sha256": first.configuration_sha256,
            "result_schema_sha256": first.result_schema_sha256,
            "provenance_sha256": first.provenance_sha256,
            "rows_sha256": rows_sha256(materialized),
            "n_rows": len(materialized),
            "n_pairs": len({row.pair_id for row in materialized}),
            "metrics": compute_checkpoint_metrics(materialized),
        }
    )


def rows_for(
    rows: Iterable[EvalRow],
    *,
    seed: int | None = None,
    task: str | None = None,
) -> tuple[EvalRow, ...]:
    return tuple(
        row
        for row in rows
        if (seed is None or row.seed == seed)
        and (task is None or row.task == task)
    )
