#!/usr/bin/env python
"""Evaluate one standard-GPT run in both atomic-memory modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.records import QAItem
from evals.relational_generate import (
    GraphDecodeState,
    OverlayStore,
    decode_items,
)
from evals.relational_metrics import (
    EXPECTED_TASKS,
    assert_expected_counts,
    counterfactual_pair_accuracy,
    exact_accuracy,
    mask_ledger_guardrail,
    measure_shared_text_bpb,
    path_diagnostics,
    path_metrics,
    recognition_accuracy,
    route_guardrails,
    score_choice_loglikelihoods,
)
from evals.scorers import normalize_answer
from organizer.graph_store import AtomicGraphStore
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _group_rows_by_task(rows) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for row in rows:
        task = str(row["task"] if isinstance(row, dict) else row.task)
        grouped.setdefault(task, []).append(row)
    return grouped


def _load_eval_items(data_dir: Path, expected_pairs: int) -> list[QAItem]:
    originals = [
        QAItem(**row)
        for row in _read_jsonl(data_dir / "eval" / "original.jsonl")
    ]
    counterfactuals = [
        QAItem(**row)
        for row in _read_jsonl(
            data_dir / "eval" / "counterfactual.jsonl"
        )
    ]
    items = originals + counterfactuals
    rows_by_task = _group_rows_by_task(items)
    assert_expected_counts(rows_by_task, expected_pairs)
    return items


def _item_meta(item) -> dict:
    meta = item["meta"] if isinstance(item, dict) else item.meta
    if not isinstance(meta, dict):
        raise ValueError("eval item meta must be a mapping")
    return meta


def store_for_item(
    base: AtomicGraphStore,
    item,
    *,
    memory_on: bool,
) -> AtomicGraphStore | OverlayStore | None:
    """Return the only evaluator toggle: base/overlay store, or ``None``."""

    if not isinstance(memory_on, bool):
        raise TypeError("memory_on must be Boolean")
    meta = _item_meta(item)
    variant = meta["variant"]
    changed = meta["changed_row"]
    if variant == "original":
        if changed is not None:
            raise ValueError("original eval items cannot contain a changed row")
        selected = base
    elif variant == "counterfactual":
        if not isinstance(changed, dict):
            raise ValueError(
                "counterfactual eval items require one changed row"
            )
        selected = OverlayStore(base, GraphRow.from_json(changed))
    else:
        raise ValueError(f"unexpected eval variant: {variant}")
    return selected if memory_on else None


def _action_json(action: GraphAction) -> list:
    value = [
        action.source_slot,
        action.relation_id,
        action.direction,
        action.read,
        action.halt,
    ]
    if action.page:
        value.append(action.page)
    return value


def _gold_actions(item) -> list[GraphAction]:
    meta = _item_meta(item)
    raw_actions = meta["gold_actions"]
    expected_slots = int(meta.get("action_slots", len(raw_actions)))
    if (
        not isinstance(raw_actions, list)
        or expected_slots not in (6, 12)
        or len(raw_actions) != expected_slots
    ):
        raise ValueError("gold_actions must contain exactly 6 or 12 actions")
    actions = []
    legacy_fields = {
        "source_slot",
        "relation_id",
        "direction",
        "read",
        "halt",
    }
    for raw in raw_actions:
        if not isinstance(raw, dict) or set(raw) not in (
            legacy_fields,
            legacy_fields | {"page"},
        ):
            raise ValueError("gold action fields do not match the contract")
        actions.append(GraphAction(**raw))
    halt_positions = [
        index for index, action in enumerate(actions) if action.halt
    ]
    if len(halt_positions) != 1:
        raise ValueError("gold_actions require exactly one HALT")
    halt = halt_positions[0]
    if not all(action.read for action in actions[:halt]):
        raise ValueError("gold actions before HALT must be reads")
    if any(
        action.read or action.halt for action in actions[halt + 1 :]
    ):
        raise ValueError("gold actions after HALT must be NOOP")
    addresses = meta["gold_addresses"]
    read_actions = [action for action in actions if action.read]
    if len(read_actions) != len(addresses):
        raise ValueError("gold action/address counts differ")
    if any(
        action.relation_id != str(address[1])
        or action.direction != str(address[2])
        or action.page != (int(address[3]) if len(address) == 4 else 0)
        for action, address in zip(read_actions, addresses)
    ):
        raise ValueError("gold actions do not match gold addresses")
    return actions


def _states_to_rows(items, states: list[GraphDecodeState]) -> list[dict]:
    materialized = list(items)
    if len(materialized) != len(states):
        raise ValueError("every eval item requires exactly one decoded state")
    rows = []
    for item, state in zip(materialized, states):
        meta = _item_meta(item)
        expected_slots = int(meta.get("action_slots", len(state.actions)))
        if (
            expected_slots not in (6, 12)
            or len(state.actions) != expected_slots
            or len(state.rows) != expected_slots
            or len(state.provisional_answers) != expected_slots
        ):
            raise ValueError("decoded state does not match its action-slot contract")
        gold_all_actions = _gold_actions(item)
        gold_actions = [
            action for action in gold_all_actions if action.read
        ]
        gold_addresses = [
            GraphAddress(
                source if isinstance(source, str) else int(source),
                str(address[1]),
                address[2],
                int(address[3]) if len(address) == 4 else 0,
            )
            for address in meta["gold_addresses"]
            for source in [address[0]]
        ]
        read_pairs = [
            (action, row)
            for action, row in zip(state.actions, state.rows)
            if action.read
        ]
        correct_referents = []
        for index, address in enumerate(gold_addresses):
            returned = (
                read_pairs[index][1] if index < len(read_pairs) else None
            )
            correct_referents.append(
                returned is not None and returned.address == address
            )
        prediction = state.provisional_answers[-1]
        answer = item["answer"] if isinstance(item, dict) else item.answer
        qid = item["qid"] if isinstance(item, dict) else item.qid
        task = item["task"] if isinstance(item, dict) else item.task
        predicted_reads = [action for action, _ in read_pairs]
        rows.append(
            {
                "qid": qid,
                "task": task,
                "pair_id": str(meta["pair_id"]),
                "variant": str(meta["variant"]),
                "correct": (
                    normalize_answer(prediction)
                    == normalize_answer(str(answer))
                ),
                "pred": prediction,
                "answer": answer,
                "actions": [
                    _action_json(action) for action in predicted_reads
                ],
                "all_actions": [
                    _action_json(action) for action in state.actions
                ],
                "gold_actions": [
                    _action_json(action) for action in gold_actions
                ],
                "gold_all_actions": [
                    _action_json(action) for action in gold_all_actions
                ],
                "correct_referents": correct_referents,
                "misses": state.misses,
                # The constrained action grammar cannot emit malformed frames.
                "malformed": 0,
                "excess_reads": max(
                    0, len(predicted_reads) - len(gold_actions)
                ),
                "halt_step": state.halt_step,
                "n_steps": len(state.actions),
                "meta": meta,
            }
        )
    return rows


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _summary(rows: list[dict], expected_pairs: int, memory: str) -> dict:
    rows_by_task = _group_rows_by_task(rows)
    assert_expected_counts(rows_by_task, expected_pairs)
    task_summary = {}
    for task in EXPECTED_TASKS:
        task_rows = rows_by_task[task]
        task_summary[task] = {
            "counterfactual_pair_accuracy": counterfactual_pair_accuracy(
                task_rows, expected_pairs=expected_pairs
            ),
            "path": path_metrics(task_rows),
            "path_diagnostics": path_diagnostics(task_rows),
            "n_rows": len(task_rows),
            "n_pairs": expected_pairs,
        }
    composite = sum(
        task_summary[task]["counterfactual_pair_accuracy"]
        for task in EXPECTED_TASKS
    ) / len(EXPECTED_TASKS)
    return {
        "memory": memory,
        "tasks": task_summary,
        "primary_composite": composite,
        "n_rows": len(rows),
        "n_pairs_per_task": expected_pairs,
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def _mask_audit_from_committed_data(
    data_dir: Path,
    condition: str,
) -> dict:
    ledger = _read_jsonl(data_dir / "mask-ledger.jsonl")
    report = _read_json(data_dir / "report.json")
    expected = Counter(
        (row["start"], row["end"], row["fact_id"])
        for row in ledger
        if row["condition"] == "expected_split"
    )
    split = Counter(
        (row["start"], row["end"], row["fact_id"])
        for row in ledger
        if row["condition"] == "split"
    )
    if split - expected:
        raise ValueError("split ledger contains unexpected payload ranges")
    selected_condition = [
        row for row in ledger if row["condition"] == condition
    ]
    protected_roles = {
        "rule",
        "action",
        "provisional_answer",
        "final_answer",
    }
    if condition == "split":
        unmasked_external = sum((expected - split).values())
    else:
        unmasked_external = sum(expected.values())
    return {
        "unmasked_external_payloads": unmasked_external,
        "external_payload_occurrences": sum(expected.values()),
        "masked_rule_action_answer_targets": sum(
            int(row["length"])
            for row in selected_condition
            if row["role"] in protected_roles
        ),
        "rule_action_answer_targets": int(
            report["masks"]["protected_target_tokens"]
        ),
    }


def produce_guardrail_measurements(
    model,
    tok,
    data_dir: Path | str,
    *,
    condition: str,
    device,
    batch_size: int,
) -> dict:
    data_dir = Path(data_dir)
    if condition not in ("dense", "split", "random"):
        raise ValueError(f"unexpected training condition: {condition}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    eval_manifest = _read_json(data_dir / "eval-manifest.json")
    expected_items = int(eval_manifest["guardrail_items"])
    expected_shared = int(eval_manifest["shared_text_items"])
    recognition_items = _read_jsonl(
        data_dir / "eval" / "recognition.jsonl"
    )
    factual_items = [
        QAItem(**row)
        for row in _read_jsonl(data_dir / "eval" / "factual.jsonl")
    ]
    internal_items = _read_jsonl(data_dir / "eval" / "internal.jsonl")
    shared_rows = _read_jsonl(
        data_dir / "eval" / "shared_text.jsonl"
    )
    for name, values, expected in (
        ("recognition", recognition_items, expected_items),
        ("factual", factual_items, expected_items),
        ("internal", internal_items, expected_items),
        ("shared_text", shared_rows, expected_shared),
    ):
        if len(values) != expected:
            raise ValueError(
                f"{name}: expected {expected} items, got {len(values)}"
            )

    choice_score_cache = {}

    def score_choices(prompt, choices):
        key = (prompt, tuple(choices))
        if key not in choice_score_cache:
            choice_score_cache[key] = score_choice_loglikelihoods(
                model,
                tok,
                prompt,
                choices,
                device=device,
            )
        return choice_score_cache[key]

    recognition = recognition_accuracy(
        score_choices,
        recognition_items,
        expected_count=expected_items,
    )
    internal = recognition_accuracy(
        score_choices,
        internal_items,
        expected_count=expected_items,
    )
    internal["per_kind"] = {
        kind: recognition_accuracy(
            score_choices,
            [item for item in internal_items if item["kind"] == kind],
        )
        for kind in ("rule", "central_fact")
    }
    language = measure_shared_text_bpb(
        model,
        tok,
        [row["text"] for row in shared_rows],
        device=device,
    )

    factual_store = AtomicGraphStore.load(
        data_dir / "eval" / "factual-graph.jsonl"
    )
    factual_measurements = {}
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            model,
            tok,
            factual_items,
            lambda item, enabled=memory_on: store_for_item(
                factual_store,
                item,
                memory_on=enabled,
            ),
            device=device,
            batch_size=batch_size,
        )
        factual_measurements[memory] = exact_accuracy(
            _states_to_rows(factual_items, states),
            expected_count=expected_items,
        )

    route = _read_json(data_dir / "eval" / "route-audit.json")
    mask_audit = _mask_audit_from_committed_data(data_dir, condition)
    return {
        "within_run_guardrails": {
            "route": route_guardrails(route),
            "mask": mask_ledger_guardrail(
                mask_audit,
                condition=condition,
            ),
        },
        "recognition_store_off": recognition,
        "factual_recall": factual_measurements,
        "internal_accuracy": internal,
        "language": language,
    }


def _validate_guardrail_schema(value: dict) -> None:
    required = {
        "within_run_guardrails",
        "recognition_store_off",
        "factual_recall",
        "internal_accuracy",
        "language",
    }
    if set(value) != required:
        raise ValueError(
            "guardrail measurement keys mismatch; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )


def _resolve_data_dir(cfg: dict, override: str | None) -> Path:
    if override is not None:
        return Path(override)
    if "data_dir" in cfg:
        return Path(cfg["data_dir"])
    if "data_rel" in cfg:
        root = os.environ.get("DATA_ROOT")
        if root is None:
            raise ValueError("DATA_ROOT is required when config uses data_rel")
        return Path(root) / cfg["data_rel"]
    raise KeyError("run config requires data_dir or data_rel")


def _load_model(
    run: Path,
    checkpoint: str,
    device: str,
) -> tuple[GPT, dict]:
    config_path = run / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    model_value = cfg["model"]
    if isinstance(model_value, str):
        if model_value not in PRESETS:
            raise ValueError(f"unknown model preset: {model_value}")
        model_cfg = replace(PRESETS[model_value])
    elif isinstance(model_value, dict):
        model_cfg = GPTConfig(**model_value)
    else:
        raise ValueError("model config must be a preset name or mapping")
    if "ctx" in cfg:
        model_cfg.ctx = int(cfg["ctx"])
    model = GPT(model_cfg)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=10_000,
        help="frozen value is 10000; smaller values are for smoke tests",
    )
    args = parser.parse_args()

    if args.expected_pairs <= 0:
        raise ValueError("expected-pairs must be positive")
    run = Path(args.run)
    device = pick_device(args.device)
    model, cfg = _load_model(run, args.checkpoint, device)
    data_dir = _resolve_data_dir(cfg, args.data_dir)
    tok = get_tok()
    items = _load_eval_items(data_dir, args.expected_pairs)
    base_store = AtomicGraphStore.load(data_dir / "eval" / "graph.jsonl")
    measurements = produce_guardrail_measurements(
        model,
        tok=tok,
        data_dir=data_dir,
        condition=cfg["condition"],
        device=device,
        batch_size=args.batch_size,
    )
    _validate_guardrail_schema(measurements)

    output = run / "evals"
    output.mkdir(parents=True, exist_ok=True)
    mode_summaries = {}
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            model,
            tok,
            items,
            lambda item, enabled=memory_on: store_for_item(
                base_store, item, memory_on=enabled
            ),
            device=device,
            batch_size=args.batch_size,
        )
        rows = _states_to_rows(items, states)
        mode_dir = output / f"memory_{memory}"
        _write_jsonl(mode_dir / "rows.jsonl", rows)
        summary = _summary(rows, args.expected_pairs, memory)
        (mode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        mode_summaries[memory] = summary

    (output / "guardrails.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n"
    )
    combined = {
        "condition": cfg["condition"],
        "modes": mode_summaries,
        "guardrails": measurements,
    }
    (output / "relational_summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
