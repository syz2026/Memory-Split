#!/usr/bin/env python
"""Evaluate QA-held-out KQA facts for one continued dense or split run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

from corpusgen.kqa_pro import LookupFact, make_recall_probe
from corpusgen.records import QAItem
from evals.kqa import (
    fact_availability,
    save_jsonl,
    score_kqa_recall,
    score_kqa_transfer,
    summarize_transfer_rows,
)
from organizer.store import Organizer, normalize
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def load_items(path: Path) -> list[QAItem]:
    with open(path) as handle:
        return [QAItem(**json.loads(line)) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    if sidecar.exists():
        digest = sidecar.read_text().strip()
        if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            return digest
        raise ValueError(f"invalid checkpoint SHA-256 sidecar: {sidecar}")
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model(run_dir: Path, checkpoint: str | None, device: str) -> GPT:
    with open(run_dir / "config.yaml") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = (
        PRESETS[cfg["model"]]
        if isinstance(cfg["model"], str)
        else GPTConfig(**cfg["model"])
    )
    if "ctx" in cfg:
        model_cfg.ctx = cfg["ctx"]
    model = GPT(model_cfg)
    default_checkpoint = "model.pt" if (run_dir / "model.pt").exists() else "ckpt.pt"
    path = run_dir / (checkpoint or default_checkpoint)
    try:
        state = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _budget_tier(needed: int, available: int) -> int:
    """Round a per-item need up to a coarse tier so batches stay large."""
    return min(available, max(64, -(-needed // 64) * 64))


def score_recall_in_budget_groups(
    model,
    tok,
    probes: list[QAItem],
    mode: str,
    organizer,
    device: str,
    *,
    context: int,
    batch_size: int,
    floor_max_new: int = 0,
) -> tuple[list[dict], dict]:
    """Recall probes sized individually rather than by the longest fact value."""
    groups: dict[int, list[QAItem]] = defaultdict(list)
    for probe in probes:
        available_new = context - len(tok.encode(probe.prompt))
        needed = max(
            floor_max_new,
            2
            + len(tok.encode(probe.meta["query"]))
            + len(tok.encode(" " + probe.answer))
            + 9,
        )
        groups[_budget_tier(needed, available_new)].append(probe)

    rows: list[dict] = []
    total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    for max_new, group in sorted(groups.items(), reverse=True):
        group_rows, group_summary = score_kqa_recall(
            model,
            tok,
            group,
            mode,
            organizer,
            device,
            max_new=max_new,
            batch_size=batch_size,
        )
        rows.extend(group_rows)
        for key in total:
            total[key] += group_summary["lookup_stats"][key]

    order = {probe.qid: index for index, probe in enumerate(probes)}
    rows.sort(key=lambda row: order[row["qid"]])
    summary = {
        "mode": mode,
        "accuracy": (
            sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
        ),
        "n": len(rows),
        "lookup_stats": total,
        "generation_budget_min": min(groups) if groups else 0,
        "generation_budget_max": max(groups) if groups else 0,
    }
    return rows, summary


def score_in_budget_groups(
    model,
    tok,
    items: list[QAItem],
    device: str,
    fact_availability_map: dict[str, bool],
    *,
    mode: str,
    organizer,
    need_for,
    context: int,
    batch_size: int,
    floor_max_new: int = 0,
    summary_mode: str | None = None,
) -> tuple[list[dict], dict]:
    """Score with a per-item generation budget instead of one global maximum.

    A single global budget is the maximum over every item, so short questions
    inherit the longest item's allowance and keep decoding long after their
    answer. That wastes most of the evaluation's runtime and invites the model
    to emit further invented question/answer pairs.
    """
    groups: dict[int, list[QAItem]] = defaultdict(list)
    for item in items:
        prompt_tokens = len(tok.encode(item.prompt))
        available_new = context - prompt_tokens
        required_new = len(tok.encode(f"Answer: {item.answer}")) + 2
        if available_new < required_new:
            raise SystemExit(
                f"prompt plus the labeled answer cannot fit context {context} "
                f"for {item.qid}: needs {prompt_tokens + required_new}"
            )
        needed = max(need_for(item), floor_max_new, required_new)
        groups[_budget_tier(needed, available_new)].append(item)

    rows: list[dict] = []
    total_stats = {
        "n_lookups": 0,
        "n_hits": 0,
        "n_misses": 0,
        "n_malformed": 0,
    }
    for max_new, group in sorted(groups.items(), reverse=True):
        group_rows, group_summary = score_kqa_transfer(
            model,
            tok,
            group,
            mode=mode,
            organizer=organizer,
            device=device,
            fact_availability_map=fact_availability_map,
            max_new=max_new,
            batch_size=batch_size,
        )
        rows.extend(group_rows)
        for key in total_stats:
            total_stats[key] += group_summary["lookup_stats"][key]

    order = {item.qid: index for index, item in enumerate(items)}
    rows.sort(key=lambda row: order[row["qid"]])
    summary = summarize_transfer_rows(rows)
    summary.update(
        {
            "mode": summary_mode or mode,
            "lookup_stats": total_stats,
            "generation_budget_min": min(groups) if groups else 0,
            "generation_budget_max": max(groups) if groups else 0,
            "context_constrained_items": sum(
                len(group)
                for budget, group in groups.items()
                for item in group
                if budget
                < max(need_for(item), floor_max_new)
            ),
        }
    )
    return rows, summary


def build_wrong_store(
    facts_by_key: dict[str, dict],
    required: set[str],
) -> tuple[Organizer, dict[str, str], dict[str, str]]:
    """Replace required values with deterministic relation-matched alternatives."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in sorted(required):
        row = facts_by_key[key]
        grouped[(row["kind"], row["relation"])].append(key)

    corruption: dict[str, str] = {}
    strategy: dict[str, str] = {}
    for keys in grouped.values():
        values = sorted({facts_by_key[key]["value"] for key in keys})
        for key in keys:
            original = facts_by_key[key]["value"]
            alternatives = [value for value in values if value != original]
            if alternatives:
                corruption[key] = alternatives[0]
                strategy[key] = "observed_relation_matched"
                continue
            translated = []
            for char in original:
                if "0" <= char <= "9":
                    translated.append(str((int(char) + 1) % 10))
                elif "a" <= char <= "z":
                    translated.append(chr((ord(char) - ord("a") + 1) % 26 + ord("a")))
                elif "A" <= char <= "Z":
                    translated.append(chr((ord(char) - ord("A") + 1) % 26 + ord("A")))
                else:
                    translated.append(char)
            changed = "".join(translated)
            corruption[key] = (
                changed if changed != original else "<counterfactual value>"
            )
            strategy[key] = "synthetic_format_preserving"

    wrong_store = Organizer()
    for key, row in facts_by_key.items():
        wrong_store.add(
            row["name"],
            row["relation"],
            corruption.get(key, row["value"]),
        )
    return wrong_store, corruption, strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt")
    parser.add_argument("--arm", choices=["dense", "split"])
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--eval-set", choices=["transfer", "dev"], default="transfer"
    )
    parser.add_argument("--limit-qa", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help="floor on generated tokens per item; 0 sizes each item from its "
        "own support lookups and answer",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run)
    with open(run_dir / "config.yaml") as handle:
        cfg = yaml.safe_load(handle)
    arm = args.arm or cfg["arm"]
    if arm not in {"dense", "split"}:
        raise SystemExit(f"KQA evaluation requires dense/split arm, got {arm!r}")
    data_dir = Path(args.data_dir or cfg["data_dir"])
    report_path = data_dir / "continuation_report.json"
    evaluation_data_fingerprint = cfg.get("data_fingerprint")
    if report_path.exists():
        evaluation_data_fingerprint = json.loads(
            report_path.read_text()
        ).get("data_fingerprint", evaluation_data_fingerprint)
    device = pick_device(args.device)
    checkpoint_name = args.ckpt or (
        "model.pt" if (run_dir / "model.pt").exists() else "ckpt.pt"
    )
    checkpoint_path = run_dir / checkpoint_name
    checkpoint_digest = file_sha256(checkpoint_path)
    model = load_model(run_dir, checkpoint_name, device)
    tok = get_tok()
    organizer = Organizer.load(data_dir / "organizer.jsonl")

    with open(data_dir / "facts.jsonl") as handle:
        facts_by_key = {
            normalize(row["key"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }
    qa_items = load_items(
        data_dir
        / ("eval_transfer.jsonl" if args.eval_set == "transfer" else "eval_dev.jsonl")
    )
    if args.limit_qa:
        qa_items = qa_items[: args.limit_qa]
    required = {
        normalize(key)
        for item in qa_items
        for key in item.meta["support_keys"]
    }
    if args.eval_set == "transfer":
        all_probes = load_items(data_dir / "recall_transfer.jsonl")
    else:
        all_probes = [
            make_recall_probe(
                LookupFact(
                    key=row["key"],
                    name=row["name"],
                    relation=row["relation"],
                    value=row["value"],
                    kind=row["kind"],
                    value_count=row.get("value_count", 1),
                    raw_ids=tuple(row.get("raw_ids", ())),
                ),
                "seen",
            )
            for key, row in facts_by_key.items()
            if key in required
        ]
    probes = [
        probe
        for probe in all_probes
        if normalize(probe.meta["fact_id"]) in required
    ]
    missing_probes = required - {
        normalize(probe.meta["fact_id"]) for probe in probes
    }
    if missing_probes:
        raise SystemExit(
            f"{len(missing_probes)} required support facts have no recall probe"
        )
    probe_by_fact = {
        normalize(probe.meta["fact_id"]): probe for probe in probes
    }
    def lookup_need(item: QAItem) -> int:
        """Room for every support lookup this item may issue, plus its answer."""
        return (
            sum(
                1
                + len(tok.encode(probe_by_fact[normalize(key)].meta["query"]))
                + 1
                + len(tok.encode(" " + probe_by_fact[normalize(key)].answer))
                + 1
                for key in item.meta["support_keys"]
            )
            + len(tok.encode(f"Answer: {item.answer}"))
            + 32
        )

    def answer_need(item: QAItem) -> int:
        return len(tok.encode(f"Answer: {item.answer}")) + 32

    context = model.cfg.ctx
    too_long = [
        item.qid
        for item in qa_items
        if len(tok.encode(item.prompt)) + lookup_need(item) > context
    ]
    if too_long:
        raise SystemExit(
            f"{len(too_long)} QA items cannot fit their support lookups and answer "
            f"within model context {context}; first: {too_long[0]}"
        )

    default_out = "kqa_evals" if args.eval_set == "transfer" else "kqa_dev"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    recall_mode = "closed" if arm == "dense" else "on"
    recall_rows, recall_summary = score_recall_in_budget_groups(
        model,
        tok,
        probes,
        recall_mode,
        organizer if arm == "split" else None,
        device,
        context=context,
        batch_size=args.batch_size,
        floor_max_new=args.max_new,
    )
    save_jsonl(recall_rows, out_dir / f"recall_{recall_mode}.jsonl")
    availability = fact_availability(recall_rows)

    transfer_rows, transfer_summary = score_in_budget_groups(
        model,
        tok,
        qa_items,
        device,
        availability,
        mode=arm,
        organizer=organizer if arm == "split" else None,
        need_for=lookup_need,
        context=context,
        batch_size=args.batch_size,
        floor_max_new=args.max_new,
    )
    save_jsonl(transfer_rows, out_dir / "transfer_qa.jsonl")

    oracle_items = []
    for item in qa_items:
        evidence = "\n".join(
            f"Knowledge: {facts_by_key[normalize(key)]['name']}'s "
            f"{facts_by_key[normalize(key)]['relation']} is "
            f"{facts_by_key[normalize(key)]['value']}."
            for key in item.meta["support_keys"]
        )
        oracle_items.append(
            QAItem(
                qid=item.qid,
                task=item.task,
                prompt=f"Evidence:\n{evidence}\n{item.prompt}",
                answer=item.answer,
                meta=item.meta,
            )
        )
    oracle_rows, oracle_summary = score_in_budget_groups(
        model,
        tok,
        oracle_items,
        device,
        {key: True for key in required},
        mode="dense",
        organizer=None,
        need_for=answer_need,
        context=context,
        batch_size=args.batch_size,
        floor_max_new=args.max_new,
        summary_mode="oracle_context",
    )
    save_jsonl(oracle_rows, out_dir / "transfer_qa_oracle_context.jsonl")

    summary = {
        "run": run_dir.name,
        "arm": arm,
        "eval_set": args.eval_set,
        "checkpoint": args.ckpt
        or ("model.pt" if (run_dir / "model.pt").exists() else "ckpt.pt"),
        "checkpoint_sha256": checkpoint_digest,
        "data_fingerprint": evaluation_data_fingerprint,
        "source_checkpoint_sha256": cfg.get("source_checkpoint_sha256"),
        "source_pair_id": cfg.get("source_pair_id"),
        "code_commit": cfg.get("code_commit"),
        "required_transfer_facts": len(required),
        "recall": recall_summary,
        "transfer_qa": transfer_summary,
        "transfer_qa_oracle_context": oracle_summary,
    }
    if arm == "split":
        unplugged_rows, unplugged_summary = score_recall_in_budget_groups(
            model,
            tok,
            probes,
            "off",
            None,
            device,
            context=context,
            batch_size=args.batch_size,
            floor_max_new=args.max_new,
        )
        save_jsonl(unplugged_rows, out_dir / "recall_off.jsonl")
        summary["recall_off"] = unplugged_summary
        unplugged_transfer_rows, unplugged_transfer_summary = score_in_budget_groups(
            model,
            tok,
            qa_items,
            device,
            fact_availability(unplugged_rows),
            mode="split_off",
            organizer=None,
            need_for=answer_need,
            context=context,
            batch_size=args.batch_size,
            floor_max_new=args.max_new,
        )
        save_jsonl(
            unplugged_transfer_rows, out_dir / "transfer_qa_off.jsonl"
        )
        summary["transfer_qa_off"] = unplugged_transfer_summary
        if args.eval_set == "transfer":
            wrong_store, corruption, strategy = build_wrong_store(
                facts_by_key, required
            )
            wrong_rows, wrong_summary = score_in_budget_groups(
                model,
                tok,
                qa_items,
                device,
                availability,
                mode="split_wrong",
                organizer=wrong_store,
                need_for=lookup_need,
                context=context,
                batch_size=args.batch_size,
                floor_max_new=args.max_new,
            )
            save_jsonl(
                wrong_rows, out_dir / "transfer_qa_wrong_store.jsonl"
            )
            with open(out_dir / "wrong_store_values.jsonl", "w") as handle:
                for key in sorted(corruption):
                    handle.write(
                        json.dumps(
                            {
                                "fact_id": key,
                                "correct": facts_by_key[key]["value"],
                                "wrong": corruption[key],
                                "strategy": strategy[key],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            wrong_summary["changed_required_facts"] = len(corruption)
            wrong_summary["corruption_strategy_counts"] = dict(
                Counter(strategy.values())
            )
            summary["transfer_qa_wrong_store"] = wrong_summary
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
