"""Evaluation for KQA fact transfer, access, and conditional utilization."""

from __future__ import annotations

import json
from pathlib import Path

from evals.generate import generate_batch_with_events
from evals.scorers import normalize_answer, parse_answer
from organizer.store import normalize


def _add_stats(total: dict, update: dict) -> None:
    for key in total:
        total[key] += update[key]


def _completed_hit_queries(events: list[dict]) -> set[str]:
    return {
        normalize(event["query"])
        for event in events
        if event.get("hit")
        and event.get("completed")
        and event.get("before_answer")
        and not event.get("malformed")
    }


def _prefix_recall_match(generated: str, answer: str) -> bool:
    candidate = normalize_answer(generated.splitlines()[0] if generated else "")
    expected = normalize_answer(answer)
    if candidate == expected:
        return True
    if not candidate.startswith(expected):
        return False
    suffix = candidate[len(expected) :]
    return (
        bool(suffix)
        and suffix[0] in ",;:!?"
        and (len(suffix) == 1 or suffix[1].isspace())
    )


def score_kqa_recall(
    model,
    tok,
    probes,
    mode: str,
    organizer,
    device,
    *,
    max_new: int = 256,
    batch_size: int = 32,
) -> tuple[list[dict], dict]:
    if mode not in {"closed", "on", "off"}:
        raise ValueError("mode must be closed, on, or off")
    if mode == "on" and organizer is None:
        raise ValueError('mode="on" requires an organizer')
    store = organizer if mode == "on" else None
    rows: list[dict] = []
    total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    for lo in range(0, len(probes), batch_size):
        chunk = probes[lo : lo + batch_size]
        texts, stats, event_rows = generate_batch_with_events(
            model,
            tok,
            [probe.prompt for probe in chunk],
            max_new,
            store,
            device,
        )
        _add_stats(total, stats)
        for probe, text, events in zip(chunk, texts, event_rows):
            expected_key = normalize(probe.meta["fact_id"])
            completed_queries = _completed_hit_queries(events)
            correct = (
                expected_key in completed_queries
                if mode == "on"
                else _prefix_recall_match(text, probe.answer)
            )
            rows.append(
                {
                    "qid": probe.qid,
                    "fact_id": probe.meta["fact_id"],
                    "correct": correct,
                    "answer": probe.answer,
                    "generated": text,
                    "completed_expected_lookup": expected_key in completed_queries,
                    "events": events,
                    "meta": probe.meta,
                }
            )
    n = len(rows)
    summary = {
        "mode": mode,
        "accuracy": sum(row["correct"] for row in rows) / n if n else 0.0,
        "n": n,
        "lookup_stats": total,
    }
    return rows, summary


def fact_availability(recall_rows: list[dict]) -> dict[str, bool]:
    return {row["fact_id"]: bool(row["correct"]) for row in recall_rows}


def score_kqa_transfer(
    model,
    tok,
    items,
    mode: str,
    organizer,
    device,
    *,
    fact_availability_map: dict[str, bool] | None = None,
    max_new: int = 384,
    batch_size: int = 16,
) -> tuple[list[dict], dict]:
    if mode not in {"dense", "split", "split_wrong", "split_off"}:
        raise ValueError("mode must be dense, split, split_wrong, or split_off")
    if mode in {"split", "split_wrong"} and organizer is None:
        raise ValueError(f"{mode} mode requires an organizer")
    if fact_availability_map is None:
        raise ValueError("direct fact-availability results are required")
    store = organizer if mode in {"split", "split_wrong"} else None
    rows: list[dict] = []
    total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    for lo in range(0, len(items), batch_size):
        chunk = items[lo : lo + batch_size]
        texts, stats, event_rows = generate_batch_with_events(
            model,
            tok,
            [item.prompt for item in chunk],
            max_new,
            store,
            device,
        )
        _add_stats(total, stats)
        for item, generated, events in zip(chunk, texts, event_rows):
            prediction = parse_answer(generated)
            correct = prediction is not None and (
                normalize_answer(prediction) == normalize_answer(item.answer)
            )
            support = [normalize(key) for key in item.meta["support_keys"]]
            availability = {
                key: bool(fact_availability_map.get(key, False)) for key in support
            }
            completed_queries = _completed_hit_queries(events)
            query_availability = {
                key: key in completed_queries for key in support
            }
            support_coverage = (
                sum(availability.values()) / len(availability) if availability else 1.0
            )
            query_support_coverage = (
                sum(query_availability.values()) / len(query_availability)
                if query_availability
                else 1.0
            )
            rows.append(
                {
                    "qid": item.qid,
                    "correct": correct,
                    "pred": prediction,
                    "answer": item.answer,
                    "generated": generated,
                    "facts_available": all(availability.values()),
                    "support_coverage": support_coverage,
                    "availability": availability,
                    "all_support_queried": (
                        all(query_availability.values())
                        if mode in {"split", "split_wrong"}
                        else None
                    ),
                    "query_support_coverage": (
                        query_support_coverage
                        if mode in {"split", "split_wrong"}
                        else None
                    ),
                    "query_availability": (
                        query_availability
                        if mode in {"split", "split_wrong"}
                        else None
                    ),
                    "events": events,
                    "meta": item.meta,
                }
            )
    summary = summarize_transfer_rows(rows)
    summary["mode"] = mode
    summary["lookup_stats"] = total
    return rows, summary


def summarize_transfer_rows(rows: list[dict]) -> dict:
    n = len(rows)
    available = [row for row in rows if row["facts_available"]]
    summary = {
        "n": n,
        "accuracy": sum(row["correct"] for row in rows) / n if n else 0.0,
        "facts_available_n": len(available),
        "facts_available_rate": len(available) / n if n else 0.0,
        "conditional_accuracy": (
            sum(row["correct"] for row in available) / len(available)
            if available
            else 0.0
        ),
        "mean_support_coverage": (
            sum(row["support_coverage"] for row in rows) / n if n else 0.0
        ),
    }
    query_rows = [row for row in rows if row.get("all_support_queried") is not None]
    if query_rows:
        summary["all_support_queried_rate"] = sum(
            row["all_support_queried"] for row in query_rows
        ) / len(query_rows)
        summary["mean_query_support_coverage"] = sum(
            row["query_support_coverage"] for row in query_rows
        ) / len(query_rows)
    by_skill = {}
    for skill in sorted({row.get("meta", {}).get("skill", "unknown") for row in rows}):
        skill_rows = [
            row
            for row in rows
            if row.get("meta", {}).get("skill", "unknown") == skill
        ]
        by_skill[skill] = {
            "accuracy": sum(row["correct"] for row in skill_rows) / len(skill_rows),
            "n": len(skill_rows),
        }
    summary["by_skill"] = by_skill
    for field in ("program_length", "support_count"):
        grouped = {}
        values = sorted(
            {
                row.get("meta", {}).get(field)
                for row in rows
                if row.get("meta", {}).get(field) is not None
            }
        )
        for value in values:
            group_rows = [
                row for row in rows if row.get("meta", {}).get(field) == value
            ]
            grouped[str(value)] = {
                "accuracy": sum(row["correct"] for row in group_rows)
                / len(group_rows),
                "n": len(group_rows),
            }
        summary[f"by_{field}"] = grouped
    return summary


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
