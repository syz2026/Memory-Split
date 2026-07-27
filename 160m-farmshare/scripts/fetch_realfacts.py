#!/usr/bin/env python
"""Freeze PopQA into the real-fact JSONL consumed by corpusgen.realfact.

Usage:
  python scripts/fetch_realfacts.py --out data/realfacts

Downloads akariasai/PopQA (test split, ~14.3K Wikidata triples over 16
relations), cleans it (strip whitespace, drop empty fields, dedupe on the
normalized organizer key "{subj}, {prop}" keeping the first row), caps to
3,000 rows by seeded shuffle, then sorts by (prop, subj) so downstream
per-relation splits are deterministic. Writes popqa_clean.jsonl with rows
{"subj","prop","obj","question","possible_answers"}.

Reference counts from the 2026-07-20 run: raw 14,264 / cleaned 13,059 /
capped 3,000. Counts are printed, not enforced. The network path lives only
in fetch_popqa(); clean_rows/cap_rows/parse_possible_answers are pure and
unit-tested offline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from organizer.store import normalize

_FIELDS = ("subj", "prop", "obj", "question")


def parse_possible_answers(raw, obj: str) -> list[str]:
    """PopQA's possible_answers is a JSON-encoded list string; fall back to
    [obj] on anything unparseable, non-list, or empty."""
    if isinstance(raw, list):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw) if raw else None
        except (TypeError, json.JSONDecodeError):
            parsed = None
    if not isinstance(parsed, list) or not parsed:
        return [obj]
    return [str(a) for a in parsed]


def clean_rows(rows: list[dict]) -> list[dict]:
    """Strip whitespace, drop rows with empty/None subj/prop/obj/question,
    and dedupe on the normalized organizer key (keep the first row)."""
    cleaned: list[dict] = []
    seen_keys: set[str] = set()
    for row in rows:
        stripped = dict(row)
        for field in _FIELDS:
            value = stripped.get(field)
            stripped[field] = value.strip() if isinstance(value, str) else value
        if any(not stripped.get(field) for field in _FIELDS):
            continue
        key = normalize(f"{stripped['subj']}, {stripped['prop']}")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cleaned.append(stripped)
    return cleaned


def cap_rows(rows: list[dict], n: int = 3000, seed: int = 0) -> list[dict]:
    """Seeded shuffle, take the first n, THEN sort by (prop, subj) so the
    downstream per-relation splits see a canonical order."""
    capped = list(rows)
    random.Random(seed).shuffle(capped)
    capped = capped[:n]
    capped.sort(key=lambda r: (r["prop"], r["subj"]))
    return capped


def fetch_popqa() -> list[dict]:
    """Network path (huggingface only); kept out of module import."""
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA")["test"]
    rows: list[dict] = []
    for row in ds:
        rows.append({
            "subj": row.get("subj"),
            "prop": row.get("prop"),
            "obj": row.get("obj"),
            "question": row.get("question"),
            "possible_answers": parse_possible_answers(
                row.get("possible_answers"), row.get("obj") or ""
            ),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/realfacts")
    args = ap.parse_args()

    raw = fetch_popqa()
    cleaned = clean_rows(raw)
    capped = cap_rows(cleaned)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "popqa_clean.jsonl"
    with open(out_path, "w") as f:
        for row in capped:
            f.write(json.dumps({
                "subj": row["subj"],
                "prop": row["prop"],
                "obj": row["obj"],
                "question": row["question"],
                "possible_answers": row["possible_answers"],
            }) + "\n")

    per_relation = Counter(row["prop"] for row in capped)
    print(f"raw {len(raw)} / cleaned {len(cleaned)} / capped {len(capped)} "
          f"-> {out_path}")
    print(f"{len(per_relation)} relations:")
    for prop, count in sorted(per_relation.items()):
        print(f"  {prop}: {count}")


if __name__ == "__main__":
    main()
