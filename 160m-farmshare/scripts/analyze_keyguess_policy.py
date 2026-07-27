#!/usr/bin/env python
"""Governance-policy analysis for the keyguess arms (results-doc source).

Recomputes, from the per-item records and the organizer table, the two
deployable selective policies:

  P0 ship-on-hit:        splice whatever the store returns on a key hit.
  P1 hit + name echo:    additionally require the emitted name half to equal
                         the gold subject (an oracle stand-in for the
                         mention-similarity verification vote).

Precision is STRICT RETURNED-VALUE equality: normalize(store value) must be
in the normalized possible_answers set for the item (never a substring match,
never the whole continuation). Continuation-level answer accuracy (the
`answer_ok` flag from evals.keyguess) is reported separately.

Usage: python scripts/analyze_keyguess_policy.py [--data data/keyguess_local] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from organizer.store import normalize


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def policy_rows(data_dir: Path, arms: tuple[str, ...] = ("A", "B", "C", "D"),
                suffix: str = "") -> list[dict]:
    store = {r["key"]: r["value"] for r in load_jsonl(data_dir / "organizer_real.jsonl")}
    items = {r["qid"]: r for r in load_jsonl(data_dir / "eval_items.jsonl")}
    rows = []
    for arm in arms:
        path = data_dir / f"records_{arm}{suffix}.jsonl"
        if not path.exists():
            continue
        held = [r for r in load_jsonl(path) if r.get("split") == "heldout"]
        n = len(held)

        def answers(rec: dict) -> set[str]:
            meta = items[rec["qid"]]["meta"]
            pa = meta.get("possible_answers") or [meta["obj"]]
            return {normalize(a) for a in pa}

        hits = [(r, store[normalize(r["emitted_key"])])
                for r in held
                if r.get("emitted_key") and normalize(r["emitted_key"]) in store]
        p0_ok = [r for r, val in hits if normalize(val) in answers(r)]
        p1 = [(r, val) for r, val in hits if r.get("name_ok")]
        p1_ok = [r for r, val in p1 if normalize(val) in answers(r)]
        rows.append({
            "arm": arm,
            "n": n,
            "p0_cov": len(hits), "p0_ok": len(p0_ok),
            "p0_silent_wrong": len(hits) - len(p0_ok),
            "p0_wrong_referent": sum(1 for r, _ in hits if not r.get("name_ok")),
            "p1_cov": len(p1), "p1_ok": len(p1_ok),
            "p1_silent_wrong": len(p1) - len(p1_ok),
            "answer_continuation": sum(1 for r in held if r.get("answer_ok")),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/keyguess_local")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    suffix = "" if args.seed == 0 else f"_s{args.seed}"
    rows = policy_rows(Path(args.data), suffix=suffix)
    hdr = ("arm  P0 cov          P0 prec        P0 silent-wrong  wrong-ref  "
           "P1 cov          P1 prec        cont-answer")
    print(hdr)
    for r in rows:
        n = r["n"]

        def pct(k: int, d: int) -> str:
            return f"{k}/{d}={k / d:.1%}" if d else "0/0"

        print(f"{r['arm']}:  {pct(r['p0_cov'], n):<14} "
              f"{pct(r['p0_ok'], r['p0_cov']):<14} "
              f"{pct(r['p0_silent_wrong'], n):<16} "
              f"{r['p0_wrong_referent']:<9} "
              f"{pct(r['p1_cov'], n):<14} "
              f"{pct(r['p1_ok'], r['p1_cov']):<14} "
              f"{pct(r['answer_continuation'], n)}")
    out = Path(args.data) / f"policy_analysis{suffix}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
