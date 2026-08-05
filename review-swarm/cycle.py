#!/usr/bin/env python3
"""Cycle driver for the MemorySplit review swarm.

Advances the cycle counter, rotates lane assignments across the three reviewer
models, creates the report directory, and prints the dispatch block for the
orchestrator. Rotation guarantees every model sees every lane across any three
consecutive cycles, so a defect that one model is blind to is picked up by
another within three hours.

    python3 review-swarm/cycle.py            # advance and print dispatch
    python3 review-swarm/cycle.py --peek     # print current state, no advance
"""

import argparse
import json
import pathlib
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
STATE = ROOT / "state.json"

MODELS = [
    ("opus-5", "claude-opus-5-thinking-max-fast", "Opus 5"),
    ("gpt-5.6-sol", "gpt-5.6-sol-max", "GPT 5.6 Sol"),
    ("grok-4.5", "cursor-grok-4.5-high-fast", "Cursor Grok 4.5"),
]

LANES = [
    ("A", "lane-A-inference.md", "Estimand, inference, preregistration compliance"),
    ("B", "lane-B-measurement.md", "Code, measurement, silent substitution"),
    ("C", "lane-C-provenance.md", "Claims, artifacts, provenance"),
]


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cycle": 0, "next_finding_id": 1, "history": []}


def repo_fingerprint():
    """Cheap signal for whether the reviewed surface moved since last cycle."""

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "head": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": len([l for l in git("status", "--porcelain").splitlines() if l]),
    }


def assign(cycle):
    """Rotate lanes across models. Cycle 1 gives model i lane i."""
    n = len(LANES)
    return [
        (mkey, mslug, mname, *LANES[(i + cycle - 1) % n])
        for i, (mkey, mslug, mname) in enumerate(MODELS)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true")
    args = ap.parse_args()

    state = load_state()
    cycle = state["cycle"] if args.peek else state["cycle"] + 1
    if cycle == 0:
        cycle = 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = ROOT / "reports" / f"cycle-{cycle:04d}"
    fingerprint = repo_fingerprint()

    if not args.peek:
        outdir.mkdir(parents=True, exist_ok=True)
        state["cycle"] = cycle
        state.setdefault("history", []).append(
            {"cycle": cycle, "started": stamp, "fingerprint": fingerprint}
        )
        STATE.write_text(json.dumps(state, indent=2) + "\n")

    print(f"CYCLE {cycle}   {stamp}")
    print(f"branch={fingerprint['branch']} head={fingerprint['head']} "
          f"uncommitted={fingerprint['dirty_files']}")
    print(f"reports -> {outdir.relative_to(REPO)}")
    print()
    for mkey, mslug, mname, lkey, lfile, ldesc in assign(cycle):
        print(f"[{mname}]  model={mslug}")
        print(f"  lane {lkey}: {ldesc}")
        print(f"  brief:  review-swarm/lanes/{lfile}")
        print(f"  output: {(outdir / (mkey + '.md')).relative_to(REPO)}")
        print()


if __name__ == "__main__":
    main()
