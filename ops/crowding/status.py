"""Full readiness report, and the next action computed from it.

Written for resuming after a disconnect. SLURM keeps running whether or not
anyone is watching, so after a gap the first question is never "what should I
launch" but "what already happened". This answers that in one call and then
names the single next action, so the decision is not reconstructed by hand
from a queue dump at three in the morning.

Run on FarmShare:
    PYTHONPATH=. python ops/crowding/status.py --root /scratch/users/syz/crowding
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

GB = 1 << 30
SCRATCH_BUDGET_GB = 150


def _du(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def corpora(root: Path) -> list[dict]:
    out = []
    d = root / "corpora"
    if not d.is_dir():
        return out
    for c in sorted(x for x in d.iterdir() if x.is_dir()):
        man = c / "manifest.json"
        rec = {"name": c.name, "bytes": _du(c), "verified": man.exists()}
        if man.exists():
            try:
                m = json.loads(man.read_text())
                rec.update({k: m.get(k) for k in
                            ("n_tokens", "n_entities", "exposures", "mod")})
            except json.JSONDecodeError:
                rec["verified"] = False
                rec["error"] = "manifest is not valid JSON"
        out.append(rec)
    return out


def runs(root: Path) -> list[dict]:
    out = []
    d = root / "runs"
    if not d.is_dir():
        return out
    for r in sorted(x for x in d.iterdir() if x.is_dir()):
        log = r / "log.jsonl"
        step = None
        if log.exists():
            try:
                lines = [x for x in log.read_text().splitlines() if x.strip()]
                if lines:
                    step = json.loads(lines[-1]).get("step")
            except (json.JSONDecodeError, OSError):
                pass
        out.append({
            "name": r.name,
            "last_step": step,
            "has_ckpt": (r / "ckpt.pt").exists(),
            "has_eval": (r / "evals" / "summary.json").exists(),
            "n_snapshots": len(list((r / "snapshots").glob("step*.pt")))
            if (r / "snapshots").is_dir() else 0,
        })
    return out


def queue(user: str) -> list[str]:
    try:
        o = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j %T %M %R"],
                           capture_output=True, text=True, timeout=30)
        return [l for l in o.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def next_action(corp: list[dict], rn: list[dict], q: list[str],
                root: Path) -> dict:
    """Name the one thing to do next, and why.

    Ordered by what blocks what. The ladder gates everything: a confirmatory
    matrix on an endpoint that cannot do one modular addition would burn
    560 GPU-hours to measure noise.
    """
    running = [l for l in q if " RUNNING " in f" {l} "]
    if running:
        return {"action": "WAIT",
                "why": f"{len(running)} job(s) still running; let them land."}

    ladder = [r for r in rn if r["name"].startswith("ladder_mod")]
    ladder_done = [r for r in ladder if r["has_eval"]]
    if ladder and len(ladder_done) < len(ladder):
        missing = [r["name"] for r in ladder if not r["has_eval"]]
        return {"action": "RESUBMIT LADDER EVALS",
                "why": f"ladder trained but unevaluated: {', '.join(missing)}"}

    if ladder_done:
        return {
            "action": "RANK THE LADDER",
            "why": "all ladder rungs evaluated; the rank decides everything "
                   "downstream.",
            "command": "python ops/crowding/ladder.py --mode rank "
                       f"--runs {root}/runs",
        }

    if not ladder:
        return {"action": "SUBMIT THE LADDER",
                "why": "no ladder runs found; it gates the confirmatory design.",
                "command": "bash ops/crowding/ladder.sh"}

    return {"action": "REVIEW", "why": "no rule matched; inspect by hand."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--user", default=os.environ.get("USER", "syz"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    corp, rn, q = corpora(root), runs(root), queue(args.user)
    used = sum(c["bytes"] for c in corp)
    act = next_action(corp, rn, q, root)

    if args.json:
        print(json.dumps({"corpora": corp, "runs": rn, "queue": q,
                          "next": act}, indent=2))
        return 0

    print("=" * 68)
    print("CORPORA")
    for c in corp:
        flag = "ok " if c["verified"] else "PARTIAL/UNVERIFIED"
        tok = f"{c['n_tokens']:,}" if c.get("n_tokens") else "-"
        print(f"  {c['name']:<20} {c['bytes']/GB:6.1f} GB  {flag:<18} "
              f"tokens={tok} mod={c.get('mod', '-')}")
    print(f"  {'TOTAL':<20} {used/GB:6.1f} GB of ~{SCRATCH_BUDGET_GB} GB budget")

    print("\nRUNS")
    for r in rn:
        print(f"  {r['name']:<32} step={str(r['last_step']):>7}  "
              f"ckpt={'y' if r['has_ckpt'] else 'n'}  "
              f"eval={'y' if r['has_eval'] else 'n'}  "
              f"snaps={r['n_snapshots']}")

    print("\nQUEUE")
    print("  (empty)" if not q else "\n".join(f"  {l}" for l in q))

    print("\n" + "=" * 68)
    print(f"NEXT: {act['action']}")
    print(f"  {act['why']}")
    if act.get("command"):
        print(f"  $ {act['command']}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
