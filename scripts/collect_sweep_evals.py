"""Tabulate the 160M sweep evaluations once FarmShare has run them.

The twelve sweep checkpoints (3.2B tokens, 3 fact loads x 2 arms x 2 seeds) were
trained on 2026-07-21 and never evaluated: `outputs/farmshare-160m-sweep/
ANALYSIS.txt` line 51 records `[FAIL] evaluations present`. On 2026-08-05 the
six dense runs were queued with the post-padding-fix decoder as jobs
1676311-1676316.

This reads the summaries over the existing SSH ControlMaster session and writes
nothing to the cluster. If the session has expired, reopen it with
`bash cluster/connect.sh <sunetid>`.

The dense dose-response is the point. If accuracy is flat or rising from 50k to
800k entities, reasoning does not degrade under fact load and the crowding
account loses its central prediction. These replace the withdrawn 65.0 / 66.7 /
68.2, which had no artifact behind them.

Usage:  python3 scripts/collect_sweep_evals.py [--user syz]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

HOST = "rice-04.farmshare.stanford.edu"
RUNS = "/scratch/users/{user}/crowding/runs"
LOADS = ("n50k", "n200k", "n800k")
ENTITIES = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}


def fetch(user: str) -> dict[str, dict]:
    """One SSH round trip: cat every summary that exists, keyed by run id."""
    # `exit 0` because the final `[ -f ... ]` is false whenever a run has not
    # finished, which would otherwise make ssh report failure on a healthy call.
    remote = (
        f'for d in {RUNS.format(user=user)}/sweepeval_*/; do '
        f'  f="$d/evals/summary.json"; '
        f'  if [ -f "$f" ]; then echo "@@@$(basename $d)"; cat "$f"; fi; '
        f'done; exit 0'
    )
    # Expanded here: subprocess does not run a shell, so a literal "~" would be
    # handed to ssh as a relative path and the socket would never be found.
    sock = os.path.expanduser("~/.ssh/cm-%r@%h:%p")
    proc = subprocess.run(
        ["ssh", "-o", f"ControlPath={sock}", "-o", "ControlMaster=no",
         "-o", "BatchMode=yes", f"{user}@{HOST}", remote],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "ssh failed. Reopen the session with `bash cluster/connect.sh "
            f"{user}`.\n{proc.stderr.strip()}"
        )
    out: dict[str, dict] = {}
    blocks = proc.stdout.split("@@@")
    for block in blocks[1:]:
        name, _, body = block.partition("\n")
        run = name.strip().replace("sweepeval_", "")
        try:
            out[run] = json.loads(body)
        except json.JSONDecodeError:
            pass
    return out


def cell(summaries: dict, arm: str, load: str, seed: int, key: str):
    row = summaries.get(f"d160m_{arm}_{load}_s{seed}")
    return None if row is None else row.get(key)


def show(summaries: dict, key: str, title: str, baseline: str) -> None:
    print(f"\n{title}")
    print(f"  baseline: {baseline}")
    print(f"  {'load':<8}{'entities':>10}{'dense s0':>11}{'dense s1':>11}"
          f"{'split s0':>11}{'split s1':>11}")
    for load in LOADS:
        cells = [cell(summaries, a, load, s, key)
                 for a in ("dense", "split") for s in (0, 1)]
        txt = "".join(f"{v:>11.4f}" if isinstance(v, (int, float)) else f"{'-':>11}"
                      for v in cells)
        print(f"  {load:<8}{ENTITIES[load]:>10,}{txt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="syz")
    args = ap.parse_args()

    summaries = fetch(args.user)
    done = sorted(summaries)
    print(f"summaries found: {len(done)} of 12")
    for r in done:
        print(f"  {r}")
    if not done:
        print("\nNothing has finished yet. Check the queue with:")
        print(f"  ssh {args.user}@{HOST} 'squeue -u {args.user}'")
        return 0

    show(summaries, "deduction_acc", "Held-out deduction", "0.500 (constant 'no')")
    show(summaries, "igsm_acc", "Held-out iGSM",
         "empirical majority ~0.072, never 1/23")
    show(summaries, "recoverable_bits_per_entity", "Storage probe",
         "52.96 bits/entity ceiling; negative means confidently wrong")

    print("\nRead the dense columns across loads first. Crowding predicts they")
    print("fall as the fact load rises. Flat or rising is evidence against it,")
    print("and it needs no arm contrast, so the loss-normalization confound")
    print("does not touch it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
