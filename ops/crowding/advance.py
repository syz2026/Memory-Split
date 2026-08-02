"""Decide and take the next step, without a human in the loop.

Written to be run immediately on reconnect. `status.py` says what happened;
this says what to do about it and, with --execute, does it.

The ordering reflects one fact from docs/THEORY-ENDPOINT.md: the binding
constraint on Stage A was almost certainly the STEP BUDGET, not the difficulty.
Stage A ran 1,525 optimizer steps against a task the literature learns in
1e4-1e6. The 16.4B corpus gives 31,280 steps, about 17 hours on one L40S.

So the dense run on the full corpus is informative whichever way the ladder
falls, and it is therefore launched first:

  ladder clears at 23   the corpus is already the right difficulty; the run
                        confirms it at the full budget.
  ladder clears below   difficulty is not the blocker at 1,500 steps, and the
                        run tests whether the budget rescues mod 23 anyway. If
                        it does, keep 23: a harder endpoint leaves more room
                        for a crowding effect to show.
  nothing clears        the step hypothesis is the last defence, and the run
                        is exactly its test. Cheaper than rebuilding at a
                        modulus that also might not clear.

The step ladder itself is free. The trainer writes model-only snapshots, so
scoring them turns this one run into the whole ladder.

A rebuild at a lower modulus is only proposed once the step budget has been
ruled out, because it costs a 2-hour build and 65.6 GB, and 65.6 GB is most of
what the scratch budget has left.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MAIN_RUN = "stepladder_d40m_std"
SNAP_FRAC = 0.05           # 20 snapshots, 3.2 GB, ladder points every ~1,564 steps
FULL_STEPS = 31_280
SCRATCH_BUDGET_GB = 150
CORPUS_GB = 65.6


def _find(corpora: list[dict], name: str) -> dict | None:
    return next((c for c in corpora if c["name"] == name), None)


def _run(runs: list[dict], name: str) -> dict | None:
    return next((r for r in runs if r["name"] == name), None)


def decide(ladder: dict | None, corpora: list[dict], runs: list[dict],
           queue: list[str], root: str, main_corpus: str = "main",
           used_gb: float = 0.0) -> dict:
    """The single next action, with the reason and the command."""
    running = [l for l in queue if " RUNNING " in f" {l} "]
    pending = [l for l in queue if " PENDING " in f" {l} "]

    corpus = _find(corpora, main_corpus)
    if corpus is None or not corpus.get("verified"):
        if any("build" in l for l in running + pending):
            return {"action": "WAIT",
                    "why": f"{main_corpus} corpus still building."}
        return {
            "action": "REBUILD MAIN CORPUS",
            "why": f"{main_corpus} is absent or failed verification, and "
                   "nothing is building it. Everything downstream needs it.",
            "commands": ["bash ops/crowding/run_load.sh"],
        }

    main = _run(runs, MAIN_RUN)

    # 1. The step-budget test. Informative regardless of the ladder, so it does
    #    not wait on it.
    if main is None:
        if any(MAIN_RUN in l for l in running + pending):
            return {"action": "WAIT", "why": "step-ladder run is queued."}
        return {
            "action": "LAUNCH THE STEP-LADDER RUN",
            "why": (f"{FULL_STEPS:,} steps against Stage A's 1,525, on a corpus "
                    "that is already built. Tests the one explanation for the "
                    "floor that survived, and costs about 17 h on one L40S."),
            "commands": [
                f"PYTHONPATH=. $MS_PY ops/crowding/advance.py --mode gen-main "
                f"--root {root} --corpus {root}/corpora/{main_corpus}",
                f"MS_CFG={root}/configs/{MAIN_RUN}.yaml "
                "sbatch ops/crowding/train.sbatch",
            ],
        }

    if any(MAIN_RUN in l for l in running):
        step = main.get("last_step") or 0
        pct = 100.0 * step / FULL_STEPS
        return {"action": "WAIT",
                "why": f"step-ladder run at {step:,}/{FULL_STEPS:,} ({pct:.0f}%)."}

    # 2. Score the snapshots that already exist.
    n_snap = main.get("n_snapshots", 0)
    n_scored = main.get("n_snapshot_evals", 0)
    if n_snap and n_scored < n_snap:
        return {
            "action": "SCORE THE SNAPSHOTS",
            "why": f"{n_snap} snapshots, {n_scored} scored. The step ladder "
                   "reads off these and needs no further training.",
            "commands": [
                f"for s in {root}/runs/{MAIN_RUN}/snapshots/step*.pt; do "
                f"$MS_PY scripts/run_evals.py --run {root}/runs/{MAIN_RUN} "
                "--ckpt $s; done",
                f"PYTHONPATH=. $MS_PY ops/crowding/ladder.py --mode steps "
                f"--run {root}/runs/{MAIN_RUN}",
            ],
        }

    # 3. Both ladders in hand: choose the operating point.
    if ladder and ladder.get("rungs"):
        clears = ladder.get("easiest_clearing_mod")
        if clears:
            return {
                "action": f"FREEZE THE OPERATING POINT AT MOD {clears}",
                "why": (f"the modulus ladder clears at {clears} and the step "
                        "ladder is scored. Pick the HARDEST modulus that clears "
                        "at the full budget -- a harder endpoint leaves more "
                        "room for a crowding effect than an easy one."),
                "commands": [
                    f"PYTHONPATH=. $MS_PY ops/crowding/ladder.py --mode steps "
                    f"--run {root}/runs/{MAIN_RUN}",
                ],
            }
        headroom = SCRATCH_BUDGET_GB - used_gb
        return {
            "action": "STOP AND REPORT",
            "why": ("no rung clears and the step budget has now been tested "
                    "too. Both defences are spent, so the NO-GO paper is the "
                    "honest output. Do not spend 560 GPU-hours on a matrix "
                    "whose endpoint cannot do one modular addition."),
            "blocked": headroom < CORPUS_GB,
            "commands": ["cat docs/NO-GO-PAPER.md"],
        }

    return {"action": "RANK THE LADDER",
            "why": "step-ladder run is done but the modulus ladder is not "
                   "ranked; both are needed to choose an operating point.",
            "commands": [f"PYTHONPATH=. $MS_PY ops/crowding/ladder.py "
                         f"--mode rank --runs {root}/runs"]}


def main_config(corpus: str, root: str, lr: float, grad_clip: float,
                n_entities: int, mod: int, seed: int = 0) -> dict:
    return {
        "run_id": MAIN_RUN,
        "stage": "step-ladder",
        "arm": "sup",
        "model": "d40m_std",
        "ctx": 1024,
        "vocab_size": 50304,
        "train_bin": f"{corpus}/targets.bin",
        "probe_mask": f"{corpus}/factmask.bin",
        "micro_batch_size": 32,
        "tokens_per_step": 524_288,
        "max_steps": FULL_STEPS,
        "total_tokens": FULL_STEPS * 524_288,
        "lr": lr,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "grad_clip": grad_clip,
        "seed": seed,
        "n_entities": n_entities,
        "corpus_seed": 0,
        "igsm_mod": mod,
        "igsm_op": [1, 4],
        "igsm_ood_op": [5, 8],
        "out_dir": f"{root}/runs/{MAIN_RUN}",
        "device": "cuda",
        "compile": True,
        "log_every": 50,
        "eval_every": 100_000,
        "snap_frac": SNAP_FRAC,
        "ckpt_minutes": 30,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decide", "gen-main"], default="decide")
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus")
    ap.add_argument("--main-corpus", default="main")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-entities", type=int, default=1_531_800)
    ap.add_argument("--mod", type=int, default=23)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)

    if args.mode == "gen-main":
        import yaml
        cfg = main_config(args.corpus, args.root, args.lr, args.grad_clip,
                          args.n_entities, args.mod)
        out = root / "configs" / f"{MAIN_RUN}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {out}  ({FULL_STEPS:,} steps, snap_frac={SNAP_FRAC})")
        return 0

    from ops.crowding import status as st
    from ops.crowding.ladder import rank_modulus

    corpora, runs = st.corpora(root), st.runs(root)
    for r in runs:                      # snapshot evals drive the step ladder
        ev = root / "runs" / r["name"] / "evals"
        r["n_snapshot_evals"] = len(list(ev.glob("step*.json"))) if ev.is_dir() else 0
    q = st.queue(__import__("os").environ.get("USER", "syz"))
    used_gb = sum(c["bytes"] for c in corpora) / (1 << 30)

    try:
        ladder = rank_modulus(root / "runs")
    except Exception:
        ladder = None

    act = decide(ladder, corpora, runs, q, str(root), args.main_corpus, used_gb)
    print(json.dumps(act, indent=2))

    if args.execute and act.get("commands") and not act.get("blocked"):
        if act["action"].startswith(("WAIT", "STOP")):
            print("\n(not executing: action is advisory)")
            return 0
        for c in act["commands"]:
            print(f"\n$ {c}")
            rc = subprocess.run(c, shell=True, cwd=root).returncode
            if rc != 0:
                print(f"FAILED rc={rc}")
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
