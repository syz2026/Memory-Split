"""The difficulty ladder Stage A was supposed to descend, and a step ladder.

Stage A returned STOP on a single rung. `build_corpus.py` takes one `--mod`,
the pilotA manifest records mod 23, and both Stage A cells ran that same
difficulty while varying only architecture. So "the endpoint has no power to
discriminate" was demonstrated at one difficulty and one step budget, and the
op-profile in docs/THEORY-ENDPOINT.md says the atomic operation is what failed:
accuracy sits at the no-skill baseline at op=1, where compounding, depth and
capacity all predict otherwise.

Two probes separate the remaining explanations.

  modulus   Hold the budget at Stage A's 1,500 steps and make the operation
            easier: mod 23, 11, 7, 5. Table sizes fall from 9,572 bits to 232.
            If op=1 lifts at an easier modulus the operation is learnable and
            23 was simply too hard for the budget. If it stays at baseline even
            at mod 5 -- five possible answers -- the task presentation is
            broken, not the difficulty.

  steps     Hold the modulus and vary the budget. This costs no extra training:
            the trainer already writes model-only snapshots, so scoring them
            turns one run into a ladder. Reads snapshots rather than launching
            runs.

Both are dense-arm only and single-seed. They ask whether the endpoint can move
at all, which is prior to any question about arms.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

MODULI = (23, 11, 7, 5)
PROBE_STEPS = 1500


def modulus_configs(root: str, corpora: dict[int, str], model: str, lr: float,
                    grad_clip: float, n_entities: int, steps: int = PROBE_STEPS,
                    seed: int = 0) -> list[dict]:
    """One dense run per modulus, everything else held at Stage A's values.

    Each rung needs its own corpus because the modulus is baked into the text
    at build time. At 800M tokens a rung is about 3.2 GB and a few minutes,
    which is cheap next to the conclusion it protects.
    """
    out = []
    for mod, corpus in sorted(corpora.items(), reverse=True):
        rid = f"ladder_mod{mod}_{model}"
        out.append({
            "run_id": rid,
            "stage": "ladder-modulus",
            "arm": "sup",
            "model": model,
            "ctx": 1024,
            "vocab_size": 50304,
            "train_bin": f"{corpus}/targets.bin",
            "probe_mask": f"{corpus}/factmask.bin",
            "micro_batch_size": 32,
            "tokens_per_step": 524_288,
            "max_steps": steps,
            "total_tokens": steps * 524_288,
            "lr": lr,
            "warmup_steps": 150,
            "weight_decay": 0.1,
            "grad_clip": grad_clip,
            "seed": seed,
            "n_entities": n_entities,
            "corpus_seed": 0,
            "igsm_mod": mod,
            "igsm_op": [1, 4],
            "igsm_ood_op": [5, 8],
            "out_dir": f"{root}/runs/{rid}",
            "device": "cuda",
            "compile": True,
            "log_every": 20,
            "eval_every": 100_000,
            "snap_frac": 1.0,
            "ckpt_minutes": 999,
        })
    return out


def _op1(summary: dict) -> tuple[float | None, int]:
    by_op = summary.get("igsm_by_op") or {}
    if not by_op:
        return None, 0
    k = min(by_op, key=lambda x: int(x))
    return by_op[k].get("acc"), by_op[k].get("n", 0)


def rank_modulus(runs_root: Path) -> dict:
    """Did the atomic operation become learnable at any rung?

    Judged on op=1 against the no-skill baseline, not on overall exact match.
    op=1 is the cleanest read: it needs a single operation, so it isolates the
    operation from composition, and it is where every non-optimization
    explanation predicts a lift.
    """
    from theory.endpoint import OpProfile

    rows = []
    for d in sorted(runs_root.glob("ladder_mod*")):
        s = d / "evals" / "summary.json"
        if not s.exists():
            continue
        summ = json.loads(s.read_text())
        mod = int(re.search(r"mod(\d+)", d.name).group(1))
        acc1, n1 = _op1(summ)
        by_op = {int(k): v["acc"] for k, v in (summ.get("igsm_by_op") or {}).items()}
        n_by_op = {int(k): v["n"] for k, v in (summ.get("igsm_by_op") or {}).items()}
        majority = summ.get("igsm_majority_rate", 1.0 / mod)
        prof = OpProfile(by_op, 1.0 / mod, majority, n_by_op)
        rows.append({
            "run": d.name,
            "mod": mod,
            "chance": round(1.0 / mod, 4),
            "majority": round(majority, 4),
            "no_skill": round(prof.no_skill, 4),
            "acc_op1": acc1,
            "n_op1": n1,
            "op1_clears_baseline": (not prof.near_baseline(min(by_op))) if by_op else None,
            "overall": summ.get("igsm_acc"),
            "signature": prof.signature(),
        })
    rows.sort(key=lambda r: -r["mod"])
    if not rows:
        return {"error": "no ladder runs found"}

    cleared = [r for r in rows if r["op1_clears_baseline"]]
    easiest = min(rows, key=lambda r: r["mod"])
    return {
        "rungs": rows,
        "any_rung_clears": bool(cleared),
        "easiest_clearing_mod": max((r["mod"] for r in cleared), default=None),
        "verdict": (
            f"LEARNABLE at mod {max(r['mod'] for r in cleared)} and below. The "
            "atomic operation is within reach at this budget, so Stage A's "
            "floor reflects difficulty rather than an unusable endpoint. Run "
            "the confirmatory design at a modulus that clears."
            if cleared else
            f"NOT LEARNABLE even at mod {easiest['mod']}, where there are only "
            f"{easiest['mod']} possible answers and the table is trivial. This "
            "is no longer a difficulty problem. Either the step budget binds -- "
            "check the step ladder -- or the task presentation is broken."
        ),
    }


def step_ladder(run: Path) -> dict:
    """Read a step ladder off one run's snapshots.

    Expects `evals/step*.json` written by `run_evals.py --ckpt`. Costs no
    training: the snapshots already exist.
    """
    from theory.endpoint import OpProfile

    rows = []
    for f in sorted((run / "evals").glob("step*.json")):
        summ = json.loads(f.read_text())
        step = int(re.search(r"(\d+)", f.stem).group(1))
        by_op = {int(k): v["acc"] for k, v in (summ.get("igsm_by_op") or {}).items()}
        n_by_op = {int(k): v["n"] for k, v in (summ.get("igsm_by_op") or {}).items()}
        mod = summ.get("mod", 23)
        prof = OpProfile(by_op, 1.0 / mod, summ.get("igsm_majority_rate", 1.0 / mod),
                         n_by_op)
        acc1, _ = _op1(summ)
        rows.append({
            "step": step,
            "acc_op1": acc1,
            "overall": summ.get("igsm_acc"),
            "op1_clears_baseline": (not prof.near_baseline(min(by_op))) if by_op else None,
        })
    rows.sort(key=lambda r: r["step"])
    if not rows:
        return {"error": f"no snapshot evals under {run}/evals/step*.json"}

    cleared = [r for r in rows if r["op1_clears_baseline"]]
    rising = len(rows) > 1 and rows[-1]["acc_op1"] is not None and \
        rows[0]["acc_op1"] is not None and rows[-1]["acc_op1"] > rows[0]["acc_op1"]
    return {
        "run": run.name,
        "ladder": rows,
        "first_clearing_step": min((r["step"] for r in cleared), default=None),
        "verdict": (
            f"Clears the baseline by step {min(r['step'] for r in cleared):,}. "
            "Stage A's floor was a step budget, not an unusable endpoint."
            if cleared else
            ("Still at baseline at the last snapshot, but op=1 is rising, so "
             "the budget may simply need to be longer again."
             if rising else
             "Flat at baseline across the whole ladder. More steps are not the "
             "answer; the endpoint or its presentation is.")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "rank", "steps"], required=True)
    ap.add_argument("--out")
    ap.add_argument("--root")
    ap.add_argument("--runs")
    ap.add_argument("--run")
    ap.add_argument("--corpus-root", help="parent holding ladder-mod<M> corpora")
    ap.add_argument("--model", default="d40m_std")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--n-entities", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=PROBE_STEPS)
    ap.add_argument("--moduli", type=int, nargs="*", default=list(MODULI))
    args = ap.parse_args()

    if args.mode == "gen":
        corpora = {m: f"{args.corpus_root}/ladder-mod{m}" for m in args.moduli}
        cfgs = modulus_configs(args.root, corpora, args.model, args.lr,
                               args.grad_clip, args.n_entities, args.steps)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for c in cfgs:
            (out / f"{c['run_id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
            print("  ", c["run_id"], "mod", c["igsm_mod"])
        print(f"{len(cfgs)} ladder configs -> {out}")
    elif args.mode == "rank":
        r = rank_modulus(Path(args.runs))
        print(json.dumps(r, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(r, indent=2))
        return 0 if r.get("any_rung_clears") else 2
    else:
        r = step_ladder(Path(args.run))
        print(json.dumps(r, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(r, indent=2))
        return 0 if r.get("first_clearing_step") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
