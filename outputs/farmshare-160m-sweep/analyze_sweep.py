#!/usr/bin/env python
"""Parse the pulled d160m sweep logs and verify what the run README asserts.

Reads only log.jsonl + config.yaml, so it runs without the cluster checkpoints.
Emits a per-run table, the seed spread now that both seeds exist, and a set of
explicit PASS/FAIL checks on the claims the README makes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parent
LOADS = ("n50k", "n200k", "n800k")
ARMS = ("dense", "split")
SEEDS = (0, 1)

EXPECTED_STEPS = 6103
EXPECTED_TOKENS = 3_200_000_000
EXPECTED_TPS = 524_288
UNIFORM_CE = math.log(50304)  # 10.826 nats over the padded vocab


def load_run(run_id: str) -> dict | None:
    d = ROOT / run_id
    if not d.is_dir():
        return None
    rows = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines() if line.strip()]
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    return {"run_id": run_id, "cfg": cfg, "rows": rows, "last": rows[-1], "n_rows": len(rows)}


def main() -> None:
    runs: dict[tuple[str, str, int], dict] = {}
    missing = []
    for load in LOADS:
        for arm in ARMS:
            for seed in SEEDS:
                rid = f"d160m_{arm}_{load}_s{seed}"
                r = load_run(rid)
                if r is None:
                    missing.append(rid)
                else:
                    runs[(load, arm, seed)] = r

    print("=" * 78)
    print("PER-RUN SUMMARY")
    print("=" * 78)
    print(f"{'run':<26} {'steps':>6} {'loss_ema':>9} {'epoch':>6} {'masked_CE':>10} {'tok/s':>9}")
    for load in LOADS:
        for arm in ARMS:
            for seed in SEEDS:
                r = runs.get((load, arm, seed))
                if r is None:
                    continue
                last = r["last"]
                mv = last.get("loss_masked_values")
                tps = mean(x["tok_s"] for x in r["rows"][1:])  # row 0 includes warmup
                print(f"{r['run_id']:<26} {last['step']:>6} {last['loss_ema']:>9.4f} "
                      f"{last['epoch']:>6} {('-' if mv is None else f'{mv:>10.2f}')} {tps:>9.0f}")

    print()
    print("=" * 78)
    print("SEED SPREAD  (new: both seeds now exist)")
    print("=" * 78)
    print(f"{'cell':<16} {'s0':>9} {'s1':>9} {'mean':>9} {'|s0-s1|':>9}")
    seed_gaps = []
    for load in LOADS:
        for arm in ARMS:
            a, b = runs.get((load, arm, 0)), runs.get((load, arm, 1))
            if not (a and b):
                continue
            x, y = a["last"]["loss_ema"], b["last"]["loss_ema"]
            seed_gaps.append(abs(x - y))
            print(f"{load+'/'+arm:<16} {x:>9.4f} {y:>9.4f} {mean([x,y]):>9.4f} {abs(x-y):>9.4f}")
    if seed_gaps:
        print(f"\nseed-to-seed |delta| in final loss_ema: "
              f"max {max(seed_gaps):.4f}, mean {mean(seed_gaps):.4f}")

    print()
    print("=" * 78)
    print("ARM GAP  (dense - split), mean over seeds  -- NOT a treatment effect")
    print("=" * 78)
    for load in LOADS:
        ds = [runs[(load, "dense", s)]["last"]["loss_ema"] for s in SEEDS if (load, "dense", s) in runs]
        sp = [runs[(load, "split", s)]["last"]["loss_ema"] for s in SEEDS if (load, "split", s) in runs]
        if ds and sp:
            print(f"  {load:<8} dense {mean(ds):.4f}  split {mean(sp):.4f}  "
                  f"gap {mean(ds)-mean(sp):+.4f}")
    print("  Arms score different target sets (split excludes fact-value tokens),")
    print("  so this gap mixes the masking with any real difference. Not reportable.")

    print()
    print("=" * 78)
    print("CHECKS")
    print("=" * 78)
    checks: list[tuple[str, bool, str]] = []

    checks.append(("all 12 sweep runs present", len(runs) == 12,
                   f"found {len(runs)}/12" + (f", missing {missing}" if missing else "")))

    done = [r for r in runs.values() if r["last"]["step"] == EXPECTED_STEPS]
    checks.append((f"all runs reached step {EXPECTED_STEPS}", len(done) == len(runs),
                   f"{len(done)}/{len(runs)} at final step"))

    cfg_ok, cfg_bad = True, []
    for (load, arm, seed), r in runs.items():
        c = r["cfg"]
        problems = []
        if c["total_tokens"] != EXPECTED_TOKENS:
            problems.append(f"tokens={c['total_tokens']}")
        if c["tokens_per_step"] != EXPECTED_TPS:
            problems.append(f"tps={c['tokens_per_step']}")
        if c["seed"] != seed:
            problems.append(f"seed={c['seed']}")
        if c["arm"] != arm:
            problems.append(f"arm={c['arm']}")
        if c["model"] != "d160m":
            problems.append(f"model={c['model']}")
        if (c.get("train_mask") is not None) != (arm == "split"):
            problems.append(f"mask={c.get('train_mask')}")
        if problems:
            cfg_ok = False
            cfg_bad.append(f"{r['run_id']}: {','.join(problems)}")
    checks.append(("configs match the frozen spec", cfg_ok, "; ".join(cfg_bad) or "all 12 consistent"))

    lr_ok = all(r["cfg"]["lr"] == 1.5e-3 and r["cfg"]["warmup_steps"] == 300 for r in runs.values())
    checks.append(("lr 1.5e-3 / warmup 300 everywhere", lr_ok, ""))

    # gate 0: masked-value CE should sit at the uniform ceiling
    mvs = [(r["run_id"], r["last"]["loss_masked_values"]) for r in runs.values()
           if "loss_masked_values" in r["last"]]
    near = [v for _, v in mvs if v >= UNIFORM_CE - 1.5]
    checks.append((f"gate 0: masked CE near ln(50304)={UNIFORM_CE:.2f}",
                   len(mvs) == 6 and len(near) == len(mvs),
                   f"{len(mvs)} split runs, range {min(v for _,v in mvs):.2f}-{max(v for _,v in mvs):.2f}"))

    dense_ep = {r["last"]["epoch"] for k, r in runs.items() if k[1] == "dense"}
    split_ep = {r["last"]["epoch"] for k, r in runs.items() if k[1] == "split"}
    checks.append(("CONFOUND: arms saw equal passes over data",
                   dense_ep == split_ep,
                   f"dense epoch {sorted(dense_ep)} vs split epoch {sorted(split_ep)}"))

    evals = list(ROOT.glob("*/evals")) + list(ROOT.glob("*/eval*.json"))
    checks.append(("evaluations present", bool(evals),
                   "none found - training only" if not evals else str(len(evals))))

    for name, ok, note in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({note})" if note else ""))

    print()
    print("=" * 78)
    print("d1b CALIBRATION GATES")
    print("=" * 78)
    for rid in ("d1b_dense_n800k_s0_gate", "d1b_dense_n4m_s0_gate"):
        r = load_run(rid)
        if r:
            c, last = r["cfg"], r["last"]
            print(f"  {rid:<26} step {last['step']}  loss_ema {last['loss_ema']:.4f}  "
                  f"epoch {last['epoch']}  budget {c['total_tokens']:,}")


if __name__ == "__main__":
    main()
