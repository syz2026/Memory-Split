#!/usr/bin/env python
"""Quantify the dense/split epoch asymmetry in the pulled d160m sweep.

Every dense run ends at epoch 1 and every split run at epoch 0. The step where
epoch flips 0->1 reveals how many tokens the dense corpus actually held, which
gives the size of the re-read and therefore the size of the exposure imbalance
between the arms.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOADS = ("n50k", "n200k", "n800k")
SEEDS = (0, 1)
TPS = 524_288
BUDGET = 3_200_000_000
BIO_SHARE = 0.23  # biographies are 23% of the mixture


def rows(run_id: str):
    p = ROOT / run_id / "log.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def flip_step(rs) -> int | None:
    """First logged step at which epoch > 0."""
    for r in rs:
        if r["epoch"] > 0:
            return r["step"]
    return None


print("=" * 82)
print("EPOCH FLIP — where each dense run wrapped its corpus")
print("=" * 82)
print(f"{'run':<26} {'flip@step':>10} {'corpus tok':>15} {'re-read tok':>14} {'re-read %':>10}")

corpus_est = {}
for load in LOADS:
    for seed in SEEDS:
        rid = f"d160m_dense_{load}_s{seed}"
        rs = rows(rid)
        fs = flip_step(rs)
        if fs is None:
            print(f"{rid:<26} {'never':>10}")
            continue
        # flip is detected at the first log line after wrap; log_every=20 so the
        # true wrap lies in (fs-20, fs]. Use the midpoint and carry the bound.
        corpus_tok = (fs - 10) * TPS
        reread = BUDGET - corpus_tok
        corpus_est[(load, seed)] = corpus_tok
        print(f"{rid:<26} {fs:>10} {corpus_tok:>15,} {reread:>14,} {reread/BUDGET:>9.1%}")

print()
print("=" * 82)
print("WHAT THE RE-READ DOES TO EXPOSURES PER FACT")
print("=" * 82)
print("Biographies are ~23% of the mixture. A dense re-read of R tokens replays")
print("~0.23*R biography tokens, raising exposures per fact above the split arm's.")
print()
print(f"{'load':<10} {'entities':>10} {'dense passes':>13} {'split passes':>13} {'dense/split':>12}")
ENT = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}
for load in LOADS:
    ests = [corpus_est[(load, s)] for s in SEEDS if (load, s) in corpus_est]
    if not ests:
        continue
    corpus_tok = sum(ests) / len(ests)
    dense_passes = BUDGET / corpus_tok          # fractional passes over dense corpus
    split_passes = 1.0                          # split never wrapped
    print(f"{load:<10} {ENT[load]:>10,} {dense_passes:>13.3f} {split_passes:>13.3f} "
          f"{dense_passes/split_passes:>11.2f}x")

print()
print("=" * 82)
print("DIRECTION OF THE BIAS")
print("=" * 82)
print("""The dense arm re-read part of its corpus; the split arm did not. Repetition
raises memorisation, and memorisation is the burden the hypothesis says the
split arm is freed from. So the asymmetry inflates the dense arm's burden
relative to split -- it biases the comparison TOWARD the hypothesis.

Dense also saw less unique text, which if anything should hurt it on held-out
reasoning. That also biases toward the hypothesis.

Both channels favour split. The previously reported result was that split shows
no consistent advantage and is 5.5 points BEHIND at the top load. A null that
survives a bias pointing the other way is stronger, not weaker -- but the
imbalance must be declared, because it is a real violation of the matched-twin
design the writeup claims.""")
