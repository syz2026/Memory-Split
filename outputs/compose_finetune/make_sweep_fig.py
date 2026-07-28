"""OOD-vs-step (sample efficiency): dense vs split on the aligned compose corpus."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TOK = 524288  # tokens per step

dense_steps = [47,94,141,188,235,282,329,376,423,470,517,564,611,658,705,752,799,846,893,940]
dense_ood   = [0.9,0.2,0.7,12.5,53.7,61.1,64.0,70.0,76.0,75.2,78.9,77.7,81.7,82.9,86.3,86.5,86.3,85.4,86.3,86.6]
# split sweep was stopped early (already at ceiling); evaluated snapshots:
split_steps = [47,94,141,940]
split_ood   = [99.8,100.0,100.0,99.8]

fig, ax = plt.subplots(figsize=(10.5, 6))
ax.plot([s*TOK/1e6 for s in split_steps], split_ood, "o-", color="#2c7fb8", lw=2.5,
        ms=8, label="split — external store")
ax.plot([s*TOK/1e6 for s in dense_steps], dense_ood, "s-", color="#cb181d", lw=2.5,
        ms=6, label="dense — closed-book")

ax.axhline(99.2, color="#2c7fb8", ls=":", lw=1, alpha=0.6)
ax.axhline(91.8, color="#cb181d", ls=":", lw=1, alpha=0.6)
ax.text(490, 100.3, "from-scratch split anchor 99.2", color="#2c7fb8", fontsize=8)
ax.text(490, 92.4, "from-scratch dense anchor 91.8", color="#cb181d", fontsize=8)

ax.annotate("split at ceiling by ~25M tokens\n(inherited the lookup module;\nonly had to learn to copy the bridge)",
            xy=(47*TOK/1e6, 99.8), xytext=(60, 66), fontsize=9, color="#08306b",
            arrowprops=dict(arrowstyle="->", color="#08306b"))
ax.annotate("dense ramps slowly\n(must learn recall + chaining\nfrom scratch), plateaus ~86%",
            xy=(700*TOK/1e6, 86.3), xytext=(210, 40), fontsize=9, color="#67000d",
            arrowprops=dict(arrowstyle="->", color="#67000d"))

ax.set_xlabel("finetune tokens (millions)   [1 step ≈ 0.52M tokens]", fontsize=11)
ax.set_ylabel("OOD two-hop accuracy (%)", fontsize=12)
ax.set_title("Sample efficiency: split converges ~10–15× faster than dense\n"
             "two-hop composition, aligned OOD (n50k, d160m)", fontsize=13)
ax.set_ylim(0, 105); ax.grid(alpha=0.25); ax.legend(loc="center right", fontsize=11)
fig.tight_layout()
out = HERE / "sweep_ood_vs_step.png"
fig.savefig(out, dpi=160); print("wrote", out)
