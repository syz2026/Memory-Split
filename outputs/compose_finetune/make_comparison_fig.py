"""Render the dense-closed / dense-open / split comparison figure (aligned vs novel)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

groups = ["Aligned OOD\n(entities seen as facts,\ncomposition held out)",
          "Novel OOD\n(entities never seen\nat all)"]
arms = ["dense — closed-book", "dense — open (facts in context)", "split — external store"]
vals = {
    "dense — closed-book":            [86.6, 0.5],
    "dense — open (facts in context)":[94.2, 0.8],
    "split — external store":         [99.8, 53.5],
}
colors = ["#cb181d", "#fd8d3c", "#2c7fb8"]

x = np.arange(len(groups)); w = 0.26
fig, ax = plt.subplots(figsize=(11, 6.2))
for i, (arm, c) in enumerate(zip(arms, colors)):
    bars = ax.bar(x + (i - 1) * w, vals[arm], w, label=arm, color=c, edgecolor="white")
    for b, v in zip(bars, vals[arm]):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel("two-hop composition accuracy (%)", fontsize=12)
ax.set_ylim(0, 108)
ax.set_title("Two-hop composition (n50k, d160m): dense vs split, seen vs novel entities\n"
             "chance ≈ 0.5%", fontsize=13)
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(axis="y", alpha=0.25)

# takeaway annotation over the novel group
ax.annotate("On NOVEL entities, dense fails even when handed the facts (0.8%);\n"
            "only the external store composes (53.5%). → the advantage is architectural,\n"
            "not just \u201chaving the facts.\u201d",
            xy=(1.0, 53.5), xytext=(0.30, 74),
            fontsize=9.5, color="#08306b",
            arrowprops=dict(arrowstyle="->", color="#08306b", lw=1.2))
# takeaway over aligned
ax.annotate("On SEEN entities all three do well\n(dense memorized the facts).",
            xy=(0.0, 99.8), xytext=(-0.45, 60), fontsize=9.5, color="#333")

fig.tight_layout()
out = HERE / "arm_comparison_fig.png"
fig.savefig(out, dpi=160); print("wrote", out)
