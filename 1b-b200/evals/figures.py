"""The dose-response figure: reasoning composite vs fact load, both arms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_ARM_COLORS = {"dense": "tab:red", "split": "tab:blue"}


def dose_response_figure(points: list[dict], out_png) -> Path:
    """Plot per-arm mean lines over fact load with per-seed scatter.

    points rows: {"n_entities": int, "arm": "dense"|"split", "seed": int,
    "composite": float}. Returns the written path.
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arm in sorted({p["arm"] for p in points}):
        arm_pts = [p for p in points if p["arm"] == arm]
        color = _ARM_COLORS.get(arm)
        loads = sorted({p["n_entities"] for p in arm_pts})
        means = [
            float(np.mean([p["composite"] for p in arm_pts if p["n_entities"] == x]))
            for x in loads
        ]
        (line,) = ax.plot(loads, means, marker="o", label=arm, color=color)
        ax.scatter(
            [p["n_entities"] for p in arm_pts],
            [p["composite"] for p in arm_pts],
            color=line.get_color(),
            s=18,
            alpha=0.45,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("distinct entities (fact load)")
    ax.set_ylabel("reasoning composite accuracy")
    ax.legend(title="arm")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
