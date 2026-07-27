"""The dose-response figure: reasoning composite vs fact load, both arms."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_ARM_COLORS = {"dense": "tab:red", "split": "tab:blue"}
_PLOT_VALUE_FIELDS = (
    "delta",
    "interaction",
    "accuracy",
    "value",
    "pair_accuracy",
    "high_delta",
    "low_delta",
)


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


def _plot_value(row: Mapping[str, Any]) -> tuple[str, float] | None:
    for name in _PLOT_VALUE_FIELDS:
        value = row.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return name, float(value)
    for name in sorted(row):
        value = row[name]
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return name, float(value)
    return None


def _plot_label(row: Mapping[str, Any], index: int) -> str:
    fields = (
        "contrast",
        "seed",
        "arm",
        "memory_mode",
        "control",
        "guard",
        "check",
        "dataset",
        "metric",
        "hop",
        "composition",
    )
    parts = [str(row[name]) for name in fields if name in row]
    return " · ".join(parts) if parts else f"row {index + 1}"


def render_plot_svg(
    section: str,
    rows: Sequence[Mapping[str, Any]],
    out_svg: str | Path,
) -> Path:
    """Render a deterministic, metadata-free SVG for one report section."""

    if not isinstance(section, str) or not section:
        raise ValueError("plot section must be a non-empty string")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("plot rows must be a sequence")
    points: list[tuple[str, str, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("plot rows must contain mappings")
        measured = _plot_value(row)
        if measured is not None:
            field, value = measured
            points.append((_plot_label(row, index), field, value))

    width = 760
    height = max(260, 94 + 34 * max(1, len(points)))
    left = 250
    right = 34
    plot_width = width - left - right
    values = [value for _label, _field, value in points]
    lower = min([0.0, *values])
    upper = max([0.0, *values])
    if lower == upper:
        lower -= 1.0
        upper += 1.0

    def x(value: float) -> float:
        return left + (value - lower) / (upper - lower) * plot_width

    title = html.escape(section.replace("_", " ").title())
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,'
            'monospace;fill:#17202a}.title{font-size:18px;font-weight:600}'
            '.label{font-size:11px}.axis{stroke:#7f8c8d;stroke-width:1}'
            '.mark{fill:#21618c}.value{font-size:10px}</style>'
        ),
        f'<text class="title" x="24" y="32">{title}</text>',
        (
            f'<line class="axis" x1="{x(0.0):.3f}" y1="54" '
            f'x2="{x(0.0):.3f}" y2="{height - 30}"/>'
        ),
        (
            f'<text class="value" x="{left}" y="{height - 10}">'
            f'{lower:.6g}</text>'
        ),
        (
            f'<text class="value" text-anchor="end" x="{width - right}" '
            f'y="{height - 10}">{upper:.6g}</text>'
        ),
    ]
    if not points:
        lines.append(
            '<text class="label" x="24" y="78">No observations</text>'
        )
    for index, (label, field, value) in enumerate(points):
        y = 72 + 34 * index
        escaped_label = html.escape(label)
        escaped_field = html.escape(field)
        lines.extend(
            [
                (
                    f'<text class="label" text-anchor="end" x="{left - 12}" '
                    f'y="{y + 4}">{escaped_label}</text>'
                ),
                (
                    f'<line class="axis" x1="{left}" y1="{y}" '
                    f'x2="{width - right}" y2="{y}" opacity="0.22"/>'
                ),
                (
                    f'<circle class="mark" cx="{x(value):.3f}" cy="{y}" '
                    'r="4"/>'
                ),
                (
                    f'<text class="value" x="{x(value) + 8:.3f}" '
                    f'y="{y + 4}">{escaped_field}={value:.6g}</text>'
                ),
            ]
        )
    lines.append("</svg>")
    destination = Path(out_svg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
