"""H4 learning-curve dynamics from existing checkpoints (no retraining).

Pure, torch-free helpers (offline-testable) that turn per-snapshot eval
summaries and training ``log.jsonl`` files into (a) per-arm metric trajectories
vs training tokens and (b) a tokens-to-milestone table with interpolated
crossings. See ``replication/specs/nr2-h4-learning-curves.md``.

Nothing here loads a model or runs a forward pass — it only reads JSON the eval
battery already wrote, so it is fully deterministic and unit-testable without a
GPU or checkpoints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_STEP_RE = re.compile(r"step0*(\d+)")

# ---- H4 milestone config: the SINGLE source of truth shared by the battery
# reducer (scripts/analyze.py) and the single-experiment plotter
# (scripts/plot_run_curves.py), so the two never drift (NR-2 reconciliation).

# Pre-declared tokens-to-milestone grids (NR-2 spec 5.5). Chance is ~0.043
# (iGSM) / ~0.50 (deduction); 40-80% is the capacity-sensitive band (L15).
H4_THRESHOLDS = {
    "composite_knowledge_free": [0.20, 0.40, 0.60],
    "igsm.acc": [0.10, 0.20, 0.40, 0.60, 0.80],
    "deduction.acc": [0.60, 0.70, 0.80, 0.90],
}

# higher_is_better per metric (accuracy/recall rise; loss/NLL fall).
METRIC_DIRECTION = {
    "composite_knowledge_free": True,
    "igsm.acc": True,
    "deduction.acc": True,
    "recall.on": True,
    "recall.closed": True,
    "loss": False,
    "loss_masked_values": False,
}

# one representative milestone per metric for single-run plots.
PRIMARY_MILESTONE = {
    "composite_knowledge_free": 0.5,
    "igsm.acc": 0.5,
    "deduction.acc": 0.75,
    "recall.on": 0.9,
    "recall.closed": 0.5,
    "loss": 2.5,
    "loss_masked_values": 10.83,  # ~ln(50304): the uniform-CE ceiling (L24)
}


def snapshot_step(name: str) -> int | None:
    """Parse the training step out of a snapshot/ckpt tag.

    ``"step0000500"`` / ``"snapshots/step0000500.pt"`` / ``"ckpt-step0000500"``
    all -> 500. Returns None when no ``stepN`` token is present (e.g. the final
    ``ckpt.pt``).
    """
    m = _STEP_RE.search(str(name))
    return int(m.group(1)) if m else None


def _dotted(d: dict, key: str):
    """Look up ``key`` in ``d``; supports one level of dotting (``igsm.acc``).

    Returns None if any segment is missing or not a dict where a dict is needed.
    """
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _summary_point(summary: dict, key: str, fallback_step: int | None) -> dict | None:
    """Build a curve point {step, tokens, value} from one summary, or None."""
    value = _dotted(summary, key)
    if value is None:
        return None
    step = summary.get("step")
    if step is None:
        step = fallback_step
    if step is None:
        return None
    return {"step": int(step), "tokens": summary.get("tokens"), "value": float(value)}


def collect_metric_curve(run_dir, key: str) -> list[dict]:
    """Collect ``[{step, tokens, value}]`` for one metric across all checkpoints.

    Reads the final ``<run>/evals/summary.json`` plus every snapshot summary at
    ``<run>/evals/ckpt-*/summary.json`` (the non-colliding layout that
    ``run_evals.py`` writes for ``--ckpt`` runs). ``key`` may be dotted
    (``"igsm.acc"``) or flat (``"composite_knowledge_free"``). Points whose
    metric or step is missing are skipped; the result is sorted by step and
    de-duplicated (last write wins per step).
    """
    run_dir = Path(run_dir)
    evals = run_dir / "evals"
    points: dict[int, dict] = {}

    final = evals / "summary.json"
    if final.exists():
        summ = json.loads(final.read_text())
        # a final-ckpt summary may lack an explicit step; treat as the largest.
        p = _summary_point(summ, key, fallback_step=summ.get("step"))
        if p is not None:
            points[p["step"]] = p
        elif _dotted(summ, key) is not None:
            # value present but no step anywhere -> park it at +inf-ish sentinel
            points.setdefault(-1, {"step": None, "tokens": summ.get("tokens"),
                                   "value": float(_dotted(summ, key))})

    for sub in sorted(evals.glob("ckpt-*")):
        s = sub / "summary.json"
        if not s.exists():
            continue
        summ = json.loads(s.read_text())
        p = _summary_point(summ, key, fallback_step=snapshot_step(sub.name))
        if p is not None:
            points[p["step"]] = p

    ordered = [points[k] for k in sorted(points) if k is not None and k >= 0]
    return ordered


def tokens_to_threshold(
    points: list[dict],
    threshold: float,
    xkey: str = "tokens",
    higher_is_better: bool = True,
) -> dict:
    """First crossing of ``threshold`` with linear interpolation between snapshots.

    ``points``: ``[{step, tokens, value}]`` (from ``collect_metric_curve``),
    assumed sorted by step. Uses ``xkey`` ("tokens" or "step") for the x-axis; if
    a point's ``tokens`` is None it falls back to ``step`` for that point.

    Returns ``{reached, x_cross, x_key, final_value, n_points}``:
      - ``reached``  : the threshold was met at some point.
      - ``x_cross``  : interpolated x at the FIRST upward crossing (for
        ``higher_is_better``); equal to the first x when already at/above at the
        start; None when never reached.
      - ``final_value`` : the last curve value (diagnostic).

    ``higher_is_better=False`` flips the comparison (for NLL-style metrics: the
    milestone is value <= threshold).
    """
    xs, ys = [], []
    for p in points:
        x = p.get(xkey)
        if x is None:
            x = p.get("step")
        if x is None:
            continue
        xs.append(float(x))
        ys.append(float(p["value"]))

    out = {
        "reached": False,
        "x_cross": None,
        "x_key": xkey,
        "final_value": (ys[-1] if ys else None),
        "n_points": len(ys),
    }
    if not ys:
        return out

    def meets(v: float) -> bool:
        return v >= threshold if higher_is_better else v <= threshold

    if meets(ys[0]):
        out["reached"] = True
        out["x_cross"] = xs[0]
        return out

    for i in range(1, len(ys)):
        if meets(ys[i]):
            out["reached"] = True
            y0, y1 = ys[i - 1], ys[i]
            x0, x1 = xs[i - 1], xs[i]
            # linear interpolation of the crossing; guard a flat segment.
            if y1 == y0:
                out["x_cross"] = x1
            else:
                frac = (threshold - y0) / (y1 - y0)
                frac = min(1.0, max(0.0, frac))
                out["x_cross"] = x0 + frac * (x1 - x0)
            return out
    return out


def log_series(log_path, ykey: str, xkey: str = "step") -> list[tuple[float, float]]:
    """Pull ``(x, y)`` pairs from a training ``log.jsonl`` for one key.

    Rows lacking ``ykey`` (or ``xkey``) are skipped, so requesting
    ``loss_masked_values`` on a log that never recorded it returns ``[]`` rather
    than raising. Order follows the file (chronological).
    """
    path = Path(log_path)
    out: list[tuple[float, float]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if ykey in row and xkey in row:
            out.append((float(row[xkey]), float(row[ykey])))
    return out
