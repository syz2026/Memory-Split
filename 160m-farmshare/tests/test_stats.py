"""Tests for evals.stats: paired clustered bootstrap, seed summary, composite."""

import random

import numpy as np
import pytest

from evals.stats import composite, paired_delta, seed_summary


def _row(qid, correct, template):
    return {"qid": qid, "task": "igsm", "correct": bool(correct),
            "pred": "x", "answer": "x", "meta": {"template": template}}


def _synthetic_rows(n_items=2000, n_clusters=20, p_a=0.6, p_b=0.5, seed=7):
    rng = random.Random(seed)
    rows_a, rows_b = [], []
    for i in range(n_items):
        cluster = f"t{rng.randrange(n_clusters)}"
        rows_a.append(_row(f"q{i}", rng.random() < p_a, cluster))
        rows_b.append(_row(f"q{i}", rng.random() < p_b, cluster))
    return rows_a, rows_b


def test_paired_delta_recovers_known_effect():
    rows_a, rows_b = _synthetic_rows()
    out = paired_delta(rows_a, rows_b, n_boot=4000, seed=0)
    # point delta is exactly mean(a) - mean(b)
    exact = (sum(r["correct"] for r in rows_a)
             - sum(r["correct"] for r in rows_b)) / len(rows_a)
    assert out["delta"] == pytest.approx(exact)
    # within sampling noise of the true 0.10
    assert abs(out["delta"] - 0.10) < 0.05
    # 95% CI covers the truth (deterministic given fixed seeds)
    assert out["ci_lo"] < 0.10 < out["ci_hi"]
    assert out["ci_lo"] < out["delta"] < out["ci_hi"]
    assert out["n_items"] == 2000
    assert out["n_clusters"] == 20
    assert out["se"] > 0


def test_paired_delta_deterministic_and_order_insensitive():
    rows_a, rows_b = _synthetic_rows()
    out1 = paired_delta(rows_a, rows_b, n_boot=500, seed=3)
    shuffled_b = list(reversed(rows_b))  # matching is by qid, not position
    out2 = paired_delta(rows_a, shuffled_b, n_boot=500, seed=3)
    assert out1 == out2


def test_paired_delta_rejects_mismatched_qids():
    rows_a, rows_b = _synthetic_rows(n_items=10)
    rows_b[0]["qid"] = "nope"
    with pytest.raises(AssertionError):
        paired_delta(rows_a, rows_b)


def test_paired_delta_accepts_string_cluster_key():
    rows_a, rows_b = _synthetic_rows(n_items=200)
    out_str = paired_delta(rows_a, rows_b, cluster_key="template",
                           n_boot=500, seed=1)
    out_fn = paired_delta(rows_a, rows_b,
                          cluster_key=lambda r: r["meta"]["template"],
                          n_boot=500, seed=1)
    assert out_str == out_fn


def test_clustering_widens_ci_vs_naive_bootstrap():
    # Outcomes perfectly correlated within clusters: 20 clusters x 100 items.
    # Clusters 0-9: both correct (d=0); 10-11: only a correct (d=1);
    # 12-19: both wrong (d=0). a acc=0.6, b acc=0.5, delta=0.1.
    rows_a, rows_b = [], []
    n_clusters, per_cluster = 20, 100
    for c in range(n_clusters):
        a_ok = c < 12
        b_ok = c < 10
        for j in range(per_cluster):
            qid = f"q{c}_{j}"
            rows_a.append(_row(qid, a_ok, f"t{c}"))
            rows_b.append(_row(qid, b_ok, f"t{c}"))
    n_boot = 2000
    out = paired_delta(rows_a, rows_b, n_boot=n_boot, seed=0)
    assert out["delta"] == pytest.approx(0.1)
    clustered_width = out["ci_hi"] - out["ci_lo"]

    # naive item-level bootstrap on the paired diffs, computed here
    d = np.array([int(a["correct"]) - int(b["correct"])
                  for a, b in zip(rows_a, rows_b)], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(0))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    naive_width = float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5))

    assert naive_width > 0
    assert clustered_width >= 1.5 * naive_width


# ---------------------------------------------------------------- seed_summary


def test_seed_summary_consistent_positive():
    out = seed_summary([0.02, 0.05, 0.03])
    assert out["mean"] == pytest.approx(0.1 / 3)
    assert out["sign_consistent"] is True
    assert out["n_seeds"] == 3
    assert out["seed_sigma"] == pytest.approx(np.std([0.02, 0.05, 0.03], ddof=1))


def test_seed_summary_inconsistent_sign():
    out = seed_summary([0.02, -0.01, 0.03])
    assert out["sign_consistent"] is False


def test_seed_summary_zero_breaks_consistency():
    assert seed_summary([0.02, 0.0])["sign_consistent"] is False


def test_seed_summary_single_seed():
    out = seed_summary([-0.04])
    assert out["mean"] == pytest.approx(-0.04)
    assert out["sign_consistent"] is True
    assert out["seed_sigma"] == 0.0
    assert out["n_seeds"] == 1


# ------------------------------------------------------------------- composite


def test_composite_unweighted_mean():
    rows_by_task = {
        "igsm": [_row(f"i{k}", k < 8, "t") for k in range(10)],       # 0.8
        "deduction": [_row(f"d{k}", k < 1, "t") for k in range(2)],   # 0.5
        "factqa": [_row(f"f{k}", True, "t") for k in range(5)],       # ignored
    }
    assert composite(rows_by_task) == pytest.approx((0.8 + 0.5) / 2)


def test_composite_custom_tasks():
    rows_by_task = {"a": [_row("1", True, "t")], "b": [_row("2", False, "t")]}
    assert composite(rows_by_task, tasks=("a", "b")) == pytest.approx(0.5)


# ------------------------------------------------------- figure (smoke only)


def test_dose_response_figure_writes_png(tmp_path):
    from evals.figures import dose_response_figure

    points = [
        {"n_entities": n, "arm": arm, "seed": s, "composite": 0.5 + 0.01 * s}
        for n in (50_000, 200_000, 800_000)
        for arm in ("dense", "split")
        for s in (0, 1)
    ]
    out = dose_response_figure(points, tmp_path / "fig" / "dose.png")
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
