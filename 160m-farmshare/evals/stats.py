"""Paired statistics for arm contrasts: clustered bootstrap, seed summary,
reasoning composite."""

from __future__ import annotations

import numpy as np

DEFAULT_TASKS = ("igsm", "deduction")


def _default_cluster_key(row: dict):
    return row["meta"]["template"]


def paired_delta(
    rows_a: list[dict],
    rows_b: list[dict],
    cluster_key=_default_cluster_key,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired accuracy delta (a - b) with a clustered percentile bootstrap.

    Rows are matched by qid (the two sets must be identical). Per-item paired
    diff d_i = int(a.correct) - int(b.correct); point delta = mean(d). The
    bootstrap resamples CLUSTERS (unique cluster_key values, taken from
    rows_a) with replacement; each replicate's mean weights items by how
    often their cluster was drawn. 95% percentile CI; se = bootstrap sigma
    (ddof=1).

    cluster_key: callable(row) -> hashable, or a string key looked up in
    row["meta"] first, then the row itself.
    """
    if isinstance(cluster_key, str):
        name = cluster_key
        cluster_key = lambda r: r["meta"][name] if name in r["meta"] else r[name]  # noqa: E731

    a_by_qid = {r["qid"]: r for r in rows_a}
    b_by_qid = {r["qid"]: r for r in rows_b}
    assert len(a_by_qid) == len(rows_a), "duplicate qids in rows_a"
    assert len(b_by_qid) == len(rows_b), "duplicate qids in rows_b"
    assert set(a_by_qid) == set(b_by_qid), "rows_a and rows_b qid sets differ"

    qids = sorted(a_by_qid)
    d = np.array(
        [int(bool(a_by_qid[q]["correct"])) - int(bool(b_by_qid[q]["correct"])) for q in qids],
        dtype=np.float64,
    )
    labels = [cluster_key(a_by_qid[q]) for q in qids]
    uniq = sorted(set(labels), key=repr)
    index = {c: i for i, c in enumerate(uniq)}
    n_clusters = len(uniq)
    sums = np.zeros(n_clusters)
    counts = np.zeros(n_clusters)
    for di, lab in zip(d, labels):
        sums[index[lab]] += di
        counts[index[lab]] += 1

    rng = np.random.Generator(np.random.PCG64(seed))
    draw = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta": float(d.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_items": int(d.size),
        "n_clusters": n_clusters,
        "se": float(boot.std(ddof=1)),
    }


def seed_summary(per_seed_deltas: list[float]) -> dict:
    """Cross-seed summary of one contrast; sign_consistent requires every
    delta strictly on the same side of zero (a zero delta breaks it)."""
    assert per_seed_deltas, "need at least one seed delta"
    n = len(per_seed_deltas)
    return {
        "mean": float(np.mean(per_seed_deltas)),
        "sign_consistent": all(x > 0 for x in per_seed_deltas)
        or all(x < 0 for x in per_seed_deltas),
        "seed_sigma": float(np.std(per_seed_deltas, ddof=1)) if n >= 2 else 0.0,
        "n_seeds": n,
    }


def composite(rows_by_task: dict[str, list[dict]], tasks=DEFAULT_TASKS) -> float:
    """Unweighted mean of per-task accuracies (the primary endpoint)."""
    accs = []
    for task in tasks:
        rows = rows_by_task[task]
        if not rows:
            raise ValueError(f"no rows for task {task!r}")
        accs.append(sum(bool(r["correct"]) for r in rows) / len(rows))
    return float(np.mean(accs))
