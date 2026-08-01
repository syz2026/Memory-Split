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


# --------------------------------------------------------------------------
# Seed-level inference and the frozen verdict.
#
# The statistical unit is the training seed. Item bootstraps quantify
# evaluation noise only; they say nothing about run-to-run variance, which is
# the noise that matters for a training-level intervention.
# --------------------------------------------------------------------------

import math  # noqa: E402


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, iters: int = 300) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta. Kept local so the analysis has no scipy
    dependency: a frozen verdict should not move because a wheel did."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        b * math.log(1 - x) + a * math.log(x) - _log_beta(b, a)
    ) * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: int) -> float:
    """P(T > t) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    p = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return p if t > 0 else 1.0 - p


def t_ppf(p: float, df: int) -> float:
    """Inverse CDF by bisection. Enough precision for a decision rule."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1.0 - t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_delta_iid(
    rows_a: list[dict],
    rows_b: list[dict],
    metric: str = "correct",
    lower_is_better: bool = False,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Item-level paired delta on an arbitrary metric, matched by qid.

    Rows missing the metric on either side are dropped pairwise rather than
    imputed, because the continuous metrics are undefined for items with no
    worked solution and imputing would invent data.

    This quantifies EVALUATION noise. It is not the inference unit.
    """
    a = {r["qid"]: r for r in rows_a}
    b = {r["qid"]: r for r in rows_b}
    qids = [q for q in a if q in b
            and a[q].get(metric) is not None and b[q].get(metric) is not None]
    if not qids:
        return {"delta": 0.0, "n": 0, "ci": (0.0, 0.0), "se": 0.0}
    sign = -1.0 if lower_is_better else 1.0
    d = np.array([sign * (float(a[q][metric]) - float(b[q][metric])) for q in qids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "n": len(d),
        "n_dropped": len(set(a) & set(b)) - len(d),
        "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "se": float(boots.std(ddof=1)),
    }


def seed_contrast(values: list[float], alpha: float = 0.05) -> dict:
    """Inference across training seeds on the per-seed paired contrast.

    One-sided by preregistration: the hypothesis predicts a positive effect,
    and a negative one is reported descriptively rather than tested. The exact
    sign test is reported alongside because at these sample sizes its floor is
    informative -- at n=3 the smallest attainable one-sided p is 0.125, so a
    perfectly consistent result still cannot clear 0.05 on sign alone.
    """
    n = len(values)
    mean = float(np.mean(values)) if n else 0.0
    if n < 2:
        return {
            "n": n, "mean": mean, "sd": float("nan"), "se": float("nan"),
            "t": float("nan"), "p_one_sided": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "sign_consistent": n == 1 and mean != 0,
            "p_sign": float("nan"),
            "underpowered": True,
        }
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = mean / se if se > 0 else float("inf") if mean > 0 else 0.0
    crit = t_ppf(1 - alpha, df)
    n_pos = sum(1 for v in values if v > 0)
    # Exact one-sided binomial tail at p=0.5.
    p_sign = sum(
        math.comb(n, k) for k in range(n_pos, n + 1)
    ) / (2 ** n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": float(t),
        "df": df,
        "p_one_sided": float(t_sf(t, df)),
        "ci_lower": mean - crit * se,          # one-sided lower bound
        "ci_upper": mean + crit * se,          # one-sided upper bound
        "sign_consistent": n_pos == n or n_pos == 0,
        "n_positive": n_pos,
        "p_sign": p_sign,
        "min_attainable_p_sign": 1.0 / (2 ** n),
    }


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05,
                          power: float = 0.80) -> float:
    """The smallest true effect this design can detect, one-sided.

    Reported instead of asserting power for a round number. The first draft
    assumed a 2.0-point effect nobody had bounded; the pilot's NOFACT run
    measures the ceiling and this converts it into a design decision.
    """
    if n < 2 or sd <= 0:
        return float("nan")
    df = n - 1
    # Normal approximation to the noncentrality, adequate for sizing.
    z_a, z_b = t_ppf(1 - alpha, df), t_ppf(power, df)
    return (z_a + z_b) * sd / math.sqrt(n)


# Frozen thresholds. Set before the confirmatory matrix runs and not movable
# afterwards. The minimum interesting effect is NOT a round number: it is
# derived from the pilot's measured `delta`, the total reasoning cost of
# carrying the fact load, which upper-bounds any achievable treatment effect.
VALIDITY = "validity"
REPORTING = "reporting"


def _gate(name, kind, ok, got, want, note=""):
    return {"name": name, "kind": kind, "ok": bool(ok),
            "observed": got, "required": want, "note": note}


def check_gates(
    bits: dict[str, float],
    bits_lower_bound: dict[str, float],
    igsm_acc_sup: float,
    clip_ratio: dict[str, float],
    storage_floor: float,
    igsm_band: tuple[float, float],
    clip_band: float = 0.05,
) -> list[dict]:
    """Split by consequence.

    Validity gates decide whether the experiment happened at all: if the
    supervised arm never carried a memorisation burden, or the masked arm
    kept it anyway, or the endpoint sat at its floor, there is nothing to
    interpret and the answer is `invalid`.

    Reporting requirements are disclosed rather than fatal. Five all-or-
    nothing gates make `invalid` uncomfortably likely, and the riskiest of
    them -- whether the control matched on difficulty -- is the one whose
    failure is itself a finding worth publishing.
    """
    gates = [
        _gate("sup_carries_a_burden", VALIDITY,
              bits_lower_bound.get("sup", 0.0) > storage_floor,
              bits_lower_bound.get("sup"), f"> {storage_floor}",
              "95% lower bound on SUP recoverable bits/param"),
        _gate("factmask_removed_it", VALIDITY,
              bits.get("factmask", 1.0) < 0.10 * max(1e-12, bits.get("sup", 0.0)),
              bits.get("factmask"), f"< 10% of SUP ({bits.get('sup')})",
              "values stay in the input context, so leakage is possible"),
        _gate("endpoint_is_measurable", VALIDITY,
              igsm_band[0] <= igsm_acc_sup <= igsm_band[1],
              igsm_acc_sup, f"in [{igsm_band[0]}, {igsm_band[1]}]",
              "against the empirical majority-class baseline, not 1/23"),
        _gate("randpos_kept_the_burden", REPORTING,
              abs(bits.get("randpos", 0.0) - bits.get("sup", 0.0))
              <= 0.20 * max(1e-12, bits.get("sup", 0.0)),
              bits.get("randpos"), f"within 20% of SUP ({bits.get('sup')})",
              "the control masks non-values, so its storage should match SUP"),
        _gate("clipping_is_comparable", REPORTING,
              (max(clip_ratio.values()) - min(clip_ratio.values())) <= clip_band
              if clip_ratio else False,
              clip_ratio, f"spread <= {clip_band} absolute",
              "a binding clip scales the arms differently"),
    ]
    return gates


def verdict(
    per_seed_effects: list[float],
    gates: list[dict],
    min_interesting_effect: float,
    alpha: float = 0.05,
) -> dict:
    """validated | rejected | inconclusive | invalid.

    `rejected` means a practical null: the one-sided upper bound sits below
    the minimum interesting effect, so an effect worth caring about has been
    excluded. It does not mean "no effect of any size", which no finite
    design can establish.
    """
    failed_validity = [g for g in gates if g["kind"] == VALIDITY and not g["ok"]]
    failed_reporting = [g for g in gates if g["kind"] == REPORTING and not g["ok"]]
    stat = seed_contrast(per_seed_effects, alpha=alpha)

    if failed_validity:
        label = "invalid"
    elif stat["n"] < 2:
        label = "inconclusive"
    elif stat["ci_lower"] > min_interesting_effect:
        label = "validated"
    elif stat["ci_upper"] < min_interesting_effect:
        label = "rejected"
    else:
        label = "inconclusive"

    return {
        "verdict": label,
        "statistic": stat,
        "min_interesting_effect": min_interesting_effect,
        "failed_validity_gates": [g["name"] for g in failed_validity],
        "disclosed_reporting_failures": [g["name"] for g in failed_reporting],
        "gates": gates,
    }
