"""Recall probes (H2/H3) and bits-in-weights accounting."""

from __future__ import annotations

import math

from evals.generate import generate_batch_with_stats
from evals.scorers import normalize_answer

MODES = ("closed", "on", "off")


def recall_accuracy(
    model,
    tok,
    probes,
    mode: str,
    organizer,
    device,
    max_new: int = 48,
    batch_size: int = 64,
) -> dict:
    """Score recall probes under one of three store modes.

    mode="closed": dense arm / no interception (organizer forced to None).
    mode="on":     split arm with the store attached (organizer required).
    mode="off":    split arm unplugged — identical call to "closed" (the model
                   may still emit DB_* junk, which is left uninterpreted);
                   both names are kept for reporting clarity.

    A probe counts correct iff normalize_answer(probe.answer) appears as a
    plain substring of the normalized generated continuation (values can land
    mid-sentence after lookups). Probe meta must carry {"relation": attr}.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "on":
        if organizer is None:
            raise ValueError('mode="on" requires an organizer')
        org = organizer
    else:
        org = None

    total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    per_attr_counts: dict[str, list[int]] = {}
    n_correct = 0
    for lo in range(0, len(probes), batch_size):
        chunk = probes[lo : lo + batch_size]
        texts, stats = generate_batch_with_stats(
            model, tok, [p.prompt for p in chunk], max_new, org, device
        )
        for k in total:
            total[k] += stats[k]
        for probe, gen in zip(chunk, texts):
            correct = normalize_answer(probe.answer) in normalize_answer(gen)
            n_correct += correct
            attr = probe.meta["relation"]
            hit_total = per_attr_counts.setdefault(attr, [0, 0])
            hit_total[0] += correct
            hit_total[1] += 1

    n = len(probes)
    return {
        "mode": mode,
        "overall": n_correct / n if n else 0.0,
        "per_attribute": {a: c / t for a, (c, t) in per_attr_counts.items()},
        "n": n,
        "stats": total,
    }


def bits_in_weights(
    per_attribute_acc: dict[str, float],
    n_entities: int,
    pool_sizes: dict[str, float],
) -> float:
    """Simplified Physics-of-LM-3.3-style bits accounting.

        bits = sum_a max(0, (acc_a - g_a) / (1 - g_a)) * N * log2(|V_a|)

    with g_a = 1 / |V_a| the per-attribute guess rate. Accuracy at or below
    chance contributes 0 (clamped); perfect accuracy contributes the full
    N * log2(|V_a|) bits. Pool sizes may be non-integer (e.g. 27759 days in
    the birth-date range); they enter only through g_a and log2.
    """
    total = 0.0
    for attr, acc in per_attribute_acc.items():
        pool = float(pool_sizes[attr])
        g = 1.0 / pool
        stored_frac = max(0.0, (acc - g) / (1.0 - g))
        total += stored_frac * n_entities * math.log2(pool)
    return total
