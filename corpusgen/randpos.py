"""RANDPOS: the control sidecar that masks matched non-value spans.

The primary contrast is FACTMASK minus RANDPOS, so RANDPOS is what separates
"masking these particular targets helped" from "masking anything helped". All
three independent reviews of this design named it the weakest link, for one
reason: equal token count is not equal difficulty.

Fact values carry roughly 2.85 bits per token. Template text in a fixed
biography frame is near-deterministic, well under 0.5 bits per token. A
count-matched control therefore removes several times less loss mass and
gradient magnitude than the treatment, and with gradient clipping in play that
alone can move the endpoint. So spans are matched on four axes:

    count      exactly as many zeroed targets as FACTMASK
    length     the same span-length histogram
    position   the same within-document relative positions
    NLL        the same mean per-token difficulty, when a pilot NLL table is
               supplied

The fourth is the one that matters and the one that can fail. If template
positions cannot be matched to within the preregistered band, RANDPOS is
empirically invalid and `match_report` says so. That is a result to publish,
not a gate to paper over.

A second hazard the length/position constraints create on their own: the
feasible set in a ~75-token document with ~18.6 value tokens is small, so
spans drift onto the tokens immediately before each value -- the cue phrases
that predict it. Masking those removes fact-relevant supervision and biases
the control toward the treatment. `match_report` measures that overlap.
"""

from __future__ import annotations

import random

import numpy as np

# Tokens immediately preceding a value are its cue phrase ("was born in ").
CUE_WINDOW = 3
# Preregistered tolerance on mean masked-span NLL, treatment vs control.
NLL_TOLERANCE = 0.20


def spans_of(mask: np.ndarray, value: int = 0) -> list[tuple[int, int]]:
    """Contiguous runs equal to `value`, as (start, length)."""
    out: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m == value and start is None:
            start = i
        elif m != value and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def _forbidden(n: int, fact_spans: list[tuple[int, int]]) -> np.ndarray:
    """Positions RANDPOS may not occupy: the value tokens themselves."""
    bad = np.zeros(n, dtype=bool)
    for s, ln in fact_spans:
        bad[s : s + ln] = True
    return bad


def build(
    ids: np.ndarray,
    factmask: np.ndarray,
    rng: random.Random,
    token_nll: np.ndarray | None = None,
    position_tolerance: float = 0.15,
) -> np.ndarray:
    """A sidecar with the same zero count and span lengths as `factmask`,
    placed on non-value tokens at matched relative positions.

    `token_nll`, when given, is a per-token difficulty table from a pilot
    supervised run over disjoint data. Among candidate placements inside the
    position tolerance, the one whose mean NLL is closest to the target span's
    is chosen. Frozen before use: selecting placements with NLL measured on
    the run being analysed would leak the outcome into the design.
    """
    n = len(factmask)
    out = np.ones(n, dtype=np.uint8)
    fact_spans = spans_of(factmask)
    if not fact_spans:
        return out
    forbidden = _forbidden(n, fact_spans)
    taken = np.zeros(n, dtype=bool)

    # Longest first: long spans have the fewest legal homes.
    order = sorted(range(len(fact_spans)), key=lambda i: -fact_spans[i][1])
    for i in order:
        start, length = fact_spans[i]
        target_rel = start / max(1, n)
        target_nll = (
            float(np.mean(token_nll[start : start + length]))
            if token_nll is not None
            else None
        )
        candidates = []
        for cand in range(0, n - length + 1):
            window = slice(cand, cand + length)
            if forbidden[window].any() or taken[window].any():
                continue
            rel = cand / max(1, n)
            if abs(rel - target_rel) > position_tolerance:
                continue
            candidates.append(cand)
        if not candidates:
            # Fall back to any legal placement, nearest in relative position.
            for cand in range(0, n - length + 1):
                window = slice(cand, cand + length)
                if not forbidden[window].any() and not taken[window].any():
                    candidates.append(cand)
        if not candidates:
            continue  # document too dense to place this span; count check reports it
        if target_nll is not None and token_nll is not None:
            best = min(
                candidates,
                key=lambda c: abs(
                    float(np.mean(token_nll[c : c + length])) - target_nll
                ),
            )
        else:
            best = rng.choice(candidates)
        out[best : best + length] = 0
        taken[best : best + length] = True
    return out


def match_report(
    factmask: np.ndarray,
    randpos: np.ndarray,
    token_nll: np.ndarray | None = None,
) -> dict:
    """How well the control matches the treatment, on every axis that matters.

    `nll_within_tolerance` is the honest gate. If it is False the control is
    not difficulty-matched and the contrast is confounded by loss mass; report
    that rather than suppressing it.
    """
    f_spans = spans_of(factmask)
    r_spans = spans_of(randpos)
    n = len(factmask)

    f_lens = sorted(ln for _, ln in f_spans)
    r_lens = sorted(ln for _, ln in r_spans)
    f_rel = sorted(s / max(1, n) for s, _ in f_spans)
    r_rel = sorted(s / max(1, n) for s, _ in r_spans)

    # Distance from each control span to the nearest value token, and how much
    # of the control lands inside a value's cue window.
    value_pos = np.flatnonzero(factmask == 0)
    cue = np.zeros(n, dtype=bool)
    for s, _ in f_spans:
        cue[max(0, s - CUE_WINDOW) : s] = True
    overlap = int(((randpos == 0) & cue).sum())

    distances = []
    if value_pos.size:
        for s, ln in r_spans:
            mid = s + ln // 2
            distances.append(int(np.min(np.abs(value_pos - mid))))

    out = {
        "n_zeros_fact": int((factmask == 0).sum()),
        "n_zeros_randpos": int((randpos == 0).sum()),
        "count_matched": int((factmask == 0).sum()) == int((randpos == 0).sum()),
        "n_spans_fact": len(f_spans),
        "n_spans_randpos": len(r_spans),
        "length_histogram_matched": f_lens == r_lens,
        "mean_relative_position_fact": float(np.mean(f_rel)) if f_rel else 0.0,
        "mean_relative_position_randpos": float(np.mean(r_rel)) if r_rel else 0.0,
        "overlaps_value_span": int(((randpos == 0) & (factmask == 0)).sum()),
        "cue_window_overlap_tokens": overlap,
        "cue_window_overlap_frac": overlap / max(1, int((randpos == 0).sum())),
        "median_distance_to_nearest_value": (
            float(np.median(distances)) if distances else None
        ),
    }
    if token_nll is not None:
        f_nll = float(np.mean(token_nll[factmask == 0])) if (factmask == 0).any() else 0.0
        r_nll = float(np.mean(token_nll[randpos == 0])) if (randpos == 0).any() else 0.0
        out["mean_nll_fact"] = f_nll
        out["mean_nll_randpos"] = r_nll
        rel = abs(r_nll - f_nll) / max(1e-9, f_nll)
        out["nll_relative_gap"] = rel
        out["nll_within_tolerance"] = rel <= NLL_TOLERANCE
    return out
