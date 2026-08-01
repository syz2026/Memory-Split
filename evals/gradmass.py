"""Gradient-mass decomposition: which parameters each objective drives.

The behavioural contrast cannot separate capacity reallocation from gradient
interference -- both predict the same sign and the same monotone dose trend.
This is the only measurement in the design that speaks to capacity rather than
to loss composition, and it costs minutes per checkpoint.

At the final checkpoint, accumulate per-parameter squared gradients on
held-out FACT batches and held-out REASONING batches separately. Each
parameter is then attributed to whichever objective drives it harder, and the
report is the share of parameter mass on each side.

If capacity is reallocated, the masked arm should carry measurably more of its
mass in reasoning-dominant directions than its supervised twin. If the two
arms differ behaviourally but not here, the effect is loss composition rather
than reallocation -- which is a finding, and the honest one.

This is the artifact-backed descendant of the participation-ratio result that
was withdrawn for having no artifacts.
"""

from __future__ import annotations

import torch


def _accumulate(model, batches, device) -> dict[str, torch.Tensor]:
    """Sum of squared gradients per parameter over a set of batches."""
    acc: dict[str, torch.Tensor] = {}
    for x, y, w in batches:
        model.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device), weights=w.to(device))
        if loss is None:
            continue
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.detach().float().pow(2)
            if name in acc:
                acc[name] += g2
            else:
                acc[name] = g2
    model.zero_grad(set_to_none=True)
    return acc


def decompose(
    model,
    fact_batches,
    reasoning_batches,
    device,
    include: str | None = None,
) -> dict:
    """Share of parameter mass dominated by each objective.

    `include` optionally restricts to a parameter-name substring, e.g.
    "blocks" to exclude the embedding table -- worth reporting both ways,
    because a tied 40.6M model is 47.6% embedding and crowding in a lookup
    table is not the schema-versus-fact competition the hypothesis is about.
    """
    gf = _accumulate(model, fact_batches, device)
    gr = _accumulate(model, reasoning_batches, device)

    names = [n for n in gf if n in gr]
    if include:
        names = [n for n in names if include in n]
    if not names:
        return {"n_params": 0, "fact_dominant_frac": 0.0,
                "reasoning_dominant_frac": 0.0}

    n_total = 0
    n_fact = 0
    mass_fact = 0.0
    mass_reason = 0.0
    for n in names:
        f, r = gf[n], gr[n]
        n_total += f.numel()
        n_fact += int((f > r).sum())
        mass_fact += float(f.sum())
        mass_reason += float(r.sum())

    total_mass = mass_fact + mass_reason
    return {
        "n_params": n_total,
        "include": include,
        "fact_dominant_frac": n_fact / n_total,
        "reasoning_dominant_frac": (n_total - n_fact) / n_total,
        "fact_mass_share": mass_fact / total_mass if total_mass else 0.0,
        "reasoning_mass_share": mass_reason / total_mass if total_mass else 0.0,
        "note": (
            "Compare arms, not absolute values. A reallocation account "
            "predicts the masked arm carries more mass in reasoning-dominant "
            "directions than its supervised twin; equal shares alongside a "
            "behavioural difference points at loss composition instead."
        ),
    }
