"""Causal mechanistic-interpretability harness (ready-to-run).

Upgrades the correlational NR-5 (`evals/mechanism.py`: selectivity + linear
probing) with CAUSAL tools:

  * `ablate_mlp_neurons` / `neuron_ablation_effect` — zero specific MLP hidden
    units and measure the effect on any scorer. Usable NOW on the fact side:
    ablate the dense arm's top memorization neurons (from `mechanism.top_neurons`)
    and watch recall collapse while reasoning is untouched — a causal
    capacity-competition test.
  * `activation_patch_logits` — patch a layer's residual from a source run into a
    target run at chosen positions (causal tracing / activation patching).
  * `capture_residual_at` — grab residual states at arbitrary token positions for
    a reasoning-FEATURE probe (reuse `mechanism.fit_linear_probe`).

All operate via forward hooks; no edit to `train/model.py`. The reasoning-circuit
uses (L10) are BLOCKED until reasoning clears chance (nothing to probe/patch at
4%), but the harness is generic and tested now so it runs the moment P0 lands.

Note on the model contract: `Block.forward` returns a tuple `(x, new_kv)`, so
block forward-hooks see a tuple output — handled below.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import torch


def _block_out(o):
    return o[0] if isinstance(o, tuple) else o


@contextmanager
def ablate_mlp_neurons(model, neurons):
    """Zero specific MLP hidden units for the duration of the context.

    ``neurons``: iterable of ``(layer, neuron)`` in the SwiGLU gated-activation
    space (the input to ``block.mlp.w2`` — the same index space as
    ``mechanism.top_neurons``). Uses a forward_pre_hook that clones the input and
    zeros the selected columns (clone avoids in-place autograd hazards).
    """
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, neuron in neurons:
        by_layer[layer].append(neuron)
    handles = []

    def mk(idxs):
        idx = torch.tensor(sorted(set(idxs)), dtype=torch.long)

        def pre_hook(_module, inp):
            x = inp[0].clone()
            x[..., idx] = 0.0
            return (x,) + tuple(inp[1:])
        return pre_hook

    try:
        for layer, idxs in by_layer.items():
            handles.append(model.blocks[layer].mlp.w2.register_forward_pre_hook(mk(idxs)))
        yield
    finally:
        for h in handles:
            h.remove()


def neuron_ablation_effect(model, neurons, scorer) -> dict:
    """Run ``scorer(model)->float`` with and without the neurons ablated.

    Returns {baseline, ablated, delta} where delta = baseline - ablated (how much
    the ablated units were contributing to the scored metric).
    """
    baseline = float(scorer(model))
    with ablate_mlp_neurons(model, neurons):
        ablated = float(scorer(model))
    return {"baseline": baseline, "ablated": ablated, "delta": baseline - ablated}


def _capture_block_output(model, prompt_ids, layer, device):
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cap: dict = {}

    def hook(_m, _i, o):
        cap["out"] = _block_out(o).detach()

    h = model.blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.forward(x)
    finally:
        h.remove()
    return cap["out"]  # [1, T, D]


def activation_patch_logits(model, tok, source_prompt, target_prompt, layer,
                            positions=None, device="cpu") -> dict:
    """Patch ``layer``'s residual output from a source run into a target run.

    Captures the source block output, then reruns the target with a forward-hook
    that overwrites the target's block-``layer`` output at ``positions`` (default:
    all shared positions) with the source's. Returns the last-position next-token
    logits {clean, patched}. Identity check: source==target ⇒ patched==clean.
    """
    src = _capture_block_output(model, tok.encode(source_prompt) or [tok.EOT], layer, device)
    tgt_ids = tok.encode(target_prompt) or [tok.EOT]
    x = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    pos = positions if positions is not None else list(range(min(src.size(1), len(tgt_ids))))

    def hook(_m, _i, o):
        out = _block_out(o).clone()
        for p in pos:
            if p < out.size(1) and p < src.size(1):
                out[:, p, :] = src[:, p, :]
        return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out

    with torch.no_grad():
        clean, _ = model.forward(x)
    h = model.blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            patched, _ = model.forward(x)
    finally:
        h.remove()
    return {"clean": clean[0, -1].detach(), "patched": patched[0, -1].detach()}


def capture_residual_at(model, tok, items, layer=-1, device="cpu") -> torch.Tensor:
    """Residual states at chosen token positions, for a reasoning-feature probe.

    ``items``: iterable of ``(prompt, position)``; ``position=None`` ⇒ last token.
    Returns ``[N, D]`` (float, CPU). Feed to ``mechanism.fit_linear_probe`` with
    reasoning labels (e.g. an iGSM intermediate quantity) once reasoning works.
    """
    feats = []
    for prompt, position in items:
        ids = tok.encode(prompt) or [tok.EOT]
        out = _capture_block_output(model, ids, layer, device)  # [1, T, D]
        p = position if position is not None else len(ids) - 1
        p = max(0, min(p, out.size(1) - 1))
        feats.append(out[0, p, :].float().cpu())
    return torch.stack(feats) if feats else torch.empty(0)
