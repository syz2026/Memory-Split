"""NR-5 — fact-side mechanism / probing (no retraining).

Mechanistic, readout-independent evidence for "facts left the weights":

1. **Memorization localization** (L10 fact half): per MLP unit, how selective is
   its last-token activation for fact-value retrieval (vs neutral control)?
   Cohen's d per neuron. The claim-bearing statistic is the CROSS-ARM ratio —
   the dense arm's top memorization neurons should be near-silent (much lower
   |d|) in the split arm, whose weights never had to store values.

2. **Probe-based bits** (L11): a linear probe on frozen residual states that
   predicts the fact value is a readout INDEPENDENT of generative decoding. If
   dense stores the fact, the probe recovers it even where greedy recall floors;
   if split never stored it, the probe fails. Probe accuracy is converted to
   bits with the SAME clamped estimator as recall bits (``evals.recall``), so a
   dense(probe) >> dense(recall) gap directly exposes the extractability
   confound L11 warns about.

Pure forward passes + a tiny deterministic probe fit. No edit to
``train/model.py`` — activations are captured via forward hooks. See
``replication/specs/nr5-fact-mechanism.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from corpusgen.bios import RELATION_PHRASES, VALUE_POOLS
from evals.recall import bits_in_weights

# categorical attributes with a fixed pool (birth_date's 27k-way space is
# excluded from the probe; it is still usable for selectivity).
PROBE_ATTRS = tuple(a for a in RELATION_PHRASES if a in VALUE_POOLS)

_DEFAULT_CONTROL = [
    "The weather today is quite",
    "In the morning she liked to",
    "The river flowed gently past the",
    "He opened the book and began to",
    "Scientists have long studied the",
    "The recipe calls for a cup of",
    "After the meeting they walked to the",
    "The old bridge was built from",
    "Children played in the park until",
    "The story begins on a cold",
]


# ---------------------------------------------------------------- capture


def _encode_padded(tok, prompts, device):
    ids = [tok.encode(p) or [tok.EOT] for p in prompts]
    lengths = [len(x) for x in ids]
    T = max(lengths)
    x = torch.full((len(ids), T), tok.EOT, dtype=torch.long, device=device)
    for i, seq in enumerate(ids):
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return x, lengths


def capture_last_token(model, tok, prompts, device, batch_size: int = 32) -> dict:
    """Last-(real)-token MLP-neuron and residual activations per prompt.

    Returns ``{"mlp": Tensor[N, L, H], "resid": Tensor[N, L, D]}`` on CPU float.
    ``mlp`` is the SwiGLU gated activation feeding each block's down-projection
    (``block.mlp.w2`` input); ``resid`` is each block's output residual stream.
    Right-padding + per-row gather at the true last token (causal attention makes
    trailing pad tokens irrelevant to earlier positions).
    """
    n_layer = len(model.blocks)
    mlp_out: list[torch.Tensor] = []
    resid_out: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo : lo + batch_size]
        x, lengths = _encode_padded(tok, chunk, device)
        mlp_buf: list[torch.Tensor | None] = [None] * n_layer
        resid_buf: list[torch.Tensor | None] = [None] * n_layer
        handles = []

        def mk_mlp(i):
            def hook(_m, inp, _out):
                mlp_buf[i] = inp[0].detach()
            return hook

        def mk_blk(i):
            def hook(_m, _inp, out):
                resid_buf[i] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook

        try:
            for i, blk in enumerate(model.blocks):
                handles.append(blk.mlp.w2.register_forward_hook(mk_mlp(i)))
                handles.append(blk.register_forward_hook(mk_blk(i)))
            with torch.no_grad():
                model.forward(x)
        finally:
            for h in handles:
                h.remove()

        last = torch.tensor([le - 1 for le in lengths], device=device)
        # stack layers -> [B, L, dim]; gather last real token per row
        mlp_stack = torch.stack(mlp_buf, dim=1)  # [B, L, T, H]
        resid_stack = torch.stack(resid_buf, dim=1)  # [B, L, T, D]
        bidx = torch.arange(mlp_stack.size(0), device=device)
        mlp_last = mlp_stack[bidx[:, None], torch.arange(n_layer, device=device)[None, :],
                             last[:, None], :]
        resid_last = resid_stack[bidx[:, None], torch.arange(n_layer, device=device)[None, :],
                                 last[:, None], :]
        mlp_out.append(mlp_last.float().cpu())
        resid_out.append(resid_last.float().cpu())
    if was_training:
        model.train()
    return {
        "mlp": torch.cat(mlp_out, dim=0) if mlp_out else torch.empty(0),
        "resid": torch.cat(resid_out, dim=0) if resid_out else torch.empty(0),
    }


# ---------------------------------------------------------------- selectivity


def cohens_d(fact: torch.Tensor, ctrl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Signed per-unit selectivity: (mean_fact - mean_ctrl) / pooled_sd.

    ``fact``/``ctrl`` are ``[N, L, H]`` (or ``[N, H]``); reduces over N ⇒ returns
    ``[L, H]`` (or ``[H]``). Positive ⇒ the unit fires more during fact
    retrieval than on neutral control.
    """
    mf, mc = fact.mean(0), ctrl.mean(0)
    vf, vc = fact.var(0, unbiased=True), ctrl.var(0, unbiased=True)
    pooled = torch.sqrt((vf + vc) / 2.0 + eps)
    return (mf - mc) / pooled


def top_neurons(sel: torch.Tensor, k: int) -> list[tuple[int, int, float]]:
    """Top-k (layer, neuron, signed d) by |d| from a ``[L, H]`` selectivity map."""
    flat = sel.reshape(-1)
    k = min(k, flat.numel())
    idx = torch.topk(flat.abs(), k).indices
    H = sel.size(-1)
    return [(int(i // H), int(i % H), float(flat[i])) for i in idx]


def localization_report(
    mlp_by_arm: dict[str, torch.Tensor],
    ctrl_mlp_by_arm: dict[str, torch.Tensor],
    k: int = 64,
) -> dict:
    """Per-arm memorization-neuron selectivity + the cross-arm silence stat.

    For each arm, ``sel_arm = cohens_d(fact, ctrl)`` over MLP units. Reports each
    arm's top-k mean |d|, and — the claim-bearing number — takes the DENSE arm's
    top-k neuron indices and compares their mean |d| in dense vs in split. A
    large ``cross_arm_ratio`` (dense >> split at the SAME units) is mechanistic
    evidence the fact-storage units are silent in the split arm.
    """
    sel = {a: cohens_d(mlp_by_arm[a], ctrl_mlp_by_arm[a]) for a in mlp_by_arm}
    report: dict = {"k": k, "per_arm": {}}
    for a, s in sel.items():
        tops = top_neurons(s, k)
        report["per_arm"][a] = {
            "top_mean_abs_d": float(sum(abs(d) for _, _, d in tops) / max(1, len(tops))),
            "top_neurons": tops[:10],  # a peek; full list is reproducible
        }
    if "dense" in sel and "split" in sel:
        H = sel["dense"].size(-1)
        dense_top = top_neurons(sel["dense"], k)
        dflat, sflat = sel["dense"].reshape(-1), sel["split"].reshape(-1)
        idx = [layer * H + neuron for layer, neuron, _ in dense_top]
        dense_at = sum(abs(float(dflat[i])) for i in idx) / max(1, len(idx))
        split_at = sum(abs(float(sflat[i])) for i in idx) / max(1, len(idx))
        report["cross_arm"] = {
            "dense_top_mean_abs_d_in_dense": dense_at,
            "dense_top_mean_abs_d_in_split": split_at,
            "cross_arm_ratio": dense_at / split_at if split_at > 1e-9 else float("inf"),
        }
    return report


# ---------------------------------------------------------------- linear probe


def fit_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Deterministic multinomial logistic regression on frozen features.

    Standardizes X (train stats), full-batch Adam, L2. Returns
    ``{"W", "b", "mu", "sd"}`` for use by ``probe_accuracy``. Linear by design
    (bounded expressivity ⇒ measures *linear decodability*, not probe capacity).
    """
    torch.manual_seed(seed)
    X = X.float()
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    Xs = (X - mu) / sd
    d = Xs.size(1)
    W = torch.zeros(d, n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        logits = Xs @ W + b
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return {"W": W.detach(), "b": b.detach(), "mu": mu, "sd": sd}


def probe_predict(probe: dict, X: torch.Tensor) -> torch.Tensor:
    Xs = (X.float() - probe["mu"]) / probe["sd"]
    return (Xs @ probe["W"] + probe["b"]).argmax(1)


def probe_accuracy(probe: dict, X: torch.Tensor, y: torch.Tensor) -> float:
    if X.numel() == 0:
        return 0.0
    return float((probe_predict(probe, X) == y).float().mean())


def probe_bits(acc_by_attr: dict[str, float], n_eval: int,
               pool_sizes: dict[str, float]) -> dict:
    """Probe accuracy -> bits via the SAME clamped estimator as recall bits.

    Returns ``{"per_attribute_stored_frac", "total_bits", "n_eval"}``.
    Comparable to ``evals.recall.bits_in_weights``: the difference between
    probe-bits and recall-bits isolates stored-but-not-extractable facts (L11).
    """
    stored = {}
    for attr, acc in acc_by_attr.items():
        pool = float(pool_sizes[attr])
        g = 1.0 / pool
        stored[attr] = max(0.0, (acc - g) / (1.0 - g))
    total = bits_in_weights(acc_by_attr, n_eval, pool_sizes)
    return {"per_attribute_stored_frac": stored, "total_bits": total, "n_eval": n_eval}


# ---------------------------------------------------------------- orchestrator


def _probe_pool_sizes() -> dict[str, float]:
    return {a: float(len(VALUE_POOLS[a])) for a in PROBE_ATTRS}


def build_fact_probes(records, attrs=PROBE_ATTRS):
    """(prompts, entity_attr_pairs) for the fact-retrieval probe set."""
    prompts, keys = [], []
    for rec in records:
        for attr in attrs:
            prompts.append(f"{rec.name}'s {RELATION_PHRASES[attr]} is")
            keys.append((rec.entity_id, attr))
    return prompts, keys


def fact_mechanism_report(
    models_by_arm: dict,
    tok,
    records,
    device,
    control_prompts=None,
    control_records=None,
    probe_layer: int = -1,
    k: int = 64,
    test_frac: float = 0.3,
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """End-to-end fact-side mechanism report over dense+split weights.

    Localization (MLP selectivity, cross-arm silence) + a linear probe-bits table
    per arm/attribute (held-out entities). All forward passes; deterministic.

    ``control_records`` (fresh entities NEVER trained): when given, the
    selectivity control is structure-matched — "{name}'s {rel} is" over fresh
    entities — so per-arm selectivity isolates *stored-value retrieval* rather
    than prompt structure (ledger L26). Falls back to the generic
    ``_DEFAULT_CONTROL`` sentences otherwise (cross-arm ratio is robust either
    way; only per-arm |d| is confounded without a matched control).
    """
    attrs = PROBE_ATTRS
    if control_prompts is None:
        if control_records is not None:
            control_prompts = [f"{r.name}'s {RELATION_PHRASES[a]} is"
                               for r in control_records for a in attrs]
        else:
            control_prompts = _DEFAULT_CONTROL
    fact_prompts, keys = build_fact_probes(records, attrs)
    pool_index = {a: {v: i for i, v in enumerate(VALUE_POOLS[a])} for a in attrs}

    rec_by_id = {r.entity_id: r for r in records}
    labels = torch.tensor([pool_index[a][rec_by_id[eid].attrs[a]] for (eid, a) in keys])
    attr_of = [a for (_eid, a) in keys]

    # held-out split by entity (never test on trained entities)
    ent_ids = sorted({r.entity_id for r in records})
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ent_ids), generator=rng).tolist()
    n_test = max(1, int(len(ent_ids) * test_frac))
    test_ents = {ent_ids[i] for i in perm[:n_test]}
    is_test = torch.tensor([keys[i][0] in test_ents for i in range(len(keys))])

    mlp_by_arm, ctrl_by_arm = {}, {}
    probe_by_arm: dict[str, dict] = {}
    for arm, model in models_by_arm.items():
        fact_act = capture_last_token(model, tok, fact_prompts, device, batch_size)
        ctrl_act = capture_last_token(model, tok, control_prompts, device, batch_size)
        mlp_by_arm[arm] = fact_act["mlp"]
        ctrl_by_arm[arm] = ctrl_act["mlp"]

        resid = fact_act["resid"][:, probe_layer, :]  # [N, D]
        acc_by_attr = {}
        for a in attrs:
            sel = torch.tensor([i for i in range(len(keys)) if attr_of[i] == a])
            if sel.numel() == 0:
                continue
            tr = sel[~is_test[sel]]
            te = sel[is_test[sel]]
            if tr.numel() == 0 or te.numel() == 0:
                continue
            probe = fit_linear_probe(resid[tr], labels[tr], len(VALUE_POOLS[a]), seed=seed)
            acc_by_attr[a] = probe_accuracy(probe, resid[te], labels[te])
        # bits_in_weights multiplies each attribute's stored-fraction by the
        # number of ENTITIES (one value per attribute per entity). is_test.sum()
        # counts entity×attribute pairs, which over-counted bits ~n_attrs× (L25).
        n_test_entities = len(test_ents)
        probe_by_arm[arm] = {
            "probe_acc": acc_by_attr,
            "chance": {a: 1.0 / len(VALUE_POOLS[a]) for a in acc_by_attr},
            "bits": probe_bits(acc_by_attr, n_test_entities, _probe_pool_sizes()),
        }

    report = {
        "localization": localization_report(mlp_by_arm, ctrl_by_arm, k=k),
        "probe": probe_by_arm,
        "n_records": len(records),
        "n_probes": len(fact_prompts),
        "probe_layer": probe_layer,
    }
    if "dense" in probe_by_arm and "split" in probe_by_arm:
        db = probe_by_arm["dense"]["bits"]["total_bits"]
        sb = probe_by_arm["split"]["bits"]["total_bits"]
        report["probe_bits_gap_dense_minus_split"] = db - sb
    return report
