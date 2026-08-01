"""The fixed-denominator weighted loss.

The bug this pins: `F.cross_entropy(..., ignore_index=-100)` uses the default
`reduction='mean'`, which averages over SURVIVING targets only. Masking a
fraction f of a document's targets therefore multiplies every remaining
target's weight by 1/(1-f) -- 1.331 at the measured 24.89% fact-value mask
rate. A masked arm trained that way is not the same objective as its dense
twin, over and above the masking, which is large enough to produce a reasoning
difference with no capacity freed.

The fix divides by the count of ALL original target positions, including the
zero-weight ones, so masking removes weight mass and never redistributes it.
"""

import torch

from train.model import GPT, GPTConfig

CPU = torch.device("cpu")


def _tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(n_layer=1, n_head=2, d_model=32, vocab_size=64, ctx=16))


def _batch(B=2, T=8, V=64):
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, V, (B, T), generator=g)
    targets = torch.randint(0, V, (B, T), generator=g)
    return idx, targets


# ------------------------------------------------------- the bug, pinned


def test_mean_reduction_redistributes_weight_to_survivors():
    """Legacy path: masking half the targets leaves the loss ~unchanged in
    scale, because the mean is taken over survivors. This is the confound."""
    model = _tiny()
    idx, targets = _batch()
    _, full = model(idx, targets)

    half = targets.clone()
    half[:, ::2] = -100  # mask every other position
    _, masked = model(idx, half)

    # Both are means over their own surviving sets, so they sit at the same
    # scale rather than the masked one being half the full one.
    assert masked.item() > 0.5 * full.item() * 1.5


# ------------------------------------------------------- the fix


def test_weights_all_ones_matches_legacy_mean():
    """An all-ones weight vector with no ignored targets is the plain mean."""
    model = _tiny()
    idx, targets = _batch()
    _, legacy = model(idx, targets)
    w = torch.ones_like(targets, dtype=torch.float32)
    _, weighted = model(idx, targets, weights=w)
    assert torch.allclose(legacy, weighted, atol=1e-5)


def test_masking_half_the_weights_halves_the_loss():
    """The load-bearing property: zero weights remain in the denominator, so
    removing half the weight mass removes half the loss."""
    model = _tiny()
    idx, targets = _batch()
    w = torch.ones_like(targets, dtype=torch.float32)
    _, full = model(idx, targets, weights=w)

    w_half = w.clone()
    w_half[:, ::2] = 0.0
    _, half = model(idx, targets, weights=w_half)

    per_token = _per_token_ce(model, idx, targets)
    expected = (per_token * w_half).sum() / w_half.numel()
    assert torch.allclose(half, expected, atol=1e-5)
    # And it is strictly less than the unmasked loss, not renormalised back up.
    assert half.item() < full.item()


def test_zero_weight_positions_do_not_change_surviving_gradients():
    """Masking must not rescale what remains. The gradient from a weighted
    batch equals the gradient from the same batch with the masked positions
    deleted and the SAME denominator kept."""
    model = _tiny()
    idx, targets = _batch()
    w = torch.ones_like(targets, dtype=torch.float32)
    w[:, ::2] = 0.0

    model.zero_grad()
    _, loss_w = model(idx, targets, weights=w)
    loss_w.backward()
    g_weighted = model.lm_head.weight.grad.clone()

    # Same thing via ignore_index, but renormalised by hand to the full count.
    kept = targets.clone()
    kept[:, ::2] = -100
    model.zero_grad()
    _, loss_i = model(idx, kept)
    n_kept = int((kept != -100).sum())
    (loss_i * n_kept / targets.numel()).backward()
    g_manual = model.lm_head.weight.grad.clone()

    assert torch.allclose(g_weighted, g_manual, atol=1e-5)


def test_fractional_weights_are_honoured():
    model = _tiny()
    idx, targets = _batch()
    w = torch.full_like(targets, 0.5, dtype=torch.float32)
    _, half_weighted = model(idx, targets, weights=w)
    _, plain = model(idx, targets)
    assert torch.allclose(half_weighted, 0.5 * plain, atol=1e-5)


def test_ignore_index_and_weights_compose():
    """-100 targets contribute nothing but still count in the denominator."""
    model = _tiny()
    idx, targets = _batch()
    t = targets.clone()
    t[:, 0] = -100
    w = torch.ones_like(targets, dtype=torch.float32)
    _, loss = model(idx, t, weights=w)
    per_token = _per_token_ce(model, idx, t)
    expected = (per_token * w).sum() / w.numel()
    assert torch.allclose(loss, expected, atol=1e-5)


def _per_token_ce(model, idx, targets):
    """Reference per-position CE with ignored positions contributing zero."""
    with torch.no_grad():
        logits, _ = model(idx)
    ce = torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    return ce.view(targets.shape)
