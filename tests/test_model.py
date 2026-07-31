import torch

from train.model import GPT, GPTConfig, PRESETS


def tiny():
    return GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64))


def test_forward_shapes_and_loss():
    m = tiny()
    x = torch.randint(0, 50304, (2, 16))
    logits, loss = m(x)
    assert logits.shape == (2, 16, 50304) and loss is None
    y = x.clone()
    logits, loss = m(x, y)
    assert torch.isfinite(loss)


def test_ignore_index_excludes_masked_targets():
    torch.manual_seed(0)
    m = tiny()
    x = torch.randint(0, 100, (1, 16))
    y = torch.randint(0, 100, (1, 16))
    y_masked = y.clone()
    y_masked[0, 4:12] = -100
    _, loss_masked = m(x, y_masked)
    # changing the true tokens at masked positions must not change the loss
    y2 = y.clone()
    y2[0, 4:12] = (y2[0, 4:12] + 17) % 100
    y2[0, 4:12] = -100
    _, loss_masked2 = m(x, y2)
    assert torch.allclose(loss_masked, loss_masked2)
    _, loss_full = m(x, y)
    assert not torch.allclose(loss_masked, loss_full)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(1)
    m = tiny().eval()
    x = torch.randint(0, 50304, (2, 12))
    with torch.no_grad():
        full_logits, _ = m(x)
        # prefill on the first 8, then step one token at a time
        step_logits, cache = m.forward_step(x[:, :8], None)
        outs = [step_logits[:, -1]]
        for t in range(8, 12):
            lg, cache = m.forward_step(x[:, t : t + 1], cache)
            outs.append(lg[:, -1])
    for i, t in enumerate(range(7, 12)):
        assert torch.allclose(full_logits[:, t], outs[i], atol=2e-4), f"pos {t}"


def test_presets_param_counts():
    cfg = PRESETS["d160m"]
    m = GPT(cfg)
    n = m.num_params()
    assert 150e6 < n < 180e6, n


def test_device_property():
    m = tiny()
    assert m.device.type == "cpu"


def test_tied_embeddings_share_one_tensor():
    untied = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64))
    tied = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, tie_embeddings=True))
    assert tied.lm_head.weight is tied.wte.weight
    assert untied.lm_head.weight is not untied.wte.weight
    # parameters() de-duplicates shared tensors, so the count drops by exactly V*D
    assert untied.num_params() - tied.num_params() == 50304 * 64


def test_tied_model_trains_and_round_trips(tmp_path):
    m = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, tie_embeddings=True))
    x = torch.randint(0, 50304, (2, 16))
    _, loss = m(x, x.clone())
    loss.backward()
    assert m.wte.weight.grad is not None
    torch.save(m.state_dict(), tmp_path / "m.pt")
    m2 = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, tie_embeddings=True))
    m2.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    assert m2.lm_head.weight is m2.wte.weight


def test_tiny_preset_param_counts_are_exact():
    """Exact, not banded: these numbers go into configs and documents."""
    assert GPT(PRESETS["d8m"]).num_params() == 7_931_776
    assert GPT(PRESETS["d40m"]).num_params() == 40_560_000


def test_tiny_presets_are_shaped_as_designed():
    d8, d40 = PRESETS["d8m"], PRESETS["d40m"]
    assert (d8.n_layer, d8.n_head, d8.d_model) == (7, 4, 128)
    assert (d40.n_layer, d40.n_head, d40.d_model) == (12, 6, 384)
    assert d8.tie_embeddings and d40.tie_embeddings
    assert d8.effective_depth == 21 and d40.effective_depth == 24
    assert d8.head_dim == 32 and d40.head_dim == 64


def test_existing_presets_are_untouched():
    assert GPT(PRESETS["d160m"]).num_params() == 162_220_800
    assert PRESETS["d160m"].n_recurrence == 1
    assert not PRESETS["d160m"].tie_embeddings


def test_recurrence_leaves_parameter_count_unchanged():
    base = GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64)
    deep = GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, n_recurrence=3)
    assert deep.effective_depth == 6
    assert GPT(base).num_params() == GPT(deep).num_params()


def test_recurrence_one_is_the_old_behaviour():
    """R=1 must be bit-identical to the pre-change model, or the ladder breaks."""
    torch.manual_seed(0)
    a = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64))
    torch.manual_seed(0)
    b = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, n_recurrence=1))
    x = torch.randint(0, 50304, (2, 16))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.equal(la, lb)


def test_recurrence_changes_the_function():
    torch.manual_seed(0)
    a = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64))
    torch.manual_seed(0)
    b = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, n_recurrence=3))
    x = torch.randint(0, 50304, (2, 16))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert not torch.allclose(la, lb)


def test_kv_cache_matches_full_forward_under_recurrence():
    torch.manual_seed(1)
    m = GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=64, n_recurrence=3)).eval()
    x = torch.randint(0, 50304, (2, 12))
    with torch.no_grad():
        full_logits, _ = m(x)
        step_logits, cache = m.forward_step(x[:, :8], None)
        outs = [step_logits[:, -1]]
        for t in range(8, 12):
            lg, cache = m.forward_step(x[:, t : t + 1], cache)
            outs.append(lg[:, -1])
    assert len(cache.kv) == 6
    for i, t in enumerate(range(7, 12)):
        assert torch.allclose(full_logits[:, t], outs[i], atol=2e-4), f"pos {t}"
