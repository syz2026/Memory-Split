import numpy as np
import pytest
import torch
import torch.nn.functional as F

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


def test_unweighted_loss_matches_legacy_cross_entropy_exactly():
    torch.manual_seed(0)
    m = tiny()
    x = torch.randint(0, 100, (1, 8))
    targets = torch.randint(0, 100, (1, 8))
    targets[0, 3] = -100
    logits, loss = m(x, targets)
    expected = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )
    assert torch.equal(loss, expected)


def test_target_weights_normalize_by_all_positions():
    torch.manual_seed(0)
    model = tiny()
    x = torch.randint(0, 100, (1, 8))
    targets = torch.randint(0, 100, (1, 8))
    weights = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    logits, loss = model(x, targets, target_weights=weights)
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    assert torch.allclose(loss, (per_token * weights).sum() / targets.numel())


def test_target_weights_zero_ignored_labels_safely():
    torch.manual_seed(1)
    model = tiny()
    x = torch.randint(0, 100, (1, 6))
    targets = torch.randint(0, 100, (1, 6))
    targets[0, 2] = -100
    weights = torch.ones_like(targets, dtype=torch.float32)
    weights[0, 2] = torch.nan
    logits, loss = model(x, targets, target_weights=weights)
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)
    valid = targets.ne(-100)
    expected = (per_token[valid] * weights[valid]).sum() / targets.numel()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected)


def test_target_weights_shape_must_match_targets():
    model = tiny()
    x = torch.randint(0, 100, (1, 8))
    targets = torch.randint(0, 100, (1, 8))
    with pytest.raises(ValueError, match="target_weights shape must match targets"):
        model(x, targets, target_weights=torch.ones(8))


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


@pytest.mark.parametrize(
    ("name", "expected_params"),
    [
        ("d160m", 162_220_800),
        ("d360m", 356_033_536),
    ],
)
def test_protected_presets_have_exact_context_and_parameter_counts(
    name,
    expected_params,
):
    assert PRESETS[name].ctx == 1024
    with torch.device("meta"):
        model = GPT(PRESETS[name])
    assert model.num_params() == expected_params


def test_trainer_context_override_does_not_mutate_global_preset(
    monkeypatch,
    tmp_path,
):
    import train.trainer as trainer_module

    class FakeGPT(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakePackedShards:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(trainer_module, "GPT", FakeGPT)
    monkeypatch.setattr(trainer_module, "PackedShards", FakePackedShards)
    corpus = tmp_path / "unused.bin"
    np.arange(256, dtype=np.uint16).tofile(corpus)
    original_ctx = PRESETS["d160m"].ctx
    try:
        trainer = trainer_module.Trainer(
            {
                "device": "cpu",
                "seed": 7,
                "model": "d160m",
                "ctx": 128,
                "micro_batch_size": 1,
                "tokens_per_step": 128,
                "train_bin": str(corpus),
                "max_steps": 1,
                "lr": 1e-3,
                "out_dir": str(tmp_path / "run"),
            }
        )

        assert trainer.model.cfg.ctx == 128
        assert PRESETS["d160m"].ctx == original_ctx
    finally:
        PRESETS["d160m"].ctx = original_ctx


def test_device_property():
    m = tiny()
    assert m.device.type == "cpu"
