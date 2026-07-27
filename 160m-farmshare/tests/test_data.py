import numpy as np
import pytest
import torch

from train.data import PackedShards


def make_shards(tmp_path, n=5000, masked_span=(100, 160)):
    toks = (np.arange(n) % 997).astype(np.uint16)
    mask = np.ones(n, dtype=np.uint8)
    mask[masked_span[0] : masked_span[1]] = 0
    bp, mp = tmp_path / "t.bin", tmp_path / "t.mask.bin"
    toks.tofile(bp)
    mask.tofile(mp)
    return bp, mp


def make_weights(tmp_path, n=5000):
    weights = (np.arange(n) % 7).astype(np.uint8)
    wp = tmp_path / "t.weights.bin"
    weights.tofile(wp)
    return wp, weights


def test_batch_alignment_and_mask(tmp_path):
    bp, mp = make_shards(tmp_path)
    ds = PackedShards(bp, mp, ctx=32, batch_size=2, device="cpu")
    x, y = ds.next_batch()
    assert x.shape == (2, 32) and y.shape == (2, 32)
    # within each row, y[t] is the successor of x[t] wherever unmasked
    for row in range(2):
        for t in range(31):
            if y[row, t] != -100:
                assert y[row, t] == x[row, t + 1]
    # first row starts at token 0, so y[0,0] is token 1
    assert y[0, 0] == 1


def test_masked_positions_become_ignore_index(tmp_path):
    bp, mp = make_shards(tmp_path, masked_span=(10, 40))
    ds = PackedShards(bp, mp, ctx=64, batch_size=1, device="cpu")
    _, y = ds.next_batch()
    assert (y == -100).sum() > 0


def test_cursor_resume_exact(tmp_path):
    bp, mp = make_shards(tmp_path)
    a = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    for _ in range(3):
        a.next_batch()
    state = a.state_dict()
    assert state["raw_positions"] == 3 * 2 * 16
    xa, ya = a.next_batch()
    b = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    b.load_state_dict(state)
    xb, yb = b.next_batch()
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_weighted_batch_aligns_weights_to_next_token(tmp_path):
    bp, mp = make_shards(tmp_path)
    wp, raw_weights = make_weights(tmp_path)
    ds = PackedShards(
        bp,
        mp,
        ctx=4,
        batch_size=2,
        device="cpu",
        weights_path=wp,
    )
    _, _, weights = ds.next_weighted_batch()
    expected = torch.from_numpy(
        raw_weights[:10].reshape(2, 5)[:, 1:].astype(np.float32)
    )
    assert weights.dtype == torch.float32
    assert torch.equal(weights, expected)


def test_weighted_batch_without_sidecar_returns_ones(tmp_path):
    bp, mp = make_shards(tmp_path)
    ds = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    _, targets, weights = ds.next_weighted_batch()
    assert weights.dtype == torch.float32
    assert torch.equal(weights, torch.ones_like(targets, dtype=torch.float32))


def test_weighted_cursor_resume_is_exact(tmp_path):
    bp, mp = make_shards(tmp_path)
    wp, _ = make_weights(tmp_path)
    a = PackedShards(
        bp,
        mp,
        ctx=16,
        batch_size=2,
        device="cpu",
        weights_path=wp,
    )
    for _ in range(3):
        a.next_weighted_batch()
    state = a.state_dict()
    batch_a = a.next_weighted_batch()
    b = PackedShards(
        bp,
        mp,
        ctx=16,
        batch_size=2,
        device="cpu",
        weights_path=wp,
    )
    b.load_state_dict(state)
    batch_b = b.next_weighted_batch()
    assert all(torch.equal(a_item, b_item) for a_item, b_item in zip(batch_a, batch_b))


def test_weighted_batch_wraps_sidecar_with_token_cursor(tmp_path):
    bp, mp = make_shards(tmp_path, n=50)
    wp, raw_weights = make_weights(tmp_path, n=50)
    ds = PackedShards(
        bp,
        mp,
        ctx=8,
        batch_size=2,
        device="cpu",
        start_cursor=40,
        weights_path=wp,
    )
    _, _, weights = ds.next_weighted_batch()
    expected = torch.from_numpy(
        raw_weights[:18].reshape(2, 9)[:, 1:].astype(np.float32)
    )
    assert ds.epoch == 1
    assert torch.equal(weights, expected)


def test_legacy_batch_tuple_and_positional_constructor_are_unchanged(tmp_path):
    bp, mp = make_shards(tmp_path)
    ds = PackedShards(bp, mp, 16, 2, "cpu", 0, 0)
    batch = ds.next_batch()
    assert isinstance(batch, tuple)
    assert len(batch) == 2


def test_wraparound_epoch(tmp_path):
    bp, mp = make_shards(tmp_path, n=200)
    ds = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    for _ in range(20):
        ds.next_batch()
    assert ds.epoch >= 1


def test_masked_value_probe(tmp_path):
    bp, mp = make_shards(tmp_path, masked_span=(5, 90))
    ds = PackedShards(bp, mp, ctx=32, batch_size=2, device="cpu")
    probe = ds.masked_value_batch()
    assert probe is not None
    x, y = probe
    live = int((y != -100).sum())
    assert 0 < live <= 85  # only the masked span (5..90 shifted) carries labels


def test_dense_arm_no_mask_file(tmp_path):
    bp, _ = make_shards(tmp_path)
    ds = PackedShards(bp, None, ctx=16, batch_size=2, device="cpu")
    _, y = ds.next_batch()
    assert (y == -100).sum() == 0
    assert ds.masked_value_batch() is None


@pytest.mark.parametrize("sidecar", ["mask", "weights"])
def test_configured_missing_sidecar_is_rejected(tmp_path, sidecar):
    bp, mp = make_shards(tmp_path)
    kwargs = {
        "bin_path": bp,
        "mask_path": mp,
        "ctx": 16,
        "batch_size": 2,
        "device": "cpu",
    }
    missing = tmp_path / f"missing-{sidecar}.bin"
    if sidecar == "mask":
        kwargs["mask_path"] = missing
    else:
        kwargs["weights_path"] = missing

    with pytest.raises((FileNotFoundError, ValueError), match=sidecar):
        PackedShards(**kwargs)


def test_invalid_data_state_does_not_partially_mutate_cursor(tmp_path):
    bp, mp = make_shards(tmp_path)
    ds = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    ds.next_batch()
    before = ds.state_dict()
    invalid = dict(before, cursor=len(ds.tokens) + 1)

    with pytest.raises(ValueError, match="cursor"):
        ds.load_state_dict(invalid)

    assert ds.state_dict() == before
