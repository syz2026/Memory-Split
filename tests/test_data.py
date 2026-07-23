import numpy as np
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
    xa, ya = a.next_batch()
    b = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    b.load_state_dict(state)
    xb, yb = b.next_batch()
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_wraparound_epoch(tmp_path):
    bp, mp = make_shards(tmp_path, n=200)
    ds = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    for _ in range(20):
        ds.next_batch()
    assert ds.epoch >= 1


def test_wraparound_trains_tail_instead_of_skipping_it(tmp_path):
    bp, _ = make_shards(tmp_path, n=70, masked_span=(70, 70))
    ds = PackedShards(bp, None, ctx=16, batch_size=2, device="cpu")

    trained_targets = set()
    for _ in range(3):
        _, y = ds.next_batch()
        trained_targets.update(y.flatten().tolist())

    assert trained_targets == set(range(70))
    assert ds.epoch == 1


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
