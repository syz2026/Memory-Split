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
    # Adjacent rows share exactly one boundary token; no causal target is skipped.
    assert x[1, 0] == 32
    assert y[0, -1] == x[1, 0]


def test_masked_positions_become_ignore_index(tmp_path):
    bp, mp = make_shards(tmp_path, masked_span=(10, 40))
    ds = PackedShards(bp, mp, ctx=64, batch_size=1, device="cpu")
    _, y = ds.next_batch()
    assert (y == -100).sum() > 0


def test_weighted_batch_keeps_targets_and_aligns_direct_weights(tmp_path):
    bp, mp = make_shards(tmp_path, n=128, masked_span=(10, 20))
    ds = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    x, y, weights = ds.next_weighted_batch()

    assert x.shape == y.shape == weights.shape == (2, 16)
    assert (y == -100).sum() == 0
    assert torch.equal(y.flatten(), torch.arange(1, 33))
    assert torch.equal(weights.flatten(), torch.tensor(
        [0.0 if 10 <= index < 20 else 1.0 for index in range(1, 33)]
    ))


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


def test_one_epoch_consumes_every_cyclic_target_once_in_order(tmp_path):
    bp, mp = make_shards(tmp_path, n=192, masked_span=(0, 0))
    ds = PackedShards(bp, mp, ctx=16, batch_size=3, device="cpu")
    targets = []
    for _ in range(4):
        _, y = ds.next_batch()
        targets.extend(y.flatten().tolist())

    assert targets == [*range(1, 192), 0]
    assert ds.state_dict() == {"cursor": 192, "epoch": 1}


def test_sidecar_alignment_wraps_with_the_final_causal_target(tmp_path):
    bp, mp = make_shards(tmp_path, n=64, masked_span=(0, 1))
    ds = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    first_targets = []
    for _ in range(2):
        _, y = ds.next_batch()
        first_targets.extend(y.flatten().tolist())

    assert first_targets[:-1] == list(range(1, 64))
    assert first_targets[-1] == -100


def test_explicit_missing_or_symlinked_sidecar_is_rejected(tmp_path):
    bp, mp = make_shards(tmp_path)
    missing = tmp_path / "missing.bin"
    with pytest.raises(ValueError, match="target-weight"):
        PackedShards(bp, missing, ctx=16, batch_size=2)

    link = tmp_path / "weights-link.bin"
    link.symlink_to(mp)
    with pytest.raises(ValueError, match="target-weight"):
        PackedShards(bp, link, ctx=16, batch_size=2)


def test_masked_value_probe(tmp_path):
    bp, mp = make_shards(tmp_path, masked_span=(5, 90))
    ds = PackedShards(bp, mp, ctx=32, batch_size=2, device="cpu")
    probe = ds.masked_value_batch()
    assert probe is not None
    _, y = probe
    live = int((y != -100).sum())
    assert 0 < live <= 85  # only the masked span (5..90 shifted) carries labels


def test_dense_arm_no_mask_file(tmp_path):
    bp, _ = make_shards(tmp_path)
    ds = PackedShards(bp, None, ctx=16, batch_size=2, device="cpu")
    _, y = ds.next_batch()
    assert (y == -100).sum() == 0
    assert ds.masked_value_batch() is None


def test_segmented_stream_crosses_boundary_and_wraps_exactly(tmp_path):
    token_paths = []
    mask_paths = []
    for index, values in enumerate(
        (
            np.arange(24, dtype=np.uint16),
            np.arange(24, 48, dtype=np.uint16),
        )
    ):
        token_path = tmp_path / f"tokens-{index}.bin"
        mask_path = tmp_path / f"weights-{index}.bin"
        values.tofile(token_path)
        np.ones(len(values), dtype=np.uint8).tofile(mask_path)
        token_paths.append(token_path)
        mask_paths.append(mask_path)

    ds = PackedShards(
        token_paths,
        mask_paths,
        ctx=8,
        batch_size=2,
        start_cursor=16,
    )
    targets = []
    for _ in range(3):
        _, y, weights = ds.next_weighted_batch()
        targets.extend(y.flatten().tolist())
        assert weights.sum() == 16

    assert targets == [*range(17, 48), *range(17)]
    assert ds.state_dict() == {"cursor": 64, "epoch": 1}


def test_segmented_stream_rejects_misaligned_segments(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, first_mask = make_shards(first_root, n=64)
    second, second_mask = make_shards(second_root, n=64)
    with pytest.raises(ValueError, match="segment count"):
        PackedShards([first, second], [first_mask], ctx=8, batch_size=2)

    np.ones(63, dtype=np.uint8).tofile(second_mask)
    with pytest.raises(ValueError, match="segment length"):
        PackedShards(
            [first, second],
            [first_mask, second_mask],
            ctx=8,
            batch_size=2,
        )
