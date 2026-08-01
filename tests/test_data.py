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
    x, y, _ = ds.next_batch()
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
    _, y, _ = ds.next_batch()
    assert (y == -100).sum() > 0


def test_cursor_resume_exact(tmp_path):
    bp, mp = make_shards(tmp_path)
    a = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    for _ in range(3):
        a.next_batch()
    state = a.state_dict()
    xa, ya, _ = a.next_batch()
    b = PackedShards(bp, mp, ctx=16, batch_size=2, device="cpu")
    b.load_state_dict(state)
    xb, yb, _ = b.next_batch()
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


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
    _, y, _ = ds.next_batch()
    assert (y == -100).sum() == 0
    assert ds.masked_value_batch() is None


# --- segmented streams (base + extension, as every cohort config supplies) ---


def make_split_shards(tmp_path, n=5000, cut=1800, masked_span=(100, 160)):
    """Same bytes as make_shards, laid out as two files cut at `cut`."""
    toks = (np.arange(n) % 997).astype(np.uint16)
    mask = np.ones(n, dtype=np.uint8)
    mask[masked_span[0] : masked_span[1]] = 0
    paths = []
    for name, lo, hi in (("a", 0, cut), ("b", cut, n)):
        bp, mp = tmp_path / f"{name}.bin", tmp_path / f"{name}.mask.bin"
        toks[lo:hi].tofile(bp)
        mask[lo:hi].tofile(mp)
        paths.append((bp, mp))
    return [p[0] for p in paths], [p[1] for p in paths]


def test_segmented_stream_reads_as_one_logical_stream(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    bp, mp = make_shards(one)
    bins, masks = make_split_shards(two)

    whole = PackedShards(bp, mp, ctx=32, batch_size=4, device="cpu")
    parts = PackedShards(bins, masks, ctx=32, batch_size=4, device="cpu")
    assert parts.n_tokens == whole.n_tokens

    # walk well past the segment boundary at token 1800
    for step in range(20):
        xa, ya, _ = whole.next_batch()
        xb, yb, _ = parts.next_batch()
        assert torch.equal(xa, xb), f"tokens diverge at step {step}"
        assert torch.equal(ya, yb), f"labels diverge at step {step}"


def test_window_spanning_the_segment_boundary(tmp_path):
    bins, masks = make_split_shards(tmp_path, cut=1800)
    ds = PackedShards(bins, masks, ctx=64, batch_size=2, device="cpu")
    toks, msk = ds._window(1790, 40)  # straddles the boundary
    assert len(toks) == 40 and len(msk) == 40
    expected = (np.arange(1790, 1830) % 997).astype(np.uint16)
    assert np.array_equal(toks, expected)


def test_segmented_mask_length_mismatch_is_caught(tmp_path):
    bins, masks = make_split_shards(tmp_path)
    (tmp_path / "b.mask.bin").write_bytes(b"\x01" * 7)
    with pytest.raises(AssertionError, match="mask/token length mismatch"):
        PackedShards(bins, masks, ctx=32, batch_size=2, device="cpu")


def test_probe_finds_masked_region_beyond_the_first_window(tmp_path):
    """The corpus is lane-ordered and the head lane carries no masked targets."""
    n = 400_000
    toks = (np.arange(n) % 997).astype(np.uint16)
    mask = np.ones(n, dtype=np.uint8)
    mask[300_000:300_500] = 0          # far past the probe's first window
    bp, mp = tmp_path / "t.bin", tmp_path / "t.mask.bin"
    toks.tofile(bp)
    mask.tofile(mp)
    ds = PackedShards(bp, mp, ctx=64, batch_size=2, device="cpu")
    probe = ds.masked_value_batch()
    assert probe is not None, "probe gave up before reaching the masked region"
    _, y = probe
    assert int((y != -100).sum()) > 0


def test_dense_arm_probes_the_split_positions(tmp_path):
    """Both arms must score the same offloaded positions, or the contrast is
    not a contrast. Dense's own sidecar is all ones and carries no signal."""
    n = 400_000
    toks = (np.arange(n) % 997).astype(np.uint16)
    dense = np.ones(n, dtype=np.uint8)
    split = np.ones(n, dtype=np.uint8)
    split[300_000:300_500] = 0
    bp = tmp_path / "t.bin"
    dp, sp = tmp_path / "dense.bin", tmp_path / "split.bin"
    toks.tofile(bp)
    dense.tofile(dp)
    split.tofile(sp)

    d = PackedShards(bp, dp, ctx=64, batch_size=2, device="cpu", probe_mask_path=sp)
    s = PackedShards(bp, sp, ctx=64, batch_size=2, device="cpu", probe_mask_path=sp)
    pd, ps = d.masked_value_batch(), s.masked_value_batch()
    assert pd is not None and ps is not None
    # identical probe positions for both arms
    assert torch.equal(pd[0], ps[0]) and torch.equal(pd[1], ps[1])
    # and the dense arm's own labels are still unmasked during training
    _, y, _ = d.next_batch()
    assert int((y == -100).sum()) == 0


def test_probe_returns_none_when_nothing_is_masked_anywhere(tmp_path):
    bp, _ = make_shards(tmp_path)
    ds = PackedShards(bp, None, ctx=16, batch_size=2, device="cpu")
    assert ds.masked_value_batch() is None


def test_single_path_still_accepted(tmp_path):
    bp, mp = make_shards(tmp_path)
    for spec_bin, spec_mask in ((bp, mp), (str(bp), str(mp)), ([bp], [mp])):
        ds = PackedShards(spec_bin, spec_mask, ctx=16, batch_size=2, device="cpu")
        assert ds.n_tokens == 5000


def test_missing_mask_path_raises_instead_of_training_dense(tmp_path):
    """A configured-but-absent sidecar used to fall through to no masking,
    silently turning a masked arm into a dense one. It must fail closed."""
    import pytest

    bin_path = tmp_path / "train.bin"
    np.arange(40000, dtype=np.uint16).tofile(bin_path)
    with pytest.raises(FileNotFoundError, match="train_mask configured but missing"):
        PackedShards(bin_path, tmp_path / "absent.bin", ctx=16, batch_size=2)


def test_missing_probe_mask_path_raises(tmp_path):
    import pytest

    bin_path = tmp_path / "train.bin"
    np.arange(40000, dtype=np.uint16).tofile(bin_path)
    with pytest.raises(FileNotFoundError, match="probe_mask configured but missing"):
        PackedShards(
            bin_path, None, ctx=16, batch_size=2,
            probe_mask_path=tmp_path / "absent.bin",
        )


def test_weights_are_zero_exactly_where_the_mask_is(tmp_path):
    bin_path = tmp_path / "train.bin"
    mask_path = tmp_path / "mask.bin"
    n = 40000
    np.arange(n, dtype=np.uint16).tofile(bin_path)
    m = np.ones(n, dtype=np.uint8)
    m[::3] = 0
    m.tofile(mask_path)
    ds = PackedShards(bin_path, mask_path, ctx=16, batch_size=2)
    _, y, w = ds.next_batch()
    assert w.shape == y.shape
    assert ((w == 0.0) == (y == -100)).all(), "weights must zero exactly the masked targets"
    assert w.numel() == y.numel(), "denominator must count every target position"
