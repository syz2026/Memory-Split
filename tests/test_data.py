import hashlib
import os

import numpy as np
import pytest
import torch

import train.data as data_module
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


def test_batches_cover_each_global_causal_target_once_in_order(tmp_path):
    bp, _ = make_shards(tmp_path, n=100, masked_span=(100, 100))
    ds = PackedShards(bp, None, ctx=4, batch_size=2, device="cpu")

    first_targets = ds.next_batch()[1].flatten()
    second_targets = ds.next_batch()[1].flatten()

    assert torch.equal(first_targets, torch.arange(1, 9))
    assert torch.equal(second_targets, torch.arange(9, 17))


def test_corpus_exactly_one_batch_window_is_accepted(tmp_path):
    bp, _ = make_shards(tmp_path, n=9, masked_span=(9, 9))
    shard = PackedShards(bp, None, ctx=4, batch_size=2)

    _, targets = shard.next_batch()

    assert torch.equal(targets.flatten(), torch.arange(1, 9))


def test_unbound_multiple_files_are_rejected(tmp_path):
    token_paths = []
    weight_paths = []
    for shard_index, (start, stop) in enumerate(((0, 9), (9, 18))):
        token_path = tmp_path / f"tokens-{shard_index}.bin"
        weight_path = tmp_path / f"weights-{shard_index}.bin"
        np.arange(start, stop, dtype=np.uint16).tofile(token_path)
        np.arange(start, stop, dtype=np.uint8).tofile(weight_path)
        token_paths.append(token_path)
        weight_paths.append(weight_path)

    with pytest.raises(ValueError, match="verified parallel corpus"):
        PackedShards(
            token_paths,
            None,
            ctx=4,
            batch_size=2,
            weights_path=weight_paths,
        )


def test_wrap_continues_through_final_target_without_skip_or_duplicate(tmp_path):
    bp, _ = make_shards(tmp_path, n=17, masked_span=(17, 17))
    ds = PackedShards(bp, None, ctx=4, batch_size=2, device="cpu")

    observed = [ds.next_batch()[1].flatten() for _ in range(3)]

    assert torch.equal(observed[0], torch.arange(1, 9))
    assert torch.equal(observed[1], torch.arange(9, 17))
    assert torch.equal(observed[2], torch.arange(0, 8))
    assert ds.epoch == 1
    assert ds.cursor == 7


def test_validated_update_cursor_advances_only_by_exact_global_quota(tmp_path):
    bp, _ = make_shards(tmp_path, n=64, masked_span=(64, 64))
    ds = PackedShards(bp, None, ctx=4, batch_size=2)
    ds.validate_update_alignment(8)

    with pytest.raises(ValueError, match="exactly"):
        ds.advance(4)

    ds.advance(8)
    assert ds.global_cursor == 8


def test_cursor_state_rejects_different_shard_provenance(tmp_path):
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    np.arange(100, dtype=np.uint16).tofile(first_path)
    np.arange(99, -1, -1, dtype=np.uint16).tofile(second_path)
    first = PackedShards(first_path, None, ctx=4, batch_size=2)
    first.next_batch()

    second = PackedShards(second_path, None, ctx=4, batch_size=2)
    with pytest.raises(ValueError, match="provenance"):
        second.load_state_dict(first.state_dict())


def test_cursor_state_rejects_same_path_rewritten_in_place(tmp_path):
    token_path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.uint16).tofile(token_path)
    original = PackedShards(token_path, None, ctx=4, batch_size=2)
    state = original.state_dict()

    np.arange(99, -1, -1, dtype=np.uint16).tofile(token_path)
    rewritten = PackedShards(token_path, None, ctx=4, batch_size=2)

    with pytest.raises(ValueError, match="provenance"):
        rewritten.load_state_dict(state)


def test_legacy_loader_hashes_and_maps_the_same_pinned_descriptor(
    tmp_path,
    monkeypatch,
):
    token_path = tmp_path / "tokens.bin"
    replacement = tmp_path / "replacement.bin"
    original = np.arange(64, dtype=np.uint16)
    malicious = np.arange(63, -1, -1, dtype=np.uint16)
    original.tofile(token_path)
    malicious.tofile(replacement)
    original_hash = hashlib.sha256(original.tobytes()).hexdigest()
    real_hash = data_module._sha256_fd
    swapped = False

    def replace_during_hash(fd):
        nonlocal swapped
        if not swapped:
            os.replace(replacement, token_path)
            swapped = True
        return real_hash(fd)

    monkeypatch.setattr(data_module, "_sha256_fd", replace_during_hash)
    shard = PackedShards(token_path, None, ctx=4, batch_size=2)
    _, targets = shard.next_batch()

    assert shard.provenance["tokens"]["sha256"] == original_hash
    assert torch.equal(targets.flatten(), torch.arange(1, 9))
    assert shard._open_files
    assert all(not pinned.handle.closed for pinned in shard._open_files)


def test_legacy_loader_rejects_symlinked_parent_component(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    token_path = real / "tokens.bin"
    np.arange(64, dtype=np.uint16).tofile(token_path)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        PackedShards(linked_parent / "tokens.bin", None, ctx=4, batch_size=2)


def test_cursor_state_rejects_incompatible_state_version(tmp_path):
    bp, _ = make_shards(tmp_path)
    shard = PackedShards(bp, None, ctx=4, batch_size=2)
    state = shard.state_dict()
    state["format_version"] = 999

    with pytest.raises(ValueError, match="version"):
        shard.load_state_dict(state)


def test_unverified_multiple_shards_fail_closed(tmp_path):
    token_paths = []
    for index, length in enumerate((9, 16)):
        path = tmp_path / f"tokens-{index}.bin"
        np.arange(length, dtype=np.uint16).tofile(path)
        token_paths.append(path)

    with pytest.raises(ValueError, match="verified parallel corpus"):
        PackedShards(token_paths, None, ctx=4, batch_size=2)


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
        raw_weights[1:9].reshape(2, 4).astype(np.float32)
    )
    assert weights.dtype == torch.float32
    assert torch.equal(weights, expected)


def test_nonzero_cursor_sidecar_alignment_uses_the_same_token_window(tmp_path):
    bp, mp = make_shards(tmp_path)
    wp, raw_weights = make_weights(tmp_path)
    start_cursor = 37
    ds = PackedShards(
        bp,
        mp,
        ctx=4,
        batch_size=2,
        device="cpu",
        start_cursor=start_cursor,
        weights_path=wp,
    )

    x, _, weights = ds.next_weighted_batch()

    token_window = np.memmap(bp, dtype=np.uint16, mode="r")[
        start_cursor : start_cursor + 9
    ]
    expected_x = torch.from_numpy(
        np.asarray(token_window[:-1].reshape(2, 4), dtype=np.int64).copy()
    )
    expected_weights = torch.from_numpy(
        raw_weights[start_cursor + 1 : start_cursor + 9]
        .reshape(2, 4)
        .astype(np.float32)
    )
    assert torch.equal(x, expected_x)
    assert torch.equal(weights, expected_weights)


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
        np.concatenate((raw_weights[41:], raw_weights[:7]))
        .reshape(2, 8)
        .astype(np.float32)
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
