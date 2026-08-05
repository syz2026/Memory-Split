"""The frozen RANDPOS difficulty table.

This runs on the cluster against a real checkpoint, where a bug costs GPU
hours and a rebuilt table invalidates every corpus placed with it. The
accumulation and finalisation are pure, so they are pinned here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "nll_table", ROOT / "ops" / "crowding" / "nll_table.py")
nt = importlib.util.module_from_spec(_SPEC)
sys.modules["nll_table"] = nt
_SPEC.loader.exec_module(nt)

VOCAB = 32


class FixedLogits:
    """A model that always predicts `favoured`, so NLL is analytic."""

    def __init__(self, favoured: int, margin: float = 10.0):
        self.favoured, self.margin = favoured, margin

    def __call__(self, idx):
        b, t = idx.shape
        logits = torch.zeros(b, t, VOCAB)
        logits[:, :, self.favoured] = self.margin
        return logits, None


def test_accumulate_counts_every_predicted_token_once():
    docs = [np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([5, 6], dtype=np.int64)]
    sums, counts = nt.accumulate(FixedLogits(0), docs, VOCAB,
                                 torch.device("cpu"), ctx=8, batch_size=2)
    # The first token of each document has no predictor, so 4+2 tokens give
    # 3+1 = 4 scored targets.
    assert counts.sum() == 4
    for tid in (2, 3, 4, 6):
        assert counts[tid] == 1, tid
    assert counts[1] == 0 and counts[5] == 0, "first tokens are not targets"


def test_padding_is_not_scored():
    """Right-padding to a rectangle must not add phantom zero-id targets."""
    docs = [np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
            np.array([7, 8], dtype=np.int64)]
    _, counts = nt.accumulate(FixedLogits(0), docs, VOCAB,
                              torch.device("cpu"), ctx=8, batch_size=2)
    assert counts.sum() == 5 + 1
    assert counts[0] == 0, "padding id 0 was scored as a real target"


def test_a_favoured_token_scores_far_lower_than_the_rest():
    """The table has to separate easy tokens from hard ones, since that is
    the whole basis on which RANDPOS picks a matched placement."""
    docs = [np.array([9, 7, 9, 7, 9], dtype=np.int64)]
    sums, counts = nt.accumulate(FixedLogits(7), docs, VOCAB,
                                 torch.device("cpu"), ctx=8, batch_size=1)
    table, _ = nt.finalise(sums, counts)
    assert table[7] < table[9], "the predicted token must be the cheaper one"
    assert table[7] < 0.1 and table[9] > 5.0


def test_unseen_ids_are_imputed_at_the_global_mean():
    """Zero would make every unseen token look maximally easy and pull
    control spans onto it; the global mean is the neutral choice."""
    sums = np.zeros(VOCAB)
    counts = np.zeros(VOCAB, dtype=np.int64)
    sums[3], counts[3] = 8.0, 2      # mean 4.0
    sums[4], counts[4] = 2.0, 1      # mean 2.0
    table, stats = nt.finalise(sums, counts)
    assert table[3] == pytest.approx(4.0)
    assert table[4] == pytest.approx(2.0)
    global_mean = 10.0 / 3
    assert table[0] == pytest.approx(global_mean)
    assert stats["ids_observed"] == 2
    assert stats["ids_imputed_at_global_mean"] == VOCAB - 2
    assert stats["tokens_scored"] == 3


def test_finalise_survives_an_empty_tally():
    table, stats = nt.finalise(np.zeros(VOCAB), np.zeros(VOCAB, dtype=np.int64))
    assert table.shape == (VOCAB,)
    assert stats["ids_observed"] == 0


def test_the_table_is_float32_and_vocab_shaped():
    """`build_corpus` indexes it with raw token ids, so a shape or dtype
    mismatch would surface as a corpus-wide failure two hours into a build."""
    sums = np.full(VOCAB, 3.0)
    counts = np.ones(VOCAB, dtype=np.int64)
    table, _ = nt.finalise(sums, counts)
    assert table.dtype == np.float32
    assert table.shape == (VOCAB,)
    ids = np.array([0, 5, VOCAB - 1], dtype=np.int64)
    assert table[ids].shape == (3,)


def test_burned_pilot_seeds_match_the_preregistration():
    """§6: pilot seeds 9001-9003 are burned and may not appear in the matrix.
    Building the table from data that also appears in the matrix would leak
    the outcome into the design."""
    assert nt.BURNED_PILOT_SEEDS == (9001, 9002, 9003)
