"""Controls for the recoverable-bits probe.

Section 6 of the paper rests on a negative: a model shown 1.5M facts a hundred
times each scores the same on them as on facts it never saw. That is only
evidence if the probe can detect knowledge when knowledge is present. These
tests pin the decision rule, so the paper cannot quietly keep a claim whose
positive control failed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "probe_control", ROOT / "ops" / "crowding" / "probe_control.py")
pc = importlib.util.module_from_spec(_SPEC)
sys.modules["probe_control"] = pc
_SPEC.loader.exec_module(pc)


def test_a_probe_that_cannot_see_knowledge_invalidates_the_null():
    """If trained and unseen entities score the same even in the phrasing the
    model trained on, the instrument is blind and the held-out null is not
    evidence of anything."""
    v = pc.verdict(training_gap=0.4, heldout_gap=-0.6)
    assert v["probe_detects_knowledge"] is False
    assert "POSITIVE CONTROL FAILS" in v["reading"]
    assert "withdrawn" in v["reading"]


def test_the_reported_configuration_is_the_interesting_one():
    """Probe sees knowledge in training phrasing, sees none under a held-out
    phrasing. That is the claim section 6 makes."""
    v = pc.verdict(training_gap=31.0, heldout_gap=-0.61)
    assert v["probe_detects_knowledge"] is True
    assert "accessibility" in v["reading"]
    assert "not about the instrument" in v["reading"]


def test_retrievable_knowledge_is_recognised_and_kills_the_null():
    """If the facts turn out to be addressable under held-out phrasing, the
    capacity argument survives and section 6 is wrong."""
    v = pc.verdict(training_gap=40.0, heldout_gap=22.0)
    assert v["probe_detects_knowledge"] is True
    assert "still live" in v["reading"]


def test_the_detection_floor_is_explicit_and_not_hidden():
    """A reviewer must be able to see what counts as detection, and it must not
    be tuned after the fact."""
    assert pc.DETECTION_FLOOR_BITS == 5.0
    # A gap just under the floor must not be read as detection.
    assert pc.verdict(4.9, -0.6)["probe_detects_knowledge"] is False
    assert pc.verdict(5.1, -0.6)["probe_detects_knowledge"] is True


def test_probe_bits_separates_a_model_that_knows_from_one_that_does_not():
    """End to end on a real tokenizer and two stub models: one that puts all
    mass on the true value, one uniform. The knowing model must score far
    higher, or the metric cannot support either direction of claim.
    """
    import torch
    from corpusgen import bios
    from train.tokenizer import get_tok

    tok = get_tok()
    recs = bios.generate_records(4, 0)
    V = 50304

    class Omniscient:
        """Assigns almost all mass to whatever token comes next."""
        cfg = type("C", (), {"ctx": 1024})()

        def __call__(self, idx, targets=None):
            b, t = idx.shape
            logits = torch.full((b, t, V), -12.0)
            # position i predicts token i+1
            for r in range(b):
                for i in range(t - 1):
                    logits[r, i, int(idx[r, i + 1])] = 12.0
            return logits, None

    class Uniform:
        cfg = type("C", (), {"ctx": 1024})()

        def __call__(self, idx, targets=None):
            b, t = idx.shape
            return torch.zeros(b, t, V), None

    dev = torch.device("cpu")
    known, _, _ = pc.probe_bits(Omniscient(), tok, recs, dev, "training", 4)
    blind, _, _ = pc.probe_bits(Uniform(), tok, recs, dev, "training", 4)
    assert known > blind + 5.0, (known, blind)
    assert known > 0, "a model that knows the value must recover positive bits"
