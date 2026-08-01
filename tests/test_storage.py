"""Recoverable-bits measurement: matched baseline/ceiling and held-out queries."""

import math

import pytest
import torch

from corpusgen import bios
from corpusgen.records import ATTRIBUTES
from evals.storage import (
    PROBE_TEMPLATES,
    baseline_bits,
    probe_queries,
    recoverable_bits,
    unconditional_bits,
)
from train.tokenizer import get_tok

TOK = get_tok()
CPU = torch.device("cpu")
V = TOK.VOCAB_SIZE


# ------------------------------------------------------- the two baselines


def test_unconditional_pool_entropy_is_52_96_bits_per_entity():
    total = sum(unconditional_bits(a) for a in ATTRIBUTES)
    assert abs(total - 52.96) < 0.01, total


def test_length_conditioned_baseline_is_45_30_bits_per_entity():
    """Token length leaks 7.66 bits, so the conditional ceiling is lower.
    Mixing this baseline with the unconditional ceiling would overstate
    recovery by exactly that gap."""
    recs = bios.generate_records(400, 0)
    per_entity = 0.0
    for a in ATTRIBUTES:
        vals = [r.attrs[a] for r in recs]
        per_entity += sum(baseline_bits(a, v, TOK, "length") for v in vals) / len(vals)
    assert abs(per_entity - 45.30) < 0.35, per_entity


def test_conditional_baseline_never_exceeds_unconditional():
    recs = bios.generate_records(200, 0)
    for r in recs[:50]:
        for a in ATTRIBUTES:
            cond = baseline_bits(a, r.attrs[a], TOK, "length")
            assert cond <= unconditional_bits(a) + 1e-9


def test_unknown_baseline_name_raises():
    with pytest.raises(ValueError, match="baseline must be"):
        baseline_bits("major", "Physics", TOK, "nonsense")


# ------------------------------------------------------- held-out queries


def test_probe_templates_are_disjoint_from_training_templates():
    """Scoring a training surface form would measure template memorisation."""
    training = {
        prefix.strip() for a in ATTRIBUTES for prefix, _ in bios.BIO_TEMPLATES[a]
    }
    for attr, (prefix, _) in PROBE_TEMPLATES.items():
        assert prefix.strip() not in training, f"{attr} probe reuses a training template"


def test_probe_prompt_text_never_appears_in_any_rendering():
    """Stronger: the probe's distinctive stem must not occur in the corpus."""
    recs = bios.generate_records(20, 0)
    corpus = "".join(
        "".join(t for t, _ in bios.render_bio_marked(r, e))
        for r in recs
        for e in range(25)
    )
    for attr, (prefix, _) in PROBE_TEMPLATES.items():
        stem = prefix.split("{name}")[-1].strip()
        assert stem not in corpus, f"{attr} probe stem {stem!r} occurs in training text"


def test_one_query_per_entity_attribute_pair():
    recs = bios.generate_records(7, 0)
    qs = probe_queries(recs)
    assert len(qs) == 7 * len(ATTRIBUTES)
    assert {q["attr"] for q in qs} == set(ATTRIBUTES)
    assert all(q["value"] for q in qs)


# ------------------------------------------------------- the two stubs


class UniformStub:
    """Flat over the vocabulary: knows nothing at all, not even the pool."""

    def __init__(self):
        self.cfg = None

    def __call__(self, ids):
        B, T = ids.shape
        return torch.zeros(B, T, V), None


class PerfectStub:
    """Probability ~1.0 on the true next token."""

    def __init__(self, tok, queries):
        self.tok = tok
        self.cfg = None

    def __call__(self, ids):
        B, T = ids.shape
        logits = torch.zeros(B, T, V)
        for b in range(B):
            row = ids[b].tolist()
            for t in range(T - 1):
                logits[b, t, row[t + 1]] = 60.0  # ~probability 1 after softmax
        return logits, None


def test_perfect_model_recovers_the_unconditional_ceiling():
    recs = bios.generate_records(6, 0)
    qs = probe_queries(recs)
    out = recoverable_bits(PerfectStub(TOK, qs), TOK, recs, CPU, baseline="unconditional")
    assert abs(out["bits_per_entity"] - 52.96) / 52.96 < 0.01, out["bits_per_entity"]


def test_perfect_model_recovers_the_length_conditioned_ceiling():
    recs = bios.generate_records(6, 0)
    out = recoverable_bits(PerfectStub(TOK, probe_queries(recs)), TOK, recs, CPU,
                           baseline="length")
    assert abs(out["bits_per_entity"] - 45.30) / 45.30 < 0.02, out["bits_per_entity"]


def test_a_model_at_the_prior_recovers_about_nothing():
    """A uniform model is worse than the pool prior, so recovery must be
    negative rather than clamped to zero -- clamping would bias every arm up."""
    recs = bios.generate_records(6, 0)
    out = recoverable_bits(UniformStub(), TOK, recs, CPU)
    assert out["bits_per_entity"] < 0.0


def test_report_shape_and_per_param():
    recs = bios.generate_records(4, 0)
    out = recoverable_bits(PerfectStub(TOK, probe_queries(recs)), TOK, recs, CPU,
                           n_params=40_560_000)
    assert out["n_entities"] == 4
    assert out["n_queries"] == 4 * len(ATTRIBUTES)
    assert set(out["per_attribute"]) == set(ATTRIBUTES)
    assert out["bits_per_param"] == pytest.approx(
        out["bits_total"] / 40_560_000, rel=1e-9
    )
    assert out["baseline"] == "unconditional"
