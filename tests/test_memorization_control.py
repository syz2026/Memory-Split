"""The control that distinguishes a deaf probe from an empty model.

`probe_control` established that the bare-prefix probe cannot separate trained
from unseen entities even in training phrasing, which left two live readings:
the probe is insensitive, or nothing was stored. `memorization_control` scores
value tokens the way gate 0 does -- full document, teacher-forced -- so it can
tell those apart. These tests hold its mechanics and its verdict boundaries.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from corpusgen import bios
from ops.crowding.memorization_control import (
    BINDING_FLOOR_NATS,
    cohort_nll,
    doc_value_nll,
    verdict,
)
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok


def _tiny():
    """A fresh config, never `PRESETS[...]`.

    The presets are shared mutable dataclasses and other modules in this suite
    assign to their fields (`test_run_evals.py` sets `PRESETS["d8m"].ctx = 64`
    to force a truncation path). Under `pytest-randomly` that mutation can land
    before these tests and silently shrink the context below one biography.
    """
    torch.manual_seed(0)
    return GPT(GPTConfig(n_layer=2, n_head=2, d_model=64, ctx=512)).eval()


def test_scores_only_value_tokens_and_counts_them():
    """The measurement is defined at marked value positions, nowhere else."""
    model, tok = _tiny(), get_tok()
    rec = bios.generate_records(1, 0)[0]

    total, n = doc_value_nll(model, tok, rec, 0, torch.device("cpu"))

    marked = bios.render_bio_marked(rec, 0)
    expected = sum(len(tok.encode(t)) for t, m in marked if m)
    # Position 0 has no predictor; every value token sits after the name, so
    # none is lost to that edge and the counts must agree exactly.
    assert n == expected
    assert n > 0
    assert total > 0
    # An untrained model is near-uniform over a 50k vocabulary, so per-token
    # cost must be large but finite.
    assert 1.0 < total / n < 20.0


def test_reads_the_same_text_training_saw():
    """A different render would measure a document the model never trained on."""
    rec = bios.generate_records(1, 0)[0]
    a = "".join(t for t, _ in bios.render_bio_marked(rec, 3))
    b = bios.render_bio_doc(rec, 3).dense_text()
    assert a == b


def test_untrained_model_shows_no_binding():
    """The null case must read as null, or the control cannot clear a model.

    A randomly initialised model has memorised nothing, so trained and unseen
    cohorts must cost the same and the verdict must be NO BINDINGS.
    """
    model, tok, dev = _tiny(), get_tok(), torch.device("cpu")
    trained = bios.generate_records(6, 0)
    unseen = bios.generate_records(6, 987654321)

    t = cohort_nll(model, tok, trained, [0], dev)
    u = cohort_nll(model, tok, unseen, [0], dev)
    out = verdict(t, u)

    assert not out["bindings_learned"]
    assert "NO BINDINGS" in out["reading"]
    assert abs(out["gap_nats_unseen_minus_trained"]) < 1.0
    assert t["n_documents"] == 6 and t["n_value_tokens"] > 0
    assert math.isclose(t["bits_per_value_token"],
                        t["nats_per_value_token"] / math.log(2))


def test_standard_error_is_clustered_by_person():
    """Two renderings of one person are not two independent observations.

    They share a name and all six attribute values, so pooling them as
    documents divides by sqrt(400) where only 200 people were sampled. That
    reports a tighter interval than the design earns, which matters because the
    paper quotes the interval as the bound on any storage advantage.
    """
    model, tok, dev = _tiny(), get_tok(), torch.device("cpu")
    recs = bios.generate_records(8, 0)

    one = cohort_nll(model, tok, recs, [0], dev)
    two = cohort_nll(model, tok, recs, [0, 1], dev)

    assert one["n_people"] == two["n_people"] == 8
    assert one["n_documents"] == 8 and two["n_documents"] == 16

    # With one exposure the two estimators coincide; with two they must not,
    # and the clustered one must be the wider (more conservative) of the pair.
    assert math.isclose(one["se"], one["se_by_document_unclustered"], rel_tol=1e-9)
    assert two["se"] > two["se_by_document_unclustered"], (
        "clustering by person should widen the interval, not narrow it")


def test_verdict_requires_both_a_floor_and_significance():
    """A gap must clear the floor and stand clear of its own noise."""
    def cell(mean, se):
        return {"nats_per_value_token": mean, "se": se}

    big = BINDING_FLOOR_NATS * 10
    # Large and precise: bindings.
    assert verdict(cell(1.0, 0.01), cell(1.0 + big, 0.01))["bindings_learned"]
    # Large but swamped by noise: not a finding.
    assert not verdict(cell(1.0, big), cell(1.0 + big, big))["bindings_learned"]
    # Precise but below the floor: not a finding.
    tiny_gap = BINDING_FLOOR_NATS / 10
    assert not verdict(cell(1.0, 1e-6), cell(1.0 + tiny_gap, 1e-6))["bindings_learned"]


def test_verdict_reports_direction_not_just_magnitude():
    """Trained cheaper than unseen is the only direction that means storage."""
    def cell(mean, se):
        return {"nats_per_value_token": mean, "se": se}

    big = BINDING_FLOOR_NATS * 10
    # Trained *more expensive* than unseen cannot be evidence of memorisation.
    backwards = verdict(cell(1.0 + big, 0.01), cell(1.0, 0.01))
    assert not backwards["bindings_learned"]
    assert backwards["gap_nats_unseen_minus_trained"] < 0


def test_cohorts_are_disjoint_by_construction():
    """If the seeds collided the control would compare a cohort to itself."""
    trained = bios.generate_records(50, 0)
    unseen = bios.generate_records(50, 987654321)
    assert {r.name for r in trained}.isdisjoint({r.name for r in unseen})


def test_trained_cohort_is_a_prefix_of_the_corpus():
    """The control probes the first N entities and calls them trained.

    That is only true if the generator is prefix-stable; if it is not, the
    'trained' cohort was never in the corpus and the comparison is
    unseen-against-unseen, which would fake a null.
    """
    assert bios.generate_records(2000, 0)[:25] == bios.generate_records(25, 0)
