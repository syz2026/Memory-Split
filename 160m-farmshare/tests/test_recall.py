"""Tests for evals.recall: substring scoring, per-attribute breakdown, and
the bits-in-weights formula."""

import math

import pytest
import torch

from corpusgen.records import QAItem
from evals.recall import bits_in_weights, recall_accuracy
from train.tokenizer import get_tok

TOK = get_tok()
V = TOK.VOCAB_SIZE
CPU = torch.device("cpu")


class OpenLoopStub:
    def __init__(self, programs):
        self.programs = [list(p) for p in programs]
        self._cursor = 0
        self.device = CPU

    def forward_step(self, idx, cache):
        B, T = idx.shape
        if cache is None:
            progs = self.programs[self._cursor : self._cursor + B]
            assert len(progs) == B, "stub ran out of programs"
            self._cursor += B
            cache = {"progs": progs, "step": 0}
        else:
            cache["step"] += 1
        logits = torch.zeros(B, T, V)
        for b, prog in enumerate(cache["progs"]):
            i = cache["step"]
            fav = prog[i] if i < len(prog) else TOK.EOT
            logits[b, -1, fav] = 1.0
        return logits, cache


def _probe(qid, answer, relation):
    return QAItem(qid=qid, task="recall", prompt=f"{qid} prompt is",
                  answer=answer, meta={"relation": relation})


def test_recall_substring_and_per_attribute():
    probes = [
        _probe("r1", "Communications", "major"),
        _probe("r2", "Tacoma", "birth_city"),
        _probe("r3", "Stanford University", "university"),
    ]
    programs = [
        # value mid-sentence (substring must match, not exact match)
        TOK.encode(" Communications, as noted earlier.") + [TOK.EOT],
        # case difference only: normalization must equate
        TOK.encode(" tacoma") + [TOK.EOT],
        # wrong value
        TOK.encode(" Harvard University") + [TOK.EOT],
    ]
    out = recall_accuracy(OpenLoopStub(programs), TOK, probes, mode="closed",
                          organizer=None, device=CPU)
    assert out["n"] == 3
    assert out["overall"] == pytest.approx(2 / 3)
    assert out["per_attribute"] == {
        "major": 1.0, "birth_city": 1.0, "university": 0.0
    }
    assert out["stats"]["n_lookups"] == 0


def test_recall_plain_containment_semantics():
    # Scoring is plain string containment on normalized text (by design; no
    # word-boundary logic): "tac" inside "tacoma" counts.
    probes = [_probe("r1", "tac", "major")]
    programs = [TOK.encode(" tacoma") + [TOK.EOT]]
    out = recall_accuracy(OpenLoopStub(programs), TOK, probes, mode="closed",
                          organizer=None, device=CPU)
    assert out["overall"] == 1.0  # plain containment, by design


def test_recall_mode_validation_and_off_equals_closed_signature():
    probes = [_probe("r1", "Tacoma", "birth_city")]
    programs = [TOK.encode(" Tacoma") + [TOK.EOT]]
    out = recall_accuracy(OpenLoopStub(programs), TOK, probes, mode="off",
                          organizer=None, device=CPU)
    assert out["overall"] == 1.0
    with pytest.raises(ValueError):
        recall_accuracy(OpenLoopStub(programs), TOK, probes, mode="banana",
                        organizer=None, device=CPU)


def test_recall_mode_on_uses_interception():
    from organizer.store import Organizer

    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    probes = [_probe("r1", "Communications", "major")]
    program = [
        [TOK.DB_START]
        + TOK.encode("Kai Nakamura, major")
        + [TOK.DB_RETRIEVE]
        # open-loop continuation after forced " Communications<|db_end|>":
        # positions shift by the forced length; pad with EOTs via program end
    ]
    # Open-loop stub keeps emitting per its counter; after the forced value the
    # counter has advanced past program end => EOT => stop. The forced value
    # itself supplies the substring.
    out = recall_accuracy(OpenLoopStub(program), TOK, probes, mode="on",
                          organizer=org, device=CPU, max_new=48)
    assert out["overall"] == 1.0
    assert out["stats"] == {"n_lookups": 1, "n_hits": 1, "n_misses": 0,
                            "n_malformed": 0}


# -------------------------------------------------------------- bits_in_weights


def test_bits_perfect_accuracy_hand_checked():
    # acc=1.0, N=100, pool=1024: (1 - 1/1024)/(1 - 1/1024) * 100 * 10 = 1000
    bits = bits_in_weights({"major": 1.0}, n_entities=100,
                           pool_sizes={"major": 1024})
    assert bits == pytest.approx(1000.0)


def test_bits_chance_accuracy_is_zero():
    g = 1 / 1024
    bits = bits_in_weights({"major": g}, n_entities=100,
                           pool_sizes={"major": 1024})
    assert bits == pytest.approx(0.0)


def test_bits_below_chance_clamped_to_zero():
    bits = bits_in_weights({"major": 0.0}, n_entities=100,
                           pool_sizes={"major": 1024})
    assert bits == 0.0


def test_bits_sums_attributes_and_noninteger_pools():
    per_attr = {"major": 1.0, "birth_date": 1.0}
    pools = {"major": 1024, "birth_date": 27759.0}  # non-integer-friendly
    bits = bits_in_weights(per_attr, n_entities=10, pool_sizes=pools)
    expected = 10 * math.log2(1024) + 10 * math.log2(27759.0)
    assert bits == pytest.approx(expected)


def test_bits_partial_accuracy():
    pool = 100
    g = 1 / pool
    acc = 0.5
    bits = bits_in_weights({"employer": acc}, n_entities=1000,
                           pool_sizes={"employer": pool})
    expected = (acc - g) / (1 - g) * 1000 * math.log2(pool)
    assert bits == pytest.approx(expected)
