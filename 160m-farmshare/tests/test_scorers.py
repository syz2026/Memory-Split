"""Tests for evals.scorers (answer parsing, end-to-end scoring, jsonl I/O)
and the pure log-likelihood helper from evals.natural (offline)."""

import json
import math

import pytest
import torch

from corpusgen.records import QAItem
from evals.natural import loglikelihood_choice_scores
from evals.scorers import normalize_answer, parse_answer, save_results, score_items
from train.tokenizer import get_tok

TOK = get_tok()
V = TOK.VOCAB_SIZE
CPU = torch.device("cpu")


class OpenLoopStub:
    """Favors program[i] at decode step i regardless of input; programs are
    assigned to batch rows in prefill order (supports chunked batches)."""

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


# ---------------------------------------------------------------- parse_answer


def test_parse_answer_missing():
    assert parse_answer("Reasoning: no conclusion reached") is None


def test_parse_answer_simple():
    assert parse_answer("Reasoning: 12 + 7 = 19.\nAnswer: 19") == "19"


def test_parse_answer_takes_last():
    text = "Answer: 1\nintermediate text\nAnswer: 2\ntrailing line"
    assert parse_answer(text) == "2"


def test_parse_answer_stops_at_eot_marker():
    assert parse_answer("Answer: 7<|eot|>garbage after eot") == "7"


def test_parse_answer_stops_at_newline():
    assert parse_answer("Answer: Tacoma\nExtra: junk") == "Tacoma"


def test_parse_answer_empty_after_colon():
    assert parse_answer("Answer:") == ""


def test_parse_answer_strips_whitespace():
    assert parse_answer("Answer:   yes  \nmore") == "yes"


# ------------------------------------------------------------ normalize_answer


def test_normalize_answer():
    assert normalize_answer("  The  Answer. ") == "the answer"
    assert normalize_answer("42.") == "42"
    assert normalize_answer("YES") == "yes"
    assert normalize_answer("a\tb\n c") == "a b c"
    assert normalize_answer("") == ""


# ------------------------------------------------------------------ score_items


def _items():
    return [
        QAItem(qid="q1", task="igsm", prompt="P is 12 + 7. Reasoning:",
               answer="19", meta={"template": "t0"}),
        QAItem(qid="q2", task="igsm", prompt="Q is 3 * 4. Reasoning:",
               answer="12", meta={"template": "t1"}),
    ]


def _programs():
    good = TOK.encode(" compute 12 + 7 = 19.\nAnswer: 19") + [TOK.EOT]
    bad = TOK.encode(" compute 3 * 4 = 11.\nAnswer: 11") + [TOK.EOT]
    return [good, bad]


def test_score_items_end_to_end():
    model = OpenLoopStub(_programs())
    rows, stats = score_items(model, TOK, _items(), organizer=None, device=CPU)
    assert [r["qid"] for r in rows] == ["q1", "q2"]
    assert [r["correct"] for r in rows] == [True, False]
    assert rows[0]["pred"] == "19" and rows[0]["answer"] == "19"
    assert rows[1]["pred"] == "11" and rows[1]["answer"] == "12"
    assert rows[0]["task"] == "igsm"
    assert rows[0]["meta"] == {"template": "t0"}
    assert stats == {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}


def test_score_items_batches_chunked():
    # batch_size=1 forces two prefills; stub cursor hands out programs in order
    model = OpenLoopStub(_programs())
    rows, _ = score_items(
        model, TOK, _items(), organizer=None, device=CPU, batch_size=1
    )
    assert [r["correct"] for r in rows] == [True, False]


def test_score_items_unparseable_prediction():
    prog = TOK.encode(" rambling with no final line") + [TOK.EOT]
    item = QAItem(qid="q", task="igsm", prompt="P. Reasoning:", answer="5", meta={})
    rows, _ = score_items(OpenLoopStub([prog]), TOK, [item], organizer=None, device=CPU)
    assert rows[0]["correct"] is False
    assert rows[0]["pred"] is None


# ----------------------------------------------------------------- save_results


def test_save_results_round_trip(tmp_path):
    rows = [
        {"qid": "a", "task": "igsm", "correct": True, "pred": "1", "answer": "1",
         "meta": {"template": "t0"}},
        {"qid": "b", "task": "igsm", "correct": False, "pred": None, "answer": "2",
         "meta": {}},
    ]
    path = tmp_path / "results.jsonl"
    save_results(rows, path)
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == rows


# ------------------------------------------- natural: pure scoring helper


class UniformModel:
    """forward() returns all-zero logits => uniform distribution over V."""

    device = CPU

    def forward(self, idx):
        B, T = idx.shape
        return torch.zeros(B, T, V), None


def test_loglikelihood_choice_scores_uniform():
    choices = [" yes", " absolutely certain today"]
    out = loglikelihood_choice_scores(UniformModel(), TOK, "The verdict is", choices, CPU)
    lp_tok = -math.log(V)
    n0 = len(TOK.encode(choices[0]))
    n1 = len(TOK.encode(choices[1]))
    assert n0 != n1  # exercises mixed lengths in one padded batch
    (s0, m0), (s1, m1) = out
    assert math.isclose(s0, n0 * lp_tok, rel_tol=1e-5)
    assert math.isclose(s1, n1 * lp_tok, rel_tol=1e-5)
    assert math.isclose(m0, lp_tok, rel_tol=1e-5)
    assert math.isclose(m1, lp_tok, rel_tol=1e-5)


class BiasedModel:
    """Puts extra logit mass on one token id at every position."""

    device = CPU

    def __init__(self, fav_id):
        self.fav_id = fav_id

    def forward(self, idx):
        B, T = idx.shape
        logits = torch.zeros(B, T, V)
        logits[:, :, self.fav_id] = 5.0
        return logits, None


def test_loglikelihood_choice_scores_prefers_favored_token():
    fav = TOK.encode(" yes")[0]
    out = loglikelihood_choice_scores(BiasedModel(fav), TOK, "The verdict is",
                                      [" yes", " no"], CPU)
    assert out[0][0] > out[1][0]


@pytest.mark.slow
def test_run_natural_suite_smoke():
    """Needs network (HF datasets). Excluded by default via -m 'not slow'."""
    from evals.natural import run_natural_suite

    out = run_natural_suite(UniformModel(), TOK, CPU, tasks=("piqa",), limit=4)
    assert "piqa" in out
    for key in ("acc", "acc_norm", "correct_prob", "n"):
        assert key in out["piqa"]
    assert 0.0 <= out["piqa"]["acc"] <= 1.0
