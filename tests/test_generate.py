"""Tests for evals.generate: batched greedy decoding.

Stubs follow the model contract:
    forward_step(idx[B, T], cache) -> (logits[B, T, V], cache)
    first call cache=None with the full (left-padded) prompt, then T=1 steps.

OpenLoopStub emits program[i] where i is the number of decode steps so far,
regardless of input. TableStub is closed loop: the favored next token is a
pure function of the previously fed token, which proves tokens really pass
through forward_step rather than being spliced into the output.
"""

import torch

from evals.generate import generate_batch
from train.tokenizer import get_tok

TOK = get_tok()
V = TOK.VOCAB_SIZE
CPU = torch.device("cpu")


class OpenLoopStub:
    """Favors program[i] at decode step i, ignoring all input.

    Programs are assigned to rows in order across prefill calls, so chunked
    callers (e.g. score_items batches) consume them sequentially.
    """

    def __init__(self, programs, vocab=V):
        self.programs = [list(p) for p in programs]
        self._cursor = 0
        self.V = vocab
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
        logits = torch.zeros(B, T, self.V)
        for b, prog in enumerate(cache["progs"]):
            i = cache["step"]
            fav = prog[i] if i < len(prog) else TOK.EOT
            logits[b, -1, fav] = 1.0
        return logits, cache


class TableStub:
    """Closed loop: favored next token = tables[row][last input token id]."""

    def __init__(self, tables, vocab=V):
        self.tables = tables
        self.V = vocab
        self.device = CPU

    def forward_step(self, idx, cache):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.V)
        for b in range(B):
            fav = self.tables[b].get(int(idx[b, -1]), TOK.EOT)
            logits[b, -1, fav] = 1.0
        return logits, {}


def _chain(ids):
    """Transition dict following ids in order; requires distinct keys."""
    keys = ids[:-1]
    assert len(set(keys)) == len(keys), f"chain needs distinct keys: {keys}"
    return dict(zip(ids, ids[1:]))


def test_open_loop_emits_scripted_text():
    prog = TOK.encode(" Answer: 7") + [TOK.EOT]
    texts = generate_batch(
        OpenLoopStub([prog]), TOK, ["Question:"], max_new=32, device=CPU
    )
    assert texts[0] == " Answer: 7"


def test_eot_stops_generation_early():
    prog = TOK.encode(" a") + [TOK.EOT] + TOK.encode(" never")
    texts = generate_batch(OpenLoopStub([prog]), TOK, ["p"], max_new=64, device=CPU)
    assert texts[0] == " a"


def test_max_new_caps_generation():
    ids = TOK.encode(" one two three four five six")
    texts = generate_batch(OpenLoopStub([ids]), TOK, ["p"], max_new=4, device=CPU)
    assert len(TOK.encode(texts[0])) == 4


def test_stop_at_eot_false_keeps_decoding():
    prog = TOK.encode(" a") + [TOK.EOT] + TOK.encode(" b")
    texts = generate_batch(
        OpenLoopStub([prog]), TOK, ["p"], max_new=8, device=CPU, stop_at_eot=False
    )
    assert "b" in texts[0]


def test_closed_loop_tokens_pass_through_forward_step():
    """The continuation is reachable only if each emitted token is fed back."""
    ids = TOK.encode(" x y z") + [TOK.EOT]
    start = TOK.encode("p")[-1]
    table = _chain([start, *ids])
    texts = generate_batch(TableStub([table]), TOK, ["p"], max_new=64, device=CPU)
    assert texts[0] == " x y z"


def test_batch_rows_are_independent():
    a = TOK.encode(" alpha") + [TOK.EOT]
    b = TOK.encode(" beta") + [TOK.EOT]
    texts = generate_batch(OpenLoopStub([a, b]), TOK, ["p", "q"], max_new=32, device=CPU)
    assert texts == [" alpha", " beta"]


def test_empty_prompt_list():
    assert generate_batch(OpenLoopStub([]), TOK, [], max_new=8, device=CPU) == []
