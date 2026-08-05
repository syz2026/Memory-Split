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


def test_padded_vocab_ids_are_never_generated():
    """Ids 50261..50303 pad the vocab to a multiple of 64 and map to no
    token. An undertrained model puts real mass there; selecting one used to
    crash the tokenizer partway through an evaluation."""
    pad_id = TOK.VOCAB_SIZE - 1
    assert pad_id >= TOK.N_REAL_TOKENS

    class AlwaysPad:
        cfg = None

        def forward_step(self, idx, cache):
            B, T = idx.shape
            logits = torch.zeros(B, T, TOK.VOCAB_SIZE)
            logits[:, -1, pad_id] = 100.0  # overwhelmingly favour the pad id
            return logits, {}

    texts = generate_batch(AlwaysPad(), TOK, ["p"], max_new=4, device=CPU)
    assert isinstance(texts[0], str)  # decoded rather than raised


def test_decode_drops_unmapped_padding_ids():
    real = TOK.encode(" hello")
    assert TOK.decode(real + [TOK.VOCAB_SIZE - 1]) == TOK.decode(real)


# ---------------- the padding bug that produced four generations of "floor"


def test_batched_decoding_equals_unbatched_at_mixed_lengths():
    """`generate_batch` used to left-pad every row to the batch maximum with
    EOT and attend over the pads. EOT is the document separator, so a short
    prompt behind 32 of them read as a fresh document and the model
    hallucinated its premises.

    Measured on the real 31,280-step checkpoint over 64 held-out iGSM items:
    3/64 batched against 60/64 one at a time. That gap is the entire
    "the endpoint floors at every scale and difficulty" result.

    Batching must never change a generation.
    """
    import torch
    from evals.generate import generate_batch
    from train.model import GPT, PRESETS
    from train.tokenizer import get_tok

    torch.manual_seed(0)
    cfg = PRESETS["d8m"]
    cfg.ctx = 256
    model = GPT(cfg).eval()
    tok = get_tok()
    # Deliberately ragged: the shortest is a fraction of the longest.
    prompts = [
        "The number of hollow barrels in the Armory is",
        "The number of dusty whistles in the Wharf is 3. The number of pallid "
        "buckets in the Forge is",
        "A. " * 40 + "The number of crimson mallets in the Terrace is",
        "Q:",
    ]
    together = generate_batch(model, tok, prompts, 24, torch.device("cpu"))
    alone = [generate_batch(model, tok, [p], 24, torch.device("cpu"))[0]
             for p in prompts]
    for i, (a, b) in enumerate(zip(together, alone)):
        assert a == b, (
            f"prompt {i} decoded differently in a batch than alone.\n"
            f"  batched : {a!r}\n  alone   : {b!r}"
        )


def test_equal_length_prompts_still_share_one_batch():
    """The fix groups by exact length; prompts that already agree must not be
    split into singletons, or evaluation throughput collapses."""
    import torch
    from evals.generate import generate_batch
    from train.model import GPT, PRESETS
    from train.tokenizer import get_tok

    torch.manual_seed(0)
    cfg = PRESETS["d8m"]
    cfg.ctx = 128
    model = GPT(cfg).eval()
    tok = get_tok()
    # Built from token ids so the lengths agree by construction rather than by
    # a guess about how the BPE splits English words.
    prompts = [tok.decode([tid] * 5) for tid in (262, 290, 318)]
    assert len({len(tok.encode(p)) for p in prompts}) == 1, "fixture must agree"
    together = generate_batch(model, tok, prompts, 12, torch.device("cpu"))
    alone = [generate_batch(model, tok, [p], 12, torch.device("cpu"))[0]
             for p in prompts]
    assert together == alone
