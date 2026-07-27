"""Tests for evals.generate: batched greedy decoding + lookup interception.

Stubs follow the Task-4 model contract:
    forward_step(idx[B, T], cache) -> (logits[B, T, V], cache)
    first call cache=None with the full (left-padded) prompt, then T=1 steps.

Two stub flavors:
- OpenLoopStub: emits program[i] where i = number of decode steps so far,
  regardless of input (open loop).
- TableStub: closed loop; the favored next token is a pure function of the
  previously fed token (table[last_input]). This proves forced tokens really
  pass through forward_step: the hit continuation is reachable only via a
  <|db_end|> input, the miss continuation only via a <|db_retrieve|> input.
"""

import torch

from evals.generate import generate_batch, generate_batch_with_stats
from organizer.store import Organizer
from train.tokenizer import get_tok

TOK = get_tok()
V = TOK.VOCAB_SIZE
CPU = torch.device("cpu")


def _stats(**over):
    base = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    base.update(over)
    return base


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


def _merge(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            assert k not in out, f"transition key collision: {k}"
            out[k] = v
    return out


# ---------------------------------------------------------------- plain greedy


def test_open_loop_emits_scripted_text():
    prog = TOK.encode(" The answer is 42.") + [TOK.EOT]
    model = OpenLoopStub([prog])
    texts, stats = generate_batch_with_stats(
        model, TOK, ["Question:"], max_new=32, organizer=None, device=CPU
    )
    assert texts == [" The answer is 42."]
    assert stats == _stats()


def test_generate_batch_is_thin_wrapper():
    prog = TOK.encode(" hi") + [TOK.EOT]
    texts = generate_batch(
        OpenLoopStub([prog]), TOK, ["p"], max_new=8, organizer=None, device=CPU
    )
    assert texts == [" hi"]


def test_eot_stops_generation_early():
    prog = TOK.encode(" brief") + [TOK.EOT] + TOK.encode(" should never appear")
    texts = generate_batch(
        OpenLoopStub([prog]), TOK, ["p"], max_new=64, organizer=None, device=CPU
    )
    assert texts == [" brief"]


def test_max_new_caps_generation():
    ids = TOK.encode(" one two three four five six seven eight nine ten")
    assert len(ids) >= 8
    texts = generate_batch(
        OpenLoopStub([ids]), TOK, ["p"], max_new=4, organizer=None, device=CPU
    )
    assert texts == [TOK.decode(ids[:4])]


def test_stop_at_eot_false_keeps_decoding():
    prog = TOK.encode(" a") + [TOK.EOT] + TOK.encode(" b")
    texts = generate_batch(
        OpenLoopStub([prog]),
        TOK,
        ["p"],
        max_new=len(prog),
        organizer=None,
        device=CPU,
        stop_at_eot=False,
    )
    assert texts == [" a<|eot|> b"]


def test_store_off_db_tokens_inert():
    prog = (
        [TOK.DB_START]
        + TOK.encode("Kai Nakamura, major")
        + [TOK.DB_RETRIEVE]
        + TOK.encode(" guess")
        + [TOK.EOT]
    )
    texts, stats = generate_batch_with_stats(
        OpenLoopStub([prog]), TOK, ["p"], max_new=64, organizer=None, device=CPU
    )
    assert texts == ["<|db_start|>Kai Nakamura, major<|db_retrieve|> guess"]
    assert stats == _stats()


# ---------------------------------------------------------------- interception


def _hit_table(prompt, confirm=" yes", wrong=" wrong"):
    """Table emitting DB_START + query + DB_RETRIEVE, then branching on the
    fed-back token: <|db_end|> => confirm (hit path); <|db_retrieve|> => wrong
    (only reachable if forcing did NOT happen)."""
    p_last = TOK.encode(prompt)[-1]
    q_ids = TOK.encode("Kai Nakamura, major")
    confirm_ids = TOK.encode(confirm)
    wrong_ids = TOK.encode(wrong)
    table = _merge(
        _chain([p_last, TOK.DB_START] + q_ids + [TOK.DB_RETRIEVE]),
        _chain([TOK.DB_END] + confirm_ids + [TOK.EOT]),
        _chain([TOK.DB_RETRIEVE] + wrong_ids + [TOK.EOT]),
    )
    return table, q_ids, confirm, wrong


def test_interception_hit_forces_value():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    prompt = "Kai Nakamura majored in"
    table, _, confirm, wrong = _hit_table(prompt)
    texts, stats = generate_batch_with_stats(
        TableStub([table]), TOK, [prompt], max_new=64, organizer=org, device=CPU
    )
    assert texts == [
        "<|db_start|>Kai Nakamura, major<|db_retrieve|> Communications<|db_end|>" + confirm
    ]
    assert " Communications<|db_end|>" in texts[0]
    assert wrong not in texts[0]  # model's own post-retrieve logits were bypassed
    assert stats == _stats(n_lookups=1, n_hits=1)


def test_interception_miss_continues_model_program():
    org = Organizer()
    org.add("Someone Else", "major", "History")  # queried key absent
    prompt = "Kai Nakamura majored in"
    p_last = TOK.encode(prompt)[-1]
    q_ids = TOK.encode("Kai Nakamura, major")
    unk_ids = TOK.encode(" unknown")
    table = _merge(
        _chain([p_last, TOK.DB_START] + q_ids + [TOK.DB_RETRIEVE]),
        _chain([TOK.DB_RETRIEVE] + unk_ids + [TOK.EOT]),
    )
    texts, stats = generate_batch_with_stats(
        TableStub([table]), TOK, [prompt], max_new=64, organizer=org, device=CPU
    )
    assert texts == ["<|db_start|>Kai Nakamura, major<|db_retrieve|> unknown"]
    assert stats == _stats(n_lookups=1, n_misses=1)


def test_malformed_query_cap():
    org = Organizer()
    junk = TOK.encode(" x" * 40)
    assert len(junk) >= 40
    prog = [TOK.DB_START] + junk + [TOK.EOT]
    texts, stats = generate_batch_with_stats(
        OpenLoopStub([prog]), TOK, ["p"], max_new=64, organizer=org, device=CPU
    )
    assert stats == _stats(n_malformed=1)
    # generation continued from the model program after the cap
    assert texts == [TOK.decode([TOK.DB_START] + junk)]


def test_forced_tokens_count_toward_max_new():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    prompt = "Kai Nakamura majored in"
    table, q_ids, _, _ = _hit_table(prompt)
    # budget: DB_START + query + DB_RETRIEVE + exactly ONE forced token
    max_new = 1 + len(q_ids) + 1 + 1
    value_ids = TOK.encode(" Communications")
    texts, stats = generate_batch_with_stats(
        TableStub([table]), TOK, [prompt], max_new=max_new, organizer=org, device=CPU
    )
    assert stats == _stats(n_lookups=1, n_hits=1)
    expected = TOK.decode(
        [TOK.DB_START] + q_ids + [TOK.DB_RETRIEVE] + value_ids[:1]
    )
    assert texts == [expected]
    assert "<|db_end|>" not in texts[0]  # truncated mid-force


# ---------------------------------------------------------------- batching


def test_batch_mixed_prompt_lengths_and_programs():
    prompts = ["Hi", "A much longer prompt about nothing at all", "Mid size prompt"]
    p0 = TOK.encode(" one") + [TOK.EOT]
    p1 = TOK.encode(" two two two two two two")  # no EOT: capped at max_new
    p2 = TOK.encode(" three three") + [TOK.EOT] + TOK.encode(" junk")
    model = OpenLoopStub([p0, p1, p2])
    texts = generate_batch(model, TOK, prompts, max_new=5, organizer=None, device=CPU)
    assert texts[0] == " one"
    assert texts[1] == TOK.decode(p1[:5])
    assert texts[2] == " three three"


def test_batch_hit_and_miss_together():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    prompt_hit = "Kai Nakamura majored in"
    prompt_miss = "Mia Okafor is employed by"
    table_hit, _, confirm, _ = _hit_table(prompt_hit)
    p_last = TOK.encode(prompt_miss)[-1]
    q_miss = TOK.encode("Mia Okafor, employer")
    dunno = TOK.encode(" dunno")
    table_miss = _merge(
        _chain([p_last, TOK.DB_START] + q_miss + [TOK.DB_RETRIEVE]),
        _chain([TOK.DB_RETRIEVE] + dunno + [TOK.EOT]),
    )
    texts, stats = generate_batch_with_stats(
        TableStub([table_hit, table_miss]),
        TOK,
        [prompt_hit, prompt_miss],
        max_new=64,
        organizer=org,
        device=CPU,
    )
    assert texts[0] == (
        "<|db_start|>Kai Nakamura, major<|db_retrieve|> Communications<|db_end|>" + confirm
    )
    assert texts[1] == "<|db_start|>Mia Okafor, employer<|db_retrieve|> dunno"
    assert stats == _stats(n_lookups=2, n_hits=1, n_misses=1)
