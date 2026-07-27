"""Tests for evals.constrain: copy-constrained query decoding.

The held-out failure this guards against: free greedy decoding fills the
name half of a "<name>, <relation>" query with a MEMORIZED training name
instead of copying the prompt's name. With query_tries, IN_QUERY tokens
must walk a trie of prompt spans x relations, so the only emittable keys
are copies from the prompt.

Stubs mirror tests/test_generate.py:
- OpenLoopStub: favors program[i] at decode step i, input-blind.
- TableStub: closed loop; favored next token = tables[row][last input].
"""

import torch

from evals.constrain import QueryTrie, QueryWalker, build_query_tries, extract_spans
from evals.generate import generate_batch_with_stats
from organizer.store import Organizer
from train.tokenizer import get_tok

TOK = get_tok()
V = TOK.VOCAB_SIZE
CPU = torch.device("cpu")


def _stats(**over):
    base = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    base.update(over)
    return base


def _cstats(**over):
    base = _stats(n_constrained_queries=0, n_constraint_dead_ends=0)
    base.update(over)
    return base


class OpenLoopStub:
    """Favors program[i] at decode step i, ignoring all input."""

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


# ---------------------------------------------------------------- extract_spans


def test_extract_spans_ngrams_and_punctuation_stripping():
    spans = extract_spans("Who wrote The Ritual?")
    assert "Ritual" in spans  # trailing '?' progressively stripped
    assert "Ritual?" in spans  # raw boundary form kept as a variant too
    assert "The Ritual" in spans
    assert "wrote The Ritual" in spans
    assert "Who wrote The Ritual" in spans


def test_extract_spans_progressive_boundary_variants():
    # N-grams are built over ORIGINAL words; boundary punctuation is then
    # stripped progressively (one char + exposed whitespace per stage) and
    # every stage is kept, so quoted titles appear both raw and clean while
    # punctuation internal to the joined gram (the comma after "Dune")
    # survives. No variant may carry boundary whitespace.
    spans = extract_spans('She read "Dune, Part Two" twice')
    assert "Dune" in spans  # '"Dune,' fully cleaned at both boundaries
    assert "Dune, Part" in spans  # comma is internal to the gram: kept
    assert "Dune, Part Two" in spans  # quotes at the boundaries stripped
    assert '"Dune, Part Two"' in spans  # raw gram kept as a variant
    for v in spans:
        assert v == v.strip()
        assert any(c.isalnum() for c in v)


def test_extract_spans_standalone_punctuation_word():
    # Standalone punctuation words may survive inside raw variants
    # ("( Ritual"), but "Ritual" itself must be present and no variant may
    # have boundary whitespace or consist of punctuation only.
    spans = extract_spans("( Ritual )")
    assert "Ritual" in spans
    for v in spans:
        assert v == v.strip()
        assert any(c.isalnum() for c in v)


def test_extract_spans_keeps_internal_punctuation():
    # Punctuation INSIDE a single word (not at an edge) survives cleaning.
    spans = extract_spans("the state-of-the-art design")
    assert "state-of-the-art" in spans
    assert "state-of-the-art design" in spans


def test_extract_spans_keeps_initialism_trailing_period():
    # Real PopQA subject "Aston Villa F.C.": the gold key needs the span
    # WITH its final period, which word-level cleaning used to destroy.
    # The question has "F.C. from?" (trailing period + space), so the raw
    # gram "Aston Villa F.C." must survive as an exact variant.
    spans = extract_spans("What country is Aston Villa F.C. from?")
    assert "Aston Villa F.C." in spans
    assert "Aston Villa F.C" in spans  # progressive stage also present


def test_extract_spans_keeps_trailing_exclamation():
    # Real PopQA subject "Home Free!": glued to the question mark as
    # "Home Free!?", the progressive stages must include the exact form.
    spans = extract_spans("Who is the author of Home Free!?")
    assert "Home Free!?" in spans  # raw gram
    assert "Home Free!" in spans  # one stage stripped: the gold subject
    assert "Home Free" in spans  # fully stripped


def test_extract_spans_candidate_count_sane():
    q = 'Who was the director of "The Long, Strange Trip (Part Two)!?" back in early 1999?'
    assert len(q.split()) == 15
    assert len(extract_spans(q)) < 2000


def test_extract_spans_dedupe_and_max_words():
    spans = extract_spans("go go go", max_words=2)
    assert spans.count("go") == 1
    assert spans.count("go go") == 1
    assert "go go go" not in spans


def test_extract_spans_drops_empty_long_and_pure_punctuation():
    long_word = "x" * 61
    spans = extract_spans(f"a {long_word} ... b")
    assert long_word not in spans
    assert "..." not in spans
    assert "" not in spans
    assert "a" in spans
    assert "b" in spans


# ---------------------------------------------------------------- trie / walker


def test_walker_permits_only_candidate_paths():
    trie = QueryTrie(TOK, ["The Ritual", "Frog"], ["author"])
    k1 = TOK.encode("The Ritual, author")
    k2 = TOK.encode("Frog, author")

    w = trie.walker()
    assert set(w.allowed()) == {k1[0], k2[0]}
    for tid in k1:
        allowed = w.allowed()
        assert tid in allowed
        assert TOK.DB_RETRIEVE not in allowed  # key not complete yet
        w.advance(tid)
    assert set(w.allowed()) == {TOK.DB_RETRIEVE}  # complete key: retrieve only

    w.advance(TOK.DB_RETRIEVE)  # finishing a query returns to the root
    assert set(w.allowed()) == {k1[0], k2[0]}
    for tid in k2:
        assert tid in w.allowed()
        w.advance(tid)
    assert set(w.allowed()) == {TOK.DB_RETRIEVE}


def test_walker_unknown_advance_and_reset_return_to_root():
    trie = QueryTrie(TOK, ["The Ritual", "Frog"], ["author"])
    k1 = TOK.encode("The Ritual, author")
    w = trie.walker()
    root_allowed = set(w.allowed())

    w.advance(k1[0])
    assert set(w.allowed()) == {k1[1]}
    w.advance(TOK.EOT)  # off-trie id
    assert set(w.allowed()) == root_allowed

    w.advance(k1[0])
    w.reset()
    assert set(w.allowed()) == root_allowed


def test_build_query_tries_one_per_prompt():
    tries = build_query_tries(["Who wrote The Ritual?", "Ping"], ["author"], TOK)
    assert len(tries) == 2

    w = tries[0].walker()
    for tid in TOK.encode("The Ritual, author"):
        assert tid in w.allowed()
        w.advance(tid)
    assert TOK.DB_RETRIEVE in w.allowed()

    w2 = tries[1].walker()
    assert set(w2.allowed()) == {TOK.encode("Ping, author")[0]}


# ---------------------------------------------------------------- decoding


def _memorized_table(prompt):
    """Table whose free argmax emits the MEMORIZED key "Frog Book, author":
    prompt end -> DB_START -> wrong key -> DB_RETRIEVE; then <|db_retrieve|>
    fed back => " wrong" (miss path), <|db_end|> fed back => " indeed"
    (hit path, only reachable when forcing spliced a value)."""
    p_last = TOK.encode(prompt)[-1]
    wrong_ids = TOK.encode("Frog Book, author")
    return _merge(
        _chain([p_last, TOK.DB_START] + wrong_ids + [TOK.DB_RETRIEVE]),
        _chain([TOK.DB_RETRIEVE] + TOK.encode(" wrong") + [TOK.EOT]),
        _chain([TOK.DB_END] + TOK.encode(" indeed") + [TOK.EOT]),
    )


def test_constrained_query_copies_prompt_span_and_hits():
    prompt = "Who wrote The Ritual?"
    org = Organizer()
    org.add("The Ritual", "author", "Mara Venn")
    table = _memorized_table(prompt)

    # Premise: unconstrained, the model emits the memorized key and misses.
    free_texts, free_stats = generate_batch_with_stats(
        TableStub([table]), TOK, [prompt], max_new=64, organizer=org, device=CPU
    )
    assert free_texts == ["<|db_start|>Frog Book, author<|db_retrieve|> wrong"]
    assert free_stats == _stats(n_lookups=1, n_misses=1)

    trie = QueryTrie(TOK, ["The Ritual"], ["author"])
    texts, stats = generate_batch_with_stats(
        TableStub([table]),
        TOK,
        [prompt],
        max_new=64,
        organizer=org,
        device=CPU,
        query_tries=[trie],
    )
    emitted_key = texts[0].split("<|db_start|>")[1].split("<|db_retrieve|>")[0]
    name, _, relation = emitted_key.rpartition(", ")
    assert name in prompt  # the name half is a literal copy of a prompt span
    assert relation == "author"
    assert " Mara Venn<|db_end|>" in texts[0]
    assert texts == [
        "<|db_start|>The Ritual, author<|db_retrieve|> Mara Venn<|db_end|> indeed"
    ]
    assert stats == _cstats(n_lookups=1, n_hits=1, n_constrained_queries=1)


def test_query_tries_none_matches_param_absent():
    prompt = "Who wrote The Ritual?"

    def run(**kw):
        org = Organizer()
        org.add("Frog Book", "author", "Nadia Quill")
        return generate_batch_with_stats(
            TableStub([_memorized_table(prompt)]),
            TOK,
            [prompt],
            max_new=64,
            organizer=org,
            device=CPU,
            **kw,
        )

    texts_absent, stats_absent = run()
    texts_none, stats_none = run(query_tries=None)
    assert texts_none == texts_absent
    assert stats_none == stats_absent
    assert texts_absent == [
        "<|db_start|>Frog Book, author<|db_retrieve|> Nadia Quill<|db_end|> indeed"
    ]
    assert stats_absent == _stats(n_lookups=1, n_hits=1)
    assert "n_constrained_queries" not in stats_absent
    assert "n_constraint_dead_ends" not in stats_absent


def test_walker_resets_between_two_lookups():
    org = Organizer()
    org.add("The Ritual", "author", "Mara Venn")
    trie = QueryTrie(TOK, ["The Ritual"], ["author"])
    key = TOK.encode("The Ritual, author")
    val = TOK.encode(" Mara Venn")
    junk = TOK.encode(" junk")[0]

    # Per query: key + <|db_retrieve|> + forced (value + <|db_end|>); the
    # junk steps are all overridden by the constraint or the force queue.
    seg = len(key) + 1 + len(val) + 1
    prog = ([TOK.DB_START] + [junk] * seg) * 2 + [TOK.EOT]
    texts, stats = generate_batch_with_stats(
        OpenLoopStub([prog]),
        TOK,
        ["Who wrote The Ritual?"],
        max_new=64,
        organizer=org,
        device=CPU,
        query_tries=[trie],
    )
    one = "<|db_start|>The Ritual, author<|db_retrieve|> Mara Venn<|db_end|>"
    assert texts == [one + one]  # second query walks the full key from the root
    assert stats == _cstats(n_lookups=2, n_hits=2, n_constrained_queries=2)


def test_empty_trie_dead_end_falls_back_unconstrained():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    q_ids = TOK.encode("Kai Nakamura, major")
    val = TOK.encode(" Communications")
    junk = TOK.encode(" junk")[0]
    prog = (
        [TOK.DB_START]
        + q_ids
        + [TOK.DB_RETRIEVE]
        + [junk] * (len(val) + 1)
        + [TOK.EOT]
    )
    trie = QueryTrie(TOK, [], ["author"])  # no spans -> no candidates
    texts, stats = generate_batch_with_stats(
        OpenLoopStub([prog]),
        TOK,
        ["p"],
        max_new=64,
        organizer=org,
        device=CPU,
        query_tries=[trie],
    )
    assert texts == [
        "<|db_start|>Kai Nakamura, major<|db_retrieve|> Communications<|db_end|>"
    ]
    # Every IN_QUERY step (query tokens + retrieve) fell back and was counted.
    assert stats == _cstats(
        n_lookups=1,
        n_hits=1,
        n_constrained_queries=1,
        n_constraint_dead_ends=len(q_ids) + 1,
    )


# ---------------------------------------------------------------- possessives


def test_extract_spans_emits_possessive_and_nonpossessive_variant():
    # PopQA occupation prompts say "George Rankin's occupation"; cleaning
    # yields "Rankin's" but never "Rankin", so the gold key
    # "George Rankin, occupation" would be unemittable without a
    # non-possessive variant. Both forms must appear as spans.
    spans = extract_spans("What is George Rankin's occupation?")
    assert "George Rankin's" in spans
    assert "George Rankin" in spans
    assert "Rankin's" in spans
    assert "Rankin" in spans
    # The unicode right-single-quote form behaves the same way.
    uni = extract_spans("What is George Rankin\u2019s occupation?")
    assert "George Rankin\u2019s" in uni
    assert "George Rankin" in uni


def test_possessive_prompt_trie_can_emit_nonpossessive_gold_key():
    # End-to-end: a trie built from the possessive prompt's spans must let
    # the model walk the NON-possessive gold key "George Rankin, occupation"
    # token by token, with DB_RETRIEVE allowed only at its terminal node.
    spans = extract_spans("What is George Rankin's occupation?")
    trie = QueryTrie(TOK, spans, ["occupation"])
    gold = TOK.encode("George Rankin, occupation")

    w = trie.walker()
    for i, tid in enumerate(gold):
        allowed = w.allowed()
        assert tid in allowed, f"token {i} ({tid}) not allowed"
        if i < len(gold) - 1:
            assert TOK.DB_RETRIEVE not in allowed  # key not complete yet
        w.advance(tid)
    # At the terminal node the only allowed token is DB_RETRIEVE.
    assert set(w.allowed()) == {TOK.DB_RETRIEVE}


# ---------------------------------------------------------------- cap-hit reset


class _SpyTrie(QueryTrie):
    """QueryTrie that records the most recent walker it handed out, so a
    test can inspect the cursor state after generation ends."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_walker: QueryWalker | None = None

    def walker(self) -> "QueryWalker":
        self.last_walker = super().walker()
        return self.last_walker


def test_constrained_cap_hit_resets_walker_to_root():
    # A constrained query that dies via QUERY_TOKEN_CAP (stub never emits
    # DB_RETRIEVE) must leave the row's walker back at the root when the
    # query ends, without requiring a later DB_START to mask the reset.
    # The single candidate is long enough (>32 tokens) that the walker is
    # genuinely mid-trie when the cap fires.
    long_span = " ".join(f"w{i}" for i in range(40))
    assert len(TOK.encode(f"{long_span}, author")) > 32  # cap fires mid-trie

    trie = _SpyTrie(TOK, [long_span], ["author"])
    root_allowed = set(trie.walker().allowed())  # fresh walker at root
    assert root_allowed  # sanity: the candidate's first token

    # Stub program: DB_START, then 32 junk IN_QUERY steps (the walker picks
    # the only allowed path token at each, so DB_RETRIEVE is never emitted),
    # then EOT to end -- no second DB_START.
    junk = TOK.encode(" zzz")[0]
    assert junk != TOK.DB_RETRIEVE
    prog = [TOK.DB_START] + [junk] * 32 + [TOK.EOT]

    org = Organizer()  # store ON, but no entry -> irrelevant (query never retrieves)
    texts, stats = generate_batch_with_stats(
        OpenLoopStub([prog]),
        TOK,
        ["p"],
        max_new=64,
        organizer=org,
        device=CPU,
        query_tries=[trie],
    )

    # The malformed query was recorded; no retrieve ever happened.
    assert stats == _cstats(n_malformed=1)
    # The walker used for row 0 is back at the root's allowed set, with no
    # second DB_START having fired (the program jumps straight to EOT).
    used = trie.last_walker
    assert used is not None
    assert set(used.allowed()) == root_allowed
