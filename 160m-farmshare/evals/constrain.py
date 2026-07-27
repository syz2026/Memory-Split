"""Copy-constrained query decoding: restrict query tokens to prompt spans.

Measured failure this addresses (2.5% held-out name accuracy): at query time
the model fills the name half of "<name>, <relation>" with a MEMORIZED
training name instead of copying the prompt's name. The fix constrains query
emission to a token-level trie of candidate keys f"{span}, {relation}",
where span is a word n-gram of the prompt and relation comes from a closed
set — non-copyable names become unemittable.

Candidates are encoded exactly as queries appear between <|db_start|> and
<|db_retrieve|> in training data: tok.encode(f"{span}, {relation}") with NO
leading space (see corpusgen/records.py::lookup_segments).
"""

from __future__ import annotations

_PUNCT = ".,?!:;'\"()[]"
_MAX_SPAN_CHARS = 60

# Possessive suffixes stripped from an n-gram's final word to emit a
# non-possessive variant span (ascii apostrophe + unicode right quote).
_POSSESSIVE_SUFFIXES = ("'s", "\u2019s")


def _leading_stages(s: str) -> list[str]:
    """s plus each progressive left stage: strip ONE leading _PUNCT char,
    then any exposed whitespace, repeating while the boundary is punct."""
    stages = [s]
    while s and s[0] in _PUNCT:
        s = s[1:].lstrip()
        stages.append(s)
    return stages


def _trailing_stages(s: str) -> list[str]:
    """s plus each progressive right stage (mirror of _leading_stages)."""
    stages = [s]
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
        stages.append(s)
    return stages


def extract_spans(prompt: str, max_words: int = 10) -> list[str]:
    """Word n-grams (1..max_words) over the whitespace-split prompt, plus
    progressive boundary-punctuation variants of each.

    N-grams are built over the ORIGINAL words (no per-word cleaning), so
    punctuation internal to the joined string survives — real subjects
    like "Aston Villa F.C." and "Home Free!" stay emittable. For each
    n-gram, variants are the cross-product of progressive left x right
    boundary stages, where each stage strips one more _PUNCT char (plus
    exposed whitespace) from that end; the raw gram is always a variant
    (e.g. "Home Free!?" also yields "Home Free!" and "Home Free").

    Possessives apply to every variant: one ending with 's or \\u2019s
    (unicode right single quote) also emits its base with the suffix
    removed and trailing stages re-applied ("F.C.'s" -> "F.C." -> "F.C"),
    so gold keys like "George Rankin, occupation" stay emittable from
    prompts that only contain "George Rankin's".

    Variants that are empty, longer than 60 chars, or contain no
    alphanumeric character (pure punctuation) are dropped; spans never
    begin or end with whitespace. Result is deduped preserving order.
    """
    words = prompt.split()
    spans: list[str] = []
    seen: set[str] = set()

    def emit(v: str) -> None:
        # The stage discipline lstrips/rstrips exposed whitespace, so
        # boundary whitespace is impossible; assert rather than filter.
        assert v == v.strip(), f"boundary whitespace survived: {v!r}"
        if not v or len(v) > _MAX_SPAN_CHARS:
            return
        if not any(ch.isalnum() for ch in v):
            return
        if v not in seen:
            seen.add(v)
            spans.append(v)

    for n in range(1, max_words + 1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            for lead in _leading_stages(gram):
                for v in _trailing_stages(lead):
                    emit(v)
                    for suf in _POSSESSIVE_SUFFIXES:
                        if v.endswith(suf):
                            base = v[: -len(suf)].rstrip()
                            for w in _trailing_stages(base):
                                emit(w)
                            break
    return spans


class _Node:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.terminal = False


class QueryTrie:
    """Token-level trie over tok.encode(f"{span}, {relation}") candidates.

    At any node whose path spells a complete candidate key, DB_RETRIEVE is
    additionally allowed, so a query can only terminate on a full key.
    """

    def __init__(self, tok, prompts_spans: list[str], relations: list[str]) -> None:
        self._retrieve: int = tok.DB_RETRIEVE
        self._root = _Node()
        for span in prompts_spans:
            for relation in relations:
                node = self._root
                for tid in tok.encode(f"{span}, {relation}"):
                    node = node.children.setdefault(tid, _Node())
                node.terminal = True

    def walker(self) -> "QueryWalker":
        return QueryWalker(self)


class QueryWalker:
    """Mutable cursor into a QueryTrie; one per decoding sequence."""

    def __init__(self, trie: QueryTrie) -> None:
        self._trie = trie
        self._node = trie._root

    def allowed(self) -> list[int]:
        ids: list[int] = list(self._node.children)
        if self._node.terminal:
            ids.append(self._trie._retrieve)
        return ids

    def advance(self, tok_id: int) -> None:
        if tok_id == self._trie._retrieve:
            self._node = self._trie._root
            return
        child = self._node.children.get(tok_id)
        self._node = child if child is not None else self._trie._root

    def reset(self) -> None:
        self._node = self._trie._root


def build_query_tries(prompts: list[str], relations: list[str], tok) -> list[QueryTrie]:
    """One trie per prompt, built from that prompt's spans only."""
    return [QueryTrie(tok, extract_spans(p), relations) for p in prompts]
