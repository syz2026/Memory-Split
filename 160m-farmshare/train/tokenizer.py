"""GPT-2 BPE tokenizer extended with memory-split and graph-control tokens.

Special token ids are frozen (see plan, Global Constraints):
    <|db_start|>=50257  <|db_retrieve|>=50258  <|db_end|>=50259  <|eot|>=50260
Graph-control token ids occupy the reserved range 50261-50291.
Vocab is padded to 50304 (multiple of 64) at the model level.

`encode_segments` encodes each segment independently so masked spans map
exactly onto token boundaries (no BPE merges across a mask edge).
"""

from __future__ import annotations

import os
from pathlib import Path

# Pin the BPE file cache inside the repo (committed) so training jobs and
# sandboxed processes never need network for tokenizer setup.
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".tiktoken_cache"
if "TIKTOKEN_CACHE_DIR" not in os.environ:
    _CACHE_DIR.mkdir(exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_CACHE_DIR)

import tiktoken

from corpusgen.records import Segment

DB_SPECIAL_TOKENS = {
    "<|db_start|>": 50257,
    "<|db_retrieve|>": 50258,
    "<|db_end|>": 50259,
    "<|eot|>": 50260,
}
GRAPH_SPECIAL_TOKENS = {
    "<|graph_start|>": 50261,
    "<|graph_read|>": 50262,
    "<|graph_return|>": 50263,
    "<|graph_end|>": 50264,
    "<|graph_halt|>": 50265,
    "<|graph_noop|>": 50266,
    "<|slot_0|>": 50267,
    "<|slot_1|>": 50268,
    "<|slot_2|>": 50269,
    "<|slot_3|>": 50270,
    "<|dir_out|>": 50271,
    "<|dir_in|>": 50272,
    "<|graph_step|>": 50273,
    "<|answer_state|>": 50274,
    "<|graph_miss|>": 50275,
    **{f"<|rel_{i}|>": 50276 + i for i in range(16)},
}
SPECIAL_TOKENS = {**DB_SPECIAL_TOKENS, **GRAPH_SPECIAL_TOKENS}

VOCAB_SIZE = 50304


class Tok:
    VOCAB_SIZE = VOCAB_SIZE
    RELATION_DIGITS = tuple(
        GRAPH_SPECIAL_TOKENS[f"<|rel_{index}|>"] for index in range(16)
    )

    def __init__(self) -> None:
        base = tiktoken.get_encoding("gpt2")
        self._enc = tiktoken.Encoding(
            name="gpt2_memsplit",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
        )
        self.DB_START = DB_SPECIAL_TOKENS["<|db_start|>"]
        self.DB_RETRIEVE = DB_SPECIAL_TOKENS["<|db_retrieve|>"]
        self.DB_END = DB_SPECIAL_TOKENS["<|db_end|>"]
        self.EOT = DB_SPECIAL_TOKENS["<|eot|>"]
        self.graph_special_tokens = dict(GRAPH_SPECIAL_TOKENS)
        self.GRAPH_START = GRAPH_SPECIAL_TOKENS["<|graph_start|>"]
        self.GRAPH_READ = GRAPH_SPECIAL_TOKENS["<|graph_read|>"]
        self.GRAPH_RETURN = GRAPH_SPECIAL_TOKENS["<|graph_return|>"]
        self.GRAPH_END = GRAPH_SPECIAL_TOKENS["<|graph_end|>"]
        self.GRAPH_HALT = GRAPH_SPECIAL_TOKENS["<|graph_halt|>"]
        self.GRAPH_NOOP = GRAPH_SPECIAL_TOKENS["<|graph_noop|>"]
        self.GRAPH_STEP = GRAPH_SPECIAL_TOKENS["<|graph_step|>"]
        self.ANSWER_STATE = GRAPH_SPECIAL_TOKENS["<|answer_state|>"]
        self.GRAPH_MISS = GRAPH_SPECIAL_TOKENS["<|graph_miss|>"]
        self.DIR_OUT = GRAPH_SPECIAL_TOKENS["<|dir_out|>"]
        self.DIR_IN = GRAPH_SPECIAL_TOKENS["<|dir_in|>"]
        self.SLOTS = tuple(GRAPH_SPECIAL_TOKENS[f"<|slot_{i}|>"] for i in range(4))

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def encode_segments(
        self, segments: list[Segment], add_eot: bool = True
    ) -> tuple[list[int], list[int]]:
        """Returns (ids, loss_mask); mask[i]=1 means loss ON at token i."""
        ids: list[int] = []
        mask: list[int] = []
        for text, masked in segments:
            seg_ids = self._enc.encode(text, allowed_special="all")
            ids.extend(seg_ids)
            mask.extend([0 if masked else 1] * len(seg_ids))
        if add_eot:
            ids.append(self.EOT)
            mask.append(1)
        return ids, mask

    def encode_tagged_segments(self, segments):
        ids: list[int] = []
        roles: list[str] = []
        fact_ids: list[str | None] = []
        for segment in segments:
            segment_ids = self._enc.encode(segment.text, allowed_special="all")
            ids.extend(segment_ids)
            roles.extend([segment.role] * len(segment_ids))
            fact_ids.extend([segment.fact_id] * len(segment_ids))
        return ids, roles, fact_ids


_TOK: Tok | None = None


def get_tok() -> Tok:
    global _TOK
    if _TOK is None:
        _TOK = Tok()
    return _TOK
