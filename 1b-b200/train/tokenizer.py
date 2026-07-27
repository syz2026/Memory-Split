"""GPT-2 BPE tokenizer extended with the four memory-split special tokens.

Special token ids are frozen (see plan, Global Constraints):
    <|db_start|>=50257  <|db_retrieve|>=50258  <|db_end|>=50259  <|eot|>=50260
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

SPECIAL_TOKENS = {
    "<|db_start|>": 50257,
    "<|db_retrieve|>": 50258,
    "<|db_end|>": 50259,
    "<|eot|>": 50260,
}

VOCAB_SIZE = 50304


class Tok:
    VOCAB_SIZE = VOCAB_SIZE

    def __init__(self) -> None:
        base = tiktoken.get_encoding("gpt2")
        self._enc = tiktoken.Encoding(
            name="gpt2_memsplit",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
        )
        self.DB_START = SPECIAL_TOKENS["<|db_start|>"]
        self.DB_RETRIEVE = SPECIAL_TOKENS["<|db_retrieve|>"]
        self.DB_END = SPECIAL_TOKENS["<|db_end|>"]
        self.EOT = SPECIAL_TOKENS["<|eot|>"]

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


_TOK: Tok | None = None


def get_tok() -> Tok:
    global _TOK
    if _TOK is None:
        _TOK = Tok()
    return _TOK
