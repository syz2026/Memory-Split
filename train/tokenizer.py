"""GPT-2 BPE tokenizer extended with memory-split and graph-control tokens.

Special token ids are frozen (see plan, Global Constraints):
    <|db_start|>=50257  <|db_retrieve|>=50258  <|db_end|>=50259  <|eot|>=50260
Graph-control token ids occupy the reserved range 50261-50295.
Vocab is padded to 50304 (multiple of 64) at the model level.

`encode_segments` encodes each segment independently so masked spans map
exactly onto token boundaries (no BPE merges across a mask edge).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Pin and authenticate the committed GPT-2 assets before tiktoken can attempt
# any cache fill. Bundles preserve these paths and hash them in manifest.json.
_CACHE_DIR = Path(__file__).resolve().parent.parent / "vendor" / "tiktoken"
_CACHE_ASSET_SHA256 = {
    "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6": (
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
    ),
    "6d1cbeee0f20b3d9449abfede4726ed8212e3aee": (
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
    ),
}


def _verify_tiktoken_assets() -> None:
    if not _CACHE_DIR.is_dir() or _CACHE_DIR.is_symlink():
        raise RuntimeError(f"vendored tiktoken cache is missing: {_CACHE_DIR}")
    for name, expected_sha256 in _CACHE_ASSET_SHA256.items():
        path = _CACHE_DIR / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"vendored tiktoken asset is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"vendored tiktoken asset hash mismatch: {path}")


_verify_tiktoken_assets()
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
    "<|relation_start|>": 50292,
    "<|relation_end|>": 50293,
    "<|page_start|>": 50294,
    "<|page_end|>": 50295,
}
SPECIAL_TOKENS = {**DB_SPECIAL_TOKENS, **GRAPH_SPECIAL_TOKENS}

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
        self.RELATION_START = GRAPH_SPECIAL_TOKENS["<|relation_start|>"]
        self.RELATION_END = GRAPH_SPECIAL_TOKENS["<|relation_end|>"]
        self.PAGE_START = GRAPH_SPECIAL_TOKENS["<|page_start|>"]
        self.PAGE_END = GRAPH_SPECIAL_TOKENS["<|page_end|>"]
        self.SLOTS = tuple(GRAPH_SPECIAL_TOKENS[f"<|slot_{i}|>"] for i in range(4))
        self.RELATIONS = {
            f"r{i}": GRAPH_SPECIAL_TOKENS[f"<|rel_{i}|>"] for i in range(16)
        }

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
