"""Packed-sequence dataloader over uint16 token shards + uint8 loss-mask shards.

The corpus builder writes, per arm, a flat stream of token ids (uint16) and
a parallel loss mask (uint8, 1 = loss ON). Batches are contiguous windows;
the target at position t is token t+1, and its label is -100 wherever the
NEXT token's mask is 0 (fact values in the split arm).

The cursor is a single integer (token offset), saved into checkpoints so a
resumed run continues on the exact next batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PackedShards:
    def __init__(
        self,
        bin_path: str | Path,
        mask_path: str | Path | None,
        ctx: int,
        batch_size: int,
        device: str = "cpu",
        start_cursor: int = 0,
        seed: int = 0,
    ):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if mask_path is not None and Path(mask_path).exists():
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
            assert len(self.mask) == len(self.tokens), "mask/token length mismatch"
        else:
            self.mask = None
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.cursor = start_cursor
        self.n_tokens = len(self.tokens)
        self.epoch = 0
        span = self.batch_size * self.ctx + 1
        assert self.n_tokens > span, "corpus smaller than one batch"

    def _window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray | None]:
        end = start + length
        if end <= self.n_tokens:
            toks = np.asarray(self.tokens[start:end])
            msk = (
                np.asarray(self.mask[start:end]) if self.mask is not None else None
            )
            return toks, msk
        wrapped = end - self.n_tokens
        toks = np.concatenate(
            (np.asarray(self.tokens[start:]), np.asarray(self.tokens[:wrapped]))
        )
        msk = (
            np.concatenate(
                (np.asarray(self.mask[start:]), np.asarray(self.mask[:wrapped]))
            )
            if self.mask is not None
            else None
        )
        return toks, msk

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self.batch_size * self.ctx
        span = targets + 1
        start = self.cursor
        toks, msk = self._window(self.cursor, span)
        self.cursor = (self.cursor + targets) % self.n_tokens
        self.epoch += (start + targets) // self.n_tokens
        toks = toks.astype(np.int64)
        x = torch.from_numpy(toks[:-1].reshape(self.batch_size, self.ctx).copy())
        y = torch.from_numpy(toks[1:].reshape(self.batch_size, self.ctx).copy())
        if msk is not None:
            m = msk[1:].reshape(self.batch_size, self.ctx)
            y[torch.from_numpy((m == 0).copy())] = -100
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def masked_value_batch(self, max_batches: int = 8) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fixed probe batches over MASKED positions only (loss_masked_values metric).

        Returns (x, y) where y is -100 everywhere EXCEPT masked-value targets —
        the complement of the training labels — sampled from the shard head.
        None when the corpus has no masked positions (dense arm).
        """
        if self.mask is None:
            return None
        max_targets = self.batch_size * self.ctx * max_batches
        sample_length = min(self.n_tokens, max_targets + 1)
        toks = np.asarray(self.tokens[:sample_length])
        msk = np.asarray(self.mask[:sample_length])
        if (msk == 0).sum() == 0:
            return None
        usable_targets = ((len(toks) - 1) // self.ctx) * self.ctx
        toks = toks[: usable_targets + 1].astype(np.int64)
        msk = msk[: usable_targets + 1]
        x = torch.from_numpy(toks[:-1].reshape(-1, self.ctx).copy())
        y = torch.from_numpy(toks[1:].reshape(-1, self.ctx).copy())
        keep = torch.from_numpy((msk[1:].reshape(-1, self.ctx) == 0).copy())
        y[~keep] = -100
        rows = keep.any(dim=1)
        if not rows.any():
            return None
        return x[rows], y[rows]

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = state["cursor"]
        self.epoch = state.get("epoch", 0)
