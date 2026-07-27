"""Packed-sequence dataloader over token, loss-mask, and target-weight shards.

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

from experiment.artifacts import require_regular_file


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
        weights_path: str | Path | None = None,
    ):
        token_file = require_regular_file(bin_path, name="corpus")
        self.tokens = np.memmap(token_file, dtype=np.uint16, mode="r")
        if mask_path is not None:
            mask_file = require_regular_file(mask_path, name="mask sidecar")
            self.mask = np.memmap(mask_file, dtype=np.uint8, mode="r")
            if len(self.mask) != len(self.tokens):
                raise ValueError("mask/token length mismatch")
        else:
            self.mask = None
        if weights_path is not None:
            weights_file = require_regular_file(
                weights_path,
                name="weights sidecar",
            )
            self.target_weights = np.memmap(
                weights_file,
                dtype=np.uint8,
                mode="r",
            )
            if len(self.target_weights) != len(self.tokens):
                raise ValueError("weights/token length mismatch")
        else:
            self.target_weights = None
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.cursor = start_cursor
        self.n_tokens = len(self.tokens)
        self.epoch = 0
        self.raw_positions = 0
        self.seed = seed
        span = self.batch_size * (self.ctx + 1)
        if self.n_tokens <= span:
            raise ValueError("corpus smaller than one batch")

    def _window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray | None]:
        toks = np.asarray(self.tokens[start : start + length])
        msk = np.asarray(self.mask[start : start + length]) if self.mask is not None else None
        return toks, msk

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.batch_size * (self.ctx + 1)
        cursor = self.cursor
        epoch = self.epoch
        if cursor + span >= self.n_tokens:
            cursor = 0
            epoch += 1
        toks, msk = self._window(cursor, span)
        stride = self.batch_size * self.ctx
        toks = toks.astype(np.int64).reshape(self.batch_size, self.ctx + 1)
        x = torch.from_numpy(toks[:, :-1].copy())
        y = torch.from_numpy(toks[:, 1:].copy())
        if msk is not None:
            m = msk.reshape(self.batch_size, self.ctx + 1)[:, 1:]
            y[torch.from_numpy((m == 0).copy())] = -100
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
        self.cursor = cursor + stride  # overlap of 1 keeps every target trained
        self.epoch = epoch
        self.raw_positions += stride
        return x, y

    def _aligned_next_token_weights_for_last_batch(self) -> torch.Tensor:
        assert self.target_weights is not None
        span = self.batch_size * (self.ctx + 1)
        start = self.cursor - self.batch_size * self.ctx
        raw = np.asarray(self.target_weights[start : start + span])
        raw = raw.reshape(self.batch_size, self.ctx + 1)[:, 1:]
        weights = torch.from_numpy(raw.astype(np.float32, copy=True))
        if self.device == "cuda":
            weights = weights.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            weights = weights.to(self.device)
        return weights

    def next_weighted_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets = self.next_batch()
        if self.target_weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
        else:
            weights = self._aligned_next_token_weights_for_last_batch()
        return x, targets, weights

    def masked_value_batch(self, max_batches: int = 8) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fixed probe batches over MASKED positions only (loss_masked_values metric).

        Returns (x, y) where y is -100 everywhere EXCEPT masked-value targets —
        the complement of the training labels — sampled from the shard head.
        None when the corpus has no masked positions (dense arm).
        """
        if self.mask is None:
            return None
        span = self.batch_size * (self.ctx + 1)
        toks, msk = self._window(0, span * max_batches)
        if (msk == 0).sum() == 0:
            return None
        usable = (len(toks) // (self.ctx + 1)) * (self.ctx + 1)
        toks = toks[:usable].astype(np.int64).reshape(-1, self.ctx + 1)
        msk = msk[:usable].reshape(-1, self.ctx + 1)
        x = torch.from_numpy(toks[:, :-1].copy())
        y = torch.from_numpy(toks[:, 1:].copy())
        keep = torch.from_numpy((msk[:, 1:] == 0).copy())
        y[~keep] = -100
        rows = keep.any(dim=1)
        if not rows.any():
            return None
        return x[rows], y[rows]

    def state_dict(self) -> dict:
        return {
            "schema_version": 2,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "raw_positions": self.raw_positions,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "cursor",
            "epoch",
            "raw_positions",
        }:
            raise ValueError("data state fields are not exact")
        if state["schema_version"] != 2:
            raise ValueError("data state schema version is incompatible")
        cursor = state["cursor"]
        epoch = state["epoch"]
        raw_positions = state["raw_positions"]
        for name, value in (
            ("cursor", cursor),
            ("epoch", epoch),
            ("raw positions", raw_positions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"data {name} must be a nonnegative integer")
        if cursor > self.n_tokens:
            raise ValueError("data cursor is outside the corpus")
        stride = self.batch_size * self.ctx
        if cursor % stride:
            raise ValueError("data cursor is not batch aligned")
        if raw_positions % stride:
            raise ValueError("data raw positions are not batch aligned")
        self.cursor = cursor
        self.epoch = epoch
        self.raw_positions = raw_positions
