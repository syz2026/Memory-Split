"""Packed-sequence dataloader over uint16 tokens and uint8 target weights.

The corpus builder writes, per arm, a flat stream of token ids (uint16) and
a parallel binary sidecar. Legacy callers receive ``-100`` labels at zero
weights; receipt-v2 callers consume the same bytes as direct target weights.
Batches read ``batch_size * ctx + 1`` cyclic tokens and overlap adjacent rows
at one boundary token, so every causal target occurs once and in order.

The cursor is a monotonic target offset saved into checkpoints, so a resumed
run continues on the exact next batch.
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
        del seed
        token_path = Path(bin_path)
        if not token_path.is_file() or token_path.is_symlink():
            raise ValueError("token stream is missing, symlinked, or unsafe")
        self.tokens = np.memmap(token_path, dtype=np.uint16, mode="r")
        if mask_path is not None:
            target_mask_path = Path(mask_path)
            if not target_mask_path.is_file() or target_mask_path.is_symlink():
                raise ValueError("target-weight stream is missing, symlinked, or unsafe")
            self.mask = np.memmap(target_mask_path, dtype=np.uint8, mode="r")
            if len(self.mask) != len(self.tokens):
                raise ValueError("mask/token length mismatch")
        else:
            self.mask = None
        if (
            isinstance(ctx, bool)
            or not isinstance(ctx, int)
            or ctx <= 0
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("ctx and batch_size must be positive integers")
        if (
            isinstance(start_cursor, bool)
            or not isinstance(start_cursor, int)
            or start_cursor < 0
        ):
            raise ValueError("start_cursor must be a non-negative integer")
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.cursor = start_cursor
        self.n_tokens = len(self.tokens)
        self.epoch = start_cursor // self.n_tokens if self.n_tokens else 0
        target_count = self.batch_size * self.ctx
        if self.n_tokens < target_count:
            raise ValueError("corpus smaller than one batch")

    def _window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray | None]:
        def read(stream: np.memmap) -> np.ndarray:
            result = np.empty(length, dtype=stream.dtype)
            position = start % self.n_tokens
            written = 0
            while written < length:
                take = min(length - written, self.n_tokens - position)
                result[written : written + take] = stream[position : position + take]
                written += take
                position = (position + take) % self.n_tokens
            return result

        return read(self.tokens), read(self.mask) if self.mask is not None else None

    def _next_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        target_count = self.batch_size * self.ctx
        toks, msk = self._window(self.cursor, target_count + 1)
        self.cursor += target_count
        self.epoch = self.cursor // self.n_tokens
        shape = (self.batch_size, self.ctx)
        x = toks[:-1].astype(np.int64).reshape(shape)
        y = toks[1:].astype(np.int64).reshape(shape)
        weights = msk[1:].reshape(shape) if msk is not None else None
        return x, y, weights

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_values, y_values, weights = self._next_arrays()
        x = torch.from_numpy(x_values.copy())
        y = torch.from_numpy(y_values.copy())
        if weights is not None:
            y[torch.from_numpy((weights == 0).copy())] = -100
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def next_weighted_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return causal targets plus direct per-target objective weights."""

        x_values, y_values, weights = self._next_arrays()
        x = torch.from_numpy(x_values.copy())
        y = torch.from_numpy(y_values.copy())
        if weights is not None and np.any((weights != 0) & (weights != 1)):
            raise ValueError("target-weight stream contains non-binary values")
        weight_values = (
            np.ones_like(y_values, dtype=np.float32)
            if weights is None
            else weights.astype(np.float32)
        )
        target_weights = torch.from_numpy(weight_values.copy())
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            target_weights = target_weights.pin_memory().to(
                self.device,
                non_blocking=True,
            )
        elif self.device != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
            target_weights = target_weights.to(self.device)
        return x, y, target_weights

    def masked_value_batch(self, max_batches: int = 8) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fixed probe batches over MASKED positions only (loss_masked_values metric).

        Returns (x, y) where y is -100 everywhere EXCEPT masked-value targets —
        the complement of the training labels — sampled from the shard head.
        None when the corpus has no masked positions (dense arm).
        """
        if self.mask is None:
            return None
        target_count = self.batch_size * self.ctx * max_batches
        toks, msk = self._window(0, target_count + 1)
        assert msk is not None
        if (msk == 0).sum() == 0:
            return None
        x = torch.from_numpy(toks[:-1].astype(np.int64).reshape(-1, self.ctx))
        y = torch.from_numpy(toks[1:].astype(np.int64).reshape(-1, self.ctx))
        keep = torch.from_numpy((msk[1:].reshape(-1, self.ctx) == 0).copy())
        y[~keep] = -100
        rows = keep.any(dim=1)
        if not rows.any():
            return None
        return x[rows], y[rows]

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        cursor = state["cursor"]
        epoch = state.get("epoch", 0)
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError("checkpoint data cursor is invalid")
        if epoch and cursor < self.n_tokens:
            cursor += epoch * self.n_tokens
        if epoch != cursor // self.n_tokens:
            raise ValueError("checkpoint data epoch is inconsistent with its cursor")
        self.cursor = cursor
        self.epoch = epoch
