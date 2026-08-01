"""Packed-sequence dataloader over uint16 token shards + uint8 loss-mask shards.

The corpus builder writes, per arm, a flat stream of token ids (uint16) and
a parallel loss mask (uint8, 1 = loss ON). Batches are contiguous windows;
the target at position t is token t+1, and its label is -100 wherever the
NEXT token's mask is 0 (fact values in the split arm).

A stream may be given as one path or as an ordered list of paths. A list is
read as one logical stream, concatenated in the order given and never
interleaved — the segmented corpora (base then reasoning extension) rely on
that order, since the extension occupies the tail of the stream and a run that
stops short of it never sees a reasoning token.

The cursor is a single integer (token offset), saved into checkpoints so a
resumed run continues on the exact next batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

StreamPaths = str | Path | Sequence[str | Path]


def _as_paths(spec: StreamPaths) -> list[Path]:
    if isinstance(spec, (str, Path)):
        return [Path(spec)]
    return [Path(p) for p in spec]


class ConcatMemmap:
    """Ordered, read-only concatenation of same-dtype memmaps as one stream.

    Segments stay separately mapped: the corpus is tens of gigabytes and
    materialising the joined stream would defeat the point of memmapping it.
    """

    def __init__(self, paths: Sequence[Path], dtype):
        self.parts = [np.memmap(p, dtype=dtype, mode="r") for p in paths]
        self.lengths = [len(part) for part in self.parts]
        self.offsets = np.cumsum([0] + self.lengths)
        self.dtype = np.dtype(dtype)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        """Gather [start, start+length) across segment boundaries."""
        assert start >= 0 and start + length <= len(self), "read past end of stream"
        i = int(np.searchsorted(self.offsets, start, side="right")) - 1
        if start + length <= self.offsets[i + 1]:
            lo = start - int(self.offsets[i])
            return np.asarray(self.parts[i][lo : lo + length])
        out = np.empty(length, dtype=self.dtype)
        pos = 0
        while pos < length:
            lo = start + pos - int(self.offsets[i])
            take = min(length - pos, self.lengths[i] - lo)
            out[pos : pos + take] = self.parts[i][lo : lo + take]
            pos += take
            i += 1
        return out


class PackedShards:
    def __init__(
        self,
        bin_path: StreamPaths,
        mask_path: StreamPaths | None,
        ctx: int,
        batch_size: int,
        device: str = "cpu",
        start_cursor: int = 0,
        seed: int = 0,
        probe_mask_path: StreamPaths | None = None,
    ):
        self.tokens = ConcatMemmap(_as_paths(bin_path), np.uint16)
        # Fail closed. A configured-but-absent sidecar used to fall through to
        # self.mask = None, which silently turns a masked arm into a dense one
        # and produces a whole cohort of runs that look valid and are not.
        mask_paths = _as_paths(mask_path) if mask_path is not None else []
        if mask_paths:
            missing = [str(p) for p in mask_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "train_mask configured but missing: "
                    + ", ".join(missing)
                    + " -- refusing to train an unmasked arm under a masked config"
                )
            self.mask = ConcatMemmap(mask_paths, np.uint8)
            assert len(self.mask) == len(self.tokens), "mask/token length mismatch"
        else:
            self.mask = None
        probe_paths = _as_paths(probe_mask_path) if probe_mask_path is not None else []
        if probe_paths:
            missing = [str(p) for p in probe_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "probe_mask configured but missing: " + ", ".join(missing)
                )
            self.probe_mask = ConcatMemmap(probe_paths, np.uint8)
            assert len(self.probe_mask) == len(self.tokens), "probe/token length mismatch"
        else:
            self.probe_mask = None
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.cursor = start_cursor
        self.n_tokens = len(self.tokens)
        self.epoch = 0
        span = self.batch_size * (self.ctx + 1)
        assert self.n_tokens > span, "corpus smaller than one batch"

    def _window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray | None]:
        length = min(length, self.n_tokens - start)
        toks = self.tokens.read(start, length)
        msk = self.mask.read(start, length) if self.mask is not None else None
        return toks, msk

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x, y, w).

        `w` is the per-target loss weight: 1.0 where supervised, 0.0 where the
        sidecar masks the target. Masked positions are additionally set to
        -100 in `y`, so they contribute nothing through either mechanism.

        `w` always has the full target shape, so `w.numel()` is identical
        across arms reading the same stream. That is what keeps the loss
        denominator fixed and stops masking from reweighting the survivors.
        """
        span = self.batch_size * (self.ctx + 1)
        if self.cursor + span >= self.n_tokens:
            self.cursor = 0
            self.epoch += 1
        toks, msk = self._window(self.cursor, span)
        self.cursor += self.batch_size * self.ctx  # overlap of 1 keeps every target trained
        toks = toks.astype(np.int64).reshape(self.batch_size, self.ctx + 1)
        x = torch.from_numpy(toks[:, :-1].copy())
        y = torch.from_numpy(toks[:, 1:].copy())
        w = torch.ones_like(y, dtype=torch.float32)
        if msk is not None:
            m = msk.reshape(self.batch_size, self.ctx + 1)[:, 1:]
            zero = torch.from_numpy((m == 0).copy())
            y[zero] = -100
            w[zero] = 0.0
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            w = w.pin_memory().to(self.device, non_blocking=True)
        elif self.device != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
        return x, y, w

    def _probe_stream(self) -> ConcatMemmap | None:
        """Stream that defines the offloaded positions. Falls back to this
        arm's own mask so the split arm works without extra configuration."""
        return self.probe_mask if self.probe_mask is not None else self.mask

    def _find_masked_window(self, width: int) -> int | None:
        """First offset whose window contains a masked target.

        The corpus is lane-ordered and its head lane is entirely unmasked, so a
        probe anchored at offset 0 finds nothing and silently disables itself.
        """
        probe = self._probe_stream()
        if probe is None:
            return None
        off = 0
        while off + width <= len(probe):
            if (probe.read(off, width) == 0).any():
                return off
            off += width
        return None

    def masked_value_batch(self, max_batches: int = 8) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fixed probe batches over MASKED positions only (loss_masked_values).

        Returns (x, y) where y is -100 everywhere EXCEPT masked-value targets —
        the complement of the training labels. Both arms should be given the
        same probe stream so their scores are comparable: the split arm should
        sit near ln(vocab), and the dense arm's score is how much of the
        offloaded content it absorbed.
        """
        probe = self._probe_stream()
        if probe is None:
            return None
        width = self.batch_size * (self.ctx + 1) * max_batches
        start = self._find_masked_window(width)
        if start is None:
            return None
        length = min(width, self.n_tokens - start)
        toks = self.tokens.read(start, length)
        msk = probe.read(start, length)
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
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = state["cursor"]
        self.epoch = state.get("epoch", 0)
