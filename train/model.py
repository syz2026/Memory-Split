"""Minimal decoder-only GPT: RMSNorm pre-norm, RoPE, SwiGLU, untied embeddings.

Two entry points (contract shared with evals/):
    forward(idx, targets=None, target_weights=None, loss_reduction="mean")
        -> (logits, loss | None)
    forward_step(idx, cache)    -> (logits, cache)         # kv-cache greedy decode
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    d_model: int
    vocab_size: int = 50304
    ctx: int = 2048
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


PRESETS: dict[str, GPTConfig] = {
    "toy": GPTConfig(n_layer=4, n_head=4, d_model=256),
    "d135m": GPTConfig(n_layer=10, n_head=12, d_model=720, ctx=1024),
    "d160m": GPTConfig(n_layer=12, n_head=12, d_model=768),
    "d410m": GPTConfig(n_layer=24, n_head=16, d_model=1024),
    "d1b": GPTConfig(n_layer=22, n_head=14, d_model=1792),
}


def _mlp_hidden(d_model: int) -> int:
    h = int(8 * d_model / 3)
    return ((h + 63) // 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rope_cos_sin(head_dim: int, max_pos: int, base: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, kv=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if kv is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
            new_kv = None
        else:
            k_all = torch.cat([kv[0], k], dim=2) if kv[0] is not None else k
            v_all = torch.cat([kv[1], v], dim=2) if kv[1] is not None else v
            # prefill (T>1, empty cache) needs a causal mask; decode steps see all
            y = F.scaled_dot_product_attention(
                q, k_all, v_all, is_causal=(T > 1 and kv[0] is None)
            )
            new_kv = (k_all, v_all)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y), new_kv


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = _mlp_hidden(cfg.d_model)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, kv=None):
        a, new_kv = self.attn(self.ln1(x), cos, sin, kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class KVCache:
    """Opaque cache: per-layer (k, v) plus current position offset."""

    def __init__(self, n_layer: int):
        self.kv: list[tuple | None] = [(None, None)] * n_layer
        self.pos: int = 0


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = _rope_cos_sin(cfg.head_dim, cfg.ctx, cfg.rope_base, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self):
        return self.wte.weight.device

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        target_weights: torch.Tensor | None = None,
        loss_reduction: str = "mean",
    ):
        B, T = idx.shape
        assert T <= self.cfg.ctx, f"sequence length {T} > ctx {self.cfg.ctx}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.wte(idx)
        for block in self.blocks:
            x, _ = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if loss_reduction not in {"mean", "sum"}:
                raise ValueError("loss_reduction must be 'mean' or 'sum'")
            if target_weights is None:
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    reduction=loss_reduction,
                )
            else:
                if target_weights.shape != targets.shape:
                    raise ValueError("target_weights shape must match targets")
                per_token = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(targets)
                valid_weights = torch.where(
                    targets.ne(-100),
                    target_weights,
                    torch.zeros_like(target_weights),
                )
                loss = (per_token * valid_weights).sum()
                if loss_reduction == "mean":
                    loss = loss / targets.numel()
        return logits, loss

    @torch.no_grad()
    def forward_step(self, idx: torch.Tensor, cache: KVCache | None):
        """Prefill with cache=None and idx [B, T]; then step with idx [B, 1]."""
        B, T = idx.shape
        if cache is None:
            cache = KVCache(self.cfg.n_layer)
        pos = cache.pos
        assert pos + T <= self.cfg.ctx, "kv cache exceeded model context"
        cos, sin = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]
        x = self.wte(idx)
        for i, block in enumerate(self.blocks):
            x, new_kv = block(x, cos, sin, kv=cache.kv[i])
            cache.kv[i] = new_kv
        x = self.ln_f(x)
        logits = self.lm_head(x)
        cache.pos = pos + T
        return logits, cache
