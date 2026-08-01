"""Batched greedy decoding.

Model contract (train/model.py, tested here against scripted stubs):
    logits, cache = model.forward_step(idx[B, T], cache)
First call passes the full (left-padded) prompt with cache=None (prefill);
every later call passes exactly one token per row. The cache is opaque.

There is no external store. The lookup-interception state machine that used
to live here belonged to the organizer experiment, whose endpoint compared a
model plus an oracle against a closed-book model — a system comparison, not a
capacity one. Nothing in the current design retrieves at train or eval time.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class _Seq:
    __slots__ = ("generated", "done")

    def __init__(self) -> None:
        self.generated: list[int] = []
        self.done = False


def generate_batch(
    model,
    tok,
    prompts: list[str],
    max_new: int,
    device,
    stop_at_eot: bool = True,
) -> list[str]:
    """Greedy-decode continuations for all prompts in one batch.

    Returns texts excluding the prompt and the stopping EOT. Over-long
    prompts are truncated from the LEFT so the question end survives.
    """
    if not prompts:
        return []

    prompt_ids = [tok.encode(p) for p in prompts]
    ctx = getattr(getattr(model, "cfg", None), "ctx", None)
    if ctx is not None:
        keep = max(8, ctx - min(max_new, ctx // 2))
        if any(len(p) > keep for p in prompt_ids):
            logger.warning("left-truncating %d prompt(s) to %d tokens",
                           sum(len(p) > keep for p in prompt_ids), keep)
            prompt_ids = [p[-keep:] for p in prompt_ids]
    pad_to = max(1, max(len(p) for p in prompt_ids))
    steps_budget = max_new if ctx is None else min(max_new, ctx - pad_to)
    # Left-pad with EOT so logits[:, -1, :] is the next-token distribution for
    # every row after a single prefill. With RoPE a constant left shift is
    # harmless for greedy decoding at these scales; the pads are ordinary EOTs
    # the model has seen as separators.
    padded = [[tok.EOT] * (pad_to - len(p)) + p for p in prompt_ids]
    x = torch.tensor(padded, dtype=torch.long, device=device)

    seqs = [_Seq() for _ in prompts]
    with torch.no_grad():
        logits, cache = model.forward_step(x, None)
        for _ in range(steps_budget):
            choices = logits[:, -1, :].argmax(dim=-1).tolist()
            next_ids: list[int] = []
            for b, s in enumerate(seqs):
                if s.done:
                    next_ids.append(tok.EOT)  # cache filler for finished rows
                    continue
                nid = int(choices[b])
                if stop_at_eot and nid == tok.EOT:
                    s.done = True
                    next_ids.append(tok.EOT)
                    continue
                s.generated.append(nid)
                if len(s.generated) >= max_new:
                    s.done = True
                next_ids.append(nid)
            if all(s.done for s in seqs):
                break
            x = torch.tensor(next_ids, dtype=torch.long, device=device)
            logits, cache = model.forward_step(x.unsqueeze(1), cache)

    return [tok.decode(s.generated) for s in seqs]
