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
    # Prompts of unequal length are decoded in separate, equal-length groups.
    #
    # This function used to left-pad every row to the batch maximum with EOT,
    # on the reasoning that "with RoPE a constant left shift is harmless" and
    # "the pads are ordinary EOTs the model has seen as separators". Both are
    # false. There is no attention mask here, so the pads are attended to as
    # real context, and EOT is the document separator -- a prompt behind 32 EOTs
    # reads as a fresh document and the model hallucinates its premises.
    #
    # Measured on stepladder_d40m_std at step 31,280, 64 held-out iGSM items:
    #
    #     batched, padded to the batch maximum ...  3/64 =  4.7%
    #     one prompt at a time .................... 60/64 = 93.8%
    #
    # The corruption is monotone in pad length -- exact at 0-16 pads, broken by
    # 32 -- so it fell hardest on the shortest prompts. That is the whole of the
    # "endpoint floors at every scale and difficulty" result this project spent
    # four experiment generations on, and it is why accuracy rose with operation
    # count: more operations means a longer prompt means less padding.
    #
    # Grouping by exact length gives zero padding and therefore no need for a
    # mask. It costs throughput when lengths are diverse; correctness first.
    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
    groups: list[list[int]] = []
    for i in order:
        if groups and len(prompt_ids[groups[-1][0]]) == len(prompt_ids[i]):
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1:
        out: list[str] = [""] * len(prompts)
        for g in groups:
            texts = generate_batch(model, tok, [prompts[i] for i in g],
                                   max_new, device, stop_at_eot)
            for i, t in zip(g, texts):
                out[i] = t
        return out

    pad_to = max(1, max(len(p) for p in prompt_ids))
    steps_budget = max_new if ctx is None else min(max_new, ctx - pad_to)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # The vocabulary is padded to a multiple of 64; the tail ids map to no
    # token. Never select one: an undertrained model puts real mass there and
    # the tokenizer cannot decode it.
    n_real = getattr(tok, "N_REAL_TOKENS", None)

    seqs = [_Seq() for _ in prompts]
    with torch.no_grad():
        logits, cache = model.forward_step(x, None)
        for _ in range(steps_budget):
            step_logits = logits[:, -1, :]
            if n_real is not None and step_logits.size(-1) > n_real:
                step_logits = step_logits[:, :n_real]
            choices = step_logits.argmax(dim=-1).tolist()
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
