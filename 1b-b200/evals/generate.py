"""Batched greedy decoding with organizer lookup interception.

Model contract (train/model.py, tested here against scripted stubs):
    logits, cache = model.forward_step(idx[B, T], cache)
First call passes the full (left-padded) prompt with cache=None (prefill);
every later call passes exactly one token per row. The cache is opaque.

Interception protocol (store ON, i.e. organizer is not None):
    FREE      --<|db_start|>-->  IN_QUERY (start recording query ids)
    IN_QUERY  --<|db_retrieve|>--> decode query, organizer.lookup(query);
              hit  -> force-append tok.encode(" " + value) + [<|db_end|>]
              miss -> log, back to FREE (model decodes freely)
    IN_QUERY  --32 query tokens without <|db_retrieve|>--> malformed: log,
              back to FREE (emitted tokens stay in the output).
Forced tokens bypass the model's own argmax but are still fed through
forward_step so the KV cache stays consistent; they count toward max_new.

Stats semantics: n_lookups == n_hits + n_misses (queries that reached
<|db_retrieve|>); cap-hit queries count only as n_malformed.
"""

from __future__ import annotations

import logging
from collections import deque

import torch

logger = logging.getLogger(__name__)

QUERY_TOKEN_CAP = 32

_FREE = 0
_IN_QUERY = 1


class _Seq:
    __slots__ = ("generated", "state", "query_ids", "force", "done")

    def __init__(self) -> None:
        self.generated: list[int] = []
        self.state = _FREE
        self.query_ids: list[int] = []
        self.force: deque[int] = deque()
        self.done = False


def _advance_state(s: _Seq, nid: int, tok, organizer, stats: dict) -> None:
    """State machine for model-chosen tokens when the store is ON."""
    if s.state == _FREE:
        if nid == tok.DB_START:
            s.state = _IN_QUERY
            s.query_ids = []
        return
    # _IN_QUERY
    if nid == tok.DB_RETRIEVE:
        query = tok.decode(s.query_ids)
        stats["n_lookups"] += 1
        value = organizer.lookup(query)
        if value is None:
            stats["n_misses"] += 1
            logger.debug("lookup miss: %r", query)
        else:
            stats["n_hits"] += 1
            s.force.extend(tok.encode(" " + value) + [tok.DB_END])
        s.state = _FREE
    else:
        s.query_ids.append(nid)
        if len(s.query_ids) >= QUERY_TOKEN_CAP:
            stats["n_malformed"] += 1
            logger.debug(
                "malformed lookup: no <|db_retrieve|> within %d tokens: %r",
                QUERY_TOKEN_CAP,
                tok.decode(s.query_ids),
            )
            s.state = _FREE


def generate_batch_with_stats(
    model,
    tok,
    prompts: list[str],
    max_new: int,
    organizer,
    device,
    stop_at_eot: bool = True,
) -> tuple[list[str], dict]:
    """Greedy-decode continuations for all prompts in one batch.

    Returns (texts, stats). Texts exclude the prompt and the stopping EOT
    but include any forced lookup tokens. max_new counts every appended
    token, forced ones included. organizer=None means store OFF: DB_*
    tokens get no special handling.
    """
    stats = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    if not prompts:
        return [], stats

    prompt_ids = [tok.encode(p) for p in prompts]
    # Clamp to the model's context: leave room for generation, and truncate
    # over-long prompts from the LEFT (keep the question end).
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
    # every row after a single prefill. With RoPE (relative positions) a
    # constant left shift is harmless for greedy decoding at these scales;
    # the pad tokens are ordinary EOTs the model has seen as separators.
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
                if s.force:
                    nid = s.force.popleft()
                    forced = True
                else:
                    nid = int(choices[b])
                    forced = False
                    if stop_at_eot and nid == tok.EOT:
                        s.done = True
                        next_ids.append(tok.EOT)
                        continue
                s.generated.append(nid)
                if not forced and organizer is not None:
                    _advance_state(s, nid, tok, organizer, stats)
                if len(s.generated) >= max_new:
                    s.done = True
                next_ids.append(nid)
            if all(s.done for s in seqs):
                break
            x = torch.tensor(next_ids, dtype=torch.long, device=device)
            logits, cache = model.forward_step(x.unsqueeze(1), cache)

    texts = [tok.decode(s.generated) for s in seqs]
    return texts, stats


def generate_batch(
    model,
    tok,
    prompts: list[str],
    max_new: int,
    organizer,
    device,
    stop_at_eot: bool = True,
) -> list[str]:
    """Thin wrapper over generate_batch_with_stats returning texts only."""
    texts, _ = generate_batch_with_stats(
        model, tok, prompts, max_new, organizer, device, stop_at_eot=stop_at_eot
    )
    return texts
