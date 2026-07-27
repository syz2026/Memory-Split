"""Continuous (probability-based) reasoning metrics on existing checkpoints.

No retraining. This module is ADDITIVE to the discrete accuracy pipeline
(``evals/scorers.py``): it imports from it but modifies nothing there.

Metrics (see replication/specs/nr1-continuous-eval.md):
  M1 reference-trace NLL   : mean per-token -log p(gold CoT+answer | prompt).
                             Primary. LMLM-style "lower perplexity", restricted
                             to reasoning traces.
  M2 answer-given-gold-CoT : -log p(gold answer | prompt + gold CoT). Isolates
                             the final reasoning step from trace modeling.
  M3 answer-at-gen-slot    : p(gold answer | prompt + model's OWN decoded CoT),
                             at the post-"Answer:" slot. Soft accuracy; needs no
                             extra data but depends on the model's own (possibly
                             wrong) CoT.

M1/M2 require ``meta["solution"]`` — the exact continuation appended to the
prompt in the training doc (i.e. training_text == prompt + solution). They are
apples-to-apples only where both arms share the rendering: iGSM and deduction
(knowledge-free docs have IDENTICAL dense/split text). They are silently skipped
when the field is absent (e.g. factqa, whose renderings differ across arms, and
any pre-existing eval JSONL generated before this field was added). M3 works
everywhere.

Lower NLL is better (M1/M2); higher correct-prob is better (M3). Feed the
resulting per-item values to ``evals.stats.paired_delta_iid`` with the right
``lower_is_better`` flag for the split-minus-dense contrast.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_ANSWER_TAG = "Answer:"
_DEFAULT_CTX = 2048


def _model_ctx(model) -> int:
    return getattr(getattr(model, "cfg", None), "ctx", _DEFAULT_CTX)


def _batched_seq_logprob(
    model, tok, pairs: list[tuple[str, str]], device, batch_size: int = 16
) -> list[tuple[float, float, int]]:
    """Teacher-forced scoring of many (context, continuation) pairs.

    Returns, per pair, (sum_logprob, mean_logprob_per_token, n_continuation_tok).
    One batched forward per chunk; right-padded with EOT (padding sits AFTER every
    scored position, so causal attention never sees it). If a sequence would
    exceed the model context, the CONTEXT is truncated from the left (the
    continuation — the thing being scored — is preserved intact).
    """
    ctx_limit = _model_ctx(model)
    out: list[tuple[float, float, int]] = [(0.0, 0.0, 0)] * len(pairs)
    for lo in range(0, len(pairs), batch_size):
        chunk = pairs[lo : lo + batch_size]
        enc: list[tuple[list[int], list[int]]] = []
        for ctx, cont in chunk:
            ctx_ids = tok.encode(ctx) or [tok.EOT]
            cont_ids = tok.encode(cont)
            # keep continuation whole; left-truncate context if over the limit
            if len(ctx_ids) + len(cont_ids) > ctx_limit:
                keep = max(1, ctx_limit - len(cont_ids))
                ctx_ids = ctx_ids[-keep:]
            enc.append((ctx_ids, cont_ids))
        max_len = max(len(c) + len(k) for c, k in enc)
        x = torch.full((len(enc), max_len), tok.EOT, dtype=torch.long, device=device)
        for i, (c, k) in enumerate(enc):
            seq = c + k
            x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model.forward(x)
            logprobs = F.log_softmax(logits.float(), dim=-1)
        for i, (c, k) in enumerate(enc):
            if not k:
                out[lo + i] = (0.0, 0.0, 0)
                continue
            cpos = len(c)
            positions = torch.arange(cpos - 1, cpos - 1 + len(k), device=device)
            targets = torch.tensor(k, dtype=torch.long, device=device)
            s = logprobs[i, positions, targets].sum().item()
            out[lo + i] = (s, s / len(k), len(k))
    return out


def sequence_logprob(model, tok, context: str, continuation: str, device):
    """Single-pair teacher-forced logprob: (sum_lp, mean_lp, n_tokens).

    Thin wrapper over ``_batched_seq_logprob`` for one (context, continuation).
    ``continuation`` must encode to >= 1 token.
    """
    if not tok.encode(continuation):
        raise ValueError("continuation must encode to at least one token")
    return _batched_seq_logprob(model, tok, [(context, continuation)], device)[0]


def _split_solution(prompt: str, solution: str):
    """(context, answer_continuation) for M2, splitting on the LAST 'Answer:'.

    ``solution`` is the exact text appended to ``prompt`` in the training doc.
    Returns None if there is no 'Answer:' marker.
    """
    idx = solution.rfind(_ANSWER_TAG)
    if idx == -1:
        return None
    cut = idx + len(_ANSWER_TAG)
    return prompt + solution[:cut], solution[cut:]


def score_items_continuous(
    model,
    tok,
    items,
    device,
    batch_size: int = 16,
    metrics: tuple[str, ...] = ("m1", "m2", "m3"),
    organizer=None,
    max_new: int = 384,
) -> tuple[list[dict], dict]:
    """Continuous metrics per QAItem. Returns (rows, agg).

    rows: one dict per item with {qid, task, meta} plus any of
    {m1_nll, m2_nll, m3_correct_prob} that were computable. A metric key is
    OMITTED (not None) when it could not be computed for that item, so
    downstream ``paired_delta_iid`` drops it pairwise.

    agg: {metric: {task: {mean, n}}} — per-arm means (this call scores one arm);
    the split-minus-dense contrast is formed later from two arms' rows via
    ``paired_delta_iid``.
    """
    rows: list[dict] = [
        {"qid": it.qid, "task": it.task, "meta": dict(it.meta)} for it in items
    ]

    if "m1" in metrics:
        idxs, pairs = [], []
        for i, it in enumerate(items):
            sol = it.meta.get("solution")
            if sol:
                idxs.append(i)
                pairs.append((it.prompt, sol))
        for i, (_s, mean_lp, _n) in zip(
            idxs, _batched_seq_logprob(model, tok, pairs, device, batch_size)
        ):
            rows[i]["m1_nll"] = -mean_lp

    if "m2" in metrics:
        idxs, pairs = [], []
        for i, it in enumerate(items):
            sol = it.meta.get("solution")
            if not sol:
                continue
            split = _split_solution(it.prompt, sol)
            if split is None:
                continue
            idxs.append(i)
            pairs.append(split)
        for i, (_s, mean_lp, _n) in zip(
            idxs, _batched_seq_logprob(model, tok, pairs, device, batch_size)
        ):
            rows[i]["m2_nll"] = -mean_lp

    if "m3" in metrics:
        from evals.generate import generate_batch_with_stats

        for lo in range(0, len(items), batch_size):
            chunk = items[lo : lo + batch_size]
            texts, _ = generate_batch_with_stats(
                model, tok, [it.prompt for it in chunk], max_new, organizer, device
            )
            answer_pairs, answer_idxs = [], []
            for j, (it, gen) in enumerate(zip(chunk, texts)):
                gi = gen.rfind(_ANSWER_TAG)
                if gi == -1:
                    continue  # model never produced an answer slot ⇒ omit M3
                ctx = it.prompt + gen[: gi + len(_ANSWER_TAG)]
                cont = " " + it.answer  # training answer line is "Answer: {answer}"
                answer_idxs.append(lo + j)
                answer_pairs.append((ctx, cont))
            for i, (_sum_lp, mean_lp, n) in zip(
                answer_idxs,
                _batched_seq_logprob(model, tok, answer_pairs, device, batch_size),
            ):
                # per-token (geometric-mean) prob, length-normalized so the
                # cross-item aggregate isn't biased by answer length (L29); the
                # paired split−dense delta is unaffected (same item, same length).
                rows[i]["m3_correct_prob"] = math.exp(mean_lp) if n else 0.0

    agg: dict = {}
    for key in ("m1_nll", "m2_nll", "m3_correct_prob"):
        by_task: dict[str, list[float]] = {}
        for r in rows:
            if key in r:
                by_task.setdefault(r["task"], []).append(r[key])
        if by_task:
            agg[key] = {
                t: {"mean": sum(v) / len(v), "n": len(v)} for t, v in by_task.items()
            }
    return rows, agg
