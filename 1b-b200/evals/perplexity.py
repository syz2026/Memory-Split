"""NR-6 — held-out bits-per-byte on stratified slices (no retraining).

Supporting evidence for the "no capability regression + facts live outside the
weights" story (spec §6). Prediction: with the store unplugged the split arm's
bpb rises specifically on the FACTUAL (bio) slice, not on bed/reasoning — the
factual slice is where externalized knowledge would have gone.

bits-per-byte is tokenizer-independent (normalizes by UTF-8 bytes, not tokens),
so dense and split are comparable even though the split rendering tokenizes
differently — as long as the SAME text is scored for both (we score the dense
rendering of every slice on both arms).

Pure teacher-forced scoring; testable offline with a stub whose forward returns
known logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LN2 = math.log(2.0)
_DEFAULT_CTX = 1024


def _model_ctx(model, fallback: int = _DEFAULT_CTX) -> int:
    return getattr(getattr(model, "cfg", None), "ctx", fallback)


def _doc_nll(model, tok, text: str, device, max_ctx: int) -> tuple[float, int, int]:
    """Teacher-forced NLL (nats) of one text plus its UTF-8 byte count.

    Returns (nll_nats, n_bytes, n_scored). Tokens are scored in non-overlapping
    windows of <= max_ctx; the first token of each window is unscored (no
    preceding context). Returns (0.0, n_bytes, 0) for texts too short to score.
    """
    ids = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    if len(ids) < 2 or n_bytes == 0:
        return 0.0, n_bytes, 0
    total = 0.0
    scored = 0
    for i in range(0, len(ids), max_ctx):
        chunk = ids[i : i + max_ctx]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model.forward(x)
            logprobs = F.log_softmax(logits.float(), dim=-1)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=device)
        positions = torch.arange(len(chunk) - 1, device=device)
        lp = logprobs[0, positions, targets].sum().item()
        total += -lp
        scored += len(chunk) - 1
    return total, n_bytes, scored


def bits_per_byte(model, tok, text: str, device, max_ctx: int | None = None) -> float | None:
    """bits-per-byte of one text (None if too short to score)."""
    max_ctx = max_ctx or _model_ctx(model)
    nll, n_bytes, scored = _doc_nll(model, tok, text, device, max_ctx)
    if scored == 0 or n_bytes == 0:
        return None
    return (nll / _LN2) / n_bytes


def fact_value_nll(model, tok, records, device, attrs=None, batch_size: int = 64) -> dict:
    """Per-token NLL (nats) of attribute VALUES given a fixed dense probe context.

    The clean factual-slice instrument (ledger L27): scores only the value tokens
    of "{name}'s {relation} is → {value}", with an IDENTICAL dense context for
    both arms. This removes (a) the dilution of averaging value tokens with
    template tokens and (b) the format confound of scoring the split arm on dense
    bio prose it never saw. Use FRESH (held-out) entities. Higher NLL ⇒ the value
    is less predictable from the weights alone (facts externalized).

    Returns {nll_per_token, per_attribute, n}. Lower is better; for the split arm
    (store OFF) this should sit near the uniform ceiling, dense well below it.
    """
    from corpusgen.bios import RELATION_PHRASES
    from evals.continuous import _batched_seq_logprob

    attrs = attrs or tuple(RELATION_PHRASES)
    pairs, keys = [], []
    for rec in records:
        for a in attrs:
            pairs.append((f"{rec.name}'s {RELATION_PHRASES[a]} is", " " + rec.attrs[a]))
            keys.append(a)
    if not pairs:
        return {"nll_per_token": None, "per_attribute": {}, "n": 0}

    res = _batched_seq_logprob(model, tok, pairs, device, batch_size)
    total_nll = 0.0
    total_tok = 0
    per: dict[str, list[float]] = {}
    for (sum_lp, _mean, n), a in zip(res, keys):
        total_nll += -sum_lp
        total_tok += n
        pa = per.setdefault(a, [0.0, 0])
        pa[0] += -sum_lp
        pa[1] += n
    return {
        "nll_per_token": total_nll / total_tok if total_tok else None,
        "per_attribute": {a: (v[0] / v[1] if v[1] else None) for a, v in per.items()},
        "n": len(pairs),
    }


def slice_bits_per_byte(model, tok, texts, device, max_ctx: int | None = None) -> dict:
    """Aggregate bpb over a slice (a list of texts).

    Accumulates total NLL and total bytes across docs (not a mean of per-doc
    bpb), so long and short docs are weighted by size. Returns
    {bpb, n_docs, n_bytes, n_scored}.
    """
    max_ctx = max_ctx or _model_ctx(model)
    total_nll = 0.0
    total_bytes = 0
    total_scored = 0
    n_docs = 0
    for text in texts:
        nll, n_bytes, scored = _doc_nll(model, tok, text, device, max_ctx)
        if scored == 0:
            continue
        total_nll += nll
        total_bytes += n_bytes
        total_scored += scored
        n_docs += 1
    bpb = (total_nll / _LN2) / total_bytes if total_bytes else None
    return {"bpb": bpb, "n_docs": n_docs, "n_bytes": total_bytes, "n_scored": total_scored}
