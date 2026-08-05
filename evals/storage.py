"""Recoverable predictive bits about the fact set.

This measures how much information about the entity→value mapping a model
makes available under a held-out query prompt. It is deliberately NOT called
"storage": prompt-conditioned NLL cannot separate what a model has stored from
what it can be induced to emit, and the previous accuracy-to-bits conversion
conflated the two badly enough that its ledger had to be withdrawn.

Two specification rules, both of which the first draft got wrong:

Baseline and ceiling share their conditioning. Against an unconditional pool
baseline a perfect model recovers 52.96 bits/entity. Against a
length-conditioned baseline it recovers 45.30, because a value's token length
already leaks 7.66 bits about which value it is. Mixing a conditional baseline
with an unconditional ceiling overstates recovery by that gap.

Queries are held-out paraphrases. Scoring a training surface form measures
template memorisation, not the fact. The probe phrasings below never appear in
BIO_TEMPLATES, which `tests/test_storage.py` asserts directly.
"""

from __future__ import annotations

import collections
import datetime
import statistics
import math

import torch

from corpusgen import bios
from corpusgen.records import ATTRIBUTES, BioRecord

# Query phrasings held out from training. Each is (prefix, suffix) around the
# value, matching the BIO_TEMPLATES shape so the seam is identical, but none
# of these strings occurs in BIO_TEMPLATES.
PROBE_TEMPLATES: dict[str, tuple[str, str]] = {
    "birth_date": ("Record lookup. {name} has date of birth", "."),
    "birth_city": ("Record lookup. {name} has city of birth", "."),
    "university": ("Record lookup. {name} has alma mater", "."),
    "major": ("Record lookup. {name} has degree subject", "."),
    "employer": ("Record lookup. {name} has place of work", "."),
    "current_city": ("Record lookup. {name} has city of residence", "."),
}

N_BIRTH_DAYS = (bios.BIRTH_DATE_MAX - bios.BIRTH_DATE_MIN).days + 1


def _birth_date_pool() -> list[str]:
    """Every representable birth date. Used only for the length histogram."""
    step = max(1, N_BIRTH_DAYS // 2000)
    return [
        bios.format_date(bios.BIRTH_DATE_MIN + datetime.timedelta(days=d))
        for d in range(0, N_BIRTH_DAYS, step)
    ]


def pool_size(attr: str) -> int:
    if attr == "birth_date":
        return N_BIRTH_DAYS
    return len(bios.VALUE_POOLS[attr])


def unconditional_bits(attr: str) -> float:
    """log2 |pool|. Sums to 52.96 over the six attributes."""
    return math.log2(pool_size(attr))


def length_conditioned_bits(attr: str, tok) -> dict[int, float]:
    """log2 of the number of pool values sharing each token length.

    A model that knows only the pool and the value's length already has
    log2(|pool|) - this many bits, so this is what must be subtracted to avoid
    crediting length as knowledge.
    """
    pool = _birth_date_pool() if attr == "birth_date" else bios.VALUE_POOLS[attr]
    by_len: collections.Counter[int] = collections.Counter(
        len(tok.encode(" " + v)) for v in pool
    )
    if attr == "birth_date":
        # The pool is subsampled for speed; every date has the same token
        # length, so the conditional baseline equals the unconditional one.
        if len(by_len) == 1:
            return {next(iter(by_len)): unconditional_bits(attr)}
        scale = N_BIRTH_DAYS / sum(by_len.values())
        return {L: math.log2(c * scale) for L, c in by_len.items()}
    return {L: math.log2(c) for L, c in by_len.items()}


def baseline_bits(attr: str, value: str, tok, baseline: str) -> float:
    if baseline == "unconditional":
        return unconditional_bits(attr)
    if baseline == "length":
        table = _LENGTH_TABLES.setdefault(
            (attr, id(tok)), length_conditioned_bits(attr, tok)
        )
        n_tok = len(tok.encode(" " + value))
        # A length never seen in the pool identifies the value outright.
        return table.get(n_tok, 0.0)
    raise ValueError(f"baseline must be 'unconditional' or 'length', got {baseline!r}")


_LENGTH_TABLES: dict[tuple[str, int], dict[int, float]] = {}


def probe_queries(
    records: list[BioRecord], attrs: tuple[str, ...] = ATTRIBUTES
) -> list[dict]:
    """One held-out query per (entity, attribute)."""
    out = []
    for rec in records:
        for attr in attrs:
            prefix, _ = PROBE_TEMPLATES[attr]
            out.append(
                {
                    "entity_id": rec.entity_id,
                    "attr": attr,
                    "prompt": prefix.format(name=rec.name),
                    "value": rec.attrs[attr],
                }
            )
    return out


@torch.no_grad()
def _value_nll_bits(model, tok, queries: list[dict], device, batch_size: int) -> list[float]:
    """Teacher-forced -log2 p(value tokens | prompt), summed over the value."""
    out: list[float] = []
    for lo in range(0, len(queries), batch_size):
        chunk = queries[lo : lo + batch_size]
        prompts = [tok.encode(q["prompt"]) for q in chunk]
        values = [tok.encode(" " + q["value"]) for q in chunk]
        widths = [len(p) + len(v) for p, v in zip(prompts, values)]
        width = max(widths)
        ids = torch.full((len(chunk), width), tok.EOT, dtype=torch.long)
        for b, (p, v) in enumerate(zip(prompts, values)):
            ids[b, : len(p) + len(v)] = torch.tensor(p + v, dtype=torch.long)
        logits, _ = model(ids.to(device))
        logprobs = torch.log_softmax(logits.float(), dim=-1).cpu()
        for b, (p, v) in enumerate(zip(prompts, values)):
            total = 0.0
            for j, vid in enumerate(v):
                # position len(p)+j-1 predicts value token j
                total += float(logprobs[b, len(p) + j - 1, vid])
            out.append(-total / math.log(2.0))
    return out


def recoverable_bits(
    model,
    tok,
    records: list[BioRecord],
    device,
    baseline: str = "unconditional",
    batch_size: int = 32,
    n_params: int | None = None,
) -> dict:
    """Bits about the entity→value mapping recoverable under a held-out query.

    Per query: baseline_bits - value_nll_bits. Positive means the model beats
    the pool prior. Values are NOT clamped at zero per item; a model at the
    prior should average to zero, and clamping would bias every arm upward,
    which is one of the ways the withdrawn ledger overstated storage.
    """
    queries = probe_queries(records)
    nll = _value_nll_bits(model, tok, queries, device, batch_size)

    per_attr: dict[str, list[float]] = collections.defaultdict(list)
    base_by_attr: dict[str, list[float]] = collections.defaultdict(list)
    per_entity: dict[object, float] = collections.defaultdict(float)
    for q, n in zip(queries, nll):
        b = baseline_bits(q["attr"], q["value"], tok, baseline)
        per_attr[q["attr"]].append(b - n)
        base_by_attr[q["attr"]].append(b)
        per_entity[q["entity_id"]] += b - n

    n_entities = len(records)
    bits_total = sum(sum(v) for v in per_attr.values())
    base_total = sum(sum(v) for v in base_by_attr.values())
    out = {
        "baseline": baseline,
        "n_entities": n_entities,
        "n_queries": len(queries),
        "bits_total": bits_total,
        "bits_per_entity": bits_total / max(1, n_entities),
        "baseline_bits": base_total,
        "baseline_bits_per_entity": base_total / max(1, n_entities),
        "per_attribute": {
            a: {"bits_total": sum(v), "bits_per_entity": sum(v) / max(1, n_entities)}
            for a, v in per_attr.items()
        },
    }
    # Entity-sampling uncertainty on bits_per_entity.
    #
    # The probe scores a few hundred entities and `run_evals` multiplies the
    # result by the corpus entity count -- ~2,000x at the operating point. That
    # extrapolation carries the sampling error with it, and the error is
    # invisible to the analyzer's across-seed interval because `corpus_seed` is
    # pinned, so every seed probes the SAME entities and their errors are
    # perfectly correlated. Preregistration §7 requires gates to use confidence
    # bounds; without this the bound omits its dominant term.
    vals = list(per_entity.values())
    if len(vals) >= 2:
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals)
        se = sd / math.sqrt(len(vals))
        out["bits_per_entity_sd"] = sd
        out["bits_per_entity_se"] = se
        out["bits_per_entity_ci95"] = [mean - 1.96 * se, mean + 1.96 * se]
        out["bits_per_entity_rel_se"] = abs(se / mean) if mean else float("inf")
    out["n_entities_probed"] = len(vals)

    if n_params:
        out["bits_per_param"] = bits_total / n_params
    return out
