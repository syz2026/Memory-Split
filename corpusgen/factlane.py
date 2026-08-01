"""The fact lane: N entities rendered E times each, one stream plus a mask.

The lane exists to put a measurable memorisation burden on the dense arm. Two
properties do that work, and the previous corpora had neither:

Exposure. Each fact is rendered `exposures` times in a different surface form
every time (120 paraphrase templates, 8 orderings, 3 name forms), so the model
must learn the fact rather than a sentence. Exposures are spread in grouped
rounds across the whole stream, never massed, because massed repetition is
memorised and then decayed.

Unpredictability. Values are drawn independently of context from invented
pools, so no amount of surrounding text predicts them. The reasoning-v3 corpus
failed here: its offloaded spans were 94.2% recoverable from context, so
masking removed a burden that did not exist.

Load varies by trading entities against exposures at a fixed document count
(see `load_variants`), which keeps tokens, optimizer steps and mask mass
identical across loads so the only thing moving is the unique-entropy ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from corpusgen import bios
from corpusgen.records import ATTRIBUTES, BioRecord

# Measured on the shipped generator with the repo tokenizer, 2026-08-01.
TOKENS_PER_DOC = 74.91
# Analytic from the pool sizes: log2(27759) + log2(200) + log2(300) +
# log2(100) + log2(263) + log2(200). Do not substitute an empirical estimate;
# it underestimates birth_date badly at any tractable sample size.
BITS_PER_ENTITY = 52.96


def plan(
    bits_per_param: float,
    n_params: int,
    exposures: int,
    n_nonembed: int | None = None,
) -> dict:
    """Entities and tokens needed to demand `bits_per_param` of a model.

    The three quantities trade off exactly:
        fact_tokens = entities x exposures x tokens_per_doc
    Raising exposure to guarantee memorisation shrinks the fact universe, and
    the universe is what sets the storage burden.
    """
    bits_total = bits_per_param * n_params
    n_entities = int(round(bits_total / BITS_PER_ENTITY))
    fact_tokens = int(round(n_entities * exposures * TOKENS_PER_DOC))
    out = {
        "bits_per_param": bits_per_param,
        "n_params": n_params,
        "exposures": exposures,
        "n_entities": n_entities,
        "n_docs": n_entities * exposures,
        "fact_tokens": fact_tokens,
        "bits_total": n_entities * BITS_PER_ENTITY,
    }
    if n_nonembed:
        out["bits_per_nonembed"] = out["bits_total"] / n_nonembed
    return out


def load_variants(n_entities: int, exposures: int, fractions: tuple[float, ...]) -> list[dict]:
    """Fact loads that differ ONLY in unique entropy.

    High load is `N` entities at `E` exposures; a load at fraction `f` is
    `f*N` entities at `E/f` exposures. Document count, and therefore total
    tokens, optimizer steps, mask mass and mask positions, are identical
    across every variant -- so a load-by-arm interaction cannot be confounded
    with training duration. Shrinking the lane instead, which the first draft
    did, moved all of those at once.
    """
    total_docs = n_entities * exposures
    divisors = _divisors(total_docs)
    out = []
    for f in fractions:
        # Snap to an exact divisor. Rounding n and e independently leaves
        # n*e off by a fraction of a percent, which would put the loads on
        # different token budgets and different step counts -- reintroducing
        # the confound this whole construction exists to remove.
        want = max(1, n_entities * f)
        n = min(divisors, key=lambda d: (abs(d - want), d))
        e = total_docs // n
        assert n * e == total_docs, (n, e, total_docs)
        out.append(
            {
                "fraction": f,
                "n_entities": n,
                "exposures": e,
                "n_docs": n * e,
                "bits_total": n * BITS_PER_ENTITY,
            }
        )
    return out


def choose_entity_count(target: int, exposures: int,
                        fractions: tuple[float, ...],
                        search: int = 2000) -> int:
    """An entity count near `target` whose document total divides cleanly.

    `load_variants` snaps each load to an exact divisor of N*E. If N*E has a
    sparse divisor lattice the snap can miss the requested fraction badly --
    300,000 x 100 puts the 0.3 load 13% off. Searching a small neighbourhood
    for a count whose divisors bracket every fraction tightly costs nothing
    at build time and keeps the dose spacing as designed.
    """
    best, best_err = target, float("inf")
    for cand in range(max(1, target - search), target + search + 1):
        divs = _divisors(cand * exposures)
        err = 0.0
        for f in fractions:
            want = max(1.0, cand * f)
            got = min(divs, key=lambda d: (abs(d - want), d))
            err = max(err, abs(got - want) / want)
        if err < best_err:
            best, best_err = cand, err
        if err == 0.0:
            break
    return best


def _divisors(n: int) -> list[int]:
    out = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.add(i)
            out.add(n // i)
        i += 1
    return sorted(out)


def doc_order(n_entities: int, exposures: int) -> Iterator[tuple[int, int]]:
    """(entity, exposure) in grouped rounds: every entity once at exposure 0,
    then every entity at exposure 1, and so on.

    Spacing is the point. All E copies of one fact adjacent would be memorised
    and then decayed; one round is n_entities documents wide, so a fact's
    exposures land evenly across the whole run.
    """
    for exposure in range(exposures):
        for entity in range(n_entities):
            yield entity, exposure


def verify_doc(rec: BioRecord, exposure: int, tok) -> None:
    """Assert the invariants a sidecar depends on.

    Masked spans must decode to exactly the six attribute values, and no value
    may also appear unmasked. A template that leaked a value into plain text
    would let the masked arm read the answer it was denied -- which is the
    failure mode that made reasoning-v3's offloaded spans recoverable.
    """
    segs = bios.render_bio_marked(rec, exposure)
    ids, mask = tok.encode_segments(segs, add_eot=True)
    masked_runs: list[str] = []
    run: list[int] = []
    for i, m in enumerate(mask):
        if m == 0:
            run.append(ids[i])
        elif run:
            masked_runs.append(tok.decode(run).strip())
            run = []
    if run:
        masked_runs.append(tok.decode(run).strip())
    got = sorted(s for s in masked_runs if s)
    want = sorted(rec.attrs[a] for a in ATTRIBUTES)
    if got != want:
        raise AssertionError(
            f"entity {rec.entity_id} exposure {exposure}: masked spans {got} != {want}"
        )
    plain = "".join(t for t, m in segs if not m)
    for a in ATTRIBUTES:
        if rec.attrs[a] in plain:
            raise AssertionError(
                f"entity {rec.entity_id} exposure {exposure}: {a} value leaks unmasked"
            )


def emit(
    records: list[BioRecord],
    exposures: int,
    tok,
    verify_every: int = 10_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (ids uint16, factmask uint8) per document.

    factmask is 1 = supervised, 0 = offloaded, matching train/data.py. The
    dense sidecar for the same ids is all ones and is written by the caller,
    because both arms share these exact tokens.

    `verify_doc` runs on a sample rather than never: it existed in the
    previous builder and was called only from its unit tests, so no production
    corpus ever asserted the invariant it protects.
    """
    for n, (entity, exposure) in enumerate(doc_order(len(records), exposures)):
        rec = records[entity]
        if verify_every and n % verify_every == 0:
            verify_doc(rec, exposure, tok)
        segs = bios.render_bio_marked(rec, exposure)
        ids, mask = tok.encode_segments(segs, add_eot=True)
        yield np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


def measure_geometry(records: list[BioRecord], tok, n_sample: int = 300) -> dict:
    """Actual tokens/doc and masked fraction, for sizing and for the manifest."""
    lens, masked = [], 0
    total = 0
    for rec in records[:n_sample]:
        for e in range(3):
            ids, mask = tok.encode_segments(
                bios.render_bio_marked(rec, e), add_eot=True
            )
            lens.append(len(ids))
            masked += sum(1 for m in mask if m == 0)
            total += len(mask)
    return {
        "tokens_per_doc": sum(lens) / len(lens),
        "masked_fraction": masked / total,
        "masked_tokens_per_doc": masked / len(lens),
        "n_docs_sampled": len(lens),
    }


def entities_for_budget(fact_tokens: int, exposures: int) -> int:
    return max(1, int(fact_tokens // (exposures * TOKENS_PER_DOC)))


def bits_per_param(n_entities: int, n_params: int) -> float:
    return n_entities * BITS_PER_ENTITY / n_params


def occupancy_report(n_entities: int, presets: dict[str, int]) -> dict:
    """bits/param against each model size. Reported against TOTAL parameters:
    the Allen-Zhu & Li 2 bits/param figure is a 1000-exposure result over
    total parameters, so a non-embedding denominator is not comparable to it.
    """
    bits = n_entities * BITS_PER_ENTITY
    return {
        "n_entities": n_entities,
        "bits_total": bits,
        "bits_per_param": {k: bits / v for k, v in presets.items()},
        "log2_pool_check": round(sum(
            math.log2(s) for s in (27759, 200, 300, 100, 263, 200)
        ), 4),
    }
