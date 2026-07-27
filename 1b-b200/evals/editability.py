"""E2 — editability / unlearning (no retraining).

An independent backing of H3 ("facts live in the store, not the weights"): edit
an organizer entry and the split arm's answer changes with ZERO weight update,
while the dense (closed-book) arm's answer is fixed in weights and can only be
changed by retraining. Precedent: LMLM (instant unlearning by DB deletion),
Larimar (side-effect-free editing); CLS "fast store is updatable" corollary.

Two metrics:
  * edit_success_rate — after swapping a fact's value in the organizer, does the
    split arm (store ON) now emit the NEW value? (Should be ~ store hit-rate,
    since store-OFF recall≈0 means no memorized value competes.)
  * locality_rate — do UNEDITED entities still answer correctly under the edited
    store? (Exact-match store ⇒ edits are local by construction; measured, not
    assumed.)

Reuses `evals.recall.recall_accuracy` with a mutated organizer; no model changes.
"""

from __future__ import annotations

import datetime
import random

from corpusgen.bios import (
    BIRTH_DATE_MAX,
    BIRTH_DATE_MIN,
    RELATION_PHRASES,
    VALUE_POOLS,
    format_date,
)
from corpusgen.records import QAItem
from evals.recall import recall_accuracy
from organizer.store import Organizer

_N_DAYS = (BIRTH_DATE_MAX - BIRTH_DATE_MIN).days + 1
_PROBE_ATTRS = tuple(a for a in RELATION_PHRASES if a == "birth_date" or a in VALUE_POOLS)


def new_value(attr: str, old: str, rng: random.Random) -> str:
    """A pool-valid value for ``attr`` that differs from ``old``."""
    if attr == "birth_date":
        while True:
            v = format_date(BIRTH_DATE_MIN + datetime.timedelta(days=rng.randrange(_N_DAYS)))
            if v != old:
                return v
    return rng.choice([v for v in VALUE_POOLS[attr] if v != old])


def make_edited_organizer(organizer: Organizer, edits) -> Organizer:
    """A NEW organizer = copy of ``organizer`` with ``edits`` applied.

    ``edits``: iterable of ``(name, relation, new_value)``. The input organizer is
    left unmutated (returns a fresh copy).
    """
    edited = Organizer()
    edited._table = dict(organizer._table)  # shallow copy of the key->value map
    for name, relation, value in edits:
        edited.add(name, relation, value)
    return edited


def editability_eval(model, tok, organizer, records, device,
                     n_edits: int = 200, seed: int = 0, attrs=_PROBE_ATTRS,
                     max_new: int = 48, batch_size: int = 64) -> dict:
    """Edit-success + locality for the split arm under a mutated store.

    Samples disjoint edited / locality entity sets. Returns
    {edit_success_rate, n_edits, locality_rate, n_locality, dense_editable, note}.
    """
    rng = random.Random(seed)
    sample = rng.sample(records, min(2 * n_edits, len(records)))
    edited_recs = sample[:n_edits]
    locality_recs = sample[n_edits:2 * n_edits]

    edits, edited_probes = [], []
    for rec in edited_recs:
        attr = rng.choice(list(attrs))
        val = new_value(attr, rec.attrs[attr], rng)
        edits.append((rec.name, attr, val))
        edited_probes.append(QAItem(
            qid=f"edit-{rec.entity_id}-{attr}", task="recall",
            prompt=f"{rec.name}'s {RELATION_PHRASES[attr]} is",
            answer=val, meta={"relation": attr}))
    edited_org = make_edited_organizer(organizer, edits)

    es = recall_accuracy(model, tok, edited_probes, "on", edited_org, device,
                         max_new=max_new, batch_size=batch_size)

    loc_probes = [
        QAItem(qid=f"loc-{rec.entity_id}-{a}", task="recall",
               prompt=f"{rec.name}'s {RELATION_PHRASES[a]} is",
               answer=rec.attrs[a], meta={"relation": a})
        for rec in locality_recs
        for a in [rng.choice(list(attrs))]
    ]
    loc = recall_accuracy(model, tok, loc_probes, "on", edited_org, device,
                          max_new=max_new, batch_size=batch_size) if loc_probes else {"overall": None}

    return {
        "edit_success_rate": es["overall"],
        "n_edits": len(edited_probes),
        "locality_rate": loc["overall"],
        "n_locality": len(loc_probes),
        "dense_editable": False,  # closed-book answer is fixed in weights
        "note": "split follows the edited store with zero weight update; "
                "dense requires retraining to change a fact",
    }
