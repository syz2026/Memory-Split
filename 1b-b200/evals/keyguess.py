"""Key-generation generalization eval — in-schema, OOD-entity version.

Question: can a split-arm model construct the *exactly correct* organizer key
`"{name}, {relation}"` for entities it never trained on? This is the addressing
skill behind the whole externalization mechanism, decomposed into:

  - name-half  : did it copy the correct entity out of the prompt? (a COPY op)
  - relation-half: did it emit the correct relation? (a 6-way classification)

and a mutually-exclusive outcome taxonomy (no_lookup / malformed / correct_key /
wrong_key_same_name / wrong_key_other). "Generalization" = held-out (fresh,
never-trained) entities vs a "seen" (trained-entity) control.

Model-agnostic given the two repo contracts it reuses:
  - `evals.generate.generate_batch_with_stats` (the <|db_start|>…<|db_retrieve|>
    …<|db_end|> decode + organizer interception), and
  - `organizer.store.normalize` (lowercase + whitespace-collapse), which must be
    the SAME canonicalization the store/keys use.

The emitted key is recovered from the decoded text: the query tokens sit between
the literal `<|db_start|>` and `<|db_retrieve|>` strings that tiktoken emits.
"""

from __future__ import annotations

from evals.generate import generate_batch_with_stats
from evals.scorers import normalize_answer
from organizer.store import normalize

DB_START = "<|db_start|>"
DB_RETRIEVE = "<|db_retrieve|>"


def extract_key(text: str) -> tuple[str | None, str]:
    """Return (raw_key_string, status) from a decoded generation.

    status in {"lookup", "no_lookup", "malformed"}:
      - no_lookup : never opened a lookup (<|db_start|> absent)
      - malformed : opened a lookup but never reached <|db_retrieve|>
      - lookup    : a well-formed query span was found (key = text between them)
    Uses the FIRST lookup (recall probes target a single fact).
    """
    i = text.find(DB_START)
    if i == -1:
        return None, "no_lookup"
    j = text.find(DB_RETRIEVE, i + len(DB_START))
    if j == -1:
        return None, "malformed"
    return text[i + len(DB_START) : j].strip(), "lookup"


def score_item(gen_text: str, name: str, relation: str, value: str) -> dict:
    """Score one probe. `relation` is the RAW key the model emits (e.g.
    'birth_city'), NOT the phrase used in the prompt."""
    gold = normalize(f"{name}, {relation}")
    key, status = extract_key(gen_text)
    res = {
        "outcome": status if status != "lookup" else None,
        "emitted_key": key,
        "correct_key": 0,
        "name_ok": 0,
        "rel_ok": 0,
        # answer-half: does the gold value show up (a correct key force-decodes it)
        "answer_ok": int(normalize_answer(value) in normalize_answer(gen_text)),
    }
    if status != "lookup":
        return res

    nkey = normalize(key)
    # split emitted key into name / relation on the LAST comma-space
    if ", " in key:
        e_name, e_rel = key.rsplit(", ", 1)
    else:
        e_name, e_rel = key, ""
    name_ok = normalize(e_name) == normalize(name)
    rel_ok = normalize(e_rel) == normalize(relation)
    res["name_ok"] = int(name_ok)
    res["rel_ok"] = int(rel_ok)
    if nkey == gold:
        res["outcome"] = "correct_key"
        res["correct_key"] = 1
    elif name_ok and not rel_ok:
        res["outcome"] = "wrong_key_same_name"
    else:
        res["outcome"] = "wrong_key_other"
    return res


_OUTCOMES = ("correct_key", "wrong_key_same_name", "wrong_key_other",
             "no_lookup", "malformed")


def keyguess_eval(model, tok, records, attributes, relation_phrases, organizer,
                  device, max_new: int = 24, batch_size: int = 64) -> dict:
    """Score key-generation over `records` × `attributes`.

    Prompts use the training-consistent recall stub "{name}'s {phrase} is"; the
    gold key uses the RAW attribute string (what the model emits in its query).
    The organizer must already contain the probed entities' facts so a correct
    key retrieves the value (enables answer-half). Returns a summary dict.
    """
    probes = [(rec, attr) for rec in records for attr in attributes]
    prompts = [f"{rec.name}'s {relation_phrases[attr]} is" for rec, attr in probes]

    per_item: list[dict] = []
    stats_total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    for lo in range(0, len(prompts), batch_size):
        texts, stats = generate_batch_with_stats(
            model, tok, prompts[lo : lo + batch_size], max_new, organizer, device
        )
        for k in stats_total:
            stats_total[k] += stats[k]
        for (rec, attr), gen in zip(probes[lo : lo + batch_size], texts):
            s = score_item(gen, rec.name, attr, rec.attrs[attr])
            s["relation"] = attr
            per_item.append(s)

    n = len(per_item) or 1
    outcomes = {o: 0 for o in _OUTCOMES}
    per_rel: dict[str, list[int]] = {}  # attr -> [n, correct, name_ok, rel_ok, answer_ok]
    tot = [0, 0, 0, 0]  # correct, name_ok, rel_ok, answer_ok
    for s in per_item:
        outcomes[s["outcome"]] += 1
        tot[0] += s["correct_key"]
        tot[1] += s["name_ok"]
        tot[2] += s["rel_ok"]
        tot[3] += s["answer_ok"]
        pr = per_rel.setdefault(s["relation"], [0, 0, 0, 0, 0])
        pr[0] += 1
        pr[1] += s["correct_key"]
        pr[2] += s["name_ok"]
        pr[3] += s["rel_ok"]
        pr[4] += s["answer_ok"]

    return {
        "n": len(per_item),
        "key_accuracy": tot[0] / n,
        "name_half_accuracy": tot[1] / n,
        "relation_half_accuracy": tot[2] / n,
        "answer_accuracy": tot[3] / n,
        "chance_relation_half": 1.0 / len(attributes),
        "outcomes": {o: c for o, c in outcomes.items()},
        "outcomes_frac": {o: c / n for o, c in outcomes.items()},
        "per_relation": {
            a: {
                "n": v[0],
                "key_accuracy": v[1] / v[0] if v[0] else 0.0,
                "name_half_accuracy": v[2] / v[0] if v[0] else 0.0,
                "relation_half_accuracy": v[3] / v[0] if v[0] else 0.0,
                "answer_accuracy": v[4] / v[0] if v[0] else 0.0,
            }
            for a, v in per_rel.items()
        },
        "stats": stats_total,
    }
