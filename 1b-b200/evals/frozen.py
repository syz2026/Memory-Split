"""Load the FROZEN entities the checkpoints were actually trained on.

Why this exists: the repo's `corpusgen.bios.generate_records` has drifted from the
corpus generator that built the training data on the cluster (same name/value
pools, different RNG sequence — verified: entity 0 and 31352 both differ, no seed
reproduces them). So any probe that *regenerates* entities queries the model about
people it never saw. To probe the model on its ACTUAL trained entities (the
correct population for any memorization/storage question), reconstruct the records
from the frozen `eval/recall.jsonl` that was written from the same records as the
training tokens.

`recall.jsonl` rows look like:
  {"qid": "...", "task": "recall",
   "prompt": "Berian Aris Birchby's birth date is", "answer": "July 1, 1994",
   "meta": {"entity_id": 31352, "relation": "birth_date", "template": "..."}}
so name = prompt up to "'s ", relation = meta.relation (raw key), value = answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from corpusgen.records import ATTRIBUTES, BioRecord


def records_from_recall_jsonl(path: str | Path) -> list[BioRecord]:
    """Reconstruct complete BioRecords (all 6 attrs) from a frozen recall.jsonl.

    Returns records sorted by entity_id. Only entities with all ATTRIBUTES present
    are returned (recall.jsonl carries every attribute per entity, so that is all
    of them). These are the true TRAINED (seen) entities.
    """
    names: dict[int, str] = {}
    attrs: dict[int, dict[str, str]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            eid = r["meta"]["entity_id"]
            rel = r["meta"]["relation"]
            # name is the prompt minus the "'s {phrase} is" tail; names contain no "'s"
            names[eid] = r["prompt"].split("'s ", 1)[0]
            attrs.setdefault(eid, {})[rel] = r["answer"]
    records: list[BioRecord] = []
    for eid in sorted(attrs):
        if all(a in attrs[eid] for a in ATTRIBUTES):
            records.append(BioRecord(entity_id=eid, name=names[eid], attrs=attrs[eid]))
    return records
