"""OOD two-fact composition generator (P3 composition endpoint).

The one place theory still predicts a split win: out-of-distribution two-hop
composition. A dense model must grok a hard parametric two-hop; an explicit
lookup makes both facts co-present and turns it into two one-hop lookups plus a
join (Grokked Transformers 2405.15071; Compositionality Gap 2210.03350).

Task. Reuse the bioS entities from ``corpusgen.bios`` and add *functional*
bridge relations (each person -> exactly one other person, e.g. mentor/advisor).
The two-hop query is "What is the {attribute} of {person}'s {relation}?" where
the bridge person is never named:
    hop 1: person --relation-->  bridge person
    hop 2: bridge person --attribute--> value

Populations (the crux; disjoint with closed relational edges, Karmim 2606.09338):
    P_comp : bridge edges point only inside P_comp; two-hop docs appear in
             training for these people.
    P_held : bridge edges point only inside P_held; these people appear ONLY as
             atomic single-hop facts, never in any two-hop training context.

Renderings, per the repo convention (see corpusgen/records.lookup_segments):
    dense : natural prose, every token graded -> must MEMORISE the composition.
    split : each fact's first citation is a loss-masked organizer lookup. Hop 2's
            query "{bridge}, {attr}" contains the bridge NAME, which the model
            only saw as hop 1's *retrieved* (masked) value -- so the split model
            must learn to COPY hop 1's answer into hop 2's query. That copy-chain
            is the capability under test.

Everything here is deterministic in (entity_id, relation, exposure) so builds are
reproducible and prefix-stable, matching corpusgen.bios.
"""

from __future__ import annotations

import random

from corpusgen.bios import RELATION_PHRASES
from corpusgen.records import (
    BioRecord,
    Doc,
    QAItem,
    Segment,
    lookup_segments,
)

# Functional bridge relations (person -> exactly one other person).
BRIDGE_RELATIONS: tuple[str, ...] = ("mentor", "advisor")
BRIDGE_PHRASES: dict[str, str] = {"mentor": "mentor", "advisor": "advisor"}

# Attributes used as hop-2 targets. birth_date is excluded: its long, punctuated
# value is noisy under exact-match and awkward to copy verbatim.
COMPOSE_ATTRS: tuple[str, ...] = (
    "birth_city",
    "university",
    "major",
    "employer",
    "current_city",
)


# --------------------------------------------------------------- bridge edges


def assign_bridges(records: list[BioRecord], seed: int) -> dict[int, dict[str, int]]:
    """Assign each record a functional bridge target *within this list*.

    Returns {entity_id: {relation: target_entity_id}}. The relation is a
    function (one target per subject); targets need not be unique across
    subjects. Never self-referential. Deterministic in (seed, relation).
    Requires len(records) >= 2.
    """
    if len(records) < 2:
        raise ValueError("assign_bridges needs at least 2 records per population")
    ids = [r.entity_id for r in records]
    edges: dict[int, dict[str, int]] = {i: {} for i in ids}
    for relation in BRIDGE_RELATIONS:
        rng = random.Random(f"bridge:{seed}:{relation}")
        for i in ids:
            tgt = rng.choice(ids)
            while tgt == i:
                tgt = rng.choice(ids)
            edges[i][relation] = tgt
    return edges


# --------------------------------------------------------------- segment utils


def _merge(segs: list[Segment]) -> list[Segment]:
    """Coalesce adjacent unmasked segments into one.

    Masked spans stay separate so a mask boundary always lands on a token
    boundary; consecutive plain text is joined so it tokenises naturally
    (encode_segments encodes each segment independently)."""
    out: list[Segment] = []
    for text, masked in segs:
        if not text:
            continue
        if not masked and out and out[-1][1] is False:
            out[-1] = (out[-1][0] + text, False)
        else:
            out.append((text, masked))
    return out


# --------------------------------------------------------------- doc renderers

# Bridge single-hop templates. sentence = prefix + BRIDGE_NAME + suffix; prefix
# ends with a space (the masked value " {name}" carries that space in split).
_BRIDGE_TEMPLATES: list[tuple[str, str]] = [
    ("{x}'s {rel} is ", "."),
    ("The {rel} of {x} is ", "."),
    ("{x} has long regarded ", " as a trusted {rel}."),
    ("Records name ", " as the {rel} of {x}."),
    ("{x} counts ", " as a {rel}."),
    ("According to the directory, {x}'s {rel} is ", "."),
    ("Listed beside {x} under \"{rel}\" is ", "."),
    ("The {rel} assigned to {x} is ", "."),
]


def render_bridge_doc(
    x_rec: BioRecord, y_name: str, relation: str, exposure: int
) -> Doc:
    """One single-hop bridge fact: "{X}'s {relation} is {Y}."

    dense: value inline, loss on. split: value wrapped in a loss-masked lookup
    keyed on ``"{x_name}, {relation}"``.
    """
    rng = random.Random(f"bridge_doc:{x_rec.entity_id}:{relation}:{exposure}")
    prefix, suffix = _BRIDGE_TEMPLATES[rng.randrange(len(_BRIDGE_TEMPLATES))]
    prefix = prefix.format(x=x_rec.name, rel=relation)
    suffix = suffix.format(x=x_rec.name, rel=relation)
    dense_text = prefix + y_name + suffix  # prefix ends with a space
    split = _merge(
        [(prefix[:-1], False)]
        + lookup_segments(x_rec.name, relation, y_name)
        + [(suffix, False)]
    )
    return Doc(
        kind="bridge",
        dense_segments=[(dense_text, False)],
        split_segments=split,
        meta={"entity_id": x_rec.entity_id, "relation": relation,
              "exposure": exposure},
    )


def compose_prompt(x_name: str, relation: str, attr: str) -> str:
    """The two-hop question, ending at 'Reasoning:' (a training-doc prefix)."""
    return (
        f"Question: What is the {RELATION_PHRASES[attr]} of "
        f"{x_name}'s {relation}?\nReasoning:"
    )


def render_compose_doc(
    x_rec: BioRecord, relation: str, y_rec: BioRecord, attr: str, exposure: int
) -> Doc:
    """One two-hop composition doc for (x --relation--> y --attr--> value).

    Reasoning states hop 1 (x's relation is y), then hop 2 via "Their {attr}",
    which never re-names y -- so in the split arm y is only available as hop 1's
    retrieved value and must be copied into hop 2's lookup key.
    """
    y_name = y_rec.name
    value = y_rec.attrs[attr]
    attr_phrase = RELATION_PHRASES[attr]
    prompt = compose_prompt(x_rec.name, relation, attr) + " "  # doc: space then CoT
    tail = f". So the answer is {value}.\nAnswer: {value}"

    dense_text = (
        prompt
        + f"{x_rec.name}'s {relation} is {y_name}. "
        + f"Their {attr_phrase} is {value}"
        + tail
    )
    split = _merge(
        [(prompt + f"{x_rec.name}'s {relation} is", False)]
        + lookup_segments(x_rec.name, relation, y_name)
        + [(f". Their {attr_phrase} is", False)]
        + lookup_segments(y_name, attr, value)
        + [(tail, False)]
    )
    return Doc(
        kind="compose",
        dense_segments=[(dense_text, False)],
        split_segments=split,
        meta={
            "x_id": x_rec.entity_id, "y_id": y_rec.entity_id,
            "relation": relation, "attr": attr, "exposure": exposure,
        },
    )


# --------------------------------------------------------------- eval items


def compose_item(
    x_rec: BioRecord, relation: str, y_rec: BioRecord, attr: str, population: str
) -> QAItem:
    """Held-out two-hop QA item (prompt ends at 'Reasoning:')."""
    return QAItem(
        qid=f"comp-{population}-{x_rec.entity_id}-{relation}-{attr}",
        task="compose",
        prompt=compose_prompt(x_rec.name, relation, attr),
        answer=y_rec.attrs[attr],
        meta={
            "population": population,
            "x_id": x_rec.entity_id, "x_name": x_rec.name,
            "y_id": y_rec.entity_id, "y_name": y_rec.name,
            "relation": relation, "attr": attr,
            # raw organizer keys the split arm must emit, per hop
            "hop1_key": f"{x_rec.name}, {relation}",
            "hop2_key": f"{y_rec.name}, {attr}",
            "template": f"comp-{relation}-{attr}",
        },
    )


def bridge_probe(x_rec: BioRecord, y_rec: BioRecord, relation: str,
                 population: str) -> QAItem:
    """Single-hop fact-access probe for the bridge edge (hop 1)."""
    return QAItem(
        qid=f"hop1-{population}-{x_rec.entity_id}-{relation}",
        task="singlehop",
        prompt=f"{x_rec.name}'s {relation} is",
        answer=y_rec.name,
        meta={"population": population, "hop": 1, "relation": relation,
              "entity_id": x_rec.entity_id, "template": f"hop1-{relation}"},
    )


def attr_probe(y_rec: BioRecord, attr: str, population: str) -> QAItem:
    """Single-hop fact-access probe for the attribute (hop 2)."""
    return QAItem(
        qid=f"hop2-{population}-{y_rec.entity_id}-{attr}",
        task="singlehop",
        prompt=f"{y_rec.name}'s {RELATION_PHRASES[attr]} is",
        answer=y_rec.attrs[attr],
        meta={"population": population, "hop": 2, "relation": attr,
              "entity_id": y_rec.entity_id, "template": f"hop2-{attr}"},
    )
