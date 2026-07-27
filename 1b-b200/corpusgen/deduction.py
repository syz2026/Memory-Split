"""RuleTaker-style closed-world Horn deduction over nonsense predicates.

Facts are atoms "X is p." over fictional-sounding person names and invented
adjectives; rules are single-variable Horn clauses "If someone is p (and q),
then they are r." with 1-2 antecedents. YES items derive the queried atom in
exactly `depth` forward-chaining steps (verified); NO items are verified
underivable while the query predicate still appears in some rule, so "no" is
never lexically detectable. `forward_chain` is the oracle: generation
verifies against it and the tests call it independently.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .records import Doc, QAItem, plain

NAMES = (
    "Milo", "Zorin", "Kelvra", "Tobbin", "Sarnel", "Quilla", "Dremmet",
    "Fenlow", "Garrick", "Hespera", "Ilvan", "Jorvik", "Kastrel", "Lumina",
    "Marnix", "Nerissa", "Oberin", "Pellam", "Quorra", "Ravik", "Sylvane",
    "Tamsin", "Ulric", "Vespera", "Wrenna", "Xandor", "Yorvin", "Zephrine",
    "Aldric", "Brynna", "Corvin", "Delphia",
)

PREDICATES = (
    "blimpy", "quorful", "snarly", "glarpy", "zibbly", "dorvish", "plinky",
    "wubbly", "crontish", "flarny", "gribbly", "snorpish", "quaddly",
    "vexful", "blorty", "crumpish", "dwindly", "fropish", "glumbly",
    "hifflish", "jorply", "klemmish", "lorbly", "morvish", "norply",
    "oblish", "prindly", "quolby", "rasply", "skorbish", "trundly",
    "umbly", "vorpish", "wemblish", "xarply", "yindly", "zorbly",
    "femmish", "gropply", "hurbly", "jasply", "kwirly",
)

Atom = tuple[str, str]  # (person, predicate)
Rule = tuple[tuple[str, ...], str]  # (antecedent predicates, head predicate)

_MAX_TRIES = 500


@dataclass
class DedProblem:
    pid: str
    depth: int
    prompt: str
    cot: str
    answer: str  # "yes" | "no"
    structure_hash: str


def forward_chain(facts: set[Atom], rules: list[Rule]) -> set[Atom]:
    """Fixpoint closure of the fact base under the Horn rules (the oracle)."""
    closure = set(facts)
    changed = True
    while changed:
        changed = False
        people = {person for person, _ in closure}
        for ants, head in rules:
            for person in people:
                atom = (person, head)
                if atom not in closure and all(
                    (person, a) in closure for a in ants
                ):
                    closure.add(atom)
                    changed = True
    return closure


def _closure_level(facts: set[Atom], rules: list[Rule], query: Atom) -> int | None:
    """First parallel forward-chaining level at which query holds (None if never)."""
    cur = set(facts)
    if query in cur:
        return 0
    level = 0
    while True:
        people = {person for person, _ in cur}
        fired = {
            (person, head)
            for ants, head in rules
            for person in people
            if all((person, a) in cur for a in ants)
        }
        nxt = cur | fired
        if nxt == cur:
            return None
        cur = nxt
        level += 1
        if query in cur:
            return level


def _fact_sentence(atom: Atom) -> str:
    return f"{atom[0]} is {atom[1]}."


def _rule_sentence(rule: Rule) -> str:
    ants, head = rule
    if len(ants) == 1:
        return f"If someone is {ants[0]}, then they are {head}."
    return f"If someone is {ants[0]} and {ants[1]}, then they are {head}."


def _build(
    depth: int, rng: random.Random, answer_yes: bool
) -> tuple[Atom, set[Atom], list[Rule], list[Rule], list[Atom]]:
    """One candidate problem: (query, facts, rules, chain_rules, chain_base_facts)."""
    person = rng.choice(NAMES)
    others = rng.sample([n for n in NAMES if n != person], rng.randint(1, 3))

    working = rng.sample(PREDICATES, 2 * depth + 12)
    chain = working[: depth + 1]  # p0 .. p_depth; query predicate is chain[-1]
    helper_pool = working[depth + 1 : 2 * depth + 1]
    extra = working[2 * depth + 1 :]  # distractor-only predicates

    facts: set[Atom] = set()
    chain_base: list[Atom] = []  # base facts the YES derivation consumes
    chain_rules: list[Rule] = []
    for i in range(1, depth + 1):
        if rng.random() < 0.5:
            helper = helper_pool[i - 1]
            ants = [chain[i - 1], helper]
            rng.shuffle(ants)
            chain_rules.append((tuple(ants), chain[i]))
            facts.add((person, helper))
            chain_base.append((person, helper))
        else:
            chain_rules.append(((chain[i - 1],), chain[i]))

    if answer_yes:
        facts.add((person, chain[0]))
        chain_base.insert(0, (person, chain[0]))
    else:
        # Chain rules present, start fact absent: the chain never fires for
        # `person`. With p=0.5 a mirror person derives the query predicate,
        # so answering "no" requires tracking who the facts are about.
        facts.add((person, extra[0]))
        if rng.random() < 0.5:
            mirror = others[0]
            facts.add((mirror, chain[0]))
            for ants, _ in chain_rules:
                for a in ants:
                    if a in helper_pool:
                        facts.add((mirror, a))

    rules: list[Rule] = list(chain_rules)
    unsafe_heads = set(chain) | set(helper_pool)
    # Difficulty floor (2026-07-20 gate-A decision): shallow problems get
    # small rule/fact bases so the chaining schema is learnable before the
    # distraction load grows.
    max_rules = 4 if depth <= 2 else 8
    n_rules = rng.randint(max(3, len(rules)), max(max_rules, len(rules)))
    tries = 0
    while len(rules) < n_rules and tries < 300:
        tries += 1
        n_ants = rng.randint(1, 2)
        ants = tuple(rng.sample(working, n_ants))
        head = rng.choice(extra[1:])
        if head in ants or head in unsafe_heads:
            continue
        rule = (ants, head)
        if rule not in rules:
            rules.append(rule)

    # Lower bound clamped so an overfull candidate (possible at OOD depths
    # 5-6 in the mirror branch) falls through to validation and retries
    # instead of raising in randint.
    max_facts = 6 if depth <= 2 else 10
    n_facts = rng.randint(min(max(4, len(facts)), max_facts), max_facts)
    tries = 0
    while len(facts) < n_facts and tries < 300:
        tries += 1
        who = rng.choice([person] + others)
        if who == person:
            # Never hand the queried person a chain/helper predicate for free.
            pred = rng.choice(extra)
        else:
            pred = rng.choice(working)
        facts.add((who, pred))

    query = (person, chain[-1])
    return query, facts, rules, chain_rules, chain_base


def generate_problem(depth: int, rng: random.Random, answer_yes: bool) -> DedProblem:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    for _ in range(_MAX_TRIES):
        query, facts, rules, chain_rules, chain_base = _build(depth, rng, answer_yes)
        max_facts = 6 if depth <= 2 else 10
        max_rules = 4 if depth <= 2 else 8
        if not (4 <= len(facts) <= max_facts and 3 <= len(rules) <= max_rules):
            continue
        closure = forward_chain(facts, rules)
        if answer_yes:
            if query not in closure:
                continue
            if _closure_level(facts, rules, query) != depth:
                continue
            without_last = [r for r in rules if r != chain_rules[-1]]
            if query in forward_chain(facts, without_last):
                continue  # a distractor shortcuts the chain
        else:
            if query in closure:
                continue
            rule_preds = {h for _, h in rules} | {a for ants, _ in rules for a in ants}
            if query[1] not in rule_preds:
                continue

        person, target = query
        statements = [_fact_sentence(f) for f in sorted(facts)] + [
            _rule_sentence(r) for r in rules
        ]
        rng.shuffle(statements)
        question = f"Question: Is {person} {target}?"
        prompt = " ".join(statements) + f"\n{question}\nReasoning:"

        if answer_yes:
            answer = "yes"
            stated: set[Atom] = set()
            base = set(chain_base)
            parts: list[str] = []
            for ants, head in chain_rules:
                for a in ants:
                    atom = (person, a)
                    if atom in base and atom not in stated:
                        parts.append(_fact_sentence(atom))
                        stated.add(atom)
                parts.append(_rule_sentence((ants, head)))
                parts.append(f"So {person} is {head}.")
            cot = " ".join(parts) + "\nAnswer: yes"
        else:
            answer = "no"
            cot = (
                f"No chain of rules concludes that {person} is {target}."
                "\nAnswer: no"
            )

        canon = "\n".join(sorted(statements)) + "\n" + question
        structure_hash = hashlib.sha1(canon.encode("utf-8")).hexdigest()
        return DedProblem(
            pid=f"ded-{structure_hash[:12]}",
            depth=depth,
            prompt=prompt,
            cot=cot,
            answer=answer,
            structure_hash=structure_hash,
        )
    raise RuntimeError(
        f"could not build a valid deduction problem (depth={depth}, yes={answer_yes})"
    )


def _problem_stream(rng: random.Random, depth_lo: int, depth_hi: int):
    attempt = 0
    while True:
        depth = rng.randint(depth_lo, depth_hi)
        yield generate_problem(depth, rng, answer_yes=(attempt % 2 == 0))
        attempt += 1


def generate_deduction_docs(
    n_docs: int, depth_lo: int, depth_hi: int, seed: int
) -> list[Doc]:
    rng = random.Random(seed)
    seen: set[str] = set()
    docs: list[Doc] = []
    if n_docs <= 0:
        return docs
    for p in _problem_stream(rng, depth_lo, depth_hi):
        if p.structure_hash in seen:
            continue
        seen.add(p.structure_hash)
        full_text = p.prompt + " " + p.cot
        docs.append(
            Doc(
                kind="deduction",
                dense_segments=[plain(full_text)],
                split_segments=[plain(full_text)],
                meta={"depth": p.depth, "structure_hash": p.structure_hash},
            )
        )
        if len(docs) == n_docs:
            return docs
    return docs


def generate_deduction_eval(
    n_items: int, depth_lo: int, depth_hi: int, seed: int, exclude: set[str]
) -> list[QAItem]:
    rng = random.Random(seed)
    seen: set[str] = set()
    items: list[QAItem] = []
    if n_items <= 0:
        return items
    for p in _problem_stream(rng, depth_lo, depth_hi):
        if p.structure_hash in exclude or p.structure_hash in seen:
            continue
        seen.add(p.structure_hash)
        items.append(
            QAItem(
                qid=f"ded-{len(items)}",
                task="deduction",
                prompt=p.prompt,
                answer=p.answer,
                meta={
                    "depth": p.depth,
                    "structure_hash": p.structure_hash,
                    "template": f"ded-d{p.depth}",
                },
            )
        )
        if len(items) == n_items:
            return items
    return items
