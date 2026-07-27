"""Tests for corpusgen.deduction (written first, per plan Task 3).

Prompt parsing and the minimal-depth computation here are test-local so they
cannot share bugs with the generator's own verification code.
"""

import random
import re

from corpusgen.deduction import (
    NAMES,
    PREDICATES,
    DedProblem,
    forward_chain,
    generate_deduction_docs,
    generate_deduction_eval,
    generate_problem,
)

_FACT = re.compile(r"([A-Z][a-z]+) is ([a-z]+)\.")
_RULE = re.compile(r"If someone is ([a-z]+)(?: and ([a-z]+))?, then they are ([a-z]+)\.")
_Q = re.compile(r"Question: Is ([A-Z][a-z]+) ([a-z]+)\?")


def _parse(prompt: str):
    head = prompt.split("\nQuestion:")[0]
    facts: set[tuple[str, str]] = set()
    rules: list[tuple[tuple[str, ...], str]] = []
    for sent in re.split(r"(?<=\.)\s+", head.strip()):
        sent = sent.strip()
        if not sent:
            continue
        m = _RULE.fullmatch(sent)
        if m:
            ants = (m.group(1),) if m.group(2) is None else (m.group(1), m.group(2))
            rules.append((ants, m.group(3)))
            continue
        m = _FACT.fullmatch(sent)
        assert m is not None, f"unparseable statement: {sent!r}"
        facts.add((m.group(1), m.group(2)))
    qm = _Q.search(prompt)
    assert qm is not None
    return facts, rules, (qm.group(1), qm.group(2))


def _min_depth(facts, rules, query):
    """Test-local iterative closure levels: first level at which query holds."""
    cur = set(facts)
    if query in cur:
        return 0
    for step in range(1, 64):
        people = {p for p, _ in cur}
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
        if query in cur:
            return step
    return None


def test_pools_large_and_unique():
    assert len(set(NAMES)) == len(NAMES) >= 30
    assert len(set(PREDICATES)) == len(PREDICATES) >= 40
    assert all(re.fullmatch(r"[A-Z][a-z]+", n) for n in NAMES)
    assert all(re.fullmatch(r"[a-z]+", p) for p in PREDICATES)


def test_400_problems_oracle_depth_and_counts():
    rng = random.Random(5)
    for i in range(400):
        depth = 1 + (i % 4)
        want_yes = (i % 8) < 4  # every (depth, yes/no) combination is exercised
        p = generate_problem(depth, rng, answer_yes=want_yes)
        assert isinstance(p, DedProblem)
        assert p.depth == depth
        assert p.answer == ("yes" if want_yes else "no")
        assert p.prompt.endswith("Reasoning:")
        assert p.cot.endswith(f"\nAnswer: {p.answer}")

        facts, rules, query = _parse(p.prompt)
        assert 4 <= len(facts) <= 10
        assert 3 <= len(rules) <= 8
        for person, pred in facts:
            assert person in NAMES and pred in PREDICATES
        for ants, head in rules:
            assert head in PREDICATES and all(a in PREDICATES for a in ants)

        derived = query in forward_chain(facts, rules)
        assert derived == want_yes
        if want_yes:
            assert _min_depth(facts, rules, query) == depth
        else:
            rule_preds = {head for _, head in rules} | {
                a for ants, _ in rules for a in ants
            }
            assert query[1] in rule_preds  # "no" is not lexically detectable


def test_problem_determinism():
    a = generate_problem(3, random.Random(9), answer_yes=True)
    b = generate_problem(3, random.Random(9), answer_yes=True)
    assert a == b
    c = generate_problem(3, random.Random(10), answer_yes=True)
    assert a != c


def test_docs_shape_balance_dedupe_determinism():
    docs = generate_deduction_docs(400, 1, 4, seed=3)
    assert len(docs) == 400
    hashes = [d.meta["structure_hash"] for d in docs]
    assert len(set(hashes)) == len(hashes)
    n_yes = 0
    for d in docs:
        assert d.kind == "deduction"
        text = d.dense_text()
        assert d.dense_segments == d.split_segments == [(text, False)]
        assert 1 <= d.meta["depth"] <= 4
        assert "\nQuestion: Is " in text
        assert "\nReasoning: " in text
        if text.endswith("\nAnswer: yes"):
            n_yes += 1
        else:
            assert text.endswith("\nAnswer: no")
            assert "No chain of rules concludes that " in text
    assert 0.45 <= n_yes / 400 <= 0.55
    again = generate_deduction_docs(400, 1, 4, seed=3)
    assert [d.dense_text() for d in again] == [d.dense_text() for d in docs]
    assert [d.meta for d in again] == [d.meta for d in docs]
    other = generate_deduction_docs(400, 1, 4, seed=4)
    assert [d.dense_text() for d in other] != [d.dense_text() for d in docs]


def test_eval_disjoint_from_train_and_fields():
    docs = generate_deduction_docs(80, 1, 4, seed=11)
    train_hashes = {d.meta["structure_hash"] for d in docs}
    # Same seed regenerates the training problems first, so the exclusion
    # path is exercised for real.
    items = generate_deduction_eval(60, 1, 4, seed=11, exclude=train_hashes)
    assert len(items) == 60
    seen = set()
    for i, it in enumerate(items):
        assert it.qid == f"ded-{i}"
        assert it.task == "deduction"
        assert it.answer in ("yes", "no")
        assert it.prompt.endswith("Reasoning:")
        depth = it.meta["depth"]
        assert 1 <= depth <= 4
        assert it.meta["template"] == f"ded-d{depth}"
        h = it.meta["structure_hash"]
        assert h not in train_hashes
        assert h not in seen
        seen.add(h)
        facts, rules, query = _parse(it.prompt)
        assert (query in forward_chain(facts, rules)) == (it.answer == "yes")
    again = generate_deduction_eval(60, 1, 4, seed=11, exclude=train_hashes)
    assert [it.prompt for it in again] == [it.prompt for it in items]


def test_forward_chain_handwritten():
    facts = {("Milo", "blimpy"), ("Milo", "snarly"), ("Vex", "blimpy")}
    rules = [
        (("blimpy", "snarly"), "quorful"),
        (("quorful",), "glarpy"),
    ]
    closure = forward_chain(facts, rules)
    assert ("Milo", "quorful") in closure
    assert ("Milo", "glarpy") in closure
    assert ("Vex", "quorful") not in closure  # Vex lacks snarly
    assert facts <= closure
