"""Tests for corpusgen.igsm_lite (written first, per plan Task 3).

All parsing here is test-local (regex on prompt text) so it cannot share bugs
with the generator or with corpusgen.igsm_lite.solve_from_prompt.
"""

import random
import re
from collections import Counter

from corpusgen.igsm_lite import (
    ADJECTIVES,
    NOUNS,
    PLACES,
    IgsmProblem,
    generate_igsm_docs,
    generate_igsm_eval,
    generate_problem,
    solve_from_prompt,
)

_LEAF = re.compile(r"The number of (.+?) is (\d+)\.")
_QQ = re.compile(
    r"The number of (.+?) equals the number of (.+?) (plus|minus|times) "
    r"the number of (.+?), modulo 23\."
)
_QC = re.compile(
    r"The number of (.+?) equals the number of (.+?) (plus|minus|times) (\d+), modulo 23\."
)


def _statements(prompt: str) -> list[str]:
    head = prompt.split("\nQuestion:")[0]
    return [s.strip() for s in re.split(r"(?<=\.)\s+", head.strip()) if s.strip()]


def _defined_names(prompt: str) -> list[str]:
    names = []
    for sent in _statements(prompt):
        m = _QQ.fullmatch(sent) or _QC.fullmatch(sent) or _LEAF.fullmatch(sent)
        assert m is not None, f"unparseable statement: {sent!r}"
        names.append(m.group(1))
    return names


def test_pools_large_and_unique():
    assert len(set(ADJECTIVES)) == len(ADJECTIVES) >= 40
    assert len(set(NOUNS)) == len(NOUNS) >= 40
    assert len(set(PLACES)) == len(PLACES) >= 30


def test_solver_semantics_handwritten():
    # Pins the statement grammar and the mod-23 semantics (negative and
    # product intermediates) independently of the generator.
    prompt = (
        "The number of amber kites in the Loft is 5. "
        "The number of crimson satchels in the Annex equals the number of "
        "amber kites in the Loft minus 9, modulo 23. "
        "The number of dusty flasks in the Depot equals the number of "
        "crimson satchels in the Annex times the number of amber kites in the Loft, modulo 23."
        "\nQuestion: What is the number of dusty flasks in the Depot, modulo 23?"
        "\nReasoning:"
    )
    # 5 - 9 = -4 -> 19 (mod 23); 19 * 5 = 95 -> 3 (mod 23)
    assert solve_from_prompt(prompt) == 3


def test_oracle_matches_answer_on_500_random_problems():
    rng = random.Random(1234)
    op_counts = Counter()
    for _ in range(500):
        op = rng.randint(2, 8)
        p = generate_problem(op, rng)
        assert isinstance(p, IgsmProblem)
        op_counts[op] += 1
        assert p.op == op
        assert 0 <= p.answer <= 22
        assert p.prompt.endswith("Reasoning:")
        assert solve_from_prompt(p.prompt) == p.answer
        m = re.search(r"\nAnswer: (\d+)$", p.cot)
        assert m is not None and int(m.group(1)) == p.answer
    assert set(op_counts) == set(range(2, 9))


def test_problem_determinism():
    a = generate_problem(5, random.Random(77))
    b = generate_problem(5, random.Random(77))
    assert a == b
    c = generate_problem(5, random.Random(78))
    assert a != c


def test_docs_shape_dedupe_and_determinism():
    docs = generate_igsm_docs(120, 2, 8, seed=9)
    assert len(docs) == 120
    hashes = [d.meta["structure_hash"] for d in docs]
    assert len(set(hashes)) == len(hashes)
    assert {d.meta["op"] for d in docs} == set(range(2, 9))
    for d in docs:
        assert d.kind == "igsm"
        text = d.dense_text()
        assert d.dense_segments == d.split_segments == [(text, False)]
        assert "\nQuestion: What is the number of " in text
        assert "\nReasoning: The number of " in text  # prompt/cot seam
        assert re.search(r"\nAnswer: \d+$", text)
    again = generate_igsm_docs(120, 2, 8, seed=9)
    assert [d.dense_text() for d in again] == [d.dense_text() for d in docs]
    assert [d.meta for d in again] == [d.meta for d in docs]
    other = generate_igsm_docs(120, 2, 8, seed=10)
    assert [d.dense_text() for d in other] != [d.dense_text() for d in docs]


def test_eval_disjoint_from_train_and_fields():
    docs = generate_igsm_docs(80, 2, 6, seed=21)
    train_hashes = {d.meta["structure_hash"] for d in docs}
    # Same seed as the docs: the eval stream regenerates the very same
    # problems first, so the exclusion path is exercised for real.
    items = generate_igsm_eval(60, 2, 6, seed=21, exclude=train_hashes)
    assert len(items) == 60
    seen = set()
    for i, it in enumerate(items):
        assert it.qid == f"igsm-{i}"
        assert it.task == "igsm"
        assert it.prompt.endswith("Reasoning:")
        assert it.answer == str(solve_from_prompt(it.prompt))
        assert 0 <= int(it.answer) <= 22
        op = it.meta["op"]
        assert 2 <= op <= 6
        assert it.meta["template"] == f"igsm-op{op}"
        h = it.meta["structure_hash"]
        assert h not in train_hashes
        assert h not in seen
        seen.add(h)
    again = generate_igsm_eval(60, 2, 6, seed=21, exclude=train_hashes)
    assert [it.prompt for it in again] == [it.prompt for it in items]


def test_some_problems_contain_distractors():
    rng = random.Random(4242)
    with_distractor = 0
    for _ in range(50):
        p = generate_problem(4, rng)
        names = _defined_names(p.prompt)
        assert len(set(names)) == len(names)  # quantity names unique in-problem
        query = re.search(r"What is the number of (.+?), modulo 23\?", p.prompt).group(1)
        assert query in names
        assert f"The number of {query} is (" in p.cot  # query gets a compute sentence
        # A needed node appears in the CoT as exactly one "The number of {n} is"
        # sentence; a distractor definition appears nowhere in the CoT.
        if any(f"The number of {n} is " not in p.cot for n in names):
            with_distractor += 1
    assert with_distractor >= 5
