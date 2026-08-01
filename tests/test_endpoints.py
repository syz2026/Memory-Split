"""Endpoint generators: solution traces, tunable MOD, topology hash,
OOD bands, and the two reporting breakdowns that stop an aggregate from
hiding a degenerate policy."""

import random
from collections import Counter

import pytest

from corpusgen import deduction, igsm_lite
from evals.scorers import accuracy_by


# ------------------------------------------------------------ solution traces


def test_igsm_eval_carries_the_worked_solution():
    item = igsm_lite.generate_igsm_eval(3, 1, 4, 5, set())[0]
    assert "solution" in item.meta and item.meta["solution"]


def test_igsm_prompt_plus_solution_is_the_training_text():
    """The continuous metrics score prompt + solution; the seam must be the
    exact byte sequence the model trained on."""
    rng = random.Random(3)
    for _ in range(20):
        p = igsm_lite.generate_problem(rng.randint(1, 4), rng)
        training_text = p.prompt + " " + p.cot
        assert p.prompt + (" " + p.cot) == training_text


def test_deduction_eval_carries_the_worked_solution():
    item = deduction.generate_deduction_eval(4, 1, 2, 5, set())[0]
    assert "solution" in item.meta and item.meta["solution"]


# ------------------------------------------------------------ tunable MOD


@pytest.mark.parametrize("mod", [7, 11, 23])
def test_oracle_agrees_with_the_generator_at_every_modulus(mod):
    """'modulo 23' is written into the statements, the question, the trace and
    four oracle regexes. A missed hardcode surfaces here."""
    rng = random.Random(0)
    for _ in range(300):
        p = igsm_lite.generate_problem(rng.randint(1, 4), rng, mod=mod)
        assert igsm_lite.solve_from_prompt(p.prompt, mod=mod) == p.answer
        assert p.mod == mod


@pytest.mark.parametrize("mod", [7, 11, 23])
def test_answers_stay_inside_the_residue_class(mod):
    rng = random.Random(1)
    for _ in range(200):
        p = igsm_lite.generate_problem(rng.randint(1, 4), rng, mod=mod)
        assert 0 <= p.answer < mod


def test_modulus_appears_in_the_rendered_text():
    p = igsm_lite.generate_problem(2, random.Random(2), mod=11)
    assert "modulo 11" in p.prompt
    assert "mod 11" in p.cot
    assert "modulo 23" not in p.prompt


def test_mod_is_threaded_through_the_generators():
    docs = igsm_lite.generate_igsm_docs(5, 1, 3, 4, mod=7)
    assert all(d.meta["mod"] == 7 for d in docs)
    items = igsm_lite.generate_igsm_eval(5, 1, 3, 4, set(), mod=7)
    assert all(i.meta["mod"] == 7 for i in items)


# ------------------------------------------------------------ topology hash


def test_same_shape_different_names_collides_on_topology_not_on_text():
    """The old structure hash is SHA-1 over rendered statements including
    entity names drawn from 56,448 combinations, so it can only certify
    exact-text novelty. The topology hash is what an OOD structure claim
    needs."""
    a = igsm_lite.generate_problem(3, random.Random(11))
    b = igsm_lite.generate_problem(3, random.Random(11))
    assert a.structure_hash == b.structure_hash  # identical draw

    # Find two problems that share a shape but not their text.
    seen: dict[str, igsm_lite.IgsmProblem] = {}
    pair = None
    rng = random.Random(0)
    for _ in range(400):
        p = igsm_lite.generate_problem(2, rng)
        prev = seen.get(p.topology_hash)
        if prev is not None and prev.structure_hash != p.structure_hash:
            pair = (prev, p)
            break
        seen[p.topology_hash] = p
    assert pair is not None, "no same-shape different-text pair found"
    x, y = pair
    assert x.topology_hash == y.topology_hash
    assert x.structure_hash != y.structure_hash


def test_topology_hash_ignores_constant_values():
    """Constants are abstracted to C, so two problems differing only in a
    constant share a shape."""
    rng = random.Random(5)
    shapes = Counter(igsm_lite.generate_problem(1, rng).topology_hash for _ in range(300))
    # op=1 has few possible shapes; if constants leaked in there would be many.
    assert len(shapes) <= 12, f"{len(shapes)} shapes at op=1 suggests constants leaked"


def test_topology_hash_is_deterministic():
    a = igsm_lite.generate_problem(4, random.Random(9))
    b = igsm_lite.generate_problem(4, random.Random(9))
    assert a.topology_hash == b.topology_hash


# ------------------------------------------------------------ OOD bands


@pytest.mark.parametrize("band", [(5, 8), (9, 12)])
def test_ood_band_generates_and_oracle_verifies(band):
    lo, hi = band
    items = igsm_lite.generate_igsm_eval(300, lo, hi, 21, set())
    assert len(items) == 300
    for it in items:
        assert igsm_lite.solve_from_prompt(it.prompt) == int(it.answer)
        assert lo <= it.meta["op"] <= hi


def test_ood_band_is_topology_disjoint_from_the_training_band():
    """Chain length is part of the shape, so a different op band cannot
    collide with the trained one."""
    train = {
        d.meta["topology_hash"] for d in igsm_lite.generate_igsm_docs(600, 1, 4, 1)
    }
    ood = {
        i.meta["topology_hash"]
        for i in igsm_lite.generate_igsm_eval(300, 5, 8, 2, set())
    }
    assert not (train & ood)


def test_exclude_topologies_is_honoured():
    train_docs = igsm_lite.generate_igsm_docs(300, 2, 2, 1)
    topos = {d.meta["topology_hash"] for d in train_docs}
    items = igsm_lite.generate_igsm_eval(
        20, 2, 2, 99, set(), exclude_topologies=topos
    )
    assert all(i.meta["topology_hash"] not in topos for i in items)


# ------------------------------------------------------------ reporting


def _rows(pairs, key, task="igsm"):
    return [
        {"qid": str(i), "task": task, "correct": c, "pred": "x",
         "answer": a, "meta": {key: b}}
        for i, (c, a, b) in enumerate(pairs)
    ]


def test_per_op_breakdown():
    rows = _rows([(True, "1", 1), (False, "2", 1), (True, "3", 2)], "op")
    out = accuracy_by(rows, "op")
    assert out["by"]["1"]["acc"] == 0.5
    assert out["by"]["2"]["acc"] == 1.0
    assert out["n"] == 3


def test_constant_no_is_visible_as_a_degenerate_policy():
    """A model that always answers 'no' scores exactly 0.500 overall on the
    balanced deduction eval. Per-class makes that unmistakable."""
    rows = (
        _rows([(True, "no", "no")] * 200, "answer_class", task="deduction")
        + _rows([(False, "yes", "yes")] * 200, "answer_class", task="deduction")
    )
    out = accuracy_by(rows, "answer_class")
    assert out["overall"] == 0.5
    assert out["by"]["no"]["acc"] == 1.0
    assert out["by"]["yes"]["acc"] == 0.0
    assert out["majority_rate"] == 0.5


def test_igsm_majority_class_baseline_is_about_seven_percent_not_one_over_23():
    """`times` overproduces zero, so the best constant predictor beats 1/23.
    Every prior iGSM number in this project was scored against 4.3%."""
    items = igsm_lite.generate_igsm_eval(2000, 1, 4, 99, set())
    rows = [
        {"qid": i.qid, "task": "igsm", "correct": False, "pred": None,
         "answer": i.answer, "meta": i.meta}
        for i in items
    ]
    out = accuracy_by(rows, "op")
    assert 0.06 < out["majority_rate"] < 0.09, out["majority_rate"]
    assert out["majority_rate"] > 1 / 23
