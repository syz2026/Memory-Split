"""Tests for scripts.run_keyguess_local (T4) — pure helpers only.

No training, no network, fast. Imports the orchestrator's pure helpers and
exercises them on tiny synthetic fixtures:
  1. corpus assembly writes aligned bin/mask files with a sane masked
     fraction and is byte-identical across two builds;
  2. arm_plan routes A/B -> runs/a, C/D -> runs/c, with B/D constrained;
  3. eval items round-trip through JSONL into QAItem with meta intact.
"""

import numpy as np
import pytest

from corpusgen import bios, realfact
from corpusgen.records import QAItem
from scripts.run_keyguess_local import (
    TRAINER_BASE_SEED,
    _trainer_cfg,
    accumulate_stats,
    arm_plan,
    assemble_corpus,
    assert_split_contract,
    check_emittability,
    corpus_complete,
    load_eval_items,
    relation_counts,
    require_corpus_complete,
    seed_suffix,
    write_eval_items,
)
from train.tokenizer import get_tok

TOK = get_tok()


def _facts():
    return [
        realfact.RealFact(
            subj=f"Entity {i}", prop="author", obj=f"obj-{i}",
            question=f"What is the author of Entity {i}?",
            possible_answers=(f"obj-{i}",),
        )
        for i in range(3)
    ]


def _records():
    return bios.generate_records(2, seed=7)


# ----------------------------------------------------------------- arm_plan


def test_arm_plan_routes_runs_and_constraints():
    plan = arm_plan(["A", "B", "C", "D"])
    assert [p["arm"] for p in plan] == ["A", "B", "C", "D"]
    for p in plan:
        assert p["run"] in ("a", "c")
        assert p["corpus"] == p["run"]
    # A/B share runs/a; C/D share runs/c
    assert plan[0]["run"] == "a" and plan[1]["run"] == "a"
    assert plan[2]["run"] == "c" and plan[3]["run"] == "c"
    # constrained flags B, D only
    assert [p["constrained"] for p in plan] == [False, True, False, True]


def test_arm_plan_subset_preserves_routing():
    plan = arm_plan(["B", "D"])
    assert [p["arm"] for p in plan] == ["B", "D"]
    assert [p["run"] for p in plan] == ["a", "c"]
    assert [p["constrained"] for p in plan] == [True, True]


def test_arm_plan_rejects_unknown_arm():
    import pytest
    with pytest.raises(ValueError):
        arm_plan(["E"])


# --------------------------------------------------------- corpus assembly


def test_assemble_corpus_writes_aligned_bin_mask_deterministic(tmp_path):
    facts = _facts()
    records = _records()
    out_a = tmp_path / "corpus_a"
    out_b = tmp_path / "corpus_b"

    rep = assemble_corpus(
        facts, records, TOK, out_a,
        n_exposures=1, seed=0, substitution_frac=0.0, fresh_flood=0,
        n_factqa_docs=0, factqa_seed=7,
    )
    assert rep["n_docs"] == len(records) * 6 + 0 + len(facts) * 1

    bin_path = out_a / "train.bin"
    mask_path = out_a / "train.mask.bin"
    assert bin_path.exists() and mask_path.exists()
    ids = np.fromfile(bin_path, dtype=np.uint16)
    mask = np.fromfile(mask_path, dtype=np.uint8)
    assert len(ids) == len(mask) == rep["n_tokens"]
    assert set(np.unique(mask)) <= {0, 1}
    frac0 = float((mask == 0).mean())
    assert 0.0 < frac0 < 0.5
    assert rep["masked_token_frac"] == pytest.approx(frac0)

    # Second build with identical args is byte-identical.
    assemble_corpus(
        facts, records, TOK, out_b,
        n_exposures=1, seed=0, substitution_frac=0.0, fresh_flood=0,
        n_factqa_docs=0, factqa_seed=7,
    )
    ids2 = np.fromfile(out_b / "train.bin", dtype=np.uint16)
    mask2 = np.fromfile(out_b / "train.mask.bin", dtype=np.uint8)
    assert np.array_equal(ids, ids2)
    assert np.array_equal(mask, mask2)


def test_assemble_corpus_substitution_and_flood_change_docs(tmp_path):
    facts = _facts()
    records = _records()
    plain = assemble_corpus(
        facts, records, TOK, tmp_path / "plain",
        n_exposures=1, seed=0, substitution_frac=0.0, fresh_flood=0,
        n_factqa_docs=0, factqa_seed=7,
    )
    flooded = assemble_corpus(
        facts, records, TOK, tmp_path / "flooded",
        n_exposures=1, seed=0, substitution_frac=0.5, fresh_flood=5,
        n_factqa_docs=0, factqa_seed=7,
    )
    assert flooded["n_docs"] == plain["n_docs"] + 5
    assert flooded["n_tokens"] != plain["n_tokens"]


# ------------------------------------------------------- eval-items round-trip


def test_eval_items_roundtrip(tmp_path):
    facts = _facts()
    items = realfact.realfact_eval_items(facts, "heldout")
    assert len(items) == len(facts)
    path = tmp_path / "eval_items.jsonl"
    write_eval_items(items, path)
    reloaded = load_eval_items(path)
    assert len(reloaded) == len(items)
    for orig, got in zip(items, reloaded):
        assert isinstance(got, QAItem)
        assert got.qid == orig.qid
        assert got.task == orig.task
        assert got.prompt == orig.prompt
        assert got.answer == orig.answer
        assert got.meta == orig.meta
        # spot-check the meta fields the scorer relies on
        assert got.meta["subj"] == orig.meta["subj"]
        assert got.meta["prop"] == orig.meta["prop"]
        assert got.meta["split"] == "heldout"


# --------------------------------------------------------- emittability gate


def test_check_emittability_passes_for_well_formed_prompts():
    facts = _facts()
    items = realfact.realfact_eval_items(facts, "heldout")
    coverage, misses = check_emittability(items)
    assert coverage == 1.0
    assert misses == []


def test_check_emittability_records_misses_when_subj_absent():
    # Prompt that never contains the subject span -> gold key unemittable.
    item = QAItem(
        qid="rf-x-0", task="realfact",
        prompt="Question: Who is the author of Someone Else?\nReasoning:",
        answer="obj",
        meta={"subj": "Hidden Subject", "prop": "author", "obj": "obj",
              "possible_answers": ["obj"], "split": "heldout"},
    )
    coverage, misses = check_emittability([item])
    assert coverage == 0.0
    assert len(misses) == 1
    assert misses[0]["subj"] == "Hidden Subject"
    assert misses[0]["question"] == "Who is the author of Someone Else?"


# -------------------------------------------------- FIX 1: split contract


def _multi_relation_facts():
    facts = []
    for i in range(5):
        facts.append(realfact.RealFact(
            subj=f"Author Entity {i}", prop="author", obj=f"a-obj-{i}",
            question=f"What is the author of Author Entity {i}?",
            possible_answers=(f"a-obj-{i}",),
        ))
    for i in range(10):
        facts.append(realfact.RealFact(
            subj=f"Director Entity {i}", prop="director", obj=f"d-obj-{i}",
            question=f"What is the director of Director Entity {i}?",
            possible_answers=(f"d-obj-{i}",),
        ))
    return facts


def test_split_contract_holds_on_tiny_fixture():
    facts = _multi_relation_facts()
    seen, heldout = realfact.split_by_relation(facts, 0.8, seed=0)
    # structural rules, not absolute counts: partition + floor rule + eval count
    assert len(seen) + len(heldout) == len(facts)
    for prop, c in relation_counts(seen, heldout).items():
        total = c["seen"] + c["heldout"]
        assert c["seen"] == int(0.8 * total)
    eval_items = (realfact.realfact_eval_items(heldout, "heldout")
                 + realfact.realfact_eval_items(seen[:200], "seen"))
    assert_split_contract(seen, heldout, eval_items, n_total=len(facts))
    assert len(eval_items) == len(heldout) + min(200, len(seen))


def test_split_contract_rejects_wrong_total():
    facts = _multi_relation_facts()
    seen, heldout = realfact.split_by_relation(facts, 0.8, seed=0)
    eval_items = (realfact.realfact_eval_items(heldout, "heldout")
                 + realfact.realfact_eval_items(seen[:200], "seen"))
    with pytest.raises(AssertionError):
        assert_split_contract(seen, heldout, eval_items, n_total=len(facts) + 1)


def test_split_contract_rejects_wrong_eval_count():
    facts = _multi_relation_facts()
    seen, heldout = realfact.split_by_relation(facts, 0.8, seed=0)
    # wrong eval count: only heldout items, missing the seen slice
    eval_items = realfact.realfact_eval_items(heldout, "heldout")
    with pytest.raises(AssertionError):
        assert_split_contract(seen, heldout, eval_items, n_total=len(facts))


# ----------------------------------------- FIX 2: corpus completeness


def test_corpus_complete_bin_only_not_complete(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "train.bin").write_bytes(b"\x00" * 100)
    assert not corpus_complete(d)


def test_corpus_complete_wrong_ratio_not_complete(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "train.bin").write_bytes(b"\x00" * 100)
    (d / "train.mask.bin").write_bytes(b"\x00" * 60)
    assert not corpus_complete(d)


def test_corpus_complete_correct_ratio_is_complete(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "train.bin").write_bytes(b"\x00" * 100)
    (d / "train.mask.bin").write_bytes(b"\x00" * 50)
    assert corpus_complete(d)


def test_require_corpus_complete_raises_on_missing_mask(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "train.bin").write_bytes(b"\x00" * 100)
    with pytest.raises(SystemExit):
        require_corpus_complete(d, stage="train")


# ------------------------------------------------ replication seed bundles


def test_seed_suffix_zero_is_original_paths():
    assert seed_suffix(0) == ""
    assert seed_suffix(3) == "_s3"
    with pytest.raises(ValueError):
        seed_suffix(-1)


def test_trainer_cfg_seed_bundle_offsets_trainer_seed(tmp_path):
    cfg0 = _trainer_cfg("a", tmp_path, tmp_path / "out", 800, "cpu", seed=0)
    cfg2 = _trainer_cfg("a", tmp_path, tmp_path / "out2", 800, "cpu", seed=2)
    assert cfg0["seed"] == TRAINER_BASE_SEED
    assert cfg0["run_id"] == "keyguess_a"
    assert cfg2["seed"] == TRAINER_BASE_SEED + 2
    assert cfg2["run_id"] == "keyguess_a_s2"
    # everything else identical except out_dir
    for key in ("model", "micro_batch_size", "tokens_per_step", "lr",
                "warmup_steps", "max_steps"):
        assert cfg0[key] == cfg2[key]


def test_assemble_corpus_shuffle_seed_changes_stream(tmp_path):
    facts = _facts()
    records = _records()
    kwargs = dict(n_exposures=1, seed=0, substitution_frac=0.0, fresh_flood=0,
                  n_factqa_docs=0, factqa_seed=7)
    rep0 = assemble_corpus(facts, records, TOK, tmp_path / "s0",
                           shuffle_seed=123, **kwargs)
    rep1 = assemble_corpus(facts, records, TOK, tmp_path / "s1",
                           shuffle_seed=124, **kwargs)
    # same content, different order: token counts equal, bytes differ
    assert rep0["n_tokens"] == rep1["n_tokens"]
    ids0 = np.fromfile(tmp_path / "s0" / "train.bin", dtype=np.uint16)
    ids1 = np.fromfile(tmp_path / "s1" / "train.bin", dtype=np.uint16)
    assert not np.array_equal(ids0, ids1)


# ----------------------------------------- FIX 3: stats aggregation


def test_accumulate_stats_aggregates_extra_key():
    acc: dict[str, int] = {}
    batches = [
        {"n_lookups": 2, "n_hits": 1, "n_misses": 1, "n_malformed": 0},
        {"n_lookups": 3, "n_hits": 2, "n_misses": 1, "n_malformed": 0,
         "n_constrained_queries": 3, "n_constraint_dead_ends": 1},
        {"n_lookups": 1, "n_hits": 1, "n_misses": 0, "n_malformed": 0,
         "n_constrained_queries": 1, "n_constraint_dead_ends": 0},
    ]
    for bs in batches:
        accumulate_stats(acc, bs)
    assert acc["n_lookups"] == 6
    assert acc["n_hits"] == 4
    assert acc["n_misses"] == 2
    assert acc["n_malformed"] == 0
    # extra constrained-only keys reach the accumulator even though the
    # first batch did not report them (unseen initialized to 0)
    assert acc["n_constrained_queries"] == 4
    assert acc["n_constraint_dead_ends"] == 1
