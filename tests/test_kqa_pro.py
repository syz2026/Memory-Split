import json

import numpy as np

from corpusgen.kqa_build import KQAContinuationConfig, build_kqa_continuation
from corpusgen.kqa_pro import (
    KQAKnowledgeBase,
    SplitConfig,
    build_transfer_split,
    program_skeleton,
    render_fact_doc,
    render_qa_doc,
    write_prepared_data,
)
from organizer.store import normalize
from train.tokenizer import get_tok


def _value(kind, value, unit=""):
    row = {"type": kind, "value": value}
    if kind == "quantity":
        row["unit"] = unit
    return row


def _mini_kb():
    return {
        "concepts": {
            "C_PERSON": {"name": "person", "instanceOf": []},
            "C_CITY": {"name": "city", "instanceOf": []},
        },
        "entities": {
            "E_ALICE": {
                "name": "Alice",
                "instanceOf": ["C_PERSON"],
                "attributes": [
                    {
                        "key": "age",
                        "value": _value("quantity", 31, "year"),
                        "qualifiers": {},
                    }
                ],
                "relations": [
                    {
                        "predicate": "lives in",
                        "object": "E_PARIS",
                        "direction": "forward",
                        "qualifiers": {
                            "since": [_value("year", 2019)],
                        },
                    },
                    {
                        "predicate": "associated with",
                        "object": "C_CITY",
                        "direction": "forward",
                        "qualifiers": {},
                    },
                ],
            },
            "E_BOB": {
                "name": "Bob",
                "instanceOf": ["C_PERSON"],
                "attributes": [
                    {
                        "key": "age",
                        "value": _value("quantity", 42, "year"),
                        "qualifiers": {},
                    }
                ],
                "relations": [
                    {
                        "predicate": "lives in",
                        "object": "E_ROME",
                        "direction": "forward",
                        "qualifiers": {
                            "since": [_value("year", 2021)],
                        },
                    }
                ],
            },
            "E_PARIS": {
                "name": "Paris",
                "instanceOf": ["C_CITY"],
                "attributes": [],
                "relations": [],
            },
            "E_ROME": {
                "name": "Rome",
                "instanceOf": ["C_CITY"],
                "attributes": [],
                "relations": [],
            },
        },
    }


def _attr_item(name, answer, source):
    return {
        "question": f"What is {name}'s age?",
        "answer": answer,
        "choices": [],
        "program": [
            {"function": "Find", "dependencies": [], "inputs": [name]},
            {"function": "QueryAttr", "dependencies": [0], "inputs": ["age"]},
        ],
        "_source": source,
    }


def _relation_item(name, answer, source):
    return {
        "question": f"Where does {name} live?",
        "answer": answer,
        "choices": [],
        "program": [
            {"function": "Find", "dependencies": [], "inputs": [name]},
            {
                "function": "Relate",
                "dependencies": [0],
                "inputs": ["lives in", "forward"],
            },
            {"function": "What", "dependencies": [1], "inputs": []},
        ],
        "_source": source,
    }


def test_skeleton_strips_inputs_but_keeps_dependencies():
    item = _relation_item("Alice", "Paris", "train")
    assert (
        program_skeleton(item["program"])
        == "Find[]|Relate[0]{direction=forward}|What[1]"
    )
    changed = _relation_item("Bob", "Rome", "val")
    assert program_skeleton(changed["program"]) == program_skeleton(item["program"])


def test_skeleton_retains_control_operators():
    greater = [
        {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
        {
            "function": "FilterYear",
            "dependencies": [0],
            "inputs": ["year", "2000", ">"],
        },
    ]
    less = [
        greater[0],
        {
            "function": "FilterYear",
            "dependencies": [0],
            "inputs": ["year", "2000", "<"],
        },
    ]
    assert program_skeleton(greater) != program_skeleton(less)


def test_executor_returns_answer_and_exact_lookup_support():
    kb = KQAKnowledgeBase(_mini_kb())

    attr = kb.trace(_attr_item("Alice", "31 year", "train"))
    assert attr.answer == "31 year"
    assert attr.support_keys == (normalize("Alice, age"),)

    relation = kb.trace(_relation_item("Bob", "Rome", "val"))
    assert relation.answer == "Rome"
    assert relation.support_keys == (normalize("Bob, lives in (forward)"),)


def test_qualifier_projection_is_addressable():
    kb = KQAKnowledgeBase(_mini_kb())
    item = {
        "question": "Since when has Alice lived in Paris?",
        "answer": "2019",
        "program": [
            {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
            {"function": "Find", "dependencies": [], "inputs": ["Paris"]},
            {
                "function": "QueryRelationQualifier",
                "dependencies": [0, 1],
                "inputs": ["lives in", "since"],
            },
        ],
    }
    trace = kb.trace(item)
    assert trace.answer == "2019"
    assert normalize("Alice, lives in (forward), qualifier since") in trace.support_keys


def test_relate_from_concept_preserves_inverse_relation_support():
    kb = KQAKnowledgeBase(_mini_kb())
    item = {
        "question": "Who is associated with the concept city?",
        "answer": "Alice",
        "program": [
            {"function": "Find", "dependencies": [], "inputs": ["city"]},
            {
                "function": "Relate",
                "dependencies": [0],
                "inputs": ["associated with", "backward"],
            },
            {"function": "What", "dependencies": [1], "inputs": []},
        ],
    }
    trace = kb.trace(item)
    assert trace.answer == "Alice"
    assert trace.support_keys == (
        normalize("city, associated with (backward)"),
    )


def test_entity_inverse_records_share_one_semantic_relation_atom():
    raw = _mini_kb()
    raw["entities"]["E_PARIS"]["relations"] = [
        {
            "predicate": "lives in",
            "object": "E_ALICE",
            "direction": "backward",
            "qualifiers": {
                "since": [_value("year", 2019)],
            },
        }
    ]
    kb = KQAKnowledgeBase(raw)
    backward_item = {
        "question": "Who lives in Paris?",
        "answer": "Alice",
        "program": [
            {"function": "Find", "dependencies": [], "inputs": ["Paris"]},
            {
                "function": "Relate",
                "dependencies": [0],
                "inputs": ["lives in", "backward"],
            },
            {"function": "What", "dependencies": [1], "inputs": []},
        ],
    }

    forward = kb.trace(_relation_item("Alice", "Paris", "train"))
    backward = kb.trace(backward_item)

    assert forward.support_atoms == backward.support_atoms
    assert kb.lookup_facts[
        normalize("Alice, lives in (forward)")
    ].raw_ids == kb.lookup_facts[
        normalize("Paris, lives in (backward)")
    ].raw_ids


def test_transfer_split_has_no_support_leakage_and_matches_skeletons():
    kb = KQAKnowledgeBase(_mini_kb())
    train = [
        _attr_item("Alice", "31 year", "train"),
        _relation_item("Alice", "Paris", "train"),
    ]
    val = [
        _attr_item("Bob", "42 year", "val"),
        _relation_item("Bob", "Rome", "val"),
    ]
    prepared = build_transfer_split(
        kb,
        train,
        val,
        SplitConfig(
            train_limit=2,
            dev_limit=0,
            test_limit=2,
            min_train_per_skeleton=1,
            max_support_facts=4,
            seed=3,
        ),
    )

    seen = set(prepared.fact_buckets["seen"])
    transfer = set(prepared.fact_buckets["transfer"])
    assert seen
    assert transfer
    assert seen.isdisjoint(transfer)
    seen_atoms = {atom for q in prepared.train for atom in q["support_atoms"]}
    transfer_atoms = {atom for q in prepared.test for atom in q["support_atoms"]}
    assert seen_atoms.isdisjoint(transfer_atoms)
    assert {q["skeleton"] for q in prepared.test} <= {
        q["skeleton"] for q in prepared.train
    }
    assert all(set(q["support_keys"]) <= transfer for q in prepared.test)
    assert all(not (set(q["support_keys"]) & transfer) for q in prepared.train)


def test_dev_quota_is_filled_from_unseen_qa_with_seen_facts():
    kb = KQAKnowledgeBase(_mini_kb())
    prepared = build_transfer_split(
        kb,
        [
            _attr_item("Alice", "31 year", "train"),
            _attr_item("Alice", "31 year", "train"),
            _attr_item("Alice", "31 year", "train"),
        ],
        [_attr_item("Bob", "42 year", "val")],
        SplitConfig(
            train_limit=2,
            dev_limit=1,
            test_limit=1,
            min_train_per_skeleton=1,
            seed=4,
        ),
    )

    assert len(prepared.dev) == 1
    assert {item["qid"] for item in prepared.train}.isdisjoint(
        item["qid"] for item in prepared.dev
    )
    assert set(prepared.dev[0]["support_atoms"]) <= set(
        prepared.train[0]["support_atoms"]
    )


def test_transfer_split_blocks_aliases_of_the_same_raw_relation():
    kb = KQAKnowledgeBase(_mini_kb())
    query_relation = {
        "question": "What is Alice's relation to Paris?",
        "answer": "lives in",
        "program": [
            {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
            {"function": "Find", "dependencies": [], "inputs": ["Paris"]},
            {"function": "QueryRelation", "dependencies": [0, 1], "inputs": []},
        ],
    }
    prepared = build_transfer_split(
        kb,
        [_relation_item("Bob", "Rome", "train"), query_relation],
        [_relation_item("Alice", "Paris", "val")],
        SplitConfig(
            train_limit=1,
            dev_limit=0,
            test_limit=1,
            min_train_per_skeleton=1,
            seed=5,
        ),
    )
    assert prepared.train[0]["question"] == "Where does Bob live?"
    assert set(prepared.train[0]["support_atoms"]).isdisjoint(
        prepared.test[0]["support_atoms"]
    )


def test_renderings_are_answer_only_and_mask_only_split_fact_value():
    kb = KQAKnowledgeBase(_mini_kb())
    fact = kb.lookup_facts[normalize("Bob, age")]
    fact_doc = render_fact_doc(fact, bucket="transfer")

    assert "42 year" in fact_doc.dense_text()
    assert [text for text, masked in fact_doc.split_segments if masked] == [" 42 year"]
    assert "<|db_start|>Bob, age<|db_retrieve|>" in fact_doc.split_text()

    qa = {
        "qid": "q1",
        "question": "What is Alice's age?",
        "answer": "31 year",
        "skeleton": "Find[]|QueryAttr[0]",
        "support_keys": [normalize("Alice, age")],
    }
    qa_doc = render_qa_doc(qa)
    assert qa_doc.dense_text() == qa_doc.split_text()
    assert qa_doc.dense_text() == "Question: What is Alice's age?\nAnswer: 31 year"
    assert "Reasoning:" not in qa_doc.dense_text()
    assert "<|db_" not in qa_doc.dense_text()


def test_prepared_data_round_trip(tmp_path):
    kb = KQAKnowledgeBase(_mini_kb())
    prepared = build_transfer_split(
        kb,
        [_attr_item("Alice", "31 year", "train")],
        [_attr_item("Bob", "42 year", "val")],
        SplitConfig(
            train_limit=1,
            dev_limit=0,
            test_limit=1,
            min_train_per_skeleton=1,
            seed=1,
        ),
    )
    report = write_prepared_data(prepared, kb, tmp_path)
    assert report["checks"]["seen_transfer_disjoint"]
    assert report["checks"]["test_requires_transfer"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "qa_test.jsonl").read_text().splitlines()
    ]
    assert rows[0]["question"] == "What is Bob's age?"
    assert "program" not in rows[0]
    assert (tmp_path / "facts.jsonl").exists()


def test_continuation_builder_writes_paired_shards_and_masks(tmp_path):
    kb = KQAKnowledgeBase(_mini_kb())
    prepared = build_transfer_split(
        kb,
        [
            _attr_item("Alice", "31 year", "train"),
            _relation_item("Alice", "Paris", "train"),
        ],
        [
            _attr_item("Bob", "42 year", "val"),
            _relation_item("Bob", "Rome", "val"),
        ],
        SplitConfig(
            train_limit=2,
            dev_limit=0,
            test_limit=2,
            min_train_per_skeleton=1,
            seed=1,
        ),
    )
    prepared_dir = tmp_path / "prepared"
    write_prepared_data(prepared, kb, prepared_dir)

    out = tmp_path / "corpus"
    report = build_kqa_continuation(
        prepared_dir,
        get_tok(),
        out,
        KQAContinuationConfig(
            total_tokens=4_000,
            fact_share=0.7,
            seed=9,
            min_fact_exposures=1,
        ),
    )
    assert report["checks"]["equal_total_token_target"]
    assert report["checks"]["all_facts_exposed"]
    assert len(report["arms"]["dense"]["artifact_sha256"]["train_bin"]) == 64
    assert len(report["arms"]["split"]["artifact_sha256"]["train_mask"]) == 64
    assert (out / "organizer.jsonl").exists()
    assert (out / "eval_transfer.jsonl").exists()

    dense_mask = np.memmap(out / "dense/train.mask.bin", dtype=np.uint8, mode="r")
    split_mask = np.memmap(out / "split/train.mask.bin", dtype=np.uint8, mode="r")
    assert dense_mask.min() == 1
    assert split_mask.min() == 0
    assert split_mask.max() == 1
