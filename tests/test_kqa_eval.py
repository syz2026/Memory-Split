import pytest
import torch

from corpusgen.records import QAItem
from evals.kqa import (
    fact_availability,
    score_kqa_recall,
    score_kqa_transfer,
    summarize_transfer_rows,
)
from organizer.store import Organizer
from scripts.run_kqa_evals import build_wrong_store, score_oracle_context
from train.tokenizer import get_tok


TOK = get_tok()
CPU = torch.device("cpu")


class _OpenLoop:
    def __init__(self, programs):
        self.programs = programs
        self.cursor = 0

    def forward_step(self, idx, cache):
        batch, width = idx.shape
        if cache is None:
            selected = self.programs[self.cursor : self.cursor + batch]
            self.cursor += batch
            cache = {"programs": selected, "step": 0}
        else:
            cache["step"] += 1
        logits = torch.zeros(batch, width, TOK.VOCAB_SIZE)
        for row, program in enumerate(cache["programs"]):
            step = cache["step"]
            token = program[step] if step < len(program) else TOK.EOT
            logits[row, -1, token] = 1
        return logits, cache


def test_fact_availability_maps_probe_results():
    rows = [
        {"fact_id": "alice, age", "correct": True},
        {"fact_id": "bob, age", "correct": False},
    ]
    assert fact_availability(rows) == {"alice, age": True, "bob, age": False}


def test_transfer_summary_reports_conditional_accuracy():
    rows = [
        {"correct": True, "facts_available": True, "support_coverage": 1.0},
        {"correct": False, "facts_available": True, "support_coverage": 1.0},
        {"correct": False, "facts_available": False, "support_coverage": 0.5},
    ]
    summary = summarize_transfer_rows(rows)
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["facts_available_rate"] == pytest.approx(2 / 3)
    assert summary["conditional_accuracy"] == pytest.approx(0.5)
    assert summary["mean_support_coverage"] == pytest.approx(5 / 6)


def test_dense_transfer_uses_direct_recall_as_availability():
    items = [
        QAItem(
            qid="q1",
            task="kqa",
            prompt="Question: What is Bob's age?\n",
            answer="42 year",
            meta={"support_keys": ["bob, age"], "template": "Find|QueryAttr"},
        ),
        QAItem(
            qid="q2",
            task="kqa",
            prompt="Question: Where does Bob live?\n",
            answer="Rome",
            meta={"support_keys": ["bob, lives in (forward)"], "template": "Find|Relate"},
        ),
    ]
    programs = [
        TOK.encode("Answer: 42 year") + [TOK.EOT],
        TOK.encode("Answer: Milan") + [TOK.EOT],
    ]
    rows, summary = score_kqa_transfer(
        _OpenLoop(programs),
        TOK,
        items,
        mode="dense",
        organizer=None,
        device=CPU,
        fact_availability_map={
            "bob, age": True,
            "bob, lives in (forward)": False,
        },
        max_new=32,
        batch_size=2,
    )
    assert [row["correct"] for row in rows] == [True, False]
    assert [row["facts_available"] for row in rows] == [True, False]
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["conditional_accuracy"] == 1.0


def test_oracle_context_reduces_generation_budget_instead_of_truncating_prompt():
    item = QAItem(
        qid="q1",
        task="kqa",
        prompt="Evidence:\n" + ("long evidence " * 20) + "\nQuestion: value?\n",
        answer="x",
        meta={
            "support_keys": ["item, value"],
            "template": "Find|QueryAttr",
            "skill": "What",
            "program_length": 2,
            "support_count": 1,
        },
    )
    required = len(TOK.encode("Answer: x")) + 2
    context = len(TOK.encode(item.prompt)) + required
    rows, summary = score_oracle_context(
        _OpenLoop([TOK.encode("Answer: x") + [TOK.EOT]]),
        TOK,
        [item],
        CPU,
        {"item, value": True},
        context=context,
        requested_max_new=384,
        batch_size=1,
    )

    assert rows[0]["correct"] is True
    assert summary["context_constrained_items"] == 1
    assert summary["generation_budget_min"] == required


def test_wrong_store_changes_every_required_value():
    facts = {
        "alice, lives in": {
            "key": "alice, lives in",
            "name": "Alice",
            "relation": "lives in",
            "value": "Paris",
            "kind": "relation",
        },
        "bob, lives in": {
            "key": "bob, lives in",
            "name": "Bob",
            "relation": "lives in",
            "value": "Rome",
            "kind": "relation",
        },
        "alice, age": {
            "key": "alice, age",
            "name": "Alice",
            "relation": "age",
            "value": "31 year",
            "kind": "attribute",
        },
    }
    store, corruption, strategy = build_wrong_store(facts, set(facts))

    assert all(corruption[key] != row["value"] for key, row in facts.items())
    assert store.lookup("Alice, lives in") == corruption["alice, lives in"]
    assert store.lookup("Alice, age") == corruption["alice, age"]
    assert strategy["alice, lives in"] == "observed_relation_matched"
    assert strategy["alice, age"] == "synthetic_format_preserving"


def test_dense_recall_does_not_use_arbitrary_substring_matches():
    probe = QAItem(
        qid="r1",
        task="kqa_recall",
        prompt="Knowledge: Item's count is",
        answer="1",
        meta={"fact_id": "item, count", "relation": "count"},
    )
    rows, summary = score_kqa_recall(
        _OpenLoop([TOK.encode(" 10") + [TOK.EOT]]),
        TOK,
        [probe],
        "closed",
        None,
        CPU,
        max_new=8,
    )
    assert rows[0]["correct"] is False
    assert summary["accuracy"] == 0.0


@pytest.mark.parametrize("generated", [" 1.5", " 1-02-03", " 1 year"])
def test_dense_recall_rejects_numeric_and_date_prefixes(generated):
    probe = QAItem(
        qid="r1",
        task="kqa_recall",
        prompt="Knowledge: Item's count is",
        answer="1",
        meta={"fact_id": "item, count", "relation": "count"},
    )
    rows, _ = score_kqa_recall(
        _OpenLoop([TOK.encode(generated) + [TOK.EOT]]),
        TOK,
        [probe],
        "closed",
        None,
        CPU,
        max_new=8,
    )
    assert rows[0]["correct"] is False


def test_split_recall_requires_completed_expected_query():
    organizer = Organizer()
    organizer.add("Bob", "age", "42 year")
    probe = QAItem(
        qid="r1",
        task="kqa_recall",
        prompt="Knowledge: Bob's age is",
        answer="42 year",
        meta={
            "fact_id": "bob, age",
            "query": "Bob, age",
            "relation": "age",
        },
    )
    query = [TOK.DB_START] + TOK.encode("Bob, age") + [TOK.DB_RETRIEVE]
    rows, _ = score_kqa_recall(
        _OpenLoop([query]),
        TOK,
        [probe],
        "on",
        organizer,
        CPU,
        max_new=32,
    )
    assert rows[0]["correct"] is True
    assert rows[0]["completed_expected_lookup"] is True
