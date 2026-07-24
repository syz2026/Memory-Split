from __future__ import annotations

from types import SimpleNamespace

import pytest

from corpusgen.v2_objective import (
    _ReasoningGym,
    _SEED_ATTEMPTS,
    _seed,
)


class _Expression:
    @staticmethod
    def from_string(value: str) -> str:
        return value


class _Dataset:
    def __init__(self, row: dict, *, valid: bool, trivial: bool = False):
        self.row = row
        self.valid = valid
        self.trivial = trivial

    def __getitem__(self, index: int) -> dict:
        assert index == 0
        return self.row

    def _is_valid_conclusion(self, premises, answer) -> bool:
        assert premises
        assert answer
        return self.valid

    def _is_trivial(self, answer) -> bool:
        assert answer
        return self.trivial

    def score_answer(self, answer: str, row: dict) -> float:
        return float(answer == row["metadata"]["example_answer"])


class _Factory:
    def __init__(self, datasets: list[_Dataset]):
        self.datasets = datasets
        self.seeds: list[int] = []

    def create_dataset(self, name: str, *, seed: int, size: int) -> _Dataset:
        assert name == "propositional_logic"
        assert size == 1
        self.seeds.append(seed)
        return self.datasets[min(len(self.seeds) - 1, len(self.datasets) - 1)]


def _row(question: str, answer: str) -> dict:
    return {
        "answer": None,
        "metadata": {
            "example_answer": answer,
            "premises": ["P"],
        },
        "question": question,
    }


def _provider(factory: _Factory) -> _ReasoningGym:
    provider = _ReasoningGym.__new__(_ReasoningGym)
    provider._factory = factory
    provider._dataset_modules = {
        "propositional_logic": SimpleNamespace(Expression=_Expression)
    }
    return provider


def test_reasoning_gym_rejects_invalid_native_fallback_and_retries():
    factory = _Factory(
        [
            _Dataset(_row("rejected question", "P"), valid=False),
            _Dataset(_row("accepted question", "(P ∨ R)"), valid=True),
        ]
    )

    result = _provider(factory).generate(3)

    assert factory.seeds == [
        _seed("reasoning_gym_exact_answer", 3, 0),
        _seed("reasoning_gym_exact_answer", 3, 1),
    ]
    assert result["question"] == "accepted question"
    assert result["answer"] == "(P ∨ R)"
    assert result["metadata"] == {
        "attempt": 1,
        "dataset": "propositional_logic",
        "native_metadata": {
            "example_answer": "(P ∨ R)",
            "premises": ["P"],
        },
        "oracle_field": "metadata.example_answer",
        "rejected_native_samples": 1,
        "rejection_policy": "invalid_or_trivial_metadata_oracle",
        "seed": _seed("reasoning_gym_exact_answer", 3, 1),
    }


def test_reasoning_gym_fails_closed_after_retry_domain_exhaustion():
    factory = _Factory([_Dataset(_row("invalid", "P"), valid=False)])

    with pytest.raises(RuntimeError, match="after 128 attempts"):
        _provider(factory).generate(3)

    assert factory.seeds == [
        _seed("reasoning_gym_exact_answer", 3, attempt)
        for attempt in range(_SEED_ATTEMPTS)
    ]


def test_reasoning_gym_valid_attempt_zero_is_unchanged():
    factory = _Factory([_Dataset(_row("valid question", "(P ∨ R)"), valid=True)])

    result = _provider(factory).generate(3)

    assert factory.seeds == [_seed("reasoning_gym_exact_answer", 3, 0)]
    assert result == {
        "answer": "(P ∨ R)",
        "metadata": {
            "dataset": "propositional_logic",
            "native_metadata": {
                "example_answer": "(P ∨ R)",
                "premises": ["P"],
            },
            "oracle_field": "metadata.example_answer",
            "seed": _seed("reasoning_gym_exact_answer", 3, 0),
        },
        "question": "valid question",
    }
