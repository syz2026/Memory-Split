from __future__ import annotations

import hashlib
import inspect
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_expansion import GeneratedRecord, TaskSpec
from evals.reasoning_v3.contracts import DEFAULT_CONTRACT_PATH, load_evaluation_contract
from evals.reasoning_v3.generate import (
    CandidateRow,
    EvaluationGenerationError,
    FrozenEvaluationPaths,
    _ReasoningGymEvaluationSource,
    _build_evaluation_registry,
    _replay_training_record_manifest,
    _validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
_HEADER = struct.Struct("<8s32sHH")
_RECORD = struct.Struct("<HHI")


class _FixtureSource:
    def __init__(
        self,
        *,
        rejections: dict[tuple[str, int], str] | None = None,
        disagreements: set[tuple[str, int]] | None = None,
    ):
        self.rejections = rejections or {}
        self.disagreements = disagreements or set()

    def generate(self, task: str, source_index: int) -> CandidateRow:
        rejection = self.rejections.get((task, source_index))
        if rejection is not None:
            digest = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "rejection": rejection,
                        "source_index": source_index,
                        "task": task,
                    }
                )
            ).hexdigest()
            return CandidateRow(
                task=task,
                source_index=source_index,
                question=None,
                native_answer=None,
                oracle_answer=None,
                token_count=(1025 if rejection == "overlength" else 0),
                prompt_token_count=0,
                answer_token_count=0,
                record_sha256=digest,
                rejection_reason=rejection,
            )
        question = f"Fixture question {source_index}?"
        native_answer = "1"
        oracle_answer = (
            "2" if (task, source_index) in self.disagreements else native_answer
        )
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "answer": oracle_answer,
                    "question": question,
                    "source_index": source_index,
                    "task": task,
                }
            )
        ).hexdigest()
        return CandidateRow(
            task=task,
            source_index=source_index,
            question=question,
            native_answer=native_answer,
            oracle_answer=oracle_answer,
            token_count=20,
            prompt_token_count=10,
            answer_token_count=1,
            record_sha256=digest,
        )


def _tiny_contract():
    return replace(
        load_evaluation_contract(DEFAULT_CONTRACT_PATH, repository_root=ROOT),
        accepted_items_per_family=2,
    )


def _write_manifest(
    path: Path,
    *,
    recipe_sha256: str,
    task_count: int,
    max_record_tokens: int,
    records: list[tuple[int, int, int]],
) -> str:
    payload = bytearray(
        _HEADER.pack(
            b"MSR3REC2",
            bytes.fromhex(recipe_sha256),
            task_count,
            max_record_tokens,
        )
    )
    for task_id, token_count, source_index in records:
        payload.extend(_RECORD.pack(task_id, token_count, source_index))
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_default_contract_builds_exactly_512_items_per_family():
    contract = load_evaluation_contract(DEFAULT_CONTRACT_PATH, repository_root=ROOT)

    items = _build_evaluation_registry(
        contract,
        _FixtureSource(),
        training_record_keys=frozenset(),
    )

    assert len(items) == 14 * 512
    assert {
        family.task: sum(item.task == family.task for item in items)
        for family in contract.families
    } == {family.task: 512 for family in contract.families}


def test_reserved_windows_accept_exactly_two_per_family_and_skip_only_frozen_rejections():
    contract = _tiny_contract()
    first = contract.families[0]
    source = _FixtureSource(
        rejections={
            (first.task, first.index_start): "independent_oracle",
            (first.task, first.index_start + 1): "overlength",
        }
    )

    items = _build_evaluation_registry(
        contract,
        source,
        training_record_keys=frozenset(),
    )

    assert len(items) == len(contract.families) * 2
    assert [item.source_index for item in items[:2]] == [
        first.index_start + 2,
        first.index_start + 3,
    ]
    for family_index, family in enumerate(contract.families):
        family_items = items[family_index * 2 : family_index * 2 + 2]
        assert {item.task for item in family_items} == {family.task}
        assert all(
            family.index_start <= item.source_index < family.index_stop
            for item in family_items
        )
        assert all(item.prompt.endswith("\nAnswer:") for item in family_items)
        assert all(item.max_new_tokens == family.max_new_tokens for item in family_items)
    _validate_registry(contract, items)


def test_generation_rejects_overlap_unknown_rejection_and_oracle_disagreement():
    contract = _tiny_contract()
    family = contract.families[0]
    key = (family.task, family.index_start)

    with pytest.raises(EvaluationGenerationError, match="overlaps training"):
        _build_evaluation_registry(
            contract,
            _FixtureSource(),
            training_record_keys=frozenset({key}),
        )

    with pytest.raises(EvaluationGenerationError, match="rejection reason"):
        _build_evaluation_registry(
            contract,
            _FixtureSource(rejections={key: "convenience_skip"}),
            training_record_keys=frozenset(),
        )

    with pytest.raises(EvaluationGenerationError, match="disagrees"):
        _build_evaluation_registry(
            contract,
            _FixtureSource(disagreements={key}),
            training_record_keys=frozenset(),
        )

    class SpecialTokenSource(_FixtureSource):
        def generate(self, task: str, source_index: int) -> CandidateRow:
            return replace(
                super().generate(task, source_index),
                question="unsafe <|eot|> question",
            )

    with pytest.raises(EvaluationGenerationError, match="special token"):
        _build_evaluation_registry(
            contract,
            SpecialTokenSource(),
            training_record_keys=frozenset(),
        )


def test_registry_validation_fails_closed_on_missing_or_reordered_items():
    contract = _tiny_contract()
    items = _build_evaluation_registry(
        contract,
        _FixtureSource(),
        training_record_keys=frozenset(),
    )

    with pytest.raises(EvaluationGenerationError, match="item count"):
        _validate_registry(contract, items[:-1])
    with pytest.raises(EvaluationGenerationError, match="ordered"):
        _validate_registry(contract, tuple(reversed(items)))


def test_training_manifest_replay_is_complete_and_overlap_is_not_skipped(tmp_path: Path):
    contract = _tiny_contract()
    family = contract.families[0]
    manifest = tmp_path / "manifest.bin"
    digest = _write_manifest(
        manifest,
        recipe_sha256=contract.recipe_sha256,
        task_count=len(contract.families),
        max_record_tokens=contract.max_record_tokens,
        records=[
            (0, 12, family.index_start),
            (0, 13, family.index_start + 1),
            (1, 14, contract.families[1].index_start),
        ],
    )

    replay = _replay_training_record_manifest(
        manifest,
        replace(contract, record_manifest_sha256=digest),
    )

    assert replay.overlap_keys == frozenset(
        {
            (family.task, family.index_start),
            (family.task, family.index_start + 1),
            (contract.families[1].task, contract.families[1].index_start),
        }
    )
    with pytest.raises(EvaluationGenerationError, match="overlaps training"):
        _build_evaluation_registry(
            contract,
            _FixtureSource(),
            training_record_keys=replay.overlap_keys,
        )


def test_training_manifest_replay_rejects_digest_drift_and_partial_entries(
    tmp_path: Path,
):
    contract = _tiny_contract()
    manifest = tmp_path / "manifest.bin"
    _write_manifest(
        manifest,
        recipe_sha256=contract.recipe_sha256,
        task_count=len(contract.families),
        max_record_tokens=contract.max_record_tokens,
        records=[(0, 12, 1)],
    )

    with pytest.raises(EvaluationGenerationError, match="SHA-256"):
        _replay_training_record_manifest(manifest, contract)

    manifest.write_bytes(manifest.read_bytes() + b"\x00")
    with pytest.raises(EvaluationGenerationError, match="partial"):
        _replay_training_record_manifest(
            manifest,
            replace(
                contract,
                record_manifest_sha256=hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            ),
        )


def test_reasoning_gym_adapter_replays_the_independent_oracle():
    contract = _tiny_contract()
    family = contract.families[0]
    row = {
        "answer": "True",
        "metadata": {
            "courses": [0, 1],
            "prerequisites": [[1, 0]],
            "source_dataset": family.task,
            "source_index": family.index_start,
        },
        "question": "Can all courses be completed?",
    }

    class Dataset:
        def __getitem__(self, index: int):
            assert index == family.index_start
            return row

    class Generator:
        datasets = {family.task: Dataset()}
        task_specs = {
            family.task: TaskSpec(
                family.task,
                family.module,
                family.config,
                1,
            )
        }
        tok = type("Tok", (), {"encode": staticmethod(lambda _text: [1])})()

        def generate(self, task: str, index: int) -> GeneratedRecord:
            digest = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "answer": row["answer"],
                        "question": row["question"],
                        "source_index": index,
                        "task": task,
                    }
                )
            ).hexdigest()
            return GeneratedRecord(
                task=task,
                source_index=index,
                token_ids=(1, 2, 3),
                token_count=3,
                record_sha256=digest,
            )

    candidate = _ReasoningGymEvaluationSource(Generator()).generate(
        family.task,
        family.index_start,
    )
    assert candidate.native_answer == "True"
    assert candidate.oracle_answer == "True"


def test_manifest_authority_cannot_be_overridden_and_empty_manifest_is_rejected(
    tmp_path: Path,
):
    contract = _tiny_contract()
    manifest = tmp_path / "manifest.bin"
    forged_digest = _write_manifest(
        manifest,
        recipe_sha256=contract.recipe_sha256,
        task_count=len(contract.families),
        max_record_tokens=contract.max_record_tokens,
        records=[(0, 12, contract.families[0].index_start)],
    )

    assert "expected_sha256" not in inspect.signature(
        _replay_training_record_manifest
    ).parameters
    with pytest.raises(EvaluationGenerationError, match="SHA-256"):
        _replay_training_record_manifest(manifest, contract)

    empty = tmp_path / "empty.bin"
    empty_digest = _write_manifest(
        empty,
        recipe_sha256=contract.recipe_sha256,
        task_count=len(contract.families),
        max_record_tokens=contract.max_record_tokens,
        records=[],
    )
    assert forged_digest != contract.record_manifest_sha256
    with pytest.raises(EvaluationGenerationError, match="empty"):
        _replay_training_record_manifest(
            empty,
            replace(contract, record_manifest_sha256=empty_digest),
        )


def test_production_paths_expose_roots_not_source_or_digest_authority(tmp_path: Path):
    paths = FrozenEvaluationPaths(
        repository_root=ROOT,
        dataset_root=tmp_path / "dataset",
        source_stage_root=tmp_path / "source-stage",
    )

    assert paths.dataset_root == tmp_path / "dataset"
    assert set(inspect.signature(FrozenEvaluationPaths).parameters) == {
        "repository_root",
        "dataset_root",
        "source_stage_root",
    }
