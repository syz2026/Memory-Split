from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.contracts import (
    DEFAULT_CONTRACT_PATH,
    load_evaluation_contract,
)
from evals.reasoning_v3.generate import (
    CandidateRow,
    ProvenanceCommitment,
    _build_evaluation_registry,
)
from evals.reasoning_v3.sealing import (
    EvaluationSealingError,
    _build_release_bundle,
    _double_generate_release,
    _parse_model_visible_release,
    _parse_sealed_gold_release,
    _validate_release_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


class _FixtureSource:
    def generate(self, task: str, source_index: int) -> CandidateRow:
        question = f"Fixture question {source_index}?"
        answer = "café"
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "answer": answer,
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
            native_answer=answer,
            oracle_answer=answer,
            token_count=20,
            prompt_token_count=10,
            answer_token_count=2,
            record_sha256=digest,
        )


def _fixture():
    contract = replace(
        load_evaluation_contract(DEFAULT_CONTRACT_PATH, repository_root=ROOT),
        accepted_items_per_family=2,
    )
    items = _build_evaluation_registry(
        contract,
        _FixtureSource(),
        training_record_keys=frozenset(),
    )
    provenance = ProvenanceCommitment(
        corpus_receipt_sha256=contract.corpus_receipt_sha256,
        source_stage_receipt_sha256=contract.source_stage_receipt_sha256,
        source_tree_commitment_sha256="a" * 64,
        record_manifest_sha256=contract.record_manifest_sha256,
        record_count=7_530_527,
        generator_artifacts_sha256="b" * 64,
        runtime_lock_sha256="c" * 64,
    )
    return contract, items, provenance


def _field_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_field_names, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_field_names, value)))
    return set()


def test_release_build_is_byte_deterministic_and_model_visible_has_no_gold():
    contract, items, provenance = _fixture()

    first = _build_release_bundle(contract, items, provenance)
    second = _build_release_bundle(contract, items, provenance)

    assert first == second
    public = json.loads(first.model_visible_bytes)
    sealed = json.loads(first.sealed_gold_bytes)
    assert public["release_kind"] == "model_visible"
    assert sealed["release_kind"] == "sealed_gold"
    assert public["registry_sha256"] == sealed["registry_sha256"]
    assert public["registry_sha256"] == first.registry_sha256
    assert len(public["items"]) == len(contract.families) * 2
    assert set(public["items"][0]) == {
        "item_id",
        "max_new_tokens",
        "prompt",
        "scorer_id",
        "source_index",
        "task",
    }
    assert set(sealed["items"][0]) == {
        "canonical_answer",
        "item_id",
        "oracle_replay",
        "source_index",
        "task",
    }
    assert not {
        "answer",
        "gold",
        "canonical_answer",
        "oracle_replay",
    } & _field_names(public)


def test_public_parser_rejects_noncanonical_json_unknown_fields_and_gold_leakage(
    tmp_path: Path,
):
    contract, items, provenance = _fixture()
    bundle = _build_release_bundle(contract, items, provenance)
    public = json.loads(bundle.model_visible_bytes)

    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(
        json.dumps(public, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationSealingError, match="canonical JSON"):
        _parse_model_visible_release(noncanonical, contract)

    public["items"][0]["answer"] = "café"
    leaking = tmp_path / "leaking.json"
    leaking.write_bytes(canonical_json_bytes(public))
    with pytest.raises(EvaluationSealingError, match="fields|gold"):
        _parse_model_visible_release(leaking, contract)


def test_pair_validation_rejects_gold_tampering_and_missing_items(
    tmp_path: Path,
):
    contract, items, provenance = _fixture()
    bundle = _build_release_bundle(contract, items, provenance)
    sealed_path = tmp_path / "sealed.json"
    sealed = json.loads(bundle.sealed_gold_bytes)
    sealed["items"][0]["canonical_answer"] = "tampered"
    sealed_path.write_bytes(canonical_json_bytes(sealed))

    with pytest.raises(
        EvaluationSealingError,
        match="registry commitment|oracle evidence",
    ):
        _validate_release_bundle(
            replace(bundle, sealed_gold_bytes=sealed_path.read_bytes()),
            contract,
            provenance,
        )

    sealed["items"].pop()
    sealed["item_count"] -= 1
    sealed_path.write_bytes(canonical_json_bytes(sealed))
    with pytest.raises(EvaluationSealingError, match="item count|identity"):
        _parse_sealed_gold_release(sealed_path, contract)


def test_double_generation_detects_independent_source_drift():
    contract, items, provenance = _fixture()
    stable = _build_release_bundle(contract, items, provenance)
    drifted_items = list(items)
    first = drifted_items[0]
    question = first.prompt.split(
        "\nQuestion: ",
        1,
    )[1].removesuffix("\nAnswer:")
    different_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "answer": "different",
                "question": question,
                "source_index": first.source_index,
                "task": first.task,
            }
        )
    ).hexdigest()
    drifted_items[0] = replace(
        first,
        canonical_answer="different",
        oracle_replay=replace(
            first.oracle_replay,
            record_sha256=different_digest,
            native_answer_sha256=hashlib.sha256(b"different").hexdigest(),
            independent_answer_sha256=hashlib.sha256(b"different").hexdigest(),
        ),
    )
    drifted = _build_release_bundle(contract, drifted_items, provenance)
    builds = iter((stable, drifted))

    with pytest.raises(EvaluationSealingError, match="independent generation"):
        _double_generate_release(lambda: next(builds))
