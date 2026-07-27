from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.contracts import (
    DEFAULT_CONTRACT_PATH,
    FAMILY_ORDER,
    EvaluationContractError,
    _validate_corpus_receipt,
    load_evaluation_contract,
    read_canonical_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_contract(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(raw)
    mutate(changed)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    return path


def test_frozen_contract_closes_exact_families_windows_and_bindings():
    contract = load_evaluation_contract(DEFAULT_CONTRACT_PATH, repository_root=ROOT)

    assert contract.contract_id == "memorysplit-reasoning-v3-eval-v1"
    assert contract.family_names == FAMILY_ORDER
    assert contract.accepted_items_per_family == 512
    assert contract.total_items == 14 * 512
    assert [family.index_start for family in contract.families] == [
        2_000_000_000 + index * 10_000 for index in range(14)
    ]
    assert all(family.index_stop - family.index_start == 10_000 for family in contract.families)
    assert contract.corpus_receipt_sha256 == (
        "b1eabb1719f66876ab54cc0791b857ccdbbbddb0ffb8c5986ac2aaa7bf33b80d"
    )
    assert contract.recipe_sha256 == (
        "276c0f85e2551af5df66b866b2c424118e7a8ddbf9efe2eff805196ba4d60a2f"
    )
    assert contract.transfer_manifest_sha256 == (
        "84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb"
    )
    assert contract.source_commit == "4fbdd59860198f2ccc623dc2cdd1aeb5af254afa"
    assert contract.source_tree == "a6f4f51213acaf57f63371e878c247e0179c345e"
    assert contract.corpus_receipt_path == "extension/receipt.json"
    assert contract.source_stage_receipt_path == "source-stage-receipt.json"
    assert contract.source_relative_path == (
        "objective_auxiliary/reasoning_gym_exact_answer/reasoning_gym"
    )
    assert contract.evaluator_role_arn == (
        "arn:aws:iam::${AWS_ACCOUNT_ID}:role/memorysplit-reasoning-v3-evaluator"
    )
    assert contract.signer_key_alias == (
        "alias/memorysplit-reasoning-v3-evaluator-v1"
    )
    assert contract.sealed_gold_kms_key_alias == (
        "alias/memorysplit-reasoning-v3-evaluator-sealed-v1"
    )
    assert contract.aws_region == "us-east-1"
    assert contract.storage_bucket == (
        "${MS_S3_BUCKET}"
    )
    assert contract.storage_prefix == "evaluations/reasoning-v3/v1"
    assert contract.aws_boundary_path == (
        "configs/preregistration-135m-reasoning-v3-eval-v1-aws-boundary.json"
    )
    assert contract.aws_boundary_sha256 == (
            "7660616406f1985b4d2506bb0cb58af8f25b5e1085fd8ca4f597503cfd5ba903"
    )
    assert contract.no_inspection_attestation == {
        "evaluation_items_or_gold_inspected": False,
        "protected_training_started": False,
        "selection_depends_on_observed_results": False,
    }
    assert contract.skip_reasons == ("independent_oracle", "overlength")


def test_contract_rejects_unknown_fields_bool_integer_aliases_and_unsafe_paths(
    tmp_path: Path,
):
    unknown = _write_contract(tmp_path, lambda raw: raw.update({"unknown": "field"}))
    with pytest.raises(EvaluationContractError, match="fields"):
        load_evaluation_contract(unknown, repository_root=ROOT)

    boolean_count = _write_contract(
        tmp_path,
        lambda raw: raw["generation"].update({"accepted_items_per_family": True}),
    )
    with pytest.raises(EvaluationContractError, match="positive integer"):
        load_evaluation_contract(boolean_count, repository_root=ROOT)

    unsafe_recipe = _write_contract(
        tmp_path,
        lambda raw: raw["corpus"]["recipe"].update({"path": "../reasoning.json"}),
    )
    with pytest.raises(EvaluationContractError, match="unsafe"):
        load_evaluation_contract(unsafe_recipe, repository_root=ROOT)

    aliased = tmp_path / "aliased.yaml"
    aliased.write_text(
        DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8")
        .replace("contract_status: frozen", "contract_status: &status frozen")
        .replace(
            "scientific_scope: prospectively_frozen_exploratory_n10",
            "scientific_scope: *status",
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationContractError, match="aliases"):
        load_evaluation_contract(aliased, repository_root=ROOT)


def test_contract_rejects_family_reordering_and_false_no_inspection_attestation(
    tmp_path: Path,
):
    reordered = _write_contract(
        tmp_path,
        lambda raw: raw["families"].__setitem__(
            slice(0, 2), list(reversed(raw["families"][:2]))
        ),
    )
    with pytest.raises(EvaluationContractError, match="family order"):
        load_evaluation_contract(reordered, repository_root=ROOT)

    inspected = _write_contract(
        tmp_path,
        lambda raw: raw["no_inspection_attestation"].update(
            {"evaluation_items_or_gold_inspected": True}
        ),
    )
    with pytest.raises(EvaluationContractError, match="no-inspection"):
        load_evaluation_contract(inspected, repository_root=ROOT)

    wrong_authority = _write_contract(
        tmp_path,
        lambda raw: raw["authority"].update(
            {
                "evaluator_role_arn": (
                    "arn:aws:iam::${AWS_ACCOUNT_ID}:role/"
                    "memorysplit-reasoning-v3-trainer"
                )
            }
        ),
    )
    with pytest.raises(EvaluationContractError, match="authority"):
        load_evaluation_contract(wrong_authority, repository_root=ROOT)


def test_canonical_json_reader_rejects_duplicate_and_noncanonical_json(tmp_path: Path):
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="canonical JSON"):
        read_canonical_json(noncanonical, label="fixture")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":1}\n', encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="repeats key"):
        read_canonical_json(duplicate, label="fixture")

    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps({"a": 1, "b": 2}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert read_canonical_json(canonical, label="fixture") == {"a": 1, "b": 2}


def test_injected_corpus_receipt_is_canonical_closed_and_manifest_bound(
    tmp_path: Path,
):
    contract = load_evaluation_contract(DEFAULT_CONTRACT_PATH, repository_root=ROOT)
    receipt = {
        "base_corpus": {
            "contract_id": "memorysplit-parallel-corpus-v2",
            "logical_tokens": 7_120_879_616,
            "ordered_stream_sha256": (
                "ae7a334cd2d73689d7ff439d9b736ebf06e5f992ae59b52ebdd32a1426bac40f"
            ),
            "packed_stream_sha256": (
                "dc0134131c57ec339997f9cee9c22f14a7414200671805c63d7cd7a7a3d5738d"
            ),
            "receipt_sha256": (
                "65eda881f59acac356df0c3fbbb62df46d60bc013df05b72e8baacba7640b027"
            ),
        },
        "composite": {
            "ordering": "frozen-v2-prefix-then-reasoning-extension",
            "raw_target_tokens": 8_169_455_616,
            "stream_sha256": {
                "dense_target_weights": (
                    "917768b13ec169728cec51dc8294d118a113aee3c370ecd8c16ef0529f63f56e"
                ),
                "packed_targets": (
                    "035ee111c329eb615c642eae9b9a7075314932ff8175e989aabb3317d6a4ef6f"
                ),
                "split90_target_weights": (
                    "8a9c84c900e503d1742342b6a21092292c2968313087d0873e429b4268757144"
                ),
            },
            "terminal_updates": 15_582,
        },
        "contract_id": "memorysplit-reasoning-dataset-v3",
        "extension": {
            "artifacts": [
                {
                    "bytes": 2_097_152_000,
                    "path": "packed/targets.bin",
                    "sha256": (
                        "e09d08cdede2317ce0841264faa247322c2fe48ead38a3d81a26572a8effdd31"
                    ),
                },
                {
                    "bytes": 60_244_260,
                    "path": "records/manifest.bin",
                    "sha256": contract.record_manifest_sha256,
                },
                {
                    "bytes": 1_048_576_000,
                    "path": "sidecars/shared_target_weights.bin",
                    "sha256": (
                        "936bed85cfae5dea666e42a3f35f3a86ae1ac8ca6aa0bba49871980ef04df7e9"
                    ),
                },
            ],
            "format": "memorysplit-reasoning-extension-v2",
            "max_record_tokens": 1024,
            "packed_stream_sha256": (
                "e09d08cdede2317ce0841264faa247322c2fe48ead38a3d81a26572a8effdd31"
            ),
            "probes": {
                task: [
                    {
                        "accepted": True,
                        "record_sha256": "b" * 64,
                        "rejection_reason": None,
                        "source_index": index,
                        "token_count": 1,
                    }
                    for index in (0, 1, 17, 127)
                ]
                for task in FAMILY_ORDER
            },
            "raw_target_tokens": 1_048_576_000,
            "record_count": 7_530_527,
            "record_stream_sha256": "f" * 64,
            "shared_target_weights_sha256": (
                "936bed85cfae5dea666e42a3f35f3a86ae1ac8ca6aa0bba49871980ef04df7e9"
            ),
            "target_weight_policy": (
                "all_extension_reasoning_targets_are_internal_in_both_arms"
            ),
            "targets_per_update": 524_288,
            "task_stats": [
                {
                    "dataset": task,
                    "emitted_records": 1,
                    "emitted_tokens": 1,
                    "examined_records": 1,
                    "oracle_rejections": 0,
                    "overlength_rejections": 0,
                    "source_cursor": 1,
                    "target_quota": 1,
                }
                for task in FAMILY_ORDER
            ],
            "terminal_updates": 2_000,
        },
        "format": "memorysplit-reasoning-composite-v3",
        "generator_artifacts": {
            "configs/reasoning-dataset-v3.json": contract.recipe_sha256,
            "corpusgen/parallel/canonical.py": "c" * 64,
            "corpusgen/reasoning_expansion.py": "d" * 64,
            "corpusgen/reasoning_oracles.py": "e" * 64,
            "train/tokenizer.py": "f" * 64,
        },
        "recipe_sha256": contract.recipe_sha256,
        "runtime_lock": dict(contract.runtime_lock),
        "schema_version": 2,
        "source": {
            "reasoning_gym_tree_commitment_sha256": "a" * 64,
            "reasoning_gym_version": "0.1.19",
            "source_stage_receipt_sha256": contract.source_stage_receipt_sha256,
        },
    }
    payload = canonical_json_bytes(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(payload)
    fixture_contract = replace(
        contract,
        corpus_receipt_sha256=hashlib.sha256(payload).hexdigest(),
    )

    loaded = _validate_corpus_receipt(path, fixture_contract)
    assert loaded["extension"]["artifacts"][1]["sha256"] == (
        contract.record_manifest_sha256
    )

    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="canonical JSON"):
        _validate_corpus_receipt(path, fixture_contract)
