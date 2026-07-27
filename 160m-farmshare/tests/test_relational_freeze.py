from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evals.relational_gates as relational_gates
from corpusgen.relational_build import RelationalBuildConfig, build_relational_corpus
from evals.relational_design import (
    DESIGN_EFFECT,
    DESIGN_PAIRS,
    DESIGN_SIMULATION_SEED,
    DESIGN_SIMULATION_VERSION,
    DESIGN_STUDIES,
    BlindedDevelopmentRows,
    DesignPowerSimulation,
    _make_receipt,
)
from evals.relational_gates import (
    D160_PARAMETERS,
    build_gate_0_receipt,
    build_gate_5_receipt,
    evaluate_development_gates,
    validate_gate_receipt,
)
from experiment.provenance import SourceProvenance
from experiment.artifacts import canonical_json_bytes
from scripts.build_relational_corpus import _validate_frozen_source_lock
from scripts.freeze_relational_study import (
    FIXTURE_WATERMARK,
    FreezeManifest,
    build_freeze_manifest,
    load_freeze_manifest,
    make_fixture_freeze,
    validate_freeze_manifest,
    write_freeze_manifest,
)
from tests.test_relational_gates import (
    common_input_hashes,
    development_fixture,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def valid_source_provenance(*, clean: bool = True) -> SourceProvenance:
    common = common_input_hashes()
    return SourceProvenance(
        git_revision="1" * 40,
        source_tree_sha256=common["source_tree_sha256"],
        clean_tree=clean,
        python_version="3.12.4",
        python_implementation="CPython",
        platform="test-platform",
        artifact_sha256={
            "source_lock": common["source_lock_sha256"],
            "relation_schema": common["relation_schema_sha256"],
            "preregistration": common["preregistration_sha256"],
            "evaluator": common["evaluator_sha256"],
            "analysis": common["analysis_sha256"],
            "corpus_recipe": _digest("corpus-recipe"),
            "dense_sidecar_recipe": _digest("dense-sidecar-recipe"),
            "split_sidecar_recipe": _digest("split-sidecar-recipe"),
            "random_sidecar_recipe": _digest("random-sidecar-recipe"),
        },
    )


def valid_design_receipt():
    development = BlindedDevelopmentRows(
        rows_by_label={"arm_a": (), "arm_b": ()},
        source_arms={"arm_a": "split", "arm_b": "dense"},
        source_paths={
            "arm_a": Path("/development/arm-a.jsonl"),
            "arm_b": Path("/development/arm-b.jsonl"),
        },
        permutation_commitment=_digest("permutation"),
        commitment_sha256=_digest("commitment"),
        blinded_input_hashes={
            "arm_a": _digest("arm-a"),
            "arm_b": _digest("arm-b"),
        },
    )
    simulation = DesignPowerSimulation(
        development_seeds=(11, 12, 13),
        blinded_seed_deltas=(0.01, 0.02, 0.03),
        variance_estimate=0.0001,
        effect=DESIGN_EFFECT,
        pairs=DESIGN_PAIRS,
        studies=DESIGN_STUDIES,
        successes=9_000,
        power=0.9,
        power_ci_lo=0.89,
        power_ci_hi=0.91,
        passed=True,
        simulation_seed=DESIGN_SIMULATION_SEED,
        simulation_version=DESIGN_SIMULATION_VERSION,
    )
    return _make_receipt(development, simulation)


def valid_design_binding(
    design_receipt,
    *,
    high_entities: int = 800_000,
    tokens_per_parameter: int = 10,
) -> dict:
    load_label = {
        50_000: "load-50k",
        200_000: "load-200k",
        800_000: "load-800k",
    }[high_entities]
    return {
        "record_type": "gate5_development_binding",
        "schema_version": 1,
        "model_id": "d160m",
        "parameter_count": D160_PARAMETERS,
        "high_entities": high_entities,
        "tokens_per_parameter": tokens_per_parameter,
        "raw_token_count": D160_PARAMETERS * tokens_per_parameter,
        "inputs": {
            label: {
                "rows_sha256": rows_hash,
                "checkpoint_sha256": _digest(f"{label}-checkpoint"),
                "data_sha256": _digest(load_label),
            }
            for label, rows_hash in design_receipt.blinded_input_hashes.items()
        },
    }


def _smoke_report() -> dict:
    return {
        "bundle_byte_deterministic": True,
        "bundle_verified": True,
        "controls": [
            "correct",
            "shuffled_returns",
            "relevant_edge",
            "irrelevant_edge",
            "gold_path",
            "gold_returns",
            "no_query",
            "explicit_miss",
            "handle_swap",
            "entity_rename",
            "graph_isomorphism",
        ],
        "corpus_builds": 2,
        "corpus_byte_deterministic": True,
        "corpus_sha256": _digest("corpus"),
        "dense_steps": 2,
        "eval_cells": 22,
        "extracted_bundle_verified": True,
        "matrix_runs": 35,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
        "resume_compared_next_update": True,
        "resume_exact": True,
        "schemas_validated": [
            "freeze-v1.schema.json",
            "relational-asset-receipt-v1.schema.json",
            "relational-result-v1.schema.json",
            "run-config-v1.schema.json",
            "run-manifest-v1.schema.json",
        ],
        "shared_stream": True,
        "sidecar_sha256": {
            label: _digest(f"{label}-sidecar")
            for label in ("dense", "random", "selective", "split")
        },
        "sidecars": ["dense", "random", "selective", "split"],
        "split_steps": 2,
        "synthetic_run_count": 35,
        "verdict_branches": [
            "validated",
            "practical_null",
            "inconclusive",
            "invalid",
        ],
    }


def valid_gate_receipts(development: dict | None = None) -> dict[str, dict]:
    common = common_input_hashes()
    receipts = evaluate_development_gates(
        development_fixture() if development is None else development
    )
    design_receipt = valid_design_receipt()
    return {
        "gate_0": build_gate_0_receipt(
            _smoke_report(),
            input_hashes=common,
        ),
        **receipts,
        "gate_5": build_gate_5_receipt(
            design_receipt,
            input_hashes=common,
            gate_3_receipt=receipts["gate_3"],
            gate_4_receipt=receipts["gate_4"],
            development_binding=valid_design_binding(
                design_receipt,
                high_entities=receipts["gate_3"]["high_entities"],
                tokens_per_parameter=receipts["gate_4"][
                    "tokens_per_parameter"
                ],
            ),
        ),
    }


def valid_frozen_freeze(**overrides) -> FreezeManifest:
    low = overrides.pop("low_entities", 50_000)
    high = overrides.pop("high_entities", 800_000)
    confirmation = overrides.pop("confirmation_entities", 1_800_000)
    development = development_fixture()
    if low == 200_000:
        development["loads"][0]["dense_reasoning_composite"] = 0.70
        development["loads"][1]["dense_fact_recall"] = 0.85
    elif low != 50_000:
        raise ValueError("test helper supports only frozen Gate-3 low loads")
    receipts = valid_gate_receipts(development)
    assert receipts["gate_3"]["high_entities"] == high
    assert receipts["gate_3"]["confirmation_entities"] == confirmation
    return build_freeze_manifest(
        valid_source_provenance(),
        receipts,
        **overrides,
    )


def test_fixture_freeze_is_deterministic_and_explicitly_nonlaunchable():
    first = make_fixture_freeze()
    second = make_fixture_freeze()

    assert first == second
    assert first.status == "fixture"
    assert first.watermark == FIXTURE_WATERMARK == "NONLAUNCHABLE_FIXTURE"
    assert first.launchable is False
    assert first.gate_receipts == {}
    assert validate_freeze_manifest(first.to_dict()) == first


def test_real_freeze_binds_all_gate_receipts_and_selections():
    freeze = valid_frozen_freeze()

    assert freeze.status == "frozen"
    assert freeze.watermark is None
    assert freeze.launchable
    assert set(freeze.gate_receipts) == {
        "gate_0",
        "gate_1",
        "gate_2",
        "gate_3",
        "gate_4",
        "gate_5",
    }
    assert freeze.selected_mixture == (0.70, 0.15, 0.15)
    assert freeze.low_entities == 50_000
    assert freeze.high_entities == 800_000
    assert freeze.confirmation_entities == 1_800_000
    assert freeze.tokens_per_parameter == 10
    assert freeze.seeds == (1001, 1002, 1003, 1004, 1005)
    assert validate_freeze_manifest(freeze.to_dict()) == freeze


def test_gate_5_binds_design_rows_to_selected_load_and_budget():
    receipts = evaluate_development_gates(development_fixture())
    design = valid_design_receipt()
    binding = valid_design_binding(
        design,
        high_entities=receipts["gate_3"]["high_entities"],
        tokens_per_parameter=receipts["gate_4"]["tokens_per_parameter"],
    )
    binding["inputs"]["arm_a"]["data_sha256"] = _digest("wrong-load")

    with pytest.raises(ValueError, match="Gate 3|load|data"):
        build_gate_5_receipt(
            design,
            input_hashes=common_input_hashes(),
            gate_3_receipt=receipts["gate_3"],
            gate_4_receipt=receipts["gate_4"],
            development_binding=binding,
        )


def test_gate_5_rejects_the_exact_rounding_error_boundary(monkeypatch):
    design = valid_design_receipt()
    binding = valid_design_binding(design)
    monkeypatch.setattr(relational_gates, "D160_PARAMETERS", 500)
    binding["parameter_count"] = 500
    binding["raw_token_count"] = 5_001

    with pytest.raises(ValueError, match="rounding"):
        relational_gates._normalize_gate_5_development_binding(
            binding,
            design,
        )


@pytest.mark.parametrize("missing", [f"gate_{number}" for number in range(6)])
def test_real_freeze_requires_every_gate(missing):
    receipts = valid_gate_receipts()
    receipts.pop(missing)
    with pytest.raises(ValueError, match="gate|0.*5|exact"):
        build_freeze_manifest(valid_source_provenance(), receipts)


def test_real_freeze_rejects_failed_inconsistent_or_drifted_receipts():
    failed = valid_gate_receipts()
    failed_gate = copy.deepcopy(failed["gate_3"])
    failed_gate["passed"] = False
    failed_gate["decision"] = {
        **failed_gate["decision"],
        "passed": False,
    }
    from evals.relational_gates import rehash_gate_receipt

    failed["gate_3"] = rehash_gate_receipt(failed_gate)
    with pytest.raises(ValueError, match="gate_3.*pass|passed"):
        build_freeze_manifest(valid_source_provenance(), failed)

    inconsistent = valid_gate_receipts()
    gate_4 = copy.deepcopy(inconsistent["gate_4"])
    gate_4["input_hashes"]["analysis_sha256"] = _digest("other-analysis")
    inconsistent["gate_4"] = rehash_gate_receipt(gate_4)
    with pytest.raises(ValueError, match="input hash|agree|consistent"):
        build_freeze_manifest(valid_source_provenance(), inconsistent)

    wrong_load_binding = valid_gate_receipts()
    changed_gate_5 = copy.deepcopy(wrong_load_binding["gate_5"])
    changed_gate_5["gate_3_receipt_sha256"] = "f" * 64
    wrong_load_binding["gate_5"] = rehash_gate_receipt(
        changed_gate_5
    )
    with pytest.raises(ValueError, match="Gate 3|gate_3|high.load|binding"):
        build_freeze_manifest(
            valid_source_provenance(),
            wrong_load_binding,
        )

    drifted = valid_gate_receipts()
    drifted["gate_2"]["selected_mixture"][0] = 0.69
    with pytest.raises(ValueError, match="hash|decision"):
        build_freeze_manifest(valid_source_provenance(), drifted)


def test_real_freeze_rejects_dirty_revision_and_provenance_hash_mismatch():
    with pytest.raises(ValueError, match="clean|dirty"):
        build_freeze_manifest(
            valid_source_provenance(clean=False),
            valid_gate_receipts(),
        )

    provenance = valid_source_provenance()
    raw = provenance.to_dict()
    raw["artifact_sha256"]["analysis"] = _digest("wrong-analysis")
    with pytest.raises(ValueError, match="analysis|input hash|provenance"):
        build_freeze_manifest(
            SourceProvenance.from_dict(raw),
            valid_gate_receipts(),
        )


def test_freeze_publication_is_canonical_hash_bound_and_symlink_safe(tmp_path):
    freeze = valid_frozen_freeze()
    path = tmp_path / "freeze.json"
    write_freeze_manifest(path, freeze)

    assert load_freeze_manifest(path) == freeze
    assert json.loads(path.read_bytes()) == freeze.to_dict()

    raw = json.loads(path.read_bytes())
    raw["tokens_per_parameter"] = 20
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="hash|decision"):
        load_freeze_manifest(path)

    target = tmp_path / "target.json"
    target.write_text("{}\n")
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|regular|canonical"):
        write_freeze_manifest(link, freeze)


def test_protected_corpus_generation_rejects_missing_or_fixture_freeze_before_write(
    tmp_path,
):
    cfg = RelationalBuildConfig(
        n_entities=50_000,
        total_tokens=1_000,
        data_seed=1001,
        artifact_mode="protected",
    )
    destination = tmp_path / "protected-corpus"

    with pytest.raises(ValueError, match="freeze|required"):
        build_relational_corpus(
            cfg,
            object(),
            iter(()),
            destination,
            freeze_manifest=None,
        )
    assert not destination.exists()

    with pytest.raises(ValueError, match="nonlaunchable fixture|frozen"):
        build_relational_corpus(
            cfg,
            object(),
            iter(()),
            destination,
            freeze_manifest=make_fixture_freeze(),
        )
    assert not destination.exists()

    failed = valid_frozen_freeze().to_dict()
    failed["gate_receipts"]["gate_4"]["passed"] = False
    with pytest.raises(ValueError, match="Gate|gate|decision|hash|passed"):
        build_relational_corpus(
            cfg,
            object(),
            iter(()),
            destination,
            freeze_manifest=failed,
        )
    assert not destination.exists()


def test_protected_corpus_rejects_unfrozen_build_settings_before_write(
    tmp_path,
):
    from scripts.make_relational_manifest import round_raw_positions

    freeze = valid_frozen_freeze()
    total_tokens = round_raw_positions(
        freeze.model_parameters["d160m"] * freeze.tokens_per_parameter,
        tokens_per_step=freeze.tokens_per_step,
    ).actual_raw_positions
    cfg = RelationalBuildConfig(
        n_entities=freeze.low_entities,
        total_tokens=total_tokens,
        data_seed=freeze.seeds[0],
        world_size=32,
        artifact_mode="protected",
    )
    destination = tmp_path / "protected-corpus"

    with pytest.raises(ValueError, match="protected|setting|world"):
        build_relational_corpus(
            cfg,
            object(),
            iter(()),
            destination,
            relation_schema=object(),
            freeze_manifest=freeze,
        )
    assert not destination.exists()


def test_protected_corpus_manifest_metadata_binds_exact_matrix_build():
    from corpusgen.relational_build import _protected_build_metadata
    from scripts.make_relational_manifest import (
        matrix_plan_sha256,
        protected_build_metadata,
        round_raw_positions,
    )

    freeze = valid_frozen_freeze()
    total_tokens = round_raw_positions(
        freeze.model_parameters["d160m"] * freeze.tokens_per_parameter,
        tokens_per_step=freeze.tokens_per_step,
    ).actual_raw_positions
    cfg = RelationalBuildConfig(
        n_entities=freeze.low_entities,
        total_tokens=total_tokens,
        data_seed=freeze.seeds[0],
        artifact_mode="protected",
    )

    metadata = protected_build_metadata(
        freeze,
        model="d160m",
        load="low",
        entities=freeze.low_entities,
        seed=freeze.seeds[0],
    )
    assert _protected_build_metadata(cfg, freeze) == metadata
    assert metadata["matrix_plan_sha256"] == matrix_plan_sha256(freeze)


def test_protected_corpus_cli_binds_bed_lock_to_freeze(tmp_path):
    source_lock = tmp_path / "bed-lock.json"
    source_lock.write_bytes(b'{"snapshot":"pinned"}\n')
    digest = hashlib.sha256(source_lock.read_bytes()).hexdigest()
    freeze = SimpleNamespace(
        artifact_sha256={"source_lock": digest},
    )

    _validate_frozen_source_lock(source_lock, freeze)
    source_lock.write_bytes(b'{"snapshot":"different"}\n')
    with pytest.raises(ValueError, match="source lock|BED|freeze|hash"):
        _validate_frozen_source_lock(source_lock, freeze)


def test_gate_receipts_remain_individually_valid_in_frozen_manifest():
    freeze = valid_frozen_freeze()
    for name, receipt in freeze.gate_receipts.items():
        gate = int(name.removeprefix("gate_"))
        assert validate_gate_receipt(
            receipt,
            expected_gate=gate,
        ) == json.loads(canonical_json_bytes(receipt))
