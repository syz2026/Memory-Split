import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import corpusgen.parallel.production as production_module
from corpusgen.parallel import (
    FROZEN_LANES,
    FROZEN_TOKEN_QUOTAS,
    FROZEN_TOTAL_TOKENS,
    ProductionPreflightError,
    ProductionRecipe,
    ProductionRenderer,
    build_production_catalog,
    build_production_corpus,
    load_production_recipe,
    load_production_source_manifest,
    production_preflight,
    render_metadata,
    require_production_preflight,
    seal_production_sources,
    verify_production_corpus,
)
from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning.proofs import EqualityPremise, solve_slot_equality


def _testing_recipe() -> ProductionRecipe:
    quotas = {lane: 4 for lane in FROZEN_LANES}
    locks = {lane: (f"test_{lane}",) for lane in FROZEN_LANES}
    return ProductionRecipe.for_testing(
        quotas,
        update_tokens=8,
        required_source_locks=locks,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _solver_verification(lane: str) -> dict:
    premises = (
        EqualityPremise(f"{lane}:left", 0, "same"),
        EqualityPremise(f"{lane}:right", 1, "same"),
    )
    return {
        "family": "slot_equality",
        "kind": "solver",
        "premises": [
            {
                "fact_id": premise.fact_id,
                "slot": premise.slot,
                "type": "equality",
                "value": premise.value,
            }
            for premise in premises
        ],
        "proof": solve_slot_equality(premises).as_dict(),
    }


def _materialize_testing_source_root(
    root: Path,
    recipe: ProductionRecipe,
) -> None:
    materialized = root / "materialized"
    ledgers = root / "ledgers"
    locks = root / "locks"
    materialized.mkdir(parents=True)
    ledgers.mkdir()
    locks.mkdir()

    source_ids = {
        source_id
        for _lane, lane_locks in recipe.required_source_locks
        for source_id in lane_locks
    }
    for source_id in sorted(source_ids):
        source_payload = source_id.encode("utf-8")
        lock = {
            "artifacts": [
                {
                    "bytes": len(source_payload),
                    "path": f"{source_id}.source",
                    "sha256": hashlib.sha256(source_payload).hexdigest(),
                }
            ],
            "format": "memorysplit-v2-source-lock-v1",
            "kind": "generator",
            "policy": {"testing": True},
            "repository": f"https://example.invalid/{source_id}",
            "revision": hashlib.sha1(source_payload).hexdigest(),
            "source_id": source_id,
        }
        (locks / f"{source_id}.lock.json").write_bytes(
            canonical_json_bytes(lock)
        )

    for lane_index, lane in enumerate(recipe.lanes):
        quota = recipe.quota_by_lane[lane]
        tokens = tuple(lane_index * 10 + index for index in range(quota))
        (materialized / f"{lane}.tokens.bin").write_bytes(
            struct.pack(f"<{quota}H", *tokens)
        )
        (materialized / f"{lane}.split90.weights.bin").write_bytes(
            bytes(quota)
        )
        fact_id = f"{lane}:fact"
        _write_jsonl(
            ledgers / f"{lane}.routes.jsonl",
            [
                {
                    "burden_bits": {"denominator": 1, "numerator": 1},
                    "external": True,
                    "fact_id": fact_id,
                }
            ],
        )
        _write_jsonl(
            ledgers / f"{lane}.masks.jsonl",
            [{"end": quota, "fact_id": fact_id, "start": 0}],
        )
        if lane in recipe.reasoning_lanes:
            verification = _solver_verification(lane)
        elif lane == recipe.objective_lane:
            verification = {
                "answer": {"value": lane},
                "kind": "objective_answer",
                "reference_answer": {"value": lane},
                "validator": "canonical_exact_match",
            }
        else:
            continue
        _write_jsonl(
            ledgers / f"{lane}.verification.jsonl",
            [
                {
                    "record_id": f"{lane}:record",
                    "source_id": recipe.locks_by_lane[lane][0],
                    "token_end": quota,
                    "token_start": 0,
                    "verification": verification,
                }
            ],
        )


def _sidecar_bytes(root: Path, receipt: dict, name: str) -> bytes:
    sidecar = next(item for item in receipt["sidecar_sets"] if item["name"] == name)
    return b"".join(
        (root / artifact["path"]).read_bytes()
        for artifact in sidecar["artifacts"]
    )


def test_authoritative_recipe_retains_exact_eight_lane_budget():
    recipe = load_production_recipe()

    assert recipe.lanes == FROZEN_LANES
    assert recipe.token_quotas == FROZEN_TOKEN_QUOTAS
    assert recipe.total_tokens == FROZEN_TOTAL_TOKENS == 7_120_879_616
    assert recipe.update_tokens * recipe.optimizer_steps == recipe.total_tokens
    assert recipe.reasoning_lanes == {
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
    }


def test_missing_production_sources_fail_with_actionable_preflight(tmp_path):
    recipe = _testing_recipe()
    source_root = tmp_path / "source"
    source_root.mkdir()

    report = production_preflight(source_root, recipe=recipe)

    assert report["ready"] is False
    assert report["issues"][0]["code"] == "source_manifest_unavailable"
    assert "source-production" in report["issues"][0]["action"]
    assert set(report["required_paths"])
    with pytest.raises(ProductionPreflightError) as captured:
        require_production_preflight(source_root, recipe=recipe)
    assert captured.value.report == report
    with pytest.raises(ProductionPreflightError) as seal_error:
        seal_production_sources(source_root, recipe=recipe)
    assert seal_error.value.report["ready"] is False
    assert not (source_root / "source-manifest.json").exists()


@pytest.mark.parametrize("corruption", ["cycle_fill", "solver_proof"])
def test_source_sealing_rejects_reasoning_reuse_or_failed_solver(
    tmp_path,
    corruption,
):
    recipe = _testing_recipe()
    source_root = tmp_path / "source"
    _materialize_testing_source_root(source_root, recipe)
    lane = "verified_synthetic_multihop"
    verification_path = source_root / "ledgers" / f"{lane}.verification.jsonl"
    verification = _solver_verification(lane)
    if corruption == "cycle_fill":
        rows = [
            {
                "record_id": f"{lane}:reused",
                "source_id": recipe.locks_by_lane[lane][0],
                "token_end": 2,
                "token_start": 0,
                "verification": verification,
            },
            {
                "record_id": f"{lane}:reused",
                "source_id": recipe.locks_by_lane[lane][0],
                "token_end": 4,
                "token_start": 2,
                "verification": verification,
            },
        ]
    else:
        verification["proof"]["conclusion"]["equal"] = False
        rows = [
            {
                "record_id": f"{lane}:record",
                "source_id": recipe.locks_by_lane[lane][0],
                "token_end": 4,
                "token_start": 0,
                "verification": verification,
            }
        ]
    _write_jsonl(verification_path, rows)

    with pytest.raises(ProductionPreflightError) as captured:
        seal_production_sources(source_root, recipe=recipe)

    messages = [
        issue["message"]
        for issue in captured.value.report["issues"]
        if issue.get("lane") == lane
    ]
    assert messages
    expected = "cycle-fill" if corruption == "cycle_fill" else "solver replay"
    assert expected in messages[0]
    assert not (source_root / "source-manifest.json").exists()


def test_tiny_production_build_is_locked_exact_and_idempotent(tmp_path):
    recipe = _testing_recipe()
    source_root = tmp_path / "source"
    _materialize_testing_source_root(source_root, recipe)

    report = seal_production_sources(source_root, recipe=recipe)
    assert report["ready"] is True
    assert report["observed_total_tokens"] == recipe.total_tokens == 32
    assert report["solver_verified_records"] == 3
    assert report["objective_verified_records"] == 1
    assert report["split90_zero_tokens"] == recipe.total_tokens

    manifest = load_production_source_manifest(source_root, recipe=recipe)
    catalog = build_production_catalog(
        manifest,
        recipe=recipe,
        chunk_tokens=2,
    )
    assert len(catalog.records) == 16
    assert tuple(dict.fromkeys(record.lane for record in catalog.records)) == FROZEN_LANES
    with ProductionRenderer(manifest, recipe) as renderer:
        metadata = render_metadata(catalog, renderer, workers=3)
        assert sum(record.token_length for record in metadata) == recipe.total_tokens
        assert all("source-locked" in record.flags for record in metadata)

    destination = tmp_path / "corpus"
    work_dir = tmp_path / "work"
    receipt = build_production_corpus(
        source_root,
        destination,
        work_dir,
        recipe=recipe,
        workers=3,
        shard_count=2,
        chunk_tokens=2,
    )

    assert receipt["format"] == "memorysplit-parallel-corpus-v2"
    assert receipt["logical_tokens"] == receipt["packed_tokens"] == 32
    assert receipt["padding_tokens"] == 0
    assert [item["name"] for item in receipt["sidecar_sets"]] == [
        "dense_target_weights",
        "split90_target_weights",
    ]
    assert _sidecar_bytes(
        destination, receipt, "dense_target_weights"
    ) == b"\x01" * 32
    assert _sidecar_bytes(
        destination, receipt, "split90_target_weights"
    ) == bytes(32)
    assert (
        verify_production_corpus(
            destination,
            recipe=recipe,
            source_manifest_sha256=manifest.sha256,
            expected_build_id=receipt["build_id"],
        )
        == receipt
    )

    rerun = build_production_corpus(
        source_root,
        destination,
        work_dir,
        recipe=recipe,
        workers=1,
        shard_count=2,
        chunk_tokens=2,
    )
    assert rerun == receipt


def test_relative_source_root_can_be_sealed(tmp_path, monkeypatch):
    recipe = _testing_recipe()
    monkeypatch.chdir(tmp_path)
    source_root = Path("source")
    _materialize_testing_source_root(source_root, recipe)

    report = seal_production_sources(source_root, recipe=recipe)

    assert report["ready"] is True
    assert Path("source/source-manifest.json").is_file()
    assert require_production_preflight(source_root, recipe=recipe)["ready"] is True


def test_checked_upstream_locks_satisfy_production_lock_schema():
    root = Path(__file__).resolve().parents[1] / "sources" / "memorysplit-v2"
    checked_locks = {
        "fineweb_edu": "fineweb-edu.lock.json",
        "finemath": "finemath.lock.json",
        "objective_auxiliary": "objective-auxiliaries.lock.json",
        "wikidata5m": "wikidata5m-complete-once.lock.json",
    }

    for source_id, filename in checked_locks.items():
        production_module._validate_source_lock(
            source_id,
            (root / filename).read_bytes(),
        )


def test_checked_source_locks_are_explicitly_partial_and_hash_bound():
    root = Path(__file__).resolve().parents[1]
    lock_root = root / "sources" / "memorysplit-v2"
    source_set = json.loads((lock_root / "source-set.lock.json").read_bytes())
    recipe_path = root / "configs" / "reasoning-dataset-v2.json"

    assert source_set["dataset_id"] == "memorysplit-v2-frozen-upstream-sources"
    assert source_set["production_readiness"]["ready"] is False
    assert source_set["production_readiness"]["missing_materialized_lanes"] == list(
        FROZEN_LANES
    )
    assert source_set["production_readiness"]["missing_source_locks"] == [
        "synthetic_graph_generator",
        "verified_synthetic_multihop_generator",
        "wikidata_path_reasoning_generator",
        "relational_refinement_generator",
        "reasoning_solver",
    ]
    assert source_set["scientific_requirements"]["sha256"] == hashlib.sha256(
        recipe_path.read_bytes()
    ).hexdigest()
    for lock in source_set["source_locks"]:
        assert hashlib.sha256((lock_root / lock["path"]).read_bytes()).hexdigest() == lock[
            "sha256"
        ]


def test_production_cli_reports_missing_sources_without_publication(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_parallel_corpus.py"
    wrapper = root / "scripts" / "build_memorysplit_v2_production.sh"
    source_root = tmp_path / "missing-source"
    source_root.mkdir()

    preflight = subprocess.run(
        [
            sys.executable,
            str(script),
            "preflight-production",
            "--source-root",
            str(source_root),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert preflight.returncode == 2
    assert json.loads(preflight.stdout)["ready"] is False

    source = subprocess.run(
        [
            sys.executable,
            str(script),
            "source-production",
            "--source-root",
            str(source_root),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert source.returncode == 2
    assert json.loads(source.stderr)["ready"] is False
    assert not (source_root / "source-manifest.json").exists()
    assert os.access(wrapper, os.X_OK)
