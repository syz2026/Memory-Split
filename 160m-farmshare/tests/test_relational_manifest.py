from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.checkpoint_binding import canonical_shared_configuration_sha256
from evals.relational_controls import ControlID, EvalMode
from experiment.artifacts import canonical_sha256, sha256_file
from scripts.analyze_relational import (
    _load_complete_runs,
    _select_frozen_checkpoint,
    _validate_checkpoint_freeze_identity,
    _validate_run_config_against_manifest,
    expected_run_keys,
)
from scripts.freeze_relational_study import make_fixture_freeze
from scripts.make_relational_manifest import (
    RunConfig,
    RunManifest,
    assert_pair_fingerprints,
    build_manifest,
    entity_load_label,
    load_run_manifest,
    publish_run_configs,
    require_launchable,
    round_raw_positions,
    write_run_manifest,
)
from scripts.run_train import main as run_train_main
from scripts.run_train import resolve_relative_config
from tests.test_relational_freeze import valid_frozen_freeze
from tests.relational_asset_fixtures import stage_launchable_relational_assets
from train.trainer import ProvenanceError, validate_run_start


def test_exact_protected_matrix():
    manifest = build_manifest(make_fixture_freeze())

    assert len(manifest.runs) == 35
    assert Counter(run.condition for run in manifest.runs) == {
        "dense": 15,
        "split": 15,
        "random": 5,
    }
    assert {run.seed for run in manifest.runs} == {
        1001,
        1002,
        1003,
        1004,
        1005,
    }
    assert all(
        run.model == "d160m" and run.load == "high"
        for run in manifest.runs
        if run.condition == "random"
    )
    assert len({run.run_id for run in manifest.runs}) == 35
    assert len({run.key for run in manifest.runs}) == 35


def test_fixture_freeze_is_nonlaunchable():
    manifest = build_manifest(make_fixture_freeze())

    assert not manifest.launchable
    assert manifest.status == "fixture"
    assert manifest.asset_receipt is None
    assert all(run.stream_sha256 is None for run in manifest.runs)
    assert all(run.weights_sha256 is None for run in manifest.runs)
    assert all(
        len(run.stream_commitment_sha256) == 64
        and len(run.weights_commitment_sha256) == 64
        for run in manifest.runs
    )
    with pytest.raises(ValueError, match="nonlaunchable fixture"):
        require_launchable(manifest)


def test_launchable_manifest_requires_post_build_asset_receipt():
    with pytest.raises(ValueError, match="asset receipt"):
        build_manifest(valid_frozen_freeze())


def test_renamed_commitment_fields_preserve_task9_recipe_hashes():
    manifest = build_manifest(make_fixture_freeze())
    run = manifest.runs[0]

    assert run.stream_commitment_sha256 == canonical_sha256(
        {
            "record_type": "relational_stream_commitment",
            "schema_version": 1,
            "freeze_sha256": manifest.freeze_sha256,
            "corpus_recipe_sha256": manifest.freeze.artifact_sha256[
                "corpus_recipe"
            ],
            "model": run.model,
            "load_role": run.load,
            "entities": run.entities,
            "data_seed": run.data_seed,
            "raw_positions": run.actual_raw_positions,
        }
    )
    assert run.weights_commitment_sha256 == canonical_sha256(
        {
            "record_type": "relational_sidecar_commitment",
            "schema_version": 1,
            "stream_sha256": run.stream_commitment_sha256,
            "condition": run.condition,
            "sidecar_recipe_sha256": manifest.freeze.artifact_sha256[
                f"{run.condition}_sidecar_recipe"
            ],
        }
    )


def test_production_receipt_finalizes_exact_35_run_manifest(tmp_path):
    freeze, receipt, manifest, data_root = stage_launchable_relational_assets(
        tmp_path
    )

    assert manifest.launchable
    assert manifest.freeze_sha256 == freeze.freeze_sha256
    assert manifest.asset_receipt == receipt
    assert len(manifest.runs) == 35
    assert len(receipt.assets) == 50
    for run in manifest.runs:
        assert run.stream_sha256 == sha256_file(data_root / run.data_rel)
        assert run.weights_sha256 == sha256_file(data_root / run.weights_rel)
        assert run.stream_sha256 != run.stream_commitment_sha256
        assert run.weights_sha256 != run.weights_commitment_sha256
        assert {
            "stream_sha256",
            "stream_commitment_sha256",
        } <= set(run.pair_material())
        assert "weights_sha256" not in run.pair_material()
        assert "weights_commitment_sha256" not in run.pair_material()
    assert_pair_fingerprints(manifest)


def test_finalized_configs_are_location_independent(tmp_path):
    _freeze_a, _receipt_a, first, _root_a = (
        stage_launchable_relational_assets(tmp_path / "first")
    )
    _freeze_b, _receipt_b, second, _root_b = (
        stage_launchable_relational_assets(tmp_path / "second")
    )

    assert [run.to_dict() for run in first.runs] == [
        run.to_dict() for run in second.runs
    ]
    assert first.to_dict() == second.to_dict()


def test_configs_are_relative_and_pair_fingerprints_match():
    manifest = build_manifest(make_fixture_freeze())

    assert all(
        not Path(value).is_absolute()
        and ".." not in Path(value).parts
        and "\\" not in value
        for run in manifest.runs
        for value in run.relative_paths()
    )
    assert_pair_fingerprints(manifest)

    groups = defaultdict(list)
    for run in manifest.runs:
        groups[(run.model, run.load, run.seed)].append(run)
    for runs in groups.values():
        assert len({run.pair_fingerprint for run in runs}) == 1
        assert {run.stream_sha256 for run in runs} == {None}
        assert {run.weights_sha256 for run in runs} == {None}
        assert len({run.stream_commitment_sha256 for run in runs}) == 1
        assert len({run.weights_commitment_sha256 for run in runs}) == len(runs)


def test_generated_twins_match_existing_shared_configuration_contract():
    manifest = build_manifest(make_fixture_freeze())
    grouped = defaultdict(list)
    for run in manifest.runs:
        grouped[(run.model, run.load, run.seed)].append(run)

    for runs in grouped.values():
        assert len(
            {
                canonical_shared_configuration_sha256(run.to_dict())
                for run in runs
            }
        ) == 1


def test_pair_fingerprint_binds_every_required_pair_field():
    manifest = build_manifest(make_fixture_freeze())
    raw = manifest.to_dict()
    split = next(
        run
        for run in raw["runs"]
        if run["condition"] == "split"
        and run["model"] == "d160m"
        and run["load_role"] == "low"
    )
    split["optimizer"]["lr"] = split["optimizer"]["lr"] / 2

    with pytest.raises(
        ValueError,
        match="pair fingerprint|config.*hash|frozen contract",
    ):
        RunManifest.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_rel", "/absolute/train.bin"),
        ("weights_rel", "../outside.weights.bin"),
        ("out_rel", "runs\\windows\\path"),
    ],
)
def test_run_config_rejects_absolute_traversing_and_nonportable_paths(
    field,
    value,
):
    run = build_manifest(make_fixture_freeze()).runs[0]
    raw = run.to_dict()
    raw[field] = value

    with pytest.raises(ValueError, match="relative|traversal|portable|path"):
        RunConfig.from_dict(raw)


def test_manifest_uses_dynamic_gate_3_load_labels(tmp_path):
    freeze = valid_frozen_freeze(low_entities=200_000)
    _, _, manifest, _ = stage_launchable_relational_assets(
        tmp_path,
        freeze=freeze,
    )

    low = [run for run in manifest.runs if run.load == "low"]
    assert low
    assert {run.entities for run in low} == {200_000}
    assert {run.load_label for run in low} == {"n200k"}
    assert entity_load_label(1_800_000) == "n1p8m"
    assert {run.key for run in manifest.runs} == {
        (
            run.model,
            run.condition,
            run.load_label,
            run.seed,
        )
        for run in manifest.runs
    }
    assert expected_run_keys(manifest) == {run.key for run in manifest.runs}


def test_raw_tokens_round_to_steps_and_reject_excess_error():
    requested = 162_220_800 * 10
    budget = round_raw_positions(requested, tokens_per_step=524_288)

    assert budget.requested_raw_positions == requested
    assert budget.actual_raw_positions == budget.steps * 524_288
    assert budget.rounding_error_fraction < 0.0002

    with pytest.raises(ValueError, match="0.02%|rounding"):
        round_raw_positions(1, tokens_per_step=524_288)


def test_rounding_rejects_the_exact_frozen_error_boundary():
    with pytest.raises(ValueError, match="0.02%|rounding"):
        round_raw_positions(5_000, tokens_per_step=4_999)

    accepted = round_raw_positions(5_001, tokens_per_step=5_000)
    assert accepted.rounding_error_fraction < 0.0002


def test_run_config_schema_rejects_the_exact_rounding_error_boundary():
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "run-config-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    rounding_schema = schema["properties"]["rounding_error_fraction"]

    assert rounding_schema == {
        "type": "number",
        "minimum": 0,
        "exclusiveMaximum": 0.0002,
    }
    assert not 0.0002 < rounding_schema["exclusiveMaximum"]
    assert 0.00019999999999999998 < rounding_schema["exclusiveMaximum"]


def test_manifest_and_config_hashes_reject_any_byte_level_drift(tmp_path):
    _, _, manifest, _ = stage_launchable_relational_assets(tmp_path / "assets")
    path = tmp_path / "run-manifest.json"
    write_run_manifest(path, manifest)

    loaded = load_run_manifest(path)
    assert loaded == manifest
    assert json.loads(path.read_bytes()) == manifest.to_dict()

    raw = json.loads(path.read_bytes())
    raw["runs"][0]["decode_budget"] = 7
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="hash|fingerprint|decode"):
        load_run_manifest(path)


def test_manifest_rejects_freeze_hash_and_status_tampering(tmp_path):
    _, _, manifest, _ = stage_launchable_relational_assets(tmp_path)
    raw = manifest.to_dict()
    raw["runs"][0]["freeze_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="freeze.*hash|config.*hash"):
        RunManifest.from_dict(raw)

    fixture = build_manifest(make_fixture_freeze()).to_dict()
    fixture["launchable"] = True
    with pytest.raises(ValueError, match="fixture|launchable|hash"):
        RunManifest.from_dict(fixture)


def test_analyzer_binds_each_loaded_config_to_manifest_freeze():
    manifest = build_manifest(make_fixture_freeze())
    run = manifest.runs[0]
    loaded = {
        **run.to_dict(),
        "train_bin": f"/runtime/data/{run.data_rel}",
        "train_weights": f"/runtime/data/{run.weights_rel}",
        "out_dir": f"/runtime/out/{run.out_rel}",
        "micro_batch_size": 8,
        "tokens_per_step": run.tokens_per_step,
        "max_steps": run.steps,
        "total_tokens": run.actual_raw_positions,
        "lr": run.optimizer["lr"],
        "weight_decay": run.optimizer["weight_decay"],
        "warmup_steps": run.scheduler["warmup_steps"],
        "ctx": run.architecture["ctx"],
    }
    assert _validate_run_config_against_manifest(loaded, manifest) == run.key

    loaded["lr"] = run.optimizer["lr"] / 2
    with pytest.raises(ValueError, match="runtime|lr|manifest"):
        _validate_run_config_against_manifest(loaded, manifest)

    loaded["lr"] = run.optimizer["lr"]
    loaded["freeze_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="freeze|manifest|config"):
        _validate_run_config_against_manifest(loaded, manifest)


def test_analyzer_selects_the_exact_frozen_checkpoint_not_the_latest():
    key = (EvalMode.MEMORY_ON, ControlID.CORRECT)
    earlier = {key: SimpleNamespace(raw_token_count=5)}
    exact = {key: SimpleNamespace(raw_token_count=10)}
    later = {key: SimpleNamespace(raw_token_count=20)}

    assert _select_frozen_checkpoint((earlier, exact, later), 10) is exact
    with pytest.raises(ValueError, match="frozen|raw-token|checkpoint"):
        _select_frozen_checkpoint((earlier, later), 10)
    with pytest.raises(ValueError, match="duplicate|frozen|checkpoint"):
        _select_frozen_checkpoint((exact, exact), 10)


def test_analyzer_binds_evaluation_relation_schema_to_freeze():
    manifest = build_manifest(make_fixture_freeze())
    anchor = SimpleNamespace(
        relation_schema_sha256=manifest.freeze.artifact_sha256[
            "relation_schema"
        ]
    )
    _validate_checkpoint_freeze_identity(anchor, manifest)

    anchor.relation_schema_sha256 = "f" * 64
    with pytest.raises(ValueError, match="relation schema|freeze"):
        _validate_checkpoint_freeze_identity(anchor, manifest)


def test_analyzer_rejects_symlinked_runs_root(tmp_path):
    real_root = tmp_path / "real-runs"
    real_root.mkdir()
    linked_root = tmp_path / "linked-runs"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|canonical"):
        _load_complete_runs(linked_root, build_manifest(make_fixture_freeze()))


def test_run_train_resolves_only_launchable_relative_configs(tmp_path):
    fixture_manifest = build_manifest(make_fixture_freeze())
    fixture_run = fixture_manifest.runs[0]
    with pytest.raises(ValueError, match="nonlaunchable fixture"):
        resolve_relative_config(
            fixture_run.to_dict(),
            run_manifest=fixture_manifest,
            environ={
                "DATA_ROOT": str(tmp_path / "data"),
                "OUT_ROOT": str(tmp_path / "out"),
            },
        )

    _, _, frozen_manifest, data_root = stage_launchable_relational_assets(
        tmp_path / "assets"
    )
    frozen_run = frozen_manifest.runs[0]
    out_root = tmp_path / "out"
    out_root.mkdir()
    resolved = resolve_relative_config(
        frozen_run.to_dict(),
        run_manifest=frozen_manifest,
        environ={
            "DATA_ROOT": str(data_root),
            "OUT_ROOT": str(out_root),
        },
    )
    assert resolved["train_bin"] == str(
        data_root / frozen_run.data_rel
    )
    assert resolved["train_weights"] == str(
        data_root / frozen_run.weights_rel
    )
    assert resolved["out_dir"] == str(out_root / frozen_run.out_rel)
    assert resolved["data_root"] == str(data_root)
    assert resolved["out_root"] == str(out_root)
    assert resolved["ledger_root"] == str(out_root)
    assert (
        resolved["source_tree_sha256"]
        == frozen_manifest.freeze.source_provenance.source_tree_sha256
    )
    train_bin = Path(resolved["train_bin"])
    validate_run_start(resolved, resume="none")
    changed = bytearray(train_bin.read_bytes())
    changed[0] ^= 1
    train_bin.write_bytes(changed)
    with pytest.raises(ProvenanceError, match="stream SHA-256.*bytes"):
        validate_run_start(resolved, resume="none")
    assert not Path(resolved["out_dir"]).exists()

    dangling_root = tmp_path / "dangling-data"
    dangling_root.symlink_to(
        tmp_path / "missing-data",
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="symlink"):
        resolve_relative_config(
            frozen_run.to_dict(),
            run_manifest=frozen_manifest,
            environ={
                "DATA_ROOT": str(dangling_root),
                "OUT_ROOT": str(tmp_path / "out"),
            },
        )

    real_data_root = tmp_path / "real-data"
    real_data_root.mkdir()
    linked_data_root = tmp_path / "linked-data"
    linked_data_root.symlink_to(real_data_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|canonical"):
        resolve_relative_config(
            frozen_run.to_dict(),
            run_manifest=frozen_manifest,
            environ={
                "DATA_ROOT": str(linked_data_root / "nested"),
                "OUT_ROOT": str(tmp_path / "out"),
            },
        )

    forged_fixture = frozen_run.to_dict()
    forged_fixture["run_id"] = f"{frozen_run.run_id}-forged"
    forged_fixture["config_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in forged_fixture.items()
            if key != "config_sha256"
        }
    )
    with pytest.raises(ValueError, match="manifest|membership|member"):
        resolve_relative_config(
            forged_fixture,
            run_manifest=frozen_manifest,
            environ={
                "DATA_ROOT": str(tmp_path / "data"),
                "OUT_ROOT": str(tmp_path / "out"),
            },
        )


def test_run_train_rejects_dirty_source_before_protected_training(
    tmp_path,
    monkeypatch,
):
    _, _, manifest, data_root = stage_launchable_relational_assets(
        tmp_path / "assets"
    )
    published = publish_run_configs(tmp_path / "published", manifest)
    run = manifest.runs[0]
    out_root = tmp_path / "out"
    out_root.mkdir()
    trained = False
    clean_requirements = []

    def reject_dirty_source(_root, _provenance, *, require_clean=False):
        clean_requirements.append(require_clean)
        if require_clean:
            raise ValueError("untracked source file")

    def fake_train(_cfg, *, resume):
        nonlocal trained
        trained = True
        return SimpleNamespace(step=0, out_dir=out_root)

    monkeypatch.setattr(
        "experiment.provenance.verify_source_provenance",
        reject_dirty_source,
    )
    monkeypatch.setattr("scripts.run_train.train", fake_train)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("OUT_ROOT", str(out_root))

    status = run_train_main(
        [
            "--config",
            str(published / "configs" / f"{run.run_id}.json"),
            "--run-manifest",
            str(published / "run-manifest.json"),
            "--resume",
            "none",
        ]
    )

    assert status == 1
    assert clean_requirements == [True]
    assert not trained


def _resolved_protected_asset_config(tmp_path):
    _, _, manifest, data_root = stage_launchable_relational_assets(
        tmp_path / "assets"
    )
    run = manifest.runs[0]
    out_root = tmp_path / "out"
    out_root.mkdir()
    resolved = resolve_relative_config(
        run.to_dict(),
        run_manifest=manifest,
        environ={
            "DATA_ROOT": str(data_root),
            "OUT_ROOT": str(out_root),
        },
    )
    return (
        resolved,
        Path(resolved["train_bin"]),
        Path(resolved["train_weights"]),
    )


@pytest.mark.parametrize(
    ("asset", "match"),
    [
        ("stream", "stream SHA-256.*bytes"),
        ("weights", "weights SHA-256.*bytes"),
    ],
)
def test_manifest_derived_config_rechecks_canonical_asset_bytes(
    tmp_path,
    asset,
    match,
):
    resolved, train_bin, train_weights = _resolved_protected_asset_config(
        tmp_path
    )
    validate_run_start(resolved, resume="none")
    path = train_bin if asset == "stream" else train_weights
    changed = bytearray(path.read_bytes())
    changed[0] ^= 1
    path.write_bytes(changed)

    with pytest.raises(ProvenanceError, match=match):
        validate_run_start(resolved, resume="none")

    assert not Path(resolved["out_dir"]).exists()


def test_config_generation_rejects_invalid_or_failed_freeze_receipt():
    raw = valid_frozen_freeze().to_dict()
    raw["gate_receipts"]["gate_4"]["passed"] = False
    with pytest.raises(ValueError, match="hash|gate|Gate|passed|decision"):
        build_manifest(raw)


def test_run_manifest_schema_documents_exact_closed_contract():
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "run-manifest-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())

    assert schema["additionalProperties"] is False
    assert schema["properties"]["runs"]["minItems"] == 35
    assert schema["properties"]["runs"]["maxItems"] == 35
    assert schema["properties"]["asset_receipt"]["oneOf"][1]["$ref"].endswith(
        "relational-asset-receipt-v1.schema.json"
    )
    assert "matrix_plan_sha256" in schema["required"]

    config_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "run-config-v1.schema.json"
        ).read_text()
    )
    assert {
        "stream_commitment_sha256",
        "weights_commitment_sha256",
    } <= set(config_schema["required"])
    assert config_schema["properties"]["stream_sha256"]["type"] == [
        "string",
        "null",
    ]
