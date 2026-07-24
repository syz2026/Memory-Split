from __future__ import annotations

import hashlib
import json

import pytest

from evals.confirmatory.contracts import canonical_json_bytes
from evals.confirmatory.run_binding import (
    RUN_BINDING_FIELDS,
    RUN_BINDING_SCHEMA,
    build_run_binding,
    validate_run_binding,
    write_run_binding,
)
from evals.confirmatory.runner import _load_run_binding


def test_run_binding_builds_evaluator_exact_metadata_and_publishes_once(tmp_path):
    run = tmp_path / "run"
    nested = run / "artifacts"
    nested.mkdir(parents=True)
    checkpoint = nested / "terminal.pt"
    configuration = run / "config.json"
    checkpoint.write_bytes(b"terminal checkpoint bytes")
    configuration.write_bytes(
        canonical_json_bytes({"condition": "split90", "seed": 9})
    )

    value = build_run_binding(
        run_root=run,
        run_id="memorysplit-v3-360m-s9-split90",
        checkpoint_path="artifacts/terminal.pt",
        configuration_path=configuration,
        route_dose_sha256="a" * 64,
        corpus_sha256="b" * 64,
        code_sha256="c" * 64,
        seed=9,
        condition_id="split90",
    )

    assert set(value) == set(RUN_BINDING_FIELDS)
    assert value["record_type"] == RUN_BINDING_SCHEMA
    assert value["checkpoint_path"] == "artifacts/terminal.pt"
    assert value["configuration_path"] == "config.json"
    assert value["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert value["configuration_sha256"] == hashlib.sha256(
        configuration.read_bytes()
    ).hexdigest()

    output = run / "run.json"
    assert write_run_binding(output, value) == output
    assert output.read_bytes() == canonical_json_bytes(value)
    loaded = _load_run_binding(run)
    assert loaded.run_id == value["run_id"]
    assert loaded.checkpoint_sha256 == value["checkpoint_sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        write_run_binding(output, value)


def test_run_binding_rejects_schema_drift_and_unsafe_checkpoint_paths(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "terminal.pt"
    configuration = run / "config.json"
    checkpoint.write_bytes(b"checkpoint")
    configuration.write_text(
        json.dumps({"condition": "dense", "seed": 0}),
        encoding="ascii",
    )
    arguments = {
        "run_root": run,
        "run_id": "memorysplit-v3-360m-s0-dense",
        "checkpoint_path": checkpoint,
        "configuration_path": configuration,
        "route_dose_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "code_sha256": "c" * 64,
        "seed": 0,
        "condition_id": "dense",
    }
    value = build_run_binding(**arguments)
    with pytest.raises(ValueError, match="fields"):
        validate_run_binding({**value, "unexpected": True})

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="inside"):
        build_run_binding(**{**arguments, "checkpoint_path": outside})

    alias = run / "alias.pt"
    alias.symlink_to(checkpoint)
    with pytest.raises(OSError):
        build_run_binding(**{**arguments, "checkpoint_path": alias})
