from __future__ import annotations

import json

import pytest

from scripts.verify_v2_readiness import REQUIRED_GATES, evaluate_readiness, main


def _evidence(state: str = "passed") -> dict:
    return {
        "schema_version": 1,
        "artifact_hashes": {
            "code": "a" * 64,
            "corpus": "b" * 64,
            "evaluation": "c" * 64,
            "preregistration": "d" * 64,
        },
        "gates": {
            gate: {"state": state, "evidence": f"{gate}.json"}
            for gate in REQUIRED_GATES
        },
    }


def test_readiness_requires_every_gate_and_bound_hash():
    result = evaluate_readiness(_evidence())

    assert result == {
        "schema_version": 1,
        "status": "ready_for_protected_launch",
        "protected_launch_allowed": True,
        "failed_gates": [],
        "missing_gates": [],
        "unbound_artifacts": [],
    }


def test_missing_or_pending_evidence_is_incomplete():
    evidence = _evidence()
    del evidence["gates"]["six_29m_diagnostics"]
    evidence["gates"]["ood_seal"]["state"] = "pending"
    evidence["artifact_hashes"]["corpus"] = None

    result = evaluate_readiness(evidence)

    assert result["status"] == "incomplete"
    assert result["protected_launch_allowed"] is False
    assert result["missing_gates"] == ["ood_seal", "six_29m_diagnostics"]
    assert result["unbound_artifacts"] == ["corpus"]


def test_measured_failure_precedes_incomplete():
    evidence = _evidence()
    evidence["gates"]["semantic_closure"]["state"] = "failed"
    evidence["gates"]["ood_seal"]["state"] = "pending"

    result = evaluate_readiness(evidence)

    assert result["status"] == "invalid"
    assert result["failed_gates"] == ["semantic_closure"]
    assert result["protected_launch_allowed"] is False


def test_contract_is_strict_and_hashes_are_real_sha256():
    evidence = _evidence()
    evidence["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        evaluate_readiness(evidence)

    evidence = _evidence()
    evidence["artifact_hashes"]["code"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_readiness(evidence)


def test_cli_emits_one_json_object(tmp_path, capsys):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")

    assert main(["--evidence", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["protected_launch_allowed"] is True
