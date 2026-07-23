#!/usr/bin/env python
"""Fail-closed protected-launch readiness gate for MemorySplit v2."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


REQUIRED_GATES = (
    "scientific_contract",
    "route_dose",
    "semantic_closure",
    "proof_verification",
    "ood_seal",
    "corpus_identity",
    "paired_training",
    "checkpoint_resume",
    "evaluation_validity",
    "six_29m_diagnostics",
)
REQUIRED_ARTIFACTS = ("code", "corpus", "evaluation", "preregistration")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_fields(value: object, expected: set[str], name: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields do not match the v2 contract")
    return value


def evaluate_readiness(raw: Mapping) -> dict:
    """Validate evidence and return the launch decision.

    A measured failure is ``invalid`` even when other evidence is missing.
    Missing, pending, or unbound evidence is ``incomplete``. Neither permits a
    protected launch.
    """

    value = _strict_fields(
        raw,
        {"schema_version", "artifact_hashes", "gates"},
        "readiness evidence",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported readiness schema_version")

    artifacts = _strict_fields(
        value["artifact_hashes"],
        set(REQUIRED_ARTIFACTS),
        "artifact_hashes",
    )
    unbound: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        digest = artifacts[name]
        if digest is None:
            unbound.append(name)
        elif not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"artifact {name} must be null or a lowercase SHA-256")

    gates = value["gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("gates must be an object")
    unknown = set(gates) - set(REQUIRED_GATES)
    if unknown:
        raise ValueError(f"gates fields contain unknown entries: {sorted(unknown)}")

    failed: list[str] = []
    missing: list[str] = []
    for name in REQUIRED_GATES:
        if name not in gates:
            missing.append(name)
            continue
        gate = _strict_fields(gates[name], {"state", "evidence"}, f"gate {name}")
        state = gate["state"]
        if state not in {"passed", "failed", "pending"}:
            raise ValueError(f"gate {name} has an unsupported state")
        evidence = gate["evidence"]
        if not isinstance(evidence, str) or not evidence:
            raise ValueError(f"gate {name} requires a non-empty evidence reference")
        if state == "failed":
            failed.append(name)
        elif state == "pending":
            missing.append(name)

    if failed:
        status = "invalid"
    elif missing or unbound:
        status = "incomplete"
    else:
        status = "ready_for_protected_launch"
    return {
        "schema_version": 1,
        "status": status,
        "protected_launch_allowed": status == "ready_for_protected_launch",
        "failed_gates": sorted(failed),
        "missing_gates": sorted(missing),
        "unbound_artifacts": sorted(unbound),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = evaluate_readiness(raw)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
