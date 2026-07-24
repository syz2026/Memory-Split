"""Audit and apply the invalid-objective-oracle compiler migration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel.production import FROZEN_LANES
from corpusgen.v2_materialize import (
    _materializer_artifact_sha256s,
    _ObjectiveWorkerClient,
)

_OBJECTIVE_ARTIFACT = "corpusgen/v2_objective.py"
_PRONTOQA_PROVIDER = "prontoqa"
_REASONING_GYM_PROVIDER = "reasoning_gym_exact_answer"


class MigrationError(RuntimeError):
    """Raised when the objective identity migration is unsafe."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _acquire_lane_locks(work_root: Path) -> list[Any]:
    handles = []
    lock_root = work_root / "lane-worker-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    try:
        for lane in FROZEN_LANES:
            handle = (lock_root / f"{lane}.lock").open("a+b")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise MigrationError(f"{lane} still has an active worker") from error
            handles.append(handle)
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return handles


def _checkpoint_evidence(work_root: Path) -> dict[str, Any]:
    evidence = {}
    for path in sorted((work_root / "lanes").glob("*/checkpoint.json")):
        lane_root = path.parent
        checkpoint = json.loads(path.read_bytes())
        evidence[lane_root.name] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256_file(path),
            "partial_file_bytes": {
                name: (lane_root / name).stat().st_size
                for name in (
                    "facts.sqlite3",
                    "tokens.partial",
                    "verification.partial.jsonl",
                )
                if (lane_root / name).exists()
            },
        }
    return evidence


def _audit_reasoning_gym(
    *,
    identity: dict[str, Any],
    objective_python: Path,
    reasoning_gym_root: Path,
    log_root: Path,
) -> dict[str, Any]:
    client = _ObjectiveWorkerClient(
        _REASONING_GYM_PROVIDER,
        reasoning_gym_root,
        python=objective_python,
        log_root=log_root,
    )
    try:
        expected_runtime = identity["objective_runtime"][_REASONING_GYM_PROVIDER]
        if client.runtime != expected_runtime["versions"]:
            raise MigrationError(
                "Reasoning Gym runtime changed during objective migration"
            )
        expected_hashes = expected_runtime["probe_record_sha256s"]
        observed_hashes = [
            _sha256(canonical_json_bytes(client.generate(index)))
            for index in expected_runtime["probe_indices"]
        ]
        if observed_hashes != expected_hashes:
            mismatches = [
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_hashes, expected_hashes)
                )
                if observed != expected
            ]
            raise MigrationError(
                f"frozen Reasoning Gym probes changed: {mismatches}"
            )
        repaired = client.generate(468)
        metadata = repaired.get("metadata", {})
        if (
            repaired.get("answer") != "(P ∨ R)"
            or metadata.get("attempt") != 1
            or metadata.get("rejected_native_samples") != 1
            or metadata.get("rejection_policy")
            != "invalid_or_trivial_metadata_oracle"
            or metadata.get("seed") != 3_632_411_432
        ):
            raise MigrationError(
                "Reasoning Gym record 468 did not use the reviewed native retry"
            )
        repeated = client.generate(468)
        if canonical_json_bytes(repaired) != canonical_json_bytes(repeated):
            raise MigrationError("Reasoning Gym record 468 is not deterministic")
        return {
            "record_468_sha256": _sha256(canonical_json_bytes(repaired)),
            "record_468_validation": {
                "answer": repaired["answer"],
                "attempt": metadata["attempt"],
                "rejected_native_samples": metadata["rejected_native_samples"],
                "rejection_policy": metadata["rejection_policy"],
                "seed": metadata["seed"],
            },
            "unchanged_probe_count": len(observed_hashes),
        }
    finally:
        client.close()


def _audit_prontoqa(
    *,
    identity: dict[str, Any],
    objective_python: Path,
    prontoqa_root: Path,
    log_root: Path,
) -> dict[str, Any]:
    client = _ObjectiveWorkerClient(
        _PRONTOQA_PROVIDER,
        prontoqa_root,
        python=objective_python,
        log_root=log_root,
    )
    try:
        expected_runtime = identity["objective_runtime"][_PRONTOQA_PROVIDER]
        if client.runtime != expected_runtime["versions"]:
            raise MigrationError("ProntoQA runtime changed during objective migration")
        expected_hashes = expected_runtime["probe_record_sha256s"]
        observed_hashes = [
            _sha256(canonical_json_bytes(client.generate(index)))
            for index in expected_runtime["probe_indices"]
        ]
        if observed_hashes != expected_hashes:
            mismatches = [
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_hashes, expected_hashes)
                )
                if observed != expected
            ]
            raise MigrationError(f"frozen ProntoQA probes changed: {mismatches}")
        repaired = client.generate(2317)
        metadata = repaired.get("metadata", {})
        repaired_sha256 = _sha256(canonical_json_bytes(repaired))
        if (
            repaired.get("answer") != "False"
            or metadata.get("attempt") != 32
            or metadata.get("deduction_steps") != 2
            or metadata.get("depth_schedule_pass") != 1
            or metadata.get("native_proof_trace_sha256")
            != "c6b4d17c8efbd80a7211649219217adacd1569b19052047268592a07344bb20d"
            or metadata.get("rejected_native_samples") != 160
            or metadata.get("rejection_policy")
            != "exhaust_primary_then_rotate_seed_depth_pairing"
            or metadata.get("seed") != 187_038_367
            or repaired_sha256
            != "564e951df554411c9dbed55b29aeee2165acca4003bd27fe3e283a4492de470a"
        ):
            raise MigrationError(
                "ProntoQA record 2317 did not use the reviewed native retry"
            )
        repeated = client.generate(2317)
        if canonical_json_bytes(repaired) != canonical_json_bytes(repeated):
            raise MigrationError("ProntoQA record 2317 is not deterministic")
        return {
            "record_2317_sha256": repaired_sha256,
            "record_2317_validation": {
                "answer": repaired["answer"],
                "attempt": metadata["attempt"],
                "deduction_steps": metadata["deduction_steps"],
                "depth_schedule_pass": metadata["depth_schedule_pass"],
                "native_proof_trace_sha256": (
                    metadata["native_proof_trace_sha256"]
                ),
                "rejected_native_samples": metadata["rejected_native_samples"],
                "rejection_policy": metadata["rejection_policy"],
                "seed": metadata["seed"],
            },
            "unchanged_probe_count": len(observed_hashes),
        }
    finally:
        client.close()


def migrate_identity(
    work_root: Path,
    *,
    migration: str,
    expected_old_objective_sha256: str,
    old_revision: str,
    objective_python: Path,
    reasoning_gym_root: Path | None,
    prontoqa_root: Path | None,
) -> dict[str, Any]:
    locks = _acquire_lane_locks(work_root)
    try:
        identity_path = work_root / "materialization-identity.json"
        old_bytes = identity_path.read_bytes()
        old_identity = json.loads(old_bytes)
        old_artifacts = old_identity.get("compiler_artifact_sha256s")
        if not isinstance(old_artifacts, dict):
            raise MigrationError("materialization identity has no compiler artifacts")
        if old_artifacts.get(_OBJECTIVE_ARTIFACT) != (
            expected_old_objective_sha256
        ):
            raise MigrationError("unexpected previous objective artifact hash")

        current_artifacts = _materializer_artifact_sha256s()
        if set(old_artifacts) != set(current_artifacts):
            raise MigrationError("compiler artifact set changed")
        drift = {
            path: (old_artifacts[path], current_artifacts[path])
            for path in sorted(old_artifacts)
            if old_artifacts[path] != current_artifacts[path]
        }
        if set(drift) != {_OBJECTIVE_ARTIFACT}:
            raise MigrationError(
                f"migration contains unrelated compiler drift: {drift}"
            )

        checkpoints = _checkpoint_evidence(work_root)
        objective = checkpoints.get("objective_auxiliary", {}).get("checkpoint")
        if (
            not isinstance(objective, dict)
            or objective.get("complete") is not False
            or objective.get("tokens") != 0
            or objective.get("records") != 0
            or objective.get("files") != {"tokens": 0, "verification": 0}
        ):
            raise MigrationError(
                "objective lane has durable output; use a fresh work root"
            )

        if migration == "reasoning-gym-invalid-oracle":
            if reasoning_gym_root is None:
                raise MigrationError("Reasoning Gym root is required")
            audit = _audit_reasoning_gym(
                identity=old_identity,
                objective_python=objective_python,
                reasoning_gym_root=reasoning_gym_root,
                log_root=work_root / "logs" / "objective-migration",
            )
            reason = (
                "reject invalid or trivial pinned Reasoning Gym metadata "
                "oracles and retry complete native rows"
            )
            semantic_scope = (
                "objective_auxiliary.invalid_native_oracle_rejection"
            )
        elif migration == "prontoqa-depth-rotation":
            if prontoqa_root is None:
                raise MigrationError("ProntoQA root is required")
            audit = _audit_prontoqa(
                identity=old_identity,
                objective_python=objective_python,
                prontoqa_root=prontoqa_root,
                log_root=work_root / "logs" / "objective-migration",
            )
            reason = (
                "rotate pinned ProntoQA seed/depth pairings only after "
                "primary rejection exhaustion"
            )
            semantic_scope = (
                "objective_auxiliary.prontoqa.seed_depth_retry_exhaustion"
            )
        else:
            raise MigrationError(f"unsupported objective migration: {migration}")
        new_identity = dict(old_identity)
        new_identity["compiler_artifact_sha256s"] = current_artifacts
        new_bytes = canonical_json_bytes(new_identity)
        repository_root = Path(__file__).resolve().parents[1]
        receipt = {
            "checkpoints": checkpoints,
            "compiler_drift": {
                _OBJECTIVE_ARTIFACT: {
                    "after_sha256": current_artifacts[_OBJECTIVE_ARTIFACT],
                    "before_sha256": expected_old_objective_sha256,
                }
            },
            "format": "memorysplit-v2-objective-identity-migration-v1",
            "new_identity_sha256": _sha256(new_bytes),
            "new_revision": _revision(repository_root),
            "objective_audit": audit,
            "old_identity_sha256": _sha256(old_bytes),
            "old_revision": old_revision,
            "reason": reason,
            "semantic_change": True,
            "semantic_scope": semantic_scope,
        }
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_path = (
            work_root
            / "identity-migrations"
            / (
                f"{expected_old_objective_sha256[:12]}-to-"
                f"{current_artifacts[_OBJECTIVE_ARTIFACT][:12]}.json"
            )
        )
        if receipt_path.exists() and receipt_path.read_bytes() != receipt_bytes:
            raise MigrationError(f"migration receipt collision: {receipt_path}")
        _write_atomic(receipt_path, receipt_bytes)
        _write_atomic(identity_path, new_bytes)
        return {
            "identity": str(identity_path),
            "identity_sha256": _sha256(new_bytes),
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_bytes),
        }
    finally:
        for handle in locks:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument(
        "--migration",
        choices=(
            "prontoqa-depth-rotation",
            "reasoning-gym-invalid-oracle",
        ),
        required=True,
    )
    parser.add_argument("--expected-old-objective-sha256", required=True)
    parser.add_argument("--old-revision", required=True)
    parser.add_argument("--objective-python", required=True, type=Path)
    parser.add_argument("--reasoning-gym-root", type=Path)
    parser.add_argument("--prontoqa-root", type=Path)
    args = parser.parse_args()
    report = migrate_identity(
        args.work_root.resolve(),
        migration=args.migration,
        expected_old_objective_sha256=args.expected_old_objective_sha256,
        old_revision=args.old_revision,
        objective_python=Path(
            os.path.abspath(os.path.expanduser(str(args.objective_python)))
        ),
        reasoning_gym_root=(
            None
            if args.reasoning_gym_root is None
            else args.reasoning_gym_root.resolve()
        ),
        prontoqa_root=(
            None if args.prontoqa_root is None else args.prontoqa_root.resolve()
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
