"""Audit and apply the byte-equivalent objective-prefetch migration."""

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
    _ObjectiveSource,
)
from corpusgen.v2_sources import (
    DEFAULT_V2_SOURCE_LOCK,
    load_v2_source_lock,
)

_MATERIALIZER_ARTIFACT = "corpusgen/v2_materialize.py"


class MigrationError(RuntimeError):
    """Raised when objective prefetch cannot preserve the frozen stream."""


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


def _record_payload(record: Any, cursor: dict[str, Any]) -> bytes:
    return canonical_json_bytes(
        {
            "cursor": cursor,
            "record": {
                "record_id": record.record_id,
                "segments": [
                    {"fact_id": segment.fact_id, "text": segment.text}
                    for segment in record.segments
                ],
                "source_id": record.source_id,
                "verification": record.verification,
            },
        }
    )


def _audit_prefetch(
    *,
    stage_root: Path,
    work_root: Path,
    identity: dict[str, Any],
    objective_python: Path,
    source_lock_path: Path,
    cursor: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    source_lock = load_v2_source_lock(source_lock_path)
    kwargs = {
        "stage_root": stage_root,
        "lock": source_lock,
        "objective_python": objective_python,
        "expected_runtime": identity["objective_runtime"],
    }
    serial = _ObjectiveSource(
        work_root=work_root / "logs" / "prefetch-audit-serial",
        prefetch=False,
        **kwargs,
    )
    parallel = _ObjectiveSource(
        work_root=work_root / "logs" / "prefetch-audit-parallel",
        prefetch=True,
        **kwargs,
    )
    serial_cursor = dict(cursor)
    parallel_cursor = dict(cursor)
    digest = hashlib.sha256()
    first_sha256 = None
    last_sha256 = None
    try:
        for _ in range(sample_count):
            serial_record, serial_cursor = serial.next(serial_cursor)
            parallel_record, parallel_cursor = parallel.next(parallel_cursor)
            serial_payload = _record_payload(serial_record, serial_cursor)
            parallel_payload = _record_payload(parallel_record, parallel_cursor)
            if serial_payload != parallel_payload:
                raise MigrationError(
                    "objective prefetch changed record or cursor ordering"
                )
            record_sha256 = _sha256(serial_payload)
            if first_sha256 is None:
                first_sha256 = record_sha256
            last_sha256 = record_sha256
            digest.update(serial_payload)
    finally:
        serial.close()
        parallel.close()
    return {
        "first_record_sha256": first_sha256,
        "last_record_sha256": last_sha256,
        "sample_count": sample_count,
        "sample_stream_sha256": digest.hexdigest(),
        "serial_final_cursor": serial_cursor,
    }


def migrate_identity(
    work_root: Path,
    *,
    stage_root: Path,
    expected_old_identity_sha256: str,
    expected_old_materializer_sha256: str,
    old_revision: str,
    objective_python: Path,
    source_lock_path: Path,
    sample_count: int,
) -> dict[str, Any]:
    locks = _acquire_lane_locks(work_root)
    try:
        identity_path = work_root / "materialization-identity.json"
        old_bytes = identity_path.read_bytes()
        if _sha256(old_bytes) != expected_old_identity_sha256:
            raise MigrationError("unexpected previous materialization identity")
        old_identity = json.loads(old_bytes)
        old_artifacts = old_identity.get("compiler_artifact_sha256s")
        if not isinstance(old_artifacts, dict):
            raise MigrationError("materialization identity has no compiler artifacts")
        if old_artifacts.get(_MATERIALIZER_ARTIFACT) != (
            expected_old_materializer_sha256
        ):
            raise MigrationError("unexpected previous materializer artifact hash")

        current_artifacts = _materializer_artifact_sha256s()
        if set(old_artifacts) != set(current_artifacts):
            raise MigrationError("compiler artifact set changed")
        drift = {
            path: (old_artifacts[path], current_artifacts[path])
            for path in sorted(old_artifacts)
            if old_artifacts[path] != current_artifacts[path]
        }
        if set(drift) != {_MATERIALIZER_ARTIFACT}:
            raise MigrationError(
                f"migration contains unrelated compiler drift: {drift}"
            )

        checkpoints = _checkpoint_evidence(work_root)
        objective = checkpoints.get("objective_auxiliary", {}).get("checkpoint")
        if not isinstance(objective, dict) or objective.get("complete") is not False:
            raise MigrationError("objective lane checkpoint is missing or complete")
        audit = _audit_prefetch(
            stage_root=stage_root,
            work_root=work_root,
            identity=old_identity,
            objective_python=objective_python,
            source_lock_path=source_lock_path,
            cursor=dict(objective["cursor"]),
            sample_count=sample_count,
        )

        new_identity = dict(old_identity)
        new_identity["compiler_artifact_sha256s"] = current_artifacts
        new_bytes = canonical_json_bytes(new_identity)
        repository_root = Path(__file__).resolve().parents[1]
        receipt = {
            "checkpoints": checkpoints,
            "compiler_drift": {
                _MATERIALIZER_ARTIFACT: {
                    "after_sha256": current_artifacts[_MATERIALIZER_ARTIFACT],
                    "before_sha256": expected_old_materializer_sha256,
                }
            },
            "equivalence_audit": audit,
            "format": "memorysplit-v2-objective-prefetch-identity-migration-v1",
            "new_identity_sha256": _sha256(new_bytes),
            "new_revision": _revision(repository_root),
            "old_identity_sha256": _sha256(old_bytes),
            "old_revision": old_revision,
            "reason": (
                "prefetch independent procedural objective providers while "
                "retaining serial record and cursor order"
            ),
            "semantic_change": False,
        }
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_path = (
            work_root
            / "identity-migrations"
            / (
                f"{expected_old_materializer_sha256[:12]}-to-"
                f"{current_artifacts[_MATERIALIZER_ARTIFACT][:12]}-prefetch.json"
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
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--expected-old-identity-sha256", required=True)
    parser.add_argument("--expected-old-materializer-sha256", required=True)
    parser.add_argument("--old-revision", required=True)
    parser.add_argument("--objective-python", required=True, type=Path)
    parser.add_argument(
        "--source-lock",
        default=DEFAULT_V2_SOURCE_LOCK,
        type=Path,
    )
    parser.add_argument("--samples", default=512, type=int)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    report = migrate_identity(
        args.work_root.resolve(),
        stage_root=args.stage_root.resolve(),
        expected_old_identity_sha256=args.expected_old_identity_sha256,
        expected_old_materializer_sha256=(
            args.expected_old_materializer_sha256
        ),
        old_revision=args.old_revision,
        objective_python=Path(
            os.path.abspath(os.path.expanduser(str(args.objective_python)))
        ),
        source_lock_path=args.source_lock.resolve(),
        sample_count=args.samples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
