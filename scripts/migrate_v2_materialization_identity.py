"""Audit and apply a byte-equivalent materializer compiler migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.v2_materialize import _materializer_artifact_sha256s

_MATERIALIZER_PATH = "corpusgen/v2_materialize.py"
_OLD_NEXT_GROUP_SQL = """
    SELECT subject, relation
    FROM triples
    WHERE subject > ?
       OR (subject = ? AND relation > ?)
    GROUP BY subject, relation
    ORDER BY subject, relation
    LIMIT 1
"""
_NEXT_RELATION_SQL = """
    SELECT subject, relation
    FROM triples
    WHERE subject = ? AND relation > ?
    ORDER BY relation, target
    LIMIT 1
"""
_NEXT_SUBJECT_SQL = """
    SELECT subject, relation
    FROM triples
    WHERE subject > ?
    ORDER BY subject, relation, target
    LIMIT 1
"""


class MigrationError(RuntimeError):
    """Raised when the existing work root cannot be migrated safely."""


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


def _next_group_seek(
    connection: sqlite3.Connection,
    subject: int,
    relation: int,
) -> tuple[int, int] | None:
    row = connection.execute(
        _NEXT_RELATION_SQL,
        (subject, relation),
    ).fetchone()
    if row is None:
        row = connection.execute(
            _NEXT_SUBJECT_SQL,
            (subject,),
        ).fetchone()
    return None if row is None else tuple(map(int, row))


def _next_group_reference(
    connection: sqlite3.Connection,
    subject: int,
    relation: int,
) -> tuple[int, int] | None:
    row = connection.execute(
        _OLD_NEXT_GROUP_SQL,
        (subject, subject, relation),
    ).fetchone()
    return None if row is None else tuple(map(int, row))


def _sample_cursors(
    connection: sqlite3.Connection,
    checkpoint: Mapping[str, Any],
    sample_count: int,
) -> list[tuple[int, int]]:
    maximum = int(
        connection.execute("SELECT MAX(ordinal) FROM triples").fetchone()[0]
    )
    cursors = {(-1, -1)}
    checkpoint_cursor = checkpoint.get("cursor", {})
    if "subject" in checkpoint_cursor and "relation" in checkpoint_cursor:
        cursors.add(
            (
                int(checkpoint_cursor["subject"]),
                int(checkpoint_cursor["relation"]),
            )
        )
    for index in range(sample_count):
        ordinal = maximum * index // max(sample_count - 1, 1)
        row = connection.execute(
            """
            SELECT subject, relation
            FROM triples
            WHERE ordinal >= ?
            ORDER BY ordinal
            LIMIT 1
            """,
            (ordinal,),
        ).fetchone()
        if row is not None:
            cursors.add(tuple(map(int, row)))
    return sorted(cursors)


def _query_plan(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[int, ...],
) -> list[str]:
    return [
        str(row[3])
        for row in connection.execute(
            f"EXPLAIN QUERY PLAN {query}",
            parameters,
        )
    ]


def migrate_identity(
    work_root: Path,
    *,
    expected_old_materializer_sha256: str,
    old_revision: str,
    new_revision: str,
    sample_count: int,
) -> dict[str, Any]:
    identity_path = work_root / "materialization-identity.json"
    old_bytes = identity_path.read_bytes()
    old_identity = json.loads(old_bytes)
    old_artifacts = old_identity.get("compiler_artifact_sha256s")
    if not isinstance(old_artifacts, dict):
        raise MigrationError("materialization identity has no compiler artifacts")
    if old_artifacts.get(_MATERIALIZER_PATH) != expected_old_materializer_sha256:
        raise MigrationError("unexpected previous materializer artifact hash")

    current_artifacts = _materializer_artifact_sha256s()
    if set(old_artifacts) != set(current_artifacts):
        raise MigrationError("compiler artifact set changed")
    drift = {
        path: (old_artifacts[path], current_artifacts[path])
        for path in sorted(old_artifacts)
        if old_artifacts[path] != current_artifacts[path]
    }
    if set(drift) != {_MATERIALIZER_PATH}:
        raise MigrationError(f"migration contains unrelated compiler drift: {drift}")

    checkpoint_paths = sorted(
        (work_root / "lanes").glob("*/checkpoint.json")
    )
    checkpoints = {
        path.parent.name: {
            "checkpoint": json.loads(path.read_bytes()),
            "checkpoint_sha256": _sha256_file(path),
        }
        for path in checkpoint_paths
    }
    graph_checkpoint = checkpoints.get("wikidata_graph", {}).get("checkpoint")
    if not isinstance(graph_checkpoint, dict):
        raise MigrationError("Wikidata graph checkpoint is missing")

    index_path = work_root / "indexes" / "wikidata.sqlite3"
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        cursors = _sample_cursors(connection, graph_checkpoint, sample_count)
        comparisons = []
        for subject, relation in cursors:
            reference = _next_group_reference(connection, subject, relation)
            seek = _next_group_seek(connection, subject, relation)
            if reference != seek:
                raise MigrationError(
                    "indexed seek differs from reference traversal at "
                    f"{subject}/{relation}: {reference} != {seek}"
                )
            comparisons.append(
                {
                    "cursor": [subject, relation],
                    "next_group": None if seek is None else list(seek),
                }
            )
        cursor = graph_checkpoint.get("cursor", {})
        subject = int(cursor.get("subject", -1))
        relation = int(cursor.get("relation", -1))
        plans = {
            "next_relation": _query_plan(
                connection,
                _NEXT_RELATION_SQL,
                (subject, relation),
            ),
            "next_subject": _query_plan(
                connection,
                _NEXT_SUBJECT_SQL,
                (subject,),
            ),
        }
        if any(
            "SEARCH" not in " ".join(rows).upper()
            or "SCAN TRIPLES" in " ".join(rows).upper()
            for rows in plans.values()
        ):
            raise MigrationError(f"replacement query is not index-seeking: {plans}")
    finally:
        connection.close()

    new_identity = dict(old_identity)
    new_identity["compiler_artifact_sha256s"] = current_artifacts
    new_bytes = canonical_json_bytes(new_identity)
    receipt = {
        "checkpoints": checkpoints,
        "compiler_drift": {
            _MATERIALIZER_PATH: {
                "after_sha256": current_artifacts[_MATERIALIZER_PATH],
                "before_sha256": expected_old_materializer_sha256,
            }
        },
        "equivalence": {
            "comparisons": comparisons,
            "query_plans": plans,
            "semantic_change": False,
            "traversal_order": ["subject", "relation", "target"],
        },
        "format": "memorysplit-v2-materializer-identity-migration-v1",
        "new_identity_sha256": _sha256(new_bytes),
        "new_revision": new_revision,
        "old_identity_sha256": _sha256(old_bytes),
        "old_revision": old_revision,
        "reason": "replace quadratic group scan with byte-equivalent index seeks",
    }
    receipt_bytes = canonical_json_bytes(receipt)
    receipt_path = (
        work_root
        / "identity-migrations"
        / (
            f"{expected_old_materializer_sha256[:12]}-to-"
            f"{current_artifacts[_MATERIALIZER_PATH][:12]}.json"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--expected-old-materializer-sha256", required=True)
    parser.add_argument("--old-revision", required=True)
    parser.add_argument("--new-revision", required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    report = migrate_identity(
        args.work_root,
        expected_old_materializer_sha256=args.expected_old_materializer_sha256,
        old_revision=args.old_revision,
        new_revision=args.new_revision,
        sample_count=args.samples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
