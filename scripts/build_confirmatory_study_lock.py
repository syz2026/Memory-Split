#!/usr/bin/env python3
"""Build (and optionally publish) the receipts-driven confirmatory v3 lock.

The CLI reads one local evidence index, descriptor-reads and rehashes every
referenced payload, extracts the twenty snapshot study identities, builds
the receipt-proven StudyLockV3, and either reports it (default dry-run,
which writes nothing) or publishes exactly one ``study-lock.json`` into one
content-addressed no-replace directory. Exactly one canonical JSON object is
emitted on stdout for help, success, and every error. No network is used;
all inputs are local files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.confirmatory.aggregate import CollectionReceiptEvidence
from evals.confirmatory.contracts import (
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.lock_builder import (
    PATH_AUTHORITY,
    LockBuildError,
    SeedCollectionEvidence,
    build_study_lock_v3,
    extract_run_study_identities,
    publish_study_lock,
    read_local_evidence_bytes,
)
from evals.confirmatory.sealing import SealingError
from msctl.aws_contracts import ARMS, SEEDS, SNAPSHOT_STEPS


_INDEX_FIELDS = frozenset({"schema_version", "seeds"})
_SEED_FIELDS = frozenset(
    {
        "seed",
        "collection",
        "finalization_payload_path",
        "checkpoint_receipt_payload_path",
        "snapshots",
    }
)
_COLLECTION_FIELDS = frozenset(
    {"uri", "sha256", "version_id", "payload_path"}
)
_SNAPSHOT_FIELDS = frozenset({"arm", "step", "path"})
_EXPECTED_SNAPSHOT_ORDER = tuple(
    (arm, step) for arm in ARMS for step in SNAPSHOT_STEPS
)


class _ArgumentError(ValueError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError(message)


def _parser() -> _JsonArgumentParser:
    parser = _JsonArgumentParser(
        description=(
            "Build the receipts-driven confirmatory v3 study lock from ten "
            "local Task 3F collection receipt bodies, their Task 3D/3C "
            "receipt bodies, and 100 locally collected snapshot files. "
            "Dry-run is the default and writes nothing; --publish installs "
            "exactly one study-lock.json into one content-addressed "
            "no-replace directory under --output-root."
        ),
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--evidence-index")
    parser.add_argument("--sealed-evaluation-release-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--output-root")
    return parser


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode("utf-8"))


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise LockBuildError(f"{label} repeats field {key}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                LockBuildError(f"{label} contains non-finite {constant}")
            ),
        )
    except LockBuildError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LockBuildError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise LockBuildError(f"{label} must be one JSON object")
    return parsed


def _exact_fields(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise LockBuildError(f"{label} fields are not exact")
    return value


def _index_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LockBuildError(f"{label} must be a non-empty string")
    return value


def _resolved_input_path(value: object, *, base: Path, label: str) -> Path:
    text = _index_string(value, label)
    if "\x00" in text:
        raise LockBuildError(f"{label} must be a local filesystem path")
    candidate = Path(text)
    if any(part == ".." for part in candidate.parts):
        raise LockBuildError(f"{label} must not contain path traversal")
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def _load_evidence_index(
    index_path: str,
) -> tuple[
    list[SeedCollectionEvidence],
    dict[tuple[int, str, int], Path],
]:
    absolute = Path(os.path.abspath(index_path))
    payload = read_local_evidence_bytes(absolute, label="evidence index")
    value = _exact_fields(
        _strict_json_object(payload, label="evidence index"),
        _INDEX_FIELDS,
        "evidence index",
    )
    if type(value["schema_version"]) is not int or value[
        "schema_version"
    ] != 1:
        raise LockBuildError("evidence index schema_version must be 1")
    rows = value["seeds"]
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise LockBuildError(
            "evidence index requires exactly ten seed rows ordered 0..9"
        )
    base = absolute.parent
    evidence: list[SeedCollectionEvidence] = []
    snapshot_paths: dict[tuple[int, str, int], Path] = {}
    for seed, row in zip(SEEDS, rows, strict=True):
        record = _exact_fields(
            row,
            _SEED_FIELDS,
            f"evidence index seed row {seed}",
        )
        if record["seed"] != seed:
            raise LockBuildError(
                "evidence index seed rows must be ordered 0..9 ascending"
            )
        collection = _exact_fields(
            record["collection"],
            _COLLECTION_FIELDS,
            f"evidence index seed {seed} collection",
        )
        collection_payload = read_local_evidence_bytes(
            _resolved_input_path(
                collection["payload_path"],
                base=base,
                label=f"evidence index seed {seed} collection payload path",
            ),
            label=f"seed {seed} collection receipt payload",
        )
        finalization_payload = read_local_evidence_bytes(
            _resolved_input_path(
                record["finalization_payload_path"],
                base=base,
                label=(
                    f"evidence index seed {seed} finalization payload path"
                ),
            ),
            label=f"seed {seed} finalization receipt payload",
        )
        checkpoint_payload = read_local_evidence_bytes(
            _resolved_input_path(
                record["checkpoint_receipt_payload_path"],
                base=base,
                label=(
                    f"evidence index seed {seed} checkpoint receipt "
                    "payload path"
                ),
            ),
            label=f"seed {seed} checkpoint receipt payload",
        )
        try:
            evidence.append(
                SeedCollectionEvidence(
                    collection=CollectionReceiptEvidence(
                        payload=collection_payload,
                        uri=_index_string(
                            collection["uri"],
                            f"evidence index seed {seed} collection URI",
                        ),
                        sha256=collection["sha256"],
                        version_id=collection["version_id"],
                    ),
                    finalization_payload=finalization_payload,
                    checkpoint_receipt_payload=checkpoint_payload,
                )
            )
        except (TypeError, ValueError) as error:
            raise LockBuildError(
                f"evidence index seed {seed} collection identity is "
                f"invalid: {error}"
            ) from error
        snapshots = record["snapshots"]
        if not isinstance(snapshots, list) or len(snapshots) != len(
            _EXPECTED_SNAPSHOT_ORDER
        ):
            raise LockBuildError(
                f"evidence index seed {seed} requires exactly ten snapshot "
                "rows ordered Dense five steps then Split90 five steps"
            )
        for (arm, step), snapshot in zip(
            _EXPECTED_SNAPSHOT_ORDER,
            snapshots,
            strict=True,
        ):
            snapshot_row = _exact_fields(
                snapshot,
                _SNAPSHOT_FIELDS,
                f"evidence index seed {seed} snapshot row",
            )
            if snapshot_row["arm"] != arm or snapshot_row["step"] != step:
                raise LockBuildError(
                    f"evidence index seed {seed} snapshot rows must be "
                    "ordered Dense five steps then Split90 five steps"
                )
            snapshot_paths[(seed, arm, step)] = _resolved_input_path(
                snapshot_row["path"],
                base=base,
                label=(
                    f"evidence index seed {seed} {arm} step {step} "
                    "snapshot path"
                ),
            )
    return evidence, snapshot_paths


def _require_arguments(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--evidence-index", args.evidence_index),
            (
                "--sealed-evaluation-release-sha256",
                args.sealed_evaluation_release_sha256,
            ),
        )
        if value is None
    ]
    if missing:
        raise _ArgumentError("requires " + ", ".join(missing))
    if args.publish and args.dry_run:
        raise _ArgumentError("--publish and --dry-run are mutually exclusive")
    if args.publish and args.output_root is None:
        raise _ArgumentError("--publish requires --output-root")
    if not args.publish and args.output_root is not None:
        raise _ArgumentError("--output-root requires --publish")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    mode = "arguments"
    try:
        args = parser.parse_args(argv)
        if args.help:
            _emit(
                {
                    "schema_version": 1,
                    "ok": True,
                    "mode": "help",
                    "published": False,
                    "usage": parser.format_help(),
                }
            )
            return 0
        _require_arguments(args)
        mode = "publish" if args.publish else "dry-run"
        evidence, snapshot_paths = _load_evidence_index(args.evidence_index)
        identities = extract_run_study_identities(
            evidence,
            snapshot_paths=snapshot_paths,
        )
        lock = build_study_lock_v3(
            sealed_evaluation_release_sha256=(
                args.sealed_evaluation_release_sha256
            ),
            evidence=evidence,
            run_study_identities=identities,
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "ok": True,
            "mode": mode,
            "published": False,
            "study_lock_sha256": canonical_sha256(lock.to_dict()),
            "slot_count": len(lock.snapshots),
            "seed_count": len(lock.seed_lifecycles),
            "run_identity_count": len(identities),
            "path_authority": PATH_AUTHORITY,
        }
        if args.publish:
            published = publish_study_lock(
                lock,
                collection_receipts=tuple(
                    row.collection for row in evidence
                ),
                output_root=args.output_root,
            )
            report["published"] = True
            report["study_lock_sha256"] = (
                published.authoritative_commitment
            )
            report["output_dir"] = str(published.output_dir)
            report["study_lock_path"] = str(published.study_lock_path)
        code = 0
    except (_ArgumentError, LockBuildError, SealingError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "ok": False,
            "mode": mode,
            "published": False,
            "error": {
                "code": getattr(exc, "code", None)
                or (
                    "ARGUMENT_ERROR"
                    if isinstance(exc, _ArgumentError)
                    else "LOCK_BUILDER_INVALID"
                ),
                "message": str(exc),
            },
        }
        code = 2
    except OSError as exc:
        report = {
            "schema_version": 1,
            "ok": False,
            "mode": mode,
            "published": False,
            "error": {
                "code": "LOCAL_IO_ERROR",
                "message": str(exc),
            },
        }
        code = 2
    except Exception:
        report = {
            "schema_version": 1,
            "ok": False,
            "mode": mode,
            "published": False,
            "error": {
                "code": "LOCK_BUILDER_INTERNAL_ERROR",
                "message": "unexpected local study-lock builder failure",
            },
        }
        code = 70
    _emit(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
