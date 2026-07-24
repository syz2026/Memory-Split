"""Materialize one frozen v2 lane under an exclusive work-root lock."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel.production import (
    DEFAULT_RECIPE_PATH,
    FROZEN_LANES,
    load_production_recipe,
)
from corpusgen.v2_materialize import (
    _compile_exact_lane,
    _compile_wikidata_graph,
    _LaneWriter,
    _main_runtime_versions,
    _materializer_artifact_sha256s,
    _ObjectiveSource,
    _RelationalRefinementSource,
    _SyntheticGraphSource,
    _SyntheticMultihopSource,
    _WikidataPathSource,
)
from corpusgen.v2_sources import (
    DEFAULT_V2_SOURCE_LOCK,
    load_v2_source_lock,
)


class LaneWorkerError(RuntimeError):
    """Raised when a lane cannot be compiled under the frozen identity."""


class _ReadOnlyWikidataIndex:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.connection.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
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


def _assert_identity(
    identity: dict[str, Any],
    *,
    recipe_sha256: str,
    source_lock_sha256: str,
) -> None:
    expected = {
        "compiler_artifact_sha256s": _materializer_artifact_sha256s(),
        "main_runtime": _main_runtime_versions(),
        "recipe_sha256": recipe_sha256,
        "source_set_lock_sha256": source_lock_sha256,
    }
    drift = {
        key: {"actual": identity.get(key), "expected": value}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if drift:
        raise LaneWorkerError(f"materialization identity drift: {drift}")


def _source_for_lane(
    lane: str,
    *,
    stage_root: Path,
    work_root: Path,
    source_lock: Any,
    objective_python: Path,
    objective_runtime: dict[str, Any],
    wikidata: _ReadOnlyWikidataIndex | None,
) -> Any:
    if lane == "synthetic_graph":
        return _SyntheticGraphSource()
    if lane == "verified_synthetic_multihop":
        return _SyntheticMultihopSource()
    if lane == "wikidata_path_reasoning":
        if wikidata is None:
            raise AssertionError("Wikidata path lane requires its frozen index")
        return _WikidataPathSource(wikidata)
    if lane == "relational_refinement":
        return _RelationalRefinementSource()
    if lane == "objective_auxiliary":
        return _ObjectiveSource(
            stage_root,
            source_lock,
            objective_python=objective_python,
            work_root=work_root,
            expected_runtime=objective_runtime,
        )
    raise LaneWorkerError(f"unsupported independent lane: {lane}")


def materialize_lane(
    lane: str,
    *,
    stage_root: Path,
    work_root: Path,
    objective_python: Path,
    source_lock_path: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    if lane not in FROZEN_LANES:
        raise LaneWorkerError(f"unknown frozen lane: {lane}")
    if lane in {"fineweb_edu", "finemath"}:
        raise LaneWorkerError(f"{lane} must use the stock source materializer")
    if (work_root / "routing-receipt.json").exists():
        raise LaneWorkerError("global routing already started")

    recipe = load_production_recipe(recipe_path)
    source_lock = load_v2_source_lock(source_lock_path)
    identity_path = work_root / "materialization-identity.json"
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    _assert_identity(
        identity,
        recipe_sha256=recipe.recipe_sha256,
        source_lock_sha256=source_lock.sha256,
    )

    lock_path = work_root / "lane-worker-locks" / f"{lane}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaneWorkerError(f"{lane} already has an active worker") from error

        wikidata = None
        if lane in {"wikidata_graph", "wikidata_path_reasoning"}:
            wikidata = _ReadOnlyWikidataIndex(
                work_root / "indexes" / "wikidata.sqlite3"
            )
        writer = _LaneWriter(
            work_root,
            lane,
            recipe.quota_by_lane[lane],
            checkpoint_tokens=int(identity["checkpoint_tokens"]),
            verification_required=(
                lane in recipe.reasoning_lanes or lane == recipe.objective_lane
            ),
        )
        succeeded = False
        try:
            if lane == "wikidata_graph":
                assert wikidata is not None
                _compile_wikidata_graph(
                    writer,
                    wikidata,
                    stage_root,
                    finish_window=int(identity["finish_window"]),
                    max_finish_candidates=int(identity["max_finish_candidates"]),
                )
            else:
                source = _source_for_lane(
                    lane,
                    stage_root=stage_root,
                    work_root=work_root,
                    source_lock=source_lock,
                    objective_python=objective_python,
                    objective_runtime=dict(identity["objective_runtime"]),
                    wikidata=wikidata,
                )
                _compile_exact_lane(
                    writer,
                    source,
                    finish_window=int(identity["finish_window"]),
                    max_finish_candidates=int(identity["max_finish_candidates"]),
                )
            succeeded = True
        finally:
            writer.close(checkpoint=succeeded)
            if wikidata is not None:
                wikidata.close()

        checkpoint_path = work_root / "lanes" / lane / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_bytes())
        if checkpoint.get("complete") is not True:
            raise LaneWorkerError(f"{lane} worker exited without completing its quota")
        repository_root = Path(__file__).resolve().parents[1]
        receipt = {
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "format": "memorysplit-v2-independent-lane-worker-v1",
            "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "lane": lane,
            "quota": recipe.quota_by_lane[lane],
            "repository_revision": _revision(repository_root),
            "script_sha256": _sha256_file(Path(__file__)),
            "tokens": int(checkpoint["tokens"]),
        }
        receipt_path = work_root / "lane-worker-receipts" / f"{lane}.json"
        _write_atomic(receipt_path, receipt)
        return {
            **receipt,
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256_file(receipt_path),
        }
    finally:
        lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=FROZEN_LANES)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--objective-python", required=True, type=Path)
    parser.add_argument(
        "--source-lock",
        default=DEFAULT_V2_SOURCE_LOCK,
        type=Path,
    )
    parser.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE_PATH,
        type=Path,
    )
    args = parser.parse_args()
    result = materialize_lane(
        args.lane,
        stage_root=args.stage_root.resolve(),
        work_root=args.work_root.resolve(),
        objective_python=args.objective_python.resolve(),
        source_lock_path=args.source_lock.resolve(),
        recipe_path=args.recipe.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
