#!/usr/bin/env python3
"""Download every reviewed source and freeze the reasoning-v2 source lock.

``scripts/resolve_source_lock.py`` proves what upstream *metadata* can prove
and refuses the rest, which leaves the git inventories and the FineMath
selection open: git names objects by SHA-1 over a length-prefixed blob header,
and the FineMath shard prefix is only decidable after the parquet text is read
and cross-deduplicated against FineWeb. Both need the bytes.

This driver closes that gap. It runs the real ``PublicSourceResolver`` over the
reviewed catalog, which downloads each source at its immutable revision, hashes
every retained file, chooses the FineMath shards under the recipe's ordering
policy, and then re-walks the materialised tree to confirm the lock describes
what is actually on disk.

The download root must not already exist for a source: the resolver refuses to
adopt bytes it did not place there itself, so a partial tree is never blessed.
Budget roughly 90 GB of transfer and a disk with room for it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from corpusgen.reasoning_v2 import load_recipe  # noqa: E402
from corpusgen.reasoning_v2.source_lock import (  # noqa: E402
    PublicSourceResolver,
    resolve_source_lock,
)

RECIPE_PATH = REPOSITORY_ROOT / "configs/reasoning-dataset-v2.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def head_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class _ProgressResolver:
    """Announce each source as it is resolved; the real work is delegated."""

    def __init__(self, delegate: object, total: int) -> None:
        self._delegate = delegate
        self._total = total
        self._done = 0

    def resolve(self, request, download_root):
        self._done += 1
        started = time.time()
        log(f"({self._done}/{self._total}) {request.source_id}: {request.transport}")
        entry = self._delegate.resolve(request, download_root)
        retained = sum(file.bytes for file in entry.files)
        log(
            f"({self._done}/{self._total}) {request.source_id}: "
            f"{len(entry.files)} files, {retained / 2**30:.2f} GiB, "
            f"rev {entry.revision[:12]}, {time.time() - started:.0f}s"
        )
        return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-root",
        required=True,
        type=Path,
        help="directory the resolver materialises sources into; must be empty",
    )
    parser.add_argument("--out", required=True, type=Path, help="source lock path")
    parser.add_argument(
        "--generator-commit",
        default=None,
        help="40-hex commit of the generator code; defaults to HEAD",
    )
    args = parser.parse_args()

    commit = args.generator_commit or head_commit()
    recipe = load_recipe(RECIPE_PATH)
    log(f"recipe    {recipe.dataset_id}")
    log(f"commit    {commit}")
    log(f"downloads {args.download_root}")

    resolver = _ProgressResolver(PublicSourceResolver(), total=11)
    started = time.time()
    lock = resolve_source_lock(
        recipe,
        resolver,
        args.download_root,
        generator_commit=commit,
    )

    payload = lock.to_bytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)

    retained = sum(file.bytes for entry in lock.sources for file in entry.files)
    files = sum(len(entry.files) for entry in lock.sources)
    log(
        f"lock frozen: {len(lock.sources)} sources, {files} files, "
        f"{retained / 2**30:.2f} GiB in {time.time() - started:.0f}s"
    )
    log(f"lock sha256 = {lock.sha256}")
    log(f"lock -> {args.out}")

    summary = {
        source.source_id: {
            "revision": source.revision,
            "files": len(source.files),
            "bytes": sum(file.bytes for file in source.files),
        }
        for source in lock.sources
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
