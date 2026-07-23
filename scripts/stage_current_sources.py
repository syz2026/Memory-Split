#!/usr/bin/env python
"""Plan or execute staging for the pinned current dataset sources."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.current_sources import load_dataset_lock, stage_current_sources


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "configs" / "current-dataset-lock.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage relational-chinchilla sources. The default is a read-only "
            "dry run; pass --execute to download, verify, and publish."
        )
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="shared data root (defaults to DATA_ROOT)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Hugging Face and Git source cache (defaults to DATA_ROOT/hf-cache)",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root or DATA_ROOT is required")

    data_root = Path(args.data_root)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else data_root / "hf-cache"
    )
    lock = load_dataset_lock(args.lock)

    previous_cache = os.environ.get("MEMORYSPLIT_SOURCE_CACHE")
    os.environ["MEMORYSPLIT_SOURCE_CACHE"] = str(cache_dir)
    try:
        result = stage_current_sources(
            lock,
            data_root,
            execute=args.execute,
        )
    finally:
        if previous_cache is None:
            os.environ.pop("MEMORYSPLIT_SOURCE_CACHE", None)
        else:
            os.environ["MEMORYSPLIT_SOURCE_CACHE"] = previous_cache

    if not args.execute:
        print("DRY RUN (no files written)")
        for operation in result["operations"]:
            print(shlex.join(operation["argv"]))
    else:
        print(result["dataset_id"])
        print(data_root / lock.dataset_id / "sources" / "source-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
