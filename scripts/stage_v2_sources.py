#!/usr/bin/env python3
"""Plan, stage, or verify the frozen upstream subset for MemorySplit v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.v2_sources import (
    DEFAULT_RESERVE_BYTES,
    DEFAULT_V2_SOURCE_LOCK,
    load_v2_source_lock,
    receipt_summary,
    stage_v2_sources,
    verify_v2_source_stage,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="destination parent (defaults to DATA_ROOT)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_V2_SOURCE_LOCK,
        help="immutable source-set lock",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="transfer cache (defaults to DATA_ROOT/.cache/memorysplit-v2)",
    )
    parser.add_argument(
        "--hf-command",
        default=os.environ.get("HF_CLI", "hf"),
        help="current huggingface_hub `hf` executable",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=DEFAULT_RESERVE_BYTES / 1024**3,
        help="free-space reserve in addition to locked data and safety margin",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="download, derive selection sidecars, verify, and publish atomically",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify an already-published source stage",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard only this lock's private partial stage before execution",
    )
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root or DATA_ROOT is required")
    if args.restart and not args.execute:
        parser.error("--restart requires --execute")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be nonnegative")

    lock = load_v2_source_lock(args.lock)
    data_root = Path(args.data_root)
    source_root = data_root.resolve() / lock.dataset_id
    if args.verify:
        receipt = verify_v2_source_stage(lock, source_root)
        result = receipt_summary(lock, source_root, receipt)
    else:
        result = stage_v2_sources(
            lock,
            data_root,
            execute=args.execute,
            cache_dir=args.cache_dir,
            hf_command=args.hf_command,
            max_workers=args.max_workers,
            reserve_bytes=int(args.reserve_gib * 1024**3),
            restart=args.restart,
        )
        if args.execute:
            result = receipt_summary(lock, source_root, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
