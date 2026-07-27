#!/usr/bin/env python
"""Lock a validated local natural-text JSONL snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.bed_snapshot import lock_bed_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local JSONL text snapshot and write its immutable lock."
        )
    )
    parser.add_argument("snapshot")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    lock = lock_bed_snapshot(
        args.snapshot,
        repo_id=args.repo_id,
        revision=args.revision,
        config=args.config,
        split=args.split,
    )
    lock.write(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
