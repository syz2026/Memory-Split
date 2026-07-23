#!/usr/bin/env python3
"""Verify or atomically stage a frozen complete-corpus Slurm mirror."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.corpus_contract import (  # noqa: E402
    stage_dataset_no_replace,
    verify_dataset_root,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="existing complete dataset mirror")
    parser.add_argument(
        "--pointer",
        default=str(ROOT / "DATASET-POINTER-SLURM-135M.json"),
    )
    parser.add_argument(
        "--source-lock",
        default=str(ROOT / "configs" / "reasoning-dataset-v2.json"),
    )
    parser.add_argument(
        "--destination",
        help="new mirror path; omitted for verification only",
    )
    args = parser.parse_args(argv)
    if args.destination:
        evidence = stage_dataset_no_replace(
            args.source,
            args.destination,
            pointer_path=args.pointer,
            source_lock_path=args.source_lock,
        )
    else:
        evidence = verify_dataset_root(
            args.source,
            pointer_path=args.pointer,
            source_lock_path=args.source_lock,
        )
    print(json.dumps(evidence.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
