#!/usr/bin/env python3
"""Create one no-replace role manifest after full-corpus verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.cohort import ROLES  # noqa: E402
from msctl.manifest import create_role_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=tuple(ROLES))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pointer",
        default=str(ROOT / "DATASET-POINTER-SLURM-135M.json"),
    )
    parser.add_argument(
        "--source-lock",
        default=str(ROOT / "configs" / "reasoning-dataset-v2.json"),
    )
    args = parser.parse_args(argv)
    output = create_role_manifest(
        args.role,
        args.output,
        dataset_root=args.dataset_root,
        pointer_path=args.pointer,
        source_lock_path=args.source_lock,
        repository_root=ROOT,
    )
    print(json.dumps({"manifest": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
