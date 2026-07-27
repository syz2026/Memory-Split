#!/usr/bin/env python3
"""Dry-run by default wrapper for paired MIT Slurm submissions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.operations import submit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_manifests", nargs="+")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--venv-root", required=True)
    parser.add_argument(
        "--mode",
        choices=("functional", "resume", "throughput", "protected"),
        default="functional",
    )
    parser.add_argument("--preflight")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report = submit(
        args.pair_manifests,
        profile_path=args.profile,
        mode=args.mode,
        venv_root=args.venv_root,
        preflight_path=args.preflight,
        apply=args.execute,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        print("dry-run; pass --execute to submit", file=sys.stderr)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
