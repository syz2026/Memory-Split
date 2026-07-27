#!/usr/bin/env python3
"""Revalidate immutable pair/profile/preflight identities on a compute node."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.adapters.slurm import load_pair_manifest
from msctl.preflight import validate_preflight
from msctl.profile import load_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--mode",
        choices=("functional", "resume", "throughput", "protected"),
        required=True,
    )
    parser.add_argument("--preflight")
    args = parser.parse_args(argv)
    pair = load_pair_manifest(args.manifest)
    profile = load_profile(args.profile)
    if pair["profile_sha256"] != profile.sha256:
        raise ValueError("pair manifest profile identity changed")
    if args.mode == "protected":
        if args.preflight is None:
            parser.error("--preflight is required for protected mode")
        validate_preflight(
            args.preflight,
            profile=profile,
            dataset_receipt_sha256=pair["dataset"]["receipt_sha256"],
            cohort_id=pair["cohort_id"],
        )
    print(
        json.dumps(
            {
                "dataset_receipt_sha256": pair["dataset"]["receipt_sha256"],
                "pair_id": pair["pair_id"],
                "profile_sha256": profile.sha256,
                "validated": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
