#!/usr/bin/env python3
"""Freeze measured functional, resume, and throughput site canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.preflight import build_preflight_receipt
from msctl.profile import load_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cohort-id")
    parser.add_argument("--dataset-receipt-sha256", required=True)
    parser.add_argument("--functional-evidence", required=True)
    parser.add_argument("--resume-evidence", required=True)
    parser.add_argument("--throughput-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = build_preflight_receipt(
        profile=load_profile(args.profile),
        dataset_receipt_sha256=args.dataset_receipt_sha256,
        functional_evidence=args.functional_evidence,
        resume_evidence=args.resume_evidence,
        throughput_evidence=args.throughput_evidence,
        output=args.output,
        **({"cohort_id": args.cohort_id} if args.cohort_id is not None else {}),
    )
    print(json.dumps({"preflight": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
