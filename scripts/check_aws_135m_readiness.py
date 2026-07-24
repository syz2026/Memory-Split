#!/usr/bin/env python3
"""Read-only AWS account, P-instance offering, and quota check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.aws.readiness import inspect_aws_readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            inspect_aws_readiness(region=args.region),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
