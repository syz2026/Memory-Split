#!/usr/bin/env python
"""Build the deterministic, non-scientific packaged current smoke corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.current_dataset import build_fixture_current_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-archive-free 131,072-token current smoke fixture."
        )
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = build_fixture_current_dataset(Path(args.out), total_tokens=131_072)
    if report["profile"] != "smoke" or report["scientific_result"] is not False:
        raise AssertionError("packaged smoke output lost its non-scientific identity")
    print(
        json.dumps(
            {
                "profile": report["profile"],
                "scientific_result": report["scientific_result"],
                "tokens": report["tokens"]["total"],
                "checks": report["checks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
