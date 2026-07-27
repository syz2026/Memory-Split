#!/usr/bin/env python
"""Build frozen, evaluation-only Wikidata5M robustness artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.relation_schema import RelationSchema
from corpusgen.wikidata_paths import build_wikidata_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build unique-only Wikidata5M valid/test robustness paths. "
            "This command emits evaluation artifacts only."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="directory containing the exact locked Wikidata5M archives",
    )
    parser.add_argument("--relation-schema", required=True)
    parser.add_argument("--split", required=True, choices=("valid", "test"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    manifest = build_wikidata_paths(
        args.source,
        RelationSchema.from_path(args.relation_schema),
        args.split,
        args.out,
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
