#!/usr/bin/env python
"""Build or freshly verify the frozen Wikidata5M hashmap archive."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.wikidata5m_hashmap import (
    build_wikidata5m_hashmap,
    verify_wikidata5m_hashmap,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or freshly verify the frozen 3,000-key Wikidata5M hashmap."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="build and atomically publish the frozen archive",
    )
    build.add_argument("--source-root", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--work-root", required=True)

    verify = subparsers.add_parser(
        "verify",
        help="freshly rebuild and byte-verify an existing archive",
    )
    verify.add_argument("--source-root", required=True)
    verify.add_argument("--archive", required=True)
    verify.add_argument("--work-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        report = build_wikidata5m_hashmap(
            args.source_root,
            args.out,
            work_root=args.work_root,
        )
    else:
        report = verify_wikidata5m_hashmap(
            args.archive,
            args.source_root,
            work_root=args.work_root,
        )
    print(
        json.dumps(
            asdict(report),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
