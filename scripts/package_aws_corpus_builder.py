#!/usr/bin/env python3
"""Build a deterministic, closed-world AWS corpus builder package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cluster.aws.corpus_builder.package import (  # noqa: E402
    build_corpus_builder_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    package = build_corpus_builder_package(args.source_root, args.output_dir)
    print(package.manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
