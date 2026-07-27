#!/usr/bin/env python
"""Download only the frozen Wikidata5M archives, then verify their bytes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.wikidata5m import WikidataLock, verify_archives


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "sources" / "wikidata5m.lock.json"


def build_download_command(
    data_root: str | Path,
    lock_path: str | Path = DEFAULT_LOCK,
) -> list[str]:
    lock = WikidataLock.from_path(lock_path)
    command = [
        "hf",
        "download",
        lock.repo_id,
        "--repo-type",
        lock.repo_type,
        "--revision",
        lock.revision,
    ]
    for name in lock.files:
        command.extend(("--include", name))
    command.extend(("--local-dir", str(Path(data_root) / "wikidata5m")))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the frozen Wikidata5M archives."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="local data root (defaults to DATA_ROOT)",
    )
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = parser.parse_args(argv)
    if not args.data_root:
        parser.error("--data-root or DATA_ROOT is required")

    lock = WikidataLock.from_path(args.lock)
    subprocess.run(
        build_download_command(args.data_root, args.lock),
        check=True,
    )
    verify_archives(Path(args.data_root) / "wikidata5m", lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
