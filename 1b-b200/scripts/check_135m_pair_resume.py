#!/usr/bin/env python3
"""Fail closed unless a protected pair can resume at one exact cursor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.adapters.slurm import load_pair_manifest  # noqa: E402
from msctl.operations import inspect_paired_resume  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    pair = load_pair_manifest(args.manifest)
    by_arm = {item["arm"]: item for item in pair["arms"]}
    state = inspect_paired_resume(
        Path(by_arm["dense"]["out_dir"]) / "ckpt.pt",
        Path(by_arm["split90"]["out_dir"]) / "ckpt.pt",
    )
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
