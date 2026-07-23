#!/usr/bin/env python3
"""Run the frozen N=10 confirmatory analysis from collected paired rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.confirmatory.inference import PairedObservation  # noqa: E402
from evals.confirmatory.reporting import (  # noqa: E402
    ValidityGates,
    build_confirmatory_report,
)
from evals.confirmatory.study_lock import load_135m_study_lock  # noqa: E402


def _load_observations(path: Path) -> list[PairedObservation]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("observation JSONL is missing or unsafe")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            raw = json.loads(line)
            rows.append(PairedObservation(**raw))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid observation at line {line_number}") from error
    return rows


def _load_validity(path: Path) -> ValidityGates:
    if not path.is_file() or path.is_symlink():
        raise ValueError("validity receipt is missing or unsafe")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("validity receipt must contain an object")
    if "failures" in raw:
        raw["failures"] = tuple(raw["failures"])
    return ValidityGates(**raw)


def _write_no_replace(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--validity", required=True)
    parser.add_argument(
        "--family-seed-differences",
        help="JSON object with graph and non_path arrays in frozen seed order",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    args = parser.parse_args(argv)
    families = (
        json.loads(Path(args.family_seed_differences).read_text())
        if args.family_seed_differences
        else None
    )
    report = build_confirmatory_report(
        _load_observations(Path(args.observations)),
        study_lock=load_135m_study_lock(args.repository_root),
        validity=_load_validity(Path(args.validity)),
        family_seed_differences=families,
    )
    _write_no_replace(Path(args.output), report)
    print(json.dumps(report["decision"], sort_keys=True))
    return 2 if report["decision"]["verdict"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
