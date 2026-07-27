#!/usr/bin/env python
"""Commit and run the blinded prospective Gate-5 design analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.relational_contracts import EvalRow
from evals.relational_design import (
    ProtectedIdentityRegistry,
    commit_arm_label_permutation,
    load_blinded_development,
    open_arm_label_permutation,
    run_prospective_design,
    write_design_receipt,
)


def _read_rows(path: Path) -> tuple[EvalRow, ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"row input must be a regular file: {path}")
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                raw = json.loads(line)
                rows.append(EvalRow.from_dict(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid EvalRow at {path}:{line_number}"
                ) from exc
    if not rows:
        raise ValueError(f"row input is empty: {path}")
    return tuple(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the temporally separated blinded Gate-5 workflow."
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    commit = subparsers.add_parser(
        "commit",
        help="persist arm-label commitment before loading outcomes",
    )
    commit.add_argument("--out", required=True)
    commit.add_argument("--development-split", required=True)
    commit.add_argument("--development-dense", required=True)
    commit.add_argument("--permutation-seed", required=True, type=int)

    analyze = subparsers.add_parser(
        "analyze",
        help="load blinded development outcomes and simulate power",
    )
    analyze.add_argument("--commitment", required=True)
    analyze.add_argument("--development-split", required=True)
    analyze.add_argument("--development-dense", required=True)
    analyze.add_argument(
        "--protected-rows",
        required=True,
        action="append",
        help="protected EvalRow JSONL; repeat for each protected source",
    )
    analyze.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    planned = {
        "split": Path(args.development_split),
        "dense": Path(args.development_dense),
    }
    if args.phase == "commit":
        commitment = commit_arm_label_permutation(
            Path(args.out),
            planned_inputs=planned,
            rng_seed=args.permutation_seed,
        )
        print(commitment.commitment_sha256)
        return

    commitment = open_arm_label_permutation(
        Path(args.commitment),
        planned_inputs=planned,
    )
    protected_paths = tuple(Path(value) for value in args.protected_rows)
    protected_rows = tuple(
        row for path in protected_paths for row in _read_rows(path)
    )
    protected = ProtectedIdentityRegistry.from_rows(
        protected_rows,
        paths=protected_paths,
    )
    development = load_blinded_development(commitment, _read_rows)
    receipt = run_prospective_design(
        development,
        protected=protected,
    )
    write_design_receipt(Path(args.out), receipt)
    print(json.dumps(receipt.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
