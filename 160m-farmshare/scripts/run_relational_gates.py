#!/usr/bin/env python
"""Evaluate development Gates 1--4 and bind optional Gate 0/5 evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.relational_gates import (
    build_gate_0_receipt,
    build_gate_5_receipt,
    evaluate_development_gates,
    load_development_gate_inputs,
)
from experiment.artifacts import atomic_write_json, load_canonical_json


def run_gates(
    development_input: str | Path,
    *,
    smoke_report: str | Path | None = None,
    gate_5_receipt: str | Path | None = None,
    gate_5_binding: str | Path | None = None,
) -> dict[str, dict]:
    inputs = load_development_gate_inputs(development_input)
    receipts = evaluate_development_gates(inputs)
    common = inputs["input_hashes"]
    if smoke_report is not None:
        receipts["gate_0"] = build_gate_0_receipt(
            load_canonical_json(smoke_report),
            input_hashes=common,
        )
    if (gate_5_receipt is None) != (gate_5_binding is None):
        raise ValueError(
            "Gate 5 receipt and development binding must be supplied together"
        )
    if gate_5_receipt is not None and gate_5_binding is not None:
        receipts["gate_5"] = build_gate_5_receipt(
            load_canonical_json(gate_5_receipt),
            input_hashes=common,
            gate_3_receipt=receipts["gate_3"],
            gate_4_receipt=receipts["gate_4"],
            development_binding=load_canonical_json(gate_5_binding),
        )
    return dict(sorted(receipts.items()))


def publish_gate_receipts(
    output: str | Path,
    receipts: dict[str, dict],
) -> Path:
    destination = Path(output)
    if destination.is_symlink() or os.path.lexists(destination):
        raise FileExistsError(
            f"gate receipt output already exists: {destination}"
        )
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError("gate output parent must be a regular directory")
    destination.mkdir()
    try:
        for name, receipt in sorted(receipts.items()):
            atomic_write_json(destination / f"{name}.json", receipt)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen relational development gates."
    )
    parser.add_argument("--development-input", required=True)
    parser.add_argument("--smoke-report")
    parser.add_argument("--gate-5-receipt")
    parser.add_argument("--gate-5-binding")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipts = run_gates(
        args.development_input,
        smoke_report=args.smoke_report,
        gate_5_receipt=args.gate_5_receipt,
        gate_5_binding=args.gate_5_binding,
    )
    publish_gate_receipts(args.out, receipts)
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
