#!/usr/bin/env python
"""Train one arm from a YAML config.

Usage: python scripts/run_train.py --config configs/foo.yaml [--resume auto|none]
"""

import argparse
import json
from pathlib import Path
import sys

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.trainer import train, trainer_capabilities


def _operational_steps(value: str) -> int:
    if (
        not value.isascii()
        or not value.isdecimal()
        or value.startswith("0")
    ):
        raise argparse.ArgumentTypeError(
            "operational-steps must be a canonical positive integer"
        )
    parsed = int(value)
    if parsed > sys.maxsize:
        raise argparse.ArgumentTypeError(
            "operational-steps exceeds the platform integer limit"
        )
    return parsed


class _StoreOperationalSteps(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest) is not None:
            parser.error("--operational-steps may be provided only once")
        setattr(namespace, self.dest, values)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config")
    mode.add_argument("--capabilities-json", action="store_true")
    ap.add_argument("--resume", default="none", choices=["auto", "none"])
    ap.add_argument("--resume-path")
    ap.add_argument("--resume-sha256")
    ap.add_argument(
        "--operational-steps",
        type=_operational_steps,
        action=_StoreOperationalSteps,
        metavar="N",
    )
    args = ap.parse_args(argv)
    if args.capabilities_json:
        if args.operational_steps is not None:
            ap.error("--operational-steps requires --config")
        print(
            json.dumps(
                trainer_capabilities(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    trainer = train(
        cfg,
        resume=args.resume,
        resume_path=args.resume_path,
        resume_sha256=args.resume_sha256,
        operational_steps=args.operational_steps,
    )
    try:
        if trainer.is_master:
            if args.operational_steps is not None:
                metrics = {
                    "end_step": trainer.step,
                    "receipt_type": "memorysplit-operational-training-v1",
                    "start_step": trainer.operational_start_step,
                    "step_tok_s": trainer.operational_step_tok_s,
                    "updates": len(trainer.operational_step_tok_s),
                }
                print(
                    json.dumps(
                        metrics,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                )
            print(f"done: step={trainer.step} out={trainer.out_dir}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
