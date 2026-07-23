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


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config")
    mode.add_argument("--capabilities-json", action="store_true")
    ap.add_argument("--resume", default="none", choices=["auto", "none"])
    ap.add_argument("--resume-path")
    ap.add_argument("--resume-sha256")
    args = ap.parse_args()
    if args.capabilities_json:
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
    )
    try:
        if trainer.is_master:
            print(f"done: step={trainer.step} out={trainer.out_dir}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
