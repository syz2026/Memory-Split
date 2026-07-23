#!/usr/bin/env python
"""Train one arm from a YAML config.

Usage: python scripts/run_train.py --config configs/foo.yaml [--resume auto|none]
"""

import argparse

import yaml

from train.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="none", choices=["auto", "none"])
    ap.add_argument("--resume-path")
    ap.add_argument("--resume-sha256")
    args = ap.parse_args()
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
