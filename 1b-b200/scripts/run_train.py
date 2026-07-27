#!/usr/bin/env python
"""Train one arm from a YAML config.

Usage: python scripts/train.py --config configs/foo.yaml [--resume auto|none]
"""

import argparse

import yaml

from train.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    trainer = train(cfg, resume=args.resume)
    print(f"done: step={trainer.step} out={trainer.out_dir}")


if __name__ == "__main__":
    main()
