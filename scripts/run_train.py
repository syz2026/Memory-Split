#!/usr/bin/env python
"""Train one arm from a YAML config.

Usage: python scripts/train.py --config configs/foo.yaml [--resume auto|none]
"""

import argparse
import sys
from pathlib import Path

import yaml

# Run from anywhere without PYTHONPATH (e.g. Colab): repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--trainer", default="v1", choices=["v1", "v2"],
                    help="v2 = token-weighted gradient accumulation (use for the "
                         "split arm; a no-op for dense). See train/trainer_v2.py.")
    args = ap.parse_args()
    if args.trainer == "v2":
        from train.trainer_v2 import train
    else:
        from train.trainer import train
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    trainer = train(cfg, resume=args.resume)
    print(f"done: step={trainer.step} out={trainer.out_dir}")


if __name__ == "__main__":
    main()
