#!/usr/bin/env python
"""Train one arm from a YAML config.

Usage:
  python scripts/run_train.py --config configs/foo.yaml [--resume auto|none]
      [--init-from /path/to/prior/ckpt.pt]

``--init-from`` loads model weights only and starts a fresh optimizer, data
cursor, and LR schedule.  It is the correct mode for continued training on a
new corpus.  ``--resume auto`` takes precedence when the new run already has
its own checkpoint.
"""

import argparse

import yaml

from train.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--init-from")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    trainer = train(cfg, resume=args.resume, init_from=args.init_from)
    print(f"done: step={trainer.step} out={trainer.out_dir}")


if __name__ == "__main__":
    main()
