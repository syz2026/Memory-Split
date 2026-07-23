#!/usr/bin/env python
"""Pack prepared KQA facts and QA into paired continuation-training shards."""

from __future__ import annotations

import argparse
import json

from corpusgen.kqa_build import KQAContinuationConfig, build_kqa_continuation
from train.tokenizer import get_tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--total-tokens", required=True, type=int)
    parser.add_argument("--fact-share", type=float, default=0.70)
    parser.add_argument("--min-fact-exposures", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_kqa_continuation(
        args.prepared_dir,
        get_tok(),
        args.out_dir,
        KQAContinuationConfig(
            total_tokens=args.total_tokens,
            fact_share=args.fact_share,
            min_fact_exposures=args.min_fact_exposures,
            seed=args.seed,
        ),
    )
    print(json.dumps(report, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit("one or more continuation-corpus invariants failed")


if __name__ == "__main__":
    main()
