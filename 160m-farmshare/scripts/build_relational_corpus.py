#!/usr/bin/env python
"""Build one relational token stream and four aligned target-weight files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.bed_snapshot import BedSnapshotLock, iter_verified_bed
from corpusgen.relation_schema import RelationSchema
from corpusgen.relational_build import (
    RelationalBuildConfig,
    build_relational_corpus,
)
from experiment.artifacts import sha256_file
from train.tokenizer import get_tok


def _validate_frozen_source_lock(path: Path | str, freeze) -> None:
    if (
        sha256_file(path)
        != freeze.artifact_sha256["source_lock"]
    ):
        raise ValueError("BED source lock hash does not match the study freeze")


def iter_bed_jsonl(
    path: Path | str,
    *,
    lock: BedSnapshotLock | None = None,
):
    path = Path(path)
    if lock is not None and lock.rows == 0:
        raise ValueError(f"{path} contains no natural-text records")
    while True:
        if lock is not None:
            yield from iter_verified_bed(path, lock)
            continue
        saw_text = False
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text") if isinstance(row, dict) else None
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"{path}:{line_number} requires a non-empty text field"
                    )
                saw_text = True
                yield text
        if not saw_text:
            raise ValueError(f"{path} contains no natural-text records")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a shared relational corpus with dense, split, "
            "matched-random, and selective target weights."
        )
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--bed-jsonl", required=True)
    parser.add_argument("--bed-lock", required=True)
    parser.add_argument("--relation-schema", required=True)
    parser.add_argument(
        "--freeze",
        help="required launchable study freeze for protected generation",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("development", "protected"),
        default="protected",
    )
    parser.add_argument(
        "--mixture-index",
        "--development-mixture-index",
        dest="development_mixture_index",
        type=int,
        default=0,
    )
    parser.add_argument("--world-size", type=int, default=64)
    parser.add_argument("--eval-pairs-per-task", type=int, default=10_000)
    parser.add_argument("--eval-pairs-per-world", type=int, default=32)
    parser.add_argument("--guardrail-items", type=int, default=10_000)
    parser.add_argument("--shared-text-eval-count", type=int, default=64)
    args = parser.parse_args(argv)
    freeze = None
    if args.artifact_mode == "protected":
        if args.freeze is None:
            raise ValueError(
                "--freeze is required for protected corpus generation"
            )
        from experiment.provenance import verify_source_provenance
        from scripts.freeze_relational_study import (
            load_freeze_manifest,
            require_launchable_freeze,
        )

        freeze = require_launchable_freeze(
            load_freeze_manifest(args.freeze)
        )
        verify_source_provenance(
            Path(__file__).resolve().parents[1],
            freeze.source_provenance,
        )
        _validate_frozen_source_lock(args.bed_lock, freeze)
    elif args.freeze is not None:
        raise ValueError("--freeze is only valid for protected generation")

    cfg = RelationalBuildConfig(
        n_entities=args.entities,
        total_tokens=args.tokens,
        data_seed=args.data_seed,
        world_size=args.world_size,
        eval_pairs_per_task=args.eval_pairs_per_task,
        eval_pairs_per_world=args.eval_pairs_per_world,
        guardrail_items=args.guardrail_items,
        shared_text_eval_count=args.shared_text_eval_count,
        artifact_mode=args.artifact_mode,
        development_mixture_index=args.development_mixture_index,
    )
    report = build_relational_corpus(
        cfg,
        get_tok(),
        iter_bed_jsonl(
            args.bed_jsonl,
            lock=BedSnapshotLock.from_path(args.bed_lock),
        ),
        Path(args.out),
        relation_schema=RelationSchema.from_path(args.relation_schema),
        freeze_manifest=freeze,
    )
    print(json.dumps(report["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
