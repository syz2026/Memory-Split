#!/usr/bin/env python3
"""Verify, materialize, and freeze the Task-4-to-135M dataset bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.corpus_contract import (  # noqa: E402
    AUTHORITATIVE_RECIPE_SHA256,
    freeze_dataset_pointer,
    materialize_135m_layout,
    sha256_file,
    verify_task4_publication,
)


def _source_document(evidence) -> dict[str, object]:
    return {
        "build_id": evidence.receipt["build_id"],
        "lane_ids": list(evidence.lane_ids),
        "merkle_root_sha256": evidence.receipt["merkle_root_sha256"],
        "ordered_stream_sha256": evidence.ordered_stream_sha256,
        "packed_stream_sha256": evidence.packed_stream_sha256,
        "raw_target_tokens": evidence.raw_target_tokens,
        "receipt_sha256": evidence.receipt_sha256,
        "sidecar_stream_sha256": dict(evidence.sidecar_stream_sha256),
        "source_recipe_sha256": evidence.source_recipe_sha256,
    }


def _add_source_lock(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-lock",
        default=str(ROOT / "configs" / "reasoning-dataset-v2.json"),
    )
    parser.add_argument(
        "--expected-source-lock-sha256",
        default=AUTHORITATIVE_RECIPE_SHA256,
    )


def _add_optional_source_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-receipt-sha256")
    parser.add_argument("--expected-ordered-sha256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify-source",
        help="verify one canonical Task-4 v2 publication",
    )
    verify.add_argument("publication")
    _add_source_lock(verify)
    _add_optional_source_identity(verify)

    materialize = commands.add_parser(
        "materialize",
        help="convert verified Task-4 shards to the flat 135M layout",
    )
    materialize.add_argument("publication")
    materialize.add_argument("destination")
    _add_source_lock(materialize)
    _add_optional_source_identity(materialize)

    freeze = commands.add_parser(
        "freeze-pointer",
        help="freeze actual source and flat-layout hashes into a new pointer",
    )
    freeze.add_argument("publication")
    freeze.add_argument("dataset")
    freeze.add_argument("output")
    freeze.add_argument(
        "--template",
        default=str(ROOT / "DATASET-POINTER-SLURM-135M.json"),
    )
    freeze.add_argument(
        "--source-lock",
        default=str(ROOT / "configs" / "reasoning-dataset-v2.json"),
    )
    freeze.add_argument("--expected-source-receipt-sha256", required=True)
    freeze.add_argument("--expected-ordered-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-source":
        evidence = verify_task4_publication(
            args.publication,
            source_lock_path=args.source_lock,
            expected_source_lock_sha256=args.expected_source_lock_sha256,
            expected_receipt_sha256=args.expected_source_receipt_sha256,
            expected_ordered_sha256=args.expected_ordered_sha256,
        )
        result = _source_document(evidence)
    elif args.command == "materialize":
        evidence = materialize_135m_layout(
            args.publication,
            args.destination,
            source_lock_path=args.source_lock,
            expected_source_lock_sha256=args.expected_source_lock_sha256,
            expected_source_receipt_sha256=args.expected_source_receipt_sha256,
            expected_ordered_sha256=args.expected_ordered_sha256,
        )
        result = evidence.as_dict()
    else:
        output = freeze_dataset_pointer(
            args.publication,
            args.dataset,
            template_path=args.template,
            output_path=args.output,
            expected_source_receipt_sha256=args.expected_source_receipt_sha256,
            expected_ordered_sha256=args.expected_ordered_sha256,
            source_lock_path=args.source_lock,
        )
        result = {
            "pointer": str(output),
            "pointer_sha256": sha256_file(output),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
