#!/usr/bin/env python3
"""Seal or verify one deterministic confirmatory evaluation release."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.confirmatory.sealing import (
    SEALED_RELEASE_MANIFEST,
    SealedReleaseResult,
    SealingError,
    VerifiedSealedRelease,
    seal_release,
    verify_release,
)
from evals.confirmatory.contracts import canonical_json_bytes


class _ArgumentError(ValueError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError(message)


def _parser() -> _JsonArgumentParser:
    parser = _JsonArgumentParser(
        description=(
            "Validate frozen v2 confirmatory records and publish or verify "
            "one content-addressed v3 sealed release."
        ),
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--source-dir")
    parser.add_argument("--preregistration")
    parser.add_argument("--out-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify-release")
    parser.add_argument("--expected-release-sha256")
    return parser


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode("utf-8"))


def _counts(value: SealedReleaseResult | VerifiedSealedRelease) -> dict[str, int]:
    return {
        "items": value.item_count,
        "pairs": value.pair_count,
        "worlds": value.world_count,
        "stores": value.store_count,
    }


def _build_payload(
    result: SealedReleaseResult,
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "mode": mode,
        "published": result.published,
        "verified": False,
        "release_id": result.release_id,
        "release_dir": str(result.release_dir),
        "release_sha256": result.release_sha256,
        "manifest": str(result.release_dir / SEALED_RELEASE_MANIFEST),
        "counts": _counts(result),
    }


def _verify_payload(result: VerifiedSealedRelease) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "mode": "verify",
        "published": False,
        "verified": True,
        "release_dir": str(result.release_dir),
        "release_sha256": result.release_sha256,
        "sealed_gold_sha256": result.sealed_gold_sha256,
        "counts": _counts(result),
    }


def _require_build_arguments(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--source-dir", args.source_dir),
            ("--preregistration", args.preregistration),
            ("--out-dir", args.out_dir),
        )
        if value is None
    ]
    if missing:
        raise _ArgumentError(
            "build mode requires " + ", ".join(missing)
        )
    if args.expected_release_sha256 is not None:
        raise _ArgumentError(
            "--expected-release-sha256 is valid only with --verify-release"
        )
    if args.apply and args.dry_run:
        raise _ArgumentError("--apply and --dry-run are mutually exclusive")


def _require_verify_arguments(args: argparse.Namespace) -> None:
    if args.expected_release_sha256 is None:
        raise _ArgumentError(
            "--verify-release requires --expected-release-sha256"
        )
    if any(
        value is not None
        for value in (args.source_dir, args.preregistration, args.out_dir)
    ) or args.apply or args.dry_run:
        raise _ArgumentError(
            "verify mode cannot be combined with build or publication flags"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    mode = "arguments"
    try:
        args = parser.parse_args(argv)
        if args.help:
            _emit(
                {
                    "schema_version": 1,
                    "ok": True,
                    "mode": "help",
                    "published": False,
                    "verified": False,
                    "usage": parser.format_help(),
                }
            )
            return 0
        if args.verify_release is not None:
            mode = "verify"
            _require_verify_arguments(args)
            report = _verify_payload(
                verify_release(
                    release_dir=args.verify_release,
                    expected_release_sha256=args.expected_release_sha256,
                )
            )
        else:
            _require_build_arguments(args)
            apply = bool(args.apply)
            mode = "publish" if apply else "dry-run"
            report = _build_payload(
                seal_release(
                    source_dir=args.source_dir,
                    preregistration_path=args.preregistration,
                    output_root=args.out_dir,
                    apply=apply,
                ),
                mode=mode,
            )
        code = 0
    except (_ArgumentError, SealingError) as exc:
        report = {
            "schema_version": 1,
            "ok": False,
            "mode": mode,
            "published": False,
            "verified": False,
            "error": {
                "code": getattr(exc, "code", "ARGUMENT_ERROR"),
                "message": str(exc),
            },
        }
        code = 2
    except OSError as exc:
        report = {
            "schema_version": 1,
            "ok": False,
            "mode": mode,
            "published": False,
            "verified": False,
            "error": {
                "code": "LOCAL_IO_ERROR",
                "message": str(exc),
            },
        }
        code = 2
    except Exception:
        report = {
            "schema_version": 1,
            "ok": False,
            "mode": mode,
            "published": False,
            "verified": False,
            "error": {
                "code": "SEALING_INTERNAL_ERROR",
                "message": "unexpected local sealing failure",
            },
        }
        code = 70
    _emit(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
