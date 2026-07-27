#!/usr/bin/env python
"""Publish the canonical post-build receipt for relational staged assets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.relational_assets import (
    AssetReceipt,
    create_asset_receipt,
    publish_asset_receipt,
)
from scripts.freeze_relational_study import (
    FreezeManifest,
    load_freeze_manifest,
    require_launchable_freeze,
)
from scripts.make_relational_manifest import (
    build_asset_specs,
    matrix_plan_sha256,
)


def create_relational_asset_receipt(
    freeze: FreezeManifest | Mapping[str, Any],
    data_root: str | Path,
) -> AssetReceipt:
    validated = require_launchable_freeze(freeze)
    return create_asset_receipt(
        data_root,
        freeze_sha256=validated.freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256(validated),
        specs=build_asset_specs(validated),
    )


def publish_relational_asset_receipt(
    path: str | Path,
    freeze: FreezeManifest | Mapping[str, Any],
    data_root: str | Path,
) -> Path:
    validated = require_launchable_freeze(freeze)
    return publish_asset_receipt(
        path,
        data_root,
        freeze_sha256=validated.freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256(validated),
        specs=build_asset_specs(validated),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash the exact staged files for a frozen relational matrix and "
            "atomically publish their receipt."
        )
    )
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze = require_launchable_freeze(load_freeze_manifest(args.freeze))
    from experiment.provenance import verify_source_provenance

    verify_source_provenance(
        Path(__file__).resolve().parents[1],
        freeze.source_provenance,
    )
    destination = publish_relational_asset_receipt(
        args.out,
        freeze,
        args.data_root,
    )
    print(json.dumps(json.loads(destination.read_bytes()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
