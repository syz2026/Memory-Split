#!/usr/bin/env python
"""Freeze the reasoning-v2 source lock from upstream metadata and report gaps.

The default is offline: point ``--metadata-record`` at a recorded transcript
and the whole resolution replays without a socket. ``--network`` is the only
switch that opens one, and it contacts nothing but the hosts printed in the
egress line of the report.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_v2.source_resolve import (
    HttpsSourceMetadataClient,
    SourcePinPlan,
    load_metadata_record,
    metadata_egress_hosts,
    resolve_source_pins,
)


def _write_new_file(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite: the artifact already exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_rows(plan: SourcePinPlan) -> list[tuple[str, ...]]:
    unpinned = set(plan.unpinned_source_ids)
    rows = []
    for source in plan.sources:
        pins = (*source.files, *source.candidate_files)
        rows.append(
            (
                "UNPINNED" if source.source_id in unpinned else "FROZEN",
                source.source_id,
                source.revision,
                source.revision_provenance,
                f"{len(source.files)}+{len(source.candidate_files)}",
                f"{sum(row.bytes for row in pins):,}",
            )
        )
    return rows


def _render_report(plan: SourcePinPlan) -> str:
    budget = plan.byte_budget()
    lines = [
        f"source pin plan sha256        {plan.sha256}",
        f"reviewed catalog sha256       {plan.source_catalog_sha256}",
        f"metadata evidence sha256      {plan.evidence_sha256}",
        f"metadata egress hosts         {', '.join(metadata_egress_hosts())}",
        "",
        f"{'STATUS':<9}{'SOURCE':<32}{'REVISION':<42}"
        f"{'REVISION FROM':<21}{'PINS':<8}PINNED BYTES",
    ]
    for status, source_id, revision, provenance, pins, byte_count in _source_rows(plan):
        lines.append(
            f"{status:<9}{source_id:<32}{revision:<42}"
            f"{provenance:<21}{pins:<8}{byte_count:>18}"
        )
    lines.append("")
    if plan.complete:
        lines.append("every reviewed binding is pinned from metadata")
    else:
        lines.append("UNPINNED bindings, each of which needs bytes:")
        for row in plan.unpinnable:
            cost = (
                "bytes unknown"
                if row.bytes_required is None
                else f"{row.bytes_required:,} bytes ({row.byte_estimate})"
            )
            target = row.path or "(whole inventory)"
            lines.append(f"  {row.source_id}  {row.binding}  {target}")
            lines.append(f"      {cost}")
            lines.append(f"      {row.reason}")
    lines.extend(
        [
            "",
            "byte budget",
            f"  already pinned, needed only to run the build   "
            f"{budget['pinned_bytes']:>18,}",
            f"  needed to freeze the lock, exactly known       "
            f"{budget['freeze_bytes_exact']:>18,}",
            f"  needed to freeze the lock, upper bound         "
            f"{budget['freeze_bytes_upper_bound']:>18,}",
            f"  bindings whose byte cost metadata cannot bound "
            f"{budget['freeze_bindings_without_estimate']:>18,}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the reviewed reasoning-v2 source catalog into a pin plan. "
            "Bindings that upstream metadata cannot prove are reported, never "
            "defaulted."
        )
    )
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument(
        "--metadata-record",
        type=Path,
        help="replay a recorded metadata transcript; makes the run offline",
    )
    transport.add_argument(
        "--network",
        action="store_true",
        help="query git ls-remote and the Hugging Face metadata API directly",
    )
    parser.add_argument("--out", type=Path, required=True, help="pin plan destination")
    parser.add_argument(
        "--record",
        type=Path,
        help="write the metadata transcript this run consumed",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit zero even while bindings stay unproven",
    )
    args = parser.parse_args(argv)

    client = (
        HttpsSourceMetadataClient()
        if args.network
        else load_metadata_record(args.metadata_record)
    )
    evidence: list[dict[str, object]] = []
    plan = resolve_source_pins(client, evidence_sink=evidence)
    _write_new_file(args.out, plan.to_bytes())
    if args.record is not None:
        _write_new_file(args.record, canonical_json_bytes(evidence[0]))
    print(_render_report(plan))
    if plan.complete or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
