from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.make_relational_manifest import (
    build_asset_specs,
    build_manifest,
)
from scripts.publish_relational_assets import create_relational_asset_receipt
from tests.test_relational_freeze import valid_frozen_freeze


_SIDECAR_BYTES = {
    "dense": b"\x01",
    "split": b"\x02",
    "random": b"\x03",
    "selective": b"\x04",
}


def stage_launchable_relational_assets(tmp_path: Path, *, freeze=None):
    freeze = valid_frozen_freeze() if freeze is None else freeze
    data_root = tmp_path / "data"
    grouped = {}
    for spec in build_asset_specs(freeze):
        grouped.setdefault(spec.build_rel, spec.build_metadata)
        assert grouped[spec.build_rel] == spec.build_metadata

    for build_rel, metadata in grouped.items():
        build = data_root / build_rel
        build.mkdir(parents=True)
        payloads = {"train.bin": b"\x00\x00" * 9_000}
        payloads.update(
            {
                f"{condition}.weights.bin": unit * 9_000
                for condition, unit in _SIDECAR_BYTES.items()
            }
        )
        for name, payload in payloads.items():
            (build / name).write_bytes(payload)
        build_manifest_payload = {
            "schema_version": 1,
            "study_freeze_sha256": freeze.freeze_sha256,
            "protected_build": {
                key: (
                    dict(value)
                    if key == "weights_commitment_sha256"
                    else value
                )
                for key, value in metadata.items()
            },
            "artifacts": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
                for name, payload in sorted(payloads.items())
            ],
        }
        (build / "manifest.json").write_text(
            json.dumps(build_manifest_payload, indent=2, sort_keys=True) + "\n"
        )

    receipt = create_relational_asset_receipt(freeze, data_root)
    manifest = build_manifest(freeze, asset_receipt=receipt)
    return freeze, receipt, manifest, data_root
