from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import experiment.relational_assets as relational_assets
from experiment.artifacts import atomic_write_json, canonical_json_bytes
from experiment.relational_assets import (
    AssetReceipt,
    StagedAssetSpec,
    create_asset_receipt,
    load_asset_receipt,
    publish_asset_receipt,
    validate_asset_receipt,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _stage(
    root: Path,
    *,
    freeze_sha256: str,
    build_metadata: dict | None = None,
) -> tuple[tuple[StagedAssetSpec, ...], dict[str, bytes]]:
    build_rel = "relational/d160m/n50k/seed-1001"
    build = root / build_rel
    build.mkdir(parents=True)
    payloads = {
        f"{build_rel}/train.bin": b"\x00\x01\x02\x03",
        f"{build_rel}/dense.weights.bin": b"\x01\x01",
    }
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)
    metadata = build_metadata or {
        "record_type": "relational_protected_build",
        "schema_version": 1,
        "freeze_sha256": freeze_sha256,
        "model": "d160m",
        "load_role": "low",
        "entities": 50_000,
        "data_seed": 1001,
        "raw_positions": 1_622_016_000,
        "stream_commitment_sha256": _digest("stream-commitment"),
        "weights_commitment_sha256": {
            "dense": _digest("dense-commitment"),
            "random": _digest("random-commitment"),
            "split": _digest("split-commitment"),
        },
    }
    manifest = {
        "schema_version": 1,
        "study_freeze_sha256": freeze_sha256,
        "protected_build": metadata,
        "artifacts": [
            {
                "path": Path(relative).name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for relative, payload in sorted(payloads.items())
        ],
    }
    (build / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return (
        (
            StagedAssetSpec(
                kind="stream",
                path=f"{build_rel}/train.bin",
                commitment_sha256=_digest("stream-commitment"),
                build_rel=build_rel,
                build_metadata=metadata,
            ),
            StagedAssetSpec(
                kind="weights",
                path=f"{build_rel}/dense.weights.bin",
                commitment_sha256=_digest("dense-commitment"),
                build_rel=build_rel,
                build_metadata=metadata,
            ),
        ),
        payloads,
    )


def test_receipt_records_actual_bytes_and_round_trips_canonically(tmp_path):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, payloads = _stage(data_root, freeze_sha256=freeze_sha256)

    receipt = create_asset_receipt(
        data_root,
        freeze_sha256=freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256,
        specs=specs,
    )
    path = publish_asset_receipt(
        tmp_path / "asset-receipt.json",
        data_root,
        freeze_sha256=freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256,
        specs=specs,
    )

    assert path.read_bytes() == canonical_json_bytes(receipt.to_dict())
    assert load_asset_receipt(path) == receipt
    assert validate_asset_receipt(
        receipt,
        freeze_sha256=freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256,
        specs=specs,
    ) == receipt
    records = {record.path: record for record in receipt.assets}
    assert set(records) == set(payloads)
    for relative, payload in payloads.items():
        assert records[relative].sha256 == hashlib.sha256(payload).hexdigest()
        assert records[relative].bytes == len(payload)
        assert records[relative].build_manifest_sha256 == hashlib.sha256(
            (data_root / Path(relative).parent / "manifest.json").read_bytes()
        ).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_receipt_validation_rejects_nonexact_asset_inventory(tmp_path, mutation):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, _ = _stage(data_root, freeze_sha256=freeze_sha256)
    raw = create_asset_receipt(
        data_root,
        freeze_sha256=freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256,
        specs=specs,
    ).to_dict()
    if mutation == "missing":
        raw["assets"].pop()
    elif mutation == "extra":
        extra = copy.deepcopy(raw["assets"][-1])
        extra["path"] = "relational/d160m/n50k/seed-1001/extra.bin"
        raw["assets"].append(extra)
        raw["assets"].sort(key=lambda item: (item["kind"], item["path"]))
    else:
        raw["assets"].append(copy.deepcopy(raw["assets"][0]))
    material = dict(raw)
    material.pop("receipt_sha256")
    from experiment.artifacts import canonical_sha256

    raw["receipt_sha256"] = canonical_sha256(material)

    if mutation == "duplicate":
        with pytest.raises(ValueError, match="duplicate"):
            AssetReceipt.from_dict(raw)
    else:
        parsed = AssetReceipt.from_dict(raw)
        with pytest.raises(ValueError, match="missing|extra|inventory"):
            validate_asset_receipt(
                parsed,
                freeze_sha256=freeze_sha256,
                matrix_plan_sha256=matrix_plan_sha256,
                specs=specs,
            )


def test_receipt_rejects_wrong_build_metadata_and_symlinked_assets(tmp_path):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, _ = _stage(data_root, freeze_sha256=freeze_sha256)
    manifest_path = data_root / specs[0].build_rel / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_bytes())
    manifest["protected_build"]["data_seed"] = 1002
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="build metadata"):
        create_asset_receipt(
            data_root,
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            specs=specs,
        )

    manifest["protected_build"]["data_seed"] = 1001
    atomic_write_json(manifest_path, manifest)
    stream = data_root / specs[0].path
    target = stream.with_suffix(".real")
    stream.rename(target)
    stream.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|canonical|regular"):
        create_asset_receipt(
            data_root,
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            specs=specs,
        )


def test_receipt_rejects_wrong_build_manifest_freeze(tmp_path):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, _ = _stage(data_root, freeze_sha256=freeze_sha256)
    manifest_path = data_root / specs[0].build_rel / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["study_freeze_sha256"] = _digest("different-freeze")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="freeze metadata"):
        create_asset_receipt(
            data_root,
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            specs=specs,
        )


def test_receipt_rejects_inconsistent_shared_stream_specs(tmp_path):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, _ = _stage(data_root, freeze_sha256=freeze_sha256)
    conflicting = StagedAssetSpec(
        kind="stream",
        path=specs[0].path,
        commitment_sha256=_digest("different-stream-commitment"),
        build_rel=specs[0].build_rel,
        build_metadata=specs[0].build_metadata,
    )

    with pytest.raises(ValueError, match="inconsistent|duplicate"):
        create_asset_receipt(
            data_root,
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            specs=(*specs, conflicting),
        )


def test_receipt_spec_rejects_path_traversal():
    with pytest.raises(ValueError, match="relative|traversal"):
        StagedAssetSpec(
            kind="stream",
            path="../train.bin",
            commitment_sha256=_digest("commitment"),
            build_rel="relational/build",
            build_metadata={},
        )


def test_receipt_rejects_duplicate_path_even_with_different_kind(tmp_path):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, _ = _stage(data_root, freeze_sha256=freeze_sha256)
    raw = create_asset_receipt(
        data_root,
        freeze_sha256=freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256,
        specs=specs,
    ).to_dict()
    raw["assets"][1]["path"] = raw["assets"][0]["path"]
    from experiment.artifacts import canonical_sha256

    material = dict(raw)
    material.pop("receipt_sha256")
    raw["receipt_sha256"] = canonical_sha256(material)

    with pytest.raises(ValueError, match="duplicate path"):
        AssetReceipt.from_dict(raw)


def test_failed_hashing_never_publishes_partial_receipt(tmp_path, monkeypatch):
    freeze_sha256 = _digest("freeze")
    matrix_plan_sha256 = _digest("matrix-plan")
    data_root = tmp_path / "data"
    specs, _ = _stage(data_root, freeze_sha256=freeze_sha256)
    destination = tmp_path / "asset-receipt.json"

    def changed_while_hashing(_path):
        raise ValueError("artifact changed while it was being hashed")

    monkeypatch.setattr(relational_assets, "sha256_file", changed_while_hashing)
    with pytest.raises(ValueError, match="changed while"):
        publish_asset_receipt(
            destination,
            data_root,
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            specs=specs,
        )
    assert not destination.exists()


def test_asset_receipt_schema_is_closed_and_documents_exact_inventory():
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "relational-asset-receipt-v1.schema.json"
        ).read_text()
    )

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    asset = schema["$defs"]["AssetRecord"]
    assert asset["additionalProperties"] is False
    assert set(asset["required"]) == set(asset["properties"])
    assert schema["properties"]["assets"]["minItems"] == 50
    assert schema["properties"]["assets"]["maxItems"] == 50
    assert schema["properties"]["assets"]["uniqueItems"] is True


def test_asset_receipt_command_runs_repo_relative():
    completed = subprocess.run(
        [sys.executable, "scripts/publish_relational_assets.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--data-root" in completed.stdout
    assert "--freeze" in completed.stdout
