"""Canonical staged-asset receipts for protected relational runs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiment.artifacts import (
    atomic_write_json,
    canonical_sha256,
    load_canonical_json,
    require_regular_file,
    sha256_file,
    validate_relative_path,
    validate_sha256,
)


_ASSET_KINDS = {"stream", "weights"}
_RECEIPT_FIELDS = {
    "record_type",
    "schema_version",
    "freeze_sha256",
    "matrix_plan_sha256",
    "assets",
    "receipt_sha256",
}
_ASSET_FIELDS = {
    "kind",
    "path",
    "commitment_sha256",
    "sha256",
    "bytes",
    "build_manifest_sha256",
}


def _absolute_without_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _regular_directory(path: str | Path, name: str) -> Path:
    candidate = Path(path)
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"{name} path cannot contain traversal")
    absolute = _absolute_without_resolution(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{name} must be a regular non-symlink directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} must be a regular directory") from exc
    if resolved != absolute:
        raise ValueError(f"{name} path is not canonical or traverses a symlink")
    metadata = candidate.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be a regular directory")
    return absolute


def _freeze_json(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("build metadata keys must be strings")
            return MappingProxyType(
                {key: freeze(child) for key, child in sorted(item.items())}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise ValueError("build metadata is not canonical JSON")

    return freeze(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class StagedAssetSpec:
    """One expected external file and the build that produced it."""

    kind: str
    path: str
    commitment_sha256: str
    build_rel: str
    build_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in _ASSET_KINDS:
            raise ValueError("staged asset kind must be stream or weights")
        path = validate_relative_path(self.path, "staged asset path")
        build_rel = validate_relative_path(
            self.build_rel,
            "staged asset build path",
        )
        if Path(path).parent.as_posix() != build_rel:
            raise ValueError("staged asset must be directly inside its build path")
        validate_sha256(
            self.commitment_sha256,
            "staged asset commitment SHA-256",
        )
        if not isinstance(self.build_metadata, Mapping):
            raise ValueError("staged asset build metadata must be an object")
        object.__setattr__(
            self,
            "build_metadata",
            _freeze_json(self.build_metadata),
        )


@dataclass(frozen=True)
class AssetRecord:
    kind: str
    path: str
    commitment_sha256: str
    sha256: str
    bytes: int
    build_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in _ASSET_KINDS:
            raise ValueError("asset receipt kind must be stream or weights")
        validate_relative_path(self.path, "asset receipt path")
        validate_sha256(
            self.commitment_sha256,
            "asset receipt commitment SHA-256",
        )
        validate_sha256(self.sha256, "asset receipt file SHA-256")
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or self.bytes < 1
        ):
            raise ValueError("asset receipt byte count must be a positive integer")
        validate_sha256(
            self.build_manifest_sha256,
            "asset receipt build manifest SHA-256",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "commitment_sha256": self.commitment_sha256,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "build_manifest_sha256": self.build_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AssetRecord":
        if not isinstance(raw, Mapping) or set(raw) != _ASSET_FIELDS:
            raise ValueError("asset receipt record fields are not exact")
        return cls(
            kind=raw["kind"],
            path=raw["path"],
            commitment_sha256=raw["commitment_sha256"],
            sha256=raw["sha256"],
            bytes=raw["bytes"],
            build_manifest_sha256=raw["build_manifest_sha256"],
        )


@dataclass(frozen=True)
class AssetReceipt:
    freeze_sha256: str
    matrix_plan_sha256: str
    assets: tuple[AssetRecord, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.freeze_sha256, "asset receipt freeze SHA-256")
        validate_sha256(
            self.matrix_plan_sha256,
            "asset receipt matrix plan SHA-256",
        )
        assets = tuple(self.assets)
        if not assets or any(not isinstance(asset, AssetRecord) for asset in assets):
            raise ValueError("asset receipt must contain asset records")
        keys = [(asset.kind, asset.path) for asset in assets]
        paths = [asset.path for asset in assets]
        if len(keys) != len(set(keys)) or len(paths) != len(set(paths)):
            raise ValueError("asset receipt contains a duplicate path")
        if keys != sorted(keys):
            raise ValueError("asset receipt records are not canonically ordered")
        object.__setattr__(self, "assets", assets)
        validate_sha256(self.receipt_sha256, "asset receipt SHA-256")

    def _without_hash(self) -> dict[str, Any]:
        return {
            "record_type": "relational_asset_receipt",
            "schema_version": 1,
            "freeze_sha256": self.freeze_sha256,
            "matrix_plan_sha256": self.matrix_plan_sha256,
            "assets": [asset.to_dict() for asset in self.assets],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._without_hash(),
            "receipt_sha256": self.receipt_sha256,
        }

    as_dict = to_dict

    @classmethod
    def create(
        cls,
        *,
        freeze_sha256: str,
        matrix_plan_sha256: str,
        assets: Sequence[AssetRecord],
    ) -> "AssetReceipt":
        provisional = cls(
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            assets=tuple(sorted(assets, key=lambda item: (item.kind, item.path))),
            receipt_sha256="0" * 64,
        )
        return cls(
            freeze_sha256=freeze_sha256,
            matrix_plan_sha256=matrix_plan_sha256,
            assets=provisional.assets,
            receipt_sha256=canonical_sha256(provisional._without_hash()),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AssetReceipt":
        if not isinstance(raw, Mapping) or set(raw) != _RECEIPT_FIELDS:
            raise ValueError("asset receipt fields are not exact")
        if (
            raw["record_type"] != "relational_asset_receipt"
            or raw["schema_version"] != 1
            or not isinstance(raw["assets"], list)
        ):
            raise ValueError("asset receipt protocol is invalid")
        receipt = cls(
            freeze_sha256=raw["freeze_sha256"],
            matrix_plan_sha256=raw["matrix_plan_sha256"],
            assets=tuple(AssetRecord.from_dict(item) for item in raw["assets"]),
            receipt_sha256=raw["receipt_sha256"],
        )
        if receipt.receipt_sha256 != canonical_sha256(receipt._without_hash()):
            raise ValueError("asset receipt hash mismatch")
        return receipt


def _normalize_specs(
    specs: Sequence[StagedAssetSpec],
) -> dict[tuple[str, str], StagedAssetSpec]:
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("staged asset specs must be a sequence")
    normalized: dict[tuple[str, str], StagedAssetSpec] = {}
    path_identity: dict[str, tuple[str, str, str, Any]] = {}
    for spec in specs:
        if not isinstance(spec, StagedAssetSpec):
            raise TypeError("staged asset spec has the wrong type")
        key = (spec.kind, spec.path)
        identity = (
            spec.kind,
            spec.commitment_sha256,
            spec.build_rel,
            _thaw(spec.build_metadata),
        )
        previous = path_identity.get(spec.path)
        if previous is not None and previous != identity:
            raise ValueError("inconsistent duplicate staged asset path")
        path_identity[spec.path] = identity
        normalized[key] = spec
    if not normalized:
        raise ValueError("staged asset plan cannot be empty")
    return normalized


def _artifact_inventory(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("build manifest artifact inventory is invalid")
    inventory: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("build manifest artifact record is invalid")
        path = validate_relative_path(item["path"], "build artifact path")
        if path in inventory:
            raise ValueError("build manifest contains a duplicate artifact path")
        validate_sha256(item["sha256"], "build artifact SHA-256")
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
        ):
            raise ValueError("build artifact byte count is invalid")
        inventory[path] = item
    return inventory


def _same_file_metadata(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        stat.S_ISREG(before.st_mode)
        and stat.S_ISREG(after.st_mode)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _load_build_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    source = require_regular_file(path, name="relational build manifest")
    before = source.stat(follow_symlinks=False)
    content = source.read_bytes()
    after_read = source.stat(follow_symlinks=False)
    if not _same_file_metadata(before, after_read):
        raise ValueError("build manifest changed while it was being read")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    "build manifest contains a duplicate JSON object key"
                )
            result[key] = value
        return result

    try:
        raw = json.loads(content, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("build manifest is invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("build manifest must contain an object")
    digest = sha256_file(source)
    after_hash = source.stat(follow_symlinks=False)
    if (
        not _same_file_metadata(before, after_hash)
        or digest != hashlib.sha256(content).hexdigest()
    ):
        raise ValueError("build manifest changed while it was being hashed")
    return raw, digest


def create_asset_receipt(
    data_root: str | Path,
    *,
    freeze_sha256: str,
    matrix_plan_sha256: str,
    specs: Sequence[StagedAssetSpec],
) -> AssetReceipt:
    root = _regular_directory(data_root, "asset data root")
    freeze_hash = validate_sha256(freeze_sha256, "asset freeze SHA-256")
    plan_hash = validate_sha256(
        matrix_plan_sha256,
        "asset matrix plan SHA-256",
    )
    expected = _normalize_specs(specs)
    builds: dict[str, tuple[Mapping[str, Any], str, dict[str, Mapping[str, Any]]]] = {}
    for spec in expected.values():
        current = builds.get(spec.build_rel)
        metadata = _thaw(spec.build_metadata)
        if current is not None:
            if _thaw(current[0]) != metadata:
                raise ValueError("inconsistent shared build metadata")
            continue
        manifest_path = root / spec.build_rel / "manifest.json"
        manifest, manifest_sha256 = _load_build_manifest(manifest_path)
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != 1
            or manifest.get("study_freeze_sha256") != freeze_hash
        ):
            raise ValueError("build manifest has wrong freeze metadata")
        if manifest.get("protected_build") != metadata:
            raise ValueError("build manifest has wrong build metadata")
        builds[spec.build_rel] = (
            spec.build_metadata,
            manifest_sha256,
            _artifact_inventory(manifest),
        )

    records: list[AssetRecord] = []
    for key in sorted(expected):
        spec = expected[key]
        source = root / spec.path
        before = source.stat(follow_symlinks=False)
        digest = sha256_file(source)
        after = source.stat(follow_symlinks=False)
        if not _same_file_metadata(before, after):
            raise ValueError("artifact changed while it was being hashed")
        relative_in_build = Path(spec.path).relative_to(spec.build_rel).as_posix()
        _metadata, manifest_sha256, inventory = builds[spec.build_rel]
        indexed = inventory.get(relative_in_build)
        if indexed is None:
            raise ValueError("build manifest is missing a staged asset")
        if indexed["sha256"] != digest or indexed["bytes"] != after.st_size:
            raise ValueError("build manifest asset digest or byte count mismatch")
        records.append(
            AssetRecord(
                kind=spec.kind,
                path=spec.path,
                commitment_sha256=spec.commitment_sha256,
                sha256=digest,
                bytes=after.st_size,
                build_manifest_sha256=manifest_sha256,
            )
        )
    return AssetReceipt.create(
        freeze_sha256=freeze_hash,
        matrix_plan_sha256=plan_hash,
        assets=records,
    )


def validate_asset_receipt(
    receipt: AssetReceipt | Mapping[str, Any],
    *,
    freeze_sha256: str,
    matrix_plan_sha256: str,
    specs: Sequence[StagedAssetSpec],
) -> AssetReceipt:
    validated = (
        receipt
        if isinstance(receipt, AssetReceipt)
        else AssetReceipt.from_dict(receipt)
    )
    validated = AssetReceipt.from_dict(validated.to_dict())
    if validated.freeze_sha256 != freeze_sha256:
        raise ValueError("asset receipt freeze hash mismatch")
    if validated.matrix_plan_sha256 != matrix_plan_sha256:
        raise ValueError("asset receipt matrix plan hash mismatch")
    expected = _normalize_specs(specs)
    records = {(item.kind, item.path): item for item in validated.assets}
    if set(records) != set(expected):
        raise ValueError("asset receipt inventory has missing or extra paths")
    for key, spec in expected.items():
        if records[key].commitment_sha256 != spec.commitment_sha256:
            raise ValueError("asset receipt commitment mismatch")
    return validated


def write_asset_receipt(
    path: str | Path,
    receipt: AssetReceipt,
) -> Path:
    destination = Path(path)
    if destination.is_symlink() or os.path.lexists(destination):
        raise FileExistsError(f"asset receipt destination already exists: {destination}")
    validated = AssetReceipt.from_dict(receipt.to_dict())
    return atomic_write_json(destination, validated.to_dict())


def publish_asset_receipt(
    path: str | Path,
    data_root: str | Path,
    *,
    freeze_sha256: str,
    matrix_plan_sha256: str,
    specs: Sequence[StagedAssetSpec],
) -> Path:
    receipt = create_asset_receipt(
        data_root,
        freeze_sha256=freeze_sha256,
        matrix_plan_sha256=matrix_plan_sha256,
        specs=specs,
    )
    return write_asset_receipt(path, receipt)


def load_asset_receipt(path: str | Path) -> AssetReceipt:
    return AssetReceipt.from_dict(load_canonical_json(path))
