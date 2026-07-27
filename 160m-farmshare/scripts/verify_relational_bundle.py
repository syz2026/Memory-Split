#!/usr/bin/env python
"""Independently verify and safely extract a Task-11 relational bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


_MAX_MEMBERS = 10_000
_MAX_MEMBER_BYTES = 64 * 1024**2
_MAX_TOTAL_BYTES = 512 * 1024**2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_HASH_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_REQUIRED_MEMBERS = {
    "BUNDLE-MANIFEST.json",
    "SOURCE-REVISION.json",
    "contracts/freeze.json",
    "contracts/run-manifest.json",
    "contracts/preregistration.md",
    "contracts/schemas/freeze-v1.schema.json",
    "contracts/schemas/relational-asset-receipt-v1.schema.json",
    "contracts/schemas/relational-result-v1.schema.json",
    "contracts/schemas/run-config-v1.schema.json",
    "contracts/schemas/run-manifest-v1.schema.json",
    "fixtures/relational-smoke.json",
    "offline_tests/verify_contracts.py",
    "requirements/base.in",
    "requirements/macos-arm64-py312.lock",
    "requirements/linux-x86_64-cuda-py312.lock",
}
_BUNDLE_FIELDS = {
    "record_type",
    "schema_version",
    "source_revision",
    "freeze_sha256",
    "run_manifest_sha256",
    "run_count",
    "eval_control_count",
    "eval_mode_count",
    "eval_cell_count",
    "external_assets_only",
    "external_assets",
    "source_members",
    "members",
    "manifest_sha256",
}
_SOURCE_FIELDS = {
    "record_type",
    "schema_version",
    "git_revision",
    "git_tree",
    "clean_tree",
    "package_source_sha256",
}
_ASSET_RECEIPT_FIELDS = {
    "record_type",
    "schema_version",
    "freeze_sha256",
    "matrix_plan_sha256",
    "assets",
    "receipt_sha256",
}
_ASSET_RECORD_FIELDS = {
    "kind",
    "path",
    "commitment_sha256",
    "sha256",
    "bytes",
    "build_manifest_sha256",
}
_BUNDLE_ASSET_FIELDS = {
    "kind",
    "path",
    "commitment_sha256",
    "sha256",
    "bytes",
    "build_manifest_sha256",
}
_FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


class BundleVerificationError(ValueError):
    """Raised for any archive, hash, extraction, or contract violation."""


def _canonicalize(value: Any, path: str = "$") -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise BundleVerificationError(f"{path} has a non-string key")
        return {
            key: _canonicalize(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [
            _canonicalize(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        import math

        if math.isfinite(value):
            return value
    raise BundleVerificationError(f"{path} is not canonical JSON")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonicalize(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_canonical_json(content: bytes, *, name: str) -> Any:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"{name} is invalid JSON") from exc
    if content != _canonical_json_bytes(value):
        raise BundleVerificationError(f"{name} is not canonical JSON")
    return value


def _portable_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BundleVerificationError("archive member name is not portable")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("./")
        or "//" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BundleVerificationError(
            "archive member name is absolute, unsafe, or contains traversal"
        )
    return value


def _regular_archive(path: str | Path) -> Path:
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise BundleVerificationError(
            "bundle must be a regular non-symlink file"
        )
    if candidate.resolve(strict=True) != absolute:
        raise BundleVerificationError(
            "bundle path is not canonical or traverses a symlink"
        )
    metadata = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise BundleVerificationError("bundle must be a regular file")
    return absolute


def _check_gzip_header(path: Path) -> None:
    header = path.read_bytes()[:10]
    if (
        len(header) != 10
        or header[:2] != b"\x1f\x8b"
        or header[2] != 8
        or header[4:8] != b"\x00\x00\x00\x00"
        or header[3] & 0x08
    ):
        raise BundleVerificationError(
            "bundle gzip metadata is not normalized"
        )


def _read_archive(path: Path) -> dict[str, bytes]:
    _check_gzip_header(path)
    try:
        with tarfile.open(path, mode="r:gz", errorlevel=2) as archive:
            members = archive.getmembers()
            if not members or len(members) > _MAX_MEMBERS:
                raise BundleVerificationError(
                    "archive member count is empty or excessive"
                )
            names: set[str] = set()
            payload: dict[str, bytes] = {}
            total = 0
            prior = ""
            for member in members:
                name = _portable_name(member.name)
                if name in names:
                    raise BundleVerificationError(
                        f"duplicate archive member: {name}"
                    )
                names.add(name)
                if prior and name <= prior:
                    raise BundleVerificationError(
                        "archive members are not in canonical sorted order"
                    )
                prior = name
                if not member.isreg():
                    raise BundleVerificationError(
                        f"archive member is not a regular file: {name}"
                    )
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.mode != 0o644
                    or member.pax_headers
                ):
                    raise BundleVerificationError(
                        f"archive metadata is not normalized: {name}"
                    )
                if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
                    raise BundleVerificationError(
                        f"archive member size is unsafe: {name}"
                    )
                total += member.size
                if total > _MAX_TOTAL_BYTES:
                    raise BundleVerificationError(
                        "archive uncompressed size is excessive"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise BundleVerificationError(
                        f"archive member cannot be read: {name}"
                    )
                content = stream.read()
                if len(content) != member.size:
                    raise BundleVerificationError(
                        f"archive member size changed while reading: {name}"
                    )
                payload[name] = content
    except BundleVerificationError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise BundleVerificationError("bundle archive is invalid") from exc
    return payload


def _validate_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BundleVerificationError(f"{name} is not a lowercase SHA-256")
    return value


def _source_digest(
    payload: Mapping[str, bytes],
    source_members: list[str],
) -> str:
    return _canonical_sha256(
        [
            {
                "path": name,
                "bytes": len(payload[name]),
                "sha256": hashlib.sha256(payload[name]).hexdigest(),
            }
            for name in source_members
        ]
    )


def _validate_payload(payload: Mapping[str, bytes]) -> dict[str, Any]:
    missing = _REQUIRED_MEMBERS - set(payload)
    if missing:
        raise BundleVerificationError(
            "bundle is missing required member(s): " + ", ".join(sorted(missing))
        )
    raw_manifest = payload["BUNDLE-MANIFEST.json"]
    bundle = _load_canonical_json(
        raw_manifest,
        name="bundle manifest",
    )
    if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_FIELDS:
        raise BundleVerificationError("bundle manifest fields are not exact")
    if (
        bundle["record_type"] != "relational_portable_bundle"
        or bundle["schema_version"] != 1
        or bundle["run_count"] != 35
        or bundle["eval_control_count"] != 11
        or bundle["eval_mode_count"] != 2
        or bundle["eval_cell_count"] != 22
        or bundle["external_assets_only"] is not True
    ):
        raise BundleVerificationError("bundle manifest protocol is invalid")
    _validate_hash(bundle["freeze_sha256"], "freeze hash")
    _validate_hash(bundle["run_manifest_sha256"], "run manifest hash")
    supplied_manifest_hash = _validate_hash(
        bundle["manifest_sha256"],
        "bundle manifest hash",
    )
    material = dict(bundle)
    material.pop("manifest_sha256")
    if supplied_manifest_hash != _canonical_sha256(material):
        raise BundleVerificationError("bundle manifest hash indicates tampering")

    index = bundle["members"]
    if not isinstance(index, list):
        raise BundleVerificationError("bundle member index must be a list")
    indexed: dict[str, tuple[int, str]] = {}
    prior = ""
    for item in index:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "bytes", "sha256"}
        ):
            raise BundleVerificationError("bundle member index fields are invalid")
        name = _portable_name(item["path"])
        if name in indexed or (prior and name <= prior):
            raise BundleVerificationError(
                "bundle member index is duplicated or unsorted"
            )
        prior = name
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleVerificationError("bundle member size is invalid")
        indexed[name] = (size, _validate_hash(item["sha256"], "member hash"))
    expected = set(payload) - {"BUNDLE-MANIFEST.json"}
    if set(indexed) != expected:
        missing_from_index = sorted(expected - set(indexed))
        missing_from_payload = sorted(set(indexed) - expected)
        raise BundleVerificationError(
            "bundle member index does not match archive inventory "
            f"(unindexed={missing_from_index}, absent={missing_from_payload})"
        )
    for name, (size, digest) in indexed.items():
        content = payload[name]
        if len(content) != size:
            raise BundleVerificationError(f"member size mismatch: {name}")
        if hashlib.sha256(content).hexdigest() != digest:
            raise BundleVerificationError(f"member hash mismatch: {name}")

    source = _load_canonical_json(
        payload["SOURCE-REVISION.json"],
        name="source revision",
    )
    if (
        not isinstance(source, Mapping)
        or set(source) != _SOURCE_FIELDS
        or source["record_type"] != "relational_source_revision"
        or source["schema_version"] != 1
        or not isinstance(source["clean_tree"], bool)
        or _GIT_HASH_RE.fullmatch(str(source["git_revision"])) is None
        or _GIT_HASH_RE.fullmatch(str(source["git_tree"])) is None
    ):
        raise BundleVerificationError("source revision contract is invalid")
    _validate_hash(source["package_source_sha256"], "source payload hash")
    if source != bundle["source_revision"]:
        raise BundleVerificationError(
            "source revision disagrees with bundle manifest"
        )
    source_members = bundle["source_members"]
    if (
        not isinstance(source_members, list)
        or source_members != sorted(set(source_members))
        or any(
            not isinstance(name, str) or name not in payload
            for name in source_members
        )
    ):
        raise BundleVerificationError("source member inventory is invalid")
    if _source_digest(payload, source_members) != source[
        "package_source_sha256"
    ]:
        raise BundleVerificationError("source payload hash mismatch")

    freeze_contract = _load_canonical_json(
        payload["contracts/freeze.json"],
        name="contracts/freeze.json",
    )
    manifest_contract = _load_canonical_json(
        payload["contracts/run-manifest.json"],
        name="contracts/run-manifest.json",
    )
    if (
        not isinstance(freeze_contract, Mapping)
        or freeze_contract.get("freeze_sha256") != bundle["freeze_sha256"]
        or not isinstance(manifest_contract, Mapping)
        or manifest_contract.get("freeze_sha256") != bundle["freeze_sha256"]
        or manifest_contract.get("manifest_sha256")
        != bundle["run_manifest_sha256"]
    ):
        raise BundleVerificationError(
            "frozen contracts disagree with bundle hashes"
        )
    if manifest_contract.get("launchable") is True:
        provenance = freeze_contract.get("source_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("git_revision") != source["git_revision"]
            or provenance.get("clean_tree") is not True
            or source["clean_tree"] is not True
        ):
            raise BundleVerificationError(
                "launchable bundle source revision differs from frozen "
                "provenance"
            )

    configs = [
        name
        for name in payload
        if name.startswith("contracts/configs/") and name.endswith(".json")
    ]
    if len(configs) != 35:
        raise BundleVerificationError(
            "bundle must contain exactly 35 run configs"
        )
    for name in configs:
        _load_canonical_json(payload[name], name=name)
    if any(Path(name).suffix.lower() in _FORBIDDEN_SUFFIXES for name in payload):
        raise BundleVerificationError(
            "bundle contains a full corpus or checkpoint artifact"
        )

    assets = bundle["external_assets"]
    if not isinstance(assets, list):
        raise BundleVerificationError("external asset inventory is invalid")
    for item in assets:
        if (
            not isinstance(item, Mapping)
            or set(item) != _BUNDLE_ASSET_FIELDS
            or item["kind"] not in {"corpus", "weights"}
        ):
            raise BundleVerificationError("external asset entry is invalid")
        _portable_name(item["path"])
        _validate_hash(
            item["commitment_sha256"],
            "external asset commitment",
        )
        _validate_hash(item["sha256"], "external asset hash")
        _validate_hash(
            item["build_manifest_sha256"],
            "external asset build manifest hash",
        )
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 1
        ):
            raise BundleVerificationError("external asset byte count is invalid")
        if item["path"] in payload:
            raise BundleVerificationError(
                "external asset was embedded in the portable bundle"
            )
    runs = manifest_contract.get("runs")
    if not isinstance(runs, list):
        raise BundleVerificationError("run manifest has no run inventory")
    receipt = manifest_contract.get("asset_receipt")
    receipt_member = "contracts/asset-receipt.json"
    expected_assets: dict[tuple[str, str, str], dict[str, Any]] = {}
    if manifest_contract.get("launchable") is True:
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != _ASSET_RECEIPT_FIELDS
            or receipt.get("record_type") != "relational_asset_receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("freeze_sha256") != bundle["freeze_sha256"]
            or receipt.get("matrix_plan_sha256")
            != manifest_contract.get("matrix_plan_sha256")
        ):
            raise BundleVerificationError(
                "launchable manifest asset receipt is invalid"
            )
        receipt_hash = _validate_hash(
            receipt.get("receipt_sha256"),
            "asset receipt hash",
        )
        receipt_material = dict(receipt)
        receipt_material.pop("receipt_sha256")
        if receipt_hash != _canonical_sha256(receipt_material):
            raise BundleVerificationError("asset receipt hash mismatch")
        if receipt_member not in payload:
            raise BundleVerificationError(
                "launchable bundle is missing its asset receipt"
            )
        standalone = _load_canonical_json(
            payload[receipt_member],
            name=receipt_member,
        )
        if standalone != receipt:
            raise BundleVerificationError(
                "standalone asset receipt disagrees with run manifest"
            )
        receipt_assets = receipt.get("assets")
        if not isinstance(receipt_assets, list) or not receipt_assets:
            raise BundleVerificationError("asset receipt inventory is empty")
        receipt_records: dict[tuple[str, str], Mapping[str, Any]] = {}
        receipt_paths: set[str] = set()
        previous: tuple[str, str] | None = None
        for record in receipt_assets:
            if (
                not isinstance(record, Mapping)
                or set(record) != _ASSET_RECORD_FIELDS
                or record.get("kind") not in {"stream", "weights"}
            ):
                raise BundleVerificationError(
                    "asset receipt record is invalid"
                )
            path = _portable_name(record.get("path"))
            key = (record["kind"], path)
            if key in receipt_records or path in receipt_paths:
                raise BundleVerificationError(
                    "asset receipt contains a duplicate path"
                )
            if previous is not None and key <= previous:
                raise BundleVerificationError(
                    "asset receipt inventory is not canonically ordered"
                )
            previous = key
            receipt_paths.add(path)
            _validate_hash(
                record.get("commitment_sha256"),
                "asset recipe commitment",
            )
            _validate_hash(record.get("sha256"), "asset file hash")
            _validate_hash(
                record.get("build_manifest_sha256"),
                "asset build manifest hash",
            )
            size = record.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise BundleVerificationError(
                    "asset receipt byte count is invalid"
                )
            receipt_records[key] = record

        referenced: set[tuple[str, str]] = set()
        for run in runs:
            if not isinstance(run, Mapping):
                raise BundleVerificationError("run manifest entry is invalid")
            for internal_kind, bundle_kind, path_key, hash_key, commitment_key in (
                (
                    "stream",
                    "corpus",
                    "data_rel",
                    "stream_sha256",
                    "stream_commitment_sha256",
                ),
                (
                    "weights",
                    "weights",
                    "weights_rel",
                    "weights_sha256",
                    "weights_commitment_sha256",
                ),
            ):
                path = run.get(path_key)
                if not isinstance(path, str):
                    raise BundleVerificationError(
                        "run manifest external asset path is invalid"
                    )
                validated_path = _portable_name(path)
                digest = _validate_hash(
                    run.get(hash_key),
                    "run manifest external asset hash",
                )
                commitment = _validate_hash(
                    run.get(commitment_key),
                    "run manifest asset commitment",
                )
                record = receipt_records.get((internal_kind, validated_path))
                if (
                    record is None
                    or record["sha256"] != digest
                    or record["commitment_sha256"] != commitment
                ):
                    raise BundleVerificationError(
                        "run asset does not match the staged-asset receipt"
                    )
                referenced.add((internal_kind, validated_path))
                key = (bundle_kind, validated_path, digest)
                expected_assets[key] = {
                    "kind": bundle_kind,
                    "path": validated_path,
                    "commitment_sha256": commitment,
                    "sha256": digest,
                    "bytes": record["bytes"],
                    "build_manifest_sha256": record[
                        "build_manifest_sha256"
                    ],
                }
        if referenced != set(receipt_records):
            raise BundleVerificationError(
                "asset receipt inventory has missing or extra paths"
            )
    else:
        if receipt is not None or receipt_member in payload:
            raise BundleVerificationError(
                "nonlaunchable fixture cannot contain an asset receipt"
            )
        for run in runs:
            if (
                not isinstance(run, Mapping)
                or run.get("stream_sha256") is not None
                or run.get("weights_sha256") is not None
            ):
                raise BundleVerificationError(
                    "fixture run mislabeled a recipe commitment as a file hash"
                )
            _validate_hash(
                run.get("stream_commitment_sha256"),
                "fixture stream commitment",
            )
            _validate_hash(
                run.get("weights_commitment_sha256"),
                "fixture weights commitment",
            )
    derived_assets = [expected_assets[key] for key in sorted(expected_assets)]
    if assets != derived_assets:
        raise BundleVerificationError(
            "external asset inventory disagrees with frozen manifest"
        )
    return {
        "source_revision": source["git_revision"],
        "source_clean": source["clean_tree"],
        "run_count": bundle["run_count"],
        "eval_cell_count": bundle["eval_cell_count"],
        "member_count": len(payload),
        "manifest_sha256": supplied_manifest_hash,
        "external_assets": derived_assets,
    }


def _validate_extract_target(path: Path) -> tuple[Path, bool]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if ".." in path.parts:
        raise BundleVerificationError("extraction path contains traversal")
    parent = absolute.parent
    if parent.is_symlink() or not parent.is_dir():
        raise BundleVerificationError(
            "extraction parent must be a regular directory"
        )
    if parent.resolve(strict=True) != parent:
        raise BundleVerificationError(
            "extraction parent is not canonical or traverses a symlink"
        )
    if os.path.lexists(absolute):
        if absolute.is_symlink() or not absolute.is_dir():
            raise BundleVerificationError(
                "extraction target must be a regular directory"
            )
        if absolute.resolve(strict=True) != absolute:
            raise BundleVerificationError("extraction target is not canonical")
        if any(absolute.iterdir()):
            raise BundleVerificationError("extraction target must be empty")
        return absolute, False
    absolute.mkdir(mode=0o755)
    return absolute, True


def _extract_payload(payload: Mapping[str, bytes], destination: Path) -> None:
    for name in sorted(payload):
        path = destination.joinpath(*PurePosixPath(name).parts)
        relative_parts = PurePosixPath(name).parts
        current = destination
        for part in relative_parts[:-1]:
            current = current / part
            if os.path.lexists(current):
                if current.is_symlink() or not current.is_dir():
                    raise BundleVerificationError(
                        f"unsafe extraction directory: {name}"
                    )
            else:
                current.mkdir(mode=0o755)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o644)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload[name])
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise BundleVerificationError(
                f"cannot safely extract member: {name}"
            ) from exc


def _read_extracted(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise BundleVerificationError(
            "extracted bundle must be a regular directory"
        )
    absolute = root if root.is_absolute() else Path.cwd() / root
    if root.resolve(strict=True) != absolute:
        raise BundleVerificationError(
            "extracted bundle path is not canonical or traverses a symlink"
        )
    payload: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BundleVerificationError("extracted bundle contains a symlink")
        if path.is_dir():
            continue
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleVerificationError(
                "extracted bundle contains a non-regular file"
            )
        name = _portable_name(path.relative_to(root).as_posix())
        payload[name] = path.read_bytes()
    return payload


def _run_offline_tests(root: Path) -> dict[str, Any]:
    script = root / "offline_tests" / "verify_contracts.py"
    environment = dict(os.environ)
    for key in list(environment):
        if key.lower().endswith("_proxy"):
            environment.pop(key, None)
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "NO_PROXY": "*",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(script), str(root)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BundleVerificationError(
            "offline contract tests could not run"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BundleVerificationError(
            f"offline contract tests failed: {detail}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BundleVerificationError(
            "offline contract tests returned invalid JSON"
        ) from exc
    if not isinstance(report, dict) or report.get("verified") is not True:
        raise BundleVerificationError(
            "offline contract tests did not report verification"
        )
    return report


def verify_extracted_bundle(
    root: str | Path,
    *,
    run_tests: bool = True,
) -> dict[str, Any]:
    extracted = Path(root)
    report = _validate_payload(_read_extracted(extracted))
    offline = _run_offline_tests(extracted) if run_tests else None
    return {
        "record_type": "relational_bundle_verification",
        "schema_version": 1,
        "verified": True,
        **report,
        "offline_tests_passed": offline is not None,
        "offline_report": offline,
    }


def verify_bundle(
    bundle: str | Path,
    *,
    extract_to: str | Path | None = None,
    run_tests: bool = True,
) -> dict[str, Any]:
    archive = _regular_archive(bundle)
    payload = _read_archive(archive)
    report = _validate_payload(payload)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if extract_to is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="relational-bundle-verify-"
        )
        extraction = Path(temporary.name).resolve(strict=True)
    else:
        extraction, _ = _validate_extract_target(Path(extract_to))
    try:
        if extract_to is None:
            # TemporaryDirectory has already created an empty canonical root.
            if any(extraction.iterdir()):
                raise BundleVerificationError(
                    "temporary extraction target is not empty"
                )
        _extract_payload(payload, extraction)
        extracted = _validate_payload(_read_extracted(extraction))
        if extracted != report:
            raise BundleVerificationError(
                "extracted verification differs from archive verification"
            )
        offline = _run_offline_tests(extraction) if run_tests else None
    finally:
        if temporary is not None:
            temporary.cleanup()
    return {
        "record_type": "relational_bundle_verification",
        "schema_version": 1,
        "verified": True,
        **report,
        "bundle_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "offline_tests_passed": offline is not None,
        "offline_report": offline,
        "extracted_to": (
            str(extraction) if extract_to is not None else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle")
    source.add_argument("--extracted")
    parser.add_argument("--extract")
    parser.add_argument("--no-tests", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.bundle:
            report = verify_bundle(
                args.bundle,
                extract_to=args.extract,
                run_tests=not args.no_tests,
            )
        else:
            if args.extract:
                raise BundleVerificationError(
                    "--extract is valid only with --bundle"
                )
            report = verify_extracted_bundle(
                args.extracted,
                run_tests=not args.no_tests,
            )
    except BundleVerificationError as exc:
        print(f"bundle verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
