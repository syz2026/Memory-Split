#!/usr/bin/env python
"""Build the deterministic, portable Task-11 relational run bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.relational_controls import ControlID, EvalMode
from experiment.artifacts import (
    atomic_write_stream,
    canonical_json_bytes,
    canonical_sha256,
    require_regular_file,
)
from experiment.provenance import tracked_source_tree_sha256
from scripts.freeze_relational_study import load_freeze_manifest
from scripts.make_relational_manifest import load_run_manifest


_SOURCE_DIRS = (
    "cluster",
    "configs",
    "corpusgen",
    "evals",
    "experiment",
    "fixtures",
    "organizer",
    "requirements",
    "schemas",
    "scripts",
    "tests",
    "train",
)
_SOURCE_FILES = ("README.md", "requirements.txt")
_SOURCE_SUFFIXES = {
    ".env",
    ".in",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".sbatch",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_REQUIRED_CONTRACT_SOURCE = {
    "preregistration": "docs/superpowers/specs/2026-07-20-preregistration.md",
    "smoke_fixture": "fixtures/relational-smoke.json",
    "offline_tests": "scripts/offline_relational_contracts.py",
}
_REQUIRED_SCHEMAS = (
    "freeze-v1.schema.json",
    "relational-asset-receipt-v1.schema.json",
    "relational-result-v1.schema.json",
    "run-config-v1.schema.json",
    "run-manifest-v1.schema.json",
)
_REQUIRED_LOCKS = (
    "requirements/base.in",
    "requirements/macos-arm64-py312.lock",
    "requirements/linux-x86_64-cuda-py312.lock",
)
_FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_SHA1_OR_SHA256_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def _git(
    source_root: Path,
    *arguments: str,
    check: bool = True,
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=check,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("source root must be a readable Git worktree") from exc
    return completed.stdout


def _source_revision(
    source_root: Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    revision = _git(source_root, "rev-parse", "HEAD").strip()
    tree = _git(source_root, "rev-parse", "HEAD^{tree}").strip()
    if _SHA1_OR_SHA256_RE.fullmatch(revision) is None:
        raise ValueError("source Git revision is not canonical")
    if _SHA1_OR_SHA256_RE.fullmatch(tree) is None:
        raise ValueError("source Git tree is not canonical")
    status = _git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    clean = not status.strip()
    if require_clean and not clean:
        raise ValueError("production packaging requires a clean source tree")
    return {
        "record_type": "relational_source_revision",
        "schema_version": 1,
        "git_revision": revision,
        "git_tree": tree,
        "clean_tree": clean,
    }


def _portable_member_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("bundle member name is not portable")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("./")
        or "//" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("bundle member name is absolute or contains traversal")
    return value


def _safe_source_file(source_root: Path, relative: str) -> Path:
    name = _portable_member_name(relative)
    path = source_root / Path(*PurePosixPath(name).parts)
    try:
        regular = require_regular_file(path, name=f"bundle source {relative}")
    except ValueError as exc:
        raise ValueError(f"bundle source is unsafe: {relative}") from exc
    if regular.resolve(strict=True) != path:
        raise ValueError(f"bundle source traverses a symlink: {relative}")
    if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"binary model/corpus artifact is forbidden: {relative}")
    return path


def _eligible_source_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    if relative in _SOURCE_FILES:
        return True
    return (
        bool(path.parts)
        and path.parts[0] in _SOURCE_DIRS
        and path.suffix.lower() in _SOURCE_SUFFIXES
        and "__pycache__" not in path.parts
        and not any(part.startswith(".") for part in path.parts[1:])
    )


def _source_paths(
    source_root: Path,
    *,
    committed_only: bool,
) -> list[str]:
    if committed_only:
        tracked = _git(
            source_root,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            "HEAD",
        ).split("\0")
        return sorted(
            {
                _portable_member_name(relative)
                for relative in tracked
                if relative and _eligible_source_path(relative)
            }
        )

    paths: set[str] = set()
    for name in _SOURCE_FILES:
        path = source_root / name
        if path.is_file() and not path.is_symlink():
            paths.add(name)
    for directory in _SOURCE_DIRS:
        root = source_root / directory
        if not root.is_dir() or root.is_symlink():
            continue
        for candidate in root.rglob("*"):
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.suffix.lower() not in _SOURCE_SUFFIXES
                or "__pycache__" in candidate.parts
                or any(part.startswith(".") for part in candidate.relative_to(root).parts)
            ):
                continue
            paths.add(candidate.relative_to(source_root).as_posix())
    return sorted(paths)


def _add(payload: dict[str, bytes], name: str, content: bytes) -> None:
    member = _portable_member_name(name)
    if member == "BUNDLE-MANIFEST.json":
        raise ValueError("bundle manifest is reserved")
    if member in payload:
        raise ValueError(f"duplicate bundle member: {member}")
    if not isinstance(content, bytes):
        raise TypeError("bundle payload must be bytes")
    payload[member] = content


def _source_payload_sha256(
    payload: dict[str, bytes],
    source_members: list[str],
) -> str:
    return canonical_sha256(
        [
            {
                "path": path,
                "bytes": len(payload[path]),
                "sha256": hashlib.sha256(payload[path]).hexdigest(),
            }
            for path in sorted(source_members)
        ]
    )


def _external_assets(manifest: Any) -> list[dict[str, Any]]:
    if manifest.asset_receipt is None:
        return []
    assets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for asset in manifest.asset_receipt.assets:
        kind = "corpus" if asset.kind == "stream" else "weights"
        key = (kind, asset.path, asset.sha256)
        assets[key] = {
            "kind": kind,
            "path": asset.path,
            "commitment_sha256": asset.commitment_sha256,
            "sha256": asset.sha256,
            "bytes": asset.bytes,
            "build_manifest_sha256": asset.build_manifest_sha256,
        }
    return [assets[key] for key in sorted(assets)]


def _build_payload(
    source_root: Path,
    *,
    freeze_path: Path,
    run_manifest_path: Path,
    revision: dict[str, Any],
) -> tuple[dict[str, bytes], list[str], Any]:
    freeze = load_freeze_manifest(freeze_path)
    manifest = load_run_manifest(run_manifest_path)
    if manifest.freeze.to_dict() != freeze.to_dict():
        raise ValueError("run manifest does not bind the supplied freeze")
    if manifest.launchable:
        provenance = freeze.source_provenance
        if provenance.git_revision != revision["git_revision"]:
            raise ValueError(
                "launchable bundle source revision differs from frozen provenance"
            )
        if not revision["clean_tree"] or not provenance.clean_tree:
            raise ValueError(
                "launchable bundle source revision must be clean"
            )
        if (
            tracked_source_tree_sha256(source_root)
            != provenance.source_tree_sha256
        ):
            raise ValueError(
                "launchable bundle source revision tree differs from "
                "frozen provenance"
            )

    payload: dict[str, bytes] = {}
    source_members: list[str] = []
    for relative in _source_paths(
        source_root,
        committed_only=revision["clean_tree"],
    ):
        path = _safe_source_file(source_root, relative)
        _add(payload, relative, path.read_bytes())
        source_members.append(relative)

    _add(payload, "contracts/freeze.json", canonical_json_bytes(freeze.to_dict()))
    _add(
        payload,
        "contracts/run-manifest.json",
        canonical_json_bytes(manifest.to_dict()),
    )
    if manifest.asset_receipt is not None:
        _add(
            payload,
            "contracts/asset-receipt.json",
            canonical_json_bytes(manifest.asset_receipt.to_dict()),
        )
    for run in manifest.runs:
        _add(
            payload,
            f"contracts/configs/{run.run_id}.json",
            canonical_json_bytes(run.to_dict()),
        )
    for schema in _REQUIRED_SCHEMAS:
        _add(
            payload,
            f"contracts/schemas/{schema}",
            _safe_source_file(source_root, f"schemas/{schema}").read_bytes(),
        )
    _add(
        payload,
        "contracts/preregistration.md",
        _safe_source_file(
            source_root,
            _REQUIRED_CONTRACT_SOURCE["preregistration"],
        ).read_bytes(),
    )
    _add(
        payload,
        "offline_tests/verify_contracts.py",
        _safe_source_file(
            source_root,
            _REQUIRED_CONTRACT_SOURCE["offline_tests"],
        ).read_bytes(),
    )
    for relative in _REQUIRED_LOCKS:
        if relative not in payload:
            _add(
                payload,
                relative,
                _safe_source_file(source_root, relative).read_bytes(),
            )
    if "fixtures/relational-smoke.json" not in payload:
        _add(
            payload,
            "fixtures/relational-smoke.json",
            _safe_source_file(
                source_root,
                _REQUIRED_CONTRACT_SOURCE["smoke_fixture"],
            ).read_bytes(),
        )

    revision = {
        **revision,
        "package_source_sha256": _source_payload_sha256(
            payload,
            source_members,
        ),
    }
    _add(payload, "SOURCE-REVISION.json", canonical_json_bytes(revision))
    return payload, source_members, manifest


def _bundle_manifest(
    payload: dict[str, bytes],
    *,
    source_members: list[str],
    manifest: Any,
) -> bytes:
    members = [
        {
            "path": name,
            "bytes": len(payload[name]),
            "sha256": hashlib.sha256(payload[name]).hexdigest(),
        }
        for name in sorted(payload)
    ]
    material = {
        "record_type": "relational_portable_bundle",
        "schema_version": 1,
        "source_revision": json.loads(payload["SOURCE-REVISION.json"]),
        "freeze_sha256": manifest.freeze_sha256,
        "run_manifest_sha256": manifest.manifest_sha256,
        "run_count": len(manifest.runs),
        "eval_control_count": len(ControlID),
        "eval_mode_count": len(EvalMode),
        "eval_cell_count": len(ControlID) * len(EvalMode),
        "external_assets_only": True,
        "external_assets": _external_assets(manifest),
        "source_members": sorted(source_members),
        "members": members,
    }
    return canonical_json_bytes(
        {**material, "manifest_sha256": canonical_sha256(material)}
    )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(_portable_member_name(name))
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _write_archive(stream: Any, payload: dict[str, bytes]) -> None:
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=stream,
        mtime=0,
        compresslevel=9,
    ) as zipped:
        with tarfile.open(
            fileobj=zipped,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            for name in sorted(payload):
                content = payload[name]
                archive.addfile(_tar_info(name, len(content)), io.BytesIO(content))


def _validate_source_root(source_root: str | Path) -> Path:
    root = Path(source_root)
    absolute = root if root.is_absolute() else Path.cwd() / root
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source root must be a regular non-symlink directory")
    if root.resolve(strict=True) != absolute:
        raise ValueError("source root is not canonical or traverses a symlink")
    return absolute


def package_run(
    output: str | Path,
    *,
    source_root: str | Path,
    freeze_path: str | Path,
    run_manifest_path: str | Path,
    require_clean: bool = True,
) -> Path:
    """Build one deterministic archive without including external assets."""

    root = _validate_source_root(source_root)
    freeze = require_regular_file(freeze_path, name="freeze manifest")
    run_manifest = require_regular_file(
        run_manifest_path,
        name="run manifest",
    )
    revision = _source_revision(root, require_clean=require_clean)
    payload, source_members, manifest = _build_payload(
        root,
        freeze_path=freeze,
        run_manifest_path=run_manifest,
        revision=revision,
    )
    payload["BUNDLE-MANIFEST.json"] = _bundle_manifest(
        payload,
        source_members=source_members,
        manifest=manifest,
    )
    destination = Path(output)
    if destination.name != "relational-run.tar.gz" and not destination.name.endswith(
        ".tar.gz"
    ):
        raise ValueError("bundle output must use a .tar.gz filename")
    return atomic_write_stream(
        destination,
        lambda stream: _write_archive(stream, payload),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--run-manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = package_run(
        args.out,
        source_root=args.source_root,
        freeze_path=args.freeze,
        run_manifest_path=args.run_manifest,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
