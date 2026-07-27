#!/usr/bin/env python
"""Independent stdlib-first checks run from an extracted relational bundle."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path, PurePosixPath


def _forbid_network() -> None:
    def blocked(*args, **kwargs):
        raise RuntimeError("offline contract tests forbid network access")

    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, *args, **kwargs):
            return blocked(*args, **kwargs)

        def connect_ex(self, *args, **kwargs):
            return blocked(*args, **kwargs)

    socket.create_connection = blocked
    socket.socket = OfflineSocket


def _canonical_json(path: Path):
    content = path.read_bytes()
    value = json.loads(content)
    expected = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if content != expected:
        raise ValueError(f"noncanonical JSON contract: {path}")
    return value


def _portable(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def verify(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    sys.path.insert(0, str(root))
    _forbid_network()

    from evals.relational_controls import ControlID, EvalMode
    from scripts.freeze_relational_study import validate_freeze_manifest
    from scripts.make_relational_manifest import RunManifest
    from scripts.platform_preflight import (
        validate_base_requirements,
        validate_lock_file,
    )

    freeze = validate_freeze_manifest(
        _canonical_json(root / "contracts" / "freeze.json")
    )
    manifest = RunManifest.from_dict(
        _canonical_json(root / "contracts" / "run-manifest.json")
    )
    if manifest.freeze.to_dict() != freeze.to_dict():
        raise ValueError("manifest freeze differs from standalone freeze")
    if len(manifest.runs) != 35:
        raise ValueError("offline manifest must contain exactly 35 runs")

    configs = root / "contracts" / "configs"
    names = {entry.name for entry in configs.iterdir() if entry.is_file()}
    expected_names = {f"{run.run_id}.json" for run in manifest.runs}
    if names != expected_names:
        raise ValueError("offline run-config inventory is not exact")
    for run in manifest.runs:
        raw = _canonical_json(configs / f"{run.run_id}.json")
        if raw != run.to_dict():
            raise ValueError(f"offline run config mismatch: {run.run_id}")
        if not all(_portable(path) for path in run.relative_paths()):
            raise ValueError(f"offline run config is not portable: {run.run_id}")

    schema_root = root / "contracts" / "schemas"
    expected_schemas = {
        "freeze-v1.schema.json",
        "relational-asset-receipt-v1.schema.json",
        "relational-result-v1.schema.json",
        "run-config-v1.schema.json",
        "run-manifest-v1.schema.json",
    }
    schema_names = {entry.name for entry in schema_root.iterdir()}
    if schema_names != expected_schemas:
        raise ValueError("offline JSON-schema inventory is not exact")
    for name in sorted(schema_names):
        schema = json.loads((schema_root / name).read_bytes())
        if (
            not isinstance(schema, dict)
            or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError(f"invalid draft-2020-12 schema: {name}")

    requirements = root / "requirements"
    base = validate_base_requirements(requirements / "base.in")
    mac = validate_lock_file(
        requirements / "macos-arm64-py312.lock",
        expected_platform="macos-arm64",
    )
    linux = validate_lock_file(
        requirements / "linux-x86_64-cuda-py312.lock",
        expected_platform="linux-x86_64-cuda",
    )
    if not set(base) <= set(mac) or not set(base) <= set(linux):
        raise ValueError("platform locks omit a direct requirement")

    forbidden = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
    included_forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden
    ]
    if included_forbidden:
        raise ValueError("bundle contains an external model or corpus asset")

    return {
        "verified": True,
        "run_count": len(manifest.runs),
        "control_count": len(ControlID),
        "eval_cell_count": len(ControlID) * len(EvalMode),
        "schemas": len(expected_schemas),
        "locks": 2,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: verify_contracts.py EXTRACTED_ROOT", file=sys.stderr)
        return 2
    try:
        report = verify(Path(arguments[0]))
    except Exception as exc:
        print(f"offline contracts failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
