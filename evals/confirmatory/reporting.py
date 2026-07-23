"""Hash-bound, fail-closed confirmatory artifact reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

from evals.confirmatory.contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.status import STATUS_PRECEDENCE


ARTIFACT_REPORT_SCHEMA = "memorysplit.confirmatory.artifact-report.v2"
REQUIRED_ARTIFACTS = (
    "checkpoints.jsonl",
    "inference.json",
    "items.jsonl",
    "metrics.json",
    "outcomes.jsonl",
    "sealed-gold.jsonl",
    "stores.jsonl",
)
_REPORT_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "status",
        "expected_cells",
        "observed_cells",
        "artifacts",
        "report_sha256",
    }
)
_BINDING_FIELDS = frozenset({"sha256", "size_bytes"})
_HEX = frozenset("0123456789abcdef")


def _hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _count(value: object, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _strict_fields(
    value: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{name} fields are not exact")
    return value


def _artifact_bindings(
    value: object,
) -> dict[str, dict[str, str | int]]:
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_ARTIFACTS):
        raise ValueError("artifact set is not exact")
    bindings = {}
    for name in REQUIRED_ARTIFACTS:
        raw = _strict_fields(
            value[name],
            _BINDING_FIELDS,
            f"artifact binding {name}",
        )
        bindings[name] = {
            "sha256": _hash(raw["sha256"], f"artifact {name} hash"),
            "size_bytes": _count(
                raw["size_bytes"],
                f"artifact {name} size",
                positive=True,
            ),
        }
    return bindings


def _report_payload(
    *,
    status: str,
    expected_cells: int,
    observed_cells: int,
    artifacts: Mapping[str, Mapping[str, str | int]],
) -> dict[str, Any]:
    return {
        "record_type": ARTIFACT_REPORT_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "status": status,
        "expected_cells": expected_cells,
        "observed_cells": observed_cells,
        "artifacts": {
            name: dict(artifacts[name]) for name in REQUIRED_ARTIFACTS
        },
    }


@dataclass(frozen=True)
class ArtifactReport:
    record_type: str
    schema_version: int
    status: str
    expected_cells: int
    observed_cells: int
    artifacts: Mapping[str, Mapping[str, str | int]]
    report_sha256: str

    def __post_init__(self) -> None:
        if (
            self.record_type != ARTIFACT_REPORT_SCHEMA
            or self.schema_version != CONTRACT_VERSION
        ):
            raise ValueError("artifact report schema identity is invalid")
        if self.status not in STATUS_PRECEDENCE:
            raise ValueError("artifact report status is invalid")
        expected = _count(self.expected_cells, "expected_cells", positive=True)
        observed = _count(self.observed_cells, "observed_cells")
        if observed > expected:
            raise ValueError("observed_cells exceeds expected_cells")
        if observed < expected and self.status != "incomplete":
            raise ValueError("incomplete cell matrix requires incomplete status")
        if observed == expected and self.status == "incomplete":
            raise ValueError("incomplete status requires a missing cell")
        bindings = _artifact_bindings(self.artifacts)
        payload = _report_payload(
            status=self.status,
            expected_cells=expected,
            observed_cells=observed,
            artifacts=bindings,
        )
        claimed = _hash(self.report_sha256, "report_sha256")
        if claimed != canonical_sha256(payload):
            raise ValueError("artifact report_sha256 does not match report")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(binding))
                    for name, binding in bindings.items()
                }
            ),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactReport":
        value = _strict_fields(raw, _REPORT_FIELDS, "artifact report")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            **_report_payload(
                status=self.status,
                expected_cells=self.expected_cells,
                observed_cells=self.observed_cells,
                artifacts=self.artifacts,
            ),
            "report_sha256": self.report_sha256,
        }


def _artifact_bytes(
    artifacts: Mapping[str, bytes],
) -> dict[str, bytes]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        REQUIRED_ARTIFACTS
    ):
        raise ValueError("artifact set is not exact")
    result = {}
    for name in REQUIRED_ARTIFACTS:
        content = artifacts[name]
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"artifact {name} must contain bytes")
        result[name] = content
    return result


def build_artifact_report(
    *,
    status: str,
    artifacts: Mapping[str, bytes],
    expected_cells: int,
    observed_cells: int,
) -> ArtifactReport:
    content = _artifact_bytes(artifacts)
    bindings = {
        name: {
            "sha256": hashlib.sha256(content[name]).hexdigest(),
            "size_bytes": len(content[name]),
        }
        for name in REQUIRED_ARTIFACTS
    }
    payload = _report_payload(
        status=status,
        expected_cells=expected_cells,
        observed_cells=observed_cells,
        artifacts=bindings,
    )
    return ArtifactReport.from_dict(
        {
            **payload,
            "report_sha256": canonical_sha256(payload),
        }
    )


def validate_artifact_report(
    report: ArtifactReport | Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> ArtifactReport:
    typed = (
        ArtifactReport.from_dict(report.to_dict())
        if isinstance(report, ArtifactReport)
        else ArtifactReport.from_dict(report)
    )
    content = _artifact_bytes(artifacts)
    for name in REQUIRED_ARTIFACTS:
        binding = typed.artifacts[name]
        if len(content[name]) != binding["size_bytes"]:
            raise ValueError(f"artifact {name} size mismatch")
        if hashlib.sha256(content[name]).hexdigest() != binding["sha256"]:
            raise ValueError(f"artifact {name} hash mismatch")
    return typed


def publish_artifact_report(
    path: str | Path,
    report: ArtifactReport | Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> Path:
    """Atomically publish a canonical report without replacing existing data."""

    typed = validate_artifact_report(report, artifacts)
    output = Path(path)
    if any(part == ".." for part in output.parts):
        raise ValueError("artifact report path cannot contain traversal")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("artifact report parent must be a regular directory")
    if parent.resolve(strict=True) != parent.absolute():
        raise ValueError("artifact report path cannot traverse symlinks")
    if os.path.lexists(output):
        raise FileExistsError(f"artifact report already exists: {output}")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(typed))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"artifact report already exists: {output}"
            ) from None
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output
