"""Hash-bound, fail-closed confirmatory artifact reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from types import MappingProxyType
from typing import Any

from evals.confirmatory.contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.status import (
    FinalInferenceConclusion,
    InterimEvidenceLabel,
    ScientificStatus,
    StatusAxes,
    classify_status,
)


ARTIFACT_REPORT_SCHEMA = "memorysplit.confirmatory.artifact-report.v2"
INFERENCE_EVIDENCE_SCHEMA = "memorysplit.confirmatory.inference-evidence.v2"
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
        "scientific_status",
        "interim_evidence_label",
        "final_inference_conclusion",
        "expected_cells",
        "observed_cells",
        "artifacts",
        "report_sha256",
    }
)
_BINDING_FIELDS = frozenset({"sha256", "size_bytes"})
_INFERENCE_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "terminal_evidence_complete",
        "measured_validity_failure",
        "observed_valid_seed_pairs",
        "required_seed_pairs",
        "same_sign_preterminal_pairs",
        "supports_effect",
        "supports_practical_null",
    }
)
_HEX = frozenset("0123456789abcdef")
_DIR_RELATIVE_PUBLICATION_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.link in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.link in os.supports_follow_symlinks
)


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


def _schema_version(value: object, name: str) -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ValueError(f"{name} schema_version is invalid")
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


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean")
    return value


@dataclass(frozen=True)
class InferenceEvidence:
    record_type: str
    schema_version: int
    terminal_evidence_complete: bool
    measured_validity_failure: bool
    observed_valid_seed_pairs: int
    required_seed_pairs: int
    same_sign_preterminal_pairs: int
    supports_effect: bool
    supports_practical_null: bool

    def __post_init__(self) -> None:
        if self.record_type != INFERENCE_EVIDENCE_SCHEMA:
            raise ValueError("inference evidence schema identity is invalid")
        _schema_version(self.schema_version, "inference evidence")
        terminal_complete = _boolean(
            self.terminal_evidence_complete,
            "terminal_evidence_complete",
        )
        measured_failure = _boolean(
            self.measured_validity_failure,
            "measured_validity_failure",
        )
        observed = _count(
            self.observed_valid_seed_pairs,
            "observed_valid_seed_pairs",
        )
        required = _count(
            self.required_seed_pairs,
            "required_seed_pairs",
            positive=True,
        )
        same_sign = _count(
            self.same_sign_preterminal_pairs,
            "same_sign_preterminal_pairs",
        )
        supports_effect = _boolean(self.supports_effect, "supports_effect")
        supports_null = _boolean(
            self.supports_practical_null,
            "supports_practical_null",
        )
        if required != 5:
            raise ValueError("required_seed_pairs must equal the frozen value five")
        if observed > required:
            raise ValueError("observed_valid_seed_pairs exceeds required_seed_pairs")
        if same_sign > observed:
            raise ValueError(
                "same_sign_preterminal_pairs exceeds observed valid pairs"
            )
        if terminal_complete and observed != required:
            raise ValueError(
                "terminal evidence requires every frozen seed pair"
            )
        if supports_effect and supports_null:
            raise ValueError("terminal conclusions are mutually exclusive")
        if (
            supports_effect or supports_null
        ) and (measured_failure or not terminal_complete):
            raise ValueError(
                "supports_effect/supports_practical_null require complete valid evidence"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InferenceEvidence":
        value = _strict_fields(
            raw,
            _INFERENCE_FIELDS,
            "inference evidence",
        )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "terminal_evidence_complete": self.terminal_evidence_complete,
            "measured_validity_failure": self.measured_validity_failure,
            "observed_valid_seed_pairs": self.observed_valid_seed_pairs,
            "required_seed_pairs": self.required_seed_pairs,
            "same_sign_preterminal_pairs": self.same_sign_preterminal_pairs,
            "supports_effect": self.supports_effect,
            "supports_practical_null": self.supports_practical_null,
        }


def _inference_evidence(content: bytes) -> InferenceEvidence:
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("inference evidence must be canonical UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("inference evidence must be an object")
    try:
        if canonical_json_bytes(raw) != content:
            raise ValueError("inference evidence must use canonical JSON encoding")
        return InferenceEvidence.from_dict(raw)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("inference evidence is invalid") from exc


def _derive_status_axes(
    evidence: InferenceEvidence,
    *,
    expected_cells: int,
    observed_cells: int,
) -> StatusAxes:
    complete = (
        evidence.terminal_evidence_complete
        and observed_cells == expected_cells
    )
    axes = classify_status(
        complete=complete,
        valid=not evidence.measured_validity_failure,
        observed_seeds=evidence.observed_valid_seed_pairs,
        required_seeds=evidence.required_seed_pairs,
        sign_consistent=evidence.same_sign_preterminal_pairs >= 3,
        supports_effect=evidence.supports_effect,
        supports_practical_null=evidence.supports_practical_null,
    )
    if (
        axes.scientific_status is not ScientificStatus.COMPLETE
        and (
            evidence.supports_effect
            or evidence.supports_practical_null
        )
    ):
        raise ValueError(
            "terminal conclusion is inconsistent with incomplete or invalid evidence"
        )
    return axes


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
    axes: StatusAxes,
    expected_cells: int,
    observed_cells: int,
    artifacts: Mapping[str, Mapping[str, str | int]],
) -> dict[str, Any]:
    return {
        "record_type": ARTIFACT_REPORT_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        **axes.to_dict(),
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
    scientific_status: ScientificStatus
    interim_evidence_label: InterimEvidenceLabel
    final_inference_conclusion: FinalInferenceConclusion
    expected_cells: int
    observed_cells: int
    artifacts: Mapping[str, Mapping[str, str | int]]
    report_sha256: str

    def __post_init__(self) -> None:
        if self.record_type != ARTIFACT_REPORT_SCHEMA:
            raise ValueError("artifact report schema identity is invalid")
        _schema_version(self.schema_version, "artifact report")
        axes = StatusAxes(
            self.scientific_status,
            self.interim_evidence_label,
            self.final_inference_conclusion,
        )
        expected = _count(self.expected_cells, "expected_cells", positive=True)
        observed = _count(self.observed_cells, "observed_cells")
        if observed > expected:
            raise ValueError("observed_cells exceeds expected_cells")
        if (
            observed < expected
            and axes.scientific_status is ScientificStatus.COMPLETE
        ):
            raise ValueError("missing cells cannot have complete scientific status")
        bindings = _artifact_bindings(self.artifacts)
        payload = _report_payload(
            axes=axes,
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
        object.__setattr__(
            self,
            "scientific_status",
            axes.scientific_status,
        )
        object.__setattr__(
            self,
            "interim_evidence_label",
            axes.interim_evidence_label,
        )
        object.__setattr__(
            self,
            "final_inference_conclusion",
            axes.final_inference_conclusion,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactReport":
        value = _strict_fields(raw, _REPORT_FIELDS, "artifact report")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            **_report_payload(
                axes=StatusAxes(
                    self.scientific_status,
                    self.interim_evidence_label,
                    self.final_inference_conclusion,
                ),
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
    artifacts: Mapping[str, bytes],
    expected_cells: int,
    observed_cells: int,
) -> ArtifactReport:
    content = _artifact_bytes(artifacts)
    expected = _count(expected_cells, "expected_cells", positive=True)
    observed = _count(observed_cells, "observed_cells")
    if observed > expected:
        raise ValueError("observed_cells exceeds expected_cells")
    evidence = _inference_evidence(content["inference.json"])
    axes = _derive_status_axes(
        evidence,
        expected_cells=expected,
        observed_cells=observed,
    )
    bindings = {
        name: {
            "sha256": hashlib.sha256(content[name]).hexdigest(),
            "size_bytes": len(content[name]),
        }
        for name in REQUIRED_ARTIFACTS
    }
    payload = _report_payload(
        axes=axes,
        expected_cells=expected,
        observed_cells=observed,
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
    evidence = _inference_evidence(content["inference.json"])
    axes = _derive_status_axes(
        evidence,
        expected_cells=typed.expected_cells,
        observed_cells=typed.observed_cells,
    )
    reported_axes = StatusAxes(
        typed.scientific_status,
        typed.interim_evidence_label,
        typed.final_inference_conclusion,
    )
    if reported_axes != axes:
        raise ValueError("artifact report status axes disagree with bound evidence")
    return typed


def _secure_directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(
        not hasattr(os, name) or not getattr(os, name)
        for name in required
    ):
        raise RuntimeError(
            "secure directory-relative publication is unsupported"
        )
    if not _DIR_RELATIVE_PUBLICATION_SUPPORTED:
        raise RuntimeError(
            "secure directory-relative publication is unsupported"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_parent_directory(output: Path) -> int:
    if any(part == ".." for part in output.parts):
        raise ValueError("artifact report path cannot contain traversal")
    if output.name in {"", ".", ".."}:
        raise ValueError("artifact report path must name a file")

    flags = _secure_directory_flags()
    parent = output.parent
    if parent.is_absolute():
        descriptor = os.open(parent.anchor, flags)
        components = parent.parts[1:]
    else:
        descriptor = os.open(".", flags)
        components = parent.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            try:
                child = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ValueError(
                    "artifact report parent cannot contain a symlink "
                    "or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(
                "artifact report parent must be a regular directory"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_temporary(
    directory_fd: int,
    content: bytes,
) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        name = f".confirmatory-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            raise
        return name
    raise FileExistsError("could not allocate a secure temporary report")


def publish_artifact_report(
    path: str | Path,
    report: ArtifactReport | Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> Path:
    """Atomically publish a canonical report without replacing existing data."""

    typed = validate_artifact_report(report, artifacts)
    output = Path(path)
    directory_fd = _open_parent_directory(output)
    temporary_name: str | None = None
    try:
        temporary_name = _create_temporary(
            directory_fd,
            canonical_json_bytes(typed),
        )
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"artifact report already exists: {output}"
            ) from None
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    return output
