#!/usr/bin/env python3
"""Immutable terminal checkpoint and paired evaluation publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.interruption_checkpoint import (
    S3ObjectStore,
    UploadedObject,
    VerifiedObjectStore,
)
_ARMS = ("dense", "split90")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OPERATION_RE = _SHA256_RE
_CHECKPOINT_FIELDS = {
    "schema_version",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "source_commit",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "profile_sha256",
    "sealed_fixture_sha256",
    "checkpoints",
}
_CHECKPOINT_ROW_FIELDS = {
    "run_id",
    "arm",
    "seed",
    "path",
    "sha256",
    "config_sha256",
    "dataset_sha256",
    "source_commit",
    "step",
    "world_size",
}
_DURABLE_CHECKPOINT_ROW_FIELDS = _CHECKPOINT_ROW_FIELDS | {
    "checkpoint_uri",
    "configuration_uri",
    "run_binding_sha256",
    "run_binding_uri",
    "checkpoint_record_sha256",
    "checkpoint_record_uri",
}
_CHECKPOINT_METADATA_FIELDS = {
    "schema_version",
    "receipt_type",
    "run_id",
    "condition",
    "seed",
    "step",
    "max_steps",
    "world_size",
    "config_fingerprint",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "terminal",
}
_RUN_BINDING_FIELDS = {
    "record_type",
    "schema_version",
    "run_id",
    "checkpoint_path",
    "checkpoint_sha256",
    "configuration_path",
    "configuration_sha256",
    "route_dose_sha256",
    "corpus_sha256",
    "code_sha256",
    "seed",
    "condition_id",
}
_EVALUATION_FIELDS = {
    "schema_version",
    "receipt_type",
    "status",
    "provider",
    "seed",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "source_commit",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "profile_sha256",
    "sealed_fixture_sha256",
    "sealed_evaluation_sha256",
    "study_lock_sha256",
    "fleet_plan_sha256",
    "fleet_wave",
    "launch_readiness_sha256",
    "operation_id",
    "checkpoint_receipt",
    "arms",
}
_EVALUATION_CHECKPOINT_FIELDS = {"sha256", "uri"}
_EVALUATION_ARM_FIELDS = {
    "run_id",
    "arm",
    "checkpoint_sha256",
    "artifacts",
}
_EVALUATION_ARTIFACT_FIELDS = {"path", "bytes", "sha256", "uri"}
EVALUATION_ARTIFACTS = frozenset(
    {
        "artifact-report.json",
        "checkpoints.jsonl",
        "inference.json",
        "items.jsonl",
        "metrics.json",
        "outcomes.jsonl",
        "sealed-gold.jsonl",
        "study-lock.json",
        "stores.jsonl",
        "validity.json",
    }
)


@dataclass(frozen=True)
class TerminalPublication:
    receipt_path: Path
    receipt_sha256: str
    receipt_uri: str
    completion_uri: str
    checkpoint_record_paths: Mapping[str, Path]
    checkpoint_record_sha256: Mapping[str, str]
    checkpoint_record_uris: Mapping[str, str]


@dataclass(frozen=True)
class EvaluationPublication:
    receipt_path: Path
    receipt_sha256: str
    receipt_uri: str
    completion_uri: str
    result_uri: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats field: {key}")
        result[key] = value
    return result


def _decode_canonical(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _portable_name(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(("/", "~"))
        or "\\" in value
        or "/" in value
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} is unsafe")
    return value


def _s3_uri(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an S3 URI")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{label} must be an S3 URI") from error
    parts = parsed.path.removeprefix("/").split("/")
    if (
        parsed.scheme != "s3"
        or parsed.hostname is None
        or parsed.netloc != parsed.hostname
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in value
        or any(character in value for character in "\n\r\x00")
    ):
        raise ValueError(f"{label} must be a canonical S3 URI")
    return value


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _directory_fd(path: Path, *, create: bool) -> int:
    """Pin every path component without ever traversing a symlink."""

    absolute = path.absolute()
    current = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current)
                    os.fsync(current)
                    next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                except OSError as error:
                    raise ValueError(
                        f"directory path is unavailable or unsafe: {path}"
                    ) from error
            except OSError as error:
                raise ValueError(
                    f"directory path is unavailable or unsafe: {path}"
                ) from error
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _open_regular(path: Path, *, label: str) -> int:
    parent = _directory_fd(path.parent, create=False)
    try:
        return os.open(path.name, _REGULAR_FLAGS, dir_fd=parent)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"{label} is unavailable or unsafe") from error
    finally:
        os.close(parent)


def _regular_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular(path: Path, *, label: str, maximum: int = 1 << 34) -> bytes:
    try:
        descriptor = _open_regular(path, label=label)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is unavailable or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise ValueError(f"{label} must be one bounded regular file")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - offset),
                offset,
            )
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _regular_identity(before) != _regular_identity(after) or offset != after.st_size:
        raise ValueError(f"{label} changed while read")
    return b"".join(chunks)


def _hash_open_regular(descriptor: int, *, label: str) -> tuple[str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
        raise ValueError(f"{label} must be one non-empty regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(1 << 20, before.st_size - offset),
            offset,
        )
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if _regular_identity(before) != _regular_identity(after) or offset != after.st_size:
        raise ValueError(f"{label} changed while hashed")
    return digest.hexdigest(), after.st_size


def _hash_regular(path: Path, *, label: str) -> tuple[str, int]:
    try:
        descriptor = _open_regular(path, label=label)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is unavailable or unsafe") from error
    try:
        return _hash_open_regular(descriptor, label=label)
    finally:
        os.close(descriptor)


def _ensure_real_directory(path: Path) -> None:
    descriptor = _directory_fd(path, create=True)
    try:
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def _write_immutable(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    parent = _directory_fd(path.parent, create=True)
    descriptor = -1
    created = False
    try:
        try:
            descriptor = os.open(path.name, _REGULAR_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            descriptor = os.open(
                path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
            created = True
        except OSError as error:
            raise ValueError(f"immutable local artifact is unsafe: {path}") from error
        if not created:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != len(payload)
                or stat.S_IMODE(before.st_mode) != mode
            ):
                raise ValueError(f"immutable local artifact conflicts: {path}")
            existing = bytearray()
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1 << 20, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    break
                existing.extend(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            if (
                _regular_identity(before) != _regular_identity(after)
                or offset != after.st_size
                or bytes(existing) != payload
            ):
                raise ValueError(f"immutable local artifact conflicts: {path}")
            return
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                os.unlink(path.name, dir_fd=parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _snapshot_regular(
    source: Path,
    destination: Path,
    *,
    label: str,
    mode: int = 0o400,
) -> tuple[str, int]:
    source_parent = _directory_fd(source.parent, create=False)
    destination_parent = _directory_fd(destination.parent, create=True)
    source_fd = -1
    destination_fd = -1
    created = False
    try:
        try:
            source_fd = os.open(source.name, _REGULAR_FLAGS, dir_fd=source_parent)
        except OSError as error:
            raise ValueError(f"{label} source is unavailable or unsafe") from error
        source_before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_nlink != 1
            or source_before.st_size <= 0
        ):
            raise ValueError(f"{label} source is not one regular file")
        try:
            destination_fd = os.open(
                destination.name,
                _REGULAR_FLAGS,
                dir_fd=destination_parent,
            )
        except FileNotFoundError:
            destination_fd = os.open(
                destination.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=destination_parent,
            )
            created = True
        except OSError as error:
            raise ValueError(f"{label} immutable snapshot is unsafe") from error
        if not created:
            destination_digest, destination_bytes = _hash_open_regular(
                destination_fd,
                label=f"{label} immutable snapshot",
            )
            source_digest, source_bytes = _hash_open_regular(
                source_fd,
                label=f"{label} source",
            )
            if (
                source_digest != destination_digest
                or source_bytes != destination_bytes
                or stat.S_IMODE(os.fstat(destination_fd).st_mode) != mode
            ):
                raise ValueError(f"{label} immutable snapshot conflicts")
            return source_digest, source_bytes
        digest = hashlib.sha256()
        offset = 0
        while offset < source_before.st_size:
            chunk = os.pread(
                source_fd,
                min(1 << 20, source_before.st_size - offset),
                offset,
            )
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                view = view[os.write(destination_fd, view) :]
            offset += len(chunk)
        source_after = os.fstat(source_fd)
        if (
            _regular_identity(source_before) != _regular_identity(source_after)
            or offset != source_after.st_size
        ):
            raise ValueError(f"{label} changed while snapshotted")
        os.fchmod(destination_fd, mode)
        os.fsync(destination_fd)
        os.fsync(destination_parent)
        return digest.hexdigest(), offset
    except BaseException:
        if destination_fd >= 0:
            os.close(destination_fd)
            destination_fd = -1
        if created:
            try:
                os.unlink(destination.name, dir_fd=destination_parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
        os.close(destination_parent)
        os.close(source_parent)


def verify_checkpoint_receipt_bytes(
    payload: bytes,
    *,
    expected: Mapping[str, object] | None = None,
    require_durable: bool = False,
) -> dict[str, object]:
    value = _decode_canonical(payload, label="paired checkpoint receipt")
    if set(value) != _CHECKPOINT_FIELDS or value["schema_version"] != 3:
        raise ValueError("paired checkpoint receipt fields do not match")
    for field in _CHECKPOINT_FIELDS - {"schema_version", "provider", "source_commit", "checkpoints"}:
        _sha256(value[field], label=f"checkpoint receipt {field}")
    if not isinstance(value["provider"], str) or not value["provider"].endswith("-v3"):
        raise ValueError("checkpoint receipt provider is invalid")
    if not isinstance(value["source_commit"], str) or _COMMIT_RE.fullmatch(value["source_commit"]) is None:
        raise ValueError("checkpoint receipt source commit is invalid")
    rows = value["checkpoints"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("checkpoint receipt must contain one pair")
    row_field_sets = {
        frozenset(row)
        for row in rows
        if isinstance(row, dict)
    }
    allowed_row_fields = {
        frozenset(_CHECKPOINT_ROW_FIELDS),
        frozenset(_DURABLE_CHECKPOINT_ROW_FIELDS),
    }
    if (
        len(row_field_sets) != 1
        or not all(isinstance(row, dict) for row in rows)
        or not row_field_sets <= allowed_row_fields
    ):
        raise ValueError("checkpoint receipt row fields do not match")
    durable = row_field_sets == {frozenset(_DURABLE_CHECKPOINT_ROW_FIELDS)}
    if require_durable and not durable:
        raise ValueError(
            "terminal checkpoint receipt omits durable artifact identities"
        )
    arms: set[str] = set()
    seeds: set[int] = set()
    steps: set[int] = set()
    artifact_root: str | None = None
    for row in rows:
        assert isinstance(row, dict)
        arm = row["arm"]
        if (
            arm not in _ARMS
            or arm in arms
            or _portable_name(
                row["run_id"],
                label="checkpoint run ID",
            )
            != row["run_id"]
            or type(row["seed"]) is not int
            or _portable_name(row["path"], label="checkpoint path") != f"{arm}.pt"
            or type(row["step"]) is not int
            or row["step"] <= 0
            or row["world_size"] != 4
            or row["dataset_sha256"] != value["dataset_sha256"]
            or row["source_commit"] != value["source_commit"]
        ):
            raise ValueError("checkpoint receipt row identity is invalid")
        for field in ("sha256", "config_sha256", "dataset_sha256"):
            _sha256(row[field], label=f"checkpoint row {field}")
        if durable:
            for field in ("run_binding_sha256", "checkpoint_record_sha256"):
                _sha256(row[field], label=f"checkpoint row {field}")
            marker = f"/checkpoints/seed-{row['seed']}/{arm}"
            checkpoint_uri = _s3_uri(
                row["checkpoint_uri"],
                label=f"{arm} checkpoint URI",
            )
            if marker not in checkpoint_uri:
                raise ValueError("checkpoint receipt artifact URI is cross-seed")
            row_root, separator, suffix = checkpoint_uri.partition(marker)
            if (
                not separator
                or suffix != f"/sha256/{row['sha256']}.pt"
                or artifact_root not in {None, row_root}
            ):
                raise ValueError(
                    "checkpoint receipt artifact URI is not canonical"
                )
            artifact_root = row_root
            expected_uris = {
                "configuration_uri": (
                    f"{row_root}{marker}/configuration/sha256/"
                    f"{row['config_sha256']}.yaml"
                ),
                "run_binding_uri": (
                    f"{row_root}{marker}/run-binding/sha256/"
                    f"{row['run_binding_sha256']}.json"
                ),
                "checkpoint_record_uri": (
                    f"{row_root}{marker}/records/"
                    f"{row['checkpoint_record_sha256']}.json"
                ),
            }
            if any(
                _s3_uri(row[field], label=f"{arm} {field}") != expected_uri
                for field, expected_uri in expected_uris.items()
            ):
                raise ValueError(
                    "checkpoint receipt artifact URI is not canonical"
                )
        arms.add(str(arm))
        seeds.add(int(row["seed"]))
        steps.add(int(row["step"]))
    if arms != set(_ARMS) or len(seeds) != 1 or len(steps) != 1:
        raise ValueError("checkpoint receipt pair is incomplete or split-step")
    if expected is not None:
        for field, expected_value in expected.items():
            if field not in value or value[field] != expected_value:
                raise ValueError(f"checkpoint receipt {field} does not match")
    return value


def verify_evaluation_receipt_bytes(
    payload: bytes,
    *,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    value = _decode_canonical(payload, label="paired evaluation receipt")
    if (
        set(value) != _EVALUATION_FIELDS
        or value["schema_version"] != 3
        or value["receipt_type"] != "memorysplit-aws-paired-evaluation-v3"
        or value["status"] != "closed"
    ):
        raise ValueError("paired evaluation receipt fields do not match")
    for field in (
        "release_sha256",
        "run_manifest_sha256",
        "dataset_sha256",
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "hardware_amendment_sha256",
        "provider_selection_sha256",
        "profile_sha256",
        "sealed_fixture_sha256",
        "sealed_evaluation_sha256",
        "study_lock_sha256",
        "fleet_plan_sha256",
        "launch_readiness_sha256",
        "operation_id",
    ):
        _sha256(value[field], label=f"evaluation receipt {field}")
    if (
        not isinstance(value["provider"], str)
        or not value["provider"].endswith("-v3")
        or not isinstance(value["source_commit"], str)
        or _COMMIT_RE.fullmatch(value["source_commit"]) is None
        or type(value["seed"]) is not int
        or value["seed"] < 0
        or type(value["fleet_wave"]) is not int
        or value["fleet_wave"] < 0
    ):
        raise ValueError("evaluation receipt identity is invalid")
    checkpoint = value["checkpoint_receipt"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != _EVALUATION_CHECKPOINT_FIELDS:
        raise ValueError("evaluation checkpoint binding fields do not match")
    _sha256(checkpoint["sha256"], label="evaluation checkpoint receipt")
    if not isinstance(checkpoint["uri"], str) or not checkpoint["uri"].startswith("s3://"):
        raise ValueError("evaluation checkpoint receipt URI is invalid")
    arms = value["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("evaluation receipt must contain one pair")
    seen: set[str] = set()
    run_ids: set[str] = set()
    for arm_row in arms:
        if not isinstance(arm_row, dict) or set(arm_row) != _EVALUATION_ARM_FIELDS:
            raise ValueError("evaluation arm fields do not match")
        arm = arm_row["arm"]
        if (
            arm not in _ARMS
            or arm in seen
            or _portable_name(
                arm_row["run_id"],
                label="evaluation run ID",
            )
            != arm_row["run_id"]
            or arm_row["run_id"] in run_ids
        ):
            raise ValueError("evaluation arm identity is invalid")
        _sha256(arm_row["checkpoint_sha256"], label="evaluation checkpoint")
        artifacts = arm_row["artifacts"]
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != len(EVALUATION_ARTIFACTS)
            or {
                row.get("path")
                for row in artifacts
                if isinstance(row, dict)
            }
            != EVALUATION_ARTIFACTS
        ):
            raise ValueError("evaluation artifact inventory is incomplete")
        for row in artifacts:
            if (
                not isinstance(row, dict)
                or set(row) != _EVALUATION_ARTIFACT_FIELDS
                or type(row["bytes"]) is not int
                or row["bytes"] <= 0
                or not isinstance(row["uri"], str)
                or not row["uri"].startswith("s3://")
            ):
                raise ValueError("evaluation artifact identity is invalid")
            _portable_name(row["path"], label="evaluation artifact path")
            _sha256(row["sha256"], label="evaluation artifact")
        seen.add(str(arm))
        run_ids.add(str(arm_row["run_id"]))
    if seen != set(_ARMS):
        raise ValueError("evaluation receipt pair is incomplete")
    if expected is not None:
        for field, expected_value in expected.items():
            if field not in value or value[field] != expected_value:
                raise ValueError(f"evaluation receipt {field} does not match")
    return value


def _verify_run_directory(
    run: Path,
    *,
    checkpoint_row: Mapping[str, object],
) -> dict[str, object]:
    binding_payload = _read_regular(
        run / "run.json",
        label=f"{checkpoint_row['arm']} evaluator run binding",
        maximum=64 * 1024,
    )
    binding = _decode_canonical(
        binding_payload,
        label=f"{checkpoint_row['arm']} evaluator run binding",
    )
    if (
        set(binding) != _RUN_BINDING_FIELDS
        or binding["record_type"] != "memorysplit.confirmatory.run-binding.v2"
        or binding["schema_version"] != 2
        or binding["run_id"] != checkpoint_row["run_id"]
        or binding["seed"] != checkpoint_row["seed"]
        or binding["condition_id"] != checkpoint_row["arm"]
        or binding["checkpoint_path"] != "ckpt.pt"
        or binding["checkpoint_sha256"] != checkpoint_row["sha256"]
        or binding["configuration_path"] != "configuration.yaml"
        or binding["configuration_sha256"] != checkpoint_row["config_sha256"]
    ):
        raise ValueError(
            f"{checkpoint_row['arm']} evaluator run binding disagrees with receipt"
        )
    for field in (
        "checkpoint_sha256",
        "configuration_sha256",
        "route_dose_sha256",
        "corpus_sha256",
        "code_sha256",
    ):
        _sha256(binding[field], label=f"run binding {field}")
    checkpoint_digest, _ = _hash_regular(
        run / "ckpt.pt",
        label=f"{checkpoint_row['arm']} evaluator checkpoint",
    )
    configuration_digest, _ = _hash_regular(
        run / "configuration.yaml",
        label=f"{checkpoint_row['arm']} evaluator configuration",
    )
    if (
        checkpoint_digest != checkpoint_row["sha256"]
        or configuration_digest != checkpoint_row["config_sha256"]
    ):
        raise ValueError(
            f"{checkpoint_row['arm']} evaluator files disagree with receipt"
        )
    return binding


def _verify_checkpoint_record(
    path: Path,
    *,
    checkpoint_row: Mapping[str, object],
    run_binding: Mapping[str, object],
) -> dict[str, object]:
    from evals.confirmatory.contracts import CheckpointRecord

    payload = _read_regular(
        path,
        label=f"{checkpoint_row['arm']} checkpoint record",
        maximum=64 * 1024,
    )
    digest = hashlib.sha256(payload).hexdigest()
    value = _decode_canonical(
        payload,
        label=f"{checkpoint_row['arm']} checkpoint record",
    )
    record = CheckpointRecord.from_dict(value)
    expected_arm = "dense" if checkpoint_row["arm"] == "dense" else "split"
    if (
        digest != checkpoint_row["checkpoint_record_sha256"]
        or record.checkpoint_sha256 != checkpoint_row["sha256"]
        or record.arm.value != expected_arm
        or record.condition_id.value != checkpoint_row["arm"]
        or record.seed != checkpoint_row["seed"]
        or record.configuration_sha256 != checkpoint_row["config_sha256"]
        or record.configuration_sha256
        != run_binding["configuration_sha256"]
        or record.route_dose_sha256 != run_binding["route_dose_sha256"]
        or record.corpus_sha256 != run_binding["corpus_sha256"]
        or record.code_sha256 != run_binding["code_sha256"]
    ):
        raise ValueError(
            f"{checkpoint_row['arm']} checkpoint record disagrees with the "
            "terminal run binding"
        )
    return value


def verify_materialized_terminal(
    *,
    receipt_path: Path,
    run_root: Path,
    record_root: Path,
    expected: Mapping[str, object],
    expected_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Verify a terminal pair rematerialized entirely from immutable S3."""

    payload = _read_regular(
        receipt_path,
        label="materialized terminal checkpoint receipt",
        maximum=1 << 20,
    )
    receipt = verify_checkpoint_receipt_bytes(
        payload,
        expected=expected,
        require_durable=True,
    )
    digest = hashlib.sha256(payload).hexdigest()
    if (
        expected_receipt_sha256 is not None
        and digest
        != _sha256(
            expected_receipt_sha256,
            label="expected terminal checkpoint receipt",
        )
    ):
        raise ValueError("materialized checkpoint receipt SHA-256 differs")
    for row in receipt["checkpoints"]:
        binding = _verify_run_directory(
            run_root / str(row["arm"]) / "run",
            checkpoint_row=row,
        )
        run_payload = _read_regular(
            run_root / str(row["arm"]) / "run" / "run.json",
            label=f"{row['arm']} evaluator run binding",
            maximum=64 * 1024,
        )
        if hashlib.sha256(run_payload).hexdigest() != row["run_binding_sha256"]:
            raise ValueError("materialized run binding SHA-256 differs")
        _verify_checkpoint_record(
            record_root / f"{row['checkpoint_record_sha256']}.json",
            checkpoint_row=row,
            run_binding=binding,
        )
    return {"value": receipt, "sha256": digest}


def _upload(
    store: VerifiedObjectStore,
    path: Path,
    uri: str,
    digest: str,
    *,
    deadline: float,
) -> UploadedObject:
    uploaded = store.put_verified(
        path,
        uri,
        expected_sha256=digest,
        deadline=deadline,
        monotonic=time.monotonic,
        wall_deadline=deadline,
        wall_monotonic=time.monotonic,
    )
    final_digest, final_bytes = _hash_regular(path, label=f"upload source {path.name}")
    if (
        uploaded is None
        or uploaded.uri != uri
        or uploaded.sha256 != digest
        or uploaded.bytes != final_bytes
        or final_digest != digest
    ):
        raise ValueError(f"immutable upload could not be verified: {uri}")
    return uploaded


def publish_terminal_pair(
    plan: object,
    *,
    object_store: VerifiedObjectStore,
    operation_id: str,
    timeout_seconds: float = 7200.0,
) -> TerminalPublication:
    """Verify, snapshot, bind, and durably publish a completed v3 pair."""

    from evals.confirmatory.contracts import (
        CHECKPOINT_SCHEMA,
        CONTRACT_VERSION,
        Arm,
        CheckpointRecord,
        ConditionId,
        canonical_json_bytes,
    )
    from evals.confirmatory.run_binding import build_run_binding

    if _OPERATION_RE.fullmatch(operation_id) is None:
        raise ValueError("terminal publication requires the operation ID")
    profile = plan.profile
    manifest = dict(plan.launch_manifest)
    if manifest.get("schema_version") != 3:
        raise ValueError("terminal publication requires a v3 launch manifest")
    receipt_root = (
        Path(plan.scratch_root)
        / "receipts"
        / "checkpoints"
        / f"seed-{plan.seed}"
    )
    _ensure_real_directory(receipt_root)
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_record_paths: dict[str, Path] = {}
    checkpoint_record_sha256: dict[str, str] = {}
    checkpoint_record_uris: dict[str, str] = {}
    terminal_artifact_paths: dict[str, dict[str, Path]] = {}
    prefix = f"{plan.runtime.s3_root.rstrip('/')}/checkpoints/seed-{plan.seed}"
    for launch in sorted(plan.arms, key=lambda item: item.arm):
        metadata_payload = _read_regular(
            launch.checkpoint_path.parent / "checkpoint-meta.json",
            label=f"{launch.arm} checkpoint metadata",
            maximum=64 * 1024,
        )
        metadata = _decode_canonical(
            metadata_payload,
            label=f"{launch.arm} checkpoint metadata",
        )
        if (
            set(metadata) != _CHECKPOINT_METADATA_FIELDS
            or metadata["schema_version"] != 1
            or metadata["receipt_type"] != "memorysplit-training-checkpoint-v1"
            or metadata["run_id"] != launch.run_id
            or metadata["condition"] != launch.condition_id
            or metadata["seed"] != plan.seed
            or metadata["step"] != launch.max_steps
            or metadata["max_steps"] != launch.max_steps
            or metadata["world_size"] != 4
            or metadata["config_fingerprint"] != launch.config_sha256
            or metadata["checkpoint_path"] != "ckpt.pt"
            or metadata["terminal"] is not True
        ):
            raise ValueError(f"{launch.arm} checkpoint is not terminal")
        snapshot = receipt_root / f"{launch.arm}.pt"
        digest, byte_count = _snapshot_regular(
            launch.checkpoint_path,
            snapshot,
            label=f"{launch.arm} terminal checkpoint",
        )
        if (
            metadata["checkpoint_sha256"] != digest
            or metadata["checkpoint_bytes"] != byte_count
        ):
            raise ValueError(
                f"{launch.arm} terminal checkpoint differs from its metadata"
            )
        configuration = launch.checkpoint_path.parent / "configuration.yaml"
        config_digest, _config_bytes = _snapshot_regular(
            launch.config_path,
            configuration,
            label=f"{launch.arm} evaluator configuration",
            mode=0o444,
        )
        if config_digest != launch.config_sha256:
            raise ValueError(f"{launch.arm} configuration binding changed")
        run_binding = build_run_binding(
            run_root=launch.checkpoint_path.parent,
            run_id=launch.run_id,
            checkpoint_path=launch.checkpoint_path,
            configuration_path=configuration,
            route_dose_sha256=launch.route_dose_sha256,
            corpus_sha256=plan.ordered_stream_sha256,
            code_sha256=plan.release_members_sha256,
            seed=plan.seed,
            condition_id=launch.condition_id,
        )
        if set(run_binding) != _RUN_BINDING_FIELDS:
            raise AssertionError("run binding schema drift")
        _write_immutable(
            launch.checkpoint_path.parent / "run.json",
            _canonical_json(run_binding),
            mode=0o444,
        )
        run_binding_path = launch.checkpoint_path.parent / "run.json"
        run_binding_sha256, _run_binding_bytes = _hash_regular(
            run_binding_path,
            label=f"{launch.arm} evaluator run binding",
        )
        checkpoint_record = CheckpointRecord(
            record_type=CHECKPOINT_SCHEMA,
            schema_version=CONTRACT_VERSION,
            checkpoint_sha256=digest,
            model_id=launch.model_id,
            arm=Arm.DENSE if launch.arm == "dense" else Arm.SPLIT,
            condition_id=ConditionId(launch.condition_id),
            seed=plan.seed,
            raw_token_count=launch.raw_token_count,
            configuration_sha256=config_digest,
            route_dose_sha256=launch.route_dose_sha256,
            corpus_sha256=plan.ordered_stream_sha256,
            code_sha256=plan.release_members_sha256,
        )
        record_payload = canonical_json_bytes(checkpoint_record.to_dict())
        record_digest = hashlib.sha256(record_payload).hexdigest()
        record_path = receipt_root / "records" / f"{record_digest}.json"
        _write_immutable(record_path, record_payload)
        checkpoint_record_paths[launch.arm] = record_path
        checkpoint_record_sha256[launch.arm] = record_digest
        checkpoint_uri = (
            f"{prefix}/{launch.arm}/sha256/{digest}.pt"
        )
        configuration_uri = (
            f"{prefix}/{launch.arm}/configuration/sha256/"
            f"{config_digest}.yaml"
        )
        run_binding_uri = (
            f"{prefix}/{launch.arm}/run-binding/sha256/"
            f"{run_binding_sha256}.json"
        )
        record_uri = (
            f"{prefix}/{launch.arm}/records/{record_digest}.json"
        )
        checkpoint_row = {
            "run_id": launch.run_id,
            "arm": launch.arm,
            "seed": plan.seed,
            "path": f"{launch.arm}.pt",
            "sha256": digest,
            "checkpoint_uri": checkpoint_uri,
            "config_sha256": launch.config_sha256,
            "configuration_uri": configuration_uri,
            "run_binding_sha256": run_binding_sha256,
            "run_binding_uri": run_binding_uri,
            "checkpoint_record_sha256": record_digest,
            "checkpoint_record_uri": record_uri,
            "dataset_sha256": plan.corpus_receipt_sha256,
            "source_commit": plan.code_commit,
            "step": launch.max_steps,
            "world_size": 4,
        }
        if _verify_run_directory(
            launch.checkpoint_path.parent,
            checkpoint_row=checkpoint_row,
        ) != run_binding:
            raise ValueError(f"{launch.arm} evaluator run binding changed")
        checkpoint_rows.append(checkpoint_row)
        terminal_artifact_paths[launch.arm] = {
            "checkpoint": snapshot,
            "configuration": configuration,
            "run_binding": run_binding_path,
            "checkpoint_record": record_path,
        }
    receipt = {
        "schema_version": 3,
        "provider": profile.provider,
        "release_sha256": plan.release_sha256,
        "run_manifest_sha256": manifest["run_manifest_sha256"],
        "dataset_sha256": plan.corpus_receipt_sha256,
        "source_commit": plan.code_commit,
        "cohort_assignment_sha256": manifest["cohort_assignment_sha256"],
        "preregistration_sha256": manifest["preregistration_sha256"],
        "hardware_amendment_sha256": manifest["hardware_amendment_sha256"],
        "provider_selection_sha256": manifest["provider_selection_sha256"],
        "profile_sha256": profile.sha256,
        "sealed_fixture_sha256": manifest["sealed_fixture_sha256"],
        "checkpoints": checkpoint_rows,
    }
    receipt_payload = _canonical_json(receipt)
    verify_checkpoint_receipt_bytes(receipt_payload, require_durable=True)
    receipt_digest = hashlib.sha256(receipt_payload).hexdigest()
    receipt_path = receipt_root / "receipt.json"
    _write_immutable(receipt_path, receipt_payload)
    deadline = time.monotonic() + float(timeout_seconds)
    for row in checkpoint_rows:
        arm = str(row["arm"])
        paths = terminal_artifact_paths[arm]
        _upload(
            object_store,
            paths["checkpoint"],
            str(row["checkpoint_uri"]),
            str(row["sha256"]),
            deadline=deadline,
        )
        _upload(
            object_store,
            paths["configuration"],
            str(row["configuration_uri"]),
            str(row["config_sha256"]),
            deadline=deadline,
        )
        _upload(
            object_store,
            paths["run_binding"],
            str(row["run_binding_uri"]),
            str(row["run_binding_sha256"]),
            deadline=deadline,
        )
        record_digest = checkpoint_record_sha256[arm]
        _upload(
            object_store,
            paths["checkpoint_record"],
            str(row["checkpoint_record_uri"]),
            record_digest,
            deadline=deadline,
        )
        checkpoint_record_uris[arm] = str(row["checkpoint_record_uri"])
    receipt_uri = f"{prefix}/receipts/{receipt_digest}.json"
    completion_uri = f"{prefix}/completions/{operation_id}.json"
    _upload(object_store, receipt_path, receipt_uri, receipt_digest, deadline=deadline)
    _upload(
        object_store,
        receipt_path,
        completion_uri,
        receipt_digest,
        deadline=deadline,
    )
    return TerminalPublication(
        receipt_path=receipt_path,
        receipt_sha256=receipt_digest,
        receipt_uri=receipt_uri,
        completion_uri=completion_uri,
        checkpoint_record_paths=dict(checkpoint_record_paths),
        checkpoint_record_sha256=dict(checkpoint_record_sha256),
        checkpoint_record_uris=dict(checkpoint_record_uris),
    )


def verify_durable_terminal(
    *,
    receipt_path: Path,
    run_root: Path,
    s3_root: str,
    object_store: VerifiedObjectStore,
    expected: Mapping[str, object],
    timeout_seconds: float = 7200.0,
) -> dict[str, object]:
    payload = _read_regular(
        receipt_path,
        label="terminal checkpoint receipt",
        maximum=1 << 20,
    )
    receipt = verify_checkpoint_receipt_bytes(
        payload,
        expected=expected,
        require_durable=True,
    )
    digest = hashlib.sha256(payload).hexdigest()
    seed = receipt["checkpoints"][0]["seed"]
    prefix = f"{s3_root.rstrip('/')}/checkpoints/seed-{seed}"
    deadline = time.monotonic() + float(timeout_seconds)
    for row in receipt["checkpoints"]:
        binding = _verify_run_directory(
            run_root / str(row["arm"]) / "run",
            checkpoint_row=row,
        )
        path = receipt_path.parent / str(row["path"])
        local_digest, _local_bytes = _hash_regular(
            path,
            label=f"{row['arm']} checkpoint",
        )
        if local_digest != row["sha256"]:
            raise ValueError("local checkpoint differs from its terminal receipt")
        _upload(
            object_store,
            path,
            str(row["checkpoint_uri"]),
            str(row["sha256"]),
            deadline=deadline,
        )
        run_directory = run_root / str(row["arm"]) / "run"
        _upload(
            object_store,
            run_directory / "configuration.yaml",
            str(row["configuration_uri"]),
            str(row["config_sha256"]),
            deadline=deadline,
        )
        _upload(
            object_store,
            run_directory / "run.json",
            str(row["run_binding_uri"]),
            str(row["run_binding_sha256"]),
            deadline=deadline,
        )
        record_path = (
            receipt_path.parent
            / "records"
            / f"{row['checkpoint_record_sha256']}.json"
        )
        _verify_checkpoint_record(
            record_path,
            checkpoint_row=row,
            run_binding=binding,
        )
        _upload(
            object_store,
            record_path,
            str(row["checkpoint_record_uri"]),
            str(row["checkpoint_record_sha256"]),
            deadline=deadline,
        )
    receipt_uri = f"{prefix}/receipts/{digest}.json"
    _upload(object_store, receipt_path, receipt_uri, digest, deadline=deadline)
    return {"value": receipt, "sha256": digest, "uri": receipt_uri}


def publish_evaluation_pair(
    *,
    evaluation_root: Path,
    checkpoint_receipt_path: Path,
    s3_root: str,
    operation_id: str,
    sealed_evaluation_sha256: str,
    study_lock_sha256: str,
    fleet_plan_sha256: str,
    fleet_wave: int,
    launch_readiness_sha256: str,
    object_store: VerifiedObjectStore,
    expected: Mapping[str, object] | None = None,
    timeout_seconds: float = 7200.0,
) -> EvaluationPublication:
    checkpoint_payload = _read_regular(
        checkpoint_receipt_path,
        label="terminal checkpoint receipt",
        maximum=1 << 20,
    )
    checkpoint = verify_checkpoint_receipt_bytes(
        checkpoint_payload,
        expected=expected,
        require_durable=True,
    )
    checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
    _sha256(sealed_evaluation_sha256, label="sealed evaluation")
    _sha256(study_lock_sha256, label="study lock")
    _sha256(fleet_plan_sha256, label="fleet plan")
    _sha256(launch_readiness_sha256, label="launch readiness")
    if type(fleet_wave) is not int or fleet_wave < 0:
        raise ValueError("fleet wave must be a nonnegative integer")
    if _OPERATION_RE.fullmatch(operation_id) is None:
        raise ValueError("evaluation publication requires the operation ID")
    seed = checkpoint["checkpoints"][0]["seed"]
    deadline = time.monotonic() + float(timeout_seconds)
    prefix = f"{s3_root.rstrip('/')}/evaluations/seed-{seed}"
    arm_rows: list[dict[str, object]] = []
    for checkpoint_row in checkpoint["checkpoints"]:
        run_id = str(checkpoint_row["run_id"])
        arm = str(checkpoint_row["arm"])
        output = evaluation_root / run_id
        try:
            output_fd = _directory_fd(output, create=False)
        except OSError as error:
            raise ValueError(f"{arm} evaluation output is unavailable") from error
        try:
            entries = set(os.listdir(output_fd))
            if entries != EVALUATION_ARTIFACTS:
                raise ValueError(
                    f"{arm} evaluation artifact inventory is not closed"
                )
            for name in entries:
                metadata = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError(
                        f"{arm} evaluation artifact inventory is not closed"
                    )
        finally:
            os.close(output_fd)
        if entries != EVALUATION_ARTIFACTS:
            raise ValueError(f"{arm} evaluation artifact inventory is not closed")
        artifacts: list[dict[str, object]] = []
        for name in sorted(EVALUATION_ARTIFACTS):
            path = output / name
            digest, artifact_bytes = _hash_regular(
                path,
                label=f"{arm} evaluation {name}",
            )
            uri = f"{prefix}/{run_id}/{name}/sha256/{digest}"
            uploaded = _upload(
                object_store,
                path,
                uri,
                digest,
                deadline=deadline,
            )
            artifacts.append(
                {
                    "path": name,
                    "bytes": artifact_bytes,
                    "sha256": digest,
                    "uri": uri,
                }
            )
        arm_rows.append(
            {
                "run_id": run_id,
                "arm": arm,
                "checkpoint_sha256": checkpoint_row["sha256"],
                "artifacts": artifacts,
            }
        )
    checkpoint_uri = (
        f"{s3_root.rstrip('/')}/checkpoints/seed-{seed}/"
        f"receipts/{checkpoint_digest}.json"
    )
    receipt = {
        "schema_version": 3,
        "receipt_type": "memorysplit-aws-paired-evaluation-v3",
        "status": "closed",
        "provider": checkpoint["provider"],
        "seed": seed,
        "release_sha256": checkpoint["release_sha256"],
        "run_manifest_sha256": checkpoint["run_manifest_sha256"],
        "dataset_sha256": checkpoint["dataset_sha256"],
        "source_commit": checkpoint["source_commit"],
        "cohort_assignment_sha256": checkpoint["cohort_assignment_sha256"],
        "preregistration_sha256": checkpoint["preregistration_sha256"],
        "hardware_amendment_sha256": checkpoint["hardware_amendment_sha256"],
        "provider_selection_sha256": checkpoint["provider_selection_sha256"],
        "profile_sha256": checkpoint["profile_sha256"],
        "sealed_fixture_sha256": checkpoint["sealed_fixture_sha256"],
        "sealed_evaluation_sha256": sealed_evaluation_sha256,
        "study_lock_sha256": study_lock_sha256,
        "fleet_plan_sha256": fleet_plan_sha256,
        "fleet_wave": fleet_wave,
        "launch_readiness_sha256": launch_readiness_sha256,
        "operation_id": operation_id,
        "checkpoint_receipt": {
            "sha256": checkpoint_digest,
            "uri": checkpoint_uri,
        },
        "arms": arm_rows,
    }
    payload = _canonical_json(receipt)
    verified_evaluation = verify_evaluation_receipt_bytes(
        payload,
        expected=expected,
    )
    checkpoint_by_arm = {
        str(row["arm"]): row for row in checkpoint["checkpoints"]
    }
    for arm_row in verified_evaluation["arms"]:
        checkpoint_row = checkpoint_by_arm[str(arm_row["arm"])]
        if (
            arm_row["run_id"] != checkpoint_row["run_id"]
            or arm_row["checkpoint_sha256"] != checkpoint_row["sha256"]
        ):
            raise ValueError("evaluation receipt does not bind its checkpoint pair")
    digest = hashlib.sha256(payload).hexdigest()
    receipt_path = evaluation_root / f"paired-seed-{seed}.json"
    _write_immutable(receipt_path, payload)
    receipt_uri = f"{prefix}/receipts/{digest}.json"
    completion_uri = f"{prefix}/completions/{operation_id}.json"
    result_uri = f"{s3_root.rstrip('/')}/results/seed-{seed}.json"
    for uri in (receipt_uri, completion_uri, result_uri):
        _upload(object_store, receipt_path, uri, digest, deadline=deadline)
    return EvaluationPublication(
        receipt_path=receipt_path,
        receipt_sha256=digest,
        receipt_uri=receipt_uri,
        completion_uri=completion_uri,
        result_uri=result_uri,
    )


def _store(region: str, kms_key_id: str, parent: Path) -> S3ObjectStore:
    _ensure_real_directory(parent)
    home = tempfile.mkdtemp(prefix=".aws-artifact-home-", dir=parent)
    os.chmod(home, 0o700)
    return S3ObjectStore(
        region=region,
        kms_key_id=kms_key_id,
        environment={
            "AWS_REGION": region,
            "HOME": home,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-terminal")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--s3-root", required=True)
    verify.add_argument("--provider", required=True)
    verify.add_argument("--run-manifest-sha256", required=True)
    verify.add_argument("--sealed-fixture-sha256", required=True)
    materialized = commands.add_parser("verify-materialized")
    materialized.add_argument("--receipt", type=Path, required=True)
    materialized.add_argument("--run-root", type=Path, required=True)
    materialized.add_argument("--record-root", type=Path, required=True)
    materialized.add_argument("--expected-receipt-sha256", required=True)
    materialized.add_argument("--provider", required=True)
    materialized.add_argument("--run-manifest-sha256", required=True)
    materialized.add_argument("--sealed-fixture-sha256", required=True)
    publish = commands.add_parser("publish-evaluation")
    publish.add_argument("--evaluation-root", type=Path, required=True)
    publish.add_argument("--checkpoint-receipt", type=Path, required=True)
    publish.add_argument("--s3-root", required=True)
    publish.add_argument("--provider", required=True)
    publish.add_argument("--run-manifest-sha256", required=True)
    publish.add_argument("--sealed-fixture-sha256", required=True)
    publish.add_argument("--sealed-evaluation-sha256", required=True)
    publish.add_argument("--study-lock-sha256", required=True)
    publish.add_argument("--fleet-plan-sha256", required=True)
    publish.add_argument("--fleet-wave", type=int, required=True)
    publish.add_argument("--launch-readiness-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    home: Path | None = None
    try:
        arguments = _parser().parse_args(argv)
        operation_id = os.environ.get("MS_OPERATION_ID", "")
        region = os.environ.get("AWS_REGION", "")
        kms_key_id = os.environ.get("MS_S3_KMS_KEY_ID", "")
        parent = (
            arguments.receipt.parent
            if arguments.command in {"verify-terminal", "verify-materialized"}
            else arguments.evaluation_root
        )
        store = _store(region, kms_key_id, parent)
        home = Path(store._environment["HOME"])
        if arguments.command == "verify-terminal":
            result = verify_durable_terminal(
                receipt_path=arguments.receipt,
                run_root=arguments.run_root,
                s3_root=arguments.s3_root,
                object_store=store,
                expected={
                    "provider": arguments.provider,
                    "run_manifest_sha256": arguments.run_manifest_sha256,
                    "sealed_fixture_sha256": (
                        arguments.sealed_fixture_sha256
                    ),
                },
            )
            report = {
                "schema_version": 1,
                "verified": True,
                "receipt_sha256": result["sha256"],
                "receipt_uri": result["uri"],
            }
        elif arguments.command == "verify-materialized":
            result = verify_materialized_terminal(
                receipt_path=arguments.receipt,
                run_root=arguments.run_root,
                record_root=arguments.record_root,
                expected_receipt_sha256=(
                    arguments.expected_receipt_sha256
                ),
                expected={
                    "provider": arguments.provider,
                    "run_manifest_sha256": arguments.run_manifest_sha256,
                    "sealed_fixture_sha256": (
                        arguments.sealed_fixture_sha256
                    ),
                },
            )
            report = {
                "schema_version": 1,
                "verified": True,
                "receipt_sha256": result["sha256"],
            }
        else:
            publication = publish_evaluation_pair(
                evaluation_root=arguments.evaluation_root,
                checkpoint_receipt_path=arguments.checkpoint_receipt,
                s3_root=arguments.s3_root,
                operation_id=operation_id,
                sealed_evaluation_sha256=arguments.sealed_evaluation_sha256,
                study_lock_sha256=arguments.study_lock_sha256,
                fleet_plan_sha256=arguments.fleet_plan_sha256,
                fleet_wave=arguments.fleet_wave,
                launch_readiness_sha256=arguments.launch_readiness_sha256,
                object_store=store,
                expected={
                    "provider": arguments.provider,
                    "run_manifest_sha256": arguments.run_manifest_sha256,
                    "sealed_fixture_sha256": (
                        arguments.sealed_fixture_sha256
                    ),
                },
            )
            report = {
                "schema_version": 1,
                "published": True,
                "receipt_sha256": publication.receipt_sha256,
                "receipt_uri": publication.receipt_uri,
                "completion_uri": publication.completion_uri,
                "result_uri": publication.result_uri,
            }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    finally:
        if home is not None:
            try:
                home.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
