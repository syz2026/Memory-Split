"""Post-training construction of the complete v3 sealed evaluator release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from collections.abc import Mapping, Sequence

from evals.confirmatory.contracts import (
    CheckpointRecord,
    ConditionId,
    canonical_json_bytes,
)
from evals.confirmatory.reporting import (
    _parse_gold,
    _parse_items,
    _parse_stores,
)
from evals.confirmatory.study_lock import (
    REQUIRED_CONTROL_IDS,
    REQUIRED_FAMILIES,
    REQUIRED_MEMORY_MODES,
    REQUIRED_RECEIPTS,
    REQUIRED_STRATA,
    STUDY_LOCK_SCHEMA_V3,
    VALIDITY_EVIDENCE_SCHEMA_V3,
    V3_CONTRACT_VERSION,
    ReceiptState,
    StudyLock,
    ValidityEvidence,
    ValidityReceipt,
    evaluate_readiness,
)

from .aws_sealed_evaluation import (
    REQUIRED_SEALED_MEMBERS,
    load_sealed_evaluation_fixture,
    load_sealed_evaluation_release,
)
from .errors import MsctlError
from .fsutil import open_directory, rename_noreplace_at
from .jsonutil import canonical_sha256, require_sha256


_MAX_RECORD_BYTES = 4 * 1024 * 1024
_MAX_FIXTURE_MEMBER_BYTES = 1 << 30


def _fail(message: str) -> None:
    raise MsctlError("SEALED_EVALUATION_FINALIZATION_INVALID", message)


def _read_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_RECORD_BYTES,
) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            _fail(f"{label} must be one bounded singly linked regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise MsctlError(
            "SEALED_EVALUATION_FINALIZATION_INVALID",
            f"{label} cannot be read safely",
        ) from error
    content = b"".join(chunks)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(content) != after.st_size:
        _fail(f"{label} changed while being read")
    return content


def _canonical_object(path: Path, *, label: str) -> Mapping[str, object]:
    content = _read_regular(path, label=label)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "SEALED_EVALUATION_FINALIZATION_INVALID",
            f"{label} must contain canonical UTF-8 JSON",
        ) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != content:
        _fail(f"{label} must contain one canonical JSON record")
    return value


def _checkpoint_records(paths: Sequence[Path | str]) -> tuple[CheckpointRecord, ...]:
    if isinstance(paths, (str, bytes)) or len(paths) != 20:
        _fail("v3 finalization requires exactly twenty checkpoint records")
    try:
        records = tuple(
            CheckpointRecord.from_dict(
                _canonical_object(
                    Path(path),
                    label=f"checkpoint record {index}",
                )
            )
            for index, path in enumerate(paths)
        )
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "SEALED_EVALUATION_FINALIZATION_INVALID",
            "checkpoint records do not satisfy the closed contract",
        ) from error
    ordered = tuple(
        sorted(records, key=lambda row: (row.seed, row.condition_id.value))
    )
    slots = tuple((row.seed, row.condition_id) for row in ordered)
    expected = tuple(
        (seed, condition)
        for seed in range(10)
        for condition in (ConditionId.DENSE, ConditionId.SPLIT90)
    )
    if slots != expected:
        _fail("checkpoint records do not cover seeds 0..9 Dense/Split90 exactly")
    hashes = tuple(row.checkpoint_sha256 for row in ordered)
    if len(set(hashes)) != len(hashes):
        _fail("checkpoint records contain duplicate checkpoint hashes")
    return ordered


def _validity_receipts(
    paths: Sequence[Path | str],
) -> tuple[ValidityReceipt, ...]:
    if isinstance(paths, (str, bytes)) or len(paths) != len(REQUIRED_RECEIPTS):
        _fail("finalization requires exactly every registered validity receipt")
    try:
        receipts = tuple(
            ValidityReceipt.from_dict(
                _canonical_object(
                    Path(path),
                    label=f"validity receipt {index}",
                )
            )
            for index, path in enumerate(paths)
        )
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "SEALED_EVALUATION_FINALIZATION_INVALID",
            "validity receipts do not satisfy the closed contract",
        ) from error
    by_id = {receipt.receipt_id: receipt for receipt in receipts}
    if (
        len(by_id) != len(receipts)
        or set(by_id) != set(REQUIRED_RECEIPTS)
        or any(receipt.state is not ReceiptState.PASSED for receipt in receipts)
    ):
        _fail("validity receipts must be exact, unique, and passed")
    return tuple(by_id[receipt_id] for receipt_id in REQUIRED_RECEIPTS)


def build_finalized_sealed_evaluation(
    *,
    fixture_root: Path | str,
    checkpoint_records: Sequence[Path | str],
    validity_receipts: Sequence[Path | str],
    preregistration_sha256: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    """Build and fully validate all post-training release bytes in memory."""

    preregistration = require_sha256(
        preregistration_sha256,
        label="v3 preregistration SHA-256",
    )
    fixture = load_sealed_evaluation_fixture(fixture_root)
    fixture_content = {
        name: _read_regular(
            fixture.root / name,
            label=f"fixture {name}",
            max_bytes=_MAX_FIXTURE_MEMBER_BYTES,
        )
        for name in fixture.members
    }
    if {
        name: hashlib.sha256(content).hexdigest()
        for name, content in fixture_content.items()
    } != dict(fixture.members):
        _fail("sealed fixture changed after validation")
    items = _parse_items(fixture_content["items.jsonl"])
    _parse_gold(fixture_content["sealed-gold.jsonl"])
    _parse_stores(fixture_content["stores.jsonl"])
    checkpoints = _checkpoint_records(checkpoint_records)
    receipts = _validity_receipts(validity_receipts)
    checkpoints_content = b"".join(
        canonical_json_bytes(checkpoint.to_dict())
        for checkpoint in checkpoints
    )
    approvals = [
        {
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "seed": checkpoint.seed,
            "condition_id": checkpoint.condition_id.value,
            "configuration_sha256": checkpoint.configuration_sha256,
            "route_dose_sha256": checkpoint.route_dose_sha256,
        }
        for checkpoint in checkpoints
    ]
    evaluation_cells = [
        {
            "item_id": item.item_id,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "seed": checkpoint.seed,
            "condition_id": checkpoint.condition_id.value,
        }
        for checkpoint in checkpoints
        for item in items
    ]
    lock_value = {
        "record_type": STUDY_LOCK_SCHEMA_V3,
        "schema_version": V3_CONTRACT_VERSION,
        "preregistration_sha256": preregistration,
        "release": {
            "items_sha256": fixture.members["items.jsonl"],
            "sealed_gold_sha256": fixture.members["sealed-gold.jsonl"],
            "stores_sha256": fixture.members["stores.jsonl"],
            "checkpoints_sha256": hashlib.sha256(
                checkpoints_content
            ).hexdigest(),
            "item_ids": [item.item_id for item in items],
            "pair_ids": sorted({item.pair_id for item in items}),
            "world_ids": sorted({item.world_id for item in items}),
            "evaluation_cells": evaluation_cells,
            "item_count": len(items),
            "pair_count": len({item.pair_id for item in items}),
            "world_count": len({item.world_id for item in items}),
            "evaluation_cell_count": len(evaluation_cells),
            "required_families": list(REQUIRED_FAMILIES),
            "required_strata": list(REQUIRED_STRATA),
            "required_memory_modes": list(REQUIRED_MEMORY_MODES),
            "required_controls": list(REQUIRED_CONTROL_IDS),
            "sealed_fixture_sha256": fixture.sha256,
        },
        "checkpoints": approvals,
        "validity_receipts": [
            {
                "receipt_id": receipt.receipt_id,
                "kind": receipt.kind,
                "state": receipt.state.value,
                "receipt_sha256": hashlib.sha256(
                    canonical_json_bytes(receipt.to_dict())
                ).hexdigest(),
            }
            for receipt in receipts
        ],
    }
    lock = StudyLock.from_dict(lock_value)
    lock_content = canonical_json_bytes(lock.to_dict())
    lock_sha256 = hashlib.sha256(lock_content).hexdigest()
    validity = ValidityEvidence.from_dict(
        {
            "record_type": VALIDITY_EVIDENCE_SCHEMA_V3,
            "schema_version": V3_CONTRACT_VERSION,
            "study_lock_sha256": lock_sha256,
            "preregistration_sha256": preregistration,
            "receipts": [receipt.to_dict() for receipt in receipts],
        }
    )
    readiness = evaluate_readiness(lock, validity)
    if not readiness.complete or not readiness.valid:
        _fail("finalized study lock validity evidence is not complete")
    content = {
        **fixture_content,
        "checkpoints.jsonl": checkpoints_content,
        "study-lock.json": lock_content,
        "validity.json": canonical_json_bytes(validity.to_dict()),
    }
    if set(content) != set(REQUIRED_SEALED_MEMBERS):
        raise AssertionError("finalized sealed release inventory drifted")
    inventory = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(content.items())
    }
    release_sha256 = canonical_sha256(
        {"schema_version": 1, "members": inventory}
    )
    return content, {
        "schema_version": 1,
        "sealed_fixture_sha256": fixture.sha256,
        "sealed_evaluation_sha256": release_sha256,
        "study_lock_sha256": lock_sha256,
        "members": inventory,
    }


def finalize_sealed_evaluation(
    *,
    fixture_root: Path | str,
    checkpoint_records: Sequence[Path | str],
    validity_receipts: Sequence[Path | str],
    preregistration_sha256: str,
    out: Path | str,
    apply: bool,
) -> dict[str, object]:
    """Plan or exclusively publish one complete N=10 sealed release."""

    content, result = build_finalized_sealed_evaluation(
        fixture_root=fixture_root,
        checkpoint_records=checkpoint_records,
        validity_receipts=validity_receipts,
        preregistration_sha256=preregistration_sha256,
    )
    destination = Path(out)
    report = {**result, "out": str(destination), "published": False}
    if not apply:
        return report
    parent_fd = open_directory(
        destination.parent,
        label="sealed evaluation output parent",
        create=True,
    )
    staging_name = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    staging = destination.parent / staging_name
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        for name, payload in content.items():
            path = staging / name
            descriptor = os.open(
                path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("sealed release write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        staged = load_sealed_evaluation_release(
            staging,
            expected_release_sha256=str(result["sealed_evaluation_sha256"]),
            expected_study_lock_sha256=str(result["study_lock_sha256"]),
            expected_preregistration_sha256=preregistration_sha256,
        )
        if staged.fixture_sha256 != result["sealed_fixture_sha256"]:
            _fail("staged release changed its fixture commitment")
        staging_fd = os.open(
            staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        try:
            rename_noreplace_at(
                parent_fd,
                staging_name,
                parent_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise MsctlError(
                "SEALED_EVALUATION_EXISTS",
                "refusing to replace an existing sealed evaluation",
            ) from error
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
        if staging.exists():
            shutil.rmtree(staging)
    load_sealed_evaluation_release(
        destination,
        expected_release_sha256=str(result["sealed_evaluation_sha256"]),
        expected_study_lock_sha256=str(result["study_lock_sha256"]),
        expected_preregistration_sha256=preregistration_sha256,
    )
    report["published"] = True
    return report
