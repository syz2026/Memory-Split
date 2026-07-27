"""Typed construction and atomic publication of Split/Dense pairing receipts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.checkpoint_binding import (
    canonical_configuration_sha256,
    canonical_shared_configuration_sha256,
)
from evals.relational_metrics import confirmatory_study_definition_sha256


PAIR_FIELDS = (
    "split_checkpoint_sha256",
    "dense_checkpoint_sha256",
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "split_configuration_sha256",
    "dense_configuration_sha256",
    "result_schema_sha256",
    "split_result_provenance_sha256",
    "dense_result_provenance_sha256",
    "study_provenance_sha256",
    "split_pair_fingerprint",
    "dense_pair_fingerprint",
)
_RECEIPT_FIELDS = (
    "record_type",
    "schema_version",
    *PAIR_FIELDS,
    "receipt_sha256",
)
_SHARED_ANCHOR_FIELDS = (
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "result_schema_sha256",
)
_HASH_FIELDS = set(PAIR_FIELDS) - {"model_id", "seed", "raw_token_count"}
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _anchor_value(anchor: object, name: str) -> Any:
    if isinstance(anchor, Mapping):
        if name not in anchor:
            raise ValueError(f"pairing anchor is missing {name}")
        return anchor[name]
    try:
        return getattr(anchor, name)
    except AttributeError as exc:
        raise ValueError(f"pairing anchor is missing {name}") from exc


def _pair_fingerprint(side: str, values: Mapping[str, object]) -> str:
    return _sha256(
        {
            "record_type": "relational_pair_fingerprint",
            "schema_version": 1,
            "side": side,
            "checkpoint_sha256": values[
                f"{side}_checkpoint_sha256"
            ],
            "model_id": values["model_id"],
            "seed": values["seed"],
            "raw_token_count": values["raw_token_count"],
            "evaluator_sha256": values["evaluator_sha256"],
            "data_sha256": values["data_sha256"],
            "relation_schema_sha256": values[
                "relation_schema_sha256"
            ],
            "configuration_sha256": values[
                f"{side}_configuration_sha256"
            ],
            "result_schema_sha256": values["result_schema_sha256"],
            "result_provenance_sha256": values[
                f"{side}_result_provenance_sha256"
            ],
            "study_provenance_sha256": values[
                "study_provenance_sha256"
            ],
        }
    )


def _study_provenance(values: Mapping[str, object]) -> str:
    return _sha256(
        {
            "record_type": "relational_paired_study_provenance",
            "schema_version": 2,
            "study_definition_sha256": (
                confirmatory_study_definition_sha256()
            ),
            "shared_anchor": {
                field: values[field] for field in _SHARED_ANCHOR_FIELDS
            },
            "split_configuration_sha256": values[
                "split_configuration_sha256"
            ],
            "dense_configuration_sha256": values[
                "dense_configuration_sha256"
            ],
        }
    )


@dataclass(frozen=True)
class PairingReceipt:
    record_type: str
    schema_version: int
    split_checkpoint_sha256: str
    dense_checkpoint_sha256: str
    model_id: str
    seed: int
    raw_token_count: int
    evaluator_sha256: str
    data_sha256: str
    relation_schema_sha256: str
    split_configuration_sha256: str
    dense_configuration_sha256: str
    result_schema_sha256: str
    split_result_provenance_sha256: str
    dense_result_provenance_sha256: str
    study_provenance_sha256: str
    split_pair_fingerprint: str
    dense_pair_fingerprint: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_receipt_mapping(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PairingReceipt":
        return cls(**dict(raw))


def _validate_receipt_mapping(raw: Mapping[str, object]) -> None:
    if set(raw) != set(_RECEIPT_FIELDS):
        raise ValueError("pairing receipt fields are invalid")
    if (
        raw["record_type"] != "paired_run_receipt"
        or raw["schema_version"] != 3
    ):
        raise ValueError("pairing receipt contract is invalid")
    if not isinstance(raw["model_id"], str) or not raw["model_id"]:
        raise ValueError("pairing receipt model_id must be nonempty")
    for field in ("seed", "raw_token_count"):
        value = raw[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"pairing receipt {field} must be nonnegative")
    for field in _HASH_FIELDS | {"receipt_sha256"}:
        _require_hash(raw[field], f"pairing receipt {field}")
    for field in (
        "checkpoint_sha256",
        "configuration_sha256",
        "result_provenance_sha256",
        "pair_fingerprint",
    ):
        if raw[f"split_{field}"] == raw[f"dense_{field}"]:
            raise ValueError(
                f"pairing receipt {field} values must be distinct"
            )
    if raw["study_provenance_sha256"] != _study_provenance(raw):
        raise ValueError("pairing receipt study provenance mismatch")
    for side in ("split", "dense"):
        expected = _pair_fingerprint(side, raw)
        if raw[f"{side}_pair_fingerprint"] != expected:
            raise ValueError(
                f"pairing receipt {side} pair fingerprint mismatch"
            )
    payload = {
        key: raw[key] for key in _RECEIPT_FIELDS if key != "receipt_sha256"
    }
    if raw["receipt_sha256"] != _sha256(payload):
        raise ValueError("pairing receipt hash mismatch")


def validate_pairing_receipt(raw: object) -> PairingReceipt:
    """Validate canonical receipt fields and both arm fingerprints."""

    if isinstance(raw, PairingReceipt):
        _validate_receipt_mapping(raw.to_dict())
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("pairing receipt must contain an object")
    _validate_receipt_mapping(raw)
    return PairingReceipt.from_dict(raw)


def build_pairing_receipt(
    split_anchor: object,
    dense_anchor: object,
    split_config: Mapping[str, Any],
    dense_config: Mapping[str, Any],
) -> PairingReceipt:
    """Build a receipt only from validated matrix anchors and run configs."""

    if not isinstance(split_config, Mapping) or not isinstance(
        dense_config, Mapping
    ):
        raise ValueError("pairing receipt run configs must be mappings")
    for side, anchor, config in (
        ("split", split_anchor, split_config),
        ("dense", dense_anchor, dense_config),
    ):
        if _anchor_value(anchor, "arm") != side:
            raise ValueError(f"pairing receipt {side} anchor arm mismatch")
        if config.get("condition") != side:
            raise ValueError(
                f"pairing receipt {side} run config condition mismatch"
            )
        actual_configuration = canonical_configuration_sha256(config)
        if actual_configuration != _anchor_value(
            anchor, "configuration_sha256"
        ):
            raise ValueError(
                f"pairing receipt {side} full configuration mismatch"
            )
    split_shared = canonical_shared_configuration_sha256(split_config)
    dense_shared = canonical_shared_configuration_sha256(dense_config)
    if split_shared != dense_shared:
        raise ValueError(
            "pairing receipt shared configuration mismatch"
        )
    for field in _SHARED_ANCHOR_FIELDS:
        if _anchor_value(split_anchor, field) != _anchor_value(
            dense_anchor, field
        ):
            raise ValueError(f"pairing receipt shared {field} mismatch")

    values: dict[str, object] = {
        "split_checkpoint_sha256": _anchor_value(
            split_anchor, "checkpoint_sha256"
        ),
        "dense_checkpoint_sha256": _anchor_value(
            dense_anchor, "checkpoint_sha256"
        ),
        **{
            field: _anchor_value(split_anchor, field)
            for field in _SHARED_ANCHOR_FIELDS
        },
        "split_configuration_sha256": _anchor_value(
            split_anchor, "configuration_sha256"
        ),
        "dense_configuration_sha256": _anchor_value(
            dense_anchor, "configuration_sha256"
        ),
        "split_result_provenance_sha256": _anchor_value(
            split_anchor, "provenance_sha256"
        ),
        "dense_result_provenance_sha256": _anchor_value(
            dense_anchor, "provenance_sha256"
        ),
    }
    values["study_provenance_sha256"] = _study_provenance(values)
    values["split_pair_fingerprint"] = _pair_fingerprint("split", values)
    values["dense_pair_fingerprint"] = _pair_fingerprint("dense", values)
    payload = {
        "record_type": "paired_run_receipt",
        "schema_version": 3,
        **values,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return validate_pairing_receipt(payload)


def publish_pairing_receipt(
    path: str | Path,
    receipt: PairingReceipt,
) -> Path:
    """Atomically create the canonical Split receipt without overwrite."""

    validated = validate_pairing_receipt(receipt)
    output = Path(path)
    if (
        output.name != "pairing-receipt.json"
        or ".." in output.parts
        or not output.is_absolute()
    ):
        raise ValueError(
            "pairing receipt output must use the canonical absolute filename"
        )
    parent = output.parent
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent.absolute()
    ):
        raise ValueError(
            "pairing receipt parent must be a canonical non-symlink directory"
        )
    if os.path.lexists(output):
        raise FileExistsError(f"pairing receipt already exists: {output}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("no-follow pairing receipt publication unsupported")

    parent_fd: int | None = None
    temporary_name: str | None = None
    lock_owned = False
    lock_name = ".pairing-receipt.publish.lock"
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        opened = os.fstat(parent_fd)
        current = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise ValueError(
                "pairing receipt parent changed during publication"
            )
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"pairing receipt already exists: {output}"
            )
        lock_fd = os.open(
            lock_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        os.close(lock_fd)
        lock_owned = True
        temporary_name = (
            f".pairing-receipt.{os.getpid()}."
            f"{os.urandom(12).hex()}.tmp"
        )
        descriptor = os.open(
            temporary_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            content = _canonical_bytes(validated.to_dict())
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        os.fsync(parent_fd)
        return output
    finally:
        if parent_fd is not None:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if lock_owned:
                try:
                    os.unlink(lock_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)
