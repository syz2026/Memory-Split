"""Validation for an externally rooted sealed-evaluation release.

The training release contains evaluator source only.  Protected items, stores,
gold, validity evidence, and the study lock are a separate release whose root
is the canonical hash of every regular member.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from evals.confirmatory.contracts import canonical_json_bytes
from evals.confirmatory.reporting import (
    _parse_checkpoints,
    _parse_gold,
    _parse_items,
    _parse_stores,
    _validate_release,
)
from evals.confirmatory.study_lock import (
    StudyLock,
    ValidityEvidence,
    evaluate_readiness,
)

from .errors import MsctlError
from .jsonutil import canonical_sha256, require_sha256


REQUIRED_SEALED_MEMBERS = frozenset(
    {
        "checkpoints.jsonl",
        "items.jsonl",
        "sealed-gold.jsonl",
        "stores.jsonl",
        "study-lock.json",
        "validity.json",
    }
)
_MAX_MEMBER_BYTES = 1 << 30


@dataclass(frozen=True)
class SealedEvaluationRelease:
    root: Path
    sha256: str
    study_lock_sha256: str
    preregistration_sha256: str
    members: Mapping[str, str]


def _fail(message: str) -> None:
    raise MsctlError("SEALED_EVALUATION_INVALID", message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"sealed JSON repeats field: {key}")
        result[key] = value
    return result


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_MEMBER_BYTES
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
            "SEALED_EVALUATION_INVALID",
            f"{label} cannot be read safely",
        ) from error
    data = b"".join(chunks)
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
    ) or len(data) != after.st_size:
        _fail(f"{label} changed while being read")
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: _fail(
                f"{label} contains non-finite {constant}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "SEALED_EVALUATION_INVALID",
            f"{label} must contain valid UTF-8 JSON",
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value


def load_sealed_evaluation_release(
    root: Path | str,
    *,
    expected_release_sha256: str | None = None,
    expected_study_lock_sha256: str | None = None,
    expected_preregistration_sha256: str | None = None,
) -> SealedEvaluationRelease:
    """Verify every release member and the evaluator's exact sealed contract."""

    candidate = Path(root)
    try:
        root_status = candidate.stat(follow_symlinks=False)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise MsctlError(
            "SEALED_EVALUATION_INVALID",
            "sealed-evaluation release root is unavailable",
        ) from error
    if candidate.is_symlink() or not stat.S_ISDIR(root_status.st_mode):
        _fail("sealed-evaluation release root must be a real directory")

    inventory: dict[str, str] = {}
    content: dict[str, bytes] = {}
    for path in sorted(resolved.rglob("*")):
        relative = path.relative_to(resolved).as_posix()
        status = path.stat(follow_symlinks=False)
        if path.is_symlink():
            _fail("sealed-evaluation release must not contain symlinks")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            _fail("sealed-evaluation release contains a special member")
        payload = _read_regular(path, label=f"sealed member {relative}")
        content[relative] = payload
        inventory[relative] = hashlib.sha256(payload).hexdigest()
    if set(inventory) != set(REQUIRED_SEALED_MEMBERS):
        _fail(
            "sealed-evaluation release member inventory is not exact; "
            f"missing={sorted(REQUIRED_SEALED_MEMBERS - set(inventory))}, "
            f"unknown={sorted(set(inventory) - REQUIRED_SEALED_MEMBERS)}"
        )

    release_sha256 = canonical_sha256(
        {"schema_version": 1, "members": inventory}
    )
    study_lock_sha256 = inventory["study-lock.json"]
    if (
        expected_release_sha256 is not None
        and release_sha256
        != require_sha256(
            expected_release_sha256,
            label="sealed-evaluation release SHA-256",
        )
    ):
        _fail("sealed-evaluation release root hash does not match")
    if (
        expected_study_lock_sha256 is not None
        and study_lock_sha256
        != require_sha256(
            expected_study_lock_sha256,
            label="sealed study-lock SHA-256",
        )
    ):
        _fail("actual study-lock.json hash does not match")

    try:
        lock = StudyLock.from_dict(
            _json_object(
                content["study-lock.json"],
                label="study-lock.json",
            )
        )
        validity = ValidityEvidence.from_dict(
            _json_object(
                content["validity.json"],
                label="validity.json",
            )
        )
        readiness = evaluate_readiness(lock, validity)
        items = _parse_items(content["items.jsonl"])
        gold = _parse_gold(content["sealed-gold.jsonl"])
        stores = _parse_stores(content["stores.jsonl"])
        checkpoints = _parse_checkpoints(content["checkpoints.jsonl"])
        _validate_release(
            content=content,
            lock=lock,
            items=items,
            gold_records=gold,
            stores=stores,
            checkpoints=checkpoints,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MsctlError(
            "SEALED_EVALUATION_INVALID",
            "sealed evaluator release does not satisfy its exact contract",
        ) from error
    if not readiness.complete or not readiness.valid:
        _fail("sealed-evaluation validity evidence is not complete and passing")
    bound = {
        "checkpoints.jsonl": lock.release.checkpoints_sha256,
        "items.jsonl": lock.release.items_sha256,
        "sealed-gold.jsonl": lock.release.sealed_gold_sha256,
        "stores.jsonl": lock.release.stores_sha256,
    }
    if any(inventory[name] != digest for name, digest in bound.items()):
        _fail("sealed members disagree with study-lock commitments")
    if canonical_json_bytes(lock.to_dict()) != content["study-lock.json"]:
        _fail("study-lock bytes are not canonical")
    if canonical_json_bytes(validity.to_dict()) != content["validity.json"]:
        _fail("validity evidence bytes are not canonical")
    if (
        expected_preregistration_sha256 is not None
        and lock.preregistration_sha256
        != require_sha256(
            expected_preregistration_sha256,
            label="sealed preregistration SHA-256",
        )
    ):
        _fail("sealed study lock binds a different frozen preregistration")

    return SealedEvaluationRelease(
        root=resolved,
        sha256=release_sha256,
        study_lock_sha256=study_lock_sha256,
        preregistration_sha256=lock.preregistration_sha256,
        members=dict(sorted(inventory.items())),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify one closed external sealed-evaluation release."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--expected-study-lock-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        release = load_sealed_evaluation_release(
            arguments.root,
            expected_release_sha256=arguments.expected_release_sha256,
            expected_study_lock_sha256=arguments.expected_study_lock_sha256,
        )
        sys.stdout.write(
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "sealed_evaluation_release_sha256": release.sha256,
                        "study_lock_sha256": release.study_lock_sha256,
                        "members": dict(release.members),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        )
        return 0
    except (MsctlError, OSError, TypeError, ValueError) as error:
        sys.stdout.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": getattr(error, "code", type(error).__name__),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
