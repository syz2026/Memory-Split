from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from corpusgen.wikidata5m import SourceDriftError


_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class BedSnapshotLock:
    repo_id: str
    revision: str
    config: str
    split: str
    rows: int
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.repo_id, "repo_id")
        _require_nonempty(self.config, "config")
        _require_nonempty(self.split, "split")
        if not isinstance(self.revision, str) or not _REVISION_RE.fullmatch(
            self.revision
        ):
            raise ValueError("revision must be a lowercase 40-hex revision")
        for value, name in ((self.rows, "rows"), (self.bytes, "bytes")):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(
            self.sha256
        ):
            raise ValueError("sha256 must be 64 lowercase hex characters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BedSnapshotLock:
        expected = {
            "repo_id",
            "revision",
            "config",
            "split",
            "rows",
            "bytes",
            "sha256",
        }
        if set(value) != expected:
            raise ValueError("invalid BED snapshot lock fields")
        return cls(
            repo_id=value["repo_id"],
            revision=value["revision"],
            config=value["config"],
            split=value["split"],
            rows=value["rows"],
            bytes=value["bytes"],
            sha256=value["sha256"],
        )

    @classmethod
    def from_path(cls, path: str | Path) -> BedSnapshotLock:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid BED snapshot lock JSON: {source}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("BED snapshot lock must be a JSON object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            + b"\n"
        )

    def write(self, path: str | Path) -> None:
        _atomic_write(Path(path), self.canonical_bytes())


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _hash_stream(stream: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := stream.read(1024 * 1024):
        byte_count += len(chunk)
        digest.update(chunk)
    return byte_count, digest.hexdigest()


def _parse_text_row(raw: bytes, path: Path, line_number: int) -> str:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"{path.name}:{line_number}: text must be a nonempty string"
        )
    return text


def _count_verified_rows(stream: BinaryIO, path: Path) -> int:
    rows = 0
    for line_number, raw in enumerate(stream, 1):
        if not raw.strip():
            continue
        _parse_text_row(raw, path, line_number)
        rows += 1
    return rows


def _iter_text_rows(stream: BinaryIO, path: Path) -> Iterator[str]:
    for line_number, raw in enumerate(stream, 1):
        if not raw.strip():
            continue
        yield _parse_text_row(raw, path, line_number)


def lock_bed_snapshot(
    path: str | Path,
    *,
    repo_id: str,
    revision: str,
    config: str,
    split: str,
) -> BedSnapshotLock:
    source = Path(path)
    _require_nonempty(repo_id, "repo_id")
    _require_nonempty(config, "config")
    _require_nonempty(split, "split")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be a lowercase 40-hex revision")

    with source.open("rb") as stream:
        byte_count, sha256 = _hash_stream(stream)
        stream.seek(0)
        rows = _count_verified_rows(stream, source)
    return BedSnapshotLock(
        repo_id=repo_id,
        revision=revision,
        config=config,
        split=split,
        rows=rows,
        bytes=byte_count,
        sha256=sha256,
    )


def iter_verified_bed(
    path: str | Path,
    lock: BedSnapshotLock,
) -> Iterator[str]:
    source = Path(path)
    with source.open("rb") as stream:
        byte_count, sha256 = _hash_stream(stream)
        if byte_count != lock.bytes or sha256 != lock.sha256:
            raise SourceDriftError(
                f"BED snapshot drift for {source}: expected "
                f"{lock.bytes} bytes/{lock.sha256}, got "
                f"{byte_count} bytes/{sha256}"
            )
        stream.seek(0)
        rows = _count_verified_rows(stream, source)
        if rows != lock.rows:
            raise SourceDriftError(
                f"BED snapshot row count drift for {source}: "
                f"expected {lock.rows}, got {rows}"
            )
        stream.seek(0)
        yield from _iter_text_rows(stream, source)
