"""Pinned verification for the canonical Task 4 parallel-corpus receipt."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping


_FORMAT = "memorysplit-parallel-corpus-v2"
_COMPILER_VERSION = "metadata-first-foundation-v1"
_TARGET_WEIGHT_NAMES = (
    "dense_target_weights",
    "split90_target_weights",
)
_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "assignments_sha256",
        "build_id",
        "catalog_sha256",
        "compiler_version",
        "config",
        "format",
        "logical_tokens",
        "merkle_root_sha256",
        "metadata_sha256",
        "ordered_stream_sha256",
        "packed_stream_sha256",
        "packed_tokens",
        "padding_tokens",
        "record_count",
        "renderer_id",
        "schedule_sha256",
        "shard_count",
        "sidecar_sets",
    }
)
_ARTIFACT_FIELDS = frozenset({"bytes", "path", "sha256"})
_SIDECAR_SET_FIELDS = frozenset(
    {"artifacts", "dtype", "items", "name", "stream_sha256"}
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "shard_count",
        "shard_index",
        "token_end",
        "token_start",
        "update_end",
        "update_start",
    }
)
_FOUNDATION_PATHS = frozenset(
    {
        "assignments.jsonl",
        "catalog.jsonl",
        "metadata.jsonl",
        "schedule.jsonl",
    }
)
_MAX_METADATA_BYTES = 256 * 1024 * 1024


class CorpusContractError(ValueError):
    """The materialized Task 4 publication does not match its receipt."""


@dataclass(frozen=True)
class PinnedCorpusFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CorpusEvidence:
    receipt: Mapping[str, object]
    files: tuple[PinnedCorpusFile, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusContractError(f"corpus JSON repeats key: {key}")
        result[key] = value
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CorpusContractError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusContractError(f"{label} must be a positive integer")
    return value


def _artifact_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("~")
        or "$" in value
    ):
        raise CorpusContractError(f"{label} is not portable")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise CorpusContractError(f"{label} is not portable")
    return path.as_posix()


def _open_directory_at(parent_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )


def _open_file(root_fd: int, relative: str) -> tuple[int, list[int]]:
    parts = PurePosixPath(relative).parts
    parent_fd = root_fd
    opened_directories: list[int] = []
    try:
        for part in parts[:-1]:
            parent_fd = _open_directory_at(parent_fd, part)
            opened_directories.append(parent_fd)
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError:
        for directory_fd in reversed(opened_directories):
            os.close(directory_fd)
        raise
    return descriptor, opened_directories


def _stream_file(
    root_fd: int,
    relative: str,
    *,
    consumer: Callable[[bytes], None] | None = None,
    collect: bool = False,
) -> tuple[int, str, bytes | None]:
    try:
        descriptor, directory_fds = _open_file(root_fd, relative)
    except OSError as error:
        raise CorpusContractError(
            f"corpus artifact is missing, symlinked, or unsafe: {relative}"
        ) from error
    chunks: list[bytes] | None = [] if collect else None
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CorpusContractError(
                f"corpus artifact is not a singly linked regular file: {relative}"
            )
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            if consumer is not None:
                consumer(chunk)
            if chunks is not None:
                if size > _MAX_METADATA_BYTES:
                    raise CorpusContractError(
                        f"corpus metadata artifact is too large: {relative}"
                    )
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or size != after.st_size
        ):
            raise CorpusContractError(
                f"corpus artifact changed while pinned: {relative}"
            )
    finally:
        os.close(descriptor)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    payload = b"".join(chunks) if chunks is not None else None
    return size, digest.hexdigest(), payload


def _namespace(root_fd: int) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        try:
            names = os.listdir(directory_fd)
        except OSError as error:
            raise CorpusContractError("corpus namespace cannot be enumerated") from error
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise CorpusContractError("corpus namespace contains an unsafe name")
            relative_parts = (*prefix, name)
            relative = PurePosixPath(*relative_parts).as_posix()
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise CorpusContractError(
                    f"corpus namespace contains symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                try:
                    child_fd = _open_directory_at(directory_fd, name)
                except OSError as error:
                    raise CorpusContractError(
                        f"corpus directory is unsafe: {relative}"
                    ) from error
                try:
                    visit(child_fd, relative_parts)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise CorpusContractError(
                    f"corpus namespace contains special entry: {relative}"
                )

    visit(root_fd, ())
    return files, directories


def _artifact_records(
    artifacts: object,
    *,
    label: str,
    require_sorted: bool,
) -> list[dict[str, object]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise CorpusContractError(f"{label} artifacts must be a non-empty list")
    result: list[dict[str, object]] = []
    paths: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise CorpusContractError(f"{label} artifact fields do not match")
        relative = _artifact_path(artifact["path"], label=f"{label} artifact path")
        byte_count = _positive_int(
            artifact["bytes"], label=f"{label} artifact bytes"
        )
        digest = _sha256(
            artifact["sha256"], label=f"{label} artifact SHA-256"
        )
        result.append({"bytes": byte_count, "path": relative, "sha256": digest})
        paths.append(relative)
    if len(paths) != len(set(paths)):
        raise CorpusContractError(f"{label} repeats an artifact path")
    if require_sorted and paths != sorted(paths):
        raise CorpusContractError(f"{label} artifact paths must be sorted")
    return result


def _assignments(payload: bytes) -> tuple[dict[str, int], ...]:
    if not payload or not payload.endswith(b"\n"):
        raise CorpusContractError("assignments must be newline-terminated")
    result: list[dict[str, int]] = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorpusContractError("assignments contain invalid JSON") from error
        if (
            not isinstance(value, dict)
            or set(value) != _ASSIGNMENT_FIELDS
            or _canonical_json_bytes(value) != line
            or any(type(item) is not int for item in value.values())
        ):
            raise CorpusContractError("assignment fields are not canonical")
        result.append(value)
    expected_count = len(result)
    token_start = 0
    update_start = 0
    for index, assignment in enumerate(result):
        if (
            assignment["shard_index"] != index
            or assignment["shard_count"] != expected_count
            or assignment["token_start"] != token_start
            or assignment["update_start"] != update_start
            or assignment["token_end"] <= token_start
            or assignment["update_end"] <= update_start
        ):
            raise CorpusContractError(
                "shard assignments are not ordered and contiguous"
            )
        token_start = assignment["token_end"]
        update_start = assignment["update_end"]
    return tuple(result)


def _default_semantic_verifier(root: Path) -> Mapping[str, object]:
    try:
        from corpusgen.parallel import verify_parallel_corpus

        value = verify_parallel_corpus(root)
    except Exception as error:
        raise CorpusContractError(
            "canonical Task 4 semantic verifier is unavailable or rejected the corpus"
        ) from error
    if not isinstance(value, dict):
        raise CorpusContractError("canonical Task 4 verifier returned invalid evidence")
    return value


def verify_canonical_corpus(
    receipt_path: Path,
    *,
    expected_sha256: str,
    expected_ordered_sha256: str,
    semantic_verifier: Callable[[Path], Mapping[str, object]]
    | None = None,
) -> CorpusEvidence:
    """Verify the complete v2 publication and defer semantic hashes to Task 4."""

    root = receipt_path.parent.resolve(strict=True)
    if receipt_path.name != "receipt.json":
        raise CorpusContractError(
            "canonical Task 4 corpus receipt must be named receipt.json"
        )
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise CorpusContractError("corpus root is missing or unsafe") from error
    try:
        receipt_size, receipt_digest, receipt_bytes = _stream_file(
            root_fd, "receipt.json", collect=True
        )
        del receipt_size
        if receipt_digest != expected_sha256:
            raise CorpusContractError("corpus receipt SHA-256 mismatch")
        assert receipt_bytes is not None
        try:
            receipt = json.loads(
                receipt_bytes.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    CorpusContractError(
                        f"corpus receipt contains non-finite {constant}"
                    )
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorpusContractError(
                "corpus receipt must contain valid UTF-8 JSON"
            ) from error
        if (
            not isinstance(receipt, dict)
            or set(receipt) != _RECEIPT_FIELDS
            or _canonical_json_bytes(receipt) != receipt_bytes
        ):
            raise CorpusContractError(
                "corpus receipt does not match the canonical Task 4 contract"
            )
        if (
            receipt["format"] != _FORMAT
            or receipt["compiler_version"] != _COMPILER_VERSION
        ):
            raise CorpusContractError("corpus receipt format identity mismatch")
        if receipt["ordered_stream_sha256"] != expected_ordered_sha256:
            raise CorpusContractError("corpus ordered stream binding mismatch")
        for field in (
            "assignments_sha256",
            "build_id",
            "catalog_sha256",
            "merkle_root_sha256",
            "metadata_sha256",
            "ordered_stream_sha256",
            "packed_stream_sha256",
            "schedule_sha256",
        ):
            _sha256(receipt[field], label=f"corpus {field}")
        logical_tokens = _positive_int(
            receipt["logical_tokens"], label="corpus logical_tokens"
        )
        packed_tokens = _positive_int(
            receipt["packed_tokens"], label="corpus packed_tokens"
        )
        padding_tokens = receipt["padding_tokens"]
        if (
            type(padding_tokens) is not int
            or padding_tokens < 0
            or packed_tokens - logical_tokens != padding_tokens
        ):
            raise CorpusContractError("corpus padding token count drift")
        _positive_int(receipt["record_count"], label="corpus record_count")
        shard_count = _positive_int(
            receipt["shard_count"], label="corpus shard_count"
        )
        if not isinstance(receipt["renderer_id"], str) or not receipt["renderer_id"]:
            raise CorpusContractError("corpus renderer_id must be non-empty")

        primary = _artifact_records(
            receipt["artifacts"],
            label="corpus primary",
            require_sorted=True,
        )
        sidecar_sets = receipt["sidecar_sets"]
        if not isinstance(sidecar_sets, list) or len(sidecar_sets) != 2:
            raise CorpusContractError(
                "corpus sidecar_sets must be the two canonical exact sets"
            )
        parsed_sidecars: list[tuple[dict[str, object], list[dict[str, object]]]] = []
        names: list[str] = []
        for sidecar_set in sidecar_sets:
            if (
                not isinstance(sidecar_set, dict)
                or set(sidecar_set) != _SIDECAR_SET_FIELDS
            ):
                raise CorpusContractError(
                    "corpus sidecar set fields do not match the contract"
                )
            name = sidecar_set["name"]
            names.append(name)
            if sidecar_set["dtype"] != "uint8":
                raise CorpusContractError(f"corpus sidecar dtype drift: {name}")
            if sidecar_set["items"] != packed_tokens:
                raise CorpusContractError(
                    f"corpus sidecar item count drift: {name}"
                )
            _sha256(
                sidecar_set["stream_sha256"],
                label=f"corpus sidecar stream {name}",
            )
            records = _artifact_records(
                sidecar_set["artifacts"],
                label=f"corpus sidecar {name}",
                require_sorted=False,
            )
            parsed_sidecars.append((sidecar_set, records))
        if tuple(names) != _TARGET_WEIGHT_NAMES:
            raise CorpusContractError(
                "corpus sidecar set names must be canonical and ordered"
            )

        all_records = [
            *primary,
            *(
                artifact
                for _sidecar, records in parsed_sidecars
                for artifact in records
            ),
        ]
        all_paths = [record["path"] for record in all_records]
        if len(all_paths) != len(set(all_paths)):
            raise CorpusContractError("corpus artifact namespace repeats a path")
        actual_files, actual_directories = _namespace(root_fd)
        expected_files = {"receipt.json", *all_paths}
        if actual_files != expected_files:
            missing = expected_files - actual_files
            if any(path.startswith("sidecars/") for path in missing):
                raise CorpusContractError(
                    "corpus sidecar artifact namespace contains missing files"
                )
            raise CorpusContractError(
                "corpus artifact namespace contains missing or extra files"
            )
        expected_directories = {
            PurePosixPath(*PurePosixPath(path).parts[:depth]).as_posix()
            for path in all_paths
            for depth in range(1, len(PurePosixPath(path).parts))
        }
        if actual_directories != expected_directories:
            raise CorpusContractError(
                "corpus artifact namespace contains missing or extra directories"
            )

        record_by_path = {record["path"]: record for record in all_records}
        verified: list[PinnedCorpusFile] = [
            PinnedCorpusFile(path=receipt_path, sha256=receipt_digest)
        ]
        metadata_payloads: dict[str, bytes] = {}
        for relative in sorted(_FOUNDATION_PATHS):
            if relative not in record_by_path:
                raise CorpusContractError(
                    "corpus is missing required foundation artifacts"
                )
            size, digest, payload = _stream_file(root_fd, relative, collect=True)
            record = record_by_path[relative]
            if size != record["bytes"] or digest != record["sha256"]:
                raise CorpusContractError(
                    f"corpus artifact digest or byte count drift: {relative}"
                )
            assert payload is not None
            metadata_payloads[relative] = payload
            verified.append(PinnedCorpusFile(root / relative, digest))
        expected_foundation_hashes = {
            "assignments.jsonl": receipt["assignments_sha256"],
            "catalog.jsonl": receipt["catalog_sha256"],
            "metadata.jsonl": receipt["metadata_sha256"],
            "schedule.jsonl": receipt["schedule_sha256"],
        }
        for relative, expected_digest in expected_foundation_hashes.items():
            if hashlib.sha256(metadata_payloads[relative]).hexdigest() != expected_digest:
                raise CorpusContractError(
                    f"corpus foundation digest drift: {relative}"
                )

        assignments = _assignments(metadata_payloads["assignments.jsonl"])
        if len(assignments) != shard_count:
            raise CorpusContractError("corpus shard count does not match assignments")
        if assignments[-1]["token_end"] != packed_tokens:
            raise CorpusContractError("corpus packed token count does not match assignments")
        config = receipt["config"]
        if not isinstance(config, dict):
            raise CorpusContractError("corpus config must be an object")
        update_tokens = _positive_int(
            config.get("update_tokens"), label="corpus update_tokens"
        )

        token_stream = hashlib.sha256()
        expected_token_paths = [
            f"shards/shard-{index:05d}-of-{shard_count:05d}.bin"
            for index in range(shard_count)
        ]
        if set(record_by_path) - _FOUNDATION_PATHS - {
            artifact["path"]
            for _sidecar, records in parsed_sidecars
            for artifact in records
        } != set(expected_token_paths):
            raise CorpusContractError(
                "corpus primary token artifact namespace does not match assignments"
            )
        for assignment, relative in zip(
            assignments, expected_token_paths, strict=True
        ):
            if (
                assignment["token_start"]
                != assignment["update_start"] * update_tokens
                or assignment["token_end"]
                != assignment["update_end"] * update_tokens
            ):
                raise CorpusContractError(
                    "corpus token assignments are not update-aligned"
                )
            expected_bytes = (
                assignment["token_end"] - assignment["token_start"]
            ) * 2
            record = record_by_path[relative]
            if record["bytes"] != expected_bytes:
                raise CorpusContractError(
                    f"corpus token shard byte count drift: {relative}"
                )
            size, digest, _ = _stream_file(
                root_fd, relative, consumer=token_stream.update
            )
            if size != expected_bytes or digest != record["sha256"]:
                raise CorpusContractError(
                    f"corpus token artifact digest drift: {relative}"
                )
            verified.append(PinnedCorpusFile(root / relative, digest))
        if token_stream.hexdigest() != receipt["packed_stream_sha256"]:
            raise CorpusContractError("corpus packed token stream digest drift")

        for sidecar_set, records in parsed_sidecars:
            name = sidecar_set["name"]
            expected_paths = [
                f"sidecars/{name}/shard-{index:05d}-of-{shard_count:05d}.bin"
                for index in range(shard_count)
            ]
            if [record["path"] for record in records] != expected_paths:
                raise CorpusContractError(
                    f"corpus sidecar artifacts are partial or reordered: {name}"
                )
            stream = hashlib.sha256()
            logical_position = 0
            for assignment, relative, record in zip(
                assignments, expected_paths, records, strict=True
            ):
                expected_bytes = (
                    assignment["token_end"] - assignment["token_start"]
                )
                if record["bytes"] != expected_bytes:
                    raise CorpusContractError(
                        f"corpus sidecar shard byte count drift: {relative}"
                    )

                def validate_chunk(chunk: bytes) -> None:
                    nonlocal logical_position
                    if any(value not in (0, 1) for value in chunk):
                        raise CorpusContractError(
                            f"corpus sidecar contains non-binary values: {name}"
                        )
                    logical_chunk = chunk[
                        : max(0, min(len(chunk), logical_tokens - logical_position))
                    ]
                    if name == "dense_target_weights" and any(
                        value != 1 for value in logical_chunk
                    ):
                        raise CorpusContractError(
                            "dense target-weight sidecar contains zero weights"
                        )
                    padding_chunk = chunk[len(logical_chunk) :]
                    if any(padding_chunk):
                        raise CorpusContractError(
                            f"corpus sidecar padding is nonzero: {name}"
                        )
                    logical_position += len(chunk)
                    stream.update(chunk)

                size, digest, _ = _stream_file(
                    root_fd, relative, consumer=validate_chunk
                )
                if size != expected_bytes or digest != record["sha256"]:
                    raise CorpusContractError(
                        f"corpus sidecar artifact digest drift: {relative}"
                    )
                verified.append(PinnedCorpusFile(root / relative, digest))
            if stream.hexdigest() != sidecar_set["stream_sha256"]:
                raise CorpusContractError(
                    f"corpus sidecar stream digest drift: {name}"
                )

        verifier = semantic_verifier or _default_semantic_verifier
        try:
            semantic_receipt = verifier(root)
        except CorpusContractError:
            raise
        except Exception as error:
            raise CorpusContractError(
                "canonical Task 4 semantic verifier rejected the corpus"
            ) from error
        if semantic_receipt != receipt:
            raise CorpusContractError(
                "canonical Task 4 semantic verifier returned different evidence"
            )
        return CorpusEvidence(receipt=receipt, files=tuple(verified))
    finally:
        os.close(root_fd)
