#!/usr/bin/env python3
"""Durable version-pinned publication of paired trainer checkpoints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import stat
import subprocess
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_ARMS = ("dense", "split90")
_SIDECARS = {
    "dense": "dense_target_weights",
    "split90": "split90_target_weights",
}
PAIR_CHECKPOINT_MAX_AGE_SECONDS = 1200
PAIR_CHECKPOINT_ATTEMPT_SECONDS = 120
PAIR_CHECKPOINT_TRIGGER_AGE_SECONDS = 1080
PAIR_CHECKPOINT_RETRY_SECONDS = 5
_METADATA_FIELDS = {
    "checkpoint_version",
    "config_fingerprint",
    "data",
    "installed",
    "receipt_type",
    "schema_version",
    "step",
    "world_size",
}
_DATA_FIELDS = {
    "build_id",
    "global_cursor",
    "ordered_stream_sha256",
    "receipt_sha256",
    "sidecar_name",
}
_INSTALLED_FIELDS = {
    "bytes",
    "ctime_ns",
    "device",
    "gid",
    "inode",
    "links",
    "mode",
    "mtime_ns",
    "uid",
}


@dataclass(frozen=True)
class TrainerCheckpointMetadata:
    checkpoint_version: int
    step: int
    world_size: int
    config_fingerprint: str
    data: dict[str, object]
    installed: dict[str, int]


@dataclass(frozen=True)
class VersionedUploadedObject:
    uri: str
    sha256: str
    bytes: int
    version_id: str


@dataclass(frozen=True)
class CheckpointReceiptRef:
    uri: str
    sha256: str
    version_id: str
    bytes: int


@dataclass(frozen=True)
class PublishedCheckpointPair:
    receipt: CheckpointReceiptRef
    checkpoints: tuple[VersionedUploadedObject, VersionedUploadedObject]
    value: dict[str, object]


class VersionedObjectStore(Protocol):
    def put_if_absent(
        self,
        path: Path,
        uri: str,
        *,
        sha256: str,
        byte_count: int,
        metadata: Mapping[str, str],
    ) -> VersionedUploadedObject | None: ...

    def head_exact(
        self,
        uri: str,
        *,
        sha256: str,
        byte_count: int,
        metadata: Mapping[str, str],
        version_id: str | None,
    ) -> VersionedUploadedObject | None: ...


def _default_command_runner(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
):
    return subprocess.run(
        list(argv),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
        or "\\" in uri
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("versioned object URI must be a safe s3:// bucket/key")
    return parsed.netloc, key


def _command_json(result: object, *, fields: set[str]) -> dict[str, object]:
    if (
        getattr(result, "returncode", None) != 0
        or getattr(result, "stderr", None) != ""
        or not isinstance(getattr(result, "stdout", None), str)
    ):
        raise ValueError("AWS command failed")
    try:
        value = json.loads(
            result.stdout,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"AWS output contains non-finite {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("AWS command output is not JSON") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("AWS command output fields do not match")
    return value


class S3VersionedObjectStore:
    """AWS CLI store requiring checksum, metadata, length, and object version."""

    def __init__(
        self,
        *,
        region: str,
        environment: Mapping[str, str],
        runner: Callable[
            [Sequence[str], Mapping[str, str], float], object
        ] = _default_command_runner,
        timeout_seconds: float = PAIR_CHECKPOINT_ATTEMPT_SECONDS,
    ) -> None:
        if not isinstance(region, str) or not region:
            raise ValueError("AWS region must be non-empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("AWS object timeout must be positive")
        self.region = region
        self.environment = dict(environment)
        self.runner = runner
        self.timeout_seconds = float(timeout_seconds)

    def _argv(self, *arguments: str, query: str) -> list[str]:
        return [
            "aws",
            "--no-cli-pager",
            "--region",
            self.region,
            *arguments,
            "--output",
            "json",
            "--query",
            query,
        ]

    @staticmethod
    def _metadata_argument(metadata: Mapping[str, str]) -> str:
        if (
            not metadata
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                or any(character in key + value for character in ",=\n\r\x00")
                for key, value in metadata.items()
            )
        ):
            raise ValueError("S3 object metadata is invalid")
        return ",".join(
            f"{key}={metadata[key]}" for key in sorted(metadata)
        )

    def put_if_absent(
        self,
        path: Path,
        uri: str,
        *,
        sha256: str,
        byte_count: int,
        metadata: Mapping[str, str],
    ) -> VersionedUploadedObject | None:
        bucket, key = _s3_parts(uri)
        if (
            not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            raise ValueError("S3 object identity is invalid")
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        argv = self._argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(path),
            "--content-length",
            str(byte_count),
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            self._metadata_argument(metadata),
            "--if-none-match",
            "*",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        result = self.runner(argv, self.environment, self.timeout_seconds)
        if getattr(result, "returncode", None) != 0:
            return None
        root = _command_json(result, fields={"object"})
        row = root["object"]
        if (
            not isinstance(row, dict)
            or set(row) != {"checksum_sha256", "version_id"}
            or row["checksum_sha256"] != checksum
            or not isinstance(row["version_id"], str)
            or row["version_id"] in {"", "null"}
        ):
            raise ValueError("S3 PUT response identity is invalid")
        return VersionedUploadedObject(
            uri=uri,
            sha256=sha256,
            bytes=byte_count,
            version_id=row["version_id"],
        )

    def head_exact(
        self,
        uri: str,
        *,
        sha256: str,
        byte_count: int,
        metadata: Mapping[str, str],
        version_id: str | None,
    ) -> VersionedUploadedObject | None:
        bucket, key = _s3_parts(uri)
        arguments = [
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
        ]
        if version_id is not None:
            if (
                not isinstance(version_id, str)
                or version_id in {"", "null"}
            ):
                return None
            arguments.extend(["--version-id", version_id])
        arguments.extend(["--checksum-mode", "ENABLED"])
        result = self.runner(
            self._argv(
                *arguments,
                query=(
                    "{object:{checksum_sha256:ChecksumSHA256,"
                    "content_length:ContentLength,metadata:Metadata,"
                    "version_id:VersionId}}"
                ),
            ),
            self.environment,
            self.timeout_seconds,
        )
        if getattr(result, "returncode", None) != 0:
            return None
        root = _command_json(result, fields={"object"})
        row = root["object"]
        checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "checksum_sha256",
                "content_length",
                "metadata",
                "version_id",
            }
            or row["checksum_sha256"] != checksum
            or type(row["content_length"]) is not int
            or row["content_length"] != byte_count
            or row["metadata"] != dict(metadata)
            or not isinstance(row["version_id"], str)
            or row["version_id"] in {"", "null"}
            or (
                version_id is not None
                and row["version_id"] != version_id
            )
        ):
            return None
        return VersionedUploadedObject(
            uri=uri,
            sha256=sha256,
            bytes=byte_count,
            version_id=row["version_id"],
        )


class PollableCheckpointAttempt(Protocol):
    def poll(self) -> tuple[bool, PublishedCheckpointPair | None]: ...

    def cancel(self) -> None: ...


class CheckpointStaleError(RuntimeError):
    """The paired durability window expired without a complete receipt."""


class ForkedCheckpointMirrorAttempt:
    """Pollable, killable process boundary around one synchronous publisher."""

    def __init__(
        self,
        publish: Callable[[], PublishedCheckpointPair],
    ) -> None:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                message = (True, publish())
            except BaseException as error:
                message = (False, f"{type(error).__name__}: {error}")
            payload = pickle.dumps(message, protocol=5)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(write_fd, view)
                    view = view[written:]
            finally:
                os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        self._pid: int | None = pid
        self._read_fd: int | None = read_fd

    def poll(self) -> tuple[bool, PublishedCheckpointPair | None]:
        if self._pid is None:
            raise RuntimeError("checkpoint mirror attempt was already consumed")
        waited, _status = os.waitpid(self._pid, os.WNOHANG)
        if waited == 0:
            return False, None
        chunks = []
        assert self._read_fd is not None
        while True:
            chunk = os.read(self._read_fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(self._read_fd)
        self._read_fd = None
        self._pid = None
        try:
            ok, value = pickle.loads(b"".join(chunks))
        except Exception:
            return True, None
        if not ok or not isinstance(value, PublishedCheckpointPair):
            return True, None
        return True, value

    def cancel(self) -> None:
        if self._pid is not None:
            try:
                os.kill(self._pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(self._pid, 0)
            except ChildProcessError:
                pass
            self._pid = None
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None


class CheckpointMirrorScheduler:
    """Nonblocking monotonic scheduler for killable mirror attempts."""

    def __init__(
        self,
        start_attempt: Callable[
            [CheckpointMirrorRequest], PollableCheckpointAttempt
        ],
        *,
        started_at: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._start_attempt = start_attempt
        self._monotonic = monotonic
        self._fresh_at = (
            monotonic() if started_at is None else float(started_at)
        )
        self._next_retry_at = (
            self._fresh_at + PAIR_CHECKPOINT_TRIGGER_AGE_SECONDS
        )
        self._attempt_started_at: float | None = None
        self._attempt: PollableCheckpointAttempt | None = None
        self._latest: PublishedCheckpointPair | None = None

    @property
    def latest(self) -> PublishedCheckpointPair | None:
        return self._latest

    @property
    def active(self) -> bool:
        return self._attempt is not None

    @property
    def fresh_at(self) -> float:
        return self._fresh_at

    def cancel_active(self) -> None:
        if self._attempt is not None:
            self._attempt.cancel()
            self._attempt = None
            self._attempt_started_at = None

    def maybe_start(
        self,
        request: CheckpointMirrorRequest,
        *,
        now: float | None = None,
        immediate: bool = False,
    ) -> bool:
        current = self._monotonic() if now is None else float(now)
        if self._attempt is not None:
            return False
        if (
            not immediate
            and (
                current
                < self._fresh_at + PAIR_CHECKPOINT_TRIGGER_AGE_SECONDS
                or current < self._next_retry_at
            )
        ):
            return False
        if current >= self._fresh_at + PAIR_CHECKPOINT_MAX_AGE_SECONDS:
            raise CheckpointStaleError(
                "CHECKPOINT_STALE: paired durability deadline expired"
            )
        self._attempt = self._start_attempt(request)
        self._attempt_started_at = current
        return True

    def poll(
        self,
        *,
        now: float | None = None,
    ) -> PublishedCheckpointPair | None:
        current = self._monotonic() if now is None else float(now)
        if self._attempt is not None:
            complete, result = self._attempt.poll()
            attempt_started = self._attempt_started_at
            if complete:
                self._attempt = None
                self._attempt_started_at = None
                if result is not None:
                    if not isinstance(result, PublishedCheckpointPair):
                        raise ValueError("checkpoint attempt returned invalid result")
                    self._latest = result
                    self._fresh_at = current
                    self._next_retry_at = (
                        current + PAIR_CHECKPOINT_TRIGGER_AGE_SECONDS
                    )
                    return result
                self._next_retry_at = (
                    current + PAIR_CHECKPOINT_RETRY_SECONDS
                )
            elif (
                attempt_started is not None
                and current - attempt_started
                >= PAIR_CHECKPOINT_ATTEMPT_SECONDS
            ):
                self._attempt.cancel()
                self._attempt = None
                self._attempt_started_at = None
                self._next_retry_at = (
                    current + PAIR_CHECKPOINT_RETRY_SECONDS
                )
        if current >= self._fresh_at + PAIR_CHECKPOINT_MAX_AGE_SECONDS:
            if self._attempt is not None:
                self._attempt.cancel()
                self._attempt = None
                self._attempt_started_at = None
            raise CheckpointStaleError(
                "CHECKPOINT_STALE: paired durability deadline expired"
            )
        return None


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC RFC3339")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"{label} must be canonical UTC RFC3339") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{label} must be canonical UTC RFC3339")
    return parsed


def _s3_root(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("S3 root must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise ValueError("S3 root must be a safe s3:// URI")
    return value.rstrip("/")


@dataclass(frozen=True)
class CheckpointMirrorRequest:
    seed: int
    reason: str
    request_id: str
    requested_at: str
    deadline_at: str
    instance_id: str
    boot_id: str
    profile_sha256: str
    environment_receipt_sha256: str
    release_sha256: str
    release_receipt_sha256: str
    run_manifest_sha256: str
    dataset_receipt_sha256: str
    dataset_build_id: str
    ordered_stream_sha256: str
    source_commit: str
    source_tree: str
    rank_zero_pids: Mapping[str, int]
    checkpoint_paths: Mapping[str, Path]
    config_sha256: Mapping[str, str]
    run_ids: Mapping[str, str]
    s3_root: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in range(10):
            raise ValueError("checkpoint seed must be an exact integer from 0 to 9")
        if self.reason not in {"periodic", "interruption"}:
            raise ValueError("checkpoint reason is invalid")
        if (
            not isinstance(self.request_id, str)
            or _REQUEST_ID_RE.fullmatch(self.request_id) is None
        ):
            raise ValueError("checkpoint request ID must be 32 lowercase hex")
        requested = _parse_utc(self.requested_at, label="requested_at")
        deadline = _parse_utc(self.deadline_at, label="deadline_at")
        if (deadline - requested).total_seconds() != PAIR_CHECKPOINT_MAX_AGE_SECONDS:
            raise ValueError("checkpoint freshness window must be exactly 1200 seconds")
        if (
            not isinstance(self.instance_id, str)
            or not self.instance_id
            or not isinstance(self.boot_id, str)
            or not self.boot_id
        ):
            raise ValueError("checkpoint instance and boot identities are required")
        for label, digest in (
            ("profile", self.profile_sha256),
            ("environment receipt", self.environment_receipt_sha256),
            ("release", self.release_sha256),
            ("release receipt", self.release_receipt_sha256),
            ("run manifest", self.run_manifest_sha256),
            ("dataset receipt", self.dataset_receipt_sha256),
            ("dataset build", self.dataset_build_id),
            ("ordered stream", self.ordered_stream_sha256),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 must be lowercase hex")
        for label, object_id in (
            ("source commit", self.source_commit),
            ("source tree", self.source_tree),
        ):
            if not isinstance(object_id, str) or _COMMIT_RE.fullmatch(object_id) is None:
                raise ValueError(f"{label} must be 40 lowercase hex")
        for label, mapping in (
            ("rank-zero PID", self.rank_zero_pids),
            ("checkpoint path", self.checkpoint_paths),
            ("config SHA-256", self.config_sha256),
            ("run ID", self.run_ids),
        ):
            if set(mapping) != set(_ARMS):
                raise ValueError(f"{label} mapping must contain the paired arms")
        if any(
            type(pid) is not int or pid <= 1
            for pid in self.rank_zero_pids.values()
        ):
            raise ValueError("rank-zero PIDs must be positive exact integers")
        if any(
            not isinstance(path, Path) for path in self.checkpoint_paths.values()
        ):
            raise ValueError("checkpoint paths must be pathlib Paths")
        if any(
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            for digest in self.config_sha256.values()
        ):
            raise ValueError("config SHA-256 values must be lowercase hex")
        if any(
            not isinstance(run_id, str) or not run_id
            for run_id in self.run_ids.values()
        ):
            raise ValueError("run IDs must be non-empty strings")
        object.__setattr__(self, "s3_root", _s3_root(self.s3_root))


@dataclass(frozen=True)
class _StagedCheckpoint:
    arm: str
    path: Path
    sha256: str
    bytes: int
    metadata: TrainerCheckpointMetadata


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"trainer checkpoint metadata repeats field: {key}")
        value[key] = item
    return value


def _read_regular(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"{label} is symlinked or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a singly linked regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns)
            or len(payload) != before.st_size
        ):
            raise ValueError(f"{label} changed while being read")
        return payload, before
    finally:
        os.close(descriptor)


def _stat_regular(path: Path, *, label: str) -> os.stat_result:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"{label} is symlinked or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{label} must be a singly linked regular file")
        return metadata
    finally:
        os.close(descriptor)


def _installed_identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "bytes": metadata.st_size,
        "ctime_ns": metadata.st_ctime_ns,
        "device": metadata.st_dev,
        "gid": metadata.st_gid,
        "inode": metadata.st_ino,
        "links": metadata.st_nlink,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "uid": metadata.st_uid,
    }


def read_trainer_checkpoint_metadata(
    checkpoint_path: str | Path,
    *,
    allow_legacy_absent: bool = False,
) -> TrainerCheckpointMetadata | None:
    """Read adjacent trainer metadata, allowing absence only when explicit."""

    if type(allow_legacy_absent) is not bool:
        raise ValueError("allow_legacy_absent must be boolean")
    checkpoint = Path(checkpoint_path)
    metadata_path = checkpoint.with_name("ckpt.meta.json")
    if not os.path.lexists(metadata_path):
        if allow_legacy_absent:
            _stat_regular(checkpoint, label="legacy trainer checkpoint")
            return None
        raise ValueError("trainer checkpoint metadata is missing")
    try:
        payload, _metadata_stat = _read_regular(
            metadata_path,
            label="trainer checkpoint metadata",
        )
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(
                    "trainer checkpoint metadata contains "
                    f"non-finite {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trainer checkpoint metadata is not canonical JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != _METADATA_FIELDS
        or (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        != payload
    ):
        raise ValueError("trainer checkpoint metadata fields are invalid")
    data = value["data"]
    installed = value["installed"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["receipt_type"] != "memorysplit-trainer-checkpoint-v1"
        or type(value["checkpoint_version"]) is not int
        or value["checkpoint_version"] != 3
        or type(value["step"]) is not int
        or value["step"] < 0
        or type(value["world_size"]) is not int
        or value["world_size"] <= 0
        or not isinstance(value["config_fingerprint"], str)
        or _SHA256_RE.fullmatch(value["config_fingerprint"]) is None
        or not isinstance(data, dict)
        or set(data) != _DATA_FIELDS
        or type(data["global_cursor"]) is not int
        or data["global_cursor"] < 0
        or any(
            item is not None
            and (
                not isinstance(item, str)
                or _SHA256_RE.fullmatch(item) is None
            )
            for item in (
                data["receipt_sha256"],
                data["build_id"],
                data["ordered_stream_sha256"],
            )
        )
        or (
            data["sidecar_name"] is not None
            and (
                not isinstance(data["sidecar_name"], str)
                or data["sidecar_name"]
                not in {
                    "dense_target_weights",
                    "split90_target_weights",
                }
            )
        )
        or not isinstance(installed, dict)
        or set(installed) != _INSTALLED_FIELDS
        or any(type(item) is not int for item in installed.values())
        or installed["bytes"] <= 0
        or installed["links"] != 1
        or not stat.S_ISREG(installed["mode"])
    ):
        raise ValueError("trainer checkpoint metadata values are invalid")
    try:
        checkpoint_stat = _stat_regular(
            checkpoint,
            label="trainer checkpoint",
        )
    except FileNotFoundError as error:
        raise ValueError("trainer checkpoint is missing") from error
    if installed != _installed_identity(checkpoint_stat):
        raise ValueError(
            "trainer checkpoint metadata does not match installed checkpoint"
        )
    return TrainerCheckpointMetadata(
        checkpoint_version=value["checkpoint_version"],
        step=value["step"],
        world_size=value["world_size"],
        config_fingerprint=value["config_fingerprint"],
        data=dict(data),
        installed=dict(installed),
    )


def _metadata_generation(checkpoint: Path) -> tuple[int, int, int, int, int] | None:
    metadata_path = checkpoint.with_name("ckpt.meta.json")
    try:
        metadata = metadata_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if (
        metadata_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError("trainer checkpoint metadata is symlinked or unsafe")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stage_generation(
    request: CheckpointMirrorRequest,
    arm: str,
    baseline: tuple[int, int, int, int, int] | None,
    staging: Path,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> _StagedCheckpoint:
    checkpoint = request.checkpoint_paths[arm]
    while monotonic() < deadline:
        generation = _metadata_generation(checkpoint)
        if generation is None or generation == baseline:
            sleep(min(0.05, max(0.0, deadline - monotonic())))
            continue
        metadata = read_trainer_checkpoint_metadata(checkpoint)
        assert metadata is not None
        data = metadata.data
        if (
            metadata.world_size != 4
            or not 0 <= metadata.step <= 13_582
            or data["receipt_sha256"] != request.dataset_receipt_sha256
            or data["build_id"] != request.dataset_build_id
            or data["ordered_stream_sha256"] != request.ordered_stream_sha256
            or data["global_cursor"] != metadata.step * 524_288
            or data["sidecar_name"] != _SIDECARS[arm]
        ):
            raise ValueError(f"{arm} trainer checkpoint metadata is incompatible")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(checkpoint, flags)
        temporary = staging / f".{arm}.stage.tmp"
        try:
            before = os.fstat(descriptor)
            if _installed_identity(before) != metadata.installed:
                raise ValueError(
                    f"{arm} trainer checkpoint generation changed before staging"
                )
            digest = hashlib.sha256()
            copied = 0
            with temporary.open("xb") as output:
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(descriptor)
            if (
                _installed_identity(after) != metadata.installed
                or copied != metadata.installed["bytes"]
            ):
                raise ValueError(f"{arm} trainer checkpoint changed while staging")
            sha256 = digest.hexdigest()
            pinned = staging / f"{arm}-{sha256}.pt"
            os.chmod(temporary, 0o400)
            os.replace(temporary, pinned)
            directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return _StagedCheckpoint(
                arm=arm,
                path=pinned,
                sha256=sha256,
                bytes=copied,
                metadata=metadata,
            )
        finally:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    raise TimeoutError(f"{arm} did not produce a fresh checkpoint generation")


def _checkpoint_metadata(
    request: CheckpointMirrorRequest,
    staged: _StagedCheckpoint,
) -> dict[str, str]:
    data = staged.metadata.data
    return {
        "arm": staged.arm,
        "checkpoint-version": str(staged.metadata.checkpoint_version),
        "config-fingerprint": staged.metadata.config_fingerprint,
        "config-sha256": request.config_sha256[staged.arm],
        "data-build-id": str(data["build_id"]),
        "data-receipt-sha256": str(data["receipt_sha256"]),
        "global-cursor": str(data["global_cursor"]),
        "ordered-stream-sha256": str(data["ordered_stream_sha256"]),
        "request-id": request.request_id,
        "run-id": request.run_ids[staged.arm],
        "seed": str(request.seed),
        "sha256": staged.sha256,
        "sidecar-name": str(data["sidecar_name"]),
        "step": str(staged.metadata.step),
        "world-size": str(staged.metadata.world_size),
    }


def _publish_exact(
    object_store: VersionedObjectStore,
    path: Path,
    uri: str,
    *,
    sha256: str,
    byte_count: int,
    metadata: Mapping[str, str],
) -> VersionedUploadedObject:
    uploaded = object_store.put_if_absent(
        path,
        uri,
        sha256=sha256,
        byte_count=byte_count,
        metadata=metadata,
    )
    expected_version = None
    if uploaded is not None:
        if (
            uploaded.uri != uri
            or uploaded.sha256 != sha256
            or uploaded.bytes != byte_count
            or not isinstance(uploaded.version_id, str)
            or uploaded.version_id in {"", "null"}
        ):
            raise ValueError("versioned object PUT response is invalid")
        expected_version = uploaded.version_id
    verified = object_store.head_exact(
        uri,
        sha256=sha256,
        byte_count=byte_count,
        metadata=metadata,
        version_id=expected_version,
    )
    if (
        verified is None
        or verified.uri != uri
        or verified.sha256 != sha256
        or verified.bytes != byte_count
        or not isinstance(verified.version_id, str)
        or verified.version_id in {"", "null"}
        or (
            expected_version is not None
            and verified.version_id != expected_version
        )
    ):
        raise ValueError("versioned object HEAD verification failed")
    return verified


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


def publish_paired_checkpoint(
    request: CheckpointMirrorRequest,
    *,
    object_store: VersionedObjectStore,
    signal_process: Callable[[int, int], None] = os.kill,
    staging_root: Path,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    staged_at: Callable[[], str] = lambda: datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
) -> PublishedCheckpointPair:
    """Signal, pin, upload, and atomically commit one complete checkpoint pair."""

    if not isinstance(request, CheckpointMirrorRequest):
        raise ValueError("checkpoint mirror request is invalid")
    baselines: dict[str, tuple[int, int, int, int, int] | None] = {}
    for arm in _ARMS:
        generation = _metadata_generation(request.checkpoint_paths[arm])
        if generation is not None:
            read_trainer_checkpoint_metadata(request.checkpoint_paths[arm])
        baselines[arm] = generation
    signal_errors = []
    for arm in _ARMS:
        try:
            signal_process(request.rank_zero_pids[arm], signal.SIGUSR1)
        except OSError as error:
            signal_errors.append((arm, error))
    if signal_errors:
        raise ValueError("both rank-zero processes must accept SIGUSR1")
    staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix=f"checkpoint-{request.request_id}-",
        dir=staging_root,
    ) as staging_text:
        staging = Path(staging_text)
        deadline = monotonic() + PAIR_CHECKPOINT_ATTEMPT_SECONDS
        staged_by_arm: dict[str, _StagedCheckpoint] = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    _stage_generation,
                    request,
                    arm,
                    baselines[arm],
                    staging,
                    deadline=deadline,
                    monotonic=monotonic,
                    sleep=sleep,
                ): arm
                for arm in _ARMS
            }
            for future in as_completed(futures):
                arm = futures[future]
                staged_by_arm[arm] = future.result()
        if set(staged_by_arm) != set(_ARMS):
            raise ValueError("checkpoint staging did not produce a complete pair")

        uploaded = []
        rows = []
        for arm in _ARMS:
            staged = staged_by_arm[arm]
            uri = (
                f"{request.s3_root}/checkpoints/seed-{request.seed}/{arm}/"
                f"sha256/{staged.sha256}.pt"
            )
            versioned = _publish_exact(
                object_store,
                staged.path,
                uri,
                sha256=staged.sha256,
                byte_count=staged.bytes,
                metadata=_checkpoint_metadata(request, staged),
            )
            uploaded.append(versioned)
            rows.append(
                {
                    "arm": arm,
                    "checkpoint_version": 3,
                    "config_fingerprint": staged.metadata.config_fingerprint,
                    "config_sha256": request.config_sha256[arm],
                    "data": dict(staged.metadata.data),
                    "object": {
                        "bytes": versioned.bytes,
                        "sha256": versioned.sha256,
                        "uri": versioned.uri,
                        "version_id": versioned.version_id,
                    },
                    "run_id": request.run_ids[arm],
                    "seed": request.seed,
                    "step": staged.metadata.step,
                    "world_size": staged.metadata.world_size,
                }
            )
        staged_timestamp = staged_at()
        staged_time = _parse_utc(staged_timestamp, label="staged_at")
        if not (
            _parse_utc(request.requested_at, label="requested_at")
            <= staged_time
            <= _parse_utc(request.deadline_at, label="deadline_at")
        ):
            raise ValueError("staged checkpoint is outside freshness window")
        receipt = {
            "boot_id": request.boot_id,
            "checkpoints": rows,
            "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
            "dataset_build_id": request.dataset_build_id,
            "dataset_receipt_sha256": request.dataset_receipt_sha256,
            "environment_receipt_sha256": request.environment_receipt_sha256,
            "freshness": {
                "deadline_at": request.deadline_at,
                "max_age_seconds": PAIR_CHECKPOINT_MAX_AGE_SECONDS,
                "requested_at": request.requested_at,
                "staged_at": staged_timestamp,
            },
            "instance_id": request.instance_id,
            "ordered_stream_sha256": request.ordered_stream_sha256,
            "profile_sha256": request.profile_sha256,
            "provider": "aws-p5.48xlarge",
            "reason": request.reason,
            "receipt_type": "memorysplit-aws-paired-checkpoint-v3",
            "release_receipt_sha256": request.release_receipt_sha256,
            "release_sha256": request.release_sha256,
            "request_id": request.request_id,
            "resumable": True,
            "run_manifest_sha256": request.run_manifest_sha256,
            "schema_version": 3,
            "seed": request.seed,
            "source_commit": request.source_commit,
            "source_tree": request.source_tree,
        }
        receipt_bytes = _canonical_json(receipt)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_path = staging / f"{receipt_sha256}.json"
        with receipt_path.open("xb") as handle:
            handle.write(receipt_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        receipt_uri = (
            f"{request.s3_root}/receipts/checkpoints/seed-{request.seed}/"
            f"sha256/{receipt_sha256}.json"
        )
        receipt_object = _publish_exact(
            object_store,
            receipt_path,
            receipt_uri,
            sha256=receipt_sha256,
            byte_count=len(receipt_bytes),
            metadata={
                "receipt-sha256": receipt_sha256,
                "receipt-type": "memorysplit-aws-paired-checkpoint-v3",
                "request-id": request.request_id,
                "seed": str(request.seed),
            },
        )
        return PublishedCheckpointPair(
            receipt=CheckpointReceiptRef(
                uri=receipt_object.uri,
                sha256=receipt_object.sha256,
                version_id=receipt_object.version_id,
                bytes=receipt_object.bytes,
            ),
            checkpoints=(uploaded[0], uploaded[1]),
            value=receipt,
        )
