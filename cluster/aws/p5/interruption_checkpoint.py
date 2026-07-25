#!/usr/bin/env python3
"""Fail-closed paired checkpoint handling for EC2 interruption notices."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROVIDER = "aws-p5.48xlarge"
INSTANCE_TYPE = "p5.48xlarge"
GRES = "gpu:h100:8"
_LEGACY_CANDIDATE_RECEIPT_TYPE = "aws-p5-interruption-candidate"
_LEGACY_INTERRUPTION_RECEIPT_TYPE = "aws-p5-paired-interruption"
_LEGACY_RESUME_COMMIT_PROTOCOL = "aws-p5-resume-commit-v1"
_NEUTRAL_CANDIDATE_RECEIPT_TYPE = "aws-gpu-interruption-candidate"
_NEUTRAL_INTERRUPTION_RECEIPT_TYPE = "aws-gpu-paired-interruption"
_NEUTRAL_RESUME_COMMIT_PROTOCOL = "aws-gpu-resume-commit-v1"
_PROFILE_CONTRACTS = {
    PROVIDER: {
        "instance_type": INSTANCE_TYPE,
        "gres": GRES,
        "assigned_seeds": (1, 2, 3, 4),
        "candidate_receipt_type": _LEGACY_CANDIDATE_RECEIPT_TYPE,
        "interruption_receipt_type": _LEGACY_INTERRUPTION_RECEIPT_TYPE,
        "resume_commit_protocol": _LEGACY_RESUME_COMMIT_PROTOCOL,
    },
    "aws-p5.48xlarge-v3": {
        "instance_type": "p5.48xlarge",
        "gres": "gpu:h100:8",
        "assigned_seeds": tuple(range(10)),
        "candidate_receipt_type": _NEUTRAL_CANDIDATE_RECEIPT_TYPE,
        "interruption_receipt_type": _NEUTRAL_INTERRUPTION_RECEIPT_TYPE,
        "resume_commit_protocol": _NEUTRAL_RESUME_COMMIT_PROTOCOL,
    },
    "aws-p6-b300.48xlarge-v3": {
        "instance_type": "p6-b300.48xlarge",
        "gres": "gpu:b300:8",
        "assigned_seeds": tuple(range(10)),
        "candidate_receipt_type": _NEUTRAL_CANDIDATE_RECEIPT_TYPE,
        "interruption_receipt_type": _NEUTRAL_INTERRUPTION_RECEIPT_TYPE,
        "resume_commit_protocol": _NEUTRAL_RESUME_COMMIT_PROTOCOL,
    },
}
RESUMABLE_EXIT_CODE = 75
NON_RESUMABLE_EXIT_CODE = 74
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_KMS_KEY_ARN_RE = re.compile(
    r"^arn:aws:kms:(?P<region>us-(?:east-1|west-2)):[0-9]{12}:"
    r"key/[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ARMS = ("dense", "split90")
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
_IMDS_ROOT = "http://169.254.169.254/latest"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class UploadedObject:
    uri: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    links: int
    uid: int
    gid: int


@dataclass(frozen=True)
class _PinnedCheckpoint:
    arm: str
    path: Path
    bytes: int
    sha256: str
    identity: _FileIdentity


class VerifiedObjectStore(Protocol):
    def put_verified(
        self,
        path: Path,
        uri: str,
        *,
        expected_sha256: str,
        deadline: float,
        monotonic: Callable[[], float],
        wall_deadline: float | None = None,
        wall_monotonic: Callable[[], float] = time.monotonic,
    ) -> UploadedObject | None: ...


@dataclass(frozen=True)
class InterruptionRequest:
    seed: int
    notice: str
    rank_zero_pids: Mapping[str, int]
    checkpoint_paths: Mapping[str, Path]
    s3_root: str
    receipt_path: Path
    release_sha256: str
    corpus_receipt_sha256: str
    code_commit: str
    config_sha256: Mapping[str, str]
    timeout_seconds: float
    upload_reserve_seconds: float
    provider: str = PROVIDER
    profile_sha256: str | None = None
    instance_type: str = INSTANCE_TYPE
    gres: str = GRES
    assigned_seeds: tuple[int, ...] = (1, 2, 3, 4)
    candidate_receipt_type: str = _LEGACY_CANDIDATE_RECEIPT_TYPE
    interruption_receipt_type: str = _LEGACY_INTERRUPTION_RECEIPT_TYPE
    resume_commit_protocol: str = _LEGACY_RESUME_COMMIT_PROTOCOL
    checkpoint_metadata_paths: Mapping[str, Path] | None = None
    checkpoint_receipt_path: Path | None = None
    run_ids: Mapping[str, str] | None = None
    run_manifest_sha256: str | None = None
    cohort_assignment_sha256: str | None = None
    preregistration_sha256: str | None = None
    hardware_amendment_sha256: str | None = None
    provider_selection_sha256: str | None = None
    sealed_fixture_sha256: str | None = None

    def __post_init__(self) -> None:
        contract = _PROFILE_CONTRACTS.get(self.provider)
        if (
            contract is None
            or self.instance_type != contract["instance_type"]
            or self.gres != contract["gres"]
            or self.assigned_seeds != contract["assigned_seeds"]
            or self.candidate_receipt_type
            != contract["candidate_receipt_type"]
            or self.interruption_receipt_type
            != contract["interruption_receipt_type"]
            or self.resume_commit_protocol
            != contract["resume_commit_protocol"]
        ):
            raise ValueError(
                "interruption request does not match a closed AWS GPU profile"
            )
        if (
            self.profile_sha256 is not None
            and (
                not isinstance(self.profile_sha256, str)
                or _SHA256_RE.fullmatch(self.profile_sha256) is None
            )
        ):
            raise ValueError("interruption profile SHA-256 must be lowercase hex")
        if self.provider != PROVIDER and self.profile_sha256 is None:
            raise ValueError(
                "v3 interruption requests require an explicit profile SHA-256"
            )
        if type(self.seed) is not int or self.seed not in self.assigned_seeds:
            choices = ", ".join(str(seed) for seed in self.assigned_seeds)
            raise ValueError(
                f"interruption seed must be assigned to the profile: {choices}"
            )
        if not isinstance(self.notice, str) or not self.notice:
            raise ValueError("interruption notice must be non-empty")
        for label, mapping in (
            ("rank-zero PID", self.rank_zero_pids),
            ("checkpoint", self.checkpoint_paths),
            ("config SHA-256", self.config_sha256),
        ):
            if set(mapping) != set(_ARMS):
                raise ValueError(f"{label} mapping must contain the paired arms")
        if any(
            type(pid) is not int or pid <= 1
            for pid in self.rank_zero_pids.values()
        ):
            raise ValueError("rank-zero PIDs must be positive process IDs")
        if any(
            not isinstance(path, Path) for path in self.checkpoint_paths.values()
        ):
            raise ValueError("checkpoint paths must be pathlib Paths")
        if not isinstance(self.receipt_path, Path):
            raise ValueError("interruption receipt path must be a pathlib Path")
        for label, digest in (
            ("release", self.release_sha256),
            ("corpus receipt", self.corpus_receipt_sha256),
            *(
                (f"{arm} config", self.config_sha256[arm])
                for arm in _ARMS
            ),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 must be lowercase hex")
        if (
            not isinstance(self.code_commit, str)
            or _COMMIT_RE.fullmatch(self.code_commit) is None
        ):
            raise ValueError("code commit must be 40 lowercase hex characters")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.timeout_seconds > 900
        ):
            raise ValueError("checkpoint timeout must be between 0 and 900 seconds")
        if (
            isinstance(self.upload_reserve_seconds, bool)
            or not isinstance(self.upload_reserve_seconds, (int, float))
            or not math.isfinite(self.upload_reserve_seconds)
            or self.upload_reserve_seconds <= 0
            or self.upload_reserve_seconds >= self.timeout_seconds
        ):
            raise ValueError(
                "upload reserve must be positive and below checkpoint timeout"
            )
        _split_s3_uri(self.s3_root + "/sentinel")
        bridge_values = (
            self.checkpoint_metadata_paths,
            self.checkpoint_receipt_path,
            self.run_ids,
            self.run_manifest_sha256,
            self.cohort_assignment_sha256,
            self.preregistration_sha256,
            self.hardware_amendment_sha256,
            self.provider_selection_sha256,
            self.sealed_fixture_sha256,
        )
        if any(value is not None for value in bridge_values):
            if self.provider == PROVIDER or any(
                value is None for value in bridge_values
            ):
                raise ValueError(
                    "checkpoint receipt bridge requires one complete v3 binding"
                )
            assert self.checkpoint_metadata_paths is not None
            assert self.checkpoint_receipt_path is not None
            assert self.run_ids is not None
            if (
                set(self.checkpoint_metadata_paths) != set(_ARMS)
                or set(self.run_ids) != set(_ARMS)
                or any(
                    not isinstance(path, Path)
                    for path in self.checkpoint_metadata_paths.values()
                )
                or not isinstance(self.checkpoint_receipt_path, Path)
                or any(
                    not isinstance(run_id, str) or not run_id
                    for run_id in self.run_ids.values()
                )
            ):
                raise ValueError("checkpoint receipt bridge pair is invalid")
            for label, digest in (
                ("run manifest", self.run_manifest_sha256),
                ("cohort assignment", self.cohort_assignment_sha256),
                ("preregistration", self.preregistration_sha256),
                ("hardware amendment", self.hardware_amendment_sha256),
                ("provider selection", self.provider_selection_sha256),
                ("sealed fixture", self.sealed_fixture_sha256),
            ):
                if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                    raise ValueError(f"{label} SHA-256 must be lowercase hex")


@dataclass(frozen=True)
class InterruptionResult:
    resumable: bool
    exit_code: int
    receipt_path: Path
    receipt_upload_verified: bool
    checkpoint_receipt_path: Path | None = None
    checkpoint_receipt_sha256: str | None = None
    checkpoint_receipt_uri: str | None = None


def _default_runner(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    key = parsed.path.removeprefix("/")
    if (
        parsed.scheme != "s3"
        or parsed.hostname is None
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or not key
        or "\\" in uri
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("object URI must be an s3:// bucket/key")
    return parsed.hostname, key


def _load_json_output(result: CommandResult) -> dict[str, object] | None:
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


class S3ObjectStore:
    """S3 uploader that verifies SHA-256 through independent object metadata."""

    def __init__(
        self,
        *,
        region: str,
        environment: Mapping[str, str],
        kms_key_id: str | None = None,
        runner: Callable[
            [Sequence[str], Mapping[str, str], float], CommandResult
        ] = _default_runner,
    ) -> None:
        if not isinstance(region, str) or not region:
            raise ValueError("S3 region must be non-empty")
        if kms_key_id is not None:
            kms_match = (
                _KMS_KEY_ARN_RE.fullmatch(kms_key_id)
                if isinstance(kms_key_id, str)
                else None
            )
            if kms_match is None or kms_match.group("region") != region:
                raise ValueError(
                    "S3 KMS key must be an immutable key ARN in the upload region"
                )
        expected_environment = {
            "AWS_REGION",
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
        }
        unsafe = sorted(set(environment) - expected_environment)
        if unsafe or set(environment) != expected_environment:
            raise ValueError(
                "S3 command environment is not the closed instance-role environment: "
                + ", ".join(unsafe)
            )
        if (
            environment.get("AWS_REGION") != region
            or environment.get("PATH") != "/usr/bin:/bin"
            or environment.get("LANG") != "C.UTF-8"
            or environment.get("LC_ALL") != "C.UTF-8"
        ):
            raise ValueError("S3 command environment has unsafe fixed values")
        home = Path(environment["HOME"])
        try:
            home_metadata = home.stat()
        except OSError as error:
            raise ValueError("S3 private HOME is unavailable") from error
        if (
            home.is_symlink()
            or not home.is_dir()
            or stat.S_IMODE(home_metadata.st_mode) != 0o700
            or home_metadata.st_uid != os.geteuid()
            or any(home.iterdir())
        ):
            raise ValueError("S3 private HOME must be owned, mode 0700, and empty")
        self._region = region
        self._kms_key_id = kms_key_id
        self._environment = dict(environment)
        self._runner = runner

    def put_verified(
        self,
        path: Path,
        uri: str,
        *,
        expected_sha256: str,
        deadline: float,
        monotonic: Callable[[], float] = time.monotonic,
        wall_deadline: float | None = None,
        wall_monotonic: Callable[[], float] = time.monotonic,
    ) -> UploadedObject | None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected object SHA-256 must be lowercase hex")
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or monotonic() >= float(deadline)
        ):
            return None
        if wall_deadline is None:
            wall_deadline = wall_monotonic() + max(
                0.0,
                float(deadline) - monotonic(),
            )
        if wall_monotonic() >= wall_deadline:
            return None
        bucket, key = _split_s3_uri(uri)
        hashed = _wait_for_io(
            _ForkIoJob(lambda: _hash_file_sync(path)),
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        )
        if (
            not isinstance(hashed, dict)
            or set(hashed) != {"bytes", "sha256"}
            or type(hashed["bytes"]) is not int
            or not isinstance(hashed["sha256"], str)
        ):
            return None
        digest = hashed["sha256"]
        size = hashed["bytes"]
        if digest != expected_sha256:
            return None
        raw_digest = bytes.fromhex(digest)
        checksum = base64.b64encode(raw_digest).decode("ascii")
        remaining = _remaining_seconds(
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        )
        if remaining <= 0:
            return None
        encryption_argv = (
            [
                "--server-side-encryption",
                "aws:kms",
                "--ssekms-key-id",
                self._kms_key_id,
            ]
            if self._kms_key_id is not None
            else []
        )
        put = None
        try:
            put = self._runner(
                [
                    "aws",
                    "s3api",
                    "put-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--body",
                    str(path),
                    "--checksum-algorithm",
                    "SHA256",
                    "--checksum-sha256",
                    checksum,
                    "--metadata",
                    f"sha256={digest}",
                    "--if-none-match",
                    "*",
                    *encryption_argv,
                    "--region",
                    self._region,
                    "--output",
                    "json",
                    "--no-cli-pager",
                ],
                self._environment,
                remaining,
            )
        except (OSError, subprocess.SubprocessError):
            put = None
        if put is not None and put.returncode == 0:
            put_value = _load_json_output(put)
            if put_value is None or put_value.get("ChecksumSHA256") != checksum:
                return None
        remaining = _remaining_seconds(
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        )
        if remaining <= 0:
            return None
        try:
            head = self._runner(
                [
                    "aws",
                    "s3api",
                    "head-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--checksum-mode",
                    "ENABLED",
                    "--region",
                    self._region,
                    "--output",
                    "json",
                    "--no-cli-pager",
                ],
                self._environment,
                remaining,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        head_value = _load_json_output(head)
        encryption_matches = (
            head_value is not None
            and (
                self._kms_key_id is None
                or (
                    head_value.get("ServerSideEncryption") == "aws:kms"
                    and head_value.get("SSEKMSKeyId") == self._kms_key_id
                )
            )
        )
        if (
            head_value is None
            or head_value.get("ChecksumSHA256") != checksum
            or head_value.get("ContentLength") != size
            or head_value.get("Metadata") != {"sha256": digest}
            or not encryption_matches
            or _remaining_seconds(
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )
            <= 0
        ):
            return None
        return UploadedObject(
            uri=uri,
            sha256=digest,
            bytes=size,
        )


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


def _remaining_seconds(
    *,
    deadline: float,
    monotonic: Callable[[], float],
    wall_deadline: float,
    wall_monotonic: Callable[[], float],
) -> float:
    return min(
        float(deadline) - monotonic(),
        float(wall_deadline) - wall_monotonic(),
    )


class _ForkIoJob:
    """Run potentially blocking local I/O in one killable child process."""

    def __init__(self, worker: Callable[[], object]) -> None:
        if not hasattr(os, "fork"):
            raise RuntimeError("bounded local I/O requires os.fork")
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                try:
                    result = worker()
                    envelope = {"ok": True, "result": result}
                except BaseException:
                    envelope = {"ok": False, "result": None}
                payload = _canonical_json(envelope)
                offset = 0
                while offset < len(payload):
                    offset += os.write(write_fd, payload[offset:])
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        os.set_blocking(read_fd, False)
        self._pid: int | None = pid
        self._read_fd: int | None = read_fd
        self._payload = bytearray()

    def _drain(self) -> None:
        if self._read_fd is None:
            return
        while True:
            try:
                chunk = os.read(self._read_fd, 65536)
            except BlockingIOError:
                return
            if not chunk:
                return
            self._payload.extend(chunk)
            if len(self._payload) > 65536:
                return

    def poll(self) -> tuple[bool, object | None]:
        if self._pid is None:
            return True, None
        self._drain()
        waited, status = os.waitpid(self._pid, os.WNOHANG)
        if waited == 0:
            return False, None
        self._pid = None
        self._drain()
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None
        if (
            not os.WIFEXITED(status)
            or os.WEXITSTATUS(status) != 0
            or len(self._payload) > 65536
        ):
            return True, None
        try:
            envelope = json.loads(bytes(self._payload).decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return True, None
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"ok", "result"}
            or envelope["ok"] is not True
        ):
            return True, None
        return True, envelope["result"]

    def cancel(self) -> None:
        if self._pid is not None:
            try:
                os.kill(self._pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            while True:
                try:
                    os.waitpid(self._pid, 0)
                    break
                except InterruptedError:
                    continue
                except ChildProcessError:
                    break
            self._pid = None
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None


def _wait_for_io(
    job: _ForkIoJob,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    wall_deadline: float,
    wall_monotonic: Callable[[], float],
) -> object | None:
    while True:
        complete, result = job.poll()
        if complete:
            return result
        remaining = _remaining_seconds(
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        )
        if remaining <= 0:
            job.cancel()
            return None
        time.sleep(min(0.01, remaining))


def _identity(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        mode=metadata.st_mode,
        links=metadata.st_nlink,
        uid=metadata.st_uid,
        gid=metadata.st_gid,
    )


def _hash_file_sync(path: Path) -> dict[str, object] | None:
    descriptor = _open_checkpoint(path)
    if descriptor is None:
        return None
    try:
        before = _identity(os.fstat(descriptor))
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = _identity(os.fstat(descriptor))
        if before != after or total != before.size:
            return None
        return {
            "bytes": total,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _open_checkpoint(path: Path) -> int | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"checkpoint path is unsafe: {path}") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"checkpoint must be a singly linked regular file: {path}")
    return descriptor


def _checkpoint_identity(path: Path) -> _FileIdentity | None:
    descriptor = _open_checkpoint(path)
    if descriptor is None:
        return None
    try:
        return _identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)


def _checkpoint_metadata(
    path: Path,
    *,
    arm: str,
    request: InterruptionRequest,
    pinned: _PinnedCheckpoint,
) -> dict[str, object] | None:
    descriptor = _open_checkpoint(path)
    if descriptor is None:
        return None
    try:
        before = os.fstat(descriptor)
        if before.st_size <= 0 or before.st_size > 64 * 1024:
            return None
        payload = b""
        while len(payload) <= before.st_size:
            chunk = os.read(descriptor, before.st_size - len(payload) + 1)
            if not chunk:
                break
            payload += chunk
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
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
    ) or len(payload) != after.st_size:
        return None
    try:
        value = _canonical_payload(payload, label=f"{arm} checkpoint metadata")
    except ValueError:
        return None
    run_ids = request.run_ids
    if (
        set(value) != _CHECKPOINT_METADATA_FIELDS
        or value["schema_version"] != 1
        or value["receipt_type"] != "memorysplit-training-checkpoint-v1"
        or run_ids is None
        or value["run_id"] != run_ids[arm]
        or value["condition"] != arm
        or value["seed"] != request.seed
        or type(value["step"]) is not int
        or value["step"] <= 0
        or type(value["max_steps"]) is not int
        or value["max_steps"] < value["step"]
        or value["world_size"] != 4
        or value["checkpoint_path"] != "ckpt.pt"
        or value["checkpoint_sha256"] != pinned.sha256
        or value["checkpoint_bytes"] != pinned.bytes
        or not isinstance(value["terminal"], bool)
        or not isinstance(value["config_fingerprint"], str)
        or _SHA256_RE.fullmatch(value["config_fingerprint"]) is None
    ):
        return None
    return value


def _generation_changed(
    baseline: _FileIdentity | None,
    current: _FileIdentity,
) -> bool:
    return baseline is None or (
        baseline.device,
        baseline.inode,
    ) != (
        current.device,
        current.inode,
    )


def _stage_checkpoint(
    arm: str,
    source: Path,
    expected: _FileIdentity,
    staging: Path,
) -> _PinnedCheckpoint | None:
    descriptor = _open_checkpoint(source)
    if descriptor is None:
        return None
    temporary = staging / f".{arm}.checkpoint.tmp"
    try:
        before = _identity(os.fstat(descriptor))
        if before != expected or before.size <= 0:
            return None
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
        after = _identity(os.fstat(descriptor))
        if after != before or copied != before.size:
            temporary.unlink(missing_ok=True)
            return None
        checkpoint_digest = digest.hexdigest()
        pinned = staging / f"{arm}-{checkpoint_digest}.pt"
        os.chmod(temporary, 0o400)
        os.rename(temporary, pinned)
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return _PinnedCheckpoint(
            arm=arm,
            path=pinned,
            bytes=copied,
            sha256=checkpoint_digest,
            identity=before,
        )
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _stage_job(
    arm: str,
    source: Path,
    expected: _FileIdentity,
    staging: Path,
) -> _ForkIoJob:
    def worker() -> object:
        pinned = _stage_checkpoint(arm, source, expected, staging)
        if pinned is None:
            return None
        return {
            "arm": pinned.arm,
            "bytes": pinned.bytes,
            "identity": dict(pinned.identity.__dict__),
            "path": str(pinned.path),
            "sha256": pinned.sha256,
        }

    return _ForkIoJob(worker)


def _pinned_result(
    arm: str,
    value: object,
    staging: Path,
) -> _PinnedCheckpoint | None:
    if not isinstance(value, dict) or set(value) != {
        "arm",
        "bytes",
        "identity",
        "path",
        "sha256",
    }:
        return None
    identity_value = value["identity"]
    if (
        value["arm"] != arm
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
        or not isinstance(value["sha256"], str)
        or _SHA256_RE.fullmatch(value["sha256"]) is None
        or not isinstance(identity_value, dict)
        or set(identity_value) != set(_FileIdentity.__dataclass_fields__)
        or any(type(item) is not int for item in identity_value.values())
    ):
        return None
    path = Path(str(value["path"]))
    expected_path = staging / f"{arm}-{value['sha256']}.pt"
    if path != expected_path:
        return None
    return _PinnedCheckpoint(
        arm=arm,
        path=path,
        bytes=value["bytes"],
        sha256=value["sha256"],
        identity=_FileIdentity(**identity_value),
    )


def _stabilize_checkpoints(
    request: InterruptionRequest,
    *,
    baselines: Mapping[str, _FileIdentity | None],
    staging: Path,
    deadline: float,
    monotonic: Callable[[], float],
    wall_deadline: float,
    wall_monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, _PinnedCheckpoint | None]:
    previous: dict[str, _FileIdentity | None] = {arm: None for arm in _ARMS}
    resolved: dict[str, _PinnedCheckpoint | None] = {
        arm: None for arm in _ARMS
    }
    jobs: dict[str, _ForkIoJob] = {}
    try:
        while _remaining_seconds(
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        ) > 0 and any(resolved[arm] is None for arm in _ARMS):
            for arm in tuple(jobs):
                complete, value = jobs[arm].poll()
                if not complete:
                    continue
                del jobs[arm]
                resolved[arm] = _pinned_result(arm, value, staging)
                if resolved[arm] is None:
                    previous[arm] = None
            for arm in _ARMS:
                if resolved[arm] is not None or arm in jobs:
                    continue
                current = _checkpoint_identity(request.checkpoint_paths[arm])
                if (
                    current is None
                    or current.size <= 0
                    or not _generation_changed(baselines[arm], current)
                ):
                    previous[arm] = None
                    continue
                if previous[arm] == current:
                    jobs[arm] = _stage_job(
                        arm,
                        request.checkpoint_paths[arm],
                        current,
                        staging,
                    )
                else:
                    previous[arm] = current
            if all(resolved[arm] is not None for arm in _ARMS):
                break
            remaining = _remaining_seconds(
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )
            if remaining <= 0:
                break
            sleep(min(0.01, remaining))
            # Yield to both killable staging workers even when tests inject a
            # logical clock whose sleep does not block the host scheduler.
            time.sleep(min(0.001, remaining))
        return resolved
    finally:
        for job in jobs.values():
            job.cancel()


def _checkpoint_uri(
    request: InterruptionRequest,
    arm: str,
    digest: str,
) -> str:
    return (
        f"{request.s3_root.rstrip('/')}/checkpoints/seed-{request.seed}/"
        f"{arm}/sha256/{digest}.pt"
    )


def _checkpoint_receipt_uri(
    request: InterruptionRequest,
    digest: str,
) -> str:
    return (
        f"{request.s3_root.rstrip('/')}/checkpoints/seed-{request.seed}/"
        f"receipts/{digest}.json"
    )


def _evidence_uri(request: InterruptionRequest, digest: str) -> str:
    return (
        f"{request.s3_root.rstrip('/')}/receipts/interruption/evidence/"
        f"seed-{request.seed}/sha256/{digest}.json"
    )


def _commit_uri(request: InterruptionRequest, nonce: str) -> str:
    return (
        f"{request.s3_root.rstrip('/')}/receipts/interruption/"
        f"resume-commits/seed-{request.seed}/{nonce}.json"
    )


def _write_immutable_json(
    path: Path,
    value: object,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    wall_deadline: float,
    wall_monotonic: Callable[[], float],
) -> tuple[bytes, str] | None:
    if _remaining_seconds(
        deadline=deadline,
        monotonic=monotonic,
        wall_deadline=wall_deadline,
        wall_monotonic=wall_monotonic,
    ) <= 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable interruption object already exists: {path}")
    payload = _canonical_json(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o400)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if _remaining_seconds(
        deadline=deadline,
        monotonic=monotonic,
        wall_deadline=wall_deadline,
        wall_monotonic=wall_monotonic,
    ) <= 0:
        return None
    return payload, hashlib.sha256(payload).hexdigest()


def _publish_local_handoff(
    source: Path,
    destination: Path,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    wall_deadline: float,
    wall_monotonic: Callable[[], float],
) -> bool:
    if (
        _remaining_seconds(
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        )
        <= 0
        or destination.exists()
        or destination.is_symlink()
    ):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError:
        return False
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if _remaining_seconds(
        deadline=deadline,
        monotonic=monotonic,
        wall_deadline=wall_deadline,
        wall_monotonic=wall_monotonic,
    ) <= 0:
        destination.unlink(missing_ok=True)
        return False
    return True


def _identity_evidence(value: _FileIdentity | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "ctime_ns": value.ctime_ns,
        "device": value.device,
        "gid": value.gid,
        "inode": value.inode,
        "mode": value.mode,
        "mtime_ns": value.mtime_ns,
        "size": value.size,
        "uid": value.uid,
    }


def handle_interruption(
    request: InterruptionRequest,
    *,
    signal_process: Callable[[int, int], None] = os.kill,
    object_store: VerifiedObjectStore,
    monotonic: Callable[[], float] = time.monotonic,
    wall_monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    commit_nonce: Callable[[], str] = lambda: secrets.token_hex(16),
) -> InterruptionResult:
    """Publish immutable evidence and one verified resume-commit handoff."""

    start = monotonic()
    wall_start = wall_monotonic()
    deadline = start + float(request.timeout_seconds)
    wall_deadline = wall_start + float(request.timeout_seconds)
    checkpoint_deadline = deadline - float(request.upload_reserve_seconds)
    checkpoint_wall_deadline = wall_deadline - float(
        request.upload_reserve_seconds
    )
    if os.path.lexists(request.receipt_path):
        raise ValueError("immutable interruption handoff already exists")
    baselines = {
        arm: _checkpoint_identity(request.checkpoint_paths[arm])
        for arm in _ARMS
    }
    signal_errors: dict[str, str] = {}
    request.receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix=".interruption-upload-",
        dir=request.receipt_path.parent,
    ) as staging_text:
        staging = Path(staging_text)
        os.chmod(staging, 0o700)
        for arm in _ARMS:
            try:
                signal_process(request.rank_zero_pids[arm], signal.SIGUSR1)
            except OSError as error:
                signal_errors[arm] = type(error).__name__

        stable = _stabilize_checkpoints(
            request,
            baselines=baselines,
            staging=staging,
            deadline=checkpoint_deadline,
            monotonic=monotonic,
            wall_deadline=checkpoint_wall_deadline,
            wall_monotonic=wall_monotonic,
            sleep=sleep,
        )
        bridge_enabled = request.checkpoint_receipt_path is not None
        bridge_metadata: dict[str, dict[str, object]] = {}
        checkpoint_rows = []
        all_verified = not signal_errors
        for arm in _ARMS:
            pinned = stable[arm]
            uploaded = None
            checkpoint_uri = (
                None
                if pinned is None
                else _checkpoint_uri(request, arm, pinned.sha256)
            )
            if pinned is not None and _remaining_seconds(
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            ) > 0:
                uploaded = object_store.put_verified(
                    pinned.path,
                    checkpoint_uri,
                    expected_sha256=pinned.sha256,
                    deadline=deadline,
                    monotonic=monotonic,
                    wall_deadline=wall_deadline,
                    wall_monotonic=wall_monotonic,
                )
            if (
                pinned is None
                or uploaded is None
                or uploaded.uri != checkpoint_uri
                or uploaded.sha256 != pinned.sha256
                or uploaded.bytes != pinned.bytes
            ):
                all_verified = False
            checkpoint_rows.append(
                {
                    "arm": arm,
                    "baseline": _identity_evidence(baselines[arm]),
                    "bytes": None if pinned is None else pinned.bytes,
                    "config_sha256": request.config_sha256[arm],
                    "generation": (
                        None
                        if pinned is None
                        else _identity_evidence(pinned.identity)
                    ),
                    "object_uri": None if uploaded is None else uploaded.uri,
                    "sha256": (
                        None
                        if uploaded is None
                        else uploaded.sha256
                    ),
                }
            )
        if bridge_enabled:
            assert request.checkpoint_metadata_paths is not None
            for arm in _ARMS:
                pinned = stable[arm]
                metadata = None
                while (
                    pinned is not None
                    and _remaining_seconds(
                        deadline=checkpoint_deadline,
                        monotonic=monotonic,
                        wall_deadline=checkpoint_wall_deadline,
                        wall_monotonic=wall_monotonic,
                    )
                    > 0
                ):
                    metadata = _checkpoint_metadata(
                        request.checkpoint_metadata_paths[arm],
                        arm=arm,
                        request=request,
                        pinned=pinned,
                    )
                    if metadata is not None:
                        break
                    sleep(0.01)
                if metadata is None:
                    all_verified = False
                else:
                    bridge_metadata[arm] = metadata
            if len(
                {
                    metadata["step"]
                    for metadata in bridge_metadata.values()
                }
            ) != 1:
                all_verified = False

        deadline_exhausted = (
            _remaining_seconds(
                deadline=checkpoint_deadline,
                monotonic=monotonic,
                wall_deadline=checkpoint_wall_deadline,
                wall_monotonic=wall_monotonic,
            )
            <= 0
            and any(stable[arm] is None for arm in _ARMS)
        ) or _remaining_seconds(
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        ) <= 0
        legacy_p5 = request.provider == PROVIDER
        candidate = {
            "checkpoints": checkpoint_rows,
            "code_commit": request.code_commit,
            "corpus_receipt_sha256": request.corpus_receipt_sha256,
            "deadline_exhausted": deadline_exhausted,
            "notice": request.notice,
            "paired": True,
            "provider": request.provider,
            "receipt_type": request.candidate_receipt_type,
            "release_sha256": request.release_sha256,
            "schema_version": 3 if legacy_p5 else 4,
            "seed": request.seed,
            "signal_errors": signal_errors,
        }
        if not legacy_p5:
            candidate.update(
                {
                    "gres": request.gres,
                    "instance_type": request.instance_type,
                    "profile_sha256": request.profile_sha256,
                }
            )
        candidate_path = staging / "candidate.json"
        candidate_written = _write_immutable_json(
            candidate_path,
            candidate,
            deadline=deadline,
            monotonic=monotonic,
            wall_deadline=wall_deadline,
            wall_monotonic=wall_monotonic,
        )
        candidate_uploaded = None
        candidate_digest = None
        candidate_uri = None
        if candidate_written is not None:
            _candidate_bytes, candidate_digest = candidate_written
            candidate_uri = _evidence_uri(request, candidate_digest)
            candidate_uploaded = object_store.put_verified(
                candidate_path,
                candidate_uri,
                expected_sha256=candidate_digest,
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )

        checkpoint_receipt_uploaded = None
        checkpoint_receipt_digest = None
        checkpoint_receipt_uri = None
        if bridge_enabled and all_verified:
            assert request.checkpoint_receipt_path is not None
            assert request.run_ids is not None
            local_checkpoints_published = all(
                stable[arm] is not None
                and _publish_local_handoff(
                    stable[arm].path,
                    request.checkpoint_receipt_path.parent / f"{arm}.pt",
                    deadline=deadline,
                    monotonic=monotonic,
                    wall_deadline=wall_deadline,
                    wall_monotonic=wall_monotonic,
                )
                for arm in _ARMS
            )
            if local_checkpoints_published:
                receipt_value = {
                    "schema_version": 3,
                    "provider": request.provider,
                    "release_sha256": request.release_sha256,
                    "run_manifest_sha256": request.run_manifest_sha256,
                    "dataset_sha256": request.corpus_receipt_sha256,
                    "source_commit": request.code_commit,
                    "cohort_assignment_sha256": (
                        request.cohort_assignment_sha256
                    ),
                    "preregistration_sha256": request.preregistration_sha256,
                    "hardware_amendment_sha256": (
                        request.hardware_amendment_sha256
                    ),
                    "provider_selection_sha256": (
                        request.provider_selection_sha256
                    ),
                    "profile_sha256": request.profile_sha256,
                    "sealed_fixture_sha256": (
                        request.sealed_fixture_sha256
                    ),
                    "checkpoints": [
                        {
                            "run_id": request.run_ids[arm],
                            "arm": arm,
                            "seed": request.seed,
                            "path": f"{arm}.pt",
                            "sha256": stable[arm].sha256,
                            "config_sha256": request.config_sha256[arm],
                            "dataset_sha256": request.corpus_receipt_sha256,
                            "source_commit": request.code_commit,
                            "step": bridge_metadata[arm]["step"],
                            "world_size": 4,
                        }
                        for arm in _ARMS
                    ],
                }
                receipt_written = _write_immutable_json(
                    request.checkpoint_receipt_path,
                    receipt_value,
                    deadline=deadline,
                    monotonic=monotonic,
                    wall_deadline=wall_deadline,
                    wall_monotonic=wall_monotonic,
                )
                if receipt_written is not None:
                    receipt_bytes, checkpoint_receipt_digest = receipt_written
                    checkpoint_receipt_uri = _checkpoint_receipt_uri(
                        request,
                        checkpoint_receipt_digest,
                    )
                    checkpoint_receipt_uploaded = object_store.put_verified(
                        request.checkpoint_receipt_path,
                        checkpoint_receipt_uri,
                        expected_sha256=checkpoint_receipt_digest,
                        deadline=deadline,
                        monotonic=monotonic,
                        wall_deadline=wall_deadline,
                        wall_monotonic=wall_monotonic,
                    )
                    if (
                        checkpoint_receipt_uploaded is None
                        or checkpoint_receipt_uploaded.uri
                        != checkpoint_receipt_uri
                        or checkpoint_receipt_uploaded.sha256
                        != checkpoint_receipt_digest
                        or checkpoint_receipt_uploaded.bytes
                        != len(receipt_bytes)
                    ):
                        all_verified = False
                else:
                    all_verified = False
            else:
                all_verified = False

        marker_path = staging / "commit.json"
        marker_uploaded = None
        marker_digest = None
        marker_uri = None
        marker_bytes = None
        if (
            all_verified
            and candidate_written is not None
            and candidate_uploaded is not None
            and candidate_uploaded.uri == candidate_uri
            and candidate_uploaded.sha256 == candidate_digest
            and candidate_uploaded.bytes == len(_candidate_bytes)
            and (
                not bridge_enabled
                or (
                    checkpoint_receipt_uploaded is not None
                    and checkpoint_receipt_uploaded.uri
                    == checkpoint_receipt_uri
                    and checkpoint_receipt_uploaded.sha256
                    == checkpoint_receipt_digest
                )
            )
            and _remaining_seconds(
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )
            > 0
        ):
            nonce = commit_nonce()
            if (
                not isinstance(nonce, str)
                or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
            ):
                raise ValueError("resume commit nonce must be 32 lowercase hex")
            marker = {
                "candidate": {
                    "bytes": candidate_uploaded.bytes,
                    "sha256": candidate_uploaded.sha256,
                    "uri": candidate_uploaded.uri,
                },
                "commit_id": nonce,
                "protocol": request.resume_commit_protocol,
                "receipt_type": request.interruption_receipt_type,
                "schema_version": 1 if legacy_p5 else 2,
                "seed": request.seed,
            }
            if not legacy_p5:
                marker.update(
                    {
                        "gres": request.gres,
                        "instance_type": request.instance_type,
                        "profile_sha256": request.profile_sha256,
                        "provider": request.provider,
                    }
                )
            marker_written = _write_immutable_json(
                marker_path,
                marker,
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )
            if marker_written is not None:
                marker_bytes, marker_digest = marker_written
                marker_uri = _commit_uri(request, nonce)
                marker_uploaded = object_store.put_verified(
                    marker_path,
                    marker_uri,
                    expected_sha256=marker_digest,
                    deadline=deadline,
                    monotonic=monotonic,
                    wall_deadline=wall_deadline,
                    wall_monotonic=wall_monotonic,
                )

        resumable = (
            marker_uploaded is not None
            and marker_uploaded.uri == marker_uri
            and marker_uploaded.sha256 == marker_digest
            and marker_bytes is not None
            and marker_uploaded.bytes == len(marker_bytes)
            and marker_digest
            == hashlib.sha256(marker_bytes).hexdigest()
            and (
                not bridge_enabled
                or checkpoint_receipt_uploaded is not None
            )
            and _publish_local_handoff(
                marker_path,
                request.receipt_path,
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )
        )
        if not resumable and candidate_written is not None:
            _publish_local_handoff(
                candidate_path,
                request.receipt_path,
                deadline=deadline,
                monotonic=monotonic,
                wall_deadline=wall_deadline,
                wall_monotonic=wall_monotonic,
            )

    return InterruptionResult(
        resumable=resumable,
        exit_code=(
            RESUMABLE_EXIT_CODE if resumable else NON_RESUMABLE_EXIT_CODE
        ),
        receipt_path=request.receipt_path,
        receipt_upload_verified=resumable,
        checkpoint_receipt_path=(
            request.checkpoint_receipt_path
            if resumable and bridge_enabled
            else None
        ),
        checkpoint_receipt_sha256=(
            checkpoint_receipt_digest
            if resumable and bridge_enabled
            else None
        ),
        checkpoint_receipt_uri=(
            checkpoint_receipt_uri
            if resumable and bridge_enabled
            else None
        ),
    )


def _canonical_payload(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def verify_resume_commit(
    *,
    candidate_bytes: bytes,
    marker_bytes: bytes,
    checkpoint_objects: Mapping[str, bytes],
    expected_provider: str = PROVIDER,
    expected_profile_sha256: str | None = None,
    expected_instance_type: str | None = None,
    expected_gres: str | None = None,
    expected_candidate_receipt_type: str = _LEGACY_CANDIDATE_RECEIPT_TYPE,
    expected_interruption_receipt_type: str = _LEGACY_INTERRUPTION_RECEIPT_TYPE,
    expected_resume_commit_protocol: str = _LEGACY_RESUME_COMMIT_PROTOCOL,
) -> bool:
    """Derive resume eligibility from immutable fetched bytes only."""

    contract = _PROFILE_CONTRACTS.get(expected_provider)
    if contract is None:
        raise ValueError("resume verifier provider is not a closed AWS GPU profile")
    instance_type = expected_instance_type or str(contract["instance_type"])
    gres = expected_gres or str(contract["gres"])
    if (
        instance_type != contract["instance_type"]
        or gres != contract["gres"]
        or expected_candidate_receipt_type
        != contract["candidate_receipt_type"]
        or expected_interruption_receipt_type
        != contract["interruption_receipt_type"]
        or expected_resume_commit_protocol
        != contract["resume_commit_protocol"]
        or (
            expected_profile_sha256 is not None
            and (
                not isinstance(expected_profile_sha256, str)
                or _SHA256_RE.fullmatch(expected_profile_sha256) is None
            )
        )
        or (
            expected_provider != PROVIDER
            and expected_profile_sha256 is None
        )
    ):
        raise ValueError("resume verifier profile binding is invalid")
    legacy_p5 = expected_provider == PROVIDER
    candidate = _canonical_payload(candidate_bytes, label="resume candidate")
    marker = _canonical_payload(marker_bytes, label="resume commit marker")
    marker_fields = {
        "candidate",
        "commit_id",
        "protocol",
        "receipt_type",
        "schema_version",
        "seed",
    }
    if not legacy_p5:
        marker_fields |= {
            "provider",
            "profile_sha256",
            "instance_type",
            "gres",
        }
    if set(marker) != marker_fields:
        raise ValueError("resume commit marker fields do not match")
    if (
        marker["protocol"] != expected_resume_commit_protocol
        or marker["receipt_type"] != expected_interruption_receipt_type
        or marker["schema_version"] != (1 if legacy_p5 else 2)
        or not isinstance(marker["commit_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", marker["commit_id"]) is None
        or (
            not legacy_p5
            and (
                marker["provider"] != expected_provider
                or marker["profile_sha256"] != expected_profile_sha256
                or marker["instance_type"] != instance_type
                or marker["gres"] != gres
            )
        )
    ):
        raise ValueError("resume commit marker identity does not match")
    candidate_ref = marker["candidate"]
    candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()
    if (
        not isinstance(candidate_ref, dict)
        or set(candidate_ref) != {"bytes", "sha256", "uri"}
        or candidate_ref["bytes"] != len(candidate_bytes)
        or candidate_ref["sha256"] != candidate_digest
        or not isinstance(candidate_ref["uri"], str)
        or not candidate_ref["uri"].endswith(
            f"/evidence/seed-{candidate.get('seed')}/sha256/"
            f"{candidate_digest}.json"
        )
    ):
        raise ValueError("resume candidate hash binding does not match")
    candidate_fields = {
        "checkpoints",
        "code_commit",
        "corpus_receipt_sha256",
        "deadline_exhausted",
        "notice",
        "paired",
        "provider",
        "receipt_type",
        "release_sha256",
        "schema_version",
        "seed",
        "signal_errors",
    }
    if not legacy_p5:
        candidate_fields |= {"profile_sha256", "instance_type", "gres"}
    if set(candidate) != candidate_fields:
        raise ValueError("resume candidate fields do not match")
    if (
        candidate["receipt_type"] != expected_candidate_receipt_type
        or candidate["provider"] != expected_provider
        or candidate["schema_version"] != (3 if legacy_p5 else 4)
        or candidate["paired"] is not True
        or candidate["deadline_exhausted"] is not False
        or candidate["signal_errors"] != {}
        or marker["seed"] != candidate["seed"]
        or (
            not legacy_p5
            and (
                candidate["profile_sha256"] != expected_profile_sha256
                or candidate["instance_type"] != instance_type
                or candidate["gres"] != gres
            )
        )
    ):
        raise ValueError("resume candidate is not eligible")
    if (
        type(candidate["seed"]) is not int
        or candidate["seed"] not in contract["assigned_seeds"]
        or not isinstance(candidate["notice"], str)
        or not candidate["notice"]
        or not isinstance(candidate["code_commit"], str)
        or _COMMIT_RE.fullmatch(candidate["code_commit"]) is None
        or not isinstance(candidate["release_sha256"], str)
        or _SHA256_RE.fullmatch(candidate["release_sha256"]) is None
        or not isinstance(candidate["corpus_receipt_sha256"], str)
        or _SHA256_RE.fullmatch(candidate["corpus_receipt_sha256"]) is None
    ):
        raise ValueError("resume candidate identity binding is invalid")
    rows = candidate["checkpoints"]
    if not isinstance(rows, list) or len(rows) != len(_ARMS):
        raise ValueError("resume candidate checkpoint pair is incomplete")
    expected_uris: set[str] = set()
    identity_fields = {
        "ctime_ns",
        "device",
        "gid",
        "inode",
        "mode",
        "mtime_ns",
        "size",
        "uid",
    }
    for arm, row in zip(_ARMS, rows, strict=True):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "arm",
                "baseline",
                "bytes",
                "config_sha256",
                "generation",
                "object_uri",
                "sha256",
            }
            or row["arm"] != arm
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or _SHA256_RE.fullmatch(row["sha256"]) is None
            or not isinstance(row["object_uri"], str)
            or not row["object_uri"].endswith(
                f"/{arm}/sha256/{row['sha256']}.pt"
            )
            or not isinstance(row["generation"], dict)
            or set(row["generation"]) != identity_fields
            or any(type(value) is not int for value in row["generation"].values())
            or not isinstance(row["config_sha256"], str)
            or _SHA256_RE.fullmatch(row["config_sha256"]) is None
        ):
            raise ValueError("resume checkpoint evidence is invalid")
        baseline = row["baseline"]
        generation = row["generation"]
        if (
            baseline is not None
            and (
                not isinstance(baseline, dict)
                or set(baseline) != identity_fields
                or any(type(value) is not int for value in baseline.values())
                or (
                    baseline.get("device"),
                    baseline.get("inode"),
                )
                == (
                    generation.get("device"),
                    generation.get("inode"),
                )
            )
        ):
            raise ValueError("resume checkpoint generation did not change")
        uri = row["object_uri"]
        payload = checkpoint_objects.get(uri)
        if (
            payload is None
            or len(payload) != row["bytes"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise ValueError("resume checkpoint object hash mismatch")
        expected_uris.add(uri)
    if set(checkpoint_objects) != expected_uris:
        raise ValueError("resume checkpoint object namespace mismatch")
    return True


class ImdsV2Client:
    """Small IMDSv2 client; every metadata GET uses a bounded token."""

    def __init__(self, *, timeout_seconds: float = 2.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._token: str | None = None

    def _token_value(self) -> str:
        if self._token is None:
            request = Request(
                f"{_IMDS_ROOT}/api/token",
                method="PUT",
                headers={
                    "X-aws-ec2-metadata-token-ttl-seconds": "21600",
                },
            )
            with urlopen(request, timeout=self._timeout_seconds) as response:
                self._token = response.read().decode("ascii")
            if not self._token:
                raise RuntimeError("IMDSv2 returned an empty token")
        return self._token

    def get(self, path: str, *, optional: bool = False) -> str | None:
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError("IMDS path must be relative")
        for attempt in range(2):
            request = Request(
                f"{_IMDS_ROOT}/{path}",
                headers={"X-aws-ec2-metadata-token": self._token_value()},
            )
            try:
                with urlopen(request, timeout=self._timeout_seconds) as response:
                    return response.read().decode("utf-8").strip()
            except HTTPError as error:
                if optional and error.code == 404:
                    return None
                if error.code == 401 and attempt == 0:
                    self._token = None
                    continue
                raise
        raise RuntimeError("IMDSv2 token refresh did not complete")

    def interruption_notice(self) -> str | None:
        rebalance = self.get(
            "meta-data/events/recommendations/rebalance", optional=True
        )
        if rebalance is not None:
            return "rebalance-recommendation"
        spot = self.get("meta-data/spot/instance-action", optional=True)
        if spot is not None:
            return "spot-instance-action"
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--gres", default=GRES)
    parser.add_argument("--assigned-seed", type=int, action="append")
    parser.add_argument(
        "--candidate-receipt-type",
        default=_LEGACY_CANDIDATE_RECEIPT_TYPE,
    )
    parser.add_argument(
        "--interruption-receipt-type",
        default=_LEGACY_INTERRUPTION_RECEIPT_TYPE,
    )
    parser.add_argument(
        "--resume-commit-protocol",
        default=_LEGACY_RESUME_COMMIT_PROTOCOL,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dense-pid", type=int, required=True)
    parser.add_argument("--split90-pid", type=int, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--split90-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-config-sha256", required=True)
    parser.add_argument("--split90-config-sha256", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--corpus-receipt-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--upload-reserve-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        client = ImdsV2Client()
        notice = None
        while notice is None:
            notice = client.interruption_notice()
            if notice is None:
                time.sleep(arguments.poll_seconds)
        region = os.environ.get("AWS_REGION", "")
        store = S3ObjectStore(
            region=region,
            kms_key_id=os.environ.get("MS_S3_KMS_KEY_ID") or None,
            environment={
                name: os.environ[name]
                for name in ("AWS_REGION", "HOME", "LANG", "LC_ALL", "PATH")
                if name in os.environ
            },
        )
        request = InterruptionRequest(
            seed=arguments.seed,
            notice=notice,
            rank_zero_pids={
                "dense": arguments.dense_pid,
                "split90": arguments.split90_pid,
            },
            checkpoint_paths={
                "dense": arguments.dense_checkpoint,
                "split90": arguments.split90_checkpoint,
            },
            s3_root=arguments.s3_root,
            receipt_path=arguments.receipt,
            release_sha256=arguments.release_sha256,
            corpus_receipt_sha256=arguments.corpus_receipt_sha256,
            code_commit=arguments.code_commit,
            config_sha256={
                "dense": arguments.dense_config_sha256,
                "split90": arguments.split90_config_sha256,
            },
            timeout_seconds=arguments.timeout_seconds,
            upload_reserve_seconds=arguments.upload_reserve_seconds,
            provider=arguments.provider,
            profile_sha256=arguments.profile_sha256,
            instance_type=arguments.instance_type,
            gres=arguments.gres,
            assigned_seeds=tuple(
                arguments.assigned_seed
                if arguments.assigned_seed is not None
                else _PROFILE_CONTRACTS.get(arguments.provider, {}).get(
                    "assigned_seeds",
                    (),
                )
            ),
            candidate_receipt_type=arguments.candidate_receipt_type,
            interruption_receipt_type=arguments.interruption_receipt_type,
            resume_commit_protocol=arguments.resume_commit_protocol,
        )
        result = handle_interruption(request, object_store=store)
        report = {
            "exit_code": result.exit_code,
            "ok": result.resumable,
            "receipt": str(result.receipt_path),
            "resumable": result.resumable,
            "schema_version": 1,
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return result.exit_code
    except (ValueError, OSError, RuntimeError, HTTPError, URLError) as error:
        report = {
            "error": type(error).__name__,
            "ok": False,
            "resumable": False,
            "schema_version": 1,
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return NON_RESUMABLE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
