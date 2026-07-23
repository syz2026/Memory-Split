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
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROVIDER = "aws-p5.48xlarge"
RESUMABLE_EXIT_CODE = 75
NON_RESUMABLE_EXIT_CODE = 74
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ARMS = ("dense", "split90")
_IMDS_ROOT = "http://169.254.169.254/latest"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class VerifiedObjectStore(Protocol):
    def put_verified(
        self,
        path: Path,
        uri: str,
        *,
        deadline: float,
        monotonic: Callable[[], float],
    ) -> bool: ...


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

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in {1, 2, 3, 4}:
            raise ValueError("interruption seed must be one of 1, 2, 3, 4")
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


@dataclass(frozen=True)
class InterruptionResult:
    resumable: bool
    exit_code: int
    receipt_path: Path
    receipt_upload_verified: bool


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
        runner: Callable[
            [Sequence[str], Mapping[str, str], float], CommandResult
        ] = _default_runner,
    ) -> None:
        if not isinstance(region, str) or not region:
            raise ValueError("S3 region must be non-empty")
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
        self._environment = dict(environment)
        self._runner = runner

    def put_verified(
        self,
        path: Path,
        uri: str,
        *,
        deadline: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> bool:
        if path.is_symlink() or not path.is_file():
            return False
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or monotonic() >= float(deadline)
        ):
            return False
        bucket, key = _split_s3_uri(uri)
        before = path.stat()
        raw_digest = hashlib.sha256(path.read_bytes()).digest()
        after_hash = path.stat()
        if (
            before.st_ino,
            before.st_dev,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after_hash.st_ino,
            after_hash.st_dev,
            after_hash.st_size,
            after_hash.st_mtime_ns,
        ):
            return False
        checksum = base64.b64encode(raw_digest).decode("ascii")
        remaining = float(deadline) - monotonic()
        if remaining <= 0:
            return False
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
            return False
        put_value = _load_json_output(put)
        if put_value is None or put_value.get("ChecksumSHA256") != checksum:
            return False
        after_upload = path.stat()
        if (
            before.st_ino,
            before.st_dev,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after_upload.st_ino,
            after_upload.st_dev,
            after_upload.st_size,
            after_upload.st_mtime_ns,
        ):
            return False
        remaining = float(deadline) - monotonic()
        if remaining <= 0:
            return False
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
            return False
        head_value = _load_json_output(head)
        return (
            head_value is not None
            and head_value.get("ChecksumSHA256") == checksum
            and monotonic() < float(deadline)
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


def _write_receipt(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ValueError("interruption receipt path must not be a symlink")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stable_checkpoint(
    path: Path,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[int, str] | None:
    previous: tuple[int, int, int] | None = None
    while monotonic() <= deadline:
        if not path.is_symlink() and path.is_file():
            first = path.stat()
            identity = (first.st_ino, first.st_size, first.st_mtime_ns)
            if first.st_size > 0 and identity == previous:
                data = path.read_bytes()
                second = path.stat()
                if (
                    (second.st_ino, second.st_size, second.st_mtime_ns)
                    == identity
                    and len(data) == first.st_size
                ):
                    return len(data), hashlib.sha256(data).hexdigest()
            previous = identity
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(0.25, remaining))
    return None


def _checkpoint_uri(request: InterruptionRequest, arm: str) -> str:
    return (
        f"{request.s3_root.rstrip('/')}/checkpoints/seed-{request.seed}/"
        f"{arm}/checkpoint.pt"
    )


def _receipt_uri(request: InterruptionRequest) -> str:
    return (
        f"{request.s3_root.rstrip('/')}/receipts/interruption/"
        f"seed-{request.seed}.json"
    )


def handle_interruption(
    request: InterruptionRequest,
    *,
    signal_process: Callable[[int, int], None] = os.kill,
    object_store: VerifiedObjectStore,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> InterruptionResult:
    """Checkpoint both arms and durably receipt them as one resumable pair."""

    start = monotonic()
    deadline = start + float(request.timeout_seconds)
    checkpoint_deadline = deadline - float(request.upload_reserve_seconds)
    signal_errors: dict[str, str] = {}
    for arm in _ARMS:
        try:
            signal_process(request.rank_zero_pids[arm], signal.SIGUSR1)
        except OSError as error:
            signal_errors[arm] = type(error).__name__

    stable_by_arm: dict[str, tuple[int, str] | None] = {}
    for arm in _ARMS:
        path = request.checkpoint_paths[arm]
        stable_by_arm[arm] = _stable_checkpoint(
            path,
            deadline=checkpoint_deadline,
            monotonic=monotonic,
            sleep=sleep,
        )
    checkpoint_rows = []
    all_verified = not signal_errors and all(
        stable_by_arm[arm] is not None for arm in _ARMS
    )
    for arm in _ARMS:
        path = request.checkpoint_paths[arm]
        stable = stable_by_arm[arm]
        uri = _checkpoint_uri(request, arm)
        if stable is None:
            row = {
                "arm": arm,
                "bytes": None,
                "config_sha256": request.config_sha256[arm],
                "path": path.name,
                "sha256": None,
                "s3_uri": uri,
                "upload_verified": False,
            }
            all_verified = False
        else:
            size, digest = stable
            uploaded = (
                monotonic() < deadline
                and object_store.put_verified(
                    path,
                    uri,
                    deadline=deadline,
                    monotonic=monotonic,
                )
            )
            row = {
                "arm": arm,
                "bytes": size,
                "config_sha256": request.config_sha256[arm],
                "path": path.name,
                "sha256": digest,
                "s3_uri": uri,
                "upload_verified": uploaded,
            }
            all_verified = all_verified and uploaded
        checkpoint_rows.append(row)

    deadline_exhausted = monotonic() >= checkpoint_deadline and any(
        stable_by_arm[arm] is None for arm in _ARMS
    )
    if monotonic() >= deadline:
        all_verified = False
        deadline_exhausted = True
    receipt = {
        "checkpoints": checkpoint_rows,
        "code_commit": request.code_commit,
        "corpus_receipt_sha256": request.corpus_receipt_sha256,
        "notice": request.notice,
        "paired": True,
        "provider": PROVIDER,
        "receipt_type": "aws-p5-paired-interruption",
        "release_sha256": request.release_sha256,
        "deadline_exhausted": deadline_exhausted,
        "resumable": False,
        "schema_version": 2,
        "seed": request.seed,
        "signal_errors": signal_errors,
        "timeout_seconds": float(request.timeout_seconds),
        "upload_reserve_seconds": float(request.upload_reserve_seconds),
    }
    _write_receipt(request.receipt_path, receipt)
    receipt_uploaded = (
        monotonic() < deadline
        and object_store.put_verified(
            request.receipt_path,
            _receipt_uri(request),
            deadline=deadline,
            monotonic=monotonic,
        )
    )
    resumable = False
    if all_verified and receipt_uploaded and monotonic() < deadline:
        receipt["resumable"] = True
        _write_receipt(request.receipt_path, receipt)
        receipt_uploaded = object_store.put_verified(
            request.receipt_path,
            _receipt_uri(request),
            deadline=deadline,
            monotonic=monotonic,
        )
        resumable = receipt_uploaded and monotonic() < deadline
        if not resumable:
            receipt["resumable"] = False
            receipt["deadline_exhausted"] = monotonic() >= deadline
            _write_receipt(request.receipt_path, receipt)
            if monotonic() < deadline:
                receipt_uploaded = object_store.put_verified(
                    request.receipt_path,
                    _receipt_uri(request),
                    deadline=deadline,
                    monotonic=monotonic,
                )
    else:
        receipt["deadline_exhausted"] = (
            receipt["deadline_exhausted"] or monotonic() >= deadline
        )
        _write_receipt(request.receipt_path, receipt)

    return InterruptionResult(
        resumable=resumable,
        exit_code=(
            RESUMABLE_EXIT_CODE if resumable else NON_RESUMABLE_EXIT_CODE
        ),
        receipt_path=request.receipt_path,
        receipt_upload_verified=receipt_uploaded,
    )


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
