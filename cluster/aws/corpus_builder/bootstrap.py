"""Render and execute the fail-closed AWS corpus-builder bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .contracts import (
    CORPUS_BUCKET,
    CORPUS_KEY_PREFIX,
    S3ObjectVersion,
    s3_object_version_from_dict,
)
from .s3 import PublicationError, S3Client, download_exact_object


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_MARKER = "# __MEMORYSPLIT_RENDERED_CONFIG__"
_SCRIPT_PATH = Path(__file__).with_name("bootstrap.sh")
_MAX_WORKERS = 64
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class BootstrapError(ValueError):
    """A fail-closed bootstrap configuration or runtime error."""


@dataclass(frozen=True)
class BootstrapConfig:
    build_id: str
    package: S3ObjectVersion
    source_manifest: S3ObjectVersion
    kms_key_arn: str
    profile_sha256: str
    launch_intent_sha256: str
    workers: int


def _object_dict(value: S3ObjectVersion) -> dict[str, object]:
    return {
        "bytes": value.bytes,
        "etag": value.etag,
        "kms_key_arn": value.kms_key_arn,
        "sha256": value.sha256,
        "sse_algorithm": value.sse_algorithm,
        "uri": value.uri,
        "version_id": value.version_id,
    }


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BootstrapError(f"{label} must be a lowercase SHA-256")
    return value


def _validated_record(
    value: object,
    *,
    label: str,
    build_id: str,
    kms_key_arn: str,
) -> S3ObjectVersion:
    if not isinstance(value, S3ObjectVersion):
        raise BootstrapError(f"{label} must be an S3ObjectVersion")
    try:
        parsed = s3_object_version_from_dict(
            _object_dict(value),
            expected_build_id=build_id,
        )
    except ValueError as error:
        raise BootstrapError(f"{label} is invalid: {error}") from error
    if parsed != value:
        raise BootstrapError(f"{label} changed during validation")
    if parsed.kms_key_arn != kms_key_arn:
        raise BootstrapError(f"{label} KMS key does not match bootstrap KMS key")
    prefix = f"s3://{CORPUS_BUCKET}/{CORPUS_KEY_PREFIX}/{build_id}/"
    if not parsed.uri.startswith(prefix) or parsed.uri == prefix:
        raise BootstrapError(f"{label} must be under the approved build prefix")
    return parsed


def _validate_config(config: BootstrapConfig) -> BootstrapConfig:
    if not isinstance(config, BootstrapConfig):
        raise BootstrapError("config must be a BootstrapConfig")
    build_id = _sha256(config.build_id, label="build_id")
    _sha256(config.profile_sha256, label="profile_sha256")
    _sha256(config.launch_intent_sha256, label="launch_intent_sha256")
    if (
        isinstance(config.workers, bool)
        or not isinstance(config.workers, int)
        or not 1 <= config.workers <= _MAX_WORKERS
    ):
        raise BootstrapError("workers must be an integer between 1 and 64")
    package = _validated_record(
        config.package,
        label="package",
        build_id=build_id,
        kms_key_arn=config.kms_key_arn,
    )
    source_manifest = _validated_record(
        config.source_manifest,
        label="source_manifest",
        build_id=build_id,
        kms_key_arn=config.kms_key_arn,
    )
    if package.uri == source_manifest.uri:
        raise BootstrapError("package and source manifest must be distinct objects")
    return config


def render_bootstrap(config: BootstrapConfig | None = None) -> str:
    """Render deterministic, build-invariant launch-template user data.

    ``config`` remains as a transition-only compatibility argument. Its values
    are deliberately neither validated nor serialized; the canonical stack
    integration calls this function with no argument. Per-build authority is
    supplied later to the installed SSM entry point.
    """

    if config is not None and not isinstance(config, BootstrapConfig):
        raise BootstrapError("config must be a BootstrapConfig or None")
    try:
        template = _SCRIPT_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise BootstrapError("bootstrap shell template cannot be read") from error
    if (
        not template.startswith("#!/usr/bin/env bash\n")
        or template.count(_CONFIG_MARKER) != 1
        or not template.endswith("\n")
    ):
        raise BootstrapError("bootstrap shell template has invalid framing")
    return template.replace(
        _CONFIG_MARKER,
        "# Build-invariant Task 6 bootstrap; per-build values arrive via SSM.",
        1,
    )


def download_bootstrap_inputs(
    s3: S3Client,
    config: BootstrapConfig,
    *,
    package_destination: Path,
    source_manifest_destination: Path,
) -> None:
    """Download both approved input records through the Task 3 helper."""

    config = _validate_config(config)
    package_path = Path(package_destination)
    manifest_path = Path(source_manifest_destination)
    if package_path == manifest_path:
        raise BootstrapError("package and source manifest destinations must differ")
    download_exact_object(s3, config.package, package_path)
    download_exact_object(s3, config.source_manifest, manifest_path)


def _stable_hash(path: Path) -> tuple[int, str]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(f"bootstrap input is not a readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapError(f"bootstrap input is not a regular file: {path}")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    except BootstrapError:
        raise
    except OSError as error:
        raise BootstrapError(f"bootstrap input cannot be hashed: {path}") from error
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or byte_count != before.st_size:
        raise BootstrapError(f"bootstrap input changed while hashing: {path}")
    return byte_count, digest.hexdigest()


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError(f"source manifest repeats JSON field: {key}")
        result[key] = value
    return result


def _verify_source_manifest(path: Path) -> None:
    try:
        size = path.stat(follow_symlinks=False).st_size
    except OSError as error:
        raise BootstrapError("source manifest cannot be inspected") from error
    if size > _MAX_MANIFEST_BYTES:
        raise BootstrapError("source manifest exceeds 64 MiB")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_json_pairs,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    BootstrapError(
                        f"source manifest contains non-finite value: {constant}"
                    )
                ),
            )
    except BootstrapError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("source manifest must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BootstrapError("source manifest must be a JSON object")


def _safe_archive_members(archive_path: Path) -> None:
    names: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    not member.name
                    or pure.is_absolute()
                    or pure.as_posix() != member.name
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or member.name in names
                    or not (member.isdir() or member.isreg())
                ):
                    raise BootstrapError(
                        f"unsafe package archive member: {member.name!r}"
                    )
                names.add(member.name)
    except BootstrapError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise BootstrapError("package archive is not a valid gzip tar") from error
    if not names:
        raise BootstrapError("package archive must not be empty")


def verify_package_inputs(
    config: BootstrapConfig,
    *,
    archive_path: Path,
    source_manifest_path: Path,
) -> None:
    """Rehash downloaded inputs and reject unsafe package archive members."""

    config = _validate_config(config)
    archive = Path(archive_path)
    manifest = Path(source_manifest_path)
    archive_bytes, archive_sha256 = _stable_hash(archive)
    manifest_bytes, manifest_sha256 = _stable_hash(manifest)
    if (
        archive_bytes != config.package.bytes
        or archive_sha256 != config.package.sha256
    ):
        raise BootstrapError("package archive authority changed after download")
    if (
        manifest_bytes != config.source_manifest.bytes
        or manifest_sha256 != config.source_manifest.sha256
    ):
        raise BootstrapError("source manifest authority changed after download")
    _verify_source_manifest(manifest)
    _safe_archive_members(archive)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise BootstrapError(f"required bootstrap environment is missing: {name}")
    return value


def _integer_environment(name: str) -> int:
    value = _required_environment(name)
    if not value.isascii() or not value.isdigit():
        raise BootstrapError(f"bootstrap environment must be an integer: {name}")
    return int(value)


def _config_from_environment() -> BootstrapConfig:
    package = S3ObjectVersion(
        uri=_required_environment("MEMORYSPLIT_PACKAGE_URI"),
        version_id=_required_environment("MEMORYSPLIT_PACKAGE_VERSION_ID"),
        bytes=_integer_environment("MEMORYSPLIT_PACKAGE_BYTES"),
        sha256=_required_environment("MEMORYSPLIT_PACKAGE_SHA256"),
        etag=_required_environment("MEMORYSPLIT_PACKAGE_ETAG"),
        sse_algorithm=_required_environment(
            "MEMORYSPLIT_PACKAGE_SSE_ALGORITHM"
        ),
        kms_key_arn=_required_environment("MEMORYSPLIT_PACKAGE_KMS_KEY_ARN"),
    )
    source_manifest = S3ObjectVersion(
        uri=_required_environment("MEMORYSPLIT_SOURCE_MANIFEST_URI"),
        version_id=_required_environment(
            "MEMORYSPLIT_SOURCE_MANIFEST_VERSION_ID"
        ),
        bytes=_integer_environment("MEMORYSPLIT_SOURCE_MANIFEST_BYTES"),
        sha256=_required_environment("MEMORYSPLIT_SOURCE_MANIFEST_SHA256"),
        etag=_required_environment("MEMORYSPLIT_SOURCE_MANIFEST_ETAG"),
        sse_algorithm=_required_environment(
            "MEMORYSPLIT_SOURCE_MANIFEST_SSE_ALGORITHM"
        ),
        kms_key_arn=_required_environment(
            "MEMORYSPLIT_SOURCE_MANIFEST_KMS_KEY_ARN"
        ),
    )
    return _validate_config(
        BootstrapConfig(
            build_id=_required_environment("MEMORYSPLIT_BUILD_ID"),
            package=package,
            source_manifest=source_manifest,
            kms_key_arn=_required_environment("MEMORYSPLIT_KMS_KEY_ARN"),
            profile_sha256=_required_environment(
                "MEMORYSPLIT_PROFILE_SHA256"
            ),
            launch_intent_sha256=_required_environment(
                "MEMORYSPLIT_LAUNCH_INTENT_SHA256"
            ),
            workers=_integer_environment("MEMORYSPLIT_WORKERS"),
        )
    )


class _TemporaryBody:
    def __init__(self, stream: BinaryIO, path: Path) -> None:
        self._stream = stream
        self._path = path
        self._closed = False

    def read(self, amount: int = -1) -> bytes:
        return self._stream.read(amount)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass


class AwsCliS3Client:
    """Minimal exact-version S3 adapter for the AL2023 AWS CLI."""

    def __init__(
        self,
        scratch_directory: Path,
        *,
        aws_path: Path = Path("/usr/bin/aws"),
    ) -> None:
        self._scratch_directory = Path(scratch_directory)
        self._aws_path = Path(aws_path)
        if (
            self._scratch_directory.is_symlink()
            or not self._scratch_directory.is_dir()
        ):
            raise BootstrapError("AWS CLI scratch path must be a real directory")
        mode = stat.S_IMODE(
            self._scratch_directory.stat(follow_symlinks=False).st_mode
        )
        if mode & 0o077:
            raise BootstrapError("AWS CLI scratch path must be owner-only")

    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "false",
            "AWS_PAGER": "",
            "AWS_REGION": "us-east-1",
            "HOME": "/root",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }

    def _run_json(self, arguments: list[str]) -> Mapping[str, object]:
        try:
            completed = subprocess.run(
                [
                    os.fspath(self._aws_path),
                    "s3api",
                    *arguments,
                    "--no-cli-pager",
                    "--output",
                    "json",
                ],
                cwd="/",
                env=self._environment(),
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise BootstrapError("AWS CLI could not be executed") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            raise BootstrapError(
                f"AWS CLI exact-version request failed: {detail or 'unknown error'}"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise BootstrapError("AWS CLI returned malformed JSON") from error
        if not isinstance(value, dict):
            raise BootstrapError("AWS CLI response must be a JSON object")
        return value

    @staticmethod
    def _request(
        kwargs: Mapping[str, object],
    ) -> tuple[str, str, str]:
        if set(kwargs) != {"Bucket", "Key", "VersionId"}:
            raise BootstrapError("S3 bootstrap request must pin one exact version")
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        version_id = kwargs["VersionId"]
        if not all(isinstance(value, str) and value for value in (bucket, key, version_id)):
            raise BootstrapError("S3 bootstrap request authority is malformed")
        return bucket, key, version_id

    def head_object(self, **kwargs: object) -> Mapping[str, object]:
        bucket, key, version_id = self._request(kwargs)
        return self._run_json(
            [
                "head-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--version-id",
                version_id,
            ]
        )

    def get_object(self, **kwargs: object) -> Mapping[str, object]:
        bucket, key, version_id = self._request(kwargs)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".s3-get-",
            dir=self._scratch_directory,
        )
        os.close(descriptor)
        path = Path(raw_path)
        try:
            response = dict(
                self._run_json(
                    [
                        "get-object",
                        "--bucket",
                        bucket,
                        "--key",
                        key,
                        "--version-id",
                        version_id,
                        os.fspath(path),
                    ]
                )
            )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            body_descriptor = os.open(path, flags)
            metadata = os.fstat(body_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(body_descriptor)
                raise BootstrapError("AWS CLI download body is not a regular file")
            response["Body"] = _TemporaryBody(
                os.fdopen(body_descriptor, "rb", closefd=True),
                path,
            )
            return response
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download-inputs")
    download.add_argument("--scratch-directory", type=Path, required=True)
    download.add_argument("--package-destination", type=Path, required=True)
    download.add_argument(
        "--source-manifest-destination",
        type=Path,
        required=True,
    )

    verify = commands.add_parser("verify-package")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--source-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _config_from_environment()
        if args.command == "download-inputs":
            client = AwsCliS3Client(args.scratch_directory)
            download_bootstrap_inputs(
                client,
                config,
                package_destination=args.package_destination,
                source_manifest_destination=args.source_manifest_destination,
            )
        elif args.command == "verify-package":
            verify_package_inputs(
                config,
                archive_path=args.archive,
                source_manifest_path=args.source_manifest,
            )
        else:  # pragma: no cover - argparse closes this branch
            raise BootstrapError("unknown bootstrap command")
    except (BootstrapError, PublicationError) as error:
        print(f"bootstrap failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
