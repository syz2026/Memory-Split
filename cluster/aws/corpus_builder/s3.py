"""Exact-version, SSE-KMS S3 publication and clean-room download."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .contracts import (
    CORPUS_BUCKET,
    CORPUS_KEY_PREFIX,
    PhaseReceipt,
    S3ObjectVersion,
    phase_receipt_to_bytes,
    s3_object_version_from_dict,
)


_CHUNK_BYTES = 1 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_METADATA_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class PublicationError(ValueError):
    """A fail-closed exact-version publication or download error."""


class S3Client(Protocol):
    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def list_object_versions(self, **kwargs: object) -> Mapping[str, object]: ...


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


def _validate_record(value: S3ObjectVersion) -> S3ObjectVersion:
    try:
        parsed = s3_object_version_from_dict(_object_dict(value))
    except ValueError as error:
        raise PublicationError(f"invalid S3 object authority: {error}") from error
    if parsed != value:
        raise PublicationError("S3 object authority changed during validation")
    return parsed


def _normalize_etag(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationError("S3 response is missing an ETag")
    if value.startswith('"') or value.endswith('"'):
        if len(value) < 2 or not (value.startswith('"') and value.endswith('"')):
            raise PublicationError("S3 response contains a malformed ETag")
        value = value[1:-1]
    return value


def _validated_metadata(
    metadata: Mapping[str, str],
    digest: str,
) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise PublicationError("S3 metadata must be a mapping")
    result: dict[str, str] = {}
    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or _METADATA_KEY_RE.fullmatch(key) is None
        ):
            raise PublicationError(
                "S3 metadata keys must be lowercase portable strings "
                "and values must be strings"
            )
        if key == "sha256":
            raise PublicationError("S3 metadata key 'sha256' is reserved")
        result[key] = value
    result["sha256"] = digest
    return result


def _split_uri(value: S3ObjectVersion) -> tuple[str, str]:
    _validate_record(value)
    parsed = urlsplit(value.uri)
    if parsed.hostname is None:
        raise PublicationError("S3 object URI has no bucket")
    return parsed.hostname, parsed.path.removeprefix("/")


def _safe_unresolved_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or "\\" in uri:
        raise PublicationError("receipt URI must be a safe S3 URI")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname != CORPUS_BUCKET
    ):
        raise PublicationError("receipt URI must use the corpus S3 bucket")
    key = parsed.path.removeprefix("/")
    parts = key.split("/")
    prefix = CORPUS_KEY_PREFIX.split("/")
    if (
        parsed.path != f"/{key}"
        or len(parts) < len(prefix) + 2
        or parts[: len(prefix)] != prefix
        or _SHA256_RE.fullmatch(parts[len(prefix)]) is None
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise PublicationError("receipt URI must use a safe corpus build key")
    return CORPUS_BUCKET, key


def _prototype(
    *,
    bucket: str,
    key: str,
    byte_count: int,
    digest: str,
    kms_key_arn: str,
) -> str:
    candidate = S3ObjectVersion(
        uri=f"s3://{bucket}/{key}",
        version_id="authority-check",
        bytes=byte_count,
        sha256=digest,
        etag="0" * 32,
        sse_algorithm="aws:kms",
        kms_key_arn=kms_key_arn,
    )
    return _validate_record(candidate).uri


def _head(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    version_id: str,
) -> Mapping[str, object]:
    try:
        response = s3.head_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
        )
    except Exception as error:
        raise PublicationError(
            f"exact S3 version is unavailable: {version_id}"
        ) from error
    if not isinstance(response, Mapping):
        raise PublicationError("S3 HEAD response must be a mapping")
    return response


def _response_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PublicationError("S3 response metadata is missing")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise PublicationError("S3 response metadata is malformed")
        result[key] = item
    return result


def _require_response_authority(
    response: Mapping[str, object],
    expected: S3ObjectVersion,
    *,
    required_metadata: Mapping[str, str],
    exact_metadata: bool,
) -> None:
    byte_count = response.get("ContentLength")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count != expected.bytes
    ):
        raise PublicationError("exact S3 version byte count drift")
    if response.get("VersionId") != expected.version_id:
        raise PublicationError("exact S3 version ID drift")
    if _normalize_etag(response.get("ETag")) != expected.etag:
        raise PublicationError("exact S3 version ETag drift")
    if response.get("ServerSideEncryption") != expected.sse_algorithm:
        raise PublicationError("exact S3 version encryption algorithm drift")
    if response.get("SSEKMSKeyId") != expected.kms_key_arn:
        raise PublicationError("exact S3 version KMS key drift")
    metadata = _response_metadata(response.get("Metadata"))
    if exact_metadata:
        if metadata != dict(required_metadata):
            raise PublicationError("exact S3 version metadata drift")
    elif any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise PublicationError("exact S3 version metadata SHA-256 drift")


def _verify_head(
    s3: S3Client,
    expected: S3ObjectVersion,
    *,
    required_metadata: Mapping[str, str],
    exact_metadata: bool,
) -> None:
    bucket, key = _split_uri(expected)
    response = _head(
        s3,
        bucket=bucket,
        key=key,
        version_id=expected.version_id,
    )
    _require_response_authority(
        response,
        expected,
        required_metadata=required_metadata,
        exact_metadata=exact_metadata,
    )


def verify_exact_object(
    s3: S3Client,
    expected: S3ObjectVersion,
) -> None:
    """HEAD and verify one immutable S3 object version exactly."""

    _verify_head(
        s3,
        expected,
        required_metadata={"sha256": expected.sha256},
        exact_metadata=False,
    )


def _record_from_head(
    *,
    uri: str,
    requested_version_id: str,
    expected_sha256: str,
    response: Mapping[str, object],
) -> S3ObjectVersion:
    byte_count = response.get("ContentLength")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise PublicationError("S3 HEAD response has an invalid byte count")
    version_id = response.get("VersionId")
    if version_id != requested_version_id:
        raise PublicationError("S3 HEAD response version ID drift")
    metadata = _response_metadata(response.get("Metadata"))
    if metadata.get("sha256") != expected_sha256:
        raise PublicationError("S3 HEAD metadata SHA-256 drift")
    sse_algorithm = response.get("ServerSideEncryption")
    kms_key_arn = response.get("SSEKMSKeyId")
    if not isinstance(sse_algorithm, str) or not isinstance(kms_key_arn, str):
        raise PublicationError("S3 HEAD encryption authority is missing")
    record = S3ObjectVersion(
        uri=uri,
        version_id=requested_version_id,
        bytes=byte_count,
        sha256=expected_sha256,
        etag=_normalize_etag(response.get("ETag")),
        sse_algorithm=sse_algorithm,
        kms_key_arn=kms_key_arn,
    )
    return _validate_record(record)


def _resolve_exact_object(
    s3: S3Client,
    *,
    uri: str,
    version_id: str,
    sha256: str,
) -> S3ObjectVersion:
    """Bootstrap a full authority record from one explicitly pinned HEAD."""

    bucket, key = _safe_unresolved_uri(uri)
    if not isinstance(version_id, str) or not version_id:
        raise PublicationError("receipt version ID must be non-empty")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise PublicationError("receipt SHA-256 must be lowercase hexadecimal")
    response = _head(
        s3,
        bucket=bucket,
        key=key,
        version_id=version_id,
    )
    record = _record_from_head(
        uri=uri,
        requested_version_id=version_id,
        expected_sha256=sha256,
        response=response,
    )
    _require_response_authority(
        response,
        record,
        required_metadata={"sha256": sha256},
        exact_metadata=False,
    )
    return record


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _hash_descriptor(
    descriptor: int,
    expected_metadata: os.stat_result,
) -> tuple[int, str]:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or _descriptor_identity(before) != _descriptor_identity(expected_metadata)
        ):
            raise PublicationError("publication source is not a stable regular file")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, _CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    except PublicationError:
        raise
    except OSError as error:
        raise PublicationError("publication source cannot be hashed") from error
    if (
        byte_count != expected_metadata.st_size
        or _descriptor_identity(after) != _descriptor_identity(expected_metadata)
    ):
        raise PublicationError("publication source changed while pinned")
    return byte_count, digest.hexdigest()


def _put_exact(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    body: object,
    byte_count: int,
    digest: str,
    kms_key_arn: str,
    metadata: Mapping[str, str],
) -> S3ObjectVersion:
    uri = _prototype(
        bucket=bucket,
        key=key,
        byte_count=byte_count,
        digest=digest,
        kms_key_arn=kms_key_arn,
    )
    uploaded_metadata = _validated_metadata(metadata, digest)
    try:
        response = s3.put_object(
            Body=body,
            Bucket=bucket,
            ContentLength=byte_count,
            Key=key,
            Metadata=uploaded_metadata,
            SSEKMSKeyId=kms_key_arn,
            ServerSideEncryption="aws:kms",
        )
    except Exception as error:
        raise PublicationError("S3 exact-version upload failed") from error
    if not isinstance(response, Mapping):
        raise PublicationError("S3 PUT response must be a mapping")
    version_id = response.get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise PublicationError("S3 PUT did not return a non-empty version ID")
    record = _validate_record(
        S3ObjectVersion(
            uri=uri,
            version_id=version_id,
            bytes=byte_count,
            sha256=digest,
            etag=_normalize_etag(response.get("ETag")),
            sse_algorithm="aws:kms",
            kms_key_arn=kms_key_arn,
        )
    )
    _verify_head(
        s3,
        record,
        required_metadata=uploaded_metadata,
        exact_metadata=True,
    )
    return record


def publish_exact_file(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    path: Path,
    kms_key_arn: str,
    metadata: Mapping[str, str],
) -> S3ObjectVersion:
    """Hash and stream one pinned regular file to an exact SSE-KMS version."""

    source = Path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise PublicationError(
            "publication source must be a readable regular file, not a symlink"
        ) from error
    try:
        initial = os.fstat(descriptor)
        byte_count, digest = _hash_descriptor(descriptor, initial)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as body:
            record = _put_exact(
                s3,
                bucket=bucket,
                key=key,
                body=body,
                byte_count=byte_count,
                digest=digest,
                kms_key_arn=kms_key_arn,
                metadata=metadata,
            )
        final_count, final_digest = _hash_descriptor(descriptor, initial)
        if final_count != byte_count or final_digest != digest:
            raise PublicationError("publication source changed during upload")
        return record
    finally:
        os.close(descriptor)


def _list_exact_key_history(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    request: dict[str, object] = {"Bucket": bucket, "Prefix": key}
    seen_markers: set[tuple[str, str]] = set()
    versions: list[Mapping[str, object]] = []
    delete_markers: list[Mapping[str, object]] = []
    while True:
        try:
            response = s3.list_object_versions(**request)
        except Exception as error:
            raise PublicationError("S3 version-history listing failed") from error
        if not isinstance(response, Mapping):
            raise PublicationError("S3 version-history response must be a mapping")
        for field, destination in (
            ("Versions", versions),
            ("DeleteMarkers", delete_markers),
        ):
            items = response.get(field, [])
            if not isinstance(items, list):
                raise PublicationError(f"S3 version-history {field} is malformed")
            for item in items:
                if not isinstance(item, Mapping):
                    raise PublicationError(
                        f"S3 version-history {field} entry is malformed"
                    )
                if item.get("Key") == key:
                    destination.append(item)
        truncated = response.get("IsTruncated", False)
        if truncated is False:
            break
        if truncated is not True:
            raise PublicationError("S3 version-history truncation flag is malformed")
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            not isinstance(next_key, str)
            or not isinstance(next_version, str)
            or not next_key
            or not next_version
        ):
            raise PublicationError("S3 version-history pagination marker is missing")
        marker = (next_key, next_version)
        if marker in seen_markers:
            raise PublicationError("S3 version-history pagination did not advance")
        seen_markers.add(marker)
        request["KeyMarker"] = next_key
        request["VersionIdMarker"] = next_version
    return versions, delete_markers


def _get(
    s3: S3Client,
    expected: S3ObjectVersion,
    *,
    required_metadata: Mapping[str, str],
    exact_metadata: bool,
) -> tuple[Mapping[str, object], object]:
    bucket, key = _split_uri(expected)
    try:
        response = s3.get_object(
            Bucket=bucket,
            Key=key,
            VersionId=expected.version_id,
        )
    except Exception as error:
        raise PublicationError(
            f"exact S3 version download failed: {expected.version_id}"
        ) from error
    if not isinstance(response, Mapping):
        raise PublicationError("S3 GET response must be a mapping")
    _require_response_authority(
        response,
        expected,
        required_metadata=required_metadata,
        exact_metadata=exact_metadata,
    )
    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise PublicationError("S3 GET response body is not readable")
    return response, body


def _close_body(body: object) -> None:
    close = getattr(body, "close", None)
    if callable(close):
        close()


def _receipt_version_matches(
    s3: S3Client,
    candidate: S3ObjectVersion,
    payload: bytes,
    metadata: Mapping[str, str],
) -> bool:
    _verify_head(
        s3,
        candidate,
        required_metadata=metadata,
        exact_metadata=True,
    )
    _response, body = _get(
        s3,
        candidate,
        required_metadata=metadata,
        exact_metadata=True,
    )
    offset = 0
    digest = hashlib.sha256()
    try:
        try:
            while True:
                chunk = body.read(_CHUNK_BYTES)
                if chunk == b"":
                    break
                if not isinstance(chunk, bytes):
                    raise PublicationError("S3 receipt body returned non-byte data")
                if payload[offset : offset + len(chunk)] != chunk:
                    return False
                digest.update(chunk)
                offset += len(chunk)
        finally:
            _close_body(body)
    except PublicationError:
        raise
    except Exception as error:
        raise PublicationError("S3 phase receipt stream failed") from error
    if offset != len(payload) or digest.hexdigest() != candidate.sha256:
        return False
    _verify_head(
        s3,
        candidate,
        required_metadata=metadata,
        exact_metadata=True,
    )
    return True


def publish_phase_receipt(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    receipt: PhaseReceipt,
    kms_key_arn: str,
) -> S3ObjectVersion:
    """Publish canonical receipt bytes once, or reuse only exact history."""

    try:
        payload = phase_receipt_to_bytes(receipt)
    except ValueError as error:
        raise PublicationError(f"phase receipt is invalid: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    uri = _prototype(
        bucket=bucket,
        key=key,
        byte_count=len(payload),
        digest=digest,
        kms_key_arn=kms_key_arn,
    )
    versions, delete_markers = _list_exact_key_history(
        s3,
        bucket=bucket,
        key=key,
    )
    if delete_markers:
        raise PublicationError("phase receipt key has a conflicting delete marker")
    if not versions:
        return _put_exact(
            s3,
            bucket=bucket,
            key=key,
            body=io.BytesIO(payload),
            byte_count=len(payload),
            digest=digest,
            kms_key_arn=kms_key_arn,
            metadata={},
        )

    expected_metadata = {"sha256": digest}
    reusable: list[S3ObjectVersion] = []
    seen_version_ids: set[str] = set()
    for version in versions:
        version_id = version.get("VersionId")
        if (
            not isinstance(version_id, str)
            or not version_id
            or version_id in seen_version_ids
        ):
            raise PublicationError("phase receipt key has conflicting version history")
        seen_version_ids.add(version_id)
        try:
            head = _head(
                s3,
                bucket=bucket,
                key=key,
                version_id=version_id,
            )
            candidate = _record_from_head(
                uri=uri,
                requested_version_id=version_id,
                expected_sha256=digest,
                response=head,
            )
            if (
                candidate.kms_key_arn != kms_key_arn
                or not _receipt_version_matches(
                    s3,
                    candidate,
                    payload,
                    expected_metadata,
                )
            ):
                raise PublicationError("phase receipt version bytes differ")
        except PublicationError as error:
            raise PublicationError(
                "phase receipt key has conflicting version history"
            ) from error
        reusable.append(candidate)
    if not reusable:
        raise PublicationError("phase receipt history did not yield an exact version")
    return reusable[0]


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise PublicationError("short write during exact-version download")
        written += count


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)
    )


def download_exact_object(
    s3: S3Client,
    expected: S3ObjectVersion,
    destination: Path,
) -> None:
    """Exclusively create, stream, fsync, hash, and re-HEAD one version."""

    expected = _validate_record(expected)
    output = Path(destination)
    if output.name in {"", ".", ".."}:
        raise PublicationError("download destination must name a file")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(output.parent, parent_flags)
    except OSError as error:
        raise PublicationError(
            "download destination parent must be a real directory"
        ) from error

    descriptor = -1
    created: os.stat_result | None = None
    success = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(output.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError as error:
            raise PublicationError(
                "download destination exists; exclusive creation required"
            ) from error
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise PublicationError("download destination is not a regular file")

        verify_exact_object(s3, expected)
        _response, body = _get(
            s3,
            expected,
            required_metadata={"sha256": expected.sha256},
            exact_metadata=False,
        )
        digest = hashlib.sha256()
        byte_count = 0
        try:
            while True:
                chunk = body.read(_CHUNK_BYTES)
                if chunk == b"":
                    break
                if not isinstance(chunk, bytes):
                    raise PublicationError("S3 body returned non-byte data")
                byte_count += len(chunk)
                if byte_count > expected.bytes:
                    raise PublicationError("S3 body exceeds its exact byte count")
                _write_all(descriptor, chunk)
                digest.update(chunk)
        finally:
            _close_body(body)
        if byte_count != expected.bytes:
            raise PublicationError("S3 body exact byte count drift")
        if digest.hexdigest() != expected.sha256:
            raise PublicationError("S3 body SHA-256 digest drift")
        os.fsync(descriptor)
        verify_exact_object(s3, expected)
        final = os.fstat(descriptor)
        named = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not _same_file(created, final)
            or not _same_file(created, named)
            or final.st_size != expected.bytes
        ):
            raise PublicationError("download destination identity or size drift")
        os.fsync(parent_fd)
        success = True
    except PublicationError:
        raise
    except OSError as error:
        raise PublicationError("exact-version download failed locally") from error
    except Exception as error:
        raise PublicationError("exact-version download stream failed") from error
    finally:
        if not success and descriptor >= 0 and created is not None:
            try:
                current_fd = os.fstat(descriptor)
                named = os.stat(
                    output.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if _same_file(created, current_fd) and _same_file(created, named):
                    os.unlink(output.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except (FileNotFoundError, OSError):
                pass
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
