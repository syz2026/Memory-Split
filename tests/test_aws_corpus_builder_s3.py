from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from cluster.aws.corpus_builder.contracts import (
    CORPUS_BUCKET,
    PHASE_RECEIPT_FORMAT,
    PhaseReceipt,
    S3ObjectVersion,
    phase_receipt_to_bytes,
)
from cluster.aws.corpus_builder.s3 import (
    PublicationError,
    download_exact_object,
    publish_exact_file,
    publish_phase_receipt,
    verify_exact_object,
)
from scripts import aws_corpus_cleanroom_verify


_BUILD_ID = "b" * 64
_OTHER_BUILD_ID = "e" * 64
_KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/01234567-89ab-cdef-0123-456789abcdef"
)
_OTHER_KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/fedcba98-7654-3210-fedc-ba9876543210"
)


class FakeClientError(RuntimeError):
    pass


class ChunkedBody:
    def __init__(
        self,
        payload: bytes,
        *,
        on_first_read: Callable[[], None] | None = None,
    ) -> None:
        self._payload = payload
        self._offset = 0
        self._on_first_read = on_first_read
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        if self._on_first_read is not None:
            callback, self._on_first_read = self._on_first_read, None
            callback()
        if self._offset >= len(self._payload):
            return b""
        requested = len(self._payload) if amount < 0 else amount
        end = min(len(self._payload), self._offset + requested, self._offset + 3)
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk

    def close(self) -> None:
        self.closed = True


class FailingBody:
    def read(self, _amount: int = -1) -> bytes:
        raise FakeClientError("stream interrupted")

    def close(self) -> None:
        pass


class FakeInputShape:
    def __init__(self, members: set[str]) -> None:
        self.members = {name: object() for name in members}


class FakeOperationModel:
    def __init__(self, members: set[str]) -> None:
        self.input_shape = FakeInputShape(members)


class FakeServiceModel:
    def __init__(self, put_object_members: set[str]) -> None:
        self._put_object_members = put_object_members

    def operation_model(self, name: str) -> FakeOperationModel:
        assert name == "PutObject"
        return FakeOperationModel(self._put_object_members)


class FakeClientMeta:
    def __init__(self, put_object_members: set[str]) -> None:
        self.service_model = FakeServiceModel(put_object_members)


class VersionedFakeS3:
    """In-memory, version-aware fake; no SDK or network dependency."""

    def __init__(self) -> None:
        self.meta = FakeClientMeta(
            {"Body", "Bucket", "IfNoneMatch", "Key"}
        )
        self._counter = 0
        self._versions: dict[tuple[str, str], list[dict[str, object]]] = {}
        self.delete_markers: list[dict[str, object]] = []
        self.put_calls: list[dict[str, object]] = []
        self.head_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.list_calls: list[dict[str, object]] = []
        self.omit_version_id = False
        self.before_put_read: Callable[[], None] | None = None
        self.get_body_override: bytes | None = None
        self.get_response_overrides: dict[str, object] = {}
        self.last_get_body: object | None = None
        self.on_get_read: Callable[[], None] | None = None
        self.fail_get_stream = False
        self.conditional_race: tuple[bytes, str, Mapping[str, str]] | None = None

    def _entry(self, bucket: str, key: str, version_id: str) -> dict[str, object]:
        for entry in self._versions.get((bucket, key), []):
            if entry["VersionId"] == version_id:
                return entry
        raise FakeClientError("NoSuchVersion")

    def put_object(self, **kwargs: object) -> Mapping[str, object]:
        self.put_calls.append(dict(kwargs))
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        body = kwargs["Body"]
        assert isinstance(bucket, str)
        assert isinstance(key, str)
        if kwargs.get("IfNoneMatch") == "*":
            if self.conditional_race is not None:
                payload, kms_key_arn, metadata = self.conditional_race
                self.conditional_race = None
                self.install(
                    bucket=bucket,
                    key=key,
                    payload=payload,
                    kms_key_arn=kms_key_arn,
                    metadata=metadata,
                )
            if self._versions.get((bucket, key)):
                raise FakeClientError("PreconditionFailed")
        assert not isinstance(body, (bytes, bytearray))
        assert hasattr(body, "read")
        if self.before_put_read is not None:
            callback, self.before_put_read = self.before_put_read, None
            callback()
        chunks = []
        while True:
            chunk = body.read(4)
            if chunk == b"":
                break
            assert isinstance(chunk, bytes)
            chunks.append(chunk)
        payload = b"".join(chunks)
        self._counter += 1
        version_id = f"version-{self._counter:04d}"
        etag = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        entry: dict[str, object] = {
            "BodyBytes": payload,
            "ContentLength": len(payload),
            "ETag": f'"{etag}"',
            "Key": key,
            "Metadata": dict(kwargs["Metadata"]),
            "SSEKMSKeyId": kwargs["SSEKMSKeyId"],
            "ServerSideEncryption": kwargs["ServerSideEncryption"],
            "VersionId": version_id,
            "HeadOverrides": {},
        }
        self._versions.setdefault((bucket, key), []).insert(0, entry)
        response: dict[str, object] = {"ETag": f'"{etag}"'}
        if not self.omit_version_id:
            response["VersionId"] = version_id
        return response

    def head_object(self, **kwargs: object) -> Mapping[str, object]:
        self.head_calls.append(dict(kwargs))
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        version_id = kwargs["VersionId"]
        assert isinstance(bucket, str)
        assert isinstance(key, str)
        assert isinstance(version_id, str)
        entry = self._entry(bucket, key, version_id)
        response = {
            field: entry[field]
            for field in (
                "ContentLength",
                "ETag",
                "Metadata",
                "SSEKMSKeyId",
                "ServerSideEncryption",
                "VersionId",
            )
        }
        response.update(entry["HeadOverrides"])
        return response

    def get_object(self, **kwargs: object) -> Mapping[str, object]:
        self.get_calls.append(dict(kwargs))
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        version_id = kwargs["VersionId"]
        assert isinstance(bucket, str)
        assert isinstance(key, str)
        assert isinstance(version_id, str)
        entry = self._entry(bucket, key, version_id)
        payload = entry["BodyBytes"]
        assert isinstance(payload, bytes)
        if self.get_body_override is not None:
            payload = self.get_body_override
        body = (
            FailingBody()
            if self.fail_get_stream
            else ChunkedBody(payload, on_first_read=self.on_get_read)
        )
        self.last_get_body = body
        response = {
            "Body": body,
            "ContentLength": entry["ContentLength"],
            "ETag": entry["ETag"],
            "Metadata": dict(entry["Metadata"]),
            "SSEKMSKeyId": entry["SSEKMSKeyId"],
            "ServerSideEncryption": entry["ServerSideEncryption"],
            "VersionId": entry["VersionId"],
        }
        response.update(self.get_response_overrides)
        return response

    def list_object_versions(self, **kwargs: object) -> Mapping[str, object]:
        self.list_calls.append(dict(kwargs))
        bucket = kwargs["Bucket"]
        prefix = kwargs["Prefix"]
        assert isinstance(bucket, str)
        assert isinstance(prefix, str)
        versions = []
        for (candidate_bucket, key), entries in self._versions.items():
            if candidate_bucket != bucket or not key.startswith(prefix):
                continue
            versions.extend(
                {
                    "ETag": entry["ETag"],
                    "Key": key,
                    "Size": entry["ContentLength"],
                    "VersionId": entry["VersionId"],
                }
                for entry in entries
            )
        return {
            "DeleteMarkers": [
                marker
                for marker in self.delete_markers
                if marker["Key"].startswith(prefix)
            ],
            "IsTruncated": False,
            "Versions": versions,
        }

    def mutate(self, expected: S3ObjectVersion, drift: str) -> None:
        parsed = urlsplit(expected.uri)
        bucket = parsed.netloc
        key = parsed.path.removeprefix("/")
        if drift == "missing-version":
            entries = self._versions[(bucket, key)]
            entries[:] = [
                entry
                for entry in entries
                if entry["VersionId"] != expected.version_id
            ]
            return
        entry = self._entry(bucket, key, expected.version_id)
        overrides = entry["HeadOverrides"]
        assert isinstance(overrides, dict)
        if drift == "wrong-kms":
            overrides["SSEKMSKeyId"] = _OTHER_KMS_ARN
        elif drift == "wrong-size":
            overrides["ContentLength"] = expected.bytes + 1
        elif drift == "wrong-sha":
            overrides["Metadata"] = {
                **dict(entry["Metadata"]),
                "sha256": "f" * 64,
            }
        elif drift == "wrong-etag":
            overrides["ETag"] = f'"{"f" * 32}"'
        elif drift == "wrong-encryption":
            overrides["ServerSideEncryption"] = "AES256"
        else:
            raise AssertionError(f"unknown drift: {drift}")

    def install(
        self,
        *,
        bucket: str,
        key: str,
        payload: bytes,
        kms_key_arn: str,
        metadata: Mapping[str, str],
    ) -> str:
        response = self.put_object(
            Body=io.BytesIO(payload),
            Bucket=bucket,
            Key=key,
            Metadata=dict(metadata),
            SSEKMSKeyId=kms_key_arn,
            ServerSideEncryption="aws:kms",
        )
        return str(response["VersionId"])


def _artifact_request(tmp_path: Path, *, name: str = "artifact.bin") -> dict[str, object]:
    path = tmp_path / name
    path.write_bytes(b"claim-bearing-corpus\n")
    return {
        "bucket": CORPUS_BUCKET,
        "key": f"v2/builds/{_BUILD_ID}/{name}",
        "kms_key_arn": _KMS_ARN,
        "metadata": {"kind": "corpus-artifact"},
        "path": path,
    }


def _phase_receipt(
    objects: tuple[S3ObjectVersion, ...],
    *,
    phase: str = "final",
) -> PhaseReceipt:
    return PhaseReceipt(
        format=PHASE_RECEIPT_FORMAT,
        schema_version=1,
        build_id=_BUILD_ID,
        phase=phase,
        package_sha256="c" * 64,
        source_lock_sha256="d" * 64,
        objects=tuple(sorted(objects, key=lambda item: item.uri)),
    )


def test_publish_requires_version_id_kms_hash_bytes_etag_and_exact_head(tmp_path):
    s3 = VersionedFakeS3()
    request = _artifact_request(tmp_path)
    payload = request["path"].read_bytes()

    record = publish_exact_file(s3, **request)

    assert record == S3ObjectVersion(
        uri=f"s3://{CORPUS_BUCKET}/v2/builds/{_BUILD_ID}/artifact.bin",
        version_id="version-0001",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        etag=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        sse_algorithm="aws:kms",
        kms_key_arn=_KMS_ARN,
    )
    put = s3.put_calls[-1]
    assert put["ServerSideEncryption"] == "aws:kms"
    assert put["SSEKMSKeyId"] == _KMS_ARN
    assert put["ContentLength"] == len(payload)
    assert put["Metadata"] == {
        "kind": "corpus-artifact",
        "sha256": record.sha256,
    }
    assert not isinstance(put["Body"], (bytes, bytearray))
    assert s3.head_calls[-1] == {
        "Bucket": CORPUS_BUCKET,
        "Key": f"v2/builds/{_BUILD_ID}/artifact.bin",
        "VersionId": record.version_id,
    }
    verify_exact_object(s3, record)


def test_publish_hashes_and_uploads_one_pinned_descriptor_when_path_is_replaced(
    tmp_path,
):
    s3 = VersionedFakeS3()
    request = _artifact_request(tmp_path)
    path = request["path"]
    original = path.read_bytes()
    replacement = b"namespace replacement\n"

    def replace_source_name() -> None:
        candidate = path.with_suffix(".replacement")
        candidate.write_bytes(replacement)
        os.replace(candidate, path)

    s3.before_put_read = replace_source_name
    record = publish_exact_file(s3, **request)

    assert path.read_bytes() == replacement
    assert record.sha256 == hashlib.sha256(original).hexdigest()
    parsed = urlsplit(record.uri)
    entry = s3._entry(
        parsed.netloc,
        parsed.path.removeprefix("/"),
        record.version_id,
    )
    assert entry["BodyBytes"] == original


def test_publish_rejects_missing_version_id(tmp_path):
    s3 = VersionedFakeS3()
    s3.omit_version_id = True

    with pytest.raises(PublicationError, match="version"):
        publish_exact_file(s3, **_artifact_request(tmp_path))


def test_publish_rejects_symlink_source_before_s3_mutation(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"target\n")
    source = tmp_path / "source.bin"
    source.symlink_to(target)
    s3 = VersionedFakeS3()

    with pytest.raises(PublicationError, match="regular|symlink"):
        publish_exact_file(
            s3,
            bucket=CORPUS_BUCKET,
            key=f"v2/builds/{_BUILD_ID}/source.bin",
            path=source,
            kms_key_arn=_KMS_ARN,
            metadata={},
        )

    assert s3.put_calls == []


@pytest.mark.parametrize(
    "drift",
    (
        "missing-version",
        "wrong-kms",
        "wrong-size",
        "wrong-sha",
        "wrong-etag",
        "wrong-encryption",
    ),
)
def test_publish_and_download_reject_every_authority_drift(tmp_path, drift):
    s3 = VersionedFakeS3()
    record = publish_exact_file(s3, **_artifact_request(tmp_path))
    s3.mutate(record, drift)
    destination = tmp_path / f"download-{drift}.bin"

    with pytest.raises(PublicationError):
        download_exact_object(s3, record, destination)

    assert not destination.exists()


def test_download_streams_one_exact_version_fsyncs_and_never_overwrites(tmp_path):
    s3 = VersionedFakeS3()
    request = _artifact_request(tmp_path)
    payload = request["path"].read_bytes()
    record = publish_exact_file(s3, **request)
    destination = tmp_path / "cleanroom.bin"

    download_exact_object(s3, record, destination)

    assert destination.read_bytes() == payload
    assert s3.get_calls == [
        {
            "Bucket": CORPUS_BUCKET,
            "Key": f"v2/builds/{_BUILD_ID}/artifact.bin",
            "VersionId": record.version_id,
        }
    ]
    assert isinstance(s3.last_get_body, ChunkedBody)
    assert s3.last_get_body.closed
    destination.write_bytes(b"foreign\n")
    with pytest.raises(PublicationError, match="exists|exclusive"):
        download_exact_object(s3, record, destination)
    assert destination.read_bytes() == b"foreign\n"


def test_download_rejects_corrupt_stream_and_removes_only_its_failed_file(tmp_path):
    s3 = VersionedFakeS3()
    record = publish_exact_file(s3, **_artifact_request(tmp_path))
    s3.get_body_override = b"x" * record.bytes
    destination = tmp_path / "corrupt.bin"

    with pytest.raises(PublicationError, match="SHA-256|digest"):
        download_exact_object(s3, record, destination)

    assert not destination.exists()


def test_download_reheads_after_stream_and_rejects_source_mutation(tmp_path):
    s3 = VersionedFakeS3()
    record = publish_exact_file(s3, **_artifact_request(tmp_path))
    s3.on_get_read = lambda: s3.mutate(record, "wrong-kms")
    destination = tmp_path / "mutated.bin"

    with pytest.raises(PublicationError, match="KMS"):
        download_exact_object(s3, record, destination)

    assert not destination.exists()


@pytest.mark.parametrize(
    "drift",
    (
        "wrong-kms",
        "wrong-size",
        "wrong-sha",
        "wrong-etag",
        "wrong-encryption",
        "wrong-version",
    ),
)
def test_get_streaming_body_closes_on_every_authority_validation_failure(
    tmp_path,
    drift,
):
    s3 = VersionedFakeS3()
    record = publish_exact_file(s3, **_artifact_request(tmp_path))
    if drift == "wrong-kms":
        s3.get_response_overrides["SSEKMSKeyId"] = _OTHER_KMS_ARN
    elif drift == "wrong-size":
        s3.get_response_overrides["ContentLength"] = record.bytes + 1
    elif drift == "wrong-sha":
        s3.get_response_overrides["Metadata"] = {"sha256": "f" * 64}
    elif drift == "wrong-etag":
        s3.get_response_overrides["ETag"] = f'"{"f" * 32}"'
    elif drift == "wrong-encryption":
        s3.get_response_overrides["ServerSideEncryption"] = "AES256"
    else:
        s3.get_response_overrides["VersionId"] = "wrong-version"
    destination = tmp_path / f"get-{drift}.bin"

    with pytest.raises(PublicationError):
        download_exact_object(s3, record, destination)

    assert isinstance(s3.last_get_body, ChunkedBody)
    assert s3.last_get_body.closed
    assert not destination.exists()


def test_download_wraps_fake_client_stream_failure_and_cleans_destination(tmp_path):
    s3 = VersionedFakeS3()
    record = publish_exact_file(s3, **_artifact_request(tmp_path))
    s3.fail_get_stream = True
    destination = tmp_path / "interrupted.bin"

    with pytest.raises(PublicationError, match="download|stream"):
        download_exact_object(s3, record, destination)

    assert not destination.exists()


def test_phase_receipt_publish_is_canonical_no_overwrite_and_exactly_reusable(
    tmp_path,
):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    puts_before_receipt = len(s3.put_calls)

    first = publish_phase_receipt(
        s3,
        bucket=CORPUS_BUCKET,
        key=f"v2/builds/{_BUILD_ID}/phase-final.json",
        receipt=receipt,
        kms_key_arn=_KMS_ARN,
    )
    second = publish_phase_receipt(
        s3,
        bucket=CORPUS_BUCKET,
        key=f"v2/builds/{_BUILD_ID}/phase-final.json",
        receipt=receipt,
        kms_key_arn=_KMS_ARN,
    )

    assert second == first
    assert len(s3.put_calls) == puts_before_receipt + 1
    assert len(s3.list_calls) == 2
    assert s3.put_calls[-1]["IfNoneMatch"] == "*"
    receipt_entry = s3._entry(
        CORPUS_BUCKET,
        f"v2/builds/{_BUILD_ID}/phase-final.json",
        first.version_id,
    )
    assert receipt_entry["Metadata"] == {"sha256": first.sha256}


def test_phase_receipt_rejects_key_build_id_before_any_s3_mutation(tmp_path):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    calls_before = (len(s3.put_calls), len(s3.list_calls))

    with pytest.raises(PublicationError, match="build"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=f"v2/builds/{_OTHER_BUILD_ID}/phase-final.json",
            receipt=receipt,
            kms_key_arn=_KMS_ARN,
        )

    assert (len(s3.put_calls), len(s3.list_calls)) == calls_before


def test_phase_receipt_rejects_kms_mismatch_before_any_s3_mutation(tmp_path):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    calls_before = (len(s3.put_calls), len(s3.list_calls))

    with pytest.raises(PublicationError, match="KMS"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=f"v2/builds/{_BUILD_ID}/phase-final.json",
            receipt=receipt,
            kms_key_arn=_OTHER_KMS_ARN,
        )

    assert (len(s3.put_calls), len(s3.list_calls)) == calls_before


@pytest.mark.parametrize(
    "capability",
    ("missing-service-model", "missing-if-none-match"),
)
def test_phase_receipt_requires_if_none_match_operation_model_before_mutation(
    tmp_path,
    capability,
):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    if capability == "missing-service-model":
        s3.meta = object()
    else:
        s3.meta = FakeClientMeta({"Body", "Bucket", "Key"})
    calls_before = (len(s3.put_calls), len(s3.list_calls))

    with pytest.raises(PublicationError, match="IfNoneMatch|service model"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=f"v2/builds/{_BUILD_ID}/phase-final.json",
            receipt=receipt,
            kms_key_arn=_KMS_ARN,
        )

    assert (len(s3.put_calls), len(s3.list_calls)) == calls_before


def test_phase_receipt_rejects_any_conflicting_history_without_put_or_delete(
    tmp_path,
):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    key = f"v2/builds/{_BUILD_ID}/phase-final.json"
    exact = publish_phase_receipt(
        s3,
        bucket=CORPUS_BUCKET,
        key=key,
        receipt=receipt,
        kms_key_arn=_KMS_ARN,
    )
    foreign = b'{"foreign":true}\n'
    foreign_version = s3.install(
        bucket=CORPUS_BUCKET,
        key=key,
        payload=foreign,
        kms_key_arn=_KMS_ARN,
        metadata={"sha256": hashlib.sha256(foreign).hexdigest()},
    )
    put_count = len(s3.put_calls)

    with pytest.raises(PublicationError, match="conflict"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=key,
            receipt=receipt,
            kms_key_arn=_KMS_ARN,
        )

    assert len(s3.put_calls) == put_count
    assert s3._entry(CORPUS_BUCKET, key, exact.version_id)
    assert s3._entry(CORPUS_BUCKET, key, foreign_version)


def test_phase_receipt_conditional_put_cannot_overwrite_a_racing_winner(tmp_path):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    key = f"v2/builds/{_BUILD_ID}/phase-final.json"
    racing_payload = b'{"racing":"winner"}\n'
    s3.conditional_race = (
        racing_payload,
        _KMS_ARN,
        {"sha256": hashlib.sha256(racing_payload).hexdigest()},
    )

    with pytest.raises(PublicationError, match="upload|condition|conflict"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=key,
            receipt=receipt,
            kms_key_arn=_KMS_ARN,
        )

    versions = s3._versions[(CORPUS_BUCKET, key)]
    assert [entry["BodyBytes"] for entry in versions] == [racing_payload]


def test_phase_receipt_reuse_fails_closed_on_fake_client_stream_error(tmp_path):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    key = f"v2/builds/{_BUILD_ID}/phase-final.json"
    publish_phase_receipt(
        s3,
        bucket=CORPUS_BUCKET,
        key=key,
        receipt=receipt,
        kms_key_arn=_KMS_ARN,
    )
    s3.fail_get_stream = True
    put_count = len(s3.put_calls)

    with pytest.raises(PublicationError, match="conflict|stream"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=key,
            receipt=receipt,
            kms_key_arn=_KMS_ARN,
        )

    assert len(s3.put_calls) == put_count


def test_phase_receipt_rejects_delete_marker_history_without_put(tmp_path):
    s3 = VersionedFakeS3()
    artifact = publish_exact_file(s3, **_artifact_request(tmp_path))
    receipt = _phase_receipt((artifact,))
    key = f"v2/builds/{_BUILD_ID}/phase-final.json"
    s3.delete_markers.append(
        {"IsLatest": True, "Key": key, "VersionId": "delete-0001"}
    )
    put_count = len(s3.put_calls)

    with pytest.raises(PublicationError, match="delete marker|conflict"):
        publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=key,
            receipt=receipt,
            kms_key_arn=_KMS_ARN,
        )

    assert len(s3.put_calls) == put_count


def test_dynamic_task1_validated_kms_arn_is_used_without_a_hardcoded_key(tmp_path):
    s3 = VersionedFakeS3()
    request = _artifact_request(tmp_path)
    request["kms_key_arn"] = _OTHER_KMS_ARN

    record = publish_exact_file(s3, **request)

    assert record.kms_key_arn == _OTHER_KMS_ARN
    assert s3.put_calls[-1]["SSEKMSKeyId"] == _OTHER_KMS_ARN


def _published_cleanroom_fixture(
    tmp_path: Path,
    *,
    phase: str = "final",
    receipt_kms_arn: str = _KMS_ARN,
) -> tuple[VersionedFakeS3, S3ObjectVersion, dict[str, bytes]]:
    s3 = VersionedFakeS3()
    payloads = {
        "receipt.json": b"parallel-corpus-receipt\n",
        "shards/00000.bin": b"\x00\x01\x02\x03",
    }
    objects = []
    for relative, payload in payloads.items():
        source = tmp_path / relative.replace("/", "-")
        source.write_bytes(payload)
        objects.append(
            publish_exact_file(
                s3,
                bucket=CORPUS_BUCKET,
                key=f"v2/builds/{_BUILD_ID}/{relative}",
                path=source,
                kms_key_arn=_KMS_ARN,
                metadata={"kind": "parallel-corpus"},
            )
        )
    receipt = _phase_receipt(tuple(objects), phase=phase)
    receipt_key = f"v2/builds/{_BUILD_ID}/phase-{phase}.json"
    if receipt_kms_arn == _KMS_ARN:
        receipt_record = publish_phase_receipt(
            s3,
            bucket=CORPUS_BUCKET,
            key=receipt_key,
            receipt=receipt,
            kms_key_arn=receipt_kms_arn,
        )
    else:
        payload = phase_receipt_to_bytes(receipt)
        digest = hashlib.sha256(payload).hexdigest()
        version_id = s3.install(
            bucket=CORPUS_BUCKET,
            key=receipt_key,
            payload=payload,
            kms_key_arn=receipt_kms_arn,
            metadata={"sha256": digest},
        )
        entry = s3._entry(CORPUS_BUCKET, receipt_key, version_id)
        receipt_record = S3ObjectVersion(
            uri=f"s3://{CORPUS_BUCKET}/{receipt_key}",
            version_id=version_id,
            bytes=len(payload),
            sha256=digest,
            etag=str(entry["ETag"]).strip('"'),
            sse_algorithm="aws:kms",
            kms_key_arn=receipt_kms_arn,
        )
    return s3, receipt_record, payloads


def _cleanroom_args(
    receipt: S3ObjectVersion,
    destination: Path,
) -> list[str]:
    return [
        "--receipt-uri",
        receipt.uri,
        "--receipt-version-id",
        receipt.version_id,
        "--receipt-sha256",
        receipt.sha256,
        "--destination",
        str(destination),
        "--expected-build-id",
        _BUILD_ID,
        "--profile",
        "sbsandbox",
        "--region",
        "us-east-1",
    ]


def _cleanroom_control_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.control")


def test_cleanroom_cli_downloads_only_receipt_bound_versions_then_verifies(
    tmp_path,
):
    s3, receipt, payloads = _published_cleanroom_fixture(tmp_path)
    destination = tmp_path / "clean-room"
    verification_calls = []

    def verifier(root: Path, *, expected_build_id: str):
        verification_calls.append((Path(root), expected_build_id))
        assert {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(Path(root).rglob("*"))
            if path.is_file()
        } == payloads
        return {"build_id": expected_build_id}

    result = aws_corpus_cleanroom_verify.main(
        _cleanroom_args(receipt, destination),
        s3=s3,
        verifier=verifier,
    )

    assert result == 0
    assert verification_calls == [(destination, _BUILD_ID)]
    assert not (destination / ".phase-receipt.json").exists()
    control = _cleanroom_control_path(destination)
    phase_receipt = control / "phase-receipt.json"
    receipt_key = urlsplit(receipt.uri).path.removeprefix("/")
    receipt_entry = s3._entry(CORPUS_BUCKET, receipt_key, receipt.version_id)
    assert phase_receipt.read_bytes() == receipt_entry["BodyBytes"]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE(control.stat().st_mode) == 0o700
    assert stat.S_IMODE(phase_receipt.stat().st_mode) == 0o600
    assert all(
        isinstance(call.get("VersionId"), str) and call["VersionId"]
        for call in s3.get_calls
    )


@pytest.mark.parametrize("foreign_kind", ("file", "symlink"))
def test_cleanroom_cli_rejects_foreign_files_and_symlinks_before_s3(
    tmp_path,
    foreign_kind,
):
    s3, receipt, _payloads = _published_cleanroom_fixture(tmp_path)
    destination = tmp_path / "unclean-root"
    destination.mkdir()
    foreign = destination / "foreign"
    if foreign_kind == "file":
        foreign.write_bytes(b"foreign\n")
    else:
        foreign.symlink_to(tmp_path)
    calls_before = (len(s3.head_calls), len(s3.get_calls))

    with pytest.raises(PublicationError, match="empty|foreign|symlink"):
        aws_corpus_cleanroom_verify.main(
            _cleanroom_args(receipt, destination),
            s3=s3,
            verifier=lambda *_args, **_kwargs: None,
        )

    assert (len(s3.head_calls), len(s3.get_calls)) == calls_before


def test_cleanroom_cli_rejects_symlinked_destination_ancestor_before_s3(tmp_path):
    s3, receipt, _payloads = _published_cleanroom_fixture(tmp_path)
    real_parent = tmp_path / "real-parent"
    (real_parent / "nested").mkdir(parents=True)
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    destination = alias / "nested" / "clean-room"
    calls_before = (len(s3.head_calls), len(s3.get_calls))

    with pytest.raises(PublicationError, match="ancestor|symlink|real directory"):
        aws_corpus_cleanroom_verify.main(
            _cleanroom_args(receipt, destination),
            s3=s3,
            verifier=lambda *_args, **_kwargs: None,
        )

    assert (len(s3.head_calls), len(s3.get_calls)) == calls_before
    assert not (real_parent / "nested" / "clean-room").exists()


def test_cleanroom_cli_rejects_non_final_receipt_before_objects_or_verifier(
    tmp_path,
):
    s3, receipt, _payloads = _published_cleanroom_fixture(
        tmp_path,
        phase="source-verify",
    )
    destination = tmp_path / "non-final"
    gets_before = len(s3.get_calls)
    verification_calls = []

    with pytest.raises(PublicationError, match="final"):
        aws_corpus_cleanroom_verify.main(
            _cleanroom_args(receipt, destination),
            s3=s3,
            verifier=lambda *_args, **_kwargs: verification_calls.append(True),
        )

    assert len(s3.get_calls) == gets_before + 1
    assert verification_calls == []
    assert not (destination / "receipt.json").exists()


@pytest.mark.parametrize("mutation", ("namespace-replacement", "receipt-swap"))
def test_cleanroom_cli_rechecks_pinned_authority_after_verification(
    tmp_path,
    mutation,
):
    s3, receipt, _payloads = _published_cleanroom_fixture(tmp_path)
    destination = tmp_path / f"clean-{mutation}"
    displaced = tmp_path / f"displaced-{mutation}"

    def mutating_verifier(root: Path, *, expected_build_id: str):
        assert Path(root) == destination
        assert expected_build_id == _BUILD_ID
        if mutation == "namespace-replacement":
            destination.rename(displaced)
            destination.mkdir(mode=0o700)
        else:
            control = _cleanroom_control_path(destination)
            phase_path = control / "phase-receipt.json"
            if phase_path.exists():
                payload = phase_path.read_bytes()
                phase_path.rename(control / "displaced-receipt.json")
            else:
                control.mkdir(mode=0o700)
                payload = b"foreign phase receipt\n"
            phase_path.write_bytes(payload)
        return {"build_id": expected_build_id}

    with pytest.raises(PublicationError, match="authority|identity|binding"):
        aws_corpus_cleanroom_verify.main(
            _cleanroom_args(receipt, destination),
            s3=s3,
            verifier=mutating_verifier,
        )


def test_cleanroom_cli_requires_receipt_and_objects_share_dynamic_kms_key(tmp_path):
    s3, receipt, _payloads = _published_cleanroom_fixture(
        tmp_path,
        receipt_kms_arn=_OTHER_KMS_ARN,
    )
    destination = tmp_path / "kms-mismatch"

    with pytest.raises(PublicationError, match="KMS"):
        aws_corpus_cleanroom_verify.main(
            _cleanroom_args(receipt, destination),
            s3=s3,
            verifier=lambda *_args, **_kwargs: None,
        )

    assert not (destination / ".phase-receipt.json").exists()
