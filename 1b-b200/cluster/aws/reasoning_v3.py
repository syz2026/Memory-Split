"""Immutable S3 transfer and local verification for the frozen v3 corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from cluster.aws.readiness import (
    B200_PROFILE_PATH,
    HardwareAdmissionError,
    admit_protected_site,
    load_hardware_profile,
)
from cluster.aws.safeio import atomic_rename_noreplace

TRANSFER_FORMAT = "memorysplit-aws-corpus-transfer-v1"
CONTRACT_ID = "memorysplit-reasoning-dataset-v3"
RAW_TARGET_TOKENS = 8_169_455_616
VIRTUAL_RECEIPT_SHA256 = (
    "b1eabb1719f66876ab54cc0791b857ccdbbbddb0ffb8c5986ac2aaa7bf33b80d"
)
TRANSFER_MANIFEST_SHA256 = (
    "84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb"
)
EXPECTED_COMPOSITE_STREAM_SHA256 = {
    "dense_target_weights": (
        "917768b13ec169728cec51dc8294d118a113aee3c370ecd8c16ef0529f63f56e"
    ),
    "packed_targets": (
        "035ee111c329eb615c642eae9b9a7075314932ff8175e989aabb3317d6a4ef6f"
    ),
    "split90_target_weights": (
        "8a9c84c900e503d1742342b6a21092292c2968313087d0873e429b4268757144"
    ),
}
EXPECTED_OBJECTS = (
    {
        "bytes": 14_241_759_232,
        "path": "base/packed/targets.bin",
        "sha256": "dc0134131c57ec339997f9cee9c22f14a7414200671805c63d7cd7a7a3d5738d",
        "source_path": "corpus-build/memorysplit-135m-dataset/packed/targets.bin",
    },
    {
        "bytes": 1_531,
        "path": "base/receipt.json",
        "sha256": "783bfb6358eb08b4cf87140e397b6c54642a470fa9b8ca38974025e2d77ef08f",
        "source_path": "corpus-build/memorysplit-135m-dataset/receipt.json",
    },
    {
        "bytes": 7_120_879_616,
        "path": "base/sidecars/dense_target_weights.bin",
        "sha256": "c68ed29c50a57f76abed9cc4d09853b3c5daad3e532236c9d58e40cb72868624",
        "source_path": (
            "corpus-build/memorysplit-135m-dataset/"
            "sidecars/dense_target_weights.bin"
        ),
    },
    {
        "bytes": 7_120_879_616,
        "path": "base/sidecars/split90_target_weights.bin",
        "sha256": "bc6498ac91a2db9e237090d8305a4dcd003f42d1437efe4edf5b758ad281e07d",
        "source_path": (
            "corpus-build/memorysplit-135m-dataset/"
            "sidecars/split90_target_weights.bin"
        ),
    },
    {
        "bytes": 2_097_152_000,
        "path": "extension/packed/targets.bin",
        "sha256": "e09d08cdede2317ce0841264faa247322c2fe48ead38a3d81a26572a8effdd31",
        "source_path": (
            "corpus-build/memorysplit-reasoning-corpus-v3/packed/targets.bin"
        ),
    },
    {
        "bytes": 15_018,
        "path": "extension/receipt.json",
        "sha256": VIRTUAL_RECEIPT_SHA256,
        "source_path": "corpus-build/memorysplit-reasoning-corpus-v3/receipt.json",
    },
    {
        "bytes": 60_244_260,
        "path": "extension/records/manifest.bin",
        "sha256": "558b70fd0ad55ba2bf91b1efeea1b49583c9fe3d0286c71571fa868686a8acc8",
        "source_path": (
            "corpus-build/memorysplit-reasoning-corpus-v3/records/manifest.bin"
        ),
    },
    {
        "bytes": 1_048_576_000,
        "path": "extension/sidecars/shared_target_weights.bin",
        "sha256": "936bed85cfae5dea666e42a3f35f3a86ae1ac8ca6aa0bba49871980ef04df7e9",
        "source_path": (
            "corpus-build/memorysplit-reasoning-corpus-v3/"
            "sidecars/shared_target_weights.bin"
        ),
    },
    {
        "bytes": 1_662,
        "path": "locks/FROZEN.json",
        "sha256": "dd4f5083c90dd4b75e9b3c2da4db34c7f6299c7b845d15f960a13a3f9f3e849f",
        "source_path": "artifacts/reasoning-corpus-v3/FROZEN.json",
    },
    {
        "bytes": 1_328,
        "path": "locks/reasoning-pointer.json",
        "sha256": "12d302edcd0cb1eb04e07b39c3b441c96813a83fd6822e452ad54086da1b40e1",
        "source_path": "corpus-build/memorysplit-reasoning-corpus-v3.pointer.json",
    },
)
STAGE_RECEIPT = "memorysplit-stage-receipt.json"
_HEX = frozenset("0123456789abcdef")
_BUCKET_RE = re.compile(
    r"^(?!xn--)(?!sthree-)(?!amzn_s3_demo_)(?!.*\.\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_KMS_RE = re.compile(
    r"^(?:alias/[A-Za-z0-9/_-]{1,240}|"
    r"arn:aws(?:-[a-z]+)?:kms:[a-z0-9-]+:\d{12}:"
    r"(?:key/[0-9a-fA-F-]{36}|alias/[A-Za-z0-9/_-]{1,240}))$"
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "contract_id",
        "raw_target_tokens",
        "virtual_receipt_sha256",
        "composite_stream_sha256",
        "objects",
    }
)
_OBJECT_FIELDS = frozenset({"bytes", "path", "sha256", "source_path"})


class AwsCorpusError(ValueError):
    """Raised when the AWS transfer or staged corpus contract is violated."""


@dataclass(frozen=True)
class TransferObject:
    path: str
    source_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class TransferManifest:
    path: Path
    sha256: str
    objects: tuple[TransferObject, ...]
    composite_stream_sha256: Mapping[str, str]


@dataclass(frozen=True)
class StagedCorpus:
    root: Path
    manifest_sha256: str
    virtual_receipt_sha256: str
    raw_target_tokens: int
    composite_stream_sha256: Mapping[str, str]
    already_present: bool = False


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AwsCorpusError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AwsCorpusError(f"{field} must be a safe relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AwsCorpusError(f"{field} must be a safe relative POSIX path")
    return pure.as_posix()


def _read_json(path: Path, *, maximum: int = 1 << 20) -> Any:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise AwsCorpusError(f"JSON artifact is missing, unsafe, or oversized: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AwsCorpusError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AwsCorpusError(f"invalid JSON artifact: {path}") from error


def _sha256_file(
    path: Path,
    *,
    consumers: Sequence[hashlib._Hash] = (),  # type: ignore[name-defined]
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
            for consumer in consumers:
                consumer.update(chunk)
    return digest.hexdigest()


def load_transfer_manifest(path: Path | str) -> TransferManifest:
    manifest_path = Path(path)
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != TRANSFER_MANIFEST_SHA256:
        raise AwsCorpusError("AWS corpus manifest digest is not the frozen v3 manifest")
    raw = _read_json(manifest_path)
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_FIELDS:
        raise AwsCorpusError("AWS corpus manifest fields do not match the schema")
    if (
        raw["schema_version"] != 1
        or isinstance(raw["schema_version"], bool)
        or raw["format"] != TRANSFER_FORMAT
        or raw["contract_id"] != CONTRACT_ID
        or raw["raw_target_tokens"] != RAW_TARGET_TOKENS
        or raw["virtual_receipt_sha256"] != VIRTUAL_RECEIPT_SHA256
        or raw["composite_stream_sha256"] != EXPECTED_COMPOSITE_STREAM_SHA256
    ):
        raise AwsCorpusError("AWS corpus manifest identity is not frozen v3")
    objects = raw["objects"]
    if not isinstance(objects, list) or not objects:
        raise AwsCorpusError("AWS corpus manifest must contain transfer objects")
    parsed = []
    for value in objects:
        if not isinstance(value, dict) or set(value) != _OBJECT_FIELDS:
            raise AwsCorpusError("AWS corpus object fields do not match the schema")
        size = value["bytes"]
        digest = value["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _is_sha(digest)
        ):
            raise AwsCorpusError("AWS corpus object size or digest is invalid")
        parsed.append(
            TransferObject(
                path=_regular_relative(value["path"], field="object path"),
                source_path=_regular_relative(
                    value["source_path"],
                    field="object source_path",
                ),
                bytes=size,
                sha256=digest,
            )
        )
    paths = [item.path for item in parsed]
    sources = [item.source_path for item in parsed]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise AwsCorpusError("AWS corpus objects must be uniquely path-sorted")
    if len(sources) != len(set(sources)):
        raise AwsCorpusError("AWS corpus source paths must be unique")
    if [
        {
            "bytes": item.bytes,
            "path": item.path,
            "sha256": item.sha256,
            "source_path": item.source_path,
        }
        for item in parsed
    ] != list(EXPECTED_OBJECTS):
        raise AwsCorpusError("AWS corpus objects differ from the frozen v3 transfer set")
    return TransferManifest(
        path=manifest_path.resolve(),
        sha256=manifest_sha256,
        objects=tuple(parsed),
        composite_stream_sha256=dict(raw["composite_stream_sha256"]),
    )


def parse_s3_uri(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or any(char in value for char in ("\x00", "\n", "\r")):
        raise AwsCorpusError("S3 URI must be a safe string")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise AwsCorpusError("S3 URI must use s3://bucket/prefix without query data")
    bucket = parsed.netloc
    if not _BUCKET_RE.fullmatch(bucket) or re.fullmatch(
        r"\d{1,3}(?:\.\d{1,3}){3}",
        bucket,
    ):
        raise AwsCorpusError("S3 bucket name is invalid or unsupported")
    prefix = parsed.path.lstrip("/").rstrip("/")
    if prefix:
        _regular_relative(prefix, field="S3 key prefix")
    return bucket, prefix


def _s3_object_uri(base_uri: str, relative: str) -> str:
    bucket, prefix = parse_s3_uri(base_uri)
    key = f"{prefix}/{relative}" if prefix else relative
    return f"s3://{bucket}/{key}"


def _expected_namespace(manifest: TransferManifest) -> set[str]:
    return {item.path for item in manifest.objects}


def _verify_namespace(
    root: Path,
    manifest: TransferManifest,
    *,
    allow_stage_receipt: bool,
) -> None:
    if not root.is_dir() or root.is_symlink():
        raise AwsCorpusError(f"staged corpus root is missing or unsafe: {root}")
    found: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AwsCorpusError(f"staged corpus contains a symlink: {path}")
        if path.is_file():
            found.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise AwsCorpusError(f"staged corpus contains a special file: {path}")
    expected = _expected_namespace(manifest)
    if allow_stage_receipt:
        expected.add(STAGE_RECEIPT)
    if found != expected:
        raise AwsCorpusError(
            "staged corpus namespace differs; "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )


def _verify_json_bindings(root: Path, manifest: TransferManifest) -> None:
    base = _read_json(root / "base" / "receipt.json")
    extension = _read_json(root / "extension" / "receipt.json")
    pointer = _read_json(root / "locks" / "reasoning-pointer.json")
    frozen = _read_json(root / "locks" / "FROZEN.json")
    if not all(isinstance(value, dict) for value in (base, extension, pointer, frozen)):
        raise AwsCorpusError("staged corpus receipts and locks must be JSON objects")
    if (
        not isinstance(base, dict)
        or base.get("contract_id") != "memorysplit-parallel-corpus-v2"
        or base.get("raw_target_tokens") != 7_120_879_616
        or not isinstance(base.get("task4_publication"), dict)
        or base["task4_publication"].get("receipt_sha256")
        != extension.get("base_corpus", {}).get("receipt_sha256")
    ):
        raise AwsCorpusError("staged v2 flat receipt does not bind the v3 prefix")
    if (
        extension.get("contract_id") != CONTRACT_ID
        or extension.get("composite", {}).get("raw_target_tokens")
        != RAW_TARGET_TOKENS
        or extension.get("composite", {}).get("stream_sha256")
        != EXPECTED_COMPOSITE_STREAM_SHA256
        or pointer.get("expected_receipt_sha256") != VIRTUAL_RECEIPT_SHA256
        or pointer.get("launch_gate_status") != "frozen"
        or pointer.get("expected_composite_stream_sha256")
        != EXPECTED_COMPOSITE_STREAM_SHA256
        or frozen.get("receipt_sha256") != VIRTUAL_RECEIPT_SHA256
        or frozen.get("pointer_sha256")
        != next(
            item.sha256
            for item in manifest.objects
            if item.path == "locks/reasoning-pointer.json"
        )
        or frozen.get("composite_stream_sha256")
        != EXPECTED_COMPOSITE_STREAM_SHA256
    ):
        raise AwsCorpusError("staged v3 receipt, pointer, or publication lock differs")


def verify_staged_corpus(
    root: Path | str,
    manifest_path: Path | str,
    *,
    allow_stage_receipt: bool = True,
) -> StagedCorpus:
    corpus_root = Path(root)
    manifest = load_transfer_manifest(manifest_path)
    _verify_namespace(
        corpus_root,
        manifest,
        allow_stage_receipt=allow_stage_receipt,
    )
    by_path = {item.path: item for item in manifest.objects}
    packed = hashlib.sha256()
    dense = hashlib.sha256()
    split90 = hashlib.sha256()
    composite_consumers = {
        "base/packed/targets.bin": (packed,),
        "extension/packed/targets.bin": (packed,),
        "base/sidecars/dense_target_weights.bin": (dense,),
        "base/sidecars/split90_target_weights.bin": (split90,),
        "extension/sidecars/shared_target_weights.bin": (dense, split90),
    }
    for relative, item in by_path.items():
        path = corpus_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.bytes
            or _sha256_file(path, consumers=composite_consumers.get(relative, ()))
            != item.sha256
        ):
            raise AwsCorpusError(f"staged corpus object differs: {relative}")
    computed = {
        "dense_target_weights": dense.hexdigest(),
        "packed_targets": packed.hexdigest(),
        "split90_target_weights": split90.hexdigest(),
    }
    if computed != EXPECTED_COMPOSITE_STREAM_SHA256:
        raise AwsCorpusError("staged corpus composite stream digest differs")
    _verify_json_bindings(corpus_root, manifest)
    if allow_stage_receipt:
        receipt = _read_json(corpus_root / STAGE_RECEIPT)
        expected_receipt = {
            "composite_stream_sha256": EXPECTED_COMPOSITE_STREAM_SHA256,
            "contract_id": CONTRACT_ID,
            "format": "memorysplit-aws-stage-receipt-v1",
            "manifest_sha256": manifest.sha256,
            "raw_target_tokens": RAW_TARGET_TOKENS,
            "schema_version": 1,
            "virtual_receipt_sha256": VIRTUAL_RECEIPT_SHA256,
        }
        if receipt != expected_receipt:
            raise AwsCorpusError("AWS stage receipt differs from the frozen identity")
    return StagedCorpus(
        root=corpus_root.resolve(),
        manifest_sha256=manifest.sha256,
        virtual_receipt_sha256=VIRTUAL_RECEIPT_SHA256,
        raw_target_tokens=RAW_TARGET_TOKENS,
        composite_stream_sha256=dict(EXPECTED_COMPOSITE_STREAM_SHA256),
    )


def admit_reasoning_v3_site(
    *,
    site_evidence: object,
    asserted_authority: object,
    profile_path: Path | str = B200_PROFILE_PATH,
    transfer_manifest_sha256: str | None = None,
    virtual_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Admit one exact B200 site and bind it to the frozen v3 corpus identity.

    Hardware admission alone is not enough: the same node must also be pinned
    to this corpus so P5, B300, or a differently seeded site can never supply
    evidence for a reasoning-v3 cell.
    """

    profile = load_hardware_profile(profile_path)
    if profile.dataset_contract_id != CONTRACT_ID:
        raise HardwareAdmissionError(
            f"hardware profile is bound to {profile.dataset_contract_id!r}, "
            f"not the frozen corpus contract {CONTRACT_ID!r}"
        )
    expected_manifest = TRANSFER_MANIFEST_SHA256
    expected_receipt = VIRTUAL_RECEIPT_SHA256
    if transfer_manifest_sha256 is None:
        transfer_manifest_sha256 = expected_manifest
    if virtual_receipt_sha256 is None:
        virtual_receipt_sha256 = expected_receipt
    if transfer_manifest_sha256 != expected_manifest:
        raise HardwareAdmissionError(
            "site admission transfer manifest digest is not the frozen v3 manifest"
        )
    if virtual_receipt_sha256 != expected_receipt:
        raise HardwareAdmissionError(
            "site admission corpus receipt digest is not the frozen v3 receipt"
        )
    receipt = admit_protected_site(
        profile=profile,
        site_evidence=site_evidence,
        asserted_authority=asserted_authority,
    )
    receipt.update(
        {
            "dataset_contract_id": CONTRACT_ID,
            "raw_target_tokens": RAW_TARGET_TOKENS,
            "transfer_manifest_sha256": transfer_manifest_sha256,
            "virtual_receipt_sha256": virtual_receipt_sha256,
        }
    )
    return receipt


def verify_upload_sources(
    repository_root: Path | str,
    manifest_path: Path | str,
) -> TransferManifest:
    repository = Path(repository_root).resolve()
    manifest = load_transfer_manifest(manifest_path)
    for item in manifest.objects:
        path = repository / item.source_path
        try:
            path.resolve().relative_to(repository)
        except ValueError as error:
            raise AwsCorpusError("AWS corpus source escapes the repository") from error
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.bytes
            or _sha256_file(path) != item.sha256
        ):
            raise AwsCorpusError(f"AWS corpus upload source differs: {item.source_path}")
    return manifest


def _default_runner(command: Sequence[str]):
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _result(result: object) -> tuple[int, str, str]:
    return (
        int(result.returncode),
        str(getattr(result, "stdout", "") or ""),
        str(getattr(result, "stderr", "") or ""),
    )


def upload_to_s3(
    repository_root: Path | str,
    manifest_path: Path | str,
    s3_uri: str,
    *,
    kms_key_id: str,
    apply: bool = False,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> dict[str, Any]:
    """Verify and upload every object; matching existing objects are idempotent."""

    if not _KMS_RE.fullmatch(kms_key_id):
        raise AwsCorpusError("kms_key_id must be a KMS key/alias ARN or alias/name")
    manifest = verify_upload_sources(repository_root, manifest_path)
    repository = Path(repository_root).resolve()
    plans = []
    for item in manifest.objects:
        uri = _s3_object_uri(s3_uri, item.path)
        bucket, key = parse_s3_uri(uri)
        head = [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            "--query",
            "Metadata.sha256",
            "--output",
            "text",
        ]
        upload = [
            "aws",
            "s3",
            "cp",
            str(repository / item.source_path),
            uri,
            "--only-show-errors",
            "--no-progress",
            "--sse",
            "aws:kms",
            "--sse-kms-key-id",
            kms_key_id,
            "--checksum-algorithm",
            "SHA256",
            "--metadata",
            f"sha256={item.sha256},memorysplit-contract={CONTRACT_ID}",
        ]
        plans.append({"head": head, "path": item.path, "upload": upload})
    report: dict[str, Any] = {
        "applied": apply,
        "commands": plans,
        "manifest_sha256": manifest.sha256,
        "object_count": len(plans),
        "s3_uri": s3_uri.rstrip("/"),
        "uploaded": 0,
        "verified_existing": 0,
    }
    if not apply:
        return report
    for plan, item in zip(plans, manifest.objects):
        returncode, stdout, stderr = _result(runner(plan["head"]))
        if returncode == 0:
            if stdout.strip() != item.sha256:
                raise AwsCorpusError(f"S3 object already exists with another identity: {item.path}")
            report["verified_existing"] += 1
            continue
        if not any(marker in f"{stdout}\n{stderr}" for marker in ("404", "Not Found", "NoSuchKey")):
            raise AwsCorpusError(f"cannot inspect S3 object {item.path}: {stderr.strip()}")
        returncode, _, stderr = _result(runner(plan["upload"]))
        if returncode:
            raise AwsCorpusError(f"S3 upload failed for {item.path}: {stderr.strip()}")
        returncode, stdout, stderr = _result(runner(plan["head"]))
        if returncode or stdout.strip() != item.sha256:
            raise AwsCorpusError(f"S3 post-upload identity failed for {item.path}: {stderr.strip()}")
        report["uploaded"] += 1
    return report


def _write_stage_receipt(root: Path, manifest: TransferManifest) -> None:
    payload = {
        "composite_stream_sha256": EXPECTED_COMPOSITE_STREAM_SHA256,
        "contract_id": CONTRACT_ID,
        "format": "memorysplit-aws-stage-receipt-v1",
        "manifest_sha256": manifest.sha256,
        "raw_target_tokens": RAW_TARGET_TOKENS,
        "schema_version": 1,
        "virtual_receipt_sha256": VIRTUAL_RECEIPT_SHA256,
    }
    path = root / STAGE_RECEIPT
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    parent_fd = os.open(
        destination.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        atomic_rename_noreplace(
            parent_fd,
            temporary.name,
            parent_fd,
            destination.name,
        )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def stage_from_s3(
    s3_uri: str,
    destination_root: Path | str,
    manifest_path: Path | str,
    *,
    apply: bool = False,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> dict[str, Any]:
    """Download, hash, and atomically publish one immutable local corpus tree."""

    parse_s3_uri(s3_uri)
    destination = Path(destination_root)
    manifest = load_transfer_manifest(manifest_path)
    commands = [
        [
            "aws",
            "s3",
            "cp",
            _s3_object_uri(s3_uri, item.path),
            str(destination.parent / f".{destination.name}.aws-stage-{os.getpid()}" / item.path),
            "--only-show-errors",
            "--no-progress",
            "--checksum-mode",
            "ENABLED",
        ]
        for item in manifest.objects
    ]
    report: dict[str, Any] = {
        "applied": apply,
        "commands": commands,
        "destination": str(destination),
        "manifest_sha256": manifest.sha256,
        "object_count": len(commands),
        "s3_uri": s3_uri.rstrip("/"),
    }
    if destination.exists() or destination.is_symlink():
        evidence = verify_staged_corpus(destination, manifest.path)
        report["already_present"] = True
        report["evidence"] = evidence
        return report
    if not apply:
        report["already_present"] = False
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.aws-stage-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"AWS staging path already exists: {temporary}")
    try:
        temporary.mkdir()
        for item, command in zip(manifest.objects, commands):
            target = temporary / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            returncode, _, stderr = _result(runner(command))
            if returncode:
                raise AwsCorpusError(f"S3 download failed for {item.path}: {stderr.strip()}")
        verify_staged_corpus(
            temporary,
            manifest.path,
            allow_stage_receipt=False,
        )
        _write_stage_receipt(temporary, manifest)
        verify_staged_corpus(temporary, manifest.path)
        _publish_no_replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    evidence = verify_staged_corpus(destination, manifest.path)
    report["already_present"] = False
    report["evidence"] = evidence
    return report
