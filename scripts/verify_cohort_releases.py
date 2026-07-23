#!/usr/bin/env python3
"""Verify Illumina and AWS handoffs as one frozen five-seed cohort."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import BinaryIO
import unicodedata
import zipfile

import yaml


SCHEMA_VERSION = 1
COHORT_ID = "memorysplit-confirmatory-v2-360m-n5"
ILLUMINA_PROVIDER = "illumina-usfc-prd"
AWS_PROVIDER = "aws-p5.48xlarge"
EXPECTED_SEEDS = {
    ILLUMINA_PROVIDER: (0,),
    AWS_PROVIDER: (1, 2, 3, 4),
}
EXPECTED_ARMS = ("dense", "split90")
SNAPSHOT_STEPS = (1_358, 3_396, 6_791, 10_187, 13_582)
ASSIGNMENT_PATH = "configs/cohort-assignment-v2.json"
CORPUS_IDENTITY_PATH = "configs/reasoning-dataset-v2.json"
EVALUATION_IDENTITY_PATH = "configs/preregistration-v2.yaml"
METADATA_PATH = "RELEASE-METADATA.json"
SUMS_PATH = "SHA256SUMS"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
_AWS_PROFILE_CONTRACT = {
    "schema_version": 1,
    "provider": AWS_PROVIDER,
    "instance_type": "p5.48xlarge",
    "purchase_model": "on_demand",
    "gpu": {
        "model": "NVIDIA H100 80GB",
        "allocated": 8,
        "seed_train_groups": [4, 4],
    },
    "cpu": {"vcpus": 192},
    "memory_bytes": 2_199_023_255_552,
    "storage": {
        "instance_store_devices": 8,
        "instance_store_device_bytes": 3_840_000_000_000,
        "scratch_root": "/mnt/memorysplit",
        "durable_uri_env": "MS_S3_ROOT",
    },
    "runtime": {
        "ami_id_env": "MS_AWS_AMI_ID",
        "container_digest_env": "MS_CONTAINER_DIGEST",
    },
    "assigned_seeds": [1, 2, 3, 4],
}


class VerificationError(ValueError):
    """A fail-closed cohort-release validation error."""


@dataclass(frozen=True)
class _RunCell:
    seed: int
    arm: str
    run_id: str
    train_corpus: str


@dataclass(frozen=True)
class _VerifiedRelease:
    provider: str
    seeds: tuple[int, ...]
    release_id: str
    archive_sha256: str
    source_commit: str
    assignment: Mapping[str, object]
    assignment_sha256: str
    corpus_sha256: str
    evaluation_sha256: str
    cells: tuple[_RunCell, ...]


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise VerificationError(f"invalid command line: {message}")


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(
        self,
        node: yaml.nodes.MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise VerificationError("run config keys must be strings")
            if key in result:
                raise VerificationError("run config contains a duplicate key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise VerificationError(f"{name} must be a lowercase SHA-256")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise VerificationError(f"{name} must be an integer")
    return value


def _strict_object(
    value: object,
    fields: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise VerificationError(f"{name} must be an object")
    if set(value) != fields:
        raise VerificationError(f"{name} fields are not exact")
    return value


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise VerificationError(f"JSON contains forbidden constant {value}")


def _json(content: bytes, name: str) -> Mapping[str, object]:
    try:
        text = content.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise VerificationError(f"{name} must contain one JSON object")
    return value


def _safe_archive_name(name: object) -> str:
    value = _string(name, "archive.path")
    if value != PurePosixPath(value).name or "\\" in value or value in {".", ".."}:
        raise VerificationError("archive.path must be a safe basename")
    return value


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not os.O_NOFOLLOW:
        raise VerificationError("secure descriptor opening is unsupported")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _regular_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not os.O_NOFOLLOW:
        raise VerificationError("secure descriptor opening is unsupported")
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_parent(receipt: Path, role: str) -> int:
    try:
        descriptor = os.open(receipt.parent or Path("."), _directory_flags())
    except OSError as error:
        raise VerificationError(
            f"{role} receipt parent is unavailable or unsafe"
        ) from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise VerificationError(f"{role} receipt parent is not a directory")
    return descriptor


def _open_regular_at(directory_fd: int, name: str, label: str) -> int:
    if (
        not isinstance(name, str)
        or not name
        or name != PurePosixPath(name).name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise VerificationError(f"{label} must be a safe sibling filename")
    try:
        descriptor = os.open(
            name,
            _regular_flags(),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise VerificationError(f"{label} is unavailable or unsafe") from error
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        os.close(descriptor)
        raise VerificationError(f"{label} must be a singly-linked regular file")
    return descriptor


def _assert_descriptor_names_entry(
    directory_fd: int,
    name: str,
    descriptor: int,
    label: str,
) -> None:
    pinned = os.fstat(descriptor)
    try:
        current = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise VerificationError(f"{label} path changed after opening") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != pinned.st_dev
        or current.st_ino != pinned.st_ino
        or current.st_size != pinned.st_size
        or current.st_nlink != 1
        or pinned.st_nlink != 1
    ):
        raise VerificationError(f"{label} path was replaced after opening")


def _read_control_file(
    directory_fd: int,
    name: str,
    label: str,
) -> bytes:
    descriptor = _open_regular_at(directory_fd, name, label)
    try:
        _assert_descriptor_names_entry(
            directory_fd,
            name,
            descriptor,
            label,
        )
        before = os.fstat(descriptor)
        if before.st_size > _MAX_CONTROL_FILE_BYTES:
            raise VerificationError(f"{label} exceeds the control-file limit")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
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
        ):
            raise VerificationError(f"{label} changed during verification")
        _assert_descriptor_names_entry(
            directory_fd,
            name,
            descriptor,
            label,
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _receipt(content: bytes, expected_provider: str) -> Mapping[str, object]:
    common_fields = {
        "schema_version",
        "release_id",
        "provider",
        "archive",
        "source",
        "members_sha256",
    }
    if expected_provider == AWS_PROVIDER:
        receipt_fields = frozenset(
            common_fields
            | {
                "package_format_version",
                "seed_assignment",
                "cohort_assignment",
                "cohort_assignment_sha256",
                "profile",
                "profile_sha256",
                "environment",
                "dataset_pointer",
                "dataset_pointer_sha256",
                "config_sha256",
            }
        )
    else:
        receipt_fields = frozenset(common_fields)
    value = _strict_object(
        _json(content, f"{expected_provider} release receipt"),
        receipt_fields,
        f"{expected_provider} release receipt",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise VerificationError("release receipt schema_version is unsupported")
    if expected_provider == AWS_PROVIDER and (
        type(value["package_format_version"]) is not int
        or value["package_format_version"] != 1
    ):
        raise VerificationError("AWS package_format_version is unsupported")
    if value["provider"] != expected_provider:
        raise VerificationError("release receipt provider is incorrect")
    release_id = _string(value["release_id"], "release_id")
    archive = _strict_object(
        value["archive"],
        frozenset({"path", "sha256", "bytes"}),
        "release archive binding",
    )
    _safe_archive_name(archive["path"])
    _hash(archive["sha256"], "archive.sha256")
    size = _integer(archive["bytes"], "archive.bytes")
    if size < 1:
        raise VerificationError("archive.bytes must be positive")
    source = _source(
        value["source"],
        "release receipt source",
        expected_provider=expected_provider,
    )
    _hash(value["members_sha256"], "members_sha256")
    release_prefix = "r1" if expected_provider == ILLUMINA_PROVIDER else "aws-p5-r1"
    if release_id != f"{release_prefix}-{value['members_sha256'][:16]}":
        raise VerificationError("release_id is not bound to members_sha256")
    archive_prefix = (
        "ms-illumina-r1" if expected_provider == ILLUMINA_PROVIDER else "ms-aws-p5-r1"
    )
    expected_archive_name = f"{archive_prefix}-{value['members_sha256'][:16]}.zip"
    if archive["path"] != expected_archive_name:
        raise VerificationError(
            "archive.path is not bound to provider and members_sha256"
        )
    if expected_provider == AWS_PROVIDER:
        _seed_assignment(
            value["seed_assignment"],
            expected_provider=expected_provider,
            expected_seeds=EXPECTED_SEEDS[expected_provider],
            label="AWS receipt seed_assignment",
        )
        _runtime_attested_environment(
            value["environment"],
            label="AWS receipt environment",
            expected_profile_sha256=_hash(
                value["profile_sha256"],
                "AWS receipt profile_sha256",
            ),
        )
        for field in (
            "cohort_assignment_sha256",
            "profile_sha256",
            "dataset_pointer_sha256",
        ):
            _hash(value[field], field)
        for field, expected_path, digest_field in (
            (
                "cohort_assignment",
                ASSIGNMENT_PATH,
                "cohort_assignment_sha256",
            ),
            (
                "profile",
                "cluster/profiles/aws-p5.48xlarge.json",
                "profile_sha256",
            ),
            (
                "dataset_pointer",
                "DATASET-POINTER-AWS.json",
                "dataset_pointer_sha256",
            ),
        ):
            binding = _strict_object(
                value[field],
                frozenset({"path", "sha256"}),
                f"AWS receipt {field}",
            )
            if binding["path"] != expected_path:
                raise VerificationError(f"AWS receipt {field} path is incorrect")
            if (
                _hash(
                    binding["sha256"],
                    f"AWS receipt {field} SHA-256",
                )
                != value[digest_field]
            ):
                raise VerificationError(f"AWS receipt {field} hash bindings disagree")
        config_hashes = value["config_sha256"]
        if not isinstance(config_hashes, Mapping) or any(
            not isinstance(key, str) for key in config_hashes
        ):
            raise VerificationError("AWS receipt config_sha256 must be an object")
        expected_config_paths = {
            f"configs/360m-v2/{arm}-s{seed}.yaml"
            for seed in EXPECTED_SEEDS[AWS_PROVIDER]
            for arm in EXPECTED_ARMS
        }
        if set(config_hashes) != expected_config_paths:
            raise VerificationError("AWS receipt config_sha256 inventory is not exact")
        for path, digest in config_hashes.items():
            _hash(digest, f"AWS receipt config hash for {path}")
    if source["dirty"] is not False:
        raise VerificationError("release receipt source must be clean")
    return value


def _source(
    value: object,
    name: str,
    *,
    expected_provider: str,
) -> Mapping[str, object]:
    fields = {"commit", "dirty"}
    if expected_provider == AWS_PROVIDER:
        fields.add("tree")
    source = _strict_object(
        value,
        frozenset(fields),
        name,
    )
    commit = source["commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise VerificationError(f"{name} commit must be a full lowercase ID")
    if not isinstance(source["dirty"], bool):
        raise VerificationError(f"{name} dirty must be Boolean")
    if expected_provider == AWS_PROVIDER:
        tree = source["tree"]
        if not isinstance(tree, str) or _OBJECT_ID.fullmatch(tree) is None:
            raise VerificationError(f"{name} tree must be a full lowercase ID")
    return source


def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or (info.is_dir() and not name.endswith("/"))
        or (not info.is_dir() and name.endswith("/"))
    ):
        raise VerificationError("ZIP contains an unsafe member name")
    core = name[:-1] if info.is_dir() else name
    parts = core.split("/")
    if (
        not core
        or any(part in {"", ".", ".."} for part in parts)
        or parts[0].endswith(":")
    ):
        raise VerificationError("ZIP contains an unsafe member path")

    mode = info.external_attr >> 16
    if info.is_dir():
        if not stat.S_ISDIR(mode):
            raise VerificationError("ZIP directory mode is not a directory")
    elif not stat.S_ISREG(mode):
        raise VerificationError("ZIP member is not a regular file")
    if info.flag_bits & 0x1:
        raise VerificationError("encrypted ZIP members are forbidden")
    return "/".join(unicodedata.normalize("NFC", part) for part in parts)


@dataclass
class _MemberTrieNode:
    kind: str | None
    children: dict[str, "_MemberTrieNode"]


def _validate_member_topology(infos: list[zipfile.ZipInfo]) -> None:
    root = _MemberTrieNode(kind="directory", children={})
    for info in infos:
        canonical = _safe_member_name(info)
        parts = canonical.split("/")
        node = root
        for index, part in enumerate(parts):
            if node.kind == "file":
                raise VerificationError("ZIP member topology has a file collision")
            child = node.children.setdefault(
                part,
                _MemberTrieNode(kind=None, children={}),
            )
            is_last = index == len(parts) - 1
            if not is_last:
                node = child
                continue
            kind = "directory" if info.is_dir() else "file"
            if child.kind is not None:
                raise VerificationError(
                    "ZIP member topology has a normalized path collision"
                )
            if kind == "file" and child.children:
                raise VerificationError("ZIP member topology has a file collision")
            child.kind = kind


def _member_bytes(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    label: str,
) -> bytes:
    if info.file_size > _MAX_CONTROL_FILE_BYTES:
        raise VerificationError(f"{label} exceeds the control-file limit")
    try:
        with archive.open(info, mode="r") as stream:
            content = stream.read(_MAX_CONTROL_FILE_BYTES + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise VerificationError(f"{label} could not be read") from error
    if len(content) > _MAX_CONTROL_FILE_BYTES:
        raise VerificationError(f"{label} exceeds the control-file limit")
    return content


def _parse_sums(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("SHA256SUMS must be UTF-8") from error
    if not text or not text.endswith("\n") or "\r" in text:
        raise VerificationError("SHA256SUMS must be newline-terminated")
    result: dict[str, str] = {}
    previous: str | None = None
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise VerificationError("SHA256SUMS contains a malformed entry")
        digest = line[:64]
        name = line[66:]
        _hash(digest, "SHA256SUMS digest")
        dummy = zipfile.ZipInfo(name)
        dummy.create_system = 3
        dummy.external_attr = (stat.S_IFREG | 0o644) << 16
        _safe_member_name(dummy)
        if name == SUMS_PATH:
            raise VerificationError("SHA256SUMS cannot hash itself")
        if name in result:
            raise VerificationError("SHA256SUMS contains a duplicate path")
        if previous is not None and name <= previous:
            raise VerificationError("SHA256SUMS paths must be sorted")
        result[name] = digest
        previous = name
    return result


def _inspect_zip(
    stream: BinaryIO,
    *,
    receipt: Mapping[str, object],
    expected_provider: str,
    expected_seeds: tuple[int, ...],
) -> _VerifiedRelease:
    try:
        with zipfile.ZipFile(stream, mode="r") as archive:
            infos = archive.infolist()
            if not infos:
                raise VerificationError("release ZIP is empty")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise VerificationError("release ZIP contains duplicate members")
            _validate_member_topology(infos)
            by_name = {info.filename: info for info in infos}
            if SUMS_PATH not in by_name or by_name[SUMS_PATH].is_dir():
                raise VerificationError("release ZIP is missing SHA256SUMS")
            sums_content = _member_bytes(
                archive,
                by_name[SUMS_PATH],
                SUMS_PATH,
            )
            sums = _parse_sums(sums_content)
            regular_names = {
                info.filename
                for info in infos
                if not info.is_dir() and info.filename != SUMS_PATH
            }
            if set(sums) != regular_names:
                raise VerificationError(
                    "SHA256SUMS does not cover every regular ZIP member"
                )
            if _sha256_bytes(sums_content) != receipt["members_sha256"]:
                raise VerificationError(
                    "members_sha256 does not match internal SHA256SUMS"
                )
            for name, expected_hash in sums.items():
                try:
                    with archive.open(by_name[name], mode="r") as member:
                        actual_hash, actual_size = _hash_stream(member)
                except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                    raise VerificationError(
                        f"ZIP member could not be authenticated: {name}"
                    ) from error
                if actual_size != by_name[name].file_size:
                    raise VerificationError(f"ZIP member size changed: {name}")
                if actual_hash != expected_hash:
                    raise VerificationError(
                        f"internal SHA-256 mismatch for ZIP member: {name}"
                    )

            profile_path = (
                "cluster/profiles/illumina-usfc-prd.json"
                if expected_provider == ILLUMINA_PROVIDER
                else "cluster/profiles/aws-p5.48xlarge.json"
            )
            required = {
                METADATA_PATH,
                ASSIGNMENT_PATH,
                CORPUS_IDENTITY_PATH,
                EVALUATION_IDENTITY_PATH,
                profile_path,
            }
            if expected_provider == AWS_PROVIDER:
                required.add("DATASET-POINTER-AWS.json")
                if "requirements-aws-p5.lock" in regular_names:
                    raise VerificationError(
                        "AWS release must not claim a static environment lock"
                    )
            missing = required - regular_names
            if missing:
                raise VerificationError(
                    f"release ZIP is missing required member: {sorted(missing)[0]}"
                )
            metadata_content = _member_bytes(
                archive,
                by_name[METADATA_PATH],
                METADATA_PATH,
            )
            assignment_content = _member_bytes(
                archive,
                by_name[ASSIGNMENT_PATH],
                ASSIGNMENT_PATH,
            )
            corpus_content = _member_bytes(
                archive,
                by_name[CORPUS_IDENTITY_PATH],
                CORPUS_IDENTITY_PATH,
            )
            evaluation_content = _member_bytes(
                archive,
                by_name[EVALUATION_IDENTITY_PATH],
                EVALUATION_IDENTITY_PATH,
            )
            metadata = _metadata(
                metadata_content,
                expected_provider=expected_provider,
                expected_seeds=expected_seeds,
                receipt=receipt,
                sums=sums,
                by_name=by_name,
            )
            _provider_profile(
                _member_bytes(
                    archive,
                    by_name[profile_path],
                    profile_path,
                ),
                expected_provider=expected_provider,
                path=profile_path,
            )
            if expected_provider == AWS_PROVIDER:
                _dataset_pointer(
                    _member_bytes(
                        archive,
                        by_name["DATASET-POINTER-AWS.json"],
                        "DATASET-POINTER-AWS.json",
                    )
                )
            assignment = _assignment(assignment_content)
            cells = _run_cells(
                archive,
                by_name=by_name,
                expected_seeds=expected_seeds,
            )
    except zipfile.BadZipFile as error:
        raise VerificationError("archive is not a valid ZIP") from error

    if metadata["seed_assignment"]["cohort_id"] != assignment["cohort_id"]:
        raise VerificationError("metadata cohort_id disagrees with assignment")
    if tuple(assignment["provider_seeds"][expected_provider]) != expected_seeds:
        raise VerificationError("archive provider seeds disagree with assignment")
    return _VerifiedRelease(
        provider=expected_provider,
        seeds=expected_seeds,
        release_id=str(receipt["release_id"]),
        archive_sha256=str(receipt["archive"]["sha256"]),
        source_commit=str(receipt["source"]["commit"]),
        assignment=assignment,
        assignment_sha256=_sha256_bytes(assignment_content),
        corpus_sha256=_sha256_bytes(corpus_content),
        evaluation_sha256=_sha256_bytes(evaluation_content),
        cells=cells,
    )


def _provider_profile(
    content: bytes,
    *,
    expected_provider: str,
    path: str,
) -> None:
    value = _json(content, path)
    if not isinstance(value, Mapping):
        raise VerificationError("provider profile must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise VerificationError("provider profile keys must be strings")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise VerificationError("provider profile schema_version is unsupported")
    if value.get("provider") != expected_provider:
        raise VerificationError("provider profile provider is incorrect")
    if "profile_id" in value and value["profile_id"] != expected_provider:
        raise VerificationError("provider profile profile_id is incorrect")
    if expected_provider == AWS_PROVIDER:
        _exact_contract(
            value,
            _AWS_PROFILE_CONTRACT,
            label="AWS P5 profile",
        )


def _exact_contract(value: object, expected: object, *, label: str) -> None:
    if isinstance(expected, Mapping):
        actual = _strict_object(
            value,
            frozenset(expected),
            label,
        )
        for field, expected_value in expected.items():
            _exact_contract(
                actual[field],
                expected_value,
                label=f"{label}.{field}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise VerificationError(f"{label} does not match the strict contract")
        for index, (item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _exact_contract(
                item,
                expected_item,
                label=f"{label}[{index}]",
            )
        return
    if type(value) is not type(expected) or value != expected:
        raise VerificationError(f"{label} does not match the strict contract")


def _runtime_attested_environment(
    value: object,
    *,
    label: str,
    expected_profile_sha256: str,
) -> Mapping[str, object]:
    expected = {
        "mode": "runtime_attested",
        "profile_sha256": expected_profile_sha256,
        "container_image_digest_env": "MS_CONTAINER_DIGEST",
        "container_image_digest_pattern": "^sha256:[0-9a-f]{64}$",
        "runtime_environment_receipt": {
            "required_at_launch": True,
            "authentication": "aws_instance_identity_document_pkcs7",
            "required_fields": [
                "schema_version",
                "profile_sha256",
                "container_image_digest",
                "aws_instance_identity_document",
                "aws_instance_identity_pkcs7",
            ],
        },
    }
    _exact_contract(value, expected, label=label)
    environment = value
    if not isinstance(environment, Mapping):
        raise VerificationError(f"{label} must be an object")
    return environment


def _dataset_pointer(content: bytes) -> None:
    value = _strict_object(
        _json(content, "DATASET-POINTER-AWS.json"),
        frozenset(
            {
                "schema_version",
                "provider",
                "dataset_id",
                "durable_uri_env",
                "receipt_relative_path",
                "full_corpus_in_release",
            }
        ),
        "DATASET-POINTER-AWS.json",
    )
    expected = {
        "schema_version": 1,
        "provider": AWS_PROVIDER,
        "dataset_id": "memorysplit-parallel-corpus-v2",
        "durable_uri_env": "MS_S3_ROOT",
        "receipt_relative_path": "dataset/corpus-receipt.json",
        "full_corpus_in_release": False,
    }
    for field, expected_value in expected.items():
        actual = value[field]
        if type(expected_value) in {bool, int} and type(actual) is not type(
            expected_value
        ):
            raise VerificationError(f"dataset pointer {field} has wrong type")
        if actual != expected_value:
            raise VerificationError(f"dataset pointer {field} is incorrect")


def _seed_assignment(
    value: object,
    *,
    expected_provider: str,
    expected_seeds: tuple[int, ...],
    label: str,
) -> Mapping[str, object]:
    assignment = _strict_object(
        value,
        frozenset({"cohort_id", "provider", "seeds", "arms"}),
        label,
    )
    if assignment["cohort_id"] != COHORT_ID:
        raise VerificationError(f"{label} cohort_id is incorrect")
    if assignment["provider"] != expected_provider:
        raise VerificationError(f"{label} provider is incorrect")
    if (
        not isinstance(assignment["seeds"], list)
        or tuple(assignment["seeds"]) != expected_seeds
        or any(type(seed) is not int for seed in assignment["seeds"])
    ):
        raise VerificationError(f"{label} seeds are incorrect")
    if assignment["arms"] != list(EXPECTED_ARMS):
        raise VerificationError(f"{label} arms are incorrect")
    return assignment


def _path_hash_binding(
    value: object,
    *,
    expected_path: str,
    sums: Mapping[str, str],
    label: str,
) -> Mapping[str, object]:
    binding = _strict_object(
        value,
        frozenset({"path", "sha256"}),
        label,
    )
    if binding["path"] != expected_path:
        raise VerificationError(f"{label} path is incorrect")
    digest = _hash(binding["sha256"], f"{label} SHA-256")
    if sums.get(expected_path) != digest:
        raise VerificationError(f"{label} is not bound to its ZIP member")
    return binding


def _metadata_members(
    value: object,
    *,
    expected_provider: str,
    sums: Mapping[str, str],
    by_name: Mapping[str, zipfile.ZipInfo],
) -> None:
    if not isinstance(value, list):
        raise VerificationError("release metadata members must be an array")
    fields = {"path", "bytes", "sha256", "git_blob"}
    if expected_provider == AWS_PROVIDER:
        fields.add("git_mode")
    member_paths: set[str] = set()
    for raw in value:
        row = _strict_object(
            raw,
            frozenset(fields),
            "release metadata member",
        )
        path = _string(row["path"], "release metadata member path")
        if path in member_paths:
            raise VerificationError("release metadata has a duplicate member")
        member_paths.add(path)
        if path not in sums or path not in by_name:
            raise VerificationError("release metadata names an absent member")
        if (
            _integer(row["bytes"], f"member bytes for {path}")
            != by_name[path].file_size
        ):
            raise VerificationError("release metadata member size is incorrect")
        if _hash(row["sha256"], f"member SHA-256 for {path}") != sums[path]:
            raise VerificationError("release metadata member hash is incorrect")
        git_blob = row["git_blob"]
        if not isinstance(git_blob, str) or _OBJECT_ID.fullmatch(git_blob) is None:
            raise VerificationError("release metadata git_blob is invalid")
        if expected_provider == AWS_PROVIDER:
            git_mode = row["git_mode"]
            if git_mode not in {"100644", "100755"}:
                raise VerificationError("release metadata git_mode is invalid")
            expected_permissions = 0o755 if git_mode == "100755" else 0o644
            actual_permissions = stat.S_IMODE(by_name[path].external_attr >> 16)
            if actual_permissions != expected_permissions:
                raise VerificationError(
                    "release metadata git_mode disagrees with ZIP mode"
                )
    expected_metadata_members = set(sums) - {METADATA_PATH}
    if member_paths != expected_metadata_members:
        raise VerificationError("release metadata member inventory is incomplete")


def _metadata(
    content: bytes,
    *,
    expected_provider: str,
    expected_seeds: tuple[int, ...],
    receipt: Mapping[str, object],
    sums: Mapping[str, str],
    by_name: Mapping[str, zipfile.ZipInfo],
) -> Mapping[str, object]:
    if expected_provider == ILLUMINA_PROVIDER:
        fields = frozenset(
            {
                "schema_version",
                "provider",
                "source",
                "profile_sha256",
                "preregistration_sha256",
                "environment_hashes",
                "members",
                "seed_assignment",
            }
        )
    else:
        fields = frozenset(
            {
                "schema_version",
                "package_format_version",
                "provider",
                "source",
                "seed_assignment",
                "cohort_assignment",
                "profile",
                "environment",
                "dataset_pointer",
                "config_sha256",
                "members",
            }
        )
    value = _strict_object(
        _json(content, METADATA_PATH),
        fields,
        METADATA_PATH,
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise VerificationError("release metadata schema_version is unsupported")
    if expected_provider == AWS_PROVIDER and (
        type(value["package_format_version"]) is not int
        or value["package_format_version"] != 1
    ):
        raise VerificationError(
            "AWS release metadata package_format_version is unsupported"
        )
    if value["provider"] != expected_provider:
        raise VerificationError("release metadata provider is incorrect")
    source = _source(
        value["source"],
        "release metadata source",
        expected_provider=expected_provider,
    )
    if source != receipt["source"] or source["dirty"] is not False:
        raise VerificationError("release metadata source binding is inconsistent")
    assignment = _seed_assignment(
        value["seed_assignment"],
        expected_provider=expected_provider,
        expected_seeds=expected_seeds,
        label="release metadata seed_assignment",
    )
    _metadata_members(
        value["members"],
        expected_provider=expected_provider,
        sums=sums,
        by_name=by_name,
    )

    profile_path = (
        "cluster/profiles/illumina-usfc-prd.json"
        if expected_provider == ILLUMINA_PROVIDER
        else "cluster/profiles/aws-p5.48xlarge.json"
    )
    if expected_provider == ILLUMINA_PROVIDER:
        profile_hash = _hash(value["profile_sha256"], "profile_sha256")
        if sums.get(profile_path) != profile_hash:
            raise VerificationError("profile_sha256 does not bind the provider profile")
        preregistration_hash = _hash(
            value["preregistration_sha256"],
            "preregistration_sha256",
        )
        if sums.get(EVALUATION_IDENTITY_PATH) != preregistration_hash:
            raise VerificationError(
                "preregistration_sha256 does not bind the preregistration member"
            )
        environments = value["environment_hashes"]
        if (
            not isinstance(environments, Mapping)
            or not environments
            or any(not isinstance(key, str) for key in environments)
        ):
            raise VerificationError("environment_hashes must be a non-empty object")
        for path, digest_value in environments.items():
            digest = _hash(digest_value, f"environment hash for {path}")
            if sums.get(path) != digest:
                raise VerificationError("environment hash is not bound to a member")
        return value

    cohort_binding = _path_hash_binding(
        value["cohort_assignment"],
        expected_path=ASSIGNMENT_PATH,
        sums=sums,
        label="AWS cohort_assignment",
    )
    profile_binding = _path_hash_binding(
        value["profile"],
        expected_path=profile_path,
        sums=sums,
        label="AWS profile",
    )
    environment_contract = _runtime_attested_environment(
        value["environment"],
        label="AWS environment",
        expected_profile_sha256=str(profile_binding["sha256"]),
    )
    dataset_binding = _path_hash_binding(
        value["dataset_pointer"],
        expected_path="DATASET-POINTER-AWS.json",
        sums=sums,
        label="AWS dataset_pointer",
    )
    config_hashes = value["config_sha256"]
    expected_config_paths = {
        f"configs/360m-v2/{arm}-s{seed}.yaml"
        for seed in expected_seeds
        for arm in EXPECTED_ARMS
    }
    if (
        not isinstance(config_hashes, Mapping)
        or any(not isinstance(key, str) for key in config_hashes)
        or set(config_hashes) != expected_config_paths
    ):
        raise VerificationError("AWS metadata config_sha256 inventory is not exact")
    for path, digest_value in config_hashes.items():
        digest = _hash(digest_value, f"AWS metadata config hash for {path}")
        if sums.get(path) != digest:
            raise VerificationError(
                "AWS metadata config hash is not bound to its member"
            )

    if receipt["package_format_version"] != value["package_format_version"]:
        raise VerificationError("AWS package format bindings disagree")
    if receipt["seed_assignment"] != assignment:
        raise VerificationError("AWS seed assignment bindings disagree")
    if receipt["cohort_assignment_sha256"] != cohort_binding["sha256"]:
        raise VerificationError("AWS cohort assignment bindings disagree")
    if receipt["cohort_assignment"] != cohort_binding:
        raise VerificationError("AWS cohort assignment objects disagree")
    if receipt["profile_sha256"] != profile_binding["sha256"]:
        raise VerificationError("AWS profile bindings disagree")
    if receipt["profile"] != profile_binding:
        raise VerificationError("AWS profile binding objects disagree")
    if receipt["environment"] != environment_contract:
        raise VerificationError("AWS runtime environment contracts disagree")
    if receipt["dataset_pointer_sha256"] != dataset_binding["sha256"]:
        raise VerificationError("AWS dataset pointer bindings disagree")
    if receipt["dataset_pointer"] != dataset_binding:
        raise VerificationError("AWS dataset pointer binding objects disagree")
    if receipt["config_sha256"] != config_hashes:
        raise VerificationError("AWS config hash bindings disagree")
    return value


def _assignment(content: bytes) -> Mapping[str, object]:
    value = _strict_object(
        _json(content, ASSIGNMENT_PATH),
        frozenset(
            {
                "schema_version",
                "cohort_id",
                "model_parameters",
                "optimizer_steps",
                "provider_seeds",
                "raw_target_tokens",
                "targets_per_update",
            }
        ),
        ASSIGNMENT_PATH,
    )
    expected_scalars = {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "raw_target_tokens": 7_120_879_616,
        "targets_per_update": 524_288,
    }
    for field, expected in expected_scalars.items():
        actual = value[field]
        if isinstance(expected, int) and type(actual) is not int:
            raise VerificationError(f"cohort assignment {field} has wrong type")
        if actual != expected:
            raise VerificationError(f"cohort assignment {field} is incorrect")
    providers = _strict_object(
        value["provider_seeds"],
        frozenset({ILLUMINA_PROVIDER, AWS_PROVIDER}),
        "cohort assignment provider_seeds",
    )
    for provider, expected in EXPECTED_SEEDS.items():
        seeds = providers[provider]
        if (
            not isinstance(seeds, list)
            or any(type(seed) is not int for seed in seeds)
            or tuple(seeds) != expected
        ):
            raise VerificationError(
                f"cohort assignment seeds are incorrect for {provider}"
            )
    illumina = set(providers[ILLUMINA_PROVIDER])
    aws = set(providers[AWS_PROVIDER])
    if illumina & aws or illumina | aws != set(range(5)):
        raise VerificationError(
            "cohort assignment must have exact disjoint five-seed coverage"
        )
    if (
        value["targets_per_update"] * value["optimizer_steps"]
        != value["raw_target_tokens"]
    ):
        raise VerificationError("cohort assignment token math is inconsistent")
    return value


_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "cohort_id",
        "run_id",
        "condition",
        "seed",
        "model",
        "ctx",
        "train_corpus",
        "sidecar_name",
        "out_dir",
        "micro_batch_size",
        "tokens_per_step",
        "max_steps",
        "total_tokens",
        "lr",
        "warmup_steps",
        "weight_decay",
        "compile",
        "device",
        "log_every",
        "eval_every",
        "snapshot_steps",
        "ckpt_minutes",
    }
)


def _yaml(content: bytes, name: str) -> Mapping[str, object]:
    try:
        value = yaml.load(content.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise VerificationError(f"{name} must be valid strict UTF-8 YAML") from error
    return _strict_object(value, _RUN_FIELDS, name)


def _run_cells(
    archive: zipfile.ZipFile,
    *,
    by_name: Mapping[str, zipfile.ZipInfo],
    expected_seeds: tuple[int, ...],
) -> tuple[_RunCell, ...]:
    prefix = "configs/360m-v2/"
    config_names = {
        name
        for name, info in by_name.items()
        if name.startswith(prefix) and not info.is_dir()
    }
    expected_names = {
        f"{prefix}{arm}-s{seed}.yaml"
        for seed in expected_seeds
        for arm in EXPECTED_ARMS
    }
    if config_names != expected_names:
        raise VerificationError(
            "archive does not contain its exact assigned Dense/Split90 configs"
        )
    cells = []
    for name in sorted(config_names):
        raw = _yaml(_member_bytes(archive, by_name[name], name), name)
        seed = _integer(raw["seed"], f"{name} seed")
        arm = _string(raw["condition"], f"{name} condition")
        if seed not in expected_seeds or arm not in EXPECTED_ARMS:
            raise VerificationError(f"{name} has an unassigned seed or arm")
        expected_path = f"{prefix}{arm}-s{seed}.yaml"
        if name != expected_path:
            raise VerificationError("run config path disagrees with its semantics")
        expected_values: dict[str, object] = {
            "schema_version": 2,
            "cohort_id": COHORT_ID,
            "run_id": f"memorysplit-v2-360m-s{seed}-{arm}",
            "model": "d360m",
            "ctx": 1024,
            "train_corpus": "dataset/corpus-receipt.json",
            "sidecar_name": (
                "dense_target_weights" if arm == "dense" else "split90_target_weights"
            ),
            "out_dir": f"runs/seed-{seed}/{arm}",
            "micro_batch_size": 8,
            "tokens_per_step": 524_288,
            "max_steps": 13_582,
            "total_tokens": 7_120_879_616,
            "lr": 0.001,
            "warmup_steps": 300,
            "weight_decay": 0.1,
            "compile": True,
            "device": "cuda",
            "log_every": 20,
            "eval_every": 250,
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "ckpt_minutes": 30,
        }
        for field, expected in expected_values.items():
            actual = raw[field]
            if isinstance(expected, bool):
                valid_type = type(actual) is bool
            elif isinstance(expected, int):
                valid_type = type(actual) is int
            elif isinstance(expected, float):
                valid_type = type(actual) is float
            elif isinstance(expected, list):
                valid_type = isinstance(actual, list) and all(
                    type(item) is int for item in actual
                )
            else:
                valid_type = isinstance(actual, str)
            if not valid_type or actual != expected:
                raise VerificationError(f"{name} has incorrect {field}")
        cells.append(
            _RunCell(
                seed=seed,
                arm=arm,
                run_id=str(raw["run_id"]),
                train_corpus=str(raw["train_corpus"]),
            )
        )
    semantic_cells = {(cell.seed, cell.arm) for cell in cells}
    expected_cells = {(seed, arm) for seed in expected_seeds for arm in EXPECTED_ARMS}
    if semantic_cells != expected_cells or len(cells) != len(semantic_cells):
        raise VerificationError("archive run configs are not complete and unique")
    return tuple(cells)


def _verify_release(
    receipt_path: Path | str,
    *,
    expected_provider: str,
) -> _VerifiedRelease:
    path = Path(receipt_path)
    if path.name in {"", ".", ".."} or "\\" in path.name:
        raise VerificationError("release receipt path is unsafe")
    parent_fd = _open_parent(path, expected_provider)
    try:
        receipt_content = _read_control_file(
            parent_fd,
            path.name,
            f"{expected_provider} release receipt",
        )
        receipt = _receipt(receipt_content, expected_provider)
        archive_binding = receipt["archive"]
        archive_name = _safe_archive_name(archive_binding["path"])
        sidecar_name = f"{archive_name}.sha256"
        expected_sidecar = f"{archive_binding['sha256']}  {archive_name}\n".encode(
            "ascii"
        )
        actual_sidecar = _read_control_file(
            parent_fd,
            sidecar_name,
            f"{expected_provider} external SHA-256",
        )
        if actual_sidecar != expected_sidecar:
            raise VerificationError(
                f"{expected_provider} external SHA-256 is inconsistent"
            )

        archive_fd = _open_regular_at(
            parent_fd,
            archive_name,
            f"{expected_provider} archive",
        )
        try:
            _assert_descriptor_names_entry(
                parent_fd,
                archive_name,
                archive_fd,
                f"{expected_provider} archive",
            )
            stream = os.fdopen(archive_fd, "rb", closefd=True)
            archive_fd = -1
            with stream:
                before = os.fstat(stream.fileno())
                archive_hash, archive_size = _hash_stream(stream)
                after_hash = os.fstat(stream.fileno())
                if archive_hash != archive_binding["sha256"]:
                    raise VerificationError(
                        f"{expected_provider} archive SHA-256 mismatch"
                    )
                if archive_size != archive_binding["bytes"]:
                    raise VerificationError(
                        f"{expected_provider} archive byte count mismatch"
                    )
                stream.seek(0)
                verified = _inspect_zip(
                    stream,
                    receipt=receipt,
                    expected_provider=expected_provider,
                    expected_seeds=EXPECTED_SEEDS[expected_provider],
                )
                after_zip = os.fstat(stream.fileno())
                fingerprints = [
                    (
                        item.st_dev,
                        item.st_ino,
                        item.st_size,
                        item.st_mtime_ns,
                        item.st_ctime_ns,
                    )
                    for item in (before, after_hash, after_zip)
                ]
                if len(set(fingerprints)) != 1:
                    raise VerificationError(
                        f"{expected_provider} archive changed during verification"
                    )
                _assert_descriptor_names_entry(
                    parent_fd,
                    archive_name,
                    stream.fileno(),
                    f"{expected_provider} archive",
                )
                return verified
        finally:
            if archive_fd >= 0:
                os.close(archive_fd)
    finally:
        os.close(parent_fd)


def verify_cohort_releases(
    *,
    illumina_release: Path | str,
    aws_release: Path | str,
) -> dict[str, object]:
    """Authenticate both releases and return one canonical cohort decision."""

    illumina = _verify_release(
        illumina_release,
        expected_provider=ILLUMINA_PROVIDER,
    )
    aws = _verify_release(
        aws_release,
        expected_provider=AWS_PROVIDER,
    )
    if illumina.source_commit != aws.source_commit:
        raise VerificationError("release source commits differ")
    if illumina.assignment != aws.assignment:
        raise VerificationError("release cohort assignments differ semantically")
    if illumina.assignment_sha256 != aws.assignment_sha256:
        raise VerificationError("release cohort assignment hashes differ")
    if illumina.corpus_sha256 != aws.corpus_sha256:
        raise VerificationError("release corpus identities differ")
    if illumina.evaluation_sha256 != aws.evaluation_sha256:
        raise VerificationError("release evaluation identities differ")

    all_cells = illumina.cells + aws.cells
    run_ids = [cell.run_id for cell in all_cells]
    if len(run_ids) != len(set(run_ids)):
        raise VerificationError("release run IDs overlap")
    cell_keys = [(cell.seed, cell.arm) for cell in all_cells]
    expected_cells = {(seed, arm) for seed in range(5) for arm in EXPECTED_ARMS}
    if len(cell_keys) != len(set(cell_keys)) or set(cell_keys) != expected_cells:
        raise VerificationError(
            "releases do not provide exact disjoint Dense/Split90 coverage"
        )
    train_corpora = {cell.train_corpus for cell in all_cells}
    if train_corpora != {"dataset/corpus-receipt.json"}:
        raise VerificationError("release run configs use different corpus identities")

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "cohort_id": COHORT_ID,
        "source_commit": illumina.source_commit,
        "cohort_assignment_sha256": illumina.assignment_sha256,
        "corpus_identity_sha256": illumina.corpus_sha256,
        "evaluation_identity_sha256": illumina.evaluation_sha256,
        "arms": list(EXPECTED_ARMS),
        "complete_cohort": list(range(5)),
        "illumina": {
            "provider": illumina.provider,
            "release_id": illumina.release_id,
            "archive_sha256": illumina.archive_sha256,
            "seeds": list(illumina.seeds),
        },
        "aws": {
            "provider": aws.provider,
            "release_id": aws.release_id,
            "archive_sha256": aws.archive_sha256,
            "seeds": list(aws.seeds),
        },
    }


def _emit(value: Mapping[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _StrictArgumentParser(
        description="Verify split-provider MemorySplit cohort releases.",
        add_help=False,
    )
    parser.add_argument("--illumina", required=True)
    parser.add_argument("--aws", required=True)
    try:
        arguments = parser.parse_args(argv)
        report = verify_cohort_releases(
            illumina_release=arguments.illumina,
            aws_release=arguments.aws,
        )
        code = 0
    except VerificationError as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": {
                "code": "COHORT_RELEASES_REJECTED",
                "message": str(error),
            },
        }
        code = 2
    except Exception:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": {
                "code": "COHORT_VERIFIER_INTERNAL_ERROR",
                "message": "unexpected local verification failure",
            },
        }
        code = 70
    _emit(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
