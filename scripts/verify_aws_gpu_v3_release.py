#!/usr/bin/env python3
"""Verify one AWS GPU v3 handoff against its selected closed profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import package_aws_gpu_handoff as _package
import verify_cohort_releases as _archive_core


SCHEMA_VERSION = 1
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "package_format_version",
        "release_id",
        "provider",
        "selected_profile_id",
        "archive",
        "source",
        "seed_assignment",
        "cohort_assignment",
        "preregistration",
        "hardware_amendment",
        "profile",
        "environment",
        "dataset_pointer",
        "container_base_lock",
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "hardware_amendment_sha256",
        "profile_sha256",
        "dataset_pointer_sha256",
        "container_base_lock_sha256",
        "config_sha256",
        "contract_locks",
        "members_sha256",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "package_format_version",
        "provider",
        "selected_profile_id",
        "source",
        "seed_assignment",
        "cohort_assignment",
        "preregistration",
        "hardware_amendment",
        "profile",
        "environment",
        "dataset_pointer",
        "container_base_lock",
        "config_sha256",
        "contract_locks",
        "members",
    }
)


class VerificationError(ValueError):
    """A fail-closed AWS GPU v3 release verification error."""


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise VerificationError(f"invalid command line: {message}")


def _fail(message: str) -> None:
    raise VerificationError(message)


def _strict_object(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{label} must be an object")
    if set(value) != fields:
        _fail(f"{label} fields are not exact")
    return dict(value)


def _json(data: bytes, *, label: str) -> dict[str, object]:
    try:
        return _package._core._load_json_object(data, label=label)
    except _package.PackageError as error:
        raise VerificationError(str(error)) from error


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _profile_relative(source: Path, profile: Path | str) -> str:
    try:
        return _package._profile_relative(source, profile)
    except _package.PackageError as error:
        raise VerificationError(str(error)) from error


def _validate_source(value: object) -> dict[str, object]:
    source = _strict_object(
        value,
        frozenset({"commit", "dirty", "tree"}),
        label="release source",
    )
    if (
        not isinstance(source["commit"], str)
        or _OBJECT_ID.fullmatch(source["commit"]) is None
        or not isinstance(source["tree"], str)
        or _OBJECT_ID.fullmatch(source["tree"]) is None
        or source["dirty"] is not False
    ):
        _fail("release source must bind one clean commit and tree")
    return source


def _validate_seed_assignment(
    value: object,
    *,
    provider: str,
) -> dict[str, object]:
    assignment = _strict_object(
        value,
        frozenset({"cohort_id", "provider", "seeds", "arms"}),
        label="seed assignment",
    )
    if assignment != {
        "cohort_id": _package.COHORT_ID,
        "provider": provider,
        "seeds": list(_package.SEEDS),
        "arms": list(_package.ARMS),
    }:
        _fail("seed assignment does not bind all ten pairs to one profile")
    return assignment


def _validate_binding(
    value: object,
    *,
    path: str,
    sums: Mapping[str, str],
    label: str,
) -> dict[str, object]:
    binding = _strict_object(
        value,
        frozenset({"path", "sha256"}),
        label=label,
    )
    digest = _require_hash(binding["sha256"], label=f"{label}.sha256")
    if binding["path"] != path or sums.get(path) != digest:
        _fail(f"{label} does not bind its exact archive member")
    return binding


def _validate_container_binding(
    value: object,
    *,
    sums: Mapping[str, str],
) -> dict[str, object]:
    binding = _strict_object(
        value,
        frozenset({"path", "sha256", "base_image", "base_digest"}),
        label="container base lock binding",
    )
    digest = _require_hash(
        binding["sha256"],
        label="container base lock binding.sha256",
    )
    expected_base_digest = (
        "sha256:3bfc0b4c9561561b4cd578cb78611ec6258e1791854000b58a279d5880afc7d6"
    )
    expected_base_image = (
        "public.ecr.aws/deep-learning-containers/"
        f"pytorch:2.12.1-cu130-amzn2023@{expected_base_digest}"
    )
    if (
        binding["path"] != _package.CONTAINER_LOCK_PATH
        or sums.get(_package.CONTAINER_LOCK_PATH) != digest
        or binding["base_digest"] != expected_base_digest
        or binding["base_image"] != expected_base_image
    ):
        _fail("container base lock binding is stale or mutable")
    return binding


def _validate_environment(value: object, *, profile_sha256: str) -> dict[str, object]:
    expected = _package._runtime_environment_contract(profile_sha256)
    if not _package._same_typed_value(value, expected):
        _fail("runtime environment contract is not bound to the selected profile")
    return dict(expected)


def _validate_contract_locks(
    value: object,
    *,
    payload: Mapping[str, bytes],
) -> dict[str, dict[str, object]]:
    if (
        not isinstance(value, dict)
        or set(value) != set(_package.CONTRACT_GROUPS)
        or any(not isinstance(key, str) for key in value)
    ):
        _fail("contract lock inventory is not exact")
    expected = _package._contract_locks(dict(payload))
    if not _package._same_typed_value(value, expected):
        _fail("selection/fleet/lifecycle/canary/container lock is stale")
    return expected


def _validate_member_rows(
    value: object,
    *,
    expected_members: frozenset[str],
    sums: Mapping[str, str],
    by_name: Mapping[str, zipfile.ZipInfo],
) -> None:
    if not isinstance(value, list):
        _fail("release metadata members must be an array")
    paths: list[str] = []
    for raw in value:
        row = _strict_object(
            raw,
            frozenset({"path", "bytes", "sha256", "git_blob", "git_mode"}),
            label="release metadata member",
        )
        path = row["path"]
        if not isinstance(path, str) or path not in expected_members:
            _fail("release metadata names an unexpected member")
        digest = _require_hash(row["sha256"], label=f"member hash for {path}")
        size = row["bytes"]
        git_blob = row["git_blob"]
        git_mode = row["git_mode"]
        if (
            path in paths
            or type(size) is not int
            or size != by_name[path].file_size
            or digest != sums[path]
            or not isinstance(git_blob, str)
            or _OBJECT_ID.fullmatch(git_blob) is None
            or git_mode not in {"100644", "100755"}
        ):
            _fail("release metadata member identity is invalid")
        expected_mode = 0o755 if git_mode == "100755" else 0o644
        if stat.S_IMODE(by_name[path].external_attr >> 16) != expected_mode:
            _fail("release metadata Git mode disagrees with the archive")
        paths.append(path)
    if paths != sorted(expected_members):
        _fail("release metadata member inventory is not sorted and complete")


def _compare_receipt_metadata(
    receipt: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    shared = (
        "package_format_version",
        "provider",
        "selected_profile_id",
        "source",
        "seed_assignment",
        "cohort_assignment",
        "preregistration",
        "hardware_amendment",
        "profile",
        "environment",
        "dataset_pointer",
        "container_base_lock",
        "config_sha256",
        "contract_locks",
    )
    if any(receipt[field] != metadata[field] for field in shared):
        _fail("release receipt does not bind internal package metadata")
    aliases = {
        "cohort_assignment_sha256": "cohort_assignment",
        "preregistration_sha256": "preregistration",
        "hardware_amendment_sha256": "hardware_amendment",
        "profile_sha256": "profile",
        "dataset_pointer_sha256": "dataset_pointer",
        "container_base_lock_sha256": "container_base_lock",
    }
    if any(
        receipt[digest_field] != metadata[binding_field]["sha256"]
        for digest_field, binding_field in aliases.items()
    ):
        _fail("release receipt path/hash aliases disagree")


def _validate_local_source(
    source_root: Path,
    *,
    source_binding: Mapping[str, object],
    expected_members: frozenset[str],
    payload: Mapping[str, bytes],
    metadata: Mapping[str, object],
) -> None:
    dot_git = source_root / ".git"
    if dot_git.exists() or dot_git.is_file():
        try:
            repository = _package._core._open_repository(source_root)
            try:
                revision = _package._core._clean_revision(repository)
                tree_id = _package._core._commit_tree(repository, revision)
                if (
                    revision != source_binding["commit"]
                    or tree_id != source_binding["tree"]
                ):
                    _fail("release is stale for the current source commit or tree")
                tracked = _package._core._tracked_files(repository, tree_id)
                by_path = {item.path: item for item in tracked}
                if not expected_members <= set(by_path):
                    _fail("current source is missing a release member")
                selected = [by_path[path] for path in sorted(expected_members)]
                snapshot = _package._core._read_git_blobs(repository, selected)
                if any(snapshot[path] != payload[path] for path in expected_members):
                    _fail("release member bytes differ from the bound Git source")
                rows = {row["path"]: row for row in metadata["members"]}
                if any(
                    rows[item.path]["git_blob"] != item.object_id
                    or rows[item.path]["git_mode"] != item.mode
                    for item in selected
                ):
                    _fail("release member Git identities differ from the source tree")
            finally:
                repository.close()
        except VerificationError:
            raise
        except _package.PackageError as error:
            raise VerificationError(str(error)) from error
        return

    for relative in sorted(expected_members):
        candidate = source_root / relative
        try:
            before = candidate.stat(follow_symlinks=False)
            data = candidate.read_bytes()
            after = candidate.stat(follow_symlinks=False)
        except OSError as error:
            raise VerificationError(
                f"local source member is unavailable: {relative}"
            ) from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or data != payload[relative]
        ):
            _fail(f"local source member is stale or unsafe: {relative}")


def _inspect_archive(
    stream,
    *,
    receipt: Mapping[str, object],
    profile_path: str,
    source_root: Path,
) -> dict[str, object]:
    expected_source = _package.required_members(profile_path)
    expected_files = expected_source | {
        _package.METADATA_PATH,
        _package.SUMS_PATH,
    }
    try:
        with zipfile.ZipFile(stream, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                _fail("release archive contains duplicate members")
            _archive_core._validate_member_topology(infos)
            by_name = {info.filename: info for info in infos}
            actual_files = frozenset(
                info.filename for info in infos if not info.is_dir()
            )
            expected_directories = _package._core._planned_directories(
                iter(expected_files)
            )
            actual_directories = {
                info.filename for info in infos if info.is_dir()
            }
            if (
                actual_files != expected_files
                or actual_directories != expected_directories
            ):
                _fail("release archive member allowlist is not exact")
            if (set(_package.PROFILE_PATHS) - {profile_path}) & actual_files:
                _fail("release archive mixes P5 and P6 profiles")
            for info in infos:
                if (
                    info.date_time != _package._core.NORMALIZED_TIME
                    or info.create_system != 3
                    or info.flag_bits != 0
                ):
                    _fail("release archive metadata is not normalized")

            sums_bytes = _archive_core._member_bytes(
                archive,
                by_name[_package.SUMS_PATH],
                _package.SUMS_PATH,
            )
            sums = _archive_core._parse_sums(sums_bytes)
            if (
                set(sums) != expected_files - {_package.SUMS_PATH}
                or _sha256(sums_bytes) != receipt["members_sha256"]
            ):
                _fail("internal checksum inventory is not exact")
            payload: dict[str, bytes] = {}
            for path in sorted(expected_files):
                data = _archive_core._member_bytes(
                    archive,
                    by_name[path],
                    path,
                )
                if path != _package.SUMS_PATH and _sha256(data) != sums[path]:
                    _fail(f"release member checksum mismatch: {path}")
                payload[path] = data
    except VerificationError:
        raise
    except (
        KeyError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        _archive_core.VerificationError,
    ) as error:
        raise VerificationError("release archive is invalid") from error

    metadata = _strict_object(
        _json(payload[_package.METADATA_PATH], label=_package.METADATA_PATH),
        _METADATA_FIELDS,
        label=_package.METADATA_PATH,
    )
    spec = _package.PROFILE_SPECS[profile_path]
    if (
        metadata["schema_version"] != 1
        or metadata["package_format_version"] != _package.PACKAGE_FORMAT_VERSION
        or metadata["provider"] != spec["provider"]
        or metadata["selected_profile_id"] != spec["provider"]
    ):
        _fail("release metadata identifies a different package or profile")
    source_binding = _validate_source(metadata["source"])
    _validate_seed_assignment(
        metadata["seed_assignment"],
        provider=str(spec["provider"]),
    )
    _validate_member_rows(
        metadata["members"],
        expected_members=expected_source,
        sums=sums,
        by_name=by_name,
    )

    bindings = {
        "cohort_assignment": _validate_binding(
            metadata["cohort_assignment"],
            path=_package.COHORT_PATH,
            sums=sums,
            label="cohort assignment binding",
        ),
        "preregistration": _validate_binding(
            metadata["preregistration"],
            path=_package.PREREGISTRATION_PATH,
            sums=sums,
            label="preregistration binding",
        ),
        "hardware_amendment": _validate_binding(
            metadata["hardware_amendment"],
            path=_package.AMENDMENT_PATH,
            sums=sums,
            label="hardware amendment binding",
        ),
        "profile": _validate_binding(
            metadata["profile"],
            path=profile_path,
            sums=sums,
            label="selected profile binding",
        ),
        "dataset_pointer": _validate_binding(
            metadata["dataset_pointer"],
            path=_package.DATASET_POINTER_PATH,
            sums=sums,
            label="dataset pointer binding",
        ),
        "container_base_lock": _validate_container_binding(
            metadata["container_base_lock"],
            sums=sums,
        ),
    }
    if bindings["profile"]["sha256"] != spec["sha256"]:
        _fail("release selected profile hash is stale")
    _validate_environment(
        metadata["environment"],
        profile_sha256=str(spec["sha256"]),
    )
    config_hashes = metadata["config_sha256"]
    if (
        not isinstance(config_hashes, dict)
        or set(config_hashes) != set(_package.EXPECTED_CONFIGS)
        or any(
            _require_hash(digest, label=f"config hash for {path}")
            != sums.get(path)
            for path, digest in config_hashes.items()
        )
    ):
        _fail("v3 config hash inventory is not exact")
    _validate_contract_locks(metadata["contract_locks"], payload=payload)

    try:
        _package._validate_cohort(payload)
        _package._validate_profile_and_amendment(payload, profile_path)
        _package._validate_container_lock(payload)
        _package._validate_dataset_pointer(
            payload[_package.DATASET_POINTER_PATH]
        )
        _package._validate_selection_schema(
            payload[_package.SELECTION_SCHEMA_PATH]
        )
        for path in expected_source:
            _package._core._scan_secret(path, payload[path])
            if _package._MUTABLE_IMAGE_TAG.search(payload[path]):
                _fail(f"mutable image tag is forbidden: {path}")
    except _package.PackageError as error:
        raise VerificationError(str(error)) from error

    _compare_receipt_metadata(receipt, metadata)
    _validate_local_source(
        source_root,
        source_binding=source_binding,
        expected_members=expected_source,
        payload=payload,
        metadata=metadata,
    )
    return {
        "metadata": metadata,
        "bindings": bindings,
        "source": source_binding,
    }


def verify_release(
    *,
    release: Path | str,
    profile: Path | str,
    source_root: Path | str,
) -> dict[str, object]:
    """Authenticate one release and return a canonical profile decision."""

    source = Path(os.path.abspath(os.fspath(source_root)))
    profile_path = _profile_relative(source, profile)
    spec = _package.PROFILE_SPECS[profile_path]
    release_path = Path(release)
    if release_path.name != _package.RELEASE_RECEIPT_NAME:
        _fail(f"release receipt must be {_package.RELEASE_RECEIPT_NAME}")
    parent_fd = _archive_core._open_parent(release_path, "AWS GPU v3")
    try:
        receipt_bytes = _archive_core._read_control_file(
            parent_fd,
            release_path.name,
            "AWS GPU v3 release receipt",
        )
        receipt = _strict_object(
            _json(receipt_bytes, label="AWS GPU v3 release receipt"),
            _RECEIPT_FIELDS,
            label="AWS GPU v3 release receipt",
        )
        if (
            receipt["schema_version"] != 1
            or receipt["package_format_version"]
            != _package.PACKAGE_FORMAT_VERSION
            or receipt["provider"] != spec["provider"]
            or receipt["selected_profile_id"] != spec["provider"]
        ):
            _fail("release receipt identifies a different package or profile")
        _validate_source(receipt["source"])
        _validate_seed_assignment(
            receipt["seed_assignment"],
            provider=str(spec["provider"]),
        )
        members_sha256 = _require_hash(
            receipt["members_sha256"],
            label="members_sha256",
        )
        expected_release_id = (
            f"aws-gpu-v3-{spec['slug']}-{members_sha256[:16]}"
        )
        if receipt["release_id"] != expected_release_id:
            _fail("release ID is not bound to the member inventory")
        archive_binding = _strict_object(
            receipt["archive"],
            frozenset({"path", "sha256", "bytes"}),
            label="archive binding",
        )
        archive_hash = _require_hash(
            archive_binding["sha256"],
            label="archive.sha256",
        )
        archive_name = (
            f"memorysplit-aws-gpu-v3-{spec['slug']}-"
            f"{members_sha256[:16]}.zip"
        )
        if (
            archive_binding["path"] != archive_name
            or type(archive_binding["bytes"]) is not int
            or archive_binding["bytes"] <= 0
        ):
            _fail("archive path or byte binding is invalid")
        checksum = _archive_core._read_control_file(
            parent_fd,
            f"{archive_name}.sha256",
            "AWS GPU v3 external checksum",
        )
        if checksum != f"{archive_hash}  {archive_name}\n".encode("ascii"):
            _fail("external archive checksum is inconsistent")

        archive_fd = _archive_core._open_regular_at(
            parent_fd,
            archive_name,
            "AWS GPU v3 archive",
        )
        try:
            _archive_core._assert_descriptor_names_entry(
                parent_fd,
                archive_name,
                archive_fd,
                "AWS GPU v3 archive",
            )
            stream = os.fdopen(archive_fd, "rb", closefd=True)
            archive_fd = -1
            with stream:
                before = os.fstat(stream.fileno())
                actual_hash, actual_bytes = _archive_core._hash_stream(stream)
                if (
                    actual_hash != archive_hash
                    or actual_bytes != archive_binding["bytes"]
                ):
                    _fail("archive hash or byte count does not match the receipt")
                stream.seek(0)
                inspected = _inspect_archive(
                    stream,
                    receipt=receipt,
                    profile_path=profile_path,
                    source_root=source,
                )
                after = os.fstat(stream.fileno())
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
                    _fail("archive changed during verification")
                _archive_core._assert_descriptor_names_entry(
                    parent_fd,
                    archive_name,
                    stream.fileno(),
                    "AWS GPU v3 archive",
                )
        finally:
            if archive_fd >= 0:
                os.close(archive_fd)
    except _archive_core.VerificationError as error:
        raise VerificationError(str(error)) from error
    finally:
        os.close(parent_fd)

    bindings = inspected["bindings"]
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "package_format_version": _package.PACKAGE_FORMAT_VERSION,
        "release_id": receipt["release_id"],
        "provider": spec["provider"],
        "selected_profile_id": spec["provider"],
        "source_commit": inspected["source"]["commit"],
        "source_tree": inspected["source"]["tree"],
        "archive_sha256": archive_hash,
        "members_sha256": members_sha256,
        "cohort_assignment_sha256": bindings["cohort_assignment"]["sha256"],
        "preregistration_sha256": bindings["preregistration"]["sha256"],
        "hardware_amendment_sha256": bindings["hardware_amendment"]["sha256"],
        "profile_sha256": bindings["profile"]["sha256"],
        "container_base_lock_sha256": bindings["container_base_lock"]["sha256"],
        "mixed_profiles": False,
        "seeds": list(_package.SEEDS),
        "arms": list(_package.ARMS),
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
        description="Verify one profile-selected AWS GPU v3 release.",
        add_help=False,
    )
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--profile",
        required=True,
        choices=_package.PROFILE_PATHS,
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    try:
        arguments = parser.parse_args(argv)
        report = verify_release(
            release=arguments.release,
            profile=arguments.profile,
            source_root=arguments.source_root,
        )
        code = 0
    except VerificationError as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": {
                "code": "AWS_GPU_V3_RELEASE_REJECTED",
                "message": str(error),
            },
        }
        code = 2
    except Exception:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": {
                "code": "AWS_GPU_V3_VERIFIER_INTERNAL_ERROR",
                "message": "unexpected local verification failure",
            },
        }
        code = 70
    _emit(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
