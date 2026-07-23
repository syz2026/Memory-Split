#!/usr/bin/env python3
"""Verify five closed, deterministic 135M collaborator ZIP releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.cohort import (  # noqa: E402
    ARMS,
    COHORT_ID,
    ROLES,
    SEEDS,
    validate_run_config,
)
from scripts.package_135m_slurm_cohort import (  # noqa: E402
    AWS_BUNDLE_NAME,
    FIXED_ZIP_TIME,
    PROFILE_BY_ROLE,
    _aws_bundle_manifest,
    _identity,
    _read_source,
    _sha,
    assignment_document,
    aws_bundle_readme,
    common_source_paths,
    expected_member_paths,
)


MAX_ARCHIVE_BYTES = 1 << 30
MAX_MEMBER_BYTES = 64 << 20
MAX_TOTAL_BYTES = 512 << 20
_SECRET_MARKERS = (
    b"AK" + b"IA",
    b"AS" + b"IA",
    b"ghp" + b"_",
    b"github" + b"_pat_",
    b"xox" + b"b-",
    b"BEGIN " + b"PRIVATE KEY",
    b"aws_" + b"secret_access_key",
)


class ReleaseVerificationError(ValueError):
    """A release differs from the closed collaborator contract."""


def _portable(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and "\\" not in name
        and not path.is_absolute()
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == name
    )


def _read_archive(path: Path) -> tuple[list[zipfile.ZipInfo], dict[str, bytes]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > MAX_ARCHIVE_BYTES
    ):
        raise ReleaseVerificationError(f"release archive is missing or unsafe: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ReleaseVerificationError("release contains duplicate members")
            payload = {}
            total = 0
            for info in infos:
                if not _portable(info.filename):
                    raise ReleaseVerificationError(
                        f"release member path is unsafe: {info.filename!r}"
                    )
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ReleaseVerificationError(
                        f"release contains a symlink: {info.filename}"
                    )
                if info.is_dir() or not stat.S_ISREG(mode):
                    raise ReleaseVerificationError(
                        f"release member is not a regular file: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise ReleaseVerificationError("encrypted release members are forbidden")
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ReleaseVerificationError("release member exceeds size limit")
                total += info.file_size
                if total > MAX_TOTAL_BYTES:
                    raise ReleaseVerificationError("release exceeds expanded size limit")
                payload[info.filename] = archive.read(info)
    except zipfile.BadZipFile as error:
        raise ReleaseVerificationError("release is not a valid ZIP") from error
    return infos, payload


def _json_member(payload: dict[str, bytes], name: str) -> dict:
    try:
        value = json.loads(payload[name])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"{name} is missing or invalid") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"{name} must contain an object")
    return value


def _verify_inventory(payload: dict[str, bytes]) -> None:
    try:
        text = payload["SHA256SUMS"].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as error:
        raise ReleaseVerificationError("internal SHA256SUMS is missing or invalid") from error
    rows = [line for line in text.splitlines() if line]
    expected_names = sorted(set(payload) - {"SHA256SUMS"})
    observed = {}
    for line in rows:
        if len(line) < 67 or line[64:66] != "  ":
            raise ReleaseVerificationError("internal SHA256SUMS row is malformed")
        digest, name = line[:64], line[66:]
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not _portable(name)
            or name in observed
        ):
            raise ReleaseVerificationError("internal SHA256SUMS row is unsafe")
        observed[name] = digest
    if list(observed) != expected_names:
        raise ReleaseVerificationError("internal SHA256SUMS does not cover sorted members")
    for name, digest in observed.items():
        if _sha(payload[name]) != digest:
            raise ReleaseVerificationError(f"member checksum differs: {name}")


def _verify_one(path: Path, *, source_root: Path) -> dict:
    infos, payload = _read_archive(path)
    assignment = _json_member(payload, "assignment.json")
    role = assignment.get("operator")
    if role not in ROLES:
        raise ReleaseVerificationError("assignment names an unknown role")

    for name, data in payload.items():
        if any(marker in data for marker in _SECRET_MARKERS):
            raise ReleaseVerificationError(f"release contains a secret marker: {name}")
    expected = expected_member_paths(source_root, role)
    missing, extra = expected - set(payload), set(payload) - expected
    if missing or extra:
        raise ReleaseVerificationError(
            f"release has missing or extra members; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if [info.filename for info in infos] != sorted(payload):
        raise ReleaseVerificationError("release members are not deterministically sorted")
    for info in infos:
        if (
            info.date_time != FIXED_ZIP_TIME
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.create_system != 3
            or info.external_attr >> 16 != 0o100644
        ):
            raise ReleaseVerificationError(
                f"release metadata is not deterministic: {info.filename}"
            )
    _verify_inventory(payload)

    receipt = _json_member(payload, "release-receipt.json")
    revision = receipt.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ReleaseVerificationError("release source revision is invalid")
    expected_assignment = assignment_document(
        role,
        source_root=source_root,
        revision=revision,
    )
    if assignment != expected_assignment:
        raise ReleaseVerificationError("role assignment differs from the frozen matrix")
    if path.name != f"{role}.zip":
        raise ReleaseVerificationError("release filename does not match its role")

    common = common_source_paths(source_root)
    for member in common + assignment["config_paths"] + [PROFILE_BY_ROLE[role]]:
        if payload[member] != _read_source(source_root, member):
            raise ReleaseVerificationError(
                f"release member differs from source commit surface: {member}"
            )
    cells = []
    for relative in assignment["config_paths"]:
        try:
            cfg = yaml.safe_load(payload[relative])
        except yaml.YAMLError as error:
            raise ReleaseVerificationError(f"invalid YAML config: {relative}") from error
        try:
            validated = validate_run_config(cfg, relative_path=relative)
        except ValueError as error:
            raise ReleaseVerificationError(f"noncanonical run config: {relative}") from error
        cells.append((validated["arm"], validated["seed"]))
    expected_cells = [
        (arm, seed)
        for seed in ROLES[role]["seeds"]
        for arm in ARMS
    ]
    if cells != expected_cells:
        raise ReleaseVerificationError("release cells differ from role ownership")

    eval_paths = [member for member in common if member.startswith("evals/")]
    scientific = [
        "configs/power-sensitivity-135m-n10.json",
        "configs/preregistration-135m-v1.yaml",
        "configs/reasoning-dataset-v2.json",
        "DATASET-POINTER-SLURM-135M.json",
    ]
    expected_receipt = {
        "cells": assignment["cells"],
        "cohort_assignment_sha256": assignment["cohort_assignment_sha256"],
        "cohort_id": COHORT_ID,
        "cohort_identity_sha256": _identity(payload, scientific),
        "dataset_identity_sha256": _identity(
            payload,
            [
                "DATASET-POINTER-SLURM-135M.json",
                "configs/reasoning-dataset-v2.json",
            ],
        ),
        "evaluation_identity_sha256": _identity(payload, eval_paths),
        "external_launch_gates": {
            "gpu_preflight": "pending_external",
            "production_dataset_receipt": "unfrozen",
        },
        "operator": role,
        "platform": ROLES[role]["platform"],
        "profile_member": PROFILE_BY_ROLE[role],
        "profile_sha256": _sha(payload[PROFILE_BY_ROLE[role]]),
        "provider": ROLES[role]["provider"],
        "release_format": "memorysplit-135m-slurm-role-v1",
        "schema_version": 1,
        "source_identity_sha256": _identity(payload, common),
        "source_revision": revision,
    }
    if receipt != expected_receipt:
        raise ReleaseVerificationError("release receipt identities are inconsistent")
    return {
        "archive": path,
        "archive_sha256": _sha(path.read_bytes()),
        "cells": cells,
        "cohort_identity_sha256": receipt["cohort_identity_sha256"],
        "dataset_identity_sha256": receipt["dataset_identity_sha256"],
        "evaluation_identity_sha256": receipt["evaluation_identity_sha256"],
        "role": role,
        "source_identity_sha256": receipt["source_identity_sha256"],
        "source_revision": revision,
    }


def _verify_outer_index(path: Path, releases: list[dict]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ReleaseVerificationError("outer checksum index is missing or unsafe")
    expected = "".join(
        f"{release['archive_sha256']}  {release['role']}.zip\n"
        for release in releases
    )
    if path.read_text() != expected:
        raise ReleaseVerificationError("outer checksum index differs from release set")


def verify_release_set(
    archives: list[Path | str],
    *,
    source_root: Path | str = ROOT,
    outer_index: Path | str | None = None,
) -> dict:
    if len(archives) != len(ROLES):
        raise ReleaseVerificationError("exactly five release archives are required")
    source = Path(source_root)
    releases = [_verify_one(Path(path), source_root=source) for path in archives]
    by_role = {release["role"]: release for release in releases}
    if len(by_role) != len(ROLES) or set(by_role) != set(ROLES):
        raise ReleaseVerificationError("release roles overlap or are incomplete")
    ordered = [by_role[role] for role in ROLES]
    identity_fields = (
        "source_revision",
        "source_identity_sha256",
        "dataset_identity_sha256",
        "evaluation_identity_sha256",
        "cohort_identity_sha256",
    )
    for field in identity_fields:
        if len({release[field] for release in ordered}) != 1:
            raise ReleaseVerificationError(f"release {field} is not shared")
    cells = [cell for release in ordered for cell in release["cells"]]
    expected_cells = [(arm, seed) for seed in SEEDS for arm in ARMS]
    if len(cells) != len(set(cells)):
        raise ReleaseVerificationError("release cells overlap")
    if set(cells) != set(expected_cells):
        raise ReleaseVerificationError("release cells are incomplete")
    if outer_index is not None:
        _verify_outer_index(Path(outer_index), ordered)
    return {
        "cells": [list(cell) for cell in sorted(cells, key=lambda item: (item[1], item[0]))],
        "identities": {field: ordered[0][field] for field in identity_fields},
        "roles": list(ROLES),
        "schema_version": 1,
        "verified": True,
    }


def verify_aws_bundle(
    path: Path | str,
    *,
    source_root: Path | str = ROOT,
) -> dict:
    bundle = Path(path)
    infos, payload = _read_archive(bundle)
    expected = {
        "README-AWS.md",
        "SHA256SUMS",
        "bundle-manifest.json",
        "roles/SHA256SUMS",
        *(f"roles/{role}.zip" for role in ROLES),
    }
    if set(payload) != expected:
        raise ReleaseVerificationError(
            "AWS bundle has missing or extra members; "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    if bundle.name != AWS_BUNDLE_NAME:
        raise ReleaseVerificationError("AWS bundle filename is not canonical")
    if [info.filename for info in infos] != sorted(payload):
        raise ReleaseVerificationError(
            "AWS bundle members are not deterministically sorted"
        )
    for info in infos:
        if (
            info.date_time != FIXED_ZIP_TIME
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.create_system != 3
            or info.external_attr >> 16 != 0o100644
        ):
            raise ReleaseVerificationError(
                f"AWS bundle metadata is not deterministic: {info.filename}"
            )
    _verify_inventory(payload)

    with tempfile.TemporaryDirectory(prefix="ms135-aws-bundle-") as temporary:
        role_dir = Path(temporary) / "roles"
        role_dir.mkdir()
        archives = {}
        for role in ROLES:
            archive = role_dir / f"{role}.zip"
            archive.write_bytes(payload[f"roles/{role}.zip"])
            archives[role] = archive
        outer = role_dir / "SHA256SUMS"
        outer.write_bytes(payload["roles/SHA256SUMS"])
        release_report = verify_release_set(
            list(archives.values()),
            source_root=source_root,
            outer_index=outer,
        )
        revision = release_report["identities"]["source_revision"]
        expected_manifest = _aws_bundle_manifest(
            archives,
            revision=revision,
        )
    if _json_member(payload, "bundle-manifest.json") != expected_manifest:
        raise ReleaseVerificationError("AWS bundle manifest is inconsistent")
    if payload["README-AWS.md"] != aws_bundle_readme(revision):
        raise ReleaseVerificationError("AWS bundle run instructions differ")
    return {
        "archive": str(bundle.resolve()),
        "archive_sha256": _sha(bundle.read_bytes()),
        "identities": release_report["identities"],
        "roles": release_report["roles"],
        "schema_version": 1,
        "verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-dir",
        default="artifacts/135m-slurm-releases",
    )
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--bundle")
    args = parser.parse_args(argv)
    if args.bundle:
        print(
            json.dumps(
                verify_aws_bundle(
                    args.bundle,
                    source_root=args.source_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    directory = Path(args.release_dir)
    report = verify_release_set(
        [directory / f"{role}.zip" for role in ROLES],
        source_root=args.source_root,
        outer_index=directory / "SHA256SUMS",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
