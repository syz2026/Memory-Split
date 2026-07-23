#!/usr/bin/env python3
"""Build five role-scoped releases and one deterministic AWS transfer bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.cohort import (  # noqa: E402
    COHORT_ID,
    ROLES,
    canonical_assignment,
    role_config_paths,
)


COMMON_PREFIXES = (
    "corpusgen/",
    "evals/",
    "msctl/",
    "organizer/",
    "train/",
)
COMMON_FILES = frozenset(
    {
        "DATASET-POINTER-SLURM-135M.json",
        "cluster/__init__.py",
        "cluster/aws/__init__.py",
        "cluster/aws/p5/__init__.py",
        "cluster/aws/p5/corpus_contract.py",
        "cluster/corpus_contract.py",
        "cluster/mit/__init__.py",
        "cluster/mit/probe_cluster.py",
        "cluster/mit/profile.py",
        "cluster/mit/run_135m_manifest.py",
        "cluster/profiles/paired-slurm-profile-v1.schema.json",
        "cluster/slurm/v2_pair_evaluate.sbatch",
        "cluster/slurm/v2_pair_train.sbatch",
        "configs/power-sensitivity-135m-n10.json",
        "configs/preregistration-135m-v1.yaml",
        "configs/reasoning-dataset-v2.json",
        "docs/SLURM-135M-RUNBOOK.md",
        "requirements.txt",
        "scripts/bridge_135m_dataset.py",
        "scripts/build_135m_preflight.py",
        "scripts/check_135m_pair_resume.py",
        "scripts/evaluate_135m_pair.py",
        "scripts/make_135m_role_manifest.py",
        "scripts/package_135m_slurm_cohort.py",
        "scripts/run_135m_pair.py",
        "scripts/run_confirmatory_135m.py",
        "scripts/run_evals.py",
        "scripts/run_train.py",
        "scripts/stage_135m_dataset.py",
        "scripts/validate_135m_launch.py",
        "scripts/verify_135m_slurm_releases.py",
    }
)
PROFILE_BY_ROLE = {
    "farmshare-lead": "cluster/profiles/farmshare-l40s.json",
    "farmshare-collaborator-1": "cluster/profiles/farmshare-l40s.json",
    "farmshare-collaborator-2": "cluster/profiles/farmshare-l40s.json",
    "mit-collaborator-a": "cluster/profiles/mit-collaborator-a.example.json",
    "mit-collaborator-b": "cluster/profiles/mit-collaborator-b.example.json",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
AWS_BUNDLE_NAME = "memorysplit-135m-n10-all-roles-aws.zip"


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def source_revision(source_root: Path, *, require_clean: bool) -> str:
    revision = _git(source_root, "rev-parse", "--verify", "HEAD").strip()
    if len(revision) != 40:
        raise ValueError("source revision is not a full commit id")
    if require_clean:
        status = _git(
            source_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        if status:
            raise ValueError("production releases require a clean source commit")
    return revision


def _portable_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} is not a portable relative path: {value!r}")
    return value


def _read_source(source_root: Path, relative: str) -> bytes:
    relative = _portable_path(relative, label="release member")
    root = source_root.resolve(strict=True)
    current = source_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"release input traverses a symlink: {relative}")
    try:
        current.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"release input is missing or unsafe: {relative}") from error
    if not current.is_file():
        raise ValueError(f"release input is not a regular file: {relative}")
    return current.read_bytes()


def common_source_paths(source_root: Path) -> list[str]:
    tracked = set(_git(source_root, "ls-files").splitlines())
    selected = {
        path
        for path in tracked
        if path.endswith(".py") and path.startswith(COMMON_PREFIXES)
    } | set(COMMON_FILES)
    missing = [
        path
        for path in sorted(selected)
        if not (source_root / path).is_file() or (source_root / path).is_symlink()
    ]
    if missing:
        raise ValueError(f"required common release members are missing: {missing}")
    return sorted(selected)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(payload: Mapping[str, bytes], paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        data = payload[path]
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def assignment_document(
    role: str,
    *,
    source_root: Path,
    revision: str,
) -> dict:
    spec = ROLES[role]
    config_paths = role_config_paths(role)
    return {
        "cells": [
            {"arm": arm, "config_path": f"configs/135m-v2/{arm}-s{seed}.yaml", "seed": seed}
            for seed in spec["seeds"]
            for arm in ("dense", "split90")
        ],
        "cohort_assignment_sha256": _sha(
            _read_source(source_root, "configs/cohort-assignment-135m-n10.json")
        ),
        "cohort_id": COHORT_ID,
        "config_paths": config_paths,
        "dataset_pointer_sha256": _sha(
            _read_source(source_root, "DATASET-POINTER-SLURM-135M.json")
        ),
        "operator": role,
        "platform": spec["platform"],
        "preregistration_sha256": _sha(
            _read_source(source_root, "configs/preregistration-135m-v1.yaml")
        ),
        "profile": PROFILE_BY_ROLE[role],
        "provider": spec["provider"],
        "schema_version": 1,
        "seeds": list(spec["seeds"]),
        "source_revision": revision,
    }


def expected_member_paths(source_root: Path, role: str) -> set[str]:
    return (
        set(common_source_paths(source_root))
        | set(role_config_paths(role))
        | {PROFILE_BY_ROLE[role], "assignment.json", "release-receipt.json", "SHA256SUMS"}
    )


def _release_payload(
    role: str,
    *,
    source_root: Path,
    revision: str,
) -> dict[str, bytes]:
    common = common_source_paths(source_root)
    source_payload = {path: _read_source(source_root, path) for path in common}
    payload = dict(source_payload)
    for path in role_config_paths(role):
        payload[path] = _read_source(source_root, path)
    profile_path = PROFILE_BY_ROLE[role]
    payload[profile_path] = _read_source(source_root, profile_path)
    assignment = assignment_document(role, source_root=source_root, revision=revision)
    payload["assignment.json"] = _json_bytes(assignment)

    eval_paths = [path for path in common if path.startswith("evals/")]
    scientific_paths = [
        "configs/power-sensitivity-135m-n10.json",
        "configs/preregistration-135m-v1.yaml",
        "configs/reasoning-dataset-v2.json",
        "DATASET-POINTER-SLURM-135M.json",
    ]
    receipt = {
        "cells": assignment["cells"],
        "cohort_assignment_sha256": assignment["cohort_assignment_sha256"],
        "cohort_id": COHORT_ID,
        "cohort_identity_sha256": _identity(payload, scientific_paths),
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
        "profile_member": profile_path,
        "profile_sha256": _sha(payload[profile_path]),
        "provider": ROLES[role]["provider"],
        "release_format": "memorysplit-135m-slurm-role-v1",
        "schema_version": 1,
        "source_identity_sha256": _identity(payload, common),
        "source_revision": revision,
    }
    payload["release-receipt.json"] = _json_bytes(receipt)
    inventory = "".join(
        f"{_sha(payload[path])}  {path}\n"
        for path in sorted(payload)
    ).encode()
    payload["SHA256SUMS"] = inventory
    expected = expected_member_paths(source_root, role)
    if set(payload) != expected:
        raise ValueError(
            f"release payload differs from closed allowlist: "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    return payload


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0x800
    return info


def _write_zip(path: Path, payload: Mapping[str, bytes]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace release archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(payload):
                archive.writestr(
                    _zip_info(name),
                    payload[name],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.rename(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def aws_bundle_readme(revision: str) -> bytes:
    return f"""# MemorySplit 135M N=10 all-role AWS transfer bundle

Source revision: `{revision}`

This archive transports all five role-scoped paired-Slurm releases through
Amazon S3 or an AWS instance filesystem. It does not contain AWS credentials,
the external production corpus, checkpoints, or an AWS-native scheduler.

## Transfer with Amazon S3

Upload from a machine with AWS credentials:

```bash
aws s3 cp {AWS_BUNDLE_NAME} s3://YOUR-BUCKET/YOUR-PREFIX/{AWS_BUNDLE_NAME}
```

Download on the target machine:

```bash
aws s3 cp s3://YOUR-BUCKET/YOUR-PREFIX/{AWS_BUNDLE_NAME} .
unzip {AWS_BUNDLE_NAME} -d memorysplit-135m-n10
cd memorysplit-135m-n10
sha256sum -c SHA256SUMS
```

Cross-verify all role releases using the verifier shipped in any role:

```bash
unzip roles/farmshare-lead.zip -d verifier
python verifier/scripts/verify_135m_slurm_releases.py --release-dir roles
```

Each role ZIP contains two assigned seeds; each seed is one Dense/Split90
two-GPU pair. Extract only the intended operator ZIP and follow its
`docs/SLURM-135M-RUNBOOK.md`.

## External launch gates

The production `memorysplit-parallel-corpus-v2` receipt is not included and
remains required. Real site canaries remain required. MIT operators must bind
their discovered Slurm profile. These role packages target paired Slurm; merely
copying this bundle to AWS does not turn them into AWS-native jobs.
""".encode()


def _aws_bundle_manifest(
    archives: Mapping[str, Path],
    *,
    revision: str,
) -> dict:
    return {
        "aws_access": {
            "credentials_included": False,
            "transport": "amazon-s3-or-instance-filesystem",
        },
        "bundle_format": "memorysplit-135m-n10-all-roles-aws-v1",
        "cohort_id": COHORT_ID,
        "execution_contract": "paired-slurm-role-releases",
        "external_launch_gates": {
            "mit_site_profiles": "operator_binding_required",
            "production_dataset_receipt": "unfrozen",
            "site_gpu_preflights": "pending_external",
        },
        "role_archives": [
            {
                "archive": f"roles/{role}.zip",
                "platform": ROLES[role]["platform"],
                "provider": ROLES[role]["provider"],
                "role": role,
                "seeds": list(ROLES[role]["seeds"]),
                "sha256": _sha(archives[role].read_bytes()),
            }
            for role in ROLES
        ],
        "schema_version": 1,
        "source_revision": revision,
    }


def build_aws_bundle(
    output: Path,
    archives: Mapping[str, Path],
    *,
    revision: str,
) -> Path:
    if set(archives) != set(ROLES):
        raise ValueError("AWS transfer bundle requires all five role archives")
    payload = {
        "README-AWS.md": aws_bundle_readme(revision),
        "bundle-manifest.json": _json_bytes(
            _aws_bundle_manifest(archives, revision=revision)
        ),
        "roles/SHA256SUMS": "".join(
            f"{_sha(archives[role].read_bytes())}  {role}.zip\n"
            for role in ROLES
        ).encode(),
    }
    for role in ROLES:
        payload[f"roles/{role}.zip"] = archives[role].read_bytes()
    payload["SHA256SUMS"] = "".join(
        f"{_sha(payload[path])}  {path}\n"
        for path in sorted(payload)
    ).encode()
    path = output / AWS_BUNDLE_NAME
    _write_zip(path, payload)
    return path


def package_all(
    out_dir: Path | str,
    *,
    source_root: Path | str = ROOT,
    require_clean: bool = True,
) -> dict[str, Path]:
    source = Path(source_root)
    revision = source_revision(source, require_clean=require_clean)
    output = Path(out_dir)
    archives = {}
    for role in ROLES:
        path = output / f"{role}.zip"
        _write_zip(
            path,
            _release_payload(role, source_root=source, revision=revision),
        )
        archives[role] = path
    outer = output / "SHA256SUMS"
    if outer.exists() or outer.is_symlink():
        raise FileExistsError(f"refusing to replace outer checksum index: {outer}")
    outer.write_text(
        "".join(
            f"{_sha(path.read_bytes())}  {path.name}\n"
            for path in archives.values()
        )
    )
    build_aws_bundle(output, archives, revision=revision)
    return archives


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="artifacts/135m-slurm-releases",
    )
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument(
        "--test-allow-dirty",
        action="store_true",
        help="test fixtures only; production releases require a clean commit",
    )
    args = parser.parse_args(argv)
    archives = package_all(
        args.out_dir,
        source_root=args.source_root,
        require_clean=not args.test_allow_dirty,
    )
    bundle = Path(args.out_dir) / AWS_BUNDLE_NAME
    print(
        json.dumps(
            {
                "all_roles_aws_bundle": {
                    "archive": str(bundle),
                    "sha256": _sha(bundle.read_bytes()),
                },
                "roles": {
                    role: {
                        "archive": str(path),
                        "sha256": _sha(path.read_bytes()),
                    }
                    for role, path in archives.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
