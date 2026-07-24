#!/usr/bin/env python3
"""Build and verify the deterministic reasoning-v3 AWS execution package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msctl.reasoning_cohort import (
    COHORT_ID,
    SEEDS,
    TRANSFER_MANIFEST_SHA256,
    VIRTUAL_RECEIPT_SHA256,
    role_config_paths,
)
from scripts.package_135m_slurm_cohort import (
    _git,
    _identity,
    _json_bytes,
    _read_source,
    _sha,
    _write_zip,
)

ARCHIVE_NAME = "memorysplit-135m-reasoning-v3-aws.zip"
PYTHON_PREFIXES = (
    "cluster/",
    "corpusgen/",
    "evals/",
    "msctl/",
    "organizer/",
    "train/",
)
REQUIRED_FILES = frozenset(
    {
        "DATASET-POINTER-AWS-135M-V3.json",
        "artifacts/reasoning-corpus-v3/FROZEN.json",
        "cluster/aws/parallelcluster/memorysplit-v3-p5.example.yaml",
        "cluster/aws/reasoning-v3-corpus-manifest.json",
        "cluster/profiles/aws-p5-p6.example.json",
        "cluster/profiles/paired-slurm-profile-v1.schema.json",
        "cluster/slurm/v2_pair_evaluate.sbatch",
        "cluster/slurm/v2_pair_train.sbatch",
        "configs/cohort-assignment-135m-v3-aws-n10.json",
        "configs/reasoning-dataset-v3.json",
        "docs/AWS-135M-REASONING-V3-RUNBOOK.md",
        "requirements.txt",
        "scripts/build_135m_preflight.py",
        "scripts/check_aws_135m_readiness.py",
        "scripts/check_135m_pair_resume.py",
        "scripts/evaluate_135m_pair.py",
        "scripts/evaluate_reasoning_v3_run.py",
        "scripts/generate_aws_reasoning_configs.py",
        "scripts/package_135m_slurm_cohort.py",
        "scripts/package_aws_reasoning_v3.py",
        "scripts/run_135m_pair.py",
        "scripts/run_evals.py",
        "scripts/run_train.py",
        "scripts/validate_135m_launch.py",
        "tests/test_aws_reasoning_v3.py",
        "tests/test_data.py",
        "tests/test_trainer.py",
    }
)
GENERATED_FILES = frozenset({"SHA256SUMS", "release-receipt.json"})
_SECRET_NAMES = (
    ".aws/",
    ".env",
    "credentials",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".key",
)


def _source_revision(source_root: Path, *, require_clean: bool) -> str:
    if not (source_root / ".git").exists():
        if require_clean:
            raise ValueError("production AWS releases require a Git checkout")
        inventory = _packaged_inventory(source_root)
        receipt = json.loads(_read_source(source_root, "release-receipt.json"))
        revision = receipt.get("source_revision")
        if (
            inventory.get("release-receipt.json")
            != _sha(_read_source(source_root, "release-receipt.json"))
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError("packaged source revision is invalid")
        return revision
    revision = _git(source_root, "rev-parse", "--verify", "HEAD").strip()
    if len(revision) != 40:
        raise ValueError("source revision is not a full commit id")
    if not require_clean:
        return revision
    changed = _git(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).splitlines()
    disallowed = [
        line
        for line in changed
        if not line.startswith("?? corpus-build/")
        and not line.startswith("?? artifacts/aws-reasoning-v3/")
    ]
    if disallowed:
        raise ValueError(
            "production AWS releases require clean tracked sources; "
            f"dirty={disallowed}"
        )
    return revision


def _packaged_inventory(source_root: Path) -> dict[str, str]:
    try:
        lines = _read_source(source_root, "SHA256SUMS").decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("packaged checksum inventory is not UTF-8") from error
    inventory: dict[str, str] = {}
    for line in lines:
        digest, separator, path = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("packaged checksum inventory is malformed")
        _portable_member(path)
        if path in inventory:
            raise ValueError("packaged checksum inventory contains duplicate paths")
        inventory[path] = digest
    if "release-receipt.json" not in inventory:
        raise ValueError("packaged checksum inventory omits the release receipt")
    for path, expected in inventory.items():
        if _sha(_read_source(source_root, path)) != expected:
            raise ValueError(f"packaged source checksum differs: {path}")
    return inventory


def source_paths(source_root: Path) -> list[str]:
    if (source_root / ".git").exists():
        tracked = set(_git(source_root, "ls-files").splitlines())
        selected = {
            path
            for path in tracked
            if path.endswith(".py") and path.startswith(PYTHON_PREFIXES)
        } | set(REQUIRED_FILES) | set(role_config_paths("aws-operator"))
    else:
        selected = set(_packaged_inventory(source_root)) - {
            "release-receipt.json",
        }
    if not REQUIRED_FILES <= selected:
        raise ValueError(
            "AWS release source inventory omits required members: "
            f"{sorted(REQUIRED_FILES - selected)}"
        )
    missing = [
        path
        for path in sorted(selected)
        if not (source_root / path).is_file() or (source_root / path).is_symlink()
    ]
    if missing:
        raise ValueError(f"required AWS release members are missing: {missing}")
    return sorted(selected)


def _payload(source_root: Path, revision: str) -> dict[str, bytes]:
    paths = source_paths(source_root)
    payload = {path: _read_source(source_root, path) for path in paths}
    receipt = {
        "archive_format": "memorysplit-reasoning-v3-aws-execution-v1",
        "cohort_id": COHORT_ID,
        "corpus_bytes_included": False,
        "dataset_pointer_sha256": _sha(
            payload["DATASET-POINTER-AWS-135M-V3.json"]
        ),
        "external_launch_gates": {
            "aws_account_identity": "operator_required",
            "site_gpu_preflight": "operator_required",
        },
        "offline_contract_complete": True,
        "schema_version": 1,
        "seeds": list(SEEDS),
        "source_identity_sha256": _identity(payload, paths),
        "source_revision": revision,
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA256,
        "virtual_corpus_receipt_sha256": VIRTUAL_RECEIPT_SHA256,
    }
    payload["release-receipt.json"] = _json_bytes(receipt)
    payload["SHA256SUMS"] = "".join(
        f"{_sha(payload[path])}  {path}\n" for path in sorted(payload)
    ).encode()
    return payload


def build_package(
    output: Path | str,
    *,
    source_root: Path | str = ROOT,
    require_clean: bool = True,
) -> Path:
    source = Path(source_root)
    revision = _source_revision(source, require_clean=require_clean)
    destination = Path(output)
    if destination.suffix != ".zip":
        destination = destination / ARCHIVE_NAME
    _write_zip(destination, _payload(source, revision))
    return destination


def _portable_member(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or pure.is_absolute()
        or pure.as_posix() != name
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe AWS release member: {name!r}")
    lowered = name.lower()
    if any(marker in lowered for marker in _SECRET_NAMES):
        raise ValueError(f"secret-like AWS release member: {name!r}")


def verify_package(
    archive_path: Path | str,
    *,
    source_root: Path | str = ROOT,
) -> dict:
    archive = Path(archive_path)
    expected = set(source_paths(Path(source_root))) | set(GENERATED_FILES)
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("AWS release members must be sorted and unique")
        for info in infos:
            _portable_member(info.filename)
            mode = info.external_attr >> 16
            if mode != 0o100644 or info.is_dir():
                raise ValueError("AWS release contains a non-regular member")
        if set(names) != expected:
            raise ValueError(
                "AWS release namespace differs; "
                f"missing={sorted(expected - set(names))}, "
                f"extra={sorted(set(names) - expected)}"
            )
        payload: Mapping[str, bytes] = {
            name: handle.read(name)
            for name in names
        }
    inventory = payload["SHA256SUMS"].decode()
    expected_inventory = "".join(
        f"{_sha(payload[path])}  {path}\n"
        for path in sorted(set(payload) - {"SHA256SUMS"})
    )
    if inventory != expected_inventory:
        raise ValueError("AWS release checksum inventory differs")
    receipt = json.loads(payload["release-receipt.json"])
    source_members = sorted(set(payload) - GENERATED_FILES)
    if (
        receipt.get("archive_format")
        != "memorysplit-reasoning-v3-aws-execution-v1"
        or receipt.get("cohort_id") != COHORT_ID
        or receipt.get("corpus_bytes_included") is not False
        or receipt.get("offline_contract_complete") is not True
        or receipt.get("seeds") != list(SEEDS)
        or receipt.get("transfer_manifest_sha256")
        != TRANSFER_MANIFEST_SHA256
        or receipt.get("virtual_corpus_receipt_sha256")
        != VIRTUAL_RECEIPT_SHA256
        or receipt.get("source_identity_sha256")
        != _identity(payload, source_members)
    ):
        raise ValueError("AWS release receipt differs from the frozen contract")
    return {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "cohort_id": COHORT_ID,
        "member_count": len(payload),
        "source_revision": receipt["source_revision"],
        "verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", default=f"artifacts/aws-reasoning-v3/{ARCHIVE_NAME}")
    build.add_argument("--source-root", default=str(ROOT))
    build.add_argument("--test-allow-dirty", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("archive")
    verify.add_argument("--source-root", default=str(ROOT))
    args = parser.parse_args(argv)
    if args.command == "build":
        archive = build_package(
            args.output,
            source_root=args.source_root,
            require_clean=not args.test_allow_dirty,
        )
    else:
        archive = Path(args.archive)
    report = verify_package(archive, source_root=args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
