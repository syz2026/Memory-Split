#!/usr/bin/env python3
"""Build and verify the deterministic reasoning-v3 AWS execution package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import yaml

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
B200_PROFILE_PATH = "cluster/profiles/aws-p6-b200.48xlarge-135m-v1.json"
EVAL_CONTRACT_PATH = "configs/preregistration-135m-reasoning-v3-eval-v1.yaml"
EVAL_AUTHORITY_PATH = (
    "configs/preregistration-135m-reasoning-v3-eval-v1-authority.json"
)
EVAL_BOUNDARY_PATH = (
    "configs/preregistration-135m-reasoning-v3-eval-v1-aws-boundary.json"
)
EVAL_RELEASE_POINTER = "EVALUATION-RELEASE-POINTER.json"
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
        "pytest.ini",
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
        "scripts/run_reasoning_v3_evals.py",
        "scripts/run_reasoning_v3_inference.py",
        "scripts/run_train.py",
        "scripts/validate_135m_launch.py",
        "tests/test_aws_p6_b200_135m.py",
        "tests/test_aws_reasoning_v3.py",
        "tests/test_data.py",
        "tests/test_reasoning_v3_eval_contract.py",
        "tests/test_reasoning_v3_eval_generation.py",
        "tests/test_reasoning_v3_eval_sealing.py",
        "tests/test_reasoning_v3_eval_security.py",
        "tests/test_reasoning_v3_inference.py",
        "tests/test_reasoning_v3_reporting.py",
        "tests/test_reasoning_v3_runner.py",
        "tests/test_trainer.py",
        B200_PROFILE_PATH,
        EVAL_AUTHORITY_PATH,
        EVAL_BOUNDARY_PATH,
        EVAL_CONTRACT_PATH,
    }
)
GENERATED_FILES = frozenset(
    {"SHA256SUMS", "release-receipt.json", EVAL_RELEASE_POINTER}
)
_SECRET_NAMES = (
    ".aws/",
    ".env",
    "credentials",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".key",
)
_CORPUS_SUFFIXES = (
    ".arrow",
    ".bin",
    ".ckpt",
    ".gz",
    ".idx",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".safetensors",
    ".tar",
    ".zip",
    ".zst",
)
_CACHE_MARKERS = (
    ".DS_Store",
    ".git/",
    ".gitmodules",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tiktoken_cache/",
    ".venv/",
    "__pycache__/",
    "node_modules/",
)
_DATA_PREFIXES = (
    "artifacts/135m-slurm-releases/",
    "checkpoints/",
    "corpus-build/",
    "data/",
    "outputs/",
    "results/",
    "wandb/",
)
# No packaged member may exceed this size. The corpus, its sidecars, and any
# checkpoint are orders of magnitude larger, so an oversized member means the
# selection logic admitted protected bytes.
MAX_MEMBER_BYTES = 4 << 20
_CREDENTIAL_PATTERNS = (
    re.compile(rb"(?:AKIA|ASIA|AIDA|AROA|ANPA)[0-9A-Z]{16}"),
    re.compile(rb"(?i)aws_secret_access_key\s*[=:]"),
    re.compile(rb"(?i)aws_session_token\s*[=:]"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"(?i)\bx-amz-security-token\s*[=:]"),
)
_ACCOUNT_PATTERN = re.compile(rb"(?<![0-9])[0-9]{12}(?![0-9])")
# Every twelve-digit identifier the closed package is allowed to carry. The
# real account appears only inside the hash-committed evaluator contract and
# the evaluator modules that assert it; the rest are test fixtures. Any new or
# missing identifier fails the build closed.
DECLARED_ACCOUNT_IDENTIFIERS = (
    "000000000000",
    "${AWS_ACCOUNT_ID}",
    "111111111111",
    "123456789012",
    "222222222222",
    "333333333333",
    "999999999999",
)
# Sealed gold never leaves the evaluator. Data members carrying answer-bearing
# field names are rejected; the frozen contract is exempt because it only
# *names* the sealed fields it withholds.
_SEALED_GOLD_MARKERS = (b"canonical_answer", b"oracle_replay")
_SEALED_GOLD_EXEMPT = frozenset({EVAL_CONTRACT_PATH})
_SEALED_GOLD_DATA_SUFFIXES = (".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv")


def _object_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"packaged {label} is invalid")
    return value


def _source_provenance(
    source_root: Path,
    *,
    require_clean: bool,
) -> tuple[str, str]:
    """Return the ``(commit, tree)`` object ids the package is built from."""

    if not (source_root / ".git").exists():
        if require_clean:
            raise ValueError("production AWS releases require a Git checkout")
        inventory = _packaged_inventory(source_root)
        receipt = json.loads(_read_source(source_root, "release-receipt.json"))
        if inventory.get("release-receipt.json") != _sha(
            _read_source(source_root, "release-receipt.json")
        ):
            raise ValueError("packaged release receipt checksum differs")
        return (
            _object_id(receipt.get("source_revision"), "source revision"),
            _object_id(receipt.get("source_tree"), "source tree"),
        )
    revision = _object_id(
        _git(source_root, "rev-parse", "--verify", "HEAD").strip(),
        "source revision",
    )
    # ``git write-tree`` records the staged tree, so a release built from an
    # overlay that is staged but deliberately not committed still carries a
    # real, reproducible tree id instead of an unverifiable claim.
    tree = _object_id(_git(source_root, "write-tree").strip(), "source tree")
    if not require_clean:
        return revision, tree
    disallowed = [
        line
        for line in _git(
            source_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ).splitlines()
        # Columns are ``XY path``: ``X`` is the index state and ``Y`` the
        # worktree state. Staged edits are captured by the tree id above;
        # unstaged edits are not, so only ``Y == ' '`` may pass.
        if not (line[:1] in {"M", "A", "D", "R", "C"} and line[1:2] == " ")
        and not line.startswith("?? corpus-build/")
        and not line.startswith("?? artifacts/aws-reasoning-v3/")
    ]
    if disallowed:
        raise ValueError(
            "production AWS releases require a fully staged source tree; "
            f"unstaged={disallowed}"
        )
    return revision, tree


def _reject_excluded(payload: Mapping[str, bytes]) -> list[str]:
    """Fail closed on anything the closed package must never carry."""

    accounts: set[str] = set()
    for name in sorted(payload):
        data = payload[name]
        lowered = name.lower()
        if any(marker in f"{name}/" for marker in _CACHE_MARKERS):
            raise ValueError(f"cache or VCS member in AWS release: {name!r}")
        if any(lowered.startswith(prefix) for prefix in _DATA_PREFIXES):
            raise ValueError(f"protected data member in AWS release: {name!r}")
        if lowered.endswith(_CORPUS_SUFFIXES):
            raise ValueError(
                f"corpus or checkpoint member in AWS release: {name!r}"
            )
        if len(data) > MAX_MEMBER_BYTES:
            raise ValueError(
                f"oversized AWS release member: {name!r} ({len(data)} bytes)"
            )
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(data):
                raise ValueError(f"credential-like AWS release member: {name!r}")
        if (
            lowered.endswith(_SEALED_GOLD_DATA_SUFFIXES)
            and name not in _SEALED_GOLD_EXEMPT
            and any(marker in data for marker in _SEALED_GOLD_MARKERS)
        ):
            raise ValueError(f"sealed-gold member in AWS release: {name!r}")
        accounts.update(match.decode() for match in _ACCOUNT_PATTERN.findall(data))
    if sorted(accounts) != sorted(DECLARED_ACCOUNT_IDENTIFIERS):
        raise ValueError(
            "AWS release account bindings differ from the declared set; "
            f"found={sorted(accounts)}"
        )
    return sorted(accounts)


def _evaluation_release_pointer(payload: Mapping[str, bytes]) -> dict:
    """Bind the model-visible evaluation release the package points at."""

    contract = yaml.safe_load(payload[EVAL_CONTRACT_PATH].decode("utf-8"))
    release = contract["release"]
    authority = json.loads(payload[EVAL_AUTHORITY_PATH])
    boundary = json.loads(payload[EVAL_BOUNDARY_PATH])
    contract_sha256 = _sha(payload[EVAL_CONTRACT_PATH])
    if (
        authority["contract_path"] != EVAL_CONTRACT_PATH
        or authority["contract_sha256"] != contract_sha256
        or authority["aws_boundary_path"] != EVAL_BOUNDARY_PATH
        or authority["aws_boundary_sha256"] != _sha(payload[EVAL_BOUNDARY_PATH])
    ):
        raise ValueError("evaluator authority record does not bind the contract")
    commitments = []
    for entry in contract["evaluator_code"]:
        path = entry["path"]
        if path not in payload:
            raise ValueError(f"committed evaluator module is not packaged: {path}")
        if _sha(payload[path]) != entry["sha256"]:
            raise ValueError(f"committed evaluator module differs: {path}")
        commitments.append({"path": path, "sha256": entry["sha256"]})
    return {
        "contract_id": authority["contract_id"],
        "contract_path": EVAL_CONTRACT_PATH,
        "contract_sha256": contract_sha256,
        "evaluator_code_commitments": commitments,
        "format": "memorysplit-reasoning-v3-eval-release-pointer-v1",
        "model_visible": {
            "authorization": release["model_visible_authorization"],
            "bytes_included": False,
            "fields": list(release["model_visible_fields"]),
            "publication_prefix": boundary["storage"]["model_visible_prefix"],
            "release_format": release["format"],
        },
        "schema_version": 1,
        "sealed_gold": {
            "authorization": release["sealed_gold_authorization"],
            "bytes_included": False,
            "publication_prefix": boundary["storage"]["sealed_gold_prefix"],
            "withheld_fields": list(release["sealed_gold_fields"]),
        },
    }


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
        selected = set(_packaged_inventory(source_root)) - set(GENERATED_FILES)
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


def _payload(source_root: Path, revision: str, tree: str) -> dict[str, bytes]:
    paths = source_paths(source_root)
    payload = {path: _read_source(source_root, path) for path in paths}
    accounts = _reject_excluded(payload)
    pointer = _evaluation_release_pointer(payload)
    payload[EVAL_RELEASE_POINTER] = _json_bytes(pointer)
    receipt = {
        "archive_format": "memorysplit-reasoning-v3-aws-execution-v1",
        "cohort_id": COHORT_ID,
        "corpus_bytes_included": False,
        "dataset_pointer_sha256": _sha(
            payload["DATASET-POINTER-AWS-135M-V3.json"]
        ),
        "declared_account_identifiers": accounts,
        "evaluation_release_pointer_sha256": _sha(payload[EVAL_RELEASE_POINTER]),
        "evaluator_code_commitments": pointer["evaluator_code_commitments"],
        "evaluation_contract_sha256": pointer["contract_sha256"],
        "execution_profile_path": B200_PROFILE_PATH,
        "execution_profile_sha256": _sha(payload[B200_PROFILE_PATH]),
        "external_launch_gates": {
            "aws_account_identity": "operator_required",
            "site_gpu_preflight": "operator_required",
        },
        "member_sha256": {
            path: _sha(payload[path]) for path in sorted(payload)
        },
        "offline_contract_complete": True,
        "run_config_paths": role_config_paths("aws-operator"),
        "schema_version": 2,
        "sealed_gold_included": False,
        "seeds": list(SEEDS),
        "source_identity_sha256": _identity(payload, paths),
        "source_revision": revision,
        "source_tree": tree,
        "transfer_manifest_path": "cluster/aws/reasoning-v3-corpus-manifest.json",
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
    revision, tree = _source_provenance(source, require_clean=require_clean)
    destination = Path(output)
    if destination.suffix != ".zip":
        destination = destination / ARCHIVE_NAME
    _write_zip(destination, _payload(source, revision, tree))
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
    accounts = _reject_excluded({name: payload[name] for name in source_members})
    pointer = _evaluation_release_pointer(payload)
    if (
        receipt.get("archive_format")
        != "memorysplit-reasoning-v3-aws-execution-v1"
        or receipt.get("cohort_id") != COHORT_ID
        or receipt.get("corpus_bytes_included") is not False
        or receipt.get("sealed_gold_included") is not False
        or receipt.get("offline_contract_complete") is not True
        or receipt.get("seeds") != list(SEEDS)
        or receipt.get("declared_account_identifiers") != accounts
        or receipt.get("run_config_paths") != role_config_paths("aws-operator")
        or receipt.get("execution_profile_path") != B200_PROFILE_PATH
        or receipt.get("execution_profile_sha256")
        != _sha(payload[B200_PROFILE_PATH])
        or receipt.get("evaluation_contract_sha256") != pointer["contract_sha256"]
        or receipt.get("evaluator_code_commitments")
        != pointer["evaluator_code_commitments"]
        or receipt.get("evaluation_release_pointer_sha256")
        != _sha(payload[EVAL_RELEASE_POINTER])
        or json.loads(payload[EVAL_RELEASE_POINTER]) != pointer
        or receipt.get("member_sha256")
        != {
            path: _sha(payload[path])
            for path in sorted(set(payload) - {"SHA256SUMS", "release-receipt.json"})
        }
        or receipt.get("transfer_manifest_sha256")
        != TRANSFER_MANIFEST_SHA256
        or receipt.get("virtual_corpus_receipt_sha256")
        != VIRTUAL_RECEIPT_SHA256
        or receipt.get("source_identity_sha256")
        != _identity(payload, source_members)
    ):
        raise ValueError("AWS release receipt differs from the frozen contract")
    _object_id(receipt.get("source_tree"), "source tree")
    return {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "cohort_id": COHORT_ID,
        "evaluation_contract_sha256": receipt["evaluation_contract_sha256"],
        "execution_profile_sha256": receipt["execution_profile_sha256"],
        "member_count": len(payload),
        "source_revision": receipt["source_revision"],
        "source_tree": receipt["source_tree"],
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
