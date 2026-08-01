#!/usr/bin/env python3
"""Build the deterministic 1B corpus handoff package.

The corpus itself is 31.7 GB and stays in S3. This package ships everything
needed to fetch it, prove it is the right bytes, and understand or rebuild how
it was made: the transfer manifest and locks, the receipts recovered from S3,
both recipe files, both producer source trees, the 1B cohort scripts, and the
methodology docs.

The producer code is uncommitted and sharded across `.worktrees/`, so the source
map below is explicit rather than derived from `git ls-files`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "memorysplit-1b-corpus-handoff"
ARCHIVE_NAME = f"{PREFIX}.zip"
NORMALIZED_TIME = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644

WT_V2 = ".worktrees/swarm-integration"
WT_V3 = ".worktrees/corpus-preservation"
WT_EVIDENCE = ".worktrees/135m-p6-evidence"
WT_360M = ".worktrees/360m-corpus"
STAGED_RECEIPTS = "artifacts/1b-corpus-handoff/receipts"

GENERATED = ("README.md", "build/README.md", "SHA256SUMS", "package-receipt.json")

# Deliberately narrow: "token" would reject train/tokenizer.py and vendor/tiktoken/.
_SECRET_MARKERS = (
    ".aws/", ".env", ".netrc", "credentials", "id_rsa", "id_ed25519",
    ".pem", ".key", "secrets/", "private_key",
)

# member path in package -> path relative to the repository root
FILE_MAP: dict[str, str] = {
    # ---- the corpus contract -------------------------------------------------
    "corpus/reasoning-v3-corpus-manifest.json":
        f"{WT_EVIDENCE}/cluster/aws/reasoning-v3-corpus-manifest.json",
    "corpus/DATASET-POINTER-AWS-135M-V3.json":
        f"{WT_V3}/DATASET-POINTER-AWS-135M-V3.json",
    "corpus/FROZEN.json":
        f"{WT_V3}/artifacts/reasoning-corpus-v3/FROZEN.json",
    "corpus/360m-source-pin-plan.json":
        "artifacts/360m-source-freeze/pin-plan.json",
    "corpus/corpus_fetch_1b.py": "scripts/corpus_fetch_1b.py",
    "corpus/verify_corpus_provenance.py": "scripts/verify_corpus_provenance.py",
    "corpus/receipts/base-receipt.json": f"{STAGED_RECEIPTS}/base-receipt.json",
    "corpus/receipts/extension-receipt.json": f"{STAGED_RECEIPTS}/extension-receipt.json",
    "corpus/receipts/reasoning-pointer.json": f"{STAGED_RECEIPTS}/reasoning-pointer.json",
    "corpus/recipes/reasoning-dataset-v2.json": f"{WT_V2}/configs/reasoning-dataset-v2.json",
    "corpus/recipes/reasoning-dataset-v3.json": f"{WT_V3}/configs/reasoning-dataset-v3.json",
    # ---- documentation -------------------------------------------------------
    "docs/1B-CORPUS-CONSTRUCTION.md": "docs/1B-CORPUS-CONSTRUCTION.md",
    "docs/1b-cohort-aws-run-record.md": "docs/2026-07-25-1b-cohort-aws-run-record.md",
    "docs/Memory-split-design.md": "Memory-split-design.md",
    "docs/360M-CORPUS-BUILD-PLAN.md": f"{WT_360M}/docs/360M-CORPUS-BUILD-PLAN.md",
    "docs/CORPUS-PRESERVATION-RUNBOOK.md": f"{WT_V3}/docs/CORPUS-PRESERVATION-RUNBOOK.md",
    "docs/AWS-135M-REASONING-V3-RUNBOOK.md": f"{WT_V3}/docs/AWS-135M-REASONING-V3-RUNBOOK.md",
    "docs/PROVENANCE.md": f"{WT_V3}/PROVENANCE.md",
    # ---- base-segment producer (v2) -----------------------------------------
    # cluster/ and cluster/aws/ are namespace packages in this tree; no __init__.py
    "build/v2-base/requirements.txt": f"{WT_V2}/requirements.txt",
    "build/v2-base/train/__init__.py": f"{WT_V2}/train/__init__.py",
    "build/v2-base/train/tokenizer.py": f"{WT_V2}/train/tokenizer.py",
    # ---- extension producer (v3) --------------------------------------------
    "build/v3-extension/requirements.txt": f"{WT_V3}/requirements.txt",
    "build/v3-extension/configs/reasoning-dataset-v3.json":
        f"{WT_V3}/configs/reasoning-dataset-v3.json",
    "build/v3-extension/cluster/__init__.py": f"{WT_V3}/cluster/__init__.py",
    "build/v3-extension/cluster/corpus_contract.py": f"{WT_V3}/cluster/corpus_contract.py",
    "build/v3-extension/cluster/aws/__init__.py": f"{WT_V3}/cluster/aws/__init__.py",
    "build/v3-extension/cluster/aws/p5/__init__.py": f"{WT_V3}/cluster/aws/p5/__init__.py",
    "build/v3-extension/cluster/aws/p5/corpus_contract.py":
        f"{WT_V3}/cluster/aws/p5/corpus_contract.py",
    "build/v3-extension/msctl/__init__.py": f"{WT_V3}/msctl/__init__.py",
    "build/v3-extension/msctl/cohort.py": f"{WT_V3}/msctl/cohort.py",
    "build/v3-extension/train/__init__.py": f"{WT_V3}/train/__init__.py",
    "build/v3-extension/train/tokenizer.py": f"{WT_V3}/train/tokenizer.py",
    # ---- 1B cohort operations ------------------------------------------------
    "cohort-1b/gen_configs_1b.py": ".swarm/tmp/gen_configs_1b.py",
    "cohort-1b/preflight_1b.py": ".swarm/tmp/preflight_1b.py",
    "cohort-1b/run_cohort.sh": ".swarm/tmp/run_cohort.sh",
    "cohort-1b/sync_daemon.sh": ".swarm/tmp/sync_daemon.sh",
}

# (package prefix, repo directory, glob, recursive) -- bulk copies
TREE_MAP: tuple[tuple[str, str, str, bool], ...] = (
    ("build/v2-base/corpusgen", f"{WT_V2}/corpusgen", "*.py", True),
    ("build/v2-base/cluster/aws/corpus_builder", f"{WT_V2}/cluster/aws/corpus_builder",
     "*.py", True),
    ("build/v2-base/sources", f"{WT_V2}/sources", "*", False),
    ("build/v2-base/vendor/tiktoken", f"{WT_V2}/vendor/tiktoken", "*", False),
    ("build/v3-extension/corpusgen", f"{WT_V3}/corpusgen", "*.py", True),
)

# individually named files pulled from a worktree's scripts/ or configs/
SCRIPT_MAP: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("build/v2-base/scripts", f"{WT_V2}/scripts", (
        "build_parallel_corpus.py",
        "resolve_source_lock.py",
        "stage_current_sources.py",
        "aws_corpus_builder_preflight.py",
        "aws_corpus_builder_launch.py",
        "aws_corpus_cleanroom_verify.py",
        "package_aws_corpus_builder.py",
    )),
    ("build/v2-base/configs", f"{WT_V2}/configs", (
        "reasoning-dataset-v2.json",
        "current-dataset-lock.json",
        "route-policy.json",
        "preregistration-v2.yaml",
        "preregistration-v3.yaml",
        "objective-controls-amendment-v3.yaml",
        "aws-hardware-amendment-v3.json",
    )),
    ("build/v3-extension/scripts", f"{WT_V3}/scripts", (
        "build_reasoning_corpus_v3.py",
        "bridge_135m_dataset.py",
        "stage_135m_dataset.py",
        "build_135m_preflight.py",
    )),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _portable(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or pure.is_absolute()
        or pure.as_posix() != name
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe member path: {name!r}")
    lowered = name.lower()
    for marker in _SECRET_MARKERS:
        if marker in lowered:
            raise ValueError(f"secret-like member path: {name!r}")


def _read(root: Path, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"refusing to package a symlink: {relative}")
    if not path.is_file():
        raise ValueError(f"source file is missing: {relative}")
    return path.read_bytes()


def collect(root: Path) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}

    def add(member: str, data: bytes) -> None:
        _portable(member)
        if member in payload:
            raise ValueError(f"duplicate member: {member}")
        payload[member] = data

    for member, source in FILE_MAP.items():
        add(member, _read(root, source))

    for prefix, directory, pattern, recursive in TREE_MAP:
        base = root / directory
        if not base.is_dir():
            raise ValueError(f"source directory is missing: {directory}")
        globber = base.rglob if recursive else base.glob
        found = 0
        for path in sorted(globber(pattern)):
            if not path.is_file() or path.is_symlink():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            add(f"{prefix}/{path.relative_to(base).as_posix()}", path.read_bytes())
            found += 1
        if not found:
            raise ValueError(f"no files matched {pattern} under {directory}")

    for prefix, directory, names in SCRIPT_MAP:
        for name in names:
            add(f"{prefix}/{name}", _read(root, f"{directory}/{name}"))

    return payload


def _revision(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _corpus_facts(payload: dict[str, bytes]) -> dict:
    manifest = json.loads(payload["corpus/reasoning-v3-corpus-manifest.json"])
    prefix = _sha(payload["corpus/reasoning-v3-corpus-manifest.json"])
    return {
        "bucket": "memorysplit-stephen-056956104102-us-east-1",
        "bytes_total": sum(entry["bytes"] for entry in manifest["objects"]),
        "contract_id": manifest["contract_id"],
        "included_in_archive": False,
        "object_count": len(manifest["objects"]),
        "raw_target_tokens": manifest["raw_target_tokens"],
        "s3_prefix": f"corpus/{prefix}/",
        "transfer_manifest_sha256": prefix,
    }


def _readme(payload: dict[str, bytes], corpus: dict) -> bytes:
    gb = corpus["bytes_total"] / 10**9
    counts: dict[str, int] = {}
    for member in payload:
        group = member.split("/")[0] if "/" in member else "(root)"
        counts[group] = counts.get(group, 0) + 1
    # account for members generated after this table is rendered
    counts["(root)"] = counts.get("(root)", 0) + 3  # README, SHA256SUMS, receipt
    counts["build"] = counts.get("build", 0) + 1  # build/README.md
    order = ["corpus", "docs", "build", "cohort-1b", "(root)"]
    table = "\n".join(
        f"| `{name}{'' if name == '(root)' else '/'}` | {counts[name]} |"
        for name in order
        if name in counts
    )
    return f"""# MemorySplit 1B corpus handoff

Everything needed to obtain, verify, and understand the corpus behind the
billion-parameter cohort `memorysplit-exploratory-v3-1b-aws-n8`.

**The corpus bytes are not in this archive.** They are {gb:.1f} GB across
{corpus['object_count']} objects and live in S3. This package carries the
contract — manifest, locks, receipts, checksums — plus the tools to fetch and
verify the real thing, and the full producer source.

## Start here

1. Read `docs/1B-CORPUS-CONSTRUCTION.md`. It explains what the corpus is, how
   the two segments were built, and what the numbers mean.
2. Prove the package is internally consistent, offline and in about a second:

   ```bash
   python3 corpus/verify_corpus_provenance.py
   ```

3. Fetch the bytes. You need AWS credentials for account `056956104102` and
   about {gb:.0f} GB of free space.

   ```bash
   export AWS_PROFILE=<your-profile> AWS_REGION=us-east-1
   python3 corpus/corpus_fetch_1b.py plan --dest /mnt/nvme/corpus
   python3 corpus/corpus_fetch_1b.py fetch --dest /mnt/nvme/corpus
   python3 corpus/corpus_fetch_1b.py verify --root /mnt/nvme/corpus
   ```

   `fetch` verifies each object's SHA-256 on arrival. `verify` also recomputes
   the three composite stream digests that training actually reads.

## The corpus in one table

| | |
|---|---|
| Contract | `{corpus['contract_id']}` |
| Tokens | {corpus['raw_target_tokens']:,} |
| Bytes | {corpus['bytes_total']:,} ({gb:.2f} GB) |
| Objects | {corpus['object_count']} |
| Location | `s3://{corpus['bucket']}/{corpus['s3_prefix']}` |
| Manifest SHA-256 | `{corpus['transfer_manifest_sha256']}` |

The S3 prefix is the SHA-256 of the transfer manifest, so the data's address is
a function of its description.

## Layout

| Path | Files |
|---|---:|
{table}

- `corpus/` — the contract: transfer manifest, dataset pointer, `FROZEN.json`,
  the receipts recovered from S3, both recipes, and the fetch/verify tools.
- `docs/` — methodology, the 1B run record, the design doc, and the runbooks.
- `build/v2-base/` — producer for the 7.12B-token base segment.
- `build/v3-extension/` — producer for the 1.049B-token reasoning extension.
- `cohort-1b/` — config generator, corpus preflight, training supervisor, S3 sync
  daemon, exactly as used on the B200 node.

See `build/README.md` before running anything in `build/`.

## Integrity

`SHA256SUMS` lists every other member. `package-receipt.json` records what this
archive claims to be. To check the archive against itself:

```bash
cd {PREFIX} && shasum -a 256 -c SHA256SUMS
```

## Caveats

- Scope is `successor_exploratory_unpreregistered`. These are not the
  preregistered 135M/360M confirmation runs and must not be reported as such.
- The base segment cannot currently be rebuilt from upstream sources; its source
  freeze is incomplete (`corpus/360m-source-pin-plan.json`). The S3 bytes are the
  artifact. The extension *is* deterministically rebuildable.
- The producer code was never committed to a git branch. These copies are the
  preserved record; the extension receipt pins digests for five of the files.
""".encode()


def _build_readme() -> bytes:
    return """# Producer source trees

Two independent source roots. **Do not merge them onto one `PYTHONPATH`.**

They share module names (`corpusgen`, `cluster`, `train`) but the files differ.
`train/tokenizer.py` is the clearest example: the copies are not the same, and
only the `v3-extension` one matches the digest pinned in the extension receipt.
Importing the wrong one silently changes tokenization.

| Tree | Builds | Tokens |
|---|---|---:|
| `v2-base/` | base segment, 8 lanes | 7,120,879,616 |
| `v3-extension/` | reasoning extension, 14 tasks | 1,048,576,000 |

Run each from its own directory:

```bash
cd build/v3-extension
PYTHONHASHSEED=0 python3 scripts/build_reasoning_corpus_v3.py --help
```

```bash
cd build/v2-base
PYTHONHASHSEED=0 python3 scripts/build_parallel_corpus.py --help
```

## Runtime lock

The builders fail closed outside this environment, and so should you:

| | |
|---|---|
| Python | CPython 3.12.3 |
| `PYTHONHASHSEED` | `0` (hash randomization off) |
| tiktoken | 0.13.0 |
| Reasoning-Gym | 0.1.19 |
| Byte order | little-endian |

`v2-base/vendor/tiktoken/` holds the vendored GPT-2 BPE assets so the tokenizer
loads with no network access.

## What actually rebuilds

**The extension is reproducible.** Given Reasoning-Gym 0.1.19 and the pinned
runtime, `scripts/build_reasoning_corpus_v3.py` regenerates all 1,048,576,000
tokens and the receipt's digests should match exactly.

**The base segment is not, today.** Its eleven upstream sources are not fully
pinned — see `corpus/360m-source-pin-plan.json`, which reports `complete: false`
with 12 unpinnable entries. `scripts/resolve_source_lock.py` is the tool that
would close that gap. Until then, treat the base bytes in S3 as the artifact
rather than as a cache of something you can regenerate.

## Entry points

`v2-base/`
- `scripts/build_parallel_corpus.py build-production` — the real builder.
- `scripts/resolve_source_lock.py` — freeze upstream sources. Offline unless
  given `--network`.
- `scripts/aws_corpus_builder_preflight.py` / `_launch.py` — the 14-gate
  preflight and digest-approved launch for the air-gapped builder instance.
- `scripts/aws_corpus_cleanroom_verify.py` — re-download from S3 and re-verify.
- `corpusgen/reasoning_v2/` — lane catalogs, renderers, source locks, Wikidata.
- `corpusgen/parallel/` — sharded scheduling, atomic publication.

`v3-extension/`
- `scripts/build_reasoning_corpus_v3.py` — build or verify the extension.
- `corpusgen/reasoning_expansion.py` — the producer.
- `corpusgen/reasoning_oracles.py` — independent answer validators. These are
  what drove oracle rejections to zero across all 14 tasks.
""".encode()


def build_payload(root: Path) -> dict[str, bytes]:
    payload = collect(root)
    corpus = _corpus_facts(payload)
    payload["README.md"] = _readme(payload, corpus)
    payload["build/README.md"] = _build_readme()
    receipt = {
        "archive_format": "memorysplit-1b-corpus-handoff-v1",
        "cohort_id": "memorysplit-exploratory-v3-1b-aws-n8",
        "corpus": corpus,
        "member_count": len(payload) + 2,
        "schema_version": 1,
        "scientific_scope": "successor_exploratory_unpreregistered",
        "source_identity_sha256": _sha(
            "".join(f"{_sha(payload[m])}  {m}\n" for m in sorted(payload)).encode()
        ),
        "source_revision": _revision(root),
    }
    payload["package-receipt.json"] = _json_bytes(receipt)
    payload["SHA256SUMS"] = "".join(
        f"{_sha(payload[m])}  {m}\n" for m in sorted(payload)
    ).encode()
    return payload


def write_zip(destination: Path, payload: dict[str, bytes]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for member in sorted(payload):
            info = zipfile.ZipInfo(f"{PREFIX}/{member}", date_time=NORMALIZED_TIME)
            info.external_attr = FILE_MODE << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, payload[member])
    with zipfile.ZipFile(temporary) as handle:
        if handle.testzip() is not None:
            raise ValueError("archive failed its CRC check")
    temporary.replace(destination)
    return destination


def verify(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names):
            raise ValueError("members are not sorted")
        if len(names) != len(set(names)):
            raise ValueError("members are not unique")
        for info in infos:
            if info.is_dir():
                raise ValueError(f"directory entry present: {info.filename}")
            if info.external_attr >> 16 != FILE_MODE:
                raise ValueError(f"unexpected mode on {info.filename}")
            if info.date_time != NORMALIZED_TIME:
                raise ValueError(f"unexpected timestamp on {info.filename}")
            if not info.filename.startswith(f"{PREFIX}/"):
                raise ValueError(f"member outside the archive prefix: {info.filename}")
            _portable(info.filename[len(PREFIX) + 1:])
        payload = {
            info.filename[len(PREFIX) + 1:]: handle.read(info.filename)
            for info in infos
        }

    expected = "".join(
        f"{_sha(payload[m])}  {m}\n" for m in sorted(set(payload) - {"SHA256SUMS"})
    )
    if payload["SHA256SUMS"].decode() != expected:
        raise ValueError("SHA256SUMS does not describe the archive contents")

    receipt = json.loads(payload["package-receipt.json"])
    if receipt["archive_format"] != "memorysplit-1b-corpus-handoff-v1":
        raise ValueError("unexpected archive format")
    if receipt["corpus"]["included_in_archive"] is not False:
        raise ValueError("receipt claims corpus bytes are included")

    manifest_sha = _sha(payload["corpus/reasoning-v3-corpus-manifest.json"])
    if manifest_sha != receipt["corpus"]["transfer_manifest_sha256"]:
        raise ValueError("manifest digest disagrees with the receipt")

    missing = [m for m in GENERATED if m not in payload]
    if missing:
        raise ValueError(f"generated members missing: {missing}")

    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha(archive.read_bytes()),
        "corpus_bytes_included": False,
        "corpus_s3_prefix": receipt["corpus"]["s3_prefix"],
        "member_count": len(payload),
        "transfer_manifest_sha256": manifest_sha,
        "uncompressed_bytes": sum(len(v) for v in payload.values()),
        "verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", default=f"artifacts/{ARCHIVE_NAME}")
    build.add_argument("--source-root", default=str(ROOT))
    verify_cmd = commands.add_parser("verify")
    verify_cmd.add_argument("archive")
    args = parser.parse_args(argv)

    if args.command == "build":
        root = Path(args.source_root).resolve()
        destination = Path(args.output)
        if not destination.is_absolute():
            destination = root / destination
        archive = write_zip(destination, build_payload(root))
    else:
        archive = Path(args.archive).resolve()

    print(json.dumps(verify(archive), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
