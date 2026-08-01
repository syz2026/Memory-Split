#!/usr/bin/env python3
"""Fetch and verify the frozen memorysplit-reasoning-dataset-v3 corpus.

The corpus is 31.7 GB and lives in S3 under a prefix equal to the SHA-256 of its
own transfer manifest. Every object is checked against that manifest on arrival,
and `verify` additionally recomputes the three composite stream digests that
training actually reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "memorysplit-stephen-056956104102-us-east-1"
DEFAULT_REGION = "us-east-1"
CONTRACT_ID = "memorysplit-reasoning-dataset-v3"
CHUNK = 8 * 1024 * 1024

HERE = Path(__file__).resolve().parent
MANIFEST_CANDIDATES = (
    HERE / "reasoning-v3-corpus-manifest.json",
    HERE.parent / "corpus" / "reasoning-v3-corpus-manifest.json",
    HERE.parent / "cluster" / "aws" / "reasoning-v3-corpus-manifest.json",
)
POINTER_CANDIDATES = (
    HERE / "DATASET-POINTER-AWS-135M-V3.json",
    HERE.parent / "corpus" / "DATASET-POINTER-AWS-135M-V3.json",
    HERE.parent / "DATASET-POINTER-AWS-135M-V3.json",
)


class VerificationError(RuntimeError):
    """Raised when the corpus does not match its frozen description."""


def _locate(candidates: tuple[Path, ...], label: str, override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise VerificationError(f"{label} not found: {path}")
        return path
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise VerificationError(
        f"{label} not found; looked in {[str(c) for c in candidates]}"
    )


def _sha256_file(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress and total > (256 << 20):
                _progress(path.name, done, total)
    if progress and total > (256 << 20):
        sys.stderr.write("\r" + " " * 78 + "\r")
        sys.stderr.flush()
    return digest.hexdigest()


def _sha256_concat(paths: list[Path], *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    total = sum(p.stat().st_size for p in paths)
    done = 0
    for path in paths:
        with path.open("rb") as handle:
            while True:
                block = handle.read(CHUNK)
                if not block:
                    break
                digest.update(block)
                done += len(block)
                if progress:
                    _progress("composite", done, total)
    if progress:
        sys.stderr.write("\r" + " " * 78 + "\r")
        sys.stderr.flush()
    return digest.hexdigest()


def _progress(label: str, done: int, total: int) -> None:
    pct = 100.0 * done / total if total else 100.0
    sys.stderr.write(f"\r  hashing {label[:28]:<28} {pct:5.1f}%  {done / 2**30:7.2f} GiB")
    sys.stderr.flush()


def load_manifest(path: Path) -> dict:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("contract_id") != CONTRACT_ID:
        raise VerificationError(
            f"manifest contract_id is {manifest.get('contract_id')!r}, "
            f"expected {CONTRACT_ID!r}"
        )
    manifest["_self_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def prefix_for(manifest: dict) -> str:
    """The S3 prefix is the content address of the manifest itself."""
    return manifest["_self_sha256"]


def _aws() -> str:
    binary = shutil.which("aws")
    if binary is None:
        raise VerificationError(
            "the AWS CLI is required; install it or download the objects manually "
            "using the URIs printed by `plan`"
        )
    return binary


def cmd_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(_locate(MANIFEST_CANDIDATES, "transfer manifest", args.manifest))
    prefix = prefix_for(manifest)
    objects = manifest["objects"]
    total = sum(entry["bytes"] for entry in objects)
    print(f"contract       {manifest['contract_id']}")
    print(f"tokens         {manifest['raw_target_tokens']:,}")
    print(f"objects        {len(objects)}")
    print(f"total bytes    {total:,}  ({total / 10**9:.2f} GB / {total / 2**30:.2f} GiB)")
    print(f"manifest sha   {prefix}")
    print(f"s3 prefix      s3://{args.bucket}/corpus/{prefix}/")
    print()
    print(f"{'bytes':>14}  {'sha256':<16}  path")
    print("-" * 78)
    for entry in objects:
        print(f"{entry['bytes']:>14,}  {entry['sha256'][:16]}  {entry['path']}")
    print()
    free = shutil.disk_usage(args.dest if args.dest else ".").free
    print(f"free space at {args.dest or '.'}: {free / 10**9:.2f} GB", end="")
    print("  (INSUFFICIENT)" if free < total else "  (ok)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest(_locate(MANIFEST_CANDIDATES, "transfer manifest", args.manifest))
    prefix = prefix_for(manifest)
    dest = Path(args.dest).resolve()
    objects = manifest["objects"]
    if args.object:
        wanted = set(args.object)
        objects = [entry for entry in objects if entry["path"] in wanted]
        if not objects:
            raise VerificationError(f"no manifest object matches {sorted(wanted)}")

    total = sum(entry["bytes"] for entry in objects)
    free = shutil.disk_usage(dest.parent if not dest.exists() else dest).free
    if free < total and not args.force:
        raise VerificationError(
            f"need {total / 10**9:.2f} GB but only {free / 10**9:.2f} GB free at {dest}; "
            "pass --force to try anyway"
        )

    aws = _aws()
    env = dict(os.environ)
    env.setdefault("AWS_REGION", args.region)
    failures: list[str] = []

    for index, entry in enumerate(objects, start=1):
        target = dest / entry["path"]
        uri = f"s3://{args.bucket}/corpus/{prefix}/{entry['path']}"
        label = f"[{index}/{len(objects)}] {entry['path']}"

        if target.is_file() and target.stat().st_size == entry["bytes"]:
            if args.skip_existing:
                print(f"{label}: present, size ok, skipped (--skip-existing)")
                continue
            if _sha256_file(target, progress=True) == entry["sha256"]:
                print(f"{label}: already present and verified")
                continue
            print(f"{label}: present but digest differs, re-downloading")

        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"{label}: downloading {entry['bytes'] / 10**9:.2f} GB")
        if args.dry_run:
            print(f"    would run: aws s3 cp {uri} {target}")
            continue

        result = subprocess.run(
            [aws, "s3", "cp", uri, str(target), "--only-show-errors"],
            env=env,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"{entry['path']}: aws s3 cp exited {result.returncode}")
            continue

        actual_size = target.stat().st_size
        if actual_size != entry["bytes"]:
            failures.append(
                f"{entry['path']}: size {actual_size} != expected {entry['bytes']}"
            )
            continue
        actual = _sha256_file(target, progress=True)
        if actual != entry["sha256"]:
            failures.append(f"{entry['path']}: sha256 {actual} != {entry['sha256']}")
            continue
        print(f"{label}: verified")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        raise VerificationError(f"{len(failures)} object(s) failed to fetch cleanly")
    if args.dry_run:
        print("\ndry run complete; nothing downloaded")
        return 0
    print(f"\nall {len(objects)} object(s) fetched and verified into {dest}")
    print(f"now run: {sys.argv[0]} verify --root {dest}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(_locate(MANIFEST_CANDIDATES, "transfer manifest", args.manifest))
    root = Path(args.root).resolve()
    problems: list[str] = []

    print(f"verifying {root}")
    print(f"contract  {manifest['contract_id']}")
    print(f"manifest  {prefix_for(manifest)}")
    print()

    print("per-object digests")
    for entry in manifest["objects"]:
        path = root / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            print(f"  MISSING  {entry['path']}")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            problems.append(f"size differs: {entry['path']} ({size} != {entry['bytes']})")
            print(f"  BAD SIZE {entry['path']}")
            continue
        if args.quick:
            print(f"  size ok  {entry['path']}")
            continue
        actual = _sha256_file(path, progress=True)
        if actual != entry["sha256"]:
            problems.append(f"digest differs: {entry['path']}")
            print(f"  BAD SHA  {entry['path']}")
        else:
            print(f"  ok       {entry['path']}")

    if args.quick:
        print("\nquick mode: composite stream digests not checked")
    else:
        pointer_path = _locate(POINTER_CANDIDATES, "dataset pointer", args.pointer)
        pointer = json.loads(pointer_path.read_bytes())
        print("\ncomposite stream digests (what training reads)")
        for name in sorted(pointer["streams"]):
            stream = pointer["streams"][name]
            paths = [root / rel for rel in stream["paths"]]
            missing = [p for p in paths if not p.is_file()]
            if missing:
                problems.append(f"stream {name}: missing members")
                print(f"  MISSING  {name}")
                continue
            actual = _sha256_concat(paths, progress=True)
            if actual != stream["sha256"]:
                problems.append(f"stream {name}: digest differs")
                print(f"  BAD SHA  {name}")
            else:
                print(f"  ok       {name}  ({' + '.join(stream['paths'])})")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        raise VerificationError(f"{len(problems)} verification failure(s)")
    print(f"corpus verified: {manifest['raw_target_tokens']:,} tokens, "
          f"{len(manifest['objects'])} objects")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("MS_CORPUS_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--manifest", help="path to reasoning-v3-corpus-manifest.json")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="show what would be downloaded")
    plan.add_argument("--dest", help="check free space at this path")
    plan.set_defaults(func=cmd_plan)

    fetch = commands.add_parser("fetch", help="download and verify every object")
    fetch.add_argument("--dest", required=True, help="corpus root to write into")
    fetch.add_argument("--object", action="append", help="fetch only this manifest path")
    fetch.add_argument("--skip-existing", action="store_true",
                       help="trust size alone for files already present")
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--force", action="store_true", help="ignore the free-space check")
    fetch.set_defaults(func=cmd_fetch)

    verify = commands.add_parser("verify", help="verify a local corpus root")
    verify.add_argument("--root", required=True)
    verify.add_argument("--pointer", help="path to DATASET-POINTER-AWS-135M-V3.json")
    verify.add_argument("--quick", action="store_true", help="check sizes only")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
