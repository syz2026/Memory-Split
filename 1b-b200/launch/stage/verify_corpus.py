#!/usr/bin/env python3
"""Re-hash the staged corpus against the transfer manifest.

Two independent levels of checking:

1. Per-object: every entry in ``objects`` must match on both size and sha256.
2. Per-stream: the ``composite_stream_sha256`` values are hashes over the
   *concatenation* of each stream's parts in declared order, which is what the
   trainer actually reads. Per-object hashes passing does not by itself prove
   the stream ordering is right, so both are checked.

Hashing 31.69 GB is I/O bound, so objects are hashed in parallel across the
box's many cores; the composite pass is inherently sequential per stream but
the three streams run concurrently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 16 * 1024 * 1024


def sha256_files(paths: list[Path]) -> str:
    """sha256 over the byte concatenation of ``paths``, in order."""
    h = hashlib.sha256()
    for p in paths:
        with p.open("rb") as fh:
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                h.update(block)
    return h.hexdigest()


def check_object(root: Path, entry: dict) -> tuple[bool, str]:
    path = root / entry["path"]
    if not path.is_file():
        return False, f"MISSING  {entry['path']}"

    size = path.stat().st_size
    if size != entry["bytes"]:
        return False, f"SIZE     {entry['path']} got {size} want {entry['bytes']}"

    digest = sha256_files([path])
    if digest != entry["sha256"]:
        return False, f"SHA256   {entry['path']} got {digest} want {entry['sha256']}"

    return True, f"ok       {entry['path']} ({size} bytes)"


def check_stream(root: Path, name: str, paths: list[str], expected: str) -> tuple[bool, str]:
    resolved = [root / p for p in paths]
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        return False, f"MISSING  stream {name}: {missing}"

    digest = sha256_files(resolved)
    if digest != expected:
        return False, f"SHA256   stream {name} got {digest} want {expected}"
    return True, f"ok       stream {name} ({' + '.join(paths)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--pointer", type=Path, default=None,
                    help="optional DATASET-POINTER json, to cross-check stream hashes")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    root = args.root
    ok = True

    print(f"contract_id       : {manifest['contract_id']}")
    print(f"raw_target_tokens : {manifest['raw_target_tokens']}")
    print(f"objects declared  : {len(manifest['objects'])}")
    print()

    print("--- per-object ---")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for good, msg in pool.map(lambda e: check_object(root, e), manifest["objects"]):
            print(msg)
            ok &= good

    # Stream membership lives in the dataset pointer, not the transfer manifest,
    # so reconstruct it from the well-known layout the pointer declares.
    streams = {
        "packed_targets": ["base/packed/targets.bin", "extension/packed/targets.bin"],
        "dense_target_weights": [
            "base/sidecars/dense_target_weights.bin",
            "extension/sidecars/shared_target_weights.bin",
        ],
        "split90_target_weights": [
            "base/sidecars/split90_target_weights.bin",
            "extension/sidecars/shared_target_weights.bin",
        ],
    }
    if args.pointer and args.pointer.is_file():
        pointer = json.loads(args.pointer.read_text())
        streams = {k: v["paths"] for k, v in pointer["streams"].items()}

    print()
    print("--- composite streams ---")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            name: pool.submit(
                check_stream, root, name, paths,
                manifest["composite_stream_sha256"][name],
            )
            for name, paths in streams.items()
        }
        for name, fut in futures.items():
            good, msg = fut.result()
            print(msg)
            ok &= good

    # The extension receipt doubles as the corpus-level virtual receipt; if this
    # disagrees the staged tree is not the frozen corpus the preregistration names.
    print()
    print("--- virtual receipt ---")
    receipt = root / "extension/receipt.json"
    if receipt.is_file():
        digest = sha256_files([receipt])
        want = manifest["virtual_receipt_sha256"]
        good = digest == want
        print(f"{'ok      ' if good else 'MISMATCH'} virtual_receipt {digest}")
        ok &= good
    else:
        print("MISSING  extension/receipt.json")
        ok = False

    print()
    print("CORPUS VERIFICATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
