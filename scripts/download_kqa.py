#!/usr/bin/env python
"""Download the public KQA Pro JSON files from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/drt/kqa_pro/resolve/main"
FILES = {
    "kb.json": "04da7408320c5cb7023c44372cce32846d56d369d8865d2e61a18c3956661a7c",
    "train.json": "e9fbe4c1cdf207aac83ae0d5e4a1a53a9965a2b13b403de699ca6d5dae6e4510",
    "val.json": "b4aed6ab3d7ad071722064fe3bb02bc028cfbeb15da5f7115d57a1e2d198f3bb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    expected_sha256: str,
    force: bool = False,
) -> None:
    if (
        destination.exists()
        and destination.stat().st_size > 0
        and _sha256(destination) == expected_sha256
        and not force
    ):
        print(f"exists: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "memory-split/1.0"})
    print(f"download: {url}")
    with urllib.request.urlopen(request) as response, open(temporary, "wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = _sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {destination.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    os.replace(temporary, destination)
    print(f"saved: {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for filename, checksum in FILES.items():
        download_file(
            f"{BASE_URL}/{filename}",
            out_dir / filename,
            checksum,
            force=args.force,
        )


if __name__ == "__main__":
    main()
