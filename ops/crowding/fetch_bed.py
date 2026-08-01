"""Pin the natural-text bed to a hashed JSONL.

The bed is the only lane the generators do not produce, so it is the only one
whose provenance can drift. Every previous corpus in this project either used
an unpinned stream or a synthetic filler, and one of them could not be rebuilt
from upstream at all -- the S3 bytes were the artifact.

This writes a fixed number of documents from a fixed dataset revision, then
records the byte count, line count and SHA-256. `build_corpus.py --bed-jsonl`
consumes it, and the corpus manifest carries the path; a manifest that says
SYNTHETIC-REHEARSAL-ONLY is a build that must not carry a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SPLIT = "train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=400_000)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--revision", default=None,
                    help="dataset revision to pin; recorded either way")
    args = ap.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET, name=CONFIG, split=SPLIT, streaming=True,
                      revision=args.revision)
    n = 0
    with open(out, "w") as fh:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            fh.write(json.dumps({"text": text}) + "\n")
            n += 1
            if n % 25_000 == 0:
                print(f"  {n:,} docs", flush=True)
            if n >= args.n_docs:
                break

    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": args.revision,
        "n_docs": n,
        "min_chars": args.min_chars,
        "bytes": out.stat().st_size,
        "sha256": h.hexdigest(),
        "path": str(out),
    }
    man_path = out.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nbed pinned -> {out}\nmanifest   -> {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
