"""Build reasoning-v4: the frozen base, thinned, interleaved with a
high-exposure fact segment, with the reasoning extension left at the tail.

    [ base documents (sampled) + fact documents (interleaved) ] [ extension ]
    |<-------------- 7,120,879,616 tokens -------------------->|<- 1.049B ->|
                        total 8,169,455,616

Why interleave rather than append. The reasoning extension is already the tail
of the stream; putting facts after it would leave them barely consolidated, and
putting them before it in one block would mass all 200 exposures of every fact
into one stretch of training. Interleaving spreads each fact's exposures across
the whole run, which is what makes repetition stick.

Why thin the base. The budget is fixed at 8,169,455,616 tokens so `max_steps`
stays 15,582 and every downstream config, snapshot step and analysis window is
unchanged. Facts have to displace base tokens rather than add to them. Base
documents are sampled systematically across the whole segment so lane
proportions survive the thinning -- taking a prefix instead would over-weight
whichever lane happens to come first, and the head lane is entirely unmasked.

Output matches the reasoning-v3 layout so the trainer, config generator and
analysis need no changes:

    base/packed/targets.bin                     uint16 tokens
    base/sidecars/dense_target_weights.bin      uint8, all ones
    base/sidecars/split90_target_weights.bin    uint8, zero on offloaded targets
    extension/...                               copied byte-for-byte from v3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Invoked by path on the cluster, so sys.path[0] is this directory and the repo
# packages are not importable without help. Same trap that bit run_train.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpusgen import bios, fact_segment as fs  # noqa: E402
from train.tokenizer import get_tok  # noqa: E402

BASE_TOKENS = 7_120_879_616
EXTENSION_TOKENS = 1_048_576_000
TOTAL_TOKENS = BASE_TOKENS + EXTENSION_TOKENS
EOT = 50260
CHUNK = 1 << 26          # 64 Mi tokens per streaming pass


def index_documents(path: Path, limit: int | None = None) -> np.ndarray:
    """Offsets of document starts in the frozen base, found via <|eot|>.

    One linear pass. The producer is not in this repository, so the separator
    token is the only document structure available.
    """
    toks = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(toks) if limit is None else min(limit, len(toks))
    starts = [0]
    pos = 0
    while pos < n:
        end = min(pos + CHUNK, n)
        blk = np.asarray(toks[pos:end])
        hits = np.flatnonzero(blk == EOT)
        if hits.size:
            starts.extend((pos + hits + 1).tolist())
        pos = end
    if starts and starts[-1] >= n:
        starts.pop()
    return np.asarray(starts, dtype=np.int64)


def select_base_documents(starts: np.ndarray, n_tokens: int, target: int) -> np.ndarray:
    """Systematic sample of document indices totalling ~`target` tokens.

    Uniform stride over the whole segment, so every lane contributes in
    proportion to its size.
    """
    lengths = np.diff(np.append(starts, n_tokens))
    total = int(lengths.sum())
    if target >= total:
        return np.arange(len(starts))
    keep_frac = target / total
    stride = 1.0 / keep_frac
    idx = np.floor(np.arange(0, len(starts), stride)).astype(np.int64)
    return idx[idx < len(starts)]


class StreamWriter:
    """Writes tokens plus both sidecars, stopping exactly at `limit` tokens."""

    def __init__(self, out: Path, limit: int):
        self.limit = limit
        self.written = 0
        (out / "base/packed").mkdir(parents=True, exist_ok=True)
        (out / "base/sidecars").mkdir(parents=True, exist_ok=True)
        self.f_tok = open(out / "base/packed/targets.bin", "wb")
        self.f_dense = open(out / "base/sidecars/dense_target_weights.bin", "wb")
        self.f_split = open(out / "base/sidecars/split90_target_weights.bin", "wb")

    def write(self, ids: np.ndarray, split_mask: np.ndarray) -> bool:
        """Returns False once the budget is exhausted."""
        room = self.limit - self.written
        if room <= 0:
            return False
        if len(ids) > room:
            ids, split_mask = ids[:room], split_mask[:room]
        self.f_tok.write(ids.tobytes())
        self.f_dense.write(np.ones(len(ids), dtype=np.uint8).tobytes())
        self.f_split.write(split_mask.tobytes())
        self.written += len(ids)
        return self.written < self.limit

    def close(self) -> None:
        for f in (self.f_tok, self.f_dense, self.f_split):
            f.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="frozen reasoning-v3 corpus root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fact-tokens", type=int, default=4_000_000_000)
    ap.add_argument("--exposures", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-base", type=int, default=0,
                    help="debug: only index this many base tokens")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tok()

    # --limit-base shrinks the source, so the segment target has to shrink with
    # it or the ratio degenerates and the streams concatenate instead of
    # interleaving -- which would make a small-scale rehearsal misleading.
    segment_tokens = args.limit_base + args.fact_tokens if args.limit_base else BASE_TOKENS
    base_budget = segment_tokens - args.fact_tokens
    if base_budget <= 0:
        return print(f"fact budget {args.fact_tokens:,} leaves no room in "
                     f"{segment_tokens:,}") or 1
    plan = fs.plan(args.fact_tokens, args.exposures)
    print(f"fact segment : {plan['n_entities']:,} entities x {args.exposures} exposures "
          f"= {plan['n_docs']:,} docs, ~{args.fact_tokens:,} tokens")
    print(f"               {plan['total_bits']/1e6:.1f} Mbit -> "
          + "  ".join(f"{k} {v:.2f} b/param" for k, v in plan["bits_per_param"].items()))
    print(f"base segment : {base_budget:,} tokens sampled from "
          f"{args.limit_base or BASE_TOKENS:,}")
    print(f"extension    : {EXTENSION_TOKENS:,} tokens copied unchanged")

    t0 = time.time()
    src_base = src / "base/packed/targets.bin"
    src_split = src / "base/sidecars/split90_target_weights.bin"
    limit = args.limit_base or None
    print("\nindexing base documents ...", flush=True)
    starts = index_documents(src_base, limit)
    n_base = limit or (src_base.stat().st_size // 2)
    print(f"  {len(starts):,} documents in {n_base:,} tokens "
          f"({time.time()-t0:.0f}s, mean {n_base/max(1,len(starts)):.0f} tok/doc)")

    keep = select_base_documents(starts, n_base, base_budget)
    print(f"  keeping {len(keep):,} of {len(starts):,} documents "
          f"({len(keep)/max(1,len(starts))*100:.1f}%)")

    src_toks = np.memmap(src_base, dtype=np.uint16, mode="r")
    src_msk = np.memmap(src_split, dtype=np.uint8, mode="r")
    bounds = np.append(starts, n_base)

    records = bios.generate_records(plan["n_entities"], args.seed)
    facts = fs.encode_docs(records, fs.doc_order(plan["n_entities"], args.exposures), tok)

    writer = StreamWriter(out, segment_tokens)
    fact_tok = base_tok = 0
    ratio = args.fact_tokens / max(1, base_budget)
    ki = 0
    going = True
    print("\ninterleaving ...", flush=True)
    while going:
        # emit from whichever stream is behind its share
        want_fact = (fact_tok < ratio * base_tok) or ki >= len(keep)
        if want_fact:
            nxt = next(facts, None)
            if nxt is None:
                if ki >= len(keep):
                    break
                want_fact = False
            else:
                ids, msk = nxt
                fact_tok += len(ids)
                going = writer.write(ids, msk)
                continue
        d = int(keep[ki]); ki += 1
        lo, hi = int(bounds[d]), int(bounds[d + 1])
        ids = np.asarray(src_toks[lo:hi])
        msk = np.asarray(src_msk[lo:hi])
        base_tok += len(ids)
        going = writer.write(ids, msk)
        if writer.written % (1 << 29) < (hi - lo):
            print(f"  {writer.written/1e9:.2f}B tokens "
                  f"(fact {fact_tok/1e9:.2f}B, base {base_tok/1e9:.2f}B) "
                  f"{time.time()-t0:.0f}s", flush=True)
    writer.close()

    print(f"\nwrote {writer.written:,} tokens to the interleaved segment "
          f"(target {segment_tokens:,})")
    print(f"  fact {fact_tok:,}  base {base_tok:,}")

    manifest = {
        "contract_id": "memorysplit-reasoning-dataset-v4-highexposure",
        "derived_from": "memorysplit-reasoning-dataset-v3",
        "scientific_scope": "successor_exploratory_unpreregistered",
        "total_tokens": TOTAL_TOKENS,
        "interleaved_tokens": writer.written,
        "extension_tokens": EXTENSION_TOKENS,
        "fact_tokens_emitted": fact_tok,
        "base_tokens_emitted": base_tok,
        "n_entities": plan["n_entities"],
        "n_exposures": args.exposures,
        "bits_total": plan["total_bits"],
        "bits_per_param": plan["bits_per_param"],
        "seed": args.seed,
        "eot_token": EOT,
    }
    (out / "v4-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {out/'v4-manifest.json'}")
    print("next: copy extension/ from the v3 root, then verify with verify_v4.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
