"""Build one corpus: a single token stream plus the two mask sidecars.

    [ fact lane | iGSM | deduction | natural bed ]  interleaved, never blocked

Output layout, per fact load:

    targets.bin     uint16 tokens
    factmask.bin    uint8, 0 on fact-value targets
    randpos.bin     uint8, 0 on matched non-value spans
    manifest.json   sizes, hashes, occupancy, and the mask audit

SUP is implicit: an all-ones sidecar is not written, because the loader
defaults to full supervision when no mask path is configured, and writing a
gigabyte of ones per run would be the largest file in the corpus.

Lanes are interleaved by running-token ratio rather than concatenated. A
blocked layout puts the reasoning lane at one end of the schedule, which is
what made the previous corpus's endpoint a threshold coin-flip: both arms saw
reasoning data only in the last 12.8% of training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpusgen import bios, deduction, factlane, igsm_lite, randpos  # noqa: E402
from train.tokenizer import get_tok  # noqa: E402

LANES = ("fact", "igsm", "deduction", "bed")


class StreamWriter:
    """Writes the token stream and both sidecars, stopping at `limit`."""

    def __init__(self, out: Path, limit: int):
        self.limit = limit
        self.written = 0
        out.mkdir(parents=True, exist_ok=True)
        self.f_tok = open(out / "targets.bin", "wb")
        self.f_fact = open(out / "factmask.bin", "wb")
        self.f_rand = open(out / "randpos.bin", "wb")

    def write(self, ids: np.ndarray, fact: np.ndarray, rand: np.ndarray) -> bool:
        room = self.limit - self.written
        if room <= 0:
            return False
        if len(ids) > room:
            ids, fact, rand = ids[:room], fact[:room], rand[:room]
        self.f_tok.write(ids.tobytes())
        self.f_fact.write(fact.tobytes())
        self.f_rand.write(rand.tobytes())
        self.written += len(ids)
        return self.written < self.limit

    def close(self):
        for f in (self.f_tok, self.f_fact, self.f_rand):
            f.close()


def _plain_doc(tok, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A non-fact document: fully supervised in every arm."""
    ids = np.asarray(tok.encode(text) + [tok.EOT], dtype=np.uint16)
    ones = np.ones(len(ids), dtype=np.uint8)
    return ids, ones, ones.copy()


def fact_stream(records, exposures, tok, rng, token_nll=None):
    for ids, fmask in factlane.emit(records, exposures, tok):
        rmask = randpos.build(ids, fmask, rng, token_nll=None)
        yield ids, fmask, rmask


def igsm_stream(tok, op_lo, op_hi, seed, mod):
    rng = random.Random(seed)
    while True:
        p = igsm_lite.generate_problem(
            rng.choices(
                list(range(op_lo, op_hi + 1)),
                [1.0 / k for k in range(op_lo, op_hi + 1)],
            )[0],
            rng,
            mod=mod,
        )
        yield _plain_doc(tok, p.prompt + " " + p.cot)


def deduction_stream(tok, depth_lo, depth_hi, seed):
    rng = random.Random(seed)
    attempt = 0
    while True:
        depth = rng.randint(depth_lo, depth_hi)
        p = deduction.generate_problem(depth, rng, answer_yes=(attempt % 2 == 0))
        attempt += 1
        yield _plain_doc(tok, p.prompt + " " + p.cot)


def bed_stream(tok, path: Path | None, seed: int):
    """Natural text. A pinned JSONL if given, otherwise a deterministic
    synthetic filler so the builder is testable offline. Synthetic filler is
    for rehearsal only and the manifest records which was used."""
    if path is not None:
        while True:
            with open(path) as fh:
                for line in fh:
                    text = json.loads(line).get("text", "").strip()
                    if text:
                        yield _plain_doc(tok, text)
    else:
        rng = random.Random(seed)
        words = list(igsm_lite.ADJECTIVES + igsm_lite.NOUNS + igsm_lite.PLACES)
        while True:
            n = rng.randint(40, 160)
            yield _plain_doc(tok, " ".join(rng.choice(words) for _ in range(n)))


def build(
    out: Path,
    n_entities: int,
    exposures: int,
    shares: dict[str, float],
    total_tokens: int,
    seed: int,
    mod: int = 23,
    op_band: tuple[int, int] = (1, 4),
    depth_band: tuple[int, int] = (1, 2),
    bed_jsonl: Path | None = None,
    progress_every: int = 0,
) -> dict:
    tok = get_tok()
    rng = random.Random(seed)
    records = bios.generate_records(n_entities, seed)

    budgets = {k: int(total_tokens * v) for k, v in shares.items()}
    streams = {
        "fact": fact_stream(records, exposures, tok, rng),
        "igsm": igsm_stream(tok, op_band[0], op_band[1], seed * 7 + 1, mod),
        "deduction": deduction_stream(tok, depth_band[0], depth_band[1], seed * 7 + 2),
        "bed": bed_stream(tok, bed_jsonl, seed * 7 + 3),
    }
    emitted = dict.fromkeys(LANES, 0)
    exhausted: set[str] = set()

    writer = StreamWriter(out, total_tokens)
    going = True
    while going:
        # Interleave by largest remaining share deficit, so every lane is
        # spread across the whole schedule rather than blocked at one end.
        live = [
            k for k in LANES
            if k not in exhausted and emitted[k] < budgets[k]
        ]
        if not live:
            break
        lane = max(live, key=lambda k: (budgets[k] - emitted[k]) / max(1, budgets[k]))
        nxt = next(streams[lane], None)
        if nxt is None:
            # A finite lane ran out. Give its remaining budget to the bed so
            # the total stays exact: a short corpus changes max_steps, and
            # every arm and load in this design must share a step count.
            exhausted.add(lane)
            budgets["bed"] += budgets[lane] - emitted[lane]
            budgets[lane] = emitted[lane]
            continue
        ids, fmask, rmask = nxt
        emitted[lane] += len(ids)
        going = writer.write(ids, fmask, rmask)
        if progress_every and writer.written % progress_every < len(ids):
            print(f"  {writer.written/1e6:.1f}M tokens {dict(emitted)}", flush=True)
    writer.close()
    if writer.written != total_tokens:
        raise RuntimeError(
            f"corpus is short: wrote {writer.written:,} of {total_tokens:,} tokens"
        )

    return _manifest(out, writer.written, emitted, budgets, n_entities,
                     exposures, seed, mod, op_band, depth_band, bed_jsonl)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(out, written, emitted, budgets, n_entities, exposures, seed,
              mod, op_band, depth_band, bed_jsonl) -> dict:
    fact = np.memmap(out / "factmask.bin", dtype=np.uint8, mode="r")
    rand = np.memmap(out / "randpos.bin", dtype=np.uint8, mode="r")
    n_fact_zero = int((fact == 0).sum())
    n_rand_zero = int((rand == 0).sum())
    overlap = int(((fact == 0) & (rand == 0)).sum())

    man = {
        "n_tokens": written,
        "n_entities": n_entities,
        "exposures": exposures,
        "seed": seed,
        "mod": mod,
        "op_band": list(op_band),
        "depth_band": list(depth_band),
        "bed": str(bed_jsonl) if bed_jsonl else "SYNTHETIC-REHEARSAL-ONLY",
        "lane_tokens": dict(emitted),
        "lane_budgets": dict(budgets),
        "bits_total": n_entities * factlane.BITS_PER_ENTITY,
        "bits_per_param": {
            "d8m": n_entities * factlane.BITS_PER_ENTITY / 7_931_776,
            "d40m": n_entities * factlane.BITS_PER_ENTITY / 40_560_000,
            "d160m": n_entities * factlane.BITS_PER_ENTITY / 162_220_800,
        },
        "mask_audit": {
            "factmask_zeros": n_fact_zero,
            "randpos_zeros": n_rand_zero,
            "mass_matched": n_fact_zero == n_rand_zero,
            "overlap": overlap,
            "factmask_frac": n_fact_zero / max(1, written),
        },
        "sha256": {
            p.name: _sha256(p)
            for p in sorted(out.glob("*.bin"))
        },
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man


def verify(out: Path, expect_tokens: int | None = None) -> list[str]:
    """Fail-closed checks. A corpus that is silently short, or whose arms are
    not nested, would waste an entire cohort."""
    fails: list[str] = []
    tok_path, f_path, r_path = (out / "targets.bin", out / "factmask.bin",
                                out / "randpos.bin")
    n = tok_path.stat().st_size // 2
    if expect_tokens is not None and n != expect_tokens:
        fails.append(f"tokens {n} != expected {expect_tokens}")
    for label, p in (("factmask", f_path), ("randpos", r_path)):
        if p.stat().st_size != n:
            fails.append(f"{label} sidecar {p.stat().st_size} bytes != {n} tokens")
    if fails:
        return fails

    fact = np.memmap(f_path, dtype=np.uint8, mode="r")
    rand = np.memmap(r_path, dtype=np.uint8, mode="r")
    nf, nr = int((fact == 0).sum()), int((rand == 0).sum())
    if nf == 0:
        fails.append("CRITICAL: factmask masks nothing")
    if nr == 0:
        fails.append("CRITICAL: randpos masks nothing")
    if nf != nr:
        fails.append(f"mask mass differs: factmask {nf} vs randpos {nr}")
    if int(((fact == 0) & (rand == 0)).sum()):
        fails.append("randpos overlaps a fact-value span")
    # Both arms must be strict subsets of SUP, which is all ones by definition,
    # so any value other than 0/1 is a corrupt sidecar.
    for label, arr in (("factmask", fact), ("randpos", rand)):
        if not np.isin(np.unique(np.asarray(arr[: min(len(arr), 1 << 24)])), [0, 1]).all():
            fails.append(f"{label} contains values outside {{0,1}}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--entities", type=int, required=True)
    ap.add_argument("--exposures", type=int, required=True)
    ap.add_argument("--total-tokens", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mod", type=int, default=23)
    ap.add_argument("--op-lo", type=int, default=1)
    ap.add_argument("--op-hi", type=int, default=4)
    ap.add_argument("--bed-jsonl", default=None)
    ap.add_argument("--fact-share", type=float, default=0.50)
    ap.add_argument("--igsm-share", type=float, default=0.30)
    ap.add_argument("--deduction-share", type=float, default=0.10)
    ap.add_argument("--bed-share", type=float, default=0.10)
    args = ap.parse_args()

    shares = {
        "fact": args.fact_share, "igsm": args.igsm_share,
        "deduction": args.deduction_share, "bed": args.bed_share,
    }
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        print(f"shares must sum to 1.0, got {sum(shares.values())}")
        return 1

    out = Path(args.out)
    man = build(
        out, args.entities, args.exposures, shares, args.total_tokens,
        args.seed, mod=args.mod, op_band=(args.op_lo, args.op_hi),
        bed_jsonl=Path(args.bed_jsonl) if args.bed_jsonl else None,
        progress_every=1 << 28,
    )
    print(json.dumps(
        {k: man[k] for k in ("n_tokens", "n_entities", "exposures",
                             "lane_tokens", "bits_per_param", "mask_audit")},
        indent=2))
    fails = verify(out, expect_tokens=args.total_tokens)
    if fails:
        print("\nVERIFY FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nVERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
