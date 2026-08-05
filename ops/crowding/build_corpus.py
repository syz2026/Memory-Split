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

The fact lane is finite -- `n_entities * exposures` documents and no more --
while its share is a fraction of the total. When the share asks for more than
the lane can emit, the remainder goes to the bed so the token count stays
exact and every arm keeps the same step count. That reallocation is correct
and deliberate, but until 2026-08-02 it was also silent, and Stage A walked
into it: 20,000 entities at 20 exposures can emit ~30M tokens against a 50%
share of 400M, so the fact lane realised 3.75% and the bed absorbed 46.3% of
the corpus. Nothing failed and the manifest recorded it where nobody looked.
`plan_lanes` now refuses that build up front, and `verify` fails closed on a
realised share that drifts from the requested one.
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

# A realised lane share may drift from the requested one by this much, absolute,
# before the corpus counts as a different experiment. The interleaver lands
# within a few thousand tokens of budget on every lane it can fill, so anything
# above a fraction of a point means a lane ran dry.
SHARE_TOLERANCE = 0.01


class LaneUnderfilled(RuntimeError):
    """The fact lane cannot fill its share, so the bed would absorb the rest."""


def fact_lane_capacity(records, exposures: int, tok, sample: int = 256
                       ) -> tuple[int, float]:
    """Total tokens the fact lane can emit, measured rather than assumed.

    Renders a spread sample of the lane's documents and scales by the document
    count. Measuring beats the 74.91 tokens/document constant in the
    preregistration because the figure moves with the record generator, and a
    stale constant here would reintroduce exactly the failure this guards.
    """
    n_docs = len(records) * exposures
    if n_docs == 0:
        return 0, 0.0
    step = max(1, n_docs // sample)
    lens = [len(factlane.render_one(records, k, tok)[0])
            for k in range(0, n_docs, step)][:sample]
    mean = float(np.mean(lens))
    return int(mean * n_docs), mean


def plan_lanes(records, exposures: int, tok, shares: dict[str, float],
               total_tokens: int) -> dict:
    """What the fact lane can deliver against what its share demands.

    Called before a byte is written. A 16.4B-token build takes about two hours
    and 65.6 GB, so discovering an underfilled lane afterwards costs a rebuild
    and, if nobody reads the manifest, an entire cohort.
    """
    capacity, mean_doc = fact_lane_capacity(records, exposures, tok)
    demand = int(total_tokens * shares["fact"])
    deficit = max(0, demand - capacity)
    needed_docs = demand / mean_doc if mean_doc else float("inf")
    # Refuse exactly when `verify` would fail, and no sooner. Both operating
    # points in the repository are sized so the fact lane fills its share to
    # within 0.01%, and the capacity estimate is a sample mean, so a stricter
    # rule would refuse the intended design on measurement noise. Stage A's
    # deficit was 46% of the corpus, which is nowhere near this boundary.
    slack = SHARE_TOLERANCE * total_tokens
    return {
        "fact_documents": len(records) * exposures,
        "mean_tokens_per_fact_doc": mean_doc,
        "fact_lane_capacity_tokens": capacity,
        "fact_lane_demand_tokens": demand,
        "deficit_tokens": deficit,
        "deficit_share_of_corpus": deficit / max(1, total_tokens),
        "tolerated_deficit_tokens": int(slack),
        "feasible": deficit <= slack,
        "realised_fact_share": min(capacity, demand) / max(1, total_tokens),
        "documents_needed": int(needed_docs) if mean_doc else None,
        "exposures_needed_at_this_entity_count": (
            int(-(-needed_docs // max(1, len(records)))) if mean_doc else None
        ),
    }


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


def randpos_seed(entity: int, exposure: int) -> random.Random:
    """A per-document RNG for the control sidecar.

    This used to be one RNG threaded through the whole fact lane, which made
    a document's control mask depend on how far the shared stream had
    advanced -- so the corpus depended on generation order and could not be
    produced in parallel without changing it. Seeding per document makes
    randpos a pure function of its coordinates, exactly like the content, and
    the corpus byte-identical regardless of worker count.
    """
    return random.Random(f"randpos:{entity}:{exposure}")


def fact_stream(records, exposures, tok, workers: int = 1,
                chunk_docs: int = 20_000, nll_table: np.ndarray | None = None):
    """Fact documents in canonical order, optionally generated in parallel.

    The fact lane is ~98% of the build cost: 11.5B tokens at ~1,640 docs/s on
    one core is 26 hours, which does not fit a wall-clock limit. Documents are
    independent given their coordinates, so chunks are farmed out and consumed
    in order; `imap` preserves ordering, so the output is unchanged.

    `nll_table` is the frozen per-token-id difficulty table. It is indexed by
    `ids` to give RANDPOS a per-position difficulty signal, which is the fourth
    matching axis and the only one that was never wired up.
    """
    n_docs = len(records) * exposures
    if workers <= 1:
        for ids, fmask, entity, exposure in factlane.emit_indexed(
            records, exposures, tok
        ):
            yield ids, fmask, randpos.build(
                ids, fmask, randpos_seed(entity, exposure),
                token_nll=_doc_nll(ids, nll_table),
            )
        return

    import multiprocessing as mp

    chunks = [(lo, min(lo + chunk_docs, n_docs))
              for lo in range(0, n_docs, chunk_docs)]
    ctx = mp.get_context("fork")  # children inherit `records` copy-on-write
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(records, exposures, nll_table)) as pool:
        for batch in pool.imap(_gen_chunk, chunks):
            yield from batch


_W: dict = {}


def _doc_nll(ids: np.ndarray, nll_table: np.ndarray | None):
    """Per-position difficulty for one document, or None if no table is frozen."""
    return None if nll_table is None else nll_table[np.asarray(ids, dtype=np.int64)]


def _init_worker(records, exposures, nll_table=None):
    _W["records"] = records
    _W["exposures"] = exposures
    _W["nll_table"] = nll_table
    _W["tok"] = get_tok()


def _gen_chunk(bounds):
    lo, hi = bounds
    records, tok = _W["records"], _W["tok"]
    nll_table = _W.get("nll_table")
    out = []
    for k in range(lo, hi):
        ids, fmask, entity, exposure = factlane.render_one(records, k, tok)
        out.append((
            ids, fmask,
            randpos.build(ids, fmask, randpos_seed(entity, exposure),
                          token_nll=_doc_nll(ids, nll_table)),
        ))
    return out


class NLLMatchAccumulator:
    """Corpus-level difficulty match between FACTMASK and RANDPOS.

    Preregistration §2 makes this the one axis whose failure is itself the
    result: "If mean masked-span NLL cannot be matched within 20% relative, the
    control is empirically invalid and that is reported as the result, not
    gated away." So this measures and records; it never blocks a build.

    Sums are exact over every fact token written, not sampled, because the
    quantity is a mean over ~61M masked positions and a sample would add noise
    to the one number the primary contrast rests on. The richer per-document
    diagnostics -- cue-window overlap, length histograms -- come from
    `randpos.match_report` on a bounded sample, since those need Python loops.
    """

    def __init__(self, nll_table: np.ndarray | None, sample_docs: int = 512):
        self.table = nll_table
        self.sample_docs = sample_docs
        self.fact_sum = self.rand_sum = 0.0
        self.fact_n = self.rand_n = 0
        self.samples: list[dict] = []

    def observe(self, ids, fmask, rmask) -> None:
        if len(self.samples) < self.sample_docs:
            self.samples.append(randpos.match_report(
                fmask, rmask, token_nll=_doc_nll(ids, self.table)))
        if self.table is None or len(ids) == 0:
            return
        nll = _doc_nll(ids, self.table)
        f, r = fmask == 0, rmask == 0
        self.fact_sum += float(nll[f].sum())
        self.fact_n += int(f.sum())
        self.rand_sum += float(nll[r].sum())
        self.rand_n += int(r.sum())

    def report(self) -> dict:
        if self.table is None:
            return {
                "status": "NOT-MEASURED",
                "note": "no frozen NLL table was supplied, so RANDPOS matched "
                        "count, span length and relative position but not "
                        "difficulty. Preregistration §2 requires the match to "
                        "be measured; an unmeasured control cannot be reported "
                        "as valid or invalid.",
                **_sample_diagnostics(self.samples),
            }
        f = self.fact_sum / max(1, self.fact_n)
        r = self.rand_sum / max(1, self.rand_n)
        gap = abs(r - f) / max(1e-9, f)
        return {
            "status": "MEASURED",
            "mean_nll_factmask": f,
            "mean_nll_randpos": r,
            "relative_gap": gap,
            "tolerance": randpos.NLL_TOLERANCE,
            "within_tolerance": gap <= randpos.NLL_TOLERANCE,
            "n_masked_factmask": self.fact_n,
            "n_masked_randpos": self.rand_n,
            **_sample_diagnostics(self.samples),
        }


def _sample_diagnostics(samples: list[dict]) -> dict:
    """Cue-window overlap and structural match, averaged over sampled documents.

    Cue overlap matters because the feasible set in a ~75-token document is
    small, so control spans drift onto the tokens that predict a value. Masking
    those removes fact-relevant supervision and biases the control toward the
    treatment.
    """
    if not samples:
        return {"n_sampled_documents": 0}
    return {
        "n_sampled_documents": len(samples),
        "mean_cue_window_overlap_frac": float(np.mean(
            [s["cue_window_overlap_frac"] for s in samples])),
        "frac_documents_length_matched": float(np.mean(
            [1.0 if s["length_histogram_matched"] else 0.0 for s in samples])),
        "frac_documents_count_matched": float(np.mean(
            [1.0 if s["count_matched"] else 0.0 for s in samples])),
        "median_distance_to_nearest_value": float(np.median(
            [s["median_distance_to_nearest_value"] for s in samples
             if s["median_distance_to_nearest_value"] is not None] or [0.0])),
    }


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
    workers: int = 1,
    nll_table: np.ndarray | None = None,
    allow_short_fact_lane: bool = False,
    sup_only: bool = False,
) -> dict:
    tok = get_tok()
    rng = random.Random(seed)
    records = bios.generate_records(n_entities, seed)

    plan = plan_lanes(records, exposures, tok, shares, total_tokens)
    if not plan["feasible"] and not allow_short_fact_lane:
        raise LaneUnderfilled(
            f"fact lane can emit ~{plan['fact_lane_capacity_tokens']:,} tokens "
            f"({plan['fact_documents']:,} documents x "
            f"{plan['mean_tokens_per_fact_doc']:.1f}) but its "
            f"{shares['fact']:.0%} share demands "
            f"{plan['fact_lane_demand_tokens']:,}. The bed would absorb the "
            f"{plan['deficit_tokens']:,}-token deficit "
            f"({plan['deficit_share_of_corpus']:.1%} of the corpus, against a "
            f"{SHARE_TOLERANCE:.0%} tolerance) and the fact share would "
            f"realise {plan['realised_fact_share']:.2%} instead of "
            f"{shares['fact']:.0%} -- a different experiment. Raise exposures to "
            f"~{plan['exposures_needed_at_this_entity_count']} at this entity "
            "count, raise the entity count, or pass allow_short_fact_lane=True "
            "if this is a rehearsal."
        )

    budgets = {k: int(total_tokens * v) for k, v in shares.items()}
    streams = {
        "fact": fact_stream(records, exposures, tok, workers=workers,
                            nll_table=nll_table),
        "igsm": igsm_stream(tok, op_band[0], op_band[1], seed * 7 + 1, mod),
        "deduction": deduction_stream(tok, depth_band[0], depth_band[1], seed * 7 + 2),
        "bed": bed_stream(tok, bed_jsonl, seed * 7 + 3),
    }
    emitted = dict.fromkeys(LANES, 0)
    exhausted: set[str] = set()
    nll = NLLMatchAccumulator(nll_table)

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
        room = writer.limit - writer.written
        if lane == "fact" and len(ids) > room:
            # Truncating a fact document cuts its two sidecars at different
            # places, so the corpus ends with unequal mask mass: the arms zero
            # a different number of targets and therefore train a different
            # effective objective, which is the confound the fixed-denominator
            # loss exists to remove. The bed is all-ones in both sidecars, so
            # truncating it costs nothing. Spend the tail there instead.
            ids, fmask, rmask = next(streams["bed"])
            lane = "bed"
        emitted[lane] += min(len(ids), max(0, room))
        if lane == "fact":
            nll.observe(ids, fmask, rmask)
        going = writer.write(ids, fmask, rmask)
        if progress_every and writer.written % progress_every < len(ids):
            print(f"  {writer.written/1e6:.1f}M tokens {dict(emitted)}", flush=True)
    writer.close()
    if writer.written != total_tokens:
        raise RuntimeError(
            f"corpus is short: wrote {writer.written:,} of {total_tokens:,} tokens"
        )

    return _manifest(out, writer.written, emitted, budgets, n_entities,
                     exposures, seed, mod, op_band, depth_band, bed_jsonl,
                     shares=shares, plan=plan, randpos_validity=nll.report(),
                     sup_only=sup_only)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(out, written, emitted, budgets, n_entities, exposures, seed,
              mod, op_band, depth_band, bed_jsonl, shares=None, plan=None,
              randpos_validity=None, sup_only=False) -> dict:
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
        # `lane_budgets` is mutated by the exhausted-lane reallocation, so it
        # records where tokens ended up, not what was asked for. The requested
        # shares are what a reader needs to see the substitution.
        "lane_budgets": dict(budgets),
        "requested_shares": dict(shares) if shares else None,
        "realised_shares": {k: emitted[k] / max(1, written) for k in LANES},
        "lane_plan": plan,
        "randpos_validity": randpos_validity,
        # Stage B's NOFACT arm: the fact lane is switched off outright rather
        # than starved and left to the bed. Expressing it as a zero share means
        # the manifest states the intent, and `verify` can then require empty
        # sidecars instead of flagging them as an inert-mask bug.
        "lane_profile": ("NOFACT" if shares and shares.get("fact", 0) == 0
                         else "standard"),
        # A SUP-only probe corpus never trains a masked arm, so its control
        # sidecar is unused and its difficulty match is not applicable. Recorded
        # so the corpus cannot later be handed to a FACTMASK or RANDPOS arm on
        # the assumption that it was validated for one.
        "valid_for": "SUP-only" if sup_only else "all arms",
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


def verify(out: Path, expect_tokens: int | None = None,
           allow_synthetic_bed: bool = False,
           allow_share_drift: bool = False,
           require_randpos_nll: bool = True) -> list[str]:
    """Fail-closed checks. A corpus that is silently short, or whose arms are
    not nested, would waste an entire cohort.

    Every gate defaults to strict, so a caller that forgets gets the protection
    rather than the hazard. Rehearsals opt out explicitly, which also makes a
    rehearsal legible as one at its call site. They are separate flags rather
    than one `--strict` because they fail for different reasons and a build may
    legitimately trip one and not the others.
    """
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

    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    nofact = man.get("lane_profile") == "NOFACT"

    fact = np.memmap(f_path, dtype=np.uint8, mode="r")
    rand = np.memmap(r_path, dtype=np.uint8, mode="r")
    nf, nr = int((fact == 0).sum()), int((rand == 0).sum())
    if nofact:
        # A NOFACT corpus has no fact values, so empty sidecars are correct and
        # a non-empty one means the lane leaked.
        if nf or nr:
            fails.append(f"NOFACT corpus has masked targets: factmask {nf}, "
                         f"randpos {nr}; the fact lane was not fully off")
    else:
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

    if not man:
        fails.append("no manifest.json; provenance cannot be checked")
        return fails

    # The failure that produced Stage A: a lane runs dry, the bed absorbs its
    # budget, the token count stays exact and nothing complains.
    requested = man.get("requested_shares")
    if not allow_share_drift:
        if not requested:
            fails.append("manifest records no requested_shares; a realised "
                         "share cannot be checked against what was asked for")
        else:
            for lane in LANES:
                want = requested.get(lane, 0.0)
                got = man["lane_tokens"].get(lane, 0) / max(1, man["n_tokens"])
                if abs(got - want) > SHARE_TOLERANCE:
                    fails.append(
                        f"lane '{lane}' realised {got:.2%} against a requested "
                        f"{want:.0%} (tolerance {SHARE_TOLERANCE:.0%}): the "
                        "corpus is not the one the design specifies"
                    )

    if not allow_synthetic_bed and "SYNTHETIC" in str(man.get("bed", "")):
        fails.append(
            "bed is SYNTHETIC-REHEARSAL-ONLY: the runbook says treat this "
            "string in a manifest as a build failure. Pin the FineWeb-Edu "
            "JSONL and record its hash."
        )

    rv = man.get("randpos_validity") or {}
    if require_randpos_nll and not nofact and rv.get("status") != "MEASURED":
        if man.get("valid_for") == "SUP-only":
            fails.append(
                "this corpus was built --sup-only, so its RANDPOS sidecar was "
                "never difficulty-matched and no masked arm may train on it. "
                "Rebuild with a frozen NLL table before using it for an arm "
                "contrast."
            )
        else:
            fails.append(
                "RANDPOS difficulty match was never measured: no frozen NLL "
                "table was supplied to the build. Preregistration §2 requires "
                "it, and without it the primary control matches mass but not "
                "difficulty. Build one with ops/crowding/nll_table.py."
            )
    return fails


def report_findings(out: Path) -> list[str]:
    """Things the preregistration says to disclose rather than gate on.

    Preregistration §2 is explicit that a control which cannot be
    difficulty-matched is "reported as the result, not gated away", so an
    out-of-tolerance NLL gap must not fail the build. It must also be
    impossible to miss, which is what this and the banner in `main` are for.
    """
    man_path = out / "manifest.json"
    if not man_path.exists():
        return []
    rv = json.loads(man_path.read_text()).get("randpos_validity") or {}
    findings: list[str] = []
    if rv.get("status") == "MEASURED" and not rv.get("within_tolerance"):
        findings.append(
            f"RANDPOS is NOT difficulty-matched: mean masked-span NLL is "
            f"{rv['mean_nll_randpos']:.4f} against FACTMASK's "
            f"{rv['mean_nll_factmask']:.4f}, a relative gap of "
            f"{rv['relative_gap']:.1%} against a {rv['tolerance']:.0%} "
            "tolerance. Preregistration §2: this is the result, not a gate. "
            "Report it; do not silently proceed as though the control held."
        )
    overlap = rv.get("mean_cue_window_overlap_frac")
    if overlap is not None and overlap > 0.10:
        findings.append(
            f"{overlap:.1%} of control-masked tokens land in a value's cue "
            "window, so RANDPOS removes fact-relevant supervision and is "
            "biased toward the treatment."
        )
    return findings


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
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel fact-lane workers; the fact lane is ~98% "
                         "of the build cost and is the only lane worth "
                         "parallelising")
    ap.add_argument("--nll-table",
                    help="frozen per-token-id NLL table (.npy) from a pilot "
                         "run on disjoint data; gives RANDPOS its fourth "
                         "matching axis. Required unless --rehearsal.")
    ap.add_argument("--sup-only", action="store_true",
                    help="this corpus will only ever train SUP arms, so the "
                         "RANDPOS difficulty match is not applicable and no "
                         "table is required. The bed and lane-share gates stay "
                         "strict. Recorded in the manifest so a FACTMASK or "
                         "RANDPOS arm cannot later be trained on it by "
                         "accident.")
    ap.add_argument("--rehearsal", action="store_true",
                    help="offline shakedown: permits a synthetic bed, an "
                         "underfilled fact lane and an unmeasured RANDPOS "
                         "match. Never valid for a run that produces a number.")
    args = ap.parse_args()

    shares = {
        "fact": args.fact_share, "igsm": args.igsm_share,
        "deduction": args.deduction_share, "bed": args.bed_share,
    }
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        print(f"shares must sum to 1.0, got {sum(shares.values())}")
        return 1

    nll_table = None
    if args.nll_table:
        nll_table = np.load(args.nll_table)
        print(f"RANDPOS difficulty table: {args.nll_table} "
              f"({nll_table.shape[0]:,} ids, mean {nll_table.mean():.4f} nats)")
    elif not (args.rehearsal or args.sup_only or args.fact_share == 0):
        print("refusing to build without --nll-table: RANDPOS would match mass "
              "but not difficulty, which preregistration §2 requires. Build a "
              "table with ops/crowding/nll_table.py, or pass --sup-only if no "
              "masked arm will ever train on this corpus, or --rehearsal.")
        return 1

    out = Path(args.out)
    try:
        man = build(
            out, args.entities, args.exposures, shares, args.total_tokens,
            args.seed, mod=args.mod, op_band=(args.op_lo, args.op_hi),
            bed_jsonl=Path(args.bed_jsonl) if args.bed_jsonl else None,
            progress_every=1 << 28, workers=args.workers,
            nll_table=nll_table, allow_short_fact_lane=args.rehearsal,
            sup_only=args.sup_only,
        )
    except LaneUnderfilled as e:
        print(f"\nBUILD REFUSED:\n  {e}")
        return 1

    print(json.dumps(
        {k: man[k] for k in ("n_tokens", "n_entities", "exposures",
                             "lane_tokens", "realised_shares", "bits_per_param",
                             "mask_audit", "randpos_validity")},
        indent=2))

    for finding in report_findings(out):
        print("\n" + "!" * 72)
        print("DISCLOSED FINDING (not a gate -- preregistration §2)")
        print(finding)
        print("!" * 72)

    fails = verify(
        out, expect_tokens=args.total_tokens,
        allow_synthetic_bed=args.rehearsal,
        allow_share_drift=args.rehearsal,
        # A --sup-only corpus is verified as what it claims to be. The masked-arm
        # refusal fires later, for whoever tries to use it as one.
        require_randpos_nll=not (args.rehearsal or args.sup_only),
    )
    if fails:
        print("\nVERIFY FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nVERIFY OK" + ("  (REHEARSAL -- not a scientific corpus)"
                           if args.rehearsal else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
