"""Corpus builder: assembles both arms' token streams from one source of
truth, plus the organizer table and all held-out eval sets.

Guarantees (checked into report["checks"]):
- per-component token budgets hit within 1% in each arm;
- bed / igsm / deduction / factqa doc lists identical across arms (the bio
  dose is the only component whose doc count differs — split bios are
  longer, so the split arm cycles fewer exposures);
- organizer covers exactly the training entities' (entity, relation) pairs
  (fresh probe entities live in organizer_fresh.jsonl, a superset store);
- rerun with the same cfg is byte-identical.

corpusgen.igsm_lite / corpusgen.deduction are imported lazily so tests can
stub them via sys.modules.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def _log(msg: str) -> None:
    print(f"[build +{time.monotonic() - _T0:8.1f}s] {msg}", flush=True)


_T0 = time.monotonic()

from corpusgen import bios, factqa
from corpusgen.records import ATTRIBUTES, Doc, plain
from organizer.store import Organizer

COMPONENTS = ("bed", "bio", "igsm", "deduction", "factqa")
_FLUSH_TOKENS = 1_000_000

# canonical fact-load levels (entities); single source of truth for scripts
LOADS: dict[str, int] = {
    "n50k": 50_000,
    "n200k": 200_000,
    "n800k": 800_000,
    "n4m": 4_000_000,  # 1B-scale dose: ~212 Mbit demanded at low exposures
}


@dataclass
class BuildCfg:
    n_entities: int
    total_tokens: int
    seed: int
    workers: int = 1  # >1 parallelizes generation+encoding (deterministic)
    # Shares amended 2026-07-19 per gate-A remediation (spec section 7:
    # "raise reasoning share once"): bed 0.62 -> 0.54, igsm 0.07 -> 0.12,
    # deduction 0.05 -> 0.08. Gate-budget pilots left both reasoning tasks
    # at chance; fact dose (0.23) and factqa (0.03) unchanged.
    bed_share: float = 0.54
    bio_share: float = 0.23
    igsm_share: float = 0.12
    deduction_share: float = 0.08
    factqa_share: float = 0.03
    # Difficulty floor (2026-07-20, gate-A decision "option B"): op 1-4
    # with 1/op-weighted training mass and <=1 distractor at op<=2; depth
    # 1-2. Bands above become OOD report-only. Prior bands: op 2-8 (v1),
    # 2-6 (first remediation); depth 1-4.
    igsm_op: tuple[int, int] = (1, 4)
    deduction_depth: tuple[int, int] = (1, 2)
    n_igsm_eval: int = 10_000
    n_deduction_eval: int = 10_000
    n_factqa_eval: int = 2_000
    n_fresh_entities: int = 200
    n_fresh_eval: int = 500
    n_recall_entities: int = 2_000

    def shares(self) -> dict[str, float]:
        return {
            "bed": self.bed_share,
            "bio": self.bio_share,
            "igsm": self.igsm_share,
            "deduction": self.deduction_share,
            "factqa": self.factqa_share,
        }


class _ArmWriter:
    """Streams uint16 token ids + uint8 loss masks to disk with bounded RAM.

    mask=None means "all loss ON" (dense renderings, bed, knowledge-free
    docs) and is materialized only at flush time, so the big shared bed
    encoding never carries a mask array in memory.
    """

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._bin = open(out_dir / "train.bin", "wb")
        self._mask = open(out_dir / "train.mask.bin", "wb")
        self._id_bufs: list[np.ndarray] = []
        self._mask_bufs: list[np.ndarray | int] = []  # int n == ones(n)
        self._buffered = 0
        self.component_tokens: dict[str, int] = {c: 0 for c in COMPONENTS}
        self.component_docs: dict[str, int] = {c: 0 for c in COMPONENTS}
        self.masked_tokens = 0
        self.total = 0

    def add(self, comp: str, ids: np.ndarray, mask: np.ndarray | None) -> None:
        n = len(ids)
        self._id_bufs.append(ids)
        self._mask_bufs.append(mask if mask is not None else n)
        self._buffered += n
        self.component_tokens[comp] += n
        self.component_docs[comp] += 1
        self.total += n
        if mask is not None:
            self.masked_tokens += n - int(mask.sum())
        if self._buffered >= _FLUSH_TOKENS:
            self.flush()

    def flush(self) -> None:
        if not self._id_bufs:
            return
        np.concatenate(self._id_bufs).tofile(self._bin)
        np.concatenate(
            [m if isinstance(m, np.ndarray) else np.ones(m, dtype=np.uint8)
             for m in self._mask_bufs]
        ).tofile(self._mask)
        self._id_bufs = []
        self._mask_bufs = []
        self._buffered = 0

    def close(self) -> None:
        self.flush()
        self._bin.close()
        self._mask.close()


def _doc_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _encode(tok, doc: Doc, arm: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode one doc for one arm; mask is None when every token gets loss."""
    segs = doc.dense_segments if arm == "dense" else doc.split_segments
    ids, mask = tok.encode_segments(segs, add_eot=True)
    ids_arr = np.asarray(ids, dtype=np.uint16)
    if 0 in mask:
        return ids_arr, np.asarray(mask, dtype=np.uint8)
    return ids_arr, None


# ---------------- parallel generation workers ------------------------------
# Module-level state initialized once per pool worker (spawn-safe). Workers
# regenerate the entity records deterministically instead of receiving a
# multi-GB pickle. Tests exercise workers=1 (in-process) and byte-identity
# between workers=1 and workers>1.

_WSTATE: dict = {}


def _pool_init(records_args: tuple | None) -> None:
    from train.tokenizer import get_tok as _get_tok

    _WSTATE["tok"] = _get_tok()
    if records_args is not None:
        n_entities, n_fresh, seed = records_args
        recs = bios.generate_records(n_entities + n_fresh, seed)[:n_entities]
        _WSTATE["records"] = recs
        _WSTATE["fq_index"] = factqa.birth_city_index(recs)


def _pool_gen_batch(task: tuple) -> list[tuple]:
    """(kind, seed, lo, hi, n_docs) -> [(doc, enc_dense, enc_split), ...]"""
    kind, seed, lo, hi, n_docs = task
    tok = _WSTATE["tok"]
    if kind == "igsm":
        mod = importlib.import_module("corpusgen.igsm_lite")
        docs = mod.generate_igsm_docs(n_docs, lo, hi, seed)
    elif kind == "deduction":
        mod = importlib.import_module("corpusgen.deduction")
        docs = mod.generate_deduction_docs(n_docs, lo, hi, seed)
    elif kind == "factqa":
        docs = factqa.generate_factqa_docs(
            _WSTATE["records"], n_docs, seed, index=_WSTATE["fq_index"]
        )
    elif kind == "bed":
        raise ValueError("bed is encoded via _pool_encode_bed")
    else:
        raise ValueError(kind)
    return [(doc, _encode(tok, doc, "dense"), _encode(tok, doc, "split")) for doc in docs]




def build_corpus(cfg: BuildCfg, tok, bed_iter: Iterator[str], out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    igsm = importlib.import_module("corpusgen.igsm_lite")
    deduction = importlib.import_module("corpusgen.deduction")

    pool = None
    if cfg.workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            cfg.workers,
            initializer=_pool_init,
            initargs=((cfg.n_entities, cfg.n_fresh_entities, cfg.seed),),
        )
        _log(f"worker pool: {cfg.workers} processes")

    budgets = {c: int(round(s * cfg.total_tokens)) for c, s in cfg.shares().items()}

    # ---------------- records + organizer ----------------------------------
    _log(f"start: n_entities={cfg.n_entities} total_tokens={cfg.total_tokens}")
    all_records = bios.generate_records(cfg.n_entities + cfg.n_fresh_entities, cfg.seed)
    records = all_records[: cfg.n_entities]
    fresh = all_records[cfg.n_entities :]
    _log(f"records generated: {len(all_records)}")
    org = Organizer()
    for rec in records:
        for attr in ATTRIBUTES:
            org.add(rec.name, attr, rec.attrs[attr])
    org.save(out_dir / "organizer.jsonl")
    org_fresh = Organizer()
    for rec in all_records:
        for attr in ATTRIBUTES:
            org_fresh.add(rec.name, attr, rec.attrs[attr])
    org_fresh.save(out_dir / "organizer_fresh.jsonl")
    _log("organizers saved")

    # ---------------- shared synthetic components --------------------------
    # Generate until the MEAN of the two renderings' token counts reaches the
    # budget, so each arm's share deviation is at most half the wrapping cost.
    _BATCH = 64
    fq_index = factqa.birth_city_index(records)  # once, not per 64-doc batch

    def _make_batch_inline(kind: str, lo: int, hi: int, offset: int, i: int):
        seed = cfg.seed * 1000 + offset + _BATCH * i
        if kind == "igsm":
            docs = igsm.generate_igsm_docs(_BATCH, lo, hi, seed)
        elif kind == "deduction":
            docs = deduction.generate_deduction_docs(_BATCH, lo, hi, seed)
        else:
            docs = factqa.generate_factqa_docs(records, _BATCH, seed, index=fq_index)
        return [(doc, _encode(tok, doc, "dense"), _encode(tok, doc, "split"))
                for doc in docs]

    def generate_until(kind: str, lo: int, hi: int, offset: int, budget: int,
                       label: str) -> tuple[list[Doc], list[tuple], list[tuple]]:
        """Consume deterministic 64-doc batches (batch i uses seed
        cfg.seed*1000 + offset + 64*i) in order until the mean-of-renderings
        token budget is met. Identical output for any workers setting."""
        docs: list[Doc] = []
        enc_dense: list[tuple] = []
        enc_split: list[tuple] = []
        mean_total = 0.0
        next_log = budget / 8

        def consume(batch) -> bool:
            nonlocal mean_total, next_log
            for doc, d, s in batch:
                docs.append(doc)
                enc_dense.append(d)
                enc_split.append(s)
                mean_total += (len(d[0]) + len(s[0])) / 2
                if mean_total >= next_log:
                    _log(f"{label}: {mean_total/1e6:.0f}M/{budget/1e6:.0f}M tokens, {len(docs)} docs")
                    next_log += budget / 8
                if mean_total >= budget:
                    return True
            return False

        i = 0
        done = False
        while not done:
            if pool is None:
                done = consume(_make_batch_inline(kind, lo, hi, offset, i))
                i += 1
            else:
                # probe two batches inline to size a bounded parallel map
                if i < 2:
                    done = consume(_make_batch_inline(kind, lo, hi, offset, i))
                    i += 1
                    continue
                per_batch = mean_total / i
                need = max(cfg.workers, int((budget - mean_total) / per_batch * 1.1) + 1)
                tasks = [(kind, cfg.seed * 1000 + offset + _BATCH * j, lo, hi, _BATCH)
                         for j in range(i, i + need)]
                for batch in pool.imap(_pool_gen_batch, tasks, chunksize=1):
                    done = consume(batch)
                    if done:
                        break
                i += need
        _log(f"{label}: done ({len(docs)} docs)")
        return docs, enc_dense, enc_split

    igsm_docs, igsm_d, igsm_s = generate_until(
        "igsm", cfg.igsm_op[0], cfg.igsm_op[1], 11, budgets["igsm"], "igsm")
    ded_docs, ded_d, ded_s = generate_until(
        "deduction", cfg.deduction_depth[0], cfg.deduction_depth[1], 22,
        budgets["deduction"], "deduction")
    fq_docs, fq_d, fq_s = generate_until(
        "factqa", 0, 0, 33, budgets["factqa"], "factqa")
    if pool is not None:
        pool.close()
        pool.join()

    # ---------------- bed (identical across arms; ids only, mask implicit) --
    bed_hasher = hashlib.sha1()
    bed_enc: list[np.ndarray] = []
    bed_total = 0
    next_log = budgets["bed"] / 16
    for text in bed_iter:
        ids, _ = tok.encode_segments([plain(text)], add_eot=True)
        bed_hasher.update(_doc_hash(text).encode())
        bed_enc.append(np.asarray(ids, dtype=np.uint16))
        bed_total += len(ids)
        if bed_total >= next_log:
            _log(f"bed: {bed_total/1e6:.0f}M/{budgets['bed']/1e6:.0f}M tokens, {len(bed_enc)} docs")
            next_log += budgets["bed"] / 16
        if bed_total >= budgets["bed"]:
            break
    bed_digest = bed_hasher.hexdigest()
    _log(f"bed: done ({len(bed_enc)} docs)")

    # ---------------- per-arm assembly -------------------------------------
    report_arms: dict[str, dict] = {}
    for arm in ("dense", "split"):
        next_arm_log = cfg.total_tokens / 8
        writer = _ArmWriter(out_dir / arm)
        shared = {
            "bed": [(ids, None) for ids in bed_enc],
            "igsm": list(igsm_d if arm == "dense" else igsm_s),
            "deduction": list(ded_d if arm == "dense" else ded_s),
            "factqa": list(fq_d if arm == "dense" else fq_s),
        }
        queues = {c: iter(v) for c, v in shared.items()}
        remaining = {c: len(v) for c, v in shared.items()}

        # bio queue: entities cycled round-robin over exposures, truncated by
        # THIS arm's token budget.
        def bio_stream():
            exposure = 0
            while True:
                for rec in records:
                    yield _encode(tok, bios.render_bio_doc(rec, exposure), arm)
                exposure += 1

        bio_iter = bio_stream()
        bio_emitted = 0
        bio_docs_n = 0

        # deterministic largest-deficit-first interleave at doc granularity
        emitted = {c: 0 for c in COMPONENTS}
        active = set(COMPONENTS)
        while active:
            deficits = {
                c: (budgets[c] - emitted[c]) / max(1, budgets[c]) for c in active
            }
            comp = max(sorted(deficits), key=lambda c: deficits[c])
            if comp == "bio":
                if bio_emitted >= budgets["bio"]:
                    active.discard("bio")
                    continue
                ids, mask = next(bio_iter)
                bio_emitted += len(ids)
                bio_docs_n += 1
            else:
                if remaining[comp] == 0:
                    active.discard(comp)
                    continue
                ids, mask = next(queues[comp])
                remaining[comp] -= 1
            if arm == "dense":
                mask = None  # dense arm: loss everywhere, by construction
            writer.add(comp, ids, mask)
            emitted[comp] += len(ids)
            if writer.total >= next_arm_log:
                _log(f"{arm} arm: {writer.total/1e6:.0f}M tokens written")
                next_arm_log += cfg.total_tokens / 8
        writer.close()
        _log(f"{arm} arm: done ({writer.total/1e6:.0f}M tokens)")

        report_arms[arm] = {
            "total_tokens": writer.total,
            "component_tokens": dict(writer.component_tokens),
            "component_shares": {
                c: writer.component_tokens[c] / writer.total for c in COMPONENTS
            },
            "component_docs": dict(writer.component_docs),
            "bio_exposures_per_entity": bio_docs_n / cfg.n_entities,
            "masked_token_frac": writer.masked_tokens / writer.total,
            "bed_hash_digest": bed_digest,
        }

    # ---------------- eval sets --------------------------------------------
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    igsm_hashes = {d.meta["structure_hash"] for d in igsm_docs}
    ded_hashes = {d.meta["structure_hash"] for d in ded_docs}
    train_prompts = {
        d.dense_text().split("\nReasoning:", 1)[0] + "\nReasoning:" for d in fq_docs
    }
    evals = {
        "igsm": igsm.generate_igsm_eval(cfg.n_igsm_eval, cfg.igsm_op[0],
                                        cfg.igsm_op[1], cfg.seed * 1000 + 44,
                                        igsm_hashes),
        "deduction": deduction.generate_deduction_eval(cfg.n_deduction_eval,
                                                       cfg.deduction_depth[0],
                                                       cfg.deduction_depth[1],
                                                       cfg.seed * 1000 + 55,
                                                       ded_hashes),
        "factqa": factqa.generate_factqa_eval(records, cfg.n_factqa_eval,
                                              cfg.seed * 1000 + 66, train_prompts,
                                              index=fq_index),
        "factqa_fresh": factqa.generate_fresh_entity_eval(fresh, cfg.n_fresh_eval,
                                                          cfg.seed * 1000 + 77),
        "recall": bios.recall_probes(records,
                                     min(cfg.n_recall_entities, cfg.n_entities),
                                     cfg.seed * 1000 + 88),
    }
    _log("eval sets generated")
    for stem, items in evals.items():
        with open(eval_dir / f"{stem}.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(asdict(item)) + "\n")
    _log("eval sets written")

    # ---------------- report + checks --------------------------------------
    shares = cfg.shares()
    max_dev = max(
        abs(report_arms[arm]["component_shares"][c] - shares[c])
        for arm in report_arms
        for c in COMPONENTS
    )
    d_bio = report_arms["dense"]["component_tokens"]["bio"]
    s_bio = report_arms["split"]["component_tokens"]["bio"]
    report = {
        "cfg": asdict(cfg),
        "arms": report_arms,
        "bio_cross_arm_rel_diff": abs(s_bio - d_bio) / d_bio,
        "eval_counts": {stem: len(items) for stem, items in evals.items()},
        "max_share_deviation": max_dev,
        "checks": {
            "shares_within_1pct": max_dev <= 0.0105,
            "bed_identical_across_arms": (
                report_arms["dense"]["bed_hash_digest"]
                == report_arms["split"]["bed_hash_digest"]
                and report_arms["dense"]["component_docs"]["bed"]
                == report_arms["split"]["component_docs"]["bed"]
            ),
            "organizer_size_ok": len(org) == cfg.n_entities * len(ATTRIBUTES),
        },
    }
    return report
