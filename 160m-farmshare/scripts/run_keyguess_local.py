#!/usr/bin/env python
"""Arm orchestrator for the held-out key-generalization experiment (T4).

Wires T1 (corpusgen.realfact), T2 (evals.keyguess), and T3
(evals.constrain / evals.generate) into one reproducible runner over four
arms (A/B/C/D) against the PopQA real-fact split. Stages are independently
re-runnable; see the brief at
.superpowers/sdd/briefs/T4-orchestrator.md and the protocol at
${HOME}/Documents/2026-07-20-heldout-key-generalization-results.md.

Usage:
  python scripts/run_keyguess_local.py --stage all
  python scripts/run_keyguess_local.py --stage data
  python scripts/run_keyguess_local.py --stage train --arms A,C
  python scripts/run_keyguess_local.py --stage eval  --arms A,B,C,D --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from corpusgen import bios, factqa, realfact
from corpusgen.records import QAItem
from evals.constrain import build_query_tries, extract_spans
from evals.generate import generate_batch_with_stats
from evals.keyguess import score_items
from organizer.store import Organizer, normalize
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok
from train.trainer import Trainer, pick_device

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "keyguess_local"
REALFACTS_PATH = ROOT / "data" / "realfacts" / "popqa_clean.jsonl"

ARMS_DEFAULT = ("A", "B", "C", "D")
EVAL_BATCH_SIZE = 32
EVAL_MAX_NEW = 64
EMITTABILITY_FLOOR = 0.99

# Protocol base seeds (seed bundle 0 == the 2026-07-21 local run exactly).
TRAINER_BASE_SEED = 7
SHUFFLE_BASE_SEED = 123
RENDER_BASE_SEED = 0


# --------------------------------------------------------------- pure helpers


def seed_suffix(seed: int) -> str:
    """Artifact suffix for a replication seed bundle. Seed 0 keeps the
    original unsuffixed paths so the first local run stays valid; seed S>0
    gets corpus_a_s{S}, runs/a_s{S}, results_A_s{S}.json, summary_s{S}.json.
    The PopQA split and eval items are seed-independent (shared): replication
    varies model init, doc order, and substitution/flood draws — never the
    held-out set."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return "" if seed == 0 else f"_s{seed}"


def arm_plan(arms: list[str]) -> list[dict]:
    """Route each requested arm to its training run and constraint mode.

    A/B share runs/a (corpus_a); C/D share runs/c (corpus_c). B/D are the
    copy-constrained arms (query_tries); A/C decode freely.
    """
    run_for = {"A": "a", "B": "a", "C": "c", "D": "c"}
    constrained_for = {"A": False, "B": True, "C": False, "D": True}
    plan: list[dict] = []
    for arm in arms:
        if arm not in run_for:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(run_for)}")
        run = run_for[arm]
        plan.append({
            "arm": arm,
            "run": run,
            "corpus": run,
            "constrained": constrained_for[arm],
        })
    return plan


def build_base_docs(records: list, n_factqa_docs: int, factqa_seed: int) -> list:
    """Shared synthetic base for both corpora, SPLIT rendering only.

    Bio docs in grouped rounds (all entities exposure 0, then exposure 1,
    ... up to 5) mirroring realfact's emission order, plus n_factqa_docs
    factqa docs. Pure: no tok, no I/O, deterministic in its arguments.
    """
    bio_docs = [
        bios.render_bio_doc(rec, exposure)
        for exposure in range(6)
        for rec in records
    ]
    fq_docs = factqa.generate_factqa_docs(records, n_factqa_docs, factqa_seed)
    return bio_docs + fq_docs


def assemble_corpus(
    facts: list,
    records: list,
    tok,
    out_dir: Path,
    *,
    n_exposures: int,
    seed: int,
    substitution_frac: float,
    fresh_flood: int,
    n_factqa_docs: int,
    factqa_seed: int,
    shuffle_seed: int = 123,
) -> dict:
    """Build one arm's token stream and write train.bin / train.mask.bin.

    Combines the shared synthetic base with render_realfact_docs(facts, ...),
    shuffles the combined doc list with random.Random(shuffle_seed), encodes
    each doc's SPLIT segments via tok.encode_segments(add_eot=True), and
    streams uint16 ids + uint8 mask to disk (same file pattern as
    corpusgen.build._ArmWriter). Deterministic in all arguments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_docs = build_base_docs(records, n_factqa_docs, factqa_seed)
    real_docs = realfact.render_realfact_docs(
        facts, n_exposures=n_exposures, seed=seed,
        substitution_frac=substitution_frac, fresh_flood=fresh_flood,
    )
    docs = list(base_docs) + list(real_docs)
    random.Random(shuffle_seed).shuffle(docs)

    id_bufs: list[np.ndarray] = []
    mask_bufs: list[np.ndarray] = []
    total = 0
    masked = 0
    for doc in docs:
        ids, mask = tok.encode_segments(doc.split_segments, add_eot=True)
        id_bufs.append(np.asarray(ids, dtype=np.uint16))
        mask_bufs.append(np.asarray(mask, dtype=np.uint8))
        total += len(ids)
        masked += len(ids) - int(np.asarray(mask).sum())
    all_ids = np.concatenate(id_bufs) if id_bufs else np.zeros(0, dtype=np.uint16)
    all_mask = np.concatenate(mask_bufs) if mask_bufs else np.zeros(0, dtype=np.uint8)
    all_ids.tofile(out_dir / "train.bin")
    all_mask.tofile(out_dir / "train.mask.bin")
    return {
        "n_docs": len(docs),
        "n_base_docs": len(base_docs),
        "n_real_docs": len(real_docs),
        "n_tokens": int(total),
        "masked_tokens": int(masked),
        "masked_token_frac": (masked / total) if total else 0.0,
    }


def write_eval_items(items: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")


def load_eval_items(path: Path) -> list[QAItem]:
    items: list[QAItem] = []
    with open(path) as f:
        for line in f:
            items.append(QAItem(**json.loads(line)))
    return items


def check_emittability(items: list[QAItem]) -> tuple[float, list[dict]]:
    """Gold-key emittability gate for the constrained arms (B/D).

    For every eval item, build the copy-constraint candidate spans from its
    prompt and assert the gold key is reachable: some span s with
    normalize(f"{s}, {prop}") == normalize(f"{subj}, {prop}"). Returns
    (coverage, misses); each miss records qid, subj, prop, question.
    """
    misses: list[dict] = []
    for item in items:
        meta = item.meta
        subj, prop = meta["subj"], meta["prop"]
        gold = normalize(f"{subj}, {prop}")
        spans = extract_spans(item.prompt)
        if not any(normalize(f"{s}, {prop}") == gold for s in spans):
            question = item.prompt
            if question.startswith("Question: "):
                question = question[len("Question: "):].split("\nReasoning:", 1)[0]
            misses.append({
                "qid": item.qid, "subj": subj, "prop": prop,
                "question": question, "prompt": item.prompt,
            })
    n = len(items)
    coverage = (n - len(misses)) / n if n else 1.0
    return coverage, misses


def relation_counts(seen: list, heldout: list) -> dict[str, dict[str, int]]:
    """Per-relation seen/heldout counts, keyed by relation (prop)."""
    counts: dict[str, dict[str, int]] = {}
    for fact in seen:
        counts.setdefault(fact.prop, {"seen": 0, "heldout": 0})["seen"] += 1
    for fact in heldout:
        counts.setdefault(fact.prop, {"seen": 0, "heldout": 0})["heldout"] += 1
    return counts


def assert_split_contract(seen: list, heldout: list, eval_items: list,
                          n_total: int, frac: float = 0.8) -> None:
    """Structural contract for the per-relation floor split + eval items.

    Asserts invariant rules rather than absolute counts, so legitimate
    upstream dataset drift does not silently break the orchestrator: the
    split partitions all n_total facts, every relation respects the floor
    rule seen == floor(frac * relation_size), and eval_items equals
    len(heldout) + min(200, len(seen)).
    """
    assert len(seen) + len(heldout) == n_total, (
        f"split does not partition input: seen({len(seen)}) + heldout"
        f"({len(heldout)}) = {len(seen) + len(heldout)} != {n_total}"
    )
    for prop, c in relation_counts(seen, heldout).items():
        total = c["seen"] + c["heldout"]
        expected_seen = int(frac * total)
        assert c["seen"] == expected_seen, (
            f"relation {prop!r}: seen={c['seen']} != floor({frac}*{total})="
            f"{expected_seen}"
        )
    assert len(eval_items) == len(heldout) + min(200, len(seen)), (
        f"eval_items={len(eval_items)} != heldout({len(heldout)}) + "
        f"min(200, seen({len(seen)}))={min(200, len(seen))}"
    )


def corpus_complete(corpus_dir: Path) -> bool:
    """A corpus dir is complete iff train.bin AND train.mask.bin exist and
    train.bin is exactly 2x train.mask.bin in bytes (uint16 ids vs uint8
    mask, one byte per token)."""
    bin_path = Path(corpus_dir) / "train.bin"
    mask_path = Path(corpus_dir) / "train.mask.bin"
    if not (bin_path.exists() and mask_path.exists()):
        return False
    return os.path.getsize(bin_path) == 2 * os.path.getsize(mask_path)


def require_corpus_complete(corpus_dir: Path, *, stage: str) -> None:
    """Raise a clear rerun instruction if a corpus dir is incomplete."""
    bin_path = Path(corpus_dir) / "train.bin"
    mask_path = Path(corpus_dir) / "train.mask.bin"
    if corpus_complete(corpus_dir):
        return
    if bin_path.exists() and not mask_path.exists():
        detail = f"missing {mask_path}"
    elif bin_path.exists() and mask_path.exists():
        detail = (f"size mismatch: train.bin={os.path.getsize(bin_path)}B "
                  f"!= 2*train.mask.bin={2 * os.path.getsize(mask_path)}B")
    elif mask_path.exists() and not bin_path.exists():
        detail = f"missing {bin_path}"
    else:
        detail = f"missing {bin_path} and {mask_path}"
    raise SystemExit(
        f"[{stage}] incomplete corpus at {corpus_dir}: {detail}; rerun "
        f"`python scripts/run_keyguess_local.py --stage data --force` to "
        f"rebuild"
    )


# -------------------------------------------------------------------- stages


def stage_data(force: bool = False, seed: int = 0) -> None:
    """Idempotent data stage: split, corpora, organizer, eval items, gate.

    Shared artifacts (split/organizer/eval items/emittability/manifest) are
    seed-independent and built once; the corpora are per-seed-bundle (doc
    shuffle + substitution/flood draws vary), suffixed via seed_suffix.
    """
    sfx = seed_suffix(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not REALFACTS_PATH.exists():
        raise SystemExit(
            f"missing {REALFACTS_PATH}; run scripts/fetch_realfacts.py first"
        )

    seen_path = DATA_DIR / "seen.jsonl"
    held_path = DATA_DIR / "heldout.jsonl"
    org_path = DATA_DIR / "organizer_real.jsonl"
    eval_path = DATA_DIR / "eval_items.jsonl"
    emit_path = DATA_DIR / "emittability.json"
    corpus_a = DATA_DIR / f"corpus_a{sfx}"
    corpus_c = DATA_DIR / f"corpus_c{sfx}"

    shared_ok = (seen_path.exists() and held_path.exists() and org_path.exists()
                 and eval_path.exists() and emit_path.exists())
    if (not force and shared_ok and corpus_complete(corpus_a)
            and corpus_complete(corpus_c)):
        print(f"[data] all outputs present for seed {seed}; skipping "
              "(use --force to rebuild)")
        return

    facts = realfact.load_realfacts(REALFACTS_PATH)
    seen, heldout = realfact.split_by_relation(facts, 0.8, seed=0)
    assert len(seen) + len(heldout) == len(facts), (
        f"split does not partition input: {len(seen)} + {len(heldout)} "
        f"= {len(seen) + len(heldout)} != {len(facts)}"
    )
    for prop, c in relation_counts(seen, heldout).items():
        total = c["seen"] + c["heldout"]
        assert c["seen"] == int(0.8 * total), (
            f"relation {prop!r}: seen={c['seen']} != floor(0.8*{total})="
            f"{int(0.8 * total)}"
        )
    print(f"[data] split: {len(seen)} seen / {len(heldout)} heldout "
          f"(total {len(facts)}; per-relation floor split)")

    def write_facts(rows, path):
        with open(path, "w") as f:
            for fact in rows:
                f.write(json.dumps({
                    "subj": fact.subj, "prop": fact.prop, "obj": fact.obj,
                    "question": fact.question,
                    "possible_answers": list(fact.possible_answers),
                }) + "\n")
    write_facts(seen, seen_path)
    write_facts(heldout, held_path)

    tok = get_tok()
    records = bios.generate_records(n_entities=300, seed=7)

    render_seed = RENDER_BASE_SEED + seed
    shuffle_seed = SHUFFLE_BASE_SEED + seed
    rep_a = assemble_corpus(
        seen, records, tok, corpus_a,
        n_exposures=6, seed=render_seed, substitution_frac=0.0, fresh_flood=0,
        n_factqa_docs=900, factqa_seed=7, shuffle_seed=shuffle_seed,
    )
    print(f"[data] {corpus_a.name}: {rep_a['n_docs']} docs, "
          f"{rep_a['n_tokens']} tokens, masked {rep_a['masked_token_frac']:.3f}")
    rep_c = assemble_corpus(
        seen, records, tok, corpus_c,
        n_exposures=6, seed=render_seed, substitution_frac=0.5, fresh_flood=2400,
        n_factqa_docs=900, factqa_seed=7, shuffle_seed=shuffle_seed,
    )
    print(f"[data] {corpus_c.name}: {rep_c['n_docs']} docs, "
          f"{rep_c['n_tokens']} tokens, masked {rep_c['masked_token_frac']:.3f}")

    realfact.build_real_organizer(seen + heldout).save(org_path)
    print(f"[data] organizer: {len(seen) + len(heldout)} real facts")

    eval_items = (realfact.realfact_eval_items(heldout, "heldout")
                  + realfact.realfact_eval_items(seen[:200], "seen"))
    assert_split_contract(seen, heldout, eval_items, n_total=len(facts))
    write_eval_items(eval_items, eval_path)
    print(f"[data] eval_items: {len(eval_items)} "
          f"({len(heldout)} heldout + {min(200, len(seen))} seen)")

    manifest_path = DATA_DIR / "data_manifest.json"
    manifest = {
        "seen": len(seen),
        "heldout": len(heldout),
        "eval_items": len(eval_items),
        "per_relation": relation_counts(seen, heldout),
        "protocol_note": (
            "lost-run snapshot was 2399/601; current snapshot 2394/606 "
            "— per-relation floor split, dataset drift documented"
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[data] manifest: {manifest_path} "
          f"(seen={manifest['seen']}, heldout={manifest['heldout']}, "
          f"eval_items={manifest['eval_items']})")

    coverage, misses = check_emittability(eval_items)
    emit = {"coverage": coverage,
            "n_misses": len(misses),
            "misses": misses[:10]}
    with open(emit_path, "w") as f:
        json.dump(emit, f, indent=2)
    print(f"[data] emittability: {coverage:.4f} ({len(misses)} misses)")
    for m in misses[:10]:
        print(f"  miss: subj={m['subj']!r} question={m['question']!r}")
    if coverage < EMITTABILITY_FLOOR:
        raise SystemExit(
            f"emittability coverage {coverage:.4f} < {EMITTABILITY_FLOOR}; "
            "constrained arms B/D cannot reach the gold key for every item"
        )


def _trainer_cfg(run: str, corpus_dir: Path, out_dir: Path, steps: int,
                 device: str, seed: int = 0) -> dict:
    return {
        "run_id": f"keyguess_{run}{seed_suffix(seed)}",
        "arm": "split",
        "model": {"n_layer": 4, "n_head": 4, "d_model": 256, "ctx": 192,
                  "vocab_size": 50304},
        "train_bin": str(corpus_dir / "train.bin"),
        "train_mask": str(corpus_dir / "train.mask.bin"),
        "micro_batch_size": 8,
        "tokens_per_step": 3072,
        "max_steps": steps,
        "lr": 1.5e-3,
        "warmup_steps": 40,
        "seed": TRAINER_BASE_SEED + seed,
        "device": device,
        "out_dir": str(out_dir),
        "log_every": 25,
        "eval_every": 200,
        "snap_frac": 0.5,
        "ckpt_minutes": 10,
    }


def stage_train(arms: list[str], steps: int, device: str, seed: int = 0) -> None:
    """Train one model per requested corpus (a and/or c); resume from ckpt."""
    sfx = seed_suffix(seed)
    plan = arm_plan(arms)
    needed_runs = sorted({p["run"] for p in plan})
    for run in needed_runs:
        corpus_dir = DATA_DIR / f"corpus_{run}{sfx}"
        require_corpus_complete(corpus_dir, stage="train")
        out_dir = DATA_DIR / "runs" / f"{run}{sfx}"
        cfg = _trainer_cfg(run, corpus_dir, out_dir, steps, device, seed)
        trainer = Trainer(cfg)
        if trainer.ckpt_path.exists():
            trainer.load_ckpt()
            print(f"[train] {run}{sfx}: resumed from step {trainer.step}")
        trainer.train_steps()
        print(f"[train] {run}{sfx}: done at step {trainer.step}")


def _load_model(run_dir: Path, device: str) -> GPT:
    import torch
    import yaml
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg)
    ckpt_path = run_dir / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def accumulate_stats(acc: dict[str, int],
                     batch_stats: dict[str, int]) -> dict[str, int]:
    """Fold one batch's stats into the running accumulator, initializing any
    unseen key to 0 so constrained-only counters (n_constrained_queries,
    n_constraint_dead_ends) reach the per-arm result even when earlier
    batches did not report them."""
    for k, v in batch_stats.items():
        acc[k] = acc.get(k, 0) + v
    return acc


def stage_eval(arms: list[str], device: str, limit: int, seed: int = 0) -> None:
    """Generate + score every requested arm; write per-arm + summary JSONs."""
    sfx = seed_suffix(seed)
    tok = get_tok()
    org = Organizer.load(DATA_DIR / "organizer_real.jsonl")
    items = load_eval_items(DATA_DIR / "eval_items.jsonl")
    if limit:
        items = items[:limit]
    facts = realfact.load_realfacts(REALFACTS_PATH)
    relations = sorted({f.prop for f in facts})

    plan = arm_plan(arms)
    results: dict = {}
    for entry in plan:
        arm = entry["arm"]
        run = entry["run"]
        constrained = entry["constrained"]
        run_dir = DATA_DIR / "runs" / f"{run}{sfx}"
        if not (run_dir / "ckpt.pt").exists():
            print(f"[eval] {arm}: no ckpt at {run_dir / 'ckpt.pt'}; skipping")
            continue
        model = _load_model(run_dir, device)

        texts: list[str] = []
        gen_stats: dict[str, int] = {}
        for batch in _batches(items, EVAL_BATCH_SIZE):
            prompts = [it.prompt for it in batch]
            qt = (build_query_tries(prompts, relations, tok)
                  if constrained else None)
            bt, bs = generate_batch_with_stats(
                model, tok, prompts, max_new=EVAL_MAX_NEW,
                organizer=org, device=device, query_tries=qt,
            )
            texts.extend(bt)
            accumulate_stats(gen_stats, bs)

        scored = score_items(items, texts)
        records = scored.pop("records")
        out = {
            "arm": arm,
            "constrained": constrained,
            "run": f"{run}{sfx}",
            "seed": seed,
            "n_items": len(items),
            "aggregates": scored,
            "generation_stats": gen_stats,
            "config": {"max_new": EVAL_MAX_NEW, "batch_size": EVAL_BATCH_SIZE,
                       "limit": limit, "device": device},
        }
        with open(DATA_DIR / f"results_{arm}{sfx}.json", "w") as f:
            json.dump(out, f, indent=2)
        with open(DATA_DIR / f"records_{arm}{sfx}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        results[arm] = out
        held = scored.get("heldout", {})
        print(f"[eval] {arm} (constrained={constrained}): "
              f"heldout full_key={held.get('full_key', 0):.3f} "
              f"name_half={held.get('name_half', 0):.3f} "
              f"relation_half={held.get('relation_half', 0):.3f} "
              f"answer={held.get('answer', 0):.3f}")

    with open(DATA_DIR / f"summary{sfx}.json", "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)


def _print_table(results: dict) -> None:
    cols = ["full_key", "name_half", "relation_half", "answer",
            "no_lookup_rate", "wrong_in_context_rate"]

    def fmt(agg: dict, c: str) -> str:
        v = agg.get(c)
        if v is None:
            return "  -  "
        lo, hi = agg.get(f"{c}_ci", (0.0, 0.0))
        return f"{v * 100:5.1f} [{lo * 100:4.1f},{hi * 100:4.1f}]"

    header = "arm   " + "  ".join(f"{c:>22}" for c in cols)
    print("\nheldout split (pct [lo,hi]):")
    print(header)
    for arm in ARMS_DEFAULT:
        if arm not in results:
            continue
        held = results[arm]["aggregates"].get("heldout", {})
        if not held:
            continue
        row = arm.ljust(5) + "  ".join(fmt(held, c).rjust(22) for c in cols)
        print(row)


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "train", "eval"])
    ap.add_argument("--arms", default=",".join(ARMS_DEFAULT),
                    help="comma-separated subset of A,B,C,D")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--force", action="store_true",
                    help="rebuild the data stage even if outputs exist")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap eval items (0 = all; for smoke)")
    ap.add_argument("--seed", type=int, default=0,
                    help="replication seed bundle (0 = the original run; "
                         "S>0 varies trainer init, doc order, and "
                         "substitution/flood draws; the PopQA split and "
                         "eval items never vary)")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS_DEFAULT:
            raise SystemExit(f"unknown arm {a!r}; expected A,B,C,D")
    device = pick_device(args.device)
    print(f"device: {device} seed: {args.seed}")

    if args.stage == "data":
        stage_data(force=args.force, seed=args.seed)
    elif args.stage == "train":
        stage_train(arms, args.steps, device, seed=args.seed)
    elif args.stage == "eval":
        stage_eval(arms, device, args.limit, seed=args.seed)
    else:  # all
        stage_data(force=args.force, seed=args.seed)
        stage_train(["A", "C"], args.steps, device, seed=args.seed)
        stage_eval(arms, device, args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
