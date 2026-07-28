"""Local exploratory eval battery for the finetuned SPLIT compose model.

Driven by the REAL corpus organizer (`organizer.jsonl`) so the store + bridge graph
exactly match training (regardless of which builder produced the corpus). Entity
names + bio attributes come from `generate_records(seed)` (proper-cased); bridges
come from the organizer. We verify the two line up before running.

Batteries:
  A. fact-lookup RETENTION  — single-hop recall, store ON vs OFF (OFF≈0, ON high => it
     still knows how to query; a drop in ON => it forgot the lookup skill).
  B. 2-hop composition       — store ON (sanity that the trained skill is present).
  C. N-hop GENERALIZATION    — 3-hop chained queries never trained on ("attr of X's
     mentor's advisor"), store ON. Does 2-hop training extend to 3 hops?
  D. free-gen example        — eyeball the trace.

Usage:
  ../../.venv/bin/python outputs/compose_finetune/local_eval.py \
      --ckpt ~/Downloads/step0000940.pt --organizer ~/Downloads/organizer.jsonl \
      --tag split_ft --n 60 --seed 1234
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from corpusgen import bios
from corpusgen.bios import RELATION_PHRASES
from organizer.store import Organizer, normalize
from evals.generate import generate_batch_with_stats
from evals.scorers import normalize_answer
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok

RELS = ("mentor", "advisor")
# attributes actually composed in this corpus (from report.json)
CA = ("birth_city", "university", "major", "employer", "current_city")


def load_model(path, device):
    st = torch.load(str(Path(path).expanduser()), map_location=device, weights_only=False)
    if "model_cfg" in st:
        cfg = GPTConfig(**st["model_cfg"])
    else:
        mc = st["cfg"]["model"]
        cfg = PRESETS[mc] if isinstance(mc, str) else GPTConfig(**mc)
        if isinstance(st.get("cfg"), dict) and "ctx" in st["cfg"]:
            cfg.ctx = st["cfg"]["ctx"]
    net = GPT(cfg).to(device).eval()
    net.load_state_dict(st["model"])
    return net, int(st.get("step", -1))


def _acc(net, tok, pairs, organizer, device, max_new):
    prompts = [p for p, _ in pairs]; golds = [g for _, g in pairs]
    texts, stats = generate_batch_with_stats(net, tok, prompts, max_new, organizer, device)
    hits = sum(normalize_answer(g) in normalize_answer(t) for t, g in zip(texts, golds))
    return hits / max(len(golds), 1), stats, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--organizer", required=True, help="path to organizer.jsonl")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-entities", type=int, default=10000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = get_tok()
    net, step = load_model(args.ckpt, dev)
    org = Organizer.load(str(Path(args.organizer).expanduser()))
    print(f"[{args.tag}] step={step} dev={dev} | organizer keys={len(org)}")

    records = bios.generate_records(args.n_entities, args.seed)
    # verify names line up with the organizer (proper-case name -> lowercased key present)
    matched = sum(1 for r in records[:200] if normalize(f"{r.name}, employer") in org._table)
    print(f"  name/organizer match: {matched}/200 (should be ~200)")
    assert matched > 150, "generate_records names don't match organizer — wrong seed?"

    bridge = lambda name, rel: org.lookup(f"{name}, {rel}")
    attrval = lambda name, attr: org.lookup(f"{name}, {attr}")
    rng = random.Random(f"localeval:{args.seed}")
    sample = rng.sample(records, min(args.n, len(records)))
    res = {"tag": args.tag, "step": step, "device": dev, "n_sample": len(sample)}

    # ---- A. single-hop retention: store ON vs OFF ----
    org.reset_counters()
    sh = [(f"Reasoning: {r.name}'s {RELATION_PHRASES[a]} is", r.attrs[a]) for r in sample for a in CA]
    on, st_on, _ = _acc(net, tok, sh, org, dev, 24)
    off, _, _ = _acc(net, tok, sh, None, dev, 24)
    res["A_single_hop"] = {"store_on": on, "store_off": off, "n": len(sh),
                           "lookups": st_on["n_lookups"], "misses": st_on["n_misses"]}
    print(f"  A single-hop: ON {on:.1%} / OFF {off:.1%}  (n={len(sh)}, miss={st_on['n_misses']})")

    # ---- B. 2-hop, store ON ----
    two = []
    for r in sample:
        for rel in RELS:
            b = bridge(r.name, rel)
            if not b: continue
            for a in CA:
                v = attrval(b, a)
                if v: two.append((f"What is the {RELATION_PHRASES[a]} of {r.name}'s {rel}? Reasoning:", v))
    acc2, st2, _ = _acc(net, tok, two, org, dev, 96)
    res["B_twohop"] = {"answer_acc": acc2, "n": len(two)}
    print(f"  B 2-hop: {acc2:.1%} (n={len(two)})")

    # ---- C. 3-hop generalization (never trained), store ON ----
    three = []
    for r in sample:
        for r1 in RELS:
            b1 = bridge(r.name, r1)
            if not b1: continue
            for r2 in RELS:
                b2 = bridge(b1, r2)
                if not b2: continue
                a = CA[(len(r.name) + len(r1)) % len(CA)]
                v = attrval(b2, a)
                if v: three.append((f"What is the {RELATION_PHRASES[a]} of {r.name}'s {r1}'s {r2}? Reasoning:", v))
    acc3, st3, ex3 = _acc(net, tok, three, org, dev, 128)
    res["C_threehop"] = {"answer_acc": acc3, "n": len(three), "lookups": st3["n_lookups"],
                         "example_gen": ex3[0][:400] if ex3 else ""}
    print(f"  C 3-hop (never trained): {acc3:.1%} (n={len(three)}, lookups={st3['n_lookups']})")

    # ---- D. free-gen sanity ----
    r = sample[0]
    d = generate_batch_with_stats(net, tok, [f"What is the employer of {r.name}'s mentor? Reasoning:"], 96, org, dev)[0][0]
    res["D_example"] = {"person": r.name, "gen": d[:400]}
    print(f"  D ({r.name}): {d[:180]!r}")

    outp = Path(__file__).parent / f"local_eval_{args.tag}.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
