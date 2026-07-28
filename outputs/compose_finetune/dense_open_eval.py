"""Dense-OPEN eval: give the (closed-book) dense model the SAME fact access split
gets, and test whether it can compose. The fairness control for dense-vs-split.

A fair "give dense the graph" cannot just hand over the two facts (the hop-2 fact
IS the answer -> trivial copy). So this builds a RAG-style **in-context KB**: the two
needed atomic facts + N distractor facts, shuffled, then the 2-hop question. Dense
must SELECT P->B then B->V from context and answer — the genuine reasoning residual,
with fact-access equalized to split.

Run on aligned (--seed 1234 + aligned organizer) and novel (--seed 2011 + novel
organizer) to compare against dense-closed (0.5% novel) and split (53.5% novel).

Caveat (interpret with care): dense-ft was trained CLOSED-BOOK (it generates the
trace from memory), so a KB preamble is a mild format shift; a low score can mean
"not trained to read provided facts," not "can't reason". The ALIGNED run is the
control — dense already knows aligned facts, so KB use there is diagnostic.

Usage:
  ../../.venv/bin/python outputs/compose_finetune/dense_open_eval.py \
    --ckpt ~/Downloads/dense_ft.pt --organizer ~/Downloads/organizer.jsonl \
    --tag dense_open_aligned --seed 1234 --n 60 --distractors 8
  # novel:
  ... --organizer ~/Downloads/organizer_novel.jsonl --tag dense_open_novel --seed 2011
"""
from __future__ import annotations

import argparse, json, random, sys
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
CA = ("birth_city", "university", "major", "employer", "current_city")


def load_model(path, device):
    st = torch.load(str(Path(path).expanduser()), map_location=device, weights_only=False)
    if "model_cfg" in st:
        cfg = GPTConfig(**st["model_cfg"])
    else:
        mc = st["cfg"]["model"]; cfg = PRESETS[mc] if isinstance(mc, str) else GPTConfig(**mc)
        if isinstance(st.get("cfg"), dict) and "ctx" in st["cfg"]:
            cfg.ctx = st["cfg"]["ctx"]
    net = GPT(cfg).to(device).eval(); net.load_state_dict(st["model"])
    return net, int(st.get("step", -1))


def fact_sentence(name, key, value):
    phrase = RELATION_PHRASES.get(key, key)   # attr -> phrase; rel stays "mentor"/"advisor"
    return f"{name}'s {phrase} is {value}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--organizer", required=True)
    ap.add_argument("--tag", default="dense_open")
    ap.add_argument("--seed", type=int, default=1234)      # 1234 aligned, 2011 (=1234+777) novel
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-entities", type=int, default=10000)
    ap.add_argument("--distractors", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = get_tok()
    net, step = load_model(args.ckpt, dev)
    org = Organizer.load(str(Path(args.organizer).expanduser()))
    recs = bios.generate_records(args.n_entities, args.seed)
    matched = sum(1 for r in recs[:200] if normalize(f"{r.name}, employer") in org._table)
    print(f"[{args.tag}] step={step} dev={dev} organizer={len(org)} name-match={matched}/200")
    assert matched > 150, "records don't match organizer — wrong --seed for this organizer?"

    bridge = lambda name, rel: org.lookup(f"{name}, {rel}")
    attrval = lambda name, attr: org.lookup(f"{name}, {attr}")
    rng = random.Random(f"denseopen:{args.seed}")
    sample = rng.sample(recs, min(args.n, len(recs)))

    # a pool of distractor facts (rendered) from the organizer
    keys = list(org._table.items())  # ("name, key" -> value) but key is normalized/lowercased
    # rebuild distractor sentences from records (proper case) instead
    distractor_pool = []
    for r in rng.sample(recs, min(2000, len(recs))):
        for a in CA:
            distractor_pool.append(fact_sentence(r.name, a, r.attrs[a]))
        for rel in RELS:
            b = bridge(r.name, rel)
            if b:
                distractor_pool.append(fact_sentence(r.name, rel, b))

    prompts, golds = [], []
    for r in sample:
        for rel in RELS:
            b = bridge(r.name, rel)
            if not b:
                continue
            for a in CA:
                v = attrval(b, a)
                if not v:
                    continue
                needed = [fact_sentence(r.name, rel, b), fact_sentence(b, a, v)]
                kb = needed + rng.sample(distractor_pool, args.distractors)
                rng.shuffle(kb)
                preamble = "Knowledge:\n" + "\n".join(kb) + "\n"
                q = f"What is the {RELATION_PHRASES[a]} of {r.name}'s {rel}? Reasoning:"
                prompts.append(preamble + q)
                golds.append(v)

    hit = 0
    for lo in range(0, len(prompts), 32):
        texts, _ = generate_batch_with_stats(net, tok, prompts[lo:lo+32], 64, None, dev)
        for t, g in zip(texts, golds[lo:lo+32]):
            hit += normalize_answer(g) in normalize_answer(t)
    acc = hit / max(len(golds), 1)
    res = {"tag": args.tag, "step": step, "n": len(golds), "distractors": args.distractors,
           "dense_open_acc": acc, "example": (prompts[0][:500] if prompts else "")}
    (Path(__file__).parent / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"  dense-open 2-hop (facts in context, {args.distractors} distractors): {acc:.1%} (n={len(golds)})")
    print(f"  wrote {args.tag}.json")


if __name__ == "__main__":
    main()
