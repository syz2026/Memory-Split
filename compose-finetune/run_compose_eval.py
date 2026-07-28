#!/usr/bin/env python
"""Evaluate a finetuned compose checkpoint (reconstruction of scripts/run_compose_eval.py).

Metrics (scored by exact match on the text after `Answer:` for two-hop; by
substring for single-hop — the CORRECT scorer, not the buggy Answer:-parser):
  * two-hop answer accuracy on the held-out in-distribution (P_comp) set and the
    OOD (P_held) set,
  * single-hop cold recall for hop-1 (bridge) and hop-2 (attribute), by group.

For the split arm pass `--arm split`: generation runs with the organizer store
attached (built from the corpus's store.json), exactly like training-time lookups.

Usage:
  python run_compose_eval.py --run <run_dir> --data <corpus_dir> [--arm split] \
      [--ckpt ckpt.pt] [--max-eval 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
for _cand in (_here, _here.parent, Path.cwd()):
    if (_cand / "train").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from evals.generate import generate_batch_with_stats
from evals.scorers import normalize_answer, parse_answer
from corpusgen.bios import RELATION_PHRASES
from organizer.store import Organizer


def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if "model_cfg" in state:
        cfg = GPTConfig(**state["model_cfg"])
    else:
        mc = state["cfg"]["model"]
        cfg = PRESETS[mc] if isinstance(mc, str) else GPTConfig(**mc)
        if "ctx" in state.get("cfg", {}):
            cfg.ctx = state["cfg"]["ctx"]
    net = GPT(cfg).to(device).eval()
    net.load_state_dict(state["model"])
    return net, cfg, int(state.get("step", -1))


def build_org(data_dir):
    store = json.loads((Path(data_dir) / "store.json").read_text())
    org = Organizer()
    for k, v in store.items():
        name, key = k.rsplit("|", 1)
        org.add(name, key, v)
    return org


def twohop_prompt(item):
    ap = RELATION_PHRASES[item["attr"]]
    return f"What is the {ap} of {item['person']}'s {item['rel']}? Reasoning:"


def eval_twohop(net, tok, items, organizer, device, max_new=64, bs=64):
    hit = n = 0
    for lo in range(0, len(items), bs):
        chunk = items[lo:lo + bs]
        texts, _ = generate_batch_with_stats(
            net, tok, [twohop_prompt(it) for it in chunk], max_new, organizer, device)
        for it, t in zip(chunk, texts):
            pred = parse_answer(t) or ""
            hit += int(normalize_answer(it["answer"]) == normalize_answer(pred))
            n += 1
    return {"answer_accuracy": hit / max(n, 1), "n": n}


def eval_singlehop(net, tok, hop1, hop2, organizer, device, max_new=16, bs=64):
    def run(items, prompt_fn, gold_key):
        hit = n = 0
        for lo in range(0, len(items), bs):
            chunk = items[lo:lo + bs]
            texts, _ = generate_batch_with_stats(
                net, tok, [prompt_fn(it) for it in chunk], max_new, organizer, device)
            for it, t in zip(chunk, texts):
                hit += int(normalize_answer(it[gold_key]) in normalize_answer(t))
                n += 1
        return hit / max(n, 1), n
    a1, n1 = run(hop1, lambda it: f"Reasoning: {it['person']}'s {it['rel']} is", "bridge")
    a2, n2 = run(hop2, lambda it: f"Reasoning: {it['person']}'s {RELATION_PHRASES[it['attr']]} is", "value")
    return {"hop1_accuracy": a1, "n_hop1": n1, "hop2_accuracy": a2, "n_hop2": n2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing ckpt.pt (or pass --ckpt)")
    ap.add_argument("--data", required=True, help="corpus dir with eval.json + store.json")
    ap.add_argument("--arm", default="dense", choices=["dense", "split"])
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (default <run>/ckpt.pt)")
    ap.add_argument("--max-eval", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tok()
    ckpt = args.ckpt or str(Path(args.run) / "ckpt.pt")
    net, cfg, step = load_model(ckpt, device)
    org = build_org(args.data) if args.arm == "split" else None
    ev = json.loads((Path(args.data) / "eval.json").read_text())

    indist = ev["twohop_indist"][: args.max_eval]
    ood = ev["twohop_ood"][: args.max_eval]
    sh = ev["singlehop"]

    print(f"[eval] arm={args.arm} step={step} store={'on' if org else 'off'} "
          f"| indist={len(indist)} ood={len(ood)}")
    res = {
        "arm": args.arm, "step": step, "store": "on" if org else "off",
        "twohop_indist": eval_twohop(net, tok, indist, org, device),
        "twohop_ood": eval_twohop(net, tok, ood, org, device),
        "singlehop_comp": eval_singlehop(net, tok, sh["comp_hop1"][: args.max_eval],
                                         sh["comp_hop2"][: args.max_eval], org, device),
        "singlehop_held": eval_singlehop(net, tok, sh["held_hop1"][: args.max_eval],
                                         sh["held_hop2"][: args.max_eval], org, device),
    }

    outdir = Path(args.run) / "compose_eval"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(res, indent=2))
    summary = (
        f"# compose eval — arm={args.arm} step={step}\n\n"
        f"- in-dist two-hop (P_comp): **{res['twohop_indist']['answer_accuracy']:.1%}**"
        f" (n={res['twohop_indist']['n']})\n"
        f"- OOD two-hop (P_held):     **{res['twohop_ood']['answer_accuracy']:.1%}**"
        f" (n={res['twohop_ood']['n']})\n"
        f"- single-hop P_comp: hop1 {res['singlehop_comp']['hop1_accuracy']:.1%}, "
        f"hop2 {res['singlehop_comp']['hop2_accuracy']:.1%}\n"
        f"- single-hop P_held: hop1 {res['singlehop_held']['hop1_accuracy']:.1%}, "
        f"hop2 {res['singlehop_held']['hop2_accuracy']:.1%}\n"
    )
    (outdir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
