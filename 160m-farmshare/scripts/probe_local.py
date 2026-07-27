#!/usr/bin/env python
"""Run every probe that is feasible with the artifacts left on this Mac.

Local artifacts (2026-07-21):
  A. outputs/pulled/step0001520.pt   dense 160M, gate round 2, n200k dose
     (single arm: Q3 storage probes + Q1 attention probes, no pair contrast)
  B. data/smoke/runs/smoke_{dense,split}/ckpt.pt   toy 4Lx256 PAIR from gate 0
     (Q2 pair contrasts at demonstration scale)
  plus local eval sets (outputs/pulled/*.jsonl, data/smoke/eval/*) and the
  deterministic corpus generators (facts regenerable from seed).

Writes JSON results to outputs/probe-local/ and prints a digest.
Usage: PYTHONPATH=. .venv/bin/python scripts/probe_local.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from corpusgen.bios import RELATION_PHRASES, generate_records
from corpusgen.records import QAItem
from evals.mechanism import capture_last_token, cohens_d, top_neurons
from probe.attention import head_ablation_effect
from probe.geometry import cross_arm_cka, weight_spectral
from probe.geometry import topk_neuron_overlap
from probe.weights import double_dissociation, fact_superposition, fact_weight_attribution
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok
from train.trainer import pick_device

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "probe-local"

GATE_CORPUS_SEED = 1234  # build_corpus.py default used for every gate corpus
GATE_N_ENTITIES = 200_000


def load_snapshot(path: Path, device: str) -> GPT:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def load_ckpt(path: Path, cfg: dict, device: str) -> GPT:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**cfg))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def qa_items(path: Path, n: int) -> list[QAItem]:
    items = [QAItem(**json.loads(l)) for l in open(path)]
    return items[:n]


def mean_answer_nll(model, tok, items, device, batch_cap=48) -> float:
    """Teacher-forced NLL of the gold answer given the prompt (cheap scorer:
    no generation). Lower = better. Used as the ablation-effect metric."""
    total, count = 0.0, 0
    for item in items[:batch_cap]:
        ctx = tok.encode(item.prompt) or [tok.EOT]
        ans = tok.encode(" " + str(item.answer))
        if not ans:
            continue
        ids = torch.tensor([ctx + ans], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model.forward(ids)
        logp = torch.log_softmax(logits.float(), dim=-1)
        pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(ans), device=device)
        tgt = torch.tensor(ans, device=device)
        total += float(-logp[0, pos, tgt].mean())
        count += 1
    return total / max(1, count)


def fact_prompts_from_records(records, n, rng) -> list[tuple[str, str]]:
    out = []
    for rec in rng.sample(records, n):
        attr = rng.choice(list(RELATION_PHRASES))
        out.append((f"{rec.name}'s {RELATION_PHRASES[attr]} is", rec.attrs[attr]))
    return out


# ---------------------------------------------------------------- part A


def probe_dense_160m(device: str, results: dict) -> None:
    snap = ROOT / "outputs" / "pulled" / "step0001520.pt"
    if not snap.exists():
        results["dense160m"] = {"skipped": "snapshot not found"}
        return
    print("== A. dense 160M (gate r2, n200k) — single-arm storage/attention probes")
    model = load_snapshot(snap, device)
    tok = get_tok()
    rng = random.Random(7)

    # A1: weight geometry vs a fresh init (where did training use capacity?)
    fresh = GPT(GPTConfig(**{
        "n_layer": model.cfg.n_layer, "n_head": model.cfg.n_head,
        "d_model": model.cfg.d_model, "vocab_size": model.cfg.vocab_size,
        "ctx": model.cfg.ctx,
    })).to(device)
    spec_trained = weight_spectral(model.state_dict())
    spec_fresh = weight_spectral(fresh.state_dict())
    delta_rank = {
        l: round(spec_trained[l]["mlp.w2"]["eff_rank"] - spec_fresh[l]["mlp.w2"]["eff_rank"], 2)
        for l in spec_trained
    }
    del fresh
    results["dense160m"] = {"eff_rank_delta_vs_init_w2": delta_rank}
    print("  eff-rank delta (trained - init), mlp.w2 by layer:", delta_rank)

    # A2: which weights store facts — attribution over 12 regenerated facts
    print("  regenerating n200k records (seed %d) ..." % GATE_CORPUS_SEED)
    records = generate_records(GATE_N_ENTITIES, GATE_CORPUS_SEED)
    facts = fact_prompts_from_records(records, 12, rng)
    per_layer_sum: dict[int, float] = {}
    for prompt, answer in facts:
        attr = fact_weight_attribution(model, tok, prompt, answer, device)
        for l, v in attr.items():
            per_layer_sum[l] = per_layer_sum.get(l, 0.0) + v
    total = sum(per_layer_sum.values()) or 1.0
    frac = {l: round(v / total, 4) for l, v in sorted(per_layer_sum.items())}
    results["dense160m"]["fact_attribution_layer_frac"] = frac
    top3 = sorted(frac, key=frac.get, reverse=True)[:3]
    results["dense160m"]["fact_attribution_top3_layers"] = top3
    print("  fact-NLL weight attribution by layer (fraction):", frac)

    # A3: activations — fact selectivity + superposition
    fact_p = [p for p, _ in fact_prompts_from_records(records, 384, rng)]
    igsm = qa_items(ROOT / "outputs" / "pulled" / "igsm.jsonl", 384)
    reason_p = [i.prompt[-1500:] for i in igsm]
    acts_f = capture_last_token(model, tok, fact_p, device)
    acts_r = capture_last_token(model, tok, reason_p, device)
    sel = cohens_d(acts_f["mlp"], acts_r["mlp"])  # [L, H] fact-vs-reasoning
    fact_units = top_neurons(sel, 60)
    sup_f = fact_superposition(acts_f["resid"][:, -1, :])
    sup_r = fact_superposition(acts_r["resid"][:, -1, :])
    results["dense160m"]["fact_selectivity_top_units"] = [
        {"layer": l, "neuron": n, "d": round(d, 2)} for l, n, d in fact_units[:10]
    ]
    results["dense160m"]["superposition"] = {
        "facts_final_resid_PR": round(sup_f["participation_ratio"], 1),
        "reasoning_final_resid_PR": round(sup_r["participation_ratio"], 1),
        "dim": sup_f["dim"],
    }
    print("  participation ratio (final resid): facts %.1f vs reasoning %.1f of %d dims"
          % (sup_f["participation_ratio"], sup_r["participation_ratio"], sup_f["dim"]))

    # A4: causal — ablate the top fact-selective units, watch recall vs reasoning NLL
    recall_items = [QAItem(qid=f"r{i}", task="recall", prompt=p, answer=a, meta={})
                    for i, (p, a) in enumerate(fact_prompts_from_records(records, 48, rng))]
    ded = qa_items(ROOT / "outputs" / "pulled" / "deduction.jsonl", 48)
    diss = double_dissociation(
        model, fact_units,
        recall_scorer=lambda m: -mean_answer_nll(m, tok, recall_items, device),
        reasoning_scorer=lambda m: -mean_answer_nll(m, tok, ded, device),
    )
    results["dense160m"]["fact_neuron_ablation"] = {
        k: {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
        for k, v in diss.items()
    }
    print("  ablate top-60 fact units: recall-NLL delta %.4f vs reasoning-NLL delta %.4f"
          % (-diss["recall"]["delta"], -diss["reasoning"]["delta"]))

    # A5: Q1 taste — per-layer attention-band ablation on deduction answers
    layer_effects = {}
    for layer in range(model.cfg.n_layer):
        heads = [(layer, h) for h in range(model.cfg.n_head)]
        eff = head_ablation_effect(
            model, heads, lambda m: -mean_answer_nll(m, tok, ded, device))
        layer_effects[layer] = round(eff["delta"], 4)
    results["dense160m"]["attention_layer_ablation_delta"] = layer_effects
    print("  attention layer-band ablation (deduction NLL worsening):", layer_effects)


# ---------------------------------------------------------------- part B


def probe_toy_pair(device: str, results: dict) -> None:
    base = ROOT / "data" / "smoke"
    dense_ck = base / "runs" / "smoke_dense" / "ckpt.pt"
    split_ck = base / "runs" / "smoke_split" / "ckpt.pt"
    if not (dense_ck.exists() and split_ck.exists()):
        results["toy_pair"] = {"skipped": "smoke pair not found"}
        return
    print("== B. toy pair (4Lx256, gate-0 smoke) — Q2 pair contrasts (demo scale)")
    cfg = {"n_layer": 4, "n_head": 4, "d_model": 256, "ctx": 512, "vocab_size": 50304}
    dense = load_ckpt(dense_ck, cfg, device)
    split = load_ckpt(split_ck, cfg, device)
    tok = get_tok()
    rng = random.Random(11)

    recall_items = qa_items(base / "eval" / "recall.jsonl", 160)
    fact_p = [i.prompt for i in recall_items]
    ded_items = qa_items(base / "eval" / "deduction.jsonl", 60)
    reason_p = [i.prompt[-1200:] for i in ded_items]

    acts_df = capture_last_token(dense, tok, fact_p, device)
    acts_sf = capture_last_token(split, tok, fact_p, device)
    acts_dr = capture_last_token(dense, tok, reason_p, device)
    acts_sr = capture_last_token(split, tok, reason_p, device)

    cka_fact = cross_arm_cka(acts_df["resid"], acts_sf["resid"])
    cka_reason = cross_arm_cka(acts_dr["resid"], acts_sr["resid"])
    results["toy_pair"] = {
        "cka_fact_prompts": {l: round(v, 3) for l, v in cka_fact.items()},
        "cka_reasoning_prompts": {l: round(v, 3) for l, v in cka_reason.items()},
    }
    print("  dense-vs-split CKA on fact prompts:", results["toy_pair"]["cka_fact_prompts"])
    print("  dense-vs-split CKA on reasoning prompts:",
          results["toy_pair"]["cka_reasoning_prompts"])

    # fact-selective units per arm (fact vs reasoning contrast), then overlap:
    sel_dense = cohens_d(acts_df["mlp"], acts_dr["mlp"])
    sel_split = cohens_d(acts_sf["mlp"], acts_sr["mlp"])
    overlap = topk_neuron_overlap(sel_dense, sel_split, 40)
    results["toy_pair"]["fact_unit_overlap_dense_vs_split"] = {
        k: round(v, 3) if isinstance(v, float) else v for k, v in overlap.items()
    }
    print("  top-40 fact-unit overlap dense vs split:", overlap)

    spec_d = weight_spectral(dense.state_dict())
    spec_s = weight_spectral(split.state_dict())
    rank_delta = {l: round(spec_d[l]["mlp.w2"]["eff_rank"] - spec_s[l]["mlp.w2"]["eff_rank"], 2)
                  for l in spec_d}
    results["toy_pair"]["w2_eff_rank_dense_minus_split"] = rank_delta
    print("  mlp.w2 eff-rank (dense - split) by layer:", rank_delta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")
    OUT.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if not args.skip_b:
        probe_toy_pair(device, results)
    if not args.skip_a:
        probe_dense_160m(device, results)

    out = OUT / "probe_local_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
