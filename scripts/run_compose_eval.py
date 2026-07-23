#!/usr/bin/env python
"""Evaluate a trained composition run: in-dist vs OOD two-hop, per-hop lookups,
fact-access conditioning, and the OOD-vs-training-step curve. Emits JSON, a
human-readable summary, and figures.

Usage:
    python scripts/run_compose_eval.py --run outputs/compose_dense --data data/compose_v1
    python scripts/run_compose_eval.py --run outputs/compose_split --data data/compose_v1 --arm split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run from anywhere without PYTHONPATH: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from corpusgen.bios import VALUE_POOLS
from evals.compose_eval import (
    aggregate_two_hop,
    load_items,
    run_singlehop,
    run_two_hop,
)
from organizer.store import Organizer
from train.model import GPT, PRESETS, GPTConfig


def load_model(path: Path, device: str) -> GPT:
    state = torch.load(path, map_location=device, weights_only=False)
    if "model_cfg" in state:               # snapshot
        cfg = GPTConfig(**state["model_cfg"])
    else:                                   # ckpt.pt
        c = state["cfg"]
        cfg = PRESETS[c["model"]] if isinstance(c["model"], str) else GPTConfig(**c["model"])
        if c.get("ctx"):
            cfg.ctx = c["ctx"]
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _snapshots(run: Path, which: str) -> list[tuple[int, Path]]:
    snaps = sorted((run / "snapshots").glob("step*.pt"))
    pairs = [(int(p.stem.replace("step", "")), p) for p in snaps]
    if not pairs and (run / "ckpt.pt").exists():
        st = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        pairs = [(int(st.get("step", 0)), run / "ckpt.pt")]
    if which == "last" and pairs:
        pairs = pairs[-1:]
    return pairs


def _answer_chance(items) -> float:
    xs = [1.0 / len(VALUE_POOLS[it.meta["attr"]]) for it in items
          if it.meta.get("attr") in VALUE_POOLS]
    return sum(xs) / len(xs) if xs else 0.0


def _arm_from_config(run: Path) -> str:
    cfg_p = run / "config.yaml"
    if cfg_p.exists():
        import yaml
        c = yaml.safe_load(cfg_p.read_text())
        if c.get("arm"):
            return c["arm"]
    return "dense" if "dense" in run.name else ("split" if "split" in run.name else "dense")


# ----------------------------------------------------------------- figures


def _make_figures(out: Path, arm: str, curve: list[dict], final: dict, chance: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[figures] matplotlib unavailable ({e}); skipping plots")
        return []
    made = []

    # 1. OOD / in-dist two-hop accuracy vs training step
    if len(curve) >= 1:
        steps = [c["step"] for c in curve]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(steps, [c["indist_acc"] for c in curve], "o-", color="#1f77b4",
                label="in-distribution (P_comp)")
        ax.plot(steps, [c["ood_acc"] for c in curve], "s-", color="#d62728",
                label="OOD (P_held)")
        ax.plot(steps, [c["ood_acc_cond"] for c in curve], "s--", color="#d62728",
                alpha=0.5, label="OOD, fact-access conditioned")
        ax.axhline(chance, ls=":", color="gray", label=f"chance ≈ {chance:.4f}")
        ax.set_xlabel("training step")
        ax.set_ylabel("two-hop answer accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Two-hop composition vs training step ({arm} arm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out / "ood_vs_step.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    # 2. Final accuracy by test set
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["in-dist\n(P_comp)", "OOD\n(P_held)", "OOD cond.\n(P_held)",
              "hop-1\naccess", "hop-2\naccess"]
    vals = [final["indist"]["answer_accuracy"],
            final["ood"]["answer_accuracy"],
            final["ood"].get("answer_accuracy_conditioned", 0.0),
            final["singlehop"]["by_group"].get("held-hop1", {}).get("accuracy", 0.0),
            final["singlehop"]["by_group"].get("held-hop2", {}).get("accuracy", 0.0)]
    colors = ["#1f77b4", "#d62728", "#d62728", "#2ca02c", "#2ca02c"]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(chance, ls=":", color="gray", label=f"answer chance ≈ {chance:.4f}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Composition & fact-access ({arm} arm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "accuracy_by_testset.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # 3. Per-hop lookup accuracy (split only)
    if arm == "split":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, s in enumerate(["indist", "ood"]):
            d = final[s]
            xs = ["hop-1 key", "hop-2 key", "both keys", "answer"]
            ys = [d["hop1_key_accuracy"], d["hop2_key_accuracy"],
                  d["both_hops_key_accuracy"], d["answer_accuracy"]]
            off = -0.2 if i == 0 else 0.2
            ax.bar([x + off for x in range(len(xs))], ys, width=0.4,
                   label=("P_comp" if s == "indist" else "P_held"),
                   color=("#1f77b4" if s == "indist" else "#d62728"), alpha=0.85)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["hop-1 key", "hop-2 key", "both keys", "answer"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title("Split per-hop lookup accuracy (hop-2 requires copying the bridge)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / "per_hop_lookup.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


# ----------------------------------------------------------------- summary


def _fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _write_summary(out: Path, arm: str, store: str, final: dict, chance: float,
                   curve: list[dict], figs: list[Path]) -> str:
    ind, ood, sh = final["indist"], final["ood"], final["singlehop"]
    h1 = sh["by_group"].get("held-hop1", {}).get("accuracy", 0.0)
    h2 = sh["by_group"].get("held-hop2", {}).get("accuracy", 0.0)
    ctrl_pass = ind["answer_accuracy"] >= 0.30
    access_ok = min(h1, h2) >= 0.85
    lines = []
    lines.append(f"# Composition eval — {arm} arm (store={store})\n")
    lines.append(f"Answer chance ≈ {chance:.4f}  (exact-match; tiny by construction)\n")
    lines.append("## Headline")
    lines.append(f"- Positive control (in-dist two-hop learnable): "
                 f"**{'PASS' if ctrl_pass else 'FAIL'}** "
                 f"— in-dist answer accuracy {_fmt_pct(ind['answer_accuracy'])}")
    lines.append(f"- Fact access (both single hops on P_held): "
                 f"**{'OK' if access_ok else 'WEAK'}** "
                 f"— hop-1 {_fmt_pct(h1)}, hop-2 {_fmt_pct(h2)}")
    lines.append(f"- OOD (P_held) two-hop: {_fmt_pct(ood['answer_accuracy'])} "
                 f"unconditioned, {_fmt_pct(ood.get('answer_accuracy_conditioned', 0.0))} "
                 f"conditioned on fact access "
                 f"(access rate {_fmt_pct(ood.get('fact_access_rate', 0.0))})\n")
    lines.append("## Table")
    lines.append("| test set | n | answer acc | hop-1 key | hop-2 key | both keys |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, d in [("in-dist (P_comp)", ind), ("OOD (P_held)", ood)]:
        lines.append(f"| {name} | {d['n']} | {_fmt_pct(d['answer_accuracy'])} "
                     f"| {_fmt_pct(d['hop1_key_accuracy'])} "
                     f"| {_fmt_pct(d['hop2_key_accuracy'])} "
                     f"| {_fmt_pct(d['both_hops_key_accuracy'])} |")
    lines.append(f"| OOD conditioned | {ood.get('n_both_hops_accessible', 0)} "
                 f"| {_fmt_pct(ood.get('answer_accuracy_conditioned', 0.0))} | — | — | — |")
    lines.append("")
    lines.append("## Single-hop fact access")
    for g, v in sorted(sh["by_group"].items()):
        lines.append(f"- {g}: {_fmt_pct(v['accuracy'])}  (n={v['n']})")
    lines.append("")
    if curve:
        lines.append("## OOD vs training step")
        lines.append("| step | in-dist | OOD | OOD cond. |")
        lines.append("|---:|---:|---:|---:|")
        for c in curve:
            lines.append(f"| {c['step']} | {_fmt_pct(c['indist_acc'])} "
                         f"| {_fmt_pct(c['ood_acc'])} | {_fmt_pct(c['ood_acc_cond'])} |")
        lines.append("")
    lines.append("## Verdict")
    if not ctrl_pass:
        lines.append("**KILL/FIX** — the task did not clear chance in-distribution. "
                     "The composition task is mis-scaled (too many relations/attrs, "
                     "too few composed exposures), not the hypothesis. Fix the task "
                     "before spending twin compute. (This is the check mod-23 iGSM skipped.)")
    elif not access_ok:
        lines.append("**INCONCLUSIVE** — in-dist composition is learnable, but single-hop "
                     "fact access on P_held is weak, so OOD failures may be a fact-access "
                     "confound. Raise atomic/bridge exposures and re-check before reading OOD.")
    else:
        lines.append("**GREENLIGHT** — task learnable and fact-access clean. The OOD number "
                     "above is a real generalization signal; proceed to the seeded 3-arm "
                     "twin (dense-closed / dense-open / split) to test H1.")
    lines.append("")
    if figs:
        lines.append("## Figures")
        for p in figs:
            lines.append(f"- {p.name}")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


# ----------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training out dir (snapshots/, ckpt.pt)")
    ap.add_argument("--data", required=True, help="build_compose out dir (eval/, organizer.jsonl)")
    ap.add_argument("--arm", choices=["dense", "split"], default=None)
    ap.add_argument("--store", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--snapshots", choices=["all", "last"], default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    run, data = Path(args.run), Path(args.data)
    arm = args.arm or _arm_from_config(run)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out) if args.out else run / "compose_eval"
    out.mkdir(parents=True, exist_ok=True)

    # store: split uses the organizer by default; dense is closed-book
    use_store = args.store == "on" or (args.store == "auto" and arm == "split")
    organizer = Organizer.load(data / "organizer.jsonl") if use_store else None
    store_str = "on" if use_store else "off"

    indist = load_items(data / "eval" / "comp_indist.jsonl")
    ood = load_items(data / "eval" / "comp_ood.jsonl")
    singles = load_items(data / "eval" / "singlehop.jsonl")
    chance = _answer_chance(ood)
    print(f"[eval] arm={arm} store={store_str} device={device} | "
          f"in-dist={len(indist)} ood={len(ood)} singlehop={len(singles)} "
          f"| answer chance≈{chance:.4f}")

    snaps = _snapshots(run, args.snapshots)
    if not snaps:
        raise SystemExit(f"no snapshots or ckpt.pt found under {run}")

    curve: list[dict] = []
    final: dict = {}
    for step, path in snaps:
        model = load_model(path, device)
        sh = run_singlehop(model, tok=_TOK, items=singles, organizer=organizer, device=device)
        ind_rows, _ = run_two_hop(model, _TOK, indist, organizer, device,
                                  args.max_new, args.batch_size)
        ood_rows, ood_stats = run_two_hop(model, _TOK, ood, organizer, device,
                                          args.max_new, args.batch_size)
        ind_agg = aggregate_two_hop(ind_rows, sh["correct"])
        ood_agg = aggregate_two_hop(ood_rows, sh["correct"])
        curve.append({"step": step,
                      "indist_acc": ind_agg["answer_accuracy"],
                      "ood_acc": ood_agg["answer_accuracy"],
                      "ood_acc_cond": ood_agg.get("answer_accuracy_conditioned", 0.0)})
        print(f"  step {step:>7}: in-dist {_fmt_pct(ind_agg['answer_accuracy'])}  "
              f"OOD {_fmt_pct(ood_agg['answer_accuracy'])}  "
              f"OOD-cond {_fmt_pct(ood_agg.get('answer_accuracy_conditioned', 0.0))}")
        final = {"step": step, "indist": ind_agg, "ood": ood_agg,
                 "singlehop": sh, "ood_lookup_stats": ood_stats}
        # keep detailed rows only for the last (best-trained) checkpoint
        final_rows = {"indist": ind_rows, "ood": ood_rows}

    (out / "results.json").write_text(json.dumps(
        {"arm": arm, "store": store_str, "chance": chance,
         "curve": curve, "final": final}, indent=2))
    with open(out / "rows_final.jsonl", "w") as f:
        for tag, rows in final_rows.items():
            for r in rows:
                f.write(json.dumps({"set": tag, **r}) + "\n")
    figs = _make_figures(out, arm, curve, final, chance)
    text = _write_summary(out, arm, store_str, final, chance, curve, figs)
    print("\n" + text)
    print(f"[eval] wrote {out}/results.json, summary.md, "
          f"{len(figs)} figure(s)")


# tokenizer is a module-global singleton (matches the rest of the repo)
from train.tokenizer import get_tok  # noqa: E402
_TOK = get_tok()

if __name__ == "__main__":
    main()
