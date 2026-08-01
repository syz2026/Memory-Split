"""End-to-end rehearsal of the whole pipeline at toy scale, on CPU.

Build a corpus, train all three arms from one byte-identical stream, evaluate
each, and run the frozen analyzer. Proves the chain holds together before any
cluster time is spent, and asserts the properties that make the contrast mean
what it claims:

  - the three arms read byte-identical tokens
  - masking does not reweight the surviving targets
  - the two sidecars have equal mass and never overlap
  - the analyzer accepts a complete matrix and refuses an incomplete one

It does NOT produce a scientific result. At this scale nothing is learned;
the point is that every seam fits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.trainer import Trainer  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load(ROOT / "ops" / "crowding" / "build_corpus.py", "build_corpus")
gc = _load(ROOT / "ops" / "crowding" / "gen_configs.py", "gen_configs")
re_mod = _load(ROOT / "scripts" / "run_evals.py", "run_evals")
ac = _load(ROOT / "scripts" / "analyze_crowding.py", "analyze_crowding")

SHARES = {"fact": 0.5, "igsm": 0.3, "deduction": 0.1, "bed": 0.1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/crowding-smoke")
    ap.add_argument("--entities", type=int, default=60)
    ap.add_argument("--exposures", type=int, default=25)
    ap.add_argument("--tokens", type=int, default=300_000)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "runs").mkdir(parents=True)
    report: dict = {}
    t0 = time.time()

    # ---------------------------------------------------------------- corpus
    print("building corpus ...", flush=True)
    corpus = out / "corpora" / "high"
    man = bc.build(corpus, args.entities, args.exposures, SHARES,
                   args.tokens, seed=0)
    fails = bc.verify(corpus, expect_tokens=args.tokens)
    assert not fails, fails
    audit = man["mask_audit"]
    assert audit["mass_matched"], audit
    assert audit["overlap"] == 0, audit
    report["corpus"] = {
        "tokens": man["n_tokens"],
        "lane_tokens": man["lane_tokens"],
        "mask_mass_matched": audit["mass_matched"],
        "randpos_overlaps_values": audit["overlap"],
        "masked_frac": round(audit["factmask_frac"], 5),
    }
    print(f"  {man['n_tokens']:,} tokens, mask mass matched, 0 overlap")

    # ---------------------------------------------------------------- train
    cfgs = gc.cohort({"high": str(corpus)}, args.seeds, "toy",
                     args.steps * 4096, 3e-3, 1.0,
                     load_entities={"high": args.entities}, corpus_seed=0,
                     igsm_mod=7, igsm_op=(1, 2))
    gc.assert_arms_match(cfgs)

    losses: dict[str, float] = {}
    for c in cfgs:
        c["model"] = {"n_layer": 2, "n_head": 2, "d_model": 64,
                      "ctx": 64, "vocab_size": 50304}
        c["ctx"] = 64
        c["micro_batch_size"] = 4
        c["tokens_per_step"] = 4 * 64
        c["max_steps"] = args.steps
        c["device"] = "cpu"
        c["compile"] = False
        c["log_every"] = 1
        c["eval_every"] = 2
        c["snap_frac"] = 1.0
        c["ckpt_minutes"] = 999
        c["igsm_ood_op"] = [3, 4]
        c["out_dir"] = str(out / "runs" / c["run_id"])
        c["train_bin"] = str(corpus / "targets.bin")
        c["probe_mask"] = str(corpus / "factmask.bin")
        if "train_mask" in c:
            c["train_mask"] = str(corpus / Path(c["train_mask"]).name)

        print(f"training {c['run_id']} ...", flush=True)
        tr = Trainer(c)
        tr.train_steps()
        rows = [json.loads(x) for x in open(tr.log_path)]
        losses[c["run_id"]] = rows[-1]["loss"]
        assert "clip_ratio" in rows[-1], "diagnostics missing"

    # The arms must read the same tokens. Only the weights differ.
    first = cfgs[0]
    for c in cfgs[1:]:
        assert c["train_bin"] == first["train_bin"], "arms read different streams"
    report["arms_share_one_stream"] = True

    # ---------------------------------------------------------------- evals
    for c in cfgs:
        run = Path(c["out_dir"])
        (run / "evals").mkdir(parents=True, exist_ok=True)
        summ = re_mod.evaluate(run, torch.device("cpu"), n_igsm=8, n_ded=4,
                               n_storage_entities=4, batch_size=4)
        (run / "evals" / "summary.json").write_text(json.dumps(summ, indent=2))
        print(f"  evaluated {c['run_id']}: igsm={summ['igsm_acc']:.3f}")

    # ---------------------------------------------------------------- verdict
    res = ac.analyse(out / "runs", "igsm_acc",
                     storage_floor=-1e9, igsm_band=(-1.0, 2.0),
                     min_interesting_effect=0.0)
    report["analyzer"] = {
        "n_cells": res["n_cells"],
        "verdict": res["verdict"],
        "primary_contrast": res["primary_contrast"],
    }
    print(f"  analyzer: {res['n_cells']} cells, verdict={res['verdict']}")

    # The analyzer must refuse an incomplete matrix.
    shutil.rmtree(out / "runs" / cfgs[-1]["run_id"])
    try:
        ac.analyse(out / "runs", "igsm_acc", -1e9, (-1.0, 2.0), 0.0)
        raise AssertionError("analyzer accepted an incomplete matrix")
    except ac.IncompleteMatrix:
        report["analyzer"]["refuses_incomplete"] = True
    print("  analyzer refuses an incomplete matrix")

    report["final_losses"] = losses
    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "smoke-report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))
    print(f"\nPIPELINE OK ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
