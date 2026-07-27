#!/usr/bin/env python
"""End-to-end toy pilot (gate 0): build a toy corpus, train both arms,
verify the mechanism, and drive the eval pipeline.

Checks:
  1. Both arms' training loss falls substantially.
  2. The split arm's CE on loss-masked fact values stays well ABOVE its own
     general next-token loss (facts are not being learned into weights,
     language is). Catches inverted/ignored masks.
  3. score_items + recall_accuracy run end-to-end (store ON / OFF) on the
     split model, including lookup interception stats.

Usage: python scripts/smoke_test.py [--device auto|cpu|mps|cuda] [--steps 500]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from corpusgen.build import BuildCfg, build_corpus
from corpusgen.records import QAItem
from evals.recall import recall_accuracy
from evals.scorers import score_items
from organizer.store import Organizer
from train.tokenizer import get_tok
from train.trainer import Trainer, pick_device

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "data" / "smoke"

# lenient floors — gate C on the cluster does the strict mechanism proof
LOSS_DROP_FRAC = 0.75          # final ema must be < first * this
MASKED_CE_FLOOR = 2.5          # absolute floor for split masked-value CE
MASKED_OVER_GENERAL = 1.15     # masked CE must exceed general loss by this factor


def toy_bed(n_docs: int = 60, seed: int = 5) -> list[str]:
    rng = random.Random(seed)
    subjects = ["The workshop", "A quiet river", "The old library", "Every morning",
                "The research team", "A traveling merchant", "The garden", "This method"]
    verbs = ["describes", "follows", "reveals", "considers", "produces", "shelters",
             "changes", "records"]
    objects = ["a pattern of small habits", "the slow work of seasons",
               "an unexpected result", "many careful measurements",
               "a route through the hills", "the shape of an argument",
               "a long tradition", "the daily flow of visitors"]
    docs = []
    for _ in range(n_docs):
        sents = [
            f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}."
            for _ in range(rng.randint(4, 8))
        ]
        docs.append(" ".join(sents))
    return docs


def train_arm(arm: str, data_dir: Path, out_dir: Path, device: str, steps: int) -> Trainer:
    cfg = {
        "run_id": f"smoke_{arm}",
        "arm": arm,
        "model": {"n_layer": 4, "n_head": 4, "d_model": 256, "ctx": 512, "vocab_size": 50304},
        "train_bin": str(data_dir / arm / "train.bin"),
        "train_mask": str(data_dir / arm / "train.mask.bin"),
        "data_dir": str(data_dir),
        "n_entities": 40,
        "micro_batch_size": 4,
        "tokens_per_step": 4 * 512,
        "max_steps": steps,
        "lr": 1.5e-3,
        "warmup_steps": 20,
        "seed": 7,
        "device": device,
        "out_dir": str(out_dir / f"smoke_{arm}"),
        "log_every": 25,
        "eval_every": 100,
        "snap_frac": 0.5,
        "ckpt_minutes": 999,
    }
    trainer = Trainer(cfg)
    trainer.train_steps()
    return trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--keep", action="store_true", help="keep data/smoke between runs")
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    if SMOKE_DIR.exists() and not args.keep:
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    tok = get_tok()
    if not (SMOKE_DIR / "organizer.jsonl").exists():
        import itertools

        cfg = BuildCfg(n_entities=40, total_tokens=240_000, seed=7,
                       igsm_op=(2, 4), deduction_depth=(1, 3),
                       n_igsm_eval=60, n_deduction_eval=60, n_factqa_eval=40,
                       n_fresh_entities=20, n_fresh_eval=20,
                       n_recall_entities=30)
        report = build_corpus(cfg, tok, itertools.cycle(toy_bed()), SMOKE_DIR)
        print("corpus report:", json.dumps({k: v for k, v in report.items()
                                            if isinstance(v, (int, float, str))}, indent=2))

    out_dir = SMOKE_DIR / "runs"
    results = {}
    for arm in ("dense", "split"):
        trainer = train_arm(arm, SMOKE_DIR, out_dir, device, args.steps)
        rows = [json.loads(l) for l in open(trainer.log_path)]
        first, last = rows[0]["loss"], rows[-1]["loss_ema"]
        masked = [r["loss_masked_values"] for r in rows if "loss_masked_values" in r]
        results[arm] = {"first": first, "last": last, "masked": masked, "trainer": trainer}
        print(f"[{arm}] loss {first:.3f} -> {last:.3f}  masked_ce={masked[-1] if masked else None}")

    # 1. training works
    for arm in ("dense", "split"):
        r = results[arm]
        assert r["last"] < r["first"] * LOSS_DROP_FRAC, f"{arm} loss did not fall: {r}"

    # 2. mechanism: split arm never learns fact values
    split = results["split"]
    assert split["masked"], "split arm produced no masked-value metric"
    final_masked = split["masked"][-1]
    assert final_masked > MASKED_CE_FLOOR, f"masked CE too low: {final_masked}"
    assert final_masked > split["last"] * MASKED_OVER_GENERAL, (
        f"masked CE {final_masked:.3f} not above general loss {split['last']:.3f} "
        f"x {MASKED_OVER_GENERAL} — mask may be inverted or ignored"
    )
    # dense arm has no masked positions
    assert not results["dense"]["masked"] or results["dense"]["masked"] == []

    # 3. eval pipeline end-to-end on the split model
    model = results["split"]["trainer"].model.eval()
    organizer = Organizer.load(SMOKE_DIR / "organizer.jsonl")

    def load_items(name, cap):
        items = [QAItem(**json.loads(l)) for l in open(SMOKE_DIR / "eval" / f"{name}.jsonl")]
        return items[:cap]

    igsm_rows, igsm_stats = score_items(model, tok, load_items("igsm", 20), None, device,
                                        max_new=160, batch_size=4)
    print(f"igsm eval ran: n={len(igsm_rows)} acc={sum(r['correct'] for r in igsm_rows)}")
    probes = load_items("recall", 24)
    on = recall_accuracy(model, tok, probes, "on", organizer, device)
    off = recall_accuracy(model, tok, probes, "off", None, device)
    print(f"recall on={on['overall']:.3f} (stats {on['stats']}) off={off['overall']:.3f}")
    assert on["overall"] >= off["overall"] - 1e-9, "store ON should never hurt recall"

    print("\nSMOKE PASS: pipeline + mechanism verified at toy scale")


if __name__ == "__main__":
    main()
