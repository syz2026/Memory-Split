"""Tiny end-to-end pipeline test (Task 6): corpus -> both arms -> mechanism
-> eval scoring. Marked slow (~40-80 s CPU); run with `pytest -m slow`."""

import itertools
import json

import pytest

from corpusgen.build import BuildCfg, build_corpus
from corpusgen.records import QAItem
from evals.recall import recall_accuracy
from evals.scorers import score_items
from organizer.store import Organizer
from train.tokenizer import get_tok
from train.trainer import Trainer

BED = [
    "The river cut through the valley long before the first roads were laid.",
    "Glass is made by melting sand together with soda ash and limestone.",
    "Most weather patterns in the region are driven by the westerly winds.",
    "Bees communicate the location of food through a series of movements.",
    "The lighthouse keeper recorded every passing ship in a leather journal.",
    "Copper conducts electricity well and has been used in wiring for a century.",
]


def bed_iter():
    for i in itertools.count():
        yield f"{BED[i % len(BED)]} This is bed passage number {i}."


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    tok = get_tok()
    cfg = BuildCfg(
        n_entities=12, total_tokens=40_000, seed=3,
        igsm_op=(2, 3), deduction_depth=(1, 2),
        n_igsm_eval=8, n_deduction_eval=8, n_factqa_eval=8,
        n_fresh_entities=4, n_fresh_eval=4, n_recall_entities=8,
    )
    report = build_corpus(cfg, tok, bed_iter(), tmp_path)
    assert report["checks"]["shares_within_1pct"]

    results = {}
    for arm in ("dense", "split"):
        trainer = Trainer({
            "arm": arm,
            "model": {"n_layer": 2, "n_head": 2, "d_model": 128, "ctx": 384,
                      "vocab_size": 50304},
            "train_bin": str(tmp_path / arm / "train.bin"),
            "train_mask": str(tmp_path / arm / "train.mask.bin"),
            "micro_batch_size": 4,
            "tokens_per_step": 4 * 384,
            "max_steps": 60,
            "lr": 2e-3,
            "warmup_steps": 10,
            "seed": 5,
            "device": "cpu",
            "out_dir": str(tmp_path / "runs" / arm),
            "log_every": 10,
            "eval_every": 30,
            "snap_frac": 1.0,
            "ckpt_minutes": 999,
        })
        trainer.train_steps()
        rows = [json.loads(l) for l in open(trainer.log_path)]
        results[arm] = {"rows": rows, "trainer": trainer}
        assert rows[-1]["loss_ema"] < rows[0]["loss"], arm

    masked = [r["loss_masked_values"] for r in results["split"]["rows"]
              if "loss_masked_values" in r]
    assert masked and masked[-1] > 2.5  # fact values stay unlearned
    assert not any("loss_masked_values" in r for r in results["dense"]["rows"])

    model = results["split"]["trainer"].model.eval()
    organizer = Organizer.load(tmp_path / "organizer.jsonl")
    items = [QAItem(**json.loads(l))
             for l in open(tmp_path / "eval" / "recall.jsonl")][:6]
    on = recall_accuracy(model, tok, items, "on", organizer, "cpu", max_new=24)
    off = recall_accuracy(model, tok, items, "off", None, "cpu", max_new=24)
    assert set(on) >= {"overall", "per_attribute", "n", "stats"}
    assert on["overall"] >= off["overall"] - 1e-9

    igsm_items = [QAItem(**json.loads(l))
                  for l in open(tmp_path / "eval" / "igsm.jsonl")][:4]
    rows, stats = score_items(model, tok, igsm_items, None, "cpu",
                              max_new=64, batch_size=2)
    assert len(rows) == 4 and {"qid", "correct", "pred"} <= set(rows[0])
