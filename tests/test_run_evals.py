"""The evaluation driver: the only producer of the file the analyzer reads."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_SPEC = importlib.util.spec_from_file_location(
    "run_evals", Path(__file__).resolve().parents[1] / "scripts" / "run_evals.py"
)
re_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(re_mod)

CPU = torch.device("cpu")


def _tiny_run(tmp_path, n_entities=8):
    """A finished run: a real tiny checkpoint, a config, and a training log."""
    from train.model import GPT, GPTConfig

    run = tmp_path / "d40m_high_sup_s0"
    run.mkdir(parents=True)
    cfg = {
        "run_id": run.name, "arm": "sup", "seed": 0,
        "model": {"n_layer": 1, "n_head": 2, "d_model": 32,
                  "ctx": 64, "vocab_size": 50304},
        "ctx": 64, "n_entities": n_entities, "corpus_seed": 0,
        "igsm_mod": 7, "igsm_op": [1, 2], "igsm_ood_op": [3, 4],
    }
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=1, n_head=2, d_model=32, ctx=64, vocab_size=50304))
    torch.save({"model": model.state_dict()}, run / "ckpt.pt")
    rows = [
        {"step": s, "loss": 3.0, "loss_ema": 3.0, "clip_ratio": 0.97,
         "clip_frac": 0.1, "grad_norm_preclip": 1.2, "adam_v_mean": 1e-6,
         "loss_masked_values": 9.5}
        for s in (10, 20, 30, 40)
    ]
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return run


def _evaluate(run, **kw):
    kw.setdefault("n_igsm", 6)
    kw.setdefault("n_ded", 4)
    kw.setdefault("n_storage_entities", 4)
    kw.setdefault("batch_size", 4)
    return re_mod.evaluate(run, CPU, kw["n_igsm"], kw["n_ded"],
                           kw["n_storage_entities"], kw["batch_size"])


def test_writes_every_field_the_analyzer_reads(tmp_path):
    """analyze_crowding indexes these by name; a rename here is a silent
    break there."""
    out = _evaluate(_tiny_run(tmp_path))
    for key in ("igsm_acc", "recoverable_bits_per_param", "clip_ratio"):
        assert key in out, f"analyzer needs {key}"


def test_reports_the_empirical_majority_baseline_not_one_over_mod(tmp_path):
    out = _evaluate(_tiny_run(tmp_path))
    assert "igsm_majority_rate" in out
    assert out["igsm_lift_over_majority"] == pytest.approx(
        out["igsm_acc"] - out["igsm_majority_rate"]
    )


def test_reports_per_op_for_the_construct_check(tmp_path):
    out = _evaluate(_tiny_run(tmp_path))
    assert out["igsm_by_op"], "monotone-in-op needs a per-op breakdown"


def test_reports_an_out_of_distribution_band(tmp_path):
    out = _evaluate(_tiny_run(tmp_path))
    assert "igsm_ood_acc" in out and "igsm_ood_by_op" in out


def test_deduction_is_only_reported_per_class(tmp_path):
    """A constant 'no' scores exactly 0.500 on the balanced eval."""
    out = _evaluate(_tiny_run(tmp_path))
    assert set(out["deduction_by_class"]).issubset({"yes", "no"})
    assert "deduction_by_class" in out


def test_reports_both_storage_baselines(tmp_path):
    """Mixing a conditional baseline with an unconditional ceiling overstates
    recovery by the 7.66 bits token length leaks."""
    out = _evaluate(_tiny_run(tmp_path))
    assert "recoverable_bits_per_entity" in out
    assert "recoverable_bits_per_entity_length" in out


def test_carries_optimizer_diagnostics_and_gate_zero_from_the_log(tmp_path):
    out = _evaluate(_tiny_run(tmp_path))
    assert out["clip_ratio"] == pytest.approx(0.97)
    assert out["clip_frac"] == pytest.approx(0.1)
    assert out["loss_masked_values_final"] == pytest.approx(9.5)
    assert out["n_gate0_points"] == 4


def test_per_item_rows_are_saved_for_post_hoc_breakdowns(tmp_path):
    run = _tiny_run(tmp_path)
    _evaluate(run)
    for name in ("igsm.jsonl", "igsm_ood.jsonl", "deduction.jsonl"):
        p = run / "evals" / name
        assert p.exists() and p.read_text().strip()
        first = json.loads(p.read_text().splitlines()[0])
        assert "answer" in first and "pred" in first


def test_a_compiled_checkpoint_loads(tmp_path):
    """torch.compile prefixes every key; production runs set compile: true."""
    run = _tiny_run(tmp_path)
    sd = torch.load(run / "ckpt.pt", weights_only=False)["model"]
    torch.save({"model": {f"_orig_mod.{k}": v for k, v in sd.items()}},
               run / "ckpt.pt")
    out = _evaluate(run)
    assert "igsm_acc" in out


def test_summary_round_trips_into_the_analyzer(tmp_path):
    """End to end: the driver's output must satisfy the analyzer's reader."""
    import shutil
    root = tmp_path / "runs"
    src = _tiny_run(tmp_path)
    out = _evaluate(src)
    (src / "evals" / "summary.json").write_text(json.dumps(out))

    aspec = importlib.util.spec_from_file_location(
        "analyze_crowding",
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_crowding.py")
    ac = importlib.util.module_from_spec(aspec)
    aspec.loader.exec_module(ac)

    for load in ("high",):
        for seed in (0, 1):
            for arm in ("sup", "factmask", "randpos"):
                d = root / f"d40m_{load}_{arm}_s{seed}"
                d.mkdir(parents=True)
                shutil.copytree(src / "evals", d / "evals")
    res = ac.analyse(root, "igsm_acc", -1.0, (-1.0, 2.0), 0.0,
                     primary_load="high")
    assert res["n_cells"] == 2


def test_format_compliance_separates_unparseable_from_wrong():
    """Headline accuracy sums two unrelated failures. On the 2026-08-02 ladder
    at MOD=5 only 51.5% of generations landed in {0..4} at all; among those,
    accuracy was 0.2500 against a 0.2573 majority rate. A model that merely
    learns to terminate with a digit would lift headline accuracy with no
    arithmetic learned, and the step ladder must be able to see the difference.
    """
    from scripts.run_evals import format_compliance

    rows = (
        [{"pred": "3", "correct": True}] * 2       # valid and right
        + [{"pred": "1", "correct": False}] * 2    # valid and wrong
        + [{"pred": "no", "correct": False}] * 3   # deduction label, not in space
        + [{"pred": None, "correct": False}] * 3   # nothing parseable
    )
    out = format_compliance(rows, mod=5, prefix="igsm")
    assert out["igsm_parsed_rate"] == 0.7            # 7 of 10 parsed
    assert out["igsm_in_answer_space_rate"] == 0.4   # 4 of 10 in {0..4}
    assert out["igsm_n_in_answer_space"] == 4
    # 2 correct out of the 4 that could have been correct, not 2 of 10.
    assert out["igsm_acc_given_valid_answer"] == 0.5


def test_format_compliance_handles_an_empty_eval():
    from scripts.run_evals import format_compliance
    assert format_compliance([], mod=23, prefix="igsm") == {}


def test_out_of_space_predictions_never_count_as_correct():
    """A 'no' can never be a right answer to a modular-arithmetic question, so
    in-answer-space count must bound the correct count."""
    from scripts.run_evals import format_compliance
    rows = [{"pred": "no", "correct": False}] * 5 + [{"pred": "2", "correct": True}]
    out = format_compliance(rows, mod=5, prefix="igsm")
    assert out["igsm_acc_given_valid_answer"] <= 1.0
    assert out["igsm_n_in_answer_space"] == 1


def test_the_continuous_endpoint_produces_numbers_at_production_ctx():
    """Exact-match accuracy is a threshold metric and has read floor in every
    run this project has done. Reference-trace NLL moves over the same items,
    which is the difference between an endpoint that can resolve a treatment
    effect and one that cannot. `evals/continuous.py` implemented it and was
    called by nothing until 2026-08-03.
    """
    import torch
    from corpusgen import igsm_lite
    from scripts.run_evals import continuous_endpoint
    from train.model import GPT, PRESETS
    from train.tokenizer import get_tok

    torch.manual_seed(0)
    cfg = PRESETS["d8m"]
    cfg.ctx = 1024
    model = GPT(cfg).eval()
    items = igsm_lite.generate_igsm_eval(6, 1, 2, 12345, set(), mod=5)

    out = continuous_endpoint(model, get_tok(), items, torch.device("cpu"),
                              batch_size=4, prefix="igsm")
    assert "igsm_continuous_error" not in out, out
    assert out["igsm_continuous_n_scored"] == 6
    assert out["igsm_continuous_n_dropped_over_ctx"] == 0
    # M1 is the primary: mean per-token NLL of the gold trace. An untrained
    # model should be near ln(vocab); what matters is that it is finite and
    # positive, so a trained model can move it.
    # Flat scalars, so the frozen analyzer can take one as --endpoint.
    m1 = out["igsm_m1_nll"]
    assert isinstance(m1, float) and 0.0 < m1 < 20.0, m1
    assert out["igsm_m1_nll_n"] == 6
    # NLL is lower-is-better; the analyzer's one-sided verdict assumes the
    # opposite, so a sign-flipped twin is emitted for use as the estimand.
    assert out["igsm_m1_nll_neg"] == -m1


def test_items_longer_than_the_context_are_dropped_not_fatal():
    """A single over-long item must not cost the metric for the whole run, and
    a thinned item set must not pass silently as a clean one."""
    import torch
    from corpusgen import igsm_lite
    from scripts.run_evals import continuous_endpoint
    from train.model import GPT, PRESETS
    from train.tokenizer import get_tok

    torch.manual_seed(0)
    cfg = PRESETS["d8m"]
    cfg.ctx = 64                      # every iGSM item exceeds this
    model = GPT(cfg).eval()
    items = igsm_lite.generate_igsm_eval(4, 1, 2, 999, set(), mod=5)
    out = continuous_endpoint(model, get_tok(), items, torch.device("cpu"),
                              batch_size=4, prefix="igsm")
    assert "igsm_continuous_error" in out
    assert "nothing scoreable" in out["igsm_continuous_error"]
