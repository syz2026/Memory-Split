"""Invariants of the tiny-cohort generator.

The generator is dependency-free so it runs under the node's bare python3 and
cannot import the model to check itself. These tests are that check.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from train.model import GPT, PRESETS

GEN = Path(__file__).resolve().parents[1] / "ops" / "cohort-tiny" / "gen_configs_tiny.py"
CORPUS_TOKENS = 8_169_455_616
BASE_TOKENS = 7_120_879_616


def load_gen():
    spec = importlib.util.spec_from_file_location("gen_configs_tiny", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return load_gen()


def test_param_counts_match_the_built_models(gen):
    for name, spec in gen.MODELS.items():
        built = GPT(PRESETS[name]).num_params()
        assert gen.param_count(**spec["dims"]) == built, name
    assert gen.param_count(**gen.MODELS["d8m"]["dims"]) == 7_931_776
    assert gen.param_count(**gen.MODELS["d40m"]["dims"]) == 40_560_000


def test_declared_dims_match_the_presets(gen):
    for name, spec in gen.MODELS.items():
        p = PRESETS[name]
        d = spec["dims"]
        assert (d["n_layer"], d["n_head"], d["d_model"]) == (p.n_layer, p.n_head, p.d_model)
        assert d["tie_embeddings"] == p.tie_embeddings


def test_recurrence_does_not_change_the_parameter_count(gen):
    """param_count takes no recurrence argument; assert that stays true."""
    assert "n_recurrence" not in gen.MODELS["d8m"]["dims"]
    assert PRESETS["d8m"].n_recurrence == 3


def test_budget_is_the_whole_corpus(gen):
    assert gen.TOTAL_TOKENS == CORPUS_TOKENS
    assert gen.MAX_STEPS * gen.TOKENS_PER_STEP == gen.TOTAL_TOKENS
    assert gen.MAX_STEPS == 15582
    assert gen.SNAPSHOTS == [1558, 3896, 7791, 11687, 15582]


def test_budget_reaches_the_reasoning_extension(gen):
    assert gen.TOTAL_TOKENS - BASE_TOKENS == 1_048_576_000


def test_arms_differ_only_in_the_base_sidecar(gen):
    for name in gen.MODELS:
        d = gen.cfg_for(name, "dense", 0)
        s = gen.cfg_for(name, "split90", 0)
        assert d["train_bin"] == s["train_bin"]
        assert d["train_mask"] != s["train_mask"]
        assert d["train_mask"][1] == s["train_mask"][1]     # shared extension
        assert d["probe_mask"] == s["probe_mask"]           # same gate-0 positions
        assert "split90" in d["probe_mask"][0]
        for k in ("model", "model_parameters", "seed", "lr", "ctx", "max_steps",
                  "tokens_per_step", "micro_batch_size", "total_tokens", "pair_id"):
            assert d[k] == s[k], (name, k)


def test_pairs_are_distinct_across_models(gen):
    a = gen.cfg_for("d8m", "dense", 0)
    b = gen.cfg_for("d40m", "dense", 0)
    assert a["pair_id"] != b["pair_id"]
    assert a["model_parameters"] != b["model_parameters"]


def test_generator_runs_standalone(tmp_path):
    corpus = tmp_path / "corpus"
    for rel in ("base/packed/targets.bin", "extension/packed/targets.bin",
                "base/sidecars/dense_target_weights.bin",
                "base/sidecars/split90_target_weights.bin",
                "extension/sidecars/shared_target_weights.bin"):
        p = corpus / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    code, runs = tmp_path / "code", tmp_path / "runs"
    env = {"MS_CORPUS": str(corpus), "MS_CODE": str(code), "MS_RUNS": str(runs),
           "PATH": "/usr/bin:/bin"}
    proc = subprocess.run([sys.executable, str(GEN)], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = code / "configs" / "tiny-v3"
    cohort = sorted(p.name for p in out.glob("*.yaml") if not p.name.startswith("smoke"))
    assert len(cohort) == 4
    cfg = yaml.safe_load((out / "d8m-dense-s0.yaml").read_text())
    assert cfg["model"] == "d8m"
    assert cfg["model_parameters"] == 7_931_776
    assert cfg["max_steps"] == 15582
    assert len(cfg["probe_mask"]) == 2
