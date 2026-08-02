import json

import pytest
import yaml

from ops.crowding import ladder


def _summary(mod, by_op, majority=None, n=400):
    return {
        "mod": mod,
        "igsm_acc": sum(by_op.values()) / len(by_op),
        "igsm_majority_rate": majority if majority is not None else 1.0 / mod,
        "igsm_by_op": {str(k): {"acc": v, "n": n} for k, v in by_op.items()},
    }


def _write(root, name, summ):
    d = root / "runs" / name / "evals"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summ))


def test_configs_cover_every_modulus_and_point_at_its_own_corpus(tmp_path):
    corpora = {m: f"/c/ladder-mod{m}" for m in (23, 11, 7, 5)}
    cfgs = ladder.modulus_configs("/root", corpora, "d40m_std", 1.5e-3, 10.0, 20_000)
    assert [c["igsm_mod"] for c in cfgs] == [23, 11, 7, 5]
    for c in cfgs:
        # The modulus is baked into the corpus text, so a rung reading the
        # wrong corpus would silently evaluate a different task than it trained on.
        assert f"ladder-mod{c['igsm_mod']}/targets.bin" == c["train_bin"].split("/", 2)[-1]


def test_configs_hold_everything_but_the_modulus_fixed(tmp_path):
    corpora = {m: f"/c/ladder-mod{m}" for m in (23, 7)}
    cfgs = ladder.modulus_configs("/root", corpora, "d40m_std", 1.5e-3, 10.0, 20_000)
    varying = {k for k in cfgs[0]
               if cfgs[0][k] != cfgs[1][k]}
    assert varying == {"run_id", "igsm_mod", "train_bin", "probe_mask", "out_dir"}


def test_configs_are_dense_only(tmp_path):
    cfgs = ladder.modulus_configs("/r", {23: "/c/a"}, "d40m_std", 1e-3, 10.0, 1)
    assert all(c["arm"] == "sup" for c in cfgs)
    assert all("train_mask" not in c for c in cfgs)


def test_rank_reports_not_learnable_when_every_rung_is_flat(tmp_path):
    for mod in (23, 11, 7, 5):
        flat = {1: 1.0 / mod, 2: 1.0 / mod, 3: 1.0 / mod, 4: 1.0 / mod}
        _write(tmp_path, f"ladder_mod{mod}_d40m_std", _summary(mod, flat))
    r = ladder.rank_modulus(tmp_path / "runs")
    assert not r["any_rung_clears"]
    assert "NOT LEARNABLE" in r["verdict"]
    assert "mod 5" in r["verdict"]


def test_rank_finds_the_easiest_clearing_rung(tmp_path):
    # Flat at 23 and 11, clearly learned at 7 and 5.
    for mod, acc1 in ((23, 0.045), (11, 0.095), (7, 0.60), (5, 0.75)):
        by_op = {1: acc1, 2: acc1 * 0.8, 3: acc1 * 0.6, 4: acc1 * 0.5}
        _write(tmp_path, f"ladder_mod{mod}_d40m_std", _summary(mod, by_op))
    r = ladder.rank_modulus(tmp_path / "runs")
    assert r["any_rung_clears"]
    assert r["easiest_clearing_mod"] == 7
    assert "LEARNABLE at mod 7" in r["verdict"]


def test_rank_judges_on_op1_against_the_no_skill_baseline(tmp_path):
    # Majority well above uniform chance. An op=1 sitting at the majority rate
    # must NOT count as clearing, which is the error the Stage A read made.
    by_op = {1: 0.071, 2: 0.05, 3: 0.085, 4: 0.06}
    _write(tmp_path, "ladder_mod23_d40m_std", _summary(23, by_op, majority=0.0713))
    r = ladder.rank_modulus(tmp_path / "runs")
    assert r["rungs"][0]["no_skill"] == pytest.approx(0.0713)
    assert not r["rungs"][0]["op1_clears_baseline"]


def test_rank_is_empty_safe(tmp_path):
    (tmp_path / "runs").mkdir(parents=True)
    assert "error" in ladder.rank_modulus(tmp_path / "runs")


def _snap(run, step, mod, by_op, majority=None, n=400):
    d = run / "evals"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"step{step:07d}.json").write_text(json.dumps(
        _summary(mod, by_op, majority, n)))


def test_step_ladder_finds_when_the_endpoint_lifts(tmp_path):
    run = tmp_path / "main"
    for step, a in ((1500, 0.045), (6000, 0.05), (25000, 0.55)):
        _snap(run, step, 23, {1: a, 2: a * 0.7, 3: a * 0.5, 4: a * 0.4},
              majority=0.0713)
    r = ladder.step_ladder(run)
    assert r["first_clearing_step"] == 25000
    assert "was a step budget" in r["verdict"]


def test_step_ladder_distinguishes_rising_from_flat(tmp_path):
    run = tmp_path / "rising"
    for step, a in ((1500, 0.045), (6000, 0.06), (25000, 0.075)):
        _snap(run, step, 23, {1: a, 2: a, 3: a, 4: a}, majority=0.0713)
    r = ladder.step_ladder(run)
    assert r["first_clearing_step"] is None
    assert "rising" in r["verdict"]

    flat = tmp_path / "flat"
    for step in (1500, 6000, 25000):
        _snap(flat, step, 23, {1: 0.045, 2: 0.045, 3: 0.045, 4: 0.045},
              majority=0.0713)
    r2 = ladder.step_ladder(flat)
    assert "More steps are not the answer" in r2["verdict"]


def test_step_ladder_is_empty_safe(tmp_path):
    run = tmp_path / "none"
    (run / "evals").mkdir(parents=True)
    assert "error" in ladder.step_ladder(run)


def test_generated_configs_are_valid_yaml(tmp_path):
    cfgs = ladder.modulus_configs("/r", {7: "/c/ladder-mod7"}, "d40m_std",
                                  1.5e-3, 10.0, 20_000)
    round_trip = yaml.safe_load(yaml.safe_dump(cfgs[0], sort_keys=False))
    assert round_trip["igsm_mod"] == 7
    assert round_trip["max_steps"] == ladder.PROBE_STEPS
