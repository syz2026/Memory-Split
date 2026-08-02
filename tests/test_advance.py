import yaml

from ops.crowding import advance

ROOT = "/scratch/x/crowding"
OK_CORPUS = [{"name": "main", "verified": True, "bytes": 65_600_000_000,
              "n_tokens": 16_400_000_000, "mod": 23}]


def _ladder(clearing=None):
    return {"rungs": [{"mod": m} for m in (23, 11, 7, 5)],
            "any_rung_clears": clearing is not None,
            "easiest_clearing_mod": clearing}


def test_missing_main_corpus_blocks_everything(advance_root=ROOT):
    a = advance.decide(None, [], [], [], ROOT)
    assert a["action"] == "REBUILD MAIN CORPUS"


def test_building_corpus_means_wait():
    q = ["1 crowding-build RUNNING 5:00 barley-01"]
    a = advance.decide(None, [], [], q, ROOT)
    assert a["action"] == "WAIT"


def test_unverified_corpus_is_not_accepted():
    corp = [{"name": "main", "verified": False, "bytes": 1}]
    a = advance.decide(None, corp, [], [], ROOT)
    assert a["action"] == "REBUILD MAIN CORPUS"


def test_step_ladder_run_launches_before_the_modulus_ladder_is_known():
    # The whole point: the step budget is the surviving explanation, so this
    # run does not wait on the ladder.
    a = advance.decide(None, OK_CORPUS, [], [], ROOT)
    assert a["action"] == "LAUNCH THE STEP-LADDER RUN"
    assert any("train.sbatch" in c for c in a["commands"])


def test_step_ladder_launches_even_when_no_rung_clears():
    a = advance.decide(_ladder(None), OK_CORPUS, [], [], ROOT)
    assert a["action"] == "LAUNCH THE STEP-LADDER RUN"


def test_running_main_reports_progress():
    runs = [{"name": advance.MAIN_RUN, "last_step": 15_640, "n_snapshots": 10}]
    q = [f"1 {advance.MAIN_RUN} RUNNING 8:00:00 oat-01"]
    a = advance.decide(_ladder(7), OK_CORPUS, runs, q, ROOT)
    assert a["action"] == "WAIT"
    assert "50%" in a["why"]


def test_unscored_snapshots_are_scored_next():
    runs = [{"name": advance.MAIN_RUN, "last_step": advance.FULL_STEPS,
             "n_snapshots": 20, "n_snapshot_evals": 3}]
    a = advance.decide(_ladder(7), OK_CORPUS, runs, [], ROOT)
    assert a["action"] == "SCORE THE SNAPSHOTS"
    assert "20 snapshots, 3 scored" in a["why"]


def test_operating_point_is_frozen_when_both_ladders_are_in():
    runs = [{"name": advance.MAIN_RUN, "last_step": advance.FULL_STEPS,
             "n_snapshots": 20, "n_snapshot_evals": 20}]
    a = advance.decide(_ladder(7), OK_CORPUS, runs, [], ROOT)
    assert a["action"] == "FREEZE THE OPERATING POINT AT MOD 7"
    assert "HARDEST modulus" in a["why"]


def test_stop_only_after_the_step_budget_has_also_been_tested():
    runs = [{"name": advance.MAIN_RUN, "last_step": advance.FULL_STEPS,
             "n_snapshots": 20, "n_snapshot_evals": 20}]
    a = advance.decide(_ladder(None), OK_CORPUS, runs, [], ROOT)
    assert a["action"] == "STOP AND REPORT"
    assert "560 GPU-hours" in a["why"]


def test_unranked_ladder_after_a_finished_run_asks_for_the_rank():
    runs = [{"name": advance.MAIN_RUN, "last_step": advance.FULL_STEPS,
             "n_snapshots": 20, "n_snapshot_evals": 20}]
    a = advance.decide(None, OK_CORPUS, runs, [], ROOT)
    assert a["action"] == "RANK THE LADDER"


def test_main_config_gives_a_real_step_ladder():
    c = advance.main_config("/c/main", ROOT, 1.5e-3, 1.0, 1_531_800, 23)
    assert c["max_steps"] == advance.FULL_STEPS
    # 20 snapshots spread over the run, not one at the end.
    assert c["snap_frac"] == advance.SNAP_FRAC
    assert int(1 / c["snap_frac"]) == 20
    assert c["arm"] == "sup" and "train_mask" not in c
    assert c["total_tokens"] == advance.FULL_STEPS * 524_288
    assert yaml.safe_load(yaml.safe_dump(c))["igsm_mod"] == 23


def test_main_config_warmup_is_a_small_fraction_at_full_length():
    # Stage A spent 19.7% of its run in warmup. At 31,280 steps this must not.
    c = advance.main_config("/c/main", ROOT, 1.5e-3, 1.0, 1_531_800, 23)
    assert c["warmup_steps"] / c["max_steps"] < 0.02
