"""The learning-rate probe. An untuned rate is the easiest way to fake this
experiment: it makes a model look capacity-crowded when it is merely badly
optimised."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "lr_probe",
    Path(__file__).resolve().parents[1] / "ops" / "crowding" / "lr_probe.py",
)
lp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lp)


def test_grid_spans_the_retained_optimum():
    """The retained probe put d40m at 8.0e-3; a grid that cannot reach it
    would rediscover the frozen 1.5e-3 by construction."""
    assert min(lp.GRID) <= 1.5e-3
    assert max(lp.GRID) >= 8.0e-3


def test_configs_are_dense_only():
    """Selection must not favour an arm, so it uses the dense arm and the
    winner is applied unchanged everywhere."""
    cfgs = lp.configs("c", "d40m_std", "/r", 1500, lp.GRID, 10.0, 100)
    assert all(c["arm"] == "sup" for c in cfgs)
    assert all("train_mask" not in c for c in cfgs)


def test_configs_differ_only_in_lr():
    cfgs = lp.configs("c", "d40m_std", "/r", 1500, [1e-3, 2e-3], 10.0, 100)
    a = {k: v for k, v in cfgs[0].items() if k not in ("lr", "run_id", "out_dir")}
    b = {k: v for k, v in cfgs[1].items() if k not in ("lr", "run_id", "out_dir")}
    assert a == b


def _write(root, model, lr, loss, steps=1500):
    d = root / f"lrprobe_{model}_{lr:g}"
    d.mkdir(parents=True)
    rows = [{"step": s, "loss_ema": loss, "clip_frac": 0.0}
            for s in range(0, steps + 1, 100)]
    (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_rank_picks_the_lowest_loss(tmp_path):
    for lr, loss in ((1.5e-3, 3.0), (6e-3, 2.5), (2.4e-2, 3.4)):
        _write(tmp_path, "d40m_std", lr, loss)
    r = lp.rank(tmp_path, "d40m_std")
    assert r["chosen_lr"] == 6e-3


def test_an_optimum_at_the_grid_edge_is_flagged(tmp_path):
    """Reporting a rate at the edge of a grid is reporting that you did not
    find the optimum."""
    for lr, loss in ((1.5e-3, 3.0), (6e-3, 2.6), (2.4e-2, 2.2)):
        _write(tmp_path, "d40m_std", lr, loss)
    r = lp.rank(tmp_path, "d40m_std")
    assert r["chosen_lr"] == 2.4e-2
    assert r["bracketed"] is False
    assert "GRID EDGE" in r["note"]


def test_a_bracketed_optimum_passes(tmp_path):
    for lr, loss in ((1.5e-3, 3.0), (3e-3, 2.7), (6e-3, 2.4),
                     (1.2e-2, 2.6), (2.4e-2, 3.1)):
        _write(tmp_path, "d40m_std", lr, loss)
    r = lp.rank(tmp_path, "d40m_std")
    assert r["bracketed"] is True
    assert r["chosen_lr"] == 6e-3


def test_rank_reports_missing_runs(tmp_path):
    assert "error" in lp.rank(tmp_path, "d40m_std")
