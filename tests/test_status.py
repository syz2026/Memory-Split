import json

from ops.crowding import status


def _run(root, name, *, step=None, ckpt=False, ev=False, snaps=0):
    d = root / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    if step is not None:
        (d / "log.jsonl").write_text(json.dumps({"step": step, "loss": 1.0}) + "\n")
    if ckpt:
        (d / "ckpt.pt").write_bytes(b"x")
    if ev:
        (d / "evals").mkdir(exist_ok=True)
        (d / "evals" / "summary.json").write_text("{}")
    if snaps:
        (d / "snapshots").mkdir(exist_ok=True)
        for i in range(snaps):
            (d / "snapshots" / f"step{i:07d}.pt").write_bytes(b"x")


def _corpus(root, name, *, verified=True, tokens=800_000_000, mod=23):
    d = root / "corpora" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "targets.bin").write_bytes(b"0" * 128)
    if verified:
        (d / "manifest.json").write_text(json.dumps(
            {"n_tokens": tokens, "n_entities": 20_000, "exposures": 20, "mod": mod}))


def test_running_jobs_mean_wait(tmp_path):
    _run(tmp_path, "ladder_mod23_d40m_std", step=100)
    q = ["1 crowding-train RUNNING 5:00 oat-01"]
    a = status.next_action([], status.runs(tmp_path), q, tmp_path)
    assert a["action"] == "WAIT"


def test_no_ladder_means_submit_it(tmp_path):
    (tmp_path / "runs").mkdir(parents=True)
    a = status.next_action([], status.runs(tmp_path), [], tmp_path)
    assert a["action"] == "SUBMIT THE LADDER"
    assert "ladder.sh" in a["command"]


def test_trained_but_unevaluated_ladder_is_caught(tmp_path):
    # The failure mode a bare queue dump hides: trains finished, evals died,
    # and nothing is running so it looks complete.
    _run(tmp_path, "ladder_mod23_d40m_std", step=1500, ckpt=True, ev=True)
    _run(tmp_path, "ladder_mod7_d40m_std", step=1500, ckpt=True, ev=False)
    a = status.next_action([], status.runs(tmp_path), [], tmp_path)
    assert a["action"] == "RESUBMIT LADDER EVALS"
    assert "ladder_mod7_d40m_std" in a["why"]


def test_fully_evaluated_ladder_means_rank_it(tmp_path):
    for m in (23, 11, 7, 5):
        _run(tmp_path, f"ladder_mod{m}_d40m_std", step=1500, ckpt=True, ev=True)
    a = status.next_action([], status.runs(tmp_path), [], tmp_path)
    assert a["action"] == "RANK THE LADDER"
    assert "--mode rank" in a["command"]


def test_corpora_report_verification_and_size(tmp_path):
    _corpus(tmp_path, "ladder-mod7", verified=True, mod=7)
    _corpus(tmp_path, "half-built", verified=False)
    c = {x["name"]: x for x in status.corpora(tmp_path)}
    assert c["ladder-mod7"]["verified"] and c["ladder-mod7"]["mod"] == 7
    assert not c["half-built"]["verified"]
    assert c["ladder-mod7"]["bytes"] > 0


def test_corrupt_manifest_is_not_reported_as_verified(tmp_path):
    d = tmp_path / "corpora" / "bad"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{not json")
    c = status.corpora(tmp_path)[0]
    assert not c["verified"]
    assert "error" in c


def test_runs_report_snapshots_for_the_step_ladder(tmp_path):
    _run(tmp_path, "main", step=31280, ckpt=True, snaps=5)
    r = status.runs(tmp_path)[0]
    assert r["n_snapshots"] == 5
    assert r["last_step"] == 31280


def test_missing_directories_are_safe(tmp_path):
    assert status.corpora(tmp_path) == []
    assert status.runs(tmp_path) == []
