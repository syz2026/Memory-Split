"""The resource ceiling has to be a gate, not a printout.

The failure this guards against already happened twice in this portfolio: a
campaign spent 3,790 calls against a 200-call ceiling that existed only in a
document, and this account was cancelled mid-battery at 261 GB of scratch
against a 150 GB budget written in a runbook. In both cases the number was
correct, known, and enforced by nothing.

So these tests do not check that the guard computes a projection. They check
that it *refuses*.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.crowding.guard import (  # noqa: E402
    BYTES_PER_TOKEN,
    MEASURED_TOK_S,
    Ceiling,
    CeilingExceeded,
    DivergenceHalt,
    ResourceGuard,
    corpus_bytes,
    projected_hours,
)

FC1_TOKENS = 21_335_900_160


@pytest.fixture
def ceiling():
    return Ceiling.from_prereg()


@pytest.fixture
def guard(ceiling, tmp_path):
    return ResourceGuard(ceiling, tmp_path / "ledger.jsonl", "fc1")


# ------------------------------------------------- the ceiling is the prereg's

def test_the_ceiling_is_read_from_the_frozen_preregistration(ceiling):
    """Not a constant in the module. If the two could differ, the code would
    enforce something the design never promised."""
    assert ceiling.scratch_bytes == 150_000_000_000
    assert ceiling.gpu_hours == 1600
    assert ceiling.soft_stop_fraction == 0.90
    assert ceiling.divergence_factor == 1.5
    assert ceiling.campaigns["fc1"] == 40


def test_a_preregistration_without_a_ceiling_block_is_refused(tmp_path):
    """There is no default. A default would be an unpreregistered ceiling."""
    p = tmp_path / "PREREG.md"
    p.write_text("# Preregistration\n\nNo ceiling here.\n")
    with pytest.raises(CeilingExceeded, match="no ```prereg-ceiling"):
        Ceiling.from_prereg(p)


def test_a_campaign_can_never_exceed_the_programme_total(ceiling):
    for name in ceiling.campaigns:
        assert ceiling.for_campaign(name) <= ceiling.gpu_hours


def test_an_unnamed_campaign_is_refused_rather_than_defaulted(ceiling):
    """Otherwise inventing a campaign name is a way to get the whole programme
    ceiling without writing anything down."""
    with pytest.raises(CeilingExceeded, match="no ceiling in the preregistration"):
        ceiling.for_campaign("a-campaign-nobody-preregistered")


def test_every_campaign_a_submission_path_uses_is_preregistered(ceiling):
    """The shell scripts name campaigns as string literals. If one names a
    campaign the prereg does not, the refusal happens on the cluster at
    submission time, which is the worst place to discover it."""
    import re as _re
    used = set()
    for sh in sorted((ROOT / "ops" / "crowding").glob("*.sh")):
        used |= set(_re.findall(r"guard_claim_cfg\s+(\w+)", sh.read_text()))
    assert used, "no submission path calls the guard at all"
    missing = sorted(used - set(ceiling.campaigns))
    assert not missing, f"campaigns used but never preregistered: {missing}"


# ------------------------------------------------------- projections are measured

def test_projections_use_measured_throughput_not_an_expected_speedup():
    """`d40m_std` at ~277,000 tok/s is 184,671 multiplied by an assumed 1.5x
    that was never observed. Budgeting at it understates fc1 by 10.7 hours."""
    assert MEASURED_TOK_S == 184_671
    measured = projected_hours(FC1_TOKENS)
    optimistic = projected_hours(FC1_TOKENS, tok_s=277_000)
    assert measured == pytest.approx(32.09, abs=0.05)
    assert optimistic == pytest.approx(21.40, abs=0.05)
    assert measured > optimistic, "the guard must budget the conservative figure"


def test_corpus_bytes_counts_the_stream_and_both_sidecars():
    assert corpus_bytes(1_000) == 1_000 * BYTES_PER_TOKEN
    assert corpus_bytes(FC1_TOKENS) == 85_343_600_640


# ------------------------------------------------------------------ it refuses

def test_the_gpu_hour_ceiling_binds_from_the_very_first_claim(ceiling, tmp_path):
    """A guard keyed to a completion fraction is undefined before anything
    completes, so the opening runs go unguarded. This one has no such window:
    the first claim over the ceiling is refused with an empty ledger."""
    g = ResourceGuard(ceiling, tmp_path / "l.jsonl", "fc1")
    assert g.committed_hours() == 0.0
    # fc1's ceiling is 40 GPU-h; four fc1-sized runs is ~128.
    with pytest.raises(CeilingExceeded, match="preregistered ceiling"):
        g.claim("too-big", FC1_TOKENS * 4)


def test_one_fc1_run_is_authorised_and_a_second_is_not(guard):
    """32.09 h fits inside 40. Two do not, and the refusal must name the
    numbers so the operator can act on it."""
    rec = guard.claim("fc1_a", FC1_TOKENS, corpus="high-e200-op8")
    assert rec["projected_hours"] == pytest.approx(32.09, abs=0.05)
    with pytest.raises(CeilingExceeded) as exc:
        guard.claim("fc1_b", FC1_TOKENS, corpus="high-e200-op8-2")
    assert "40" in str(exc.value)


def test_only_one_load_fits_at_the_current_operating_point(ceiling, tmp_path):
    """A finding, not just a bound. `ops/crowding/RUNBOOK.md` says two loads fit
    the 150 GB budget, but that was priced at 16.4B tokens and 65.6 GB per load.
    The operating point moved to 21.3B tokens, which is 85.3 GB, so **two no
    longer fit** -- 170.7 GB against 150. The runbook's "at most two resident"
    is stale and following it would reproduce the 261 GB cancellation.
    """
    g = ResourceGuard(ceiling, tmp_path / "l.jsonl", "matrix")
    g.claim("load_a", FC1_TOKENS, corpus="a")
    with pytest.raises(CeilingExceeded, match="scratch"):
        g.claim("load_b", FC1_TOKENS, corpus="b")


def test_releasing_a_corpus_returns_its_scratch(ceiling, tmp_path):
    """Build, train, delete, repeat. The guard has to believe the delete, or it
    refuses submissions it should permit and the operator learns to bypass it."""
    g = ResourceGuard(ceiling, tmp_path / "l.jsonl", "matrix")
    g.claim("load_a", FC1_TOKENS, corpus="a")
    g.release("a")
    g.claim("load_b", FC1_TOKENS, corpus="b")  # must not raise
    assert g.resident_bytes() < ceiling.scratch_bytes


# ------------------------------------------------------------------ soft stop

def test_the_soft_stop_holds_back_the_last_slice(ceiling, tmp_path):
    """At 90% of a campaign, new work stops being authorised while claimed work
    is still permitted to finish. A hard stop at 95% completion leaves
    half-finished runs that are unanalysable and fully paid for."""
    g = ResourceGuard(ceiling, tmp_path / "l.jsonl", "stage_c")
    cap = ceiling.for_campaign("stage_c")  # 220 GPU-h
    twenty_hours = int(20 * 3600 * MEASURED_TOK_S)
    assert g.may_start_new_work()

    for i in range(9):  # 180 h, which is 81.8% of 220
        g.claim(f"c{i}", twenty_hours)
    assert g.may_start_new_work(), "81.8% is below the soft stop"

    g.claim("c9", twenty_hours)  # 200 h, 90.9%
    assert g.committed_hours() == pytest.approx(200.0)
    assert g.committed_hours() < cap, "still 20 h under the hard ceiling"
    assert not g.may_start_new_work(), "but past the 90% soft stop"

    with pytest.raises(CeilingExceeded, match="soft stop"):
        g.claim("c10", twenty_hours)


# ------------------------------------------------------------ divergence breaker

def test_the_breaker_fires_on_measured_cost_not_on_a_forecast(guard):
    """The signal is that the cost model of the experiment is wrong, so it must
    key on an observation. A run projected at 32 h that takes 60 halts the
    campaign at the moment the measurement lands, not at the next submission.
    """
    guard.claim("fc1_a", FC1_TOKENS)
    with pytest.raises(DivergenceHalt, match="cost model"):
        guard.record_actual("fc1_a", 60.0)


def test_a_divergence_halt_is_durable_across_processes(ceiling, tmp_path):
    """The measurement is written before the exception is raised, so a halt
    survives the process that discovered it. Otherwise the next invocation
    starts clean and submits the very run the breaker was protecting."""
    led = tmp_path / "l.jsonl"
    g1 = ResourceGuard(ceiling, led, "fc1")
    g1.claim("fc1_a", FC1_TOKENS)
    with pytest.raises(DivergenceHalt):
        g1.record_actual("fc1_a", 60.0)

    g2 = ResourceGuard(ceiling, led, "fc1")  # fresh object, same ledger
    with pytest.raises(DivergenceHalt, match="cost model"):
        g2.claim("fc1_b", 1_000_000)


def test_a_run_inside_the_divergence_factor_does_not_halt(guard):
    guard.claim("fc1_a", FC1_TOKENS)
    guard.record_actual("fc1_a", 32.09 * 1.4)
    assert guard.status()["campaign"] == "fc1"


def test_an_overrun_is_counted_at_its_measured_cost_not_its_projection(guard):
    """Otherwise the ledger stays optimistic forever and the ceiling drifts."""
    guard.claim("fc1_a", FC1_TOKENS)
    before = guard.committed_hours()
    guard.record_actual("fc1_a", 36.0)
    assert guard.committed_hours() == pytest.approx(36.0)
    assert guard.committed_hours() > before


# ------------------------------------------------------------------- the CLI

def test_the_cli_exits_nonzero_when_it_refuses(tmp_path):
    """The shell submission paths rely on the exit code under `set -e`. If a
    refusal exited 0 the sbatch would run anyway."""
    out = subprocess.run(
        [sys.executable, "-m", "ops.crowding.guard", "claim",
         "--campaign", "fc1", "--run", "huge",
         "--tokens", str(FC1_TOKENS * 10),
         "--ledger", str(tmp_path / "l.jsonl")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 3, out.stdout + out.stderr
    assert "REFUSED" in out.stderr


def test_the_cli_authorises_a_run_that_fits_and_writes_the_ledger(tmp_path):
    ledger = tmp_path / "l.jsonl"
    out = subprocess.run(
        [sys.executable, "-m", "ops.crowding.guard", "claim",
         "--campaign", "fc1", "--run", "fc1_d40m_std_e200",
         "--tokens", str(FC1_TOKENS), "--corpus", "high-e200-op8",
         "--ledger", str(ledger)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "AUTHORISED" in out.stdout
    rec = json.loads(ledger.read_text().splitlines()[0])
    assert rec["run_id"] == "fc1_d40m_std_e200"
    assert rec["corpus_bytes"] == 85_343_600_640


# ------------------------------------------- every submission path is guarded

def test_no_submission_path_calls_sbatch_without_claiming_first():
    """A ledger only accounts for the work that goes through it. This turns
    "every submission is guarded" from a convention into something a diff
    violates visibly.

    Any shell script under ops/crowding that invokes sbatch must also invoke
    the guard. Scripts that only ever print a plan are exempt by not calling
    sbatch at all.
    """
    offenders = []
    for sh in sorted((ROOT / "ops" / "crowding").glob("*.sh")):
        text = sh.read_text()
        if "train.sbatch" not in text:
            continue  # build- and eval-only paths spend CPU, not the GPU budget
        if "guard_claim_cfg" not in text:
            offenders.append(sh.name)
    assert not offenders, (
        "these submit training to SLURM without claiming against the "
        f"preregistered ceiling first: {offenders}. Add a guard_claim_cfg "
        "before the sbatch, or the ceiling in docs/PREREGISTRATION.md §10a is "
        "decorative.")


def test_the_shell_helper_actually_reaches_this_module():
    """`guard_claim_cfg` is the only thing the submission paths call, so if it
    stopped invoking the guard every one of them would silently go unguarded
    while still passing the test above."""
    env = (ROOT / "cluster" / "config.env").read_text()
    assert "guard_claim_cfg()" in env, "the helper is not defined in config.env"
    body = env.split("guard_claim_cfg()", 1)[1]
    assert "ops.crowding.guard" in body.split("\n}", 1)[0], (
        "guard_claim_cfg does not invoke ops.crowding.guard")
