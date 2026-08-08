"""Every number in the workshop paper must match the artifact it came from.

The paper has already carried wrong numbers into a draft four times: a
first-snapshot probe value presented as the run's result, a table that spliced
one run's row onto another's, a bits-per-entity figure computed against the
wrong entity count, and an abstract claiming the model reproduced fact text
"almost perfectly" when its own arithmetic put it level with a frequency
baseline. Each survived review because checking meant opening a JSON file on a
cluster nobody could reach.

The artifacts are now mirrored under `outputs/cluster-summaries/`, so the check
is mechanical. If a number moves in either the paper or the artifact, this
fails.

The paper is `docs/paper-workshop.tex`. There was briefly also a markdown
draft; it was retired once the two diverged, because a repository holding two
versions of one paper eventually cites the wrong one.
"""

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# In docs/ rather than a paper/ tree: `paper` is on the forbidden-directory list
# in test_repo_state.py, banned when the draft built on withdrawn numbers was
# deleted. docs/ also puts the source inside the withdrawn-number scan, which
# globs docs/*.tex.
PAPER = ROOT / "docs" / "paper-workshop.tex"
PDF = ROOT / "docs" / "paper-workshop.pdf"
RUN = ROOT / "outputs" / "cluster-summaries" / "stepladder_d40m_std"

pytestmark = pytest.mark.skipif(
    not RUN.exists(), reason="cluster artifacts not mirrored into the tree")


@pytest.fixture(scope="module")
def paper() -> str:
    """The paper with runs of whitespace flattened to single spaces.

    Every check here asks whether a phrase appears. Where LaTeX happens to wrap
    a line is irrelevant to that, and matching raw text made these tests fail
    whenever a sentence was rewrapped.
    """
    return " ".join(PAPER.read_text().split())


@pytest.fixture(scope="module")
def final() -> dict:
    return json.loads((RUN / "evals" / "step0031280.json").read_text())


@pytest.fixture(scope="module")
def ladder() -> dict:
    return json.loads((RUN / "step_ladder.json").read_text())


def _pct(x: float) -> str:
    return f"{x * 100:.1f}"


def test_per_operation_table_matches_the_scored_checkpoint(paper, final):
    """All eight per-difficulty cells and their sample sizes."""
    for band in ("igsm_by_op", "igsm_ood_by_op"):
        for op, cell in final[band].items():
            assert f"{_pct(cell['acc'])}\\%" in paper, (
                f"op{op}: artifact says {_pct(cell['acc'])}%, not in the paper")
            assert str(cell["n"]) in paper, f"op{op}: n={cell['n']} missing"


def test_headline_accuracy_and_baseline(paper, final):
    assert _pct(final["igsm_acc"]) == "96.0"
    assert "96.0\\%" in paper
    assert _pct(final["igsm_majority_rate"]) == "7.5"
    assert "7.5\\%" in paper


def test_deduction_and_gate_zero(paper, final):
    assert f"{final['deduction_overall']:.3f}" == "0.950"
    assert "0.950" in paper
    assert f"{final['loss_masked_values_final']:.4f}" == "1.9732"
    assert "1.9732" in paper


def test_final_recoverable_bits_is_not_the_first_snapshot(paper, final):
    assert f"{final['recoverable_bits_per_entity']:.1f}" == "-112.7"
    # -126 was the first snapshot and was once printed as the run's result.
    assert "126" not in paper.replace("2404.05405", "").replace("2407.20311", "")


def test_probe_positive_control_cells(paper):
    d = json.loads((RUN / "probe_control.json").read_text())
    assert d["probe_detects_knowledge"] is False
    for key, shown in (("training_trained", "67.83"),
                       ("training_unseen", "67.65"),
                       ("heldout_trained", "111.93"),
                       ("heldout_unseen", "112.10")):
        got = f"{abs(d['cells'][key]['bits_per_entity']):.2f}"
        assert got == shown, f"{key}: artifact {got}, paper {shown}"
        assert shown in paper, f"{key}: {shown} missing from the paper"


def test_memorization_control_is_the_load_bearing_null(paper):
    d = json.loads((RUN / "memorization_control.json").read_text())
    assert d["bindings_learned"] is False
    t = d["cells"]["trained"]["nats_per_value_token"]
    u = d["cells"]["unseen"]["nats_per_value_token"]
    assert f"{t:.4f}" == "1.9876" and f"{u:.4f}" == "1.9866"
    assert "1.9876" in paper and "1.9866" in paper
    assert f"{d['gap_nats_unseen_minus_trained']:.4f}" == "-0.0010"
    assert "0.0010" in paper
    assert f"{d['gap_z']:.2f}" == "-0.08"
    assert "0.08" in paper


def test_storage_instrument_has_a_passing_positive_control(paper):
    """Without this, Section 6 repeats the error the paper is about."""
    p = RUN / "memorization_positive_control.json"
    if not p.exists():
        pytest.fail("positive control artifact missing; the paper's claim that "
                    "the instrument can detect storage would be unbacked")
    d = json.loads(p.read_text())
    assert d["instrument_detects_storage"] is True
    assert f"{d['gap_nats_unseen_minus_trained']:.4f}" == "1.4484"
    assert "1.4484" in paper
    assert f"{d['gap_z']:.1f}" == "18.7"
    assert "18.7" in paper
    assert d["gap_nats_unseen_minus_trained"] > 4 * d["detection_floor_nats"]


def test_the_arithmetic_reconciliation_recomputes(paper, final):
    """Section 6 derives that the model sits level with a frequency baseline.

    This is the correction that took the abstract's "almost perfectly" away, so
    it is the last derivation that should be allowed to rot. Recompute it from
    committed constants rather than trusting the typed figures.
    """
    from corpusgen import factlane

    # Measured on the shipped generator, recorded in PREREGISTRATION.md section 4
    # beside TOKENS_PER_DOC. Not a named constant in code.
    value_token_fraction = 0.2489

    value_tokens = factlane.TOKENS_PER_DOC * value_token_fraction
    assert f"{value_tokens:.2f}" == "18.65"

    gate0 = final["loss_masked_values_final"]
    nll_bits = value_tokens * gate0 / math.log(2)
    assert f"{nll_bits:.2f}" == "53.08"
    assert "53.08" in paper

    assert f"{factlane.BITS_PER_ENTITY:.2f}" == "52.96"
    assert "52.96" in paper

    # The model is WORSE than the baseline, by a small margin. If this flips
    # sign the paper's central sentence is wrong again.
    margin = nll_bits - factlane.BITS_PER_ENTITY
    assert 0 < margin < 0.5, f"margin over baseline is {margin:+.2f}"
    assert f"{margin:.2f}" == "0.12"
    assert "0.12" in paper


def test_the_paper_does_not_claim_the_model_reproduces_facts_well(paper):
    """The overstatement that survived three review rounds.

    The model assigns 13.9% to the correct fact token and sits 0.12 bits worse
    than a frequency baseline. Any phrasing implying it reproduces factual
    content accurately contradicts the paper's own arithmetic.
    """
    low = paper.lower()
    for banned in ("almost perfectly", "reproduces the text of the facts",
                   "predicts fact values in its training documents well",
                   "fits fact text"):
        assert banned not in low, (
            f"the paper claims {banned!r}, which its own reconciliation in "
            f"Section 6 contradicts")


def test_the_new_instrument_reproduces_gate_zero(final):
    """Why the replacement is trusted. If these diverge, the argument goes."""
    d = json.loads((RUN / "memorization_control.json").read_text())
    trained = d["cells"]["trained"]["nats_per_value_token"]
    gate0 = final["loss_masked_values_final"]
    assert abs(trained - gate0) < 0.05, (
        f"memorization control {trained:.4f} no longer reproduces the training "
        f"loss {gate0:.4f}; the validity argument depends on this")


def test_capacity_ratio_reads_off_an_anchor_not_an_interpolation(paper):
    """The paper says the 2x figure needs no interpolation of prior work."""
    from theory import capacity as C

    low_exposures, low_bits = C.ALPHA_ANCHORS[0]
    assert low_exposures == 100, (
        "the run trains at 100 exposures; if that is no longer an anchor the "
        "paper may not say the ratio avoids interpolation")
    assert C.bits_per_param_at(100) == low_bits

    load = C.Load(1.0, 1_531_800, 100, 52.96, 40_560_000)
    assert round(load.ratio, 2) == 2.00
    doubled = C.bits_per_param_at(100, anchors=((100, low_bits * 2), (1000, 2.0)))
    assert abs(load.demand_bits_per_param / doubled - 1.0) < 0.01
    assert "double" in paper


def test_control_matching_table(paper):
    for shown in ("1.9598", "1.2712", "35.1", "1.5667", "1.7677", "1.8644"):
        assert shown in paper, f"control table figure {shown} missing"


def test_lane_substitution_figures(paper):
    for shown in ("3.75", "56.25"):
        assert shown in paper, f"data mixture figure {shown} missing"


def test_padding_defect_figures(paper):
    for shown in ("3/64", "60/64", "0/64", "63/64", "4.7", "93.8", "47\\%"):
        assert shown in paper, f"padding figure {shown} missing"


def test_citations_resolve_to_real_identifiers(paper):
    """Fabricated citations are the classic tell; pin the two we rely on."""
    assert "arXiv:2404.05405" in paper, "capacity citation missing"
    assert "Allen-Zhu" in paper and "Knowledge Capacity Scaling Laws" in paper
    assert "arXiv:2407.20311" in paper, "iGSM citation missing"
    assert "Grade-School Math" in paper


def test_latex_body_fits_four_pages():
    """Four pages of body. Appendix and references may spill.

    Venues exclude both from the page limit, so a bare page count is the wrong
    assertion: it would either forbid a legal fifth page or silently let body
    text run over. Instead require every numbered section to sit before page
    five.
    """
    if not PDF.exists():
        pytest.skip("PDF not built")
    pypdf = pytest.importorskip("pypdf")
    pages = [(p.extract_text() or "") for p in pypdf.PdfReader(str(PDF)).pages]
    assert len(pages) <= 5, f"{len(pages)} pages is over budget even with the appendix"

    assert "Limitations" in "".join(pages[:4]), (
        "the numbered body no longer fits four pages; Limitations has been "
        "pushed past the limit")
    for i, text in enumerate(pages[4:], start=5):
        head = text.strip()[:40]
        assert not re.match(r"^[1-7]\s", head), (
            f"page {i} opens a numbered section ({head!r}); only the appendix "
            f"and references may run past page four")
