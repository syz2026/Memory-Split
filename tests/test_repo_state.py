"""The repository holds one experiment.

Everything superseded was deleted, not quarantined. These assertions exist so
it stays that way: the failure mode this project actually suffered was three
experiment generations coexisting with incompatible endpoints, and someone
reading the wrong one.

`outputs/` is exempt from the naming rules. Retained results keep the names
they were produced under, because renaming an artifact breaks its provenance.
What may be cited from there is fixed by docs/RETAINED-RESULTS.md.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Trees whose presence means a superseded line came back.
FORBIDDEN_DIRS = [
    "probe", "artifacts", "_preserved", "integrity", "_scrapped",
    "organizer", "paper", ".swarm", ".superpowers",
]

# Names that mark a versioned or superseded line. Checked on source paths.
FORBIDDEN_NAME = re.compile(
    r"(^|[/_-])(v\d+|legacy|relational|aws|b200|srgm|keyguess|cohort-tiny)([/_.-]|$)",
    re.IGNORECASE,
)

SOURCE_DIRS = ["corpusgen", "evals", "train", "scripts", "ops", "cluster", "tests"]


def _tracked(prefix=""):
    out = subprocess.run(
        ["git", "ls-files", prefix] if prefix else ["git", "ls-files"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [p for p in out if p]


def test_no_superseded_tree_survives():
    present = [d for d in FORBIDDEN_DIRS if (ROOT / d).exists()]
    assert not present, f"superseded trees came back: {present}"


def test_only_one_ops_directory():
    ops = ROOT / "ops"
    subdirs = sorted(p.name for p in ops.iterdir() if p.is_dir()) if ops.exists() else []
    assert subdirs == ["crowding"], f"expected only ops/crowding, found {subdirs}"


def test_no_versioned_or_superseded_source_paths():
    bad = []
    for d in SOURCE_DIRS:
        for p in _tracked(d):
            if FORBIDDEN_NAME.search(p):
                bad.append(p)
    assert not bad, f"versioned or superseded paths: {bad}"


def test_docs_holds_only_the_current_documents():
    docs = sorted(p.name for p in (ROOT / "docs").iterdir() if p.is_file())
    assert docs == [
        "AMENDMENT-2026-08-02.md",
        "GATE0-CEILING-IS-NOT-A-BOUND.md",
        "NO-GO-PAPER.md",
        "PAPER-MEASUREMENT.md",
        "PAPER-WORKSHOP.md",
        "POPQA-HELDOUT-KEY.md",
        "PREREGISTRATION.md",
        "RESULTS-2026-08-04.md",
        "RETAINED-RESULTS.md",
        "THEORY-CAPACITY.md",
        "THEORY-ENDPOINT.md",
        "section-null-without-crowding.md",
    ], docs


def test_retained_results_catalogue_exists_and_is_referenced():
    cat = (ROOT / "docs" / "RETAINED-RESULTS.md").read_text()
    assert "Nothing outside this file may be cited" in cat
    assert "RETAINED-RESULTS.md" in (ROOT / "README.md").read_text()


def test_every_retained_output_is_tracked():
    """The whole set existed only as untracked files; git could not have
    recovered any of it."""
    tracked = set(_tracked("outputs"))
    for must in [
        "outputs/cluster-summaries/evals/d160m_split_n200k_s0_gate/summary.json",
        "outputs/farmshare-160m-sweep/CHECKPOINTS-SHA256SUMS",
        "outputs/farmshare-160m-v3-s0/fact-exposure.json",
        "outputs/1b-heldout/evals/split90_step0015582.json",
        "outputs/RETAINED-SHA256SUMS",
    ]:
        assert must in tracked, f"{must} is not tracked"


def test_the_loss_bug_is_documented_where_it_will_be_read():
    """Every masked run in outputs/ predates the fix. If that caveat is not
    on the entry, the numbers get quoted back as arm contrasts."""
    cat = (ROOT / "docs" / "RETAINED-RESULTS.md").read_text()
    assert "reduction='mean'" in cat
    assert "not valid arm contrasts" in cat
    assert "reduction='mean'" in (ROOT / "README.md").read_text()


def test_readme_scopes_the_claim():
    readme = (ROOT / "README.md").read_text()
    assert "not identified" in readme
    assert "phi-3" in readme


def test_no_reference_to_a_deleted_module_survives():
    dead = ["evals.recall", "evals.natural", "evals.mechanism", "evals.keyguess",
            "corpusgen.build", "corpusgen.factqa", "corpusgen.fact_segment",
            "organizer.store"]
    bad = []
    for d in SOURCE_DIRS:
        for p in _tracked(d):
            if not p.endswith(".py"):
                continue
            text = (ROOT / p).read_text()
            for mod in dead:
                if f"import {mod}" in text or f"from {mod}" in text:
                    bad.append((p, mod))
    assert not bad, f"imports of deleted modules: {bad}"


# ---------------------------------------------- withdrawn numbers stay withdrawn

# Fingerprints of the ledgers RETAINED-RESULTS.md withdrew for lack of any
# committed artifact. Matched as co-occurring sets rather than lone figures, so
# an innocent "65.0" somewhere does not trip the guard.
WITHDRAWN_FINGERPRINTS = {
    "held-out deduction table at 160M / 3.2B":
        ("65.0", "63.0", "66.7", "69.8", "68.2", "62.7"),
    "bits-stored-per-entity ledger":
        ("33.1", "0.246", "0.202"),
    "four-way recognition probe":
        ("0.903", "0.26"),
    "participation ratios":
        ("6.7", "1.8", "2.6"),
    "49-to-196-exposure memorisation threshold":
        ("49 and 196",),
    # Withdrawn 2026-08-05 by the decoding defect. Unlike the five above these
    # have committed artifacts, so the temptation to cite them is much stronger
    # and the guard matters more, not less.
    "0.8B gate accuracy table (padded decoder)":
        ("0.0387", "0.3693", "0.3087"),
    "0.8B split-arm fresh-entity claim (padded decoder)":
        ("0.5813", "0.996"),
    "1B held-out arm accuracies (deleted scorer)":
        ("26.87", "32.71"),
    "1B held-out gemma reference (deleted scorer)":
        ("7.70", "12.11"),
    "PopQA key-generalization figures (deleted harness)":
        ("98.5", "2.5", "24.3"),
}

# Documents whose job is to record a withdrawal, or which are the primary
# artifact being withdrawn. They must be able to name the numbers.
FINGERPRINT_EXEMPT = {
    "RETAINED-RESULTS.md",    # the catalogue; listing them is its purpose
    "RESULTS-2026-08-04.md",  # the retraction; it quotes what it retracts
    "POPQA-HELDOUT-KEY.md",   # the primary record of the withdrawn experiment
}


def test_no_document_resurrects_a_withdrawn_number():
    """RETAINED-RESULTS.md exists so nobody re-derives these, and it has already
    failed twice: the deleted `main.tex` hardcoded the deduction table, on
    2026-08-02 a new draft reproduced four of the six withdrawn ledgers, and on
    2026-08-05 `PAPER-BRIEF` was found carrying the 0.8B gate table and the 1B
    held-out result one day after the decoding defect retracted both.

    A number without an artifact is not a number, and neither is a number whose
    artifact was produced by a decoder that attended over its own padding. If a
    document needs one of these, it must first be re-measured with the fixed
    decoder and catalogued in RETAINED-RESULTS.md.

    Typeset sources are scanned as well as prose, because the original offender
    was a `main.tex` and a .tex draft would otherwise sit outside the guard.
    """
    offences = []
    docs = sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / "docs").glob("*.tex"))
    for doc in docs:
        if doc.name in FINGERPRINT_EXEMPT:
            continue
        text = doc.read_text()
        for name, tokens in WITHDRAWN_FINGERPRINTS.items():
            if all(t in text for t in tokens):
                offences.append(f"{doc.name} reproduces the {name}")
    assert not offences, (
        "withdrawn figures are back in the documents:\n  "
        + "\n  ".join(offences)
        + "\n\nSee docs/RETAINED-RESULTS.md. Reproduce the measurement and "
          "catalogue it, or remove the claim."
    )
