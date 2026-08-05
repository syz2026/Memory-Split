# Lane C — Claims, artifacts, and provenance

You own the question of whether every number in a live document is real, is the
number it is labelled as, and is the same number everywhere it appears.

This lane is mechanical and it is the one most likely to find an S0 on any given
cycle, because the repository has withdrawn a large fraction of its own numbers
in the last three days and the drafts were written across that boundary.

## Primary reading

- `docs/RETAINED-RESULTS.md` — the sole catalogue of citable numbers, including
  its two withdrawal lists at the end
- `outputs/RETAINED-SHA256SUMS` and the `outputs/` tree
- Every live draft: `docs/PAPER-MEASUREMENT.md`, `docs/PAPER-WORKSHOP.md`,
  `docs/PAPER-BRIEF.md`, `docs/PAPER-BRIEF.tex`, `docs/NO-GO-PAPER.md`,
  `docs/RESULTS-2026-08-04.md`, `docs/section-null-without-crowding.md`
- `README.md`, `HANDOFF.md`, `ops/crowding/RUNBOOK.md`
- `tests/test_repo_state.py`

## What to attack

**Resurrection.** `RETAINED-RESULTS.md` ends with two lists of withdrawn
figures: those withdrawn 2026-08-05 by the decoding defect or a missing harness,
and those withdrawn earlier for having no artifact at all. Every one of those
numbers is now forbidden in a live document. Grep for each. The known
intentional failure is `docs/section-null-without-crowding.md`. Any *other*
document carrying a withdrawn number is an S0.

Pay particular attention to the fact that the 2026-08-05 withdrawal removed
"every per-op and per-class breakdown in `RETAINED-RESULTS.md` §1", the 1B
held-out result, and the 0.8B gate evaluations, while `PAPER-WORKSHOP.md` and
`RESULTS-2026-08-04.md` report *new* per-op tables from corrected decoding.
Confirm that no draft mixes pre-fix and post-fix per-op numbers in one table or
one sentence.

**The same number in two places.** Build a cross-document table of every
numeric claim that appears more than once and diff it. Known values to trace,
non-exhaustively: 4.7% / 3 of 64, 93.8% / 60 of 64, 1.9598 nats, 1.2712 nats,
35.1%, 20.0%, 9.8%, 4.9%, 1.5667, 1.7677, 1.8644, 1.9732 nats, 25.03, −126
bits, −112.90, −112.29, −0.61 ± 0.68, 52.96-bit ceiling, 996,408 entities,
1,531,800 entities, 200 exposures, 100 exposures, 16.4B, 21.3B, 5.737B tokens,
31,280 steps, 40,695 steps, 1,564 steps, 19.6%, 0.9993, 1.000, 0.533, 0.783,
3.75%, 56.25%, 40.6M, 21.2M, 1.04–1.55 exposures.

A value that appears with two different definitions is as bad as one that
appears with two different values. `1.000` as an in-answer-space rate and
`1.000` as F/C are different quantities and must not be conflated.

**Artifact backing.** For each number in a live draft, find its row in
`RETAINED-RESULTS.md`, then find the artifact that row points at, then confirm
the file exists and is tracked. A number with a catalogue row but no file is an
S2. A number with neither is an S0.

**Placeholders and forward references.** `PAPER-WORKSHOP.md` §3 contains
`[PLACEHOLDER: final-checkpoint per-operation table ... from the scoring run
completing 2026-08-05]`. Today is on or after that date. Check whether the run
completed, whether the numbers landed, and whether any *other* text in the draft
already assumes the values the placeholder will carry. A sentence written
against numbers that do not exist yet is an S0 waiting to happen.

Find every other placeholder, TODO, TK, and forward-dated promise across the
drafts and list them with status.

**Supersession.** `RESULTS-2026-08-04.md` declares it supersedes the endpoint
conclusions in `docs/THEORY-ENDPOINT.md` and the STOP in
`outputs/pilot/stage_a_report.json`. Check that the superseded documents say so
at the top, or that nothing cites them as current. `NO-GO-PAPER.md` rested on
the endpoint being unusable and that premise is gone. Check what state that
draft is in and whether anything points at it as the deliverable.

**Internal consistency of the withdrawal itself.** `RETAINED-RESULTS.md` §8 is
titled for the evaluation defect and §1 is marked withdrawn. Verify the file
does not cite its own withdrawn sections elsewhere, and that the SHA256 manifest
matches the files present.

## Method

Grep and read. This lane rewards exhaustiveness over cleverness. A shell loop
over the numeric literals above, across `docs/` and `README.md` and `HANDOFF.md`,
is a good first pass and will usually produce the cycle's findings directly.

Report every mismatch you find with both locations and both values. Do not try
to decide which one is correct unless the artifact settles it.
