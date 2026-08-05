# Cycle 1 synthesis — 2026-08-05T15:16Z

Branch `cleanup/core-measurements` at `efc338a`, 37 uncommitted files.
All three lanes complete. Every finding re-verified by the orchestrator before
landing in `FINDINGS.md`.

| Lane | Model | Filed | Landed |
|---|---|---|---|
| A, estimand and inference | Opus 5 | 4 | 3 (one rejected, one merged) |
| B, code and silent substitution | GPT 5.6 Sol | 4 | 5 (one split out) |
| C, claims and provenance | Cursor Grok 4.5 | 7 | 7 (one merged, one reclassified) |

**Fifteen open: six S0, six S1, two S2, one S3.**

## Verdict

The science is in better shape than the documents. Not one finding says a
measurement is wrong in a way that changes a conclusion. What they say is that
several live documents assert things the project has already withdrawn, and that
three measurements are computed against a definition adjacent to the one the text
claims. Given that the paper's entire subject is defects that survived checking,
these are the findings a reviewer would punish hardest.

Three deserve immediate attention:

**MS-011** is the charter's defect shape in its purest form and it is the worst
one found this cycle. `evals/storage.py` opens by refusing to call its metric
storage, naming the exact prior withdrawal caused by that conflation.
`occupancy.py` then assigns the value to a key called `stored_bits_per_param` and
issues "Premise 1 of Memory Split fails" off its shape. The test for it writes
synthetic values under the honest key and asserts the renamed verdicts fire, so
the test codifies the substitution. The positive control that would settle it
exists in `probe_control.py` and has never been run.

**MS-012** puts the central control table's row label in doubt. Three independent
paths in `randpos.build` drop the length or position match without failing
anything, including one where the comment claims position-nearest selection and
the code selects on NLL. A surrogate-table execution on 800 documents broke the
length histogram in 336 of them. The frozen table is not local, so this is
unverified rather than disproved, and it is the single cheapest high-value check
outstanding.

**MS-001** is the only finding that cannot be fixed by editing after the fact. The
preregistration declares itself frozen while the primary load, and therefore the
estimand, is named nowhere in it.

## What the rotation bought

MS-005 exists only because two models read the same table from different lanes.
Lane A attacked the arithmetic of the 20.0% oracle row and used the wrong
denominator. Lane C independently noticed that §7 and §10 report different
treatment means and filed it as a cosmetic S3. Neither is right alone. Merged and
re-verified, they show the §10 gap column follows from **no** single treatment
mean: rows one through three require 1.9584 and rows four and five require
1.9598. That is a reproducibility defect in three documents rather than a rounding
slip or a typo.

The lane A finding was rejected outright in synthesis. That is the mechanism
working and it is why reports are never patched from directly.

MS-012 also lands next to MS-005 and MS-006 on the same table from a third
direction. Three lanes converging on §10 from arithmetic, from semantics, and
from the placement code is the strongest signal in this cycle.

## Themes

**Documents lag the 2026-08-05 withdrawal.** MS-002, MS-003, MS-004, MS-008 and
MS-010 are one event: a large fraction of the project's numbers were withdrawn
three days ago and the drafts, the handoff, and the superseded theory notes have
not caught up. `PAPER-WORKSHOP.md` is consistently ahead of `PAPER-MEASUREMENT.md`,
so the long draft can adopt the workshop's wording in several places rather than
needing new prose.

**Historical defects have living siblings.** MS-013 is the `reduction="mean"`
defect that withdrew generation 3, surviving in the Gate 0 probe path because the
fix was applied to the training path. MS-012 is the lane-substitution shape, with
`frac_documents_length_matched` computed into a manifest field that no gate reads.
Both were found by looking for siblings rather than by reviewing generically.

**The paper claims a check it does not run.** MS-014. The appendix that is the
draft's practical contribution lists "save raw generations" as current practice,
and `score_items` discards them.

## Costs

MS-002's step-budget arithmetic is used to size the confirmatory matrix and is
wrong in both the budget and the factor. MS-009 means an outside replicator cannot
verify the post-fix headline numbers from this tree.

## Cheapest closures, in order

1. MS-002, MS-003, MS-004, MS-010, MS-015 — editing and banners, no compute.
2. MS-005 — recompute one table column against one stated denominator.
3. MS-012 — regenerate 800 documents against the frozen table. Highest value per
   minute of anything on this list.
4. MS-013, MS-014 — one regression test each.
5. MS-006 — one forward pass over a few hundred held-out fact documents.
6. MS-011 — needs `probe_control.py` on the final checkpoint, so it needs cluster
   access.

## Process notes

Both remote reviewers were blocked on the same thing: FarmShare is unreachable
from this checkout, so the frozen `high-e200` difficulty table, the production
checkpoints, and the state of the 2026-08-05 scoring job could not be inspected.
MS-008, MS-009, MS-011 and MS-012 are all partially blocked on cluster access and
several may close immediately once it returns.

The documented test baseline is stale (MS-015): 433 pass against a documented 392,
and no `.venv` in the tree. Lane B used pyenv 3.12.0 with pytest 8.4.2 and torch
2.12.0. The charter cites 392 as the sanity baseline and should be corrected
before cycle 2 so reviewers do not chase it.

## Verified clean, not to be re-examined in cycle 2

Amendment 2's circularity fix; the reduced-form estimand in §1; the §3 load
construction; `dose_response_signature`'s zero-floor fix; `responsive()` failing
from the ceiling side; F/C invariance under the op-band widening; hash-manifest
integrity as distinct from coverage; cross-document agreement on the retained core
numbers; the equal-length decode fix including divergence after EOT; right-padding
in `evals/storage.py` (measured at 1.38e-6 bits, and the stated reason is the
correct one); absence of a second decode path; absence of Python `hash()` in
source; the weighted training loss denominator.

Open question carried forward: whether §7.3's first gate clause is evaluated
per-seed on the matrix or on the run that instantiated the band. Not a defect
until the matrix runs.
