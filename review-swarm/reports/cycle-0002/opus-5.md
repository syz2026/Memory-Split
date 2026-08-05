# Cycle 2 — Lane B, code and silent substitution — Opus 5

**Verdict: every campaign ceiling in the frozen preregistration was sized at a
throughput the guard refuses to use, so the guard will refuse the project's own
plan. Stage B gets 1 of its 2 runs, Stage C 6 of 9, and the matrix 45 of 72 —
and the analyzer refuses an incomplete matrix, so those 1,444 GPU-hours would be
unanalysable.**

Cycle 1's lane B occupant was GPT 5.6 Sol. I verified MS-011 through MS-015
during cycle 1 synthesis before they landed, so I did not re-verify them here and
instead took new surface: the enforcement code, which nobody has read.

---

### B-01. The matrix ceiling was sized at a rate the guard refuses to use

- **Severity:** S1
- **Target:** `docs/PREREGISTRATION.md:288-299` and `:318-323`;
  `ops/crowding/guard.py:56,143-148,219-224,240-253`; `HANDOFF.md:307-313`
- **Claim under attack:** §10a sets `campaign:matrix = 1600` GPU-hours and states
  that "**Projections are made from measured throughput only** ... The guard
  therefore budgets at the measured `d40m` rate and treats any faster outturn as
  headroom returned." `HANDOFF.md` prices the matrix at 1,495 GPU-h for 72 runs.
- **Defect:** The two numbers come from different rates and nothing reconciles
  them. The 1,495 estimate is 20.76 h per run, which is the extrapolated
  `d40m_std` rate of about 277,000 tok/s that §10a itself says has never been
  observed. The guard projects at `MEASURED_TOK_S = 184_671`, giving **32.09 h**
  for a 21.3B-token run and **2,311 GPU-h** for the matrix against a 1,600
  ceiling.

  Because the soft stop refuses new claims once committed reaches 90% of the
  campaign cap, the run-by-run consequence is exact:

  ```
  ceiling from prereg: 1600.0   soft stop 0.9
  guard projection per 21.3B-token run at 184,671 tok/s: 32.09 GPU-h
  plan assumption (1495 GPU-h / 72 runs):                20.76 GPU-h
  72 runs at the guard rate: 2,311 GPU-h against a 1600 ceiling
  soft stop bites once committed >= 1440 GPU-h
  runs claimable before new work is refused: 45 of 72 (committed 1,444 GPU-h)
  runs that would never be claimed: 27
  ```

  `scripts/analyze_crowding.py::require_complete` refuses a matrix with a missing
  arm or a dropped seed, so the 45 authorised runs are not a partial result. They
  are 1,444 GPU-hours of unanalysable spend, which is precisely the outcome §10a's
  soft stop was written to prevent: "a ceiling hit at 95% completion leaves
  half-finished runs that are unanalysable and fully paid for."

  **This is not specific to the matrix. Every remaining campaign ceiling is
  undersized under the guard's own projection, and the next one due to run fails
  first:**

  ```
  campaign    runs     need  ceiling    soft   claimable
  stage_b        2       64       60      54    1/2   SHORT by 1
  stage_c        9      289      220     198    6/9   SHORT by 3
  matrix        72     2311     1600    1440   45/72  SHORT by 27
  ```

  Stage B is two runs and only one can be authorised. It is the immediate next
  action in both branches of the plan, so the first submission after this review
  would be refused by the project's own guard.

  Every component here is individually correct and individually documented. The
  conservative projection rule is right and its reasoning is good. The ceiling is
  frozen where it should be. What is missing is any check that the ceilings were
  derived under the projection rule the guard enforces, and the whole block was
  derived under the other one.

  There is a methodological consequence beyond the operational one. §10a is
  frozen and the ceilings can only move by amendment, so the fix is an amendment
  that raises resource ceilings after discovering the work does not fit. That is
  defensible here because the cause is a projection-rate mismatch identified
  before any campaign ran and not an overrun discovered mid-spend, but it needs
  to be filed that way, with the derivation shown, rather than as a quiet edit.
- **Why existing checks miss it:** `tests/test_guard.py` exercises the guard
  against synthetic ceilings and claim sequences, so it verifies the mechanism.
  Nothing instantiates the *real* preregistered ceiling against the *real*
  campaign shape and asks whether the campaign fits. The guard cannot detect this
  because from its side both numbers are inputs.
- **Falsifiable check:** the block above reproduces with
  `PYTHONPATH=. python3 -c "from ops.crowding.guard import *; c=Ceiling.from_prereg(); print(projected_hours(21_335_900_160), c.for_campaign('matrix'))"`.
  Add a test that, for each named campaign, multiplies its planned run count by
  `projected_hours(total_tokens)` and asserts the result is below
  `soft_stop_fraction * campaign_ceiling`. That converts a mismatch between the
  plan and the ceiling into a failing test rather than a refusal 45 runs in.
- **What would prove me wrong:** the matrix being submitted in fewer, larger
  claims than one per run, or a per-run token count materially below 21.3B, or an
  amendment raising `campaign:matrix` above about 2,570 (2,311 / 0.9).

---

### B-02. The release path never returns checkpoint bytes, so scratch only ever grows

- **Severity:** S2
- **Target:** `ops/crowding/guard.py:204-217,283-288,358-361,394-396`
- **Claim under attack:** `release` "gives scratch back", and `resident_bytes` is
  "scratch currently held: corpora claimed and not released, plus checkpoints for
  every claimed run."
- **Defect:** `release(corpus, checkpoint_bytes=0)` subtracts whatever
  `checkpoint_bytes` the ledger record carries, but the CLI is the documented
  path and it cannot supply one. The `release` subparser declares only
  `--campaign`, `--corpus` and `--ledger`, and `main` calls
  `guard.release(args.corpus)`, so every CLI release writes `checkpoint_bytes: 0`
  and the `ckpt` accumulator never decreases. Each claim adds
  `CHECKPOINT_BYTES * n_checkpoints` = 487 MB and nothing ever gives it back.

  Over 72 matrix runs that is about 35 GB of scratch the guard believes is
  resident and which does not exist. Against a 150 GB ceiling holding one 85 GB
  corpus at a time, the phantom reaches 120 GB of apparent occupancy by the end
  and will refuse a legitimate build before then.

  The direction is safe, since the guard refuses too much rather than too little.
  But `release`'s own docstring warns that "Deleting a corpus on disk without
  recording it here leaves the guard refusing submissions it should permit", and
  the CLI makes exactly that unavoidable for checkpoints.
- **Why existing checks miss it:** `release` is correct as a Python API and is
  presumably tested that way. No test drives the argparse path, so the gap between
  the function's signature and the CLI's flags is invisible.
- **Falsifiable check:** add `--checkpoint-bytes` to the `release` subparser and
  pass it through, or have `release` look up the claim record for that corpus and
  subtract the checkpoint bytes it recorded. Then assert that a
  claim-then-release cycle returns `resident_bytes()` to its starting value.
- **What would prove me wrong:** an operational path that calls `release` as a
  Python API with the byte count rather than through the CLI, in which case this
  is a CLI gap only.

---

## Cross-verification of cycle 1 lane B

MS-011 through MS-015 were verified by me during cycle 1 synthesis before they
were admitted, against the primary sources rather than against Sol's report:
`evals/storage.py:1-20` for the storage-versus-recoverable-bits refusal,
`ops/crowding/occupancy.py:88,107-167` for the relabelling and the verdict
strings, `corpusgen/randpos.py:94-151` for all three match-dropping paths,
`train/trainer.py:146-167` for the mean-of-microbatch-means, `evals/scorers.py:37-72`
for the discarded generations, and a full 12-minute suite run for the baseline
(433 passed, 1 intentional failure, confirming MS-015).

All five stand as written. MS-014 was raised from S2 to S1 at that time. I have
added no confirmation counts, because verifying a finding I admitted is not
independent and should not be recorded as though it were. Independent
confirmation of lane B belongs to whichever model draws it in cycle 3.

## Checked and found clean

- **The lane-substitution fix is genuinely fail-closed, and it is the best piece
  of defensive code in the repository.** `plan_lanes` measures fact-lane capacity
  by rendering a spread sample rather than trusting the 74.91 tokens/document
  constant, and refuses up front with `LaneUnderfilled`. `verify` independently
  fails closed on realised-versus-requested drift. `requested_shares` is stored
  separately from `lane_budgets` precisely because the reallocation mutates the
  latter, and the manifest comment says so. `SHARE_TOLERANCE = 0.01` is used
  consistently as a token slack in `plan_lanes` and an absolute share difference
  in `verify`, and the docstring explains why the two must refuse at the same
  point. Stage A's 46% deficit would fire both. The escape hatches
  (`--rehearsal`, `--sup-only`, `allow_share_drift`) are each narrow and each
  labels the resulting corpus as non-scientific.
- **`Ceiling.from_prereg` refuses rather than defaulting.** No ceiling block means
  an exception, not a fallback constant, and an unnamed campaign is refused rather
  than inheriting the programme total. The reasoning given (inventing a campaign
  name would otherwise buy 1,600 unlogged GPU-hours) is correct and the code
  implements it.
- **`committed_hours` counts claimed-but-unfinished work at projection and
  replaces it with measured cost once known.** That closes the window where an
  unbounded number of runs could be authorised before the first lands, which is
  the defect the independent-research standard names.
- **The divergence breaker fires on measured cost per run**, not on a
  completion-fraction forecast, so it is defined from the first completed run.
- **`nll_table.accumulate` right-pads with zeros and excludes pads from the
  tally** via `keep[:, 1:]`. Real tokens sit at positions 0..n-1 and causal
  attention excludes the right tail, so the marginal table is not contaminated by
  padding. This is the same argument Sol measured for `evals/storage.py` and it
  holds here for the same reason.

## Handed forward

B-01 is a preregistration-arithmetic finding as much as a code one and would sit
equally well in lane A. Whoever draws lane A in cycle 3 should check whether the
other campaign ceilings (`fc1` 40, `stage_a` 20, `ladder` 80, `stage_b` 60,
`stage_c` 220) fit their planned run counts under `projected_hours` at the
measured rate, since I only checked `matrix`.
