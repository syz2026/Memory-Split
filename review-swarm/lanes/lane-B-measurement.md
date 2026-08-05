# Lane B — Code, measurement, and silent substitution

You own the question of whether the code computes the quantity the paper says it
computes. This is the lane where all four withdrawn generations died. Assume
there is a fifth defect and that it currently passes every test in `tests/`.

## Primary reading

- `evals/generate.py` — the padding defect lived here; the fix decodes in groups
  of equal prompt length
- `evals/storage.py` — recoverable bits; pads on the right, claimed safe under
  causal attention
- `evals/` scorers and the frozen verdict
- `ops/crowding/build_corpus.py` — 438 lines changed; the lane-substitution
  defect lived here
- `corpusgen/randpos.py` — the matched control
- `ops/crowding/nll_table.py`, `occupancy.py`, `probe_control.py` — new,
  uncommitted, and therefore unreviewed
- `train/` loader, trainer, and the weighted loss
- The whole of `tests/`, read adversarially: what does each test actually assert
  versus what its name implies

## What to attack

**The fix for the padding defect.** It is the newest and most consequential
change in the repo and the entire paper rests on it. Verify by execution, not by
reading:

- Does `tests/test_generate.py` actually assert that batching never changes a
  generation, across mixed prompt lengths, or only for a case that happens to
  pass? Run it.
- The equal-length grouping removes padding for the *prompt*. What happens
  during generation, when sequences in a group diverge in length as some emit
  EOT earlier than others? Is there any point after the prefill at which a
  shorter sequence sits behind a pad?
- `evals/storage.py` right-pads and argues causal attention makes it safe.
  Check the position ids. Causal attention protects against attending *forward*,
  but if position indices are assigned over the padded tensor, a right-padded
  batch can still shift the positional encoding of nothing, while a *left*-padded
  one shifts everything. Confirm which the code does and that the argument in
  `RESULTS-2026-08-04.md` is the correct argument and not a correct conclusion
  from a wrong premise.
- Is there any other decode path in the repo, in `scripts/` or `ops/`, that did
  not receive the fix?

**Where else does a count-preserving substitution hide.** The lane-substitution
bug kept the token total exact while changing what the tokens were. Look for the
same move anywhere a budget, a share, a mass, or a count is conserved by
reallocation. Check that `build_corpus.py` now stores requested proportions
separately from the mutated budget, that the check fires on the difference, and
that a test exercises the failing case rather than only the passing one.

**Masks and loss.** The `reduction="mean"` defect re-weighted surviving targets
by `1/(1-f)`. Verify the current loss is summed over a denominator that is
identical across arms, and that a test would fail if someone changed it back.

**Determinism and seeding.** Any use of Python's `hash()` on strings is
salted per process and silently non-reproducible. Check for it. Check that
seeds are derived reproducibly, and that a rerun of any scoring path is
byte-identical.

**Uncommitted modules.** `nll_table.py`, `occupancy.py`, `probe_control.py`
and their tests are untracked. They produce numbers that appear in
`RETAINED-RESULTS.md` §9, §10 and in the papers. Read them as if they had never
been reviewed, because they have not been.

**Tests that assert less than their name.** For each test file, pick the tests
whose names make the strongest claims and check the body delivers it. A test
named for an invariant that only exercises a single hard-coded case is how
defect 2 survived.

## Method

Run things. You have shell access and the repo has a working venv path documented
in `HANDOFF.md`:

```
export PYTHONPATH=.
python3 -m pytest tests -q
```

Measured baseline as of cycle 1: **434 collected, 433 pass, 1 intentional fail**
(`test_no_document_resurrects_a_withdrawn_number`). If your run disagrees,
establish why before reporting anything else, because a changed baseline changes
the meaning of everything downstream.

Note that `HANDOFF.md` documents 392 pass and a `.venv/bin/python` that does not
exist in the working tree. Both are stale and are already filed as MS-015. Do not
re-file them, and do not use 392 as your sanity check.

Do not run anything that writes to `outputs/`, submits a job, or touches git
state. Scoring and corpus builds are expensive and some are running. If a check
needs a write, describe the check instead of running it.
