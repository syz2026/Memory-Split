# Review swarm

Three frontier models reviewing the MemorySplit experiments and paper drafts on
an hourly cycle, against a shared contract, into one deduplicated work queue.

## What to read

| You want | Read |
|---|---|
| What needs fixing, right now | `FINDINGS.md` |
| What happened this hour | `reports/cycle-NNNN/SYNTHESIS.md` |
| What one model actually said | `reports/cycle-NNNN/<model>.md` |
| The rules reviewers work under | `CHARTER.md` |
| What each lane covers | `lanes/` |

`FINDINGS.md` is the authoritative artifact. The per-model reports are raw
input and are kept for audit, not for action.

## The reviewers and the rotation

| Model | Reviews |
|---|---|
| Opus 5 | rotating |
| GPT 5.6 Sol | rotating |
| Cursor Grok 4.5 | rotating |

| Lane | Surface |
|---|---|
| A | Estimand, inference, preregistration compliance |
| B | Code, measurement, silent substitution |
| C | Claims, artifacts, provenance |

Lanes rotate one position per cycle, so every model covers every lane across any
three consecutive hours. This is deliberate. A single model has consistent blind
spots, and rotating the same surface past three different ones inside three hours
is what turns a review into a swarm rather than three parallel opinions. It also
means a finding filed by one model gets independently re-examined by another two
hours later, which is where the `Confirmations` count in the ledger comes from.

`python3 review-swarm/cycle.py --peek` prints the current assignment without
advancing. Dropping `--peek` advances the cycle and creates the report directory.

## For a patching agent

Read the "How to use this as a patching agent" section at the top of
`FINDINGS.md`. The short version: work S0 first, run the finding's **Check**
before you patch, never delete a row, and set `DISPUTED` rather than silently
skipping something you think is wrong. Disputed findings get routed to a
different model on the next cycle.

Do not patch straight from a per-model report. Reports are unverified and
undeduplicated by design.

## Operating notes

**The loop is session-bound.** It runs as a background shell in the Cursor
session that started it. Closing the session stops it. Restarting is one command
and the state in `state.json` survives, so cycles resume at the right number.

**Cost is real.** Three frontier models doing a repository-wide review every
hour is a meaningful spend. If cycles are producing thin deltas, widen the
interval rather than trimming the lanes, because a shallow review at high
frequency is worse than a deep one at low frequency.

**Reviewers are read-only.** The charter forbids them from touching anything
outside `review-swarm/`, from writing to `outputs/`, from submitting cluster
jobs, and from changing git state. Patching is a separate role run by a separate
agent, deliberately, so that the thing proposing a change is never the thing
making it.

**Stopping.** Ask the orchestrating session to stop the swarm, or kill the loop
shell directly. `state.json` records where it left off.

## Why this exists

Four generations of this experiment ran and all four were withdrawn. Not one
failed on the hypothesis. Each failed on a defect that produced a plausible null
and passed every check that existed at the time: a loss reduction that
re-weighted the survivors, a data lane that silently became another lane, a
control that was never difficulty-matched, and a decoder that padded with the
document separator and attended over the pads. That last one cost sixteen days
and read 4.7% where the true number was 93.8%.

All four have the same shape. A quantity was computed correctly against an
invariant that was not the one the claim depended on. The counts matched, the
bytes aligned, the totals were exact, and nothing was watching the meaning.

The swarm exists to look for the fifth one.
