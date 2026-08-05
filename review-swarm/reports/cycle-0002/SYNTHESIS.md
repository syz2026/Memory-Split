# Cycle 2 synthesis — 2026-08-05T16:17Z

Branch `cleanup/core-measurements` at `efc338a`, unchanged since cycle 1.

| Lane | Model | Held findings filed in cycle 1 by | Outcome |
|---|---|---|---|
| A, estimand and inference | Grok 4.5 | Opus 5 | 3 confirmed, 1 refined, 3 new |
| B, code and substitution | Opus 5 | GPT 5.6 Sol | verified at synthesis, 2 new |
| C, claims and provenance | GPT 5.6 Sol | Grok 4.5 | 2 confirmed, 4 refined, 4 new |

**24 findings open: seven S0, eight S1, six S2, three S3.**

## The headline

Nothing changed in the repository this hour, so the cycle was redirected to
adversarial cross-verification. Every model was handed a lane whose findings a
different model had filed and told its primary job was to break them.

**Ten findings were attacked and none was retired.** Five stand exactly as filed.
Five were refined, and in three cases that meant a severity reduction. That is the
outcome you want from a second pass: the surviving set is smaller in claimed
severity and much better evidenced than the set cycle 1 produced.

## What cross-verification actually changed

The refinements matter more than the confirmations, because each one is a place
where cycle 1 overclaimed and a fresh model caught it.

- **MS-006** was too wide. I filed it saying the papers misdescribe the 1.9598 and
  1.2712 figures as "mean masked-span NLL". Grok 4.5 found that preregistration §2
  *defines* that quantity using the frozen token table, so the label is entitled.
  What survives is the narrower and still real charge that three places promote
  the figures to "learnable signal removed", which the marginal table cannot
  support.
- **MS-002** carried an arithmetic charge I added, that "generous by roughly a
  factor of three" was wrong under every reading. GPT 5.6 Sol found the charitable
  reading, 40,695 / 12,512 = 3.25 against the last displayed curve point, and the
  charge is withdrawn. The wrong-corpus S0 underneath it stands.
- **MS-004** drops from S0 to S2. `HANDOFF.md` is a dated 2026-08-03 snapshot and
  the reversal is dated 2026-08-04, so it went stale rather than resurrecting
  anything.
- **MS-008** drops to S3 on a second look, including at my own reasoning for
  raising it.
- **MS-010** loses most of its scope. Only the missing banner on
  `THEORY-ENDPOINT.md` survives.

That last one produced the best single judgement of the cycle. Sol removed
`outputs/pilot/stage_a_report.json` from the target list on the grounds that it is
a raw measurement artifact whose recorded `"decision": "STOP"` should stay
byte-faithful, and that supersession belongs in a banner on the consuming document
rather than in a rewrite of the measurement. A patching agent working from cycle
1's version of that finding would have edited a stored result. The ledger now says
explicitly not to.

## The two new S0s

Both are in the confirmatory path, which cycle 1 never reached, and both are the
same kind of defect: the preregistration promises something the code does not
implement.

**MS-018.** §7 ends "All gates use confidence bounds, not point estimates."
`check_gates` implements five gates and one of them takes a bound. §7's sixth
gate, per-arm unmasked-token NLL, does not exist at all, so the gate that would
expose the entropy asymmetry between arms cannot fire, and that asymmetry is the
subject of Section 4 of the paper. §7.3's two monotone clauses exist only in the
pilot path.

**MS-019.** §5 freezes the primary scoring band as the contiguous run of ops where
SUP accuracy falls inside [0.17, 0.85], and excludes ops outside it because
averaging a ceiling op shrinks the effect toward zero. The analyzer never
instantiates it. `igsm_by_op` is produced by `run_evals.py` and consumed by
`ladder.py`, and `analyze_crowding.py` never reads it. This is sharpest against the
2026-08-05 decision to widen the corpus to op [1,8] because op 1–4 sat at 93.8%: an
aggregate over [1,8] can sit inside the interval while the estimand still averages
the ceiling ops the rule exists to exclude.

MS-019 also settles the question cycle 1 carried forward, in the worse direction.
§7.3 is not circular against §5. It is unimplemented.

## Two couplings a patching agent must respect

**MS-023 changes MS-016's Stage B row.** The runbook builds `high-e200` at 21.3B
tokens and then points Stage B and C at `corpora/high` at 16.4B. At the stale token
count the guard projects 24.67 h per run and Stage B's two runs fit the 60 h
ceiling; at the intended count they do not. So MS-016's Stage B shortfall is
contingent on resolving MS-023 first. The matrix fails under both.

**MS-021 gates MS-011.** One says the occupancy ladder measures the wrong
quantity, the other that its decision thresholds are unfrozen and live in an
untracked file. Neither closes alone.

## An instructive near-miss

Grok 4.5 examined §10a against `guard.py` this cycle and recorded it clean, which
is correct: the parsing does match the document. I filed MS-016 an hour earlier
because the document's numbers do not fit each other. Two models read the same file
in the same cycle and only one asked whether the numbers were mutually consistent
rather than whether the code matched them. Worth adding to the lane A brief as a
standing question: *does this document agree with itself, not just with the code?*

## Blocked, unchanged from cycle 1

FarmShare is unreachable, so the frozen difficulty table, the production
checkpoints, and the state of the 2026-08-05 scoring job remain uninspectable.
MS-008, MS-009, MS-011, MS-012 and MS-019 are all partially gated on it. MS-008
should not be pressed until 2026-08-06, since a promise to complete on 2026-08-05
is not yet overdue.

## Resolved and closed out

`docs/PAPER-BRIEF.md` and `.tex` do not exist, are untracked, and have no status
entry. Cycle 1 reported them absent, which conflicted with a stale git-status
snapshot the orchestrator was working from. Confirmed absent at `efc338a`. Not a
finding.

## For cycle 3

The rotation returns each model to a third lane, so lane B gets its first
independent cross-verification (MS-011 through MS-017, currently all at one
confirmation, verified only by the orchestrator who admitted them). That is the
weakest evidence in the ledger and should be cycle 3's priority.

If the repository is still unchanged at cycle 3, the marginal value of a fourth
pass over the same surface is low. Consider either widening the interval or
switching the swarm to verifying patches once a patching agent starts work.
