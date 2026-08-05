# Lane A — Estimand, inference, and preregistration compliance

You own the question of whether the statistical claims are entitled to be made.
Not whether the code runs, and not whether the artifact exists. Whether the
inference is licensed by the design.

## Primary reading

- `docs/PREREGISTRATION.md` — the frozen design: arms, gates, inference,
  GO/NO-GO thresholds
- `docs/AMENDMENT-2026-08-02.md` — seven defects found in the unrun pipeline,
  and which choices are preregistered open slots rather than post-hoc
- `docs/PAPER-MEASUREMENT.md` §4, §5, §7, §8 and `docs/PAPER-WORKSHOP.md`
  §3, §4, §6, §7
- `scripts/analyze_crowding.py`, `evals/` statistics modules,
  `tests/test_stats_crowding.py`, `tests/test_stats.py`
- `theory/capacity.py` and `tests/test_theory_capacity.py`

## The standing question, added after cycle 2

**Does this document agree with itself, not just with the code?**

In cycle 2 two models read `docs/PREREGISTRATION.md` §10a against
`ops/crowding/guard.py` in the same hour. One checked whether the code implements
what the document says and correctly recorded it clean. The other asked whether
the document's own numbers are mutually consistent and found that every campaign
ceiling was sized at a throughput the same section forbids the guard from using,
so the guard will refuse the project's own plan (MS-016).

Verifying that code matches a specification is the easier half and it is the half
that gets done. Ask of any frozen document: were all of its numbers derived under
the rules it states elsewhere in itself?

## What to attack

**Unit of independence.** The confirmatory analysis previously pooled 3 loads x
8 seeds as n=24 when the same eight seeds recur at every load. That specific
defect is catalogued. Check whether its siblings survive anywhere: any bootstrap,
any interval, any test whose resampling unit is finer than the unit that was
independently randomized. A gate stated at the wrong unit fires, which is worse
than a gate that does nothing.

**Ceiling and floor.** The endpoint now fails `capacity.responsive()` from the
ceiling side at 93.8% in-band. A measure pinned at either end cannot show a
treatment effect. Check every outcome variable the papers report for whether it
has room to move in the direction the design needs, and whether the text
acknowledges the side it is pinned on. Note that op 5–8 was never trained on, so
those columns measure length generalization and not the trained endpoint. Verify
the papers do not silently pool the two.

**Per-cell n.** `PAPER-WORKSHOP.md` §3 reports per-op cells with 8 to 17 items
and calls them provisional. `RESULTS-2026-08-04.md` reports the same cells.
Check whether any inference, gate decision, or forward-looking design choice is
resting on cells that small, and whether stated intervals reflect that n.

**Frozen thresholds versus observed numbers.** Every gate threshold must predate
the number that meets it. Where a threshold appears in a paper, trace it to the
preregistration or to a dated amendment. The 20% NLL tolerance and the
oracle-matched control landing "exactly on the tolerance boundary" at 4.9% and
20.0% deserves specific attention: check whether the 20.0% figure is a
coincidence, a construction, or a rounding artifact, and whether the text's
reading of it is entitled.

**The difficulty table's provenance.** `PAPER-WORKSHOP.md` §8 states the
difficulty table in §4 was frozen from an early checkpoint of the run later
analysed, and asserts "the gap between arms is unbiased but the absolute values
are not independent of the model". Check that claim. Is the gap actually
unbiased under that construction, or does using the same model to score both
arms induce a correlated error that does not cancel?

**Estimand drift.** The papers are now about measurement defects rather than the
crowding hypothesis. Check that no sentence quietly promotes a defect finding
into evidence about the hypothesis, and that §7 / §8 claims about "the first
premise failed here" are scoped to the one corpus and one exposure regime where
they were measured.

**The primary load is still unfrozen.** Confirm nothing in a live document
picks it, implies it, or reports an analysis that presupposes it.

## Specific open items flagged by the project itself

- Which operating point is the preregistered one. Two internally coherent
  designs exist: 5.737B tokens with 50/30/10/10 lane shares, versus 16.4B–21.3B
  with 70/12.3/4.3/13.4. Only the first is in `PREREGISTRATION.md` and the
  second is the one being run. Check what each live paper claims, and whether
  any paper cites a number from one design while describing the other.
- `PAPER-WORKSHOP.md` §2 states Sections 3 and 6 use the 16.4B corpus and
  Section 4 uses the 21.3B corpus. Verify every number in those sections against
  that assignment. A number from the wrong corpus is an S0.
