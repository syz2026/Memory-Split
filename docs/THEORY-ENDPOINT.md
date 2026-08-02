# Why iGSM is at floor, and what Stage A did not test

Derivations in `theory/endpoint.py`, assertions in
`tests/test_theory_endpoint.py`, measurements from
`outputs/pilot/stage_a_report.json`.

Stage A returned **STOP**, on the rationale that "the endpoint has no power to
discriminate anything, so a null from the matrix would be uninformative." The
floor is real. The rationale does not follow from what was run.

## Eliminating the candidates

The accuracy-by-operation-count profile distinguishes the explanations, because
each leaves a different fingerprint.

| explanation | prediction | verdict |
|---|---|---|
| capacity | needs table_bits > model capacity | **ruled out.** 9,572 bits against 40.56 M parameters |
| expressivity | needs the answer without intermediate steps | **ruled out.** eval is generative, `max_new=384`, prompt ends at `Reasoning:` |
| output format | unparseable generations | **ruled out.** deduction scores 55% parseable from the same checkpoint |
| compounding | high at op=1, decaying as p^op | **ruled out.** op=1 is at baseline |
| optimization | flat at baseline at every op **including op=1** | **consistent** |

The measured profile, `d40m_std`:

| op | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| acc | 5.4% | 2.9% | 7.2% | 5.0% |

Flat at the no-skill baseline everywhere, op=1 included. The model cannot
perform **one** modular addition.

A note on the baseline. The right reference is the **majority rate of 7.13%**,
not uniform chance of 4.35%, because always emitting the modal answer earns the
majority rate for free. Read against uniform, the op=3 cell at 8.5% looks like
3.8 sigma of signal; read against majority with the correct n≈388 it is 1.3
sigma of nothing. `OpProfile` uses two binomial standard errors at the measured
cell size rather than a fixed window, since at n≈380 the standard error near 5%
is 1.1 points and a flat 3-point tolerance means different things at different
cell sizes.

## The binding constraint is the step budget

Stage A ran 800 M tokens at 524,288 tokens per step:

**1,525 optimizer steps, 300 of them (19.7%) warmup.**

Modular arithmetic is slow for gradient descent. The grokking literature
(Power et al. 2022; Nanda et al. 2023) reports 10⁴–10⁶ steps on far smaller
models. Those are fixed-table settings and this one streams fresh problems,
which removes the memorisation phase but not the slowness. 1,525 steps is
about **7× short of the low end** before any of this project's specifics enter.

The corpus now building reaches **31,280 steps** with warmup at 1.0%, which is
inside that range. That is a real argument for the larger corpus, independent
of the capacity accounting in `docs/THEORY-CAPACITY.md`.

## Two defects in what Stage A actually ran

**The difficulty ladder was never built.** `stage_a_configs` documents itself
as "the difficulty cells live inside the corpus and are separated at evaluation
by meta tags, so a single run measures the whole ladder." That is not the case:
`build_corpus.py` takes a single `--mod` defaulting to 23, the pilotA manifest
records `mod: 23, op_band: [1,4]`, and `DIFFICULTY_CELLS` does not exist in
`analyze_pilot.py`. Both Stage A cells ran the same difficulty and varied only
architecture. The STOP therefore rests on one rung of a ladder that was
supposed to be descended until something became learnable.

**The learning-rate probe cannot answer the question it was launched for.** It
runs 1,500 steps at MOD=23 — the same budget and the same difficulty as the run
whose floor it is meant to explain. A one-factor probe identifies its factor
only when the others are slack, and here it holds the suspected binding
constraint fixed. It will read floor at every rate and invite the conclusion
that the rate is exonerated and the endpoint is dead.

It remains valid for choosing a learning rate by training loss, which is worth
having. At step 880 the loss ordering is already clean and unfavourable to
higher rates: 1.80, 1.82, 1.89, 1.97 for 1.5e-3 through 1.2e-2.
`probe_is_confounded` states the condition.

## What would settle it

Two cheap probes, neither requiring the large corpus.

**Descend the modulus** at the existing 1,500-step budget. One 800 M-token
corpus per rung, roughly 3.2 GB and a few minutes to build each.

| mod | chance | table bits | clears if op=1 exceeds |
|---:|---:|---:|---:|
| 23 | 0.043 | 9,572 | 0.143 |
| 11 | 0.091 | 1,674 | 0.191 |
| 7 | 0.143 | 550 | 0.243 |
| 5 | 0.200 | 232 | 0.300 |

If op=1 clears at MOD=7 or MOD=5, the operation is learnable and MOD=23 is
simply too hard for the budget. If it stays at baseline even at MOD=5 — a
232-bit table, five possible answers — then something is broken in how the
task is presented, not in how hard it is.

**Extend the steps** at the loss-optimal rate on the existing corpus, at 1.5k /
6k / 25k steps, watching op=1. This separates the step hypothesis directly and
prices the full run before committing to it.

## What this does and does not change

It does not overturn the floor: iGSM at 1,525 steps and MOD=23 is genuinely
unusable, and nothing here suggests otherwise. Responsiveness has to hold
before a crowding experiment means anything, and it does not hold here.

What changes is the inference drawn from it. "The endpoint has no power to
discriminate" is a claim about the endpoint. What was demonstrated is that
**this endpoint at this difficulty and this step budget** has no power, on a
ladder with one rung, with a step budget about 7× short of the literature and
a probe that cannot see the difference. The NO-GO paper should not be written
on that basis until the two probes above have run.
