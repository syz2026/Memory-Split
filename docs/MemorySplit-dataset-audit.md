# MemorySplit v2 Dataset Audit

- **Audit date:** 2026-07-23
- **Base revision:** `feat/memorysplit-v2-integration` at `4ecff3a`
- **Current scientific status:** `incomplete`
- **Protected launch allowed:** `false`
- **Contract files:** `configs/reasoning-dataset-v2.json` and
  `configs/preregistration-v2.yaml`

## Status decision

No valid claim-bearing MemorySplit v2 experiment has finished. The correct
current status is `incomplete`, not “failed to reject,” because the required
corpus, sealed evaluation, diagnostic gate, protected cohort, and bound
provenance do not yet exist.

`invalid` takes precedence over `incomplete` when a measured protocol,
provenance, endpoint, or instrument failure coexists with missing evidence.
Accordingly, a result produced with the currently mismatched compiler and
evaluator would be `invalid`; it would not become negative evidence for the
hypothesis. No current output is promoted to a v2 measurement by this audit.

The canonical hypothesis is:

> At matched parameters, initialization, raw tokens, token order, optimizer
> schedule and inference budget, removing direct next-token loss from selected
> factual payloads improves OOD relational reasoning.

## Audited current dataset

The current nominal seven-lane compiler targets:

- 40% FineWeb-Edu;
- 20% Wikidata graph serialization;
- 10% synthetic graph exposure;
- 15% synthetic multi-hop reasoning;
- 7.5% Wikidata reasoning;
- 2.5% relational refinement; and
- 5% ARC/ConceptARC auxiliary material.

This is nominally 40% natural language, 30% graph exposure, 25% relational
traces, and 5% puzzles. Mixture percentages alone overstate the diversity and
verification quality of the current reasoning signal.

The update-aligned 20-token-per-parameter floors currently recorded are:

- 29M: 579,862,528 raw target tokens, of which 144,965,632 are nominally in
  the three relational lanes;
- 160M: 3,244,818,432 raw target tokens, of which 811,204,608 are nominally
  relational; and
- 360M: 7,120,879,616 raw target tokens, of which 1,780,219,904 are nominally
  relational.

These counts measure scheduled lane mass, not unique verified reasoning. They
must not be described as a reasoning-optimal corpus.

## Blocking findings

1. Current synthetic reasoning cycles through only six relation programs.
   Entity renaming does not make repeated canonical programs independent
   reasoning content.
2. Wikidata reasoning is one hop, so it does not test acquisition of reusable
   multi-step procedures.
3. Relational refinement has one refinement template, which makes template
   memorization a plausible shortcut.
4. There is no production proof verifier and no sealed OOD suite spanning IID,
   composition OOD, length OOD, and joint OOD.
5. The effective intervention uses a 50% hash route rather than proving at
   least 90% coverage of both distinct offloadable facts and train-only
   information burden.
6. Routed facts can reappear in supervised candidate and final-answer copies.
   The current masking ledger therefore does not prove semantic closure.
7. The random control can mask repeated filler and no-op material rather than
   a scientifically matched target burden.
8. The current loader can create duplicate target positions across batch
   boundaries, so exact consumed-token identity is not established.
9. The required six-run 29M diagnostic gate is absent: full-corpus Dense and
   Split90, no-ARC/ConceptARC Dense and Split90, and no-refinement Dense and
   Split90 have not completed.
10. The same-load smaller-Split/larger-Dense comparison required for a capacity
    substitution claim is absent.
11. Only a 131,072-token smoke corpus exists locally. Existing collaborator
    ZIPs contain code, builders, and launchers rather than the full protected
    corpus.
12. Existing cluster kits do not bind code, environment, sources, corpus,
    route, graph memory, evaluation, run, and checkpoint identities end to
    end. They are not an Illumina v2 release.

Do not scale the current compiler beyond 20 tokens per parameter. Additional
tokens would amplify semantic replay, filler/no-ops, and finite-template
memorization rather than repair these findings.

## Frozen v2 replacement

The fixed sprint is a 20-token-per-parameter, reasoning-maximized recipe. It is
not labeled “reasoning optimal”; that label remains reserved for a later
development-only 20/40/80/160 search.

The replacement mixture is exactly:

- 25% expanded FineWeb-Edu;
- 15% FineMath, consuming cross-deduplicated `finemath-4plus` before the
  cross-deduplicated `3plus` remainder;
- 20% complete Wikidata5M training graph;
- 10% synthetic graph;
- 15% solver-verified synthetic multi-hop reasoning;
- 7.5% solver-verified Wikidata path reasoning;
- 2.5% solver-verified relational refinement; and
- 5% objective auxiliary reasoning.

ARC/ConceptARC is capped at 0.25% of the full stream. Teacher-generated
chain-of-thought is capped at 0.5% of the full stream and requires independent
answer validation, trace validation, and contamination review. The recipe
guarantees at least 40% human-language tokens and at least 25% broad
general-language tokens.

The claim-bearing comparison is Dense versus Split90 over identical model
parameters, initialization, raw token bytes, token order, packing, optimizer
and schedule, optimizer-step count, inference budget, evaluation items, and
exact graph-memory bytes. Split90 must offload at least 90% of distinct
offloadable atomic facts and at least 90% of their train-only
information-weighted burden. Rules, operators, schemas, and proof procedures
remain internal.

The primary endpoint is counterfactual pair-and-proof accuracy. A pair receives
credit only when both twins have correct answers and verifier-accepted proofs.
Model-visible items and sealed gold are separate, and IID, composition OOD,
length OOD, and joint OOD releases are evaluation-only and structurally
disjoint from training.

## Confirmatory run freeze

The protected 360M cohort contains five precommitted Dense/Split90 pairs with
seeds 0–4. Seed 0 remains in the terminal cohort regardless of its effect and,
while it is the only completed pair, can be reported only as
`directional_only (1/5)`. Continuation may stop for validity or infrastructure
failure, never because of the observed treatment direction.

Each protected run uses 524,288 targets per optimizer update for 13,582
optimizer steps, or 7,120,879,616 raw target tokens. The seed-0 allocation is
seven A100 80 GB devices: three for Dense, three for Split90, and the seventh
for noninterfering evaluation or verification. A four-versus-three training
allocation is forbidden.

Inference uses paired training bundles as the independent unit, a
20,000-draw seed/world/counterfactual-pair hierarchical bootstrap, exact
paired sign-flip and one-sided sign procedures, Holm correction,
non-inferiority/equivalence tests, and fixed-checkpoint AULC. Five all-positive
pairs yield the minimum exact one-sided sign probability of 0.03125; three
same-sign pairs alone are at most `sign_consistent_only`.

## Protected-launch closure

All of the following gates are required and currently pending:

1. scientific contract;
2. route dose;
3. semantic mask closure;
4. proof verification;
5. OOD seal;
6. corpus identity;
7. paired training identity;
8. exact checkpoint resume;
9. evaluation validity; and
10. completion of all six 29M diagnostics.

Every future evidence binding is deliberately `null` and `unfrozen`: expanded
FineWeb snapshot, FineMath snapshot, reasoning corpus, route manifest, exact
graph memory, model-visible evaluation, sealed-gold evaluation, environment
lock, and run manifest. Those artifacts do not yet exist in protected form.
No digest is inferred from a path, copied from a different artifact, or
invented as a placeholder.

`protected_launch_allowed` remains `false` until every gate passes and every
required artifact is built, independently verified, and bound to its real
digest.
