# Held-Out Key Generalization — Fix Results (Copy-Constrained Decoding)

2026-07-21. Local CPU experiment, single seed (seed bundle 0). Follow-up to
`2026-07-20-heldout-key-generalization-results.md` (the 2.5% name-half
failure). The lost harness was recreated from that doc's protocol and is now
committed (`corpusgen/realfact.py`, `evals/keyguess.py`, `evals/constrain.py`,
`scripts/run_keyguess_local.py`); reproduce with
`scripts/run_keyguess_local.py --stage all` against the COMMITTED input
snapshot `data/realfacts/popqa_clean.jsonl` (SHA-256 pinned in Caveats; do
not refetch — upstream drifts). Replication seeds run via `--seed N`
(FarmShare: `cluster/slurm/keyguess_cpu.sbatch`); policy tables regenerate
via `scripts/analyze_keyguess_policy.py`.

## Question

Does making the copy route structural fix the measured failure — the model
memorizing entity names into keys instead of copying them from the prompt?
Two candidate fixes, isolated and combined:

- **Inference fix:** copy-constrained query decoding — during
  `<|db_start|>…<|db_retrieve|>`, the name half may only extend a span of the
  live prompt (token trie with boundary-punctuation/possessive variants), the
  relation half comes from the closed 16-relation grammar. Non-copyable names
  become unemittable.
- **Training fix:** copy-dominance data — counterfactual name substitution on
  ~50% of real-fact traces (name swapped consistently across question + key
  within relation, per-exposure permutations) + 2,400 fresh-name flood docs
  (each name appears exactly once).

## What we did

Protocol as on 07-20, recreated: PopQA raw 14,267 → cleaned 13,063 → capped
3,000; per-relation floor split **2,394 seen / 606 held-out** (07-20 snapshot:
2,399/601 — upstream dataset drift, documented in `data_manifest.json`);
6 exposures per seen fact as single-hop Question/Reasoning/Answer traces with
values loss-masked; shared synthetic base (2,700 docs; 300 bios entities x 6
exposures + 900 factqa docs); toy 4L/256d ~29M, ctx 192, 800 steps, CPU;
organizer holds all 3,000 facts at eval. Corpus A ≈ 1.07M tokens; corpus C
(substitution + flood) ≈ 1.21M tokens; masks verified byte-exact around
`<|db_retrieve|>` spans in review.

Four arms; B/D differ from A/C only at decode time:

| Arm | Training corpus | Decoding |
|---|---|---|
| A | baseline (as 07-20) | free argmax |
| B | A's checkpoint | copy-constrained |
| C | + substitution + flood | free argmax |
| D | C's checkpoint | copy-constrained |

Gold-key emittability through the span trie (hard gate for B/D): **804/806 =
99.75%** — 2 structural misses (11-word subjects beyond the n-gram cap), so
span extraction caps the constrained-arm ceiling at 99.75%, not a model limit.
Both trainings kept the mechanism intact (masked-value CE ≈ 10 vs general
loss ≈ 0.7 at step 800: fact values were never learned into weights).

## Results — emitted-key decomposition (held-out, n = 606, Wilson 95%)

| Arm | Full key | Name-half | Relation-half | Answer | No-lookup | Wrong-in-context name |
|---|---|---|---|---|---|---|
| A | 0.0 [0.0, 0.6] | 0.0 [0.0, 0.6] | 96.0 [94.2, 97.3] | 2.1 [1.3, 3.6] | 1.0 | 5.4 |
| B | **24.3 [21.0, 27.8]** | 24.4 [21.2, 28.0] | 97.2 [95.6, 98.2] | **26.2 [22.9, 29.9]** | 1.0 | 74.6 |
| C | 0.2 [0.0, 0.9] | 0.2 [0.0, 0.9] | 94.2 [92.1, 95.8] | 2.1 [1.3, 3.6] | 2.1 | 4.0 |
| D | 23.6 [20.4, 27.1] | 24.6 [21.3, 28.2] | 94.1 [91.9, 95.7] | 25.7 [22.4, 29.4] | 2.1 | 73.3 |

Seen split (n = 200): A 5.0% full key, B 63.0%, C 4.0%, **D 71.5%** (all
relation-half ≥ 93.5%).

## Results — the governance view (selective shipping policies)

Two deployable policies, recomputed by `scripts/analyze_keyguess_policy.py`
(persisted in `policy_analysis.json`). **Matching rule:** a shipped answer is
correct iff `normalize(store-returned value)` is exactly in the item's
normalized `possible_answers` set — strict returned-value equality, never a
substring, never the whole continuation. (The emitted-key table above uses
continuation-level answer accuracy, a different, looser metric.)

P0 = ship on store hit. P1 = ship on hit AND emitted name == gold subject
(oracle stand-in for one mention-similarity verification vote).

| Arm | P0 coverage | P0 precision | P0 silent-wrong | Wrong-referent keys among hits | P1 coverage | P1 precision |
|---|---|---|---|---|---|---|
| A | 209/606 = 34.5% | 3/209 = **1.4%** | 206/606 = **34.0%** | 209/209 | 0% | — |
| B | 154/606 = 25.4% | 147/154 = **95.5%** | 7/606 = **1.2%** | 7 | 147/606 = 24.3% | 147/147 = **100%** |
| C | 213/606 = 35.1% | 7/213 = 3.3% | 206/606 = 34.0% | 212/213 | 0.2% | 1/1 |
| D | 149/606 = 24.6% | 143/149 = **96.0%** | 6/606 = **1.0%** | 6 | 143/606 = 23.6% | 143/143 = **100%** |

The baseline is not merely unhelpful — it is actively dangerous: every one of
its 209 held-out store hits is a memorized key for the WRONG referent (the 3
"correct" shipments are coincidences where the wrong referent shares the gold
value), so a naive splice pipeline ships silently wrong values on a third of
all queries. The copy constraint converts that into a selective system, per
arm [I, arithmetic on the measured counts]: within arm B, silent error falls
**34.0% → 1.2% (~29x)** at 25.4% coverage and 95.5% precision; within arm D,
**34.0% → 1.0% (~34x)** at 24.6% coverage and 96.0% precision. Stacking the
name-echo vote (P0 → P1 within the same arm) removes ALL residual
silent-wrongs — 100% precision — at ~1.1pp coverage cost (B: 154→147 shipped;
D: 149→143). The residual P0 wrong-referent hits (7 and 6) are exactly the
valid-but-wrong-key class the proposal's mention-similarity + discriminator
verification targets; the P0→P1 delta is the measured value of one such vote
at this scale.

## Reading

1. **Arm A reproduces the failure signature qualitatively** [I]: name-half
   0.0 [0.0, 0.6] here vs 2.5 [1.5, 4.1] on 07-20; relation-half 96.0 vs
   98.5 (different PopQA snapshot, seeds, and batch schedule — the numbers
   are not expected to be identical). The signature — relation transfers at
   ceiling, name-copy at chance-or-below — is unambiguous in both.
2. **The inference-side constraint is the fix that matters at this scale**
   [I from the measured arm contrast]: full-key 0.0% → 24.3% and end-to-end
   answer 2.1% → 26.2% from the SAME checkpoint — for a quarter of items the
   failure was emission, not knowledge. The training-side fix alone moved
   nothing held-out (C: 1/606 = 0.2% vs A: 0/606): at 29M/800 steps,
   substitution data does not teach unconstrained copying.
3. **Seen-split interaction, directional only** [I, single seed,
   exploratory]: with the constraint on, the substitution-trained checkpoint
   ranks spans better where knowledge exists (D 143/200 = 71.5% vs B 126/200
   = 63.0% seen full-key). Whether substitution "improves" ranking is
   deferred to replication seeds 1–4.
4. **The bottleneck relocated, as designed** [M for the rate, I for the
   diagnosis]: with unemittable junk removed, 74.6% of items fail by picking
   the WRONG in-context span ("The, author"; "Question: Who, author") — the
   model cannot rank candidate spans it never learned to score. We read this
   as a capability gap at 29M, not an architecture gap: the 07-20 doc records
   the project's 160M model reaching 100% on held-out synthetic names, and
   span ranking (pointer loss over spans) is exactly what the full Tier-S
   trains.
5. **Go-gate verdict, honest:** the pre-registered go threshold (verified
   end-to-end ≥60%, Wilson LB ≥55%) is **NOT met at 29M** — 26.2 [22.9,
   29.9]. What IS established [M counts, I framing]: the structural lift
   (0 → 24.3 full-key from the same weights; within-arm silent error down
   ~29–34x under P0) and the verification story (P1 removes all residual
   silent-wrongs at ~1pp coverage cost, both constrained arms). The
   kill-lever does not fire either (it targets Tier-C prefix accuracy, not
   Tier-S). Next rung per the experiment ladder: 160M with the
   pointer-ranking loss and seeds 1–4 (FarmShare CPU jobs packaged and
   handed off).

## Caveats

Single seed at toy scale (seeds 1–4 dispatched to a collaborator; directional
until they land). Policy precision is strict normalized equality of the
store-returned value against PopQA possible_answers — unlisted aliases may
undercount true precision; the looser continuation-level answer metric is
reported separately in the emitted-key table and never used for governance
claims. The 2 emittability misses are counted as failures in all rates.
Single-hop extraction traces only; the organizer is exact-match (no
fuzzy/alias resolution).

**Evidence (committed with this doc):** `data/keyguess_local/`
`summary.json`, `results_{A..D}.json`, `records_{A..D}.jsonl`,
`policy_analysis.json`, `data_manifest.json`, `emittability.json`,
`eval.console.log`, `runs/{a,c}/log.jsonl`, `seen.jsonl`, `heldout.jsonl`,
`eval_items.jsonl`, `organizer_real.jsonl`, plus the frozen input snapshot
`data/realfacts/popqa_clean.jsonl` (SHA-256 pinned below). The fetch script
pulls an UNPINNED upstream dataset that is known to drift — reproduce from
the committed snapshot; refetch only to extend.

`popqa_clean.jsonl` SHA-256: `d167a4dbed20d7bb3277a1b34f882b35655f0d8862700cc451ad0a2d85dd2fd7`
