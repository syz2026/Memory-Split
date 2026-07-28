# Local eval battery — finetuned split (`split_n50k_ft`, step 940)

**Date:** 2026-07-23 · run on this Mac (MPS), `outputs/compose_finetune/local_eval.py`
**Model:** `step0000940.pt` (split arm, finetuned from `split_n50k` base on 2-hop compose)
**Store/graph:** the real `organizer.jsonl` (80,000 keys) from the training corpus;
entity names + bio attributes from `generate_records(seed=1234)` — **verified to match**
the organizer (200/200 names present). Sample n = 20 entities (CPU-modest; directional).

## Method (spec)
Everything is driven by the **real organizer** so the store + bridge graph exactly
match training. Generation uses the repo's `generate_batch_with_stats` (the
`<|db_start|>…<|db_retrieve|>…<|db_end|>` decode + organizer interception). Scoring
is substring match of the gold value in the generated text.

| battery | prompt | store | scores |
|---|---|---|---|
| **A** single-hop retention | `Reasoning: {name}'s {attr} is` | ON and OFF | did it keep the lookup skill? |
| **B** 2-hop | `What is the {attr} of {name}'s {rel}? Reasoning:` | ON | trained skill intact? |
| **C** 3-hop (never trained) | `... of {name}'s {rel1}'s {rel2}?` | ON | does depth generalize? |
| **D** free-gen | one 2-hop prompt | ON | eyeball the trace |

## Results — base vs finetuned (same organizer, n=20 entities)
| battery | **split BASE** (step 6100, pre-finetune) | **split FINETUNED** (step 940) |
|---|---|---|
| A single-hop, store **ON** | **100.0%** (0 miss) | **99.0%** (0 miss) |
| A single-hop, store **OFF** | 0.0% | 0.0% |
| B **2-hop** (store on) | **0.0%** (n=200) | **100.0%** (n=200) |
| C 3-hop (store on) | 0.0% (≈no lookups emitted) | 0.0% (emits exactly **2** lookups/item; drops middle relation) |
| D free-gen on a 2-hop Q | **degenerates**: *"The employer of the employer is the employer of…"* | correct 2-hop trace |

## What finetuning did (the before/after decomposition)
1. **Single-hop lookup was pre-existing and is retained — no forgetting.** Base
   100% ON → finetuned 99% ON (noise); OFF = 0% both. The base already generalizes
   single-hop lookup to the compose corpus's **new** entities (100% via name-copy),
   so finetuning **inherited** the addressing skill rather than teaching it.
2. **Finetuning's entire contribution = 2-hop chaining (0% → 100%).** The base can
   emit one lookup but **cannot chain two** (0% 2-hop; its free-gen doesn't even
   parse the two-hop question — it loops "employer of the employer…"). The finetune
   added exactly one capability: **copy hop-1's retrieved bridge into hop-2's query.**
3. **Recursion was NOT added.** 3-hop is 0% before and after; the finetuned model
   applies a **fixed depth-2 template** and drops any extra relation.

**One-line:** *lookup = pre-existing & retained; 2-hop chaining = the learned skill;
variable-depth recursion = not learned.*

## Interpretation

**1. It did NOT forget fact lookup — retention is perfect.**
Store ON 99.0% with **0 misses**, store OFF **0.0%**. The finetuned split still
emits correct lookup keys and reads values from the store, and it holds **no** values
in its weights (OFF = 0). Finetuning on composition cost nothing on the single-hop
addressing skill.

**2. The 2-hop skill is fully intact locally (100%)** — consistent with the 99.8%
OOD from the Colab eval.

**3. Depth does NOT generalize — the model is template-bound to 2 hops.** ⭐
On a 3-hop question it scores 0%, and the failure mode is precise: it emits **exactly
two lookups** (not three) and **silently drops the middle relation**. Example, for
*"major of Brethis's **mentor's advisor**"*:
```
Brethis's mentor is → Baelen Danin Dalewell.  Their major is → Computational Architecture.
So the answer is Computational Architecture.        (WRONG: that's mentor's major, advisor skipped)
```
It pattern-matches the question to its learned **depth-2 template** (`X's <rel> is B.
Their <attr> is V.`), consumes the *first* relation, ignores the second, and answers
the 2-hop sub-query — confidently. This is **not** "it tried to chain and lost
accuracy per hop"; it has **no variable-depth mechanism** at all.

### The headline distinction
The finetuned split **generalizes across entities but not across depth**:
- **Entity generalization: yes** — 99.8% OOD on never-composed *people* (Colab) and
  100% here. The addressing/copy skill transfers to new entities.
- **Depth generalization: no** — 0% on 3-hop; it collapses any query to its trained
  2-hop template.

So the "composition skill" learned here is **a fixed 2-step routine**, not a
recursive "look up, then recurse on the result" algorithm. That's an important
scoping of the earlier GREENLIGHT: strong OOD-entity generalization ≠ general
multi-hop reasoning.

## Caveats
- **n = 20 entities** (CPU), single seed, store-on only — directional but the effects
  are saturating (99–100% and 0%), so not borderline.
- **Base compared** (`split_n50k_step0006100.pt`): single-hop 100%→99% (no
  forgetting), 2-hop 0%→100% (the added skill), 3-hop 0%→0% (recursion not added).
- **3-hop uses the same surface template** with an extra `'s {rel2}`; the model was
  never trained on that surface form, so part of the failure is surface-form OOD as
  well as depth OOD. A cleaner test would train a few mixed-depth examples and re-test.
- Other reasoning tasks (iGSM/deduction) not testable locally (no eval data on this Mac).

## Discussion — what these results do and don't imply

> **Related work:** see [`RELATED-WORK.md`](RELATED-WORK.md) — our findings largely
> **replicate** *In-Tool Learning* (arXiv:2508.20755; in-weight vs lookup on synthetic
> bios → OOD-generalizing rule + no forgetting) and *Physics of LM 3.2* (arXiv:2309.14402;
> manipulating stored facts needs explicit CoT steps), and the 3-hop failure matches the
> grokking-composition result that composition doesn't systematically generalize.

### Does this support "freeing weights from fact-recall gave the split arm capacity to reason"?
**Too much of a stretch on this evidence.** Four reasons:
1. **No contrast measured.** That claim is inherently *split vs dense on reasoning*.
   We don't yet have the dense finetune, so we can't compare.
2. **n50k is the wrong regime.** At 50k entities there is no memory pressure — the
   dense base already recalls 79% of these facts, so dense will very likely also learn
   2-hop and the arms won't separate. The freed-capacity effect is predicted only at
   high fact load (n800k), where dense memory collapses.
3. **The task is lookup-chaining, not reasoning — and split *externalizes* it.** In
   the split arm the retrieval is done by the store; the model only emits keys and
   copies. So split does *less* in-weight computation, not more. Composition-via-store
   is closer to the opposite of "freed weights doing reasoning."
4. **"Avoided catastrophic forgetting" ≠ "freed capacity for reasoning."** No-forgetting
   shows the fact representation is stable/modular; it says nothing about spare capacity
   being spent on reasoning. The earlier mechanism probe was explicit that split doesn't
   rebuild dense's fact-neurons ("freed") but **"what the freed capacity does is still
   open."** These results don't close that.

The defensible positive is weaker: composition slotted onto the split base **cheaply,
fast (converged ~step 47 ≈ 25M tokens), and without forgetting** ⇒ the split
representation is unusually **composable / modular**. "Composable interface" is a much
safer claim than "freed capacity for reasoning."

### "If it can learn this skill, can it learn others?"
Weaker evidence than it feels: **it couldn't extend the *same* skill by one hop**
(2-hop learned, 3-hop 0% — collapses to the trained template). So "learns X" did not
transfer even to X's nearest neighbor. What it shows is **plasticity** (can acquire a
narrow, shallow behavior via finetuning, quickly, no forgetting) — and this one was
easy because it rode a **pre-existing lookup module** (the new content was a copy op).
A skill requiring genuinely new computation is a different, untested ask.

### "Does the base have latent reasoning capacity, just undertrained?"
Open, but the current evidence leans **skeptical for systematic reasoning at this scale**:
- **For the hypothesis:** grokking is real — transformers can acquire implicit multi-hop
  reasoning late, after loss saturates (Wang et al. 2024, *Grokked Transformers are
  Implicit Reasoners*). "At chance now" ≠ "incapable." The substrate isn't empty (facts
  memorized; chaining learnable cheaply).
- **Against (specific):** the **3-hop failure is a template/shortcut signature, not an
  undertraining one.** That same literature finds **composition does NOT systematically
  generalize even after grokking** (in-distribution yes, novel structure no) — exactly
  our pattern (great on new *entities*, 0% on new *depth*). So more training would likely
  **sharpen the template, not build a general algorithm.** Also 162M is small — a real
  capacity ceiling for hard reasoning.
- **Read:** likely latent capacity for *shortcut-shaped, shallow* skills (elicitable via
  finetuning, as we did); the depth failure argues what forms are **templates, not
  systematic algorithms** — i.e. "capable of memorizing another shortcut," not "a reasoner
  waiting to be unlocked."

### The decisive, cheap experiment
**Mixed-depth finetune:** train on 2- *and* 3-hop, then test **4-hop** (held-out depth).
- Learns 3-hop **and** generalizes to 4-hop ⇒ a variable-depth (recursive) mechanism ⇒
  "capacity, just needed the right training" (supports the optimistic view).
- Learns 3-hop but **fails 4-hop** ⇒ each depth memorized as a separate template ⇒
  capacity for shortcuts, not recursion (the predicted outcome).
Complementary: train far longer on the stalled reasoning task (watch for grokking-style
late emergence); add explicit CoT scaffolding (works with CoT but not without ⇒ capacity
present, elicitation was the gap); scale the model (separates capacity from training).

### What externalization actually buys (the paper's narrower, supportable thesis)
Not "freed capacity for reasoning" but **fact-availability**: externalizing facts keeps
them accessible even when they wouldn't fit in weights, so fact-dependent downstream
tasks keep working when a memorizing model would fail. That is testable as dense-ft vs
split-ft two-hop **at n800k** (predicted: split holds, dense collapses).

## Next steps
1. **Base before/after** for A/B (quantify any forgetting delta) — needs the base snapshot.
2. **Mixed-depth finetune** (2- and 3-hop) → re-run C to see if recursion is learnable
   at all, or whether the architecture defaults to fixed-depth templates.
3. **Dense comparison:** run the same battery on the dense finetune (closed-book) — does
   dense also collapse 3-hop, and does it forget single-hop facts under finetuning?

## Provenance
`outputs/compose_finetune/local_eval.py` (organizer-driven; read-only forward passes).
Raw: `local_eval_split_ft.json`.
