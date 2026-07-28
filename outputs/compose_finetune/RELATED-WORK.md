# Related work — facts vs reasoning, and how it maps to our results

*Read closely (not just abstracts) and assessed for applicability to the
memory-split composition finetune. Verdicts: **STRONG** = essentially our setting;
**PARTIAL** = related mechanism, different regime; **BACKDROP** = context.*

The organizing question — *do you need facts to reason, and does storing facts help
or hurt reasoning?* — has a fairly clear literature answer: **facts are necessary but
not sufficient for reasoning; storing them costs finite capacity and tends to create
lookup shortcuts that pre-empt reasoning; externalizing them reliably preserves skills
and generalizes fact-access, but "freed capacity that then reasons" is largely
unproven.** Our results line up with the first three and are silent (so far) on the last.

---

## The two most on-point papers

### Allen-Zhu & Li, *Physics of LM 3.2: Knowledge Manipulation* (arXiv:2309.14402) — **STRONG**
Same synthetic-biography world we use (bioS: N people × 6 attributes), same "use stored
facts" question, same OOD-over-people eval design (their P_train/P_test = our
P_comp/P_held). Findings (their Results):
- Models **excel at retrieval** but **fail classification/comparison without CoT** —
  e.g. "born in an even month?" is at chance even with 25k QA samples and 100%-accurate
  month recall; comparison among 100 options stays ~random with **2.5M** samples
  (Results 3–5).
- **Inverse search ≈ 0%** regardless of scale/data unless the fact was pretrained in
  reverse (Results 7–9) — this *is* the reversal curse.
- **Better extraction ≠ better manipulation** (Result 5); **CoT-in-training doesn't
  fix non-CoT inference** (Result 4).

**Applies to us:** our two-hop composition *is* a knowledge-manipulation task, and the
reasoning **trace is the CoT** — which is exactly why our numbers are high (100%
in-context) while the **cold single-hop probe** and any no-trace variant crater. Their
result "manipulation needs explicit steps" predicts our cold-vs-in-context gap and the
whole reason the task is framed as a written trace. *Caveat:* their headline negatives
are for **non-CoT** manipulation; we operate in the CoT regime, so we see the "works
with CoT" side — consistent, not contradictory. Their inverse-search=0 also predicts our
model would fail "which person has employer X?" (untested, but expected).

### *Provable Benefits of In-Tool Learning* (arXiv:2508.20755) — **STRONG (nearly our exact experiment)**
Compares **in-weight** (memorize the value = our *dense*) vs **in-tool** (emit a lookup
query to an external DB = our *split*) on synthetic biographies. Findings:
- **Capacity ceiling:** params to memorize facts grow **linearly** with #facts
  (Thm 3.2); in-weight recall is bounded by size.
- **In-tool → grokking-style "memorization → rule learning":** once enough fact
  diversity is seen, the model discovers the query *format* and **generalizes to OOD
  facts, decoupled from #facts** (Fig 3).
- **In-weight finetuning degrades general skills** (HellaSwag drops, esp. small models /
  >10k facts) due to finite capacity + update interference; **in-tool preserves them
  almost perfectly** (Fig 5).
- **Structure helps memorization:** correlated facts need fewer params (α-experiment).

**Applies to us — this is the paper our finetune most directly instantiates:**
- Our split's **OOD-entity generalization (99.8%)** + **decoupling from stored values
  (store OFF = 0%)** = their in-tool "rule learning generalizes, recall decoupled from
  #facts."
- Our split's **no catastrophic forgetting** (single-hop 100%→99% across the compose
  finetune) = their "in-tool preserves general capabilities."
- Our split converging by **~step 47 (~25M tokens)** = their **grokking-style phase
  transition** to rule-based querying.
- Their prediction that **in-weight (dense) finetuning overwrites skills** is exactly
  what to check in our **dense** finetune (does dense forget the 79% single-hop facts?).
*Caveat:* their "rule" is single-hop querying; our composition adds **chaining**, and our
**3-hop failure** shows the learned rule is **depth-bounded** — a limit they don't test.

---

## Multi-hop / latent reasoning (our depth story)

### Yang et al. 2024, *Do LLMs Latently Perform Multi-Hop Reasoning?* (ACL) — **PARTIAL/STRONG**
Strong evidence for **hop-1** (~70% of the time, later layers increase bridge-entity
recall) but **weak for hop-2 / full traversal** — models often don't complete the chain
latently. **Maps to** our finding that hop-2 works *in-context* (with the written bridge)
but not cold, and that depth beyond the trained pattern breaks.

### Biran et al. 2024, *Hopping Too Late* (EMNLP) — **PARTIAL/STRONG**
Two-hop is **sequential**: bridge resolved in early layers, second hop in mid/upper
layers; the model can "**run out of layers**" for hop-2 (back-patching a late rep into
early layers fixes many failures). **Maps to** our `interp/` circuit (retrieve→move→
lookup, resolve L8–9) and to why deeper chaining is fragile. *Caveat:* they study
**latent** (no-CoT) multi-hop; our composition uses explicit steps, so our 3-hop failure
is better described as **template-boundedness** than "out of layers."

---

## Shortcuts vs reasoning (why memorizing can hurt)

### Berglund et al. 2023, *The Reversal Curse* (+ analyses aclanthology 2024.emnlp-main.754; NeurIPS 2024 theory) — **BACKDROP**
"A is B" ⇏ "B is A"; stored facts are **directional co-occurrence shortcuts**, traced to
weight asymmetry under next-token loss. **Relevance:** what looks like "knowledge" is
often a shortcut, not a relational substrate for reasoning; our split **offloads** values
(so it doesn't store these directional shortcuts in weights) — but its composition
**template is itself a shortcut** (hence 3-hop fails).

### *Reasoning or Retrieval?* (arXiv:2509.24156) — **PARTIAL**
Answers = **joint product of reasoning + memorized-retrieval**; SFT/distillation → the
model leans on **retrieval shortcuts**; **unlearning memorized answers forces reasoning
to dominate** and generalize. **Relevance:** memorizable answers actively **suppress**
the reasoning pathway — a direct statement of the "shortcut suppresses reasoning" claim,
and a recipe (remove the shortcut) for eliciting reasoning.

### Wang et al. 2024, *Grokked Transformers are Implicit Reasoners* (arXiv:2405.15071) — **PARTIAL**
Reasoning circuits can emerge **late** (grokking), **but composition specifically does
NOT systematically generalize** even after grokking (in-distribution yes, novel
structure no); comparison does. **Maps directly** to our result: great on new *entities*,
**0% on new depth** — the signature of a shortcut/template, not a systematic algorithm.

### Capacity: Physics 3.3 (arXiv:2404.05405) & "bits/param" — **BACKDROP**
~2 bits/param knowledge capacity; formalizes the finite budget that facts and other
functions compete for.

---

## The claim, grounded: "storing facts costs capacity and can create shortcuts that suppress reasoning"
- **Costs capacity:** In-Tool Thm 3.2 (linear param cost) + Physics 3.3 (~2 bits/param) +
  In-Tool Fig 5 (in-weight fact finetuning degrades general skills). Finite, shared budget.
- **Shortcuts suppress reasoning:** when the answer is memorizable, SGD prefers a lookup
  (lower loss faster); the shortcut explains the loss so no pressure builds a reasoning
  circuit (grokking's memorize-first; *Reasoning or Retrieval?* unlearning forces
  reasoning; reversal curse = the shortcut is non-relational).

## What the literature predicts for our next experiments
1. **n800k dense vs split (the decisive run):** In-Tool + capacity laws predict dense's
   in-weight recall **collapses** as facts exceed capacity while split (externalized)
   holds — the fact-availability win. This is the regime where the arms should separate
   (n50k shows no gap, as expected).
2. **Dense finetune should show forgetting** (In-Tool Fig 5) — watch dense's single-hop
   79% under compose finetuning; split showed none.
3. **Mixed-depth (2+3-hop) finetune:** grokking-composition results predict it learns
   3-hop as a new template but **fails 4-hop** (no systematic recursion). If it *does*
   generalize to 4-hop, that would be a notable positive against the literature.
4. **To elicit real reasoning, remove the shortcut** (*Reasoning or Retrieval?*): raise
   fact-load so memory can't fit, or unlearn memorized answers, then measure whether a
   reasoning procedure appears.

## Bottom line for the memory-split thesis
Our finetune results **replicate In-Tool Learning's core claims on this architecture**
(externalize → OOD-generalizing rule + no forgetting) and **confirm Physics 3.2's**
"manipulation needs explicit steps." They **do not** support the stronger "freed capacity
→ reasoning" claim, and the depth failure (with grokking-composition) suggests what's
learned is a **shortcut/template**, not a reasoner. The defensible, literature-backed
story is **fact-availability + skill-preservation + composable interface**, not
emergent reasoning.
