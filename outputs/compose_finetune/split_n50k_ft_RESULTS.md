# Compose finetune — split arm (base `n50k`) — results

**Date:** 2026-07-23
**Run:** `split_n50k_ft` (step 940) · arm = **split** · store = **on** · device = CUDA
**Base (init-from):** `split_n50k_step0006100.pt` (fact-storage experiment, ~3.2 B tokens)
**Finetune corpus:** two-hop composition, 10k entities (P_comp 8,000 / P_held 2,000),
seed 1234, ~600 M tokens, mix bio 20% / bridge 13% / compose 67%, `d160m` (162 M).
**Eval:** held-out sets from the corpus builder (`eval/*.jsonl` + `organizer.jsonl`),
exact-match on text after `Answer:`; answer chance ≈ **0.54%**.

## Headline
- **In-distribution two-hop (P_comp): 100.0%** — positive control: the task is learnable.
- **OOD two-hop (P_held): 99.8%** (conditioned on fact access: 99.8%; access rate 99.1%).
- **Verdict: GREENLIGHT** — task learnable, fact access clean, OOD is a real
  generalization signal.

For reference, the study's **from-scratch** split anchor was **99.2%** OOD — this
**finetuned** split matches/edges it, i.e. finetuning the fact-storage base onto
composition transfers the addressing+copy skill to never-composed entities.

## Results table
| test set | n | answer acc | hop-1 key | hop-2 key | both keys |
|---|---:|---:|---:|---:|---:|
| in-dist (P_comp) | 1000 | 100.0% | 100.0% | 100.0% | 100.0% |
| OOD (P_held) | 1000 | 99.8% | 99.9% | 99.8% | 99.8% |
| OOD, conditioned on fact access | 991 | 99.8% | — | — | — |

**Single-hop fact access**
| probe | acc | n |
|---|---:|---:|
| comp-hop1 | 99.7% | 989 |
| comp-hop2 | 99.0% | 980 |
| held-hop1 | 99.7% | 918 |
| held-hop2 | 99.5% | 958 |

## Why this is genuine composition (not a shortcut)
The split arm leaves a **checkable trace**: on OOD items it emits the correct
**hop-1 lookup key 99.9%** and the correct **hop-2 key 99.8%**, with **both keys
correct 99.8%** — essentially equal to the answer accuracy. So the score comes from
*actually writing the two correct lookups and copying the bridge into hop-2*, not
from a person→answer shortcut. (This is the finetuned analogue of the paper's 99.3%
both-keys-correct for from-scratch split.)

## OOD-vs-step curve
| step | in-dist | OOD | OOD cond. |
|---:|---:|---:|---:|
| 940 | 100.0% | 99.8% | 99.8% |

*(single point from `--snapshots last`; full curve from `--snapshots all` pending.)*

## Caveats / interpretation
- **This is the split arm only.** The H1 question (does split *beat* dense?) needs
  the paired **dense** finetune (`dense_n50k`, closed-book) and ideally a
  **dense-open** arm (dense given the store at eval). Run and compare OOD.
- **No split "win" expected at n50k.** At 50k entities each fact is seen ~100+ times,
  so the dense arm memorizes well and should also score high OOD — matching the
  study's "no arm difference at this fact load." Split's advantage is predicted at
  **higher fact load (n800k)**, where dense memory collapses; that pair is the
  decisive run.
- **Single seed.** Direction, not a proven effect; add seeds for error bars.
- **Eval store = on.** OOD accuracy reflects the split system (weights + organizer);
  with the store off, split recall of values is ~0 by construction.

## Provenance
- Finetune: `compose-finetune/finetune.py --trainer v2` (token-weighted grad accum),
  `micro_batch_size` lowered for GPU memory (accum compensates → identical training),
  `compile: false`.
- Eval: the repo's **original** `scripts/run_compose_eval.py` (matches the corpus's
  `organizer.jsonl` + `eval/` format), `--arm split --store on`.
- Raw outputs on Colab: `{OUT_DIR}/eval_aligned/{results.json, summary.md, *.png,
  rows_final.jsonl}`; best checkpoint copied to `MyDrive/ms/best/`.
- **Not** produced by the reconstructed `build_compose.py`/`run_compose_eval.py` in
  `compose-finetune/` — the corpus and eval used the clone's original scripts.
