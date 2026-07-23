# P3 — OOD two-fact composition run

A new training+eval endpoint that replaces the broken reasoning tasks (mod-23 iGSM
never cleared chance; deduction showed no split-vs-dense gap) with the **one place
theory still predicts a split win**: out-of-distribution two-hop composition. A dense
model must grok a hard parametric two-hop; an explicit lookup makes both facts
co-present, turning it into two one-hop lookups + a join (Grokked Transformers
2405.15071; Compositionality Gap 2210.03350). See `notes/p3-composition-build-plan.md`.

## The task

`What is the {attribute} of {person}'s {mentor|advisor}?` — the bridge person is never
named. Hop 1: person → bridge (functional relation). Hop 2: bridge → attribute.

- **Dense** memorises the composition (all tokens graded).
- **Split** emits two loss-masked organizer lookups; hop 2's query `"{bridge}, {attr}"`
  must **copy hop 1's retrieved bridge name** — the copy-chain is the capability tested.

Two disjoint populations with closed bridge edges (Karmim 2606.09338):
`P_comp` (composed in training) and `P_held` (atomic facts only, never composed → OOD).

## Tonight = the positive control (kill-gate), not H1

Confirm the task **clears chance in-distribution** (the check iGSM skipped) and read the
dense OOD curve. Only a PASS greenlights the seeded 3-arm twin.

## Run (A100 80GB)

```bash
export PYTHONPATH=. TIKTOKEN_CACHE_DIR=.tiktoken_cache

# 1. Build corpus (~min; aborts if any integrity check fails)
python scripts/build_compose.py --out data/compose_v1 \
    --n-entities 10000 --held-frac 0.2 --total-tokens 600_000_000 --n-eval 1000

# 2. Positive control: dense arm (~1-2 h on A100)
python scripts/run_train.py --config configs/compose_dense.yaml

# 3. Eval + figures + summary
python scripts/run_compose_eval.py --run outputs/compose_dense_s0 --data data/compose_v1
#   -> outputs/compose_dense_s0/compose_eval/{summary.md, results.json,
#      ood_vs_step.png, accuracy_by_testset.png}

# 4. (Stretch, if control passes) split arm — MUST use the token-weighted trainer
python scripts/run_train.py --config configs/compose_split.yaml --trainer v2
python scripts/run_compose_eval.py --run outputs/compose_split_s0 --data data/compose_v1 --arm split
```

Colab: `notebooks/colab_compose.ipynb` runs all of the above with inline figures.

## Decision gate (in `summary.md`)

- **PASS** — in-dist two-hop ≥ ~30% and single-hop access ≥ ~85% → greenlight the
  seeded 3-arm twin (dense-closed / dense-open / split), distractor sweep, 3-hop.
- **KILL/FIX** — in-dist near chance → task mis-scaled (fewer relations/attrs or more
  composed exposures) before any twin compute.

The dense OOD curve is itself a result: near-floor = room a lookup could fill; rising =
dense groks a shortcut (measure, don't assume — 2510.26745).

## What's new in the repo

| Path | Purpose |
|---|---|
| `corpusgen/compose.py` | bridge relations, two-hop dense/split docs, eval items |
| `scripts/build_compose.py` | corpus builder (bins + organizer + evals + `report.json`) |
| `evals/compose_eval.py` | two-hop scorer: per-hop lookup acc + fact-access conditioning |
| `scripts/run_compose_eval.py` | eval driver: OOD-vs-step curve, figures, `summary.md` |
| `configs/compose_{dense,split}.yaml` | A100-sized training configs |
| `notebooks/colab_compose.ipynb` | one-click Colab run |
| `tests/test_compose.py` | 10 offline tests |

## Notes on rigor (from the audit)

- Answers scored by **exact match** (`evals.scorers`), not the substring containment in
  `evals/recall.py`.
- Split arm trains with **`--trainer v2`** (token-weighted grad accumulation) to avoid the
  arm-asymmetric bias v1 introduces on masked tokens.
- OOD accuracy is reported **conditioned on both single hops being individually accessible**,
  so an OOD miss localises to composition, not fact access. Per-hop lookup accuracy is logged
  separately (two lookups at p give ~p² end-to-end).
