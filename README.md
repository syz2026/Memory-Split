# MemorySplit — Schemas in Weights, Facts in the Organizer

Does a small LM trained with facts offloaded to an external knowledge
organizer learn better reasoning than a dense twin at the same parameter
and token budget? A controlled fact-load dose-response experiment
(LMLM-style loss-masked lookups, arXiv 2505.15962) at 160M/410M scale with
a 1B stretch pair. Deliverable: a measured reasoning delta or a defensible
null.

- **Design spec:** `docs/superpowers/specs/2026-07-17-memory-split-design.md`
- **Implementation plan:** `docs/superpowers/plans/2026-07-17-memory-split.md`
- **Research dossier:** `docs/superpowers/research/2026-07-17-memory-split/`
- **Google Colab KQA continuation:** `docs/COLAB.md` and
  `notebooks/kqa_memory_transfer_colab.ipynb`

## Layout

```
corpusgen/    seeded generators: biographies (the dose), iGSM-lite math,
              rule deduction, fact-use QA; emits dense + split renderings,
              the organizer KV table, and held-out eval sets
organizer/    exact-match (entity, relation) -> value store + query grammar
train/        tokenizer (GPT-2 BPE + 4 special tokens), GPT model,
              memmap+mask dataloader, AdamW trainer with checkpoint/resume
evals/        generative scorers with lookup interception, recall probes,
              bits-in-weights accounting, natural benchmarks, paired stats,
              dose-response figure
configs/      YAML per run: {scale} x {arm} x {load} x {seed}
scripts/      build_corpus.py, run_train.py, run_evals.py, analyze.py, smoke_test.py
cluster/      FarmShare (Slurm) scaffolding: sync, env, sbatch templates
tests/        pytest suite; `python -m pytest` runs offline in <1 min
```

## Quick start (local, macOS)

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv/bin/python
export PYTHONPATH=.
.venv/bin/python -m pytest tests -q          # offline unit tests
.venv/bin/python scripts/smoke_test.py       # end-to-end toy pilot (CPU/MPS)
```

The smoke test builds a toy corpus (500 entities), trains dense and split
toy models for a few hundred steps, and asserts the mechanism: split-arm
loss on masked fact values stays near-uniform while dense bio loss falls.
