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
- **v2 source staging:** `docs/MEMORYSPLIT-V2-SOURCE-STAGING.md`
- **v2 production corpus contract:** `sources/memorysplit-v2/README.md`

## Layout

```
corpusgen/    seeded generators: biographies (the dose), iGSM-lite math,
              rule deduction, fact-use QA; emits dense + split renderings,
              the organizer KV table, and held-out eval sets
organizer/    exact-match (entity, relation) -> value store + query grammar
train/        tokenizer (GPT-2 BPE + 4 special tokens), GPT model,
              memmap+mask dataloader, AdamW trainer with checkpoint/resume
vendor/       tracked GPT-2 tiktoken cache assets for offline execution
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

Tokenizer initialization verifies and reads the tracked files in
`vendor/tiktoken/`; it does not populate a cache from the network.

The smoke test builds a toy corpus (500 entities), trains dense and split
toy models for a few hundred steps, and asserts the mechanism: split-arm
loss on masked fact values stays near-uniform while dense bio loss falls.

## Cluster (FarmShare)

```bash
bash cluster/connect.sh <sunetid>            # warm SSH control socket (Duo)
bash cluster/sync_push.sh                    # rsync repo to /scratch/users/$USER/memorysplit
ssh <sunetid>@rice-04.farmshare.stanford.edu
cd /scratch/users/$USER/memorysplit
bash cluster/setup_env.sh                    # venv + torch cu13 + deps
sbatch cluster/slurm/data_prep.sbatch        # FineWeb-Edu download + corpus build
python scripts/make_manifest.py --stage sweep   # emits run manifest
bash cluster/submit_manifest.sh outputs/manifests/sweep.tsv
```

Runs checkpoint every 30 min and are requeue-safe (2-day MaxWall). Gate
runs, the preregistration freeze, and the kill order are specified in the
design spec §7.
