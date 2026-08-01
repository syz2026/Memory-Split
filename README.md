# Does masking fact targets buy reasoning?

A controlled test of the belief that factual memorisation competes with
reasoning for parameters in small language models — the belief the phi-3 team
cites when filtering fact-heavy pages from training data. The training-time
mechanism is LMLM-style (arXiv 2505.15962): fact values are present in the
context but excluded from the loss.

## The estimand, stated exactly

The primary quantity is

```
effect = Y[FACTMASK] - Y[RANDPOS]
```

the effect of masking fact-value targets rather than an equal mass of matched
non-value targets, measured on closed-book iGSM accuracy, in a 40.6M-parameter
tied-embedding decoder with 21.2M non-embedding parameters, trained at ~141
tokens per parameter on one controlled corpus.

**Capacity reallocation is an interpretation of that quantity, not the quantity
itself, and this design does not identify it.** Masking removes competing
gradients whether or not any parameter was ever occupied, so gradient
interference predicts the same sign and the same monotone dose trend. The
mitigations are a shape test across fact loads and a gradient-mass
decomposition; both are consistency evidence, not identification.

Nothing here licenses a claim about "small language models" generally, or about
phi-3, which is 100x larger.

## Design

One byte-identical `uint16` token stream per fact load. The three arms are
three `uint8` target-weight sidecars over that stream:

| Arm | Sidecar |
|---|---|
| `SUP` | all ones — manipulation check and load main effect |
| `FACTMASK` | zero on fact-value targets |
| `RANDPOS` | zero on non-value spans matched on mass, span length, relative position and token NLL |

The arms cannot differ through anything except which targets earn gradient.
There is no external store at training or evaluation time.

Fact load varies by trading entities against exposures — high load is `N`
entities at `E` exposures, low load is `0.3N` at `E/0.3` — so document count,
total tokens, optimizer steps, mask mass and mask positions are all identical
across loads and only the unique-entropy ceiling moves. That decouples fact
load from exposures per fact, which is the confound that invalidated the
earlier dose sweep.

## Layout

```
corpusgen/   biography records (the fact load), iGSM-lite and Horn-clause
             deduction generators with independent oracles
train/       GPT-2 BPE tokenizer, decoder-only GPT, memmap + sidecar loader,
             AdamW trainer with checkpoint/resume
evals/       greedy decoding, answer scorers, recoverable-bits accounting,
             paired statistics and the frozen verdict
ops/         corpus builder, config generation, Slurm submission
cluster/     FarmShare environment setup and sync
docs/        PREREGISTRATION.md, RETAINED-RESULTS.md, POPQA-HELDOUT-KEY.md
outputs/     retained results only; see docs/RETAINED-RESULTS.md
tests/       pytest suite, runs offline
```

## Quick start

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv/bin/python
export PYTHONPATH=.
.venv/bin/python -m pytest tests -q
```

## Results

**Read `docs/RETAINED-RESULTS.md` before citing any number.** It catalogues
every artifact-backed result with its hash and states what each can and cannot
support. A number that does not appear there has no artifact behind it and is
withdrawn.

One caveat applies to everything measured before this repository state: the
loss was computed with `reduction='mean'` over surviving targets, so masking
silently multiplied every remaining target's weight by `1/(1-f)`. Every masked
arm therefore trained under a different effective objective than its dense
twin, over and above the masking. Those runs remain evidence that masking keeps
values out of the weights; they are not valid arm contrasts.
