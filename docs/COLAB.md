# KQA Pro continuation experiment on Google Colab

The notebook `notebooks/kqa_memory_transfer_colab.ipynb` runs the complete
answer-only experiment from two existing MemorySplit checkpoints.

## What is trained

- **Dense:** ordinary KQA fact documents plus answer-only reasoning QA.
  Fact values receive gradient and must be stored in weights.
- **Split:** the same fact documents use the native
  `<|db_start|>query<|db_retrieve|> value<|db_end|>` rendering. Value tokens
  remain loss-masked and the same facts are installed in the organizer.
- **Both:** identical answer-only QA text. KoPL and SPARQL are never emitted
  into model training data.

KoPL is executed only during offline preparation to identify supporting facts,
create structure-only skeleton labels, and enforce leakage checks.

## Drive layout

Each source model must be a complete run directory containing `config.yaml`
and one of: model-only `model.pt`, full `ckpt.pt`, or
`snapshots/step*.pt`. The notebook selects the latest numbered snapshot when
no root checkpoint exists; explicit snapshot overrides are available in the
first cell. The selected checkpoint must contain its training `step`
metadata. The two source configs must agree on architecture, seed, schedule,
optimizer settings, microbatching, precision, and selected completion step:

```text
MyDrive/memorysplit/compose_dense_s0/
├── config.yaml
└── snapshots/
    └── stepNNNNNNN.pt

MyDrive/memorysplit/compose_split_s0/
├── config.yaml
└── snapshots/
    └── stepNNNNNNN.pt
```

The notebook writes continued checkpoints and evaluations under:

```text
MyDrive/memorysplit_kqa_outputs/runs/
```

KQA JSON, prepared artifacts, packed `*.bin` shards, and tokenizer cache stay
under `/content` while active. Do not train directly from Drive-backed memmaps;
Drive FUSE is too slow and unreliable for random packed-shard reads.
The notebook also stages source weights onto local disk before loading them and
verifies their SHA-256 digests.

The source `memorysplit` folder may be a read-only shared-folder shortcut.
Outputs deliberately go to a separate writable folder owned by the account
mounted in Colab.

## Colab runtime

An A100 or L4 is preferred. At config creation, `precision: auto` is resolved
once to explicit `bf16` or `fp16` and stored in the run identity. A reconnect
on an incompatible GPU is rejected rather than silently changing arithmetic.
Torch compilation is disabled.

Free-tier sessions may not finish a large continuation run. The trainer writes
durable checkpoints to Drive and `--resume auto` resumes the new run after a
disconnect. Recreate the deterministic local KQA corpus at the same
`/content/...` path before resuming.

The notebook contains an immutable repository commit rather than following a
branch or a stale Drive pin. KQA source files, source checkpoints, and emitted
packed shards have SHA-256 identities. Source configs and completion steps are
checked as a matched pair, and resume refuses a checkpoint whose code,
source-model, data, precision, or training identity differs from the current
config.

## Default pilot

The notebook defaults to:

- 20,000 training QA examples
- 2,000 transfer-test questions
- 50,000 facts unused by either QA split as additional memory pressure
- answer-only continuation text
- no `FindAll` questions
- at most 16 support lookups and 16 returned values per lookup
- at least two complete fact exposures in each packed shard

Start with a modest token budget to verify the mechanism. Increase
`NOISE_FACTS`, `CONTINUATION_TOKENS`, model size, and seeds only after:

1. `report.json` passes every split invariant.
2. `continuation_report.json` confirms every selected fact was exposed at
   least twice and records the exact packed-shard hashes.
3. Dense direct recall and split organizer recall are both nontrivial.
4. The split model emits well-formed KQA lookups.

## Outputs

For each arm:

```text
<continued-run>/
├── config.yaml
├── ckpt.pt
├── model.pt
├── model.pt.sha256
├── log.jsonl
├── snapshots/
├── kqa_dev/
│   └── summary.json
└── kqa_evals/
    ├── recall_*.jsonl
    ├── transfer_qa.jsonl
    ├── transfer_qa_off.jsonl
    ├── transfer_qa_wrong_store.jsonl
    ├── transfer_qa_oracle_context.jsonl
    └── summary.json
```

The paired comparison reports:

- overall transfer-QA accuracy;
- direct fact recall / organizer lookup coverage;
- the descriptive QA accuracy conditional on both systems passing separate
  direct-access probes;
- split QA-time support-query coverage, store-off and wrong-store controls,
  and an equal oracle-context control;
- clustered paired bootstrap confidence intervals by hidden KoPL skeleton.

Wrong-store summaries separate observed relation-matched substitutions from
format-preserving synthetic fallbacks; treat this as a robustness diagnostic,
not a clean causal intervention.

Development reports are generated before the transfer set is unblinded. The
same paired development and transfer reports are generated for the
pre-continuation source models.
Oracle prompts close to the model context limit receive a smaller per-group
generation budget; evidence is never silently truncated.

## Interpretation guardrail

The transfer facts are **QA-held-out**, not unseen to the systems:

- dense receives them only as standalone fact training;
- split receives them through masked fact documents and the organizer;
- neither receives reasoning QA demonstrations using them.

This tests whether previously learned procedures transfer to independently
supplied facts. It does not test learning an entirely unseen reasoning
structure; that should be a separate split.

The primary endpoint is unconditional paired transfer-QA accuracy. Conditional
accuracy is a diagnostic, not a causal estimand: conditioning on model-specific
recall can select different subsets. Report results by KQA skill, program
length, and support count; many accepted questions are short one-lookup cases,
so aggregate accuracy alone is not evidence of complex reasoning.

The existing source checkpoints were trained with the repository's legacy v1
microbatch weighting. Because the user-provided arms are already diverged, this
workflow preserves that native method. Results compare the two legacy
end-to-end systems; they do not isolate fact storage as the sole causal
difference. Fact-token compute is matched, while dense and split fact-document
exposure counts differ because lookup wrappers make split documents longer.
Facts and answer-only QA are also introduced together. A stronger claim about
what the new QA specifically taught requires additional facts-only
continuation controls.
