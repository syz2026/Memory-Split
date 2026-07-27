# MemorySplit

Does a small language model learn better reasoning when arbitrary facts are
loss-masked and supplied by an exact external store, instead of being written
into the weights? Matched Dense and Split twins are trained on one byte-identical
token stream; only the target weights on fact payloads differ.

This branch carries the code for the two tiers that reached real hardware. They
come from **two different code generations and must not be mixed** — the metrics,
thresholds, and run instructions are not interchangeable.

## `160m-farmshare/`

The legacy exact-key battery, and the tier that ran on Stanford FarmShare.

`results/` holds what survived of those runs: six 160M gate summaries with their
training logs and per-task eval JSONL. They are single-seed pilot gates, not a
claim-bearing matrix. What they show:

- the mechanism works — Split reaches ~100% fresh-entity organizer use where
  Dense closed-book sits at 0.2–0.8%;
- the knowledge-free reasoning tasks never became learnable — iGSM stays at its
  ~4% floor and deduction at ~50% chance across every gate round.

The second point is why this endpoint was retired. It is a finding about the
instrument, not evidence for or against fact externalization.

The 160M matrix defined for the later relational design (15 configs) has never
been run.

## `1b-b200/`

The v3 reasoning code, deployed to one 8×B200 capacity-block node for
`memorysplit-exploratory-v3-1b-aws-n8` — 1.03B parameters, 8.17B tokens,
one epoch, Dense vs Split90.

**One of eight seed pairs completed.** Seeds 1–3 stopped at step ~1,140 of
15,582 about an hour in; seeds 4–7 never started. Only seed 0 has full
checkpoints, all five snapshots, and a held-out evaluation.

- `launch/` — node provisioning, staging, corpus verification, the cohort
  supervisor, and the S3 sync daemon.
- `run-seed0/` — seed 0 configs, training logs, and eval summaries.

The seed-0 evaluation is **not a result**. It is n=1 where the design requires
three paired seeds; its trajectory is non-monotonic and drops to exactly 0.0 for
both arms at step 11,687, which is an evaluator artifact; and it is scored by
teacher-forced greedy agreement rather than free generation. The scientific scope
of this cohort is `successor_exploratory_unpreregistered` — it is not the
preregistered confirmation and must not be reported as one.

## Provenance

Infrastructure identifiers (AWS account, bucket names, instance and capacity-block
IDs, cluster login) are replaced with `${VAR}` placeholders throughout.

`1b-b200/SHA256SUMS` and `release-receipt.json` are the original package
manifests and still describe the pre-scrub bytes.
