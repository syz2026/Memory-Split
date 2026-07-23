# MemorySplit Seed-Cohort Cloud Handoff Design

## Decision

Build two deterministic, hash-bound releases from one clean source commit and
one frozen scientific cohort assignment:

- the Illumina release contains only seed 0;
- the AWS P5 release contains seeds 1, 2, 3, and 4;
- every seed contains exactly one 360M Dense/Split90 matched pair;
- seed 0 remains part of the terminal five-pair cohort;
- provider assignment is frozen before any seed result is unblinded.

The AWS target is `p5.48xlarge`: eight NVIDIA H100 80 GB GPUs, 192 vCPUs,
2 TiB RAM, and eight 3.84 TB local NVMe devices.

## Scientific contract

All five seed bundles use the same:

- 356,033,536-parameter decoder architecture;
- corpus bytes and token ordering;
- 7,120,879,616 raw targets per arm;
- 524,288 targets per optimizer update;
- 13,582 optimizer updates;
- optimizer, schedule, context length, and inference budget;
- code commit, source locks, route manifest, and sealed evaluation release.

Within each seed, Dense and Split90 run simultaneously on symmetric hardware.
Illumina seed 0 uses three A100s per arm and reserves the seventh GPU for
verification. Each P5 seed uses four H100s per arm. Hardware may differ between
precommitted seed blocks, but never between arms within a block. The terminal
statistic remains the paired within-seed Split90-minus-Dense contrast.

Seed 0 alone is `incomplete` with `directional_only (1/5)`. Seeds 1–4 must
continue regardless of seed-0 direction unless a preregistered validity or
infrastructure stop condition fires.

## Release architecture

A canonical `COHORT-ASSIGNMENT.json` is included in both archives. It binds the
cohort ID, all five seeds, provider assignment, arm identities, model size,
training-token math, and source commit.

The Illumina packager must fail unless its archive contains:

- seed 0 Dense and Split90 configurations;
- no seed 1–4 run configuration;
- the Illumina profile and seed-0 launcher;
- the canonical cohort assignment.

The AWS packager must fail unless its archive contains:

- Dense and Split90 configurations for each of seeds 1–4;
- no seed-0 run configuration;
- a strict `aws-p5.48xlarge` profile;
- bootstrap, corpus-stage, paired-train, resume, evaluate, and collect entry
  points;
- the same canonical cohort assignment.

ZIP creation is deterministic and closed-allowlist. Each release is emitted
with an external SHA-256 file and `RELEASE.json`. Run manifests are finalized
outside the archive after the archive and dataset receipt hashes exist, avoiding
a self-hash cycle.

## P5 execution topology

One seed consumes one full `p5.48xlarge`:

- GPUs 0–3: four-rank Dense DDP;
- GPUs 4–7: four-rank Split90 DDP;
- distinct process groups and rendezvous ports;
- identical per-arm global update size;
- CPU and NVMe bandwidth partitioned symmetrically.

The default one-instance mode runs seeds 1, 2, 3, and 4 sequentially. A
deadline mode may launch four identical P5 instances, one preassigned seed per
instance, without changing any run manifest. A single P5 must never interleave
two seed pairs.

The eight local NVMe devices form ephemeral RAID0 scratch. Source data, corpus
publication, and environment artifacts are copied from hash-bound object
storage, verified, and staged there. Checkpoints, receipts, logs, and final
evidence are durably mirrored to S3; no irreplaceable artifact may exist only
on instance store.

Use a region-pinned AWS Deep Learning AMI or immutable container digest with
H100-compatible NVIDIA driver, CUDA, Fabric Manager, PyTorch, and NCCL. Because
training is single-node, NCCL uses NVSwitch; multi-node EFA configuration is
not a launch prerequisite.

## Lifecycle

1. Verify ZIP, release, cohort assignment, code commit, and environment lock.
2. Verify or materialize the immutable 7.12B-target corpus and Dense/Split90
   sidecars before paid training.
3. Run readiness gates and a two-GPU functional canary.
4. Run a 100-update four-plus-four throughput canary and derive a measured ETA.
5. Launch the assigned seed pair with fresh output directories.
6. Write durable, hash-bound checkpoints at least every 30 minutes and on
   termination notice.
7. Resume only from a receipt matching seed, arm, world size, config, code,
   corpus, and checkpoint bytes.
8. Evaluate both arms against the sealed release, replay proofs, collect
   evidence, and upload the immutable result bundle.
9. Proceed to the next assigned seed regardless of observed effect direction.
10. After all five bundles exist, run the preregistered exact one-sided N=5
    sign-flip test and label broader family claims as unsupported at N=5.

## Failure handling

The launcher fails closed on a wrong seed assignment, generic `split` arm,
partial pair, stale output directory, mismatched hash, absent sidecar,
non-integral update count, failed readiness gate, unverified checkpoint, or
incomplete sealed evaluation. Spot interruption is allowed only after the
resume canary passes; On-Demand or reserved capacity is the default.

## Verification

Tests must prove:

- Illumina contains seed 0 and rejects seeds 1–4;
- P5 contains exactly seeds 1–4 and rejects seed 0;
- every seed has one Dense and one explicit Split90 configuration;
- all configurations encode exactly 356,033,536 parameters,
  7,120,879,616 targets, 524,288 targets/update, and 13,582 updates;
- the two ZIPs bind the same cohort assignment, dataset, code, and evaluation
  identities;
- repeated builds are byte-identical;
- P5 launch dry-runs render symmetric `4+4` DDP commands;
- resume, interruption, malformed receipts, and partial-pair cases fail closed.

## Rejected approaches

- One archive with an ambiguous provider role: easier to misuse and harder to
  audit.
- A documentation-only P5 handoff: not executable or independently verifiable.
- Putting all five seeds in both archives: permits duplicate or selectively
  reported runs.
- Running Dense and Split90 sequentially on eight GPUs: introduces avoidable
  temporal asymmetry and is unlikely to improve this small model's wall time.
