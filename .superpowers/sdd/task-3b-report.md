# Task 3B Report: Executable P5 Qualification Canary

## Status

Complete on `feat/memorysplit-v3-aws-n10`, starting from
`6c5c964bd83d71f03cf28b92d58535fd23badac6`.

Implementation commit:

- `22ee3c4` — `feat: add executable P5 qualification canary`

The session resumed once after a PING timeout. The existing RED/GREEN trainer
work was recovered from the worktree and continued without being discarded or
duplicated.

## Delivered

- Added executable `cluster/aws/p5/canary.py` with dependency-injected command,
  object-store, and time boundaries. It validates the extracted release,
  external release receipt, schema-3 run manifest, canonical dataset receipt,
  authenticated environment receipt v2, runtime lock, selected instance, and
  live boot before qualification.
- Implemented all six phases in the required order:
  1. exact eight-device H100 80GB, Fabric Manager, NVSwitch, live boot, and
     runtime-fact checks;
  2. one eight-rank NCCL all-reduce with finite latency;
  3. one-update Dense and Split90 functional runs with isolated outputs;
  4. exact same-config checkpoint resume from step 1 to step 2;
  5. concurrent four-plus-four throughput runs with fixed GPU/CPU halves,
     distinct ports, 100 updates, ten-update exclusion, and deterministic
     medians; and
  6. deterministic checksum- and version-bound S3 roundtrip.
- Added canonical
  `memorysplit-aws-p5-qualification-v1` receipts with exact identity,
  phase, hardware, functional, resume, throughput, and S3 evidence. Local
  publication is owner-only, atomic, and no-replace; failed publication removes
  the installed inode before returning failure.
- Added operational-only `--operational-steps N` support. Inputs are canonical
  positive platform integers, duplicate/zero/negative/float/overflow forms
  fail before config access, and resume advances by exactly the requested
  additional updates capped by unchanged scientific `max_steps`.
- Preserved config bytes, config fingerprints, optimizer/snapshot schedules,
  checkpoint schema, data cursor rules, and no-option trainer behavior.
  Operational per-update metrics are emitted separately and only when the
  operational option is present.
- Added `msctl canary run` for the exact v3 profile. Dry run is local and
  mutation-free. Apply authenticates the tuple and selected instance, publishes
  a content-addressed intent, sends it through the fixed argv SSM document,
  polls with a bounded 48-hour deadline, consumes and independently parses the
  receipt, re-downloads the exact S3 object version, and publishes the final
  receipt no-replace at
  `canaries/<tuple-sha256>/<receipt-sha256>.json`.
- Extended the remote argv validator only for the closed canary operation.
  Production operations reject both `--operational-steps N` and
  `--operational-steps=N`; the canary intent may execute only
  `cluster/aws/p5/canary.py`.
- Made the canary and all imported repository runtime dependencies mandatory in
  the v3 package.

No checkpoint/state production lifecycle, production submit/resume/evaluate
enablement, evaluation, IaC, verifier, runbook, Illumina, v2 scientific file,
or `corpusgen/` behavior was changed.

## TDD evidence

Observed RED before implementation:

- bounded API: `train()` rejected the missing `operational_steps` keyword;
- operational metrics: trainer attributes and canonical output were absent;
- canary: executable module was absent;
- controller: `AwsP5Backend.canary_run` was absent;
- apply: controller stopped at `CANARY_APPLY_UNIMPLEMENTED`;
- remote boundary: canary operation was rejected by the fixed argv validator;
- CLI: `canary` was not a recognized command;
- package: five required canary/runtime members were not enforced;
- live boot: no boot command was rendered and boot drift passed;
- atomic failure: a directory-fsync error left the linked receipt;
- remote wait: controller constructor lacked injected polling boundaries.

Each slice was made GREEN before the next production slice. Focused final
results:

- canary/controller tests: **29 passed**;
- trainer/DDP tests: **84 passed**;
- launcher/package/manifest tests: **307 passed**;
- controller/environment/canary tests: **252 passed**;
- remote argv tests: **19 passed**;
- full package suite: **100 passed**.

## Required verification

The exact required combined command passed before the final bounded polling
addition:

```text
642 passed, 22 warnings in 170.60s
```

After the final polling change, the same required files were rerun in three
space-bounded groups because another process reduced the host volume to roughly
1 GiB and the combined pytest basetemp hit `ENOSPC`. The final groups were
non-overlapping and cover every file in the binding command:

```text
252 passed in 41.70s
84 passed in 36.51s
307 passed in 90.30s
```

Final aggregate: **643 passed**. The only warnings in the combined run were 22
pre-existing `os.fork()` deprecation warnings from
`interruption_checkpoint.py`.

Additional verification:

- changed-file `python -m py_compile`: passed;
- `git diff --check`: passed;
- forbidden-scope diff check for configs, `corpusgen/`, evaluation, docs, and
  checkpoint interruption implementation: passed;
- scientific v3 configs and fingerprints were not modified.

## Self-review

### Failure behavior

- Every command requires zero exit and empty stderr.
- Partial pairs, peer failure, missing/non-finite/zero throughput, incorrect
  steps/cursors/world sizes, checkpoint/config/data drift, malformed phase
  order, live boot drift, and S3 checksum/byte/version drift fail before a
  passing receipt.
- Receipt parsing independently rejects unknown, missing, wrongly typed, and
  identity-mutated fields.

### No-replace behavior

- Local publication uses same-directory staging plus atomic hard-link
  no-replace, mode `0600`, directory fsync, and inode-matched cleanup on
  post-link failure.
- Intent, roundtrip, and final receipt uploads use checksum-bound
  `If-None-Match: *`.
- An already-present final receipt is accepted only after exact checksum,
  length, metadata, and non-null version verification; it is never overwritten.

### Topology and concurrency

- Hardware requires eight indexed H100 80GB devices, active Fabric Manager, and
  a complete eight-by-eight NVLink/NVSwitch matrix.
- The current kernel boot ID must equal the attested explicit boot ID.
- Throughput uses Dense GPUs 0-3/CPUs 0-95 and Split90 GPUs 4-7/CPUs 96-191,
  distinct deterministic ports, one concurrent group, four ranks per arm, and
  exactly 100 updates.
- The default pair runner starts both processes before polling and terminates
  the peer process group on failure or timeout.

## Concerns and deliberate boundaries

- No live AWS, P5, NCCL, Docker, or S3 call was made in this task; those paths
  are exercised through strict synthetic injected boundaries.
- Real qualification still depends on Task 5 supplying the immutable AMI,
  digest-pinned image/runtime lock, SSM-installed control bundle, versioned S3
  bucket, and selected online P5 instance.
- The controller intentionally enables only qualification. Production
  submit/resume/evaluate and checkpoint/state lifecycle remain disabled for v3.

## Blocking review follow-up

All Critical and Important Task 3B review findings were closed:

1. `AwsCliObjectStore` now parses the complete AWS CLI stdout as exactly one
   closed JSON object, including realistic indented multi-line output. Duplicate,
   trailing, non-finite, missing/null-version, checksum, and version drift fail
   closed for both put and get.
2. `scripts/run_train.py` disables argparse abbreviation. The remote argv
   contract rejects the exact operational option, `=` form, and every argparse
   prefix from `--o` through `--operational-step`, outside the closed canary
   operation.
3. Dense and Split90 config bytes are rehashed from a pinned regular-file read
   immediately before every functional, resume, and throughput execution
   boundary. Any post-plan mutation fails before the next training process or
   pair starts, so receipt config hashes bind the bytes admitted for execution.
4. Every remote canary path is normalized to an absolute path during plan
   admission and the SSM argv is rendered only from those normalized plan
   paths. Relative caller inputs no longer make execution depend on SSM cwd.
5. Controlled tests now exercise the real subprocess pair runner. They prove
   both children spawn before polling and prove process-group termination on
   peer failure and timeout.

Exact review RED evidence:

```text
review canary slice: 10 failed, 6 passed, 29 deselected in 1.71s
trainer abbreviation slice: 5 failed, 8 passed in 11.03s
strict AWS version slice: 3 failed in 0.14s
```

The failures directly reproduced multi-line JSON rejection, stale config
acceptance at all three phase boundaries, relative SSM argv, abbreviated
operational-option bypasses, argparse config access after abbreviation, and
missing/null S3 version acceptance.

Exact review GREEN evidence:

```text
review canary slice: 16 passed, 29 deselected in 1.56s
trainer abbreviation slice: 13 passed in 10.75s
strict AWS version slice: 3 passed in 0.12s
real subprocess pair runner: 3 passed, 42 deselected in 1.05s
focused canary/trainer/argv suites: 146 passed in 25.10s
```

Final post-review binding command:

```text
667 passed, 22 warnings in 183.40s
```

The 22 warnings remain the pre-existing `os.fork()` deprecation warning in
`interruption_checkpoint.py`. Final changed-file `py_compile`,
`git diff --check`, and forbidden-scope diff checks passed.

No live paid AWS/P5 run was performed. External hardware and cloud
qualification remains explicitly deferred to the later execution task.

Report path:

`/Users/stephenzhang/Documents/MemorySplit/.worktrees/memorysplit-v3-aws-n10/.superpowers/sdd/task-3b-report.md`
