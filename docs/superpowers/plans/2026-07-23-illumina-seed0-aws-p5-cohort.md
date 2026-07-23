# Illumina Seed 0 and AWS P5 Cohort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one deterministic Illumina ZIP containing only seed 0 and one
deterministic AWS P5 ZIP containing only seeds 1–4, then run every 360M
Dense/Split90 pair under one frozen five-seed scientific contract.

**Architecture:** A shared canonical cohort assignment and ten immutable run
configurations are the source of truth. Provider-specific packagers select
disjoint seed subsets and prove their contents. Illumina launches seed 0 as
three A100 ranks per arm; AWS launches one assigned seed per `p5.48xlarge` as
four H100 ranks per arm, with object storage as the durable boundary around
ephemeral NVMe.

**Tech Stack:** Python 3.11, PyTorch DDP/NCCL, canonical JSON, YAML,
`pytest`, AWS EC2 `p5.48xlarge`, S3, Systems Manager, CloudWatch, NVMe RAID0,
AWS Deep Learning AMI, deterministic ZIP.

## Global Constraints

- Cohort ID: `memorysplit-confirmatory-v2-360m-n5`.
- Illumina owns exactly seed `0`; AWS owns exactly seeds `1,2,3,4`.
- Every seed has exactly one `dense` and one explicit `split90` arm.
- Model parameters: `356033536`.
- Raw targets per arm: `7120879616`.
- Targets per update: `524288`.
- Optimizer updates: `13582`.
- Context length: `1024`; sequences per update: `512`.
- Illumina train topology: symmetric `3+3`, with GPU 6 reserved.
- P5 train topology: symmetric `4+4`, using GPUs `0–3` and `4–7`.
- Dense and Split90 within a seed must share initialization, corpus bytes,
  target order, optimizer, schedule, and evaluation release.
- Provider assignment is frozen before seed 0 is unblinded.
- Seed 0 alone is `incomplete` and `directional_only (1/5)`.
- No effect direction may stop seeds 1–4.
- Production launches fail closed until DDP, sidecar publication, readiness,
  and replayable evaluation gates pass.
- No AWS credential, token, private key, sealed gold, corpus, checkpoint,
  cache, or log may enter either ZIP.

---

## File Map

**Shared cohort contract**

- Create: `configs/cohort-assignment-v2.json`
- Create: `configs/360m-v2/dense-s0.yaml` through `dense-s4.yaml`
- Create: `configs/360m-v2/split90-s0.yaml` through `split90-s4.yaml`
- Create: `tests/test_cohort_assignment_v2.py`

**Illumina release**

- Modify: `scripts/package_illumina_handoff.py`
- Modify: `tests/test_package_illumina_handoff.py`
- Modify: `AGENT-START.md`

**AWS release and runtime**

- Create: `cluster/profiles/aws-p5.48xlarge.json`
- Create: `AWS-P5-START.md`
- Create: `DATASET-POINTER-AWS.json`
- Create: `requirements-aws-p5.lock`
- Create: `cluster/aws/p5/bootstrap.sh`
- Create: `cluster/aws/p5/launch_seed_pair.py`
- Create: `cluster/aws/p5/interruption_checkpoint.py`
- Create: `scripts/package_aws_p5_handoff.py`
- Create: `tests/test_aws_p5_profile.py`
- Create: `tests/test_aws_p5_launcher.py`
- Create: `tests/test_package_aws_p5_handoff.py`

**Provider-neutral lifecycle**

- Modify: `msctl/profile.py`
- Modify: `msctl/contracts.py`
- Modify: `msctl/cli.py`
- Modify: `msctl/operations.py`
- Create: `msctl/aws_p5.py`
- Modify: `tests/test_msctl.py`

**Production data and DDP prerequisite**

- Modify: `corpusgen/parallel/publication.py`
- Modify: `scripts/build_parallel_corpus.py`
- Modify: `train/data.py`
- Modify: `train/trainer.py`
- Modify: `scripts/run_train.py`
- Modify: `tests/test_parallel_corpus.py`
- Modify: `tests/test_sharded_loader.py`
- Modify: `tests/test_ddp_trainer.py`

**Runbook and release verification**

- Create: `evals/confirmatory/__main__.py`
- Create: `evals/confirmatory/runner.py`
- Create: `tests/test_confirmatory_runner.py`
- Create: `docs/AWS-P5-360M-RUNBOOK.md`
- Create: `scripts/verify_cohort_releases.py`
- Create: `tests/test_verify_cohort_releases.py`

---

### Task 1: Freeze the disjoint five-seed cohort

**Files:**
- Create: `configs/cohort-assignment-v2.json`
- Create: `configs/360m-v2/*.yaml`
- Create: `tests/test_cohort_assignment_v2.py`

**Interfaces:**
- Consumes: `configs/preregistration-v2.yaml`
- Produces: `load_cohort_assignment(path) -> CohortAssignment` in
  `msctl/contracts.py`

- [ ] **Step 1: Write failing contract tests**

Test exact provider assignment, ten unique `(seed, arm)` cells, explicit
`split90`, and exact token math:

```python
def test_frozen_cohort_is_disjoint_and_complete(repo_root):
    cohort = load_cohort_assignment(
        repo_root / "configs/cohort-assignment-v2.json"
    )
    assert cohort.illumina_seeds == (0,)
    assert cohort.aws_p5_seeds == (1, 2, 3, 4)
    assert set(cohort.illumina_seeds).isdisjoint(cohort.aws_p5_seeds)
    assert set(cohort.illumina_seeds + cohort.aws_p5_seeds) == set(range(5))
    assert cohort.targets_per_update * cohort.optimizer_steps == 7_120_879_616
```

Parameterize malformed cases for duplicate seed, missing seed, seed 0 in AWS,
generic `split`, wrong model size, partial pair, bool-valued integer, wrong
token count, and mismatched Dense/Split90 invariants.

- [ ] **Step 2: Run the tests and confirm red**

Run:

```bash
pytest -q tests/test_cohort_assignment_v2.py
```

Expected: import/file failures because the cohort contract does not exist.

- [ ] **Step 3: Add the canonical assignment**

Use this exact semantic payload, serialized with sorted keys and a trailing
newline:

```json
{
  "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
  "model_parameters": 356033536,
  "optimizer_steps": 13582,
  "provider_seeds": {
    "aws-p5.48xlarge": [1, 2, 3, 4],
    "illumina-usfc-prd": [0]
  },
  "raw_target_tokens": 7120879616,
  "schema_version": 2,
  "targets_per_update": 524288
}
```

Use this complete Dense seed-0 shape; generate the other nine configs by
changing only `run_id`, `condition`, `seed`, `out_dir`, and `sidecar_name`:

```yaml
schema_version: 2
cohort_id: memorysplit-confirmatory-v2-360m-n5
run_id: memorysplit-v2-360m-s0-dense
condition: dense
seed: 0
model: d360m
ctx: 1024
train_corpus: dataset/corpus-receipt.json
sidecar_name: dense_target_weights
out_dir: runs/seed-0/dense
micro_batch_size: 8
tokens_per_step: 524288
max_steps: 13582
total_tokens: 7120879616
lr: 0.001
warmup_steps: 300
weight_decay: 0.1
compile: true
device: cuda
log_every: 20
eval_every: 250
snap_frac: 0.1
ckpt_minutes: 30
```

Split90 uses `sidecar_name: split90_target_weights`. Paths are logical,
relative run-root paths. The provider launcher resolves the run root before
training and verifies the corpus receipt against the external run manifest;
the checked-in config therefore contains no machine-specific path or
unresolved environment variable.

- [ ] **Step 4: Implement strict parsing and cross-config validation**

Reject unknown fields, aliases, bool integers, unresolved environment
placeholders, and conditions other than `dense` or `split90`. Recompute every
config SHA-256 from bytes and expose it from `CohortAssignment`.

- [ ] **Step 5: Run focused tests**

```bash
pytest -q tests/test_cohort_assignment_v2.py tests/test_msctl.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add configs/cohort-assignment-v2.json configs/360m-v2 \
  msctl/contracts.py tests/test_cohort_assignment_v2.py tests/test_msctl.py
git commit -m "feat: freeze disjoint five-seed cohort"
```

---

### Task 2: Make the Illumina ZIP prove it contains only seed 0

**Files:**
- Modify: `scripts/package_illumina_handoff.py`
- Modify: `tests/test_package_illumina_handoff.py`
- Modify: `AGENT-START.md`

**Interfaces:**
- Consumes: `CohortAssignment`
- Produces: `RELEASE-METADATA.json.seed_assignment`

- [ ] **Step 1: Add adversarial package tests**

Open the produced ZIP and parse—not filename-match—the assignment and configs:

```python
with ZipFile(release.archive) as archive:
    assignment = json.loads(archive.read("configs/cohort-assignment-v2.json"))
    config_names = {
        name for name in archive.namelist()
        if name.startswith("configs/360m-v2/")
    }
assert assignment["provider_seeds"]["illumina-usfc-prd"] == [0]
assert config_names == {
    "configs/360m-v2/dense-s0.yaml",
    "configs/360m-v2/split90-s0.yaml",
}
```

Add red tests proving packaging rejects seed 0 omission, seed 1 inclusion,
generic `split`, wrong token math, partial pair, and assignment/config hash
mismatch.

- [ ] **Step 2: Run and confirm red**

```bash
pytest -q tests/test_package_illumina_handoff.py
```

- [ ] **Step 3: Replace broad config inclusion with provider selection**

The Illumina allowlist includes all shared scientific locks but only the two
seed-0 run configs. `_collect_payload` must call the strict cohort parser before
writing any output. Add this immutable metadata:

```json
"seed_assignment": {
  "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
  "provider": "illumina-usfc-prd",
  "seeds": [0],
  "arms": ["dense", "split90"]
}
```

- [ ] **Step 4: Make publication no-replace**

Publish ZIP, checksum, and release receipt as one owned staging set. Refuse an
existing final path; never `os.replace` a prior release. Hash the exact open
descriptors used to construct the ZIP.

- [ ] **Step 5: Update the agent start contract**

State unambiguously that this archive may launch only seed 0 and that any
seed 1–4 request is a contract violation.

- [ ] **Step 6: Verify**

```bash
pytest -q tests/test_package_illumina_handoff.py tests/test_msctl.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/package_illumina_handoff.py \
  tests/test_package_illumina_handoff.py AGENT-START.md
git commit -m "fix: bind Illumina handoff to seed zero"
```

---

### Task 3: Define the AWS P5 profile and durable storage contract

**Files:**
- Create: `cluster/profiles/aws-p5.48xlarge.json`
- Create: `DATASET-POINTER-AWS.json`
- Create: `requirements-aws-p5.lock`
- Modify: `msctl/profile.py`
- Create: `tests/test_aws_p5_profile.py`

**Interfaces:**
- Produces: `AwsP5Profile`
- Consumes: region, S3 prefix, AMI/container digest from operator environment

- [ ] **Step 1: Write profile tests**

Require exact hardware:

```python
assert profile.instance_type == "p5.48xlarge"
assert profile.gpu_model == "NVIDIA H100 80GB"
assert profile.allocated_gpus == 8
assert profile.train_groups == (4, 4)
assert profile.instance_store_devices == 8
assert profile.instance_store_device_bytes == 3_840_000_000_000
```

Reject a one-GPU `p5.4xlarge`, Spot-by-default, static AWS keys, mutable AMI
labels, unpinned container tags, non-S3 durable roots, and seed 0.

- [ ] **Step 2: Run and confirm red**

```bash
pytest -q tests/test_aws_p5_profile.py
```

- [ ] **Step 3: Add a closed AWS profile schema**

The profile requires:

- `provider: aws-p5.48xlarge`;
- `instance_type: p5.48xlarge`;
- `purchase_model: on_demand`;
- `gpu.allocated: 8`;
- `gpu.seed_train_groups: [4, 4]`;
- `storage.scratch_root: /mnt/memorysplit`;
- `storage.durable_uri_env: MS_S3_ROOT`;
- `runtime.ami_id_env: MS_AWS_AMI_ID`;
- `runtime.container_digest_env: MS_CONTAINER_DIGEST`;
- `assigned_seeds: [1,2,3,4]`.

The code validates an `ami-*` ID and `sha256:<64 lowercase hex>` digest at
runtime; values remain external because AMI IDs are region-specific.

- [ ] **Step 4: Pin the Python environment**

Generate `requirements-aws-p5.lock` from the same PyTorch major/minor and
project dependencies used by Illumina. Record hashes. Do not pin a CUDA wheel
that conflicts with the selected DLAMI/container.

- [ ] **Step 5: Verify**

```bash
pytest -q tests/test_aws_p5_profile.py tests/test_msctl.py
```

- [ ] **Step 6: Commit**

```bash
git add cluster/profiles/aws-p5.48xlarge.json DATASET-POINTER-AWS.json \
  requirements-aws-p5.lock msctl/profile.py tests/test_aws_p5_profile.py
git commit -m "feat: define strict AWS P5 profile"
```

---

### Task 4: Complete the production sidecar and four-rank DDP prerequisite

**Files:**
- Modify: `corpusgen/parallel/publication.py`
- Modify: `scripts/build_parallel_corpus.py`
- Modify: `train/data.py`
- Modify: `train/trainer.py`
- Modify: `scripts/run_train.py`
- Modify corresponding corpus/loader/DDP tests

**Interfaces:**
- Produces: `memorysplit-parallel-corpus-v2` receipt with token and sidecar sets
- Consumes: `train_corpus` receipt plus `sidecar_name`

- [ ] **Step 1: Add a failing end-to-end Split90 loader test**

Build a tiny multi-shard publication containing:

- ordered `uint16` token shards;
- ordered Dense `uint8` target-weight shards;
- ordered Split90 `uint8` target-weight shards;
- stream SHA-256 and item count for each set;
- zero weights at every padded target.

Launch two CPU/gloo ranks from the receipt and prove the global target sequence
and weighted loss equal the single-process reference.

- [ ] **Step 2: Run and confirm red**

```bash
pytest -q tests/test_parallel_corpus.py tests/test_sharded_loader.py \
  tests/test_ddp_trainer.py
```

Expected: the current v1 publisher cannot provide verified Split90 sidecars.

- [ ] **Step 3: Publish receipt v2 atomically**

Implement the exact sidecar namespace already declared by
`PARALLEL_SIDECAR_V2_CONTRACT`. Bind ordered path, bytes, SHA-256, dtype, item
count, and whole-stream SHA-256. Verification must use descriptor-pinned
regular files and reject symlinks, extra files, wrong padding, and reordered
shards.

- [ ] **Step 4: Validate four-rank update geometry**

For `ctx=1024`, `tokens_per_step=524288`, and world size 4:

```python
assert rank_sequence_counts(512, 4) == (128, 128, 128, 128)
```

With `micro_batch_size=8`, each rank performs exactly 16 real microsteps.
Require `max_steps * tokens_per_step == total_tokens`.

- [ ] **Step 5: Run DDP integration tests**

```bash
pytest -q tests/test_parallel_corpus.py tests/test_sharded_loader.py \
  tests/test_ddp_trainer.py tests/test_trainer.py tests/test_model.py
```

- [ ] **Step 6: Commit**

```bash
git add corpusgen/parallel/publication.py scripts/build_parallel_corpus.py \
  train/data.py train/trainer.py scripts/run_train.py \
  tests/test_parallel_corpus.py tests/test_sharded_loader.py \
  tests/test_ddp_trainer.py tests/test_trainer.py
git commit -m "feat: train Split90 from verified sharded sidecars"
```

---

### Task 5: Build the single-node P5 paired launcher

**Files:**
- Create: `cluster/aws/p5/bootstrap.sh`
- Create: `cluster/aws/p5/launch_seed_pair.py`
- Create: `cluster/aws/p5/interruption_checkpoint.py`
- Create: `tests/test_aws_p5_launcher.py`

**Interfaces:**
- CLI:
  `python cluster/aws/p5/launch_seed_pair.py --seed N --manifest PATH [--apply]`
- Produces one JSON object and two supervised `torchrun` process groups

- [ ] **Step 1: Write dry-run and fail-closed tests**

For each seed 1–4, assert exact commands:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  --master_port=<dense-port> scripts/run_train.py --config <dense>
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  --master_port=<split-port> scripts/run_train.py --config <split90>
```

Reject seed 0, seed 5, partial pairs, equal ports, occupied ports, non-H100
devices, stale outputs, wrong config hashes, wrong corpus receipt, missing
sidecars, or any inherited secret variable.

- [ ] **Step 2: Run and confirm red**

```bash
pytest -q tests/test_aws_p5_launcher.py
```

- [ ] **Step 3: Implement bootstrap**

`bootstrap.sh` must:

1. verify IMDSv2 and expected instance type;
2. verify eight H100 devices and running Fabric Manager;
3. identify all eight instance-store NVMe devices by model, never device name;
4. create RAID0 and mount `/mnt/memorysplit` with owner-only permissions;
5. verify the immutable container/environment;
6. download from the assigned S3 prefix with an instance role, not keys;
7. verify ZIP, release, dataset, code, and cohort hashes;
8. emit one canonical bootstrap receipt to S3.

- [ ] **Step 4: Implement paired supervision**

Launch both arms, record child PIDs before reporting success, propagate a
failure from either arm, terminate the peer on fail-stop errors, and reject
partial completion. Use separate CPU affinity sets and equal data-loader worker
budgets.

- [ ] **Step 5: Implement interruption handling**

Poll IMDS for rebalance/interruption notices. On notice:

- signal both rank-zero processes to write checkpoints;
- wait a bounded interval;
- hash and upload both checkpoints and a paired receipt;
- terminate with a distinct resumable exit code.

Never label an unverified upload resumable.

- [ ] **Step 6: Verify**

```bash
pytest -q tests/test_aws_p5_launcher.py tests/test_ddp_trainer.py
```

- [ ] **Step 7: Commit**

```bash
git add cluster/aws/p5 tests/test_aws_p5_launcher.py
git commit -m "feat: launch matched P5 seed pairs"
```

---

### Task 6: Extend `msctl` to the P5 lifecycle

**Files:**
- Create: `msctl/aws_p5.py`
- Modify: `msctl/cli.py`
- Modify: `msctl/operations.py`
- Modify: `msctl/contracts.py`
- Modify: `tests/test_msctl.py`

**Interfaces:**
- Commands: `auth check`, `capacity check`, `env ensure`, `dataset ensure`,
  `dataset verify`, `runs instantiate`, `runs render`, `submit`, `status`,
  `resume`, `cancel`, `evaluate`, `collect`, `cleanup`
- AWS backend: S3 + SSM + EC2 APIs through AWS CLI JSON output

- [ ] **Step 1: Add provider dispatch tests**

Mock process boundaries, not internal functions. Assert dry-run default and one
strict stdout JSON object. Reject ambient credentials in rendered commands,
unknown AWS output fields, wrong instance tags, and duplicate active seed IDs.

- [ ] **Step 2: Add seed ownership tests**

Every mutating AWS operation must reject seed 0. Every Illumina mutating
operation must reject seeds 1–4.

- [ ] **Step 3: Instantiate run manifests after release publication**

Add:

```text
msctl runs instantiate --release RELEASE.json \
  --dataset-receipt dataset-receipt.json --seed N --out runs-sN.json --apply
```

This is the only run-manifest creation path. It resolves the provider-owned
seed's two checked-in configs, verifies their hashes and Dense/Split90
invariants, then binds the now-known release, dataset, cohort-assignment,
study-lock, and config SHA-256 values. Illumina accepts only `N=0`; AWS accepts
only `N` in `1..4`. Dry-run returns the exact canonical manifest and hash
without writing. Apply uses atomic no-replace publication. No manifest may
bind more than one seed pair.

- [ ] **Step 4: Implement AWS backend**

Use argument arrays with `env -i`; never invoke a shell. Require:

- region;
- exact instance ID and `p5.48xlarge` type;
- cohort/release/dataset hashes in EC2 tags;
- SSM online state;
- instance profile ARN;
- S3 durable prefix;
- explicit approval for paid launch, resume, cancel, evaluate, and cleanup.

- [ ] **Step 5: Implement paired checkpoint resume**

Pass `--resume-path` and `--resume-sha256` to `scripts/run_train.py` for each
arm. Require world size 4, matching seed/arm/config/corpus/code, and both
checkpoint hashes before mutation.

- [ ] **Step 6: Verify**

```bash
pytest -q tests/test_msctl.py tests/test_aws_p5_launcher.py
```

- [ ] **Step 7: Commit**

```bash
git add msctl tests/test_msctl.py
git commit -m "feat: manage AWS P5 runs with msctl"
```

---

### Task 7: Package the AWS seeds 1–4 handoff

**Files:**
- Create: `scripts/package_aws_p5_handoff.py`
- Create: `tests/test_package_aws_p5_handoff.py`
- Create: `AWS-P5-START.md`

**Interfaces:**
- Produces: `ms-aws-p5-r1-<digest>.zip`, checksum, and `RELEASE-AWS-P5.json`

- [ ] **Step 1: Write deterministic package tests**

Assert the archive contains exactly eight run configs:

```python
assert config_names == {
    f"configs/360m-v2/{arm}-s{seed}.yaml"
    for seed in (1, 2, 3, 4)
    for arm in ("dense", "split90")
}
```

Assert seed 0 is absent. Rebuild twice and compare bytes. Test secret scanning,
symlink rejection, unknown tracked path rejection, no-replace publication,
provider metadata, and executable modes.

- [ ] **Step 2: Run and confirm red**

```bash
pytest -q tests/test_package_aws_p5_handoff.py
```

- [ ] **Step 3: Implement the closed allowlist**

Include shared source, contracts, evaluator, tests, AWS profile/runtime,
dataset pointer, environment lock, and seeds 1–4 configs. Exclude Illumina
Slurm scripts, seed-0 configs, data, artifacts, docs history, outputs, caches,
logs, checkpoints, and credentials.

- [ ] **Step 4: Add semantic release metadata**

Bind:

- source commit;
- cohort-assignment SHA-256;
- seeds `[1,2,3,4]`;
- arms `["dense","split90"]`;
- profile SHA-256;
- environment SHA-256;
- every member hash;
- package format version.

- [ ] **Step 5: Verify**

```bash
pytest -q tests/test_package_aws_p5_handoff.py \
  tests/test_package_illumina_handoff.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/package_aws_p5_handoff.py \
  tests/test_package_aws_p5_handoff.py AWS-P5-START.md
git commit -m "feat: package AWS P5 seeds one through four"
```

---

### Task 8: Make sealed confirmatory evaluation executable

**Files:**
- Create: `evals/confirmatory/__main__.py`
- Create: `evals/confirmatory/runner.py`
- Create: `tests/test_confirmatory_runner.py`

**Interfaces:**
- CLI:
  `python -m evals.confirmatory evaluate --run RUN --sealed-release RELEASE --expected-study-lock-sha256 HASH --device cuda`
- Produces: hash-bound submissions, replayed outcomes, metrics, inference
  evidence, and artifact report

- [ ] **Step 1: Write CLI and replay tests**

Use a deterministic tiny model adapter and sealed fixture. Assert the runner:

- verifies the external study-lock hash before opening model-visible items;
- checks checkpoint/config/corpus/code/seed/arm bindings;
- keeps sealed gold outside the model adapter;
- records submitted answer plus exactly 12 action slots;
- derives correctness only through the trusted solver;
- writes canonical outcomes, metrics, inference, and report artifacts;
- refuses wrong study lock, malformed submission, missing item, duplicate item,
  generic `split`, or invalid checkpoint binding.

- [ ] **Step 2: Run and confirm red**

```bash
pytest -q tests/test_confirmatory_runner.py
```

- [ ] **Step 3: Implement the runner**

`runner.py` owns orchestration only. Reuse strict contracts, solver replay,
metrics, and reporting; do not duplicate their validation. Inject a
`ModelAdapter.generate(item) -> Submission` interface so tests need no GPU and
the production adapter can load the repository GPT checkpoint.

- [ ] **Step 4: Implement the strict CLI**

Emit one JSON object to stdout. Send logs to stderr. Default to dry-run unless
`evaluate` receives an explicit output directory and all trust roots. Refuse
to overwrite any artifact.

- [ ] **Step 5: Verify**

```bash
pytest -q tests/test_confirmatory_runner.py tests/test_confirmatory_replay.py \
  tests/test_confirmatory_reporting.py tests/test_confirmatory_validation.py
```

- [ ] **Step 6: Commit**

```bash
git add evals/confirmatory/__main__.py evals/confirmatory/runner.py \
  tests/test_confirmatory_runner.py
git commit -m "feat: run sealed confirmatory evaluation"
```

---

### Task 9: Verify the two releases as one cohort

**Files:**
- Create: `scripts/verify_cohort_releases.py`
- Create: `tests/test_verify_cohort_releases.py`
- Create: `docs/AWS-P5-360M-RUNBOOK.md`

**Interfaces:**
- CLI:
  `verify_cohort_releases.py --illumina RELEASE.json --aws RELEASE-AWS-P5.json`

- [ ] **Step 1: Add cross-release tests**

Reject overlap, missing seed, different code commits, different cohort hashes,
different corpus/evaluation identities, partial arms, or duplicate run IDs.
Accept only `{0} ∪ {1,2,3,4} = {0,1,2,3,4}`.

- [ ] **Step 2: Implement verifier**

Hash both archives through pinned descriptors, verify internal `SHA256SUMS`,
parse both assignments, and emit one canonical JSON decision. Do not trust
release receipt counts without enumerating ZIP members.

- [ ] **Step 3: Write the operator runbook**

Include exact preflight, S3 layout, bootstrap, canary, launch, status, resume,
evaluation, evidence collection, and termination commands. Every paid command
appears first in dry-run form.

- [ ] **Step 4: Verify**

```bash
pytest -q tests/test_verify_cohort_releases.py \
  tests/test_package_illumina_handoff.py \
  tests/test_package_aws_p5_handoff.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_cohort_releases.py \
  tests/test_verify_cohort_releases.py docs/AWS-P5-360M-RUNBOOK.md
git commit -m "feat: verify split-provider cohort releases"
```

---

### Task 10: End-to-end local release gate

**Files:**
- Modify only files needed to correct failures found by this gate

- [ ] **Step 1: Run the complete repository suite**

```bash
pytest -q
```

Expected: every test passes; GPU-only tests may be explicitly deselected only
when their CPU/gloo contract equivalent passes.

- [ ] **Step 2: Run static checks**

```bash
python -m compileall -q msctl corpusgen evals train scripts cluster/aws
git diff --check
```

- [ ] **Step 3: Build both releases twice**

Build from one clean commit into four empty output directories. Verify
Illumina A equals Illumina B byte-for-byte and P5 A equals P5 B byte-for-byte.

- [ ] **Step 4: Run cross-release verification**

```bash
python scripts/verify_cohort_releases.py \
  --illumina dist/illumina/RELEASE.json \
  --aws dist/aws-p5/RELEASE-AWS-P5.json
```

Expected JSON: `ok=true`, Illumina seeds `[0]`, AWS seeds `[1,2,3,4]`,
complete cohort `[0,1,2,3,4]`.

- [ ] **Step 5: Independent review**

Request specification compliance and code-quality reviews. Fix every Critical
and Important finding, rerun the complete suite, and rebuild both ZIPs.

- [ ] **Step 6: Commit release-ready fixes**

```bash
git add -A
git commit -m "fix: close split-provider release gate"
```

---

## AWS P5 Operational Run Plan

### Capacity and image

Target `p5.48xlarge` in a region with confirmed quota and capacity. Default to
On-Demand or a capacity reservation. Use Spot only after paired interruption
and resume tests pass. Resolve and record the region-specific DLAMI ID and
immutable container digest before launch.

Authoritative hardware contract:

- 8× H100 80 GB;
- 192 vCPUs;
- 2 TiB RAM;
- 8×3.84 TB NVMe;
- 900 GB/s NVSwitch.

Source: `https://aws.amazon.com/ec2/instance-types/p5/`.

### Storage layout

```text
/mnt/memorysplit/
  release/       verified extracted P5 ZIP
  dataset/       verified token and sidecar shards
  runs/seed-N/
    dense/
    split90/
  staging/       owned atomic upload staging

s3://<bucket>/<cohort-id>/
  releases/
  dataset/
  checkpoints/seed-N/{dense,split90}/
  evaluations/seed-N/{dense,split90}/
  receipts/
```

Local NVMe is scratch only. S3 is authoritative for checkpoints and evidence.

### Bring-up sequence

1. Query quota, capacity, and current price.
2. Launch with an instance role scoped to one cohort S3 prefix.
3. Connect through SSM; require no inbound SSH.
4. Run bootstrap and inspect its receipt.
5. Stage and verify release, corpus, sidecars, and sealed evaluation.
6. Run readiness and two-rank functional canaries.
7. Run 100 updates at `4+4`; record median seconds/update and tokens/s/arm.
8. Derive ETA from measured throughput.
9. Launch one assigned seed pair.
10. Mirror paired checkpoints every 30 minutes.
11. Evaluate and collect before reusing the instance for the next seed.
12. Terminate only after S3 receipts independently rehash all outputs.

### Scheduling modes

**Deadline mode:** four `p5.48xlarge` instances, one preassigned seed each.
Seeds 1–4 finish in one training wave. This is fastest and avoids changing
manifests.

**Economy mode:** one `p5.48xlarge`; run seeds 1, 2, 3, and 4 sequentially.
This has nearly the same aggregate instance-hours but four times the wall time.

### Provisional time and cost envelope

No production H100 benchmark exists. The 100-update canary is authoritative.
For planning only:

- 600k targets/s/arm: about 3.3 training hours per seed;
- 400k targets/s/arm: about 5.0 training hours per seed;
- 250k targets/s/arm: about 7.9 training hours per seed.

Allow an additional 0.75–1.5 hours per seed for staging, checkpoint upload, and
sealed evaluation. Four parallel instances therefore target roughly 4–10
hours wall time after data readiness; one sequential instance targets roughly
16–38 hours.

Public pricing trackers listed `p5.48xlarge` in `us-east-1` at approximately
`$55.04/instance-hour` on 2026-07-23, but the operator must query the AWS
Pricing API immediately before approval. At that provisional rate, 16–35
aggregate instance-hours are approximately `$880–$1,930`, excluding storage,
data transfer, and idle capacity. Do not encode this price in an approval
receipt.

### Terminal analysis

Combine Illumina seed 0 with AWS seeds 1–4 only after cross-release and
evaluation verification. Run the frozen exact one-sided exhaustive sign-flip
test over five paired seed deltas. `supports_effect` is limited to the narrow
primary omnibus; N=5 cannot establish separate superiority in both reasoning
families.
