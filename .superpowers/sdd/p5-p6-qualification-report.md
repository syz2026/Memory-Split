# Profile-Driven P5/P6 Qualification Report

## Status

DONE on `feat/aws-p5-p6-profile-qualification`, based exactly on
`701eec85367b3b84f4265888a6028d4b9916d14c`.

## Reviewed runtime integration

The four review-clean runtime patches were cherry-picked in the requested
order. Stable patch IDs match their source commits exactly:

- `c70f4ce` -> `f6b9411`
- `213bf0b` -> `359081c`
- `bee8a73` -> `2c787fc`
- `cd8bbb4` -> `09e7366`

Git applied all four without a conflict, so no overlapping constant required a
manual resolution.

## Delivered

- Added selected-profile contracts that re-enter fixed provider-selection
  authority and derive the authenticated profile and both arm scopes before
  binding hardware amendment, provider-selection version, runtime, account,
  instance, boot, AMI, image, and measured facts.
- Added closed canonical parsers for the selected environment, bootstrap, and
  qualification receipts. Profile, runtime, selection, placement, open-field,
  and P5/P6 cross-authorization drift fail closed.
- Kept `attest_environment`, `attest_legacy_p5_environment`, bootstrap P5
  functions, legacy canary `RECEIPT_TYPE`, legacy phase order, and all legacy
  receipt schemas unchanged. New selected entry points are additive.
- Added selected environment orchestration over the reviewed host/container
  attestation and an injected selected-bootstrap hardware-reader seam.
- Enforced the reviewed P6 tuple: `p6-b300.48xlarge`, x86_64, 192 vCPUs,
  4096 GiB, eight exact `NVIDIA B300` devices, eight profile-sized 3.84-TB
  instance-store devices, the supported Base DLAMI/owner, exact digest-pinned
  image, and CUDA/driver/NVLSM/kernel/EFA/OFI floors.
- Added a digest-bound qualification worker for Python/PyTorch/CUDA/cuDNN/NCCL,
  BF16, SDPA forward/backward, `torch.compile`, fused AdamW, one-step training,
  and exact checkpoint/resume checks. Its bytes are part of the image-build
  inputs and runtime SBOM provenance.
- Rendered and verified simultaneous independent Dense and Split90 four-rank
  NCCL groups with disjoint profile-derived GPU sets, profile-derived CPU
  affinity, distinct rendezvous ports, and no eight-rank process group.
- Bound NVLink/topology evidence, exact-version S3 round-trip evidence,
  per-arm throughput and peak-memory telemetry, and projected pair/cohort ETA.
  Every measured phase has its own canonical evidence hash.

## Strict TDD evidence

- Environment/bootstrap contract RED: 9 expected missing-API failures; GREEN:
  9 passed.
- Selected canary RED: 10 expected missing-plan/execution failures; GREEN:
  20 passed.
- Runtime worker binding RED: missing worker/build-input failure; GREEN:
  worker and command-rendering tests passed.
- Selected orchestration RED: 2 expected missing-entry-point failures; GREEN:
  2 passed.
- Closed receipt parsers and authenticated hardware argv were each observed
  failing before their implementations, then passed.
- Final selected qualification suite: 24 passed.

## Verification

- Focused P5/P6 attestation, bootstrap/profile, canary, runtime, hardware, and
  contract regressions: **454 passed**.
- The first contract-roundtrip run was sandbox-blocked when fixtures invoked
  `git init`; the identical full command passed outside the sandbox.
- `py_compile`, stable patch-ID checks, exact-base/protected-scope checks, and
  `git diff --check` passed.

## Scope audit

No live Docker, AWS, S3, EC2, or paid-capacity action was run; tests use only
injected command, object-store, time, identity, and hardware fixtures. Corpus,
evaluation, scientific 360M configs, provisioning, shared Task 3C checkpoint
mirroring, and interruption-journal files are unchanged.

## Critical/Important review remediation

This section supersedes the initial caller-binding, shallow-SBOM, concurrency,
toy-throughput, and checkpoint claims above.

- Public selected environment, bootstrap, and canary APIs no longer accept an
  `AuthenticatedSelectionBinding` or selected profile. They take fixed
  authority/runtime/evidence paths and production store/crypto verifiers, call
  `admit_provider_selection` independently for Dense and Split90, require one
  cohort selection, and bind the exact seed plus both arm scopes. Caller-made
  bindings are reachable only through private testable helpers.
- Runtime SBOM admission now invokes the exact closed parser in the reviewed
  runtime module. Complete dependency, inherited-base, dpkg, Python package,
  host, image, lock, and command/input provenance is mandatory. NVLSM is a
  closed host fact/floor, and `qualification_worker_sha256` must equal the
  worker copied into the digest-pinned image.
- The controller starts both four-rank processes before polling either. It
  records PID, monotonic start/end, rendezvous, GPUs, CPUs, port, and exit
  status; requires positive measured overlap and disjoint resources; and
  derives `simultaneous_groups` from that evidence. A production `Popen`
  launcher and selected dry-run/apply CLI are included; tests inject fakes.
- The worker loads the exact frozen d360m config for each seed/arm: context
  1024, micro-batch 8, 524,288 tokens/update, 13,582 steps, compile enabled,
  and the named Dense/Split90 sidecar. It runs the repository Trainer against
  the bound corpus, derives throughput from real step token counts/timings,
  and requires distinct measured sidecar streams and target-weight statistics.
- Each group measures finite BF16 model output/loss, SDPA forward/backward,
  gradients, compiled-model execution, and fused AdamW. After 100 real updates
  it checkpoints model, optimizer, data cursor, and Python/NumPy/Torch/CUDA
  RNG, compares the next uninterrupted and resumed updates, and enforces model,
  optimizer, loss, step, and cursor equivalence at frozen `1e-6` tolerance.

### Remediation RED/GREEN evidence

- Authority re-entry/scope: missing cohort API, unsafe public signatures,
  exposed forged-binding builders, and dropped arm scope each failed before
  implementation; six focused authority tests passed after correction.
- Closed SBOM: eight shallow/open/missing provenance mutations were accepted
  or unsupported in RED; all eight fail closed in GREEN.
- Process concurrency/telemetry: six absent/sequential/overlap/resource cases
  failed in RED; measured overlap and telemetry tests pass in GREEN.
- Worker/CLI: reviewed geometry, non-toy group evidence, sidecar distinction,
  and production CLI arguments each failed before implementation.
- Hardened qualification/runtime suite: **87 passed**.
- Full requested regression superset: **479 passed** (the prior 454 plus 25
  review-remediation tests).
