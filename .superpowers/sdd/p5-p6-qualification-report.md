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

- Added a side-effect-free selected-profile contract that requires an
  `AwsGpuProfile` and `AuthenticatedSelectionBinding`, then binds the hardware
  amendment, provider-selection hash and exact version, profile, runtime lock,
  runtime SBOM, prior qualification/approval evidence, account, instance,
  boot, region, availability zone, AMI, image, and measured facts.
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
