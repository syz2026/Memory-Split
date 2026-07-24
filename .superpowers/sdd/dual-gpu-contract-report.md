# Prospective P5/P6 Hardware-Selection Contract Report

## Status

DONE on `feat/aws-p5-p6-hardware-contract`, based exactly on
`b3471e0969ca2a997d33acf60d2e777720afa1c4`.

## Delivered

- Added a neutral, closed `AwsGpuProfile` loader for the unchanged legacy and
  v3 P5 profiles plus the new P6-B300 profile. Legacy P5 class, runtime, parser,
  loader, constants, and runtime-validation names remain aliases.
- Added `aws-p6-b300.48xlarge-v3.json`, freezing x86_64, 192 vCPUs, 4096 GiB
  RAM, eight NVIDIA B300 GPUs, symmetric 4+4 groups, eight 3.84-TB NVMe
  devices, Capacity Block purchase, and the `us-east-1d` offering. AMI and
  image values remain solely runtime-lock inputs.
- Added the append-only hardware amendment. It byte-binds the frozen v3
  preregistration, cohort assignment, and exact P5/P6 profiles; permits only
  provider-assignment and hardware-topology supersession; and retains seeds
  0..9, Dense/Split90, 4+4 symmetry, all non-provider scientific fields, and
  zero inspected protected outcomes.
- Added canonical provider-selection receipts with strict duplicate/open-field
  rejection, exact amendment/profile/runtime/AWS/cohort bindings, UTC
  timestamps, zero outcomes, atomic no-replace publication, and explicit
  resume/run binding validation. Cross-profile, cross-runtime, cross-placement,
  cross-seed, and cross-arm mixing fail closed.

## Frozen hashes

- Preregistration:
  `6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7`
- Cohort assignment:
  `47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c`
- P5 profile:
  `2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543`
- P6 profile:
  `f22ccf259e30b07b7ad9d848723ad10e59092cbf5ea6e3aceca7a556279ce681`
- Hardware amendment:
  `9d6bbaedfe2520bd6ce10957e2c8e923a624144c0feb4b19c3764f71040be755`

## TDD evidence

- Profile/amendment RED: `tests/test_aws_hardware.py` produced 26 expected
  failures for the absent neutral module, P6 profile, amendment, and parser.
- Profile/amendment GREEN: new contract plus existing P5 profile/cohort tests
  passed, 107 tests at that checkpoint.
- Selection RED: 33 expected failures for absent provider-selection APIs.
- Selection GREEN: all 33 selection, no-replace, and mixing tests passed.
- A compatibility regression test first reproduced an unintended legacy P5
  region restriction, then passed after leaving provider location authority in
  the new selection contract.

## Final verification

```text
tests/test_aws_hardware.py tests/test_aws_p5_profile.py
tests/test_cohort_assignment_v3.py
=> 141 passed

tests/test_aws_contract_roundtrip.py tests/test_run_manifest_v3.py
tests/test_aws_environment_receipt.py tests/test_aws_canary.py
tests/test_aws_p5_launcher.py
=> 347 passed

python -m py_compile cluster/aws/gpu_profile.py cluster/aws/p5/profile.py
  msctl/aws_hardware.py tests/test_aws_hardware.py
git diff --check
frozen-file SHA-256 assertions
=> passed
```

The first downstream run was sandbox-blocked when its fixtures invoked
`git init`; the identical suite passed outside that restriction.

## Scope audit

Only the neutral profile layer, P5 compatibility shim, P6 profile, append-only
amendment, hardware-selection module, focused tests, and this report changed.
Frozen preregistration/cohort bytes and Task 3C, launcher, shared lifecycle,
runtime-container, canary, Task 3D, evaluation, IaC, corpus-generation, and
package-script files remain unchanged. No live AWS or paid-capacity operation
was performed.
