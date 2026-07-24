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

## Review remediation (2026-07-24)

This section supersedes the initial Capacity-Block minimum, profile hashes, and
unanchored authority statements above. The original section is retained as the
reviewed history.

### Critical and Important findings closed

- The sole local authority is now
  `memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json` relative
  to its control root. The sole versioned-store key is
  `cohorts/memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json`.
  No public arbitrary-path writer remains.
- `publish_provider_selection` is the injected production seam. It always uses
  `If-None-Match: *` at the fixed key, requires exact checksum/length/version
  HEAD confirmation, recovers a lost PUT response only through exact HEAD, and
  rejects conflicting local or remote bytes before allowing another profile.
- Public dataclasses are data-only. Selection publication, local loading,
  amendment-file verification, and resume validation re-open fixed canonical
  bytes themselves. Forged amendment/receipt objects and claimed hashes cannot
  become authority.
- P6-B300 now uses the supported minimum path: On-Demand in `us-east-1d`.
  Capacity Block is not implicit or required.
- The P6 profile freezes only official software minima: CUDA 13.0, NVIDIA
  driver R580, NVLink R580, kernel 6.1, EFA 1.44.0, and OFI-NCCL 1.17.1.
  Selection parses canonical runtime-lock and runtime-evidence bytes, binds
  their hashes, verifies profile/AMI/image/placement identity, and rejects
  attested versions below any floor. No measured runtime values were invented.
- Profiles, amendments, runtime locks, runtime evidence, and local selection
  authorities use bounded descriptor-pinned `O_NOFOLLOW` reads with owner,
  mode, single-link, regular-file, pre/post descriptor, and final-path identity
  checks. Hardlinks, unsafe modes, symlinks, and path replacement fail closed.
- Legacy `AwsP5Profile` positional and keyword constructor order is restored;
  neutral architecture/location/software fields have safe defaults.

### Corrected contract hashes

- Preregistration (unchanged):
  `6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7`
- Cohort assignment (unchanged):
  `47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c`
- P5 profile (unchanged):
  `2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543`
- P6 profile:
  `6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4`
- Hardware amendment:
  `d4cf13b587c751d27756ad7881e538facb7ea79305a098990a568a7b28b6fb14`

### Remediation RED/GREEN evidence

- P6 minimum/floors: RED had two stale-contract failures; GREEN passed 21
  focused P6/amendment/selection cases.
- Legacy constructors: RED had two direct-constructor failures; GREEN passed
  positional, keyword, runtime, and P6 compatibility cases.
- Secure reads: RED had five accepted hardlink/mode/owner/TOCTOU cases; GREEN
  passed all five. The existing symlink regression then caught an error-message
  compatibility issue and passed after the `ELOOP` correction.
- Runtime bytes/floors: RED had 15 missing authority API failures; GREEN passed
  exact P5/P6 locks plus profile, AMI, image, evidence, placement, and six floor
  mutation cases.
- Fixed authority/forgery: RED had 12 missing or unanchored authority failures;
  GREEN passed fixed local/S3 publication, lost-PUT recovery, exact HEAD,
  conflict, data-only, forged-object, private-file, load, and resume cases.
- A final RED showed a pre-existing local selection could leak a conflicting
  choice to an empty remote store; GREEN added secure local preflight before
  remote publication.

### Remediation verification

```text
tests/test_aws_hardware.py tests/test_aws_p5_profile.py
tests/test_cohort_assignment_v3.py
=> 182 passed (the original 141 plus 41 review regressions)

tests/test_aws_contract_roundtrip.py tests/test_run_manifest_v3.py
tests/test_aws_environment_receipt.py tests/test_aws_canary.py
tests/test_aws_p5_launcher.py
=> 347 passed
```

No live AWS API, S3, EC2, Capacity Block, or paid-capacity mutation was
performed; versioned-store behavior used only the injected in-memory fake.

## Second re-review remediation (2026-07-24)

This section supersedes the prior self-asserted runtime-evidence and
test-only-store integration statements. All previously closed P5/P6 profile,
On-Demand, secure-read, fixed-key, and constructor findings remain closed.

### Production authority integration

- `AwsCliVersionedSelectionStore` is the production AWS CLI adapter. It uses
  the repository's injected `(argv, environment, timeout)` runner pattern and
  never creates an ambient boto client.
- Before PUT, the adapter calls `list-object-versions` for the sole fixed key.
  Any historical object version or delete marker blocks publication, including
  delete/recreate cases where current HEAD is absent.
- PUT always uses `If-None-Match: *` and checksum SHA-256. Publication then
  HEADs the exact returned version and requires exact key, checksum, byte
  count, and non-null version ID.
- The selected version ID is persisted at
  `memorysplit-confirmatory-v3-360m-n10-aws/provider-selection-version.json`.
  Replay/admission performs `get-object --version-id` and independently checks
  returned version, checksum, byte count, remote bytes, and fixed local bytes.
- `AuthenticatedSelectionBinding` is the sole launch/resume admission output.
  `admit_provider_selection` and the hardened
  `validate_resume_hardware_binding` re-verify exact remote version,
  environment/canary qualification, account, instance, boot, profile,
  runtime, seed, and arm without accepting caller-constructed authority
  objects.

### Authenticated qualification evidence

- Selection no longer trusts a JSON map because its hash matches. The
  qualification-evidence document embeds canonical environment and canary
  receipts plus an approval that signs their exact identity/fact scope.
- The environment receipt binds account, instance, boot, profile, AMI, image
  digest, runtime-lock hash, source, and measured container facts. Its AWS
  instance identity PKCS7 is verified through the repository's pinned AWS
  identity-certificate verifier.
- The passed qualification canary binds the same instance/boot/profile/runtime
  tuple, all six canary phases, measured host facts, and the environment
  receipt hash.
- The approval scope commits the environment/canary hashes, host/container
  fact hashes, account, instance, boot, profile, runtime, AMI, and image. A
  separately trusted public-key SHA-256 must match before an injected
  cryptographic verifier can accept the signature.
- `OpenSslQualificationApprovalVerifier` is the production local verifier for
  `RSASSA_PSS_SHA_256`; tests inject deterministic cryptographic-verifier
  fixtures. Forged, unsigned, wrong-key, wrong-instance, wrong-boot,
  wrong-profile, wrong-runtime, wrong-AMI, and wrong-image evidence fails.
- The P6 `nvlink: R580` floor is retained. Its code provenance states the
  official AWS P6-B300 requirements, including `NVLINK 5 R580` alongside CUDA
  13.0, driver R580, kernel 6.1, EFA 1.44.0, and OFI-NCCL 1.17.1.

### Executable dry-run/apply path

The standalone CLI is executable without touching the shared Task 3C
`msctl/aws_p5.py` lifecycle:

```text
python -m msctl.aws_hardware publish \
  --repo-root <release-root> \
  --authority-root <private-authority-root> \
  --runtime-lock <runtime-lock.json> \
  --qualification-evidence <qualified-runtime-v2.json> \
  --selection <provider-selection.json> \
  --bucket <versioned-authority-bucket> \
  --region us-east-1 \
  --approval-public-key <approval-public-key.pem> \
  --approval-public-key-sha256 <trusted-sha256>
```

The command is dry-run by default: it parses and cryptographically verifies all
anchored bytes but performs no AWS or local authority mutation. Add `--apply`
explicitly to list history, PUT/HEAD the fixed key, and install the fixed local
selection/version authorities.

The exact later controller integration call is
`admit_provider_selection(...)` with the fixed authority/release roots, runtime
lock and qualification-evidence paths, `AwsCliVersionedSelectionStore`, target
account/instance/boot/seed/arm, persisted selection version ID,
`verify_aws_instance_identity_pkcs7`,
`OpenSslQualificationApprovalVerifier`, and the separately trusted public-key
SHA-256. It returns only `AuthenticatedSelectionBinding`.

### Second re-review RED/GREEN evidence

- Production S3 adapter/history/version replay: RED had four missing or
  non-blocking authority failures; GREEN passed exact CLI argv, historical
  version, delete-marker, persisted-version, and exact-GET cases.
- Authenticated qualification: RED had twelve missing provenance/verifier
  failures; GREEN passed PKCS7/approval verification and every forged identity,
  runtime, image, key, and signature mutation.
- CLI/admission: RED had seven missing executable/admission failures; GREEN
  passed dry-run/apply, production outputs, exact-version admission, wrong
  identity/version rejection, and OpenSSL commitment checks.
- Resume admission then produced one RED for accepting no exact store/version;
  GREEN now routes resume validation through exact authenticated admission.

### Second re-review verification

```text
tests/test_aws_hardware.py tests/test_aws_p5_profile.py
tests/test_cohort_assignment_v3.py
=> 205 passed

tests/test_aws_contract_roundtrip.py tests/test_run_manifest_v3.py
tests/test_aws_environment_receipt.py tests/test_aws_canary.py
tests/test_aws_p5_launcher.py
=> 347 passed

python -m py_compile cluster/aws/gpu_profile.py cluster/aws/p5/profile.py
  msctl/aws_hardware.py tests/test_aws_hardware.py
git diff --check
frozen scientific/profile/amendment diff assertions
=> passed
```

No live AWS command or paid-capacity mutation was performed. AWS behavior was
exercised only through injected runners and in-memory versioned stores.
