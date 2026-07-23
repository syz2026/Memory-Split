# Task 3A Report: Authenticated AWS Environment Attestation

## Status

Complete on `feat/memorysplit-v3-aws-n10`, based on
`d21aa16a770cf1b75a5d371fe469aedfd94ce10f`.

Implementation commit:

- `dd7ca713daec40b4259562085f494216c935f750` —
  `feat: add authenticated AWS environment attestation`

## Delivered

- Added `cluster/aws/p5/attest_environment.py`, a dependency-light,
  argv-only instance producer that:
  - parses the complete frozen v3 profile;
  - accepts only the canonical closed runtime-lock schema;
  - obtains IMDSv2 identity and PKCS7 evidence;
  - measures the boot ID, exact local image digest, and ten pinned runtime
    versions;
  - rejects identity, hash, image, version, command, path, symlink, hardlink,
    and replacement drift;
  - supports procfs zero-size virtual files such as the production boot-ID
    source; and
  - emits canonical v2 bytes, with owner-only atomic no-replace publication.
- Replaced the unreachable AWS `env_ensure` implementation with v3 controller
  behavior that verifies canonical receipt bytes, profile/lock/control/runtime
  bindings, IID PKCS7, selected instance and AMI facts, and then publishes the
  receipt content-addressed and no-replace to S3.
- S3 publication requires the expected SHA-256 checksum, exact metadata and
  length, and a real non-`null` version ID. Uploads use a private staged copy
  of the already-verified bytes.
- Preserved credential separation:
  - static AWS key variables fail closed locally;
  - local profile/SSO configuration and controller HOME reach only local AWS
    CLI argv; and
  - remote intents reject credential/config/profile sources.
- Required and propagated exact `MS_CONTAINER_IMAGE`, digest-matched to
  `MS_CONTAINER_DIGEST`, and rejected tagged or otherwise mutable image
  references.
- Enabled the frozen v3 profile only for `runs instantiate` and `env ensure`;
  submit/resume/evaluate remain disabled for v3.
- Added the producer and its runtime dependency as required v3 package members.
- Centralized the 18 receipt fields, 10 runtime-lock fields, and 10 runtime-fact
  fields so producer, release contract, controller, and package metadata cannot
  silently diverge.

## TDD and review

Tests were added before each implementation slice and observed failing for the
missing producer, controller path, CLI arguments, package contract,
`MS_CONTAINER_IMAGE` binding, credential separation, procfs boot-ID handling,
tagged images, floating versions, S3 `null` versions, and real Fabric Manager
output. Each slice was then made green.

A focused independent review found and prompted fixes for:

- the real `/proc/sys/kernel/random/boot_id` zero-size stat behavior;
- the `/usr/bin/nv-fabricmanager` executable and its labeled version output;
- local SSO HOME propagation;
- tagged digest references and floating version suffixes; and
- S3 versioning-suspended `"null"` version IDs.

The final package/runtime/receipt consistency check reported:

`receipt=18 lock=10 runtime_facts=10; shared contracts match`

## Verification

- Required Task 3A suite: **457 passed in 155.68s**
- Changed AWS launcher fixture suite: **135 passed in 5.82s**
- `python -m py_compile` for every changed Python file: **passed**
- `git diff --check`: **passed**
- Producer `python -S --help`: **covered and passed**

Required-suite command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3a \
  tests/test_aws_environment_receipt.py \
  tests/test_aws_argv.py \
  tests/test_aws_p5_profile.py \
  tests/test_package_aws_p5_handoff.py \
  tests/test_aws_contract_roundtrip.py \
  tests/test_run_manifest_v3.py \
  tests/test_msctl.py
```

## Concerns and deliberate boundaries

- No Task 3A blocker remains.
- The default reviewed PKCS7 trust anchor remains intentionally limited to
  `us-east-1`; other regions fail closed.
- Task 5 must provide the real runtime lock, immutable AMI/image identities,
  deployment/staging, and a versioned S3 bucket. No live AWS call was made in
  this task.
- The controller interface deliberately consumes an explicit produced receipt
  and returns the exact remote attestation plan; it does not broaden this task
  into deployment or IaC orchestration.

Report path:

`/Users/stephenzhang/Documents/MemorySplit/.worktrees/memorysplit-v3-aws-n10/.superpowers/sdd/task-3a-report.md`

## Review-fix follow-up

Status: complete.

Review-fix commit:

- `22c480d457f5cd7a2ac9c2eeda9f59c06ee5849d` —
  `fix: harden AWS environment attestation`

The follow-up closes every reported Task 3A boundary:

- Python runtime facts now use the exact `/usr/bin/python3` argv with `-I -P`,
  the minimal command environment, and the explicit root-owned,
  group/world-non-writable `/usr` working directory. Writable `platform.py`
  and `torch.py` attack fixtures can no longer supply matching lock facts;
  unsupported or missing trusted runtimes fail closed.
- Version validation rejects embedded and separator/case-obfuscated floating
  markers, including digit-bearing `latest`, `main`, `master`, `head`, `dev`,
  `nightly`, `snapshot`, `rolling`, and `unknown` forms, while preserving
  concrete Python, PyTorch/CUDA, driver, release-candidate, and CPU outputs.
- `msctl.aws_contracts.validate_digest_pinned_oci_image` is now the one shared
  pure validator used by the receipt producer/parser, profile runtime
  validation, and remote argv validation. It requires one valid registry,
  nonempty canonical repository components, no tag, one exact lowercase
  SHA-256 digest, and exact image/digest agreement.
- V3 `env ensure` rejects legacy `--lock` and `--release` before backend
  construction. Existing non-v3 `env ensure --lock` behavior remains covered
  by the unchanged legacy suite.

TDD evidence:

- RED attestation slice: **11 failed, 9 passed** (probe isolation, shadow
  attacks, floating markers, malformed OCI paths).
- RED remote OCI slice: **6 failed, 4 passed**.
- RED profile OCI slice: **3 failed, 16 passed**.
- RED v3 CLI slice: backend-construction sentinel failed as expected.
- GREEN focused slices: **25 passed**, **10 passed**, **19 passed**, and
  **1 passed**, respectively.

Fresh verification:

- Required Task 3A suite: **497 passed in 140.09s**.
- Changed AWS launcher suite: **135 passed in 6.38s**.
- `python -m py_compile` for every changed Python file: **passed**.
- `git diff --check`: **passed**.
- Package/runtime/receipt consistency:
  `receipt=18 lock=10 runtime_facts=10; shared contracts match`.

Follow-up concerns:

- No Task 3A blocker remains.
- The frozen v3 runtime lock requires a Python supporting `-P`; an older
  `/usr/bin/python3` fails closed rather than weakening probe isolation.
- No live AWS call was made; the trust-anchor and Task 5 provisioning
  boundaries recorded above remain unchanged.
