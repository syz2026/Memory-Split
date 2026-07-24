# AWS GPU Runtime Build Report

## Status

DONE on `feat/aws-gpu-runtime-build`, based exactly on
`b3471e0969ca2a997d33acf60d2e777720afa1c4`.

## Delivered

- Added one operator-side runtime tree at `containers/aws-gpu/`; no
  `runtime/aws-p5/` tree was created.
- Pinned the sole Docker `FROM` to
  `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training@sha256:1414a836532f22b271c03b7ccdbdff3daa0591975b3bd9a3cf51601a45b37f4f`.
- Added fixed UID/GID `10001:10001`, a dependency-only Docker context,
  binary-only `pip --require-hashes --no-deps` installation, and no remote
  `ADD`, package-manager mutation, or fetch-pipe.
- Resolved every pip-installed project dependency and transitive dependency
  through the package index for CPython 3.12 / x86_64 manylinux 2.28. The
  checked-in lock contains exact versions and real SHA-256 distribution hashes.
  PyTorch is supplied by the digest-pinned DLC, excluded from pip installation,
  and checked as public version `2.9.0` during the image build.
- Added a closed private-ECR build renderer. Dry-run is the default; only exact
  `--apply` executes `docker build` followed by `docker push`. The renderer
  scrubs ambient credentials, binds Dockerfile/context-policy/dependency-lock
  hashes, rejects post-plan drift, and emits only the resulting digest-pinned
  private-ECR binding as authority.
- Added the immutable host candidate
  `ami-0260c4d597dcc8641` / owner `898082745236`, its supplied CUDA 13.2,
  driver 595.71.05, kernel 6.17, EFA 1.47.0, and OFI-NCCL 1.18.0 facts, and the
  official P6-B300 version floors.
- Added an offline deterministic runtime-lock/SBOM producer. Runtime-lock
  `versions` remain the current parser's host-attestation fields; container
  Python/PyTorch/CUDA/cuDNN/NCCL facts are separately represented in the SBOM.
  Produced lock bytes are canonical and are re-parsed by the existing
  `parse_runtime_lock_bytes` before release.

## TDD Evidence

- Docker/lock RED: 2 failed because the runtime tree was absent; GREEN: 2
  passed after adding the digest-pinned Dockerfile and index-resolved lock.
- Renderer RED: 7 failed because `build_image.py` was absent; GREEN: 9 passed.
- Runtime producer RED: 1 failed because `runtime_lock.py` was absent; GREEN:
  1 passed with deterministic parser-compatible bytes.
- Integrity RED: 4 failed for mutable Docker frontend/UID inputs, absent closed
  context, unbound build inputs, and same-path artifact writes; GREEN: 20
  passed.
- Binary-only install RED: 1 failed; GREEN: 1 passed after adding
  `--only-binary=:all:` and `--no-compile`.
- Host/container separation RED: 1 failed; GREEN: 1 passed after ensuring the
  current runtime lock contains only host-attestation versions while the SBOM
  records container versions independently.

## Verification

```text
112 passed in 22.89s
```

The focused command covered `tests/test_aws_gpu_runtime.py`,
`tests/test_aws_environment_receipt.py`, and
`tests/test_aws_contract_roundtrip.py`.

- Changed-file `python -m py_compile`: passed.
- `git diff --check`: passed.
- An offline `uv pip compile` rerun byte-matched `requirements.lock`.

## Files

Created:

- `containers/aws-gpu/Dockerfile`
- `containers/aws-gpu/Dockerfile.dockerignore`
- `containers/aws-gpu/requirements.in`
- `containers/aws-gpu/requirements.lock`
- `containers/aws-gpu/build_image.py`
- `containers/aws-gpu/host-candidate.json`
- `containers/aws-gpu/runtime_lock.py`
- `tests/test_aws_gpu_runtime.py`
- `.superpowers/sdd/aws-gpu-runtime-report.md`

## Deliberate Boundaries

- No Docker build, tag, push, image pull, or live image inspection occurred.
- No AWS API, ECR resource, profile, capacity, CloudFormation, launcher,
  attestation, canary, lifecycle, package, config, corpus, or evaluation code
  was changed or invoked.
- The older approved plan named PyTorch 2.12.1; the explicit task instruction
  supplied and therefore overrides it with the immutable PyTorch 2.9.0 digest.
- A private-ECR image digest and measured container patch-level facts can exist
  only after a separately approved `--apply` build/push and inspection. The
  producer intentionally requires those exact values rather than inventing
  them.
