# AWS GPU Runtime Build Report

## Status

DONE on `feat/aws-gpu-runtime-build`, based exactly on
`b3471e0969ca2a997d33acf60d2e777720afa1c4`. The original implementation is
`c70f4cead3b1c921bfac931ac9033ef29f8b7d36`; this report includes the
pinned-runtime review fixes in `213bf0b4ee69dac6b9997163f75740019d40c274`
and second re-review fixes in
`bee8a739af2ae3737babfdf9fa557e4e5dfca09a`. The final root-inventory fix is
layered on that commit.

## Delivered

- Added one operator-side runtime tree at `containers/aws-gpu/`; no
  `runtime/aws-p5/` tree was created.
- Pinned the sole Docker `FROM` to
  `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training@sha256:1414a836532f22b271c03b7ccdbdff3daa0591975b3bd9a3cf51601a45b37f4f`.
- Added fixed UID/GID `10001:10001`, a closed Docker context, binary-only
  `pip --require-hashes --no-deps --force-reinstall` installation, a pip
  install report, and no remote `ADD`, package-manager mutation, or fetch-pipe.
- Resolved every pip-installed project dependency and transitive dependency
  through the package index for CPython 3.12 / x86_64 manylinux 2.28. The
  checked-in lock contains exact versions and real SHA-256 distribution hashes.
  PyTorch is supplied by the digest-pinned DLC, excluded from pip installation,
  and checked as public version `2.9.0` during the image build.
- Added a closed private-ECR build renderer. Dry-run is the default; only exact
  `--apply` executes Docker. It verifies the actual clean Git commit and source
  tree, descriptor-pins all build inputs, rehashes them around every command,
  hashes every Git/Docker command transcript, and rechecks the repository
  before emitting authority.
- The apply path measures the base and local image before push, then reruns
  image inspection, framework measurement, and complete software inspection
  through the final pushed digest. Framework probes run explicitly as
  `10001:10001` with GPU access; all probes use `--network none`, `--pull
  never`, a read-only root, and the exact digest. Python 3.12, PyTorch 2.9.0,
  CUDA 13.0, cuDNN, NCCL, and inherited entrypoint/CMD must remain exact.
- Every framework and inventory probe uses `/opt/conda/bin/python`. Apply
  separately proves that executable exists in the base, local, and final
  digest images; `/usr/bin/python3` cannot supply container evidence.
- Added the immutable host candidate
  `ami-0260c4d597dcc8641` / owner `898082745236`, its supplied CUDA 13.2,
  driver 595.71.05, kernel 6.17, EFA 1.47.0, and OFI-NCCL 1.18.0 facts, and the
  official P6-B300 version floors.
- Added explicit attestation paths. `attest_environment` and
  `attest_legacy_p5_environment` remain the byte-compatible legacy P5 v2 path.
  `attest_selected_gpu_environment` is the neutral integration interface for a
  loaded P5/P6 profile. It never imports host Torch: framework facts are
  measured inside the exact local digest-pinned GPU container; driver, Fabric
  Manager, Docker, NVIDIA runtime, AWS CLI, CUDA toolkit, kernel, EFA, and
  OFI-NCCL remain host measurements.
- The selected-profile evidence contract rejects P5/P6 mixing, requires eight
  identical approved H100 or B300 identities, and independently checks the
  official P6-B300 CUDA 13.0, R580, kernel 6.1, EFA 1.44.0, OFI-NCCL 1.17.1,
  and NVLSM R580 floors.
  P6-B300 requires Base DLAMI `ami-0260c4d597dcc8641`, owner `898082745236`;
  framework DLAMI `ami-0b39828e6910b0bb8` is rejected.
- P6 evidence measures `/usr/bin/nvlsm --version` and requires the driver,
  Fabric Manager, and NVLSM branches to agree. Missing, below-floor, and forged
  NVLink-manager evidence fail closed.
- One shared launcher-facing product-name contract permits only
  `NVIDIA H100 80GB` and `NVIDIA H100 80GB HBM3` for P5, and only
  `NVIDIA B300` for P6. Prefixes, arbitrary suffixes, and mixed names fail.
- Replaced the dependency-only “SBOM” with a closed tool-produced inspection
  contract. It binds base/final image digests, OS release, Python runtime,
  complete installed Python inventory (including inherited DLC packages),
  RECORD/WHEEL metadata hashes, exact selected project archive hashes from the
  pip report, dependency-lock provenance, build/source/command provenance,
  inherited entrypoint, and runtime-lock hash. Runtime-lock bytes still
  round-trip through the current `parse_runtime_lock_bytes`.
- Inspection schema v2 inventories every fully installed dpkg package, every
  dpkg database file, every package-owned path, every Python distribution, and
  every RECORD-listed path with type-aware content commitments. It records
  exact METADATA, RECORD, and WHEEL file hashes. Project wheels carry the exact
  pip-report archive hash; inherited DLC distributions carry the immutable
  base-image digest and installed-file commitments, never a null or mislabeled
  archive hash.
- Only the full inventory probe runs as `0:0`. It has no GPU, network, mount,
  or secret access; sets no-new-privileges; drops every capability; and adds
  only `DAC_READ_SEARCH` so root-only dpkg/database/package files are readable.
  Framework facts are passed in from the separately measured nonroot probe,
  and the inspector emits only canonical stdout evidence.

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
- Host/container separation RED: 1 failed; the review fix now derives the five
  framework lock fields only from measured container evidence and the five
  host lock fields only from measured/pinned host evidence.
- Review attestation RED: 8 failed for the absent selected-GPU API; GREEN: 8
  passed, followed by the complete 74-test legacy/selected attestation file.
- Apply provenance RED: 11 failed for absent Git identity, descriptor pinning,
  measured container flow, and inspection contract; GREEN: 12 passed.
- Forced reinstall/report RED: 1 failed; GREEN after replacing
  `--ignore-installed` with deterministic `--force-reinstall`.
- Full SBOM RED: 6 failed because the producer accepted operator container
  claims and lacked image-binding input; GREEN: 8 focused producer/build tests.
- Independent evidence-floor and inherited-entrypoint bindings each had one
  focused RED followed by GREEN.
- Closed SBOM parser, supported-host pin, explicit full-inventory method,
  pre-apply repository check, and final local image-ID verification each had a
  focused RED followed by GREEN.
- DLC Python RED: selected attestation failed because no exact interpreter
  existence/probe API existed; GREEN: 11 focused build/inspection/attestation
  tests. The pinned inspector shebang had its own RED then GREEN.
- H100-name RED: 2 failures for the absent shared allowed-name API; GREEN: 7
  acceptance, prefix, mixing, parser, and cross-profile tests.
- NVLSM RED: 3 failures for absent evidence/floor handling; GREEN: 9 measured,
  below-floor, missing, forged, and parser tests.
- SBOM v2 RED: 6 failures for absent dpkg/file/provenance closure; GREEN: 6
  focused inspection/build/runtime tests. A forged project archive had an
  independent RED then passed after lock cross-checking.
- Root-inventory RED: the exact command still inherited the image user instead
  of the restricted root boundary. GREEN: 5 new root/nonroot and
  extra-network/capability/mount rejection cases, plus the existing inspection
  and apply checks.

## Verification

```text
392 passed in 49.83s
```

The focused command covered `tests/test_aws_gpu_runtime.py`,
`tests/test_aws_environment_receipt.py`, `tests/test_aws_contract_roundtrip.py`,
`tests/test_aws_canary.py`, `tests/test_aws_p5_profile.py`, and
`tests/test_aws_p5_launcher.py`.

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
- `containers/aws-gpu/inspect_container.py`
- `containers/aws-gpu/host-candidate.json`
- `containers/aws-gpu/runtime_lock.py`
- `tests/test_aws_gpu_runtime.py`
- `.superpowers/sdd/aws-gpu-runtime-report.md`

Modified for the review fix:

- `cluster/aws/p5/attest_environment.py`
- `msctl/aws_contracts.py`

## Deliberate Boundaries

- No Docker build, tag, push, image pull, or live image inspection occurred.
- No AWS API, ECR resource, profile, capacity, CloudFormation, launcher,
  canary, lifecycle, package, config, corpus, or evaluation code was changed
  or invoked.
- The older approved plan named PyTorch 2.12.1; the explicit task instruction
  supplied and therefore overrides it with the immutable PyTorch 2.9.0 digest.
- A private-ECR image digest and real inspection artifact can exist only after
  a separately approved `--apply`; fixtures exercise that boundary offline.
- The selected-GPU attestation API is intentionally not wired into the current
  P5 launcher/controller. That integration belongs to the dual-profile branch;
  this branch exposes and tests the profile-compatible interface without
  claiming lifecycle completion.
- EFA and OFI-NCCL measurement uses AWS's documented default files
  `/opt/amazon/efa_installed_packages` and
  `/opt/amazon/ofi-nccl/lib/libnccl-net.so`. Their exact presence remains a live
  Base-DLAMI qualification check.
- `/usr/bin/nvlsm` and the complete dpkg/Python inventories are fixture-tested
  offline; their real Base-DLAMI output remains part of deferred qualification.
- The restricted root command and nonroot framework command are argv-verified
  offline; no privileged container was executed in this task.
