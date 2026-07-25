# AWS corpus builder Task 6 report

## Status and scope

Implemented the NVMe bootstrap renderer and runtime, the hard systemd termination
watchdog, and focused tests. No CloudFormation file, package inventory, or file
outside the assigned Task 6 ownership set was modified. No AWS API or network
request was made during development or tests.

Owned implementation:

- `cluster/aws/corpus_builder/bootstrap.py`
- `cluster/aws/corpus_builder/bootstrap.sh`
- `tests/test_aws_corpus_builder_bootstrap.py`
- this report

## RED evidence

Initial contract RED:

```text
Command: python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
Exit: 2
Result: 1 collection error
Expected cause:
  ModuleNotFoundError: No module named 'cluster.aws.corpus_builder.bootstrap'
```

The failing contract was committed before production code:

```text
1065b0d0c11097d2a6eb2d92638bd1589032da37
test: define AWS corpus bootstrap contract
```

Self-review found that a failed timeout-marker upload could prevent the watchdog
from reaching shutdown. A focused regression test was added and observed RED:

```text
Command: python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py -k unconditional
Exit: 1
Result: 1 failed, 16 deselected
Expected cause:
  missing unconditional EXIT shutdown trap
```

After adding the unconditional trap and a service timeout longer than the
bounded marker-upload attempt, the same command produced:

```text
1 passed, 16 deselected
```

## GREEN implementation

`BootstrapConfig` reuses `S3ObjectVersion`. `render_bootstrap` validates both
records through the existing closed contract, binds both objects to the build
ID and KMS key, shell-quotes every rendered value, and emits deterministic
user data.

The runtime:

- selects only whole disks whose model is exactly
  `Amazon EC2 NVMe Instance Storage`;
- rejects non-four counts, repeated serials, root devices, mounts, child
  devices, filesystem signatures, RAID metadata, and active holders;
- creates four-device RAID0 at `/dev/md/memorysplit` with a 512 KiB chunk,
  formats XFS, and mounts with `noatime,nodiratime`;
- creates owner-only `work`, `output`, and `cleanroom` directories and records
  device serials plus the XFS UUID;
- enables the `23h30m` timer before any S3 bootstrap read;
- makes shutdown unconditional even when timeout-marker publication fails;
- seeds the approved package with an exact-version AWS CLI read, then uses
  Task 3 `download_exact_object` for both the package and source manifest;
- verifies hashes, authority metadata, JSON manifest framing, and safe archive
  members before the final `/opt/memorysplit` extraction;
- runs the fixed driver path under `env -i` with an explicit allowlist; and
- uploads the receipt and phase logs with SSE-KMS, then requests shutdown on
  success or failure.

Implementation commit:

```text
428ef1199104d8449e1d737a556b039a79c078ed
feat: bootstrap bounded NVMe corpus builds
```

## Verification evidence

Required focused verification after the final implementation change:

```text
python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
17 passed in 0.04s

python -m py_compile cluster/aws/corpus_builder/bootstrap.py
exit 0, no output

bash -n cluster/aws/corpus_builder/bootstrap.sh
exit 0, no output

git diff --check
exit 0, no output
```

Related no-network regression run:

```text
python -m pytest -q \
  tests/test_aws_corpus_builder_contracts.py \
  tests/test_aws_corpus_builder_s3.py \
  tests/test_aws_corpus_builder_bootstrap.py
101 passed in 0.27s
```

## Exact launch-template wiring contract

The foundation template must not duplicate bootstrap logic. Its immutable
launch-template version must use the output of this sequence:

```python
import base64
import gzip

rendered = render_bootstrap(
    BootstrapConfig(
        build_id=approved_build_id,
        package=approved_package_s3_object_version,
        source_manifest=approved_source_manifest_s3_object_version,
        kms_key_arn=approved_kms_key_arn,
        profile_sha256=approved_profile_sha256,
        launch_intent_sha256=approved_launch_intent_sha256,
        workers=approved_worker_count,
    )
).encode("utf-8")
user_data = base64.b64encode(
    gzip.compress(rendered, compresslevel=9, mtime=0)
).decode("ascii")
```

Assign `user_data` directly to `LaunchTemplateData.UserData`; it is already
base64 encoded and must not be wrapped in a second `Fn::Base64`. AL2023
cloud-init executes the gzip payload as the rendered shebang script. The
rendered script takes zero positional arguments because every approved value is
embedded and shell-quoted. Create and preflight a new explicit numeric launch
template version whenever any field changes. `RunInstances` must not override
user data.

Compression is mandatory for the EC2 16 KiB decoded user-data limit. The test
fixture rendered to 22,284 raw bytes, 6,075 deterministic gzip bytes, and 8,100
base64 bytes. Integration must reject a gzip payload over 16,384 bytes.

The launch template must retain
`InstanceInitiatedShutdownBehavior: terminate`, IMDSv2-only metadata, the
builder role, and no public IP. The AMI must provide `/usr/bin/aws` (CLI v2),
Python 3, systemd, mdadm, XFS tools, util-linux, and coreutils. IAM must permit
exact-version reads for both approved objects and SSE-KMS writes beneath
`v2/builds/{build_id}/operational/`.

## Self-review and integration concerns

- Scope review: only assigned implementation/test/report files changed; the
  untracked task brief remains untouched.
- Authority review: every bootstrap object read carries a version ID and
  SHA-256; no `aws s3 cp`, latest read, Git operation, token, or credential is
  rendered.
- Termination review: the timer starts before S3 input work, marker upload is
  bounded to 120 seconds, the service allows three minutes, and an EXIT trap
  always requests shutdown.
- Shell review: the checked-in script is executable, uses strict mode and
  owner-only runtime paths, and the final driver receives only allowlisted
  variables.
- Remaining integration action outside this task's ownership: update the
  deterministic package inventory so the approved seed contains
  `bootstrap.py`, `contracts.py`, and `s3.py`, and provide the fixed
  `/opt/memorysplit/scripts/aws_corpus_builder_driver.py` entry point.
- Remaining infrastructure action outside this task's ownership: apply the
  launch-template wiring above and ensure the selected AL2023 AMI contains AWS
  CLI v2 at `/usr/bin/aws`.
- Runtime NVMe, systemd, cloud-init gzip handling, and S3 behavior were not
  exercised on EC2 because this task explicitly prohibited AWS/network use.
