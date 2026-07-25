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
- enables the boot-anchored `23h30m` timer before runtime initialization,
  storage discovery, RAID/XFS work, or any network operation;
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

Compression is mandatory for the EC2 16 KiB decoded user-data limit. The
current test fixture renders to 26,605 raw bytes, 6,878 deterministic gzip
bytes, and 9,172 base64 bytes. Integration must reject a gzip payload over
16,384 bytes.

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
- Termination review: the timer starts before runtime/storage work and measures
  from boot; the timeout service requests shutdown before telemetry, bounds its
  upload attempt to five seconds, allows 15 seconds for the service, and retries
  shutdown from its EXIT trap.
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

## Review follow-up: boot-anchored hard deadline

### Root cause

The reviewed implementation invoked `install_watchdog` only after NVMe
discovery, RAID creation, XFS formatting/mounting, and receipt generation.
Those operations could hang before a timer existed. Its `OnActiveSec=23h30m`
trigger also started the clock at timer activation rather than instance boot,
and the watchdog's EXIT trap reached shutdown only after a 120-second marker
upload attempt.

### Follow-up RED evidence

The textual ordering check was replaced with tests that execute rendered user
data in an isolated state root with PATH-injected `systemctl`, `udevadm`,
`shutdown`, `timeout`, and `aws` commands. The ordering test also executes a
deliberately regressed orchestration order and proves no timer activation is
recorded before the storage stub halts execution. A second behavioral test
executes the generated watchdog program with a timed-out telemetry command.

```text
Command: python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
Exit: 1
Result: 3 failed, 15 passed
Expected causes:
  OnBootSec=23h30m was absent
  no timer activation preceded the storage command
  no sandboxed watchdog program was installed
```

The failing review tests were committed before the fix:

```text
11784756a73f71eab7a8ad1a2ff65b9b3ab69fa8
test: reproduce boot deadline watchdog gaps
```

### Follow-up GREEN implementation

The rendered script now performs only shell setup, path assignment, shutdown
resolution, and an emergency EXIT-trap registration before
`install_watchdog`. The watchdog is installed and enabled before
`initialize_runtime`, `udevadm`, `lsblk`, `mdadm`, `mkfs.xfs`, mounting,
downloads, or uploads.

The timer uses:

```ini
[Timer]
OnBootSec=23h30m
Persistent=true
AccuracySec=1s
Unit=memorysplit-corpus-watchdog.service
```

`OnBootSec` is measured from kernel boot, not timer activation. Enabling the
unit slightly after boot preserves the original boot-plus-23h30m deadline. If
activation occurs after that monotonic deadline, systemd considers the trigger
elapsed and starts the service immediately; it does not grant a fresh 23h30m
window.

The timeout service invokes `shutdown -h now` before constructing or uploading
the marker. Marker upload is then best effort under a hard five-second timeout,
the service has a 15-second start timeout, and the EXIT trap attempts shutdown
again regardless of marker success. The main bootstrap failure trap likewise
requests shutdown before uploading its failure log.

Fix commit:

```text
48835f75547986e5b74304adcb57767ae748911e
fix: anchor corpus watchdog to boot
```

### Follow-up verification

```text
python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
18 passed in 2.04s

python -m py_compile cluster/aws/corpus_builder/bootstrap.py
exit 0, no output

bash -n cluster/aws/corpus_builder/bootstrap.sh
exit 0, no output

git diff --check
exit 0, no output

python -m pytest -q \
  tests/test_aws_corpus_builder_contracts.py \
  tests/test_aws_corpus_builder_s3.py \
  tests/test_aws_corpus_builder_bootstrap.py
102 passed in 2.23s
```

### Updated launch-template contract and self-review

The launch-template data construction is unchanged: gzip the exact UTF-8
output of `render_bootstrap`, base64 it once, and assign that value directly to
`LaunchTemplateData.UserData`. Production cloud-init must invoke the rendered
script with zero positional arguments; the optional state-root argument exists
only so no-network tests can execute the real orchestration against isolated
paths and command doubles.

Self-review confirmed:

- no command that discovers or mutates storage can execute before successful
  timer activation;
- moving `install_watchdog` below discovery makes the behavioral regression
  test fail;
- boot-relative timing cannot be extended by delayed timer activation;
- shutdown is initiated before either watchdog-marker or failure-log
  telemetry;
- four-device instance-store selection, root exclusion, RAID0/XFS settings,
  exact-version downloads, and credential-free rendered inputs are unchanged;
  and
- no CloudFormation, package inventory, AWS resource, or network operation was
  touched by this follow-up.
