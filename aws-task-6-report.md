# AWS corpus builder Task 6 report

## Status and scope

Implemented the build-invariant NVMe bootstrap renderer and runtime, the hard
systemd termination watchdog, the fixed SSM-stage builder entry point, and
focused tests. No CloudFormation file, package inventory, or file outside the
assigned Task 6 ownership set was modified. No AWS API or network request was
made during development or tests.

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

`render_bootstrap()` emits one deterministic stack-lifetime payload. The
optional legacy `BootstrapConfig` argument is transition-only and deliberately
ignored by rendering, so differing build parameters produce byte-identical
output. The old exact-record helper remains available to Python callers but no
record, build ID, KMS ARN, profile hash, worker count, or launch-intent hash is
serialized into launch-template user data.

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
- installs `/usr/local/bin/memorysplit-corpus-builder` only after the watchdog,
  NVMe validation, RAID0/XFS creation, and owner-only directories are ready;
- leaves the healthy instance running for the Task 5 SSM command while the
  boot-anchored watchdog remains armed;
- accepts only the five Task 5 command values at the fixed entry point, resolves
  a single immutable S3 version for each content-addressed key, then pins every
  HEAD and GET to that version ID;
- verifies SSE-KMS authority, metadata SHA-256, byte count, streamed SHA-256,
  a final exact-version HEAD, JSON framing, and safe archive members before
  `/opt/memorysplit` installation;
- runs the fixed driver under an explicit environment allowlist; and
- requests shutdown after every SSM-stage success or failure, independently of
  the still-armed watchdog.

Implementation commit:

```text
428ef1199104d8449e1d737a556b039a79c078ed
feat: bootstrap bounded NVMe corpus builds
```

## Verification evidence

Initial implementation verification before the review follow-ups:

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

The foundation template must not duplicate bootstrap logic or supply any
per-build value to user data. Its immutable launch-template version must use
the output of this exact sequence:

```python
import base64
import gzip
import hashlib

rendered = render_bootstrap().encode("utf-8")
compressed = gzip.compress(rendered, compresslevel=9, mtime=0)
assert len(compressed) <= 16_384
user_data = base64.b64encode(compressed).decode("ascii")
bootstrap_user_data_sha256 = hashlib.sha256(
    user_data.encode("ascii")
).hexdigest()
parts = [
    user_data[index : index + 4096]
    for index in range(0, len(user_data), 4096)
]
```

Split `user_data` at 4,096-character boundaries into the foundation's ordered
`BuilderUserDataGzipBase64Part1` through `Part6` parameters and leave unused
tail parts empty. Assign their exact concatenation directly to
`LaunchTemplateData.UserData`; it is already base64 encoded and must not be
wrapped in a second `Fn::Base64`. Set the stack parameter and output
`BootstrapUserDataSha256` to `bootstrap_user_data_sha256`. `RunInstances` must
not override user data.

For commit `d3af78ae3bc4e5805fde06f8fa96ab41ec1bb9ec`, the authoritative
deterministic payload is:

```text
UTF-8 rendered bytes: 36,486
UTF-8 SHA-256:        7a551e1bc5bfc614c3ca699c8359b46143e4c10417a68be1f75b5b73edfdda37
gzip bytes:           9,594
gzip SHA-256:         11f5fbfd7bee7a01e42956654020d21eeec7455305a2da3c8c7e9fb37a0b139f
base64 ASCII bytes:   12,792
base64 SHA-256:       0e93ee325ff77776caf8e5e910e6a7b84850d901d5b596981ad3525fd5fff50d
part lengths:         4,096, 4,096, 4,096, 504
```

The authoritative foundation/preflight value is the base64 SHA-256
`0e93ee325ff77776caf8e5e910e6a7b84850d901d5b596981ad3525fd5fff50d`,
matching the foundation parameter's "concatenated base64 payload" contract.
Base64 decoding produces the 9,594-byte gzip member, 6,790 bytes below EC2's
16,384-byte decoded user-data limit. AL2023 cloud-init decompresses it and
executes the rendered shebang script.

The payload installs the fixed executable expected by the Task 5 SSM document:

```text
/usr/local/bin/memorysplit-corpus-builder
  --build-id '{{ BuildId }}'
  --package-uri '{{ PackageUri }}'
  --package-sha256 '{{ PackageSHA256 }}'
  --source-manifest-uri '{{ SourceManifestUri }}'
  --source-manifest-sha256 '{{ SourceManifestSHA256 }}'
```

Those five values exist only at SSM invocation time. Because the Task 5
document does not carry S3 version IDs, the fixed entry point lists the exact
content-addressed key, rejects delete markers or any history other than one
object version, captures that sole version ID, and uses it for HEAD, GET, and
the final HEAD. It never reads an unversioned "latest" object. The command
fails closed if the key was overwritten, deleted, is outside `v2/packages/` or
`v2/sources/`, or its metadata/content SHA-256 differs from the command.

The launch template must retain
`InstanceInitiatedShutdownBehavior: terminate`, IMDSv2-only metadata, the
builder role, and no public IP. The AMI must provide `/usr/bin/aws` (CLI v2),
Python 3, systemd, mdadm, XFS tools, util-linux, and coreutils. IAM must permit
`s3:ListBucketVersions`, exact-version reads for both approved objects, and
SSE-KMS writes beneath `v2/builds/{build_id}/operational/`.

## Self-review and integration concerns

- Scope review: only assigned implementation/test/report files changed; the
  untracked task brief remains untouched.
- Invariance review: tests render two configurations with different build IDs,
  object URIs, version IDs, object hashes, profile hashes, launch-intent hashes,
  and worker counts; both outputs are byte-identical and contain none of those
  values.
- Authority review: every SSM-stage object read captures one version ID before
  GET and carries that ID through two HEADs plus the download; no `aws s3 cp`,
  latest read, Git operation, token, or static credential is rendered.
- Termination review: the timer starts before runtime/storage work and measures
  from boot; the timeout service requests shutdown before telemetry, bounds its
  upload attempt to five seconds, allows 15 seconds for the service, and retries
  shutdown from its EXIT trap.
- Runtime review: the checked-in shell uses strict mode and owner-only runtime
  paths; the installed Python entry point extracts regular files/directories
  manually, never calls `extractall`, and gives the final driver only
  allowlisted variables.
- Remaining integration action outside this task's ownership: update both the
  Task 5 foundation fixture and preflight fixture from the payload pinned from
  `48835f7` to the authoritative base64 SHA-256 above, supply the new four
  base64 parts, and ensure the selected AL2023 AMI contains AWS CLI v2 at
  `/usr/bin/aws`.
- Publishing must preserve one object version per content-addressed package or
  source key. This is an intentional fail-closed consequence of the existing
  five-parameter SSM contract, which does not pass version IDs.
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
requests shutdown without waiting for remote telemetry.

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
  boot-relative termination, and credential-free user data are preserved; and
- no CloudFormation, package inventory, AWS resource, or network operation was
  touched by this follow-up.

## Integration follow-up: build-invariant user data

### Root cause and boundary

The prior payload embedded every `BootstrapConfig` field, including the
preflight-produced launch-intent hash. That made the launch-template payload
change per build and created a circular attestation: preflight could not verify
the payload until it produced the value embedded inside that payload.

The corrected boundary is:

1. build-invariant user data arms the boot-relative watchdog first;
2. it validates and mounts exactly four instance-store NVMe devices as
   RAID0/XFS and creates owner-only directories;
3. it installs the fixed SSM entry point and waits;
4. Task 5 passes the five per-build values to that executable; and
5. the entry point resolves and verifies exact S3 versions, installs the
   package, runs the driver, and requests termination.

The watchdog never depends on step 4. Before SSM supplies a build identity it
writes a local timeout marker and shuts down. After the fixed entry point has
validated the build and KMS authority, it atomically writes owner-only watchdog
context so the already-armed service may also attempt the same bounded
build-scoped marker upload.

### RED evidence

Tests were changed before production code to require no-argument canonical
rendering, byte identity across different build configurations, absence of all
fixture values, the fixed entry point, unique-version resolution, ambiguous
history rejection, and the decoded user-data size gate.

```text
Command: python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
Exit: 1
Result: 6 failed, 15 passed
Expected causes:
  rendered output still differed between two builds
  no fixed builder-entrypoint heredoc existed
  no unique-version resolver existed
  old runtime tests still described boot-stage package execution
```

Failing contract commit:

```text
92b148fa04cdcc7645a9d116bda782b3d5a41c58
test: define invariant bootstrap boundary
```

### GREEN implementation

`render_bootstrap()` now replaces its single marker with a constant comment;
the canonical call has no argument. Passing either of the legacy test
configurations is supported only for transition and cannot alter bytes.

The user-data main path ends after watchdog activation, NVMe/RAID/XFS setup,
receipt creation, and fixed-entrypoint installation. The embedded entry point
accepts exactly the Task 5 flags, resolves a sole immutable version from each
content-addressed key, performs version-pinned HEAD/GET/final-HEAD checks,
rehashes both files, validates JSON and archive safety, and then runs the fixed
driver. Its `finally` path requests shutdown whether argument processing,
authority checks, extraction, or the driver succeeds or fails.

Implementation commit:

```text
d3af78ae3bc4e5805fde06f8fa96ab41ec1bb9ec
fix: make corpus bootstrap build invariant
```

### Final verification

All commands were run after the implementation commit without AWS or network
access:

```text
python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
21 passed in 1.77s

python -m pytest -q tests/test_aws_corpus_builder*.py
105 passed in 2.33s

python -m py_compile cluster/aws/corpus_builder/bootstrap.py
exit 0, no output

bash -n cluster/aws/corpus_builder/bootstrap.sh
exit 0, no output

git diff --check
exit 0, no output
```

The payload measurement in the launch-template contract was generated twice
from `render_bootstrap()`, asserted byte-identical, compressed with
`compresslevel=9, mtime=0`, and encoded with standard base64. No AWS API,
instance, CloudFormation stack, or network service was contacted.
