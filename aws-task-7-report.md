# AWS corpus builder Task 7 report

## Status

PASS. Implemented the read-only preflight, canonical 30-minute launch intent,
owner-only intent emission, and fake-only focused coverage on
`feat/aws-corpus-preflight`.

Base commit:
`f098362ce53b85fc45bb4293022ae37c9a2c020d`

Implementation commit:
`06e4c4ab411ac7b539183984d6c5302755815d1d`
(`feat: preflight bounded AWS corpus builds`)

## Strict TDD evidence

### Initial RED

Command:

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
```

Result: exit 2, one collection error. The expected failure was
`ModuleNotFoundError: No module named 'cluster.aws.corpus_builder.preflight'`.
No production preflight or CLI file existed.

### Initial GREEN

After the first minimal implementation, the same focused command completed
with `33 passed in 0.14s`.

### Review RED/GREEN: ordered price gate and failed-write cleanup

Two review tests were added before their fixes:

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
```

The expected RED result was `2 failed, 34 passed`. It proved:

1. an over-ceiling hourly price was being rejected in check 12 instead of
   check 11; and
2. a simulated `fsync` failure could leave a partial intent file.

The implementation then moved the `$5.491/hour` comparison into check 11 and
made failed owner-only writes remove only the inode created by the CLI. GREEN
was `36 passed in 0.12s`.

### Review RED/GREEN: live AMI root-device binding

The parallel foundation uses the pinned AMI root name `/dev/sda1`. A focused
test was changed first to represent that live contract:

```text
python -m pytest -q \
  tests/test_aws_corpus_builder_preflight.py::test_preflight_emits_intent_only_after_every_read_only_gate_and_ec2_dry_run
```

The expected RED result was one failure:
`launch template root device name drift`, exposing a hard-coded `/dev/xvda`.
The AMI gate now returns the verified root device name and the launch-template
gate binds its root mapping to that value. The full focused suite returned to
`36 passed in 0.11s`.

### Final GREEN and static verification

Fresh post-commit commands:

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
```

Result: `36 passed in 0.11s`.

```text
python -m py_compile \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py
```

Result: exit 0, no output.

```text
git diff --check HEAD^ HEAD
git diff --check
```

Result: both exit 0, no output.

## Exact ordered gate list

The returned `checks` tuple is exactly:

1. `profile-canonical-sha256` — load the closed profile, require canonical
   bytes, reserialize byte-for-byte, and compute SHA-256.
2. `exact-s3-versions` — validate both `S3ObjectVersion` contracts, HEAD the
   package and source manifest by exact version, and require shared build and
   KMS identities. Require the exact package version's S3 metadata to bind its
   immutable package revision.
3. `production-software-gate` — parse the Task 2 canonical package manifest,
   bind archive bytes and SHA-256 to the package record, bind its revision to
   the exact package object's metadata, and require every package module in the
   existing production authority set.
4. `account-and-region` — require STS account `056956104102` and EC2 client
   region `us-east-1`.
5. `ami-identity` — require the exact AMI and owner, `x86_64`, `available`,
   EBS root, a valid root device name, and a non-future UTC creation date.
6. `instance-type-availability` — require an available selected subnet and
   exact `i4i.16xlarge` offerings in both `us-east-1` and the subnet AZ.
7. `launch-template-version` — require the stack launch-template ID and an
   explicit numeric version, then verify AMI, instance type, shutdown,
   termination protection, monitoring, IMDSv2, encrypted 200 GiB `gp3`, and
   the AMI-bound root device.
8. `private-network-and-security-group` — require a private subnet, one
   no-public-IP interface, only the output security group, matching VPC, and
   exactly zero ingress rules.
9. `instance-profile-and-builder-role` — bind the template profile ARN to the
   live IAM instance profile and its one exact builder role.
10. `bucket-and-kms` — require the exact bucket, enabled versioning, all four
    Block Public Access controls, BucketOwnerEnforced ownership, one
    BucketKey-enabled SSE-KMS rule, an enabled customer KMS key, and package
    and source KMS ARNs equal to the live CloudFormation `DataKeyArn`.
11. `linux-on-demand-price` — parse exactly one current Linux, shared,
    no-preinstall, Used-capacity On-Demand hourly price for
    `i4i.16xlarge` in N. Virginia and require at most `$5.491/hour`.
12. `maximum-compute-cost` — calculate the 24-hour amount to cents and require
    at most `$131.78`.
13. `ec2-run-instances-dry-run` — call only the injected EC2 client with
    `DryRun=True`, the exact template ID/version, and one-instance counts;
    accept only the AWS `DryRunOperation` confirmation (or an explicit
    fake-client dry-run acknowledgement).

No `LaunchIntent` is constructed or serialized until all 13 gates, including
the EC2 dry run, have passed.

## Intent and CLI behavior

- `not_after` is exactly 30 minutes after the supplied canonical UTC second.
- The intent uses Task 1's `LaunchIntent` and canonical serializer.
- The CLI accepts exact package/source records and stack-output JSON, supports
  injected `AwsClients` and time for tests, and lazily creates boto3 clients
  only for an explicit live invocation.
- The output is exclusively created, canonical, fsynced, mode `0600`, and its
  SHA-256 is printed. Failed gates and failed writes leave no emitted intent.
- Every focused test uses fakes. No real AWS call was made during this task.

## Self-review

- Confirmed all AWS operations before launch are read-only; the sole
  `RunInstances` call always carries `DryRun=True`.
- Confirmed the live `DataKeyArn` comparison deferred by Task 1 is enforced.
- Confirmed exact check ordering both in the public tuple and fake call trace.
- Confirmed denial, malformed response, drift, pagination, multiplicity, and
  write failures fail closed before intent emission.
- Confirmed the implementation reuses Task 1 contracts and Task 3 exact
  S3-version verifier without redefining either.
- Confirmed only the three assigned implementation files were included in the
  implementation commit; `aws-task-7-brief.md` was not staged.

## Concerns

No known blocker. The software-gate receipt is deliberately interpreted as the
canonical Task 2 package manifest because that is the existing artifact that
proves successful closed-world production packaging; it is bound to the exact
uploaded package version and the package module's production authority set.

## Controller review corrections

Fix commit:
`63f73b73679b1c6091ab5ea02905e6d1fad97753`
(`fix: harden AWS preflight launch authorization`)

### Review RED

The focused regression tests were added before the corrections:

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
```

Expected RED: `9 failed, 36 passed in 0.39s`. The failures demonstrated:

1. a manifest with the right archive SHA-256 but an unrelated revision passed
   the production software gate;
2. a mutable `stack_outputs` mapping could change the emitted subnet after
   validation;
3. `--live` did not exist, default CLI execution constructed live clients, and
   malformed local authorities did not precede client construction; and
4. the final intent path was visible before the payload was fsynced.

### Review GREEN

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
python -m py_compile \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py
git diff --check
```

Fresh result: `45 passed in 0.13s`; compilation and whitespace checks exited 0
with no output.

### Review self-review

- The package manifest's archive bytes and SHA-256 now match the package record,
  and its revision must match metadata read from that same immutable S3 object
  version after Task 3's exact-object verification.
- The CLI parses records and validates the canonical profile, record
  relationships, and software evidence before `_live_clients` can run.
  `--live` is mandatory without injected clients, and an autouse test guard
  makes accidental real-client construction fail every focused test.
- Stack outputs are copied once at the first stack-dependent gate. The
  validated subnet ID and other launch authorities are carried from that
  snapshot into the intent; the mutable request mapping is never re-read.
- Intent bytes are written mode `0600` to a randomized same-directory
  temporary, fsynced and closed under the failure guard, then atomically
  published without replacement by hard link. Interrupted writes never expose
  a partial final intent path.
- No real AWS client or network call was made. All live-path tests replace
  `_live_clients` with fakes.
- Only the three assigned Python files and this report changed. The untracked
  task brief and review package were not staged.

Operational prerequisite: package publication must preserve the packager's
immutable revision in S3 object metadata under `revision`; preflight now fails
closed if that binding is absent or malformed.

## Controller second-review corrections

Fix commit:
`de66716c3ad9f55cd893ef44c5d1e674610e8af6`
(`fix: close AWS preflight publication races`)

### Second-review RED

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
```

Expected RED: `5 failed, 45 passed in 0.27s`. The failures demonstrated:

1. missing or malformed package revision metadata did not name
   `Metadata["revision"]`;
2. an invalid programmatic clock constructed opted-in live clients before
   `run_preflight` rejected it;
3. a destination created during publication was silently replaced by rename;
4. failure of the first temporary-file `fstat` left the temporary behind; and
5. cleanup unlink failures were suppressed instead of accompanying the primary
   error.

### Second-review GREEN

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
python -m py_compile \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py
git diff --check
```

Fresh result: `50 passed in 0.18s`; compilation and whitespace checks exited 0
with no output.

### Second-review self-review

- Publication now hard-links the completed same-directory temporary to the
  destination. This operation atomically fails if any racer owns the final
  name, and the final inode is verified against the written inode before
  success.
- The writer records temporary creation before its first `fstat`. Every
  failure path attempts cleanup of each name it created, including both links
  when publication partially completed.
- Cleanup unlink failures are attached as exception notes. The original write,
  validation, or publication exception remains the raised primary error.
- Canonical UTC time validation runs with the other client-free checks before
  `_live_clients`; the direct `run_preflight` validation remains in place.
- Missing or malformed package revision metadata now identifies the exact
  required key, `Metadata["revision"]`.
- No real AWS client or network call was made.

An uncatchable kill before publication can leave a private
`.name.pid.random.tmp` file. It is not the canonical intent path and cannot be
mistaken for a published or reusable launch intent. No automatic sweep is
performed because a matching filename and owner alone do not prove that a
stale file belongs to this invocation.
