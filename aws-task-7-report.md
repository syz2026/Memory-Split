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
8. `bootstrap-user-data-sha256` — strictly decode the deployed launch-template
   `UserData`, hash its exact bytes, and require equality with both the
   `BootstrapUserDataSha256` stack output and the frozen profile's
   `bootstrap_user_data_sha256`.
9. `private-network-and-security-group` — require a private subnet, one
   no-public-IP interface, only the output security group, matching VPC, and
   exactly zero ingress rules.
10. `instance-profile-and-builder-role` — bind the template profile ARN to the
   live IAM instance profile and its one exact builder role.
11. `bucket-and-kms` — require the exact bucket, enabled versioning, all four
    Block Public Access controls, BucketOwnerEnforced ownership, one
    BucketKey-enabled SSE-KMS rule, an enabled customer KMS key, and package
    and source KMS ARNs equal to the live CloudFormation `DataKeyArn`.
12. `linux-on-demand-price` — parse exactly one current Linux, shared,
    no-preinstall, Used-capacity On-Demand hourly price for
    `i4i.16xlarge` in N. Virginia and require at most `$5.491/hour`.
13. `maximum-compute-cost` — calculate the 24-hour amount to cents and require
    at most `$131.78`.
14. `ec2-run-instances-dry-run` — call only the injected EC2 client with
    `DryRun=True`, the exact template ID/version, and one-instance counts;
    accept only the AWS `DryRunOperation` confirmation (or an explicit
    fake-client dry-run acknowledgement).

No `LaunchIntent` is constructed or serialized until all 14 gates, including
the EC2 dry run, have passed.

## Intent and CLI behavior

- `not_after` is exactly 30 minutes after the supplied canonical UTC second.
- The intent uses Task 1's `LaunchIntent` and canonical serializer.
- The CLI accepts exact package/source records and stack-output JSON, supports
  injected `AwsClients` and time for tests, and lazily creates boto3 clients
  only for an explicit live invocation.
- The CLI has no bootstrap-hash override. It rejects the removed
  `--expected-bootstrap-user-data-sha256` flag.
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

## Bootstrap user-data integrity gate

Implementation commit:
`9411945342713dee8388eabcaa0ee46e911621b0`
(`feat: verify launch template bootstrap integrity`)

### Superseded renderer choice

`cluster/aws/corpus_builder/bootstrap.py` is not present in this worktree, so
commit `9411945` initially used a caller-supplied expected-hash fallback. Review
showed that this was not independent authority because the same caller supplied
the stack-output JSON. That fallback is removed by the Critical correction
below.

### Bootstrap gate RED

The request-authority test first produced `1 failed in 0.12s` with the expected
`TypeError` because `PreflightRequest` had no expected-hash field:

```text
python -m pytest -q \
  tests/test_aws_corpus_builder_preflight.py::test_request_carries_reviewed_bootstrap_user_data_hash
```

After adding only that authority field, the focused gate tests produced
`5 failed, 1 passed, 50 deselected in 0.16s`: the old 13-gate path accepted a
mismatched payload, ignored missing and malformed stack hashes, and never
compared the foundation output with the reviewed expected hash.

The CLI RED was `2 failed in 0.21s` before CLI support existed: one missing
flag reached local validation instead of argparse, and one supplied flag was
unrecognized.

### Bootstrap gate GREEN

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
python -m py_compile \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py
git diff --check
```

Fresh result: `58 passed in 0.20s`; compilation and whitespace checks exited 0
with no output.

### Bootstrap gate self-review

- Gate 8 runs immediately after `launch-template-version` and before
  `private-network-and-security-group`; there are now exactly 14 gates.
- The existing exact launch-template lookup supplies deployed `UserData`.
  Preflight requires non-empty canonical base64, decodes it, and hashes the
  decoded bytes rather than the encoded representation.
- The review invalidated the original independence claim: its expected hash
  and stack output were both invocation-controlled. The Critical correction
  below replaces that input with a frozen profile contract.
- Mismatch coverage confirms no intent serialization, private-network lookup,
  EC2 dry run, or launch can follow the bootstrap gate.
- The launch intent already pins the exact launch-template ID and version, so
  the verified bootstrap remains bound without changing the Task 1 intent
  schema.
- No real AWS client or network call was made.

The caller-supplied fallback described in this historical section is no longer
accepted.

## CRITICAL CONTRACT CHANGE: profile-pinned bootstrap digest

All branches that consume the corpus-builder profile must pick up both
`cluster/aws/corpus_builder/contracts.py` and
`cluster/profiles/aws-i4i.16xlarge-corpus-v1.json`. The profile schema now
requires `bootstrap_user_data_sha256`; taking only one file will fail canonical
profile parsing.

Fix commit:
`32e4c81ca9422b312c418de78872bbb2d37cf140`
(`fix: pin bootstrap digest in profile contract`)

### Provisional integration value

The following digest is deliberately provisional:

```text
4a666e5a093da098a8066aadd4b4ed368dc4571a8cde54e525bf2e387db06861
```

It is named `PROVISIONAL_BOOTSTRAP_USER_DATA_SHA256` in `contracts.py` and
asserted literally by
`test_profile_bootstrap_hash_is_loudly_provisional`. When the authoritative
build-invariant payload lands, integration must update the contracts constant,
the canonical profile JSON, and that test/fixture together. Changing only one
fails loudly.

### Pinned-authority RED

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py \
  -k 'request_cannot_override or profile_bootstrap_hash or profile_missing_or_malformed or alternate_profile or bootstrap_user_data_mismatch or bootstrap_stack_hash or cli_rejects_caller'
```

Observed RED: `8 failed, 53 deselected in 0.57s`. The failures proved that:

1. `PreflightRequest` still accepted a caller hash;
2. the profile had no bootstrap digest contract;
3. missing, malformed, and alternate valid profile digests were not governed
   by the new schema and pinned value;
4. gate 8 errors still described caller/output consistency instead of profile
   authority; and
5. the CLI still accepted the override flag.

### Pinned-authority GREEN

```text
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
python -m pytest -q tests/test_aws_corpus_builder_contracts.py
python -m py_compile \
  cluster/aws/corpus_builder/contracts.py \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py
git diff --check
```

Fresh results: `61 passed in 0.78s` for preflight and `46 passed in 0.04s` for
contracts (`107 passed` total). Compilation and whitespace checks exited 0
with no output.

### Pinned-authority self-review

- The canonical profile requires the new field, validates lowercase SHA-256
  grammar, serializes it byte-for-byte, and pins it to the versioned
  provisional constant.
- Exact pinning in `contracts.py` prevents an alternate `--builder-profile`
  path containing another syntactically valid digest from becoming authority.
- Gate 8 receives the already validated `CorpusBuilderProfile`; both the
  snapshotted `BootstrapUserDataSha256` output and decoded deployed user-data
  bytes must equal `profile.bootstrap_user_data_sha256`.
- `PreflightRequest` no longer has an expected-digest field, and the CLI
  parser no longer defines an expected-digest flag. A regression asserts that
  the old flag is rejected.
- Missing or malformed profile fields fail in gate 1 before any AWS client
  call. Stack or deployed-payload disagreement fails in gate 8 before network,
  dry-run, or intent emission.
- The frozen profile's canonical bytes, and therefore every resulting
  `profile_sha256`, intentionally change with this contract addition.
- No real AWS client or network call was made.
