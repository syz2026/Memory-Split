# AWS Corpus Builder Task 5 Report

## Status

Implemented the private, no-instance CloudFormation foundation on
`feat/aws-corpus-foundation` from base
`f098362ce53b85fc45bb4293022ae37c9a2c020d`. No stack deployment, change set,
or AWS API call was made.

Implementation commit:
`afd48fa307ecf40d8584ebc060c83634f95bd805`
(`feat: add private AWS corpus builder foundation`).

## Scope

The implementation commit contains only:

- `infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml`
- `infra/aws/cfn-guard/memorysplit-corpus-builder.guard`
- `infra/aws/requirements-dev.txt`
- `tests/test_aws_corpus_builder_foundation.py`

The supplied untracked `aws-task-5-brief.md` was not modified or committed.

## TDD evidence

### Initial RED

Command:

```text
python -m pytest -q tests/test_aws_corpus_builder_foundation.py
```

Result: exit 1, `1 failed, 16 skipped in 0.13s`.

The expected failing assertion identified exactly these absent production
artifacts:

```text
infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml
infra/aws/cfn-guard/memorysplit-corpus-builder.guard
infra/aws/requirements-dev.txt
```

### Initial GREEN

After implementing the template, guard rules, and pinned validator
requirement, the focused result was:

```text
17 passed in 0.11s
```

### Validator-driven RED/GREEN

The first `cfn-lint` run found no schema error, but exited 4 with twelve
`W1020` warnings because static S3 prefixes were unnecessarily wrapped in
`Fn::Sub`. A regression invariant was added first. Its RED result was:

```text
1 failed, 17 passed in 0.16s
```

The expected failure was:

```text
assert '${' in 'v2/builds'
```

Static prefixes were then converted to plain strings and the guard queries
were updated to the same representation.

### Final GREEN

Fresh final command:

```text
python -m pytest -q tests/test_aws_corpus_builder_foundation.py
```

Result: exit 0, `18 passed in 0.11s`.

## Validator and verification evidence

### cfn-lint

`cfn-lint==1.53.2` was installed successfully from PyPI into the ignored
worktree-local `.venv`. Fresh final command:

```text
.venv/bin/cfn-lint \
  infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml
```

Result: exit 0 with no findings.

### cfn-guard

No `cfn-guard` executable was present on `PATH`. The required PyPI
installation attempt was:

```text
.venv/bin/python -m pip install cfn-guard
```

It failed definitively with exit 1:

```text
ERROR: Could not find a version that satisfies the requirement cfn-guard
       (from versions: none)
ERROR: No matching distribution found for cfn-guard
```

Therefore `cfn-guard validate` could not be executed. This is the only
required-validator gap. The focused Python suite still checks guard
non-vacuity, exact logical resource/type assertions, exact resource count,
forbidden resource types, exact builder action strings, wildcard rejection,
and required IAM/network/launch/bucket rules.

### Python compile and patch checks

Fresh commands:

```text
python -m py_compile tests/test_aws_corpus_builder_foundation.py
git diff --check
```

Both exited 0 with no output.

### Optional repository-wide suite

The initial sandboxed full-suite attempt was invalidated by repeated
`Operation not permitted` errors from tests that create temporary Git
repositories. Re-running once without sandbox filesystem restrictions gave:

```text
1908 passed, 3 failed, 2 deselected, 1 warning in 280.60s
```

The three remaining failures are outside the owned files:

- `test_route_index_open_rejects_namespace_aba_to_different_sqlite_inode`
  failed with a temporary SQLite `disk I/O error`;
- `test_accepts_release_built_by_final_aws_packager` rejected an existing
  integration artifact whose `DATASET-POINTER-AWS.json` fields were not exact;
- `test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle` expected a
  different pre-existing integration commit reference.

The required focused Task 5 suite is independently green as recorded above.

## Self-review

- Exact inventory: 24 builder-only resources; no EC2 instance, Auto Scaling,
  Fleet, Spot, Capacity Reservation, ECR, key pair, NAT, Internet Gateway,
  EIP, gateway attachment, or public route.
- Network: one operator-selected private subnet; public IP assignment is
  disabled; builder ingress is empty; endpoint ingress is HTTPS only from the
  builder security group; S3 plus the seven required interface endpoints are
  present.
- State: the exact approved bucket name, bucket versioning, Bucket Owner
  Enforced ownership, full public-access blocking, dedicated rotating KMS
  encryption, retained bucket/key/log group, and TLS/wrong-KMS deny statements
  are encoded.
- Launch safety: `i4i.16xlarge`, 200 GiB encrypted `gp3`, terminate-on-shutdown,
  deletion on termination, detailed monitoring, IMDSv2-only metadata,
  termination protection disabled, no SSH key, private networking, and
  `MemorySplitCorpusBuilder=true` instance/volume tags match the frozen
  profile.
- IAM: exactly two roles and one instance profile; the builder has only the
  fifteen approved inline actions plus only
  `AmazonSSMManagedInstanceCore`; S3 reads/writes are restricted to builds,
  packages, and sources; controller lifecycle, pass-role, SSM, tag, and KMS
  permissions are statement- and resource-scoped.
- Operations: the controller can send only
  `MemorySplit-CorpusBuilderV1` to tagged builders; the budget is fixed at
  `$200` with actual 50%, actual 80%, and forecasted 100% email alerts.
- Interface: outputs are exactly the eleven required non-exported identifiers.
- Template style and secret hygiene: long-form intrinsics only, no redundant
  substitutions, and no static secret-like material.
- Scope: no deployment and no edits outside the authorized implementation
  files and this report.

## Concerns

- The guard rules could not be executed because AWS CloudFormation Guard has
  no installable PyPI distribution and no local binary was available.
- The optional full repository suite has the three unrelated failures listed
  above; the focused Task 5 suite and `cfn-lint` are clean.

## AWS IaC MCP finding triage follow-up

### Applied findings

- `S3_BUCKET_SSL_REQUESTS_ONLY`: fixed. `DenyInsecureTransport` now denies
  `s3:*` on both the bucket and all objects when `aws:SecureTransport` is
  `false`; the other statement fields are unchanged.
- `SECURITY_GROUP_MISSING_EGRESS_RULE` and
  `SECURITY_GROUP_DESCRIPTION_RULE`: fixed. The builder group now permits
  only TCP 443 to the interface-endpoint group and to the operator-supplied
  `com.amazonaws.us-east-1.s3` managed prefix list. The endpoint group uses
  the AWS-documented `127.0.0.1/32` no-op egress rule to suppress EC2's
  create-time default allow-all rule; stateful reply traffic does not require
  a routed egress allowance. Every ingress and egress rule has a description.
  The endpoint ingress is a separate resource so the two groups do not form a
  CloudFormation dependency cycle.
- `S3_BUCKET_DEFAULT_LOCK_ENABLED`: fixed. The claim-bearing bucket has S3
  Object Lock enabled in `GOVERNANCE` mode with a default retention period of
  **365 days**.

Object Lock is compatible with the current publication and clean-room flows in
`cluster/aws/corpus_builder/s3.py`. Publication uses `PutObject` (including
`IfNoneMatch: "*"` for no-replace receipts), records returned version IDs, and
then performs version-pinned HEAD/GET/list verification. Clean-room retrieval
also uses version-pinned HEAD and GET. The module does not delete objects or
versions, modify retention, or request governance bypass, so default retention
does not block any operation it performs.

### Accepted without template changes

- `S3_BUCKET_NO_PUBLIC_RW_ACL`: false positive. `BucketOwnerEnforced` disables
  ACLs, and all four S3 public-access-block settings are enabled.
- `S3_BUCKET_REPLICATION_ENABLED`: accepted. Cross-Region replication is
  outside this builder-foundation scope and would add recurring cost for a
  rebuildable artifact.
- `IAM_NO_INLINE_POLICY_CHECK`: accepted. Inline policies deliberately match
  the reviewed sibling P5 foundation, keep exact statement allowlists
  versioned with this stack, and prevent their reuse elsewhere.

### Follow-up TDD evidence

Initial RED after adding the SSL, egress, descriptions, and Object Lock
regressions:

```text
6 failed, 14 passed in 0.19s
```

The first implementation reached `20 passed`, but a documentation check showed
that an empty endpoint egress list would not suppress EC2's create-time
allow-all rule. Tightening that regression produced the second RED:

```text
1 failed, 19 passed in 0.17s
```

After adding the documented localhost no-op rule, final GREEN was:

```text
20 passed in 0.09s
```

The final `cfn-lint` 1.53.2 run, `python -m py_compile
tests/test_aws_corpus_builder_foundation.py`, and `git diff --check` all exited
0 with no findings. CloudFormation Guard still has no installable PyPI
distribution in this environment, so the extended local guard file was not
executed here.

Implementation commit:
`c24123cbb56bd7dc56c01ae6cee514097abae6e8`.

## Full code-review follow-up

Implementation commit:
`ba8d02ed84778538ca2a2e78f42c3a12a0773013`.

### Critical 1: boot-anchored watchdog wiring

Closed. `BuilderLaunchTemplate` now receives the Task 6 `render_bootstrap`
payload as deterministic gzip encoded once as base64, without `Fn::Base64`.
CloudFormation limits each String parameter to 4,096 characters, while the
reviewed Task 6 fixture is 9,172 base64 characters and a 16,384-byte gzip
payload can require 21,848 characters. The template therefore accepts six
ordered chunks (five at 4,096 characters and one at 1,368) and joins them
directly into `LaunchTemplateData.UserData`. The deployment integration must
split the already-encoded value at those boundaries; it must not re-encode it.
Changing any chunk creates a new launch-template version for Task 7/8 to
preflight and launch by explicit numeric version.

The launch template still fixes
`InstanceInitiatedShutdownBehavior: terminate`. The frozen profile assertions
now cover the 86,400-second compute ceiling, the 84,600-second (23h30m)
boot-anchored watchdog/SSM timeout, `$5.491/hour`, and `$131.78` maximum compute
cost.

### Critical 2: controller launch authority

Partially closed at the IAM layer, with the unsupported remainder documented
rather than represented by fictitious condition keys.

Applied controls:

- `RunInstances` still requires the exact stack launch-template ARN and
  `ec2:IsLaunchTemplateResource=true`, and now also requires
  `ec2:InstanceMarketType=on-demand`.
- The existing exact instance-type, IMDSv2, and request-tag conditions remain.
- `ec2:ModifyInstanceAttribute` and `SetBuilderShutdownBehavior` were removed,
  so the controller cannot change shutdown behavior or termination protection
  after launch.

AWS's *Actions, resources, and condition keys for Amazon EC2* reference lists
`ec2:LaunchTemplate`, `ec2:IsLaunchTemplateResource`, and
`ec2:InstanceMarketType`, but no condition keys for a launch-template version,
`MinCount`, `MaxCount`, instance-initiated shutdown behavior, or termination
protection. The `RunInstances` API reference exposes version and instance
counts only as request fields. Consequently IAM cannot make a direct
`RunInstances` grant enforce one instance, one numeric template version, or
the shutdown scalar at launch.

Task 8 remains the compensating trusted launch boundary: it verifies the exact
numeric template version, calls `RunInstances` once with
`MinCount=MaxCount=1`, a deterministic client token, no user-data or other
launch overrides, and verifies/cleans up the returned instances. A principal
that can assume the controller role can still bypass that client, so the
cardinality/version/launch-time-shutdown portion of this Critical is **not
fully closed by IAM**. Closing it independently of trusted client code would
require a mediated launch service or another stateful authorization boundary,
which is outside this foundation's approved resource inventory.

### KMS and S3 hardening

- Builder cryptographic calls require
  `kms:ViaService=s3.${AWS::Region}.amazonaws.com` and the artifact-bucket
  encryption context. The context is the bucket ARN because S3 Bucket Keys
  are enabled.
- Controller EBS cryptographic calls require
  `kms:ViaService=ec2.${AWS::Region}.amazonaws.com` and the `aws:ebs:id`
  encryption-context key. `CreateGrant` also requires the EC2 service path and
  `kms:GrantIsForAWSResource=true`.
- Controller `kms:DescribeKey` remains a direct, read-only call scoped to the
  one key because Task 7 preflight uses it to verify live key metadata before
  launch; it is not a cryptographic operation.
- Both wrong-algorithm and wrong-key bucket-policy denies now cover
  `v2/builds/*`, `v2/packages/*`, and `v2/sources/*`.

### Governance-retention cleanup

Failed, partial, and quarantined build output **can be physically deleted
within the 365-day GOVERNANCE period**. Default retention applies to the
`v2/builds/*` objects, so the controller now has:

- `s3:ListBucketVersions` on the bucket, conditioned to `v2/builds/*`; and
- `s3:DeleteObjectVersion` plus `s3:BypassGovernanceRetention`, scoped only to
  `${ArtifactBucket.Arn}/v2/builds/*`.

Cleanup must address the exact version ID and send
`x-amz-bypass-governance-retention:true`; deleting without a version would
only create a delete marker and would not remove billed locked bytes.
Packages and sources are immutable inputs, not failure output, and remain
outside this cleanup grant. As expected for governance mode, the controller
can also bypass retention for a successful build version under the same
prefix; this is an explicit operator trust boundary, not an automatic
lifecycle path.

### Guard non-vacuity and TDD evidence

The guard now requires exactly one launch template, one required instance tag,
one required volume tag, every expected IAM/endpoint/bucket policy SID, both
trust-policy SIDs, one SSM shell step, one exact command, and all five exact
SSM parameters. The old broad guard-text search was replaced by 13 template
mutations covering missing/altered launch, user-data, tags, trust statements,
policy SIDs, SSM command, step, and parameters.

RED/GREEN evidence:

```text
Watchdog wiring RED:       2 failed, 19 passed
Watchdog wiring GREEN:     21 passed
IAM/KMS/S3 cleanup RED:    3 failed, 20 passed
IAM/KMS/S3 cleanup GREEN:  23 passed
Guard count RED:           1 failed, 36 passed
Guard count GREEN:         37 passed
```

The first `cfn-lint` review run correctly rejected a single 21,848-character
parameter with `E2001: 21848 is greater than the maximum of 4096`. Chunking
regressions then produced `3 failed, 34 passed`; the six-part implementation
returned the suite to `37 passed`.

Cross-component review caught that applying `kms:ViaService` to controller
`DescribeKey` would break Task 7's direct metadata check. Its focused RED was
`1 failed, 36 deselected`; the final full suite covers the corrected split
between direct metadata read and service-bound cryptographic use.

Fresh final validation:

```text
CFN_GUARD=.cfn-guard-tool/bin/cfn-guard \
  python -m pytest -q tests/test_aws_corpus_builder_foundation.py
37 passed in 1.71s

cfn-lint 1.53.2
exit 0, no findings

cfn-guard 3.2.0 validate --rules <guard> --data <template>
exit 0, no findings

python -m py_compile tests/test_aws_corpus_builder_foundation.py
git diff HEAD^ HEAD --check
git diff --check
all exit 0, no findings
```

CloudFormation Guard still has no PyPI distribution, but this follow-up
installed version 3.2.0 temporarily from crates.io. The real guard engine
validated the final template and rejected all 13 mutations; the temporary
binary was then removed.

### Package approval and standing cost

`PackageUri` remains caller-selected within `v2/packages/`; duplicating Task
7's software gate in this static template would not add authority. Task 7
first performs exact-version S3 verification, then binds the archive bytes and
SHA-256 to the canonical package manifest, binds the manifest revision to
metadata on that same immutable package object version, and requires the
closed production module authority set. The emitted launch intent therefore
authorizes the exact verified package object rather than any object that merely
matches the SSM URI pattern.

Deploying this stack starts standing non-compute charges. At the published
us-east-1 price of `$0.01` per interface-endpoint ENI-hour, seven one-AZ
interface endpoints are approximately `$51.10/month` at 730 hours. The
retained customer-managed KMS key starts at `$1/month`, for an initial standing
base of approximately **$52.10/month**, before endpoint data processing, KMS
requests, S3 storage/requests, and CloudWatch Logs. Automatic rotation adds
another `$1/month` after each of the first two rotations (capped after the
second). The S3 gateway endpoint has no endpoint-hour charge.

### Follow-up self-review

- Only the owned template, guard, test, and report files changed; no AWS API,
  deployment, or change-set operation occurred.
- `PENDING-REVIEW-FINDINGS.md` remains untracked and untouched.
- Critical 1 is closed. Critical 2's supported IAM controls are tightened, but
  the absence of EC2 condition keys for count/version/launch-time shutdown is
  an explicit remaining boundary, not silently widened policy.
- Governance cleanup removes billed object versions rather than merely adding
  delete markers, and is scoped away from package/source inputs.

## Second-review payload binding and residual-risk follow-up

Implementation commit:
`ba5579d441d8249db45d61041b8bdada3aab21a8`.

### Bootstrap payload hash authority

The Task 6 renderer is not present or importable in this foundation worktree,
and copying the renderer here would duplicate runtime logic and violate the
owned-file boundary. This follow-up therefore uses the review-approved small
fixture approach.

The committed test fixture is the SHA-256 of the ASCII bytes of the actual
base64 user-data payload produced by Task 6's `fixture_config` at reviewed
bootstrap commit `48835f7`:

```text
raw rendered bytes:  26,605
gzip bytes:           6,878
base64 characters:    9,172
base64 SHA-256:        0381b939d78fb48b4ba6593ce3f5bc1472f22ed798f9cd79388f7030a40e80aa
```

The foundation now requires `BootstrapUserDataSha256`, constrained to exactly
64 lowercase hexadecimal characters with no default, and emits the same value
as the closed, non-exported `BootstrapUserDataSha256` stack output. The test
feeds the reviewed fixture hash through the parameter constraint and asserts
that the output references that parameter, proving the authority value is
carried end to end by this stack.

This is deliberately not represented as content validation inside
CloudFormation: an authorized stack updater can still supply arbitrary chunks
and their matching digest because CloudFormation cannot hash the concatenated
parameter value. The separately assigned preflight change is the completing
control: it will hash the deployed launch template's actual user data, compare
that result with this stack output, freshly render the reviewed Task 6
bootstrap from the approved launch inputs, and require all three values to
match before authorizing a launch.

### Remaining guard non-vacuity

`launch_template_is_private_imdsv2_i4i` now requires exactly one
`NetworkInterfaces` entry and exactly one `BlockDeviceMappings` entry before
checking their nested fields. Two new mutations delete those arrays
independently. Both the Python selector model and the real CloudFormation Guard
engine reject each deletion; the mutation suite now contains 15 cases.

### Accepted SSM package-authorization residual

The SSM document is an integrity boundary, not an independent package
authorization boundary. A principal that can assume `ControllerRole` can call
`SendCommand` directly with any existing object under `v2/packages/*` and a
matching caller-supplied digest, bypassing the preflight software gate. The SSM
document cannot verify a package signature or cross-bind one parameter to
another.

Compensating controls are:

- the on-instance builder verifies the exact S3 object version and SHA-256
  before execution;
- preflight binds the production software gate to that exact verified package
  object and carries it into the approved launch authority; and
- `ControllerRole` is operator-only, separately assumed, and not attached to
  the builder instance.

The URI pattern was not narrowed to an invented build-scoped layout. The
current frozen infrastructure contract guarantees only `v2/packages/*`, and
SSM `allowedPattern` cannot require that a URI's path segment equals the
separate `BuildId`, an S3 version ID, or a signed authority. Requiring an
unfrozen path shape could reject valid approved packages without closing the
direct-call authorization gap. This residual therefore remains explicitly
accepted at the operator-role trust boundary.

### Second-review RED/GREEN and verification

Payload authority RED:

```text
3 failed, 35 passed in 0.45s
```

Payload parameter/output GREEN:

```text
38 passed in 0.34s
```

Launch-array non-vacuity RED and GREEN:

```text
RED:   1 failed, 39 passed in 0.46s
GREEN: 40 passed in 0.39s
```

Fresh real-guard mutation run:

```text
CFN_GUARD=.cfn-guard-tool/bin/cfn-guard \
  python -m pytest -q tests/test_aws_corpus_builder_foundation.py
40 passed in 2.60s

cfn-guard 3.2.0 validate --rules <guard> --data <template>
exit 0, no findings
```

`cfn-lint 1.53.2`, `python -m py_compile
tests/test_aws_corpus_builder_foundation.py`, and `git diff --check` each exited
0 with no findings. The temporary Guard binary was removed after validation.

### Second-review self-review

- The hash is explicitly defined over the concatenated base64 text, matching
  the downstream comparison contract; it is not ambiguously defined over the
  decoded gzip bytes or rendered shell text.
- The fixture proves stack plumbing but does not pretend to replace the live
  preflight render-and-compare gate.
- Both wildcard launch arrays now have required-match counts and executable
  deletion mutations.
- The SSM direct-call package risk is stated as residual authorization risk,
  not conflated with its strong object-integrity checks.
- No deployment, change set, AWS API call, push, amend, or edit outside the
  owned files occurred. `PENDING-REVIEW-FINDINGS.md` remains untracked.
