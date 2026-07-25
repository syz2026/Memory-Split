# AWS Corpus Builder Task 8 Report

## Status and scope

- Branch: `feat/aws-corpus-launch`
- Base: `f098362ce53b85fc45bb4293022ae37c9a2c020d`
- Status: complete
- Owned implementation:
  - `cluster/aws/corpus_builder/launch.py`
  - `scripts/aws_corpus_builder_launch.py`
  - `tests/test_aws_corpus_builder_launch.py`
- No real AWS call was made. Every launch, describe, attribute, pricing, and
  termination test used an injected fake.
- The supplied `aws-task-8-brief.md` remains untracked and was not committed.

## Commits

1. `e7eefc92f39df0e0ae1e4c5e8db6ec8b14fc95aa` —
   `test: define approved corpus builder launch boundary`
2. `f10c29aab255ff23e4863f9f692630f5193670dd` —
   `feat: launch explicitly approved corpus builders`

## RED/GREEN evidence

### Baseline

Command:

```text
python -m pytest -q tests/test_aws_corpus_builder_contracts.py
```

Result before Task 8 edits: exit 0, `46 passed in 0.07s`.

### Initial approval boundary

RED command:

```text
python -m pytest -q tests/test_aws_corpus_builder_launch.py
```

RED result: exit 2 during collection, exactly as planned:

```text
ModuleNotFoundError: No module named 'cluster.aws.corpus_builder.launch'
1 error in 0.12s
```

First GREEN result after the launch module and CLI were implemented:

```text
26 passed in 0.34s
```

### Real shutdown attribute and malformed multi-instance response

The botocore EC2 service model confirmed that `DescribeInstances` does not
return `InstanceInitiatedShutdownBehavior`; it must be read with
`DescribeInstanceAttribute`.

RED result after changing the fake to the real response shape and adding a
malformed second-instance case:

```text
4 failed, 23 passed in 0.44s
```

The failures were the happy launch, malformed second instance, polling path,
and CLI path. GREEN after adding injected `DescribeInstanceAttribute` and
counting every returned instance row:

```text
27 passed in 0.36s
```

### Public-address and exact CLI approval boundary

RED after adding IPv6-public-address rejection and disabling abbreviated
approval flags:

```text
2 failed, 27 passed in 0.37s
```

GREEN:

```text
29 passed in 0.33s
```

### Explicit no-public-IP launch-template gate

RED after requiring explicit
`NetworkInterfaces[0].AssociatePublicIpAddress == false`, rather than relying
only on subnet defaults:

```text
1 failed, 29 passed in 0.37s
```

GREEN:

```text
30 passed in 0.33s
```

### Actual EC2 instance response compatibility

The botocore model also confirmed that `DescribeInstances` does not return a
`LaunchTemplate` field. The exact numeric template is instead checked before
launch, and the request itself contains only that pinned template.

RED after removing the synthetic field from the fake:

```text
3 failed, 26 passed in 0.41s
```

GREEN after removing the impossible post-launch field check:

```text
29 passed in 0.33s
```

### Controller-authorized termination tags

RED after changing the expected tags to the foundation contract:
`MemorySplitCorpusBuilder=true` plus a `Name` that embeds the full approved
intent SHA-256:

```text
3 failed, 26 passed in 0.39s
```

GREEN:

```text
29 passed in 0.33s
```

This keeps RunInstances compatible with the controller role's tag-key
allowlist and ensures terminate-on-mismatch is authorized.

## Final verification

Focused tests:

```text
$ python -m pytest -q tests/test_aws_corpus_builder_launch.py
.............................                                            [100%]
29 passed in 0.32s
```

Contract regression:

```text
$ python -m pytest -q tests/test_aws_corpus_builder_contracts.py
..............................................                           [100%]
46 passed in 0.06s
```

Compilation:

```text
$ python -m py_compile cluster/aws/corpus_builder/launch.py scripts/aws_corpus_builder_launch.py
```

Result: exit 0, no output.

Whitespace:

```text
$ git diff --check
```

Result: exit 0, no output.

AWS SDK shape check: botocore parameter validation accepted all six injected
request shapes (`GetProducts`, `DescribeLaunchTemplateVersions`,
`RunInstances`, `DescribeInstances`, `DescribeInstanceAttribute`, and
`TerminateInstances`) and confirmed every inspected EC2 response field.

## Self-review

- Approval is fail-closed: lowercase exact SHA-256, constant-time comparison,
  canonical Task 1 parsing, and strict expiry before any AWS call.
- The live Linux On-Demand price must exactly equal the approved intent and
  remain at or below `$5.491/hour`.
- The immutable launch-template ID/version and safety data are re-read before
  launch, including exact AMI, `i4i.16xlarge`, profile, subnet, security group,
  explicit no-public-IP networking, IMDSv2, On-Demand tenancy, and terminating
  shutdown behavior.
- `RunInstances` is called exactly once with only `LaunchTemplate`,
  `MinCount=1`, `MaxCount=1`, and approved instance tags. There are no AMI,
  type, network, role, storage, or user-data overrides.
- Every returned instance row is counted. Any second or malformed row causes
  all known returned instance IDs to be terminated before raising.
- Post-launch checks cover exact type, AMI, subnet, one security group, one
  interface, profile, IMDSv2, tags, Linux On-Demand lifecycle, pending/running
  state, no public IPv4 or IPv6, and shutdown behavior from the real EC2
  attribute API.
- Every mismatch or read failure after a known launch invokes
  `TerminateInstances` before raising; cleanup failure is surfaced explicitly.
- The CLI requires the full approval SHA flag, exact `sbsandbox` profile and
  `us-east-1` region, disables option abbreviation, and defines no `--yes`.
- The reported terminate-at time is the 24-hour compute ceiling; the Task 6
  watchdog is expected to initiate shutdown earlier at 23 hours 30 minutes.

## Concerns

No code blocker remains. Operationally, `sbsandbox` must retain read access to
the Pricing API for the mandatory last-moment price recheck; if it does not,
the CLI fails closed before `RunInstances`.

## Review fixes after `bf4a182`

### Fix commits

1. `bac63fe1505ea03bb5b72f0a2d7e3e4ee1528bc8` —
   `test: expose corpus launch safety gaps`
2. `148b05702449f2b3c7e8aa9f05fcd0fe2b4865d3` —
   `fix: harden approved corpus launches`
3. `b92780ee60262ec4c8e2ebe8cb50261be69aae49` —
   `test: require confirmed launch cleanup`
4. `5ea16902e9738474db9b34758229c659783da474` —
   `fix: confirm failed launch termination`

### Review-finding verification

The reviewed implementation had no `ClientToken` or pre-launch tagged-instance
lookup, reused the initial `now` value after the live rechecks, made one
unconfirmed termination call, never checked STS identity or security-group
ingress, synthesized `launch_time` from `now`, and accepted arbitrary
`i-...` strings. Each finding reproduced against reviewed head `bf4a182`.

### Replay, account, expiry, ingress, and instance-ID TDD

RED after adding focused replay, deterministic token, final-clock, three-way
account, pre/post ingress, and real instance-ID tests:

```text
8 failed, 29 passed in 0.88s
```

The strict instance-ID case independently failed with the old permissive
validation:

```text
1 failed in 0.10s
```

GREEN after the defense-in-depth implementation:

```text
38 passed in 0.35s
```

The launch request now uses the approved 64-character SHA-256 as its EC2
`ClientToken`. Before launch, `DescribeInstances` searches pending, running,
stopping, and stopped instances for both `MemorySplitCorpusBuilder=true` and
the existing `Name=memorysplit-corpus-builder-<intent-sha256>` tag. Encoding
the hash in `Name` preserves the foundation controller role's two-key tag
allowlist while closing the replay lookup.

The controller now calls injected STS `GetCallerIdentity` and requires account
`056956104102`; the intent's AMI owner and the account segment of its
instance-profile ARN must match. The existing `LaunchIntent` fields were
sufficient, so no contract change was required.

### Confirmed cleanup and EC2 lifecycle-time TDD

RED after adding transient termination, never-confirmed termination, and
actual `LaunchTime` cases:

```text
3 failed in 0.11s
```

A malformed termination response also exposed an unhandled response-shape
case:

```text
1 failed in 0.09s
```

GREEN evidence:

```text
3 passed in 0.06s
1 passed in 0.03s
41 passed in 0.07s
```

Termination is retried at most three times. Cleanup then polls each known
instance ID up to 20 times until EC2 reports `shutting-down` or `terminated`;
`InvalidInstanceID.NotFound` confirms that individual ID is gone. Failure to
confirm raises an error containing every instance ID and this directly usable
manual command:

```text
aws ec2 terminate-instances --profile sbsandbox --region us-east-1 --instance-ids <instance-id>
```

The success report now uses `DescribeInstances.LaunchTime`, and computes
`terminate_at` exactly 24 hours from that EC2-reported timestamp.

### Review-fix final verification

```text
$ python -m pytest -q tests/test_aws_corpus_builder_launch.py
.........................................                                [100%]
41 passed in 0.08s

$ python -m pytest -q tests/test_aws_corpus_builder_contracts.py
..............................................                           [100%]
46 passed in 0.05s

$ python -m py_compile cluster/aws/corpus_builder/launch.py scripts/aws_corpus_builder_launch.py
# exit 0, no output

$ git diff --check
# exit 0, no output
```

Botocore parameter validation accepted all eight request shapes used by the
controller, including the new `ClientToken`, `DescribeSecurityGroups`, and STS
`GetCallerIdentity` calls. No AWS or network call was made.

### Review-fix self-review

- Repeated approvals are rejected when the intent-tagged instance is active;
  concurrent pre-check races still converge on one EC2 request through the
  deterministic `ClientToken`.
- A fresh injected clock is read after account, template, price, ingress, and
  replay rechecks and immediately before `RunInstances`.
- Account identity, AMI-owner account, and instance-profile account all fail
  closed unless they equal `056956104102`.
- The security group must have exactly zero `IpPermissions` both before
  launch and after the instance becomes describable; post-launch drift invokes
  confirmed cleanup.
- Instance IDs accept only AWS's 8- or 17-lowercase-hex forms.
- Every known non-compliant instance is retried and polled until an accepted
  shutdown state. Unconfirmed cleanup cannot return success and gives an
  operator the exact recovery command.
- The launch request still has no AMI, type, network, role, storage, or
  user-data override. Its only new launch parameter is the idempotency token.
- The CLI still rejects generic or abbreviated approval and remains fixed to
  `sbsandbox` and `us-east-1`.

### Review-fix concerns

No contract change or code blocker remains. Deployment integration must ensure
the controller role retains read-only `ec2:DescribeSecurityGroups`,
`ec2:DescribeInstances`, launch-template, and Pricing permissions. STS
`GetCallerIdentity` itself does not require an allow policy.
