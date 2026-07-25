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
