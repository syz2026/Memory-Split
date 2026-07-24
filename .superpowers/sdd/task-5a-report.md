# Task 5A Report: Native No-Instance AWS Foundation

## Status

`DONE` on `feat/memorysplit-v3-aws-task5a-foundation`, based exactly on
`b3471e0969ca2a997d33acf60d2e777720afa1c4`
(`feat/memorysplit-v3-aws-n10`).

Implementation commit:

- `5bc815f` — `feat: add private no-instance AWS foundation`

The source worktree at `/Users/stephenzhang/Documents/MemorySplit` was not
modified. All Task 5A work occurred in the isolated
`memorysplit-v3-aws-task5a-foundation` worktree.

## Delivered

- Added long-form-intrinsic CloudFormation with:
  - two private subnets and route tables with no default routes, IGW, NAT,
    public IP association, SSH key, or port 22;
  - S3 gateway plus EC2, EC2 Messages, ECR API/DKR, KMS, Logs, SSM,
    SSM Messages, and STS interface endpoints;
  - a zero-ingress train security group and endpoint-only HTTPS ingress;
  - a retained, versioned, public-blocked, SSE-KMS S3 bucket;
  - a retained, scan-on-push, immutable, KMS-encrypted ECR repository;
  - retained symmetric `SYMMETRIC_DEFAULT` data and asymmetric
    `ECC_NIST_P384` `SIGN_VERIFY` approval KMS keys;
  - distinct train, evaluator, controller, and signer roles, with only the
    train role in an instance profile and only the signer role allowed to
    call `kms:Sign`;
  - exact `MemorySplit-ArgvV1` SSM content and SHA-256
    `b022ab1f970289d8340e36cbf742a3195f894bdd97cd388590d9aa4acb8f6f2d`;
  - a p5.48xlarge launch template with IMDSv2, explicit no-public-IP
    networking, and an encrypted root volume, but no capacity resource; and
  - three monthly AWS Budget email notifications.
- Added cfn-guard rules for paid-capacity absence, private networking,
  IMDSv2, retained encryption, immutable ECR, and role separation.
- Added a pinned `cfn-lint==1.53.2` IaC development requirement.
- Added offline structural/security tests with an exact closed output set and
  secret-like-material scanning.
- Changed only AWS handoff path classification: tracked `infra/**` and
  `runtime/**` are excluded, while root `requirements-aws-p5.lock` remains
  forbidden. An end-to-end ZIP test proves neither excluded tree is emitted.

No EC2 instance, Auto Scaling group, EC2/Spot fleet, capacity reservation, or
capacity reservation fleet is present in the template.

## TDD evidence

Initial RED, before creating infrastructure or changing classification:

```text
2 failed, 10 errors
```

Observed causes were the missing foundation template, missing guard rules, and
`infra/**` being classified as `unknown`.

Additional validator-driven RED/GREEN cycles:

- cfn-lint rejected YAML 1.1 coercion of unquoted `Null`; a pytest regression
  reproduced the null mapping key before quoting it.
- cfn-lint rejected all three CIDR defaults against their original pattern; a
  pytest regression reproduced the mismatch before correcting the pattern.
- the IaC requirement pin test failed on unpinned `cfn-lint` before pinning the
  installed current version.

Final focused pytest command covered the new foundation suite, the complete AWS
handoff packager suite, and the canonical argv suite:

```text
132 passed in 56.74s
```

## Static validation

- PyYAML safe parse: passed, 32 resources.
- cfn-lint 1.53.2 (`us-east-1` schema): passed with no findings.
- cfn-guard 3.2.0: all 20 rules passed.
- The downloaded cfn-guard macOS arm64 archive matched the release SHA-256
  `6eb1d1693471499d47e453567a7d46f17171468ec31ff2aa98b91116b316b5c4`.
- `py_compile`: passed for the modified packager and tests.
- `git diff --check`: passed.

## Files

Created:

- `infra/aws/cloudformation/memorysplit-p5-foundation.yaml`
- `infra/aws/cfn-guard/memorysplit-p5-foundation.guard`
- `infra/aws/requirements-dev.txt`
- `tests/test_aws_cloudformation_foundation.py`
- `.superpowers/sdd/task-5a-report.md`

Modified:

- `scripts/package_aws_p5_handoff.py`
- `tests/test_package_aws_p5_handoff.py`

Intentionally unchanged:

- `runtime/**`
- controller, approval, and lifecycle implementation
- `corpusgen/**`, scientific configs, runbooks, and verifier behavior

## Boundaries

No deployment, AWS API operation, change set, paid-capacity creation, runtime
Dockerfile, controller/approval/lifecycle edit, corpus/config/runbook edit, or
unrelated verifier fix was performed. Live account preflight and deployment
remain later, separately approved tasks.

One initial template write hit host `ENOSPC`; only Task 5A-owned pytest scratch
directories were removed. The template, tests, and both offline AWS validators
then completed successfully.

Report path:

`.superpowers/sdd/task-5a-report.md`

## Review-fix addendum

The initial Task 5A review failed. Commit `2390b87`
(`fix: harden AWS foundation review boundaries`) fixes every Critical and
Important finding. This addendum supersedes the original packaging statement:
only the three implemented `infra/aws/**` files are explicitly omitted;
unallowlisted `infra/**` and every future `runtime/**` path now fail closed.

### Corrected boundaries

- Controller permissions now cover every AWS call rendered by the current
  `msctl/aws_p5.py`: STS identity; S3 get/versioned-get/put; EC2 image,
  instance, attribute, offering, tag, shutdown-behavior, and terminate calls;
  and SSM document discovery/create, managed-instance discovery,
  command send/reconcile/status/cancel calls.
- Read-only discovery remains action-exact. EC2 mutations require the approved
  launch template, p5.48xlarge, IMDSv2, the selected private subnet and train
  security group embedded in that template, and `MemorySplitManaged` request
  or resource tags. SSM sends require `MemorySplit-ArgvV1` plus tagged
  MemorySplit instances.
- Launch-template termination protection is explicitly disabled; the existing
  controller still enforces instance-initiated shutdown behavior and its
  evidence-gated cleanup can call `TerminateInstances` without an unavailable
  protection-disable step.
- Controller IAM and the data-key policy grant the exact documented EBS CMK
  launch actions: `Decrypt`, `DescribeKey`, `GenerateDataKeyWithoutPlaintext`,
  `ReEncryptFrom`, `ReEncryptTo`, plus `GenerateDataKey` for S3. `CreateGrant`
  is isolated and requires `kms:GrantIsForAWSResource`; wildcard grant/data-key
  actions and grant management are rejected by tests.
- `ArtifactRootPrefix` now binds IAM and the S3 endpoint to the same durable
  prefix used by `MS_S3_ROOT`. Train can read release/dataset inputs and write
  only operations, checkpoints, receipts, and canary round trips. Evaluator can
  read only sealed/checkpoint inputs and write/read only evaluation evidence
  and evaluation receipts; it cannot write checkpoints.
- `TrainingAvailabilityZone` and a distinct explicitly selected secondary AZ
  replace `Fn::GetAZs` first-two selection. The p5 launch template embeds the
  training subnet, so the operator's AZ-level p5 offering check and launch
  cannot drift.
- Forbidden-component classification now precedes top-level omission.
  Implemented IaC is secret-scanned from the pinned Git snapshot even though it
  is omitted. Unknown IaC/runtime paths are rejected and future runtime work
  must add each reviewed file explicitly. Root `requirements-aws-p5.lock`
  remains forbidden.
- cfn-guard names required logical resources directly, requires the controller
  policy resource, and checks train/evaluator prefix conditions, constrained
  launch/grant/tagged-termination/tagged-SSM statements, and signer authority.

### Review RED to GREEN evidence

- Controller call surface RED listed missing
  `DescribeInstanceAttribute`, `ModifyInstanceAttribute`, `CancelCommand`,
  `CreateDocument`, `DescribeInstanceInformation`, `ListDocuments`, and STS
  identity; GREEN: controller/role group `2 passed`.
- Termination compatibility RED observed `DisableApiTermination` was `true`;
  GREEN: `1 passed`.
- EBS CMK RED observed missing `ControllerOperationsPolicy`; GREEN: constrained
  KMS/controller group `4 passed`.
- S3 scoping RED observed missing `ListTrainPrefixes`; EC2/SSM scoping RED
  observed missing `RunApprovedLaunchTemplate`; GREEN: integrated IAM group
  `5 passed`.
- Packaging RED: `12 failed, 1 passed` for blanket IaC/runtime omission,
  forbidden-component ordering, and omitted-file secret scanning; GREEN:
  `16 passed, 98 deselected`. The pinned-snapshot compatibility regression
  then failed once and passed with both security scans (`3 passed`).
- AZ binding RED observed missing `TrainingAvailabilityZone`; GREEN: private
  network/AZ group `4 passed`.
- Guard RED observed vacuous `%launch_templates !empty`, then missing tagged
  SSM/termination invariants; GREEN: guard coverage tests passed and cfn-guard
  accepted all 21 strengthened rules.
- Durable-prefix RED observed missing `ArtifactRootPrefix`, then the broad S3
  endpoint resource; GREEN: exact train/evaluator/controller/endpoint prefix
  test `1 passed`.

### Review-fix verification

- Focused pytest: `152 passed in 64.68s`.
- Safe YAML parse: 33 resources.
- cfn-lint 1.53.2: passed with no findings.
- cfn-guard 3.2.0: all 21 rules passed.
- Guard mutation check: all 11 required-resource deletion mutants were
  rejected.
- The actual three omitted IaC files passed the packager's secret scanner.
- `py_compile` and `git diff --check`: passed.

No deployment, AWS API mutation, change set, cleanup-code edit, runtime
Dockerfile, or profile-neutral P6 work was performed. The operator must still
select an AZ where the separately executed p5.48xlarge offering check passes.

## Exact S3 role-separation addendum

Commit `2f6b4f9` (`fix: separate AWS artifact role prefixes`) closes the final
Important Task 5A review gap.

- Train list/read/write permissions now enumerate only release and dataset
  inputs, operation intents and instance-owned operation receipts, canary
  round trips, checkpoints, snapshots, logs, and the bootstrap, canary,
  checkpoint, interruption, and run receipt namespaces. Train has no evaluator,
  evaluation-receipt, or sealed namespace.
- Evaluator remains read-only on sealed, checkpoint, and checkpoint-receipt
  inputs. Its only writable namespaces are `evaluations/**` and
  `receipts/evaluations/**`.
- Controller read permissions enumerate immutable/lifecycle inputs needed by
  the current release, dataset, canary, environment, checkpoint/resume,
  operation reconciliation, evaluation collection, log/snapshot collection,
  and collection-receipt paths. Its writes are limited to operation intents,
  canary/environment receipts, dataset publication, and the current
  controller-owned checkpoint staging paths. It cannot write evaluator
  evidence or evaluator receipts and has no sealed read/write permission.
- Every role has an exact `s3:prefix` ListBucket condition. The S3 endpoint also
  enumerates named object namespaces instead of allowing the artifact-root
  wildcard.
- cfn-guard validates the named statements and all S3 read/write/list
  statements, so an extra statement cannot bypass the named-policy checks.

RED evidence:

- Exact Train/Controller tests produced `2 failed, 1 passed`: Train still listed
  broad `operations/**` and `receipts/**`, while Controller had no scoped list
  statement and retained `${ArtifactRootPrefix}/*`. Evaluator was already
  compliant.
- Endpoint RED observed the broad root resource where an enumerated list was
  required.
- Guard mutation RED showed both an injected broad Train statement and an
  injected Controller evaluator-write statement returning success before the
  all-statement queries were added.

GREEN and final evidence:

- Focused pytest: `154 passed in 115.29s`.
- Safe YAML parse: 33 resources.
- cfn-lint 1.53.2: passed with no findings.
- cfn-guard 3.2.0: all 22 rules passed.
- cfn-guard rejected 11 required-resource deletion mutants and 11 S3 policy
  mutants, including injected statements, artifact-root wildcards, Train
  evaluator access, Evaluator checkpoint writes, Controller evaluator writes,
  Controller sealed reads, broad list prefixes, and a broad endpoint resource.
- The actual omitted IaC files passed secret scanning; `py_compile` and
  `git diff --check` passed.

No deployment, AWS API mutation, change set, runtime implementation, cleanup
change, or profile-neutral P6 work was performed.

## Final endpoint-enforcement addendum

Commit `e4f8dd3` (`fix: close S3 endpoint policy bypasses`) closes the final two
Important foundation enforcement findings.

- The S3 gateway endpoint now separates unconditional
  `GetBucketLocation` from `ListBucket`/`ListBucketVersions`. Listing requires
  one of the exact named `ArtifactRootPrefix` namespaces and rejects the root,
  root wildcard, missing prefix, and unrelated prefixes.
- Python invariants iterate every Train, Evaluator, Controller, and S3 endpoint
  statement. They reject wildcard actions (including `*`, `s3:*`, and
  `s3:Get*`), case variants, `NotAction`, `NotResource`, `Resource: "*"`,
  plain or unknown resource shapes, artifact-root wildcards, unexpected
  actions/Sids, malformed conditions, and extra S3 statements.
- cfn-guard now uses case-insensitive all-S3 selectors plus whole-policy
  statement checks. Every statement must use a non-wildcard string/list
  action, no `NotAction`/`NotResource`, an allowlisted Sid/action, and only
  reviewed `Fn::Sub` or exact artifact-bucket `Fn::GetAtt` resources.
- Endpoint listing and object resources are independently allowlisted; action-
  filtered or `Fn::Sub`-only projections can no longer hide an injected
  statement or malformed resource.

RED evidence:

- Endpoint listing test failed because `MemorySplitBucketLocation` and
  conditioned `MemorySplitBucketListing` did not exist.
- Guard coverage test failed because the all-statement rejection rule was
  absent.
- Python mutation tests exposed case-insensitive `S3:GetObject` and an injected
  `Action: "*"` statement as unobserved before selectors were hardened.
- Expanded cfn-guard mutants exposed an unknown resource mapping as accepted
  before exact resource-key shape checks were added.

GREEN and final evidence:

- Focused pytest: `173 passed in 88.03s`.
- Safe YAML parse: 33 resources.
- cfn-lint 1.53.2: passed with no findings.
- cfn-guard 3.2.0: all 23 rules passed.
- Python accepted the canonical policy and rejected all 17 bypass forms.
- cfn-guard rejected 11 required-resource deletion mutants and 26 endpoint/S3
  enforcement mutants.
- The actual omitted IaC files passed secret scanning; `py_compile` and
  `git diff --check` passed.

No deployment, AWS API mutation, change set, runtime implementation, cleanup
change, or profile-neutral P6 work was performed.

## Whole-template policy-container addendum

Commit `f1c82db` (`fix: close whole-template policy bypasses`) closes the final
policy-container bypass.

- The IAM resource set is now exactly four roles and one train instance
  profile. Each role has exactly one named inline policy. Train alone retains
  the reviewed `AmazonSSMManagedInstanceCore` ARN; Evaluator, Controller, and
  Signer have no managed-policy attachments or permission boundary.
- `ControllerOperations` moved inline, eliminating all
  `AWS::IAM::Policy`/`AWS::IAM::ManagedPolicy` resources. The controller inline
  document is 8,338 compact characters, below the 10,240-character role quota.
- Data-key policy access is still role-separated without a dependency cycle:
  the account principal is constrained by
  `aws:PrincipalTag/memorysplit:role=controller`, and EBS grants still require
  `kms:GrantIsForAWSResource`.
- The bucket transport deny now enumerates only reviewed S3 actions; no
  `s3:*` remains.
- Python recursively inventories every `PolicyDocument`,
  `AssumeRolePolicyDocument`, and KMS `KeyPolicy` anywhere under Resources.
  Policy paths and IAM container topology must match the reviewed template;
  every S3-capable statement must be an exact reviewed statement.
- cfn-guard fixes total/IAM/role/profile/policy counts, inline policy names and
  counts, managed attachments, bucket-policy count, and nested policy-container
  count. It rejects standalone/managed IAM policies, extra IAM resources,
  appended managed policies, extra policy resources, nested policy documents,
  and S3 statements in trust/KMS/signer containers.

RED evidence:

- Exact IAM and canonical whole-policy tests failed on the external
  `ControllerOperationsPolicy` and bucket-policy `s3:*`.
- Initial Python container mutations exposed five unobserved attachment/IAM
  topology cases (managed-policy ARNs, extra profile, and IAM user) before the
  exact IAM inventory was incorporated.
- Guard development confirmed that unconstrained property traversal was
  insufficient; final enforcement uses exact resource counts plus explicit
  reviewed direct, inline, trust, key, and nested policy containers.

GREEN and final evidence:

- Focused pytest: `190 passed in 179.03s`.
- Safe YAML parse: 32 resources.
- cfn-lint 1.53.2: passed with no findings and no dependency cycle.
- cfn-guard 3.2.0: all 24 rules passed.
- Python rejected all 15 policy-container mutants plus all prior S3 bypass
  forms.
- cfn-guard rejected 15 policy-container mutants, 11 required-resource
  deletion mutants, and 30 S3 enforcement mutants.
- The actual omitted IaC files passed secret scanning; `py_compile` and
  `git diff --check` passed.

No deployment, AWS API mutation, change set, runtime implementation, cleanup
change, or profile-neutral P6 work was performed.
