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
