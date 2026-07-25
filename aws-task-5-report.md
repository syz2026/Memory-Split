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
