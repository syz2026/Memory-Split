# AWS Claim-Bearing Corpus Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, launch, and verify a bounded AWS `i4i.16xlarge` workflow that materializes the frozen `memorysplit-parallel-corpus-v2` corpus and publishes independently verifiable versioned S3 artifacts.

**Architecture:** Keep scientific corpus generation local-filesystem-first and deterministic. Add a narrow AWS orchestration layer that packages one clean revision, launches one private CPU/storage builder from a no-instance CloudFormation foundation, records every phase through canonical versioned S3 receipts, and declares success only after a clean-room re-download verifies the complete corpus. The existing Wikidata derived-view plan is a prerequisite and is not duplicated here.

**Tech Stack:** Python 3.12, pytest, boto3-compatible EC2/S3/KMS/SSM clients, AWS CloudFormation, cfn-lint, cfn-guard, Amazon Linux 2023, systemd, mdadm, XFS, versioned SSE-KMS S3, EC2 local NVMe.

## Global Constraints

- Binding design: `docs/superpowers/specs/2026-07-24-aws-corpus-builder-design.md`.
- Software prerequisite: all four tasks in `docs/superpowers/plans/2026-07-24-wikidata-derived-view.md` pass on the same clean revision.
- Region: `us-east-1`.
- Builder: exactly one Linux On-Demand `i4i.16xlarge`.
- Builder geometry: 64 vCPUs, 512 GiB RAM, four 3.75 TB local NVMe SSDs, 15 TB total local NVMe.
- Root storage: 200 GiB encrypted `gp3`, deleted on termination.
- Bucket: `memorysplit-corpus-056956104102-us-east-1`, versioning enabled, Block Public Access enabled, dedicated SSE-KMS key.
- Namespace: `v2/builds/{build_id}/`.
- Observed compute price ceiling: `$5.491/hour`; maximum instance lifetime 24 hours; maximum EC2 compute charge `$131.78`.
- Start the watchdog at boot and initiate shutdown at 23 hours 30 minutes.
- Set instance-initiated shutdown behavior to `terminate`.
- Require IMDSv2, no public IP, no inbound security-group rules, and SSM-only operator access.
- Do not use P4, P5, P6, Spot, Capacity Blocks, legacy pilot corpora, fixture corpora, or collaborator ZIP payloads.
- Do not launch or mutate AWS from Tasks 1–8. Those tasks implement and verify dry-run-capable software and no-instance infrastructure only.
- Task 9 may render a CloudFormation change set and EC2 launch intent, but applying the stack and launching the instance require separate explicit user approval of the exact cost and resources.
- Do not execute a Git commit step unless the user explicitly authorizes commits during implementation.

## File Map

- `cluster/profiles/aws-i4i.16xlarge-corpus-v1.json`: closed hardware, region, storage, time, and price contract.
- `cluster/aws/corpus_builder/contracts.py`: canonical profile, package, phase, object-version, build, verification, and launch-intent records.
- `cluster/aws/corpus_builder/package.py`: deterministic closed-allowlist builder package.
- `cluster/aws/corpus_builder/s3.py`: exact-version upload, HEAD verification, download, and receipt publication.
- `cluster/aws/corpus_builder/driver.py`: phase state machine and local corpus build orchestration.
- `cluster/aws/corpus_builder/bootstrap.py`: deterministic user-data and systemd unit rendering.
- `cluster/aws/corpus_builder/preflight.py`: read-only AWS and package gates plus EC2 dry-run.
- `cluster/aws/corpus_builder/launch.py`: exact-intent launch request construction and explicit-approval enforcement.
- `scripts/package_aws_corpus_builder.py`: package CLI.
- `scripts/aws_corpus_builder_preflight.py`: preflight CLI.
- `scripts/aws_corpus_builder_launch.py`: approved launch CLI.
- `scripts/aws_corpus_cleanroom_verify.py`: empty-root S3 re-download and corpus verification CLI.
- `infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml`: retained data plane, private networking, roles, SSM document, and no-instance launch template.
- `infra/aws/cfn-guard/memorysplit-corpus-builder.guard`: static safety invariants.
- `infra/aws/requirements-dev.txt`: pinned IaC validation dependency.
- `tests/test_aws_corpus_builder_contracts.py`: closed contract tests.
- `tests/test_package_aws_corpus_builder.py`: deterministic package tests.
- `tests/test_aws_corpus_builder_s3.py`: exact-version publication and clean-room tests.
- `tests/test_aws_corpus_builder_driver.py`: phase, recovery, and fail-closed orchestration tests.
- `tests/test_aws_corpus_builder_bootstrap.py`: RAID, watchdog, shutdown, and environment tests.
- `tests/test_aws_corpus_builder_foundation.py`: CloudFormation and IAM invariants.
- `tests/test_aws_corpus_builder_preflight.py`: read-only gate and dry-run tests.
- `tests/test_aws_corpus_builder_launch.py`: explicit approval and exact launch request tests.
- `docs/AWS-CORPUS-BUILDER-RUNBOOK.md`: operator preflight, approval, monitoring, verification, and teardown.

---

### Task 1: Closed Builder Profile and Canonical Contracts

**Files:**
- Create: `cluster/aws/corpus_builder/__init__.py`
- Create: `cluster/aws/corpus_builder/contracts.py`
- Create: `cluster/profiles/aws-i4i.16xlarge-corpus-v1.json`
- Create: `tests/test_aws_corpus_builder_contracts.py`

**Interfaces:**
- Consumes: canonical JSON bytes and SHA-256 strings.
- Produces:

```python
@dataclass(frozen=True)
class CorpusBuilderProfile:
    profile_id: str
    region: str
    instance_type: str
    vcpus: int
    memory_mib: int
    nvme_devices: int
    nvme_total_gib: int
    root_volume_gib: int
    max_runtime_seconds: int
    watchdog_shutdown_seconds: int
    max_hourly_usd: Decimal
    max_compute_usd: Decimal
    bucket_name: str
    key_prefix: str

@dataclass(frozen=True)
class S3ObjectVersion:
    uri: str
    version_id: str
    bytes: int
    sha256: str
    etag: str
    sse_algorithm: str
    kms_key_arn: str

@dataclass(frozen=True)
class PhaseReceipt:
    format: str
    schema_version: int
    build_id: str
    phase: str
    package_sha256: str
    source_lock_sha256: str
    objects: tuple[S3ObjectVersion, ...]

@dataclass(frozen=True)
class LaunchIntent:
    format: str
    schema_version: int
    profile_sha256: str
    package: S3ObjectVersion
    source_manifest: S3ObjectVersion
    ami_id: str
    ami_owner_id: str
    launch_template_id: str
    launch_template_version: str
    subnet_id: str
    security_group_id: str
    instance_profile_arn: str
    hourly_usd: Decimal
    max_compute_usd: Decimal
    not_after: str
```

- [ ] **Step 1: Write closed-parser RED tests**

```python
def test_builder_profile_is_exact_and_cost_bounded():
    profile = load_corpus_builder_profile(PROFILE_PATH)
    assert profile.instance_type == "i4i.16xlarge"
    assert profile.region == "us-east-1"
    assert profile.max_runtime_seconds == 86_400
    assert profile.watchdog_shutdown_seconds == 84_600
    assert profile.max_hourly_usd == Decimal("5.491")
    assert profile.max_compute_usd == Decimal("131.78")

def test_contract_parsers_reject_unknown_missing_duplicate_and_noncanonical_fields():
    for payload in malformed_contract_payloads():
        with pytest.raises(ValueError):
            phase_receipt_from_bytes(payload)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_contracts.py
```

Expected: collection fails because `cluster.aws.corpus_builder.contracts` is absent.

- [ ] **Step 3: Add the exact profile**

Create `cluster/profiles/aws-i4i.16xlarge-corpus-v1.json` with canonical JSON equivalent to:

```json
{
  "bucket_name": "memorysplit-corpus-056956104102-us-east-1",
  "instance_type": "i4i.16xlarge",
  "key_prefix": "v2/builds",
  "max_compute_usd": "131.78",
  "max_hourly_usd": "5.491",
  "max_runtime_seconds": 86400,
  "memory_mib": 524288,
  "nvme_devices": 4,
  "nvme_total_gib": 15000,
  "profile_id": "aws-i4i.16xlarge-corpus-v1",
  "region": "us-east-1",
  "root_volume_gib": 200,
  "vcpus": 64,
  "watchdog_shutdown_seconds": 84600
}
```

- [ ] **Step 4: Implement strict canonical parsers**

Implement:

```python
def load_corpus_builder_profile(path: Path) -> CorpusBuilderProfile: ...
def corpus_builder_profile_to_bytes(profile: CorpusBuilderProfile) -> bytes: ...
def s3_object_version_from_dict(value: object) -> S3ObjectVersion: ...
def phase_receipt_from_bytes(payload: bytes) -> PhaseReceipt: ...
def phase_receipt_to_bytes(receipt: PhaseReceipt) -> bytes: ...
def launch_intent_from_bytes(payload: bytes) -> LaunchIntent: ...
def launch_intent_to_bytes(intent: LaunchIntent) -> bytes: ...
```

Use UTF-8, sorted keys, compact separators, `allow_nan=False`, one terminal
newline, duplicate-key rejection, closed fields, exact decimal strings, safe
S3 URIs, lowercase SHA-256, UTC timestamps, and sorted unique object URIs.

- [ ] **Step 5: Run GREEN and regressions**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_contracts.py
python -m py_compile cluster/aws/corpus_builder/contracts.py
git diff --check
```

Expected: all tests pass and both static checks are silent.

- [ ] **Step 6: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/__init__.py \
  cluster/aws/corpus_builder/contracts.py \
  cluster/profiles/aws-i4i.16xlarge-corpus-v1.json \
  tests/test_aws_corpus_builder_contracts.py
git commit -m "feat: define bounded AWS corpus builder contracts"
```

Execute only with explicit commit authorization.

---

### Task 2: Deterministic Closed-World Builder Package

**Files:**
- Create: `cluster/aws/corpus_builder/package.py`
- Create: `scripts/package_aws_corpus_builder.py`
- Create: `tests/test_package_aws_corpus_builder.py`

**Interfaces:**
- Consumes: a clean Git root, exact revision, and output directory.
- Produces:

```python
class PackageError(ValueError):
    pass

@dataclass(frozen=True)
class BuilderPackage:
    archive: Path
    manifest: Path
    sha256_file: Path
    revision: str
    sha256: str
    bytes: int
    members: tuple[str, ...]

def build_corpus_builder_package(
    source_root: Path,
    output_dir: Path,
) -> BuilderPackage: ...
```

- [ ] **Step 1: Write package RED tests**

```python
def test_builder_package_is_byte_identical_across_two_builds(tmp_path):
    first = build_corpus_builder_package(CLEAN_REPO, tmp_path / "a")
    second = build_corpus_builder_package(CLEAN_REPO, tmp_path / "b")
    assert first.sha256 == second.sha256
    assert first.archive.read_bytes() == second.archive.read_bytes()

def test_builder_package_rejects_dirty_tree_secret_and_unreviewed_member(tmp_path):
    for mutation in ("dirty", "secret", "foreign-member"):
        with pytest.raises(PackageError):
            build_mutated_package(tmp_path, mutation)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Expected: collection fails because `cluster.aws.corpus_builder.package` is absent.

- [ ] **Step 3: Implement the closed allowlist**

Include only:

```python
REQUIRED_PREFIXES = (
    "cluster/aws/corpus_builder/",
    "cluster/profiles/",
    "configs/",
    "corpusgen/parallel/",
    "corpusgen/reasoning_v2/",
    "scripts/",
    "sources/",
    "vendor/tiktoken/",
)
REQUIRED_FILES = (
    "requirements.txt",
    "scripts/build_parallel_corpus.py",
    "scripts/package_aws_corpus_builder.py",
)
```

`requirements.txt` is the authoritative Python dependency declaration; this
repository does not introduce a second `pyproject.toml` dependency surface.
Require the completed Wikidata APIs, production adapters, source locks, frozen
recipe, tokenizer assets, tests, and builder profile. Exclude `.git`, caches,
outputs, checkpoints, pilot artifacts, credentials, symlinks, sockets, devices,
and files not present in the committed tree. Before Task 4 removes
`UnsupportedProductionRenderer`, package construction must fail with
`INCOMPLETE_PRODUCTION_PIPELINE`; deterministic real-tree package verification
is completed in Task 9 after the production path exists.

- [ ] **Step 4: Implement deterministic archive bytes**

Use sorted POSIX paths, regular files only, normalized mode `0644` or `0755`,
uid/gid zero, empty owner/group names, mtime zero, deterministic gzip metadata,
and a canonical manifest containing revision, path, mode, byte count, object
ID, and SHA-256.

- [ ] **Step 5: Add the CLI**

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    package = build_corpus_builder_package(args.source_root, args.output_dir)
    print(package.manifest.read_text(encoding="utf-8"), end="")
    return 0
```

- [ ] **Step 6: Run GREEN and deterministic rebuild verification**

Run:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-package-a
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-package-b
cmp \
  /tmp/memorysplit-corpus-package-a/memorysplit-corpus-builder.tar.gz \
  /tmp/memorysplit-corpus-package-b/memorysplit-corpus-builder.tar.gz
git diff --check
```

Expected: tests pass, `cmp` exits zero, and diff-check is silent.

- [ ] **Step 7: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/package.py \
  scripts/package_aws_corpus_builder.py \
  tests/test_package_aws_corpus_builder.py
git commit -m "feat: package deterministic AWS corpus builder"
```

Execute only with explicit commit authorization.

---

### Task 3: Exact-Version S3 Publication and Clean-Room Download

**Files:**
- Create: `cluster/aws/corpus_builder/s3.py`
- Create: `scripts/aws_corpus_cleanroom_verify.py`
- Create: `tests/test_aws_corpus_builder_s3.py`

**Interfaces:**
- Consumes: a versioned SSE-KMS S3 client, KMS key ARN, local artifact path,
  canonical receipt bytes, and exact `S3ObjectVersion` records.
- Produces:

```python
class PublicationError(ValueError):
    pass

class S3Client(Protocol):
    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def list_object_versions(self, **kwargs: object) -> Mapping[str, object]: ...

def publish_exact_file(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    path: Path,
    kms_key_arn: str,
    metadata: Mapping[str, str],
) -> S3ObjectVersion: ...

def verify_exact_object(
    s3: S3Client,
    expected: S3ObjectVersion,
) -> None: ...

def download_exact_object(
    s3: S3Client,
    expected: S3ObjectVersion,
    destination: Path,
) -> None: ...

def publish_phase_receipt(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    receipt: PhaseReceipt,
    kms_key_arn: str,
) -> S3ObjectVersion: ...
```

- [ ] **Step 1: Write exact-version RED tests**

```python
def test_publish_requires_version_id_kms_hash_bytes_etag_and_exact_head(tmp_path):
    s3 = VersionedFakeS3()
    record = publish_exact_file(s3, **artifact_request(tmp_path))
    assert record.version_id
    verify_exact_object(s3, record)

@pytest.mark.parametrize(
    "drift",
    ("missing-version", "wrong-kms", "wrong-size", "wrong-sha", "wrong-etag"),
)
def test_publish_and_download_reject_every_authority_drift(tmp_path, drift):
    with pytest.raises(PublicationError):
        exercise_drift(tmp_path, drift)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_s3.py
```

Expected: collection fails because `cluster.aws.corpus_builder.s3` is absent.

- [ ] **Step 3: Implement streaming upload and exact HEAD verification**

Hash the pinned local file descriptor before upload, upload with
`ServerSideEncryption="aws:kms"` and the exact key ARN, require a non-empty
version ID, then call `head_object` with that version. Require exact byte count,
ETag, metadata SHA-256, KMS key, and encryption algorithm.

- [ ] **Step 4: Implement no-overwrite receipt publication**

Before publishing a receipt key, list existing versions. Reuse only a version
whose bytes and metadata exactly match. Reject every conflicting version; do
not overwrite or delete it.

- [ ] **Step 5: Implement clean-room download**

Create the destination with `O_EXCL`, stream one exact version, fsync it, verify
byte count and SHA-256, and re-HEAD the source version after download. Reject
foreign files and symlinks in the destination root.

- [ ] **Step 6: Implement the CLI**

The CLI accepts:

```text
--receipt-uri
--receipt-version-id
--receipt-sha256
--destination
--expected-build-id
--profile sbsandbox
--region us-east-1
```

It downloads the final receipt and every exact object version to an empty root,
then calls `verify_parallel_corpus(destination, expected_build_id=...)`.

- [ ] **Step 7: Run GREEN**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_s3.py
python -m py_compile \
  cluster/aws/corpus_builder/s3.py \
  scripts/aws_corpus_cleanroom_verify.py
git diff --check
```

Expected: all tests pass and static checks are silent.

- [ ] **Step 8: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/s3.py \
  scripts/aws_corpus_cleanroom_verify.py \
  tests/test_aws_corpus_builder_s3.py
git commit -m "feat: publish versioned corpus artifacts exactly"
```

Execute only with explicit commit authorization.

---

### Task 4: Resumable Corpus Build Driver

**Files:**
- Create: `cluster/aws/corpus_builder/driver.py`
- Create: `tests/test_aws_corpus_builder_driver.py`
- Modify: `corpusgen/parallel/adapters.py`
- Modify: `scripts/build_parallel_corpus.py`
- Modify: `tests/test_parallel_corpus.py`

**Interfaces:**
- Consumes: completed production adapters, verified source locks, a deterministic
  package, profile, S3 client, and a private NVMe work root.
- Produces:

```python
@dataclass(frozen=True)
class VerifiedProductionInputs:
    catalog: InputCatalog
    renderer: Renderer
    sidecars: Mapping[str, Path]

def load_verified_production_inputs(
    *,
    source_lock_path: Path,
    source_root: Path,
    derived_root: Path,
    expected_generator_commit: str,
) -> VerifiedProductionInputs: ...

class CommandRunner(Protocol):
    def run(self, phase: str, action: Callable[[], object]) -> object: ...

PHASES = (
    "source-stage",
    "wikidata-view",
    "catalog",
    "render-pack",
    "local-verify",
    "s3-publish",
    "cleanroom-verify",
)

@dataclass(frozen=True)
class CorpusBuildRequest:
    build_id: str
    package_sha256: str
    source_lock_path: Path
    source_lock_sha256: str
    work_root: Path
    output_root: Path
    workers: int
    shard_count: int
    bucket: str
    prefix: str
    kms_key_arn: str

def run_corpus_build(
    request: CorpusBuildRequest,
    *,
    s3: S3Client,
    runner: CommandRunner,
) -> S3ObjectVersion: ...
```

- [ ] **Step 1: Write phase and recovery RED tests**

```python
def test_driver_runs_all_phases_and_publishes_final_receipt_last(tmp_path):
    result = run_corpus_build(request(tmp_path), s3=FakeS3(), runner=FakeRunner())
    assert result.uri.endswith("/receipts/final.json")
    assert FakeRunner.phases == list(PHASES)

def test_driver_reuses_only_exact_verified_phase_receipts(tmp_path):
    first = interrupted_build(tmp_path, after="catalog")
    resumed = resume_build(tmp_path, first.receipts)
    assert resumed.reused == ("source-stage", "wikidata-view", "catalog")
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_driver.py
```

Expected: collection fails because `cluster.aws.corpus_builder.driver` is absent.

- [ ] **Step 3: Add a real production CLI path**

Replace the `build-production` branch that constructs
`UnsupportedProductionRenderer` with an exact production factory:

```python
inputs = load_verified_production_inputs(
    source_lock_path=args.source_lock,
    source_root=args.source_root,
    derived_root=args.derived_root,
    expected_generator_commit=args.generator_commit,
)
receipt = build_parallel_corpus(
    inputs.catalog,
    inputs.renderer,
    config,
    args.output,
    workers=args.workers,
    sidecar_paths=dict(inputs.sidecars),
)
```

Add required arguments:

```text
--source-lock
--source-root
--derived-root
--generator-commit
--output
--update-tokens
--shards
--workers
```

Remove `--source`; production identity comes from verified source and derived
receipts, never a free-form renderer name.

- [ ] **Step 4: Implement the phase state machine**

For each phase:

1. Look up the expected content-addressed phase-receipt key.
2. Reuse it only after exact-version S3 and local dependency verification.
3. Run the phase into a private local temporary.
4. Verify outputs locally.
5. Upload exact immutable objects.
6. Publish the canonical phase receipt last.

Call the completed Wikidata APIs, production catalog factory,
`build_parallel_corpus`, and `verify_parallel_corpus` directly in Python.

- [ ] **Step 5: Enforce final receipt order**

The `s3-publish` phase uploads all corpus objects before its receipt.
`cleanroom-verify` downloads all exact versions into a newly created empty root,
calls `verify_parallel_corpus`, and publishes the clean-room verification
receipt. Publish `receipts/final.json` only after both receipts agree on
`build_id`, object versions, byte counts, and SHA-256 values.

- [ ] **Step 6: Run GREEN and producer regressions**

Run:

```bash
python -m pytest -q \
  tests/test_aws_corpus_builder_driver.py \
  tests/test_parallel_corpus.py \
  tests/test_reasoning_v2_wikidata_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_renderers.py
python -m py_compile \
  cluster/aws/corpus_builder/driver.py \
  scripts/build_parallel_corpus.py
git diff --check
```

Expected: all tests pass and static checks are silent.

- [ ] **Step 7: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/driver.py \
  corpusgen/parallel/adapters.py \
  scripts/build_parallel_corpus.py \
  tests/test_aws_corpus_builder_driver.py \
  tests/test_parallel_corpus.py
git commit -m "feat: orchestrate resumable production corpus builds"
```

Execute only with explicit commit authorization.

---

### Task 5: Private No-Instance CloudFormation Foundation

**Files:**
- Create: `infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml`
- Create: `infra/aws/cfn-guard/memorysplit-corpus-builder.guard`
- Create: `infra/aws/requirements-dev.txt`
- Create: `tests/test_aws_corpus_builder_foundation.py`

**Interfaces:**
- Consumes: pinned Amazon Linux 2023 AMI ID, one selected Availability Zone,
  budget email, VPC CIDRs, and exact builder profile values.
- Produces CloudFormation outputs:

```text
ArtifactBucketName
BuilderInstanceProfileArn
BuilderRoleArn
ControllerRoleArn
DataKeyArn
LaunchTemplateId
LaunchTemplateVersion
PrivateSubnetId
SecurityGroupId
SsmDocumentName
VpcId
```

- [ ] **Step 1: Write foundation RED tests**

```python
def test_foundation_has_launch_template_but_no_paid_instance_resource(template):
    assert "AWS::EC2::Instance" not in resource_types(template)
    launch = template["Resources"]["BuilderLaunchTemplate"]
    assert launch["Properties"]["LaunchTemplateData"]["InstanceType"] == "i4i.16xlarge"

def test_launch_template_is_private_imdsv2_encrypted_and_terminating(template):
    data = launch_data(template)
    assert data["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert data["MetadataOptions"]["HttpTokens"] == "required"
    assert data["NetworkInterfaces"][0]["AssociatePublicIpAddress"] is False
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_foundation.py
```

Expected: fails because the template and guard file are absent.

- [ ] **Step 3: Adapt the reviewed P5 foundation pattern**

Copy structural patterns from
`infra/aws/cloudformation/memorysplit-p5-foundation.yaml`, but create separate
builder-only resources:

- private VPC and subnet;
- zero-ingress builder security group;
- S3, EC2, EC2 Messages, KMS, Logs, SSM, SSM Messages, and STS endpoints;
- retained versioned SSE-KMS bucket with the exact approved bucket name;
- builder role and instance profile;
- separate controller role;
- SSM command document;
- no-instance `i4i.16xlarge` launch template; and
- a `$200` monthly budget with 50%, 80%, and forecasted 100% alerts.

Do not include ECR, evaluator, signer, GPU, training, Capacity Reservation, Spot,
Auto Scaling, NAT, Internet Gateway, SSH key, public route, or public IP
resources.

- [ ] **Step 4: Scope IAM exactly**

Builder role:

```text
s3:GetBucketLocation
s3:ListBucket
s3:ListBucketVersions
s3:GetObject
s3:GetObjectAttributes
s3:GetObjectVersion
s3:PutObject
s3:AbortMultipartUpload
s3:ListMultipartUploadParts
kms:Decrypt
kms:DescribeKey
kms:Encrypt
kms:GenerateDataKey
logs:CreateLogStream
logs:PutLogEvents
```

Restrict S3 to `v2/builds/*`, package objects, and source objects. Attach only
`AmazonSSMManagedInstanceCore`.

Controller role may describe state, run exactly the approved launch template
with `i4i.16xlarge`, pass only the builder role, send the exact SSM document,
modify shutdown behavior, and terminate only resources tagged
`MemorySplitCorpusBuilder=true`.

- [ ] **Step 5: Encode launch-template safety**

Use:

```yaml
InstanceInitiatedShutdownBehavior: terminate
InstanceType: i4i.16xlarge
MetadataOptions:
  HttpEndpoint: enabled
  HttpProtocolIpv6: disabled
  HttpPutResponseHopLimit: 1
  HttpTokens: required
  InstanceMetadataTags: disabled
```

Set a 200 GiB encrypted `gp3` root volume with `DeleteOnTermination: true`,
disable termination protection, enable detailed monitoring, attach no public
IP, and tag instance and volume with `MemorySplitCorpusBuilder=true`.

- [ ] **Step 6: Add cfn-guard invariants**

Require exact resource counts and types; no paid instance resources; no public
network; exact `i4i.16xlarge`; termination shutdown behavior; IMDSv2; encrypted
root; retained versioned KMS bucket; exact IAM policy containers; no wildcard
actions; no broad S3 resources; and no static secrets.

- [ ] **Step 7: Run GREEN and IaC validation**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_foundation.py
python -m pip install -r infra/aws/requirements-dev.txt
cfn-lint infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml
cfn-guard validate \
  --rules infra/aws/cfn-guard/memorysplit-corpus-builder.guard \
  --data infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml
git diff --check
```

Expected: tests and both IaC validators pass.

- [ ] **Step 8: Prepare the proposed commit**

```bash
git add \
  infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml \
  infra/aws/cfn-guard/memorysplit-corpus-builder.guard \
  infra/aws/requirements-dev.txt \
  tests/test_aws_corpus_builder_foundation.py
git commit -m "feat: add private AWS corpus builder foundation"
```

Execute only with explicit commit authorization.

---

### Task 6: NVMe Bootstrap and Hard Termination Watchdog

**Files:**
- Create: `cluster/aws/corpus_builder/bootstrap.py`
- Create: `cluster/aws/corpus_builder/bootstrap.sh`
- Create: `tests/test_aws_corpus_builder_bootstrap.py`
- Modify: `infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml`

**Interfaces:**
- Consumes: package S3 URI/version/SHA-256, source manifest URI/version/SHA-256,
  build ID, KMS key ARN, profile SHA-256, worker count, and launch-intent
  SHA-256.
- Produces:

```python
@dataclass(frozen=True)
class BootstrapConfig:
    build_id: str
    package: S3ObjectVersion
    source_manifest: S3ObjectVersion
    kms_key_arn: str
    profile_sha256: str
    launch_intent_sha256: str
    workers: int

def render_bootstrap(config: BootstrapConfig) -> str: ...
```

- [ ] **Step 1: Write bootstrap RED tests**

```python
def test_bootstrap_mounts_exactly_four_nvmes_as_raid0_xfs_and_starts_watchdog():
    text = render_bootstrap(fixture_config())
    assert "mdadm --create /dev/md/memorysplit" in text
    assert "--raid-devices=4" in text
    assert "mkfs.xfs" in text
    assert "OnActiveSec=23h30m" in text
    assert "shutdown -h now" in text
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
```

Expected: collection fails because `cluster.aws.corpus_builder.bootstrap` is absent.

- [ ] **Step 3: Implement strict NVMe discovery**

Identify only local NVMe instance-store devices whose model is
`Amazon EC2 NVMe Instance Storage`. Require exactly four devices. Reject root
EBS devices, duplicate serials, mounted devices, existing filesystems, or any
device count other than four.

- [ ] **Step 4: Build and mount scratch**

Create `/dev/md/memorysplit` as RAID0 with 512 KiB chunk size, format XFS, mount
at `/mnt/memorysplit-builder` with `noatime,nodiratime`, create owner-only
`work`, `output`, and `cleanroom` directories, and record device serials and
filesystem UUID in the bootstrap receipt.

- [ ] **Step 5: Install the watchdog before the build**

Render:

```ini
[Unit]
Description=Terminate MemorySplit corpus builder before 24-hour ceiling

[Timer]
OnActiveSec=23h30m
Persistent=true

[Install]
WantedBy=timers.target
```

The paired service uploads a timeout marker to S3 and runs
`/usr/sbin/shutdown -h now`. Enable and start the timer before package download.

- [ ] **Step 6: Download and verify exact inputs**

Use the Task 3 exact-version functions for package and source manifest. Reject
latest-version reads. Extract the package into `/opt/memorysplit` only after
archive and manifest verification. Run the driver with an allowlisted
environment and upload logs after every phase.

- [ ] **Step 7: Bind bootstrap to the launch template**

The launch template user data invokes only the rendered bootstrap script and
contains no credentials, tokens, mutable Git references, or unversioned S3
URIs.

- [ ] **Step 8: Run GREEN**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_bootstrap.py
python -m py_compile cluster/aws/corpus_builder/bootstrap.py
bash -n cluster/aws/corpus_builder/bootstrap.sh
git diff --check
```

Expected: tests pass and static checks are silent.

- [ ] **Step 9: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/bootstrap.py \
  cluster/aws/corpus_builder/bootstrap.sh \
  infra/aws/cloudformation/memorysplit-corpus-builder-foundation.yaml \
  tests/test_aws_corpus_builder_bootstrap.py
git commit -m "feat: bootstrap bounded NVMe corpus builds"
```

Execute only with explicit commit authorization.

---

### Task 7: Read-Only Preflight and Canonical Launch Intent

**Files:**
- Create: `cluster/aws/corpus_builder/preflight.py`
- Create: `scripts/aws_corpus_builder_preflight.py`
- Create: `tests/test_aws_corpus_builder_preflight.py`

**Interfaces:**
- Consumes: profile, package record, source-manifest record, stack outputs,
  EC2/IAM/S3/KMS/Pricing clients, and current UTC time.
- Produces:

```python
class PreflightError(ValueError):
    pass

@dataclass(frozen=True)
class PreflightRequest:
    profile_path: Path
    package: S3ObjectVersion
    source_manifest: S3ObjectVersion
    software_gate_receipt: Path
    stack_outputs: Mapping[str, str]
    ami_id: str
    ami_owner_id: str

@dataclass(frozen=True)
class AwsClients:
    sts: object
    ec2: object
    iam: object
    s3: object
    kms: object
    pricing: object

@dataclass(frozen=True)
class PreflightResult:
    intent: LaunchIntent
    intent_sha256: str
    checks: tuple[str, ...]

def run_preflight(
    request: PreflightRequest,
    *,
    aws: AwsClients,
    now: datetime,
) -> PreflightResult: ...
```

- [ ] **Step 1: Write preflight RED tests**

```python
def test_preflight_emits_intent_only_after_every_read_only_gate_and_ec2_dry_run():
    result = run_preflight(valid_request(), aws=FakeAws(), now=NOW)
    assert result.checks == EXPECTED_CHECKS
    assert FakeAws.run_instances_calls[-1]["DryRun"] is True

@pytest.mark.parametrize(
    "failure",
    (
        "dirty-package",
        "source-drift",
        "wrong-ami-owner",
        "instance-unavailable",
        "hourly-price-over-cap",
        "bucket-unversioned",
        "wrong-kms-key",
        "wrong-launch-template",
        "dry-run-denied",
    ),
)
def test_preflight_fails_closed_without_launch_intent(failure):
    with pytest.raises(PreflightError):
        run_preflight(request_for(failure), aws=FakeAws(), now=NOW)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
```

Expected: collection fails because `cluster.aws.corpus_builder.preflight` is absent.

- [ ] **Step 3: Implement exact checks in this order**

1. Profile canonical bytes and SHA-256.
2. Package and source-manifest exact S3 versions.
3. Completed production software-gate receipt.
4. Account `056956104102` and region `us-east-1`.
5. AMI owner, architecture, state, root device, and creation date.
6. `i4i.16xlarge` regional and subnet availability.
7. Launch-template ID and exact numeric version.
8. No public IP and exact zero-ingress security group.
9. Instance profile and builder role identity.
10. Bucket versioning, Block Public Access, ownership, and KMS encryption.
11. Current Linux On-Demand price at or below `$5.491/hour`.
12. Launch intent maximum at or below `$131.78`.
13. EC2 `RunInstances` with `DryRun=True`.

- [ ] **Step 4: Emit a short-lived intent**

Set `not_after` to 30 minutes after preflight. Canonicalize the intent, write it
with mode `0600`, and print its SHA-256. The preflight command performs no AWS
mutation other than the non-mutating EC2 dry-run.

- [ ] **Step 5: Run GREEN**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_preflight.py
python -m py_compile \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py
git diff --check
```

Expected: tests pass and static checks are silent.

- [ ] **Step 6: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/preflight.py \
  scripts/aws_corpus_builder_preflight.py \
  tests/test_aws_corpus_builder_preflight.py
git commit -m "feat: preflight bounded AWS corpus builds"
```

Execute only with explicit commit authorization.

---

### Task 8: Explicit-Approval Launch and Immediate Lifecycle Control

**Files:**
- Create: `cluster/aws/corpus_builder/launch.py`
- Create: `scripts/aws_corpus_builder_launch.py`
- Create: `tests/test_aws_corpus_builder_launch.py`

**Interfaces:**
- Consumes: canonical launch-intent bytes, exact intent SHA-256, explicit
  approval SHA-256, and EC2 client.
- Produces:

```python
class LaunchError(ValueError):
    pass

class Ec2Client(Protocol):
    def run_instances(self, **kwargs: object) -> Mapping[str, object]: ...
    def describe_instances(self, **kwargs: object) -> Mapping[str, object]: ...
    def terminate_instances(self, **kwargs: object) -> Mapping[str, object]: ...

@dataclass(frozen=True)
class BuilderLaunch:
    instance_id: str
    launch_intent_sha256: str
    launch_time: str
    terminate_at: str

def launch_approved_builder(
    intent_bytes: bytes,
    *,
    approved_intent_sha256: str,
    ec2: Ec2Client,
    now: datetime,
) -> BuilderLaunch: ...
```

- [ ] **Step 1: Write approval RED tests**

```python
def test_launch_requires_exact_unexpired_intent_hash_and_one_instance():
    launch = launch_approved_builder(
        INTENT_BYTES,
        approved_intent_sha256=sha256(INTENT_BYTES),
        ec2=FakeEc2(),
        now=NOW,
    )
    assert launch.instance_id == "i-builder"
    assert FakeEc2.last_request["MinCount"] == 1
    assert FakeEc2.last_request["MaxCount"] == 1

@pytest.mark.parametrize(
    "failure",
    ("wrong-hash", "expired", "price-drift", "template-drift", "second-instance"),
)
def test_launch_rejects_every_unapproved_change(failure):
    with pytest.raises(LaunchError):
        launch_failure(failure)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_launch.py
```

Expected: collection fails because `cluster.aws.corpus_builder.launch` is absent.

- [ ] **Step 3: Implement exact launch construction**

Require the approved intent hash, current time before `not_after`, exact profile
and package hashes, and a fresh read-only recheck of price and launch-template
version. Call `RunInstances` exactly once with:

```python
{
    "LaunchTemplate": {
        "LaunchTemplateId": intent.launch_template_id,
        "Version": intent.launch_template_version,
    },
    "MinCount": 1,
    "MaxCount": 1,
    "TagSpecifications": approved_tag_specifications(intent),
}
```

Do not accept instance-type, AMI, network, role, storage, or user-data
overrides.

- [ ] **Step 4: Verify post-launch safety before returning**

Describe the instance until it exists, then require exact instance type,
subnet, security group, profile, AMI, IMDSv2, shutdown behavior `terminate`,
tags, and no public IP. On any mismatch, terminate the tagged instance and
raise `LaunchError`.

- [ ] **Step 5: Add the CLI confirmation boundary**

The CLI requires:

```text
--intent
--approve-intent-sha256
--profile sbsandbox
--region us-east-1
```

It prints the exact instance ID, launch time, terminate-at time, and maximum
approved cost. It never accepts a generic `--yes` flag.

- [ ] **Step 6: Run GREEN**

Run:

```bash
python -m pytest -q tests/test_aws_corpus_builder_launch.py
python -m py_compile \
  cluster/aws/corpus_builder/launch.py \
  scripts/aws_corpus_builder_launch.py
git diff --check
```

Expected: tests pass and static checks are silent.

- [ ] **Step 7: Prepare the proposed commit**

```bash
git add \
  cluster/aws/corpus_builder/launch.py \
  scripts/aws_corpus_builder_launch.py \
  tests/test_aws_corpus_builder_launch.py
git commit -m "feat: launch explicitly approved corpus builders"
```

Execute only with explicit commit authorization.

---

### Task 9: Runbook, Full Verification, and Exact Operational Handoff

**Files:**
- Create: `docs/AWS-CORPUS-BUILDER-RUNBOOK.md`
- Modify: `scripts/package_aws_corpus_builder.py`
- Modify: `tests/test_package_aws_corpus_builder.py`

**Interfaces:**
- Consumes: all Task 1–8 commands and artifacts.
- Produces: one deterministic package, one preflight command, one separately
  approved launch command, monitoring commands, clean-room verification
  commands, and teardown evidence.

- [ ] **Step 1: Write runbook/package RED tests**

```python
def test_package_contains_builder_runtime_iac_tests_and_runbook():
    package = build_corpus_builder_package(CLEAN_REPO, OUTPUT)
    assert REQUIRED_BUILDER_MEMBERS <= package.members

def test_runbook_never_launches_without_exact_intent_approval():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "--approve-intent-sha256" in text
    assert "--yes" not in text
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_package_aws_corpus_builder.py \
  tests/test_aws_corpus_builder_launch.py
```

Expected: fails because the final required members and runbook are absent.

- [ ] **Step 3: Write the exact preflight sequence**

The runbook must require:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_renderers.py \
  tests/test_parallel_corpus.py \
  tests/test_aws_corpus_builder_contracts.py \
  tests/test_package_aws_corpus_builder.py \
  tests/test_aws_corpus_builder_s3.py \
  tests/test_aws_corpus_builder_driver.py \
  tests/test_aws_corpus_builder_bootstrap.py \
  tests/test_aws_corpus_builder_foundation.py \
  tests/test_aws_corpus_builder_preflight.py \
  tests/test_aws_corpus_builder_launch.py
```

Then build the package twice, compare bytes, validate CloudFormation with
cfn-lint and cfn-guard, upload exact package/source objects, and run the
read-only preflight.

- [ ] **Step 4: Document the separate approval point**

The operator must copy the printed intent SHA-256 into:

```bash
python scripts/aws_corpus_builder_launch.py \
  --intent artifacts/aws-corpus-builder/launch-intent.json \
  --approve-intent-sha256 "$EXACT_INTENT_SHA256" \
  --profile sbsandbox \
  --region us-east-1
```

The runbook states that this command creates a paid On-Demand instance and may
run only after the user approves the exact intent and `$131.78` ceiling.

- [ ] **Step 5: Document monitoring and terminal evidence**

Use SSM and S3 phase receipts. Success requires the final receipt and
clean-room receipt. On failure or timeout, verify instance termination and
retain phase receipts. Never label a phase receipt or local NVMe directory as a
built corpus.

- [ ] **Step 6: Run the full non-slow suite and package verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-final-a
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-final-b
cmp \
  /tmp/memorysplit-corpus-final-a/memorysplit-corpus-builder.tar.gz \
  /tmp/memorysplit-corpus-final-b/memorysplit-corpus-builder.tar.gz
git diff --check
git status --short
```

Expected: all tests pass, package bytes match, diff-check is silent, and only
the reviewed task files appear in status.

- [ ] **Step 7: Prepare the proposed commit**

```bash
git add \
  docs/AWS-CORPUS-BUILDER-RUNBOOK.md \
  scripts/package_aws_corpus_builder.py \
  tests/test_package_aws_corpus_builder.py
git commit -m "docs: add bounded AWS corpus build runbook"
```

Execute only with explicit commit authorization.

- [ ] **Step 8: Stop before AWS mutation**

Present:

- exact CloudFormation change-set summary;
- exact retained and hourly infrastructure cost estimate;
- exact package and source object versions and SHA-256 values;
- exact launch intent and SHA-256;
- current `i4i.16xlarge` price;
- `$131.78` compute ceiling; and
- all test and IaC validation evidence.

Do not apply the stack or launch the instance until the user separately
approves those exact resources and costs.
