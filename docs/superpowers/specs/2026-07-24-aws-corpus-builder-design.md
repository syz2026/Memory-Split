# AWS Claim-Bearing Corpus Builder Design

## Decision

Materialize the frozen MemorySplit V2/V3 claim-bearing corpus on one dedicated
Amazon EC2 `i4i.16xlarge` in `us-east-1`. The instance is a bounded build worker,
not a training host. It may launch only after the production corpus pipeline
passes the local authority and fixture gates in this design.

The selected instance provides 64 vCPUs, 512 GiB RAM, four 3.75 TB local NVMe
SSDs (15 TB total), 37.5 Gbps networking, and 20 Gbps baseline EBS bandwidth.
The current Linux On-Demand rate is $5.491/hour. A hard 24-hour lifetime limits
instance compute cost to $131.78, excluding EBS, S3, KMS, and request charges.

## Goal and non-goals

The successful output is a reusable, independently verified
`memorysplit-parallel-corpus-v2` artifact containing:

- the canonical final receipt;
- every referenced packed token shard;
- Dense and Split90 target-weight sidecars;
- route, mask, semantic-closure, graph-memory, and source-provenance records;
- the verified Wikidata derived-view receipt and commitments;
- immutable S3 object version IDs, byte counts, and SHA-256 values; and
- a clean-room re-download verification receipt.

Legacy multihop, 10M-token, 20M-dose, smoke, fixture, and collaborator-kit
corpora do not satisfy this design. This build does not launch training, alter
frozen scientific files, or convert pilot outputs into claim-bearing evidence.

## Pre-launch software gate

No paid builder may start until one clean revision provides the complete
production path:

1. Descriptor-pinned Wikidata archive authority and closed receipt parsing.
2. Canonical Wikidata streams, indexes, transactional publication, and lookup.
3. Production Wikidata catalog adapter with the distinct-edge capacity gate.
4. Indexed Wikidata renderer integration.
5. Production renderers/adapters for every other frozen corpus lane.
6. End-to-end tiny-archive fixture coverage through catalog, rendering,
   packing, Dense/Split90 sidecars, receipt publication, and verification.
7. A deterministic builder package whose two independent builds have identical
   SHA-256 values.

The exact focused suites from the Wikidata derived-view plan must pass, followed
by the complete corpus-producer suite and package verification. The current
`UnsupportedProductionRenderer` path is a launch blocker, not a warning.

## AWS resources

The builder uses:

- one On-Demand `i4i.16xlarge` in `us-east-1`;
- a 200 GiB encrypted `gp3` root volume;
- all four local NVMe devices as an ephemeral RAID0 XFS scratch filesystem;
- IMDSv2-only metadata;
- no inbound security-group rules;
- AWS Systems Manager Session Manager for operator access;
- an instance profile restricted to the corpus bucket, logs, KMS key, and
  self-observation APIs; and
- the versioned, Block-Public-Access-enabled bucket
  `memorysplit-corpus-056956104102-us-east-1`.

The bucket uses a dedicated KMS key. The instance writes only beneath
`v2/builds/{build_id}/`, where `build_id` is the existing canonical parallel
build identifier derived from frozen inputs and build configuration.

The instance is launched with instance-initiated shutdown behavior set to
`terminate`. A systemd watchdog begins shutdown at 23 hours 30 minutes even if
the build driver hangs. Normal success terminates the instance immediately
after clean-room verification.

## Data flow

1. Resolve and verify the selected clean Git commit and deterministic package.
2. Verify every frozen source-lock record before staging source archives.
3. Download exact source objects to NVMe and rehash them before use.
4. Build and verify the content-addressed Wikidata derived view.
5. Build the canonical production input catalog.
6. Render disjoint ordinal partitions with bounded multiprocessing.
7. Pack canonical token shards and both arm sidecar sets.
8. Verify the complete local corpus and publish immutable objects to S3.
9. Publish the final receipt only after every referenced object has an exact
   version ID, byte count, SHA-256, and successful HEAD verification.
10. In a fresh process with empty local caches, re-download the receipt and
    every object, verify the complete corpus, and publish the clean-room
    verification receipt.
11. Terminate the instance.

Parallelism starts at 32 workers. A bounded fixture benchmark may raise it to 64
only if throughput improves without changing output bytes or exceeding memory
and file-descriptor limits.

## Recovery and failure behavior

Each completed phase publishes a content-addressed phase receipt to S3. A retry
may reuse a phase only after complete receipt, object-version, inventory, and
checksum verification. Partial prefixes are never repaired or treated as
authority.

The final corpus does not exist unless both the final receipt and clean-room
verification receipt are present and mutually consistent. Instance loss,
watchdog termination, source drift, missing object versions, hash mismatch,
incomplete sidecars, or failed re-download leaves the build failed and
non-claim-bearing.

CloudWatch and S3 receive phase logs and failure summaries. Logs are
operational evidence only and cannot replace receipts.

## Cost and safety boundary

The maximum EC2 compute charge per attempt is $131.78 at the observed
$5.491/hour rate. Root EBS, S3 storage, KMS, and API requests are additional.
No Capacity Block or GPU instance is used.

Before launch, the operator must see and approve:

- the exact AMI, subnet, Availability Zone, security group, and instance
  profile;
- the current hourly price and $131.78 compute ceiling;
- the deterministic package SHA-256;
- the source-lock and expected build identifiers; and
- the auto-termination configuration.

## Acceptance criteria

The corpus is built only when:

- all pre-launch software gates pass on one clean commit;
- the builder package and source locks match their approved hashes;
- every required lane is production-rendered;
- every shard and Dense/Split90 sidecar is receipt-bound;
- S3 versioning and encryption are verified;
- no object is missing, mutable, or unversioned;
- local and clean-room verification produce the expected build ID; and
- the final receipt URI, version ID, byte count, and SHA-256 are recorded in
  the scientific artifact binding.
