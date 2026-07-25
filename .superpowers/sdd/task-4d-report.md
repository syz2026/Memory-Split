# Task 4D report: durable cohort evaluation-evidence collection and receipt

Brief: `.superpowers/sdd/task-4d-brief.md`
(SHA-256 `36e8c719148415cc81e6107b260fe27c4f7177e34c0b6d5cedd8e3aed1ba7440`),
executed from clean head `ac337b9` (Task 4C head `181f3ef` plus its progress
commit) on `integration/provider-aware-lifecycle`.

Task 4D implements the collection half of plan §7 for cohort evidence: one
`collect-cohort` operation downloads, independently stream-rehashes, and
receipts exactly the 1,012 durable cohort evaluation objects, replays the
complete Task 4C report recomputation as the only outcome authority, and
publishes one canonical no-replace cohort collection receipt. Selected
`evaluate` stays blocked for Task 4E; selected `cleanup` stays
unconditionally blocked even with ten per-seed collection states plus a
valid cohort collection state.

## Exact object enumeration

One cohort collection binds exactly `COHORT_COLLECTION_OBJECT_COUNT = 1012`
objects, in the frozen row order proven by the parser and pinned by tests:

- rows 0-999: the 100 outputs in frozen `EXPECTED_STUDY_SLOTS_V3` order
  (seed 0-9 ascending; `dense` then `split90`; steps 1358, 3396, 6791,
  10187, 13582), each slot contributing its ten members in the sorted
  `EVALUATION_OUTPUT_MEMBERS` order, so slot *i* occupies rows
  `10*i .. 10*i+9`;
- rows 1000-1009: the ten Task 3F per-seed collection receipts, seeds 0-9
  ascending;
- row 1010: the published StudyLockV3 object;
- row 1011: the published cohort report object.

Rows 1000-1011 must equal the receipt's top-level
`seed_collections`/`study_lock`/`cohort_report` references pairwise (the
Task 3F rows-15/16 rule generalized); every row's URI must sit at its
shared key helper under the receipt's own safe root. Publication is
per-file and slot-scoped: cohort-uniform members occupy 100 distinct keys,
so no receipt row can alias another slot's object and each row carries its
own exact S3 version ID.

## Canonical keys (`msctl/aws_contracts.py`)

- `EVALUATION_OUTPUT_MEMBERS`: the closed sorted ten-name tuple,
  mirror-pinned (test-side imports only) against
  `evals.confirmatory.aggregate._OUTPUT_NAMES` and
  `tuple(sorted((*runner._V3_OUTPUT_ARTIFACTS, "output.json")))`.
- `study_lock_object_key` → `evaluations/study-lock/sha256/{digest}.json`
- `cohort_report_object_key` →
  `evaluations/cohort-report/sha256/{digest}.json`
- `evaluation_output_member_key` →
  `evaluations/outputs/seed-{seed}/{arm}/step-{step}/{stem}/sha256/{digest}.{ext}`
  with exact-int seed/arm/step validation matching the existing helpers,
  the closed member set, and `{stem}.{ext}` split at the member's final dot.
- `cohort_collection_receipt_key` →
  `receipts/cohort-collections/sha256/{digest}.json` (cohort-scoped; no
  seed segment).

## Contracts (`msctl/aws_cohort_collect.py`)

- `COHORT_COLLECTION_RECEIPT_TYPE = "memorysplit-aws-cohort-collection-v3"`,
  `COHORT_COLLECTION_OBJECT_COUNT = 1012`,
  `COHORT_COLLECTION_RECEIPT_FIELDS` (closed 37-field schema-3 set; the
  frozen Task 3F set minus `run_receipt`/`checkpoint_receipt` plus
  `study_lock`, `cohort_report`, `seed_collections`,
  `sealed_evaluation_release_sha256`, `preregistration_sha256` — equality
  with that derivation is mirror-pinned by a test),
  `CohortCollectionError`, and the fail-closed `CohortEvidenceRef` /
  `CohortCollectedObject` dataclasses.
- `parse_cohort_evidence_index(payload, *, s3_root)`: strict schema-1
  parser for the wholly untrusted operator index (Task 4E's output
  contract). Duplicate-key and non-finite rejecting; closed field sets at
  every level; exactly 100 outputs in frozen slot order with exactly ten
  member rows each in sorted member-name order; exactly ten seed rows,
  ascending, with no cross-seed receipt aliasing; every URI at its
  canonical key under the pinned safe S3 root; `output_id` equal to the
  canonical derivation
  `snapshot-evaluation-memorysplit-v3-{profile_id}-s{seed:02d}-{arm}-step{step:05d}`
  with one consistent profile from the closed eligible set; every byte
  count a positive exact int bounded by `_MAX_EVIDENCE_OBJECT_BYTES`
  (mirror-pinned to `aggregate._MAX_OUTPUT_FILE_BYTES`, 2 GiB) so oversize
  fails before any download.
- `parse_cohort_collection_receipt_bytes(...)`: mirrors the Task 3F parser
  exactly — canonical Task 3C bytes with one trailing newline, self-hash
  against `receipt_sha256`, non-null version, closed 37-field set, seed
  pinned to the terminal 9, identity block (schema 3 / receipt type /
  `COHORT_ID` / `complete: true`), request-ID/timestamp shapes,
  profile↔provider consistency, nineteen SHA-256 field checks,
  Git/instance/boot shape validation, safe root derived from the receipt
  URI suffix, exactly 1,012 rows in canonical order at canonical keys,
  rows 1000-1011 pairwise-equal to the top-level references, and the
  optional exact lifecycle comparison through the imported frozen Task 3D
  `_RECEIPT_BINDING_FIELDS` mapping (not duplicated). The receipt carries
  no outcome, statistic, or conclusion field (tested).
- Module import is proven clean by a subprocess: importing
  `msctl.aws_cohort_collect` leaves `torch`, `train.trainer`, and
  `evals.confirmatory` unimported.

## Selected collection controller (`msctl/aws_p5.py`)

`AwsP5Backend.collect_cohort_evidence(*, release, manifest, cohort_report,
evidence_index, out, apply)` is the selected dispatch target for
`collect-cohort`. Like Task 3F collection it requires no paid-instance
approval — it creates no capacity, sends no remote command, and only
publishes one content-addressed receipt (documented in the method and the
dispatch branch). Apply ordering, with every gate preceding any local or
remote mutation:

1. `_validate_manifest` re-admits the fixed authority;
   `_selected_manifest_lifecycle` plus a non-null `lifecycle_binding`
   required (`OPERATION_UNSUPPORTED` otherwise); `manifest.seed == 9`
   required (`COHORT_COLLECT_EVIDENCE_INVALID` with zero AWS calls
   otherwise, seeds 0-8 tested); exact operator cohort-report triple at
   its canonical key under the pinned root; strict index parse plus
   index↔anchor-triple equality;
2. nonexisting, nonsymlink `--out` (before the idempotence gate, exactly
   the reviewed 3F semantics — idempotent replays need a fresh `--out`);
3. under the state lock, singleton cohort state: identical anchors (report
   triple, study-lock ref, seed-9 manifest hash) → exact-HEAD verify the
   recorded receipt and return idempotently; different anchors →
   `COHORT_COLLECT_CONFLICT` with zero AWS calls; unverifiable recorded
   receipt → `COHORT_COLLECT_INCOMPLETE`;
4. exact-version checksum-mode GET of the report through the bounded
   download runner; returned checksum/length/version equality; independent
   stream rehash; strict function-local
   `evals.confirmatory.aggregate.CohortReport.from_dict` with
   canonical-bytes equality; report↔index collection-receipt identity;
   `report.study_lock_sha256` == index lock SHA; sealed-release /
   preregistration / provider-selection consistency with the controller
   authority;
5. exact-version GET of the lock; rehash equals the report commitment;
   `StudyLockV3.from_dict` with canonical-bytes equality;
   `lock.provider_selection` equality with the cohort selection authority
   (selection SHA/version, profile ID/hash, hardware amendment, runtime
   lock, five qualification hashes, provider, cohort);
   `lock.lifecycle_binding(9)` equal to the controller binding with boot
   rebound only (a reboot between training and collection passes — proven
   in the main flow, whose controller boot differs from the lock's
   training boot; instance drift fails); lock seed lifecycles'
   collection-receipt identities equal to index (and hence report) rows;
   three-way sealed-release bind manifest == lock == report; dataset
   triple from the lock's uniform slots cross-checked against the
   manifest;
6. exact-version GETs of the ten per-seed collection receipts with
   independent rehash (their full Task 3F parse happens inside the
   replay's `plan_snapshot_evaluations`);
7. per slot in frozen order: exact-version GET of `output.json`; rehash;
   SHA must equal the report input commitment; strict parse; its nine
   `artifacts` rows must equal the index member rows (path/sha256/bytes);
   `output_id` and lock/release/selection/slot identity consistency; then
   the nine artifacts fetched bytes-opaquely by exact version through the
   bounded runner and stream-hashed at the descriptor-pinned staged file
   (drift → `COHORT_COLLECT_OBJECT_MISMATCH`); all downloads land in a
   private 0700 no-follow staging tree shaped for the replay;
8. outcome replay — the only outcome authority: function-local import of
   `evals.confirmatory.aggregate`; 100 `SnapshotOutputReference` values
   (staged output dirs + report commitments) and ten
   `CollectionReceiptEvidence` values (downloaded payloads + identities)
   drive `validate_cohort_report(report_value, outputs=…,
   collection_receipts=…, expected_study_lock_sha256=…)` — the complete
   Task 4C recomputation. Divergence →
   `COHORT_COLLECT_REPLAY_FAILED`, no PUT, no state;
9. the canonical receipt is built from the authenticated controller
   binding (through the frozen `_RECEIPT_BINDING_FIELDS` mapping), the
   seed-9 manifest provenance, the verified anchors, the ten seed rows,
   and the 1,012 object rows; self-parsed with
   `expected_binding=self.lifecycle_binding` before any publication; a
   copy staged at its key-relative path;
10. `--if-none-match "*"` PUT, checksum-bound and metadata-bound
    (`receipt-sha256`, `receipt-type`, `request-id`), then exact HEAD
    (checksum, content length, metadata, version, PUT-version equality
    when the PUT succeeded); a lost or conflicting no-replace PUT
    recovers only through the exact HEAD; drift →
    `COHORT_COLLECT_CONFLICT`;
11. atomic singleton state write, then no-replace rename of the staging
    tree (`FileExistsError` → `COHORT_COLLECT_DESTINATION_EXISTS`);
    failed runs remove the staging tree, never the destination;
12. returns `{provider, seed, run_manifest_sha256, study_lock,
    cohort_report, cohort_collection_receipt, objects_collected,
    bytes_collected, out, collected, idempotent}`.

Dry run performs zero AWS calls and renders only the first exact
cohort-report GET (`get-object --version-id … --checksum-mode ENABLED`),
returning `commands: [first_get], collected: 0, idempotent: false`.

Every body GET (all 1,012) goes through
`SubprocessAwsDownloadRunner.run_json` with
`timeout_seconds = download_timeout_seconds(index-declared bytes)`
(300-second floor, 2x expected size at 8 MiB/s, asserted per call in the
full-flow test); the fixed 60-second `AwsJsonRunner` serves only the PUT
and HEAD calls.

Destination layout (staging tree renamed to `--out`):

```text
{out}/cohort-report-{report_sha256}/cohort-report.json   (0444, 0700 dir)
{out}/study-lock-{lock_sha256}/study-lock.json           (0444, 0700 dir)
{out}/outputs/{output_id}/{10 members}                   (0700/0600) x 100
{out}/seed-collections/seed-{seed}/sha256/{sha256}.json  x 10
{out}/receipts/cohort-collections/sha256/{receipt_sha256}.json
```

The outputs deliberately deviate from 3F key-relative staging because the
aggregate replay requires owned 0700 directories containing exactly the
ten members as owned singly-linked 0600 regular files; the replay binds
identity by descriptor and `output.json` binds `output_id`.

## Cohort state (`msctl/state.py`)

`StateStore.locked()` additionally pins a `cohorts` directory. Singleton
`read_cohort_collection()` / `write_cohort_collection(value)` use the
fixed file name `{COHORT_ID}.json`, making a second conflicting cohort
receipt locally unrepresentable. Closed schema-2 record exactly as
specified (operation `collect-cohort`, seed pinned to 9,
`objects_collected` pinned to 1012, `status: Published`, all
`LIFECYCLE_BINDING_FIELDS` reconstructed through
`_reconstruct_lifecycle_binding`, the three references at their canonical
keys); rewrites may change only `updated_at`; conflicting rewrites fail
closed with `STATE_CORRUPT` — the exact 3F `write_collection` pattern.

## CLI (`msctl/cli.py`)

New top-level leaf `collect-cohort` (`--release --manifest
--cohort-report-uri --cohort-report-sha256 --cohort-report-version-id
--evidence-index --out`, the fixed selected authority argument group,
`--apply`; no `--source` exists). `"collect-cohort"` joined
`_SELECTED_LIFECYCLE_COMMANDS`; `_SELECTED_COHORT_COLLECT_ARGUMENTS`
mirrors `_SELECTED_COLLECT_ARGUMENTS`. The command is selected-only: an
absent or partial authority group fails `CLI_USAGE` with the exact missing
list, and the legacy AWS route and local route reject the command entirely
before any backend is constructed. Backend dispatch adds a
`collect-cohort` branch parallel to `collect` (complete-argument-set check
with exact missing list, `_load_bound_inputs`, route to
`collect_cohort_evidence`). Like selected `collect`, no approval argument
exists — collection creates no capacity and publishes one
content-addressed receipt (documented at both routes). Legacy/local
`collect`, `submit`, and every other existing route stay byte-identical
(existing suites unchanged and green).

## Blockers preserved

- Selected v3 `evaluate` stays `OPERATION_UNSUPPORTED`: "AWS v3 evaluation
  awaits Task 4E selected-evaluation enablement".
- Selected v3 `cleanup` stays unconditionally `OPERATION_UNSUPPORTED`:
  "AWS v3 cleanup requires all ten per-seed collection receipts, the
  cohort evaluation-evidence collection receipt, and cleanup enablement
  itself, which follows selected evaluation (Task 4E)". A dedicated test
  seeds ten per-seed collection states plus a valid cohort collection
  state and proves `evaluate`/`cleanup plan`/`cleanup apply` all stay
  blocked with zero AWS calls and no `terminate-instances`, asserting the
  updated messages.
- Selected on-instance execution remains blocked pending the later §6
  qualification task; production selected approval-resource closure
  remains the later paid-launch qualification work.
- Real production cohort evidence does not exist yet (evaluate is
  disabled); the collector is proven over payload-real fixtures that admit
  through the production parsers — the same status 4B/4C recorded.

## Strict TDD evidence

RED (all captured before implementation):

```text
python -c "import msctl.aws_cohort_collect"
  → ModuleNotFoundError: No module named 'msctl.aws_cohort_collect'
tests/test_aws_cohort_collect.py collection
  → ImportError: cannot import name 'EVALUATION_OUTPUT_MEMBERS'
    from 'msctl.aws_contracts'
tests/test_aws_contracts.py → 27 failed, 60 passed
  (AttributeError: module 'msctl.aws_contracts' has no attribute
   'study_lock_object_key', … for all four helpers and the member tuple)
tests/test_package_aws_p5_handoff.py::test_cohort_collection_closure_is_required
  → FAILED (member absent from REQUIRED_MEMBERS; synthetic packaging
    fixture entry added in the same RED step — the 4B pattern)
tests/test_msctl.py collect-cohort CLI tests → 2 FAILED
  (argparse: unknown command)
```

GREEN was reached in slices (contracts 87 passed → module parsers 90
passed → state 19 passed → controller full flow → complete file), with
four intermediate failures fixed during the controller slice:

- the idempotence conflict case had to present a coherent conflicting
  anchor-plus-index pair (the anchor↔index equality gate otherwise fires
  first);
- the `seed-receipt-identity-drift` gate case had to republish a dishonest
  report whose receipt row agrees with the drifted index so the drift is
  only caught against the lock after the second download;
- both import-boundary subprocesses were tightened to the strongest
  attainable claims (below).

## Verification

```text
group 1 (brief command: cohort collect, collect, contracts,
  msctl, paired state, seed transition, aggregate):     711 passed
  (tests/test_aws_cohort_collect.py alone: 160 passed)
confirmatory v2/v3 regression:                          209 passed
lifecycle/hardware authority (unsandboxed):             522 passed
  (495 inherited + 27 new key-helper/mirror tests)
DDP provenance:                                           8 passed
package handoff:                                        107 passed
  (106 inherited + 1 new cohort-collection closure)
full suite: 3204 passed, 2 failed*, 2 deselected, 1 warning
changed-file python -m py_compile:                      passed
git diff --check (worktree and 181f3ef..HEAD):          clean
base ancestry (181f3ef ancestor of HEAD):               confirmed
excluded-scope zero-diff over every do-not-modify path
  (evals/confirmatory production, Task 3C-3F receipt
  modules, aws_seed_transition/launcher/bootstrap/argv,
  legacy collect/cleanup/operations/approval, trainer,
  corpusgen, configs, profiles, docs, verifier):        empty
CLI subprocess matrix: collect-cohort --help exits 0
  with one JSON object; bare collect-cohort → CLI_USAGE;
  existing entries green inside tests/test_msctl.py
import-boundary subprocesses:                           passed
committed-tree package dry-run:                         ok=true, dry_run=true
frozen-science zero-diff:                               only scoped files changed
```

*The two failures are the inherited, base-reproducible v2 cohort-release
tests in `tests/test_verify_cohort_releases.py`, explicitly out of scope.

Failure-before-mutation proofs (each asserts its exact code, its exact
download count, no S3 PUT, no staging residue, no state, and no output
tree): bad/missing/foreign/wrong-key anchor triple (0 downloads), missing
or invalid index (0), index↔anchor drift (0), wrong-seed manifests 0-8
(0, zero AWS calls entirely), destination exists (0), state conflict (0),
report body/checksum drift (1), report↔index lock-SHA and receipt-row
drift (1), sealed-release↔manifest drift (1), lock body drift (2),
lock-vs-index receipt identity drift via dishonest republication (2),
instance drift against the lock (2), seed-receipt body drift (3),
`output.json` report-commitment drift (13), index↔`output.json`
artifact-row drift (13), unavailable object (13,
`COHORT_COLLECT_INCOMPLETE`). Independent stream rehash catches artifact
body, length, and returned-version drift after correct declared metadata.

Replay-only outcome authority: four hash-consistent dishonest reports
(delta, bootstrap bound, AULC, status) and two coherently republished
tampered artifacts (`outcomes.jsonl`, `metrics.json`, with index and
report hashes updated) all download cleanly — 1,012 clean fetches — and
fail only at `COHORT_COLLECT_REPLAY_FAILED` with no PUT and no state.

Publication authority: `--if-none-match "*"` with checksum and metadata
binding; lost-PUT recovery only through the exact HEAD (green when the
HEAD confirms, `COHORT_COLLECT_CONFLICT` with no state and no destination
when it drifts); post-publication HEAD version drift conflicts. Idempotent
replay with a fresh `--out` performs exactly one exact-HEAD call and zero
downloads.

## Files and commits

- `6caa483` `test: factor payload-real cohort builder into shared fixtures`
  — pure movement of the Task 4C cohort builder into
  `tests/cohort_output_fixtures.py` plus import-only edits to
  `tests/test_confirmatory_v3_aggregate.py` (its 29 tests pass unmodified).
- `d5b7ad3` `feat: collect durable cohort evaluation evidence receipts`
  - `msctl/aws_contracts.py` (member tuple + four key helpers)
  - `msctl/aws_cohort_collect.py` (new: contracts and the two parsers)
  - `msctl/aws_p5.py` (collection controller, dispatch, updated
    evaluate/cleanup messages)
  - `msctl/state.py` (`cohorts` store, singleton read/write, validator)
  - `msctl/cli.py` (`collect-cohort` leaf and selected-only routing)
  - `scripts/package_aws_p5_handoff.py` (one `REQUIRED_MEMBERS` entry)
  - `tests/test_aws_cohort_collect.py` (new, 160 tests),
    `tests/test_aws_contracts.py`, `tests/test_aws_collect.py`
    (blocked-message assertions), `tests/test_msctl.py`,
    `tests/test_package_aws_p5_handoff.py`

## Self-review and concerns

1. **Import-boundary deviation (documented).** The brief asks the module
   to import the frozen 3F field set and shape patterns while never
   importing torch at module scope, and asks the dry run to leave torch
   unimported. Those are jointly unsatisfiable today:
   `cluster/aws/p5/run_finalization.py` (frozen) imports `train.trainer`
   (and therefore torch) at module scope, so `msctl.aws_collect` and
   `msctl.aws_p5` have pulled torch on import since Task 3D/3F. The
   implemented boundary is the strongest attainable and is
   subprocess-proven: importing `msctl.aws_cohort_collect` alone leaves
   torch, `train.trainer`, and `evals.confirmatory` unimported (the
   37-field set is enumerated literally and the shape patterns/profile
   mapping are duplicated, both mirror-pinned against the frozen 3F/3D
   sets so drift fails the suite; `_RECEIPT_BINDING_FIELDS` itself is
   imported function-locally, never duplicated); the dry run leaves
   `evals.confirmatory` unimported and adds zero torch modules beyond the
   frozen controller chain; the apply-path replay imports
   `evals.confirmatory.aggregate` function-locally and adds zero torch
   modules (the replay path is snapshot-free).
2. Idempotent replays require a fresh `--out` (the reviewed 3F semantics,
   kept deliberately; already documented as 3F concern 2).
3. The index parser bounds *every* declared byte count (members, lock,
   report, seed receipts) by the mirror-pinned 2 GiB ceiling — slightly
   stricter than the aggregate, which bounds only output members; this
   fails closed and only earlier.
4. The receipt's `bytes_collected`/row bytes come from the index but every
   one of the 1,012 declarations is proven by exact-version GET, returned
   checksum/length/version equality, and independent stream rehash before
   the receipt is built.
5. The cohort receipt key is content-addressed, so distinct receipts could
   coexist on S3; the local singleton state plus the future cleanup gate
   (exact-HEAD verification of one recorded receipt) bind one, as the
   brief resolves.
6. Payload-real fixtures admit through the production parsers
   (`plan_snapshot_evaluations`, sealed-release replay, solver replay,
   exact statistics), but real production cohort evidence does not exist
   yet — the same status 4B/4C recorded.

## Task 4E handoff

Task 4E enables selected v3 evaluation against 4D's frozen contracts:

1. replace the `evaluate` block with an approval-gated selected operation
   (fresh signed deadline per plan §3/§5) running
   `evals/confirmatory/runner.py` under the sealed evaluator role over the
   100 planned bindings; the runner and its output schema must not change;
2. publish exactly 1,002 objects (the StudyLockV3, the 1,000 output
   members, the cohort report) to 4D's canonical keys under the pinned S3
   root, each `--if-none-match "*"`, checksum-bound, metadata-bound,
   exact-HEAD verified; server-side copy permitted for byte-identical
   cohort-uniform members with independent exact HEADs; collisions never
   reused or replaced (the ten seed collection receipts are already
   durable from 3F and are re-fetched by `collect-cohort` for the replay);
3. emit 4D's schema-1 cohort evidence index as the publisher's output; a
   shared round-trip test must prove the emitted index admits through
   `parse_cohort_evidence_index` and drives a green `collect-cohort`
   against an injected runner replaying the published bytes;
4. add v3 evaluation state with the same closed-schema/immutable-rewrite
   discipline; idempotent republication only via exact HEAD;
5. ordering: the publisher runs only after all 100 outputs exist and the
   local report is published (4C); `collect-cohort` runs only after the
   publisher; cleanup stays blocked throughout 4E and its enablement
   (post-4E) requires ten per-seed collection states/receipts, the 4D
   cohort collection receipt verified by exact HEAD, and a signed
   termination deadline before `terminate-instances`; 4E must keep 4D's
   blocking tests green;
6. inherited prerequisites: the canonical keys and closed member tuple in
   `msctl/aws_contracts.py`; the receipt/index parsers in
   `msctl/aws_cohort_collect.py` (reusable by the cleanup gate); the
   updated blocking messages naming Task 4E.
