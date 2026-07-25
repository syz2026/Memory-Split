# w2-cohort-verifier-v3 — cohort release verifier migrated to the v3 AWS-only N=10 contract

Branch: `cursor/cohort-verifier-v3-migration-e57b` (base `integration/provider-aware-lifecycle`, merge base `1fc65e5`)

## Verdict

The v3 verifier is correct and complete for the release shape the current
packager actually emits: `scripts/verify_cohort_releases.py` now selects a
frozen `_CohortContract` from the cohort ID the AWS receipt itself declares,
v2 is retained unchanged behind that selection, and the AWS-only N=10 v3 path
verifies a real release built by `scripts/package_aws_p5_handoff.py` from the
working-tree bytes end to end (`ok=true`, `complete_cohort` 0–9, and the three
frozen identity digests reproduced byte-for-byte from disk). Of the two
designated failures, one — `test_accepts_release_built_by_final_aws_packager` —
now passes legitimately, as does the third failure I found on arrival
(`test_accepts_release_built_by_integrated_illumina_packager`, which was pinned
to a commit that does not exist on `origin` at all). The single largest thing
missing is that **the other designated failure,
`test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle`, cannot be
fixed inside my fence and is still red**: it asserts nothing about the verifier,
it asserts the contents of `docs/AWS-P5-360M-RUNBOOK.md`, which is still an
entirely v2 document ("AWS seeds 1–4", `memorysplit-confirmatory-v2-360m-n5`,
"Illumina owns seed 0") and is outside the two files I am allowed to edit. I did
not re-pin the test constant to make it green, because that would only assert
that the runbook contains whatever the test says it contains. See Handoff 1.

A second, smaller gap worth flagging before anyone treats the v3 path as
complete: the verifier implements the **unauthenticated** v3 receipt exactly,
and will fail closed on a release built through
`build_authenticated_handoff(...)`, which injects provider-lifecycle binding
fields into both the receipt and the metadata. See Blocking gap 2.

## What I changed

Two files only. `git diff --stat 1fc65e5..HEAD` (run verbatim):

```
 scripts/verify_cohort_releases.py    | 623 ++++++++++++++++++++++-----------
 tests/test_verify_cohort_releases.py | 647 +++++++++++++++++++++++++++++++----
 2 files changed, 1021 insertions(+), 249 deletions(-)
```

`git diff --name-only 1fc65e5..HEAD -- configs/ cluster/ docs/ msctl/ evals/ train/ schemas/ DATASET-POINTER-AWS.json | wc -l` returns `0`: no frozen scientific bytes, no seed count, token budget, snapshot step or decision threshold was touched.

| Commit | Files | What |
| --- | --- | --- |
| `4d7d9db` | `tests/test_verify_cohort_releases.py` | RED. States the v3 contract as tests, adds the v3 fixture builder and the `--aws`-only invocation helper, and retargets the two historical-replay tests from dead/stale pinned SHAs to the packagers at HEAD. |
| `0a2c71b` | `scripts/verify_cohort_releases.py`, `tests/test_verify_cohort_releases.py` | GREEN. Replaces the hardcoded v2 assumptions with `_CohortContract` + explicit cohort-ID dispatch; adds `V3_CONTRACT`; makes `--illumina` optional; adds two dispatch-hardening tests. |

This report is committed separately, under `.swarm-reports/`, and touches no code.

### Decision: v2 retained, not retired

Retained, behind an explicit selection on the cohort ID declared in the AWS
receipt's `seed_assignment.cohort_id`. Reasons, in order of weight:

1. v2 artifacts are still on disk and still verifiable:
   `configs/cohort-assignment-v2.json`, `configs/preregistration-v2.yaml`,
   `configs/360m-v2/*` (10 files), `cluster/profiles/aws-p5.48xlarge.json`,
   `cluster/profiles/illumina-usfc-prd.json`, and a live v2 producer at
   `scripts/package_illumina_handoff.py`. `docs/AWS-P5-360M-RUNBOOK.md:187-196`
   still invokes the verifier with `--illumina … --aws …` and `jq -e`s
   `.illumina.seeds == [0] and .aws.seeds == [1,2,3,4]`.
2. The v2 report shape is unchanged, so that documented invocation keeps working
   (`test_accepts_exact_disjoint_five_seed_cohort_and_emits_canonical_json`
   asserts the whole report object and still passes).
3. Dispatching did not make the code ugly. Every v2/v3 difference is a value,
   not a control-flow branch: cohort ID, assignment path and `schema_version`,
   preregistration path/id, config root, `run_id` prefix, `train_corpus`,
   profile path/id and its exact contract, `package_format_version`, the
   environment-receipt field list, the provider set, the per-provider seed
   tuples, and the complete cohort. All of them live in one frozen dataclass
   (`scripts/verify_cohort_releases.py:161-268`); after the change, every
   remaining v2 string literal in the file is inside the `V2_CONTRACT`
   constructor. Verified:
   `rg -n "360m-v2|cohort-assignment-v2|preregistration-v2|range\(5\)|memorysplit-v2-360m|corpus-receipt" scripts/verify_cohort_releases.py`
   returns only lines 54–57 (the v2 constants) and 207–232 (the `V2_CONTRACT`
   body).
4. Fail-closed on the unknown: a receipt naming any third cohort ID is refused
   outright (`release receipt does not name a known cohort contract`,
   `scripts/verify_cohort_releases.py:531`) rather than defaulted to either
   contract, so an old release can never be silently re-verified under new rules.

### Every place the five-seed / two-provider assumption was encoded

Mapped by reading the whole file, not only the assertion the brief named. All
line numbers are pre-change (`1fc65e5:scripts/verify_cohort_releases.py`):

| Pre-change site | Assumption | Now |
| --- | --- | --- |
| `:2` | docstring "one frozen five-seed cohort" | rewritten to state both contracts |
| `:24` `COHORT_ID` | v2 cohort ID | `contract.cohort_id` |
| `:27-30` `EXPECTED_SEEDS` | `{illumina: (0,), aws: (1,2,3,4)}` | `contract.expected_seeds` |
| `:33-35` | `cohort-assignment-v2.json`, `preregistration-v2.yaml` | `contract.assignment_path`, `contract.evaluation_identity_path` |
| `:42-73` `_AWS_PROFILE_CONTRACT` | `profile_id == aws-p5.48xlarge`, `assigned_seeds [1,2,3,4]` | `_aws_profile_contract(profile_id=…, assigned_seeds=…)` per contract |
| `:353` and `:1030` | `package_format_version != 1` | `!= contract.package_format_version` (v3 = 2) |
| `:390` `:446` | `EXPECTED_SEEDS[provider]` in receipt validation | contract-scoped |
| `:415` | profile path literal `cluster/profiles/aws-p5.48xlarge.json` | `contract.aws_profile_path` |
| `:420` `:673` `:1106` | `DATASET-POINTER-AWS.json` literal | `contract.dataset_pointer_path` |
| `:445` `:1112` `:1254` | `configs/360m-v2/` config root | `contract.config_root` / `contract.config_paths()` |
| `:660-664` `:1057-1061` | provider profile path ternary | `contract.profile_path(provider)` |
| `:674` | `requirements-aws-p5.lock` literal | `STATIC_ENVIRONMENT_LOCK_PATH` |
| `:770` | `profile_id != expected_provider` | `!= contract.aws_profile_id` (v3 profile id is `aws-p5.48xlarge-v3`, deliberately *not* the provider) |
| `:816-832` | 5-name `required_fields` | `contract.environment_receipt_fields` (v3 = the 18-name `AWS_ENVIRONMENT_RECEIPT_V2_FIELDS`) |
| `:860-874` `_dataset_pointer` | inline expected mapping | `contract.dataset_pointer_contract` |
| `:898` `:908` | `COHORT_ID`, `EXPECTED_ARMS` in seed assignment | contract-scoped |
| `:1169` | assignment `schema_version: 2` | `contract.assignment_schema_version` (v3 = 3) |
| `:1184` | `provider_seeds` keys `{illumina, aws}` | `frozenset(contract.providers)` |
| **`:1197-1202`** | **`illumina | aws != set(range(5))`** — the assertion the brief named | generalised disjoint-union check against `contract.complete_cohort`, with a per-contract message |
| `:1282` `:1285` | `memorysplit-v2-360m-…`, `dataset/corpus-receipt.json` | `contract.run_id_prefix`, `contract.train_corpus` |
| `:1301` | `SNAPSHOT_STEPS` | `contract.snapshot_steps` (identical values; frozen, unchanged) |
| `:1429-1443` | `verify_cohort_releases` requires both releases | `illumina_release` optional; the contract decides whether it is required or forbidden |
| `:1460` `:1478` | `range(5)` cohort coverage and `complete_cohort` | `contract.complete_cohort` |
| `:1479-1484` | unconditional `illumina` report block | emitted only when an Illumina release was verified |
| `:1512` | `--illumina` `required=True` | `default=None` |

One thing v3 gains that v2 never had: the v3 AWS metadata carries no
`preregistration_sha256` binding (only Illumina metadata did), so v3 now parses
the evaluation-identity member and binds it to `schema_version: 3` /
`preregistration_id: memorysplit-confirmatory-v3`
(`scripts/verify_cohort_releases.py:_preregistration`). This check is
contract-gated and deliberately **not** enabled for v2: tightening what an
already-sealed v2 release verifies to is exactly the hazard the brief warns
about.

### Two replay tests retargeted (not deleted, not weakened)

Both pinned "final interface refs" that no longer describe anything real:

* `COHORT_AULC_REF = 70c1951fedce3a61e24b7d761fa330749b71618c` is **not reachable
  from `origin` at all**. `git fetch origin 70c1951…` →
  `fatal: remote error: upload-pack: not our ref 70c1951fedce3a61e24b7d761fa330749b71618c`.
  The replay could never run in a fresh clone; it failed here with
  `fatal: not a tree object`.
* `AWS_PACKAGE_REF = 4502db6607b673ead8613e68e8c8db49dd254b65` exists, but its
  own fixture emits a six-field `DATASET-POINTER-AWS.json`
  (`schema_version, provider, dataset_id, durable_uri_env, receipt_relative_path,
  full_corpus_in_release`, from `4502db6:tests/test_package_aws_p5_handoff.py:191-199`)
  while the packager at that commit only checked three of them
  (`4502db6:scripts/package_aws_p5_handoff.py:1322-1336`). The verifier at HEAD
  requires the exact eleven-field pointer. The test therefore asserted a lineage
  claim that is **false at HEAD, and should be false** — accepting that release
  would be the "silently changing what an old release verifies to" failure mode.

Both now build against the packagers at HEAD (`scripts/package_illumina_handoff.py`
for the v2 lineage gate, `scripts/package_aws_p5_handoff.py` for v3), which is
strictly stronger: real bytes through the shipped code, reproducible in any
clone, and self-updating instead of pinned to a dead SHA. The AWS one also now
asserts the full report (`cohort_id`, `complete_cohort`, `arms`, the whole `aws`
block, and the three identity digests read back out of the archive). I left
`AWS_PACKAGE_REF` and `COHORT_AULC_REF` at their current values so the runbook
gate's meaning is untouched.

## Commands run

Environment: `python3` 3.12.3, `pytest` 9.1.1. `pytest` and `pyyaml` were absent
on arrival (`/usr/bin/python3: No module named pytest`); installed with
`pip3 install pytest pyyaml`, then the rest with `pip3 install -r requirements.txt`
before the full-suite runs. **No AWS calls of any kind were made**, read-only or
otherwise; the only network traffic was `pip` and `git fetch origin`.

### 1. RED — before touching anything

```
$ python3 -m pytest -q -p no:cacheprovider tests/test_verify_cohort_releases.py
```

```
=========================== short test summary info ============================
FAILED tests/test_verify_cohort_releases.py::test_accepts_release_built_by_integrated_illumina_packager
FAILED tests/test_verify_cohort_releases.py::test_accepts_release_built_by_final_aws_packager
FAILED tests/test_verify_cohort_releases.py::test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle
3 failed, 44 passed in 2.89s
```

Three, not two. The three verbatim failure bodies:

```
root = PosixPath('/tmp/pytest-of-ubuntu/pytest-0/test_accepts_release_built_by_0/integrated-ref')
revision = '70c1951fedce3a61e24b7d761fa330749b71618c'
...
>       assert completed.returncode == 0, completed.stderr
E       AssertionError: fatal: not a tree object: 70c1951fedce3a61e24b7d761fa330749b71618c
E
E       assert 128 == 0
tests/test_verify_cohort_releases.py:510: AssertionError
```

```
        completed = _invoke(illumina.receipt, aws.release)

>       assert completed.returncode == 0, completed.stdout
E       AssertionError: {"error":{"code":"COHORT_RELEASES_REJECTED","message":"DATASET-POINTER-AWS.json fields are not exact"},"ok":false,"schema_version":1}
E
E       assert 2 == 0
tests/test_verify_cohort_releases.py:784: AssertionError
```

```
>           assert required in lowered
E           AssertionError: assert '4502db6607b673ead8613e68e8c8db49dd254b65' in '# aws p5 360m paired-cohort runbook\n\nthis runbook operates aws seeds 1–4 of\n`memorysplit-confirmatory-v2-360m-n5`....nfirmatory label is assigned only from the complete sealed\nfive-seed cohort after study-lock validation and replay.\n'
tests/test_verify_cohort_releases.py:1594: AssertionError
```

The two brief-designated failures are the second and third; `.superpowers/sdd/task-4b-report.md:223-226`
names exactly those two. The first is this-clone-specific and is explained above.

### 2. RED — v3 tests written before the implementation (commit `4d7d9db`)

```
$ PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/test_verify_cohort_releases.py
=========================== short test summary info ============================
FAILED tests/test_verify_cohort_releases.py::test_accepts_release_built_by_final_aws_packager
FAILED tests/test_verify_cohort_releases.py::test_accepts_v3_aws_only_cohort_and_emits_canonical_json
FAILED tests/test_verify_cohort_releases.py::test_v3_verifier_contract_mirrors_the_canonical_aws_contract_module
FAILED tests/test_verify_cohort_releases.py::test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle
4 failed, 60 passed in 4.17s
```

(The retargeted Illumina replay went green immediately, which is the point: it
proves the retained v2 path was already correct and gives a regression baseline
before I touched the verifier.)

### 3. GREEN — focused suite (commit `0a2c71b`)

```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/test_verify_cohort_releases.py
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /workspace
configfile: pytest.ini
plugins: anyio-4.14.2
collected 66 items

tests/test_verify_cohort_releases.py ................................... [ 53%]
.............................F.                                          [100%]
...
FAILED tests/test_verify_cohort_releases.py::test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle
========================= 1 failed, 65 passed in 9.08s =========================
```

44 → 65 passing. The only remaining failure is the runbook document gate.

### 4. Every v3 rejection fires for the right reason

A green "rejected" assertion is worthless if the rejection came from argument
parsing. I ran each negative fixture through the CLI and printed the actual
`error.message`:

```
rc=0  baseline v3 (expect accept)
    -> ACCEPTED
rc=2  nine seeds
    -> AWS receipt seed_assignment seeds are incorrect
rc=2  eleven seeds
    -> AWS receipt seed_assignment seeds are incorrect
rc=2  five seeds
    -> AWS receipt seed_assignment seeds are incorrect
rc=2  second provider in provider_seeds
    -> cohort assignment provider_seeds fields are not exact
rc=2  v2 assignment schema_version
    -> cohort assignment schema_version is incorrect
rc=2  v2 package_format_version
    -> AWS package_format_version is unsupported
rc=2  v2 config root
    -> AWS receipt config_sha256 inventory is not exact
rc=2  v2 provider profile
    -> provider profile profile_id is incorrect
rc=2  v2 environment receipt contract
    -> AWS receipt environment.runtime_environment_receipt.required_fields does not match the strict contract
rc=2  v2 train_corpus in one run config
    -> configs/360m-v3/split90-s7.yaml has incorrect train_corpus
rc=2  v2 run_id in one run config
    -> configs/360m-v3/dense-s3.yaml has incorrect run_id
rc=2  v2 preregistration identity
    -> preregistration schema_version is unsupported
rc=2  dataset pointer not exact
    -> dataset pointer full_corpus_in_release is incorrect
rc=2  v3 paired with an Illumina release
    -> this cohort contract is AWS-only and admits no Illumina release
rc=2  v2 cohort verified without its Illumina half
    -> this cohort contract requires an Illumina release as well
rc=0  v2 pair (expect accept)
    -> memorysplit-confirmatory-v2-360m-n5
```

### 5. End-to-end against real working-tree bytes (no fixtures)

```
$ PYTHONDONTWRITEBYTECODE=1 python3 scripts/package_aws_p5_handoff.py --out-dir /tmp/w2-release --apply
{"archive":"/tmp/w2-release/aws-p5-r1-af4d05bbbbf555af/ms-aws-p5-r1-af4d05bbbbf555af.zip","dry_run":false,"ok":true,"provider":"aws-p5.48xlarge","published":true,"release":"/tmp/w2-release/aws-p5-r1-af4d05bbbbf555af/RELEASE-AWS-P5.json","release_dir":"/tmp/w2-release/aws-p5-r1-af4d05bbbbf555af","release_id":"aws-p5-r1-af4d05bbbbf555af","schema_version":1,"sha256":"0f8ab689f2ee7dfb86cffbcc6cf7230476687817a8c549042134e4fc8c983b36","sha256_file":"/tmp/w2-release/aws-p5-r1-af4d05bbbbf555af/ms-aws-p5-r1-af4d05bbbbf555af.zip.sha256"}

$ python3 scripts/verify_cohort_releases.py --aws /tmp/w2-release/aws-p5-r1-af4d05bbbbf555af/RELEASE-AWS-P5.json
{"arms":["dense","split90"],"aws":{"archive_sha256":"0f8ab689f2ee7dfb86cffbcc6cf7230476687817a8c549042134e4fc8c983b36","provider":"aws-p5.48xlarge","release_id":"aws-p5-r1-af4d05bbbbf555af","seeds":[0,1,2,3,4,5,6,7,8,9]},"cohort_assignment_sha256":"47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c","cohort_id":"memorysplit-confirmatory-v3-360m-n10-aws","complete_cohort":[0,1,2,3,4,5,6,7,8,9],"corpus_identity_sha256":"c704baf9e07acb93ffab8b33027b310fc9b3d68e0c036a6d8cc1d45b86872ce8","evaluation_identity_sha256":"6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7","ok":true,"schema_version":1,"source_commit":"0a2c71b7e83d0f01858221e4bc79e89bceea03b2"}
rc=0

$ python3 scripts/verify_cohort_releases.py --illumina <that same v3 receipt> --aws <same>
{"error":{"code":"COHORT_RELEASES_REJECTED","message":"illumina-usfc-prd release receipt fields are not exact"},"ok":false,"schema_version":1}
rc=2
```

The three identity digests in that report are the real frozen artifacts, cross-checked independently:

```
$ sha256sum configs/cohort-assignment-v3.json configs/reasoning-dataset-v2.json configs/preregistration-v3.yaml
47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c  configs/cohort-assignment-v3.json
c704baf9e07acb93ffab8b33027b310fc9b3d68e0c036a6d8cc1d45b86872ce8  configs/reasoning-dataset-v2.json
6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7  configs/preregistration-v3.yaml
```

### 6. Full-suite regression comparison (base vs. head, same host, same flags)

Base (`1fc65e5`, in a detached `git worktree` at `/tmp/base-check`, so my branch
was never left):

```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider
51 failed, 3155 passed, 2 deselected in 668.83s (0:11:08)
```

Head (`0a2c71b`):

```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider
48 failed, 3177 passed, 2 deselected in 660.13s (0:11:00)
```

Set difference of the two `FAILED` lists:

```
=== fixed by this branch (in base, not in head) ===
tests/test_confirmatory_sealing.py::test_publish_boundary_detects_concurrent_membership_mutation[replace]
tests/test_verify_cohort_releases.py::test_accepts_release_built_by_final_aws_packager
tests/test_verify_cohort_releases.py::test_accepts_release_built_by_integrated_illumina_packager
=== newly failing (in head, not in base) ===
=== counts ===
  51 base
  48 head
```

**Zero newly failing tests.** (The `test_confirmatory_sealing` entry is a
concurrency test that flipped between the two runs — head still shows a sibling
`…::test_quarantine_detects_membership_mutation_after_rename[replace]` failure.
It is flaky on this host and I claim no credit for it.)

## Gates now passing

| Gate | Evidence |
| --- | --- |
| `test_accepts_release_built_by_final_aws_packager` — one of the two brief-designated failures | Green at `0a2c71b`. Builds a real release with `scripts/package_aws_p5_handoff.py` at HEAD via `tests/test_package_aws_p5_handoff.py::_minimal_repo`, verifies with `--aws` only, asserts `cohort_id`, `complete_cohort == list(range(10))`, `arms`, the full `aws` block, and all three identity digests read back out of the archive. |
| `test_accepts_release_built_by_integrated_illumina_packager` — third failure found on arrival | Green at `4d7d9db` (before any verifier change) and still green at `0a2c71b`; doubles as the v2 no-regression gate. |
| v3 acceptance + canonical single-line JSON | `test_accepts_v3_aws_only_cohort_and_emits_canonical_json` asserts the whole report object and re-serialises it to prove byte-canonicality. |
| v3 refuses every v2 carry-over | 12 negative tests, all confirmed to fire on the intended check (section 4 above): `…without_exact_ten_seed_coverage[short|long|five-seed]`, `…names_a_second_provider`, `…v2_assignment_schema_version`, `…v2_package_format_version`, `…v2_config_root`, `…reuses_the_v2_provider_profile`, `…v2_environment_receipt_contract`, `…v2_corpus_receipt`, `…v2_run_id`, `…non_v3_preregistration_identity`, `…dataset_pointer_is_not_exact`. |
| Contract dispatch is fail-closed | `test_rejects_release_receipt_that_names_an_unknown_cohort`, `test_rejects_v2_release_relabelled_as_the_v3_cohort`, `test_rejects_v2_cohort_release_that_omits_the_illumina_half`, `test_rejects_v3_cohort_paired_with_an_illumina_release`. |
| Verifier constants cannot drift from the canonical contract | `test_v3_verifier_contract_mirrors_the_canonical_aws_contract_module` pins `V3_CONTRACT` field-by-field to `msctl/aws_contracts.py` (cohort id, both config paths, preregistration id, config root, profile path, dataset pointer path, seeds, arms, snapshot steps, `PACKAGE_FORMAT_VERSION`, the 18 environment-receipt fields, and the full 20-path `EXPECTED_CONFIG_PATHS` set). The verifier still does **not** import `msctl`, so it remains an independent restatement. |
| v2 unchanged | Every pre-existing v2 test in the file still passes, including the full-report equality test `test_accepts_exact_disjoint_five_seed_cohort_and_emits_canonical_json`. |
| No frozen scientific bytes touched | `git diff --name-only 1fc65e5..HEAD -- configs/ cluster/ docs/ msctl/ evals/ train/ schemas/ DATASET-POINTER-AWS.json` → 0 files. |
| Fence respected | `git diff --stat 1fc65e5..HEAD` lists exactly `scripts/verify_cohort_releases.py` and `tests/test_verify_cohort_releases.py`. |
| No new regressions repo-wide | Section 6: 51 → 48 failures, empty "newly failing" set. |

## Blocking gaps

1. **`test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle` is still red
   and cannot be fixed in this fence.** Size **M** (for whoever owns the doc).
   Path: `docs/AWS-P5-360M-RUNBOOK.md`. The immediate cause is a one-line drift:
   commit `1fdc239` ("fix: align verifier provider contracts") bumped
   `AWS_PACKAGE_REF` in `tests/test_verify_cohort_releases.py:28` from
   `e01cee358288684c801c2bfff277ee0700e6034b` to
   `4502db6607b673ead8613e68e8c8db49dd254b65` without updating
   `docs/AWS-P5-360M-RUNBOOK.md:26`, which still exports the old value. (The old
   value is not even a valid object in this clone: `git log e01cee35…` →
   `fatal: bad object`.) The real cause is larger: the runbook is still wholly
   v2 — `:3-4` "operates AWS seeds 1–4 of `memorysplit-confirmatory-v2-360m-n5`.
   Illumina owns seed 0", `:32` `MS_COHORT_ID=memorysplit-confirmatory-v2-360m-n5`,
   `:46` `MS_ILLUMINA_RELEASE`, `:191-196` `jq -e '.illumina.seeds == [0] and
   .aws.seeds == [1,2,3,4] and .complete_cohort == [0,1,2,3,4]'`. Why it blocks:
   it is one of the two failures the whitelist exists for, so the whitelist
   cannot be retired until it is fixed. I deliberately did **not** re-pin the test
   constant to the runbook's value, which would have made it green while
   reducing the assertion to a tautology.
2. **Lifecycle-authenticated v3 releases are refused.** Size **S–M**. Path:
   `scripts/verify_cohort_releases.py:_receipt` (receipt field set) and
   `:_metadata` (metadata field set). `scripts/package_aws_p5_handoff.py:1895-1935`
   and `:1559-1588` splat `collected.lifecycle_fields` plus a top-level
   `profile_id` into both the receipt and `RELEASE-METADATA.json` when a release
   is built through `build_authenticated_handoff(...)`. Those fields come from
   `msctl/aws_lifecycle.py:192-…` (`account_id`, `arms`, `availability_zone`,
   `boot_id`, `cohort_id`, `hardware_amendment_sha256`, `instance_id`,
   `objective_controls_contract_sha256`, `profile_id`, `profile_sha256`,
   `provider`, `provider_selection_sha256`, `provider_selection_version_id`,
   `purchase_model`, `qualification_approval_public_key_sha256`,
   `qualification_approval_receipt_sha256`, …). The verifier's `_strict_object`
   will reject such a release with `release receipt fields are not exact`. This
   is fail-closed, not unsafe, but on the `integration/provider-aware-lifecycle`
   line it means the authenticated packaging path has no verifier. Fixing it
   properly means deciding whether those fields are *required* under a third
   contract variant or merely *permitted*, and cross-binding `profile_id` to the
   selected profile — a contract decision, not a code tweak, and I had no
   authenticated fixture to test against. It is inside my fence but out of my
   brief; flagging rather than guessing.
3. **`aws-p6-b300.48xlarge-v3` releases have no verifier path.** Size **M**.
   Path: `scripts/verify_cohort_releases.py` (`V3_CONTRACT.aws_profile_path` is
   hardcoded to the P5 profile). `msctl/aws_contracts.py:107-113` defines a
   second qualified profile and `scripts/package_aws_p5_handoff.py:1530-1535`
   can select `cluster/profiles/aws-p6-b300.48xlarge-v3.json`. Since profile
   selection only happens on the authenticated path, this is downstream of
   gap 2; both should be resolved together.
4. **The v3 AWS release has no dedicated preregistration hash binding in its
   metadata.** Size **S**. Path: `scripts/package_aws_p5_handoff.py:1559-1588`.
   The v2 *Illumina* metadata carried `preregistration_sha256`; the v3 AWS
   metadata carries `cohort_assignment`, `profile` and `dataset_pointer`
   bindings but nothing for `configs/preregistration-v3.yaml`. The bytes are
   still authenticated (SHA256SUMS → `members` → `members_sha256` → receipt), and
   I added a semantic identity check in the verifier as compensation, but there
   is no top-level path/hash binding the operator can `jq` out of the receipt the
   way `.cohort_assignment_sha256` is used at `docs/AWS-P5-360M-RUNBOOK.md:201-203`.
   Adding one requires changing the packager, which is outside my fence.

## Handoffs

1. **Owner of `docs/AWS-P5-360M-RUNBOOK.md`** — migrate the runbook to the v3
   AWS-only N=10 lifecycle. Minimum to clear gap 1: set `MS_COHORT_ID` to
   `memorysplit-confirmatory-v3-360m-n10-aws`, drop `MS_ILLUMINA_RELEASE` and the
   `--illumina` argument at `:187-190`, change the `jq -e` gate at `:191-196` to
   `.ok == true and .aws.seeds == [0,1,2,3,4,5,6,7,8,9] and .complete_cohort ==
   [0,1,2,3,4,5,6,7,8,9]` with no `.illumina` clause, and rewrite the seeds-1–4 /
   "Illumina owns seed 0" framing at `:3-6`. The v3 report keys that gate can
   rely on are exactly those shown in section 5 above. Also either re-pin
   `MS_AWS_PACKAGE_INTERFACE_REF` at `:26` to
   `4502db6607b673ead8613e68e8c8db49dd254b65` (matching
   `tests/test_verify_cohort_releases.py:28`) or, better, agree with the owner of
   that test file to retire the "integration constants" block entirely, since two
   of the five pinned SHAs (`70c1951`, `e01cee35`) no longer exist in the
   repository. Coordinate with me before touching
   `tests/test_verify_cohort_releases.py` — it is inside my fence.
2. **Owner of `scripts/package_aws_p5_handoff.py` / `msctl/aws_lifecycle.py`** —
   publish the authoritative field list for a lifecycle-authenticated v3 receipt
   and metadata (gap 2), and say whether those fields are required or optional.
   I will extend `_CohortContract` to cover them once the contract is fixed;
   a single authenticated fixture release would be enough to test against.
3. **Owner of `scripts/package_aws_p5_handoff.py`** — consider adding a
   `preregistration` path/hash binding to the v3 receipt and metadata (gap 4),
   symmetric with the existing `cohort_assignment` / `profile` /
   `dataset_pointer` bindings.
4. **Environment / infrastructure owner** — two host-level issues that cost me
   time and will cost every other agent the same:
   - `tests/test_package_aws_p5_handoff.py::test_cli_apply_uses_documented_external_output_by_default`
     fails unless `PYTHONDONTWRITEBYTECODE=1` is set. The packager subprocess
     imports `msctl.aws_contracts` from the fixture repo, which writes
     `msctl/__pycache__/`, and `_clean_revision` runs
     `git status --untracked-files=all` and sees a dirty tree. Confirmed: fails
     without the variable, passes with it. Either export it in CI or add
     `__pycache__` to the fixture's ignore set.
   - 44 of the 48 remaining full-suite failures are in `tests/test_msctl.py` and
     all reduce to `msctl.errors.MsctlError: environment Python does not match
     the pinned contract` (`msctl/environment.py:731`). They are pre-existing
     (44 of them fail identically at base `1fc65e5`) and unrelated to this work,
     but they mean "the suite is green except for two known failures" is not true
     on this VM image.
