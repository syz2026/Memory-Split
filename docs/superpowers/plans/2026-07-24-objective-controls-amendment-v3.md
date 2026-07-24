# Objective Controls Amendment V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an append-only prospective amendment that expands only the 29M
development diagnostics from six to eight runs while preserving the frozen
20-cell 360M primary cohort byte-for-byte.

**Architecture:** A canonical YAML amendment binds the existing frozen
preregistration and cohort hashes and points to one canonical JSON manifest.
Eight hash-bound YAML configs describe the development-only runs. A new
standalone `msctl.objective_controls_v3` module validates those artifacts and
future admission evidence without wiring them into the shared lifecycle.

**Tech Stack:** Python 3.11, PyYAML, canonical JSON, dataclasses, pytest.

## Global Constraints

- Base commit is exactly `b3471e0969ca2a997d33acf60d2e777720afa1c4`.
- Never modify `configs/preregistration-v3.yaml` or
  `configs/cohort-assignment-v3.json`.
- The protected cohort remains exactly ten Dense/Split90 pairs and 20 cells.
- No 360M controls, runtime, launcher, evaluator, corpus generator, study lock,
  AWS profile, package, or IaC is added.
- The 29M matrix has exactly eight runs at 28,969,216 parameters, 1,106
  updates, 524,288 targets/update, and 579,862,528 raw targets.
- The two new full-corpus controls are integrity-only and non-claim-bearing.
- Validation rejects missing or extra runs, unknown fields, bool/integer
  aliases, float/integer aliases, and cross-variant or cross-sidecar bindings.

---

### Task 1: Specify the prospective contract with failing tests

**Files:**
- Create: `tests/test_objective_controls_v3.py`

**Interfaces:**
- Consumes: the user-approved scientific decision and frozen V3 artifacts.
- Produces: executable expectations for
  `load_objective_controls_contract(path)`.

- [ ] Write tests importing the wished-for focused module and asserting the
  exact frozen hashes, protected cells, eight-run manifest, replacement
  identities, control exclusions, and config hashes.
- [ ] Add mutation tests for missing/extra runs, parent drift, config drift,
  numeric aliases, and corpus/sidecar/provenance cross-binding.
- [ ] Run:

  ```bash
  python -m pytest -q -p no:cacheprovider tests/test_objective_controls_v3.py
  ```

  Expected: collection fails because `msctl.objective_controls_v3` does not
  exist.

### Task 2: Add the amendment, manifest, and eight configs

**Files:**
- Create: `configs/objective-controls-amendment-v3.yaml`
- Create: `configs/29m-v3/manifest.json`
- Create: `configs/29m-v3/*.yaml` (exactly eight files)

**Interfaces:**
- The amendment binds SHA-256 identities of both frozen parent files.
- The manifest binds every config SHA-256 and explicit corpus, sidecar, and
  provenance IDs.

- [ ] Write the append-only amendment with the exact 20 protected cells,
  statistical exclusions, no-outcome-inspection declaration, and explicit
  disclaimer of a replicated 360M selectivity-vs-random claim.
- [ ] Write all eight configs with one shared seed and initialization,
  preserving `fineweb_edu` and
  `verified_standard_relational_records` replacements.
- [ ] Generate the canonical manifest from the final config byte hashes.

### Task 3: Implement the focused parser and validator

**Files:**
- Create: `msctl/objective_controls_v3.py`

**Interfaces:**
- `load_objective_controls_contract(path) -> ObjectiveControlsContract`
- `validate_objective_controls_admission(contract, value) -> ObjectiveControlsAdmission`
- `load_objective_controls_admission(path, contract) -> ObjectiveControlsAdmission`

- [ ] Parse YAML/JSON with duplicate-key and non-finite-number rejection.
- [ ] Require exact schemas, paths, IDs, hashes, integer/float types, run sets,
  and cross-file bindings.
- [ ] Enforce non-directional admission: finite optimization, exact resume with
  maximum delta `1e-5`, language degradation at most `1%`, all three audits,
  and strictly greater than `0.75` in every primary cell for only the six
  original runs.
- [ ] Require `null` primary-cell accuracy for both integrity-only controls.
- [ ] Run the focused tests until green, then refactor only while green.

### Task 4: Verify and report

**Files:**
- Create: `.superpowers/sdd/objective-controls-amendment-report.md`

- [ ] Run the focused test file.
- [ ] Run `python -m py_compile` on the module and focused test.
- [ ] Run `git diff --check`.
- [ ] Prove both frozen files are unchanged from the pinned base and only
  scoped files changed.
- [ ] Record RED/GREEN evidence, test output, exact hashes, exclusions, and
  concerns in the report.
- [ ] Commit the scoped implementation and report.
