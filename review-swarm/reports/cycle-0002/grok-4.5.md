# Cycle 2 — Lane A, estimand and inference — Grok 4.5

**Verdict: Opus 5's four lane-A findings all hold under adversarial re-check (MS-006 refined on scope, not retired); two new S0 defects in the frozen gate/estimand path, where the preregistration promises confidence-bound gates and a primary scoring band that `check_gates` / `analyse` never implement.**

HEAD `efc338a`. FarmShare unreachable; frozen difficulty table and production checkpoints not inspectable.

---

## Cross-verification

### MS-001 — The preregistration is banner-frozen with the primary load unnamed

- **Status:** CONFIRMED
- **Check output:**
  ```
  $ rg -n "primary load|primary_load" docs/PREREGISTRATION.md
  336:  primary load rather than pooled across loads (§6); the dose-response shape
  ```
  Only a backreference. No assignment.
- **Hard look at implied primary (`f = 1.0`):** §3 tags loads `f = 1.0, 0.3, 0.1` as `[MEASURED]` and specifies `N, E` at `f = 1.0` as the `high-e200` / F/C = 1.000 construction point. That makes `f = 1.0` the natural *reference load for building the ladder*, not the named primary estimand. `docs/AMENDMENT-2026-08-02.md:171-172` still says explicitly: "Still to be frozen before the matrix runs. The identity of the primary load. §5 does not name it, and this amendment deliberately does not choose it." `scripts/analyze_crowding.py:119-125` still hard-refuses without `--primary-load` and points at §5, which freezes the scoring *band*, not the load. Nothing in §3 or §5 converts the construction reference into a frozen primary. Severity S0 stands: the estimand is at one load, and that load is unnamed under a FROZEN banner.
- **Reasoning:** The narrow defence ("it was never `[PILOT]`") is real and is already in the finding; it does not retire it. Tried to break it via implied-primary reading; failed.

### MS-005 — The §10 control table cannot be reproduced from any single treatment mean

- **Status:** CONFIRMED
- **Check output:**
  ```
  vs 1.9584: [35.1, 23.5, 20.0, 9.7, 4.8]
  vs 1.9598: [35.1, 23.5, 20.1, 9.8, 4.9]
  1.9584 0.20001021241830064 False
  1.9598 0.20058169200938872 False
  ```
  Rows 3 vs 4–5 require different denominators. Oracle fails `rel <= 0.20` under both (`corpusgen/randpos.py:211`).
- **Reasoning:** Independently reproduced. Note ledger already has Confirmations: 2; this is a third independent run of the check, not a new filing. Opus's original A-02 (wrong single denominator) stays correctly graveyarded.

### MS-006 — The 35.1% gap is a lookup-table average, not measured masked-span loss

- **Status:** REFINED
- **Evidence traced:**
  - `ops/crowding/build_corpus.py:202-204`: `_doc_nll(ids, nll_table)` returns `nll_table[ids]` (vocab lookup) or `None`.
  - `build_corpus.py:253-254,257-262`: `match_report` and `NLLMatchAccumulator.observe` average that lookup over masked positions; corpus means are `fact_sum/fact_n` and `rand_sum/rand_n` of the same table.
  - `corpusgen/randpos.py:204-211`: `mean_nll_*` and `nll_within_tolerance` are computed from the passed `token_nll` array, which on the production path is that lookup.
  - `ops/crowding/nll_table.py:21-26`: docstring states explicitly this is "a marginal, not a contextual, difficulty" and "a deliberate approximation".
- **Paper claims under attack still stand as overclaims:** `PAPER-WORKSHOP.md:106-107` and `PAPER-MEASUREMENT.md:241` / `RETAINED-RESULTS.md:290` promote the figures to "learnable signal" without saying they are table averages. No draft scopes 1.9598 / 1.2712 as marginal table statistics.
- **Refinement (legitimate operationalization argument):** Preregistration §2 *defines* the fourth matching axis as "using a token-NLL table frozen from a pilot run … mean per-token difficulty" and makes failure of "mean masked-span NLL" within 20% the reportable result. Under that operationalization, calling the printed numbers "mean masked-span NLL" is entitled. What is not entitled is equating them with contextual model loss or "learnable signal removed." The self-optimizing matcher point (placement by `argmin |window_mean − target_nll|` on the same table the report averages) remains load-bearing and keeps severity at S1.
- **What would retire the refined finding:** drafts that say "marginal per-id table average used for placement" whenever they print 1.9598 / 1.2712 / 35.1%, and that reserve "learnable signal" for a contextual measurement — or a contextual forward-pass that agrees. FarmShare unreachable, so the contextual check could not be run here.

### MS-007 — The difficulty table's leak guard protects the data channel, not the model channel

- **Status:** CONFIRMED
- **Evidence:**
  - `nll_table.py:142-147` refuses only when `--seed ∉ BURNED_PILOT_SEEDS`.
  - `nll_table.py:158`: seed feeds `bios.generate_records(entities, seed)` only.
  - `nll_table.py:127-129,151-155`: `--run` / `--ckpt` load the model with no seed or burned-run check.
  - Provenance sidecar records `checkpoint` and `checkpoint_sha256` (`:171-178`) and nothing reads them back for a guard.
  - `PAPER-WORKSHOP.md:208-211` asserts "the gap between arms is unbiased"; `RETAINED-RESULTS.md:298-299` records the table as frozen from "the step-ladder run's step-1,564 snapshot" (model channel non-disjointness admitted in prose, not blocked in code).
- **Bias half:** FACTMASK spans are role-selected (table values unselected); RANDPOS spans are chosen to match a noisy table estimate, so winner's-curse in the control does not cancel in the gap. Asserting unbiasedness without a disjoint-checkpoint comparison is unsupported. Severity S1 stands.
- **Could not check:** the shipped `nll_table.provenance.json` on FarmShare (unreachable). Finding does not need it to stand against the code and the draft's own limitation sentence.

---

## New findings

### Confirmatory gates mostly use point estimates while the preregistration requires confidence bounds

- **Severity:** S0
- **Target:** `docs/PREREGISTRATION.md:237` against `evals/stats.py:295-340` and `scripts/analyze_crowding.py:154-163`
- **Claim under attack:** "All gates use confidence bounds, not point estimates."
- **Defect:** Only gate 1 (`sup_carries_a_burden`) receives a bound: `bits_lower_bound["sup"] = seed_contrast(...)["ci_lower"]`. Gates 2–5 in `check_gates` compare point means — `bits["factmask"]` vs `0.10 * bits["sup"]`, mean `igsm_acc_sup` vs the band, `|bits["randpos"] - bits["sup"]|`, and clip-ratio spread. Gate 6 ("Per-arm mean unmasked-token NLL") is named in §7 and is absent from `check_gates` entirely. §7.3's monotone-in-`op` and monotone-over-training clauses are also absent from the confirmatory path (they exist only in the pilot gate). A matrix can therefore pass or fail validity on seed-mean point estimates that the frozen document forbids, and the entropy-asymmetry reporting gate cannot fire at all.
- **Why existing checks miss it:** `tests/test_stats_crowding.py` asserts the split into validity/reporting and the three-way verdict, and feeds hand-built point dicts; nothing asserts that each gate's `observed` field is a confidence bound, or that gate 6 exists.
- **Falsifiable check:**
  ```bash
  PYTHONPATH=. python3 - <<'PY'
  import inspect
  from evals.stats import check_gates
  src = inspect.getsource(check_gates)
  assert "bits_lower_bound" in src
  # These must NOT decide gates on bare means if §7 holds:
  print("factmask uses bits.get:", "bits.get(\"factmask\"" in src)
  print("endpoint uses point igsm_acc_sup:", "igsm_band[0] <= igsm_acc_sup" in src)
  print("gate 6 / unmasked NLL present:", "unmasked" in src.lower())
  PY
  ```
  Expected today: `True`, `True`, `False`. Fix by threading per-seed vectors into every gate and comparing bounds, and by implementing gate 6.
- **What would prove me wrong:** `check_gates` deciding every §7 gate on a confidence bound (or one-sided interval) computed across seeds, and a sixth gate reporting per-arm unmasked-token NLL.

### The §5 primary scoring band is never instantiated, so the confirmatory estimand is not the one the preregistration defines

- **Severity:** S0
- **Target:** `docs/PREREGISTRATION.md:153-161` and `:225-226` against `scripts/analyze_crowding.py:136-165` and `evals/stats.py:325-328`
- **Claim under attack:** Primary scoring band is "the contiguous run of `op` values whose SUP accuracy falls inside [0.17, 0.85]", `[RULE]`, instantiated by the SUP arm of the first run; §7.3 requires "SUP iGSM accuracy inside the frozen band"; the primary estimand is closed-book iGSM accuracy on that band.
- **Defect:** `analyse` reads `summary.json["igsm_acc"]` (overall accuracy over whatever `igsm_op` the run config set — default `[1, 4]` in `run_evals.py:194`, widened corpora use `[1, 8]`) and never reads `igsm_by_op`, never computes a contiguous in-band op run, and never restricts `Y[FACTMASK] − Y[RANDPOS]` to those ops. `check_gates`'s `endpoint_is_measurable` checks whether the *mean aggregate* SUP accuracy lies in the CLI `--igsm-band` interval `(0.17, 0.85)`. That is the *selection interval*, not the frozen op set, and it is evaluated on a cross-seed mean rather than per-seed. After the corpus was widened because op 1–4 sat at 93.8%, an aggregate over `[1, 8]` can land inside `[0.17, 0.85]` while the estimand still averages ceiling ops the `[RULE]` exists to exclude.

  **Carried-forward circularity, settled:** §7.3 is *not* circular against §5 as implemented, because the code never instantiates the op-band from accuracy and then re-checks those ops on the same run. The gate and the estimand simply ignore the `[RULE]`. The ambiguity Opus left open is resolved in the worse direction: not "evaluated per-seed on later matrix seeds" and not "tautological on the instantiating run", but "unimplemented."
- **Why existing checks miss it:** tests pass synthetic `igsm_acc` scalars inside `(0.17, 0.85)`; nothing builds a per-op curve, freezes a contiguous band, and asserts the primary contrast uses only those ops.
- **Falsifiable check:**
  ```bash
  rg -n "igsm_by_op|primary.?scoring|contiguous" scripts/analyze_crowding.py evals/stats.py
  # expect no hits that instantiate or apply the band
  PYTHONPATH=. python3 - <<'PY'
  from pathlib import Path
  import scripts.analyze_crowding as ac
  # or importlib load; assert analyse source never references igsm_by_op
  src = Path("scripts/analyze_crowding.py").read_text()
  assert "igsm_by_op" not in src
  print("analyse does not touch igsm_by_op")
  PY
  ```
  Fix: freeze the contiguous op run from the first SUP matrix seed into the preregistration (or a dated sidecar), score the primary contrast on that band only, and make §7.3 check per-seed SUP accuracy on that band with the monotone clauses.
- **What would prove me wrong:** code that (a) derives the contiguous op set from the first SUP run's `igsm_by_op`, (b) writes it down before FACTMASK/RANDPOS are scored, and (c) computes both the primary contrast and the §7.3 gate on that set.

### Pilot §8.2 treats missing monotone evidence as a pass

- **Severity:** S1
- **Target:** `ops/crowding/analyze_pilot.py:58-62,137-143` against `docs/PREREGISTRATION.md:247` / `analyze_pilot.py:15-18`
- **Claim under attack:** GO/NO-GO item 2 requires accuracy inside the frozen band, **and** monotone decreasing in `op`, **and** monotone increasing over training.
- **Defect:** The gate accepts `monotone_in_op is not False` and `monotone_over_training is not False`. When `acc_by_op` has fewer than two ops, or `acc_over_training` has fewer than two points, `stage_a` sets those fields to `None`, and `None is not False` is true. A Stage A cell that never measured a curve therefore contributes a vacuous pass on two of the three conjuncts. Reproduced:

  ```
  chosen monotone_in_op None
  chosen monotone_over_training None
  decision GO
  endpoint_is_a_construct True {... 'monotone_in_op': None, 'monotone_over_training': None}
  ```

  Rising-in-op still correctly NO-GOs when the curve exists and fails. The defect is the silent pass under missing evidence, not the failure path.
- **Why existing checks miss it:** `tests/test_pilot.py` healthy-GO fixture supplies a two-op decreasing curve and a two-point training curve; nothing asserts that `None` fails the gate.
- **Falsifiable check:** the snippet above, or a unit test that feeds a single-op Stage A cell and expects NO-GO (or an explicit `monotone_unmeasured` failure) rather than GO.
- **What would prove me wrong:** the gate treating `None` as failure, or Stage A refusing to emit a chosen cell without both monotone measurements.

---

## Checked and found clean

- **`verdict()` three-way rule.** Matches §6: `validated` iff `ci_lower > MIE`, `rejected` iff `ci_upper < MIE`, else `inconclusive`; `invalid` on any failed validity gate. Covered by existing tests; no drift found.
- **§6 power / MIE arithmetic.** `sqrt(0.25/1500) = 0.012910`; 2× = 0.02582 matches Amendment 2 / §8.3. `min_detectable_effect` at n=8, sd≈1.31 pp is ≈1.29 pp, consistent with the orientation figure in §6. The *selection* of the smallest n in `[6, 8]` is not automated (CLI `--n-confirm` defaults to 8); that is a procedural gap short of a false claim, left unfiled.
- **§8 GO/NO-GO failability (when evidence exists).** All four pilot checks can fail under constructed inputs; tests cover exposure-limited, small delta, underpowered, and floored endpoint. Soft-pass on missing monotone is the exception, filed above.
- **§10a resource ceiling ↔ `guard.py`.** `Ceiling.from_prereg()` parses the `prereg-ceiling` block and returns `scratch_bytes=150000000000`, `gpu_hours=1600`, `soft_stop_fraction=0.9`, `divergence_factor=1.5`, and the six named campaigns. Soft stop, scratch projection, campaign binding, and divergence halt match the document. Clean.
- **MS-001 implied-primary escape hatch.** Closed; see cross-verification.
- **§7.3 / §5 circularity.** Settled as "not circular; unimplemented." See second new finding.
- **Did not re-run** Opus's cycle-1 clean list (Amendment 2 circularity fix, reduced-form estimand, load construction, `dose_response_signature` floor, `responsive()` ceiling side, F/C invariance). Repository unchanged since cycle 1.
- **Did not run** the full pytest suite (~12 min); not required for these checks. FarmShare artifacts not inspectable from this checkout.
