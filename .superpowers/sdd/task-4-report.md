# Task 4 report
- Status: `DONE_WITH_CONCERNS`.
- Branch: `feat/memorysplit-v2-task-4`; exact base: `c190429`.
- Prior commits: `f5ff56c`, `afd8038`, `78a4a61`, `7897f13`, `c514769`.
- Review fix: `2b2e77d` (exact receipt paths, strict scalars/sidecars, replay-safe resume).
- Final review: strict post-verification receipt identity and bidirectional
  Dense/Split90 selector-to-sidecar enforcement.
- Crash replay: durable-prefix log truncation, strict complete-row validation,
  owned atomic-temp/hardlink recovery, and hard-linked log rejection.
- Snapshot replay now rejects steps outside the exact `snapshot_steps` set.
- Runtime: strict capability JSON, atomic rank-zero PID, request-only SIGUSR1, synchronized checkpoint.
- Snapshots: exact strict `snapshot_steps`; preregistered `[1358,3396,6791,10187,13582]` is covered.
- TDD: all new regressions were observed failing before implementation and passing afterward.
- Deadline-mode focused evidence: loader/trainer `94 passed`; real gloo
  checkpoint/SIGUSR1 `4 passed, 3 deselected`.
- Scoped/smoke/gloo: `181 passed, 1 deselected in 57.96s`.
- Full suite unrestricted: `837 passed, 2 deselected in 195.76s`.
- Static: `py_compile`, Ruff, capability probe, and `git diff --check` passed.
- Concern: no production-scale 7.12B-target corpus or GPU run was executed locally.
- Report: `.superpowers/sdd/task-4-report.md`.
