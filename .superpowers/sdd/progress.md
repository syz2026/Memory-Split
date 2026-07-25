Execution branch: integration/provider-aware-lifecycle
Approved plan: /Users/stephenzhang/.cursor/plans/aws_n10_readiness_41a3c374.plan.md
Corpus producer remains independent: do not modify corpusgen/
Atomic Task 3C base: b7382e2, review clean
Provider lifecycle bridge: complete through bec4c79, 1166 tests and package dry-run
Task 3D provider-aware clean-run finalization: complete
  commits e9e2c74..a6e503e, review clean
  exact brief verification 493 + 337 passed
  committed package dry-run passed
Relational bundle audit fix: integrated as 1714c78, review clean, 37 passed
Task 3E safe sequential seed transitions/bootstrap-once: complete
  commit ba5495f, review clean
  parent verification 380 + 503 passed
  selected on-instance execution remains a documented later blocker
Task 3F per-seed evidence collection and cleanup safety: complete
  commits 2fd79d9 + 29616eb, review clean
  exact brief verification 488 + 537 passed, group 1 fully green
  inherited collect/_s3_head dry-run NameError closed
  committed package dry-run passed
selected evaluate/cleanup stay blocked pending plan §4 sealed evaluation
Plan 4 evaluation foundation and sealing: integrated and review clean
Task 4A provider-aware 100-snapshot evaluation bridge: complete
  commits 8fbae08..9bd99a5, review clean
  parent verification 535 + 209 + 495 + 8 passed
Task 4B receipts-driven StudyLockV3 construction/publication: complete
  commits c964264 + 955ca45 + c92980d, review clean
  exact brief verification 501 + 209 + 495 + 8 passed, package 106 passed
  full suite 2980 passed with only the two pre-existing v2 cohort-release
  failures; committed package dry-run passed
Task 4C receipts-driven cohort aggregation/report: implementation committed
  commits 73e6528 + 181f3ef, review clean
  parent verification 310 + 209 + 495 + 8 passed, package 106 passed
  full suite 3010 passed with only the two inherited v2 failures
Task 4D durable cohort evaluation/report evidence collection: complete
  commits 6caa483 + d5b7ad3, review pending
  exact brief verification 711 + 209 + 522 + 8 passed, package 107 passed
  full suite 3204 passed with only the two inherited v2 failures
  committed package dry-run passed
  cohort collect-cohort publishes one 1012-object receipt; outcomes are
  certified solely by the Task 4C report replay
selected evaluate stays blocked pending Task 4E; cleanup stays blocked
  even with ten per-seed collection states plus a cohort collection state
