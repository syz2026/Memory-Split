# Relational experiment work distribution

The four ZIP files are for four collaborators; the coordinator uses the source
worktree directly.

Each archive is self-contained and has a platform-specific closed inventory:
one assignment-specific `README.md`, `assignment.json`, `SHA256SUMS`, the
frozen `relational-run.tar.gz`, and the needed launch/build/verification
helpers.
Each archive also carries `Memory-split-design.md`, the only current design and
dataset specification. The enclosed runnable bundle and assignments are
historical pilot artifacts and must not be used to claim a run on the current
dataset.
`SHA256SUMS` enumerates every payload file except itself. The four outer ZIP
digests are recorded in `artifacts/COLLABORATOR-ZIP-SHA256SUMS`; verify that
file before distribution.

## FarmShare

- Coordinator: seed 0, both 29M gate runs, and the five seed-0 160M runs.
- FarmShare collaborator 1: all five seed-1 160M runs.
- FarmShare collaborator 2: all five seed-2 160M runs.

Each person therefore owns five protected 160M jobs (8.0B raw training
tokens). The coordinator additionally owns the paired 29M gate (0.6B raw
training tokens). Each seed owner builds the `n50k` and `n800k` corpus for that
seed; the coordinator also builds the smaller gate corpus.

## MIT

- MIT collaborator A: Dense seed 0, Split seed 0, and Dense seed 2.
- MIT collaborator B: Dense seed 1, Split seed 1, and Split seed 2.

Each collaborator owns three 360M jobs (10.8B raw training tokens), one intact
Dense/Split seed pair, and one arm of the cross-owned seed-2 pair. To balance
the non-GPU work, A stages the shared FineWeb snapshot and builds the seed-0
corpus; B builds the seed-1 and seed-2 corpora. A owns the combined six-probe
preflight and final artifact sync.

Both MIT kits must use the same reviewed profile bytes, shared `DATA_ROOT`, and
shared `OUT_ROOT`. Full 360M runs remain blocked until all six 200-step probes
pass the central MIT preflight.
