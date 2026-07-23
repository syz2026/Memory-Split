# MemorySplit v2 production source staging

The checked lock files freeze the upstream FineWeb-Edu, FineMath, Wikidata5M,
and objective-auxiliary inputs. The checked source-set lock remains
intentionally marked `production_readiness.ready: false`: it describes only
the upstream stage, not a completed production corpus. The lane materializer
creates the missing streams and derives generator/solver receipts from the
actual repository code and runtime used for that run.

Stage that frozen upstream subset with:

```bash
scripts/stage_v2_sources.py --data-root "$DATA_ROOT" --execute
```

Its receipt remains `production_corpus_ready: false`; this command does not
materialize any of the eight final lanes. See
`docs/MEMORYSPLIT-V2-SOURCE-STAGING.md` for its disk and resume contract.

## Materialize the eight lanes

After source staging succeeds, run:

```bash
python3 scripts/materialize_memorysplit_v2.py materialize \
  --stage-root "$DATA_ROOT/memorysplit-v2-frozen-upstream-sources" \
  --source-root /data/memorysplit-v2/source \
  --work-root /scratch/memorysplit-v2-materialize \
  --objective-python /opt/memorysplit-v2-objective/bin/python
```

The work root is resumable and may be much larger than memory. Lane tokens,
verification rows, and factual occurrences are checkpointed; Wikidata,
routing, uniqueness checks, and preflight verification use file-backed
SQLite. Re-run the identical command after interruption. A changed source
inventory, recipe, compiler artifact, runtime, or compiler option is rejected
instead of being mixed into an existing checkpoint.

The objective interpreter must contain the dependencies required by all five
pinned procedural generators. Before any lane is written, the command starts
each implementation from its staged source tree, generates ordinals 0 through
127 in both forward and reverse order in isolated workers, and records exact
runtime versions and probe hashes. Missing, incompatible, or order-dependent
primitives are reported by provider; no fallback generator is used.

Monitor a running or interrupted build without opening its large artifacts:

```bash
python3 scripts/materialize_memorysplit_v2.py status \
  --work-root /scratch/memorysplit-v2-materialize
```

For full-scale preflight, point SQLite temporary storage at scratch:

```bash
export SQLITE_TMPDIR=/scratch/memorysplit-v2-preflight
export TMPDIR="$SQLITE_TMPDIR"
mkdir -p "$SQLITE_TMPDIR"
```

Exercise the same 45-artifact publication, routing, solver replay, sealing, and
preflight path without claiming a scientific corpus:

```bash
python3 scripts/materialize_memorysplit_v2.py smoke \
  --source-root /tmp/memorysplit-v2-smoke-source \
  --work-root /tmp/memorysplit-v2-smoke-work
```

`source-production` never downloads, generates, repeats, or substitutes data.
It only validates a complete external source root and seals its hashes into
`source-manifest.json`.

## External source-root contract

The source root must contain:

```text
materialized/<lane>.tokens.bin
materialized/<lane>.split90.weights.bin
ledgers/<lane>.routes.jsonl
ledgers/<lane>.masks.jsonl
ledgers/<reasoning-or-objective-lane>.verification.jsonl
locks/<source-id>.lock.json
```

Run the preflight command to validate the complete path list:

```bash
python3 scripts/build_parallel_corpus.py preflight-production \
  --source-root /data/memorysplit-v2/source
```

Token files are little-endian `uint16` streams and must exactly meet the eight
frozen quotas totaling `7,120,879,616` tokens. Split90 files contain one binary
`uint8` weight per token. Route rows have `fact_id`, `external`, and exact
rational `burden_bits`; mask rows have `fact_id`, `start`, and `end`.
Canonical JSONL is compact, key-sorted, UTF-8, and newline-terminated.

Reasoning and objective verification rows cover each corresponding token file
exactly once in source order:

```json
{"record_id":"...","source_id":"...","token_end":2,"token_start":0,"verification":{}}
```

Reasoning rows must replay through the repository solver and may not repeat a
record ID. Objective rows must carry either a replayable solver proof or a
canonical exact-match answer. Reasoning cycle-fill is always rejected.

The required lock IDs are reported by preflight. The four checked upstream
locks can be installed under their contract names as follows; generator and
solver locks must come from the actual materialization run:

```text
fineweb-edu.lock.json              -> locks/fineweb_edu.lock.json
finemath.lock.json                 -> locks/finemath.lock.json
wikidata5m-complete-once.lock.json -> locks/wikidata5m.lock.json
objective-auxiliaries.lock.json    -> locks/objective_auxiliary.lock.json
```

## Seal and build

After every required artifact exists, this executable command seals the source
root, builds the corpus, and verifies exact quotas plus both v2 sidecars:

```bash
scripts/build_memorysplit_v2_production.sh \
  /data/memorysplit-v2/source \
  /data/memorysplit-v2/corpus \
  /scratch/memorysplit-v2-work \
  --workers 32 --shards 32
```

Any missing or drifting input stops before publication and emits a JSON
preflight report with a repair action. The command does not change the
preregistration or claim that absent artifacts are complete.
