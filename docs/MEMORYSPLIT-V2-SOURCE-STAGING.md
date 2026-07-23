# MemorySplit v2 frozen upstream source staging

This path stages the currently frozen upstream inputs referenced by
`configs/reasoning-dataset-v2.json`. It is deliberately incomplete: generator
and solver locks plus all eight materialized lane streams are still missing.
It does not build the final packed corpus, change the preregistration, fill any
still-unbuilt artifact hash, or permit a protected launch.

## Frozen inputs

`sources/memorysplit-v2/source-set.lock.json` hashes four child locks. Every
downloaded file is bound to an upstream revision, byte count, and real
SHA-256; the objective-source lock additionally inventories every selected
archive member and its license file. Wikidata archive members carry their own
SHA-256s in addition to the enclosing LFS archive hashes.

| Source | Revision/scope | Download bytes |
|---|---|---:|
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, complete `sample-10BT` (14 shards) | 28,518,193,415 |
| FineMath | `e92b25a616738fe95dc186b64dfb19f9c8525594`, all 64 `4plus` then all 128 `3plus` shards | 83,404,381,578 |
| Wikidata5M | `6b2b09672129e280c0c9da97ab58154e9d535e6b`, alias + inductive + transductive archives | 532,955,381 |
| Licensed objective auxiliaries | eight pinned GitHub archives implementing the seven frozen source roles | 54,692,996 |
| Total | | **112,510,223,370 bytes (112.510 GB / 104.783 GiB)** |

The default clean-stage preflight currently requires 135,186,093,314 free
bytes (135.186 GB / 125.902 GiB). That includes extraction and sidecar
allowances, an 8 GiB working reserve, and a 10% safety margin. The dry run
recomputes this value and checks the actual destination filesystem.

FineMath selection is deterministic and preserves the required precedence:

1. hash the exact UTF-8 bytes of every FineWeb-Edu `text` value;
2. consume `finemath-4plus` in locked shard and row order, keeping only the
   first exact text not already seen;
3. consume `finemath-3plus` in locked shard and row order, keeping only its
   cross-deduplicated remainder.

Each FineMath parquet receives a one-byte-per-row `.keep.u8` sidecar (`1` keep,
`0` drop). No Unicode, whitespace, or fuzzy normalization is silently added.

Wikidata uses the locked inductive-training split before the locked
transductive-training split and emits equivalent keep sidecars, so every exact
training triple is selected once. The official test and validation files are
retained byte-for-byte for provenance. They are not assumed to be clean:
staging emits a row-aligned `.eligible.u8` sidecar for each file and excludes
every exact triple present in the complete-once training union from future
sealed evaluation. A canonical exclusion-evidence file binds each excluded row
index and triple digest to the source-file hash.

The frozen revision contains substantial cross-regime contamination. Of 23,889
official sealed-source rows, 23,823 occur in the combined training graph. The
eligible remainder has 66 rows and 62 distinct triples; four rows duplicate
eligible triples across official splits. Staging records these counts rather
than silently changing training coverage or deleting source evidence.

An exact, sorted, fixed-width index commits all 20,624,513 distinct selected
training triples. Verification uses it to recompute every eligibility marker,
requires the eligible subset to have zero training overlap, rejects unjustified
exclusions, and binds the index, sidecars, evidence files, per-split counts, and
hashes into both the selection manifest and source-stage receipt. Evaluation
consumers must use the eligible sidecars and must not read the official files
as unfiltered evaluation sets.

Only training-eligible ARC/ARC-AGI-2 and ConceptARC files are extracted.
Published ProntoQA data/model-output ZIPs, CLRS model-accuracy outputs, and ARC
evaluation tasks are not placed in the objective training source tree.
Generated objective records still require the answer/solver validation
specified by the scientific contract.

## Dry run

The default is read-only:

```bash
python scripts/stage_v2_sources.py \
  --data-root "$DATA_ROOT" \
  --cache-dir "$DATA_ROOT/.cache/memorysplit-v2" \
  --hf-command "$(command -v hf)"
```

The JSON output includes the exact commands, environment, lock digest,
download bytes, remaining bytes, disk requirement, and preflight decision. A
dry run creates neither the data root nor the cache.

`huggingface_hub` 1.x rejects `hf download --local-dir ... --cache-dir ...`.
The emitted command therefore uses exact positional filenames with
`--local-dir` and sets `HF_HOME` in the subprocess environment. It never passes
the incompatible option pair.

## Execute and resume

After reviewing a passing dry run, launch:

```bash
python scripts/stage_v2_sources.py \
  --data-root "$DATA_ROOT" \
  --cache-dir "$DATA_ROOT/.cache/memorysplit-v2" \
  --hf-command "$(command -v hf)" \
  --execute
```

Re-run the same command after interruption. Downloads, per-file deduplication
transactions, and safely extracted trees resume under a private directory
keyed by the source-set lock SHA-256. Publication is one atomic rename and
never replaces an existing destination. `--restart` explicitly discards only
that private partial stage. In particular, recovery from a sealed-evaluation
audit failure must omit `--restart`; the completed FineMath and Wikidata
training transactions remain valid and are reused.

On success, the command prints the immutable receipt path and SHA-256. The
receipt is:

```text
$DATA_ROOT/memorysplit-v2-frozen-upstream-sources/source-stage-receipt.json
```

Verify an existing publication with:

```bash
python scripts/stage_v2_sources.py \
  --data-root "$DATA_ROOT" \
  --verify
```

Verification rejects missing, changed, extra, symlinked, or non-regular files
and rechecks every locked upstream file independently of the receipt. The
published tree includes a self-contained, path-preserving copy of the source
contract under `source-contract/`.
The receipt says `stage_complete: true` only for this locked upstream subset
and keeps `production_corpus_ready: false`, with explicit missing-lock and
missing-lane lists. Follow `sources/memorysplit-v2/README.md` to materialize,
seal, and build the real eight-lane corpus.

## Lock maintenance

Lock regeneration is a deliberate networked maintainer operation:

```bash
python scripts/generate_v2_source_locks.py \
  --output-dir sources/memorysplit-v2 \
  --replace
```

The generator resolves only its hard-coded revisions and derives checksums from
actual Hub LFS metadata, downloaded Wikidata archive members, and downloaded
GitHub archives. `--archive-cache DIR` may supply already-downloaded archives;
each cached byte stream is still checked against upstream metadata. Any diff in
revisions, file inventory, checksums, sizes, or licenses requires review. Never
replace a digest with a guessed value and never copy a digest from another
artifact.
