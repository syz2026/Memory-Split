#!/usr/bin/env python
"""Evaluate a frozen checkpoint on robustness-only Wikidata5M artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.records import QAItem
from corpusgen.relation_schema import RelationSchema
from corpusgen.wikidata5m import DEFAULT_WIKIDATA_LOCK_PATH, WikidataLock
from corpusgen.wikidata_paths import (
    CoverageManifest,
    private_temporary_directory,
)
from corpusgen.wikidata_robustness_source import wikidata_lock_sha256
from evals.checkpoint_binding import (
    checkpoint_sha256,
    require_claim_bearing_checkpoint,
    resolve_run_checkpoint,
    verify_checkpoint_config,
    verify_checkpoint_unchanged,
)
from evals.relational_generate import decode_items
from evals.scorers import normalize_answer
from evals.wikidata_metrics import compute_wikidata_metrics
from organizer.packed_graph_store import PackedGraphStore
from scripts.run_relational_evals import _states_to_rows, store_for_item
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluator_path_key(path: Path) -> str:
    root = Path(__file__).resolve().parents[1]
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _evaluator_sha256(paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError("at least one evaluator file is required")
    hashes = sorted(
        (_evaluator_path_key(path), _sha256_file(path))
        for path in paths
    )
    if len(hashes) == 1:
        return hashes[0][1]
    payload = json.dumps(
        hashes,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def default_evaluator_files() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "corpusgen/graph_records.py",
        "corpusgen/graph_trace.py",
        "corpusgen/mask_ledger.py",
        "corpusgen/payload_inventory.py",
        "corpusgen/records.py",
        "corpusgen/relation_codec.py",
        "corpusgen/relation_schema.py",
        "corpusgen/srgm_worlds.py",
        "corpusgen/wikidata5m.py",
        "corpusgen/wikidata_path_replay.py",
        "corpusgen/wikidata_paths.py",
        "corpusgen/wikidata_robustness_source.py",
        "corpusgen/world_splits.py",
        "evals/checkpoint_binding.py",
        "evals/generate.py",
        "evals/relational_contracts.py",
        "evals/relational_controls.py",
        "evals/relational_design.py",
        "evals/relational_generate.py",
        "evals/relational_gates.py",
        "evals/relational_metrics.py",
        "evals/relational_pairing.py",
        "evals/relational_stats.py",
        "evals/scorers.py",
        "evals/wikidata_metrics.py",
        "experiment/artifacts.py",
        "experiment/ledger.py",
        "experiment/provenance.py",
        "experiment/relational_assets.py",
        "organizer/graph_store.py",
        "organizer/packed_graph_store.py",
        "scripts/freeze_relational_study.py",
        "scripts/make_relational_manifest.py",
        "scripts/run_relational_evals.py",
        "scripts/run_wikidata_evals.py",
        "schemas/relational-result-v1.schema.json",
        "train/data.py",
        "train/model.py",
        "train/tokenizer.py",
        "train/trainer.py",
    )
    return tuple(
        root / relative
        for relative in relative_paths
    )


def build_eval_provenance(
    *,
    checkpoint: str | Path,
    source_lock: str | Path,
    schema: str | Path,
    evaluator_files: Sequence[str | Path],
    preregistration: str | Path,
) -> dict[str, Any]:
    evaluator_paths = tuple(Path(path) for path in evaluator_files)
    evaluator_keys = [
        _evaluator_path_key(path) for path in evaluator_paths
    ]
    if len(evaluator_keys) != len(set(evaluator_keys)):
        raise ValueError("evaluator file paths must be unique")
    checkpoint_hash = checkpoint_sha256(checkpoint)
    return {
        "version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "source_lock_sha256": _sha256_file(source_lock),
        "schema_sha256": _sha256_file(schema),
        "evaluator_sha256": _evaluator_sha256(evaluator_paths),
        "preregistration_sha256": _sha256_file(preregistration),
        "evaluator_files": {
            _evaluator_path_key(path): _sha256_file(path)
            for path in sorted(evaluator_paths, key=_evaluator_path_key)
        },
        "evaluation_only": True,
        "optimizer_loaded": False,
        "optimizer_steps": 0,
        "training_examples": 0,
        "fine_tuning_performed": False,
        "confirmatory_verdict_eligible": False,
        "analysis_role": "robustness_only",
    }


def _canonical_json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError(f"Wikidata evaluation artifact is empty: {path}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Wikidata JSONL rows must be objects: {path}")
    return rows


def _verify_artifacts(root: Path, manifest: CoverageManifest) -> None:
    for relative, expected in manifest.artifacts.items():
        path = root / relative
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Wikidata artifact hash mismatch: {relative}"
            )
    forbidden = (
        list(root.glob("**/train.bin"))
        + list(root.glob("**/*.weights.bin"))
        + list(root.glob("**/*optimizer*"))
        + list(root.glob("**/*checkpoint*"))
    )
    if forbidden:
        raise ValueError("Wikidata robustness artifacts contain training output")


def _load_items(
    root: Path,
    manifest: CoverageManifest,
) -> list[QAItem]:
    originals = _read_jsonl(root / "eval" / "original.jsonl")
    counterfactuals = _read_jsonl(
        root / "eval" / "counterfactual.jsonl"
    )
    if len(originals) != manifest.pair_count:
        raise ValueError("original item count does not match coverage manifest")
    if len(counterfactuals) != manifest.pair_count:
        raise ValueError(
            "counterfactual item count does not match coverage manifest"
        )
    rows = originals + counterfactuals
    if len(rows) != manifest.item_count:
        raise ValueError("item count does not match coverage manifest")
    qids = [str(row.get("qid")) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("Wikidata evaluation Q IDs must be unique")
    for row in originals:
        meta = row.get("meta")
        if not isinstance(meta, dict) or meta.get("variant") != "original":
            raise ValueError("original artifact contains a non-original item")
        if meta.get("changed_row") is not None:
            raise ValueError("original item cannot contain an overlay row")
    for row in counterfactuals:
        meta = row.get("meta")
        if (
            not isinstance(meta, dict)
            or meta.get("variant") != "counterfactual"
            or not isinstance(meta.get("changed_row"), dict)
            or meta.get("counterfactual_changed_rows") != 1
        ):
            raise ValueError(
                "counterfactual item must replace exactly one graph row"
            )
    return [QAItem(**row) for row in rows]


def _load_model(
    run: Path,
    checkpoint: Path,
    state: Mapping[str, Any],
    device: str,
) -> tuple[GPT, dict[str, Any], dict[str, Any]]:
    cfg, identities = verify_checkpoint_config(run, state)
    model_value = cfg.get("model")
    if isinstance(model_value, str):
        if model_value not in PRESETS:
            raise ValueError(f"unknown model preset: {model_value}")
        model_cfg = replace(PRESETS[model_value])
    elif isinstance(model_value, dict):
        model_cfg = GPTConfig(**model_value)
    else:
        raise ValueError("model config must be a preset name or mapping")
    if "ctx" in cfg:
        model_cfg.ctx = int(cfg["ctx"])
    model = GPT(model_cfg)
    try:
        model.load_state_dict(state["model"])
    except Exception as exc:
        raise ValueError(
            f"checkpoint model weights do not match run config: {checkpoint}"
        ) from exc
    model.to(device).eval()
    return model, cfg, identities


def _decode_mode(
    *,
    model,
    tok,
    items: Sequence[QAItem],
    store: PackedGraphStore,
    memory: str,
    device: str,
    batch_size: int,
    schema: RelationSchema,
) -> list[dict[str, Any]]:
    memory_on = memory == "on"
    states = decode_items(
        model,
        tok,
        items,
        lambda item: store_for_item(
            store,
            item,
            memory_on=memory_on,
        ),
        device=device,
        batch_size=batch_size,
        codec=schema.codec,
    )
    rows = _states_to_rows(items, states)
    for row in rows:
        row["memory"] = memory
        row["abstained"] = normalize_answer(str(row["pred"])) == "abstain"
    return rows


def run_wikidata_evaluation(
    *,
    run: str | Path,
    checkpoint: str | Path,
    artifacts: str | Path,
    relation_schema: str | Path,
    preregistration: str | Path,
    out_dir: str | Path,
    device: str = "auto",
    batch_size: int = 32,
) -> dict[str, Any]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be positive")
    run_path = Path(run)
    checkpoint_path = resolve_run_checkpoint(run_path, checkpoint)
    artifact_root = Path(artifacts)
    schema_path = Path(relation_schema)
    preregistration_path = Path(preregistration)
    source_lock_path = DEFAULT_WIKIDATA_LOCK_PATH
    destination = Path(out_dir)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Wikidata evaluation output already exists: {destination}"
        )

    source_lock_value = WikidataLock.from_path(source_lock_path)
    schema = RelationSchema.from_path(schema_path)
    manifest_path = artifact_root / "coverage-manifest.json"
    manifest = CoverageManifest.from_path(manifest_path, schema=schema)
    if not manifest.production_evaluation_eligible:
        raise ValueError(
            "non-production fixture artifacts cannot enter production evaluation"
        )
    canonical_source_lock_hash = wikidata_lock_sha256(source_lock_value)
    if manifest.source_lock_sha256 != canonical_source_lock_hash:
        raise ValueError("coverage manifest source-lock hash mismatch")
    expected_archive_hashes = {
        name: item.sha256
        for name, item in source_lock_value.files.items()
    }
    if dict(manifest.source_archive_sha256) != expected_archive_hashes:
        raise ValueError("coverage manifest source archive hashes mismatch")
    if manifest.schema_sha256 != schema.sha256():
        raise ValueError("coverage manifest schema hash mismatch")
    if manifest.recomputed_schema_sha256 != schema.sha256():
        raise ValueError("coverage manifest recomputed schema hash mismatch")
    _verify_artifacts(artifact_root, manifest)
    items = _load_items(artifact_root, manifest)

    checkpoint_before = checkpoint_sha256(checkpoint_path)
    state = require_claim_bearing_checkpoint(checkpoint_path)
    resolved_device = pick_device(device)
    model, cfg, config_identities = _load_model(
        run_path,
        checkpoint_path,
        state,
        resolved_device,
    )
    evaluator_paths = default_evaluator_files()
    provenance = build_eval_provenance(
        checkpoint=checkpoint_path,
        source_lock=source_lock_path,
        schema=schema_path,
        evaluator_files=evaluator_paths,
        preregistration=preregistration_path,
    )
    provenance["checkpoint_config_identities"] = config_identities
    provenance.update(
        {
            "source_lock_canonical_sha256": canonical_source_lock_hash,
            "source_archive_sha256": expected_archive_hashes,
            "artifact_source_sha256": manifest.source_sha256,
            "recomputed_schema_sha256": (
                manifest.recomputed_schema_sha256
            ),
        }
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with private_temporary_directory(
        destination.parent,
        f".{destination.name}.stage-",
    ) as staging:
        with closing(
            PackedGraphStore.load(
                artifact_root / "graph.store",
                schema.codec,
            )
        ) as store:
            all_rows = []
            for memory in ("off", "on"):
                rows = _decode_mode(
                    model=model,
                    tok=get_tok(),
                    items=items,
                    store=store,
                    memory=memory,
                    device=resolved_device,
                    batch_size=batch_size,
                    schema=schema,
                )
                all_rows.extend(rows)
                mode_dir = staging / f"store_{memory}"
                mode_dir.mkdir()
                (mode_dir / "rows.jsonl").write_text(
                    "".join(_canonical_json_line(row) for row in rows),
                    encoding="utf-8",
                )

            metrics = compute_wikidata_metrics(all_rows, manifest)
            integrity = verify_checkpoint_unchanged(
                checkpoint_path,
                checkpoint_before,
            )
            provenance.update(integrity)
            result = {
                "version": 1,
                "analysis_role": "robustness_only",
                "confirmatory_verdict_eligible": False,
                "condition": cfg["condition"],
                "model": cfg["model"],
                "seed": cfg["seed"],
                "coverage_manifest_sha256": _sha256_file(manifest_path),
                "metrics": metrics,
                "provenance": provenance,
            }
            (staging / "wikidata-results.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Wikidata evaluation output already exists: {destination}"
            )
        os.replace(staging, destination)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation-only Wikidata5M robustness transfer. "
            "This command never fine-tunes or writes a checkpoint."
        )
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--relation-schema", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--out")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    run_path = Path(args.run)
    checkpoint_path = resolve_run_checkpoint(run_path, args.checkpoint)
    checkpoint_hash = checkpoint_sha256(checkpoint_path)
    output = (
        Path(args.out)
        if args.out is not None
        else run_path / "evals" / "wikidata" / checkpoint_hash
    )
    result = run_wikidata_evaluation(
        run=run_path,
        checkpoint=args.checkpoint,
        artifacts=args.artifacts,
        relation_schema=args.relation_schema,
        preregistration=args.preregistration,
        out_dir=output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
