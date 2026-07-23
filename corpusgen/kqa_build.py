"""Build equal-token paired continuation shards from prepared KQA artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from corpusgen.kqa_pro import LookupFact, render_fact_doc, render_qa_doc
from corpusgen.records import QUERY_TOKEN_CAP


@dataclass(frozen=True)
class KQAContinuationConfig:
    total_tokens: int
    fact_share: float = 0.70
    seed: int = 42
    min_fact_exposures: int = 2

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not 0 < self.fact_share < 1:
            raise ValueError("fact_share must be strictly between 0 and 1")
        if self.min_fact_exposures < 1:
            raise ValueError("min_fact_exposures must be at least 1")


class _Writer:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._tokens = open(out_dir / "train.bin", "wb")
        self._masks = open(out_dir / "train.mask.bin", "wb")
        self.total = 0
        self.masked = 0
        self.component_tokens = {"facts": 0, "qa": 0}
        self.component_docs = {"facts": 0, "qa": 0}

    def add(
        self,
        component: str,
        token_ids: np.ndarray,
        loss_mask: np.ndarray | None,
    ) -> None:
        token_ids.tofile(self._tokens)
        if loss_mask is None:
            np.ones(len(token_ids), dtype=np.uint8).tofile(self._masks)
        else:
            loss_mask.tofile(self._masks)
            self.masked += len(loss_mask) - int(loss_mask.sum())
        self.total += len(token_ids)
        self.component_tokens[component] += len(token_ids)
        self.component_docs[component] += 1

    def close(self) -> None:
        self._tokens.close()
        self._masks.close()


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _encode(tok, segments) -> tuple[np.ndarray, np.ndarray | None]:
    ids, mask = tok.encode_segments(segments, add_eot=True)
    token_ids = np.asarray(ids, dtype=np.uint16)
    if 0 not in mask:
        return token_ids, None
    return token_ids, np.asarray(mask, dtype=np.uint8)


def _cycle_order(size: int, seed: int) -> list[int]:
    if size == 0:
        raise ValueError("cannot build a continuation stream from zero documents")
    order = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return order


def _copy_eval_artifacts(prepared_dir: Path, out_dir: Path) -> None:
    for filename in (
        "organizer.jsonl",
        "eval_transfer.jsonl",
        "eval_dev.jsonl",
        "recall_transfer.jsonl",
        "qa_train.jsonl",
        "qa_dev.jsonl",
        "qa_test.jsonl",
        "facts.jsonl",
    ):
        source = prepared_dir / filename
        if source.exists():
            shutil.copy2(source, out_dir / filename)
    report = prepared_dir / "report.json"
    if report.exists():
        shutil.copy2(report, out_dir / "prepared_report.json")


def _data_fingerprint(prepared_dir: Path, cfg: KQAContinuationConfig) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    for filename in (
        "facts.jsonl",
        "qa_train.jsonl",
        "qa_dev.jsonl",
        "qa_test.jsonl",
    ):
        digest.update(filename.encode())
        with open(prepared_dir / filename, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_kqa_continuation(
    prepared_dir: str | Path,
    tok,
    out_dir: str | Path,
    cfg: KQAContinuationConfig,
) -> dict:
    prepared_dir = Path(prepared_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fact_rows = _read_jsonl(prepared_dir / "facts.jsonl")
    qa_rows = _read_jsonl(prepared_dir / "qa_train.jsonl")
    if not fact_rows:
        raise ValueError("prepared facts.jsonl is empty")
    if not qa_rows:
        raise ValueError("prepared qa_train.jsonl is empty")

    facts = [
        (
            LookupFact(
                key=row["key"],
                name=row["name"],
                relation=row["relation"],
                value=row["value"],
                kind=row["kind"],
                value_count=row.get("value_count", 1),
                raw_ids=tuple(row.get("raw_ids", ())),
            ),
            row["bucket"],
        )
        for row in fact_rows
    ]
    fact_by_key = {fact.key: fact for fact, _ in facts}
    test_rows = _read_jsonl(prepared_dir / "qa_test.jsonl")
    required_transfer_keys = {
        key for row in test_rows for key in row.get("support_keys", ())
    }
    oversized_queries = sorted(
        key
        for key in required_transfer_keys
        if key in fact_by_key
        and len(
            tok.encode(
                f"{fact_by_key[key].name}, {fact_by_key[key].relation}"
            )
        )
        >= QUERY_TOKEN_CAP
    )
    if oversized_queries:
        examples = ", ".join(repr(key) for key in oversized_queries[:3])
        raise ValueError(
            f"{len(oversized_queries)} transfer lookup queries reach the native "
            f"{QUERY_TOKEN_CAP}-token cap (examples: {examples}). Rebuild the "
            "prepared split with shorter aliases/keys or intentionally raise "
            "the model's query cap."
        )

    fact_budget = int(round(cfg.total_tokens * cfg.fact_share))
    budgets = {"facts": fact_budget, "qa": cfg.total_tokens - fact_budget}
    report_arms: dict[str, dict] = {}
    fact_order = _cycle_order(len(facts), cfg.seed * 2 + 1)
    qa_order = _cycle_order(len(qa_rows), cfg.seed * 2 + 2)

    for arm in ("dense", "split"):
        print(f"encoding KQA {arm} continuation documents...", flush=True)
        encoded_facts = []
        for fact, bucket in facts:
            doc = render_fact_doc(fact, bucket)
            segments = doc.dense_segments if arm == "dense" else doc.split_segments
            encoded_facts.append(_encode(tok, segments))
        encoded_qa = []
        for row in qa_rows:
            doc = render_qa_doc(row)
            segments = doc.dense_segments if arm == "dense" else doc.split_segments
            encoded_qa.append(_encode(tok, segments))

        one_fact_pass = sum(len(encoded_facts[index][0]) for index in fact_order)
        required_fact_tokens = one_fact_pass * cfg.min_fact_exposures
        if budgets["facts"] < required_fact_tokens:
            raise ValueError(
                f"{arm} fact budget ({budgets['facts']:,}) cannot expose every fact "
                f"{cfg.min_fact_exposures} time(s); needs at least "
                f"{required_fact_tokens:,} tokens. Increase --total-tokens or "
                "--fact-share, reduce the prepared KB, or lower repetitions."
            )

        writer = _Writer(out_dir / arm)
        fact_bucket_docs: Counter[str] = Counter()
        positions = {"facts": 0, "qa": 0}
        orders = {"facts": fact_order, "qa": qa_order}
        encoded = {"facts": encoded_facts, "qa": encoded_qa}
        emitted = {"facts": 0, "qa": 0}
        active = {"facts", "qa"}
        while active:
            deficits = {
                component: (budgets[component] - emitted[component])
                / max(1, budgets[component])
                for component in active
            }
            component = max(sorted(deficits), key=deficits.get)
            if emitted[component] >= budgets[component]:
                active.remove(component)
                continue
            order = orders[component]
            index = order[positions[component] % len(order)]
            positions[component] += 1
            ids, mask = encoded[component][index]
            if component == "facts":
                fact_bucket_docs[facts[index][1]] += 1
            if arm == "dense":
                mask = None
            writer.add(component, ids, mask)
            emitted[component] += len(ids)
        writer.close()

        full_fact_passes = writer.component_docs["facts"] // len(facts)
        arm_dir = out_dir / arm
        report_arms[arm] = {
            "total_tokens": writer.total,
            "component_tokens": writer.component_tokens,
            "component_docs": writer.component_docs,
            "masked_tokens": writer.masked,
            "masked_fraction": writer.masked / writer.total,
            "fact_count": len(facts),
            "qa_count": len(qa_rows),
            "fact_tokens_one_pass": one_fact_pass,
            "full_fact_exposures": full_fact_passes,
            "fact_bucket_docs": dict(fact_bucket_docs),
            "fact_bucket_exposures": {
                bucket: fact_bucket_docs[bucket]
                / sum(1 for _, fact_bucket in facts if fact_bucket == bucket)
                for bucket in sorted({fact_bucket for _, fact_bucket in facts})
            },
            "artifact_sha256": {
                "train_bin": _sha256_file(arm_dir / "train.bin"),
                "train_mask": _sha256_file(arm_dir / "train.mask.bin"),
            },
        }
        del encoded_facts, encoded_qa

    _copy_eval_artifacts(prepared_dir, out_dir)
    total_difference = abs(
        report_arms["dense"]["total_tokens"] - report_arms["split"]["total_tokens"]
    )
    prepared_fingerprint = _data_fingerprint(prepared_dir, cfg)
    fingerprint_digest = hashlib.sha256(prepared_fingerprint.encode())
    fingerprint_digest.update(
        json.dumps(
            {
                arm: report_arms[arm]["artifact_sha256"]
                for arm in ("dense", "split")
            },
            sort_keys=True,
        ).encode()
    )
    report = {
        "config": asdict(cfg),
        "prepared_fingerprint": prepared_fingerprint,
        "data_fingerprint": fingerprint_digest.hexdigest(),
        "budgets": budgets,
        "arms": report_arms,
        "checks": {
            "equal_total_token_target": total_difference
            <= max(
                256,
                0.01
                * max(
                    report_arms["dense"]["total_tokens"],
                    report_arms["split"]["total_tokens"],
                ),
            ),
            "all_facts_exposed": all(
                arm["full_fact_exposures"] >= cfg.min_fact_exposures
                for arm in report_arms.values()
            ),
            "dense_has_no_masked_tokens": report_arms["dense"]["masked_tokens"] == 0,
            "split_has_masked_fact_values": report_arms["split"]["masked_tokens"] > 0,
            "transfer_queries_fit_cap": not oversized_queries,
        },
    }
    with open(out_dir / "continuation_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    return report
