#!/usr/bin/env python
"""Run the packaged current smoke bytes through training and memory modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from corpusgen.current_dataset import verify_current_dataset
from evals.relational_generate import decode_items
from organizer.graph_store import AtomicGraphStore
from train.safeio import read_regular_path
from train.tokenizer import get_tok
from train.trainer import Trainer


SMOKE_FIXTURE = {
    "n_entities": 32,
    "total_tokens": 40_000,
    "data_seed": 1,
    "world_size": 32,
    "eval_pairs_per_task": 4,
    "eval_pairs_per_world": 4,
    "route_stats_pairs_per_task": 64,
    "guardrail_items": 4,
    "shared_text_eval_count": 4,
}
SMOKE_STEPS = 2
_MODEL = {
    "n_layer": 1,
    "n_head": 1,
    "d_model": 8,
    "ctx": 320,
    "vocab_size": 50_304,
}
_DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "current-smoke"


def _trainer_config(
    root: Path,
    corpus: Path,
    arm: str,
    *,
    steps: int,
    device: str,
) -> dict:
    return {
        "condition": arm,
        "model": dict(_MODEL),
        "train_bin": str(corpus / "train.bin"),
        "train_weights": str(corpus / f"{arm}.weights.bin"),
        "micro_batch_size": 1,
        "tokens_per_step": _MODEL["ctx"],
        "max_steps": steps + 1,
        "lr": 1e-3,
        "warmup_steps": 1,
        "seed": 19,
        "device": device,
        "out_dir": str(root / "runs" / arm),
        "log_every": 1,
        "eval_every": 1,
        "snap_frac": 1.0,
        "ckpt_minutes": 999,
    }


def _same_state(left, right) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def _resume_is_exact(trainer: Trainer, root: Path) -> bool:
    resumed_cfg = dict(
        trainer.cfg,
        out_dir=str(root / "runs" / "resume-check"),
    )
    resumed = Trainer(resumed_cfg)
    checkpoint = read_regular_path(
        trainer.ckpt_path,
        label="smoke resume checkpoint",
    )
    resumed.load_ckpt(trainer.ckpt_path, sha256=checkpoint.sha256)
    if (
        resumed.step != trainer.step
        or resumed.data.state_dict() != trainer.data.state_dict()
        or not _same_state(trainer.model, resumed.model)
    ):
        return False

    expected_batch = trainer.data.next_weighted_batch()
    resumed_batch = resumed.data.next_weighted_batch()
    if not all(
        torch.equal(expected, actual)
        for expected, actual in zip(expected_batch, resumed_batch)
    ):
        return False

    with torch.no_grad():
        _, expected_loss = trainer.model(
            expected_batch[0],
            expected_batch[1],
            target_weights=expected_batch[2],
        )
        _, resumed_loss = resumed.model(
            resumed_batch[0],
            resumed_batch[1],
            target_weights=resumed_batch[2],
        )
    return torch.equal(expected_loss, resumed_loss)


def _evaluate_modes(
    root: Path,
    corpus: Path,
    trainer: Trainer,
    tok,
) -> list[str]:
    items = [
        json.loads(line)
        for line in (corpus / "eval" / "items.jsonl").read_text().splitlines()
        if line
    ]
    if not items:
        raise ValueError("packaged current smoke fixture has no eval items")
    base_store = AtomicGraphStore.load(corpus / "eval" / "graph.jsonl")
    modes = []
    trainer.model.eval()
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            trainer.model,
            tok,
            items,
            base_store if memory_on else None,
            device="cpu",
            batch_size=8,
        )
        rows = [
            {
                "qid": item["qid"],
                "memory": memory,
                "actions": [
                    [
                        action.source_slot,
                        action.relation_id,
                        action.direction,
                        action.page,
                        action.read,
                        action.halt,
                    ]
                    for action in state.actions
                ],
                "misses": state.misses,
                "halt_step": state.halt_step,
                "n_steps": len(state.actions),
                "prediction": state.provisional_answers[-1],
            }
            for item, state in zip(items, states)
        ]
        mode_dir = root / "evals" / f"memory_{memory}"
        mode_dir.mkdir(parents=True)
        (mode_dir / "rows.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        )
        summary = {
            "memory": memory,
            "n_items": len(rows),
            "action_slots": 12,
            "all_items_complete": all(row["n_steps"] == 12 for row in rows),
        }
        (mode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        modes.append(memory)
    return modes


def _fixture_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _resume_one_step(trainer: Trainer, root: Path) -> int:
    resumed_cfg = dict(
        trainer.cfg,
        out_dir=str(root / "runs" / "resume-one-step"),
    )
    resumed = Trainer(resumed_cfg)
    checkpoint = read_regular_path(
        trainer.ckpt_path,
        label="smoke resume checkpoint",
    )
    resumed.load_ckpt(trainer.ckpt_path, sha256=checkpoint.sha256)
    resumed.train_steps(1)
    return resumed.step


def run_smoke(
    out_dir: Path | str,
    *,
    fixture: Path | str = _DEFAULT_FIXTURE,
    steps: int = SMOKE_STEPS,
    device: str = "cpu",
) -> dict:
    """Train and evaluate directly from immutable packaged current bytes."""

    if steps != SMOKE_STEPS:
        raise ValueError("the local smoke contract requires exactly two steps")
    if device != "cpu":
        raise ValueError("the local smoke contract requires device='cpu'")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"smoke output directory must be empty: {root}")

    corpus = Path(fixture).resolve()
    build_report = verify_current_dataset(corpus, "smoke")
    fixture_before = _fixture_hashes(corpus)
    tok = get_tok()
    if not all(build_report["checks"].values()):
        raise AssertionError("packaged current smoke corpus failed build checks")

    dense = Trainer(
        _trainer_config(root, corpus, "dense", steps=steps, device=device)
    )
    split = Trainer(
        _trainer_config(root, corpus, "split", steps=steps, device=device)
    )
    initial_state = {
        name: value.detach().clone()
        for name, value in dense.model.state_dict().items()
    }
    dense.model.load_state_dict(initial_state)
    split.model.load_state_dict(initial_state)
    if not _same_state(dense.model, split.model):
        raise AssertionError("Dense and Split initial states differ")

    dense.train_steps(steps)
    split.train_steps(steps)
    resume_exact = _resume_is_exact(dense, root)
    resume_step = _resume_one_step(dense, root)
    modes = _evaluate_modes(root, corpus, split, tok)
    pairs_complete = all(
        json.loads(
            (root / "evals" / f"memory_{mode}" / "summary.json").read_text()
        )["all_items_complete"]
        for mode in modes
    )
    verify_current_dataset(corpus, "smoke")
    fixture_unchanged = _fixture_hashes(corpus) == fixture_before

    report = {
        "shared_stream": (
            dense.cfg["train_bin"] == split.cfg["train_bin"]
            and dense.data.n_tokens == split.data.n_tokens
        ),
        "dense_steps": dense.step,
        "split_steps": split.step,
        "resume_exact": resume_exact,
        "resume_step": resume_step,
        "memory_modes": modes,
        "pairs_complete": pairs_complete,
        "fixture_unchanged": fixture_unchanged,
        "profile": build_report["profile"],
        "scientific_result": build_report["scientific_result"],
    }
    if not (
        report["shared_stream"]
        and report["dense_steps"] == steps
        and report["split_steps"] == steps
        and report["resume_exact"]
        and report["resume_step"] == steps + 1
        and report["memory_modes"] == ["off", "on"]
        and report["pairs_complete"]
        and report["fixture_unchanged"]
        and report["profile"] == "smoke"
        and report["scientific_result"] is False
    ):
        raise AssertionError(f"local relational smoke failed: {report}")
    (root / "smoke-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run two-step CPU smoke directly from packaged current bytes."
    )
    parser.add_argument("--out", default="outputs/relational-smoke")
    parser.add_argument("--fixture", default=str(_DEFAULT_FIXTURE))
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)
    report = run_smoke(args.out, fixture=args.fixture, device=args.device)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
