"""Emit the tiny crowding-cohort configs: 2 models x 2 arms, one seed.

Everything that defines the experiment is held fixed against the d160m run --
same corpus, same byte-identical token stream, same two sidecars, the same
524,288-token optimizer batch and therefore the same 15,582 steps and snapshot
schedule. Only the model changes, plus the learning rate, which is tiered by
size and frozen by the probe in Task 7.

Both arms carry the same `probe_mask`, the split90 sidecar, so the gate-0
metric scores both arms on the same offloaded positions. The split arm should
sit near ln(50304) = 10.83; the dense arm's score is the memorisation burden.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CORPUS = Path(os.environ.get("MS_CORPUS", "/mnt/nvme/corpus"))
CODE = Path(os.environ.get("MS_CODE", "/mnt/nvme/code"))
RUNS = Path(os.environ.get("MS_RUNS", "/mnt/nvme/runs/tiny-v3"))
OUT = CODE / "configs" / "tiny-v3"

COHORT = "memorysplit-exploratory-v3-tiny-crowding-n1"
TOTAL_TOKENS = 8_169_455_616
TOKENS_PER_STEP = 524_288
CTX = 1024
MAX_STEPS = TOTAL_TOKENS // TOKENS_PER_STEP
SNAPSHOTS = [int(MAX_STEPS * f + 0.5) for f in (0.10, 0.25, 0.50, 0.75)] + [MAX_STEPS]
SEED = 0

# Frozen by the learning-rate probe (Task 7). Chosen on the DENSE arm only and
# then applied to both, so the choice cannot favour either arm.
LR = {"d8m": 3.0e-3, "d40m": 2.0e-3}

MODELS = {
    "d8m": {"dims": {"n_layer": 7, "n_head": 4, "d_model": 128,
                     "vocab_size": 50304, "tie_embeddings": True},
            "micro_bs": 32},
    "d40m": {"dims": {"n_layer": 12, "n_head": 6, "d_model": 384,
                      "vocab_size": 50304, "tie_embeddings": True},
             "micro_bs": 32},
}

PROVIDER = os.environ.get("MS_PROVIDER", "farmshare-slurm")
OPERATOR = os.environ.get("MS_OPERATOR", "farmshare-operator")
CKPT_MINUTES = int(os.environ.get("MS_CKPT_MINUTES", "10"))

TOKENS = [
    str(CORPUS / "base/packed/targets.bin"),
    str(CORPUS / "extension/packed/targets.bin"),
]
MASKS = {
    "dense": [
        str(CORPUS / "base/sidecars/dense_target_weights.bin"),
        str(CORPUS / "extension/sidecars/shared_target_weights.bin"),
    ],
    "split90": [
        str(CORPUS / "base/sidecars/split90_target_weights.bin"),
        str(CORPUS / "extension/sidecars/shared_target_weights.bin"),
    ],
}
PROBE_MASK = MASKS["split90"]


def mlp_hidden(d_model: int) -> int:
    return ((int(8 * d_model / 3) + 63) // 64) * 64


def param_count(n_layer: int, n_head: int, d_model: int, vocab_size: int,
                tie_embeddings: bool) -> int:
    """Analytic count for the repo's GPT. Recurrence is deliberately not an
    argument: it changes compute and effective depth, never parameters."""
    d_ff = mlp_hidden(d_model)
    embeddings = (1 if tie_embeddings else 2) * vocab_size * d_model
    per_block = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model
    return embeddings + n_layer * per_block + d_model


def cfg_for(model: str, arm: str, seed: int, *, smoke: bool = False) -> dict:
    spec = MODELS[model]
    run_id = f"{model}_{arm}_reasoning_v3_s{seed}"
    micro_bs = spec["micro_bs"]
    accum = TOKENS_PER_STEP // (micro_bs * CTX)
    assert accum * micro_bs * CTX == TOKENS_PER_STEP, f"{model}: microbatch not integral"
    out = RUNS / ("_smoke_" + run_id if smoke else run_id)
    return {
        "schema_version": 1,
        "cohort_id": COHORT,
        "run_id": run_id,
        "pair_id": f"{model}_reasoning_v3_s{seed}",
        "model": model,
        "model_parameters": param_count(**spec["dims"]),
        "arm": arm,
        "seed": seed,
        "operator": OPERATOR,
        "provider": PROVIDER,
        "ctx": CTX,
        "train_bin": TOKENS,
        "train_mask": MASKS[arm],
        "probe_mask": PROBE_MASK,
        "total_tokens": TOTAL_TOKENS,
        "tokens_per_step": TOKENS_PER_STEP,
        "max_steps": 30 if smoke else MAX_STEPS,
        "micro_batch_size": micro_bs,
        "lr": LR[model],
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "out_dir": str(out),
        "log_every": 5 if smoke else 20,
        "eval_every": 10 if smoke else 250,
        "snap_frac": 0.1,
        "ckpt_minutes": 1 if smoke else CKPT_MINUTES,
        "checkpoint_updates": [10, 30] if smoke else SNAPSHOTS,
        "dataset": {
            "contract_id": "memorysplit-reasoning-dataset-v3",
            "complete_dataset": True,
            "packed_targets": TOKENS,
            "target_weights": MASKS[arm],
            "raw_target_tokens": TOTAL_TOKENS,
            "scientific_scope": "successor_exploratory_unpreregistered",
        },
    }


def dump(cfg: dict, path: Path) -> None:
    """Minimal YAML writer: avoids depending on pyyaml being importable here."""
    lines: list[str] = []

    def emit(key: str, val, indent: int = 0) -> None:
        pad = " " * indent
        if isinstance(val, dict):
            lines.append(f"{pad}{key}:")
            for k, v in val.items():
                emit(k, v, indent + 2)
        elif isinstance(val, list):
            lines.append(f"{pad}{key}:")
            for item in val:
                lines.append(f"{pad}- {json.dumps(item) if isinstance(item, str) else item}")
        elif isinstance(val, bool):
            lines.append(f"{pad}{key}: {'true' if val else 'false'}")
        elif isinstance(val, str):
            lines.append(f"{pad}{key}: {json.dumps(val)}")
        else:
            lines.append(f"{pad}{key}: {val}")

    for k, v in cfg.items():
        emit(k, v)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for model in MODELS:
        for arm in ("dense", "split90"):
            p = OUT / f"{model}-{arm}-s{SEED}.yaml"
            dump(cfg_for(model, arm, SEED), p)
            written.append(p.name)
    for arm in ("dense", "split90"):
        dump(cfg_for("d8m", arm, SEED, smoke=True), OUT / f"smoke-d8m-{arm}-s{SEED}.yaml")

    assert MAX_STEPS * TOKENS_PER_STEP == TOTAL_TOKENS, "step budget does not close"
    for p in TOKENS + MASKS["dense"] + MASKS["split90"]:
        fp = Path(p)
        assert fp.is_file() and not fp.is_symlink(), f"bad corpus path: {p}"

    print(f"wrote {len(written)} cohort configs + 2 smoke configs to {OUT}")
    print(f"max_steps={MAX_STEPS}  snapshots={SNAPSHOTS}")
    for model, spec in MODELS.items():
        n = param_count(**spec["dims"])
        accum = TOKENS_PER_STEP // (spec["micro_bs"] * CTX)
        print(f"  {model}: {n:,} params  lr={LR[model]}  mbs={spec['micro_bs']} "
              f"accum={accum}  tokens/param={TOTAL_TOKENS/n:.0f}")


if __name__ == "__main__":
    main()
