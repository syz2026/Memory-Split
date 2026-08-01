"""Generate the run configs for one cohort.

Every arm within a seed must be byte-identical except for `train_mask`. That
is asserted here rather than trusted, because the difference between the arms
IS the experiment: a config key that drifts between them is a confound with no
symptom until the analysis.

Arms:
    sup       no train_mask -- the loader defaults to full supervision
    factmask  train_mask = factmask.bin
    randpos   train_mask = randpos.bin

All three carry `probe_mask = factmask.bin` so the gate-0 masked-value probe
scores the same positions in every arm, which is what makes the dense number
interpretable as a memorisation burden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ARMS = ("sup", "factmask", "randpos")
MASK_FILE = {"sup": None, "factmask": "factmask.bin", "randpos": "randpos.bin"}


def make_config(
    run_id: str,
    arm: str,
    seed: int,
    corpus_rel: str,
    model: str,
    total_tokens: int,
    lr: float,
    grad_clip: float,
    tokens_per_step: int = 524_288,
    micro_batch_size: int = 32,
    ctx: int = 1024,
    n_entities: int = 0,
    corpus_seed: int = 0,
    igsm_mod: int = 23,
    igsm_op: tuple[int, int] = (1, 4),
    igsm_ood_op: tuple[int, int] = (5, 8),
) -> dict:
    cfg = {
        "run_id": run_id,
        "arm": arm,
        "model": model,
        "ctx": ctx,
        "vocab_size": 50304,
        "train_bin": f"{corpus_rel}/targets.bin",
        "probe_mask": f"{corpus_rel}/factmask.bin",
        "micro_batch_size": micro_batch_size,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "lr": lr,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "grad_clip": grad_clip,
        "seed": seed,
        # Corpus metadata, so scripts/run_evals.py can regenerate the exact
        # fact set to probe and the exact difficulty band to score.
        "n_entities": n_entities,
        "corpus_seed": corpus_seed,
        "igsm_mod": igsm_mod,
        "igsm_op": list(igsm_op),
        "igsm_ood_op": list(igsm_ood_op),
        "out_rel": f"runs/{run_id}",
        "device": "cuda",
        "compile": True,
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": 0.25,
        "ckpt_minutes": 30,
    }
    mask = MASK_FILE[arm]
    if mask is not None:
        cfg["train_mask"] = f"{corpus_rel}/{mask}"
    return cfg


def cohort(
    loads: dict[str, str],
    seeds: list[int],
    model: str,
    total_tokens: int,
    lr: float,
    grad_clip: float,
    load_entities: dict[str, int] | None = None,
    corpus_seed: int = 0,
    igsm_mod: int = 23,
    igsm_op: tuple[int, int] = (1, 4),
) -> list[dict]:
    out = []
    for load_name, corpus_rel in loads.items():
        for seed in seeds:
            for arm in ARMS:
                out.append(
                    make_config(
                        run_id=f"{model}_{load_name}_{arm}_s{seed}",
                        arm=arm,
                        seed=seed,
                        corpus_rel=corpus_rel,
                        model=model,
                        total_tokens=total_tokens,
                        lr=lr,
                        grad_clip=grad_clip,
                        n_entities=(load_entities or {}).get(load_name, 0),
                        corpus_seed=corpus_seed,
                        igsm_mod=igsm_mod,
                        igsm_op=igsm_op,
                    )
                )
    return out


IGNORED_KEYS = {"run_id", "arm", "out_rel", "train_mask"}


def assert_arms_match(configs: list[dict]) -> None:
    """Within a (load, seed), the three arms must differ only in train_mask."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for c in configs:
        load = c["run_id"].split("_")[1]
        groups.setdefault((load, c["seed"]), []).append(c)
    for (load, seed), arms in groups.items():
        assert len(arms) == len(ARMS), f"{load} s{seed}: {len(arms)} arms"
        ref = {k: v for k, v in arms[0].items() if k not in IGNORED_KEYS}
        for c in arms[1:]:
            got = {k: v for k, v in c.items() if k not in IGNORED_KEYS}
            diff = {k for k in set(ref) | set(got) if ref.get(k) != got.get(k)}
            assert not diff, (
                f"{load} s{seed}: arms differ in {sorted(diff)} -- the arms "
                "must differ only in train_mask"
            )
        masks = {c.get("train_mask") for c in arms}
        assert len(masks) == len(ARMS), f"{load} s{seed}: duplicate masks {masks}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for the YAMLs")
    ap.add_argument("--model", default="d40m")
    ap.add_argument("--total-tokens", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument(
        "--load", action="append", required=True, metavar="NAME=CORPUS_REL",
        help="repeatable, e.g. --load high=corpora/high --load low=corpora/low",
    )
    ap.add_argument(
        "--entities", action="append", default=[], metavar="NAME=N",
        help="entity count per load, so the storage probe knows what to ask about",
    )
    ap.add_argument("--corpus-seed", type=int, default=0)
    ap.add_argument("--igsm-mod", type=int, default=23)
    ap.add_argument("--igsm-op", type=int, nargs=2, default=(1, 4))
    args = ap.parse_args()

    loads = dict(kv.split("=", 1) for kv in args.load)
    load_entities = {k: int(v) for k, v in
                     (kv.split("=", 1) for kv in args.entities)}
    configs = cohort(loads, args.seeds, args.model, args.total_tokens,
                     args.lr, args.grad_clip, load_entities,
                     args.corpus_seed, args.igsm_mod, tuple(args.igsm_op))
    assert_arms_match(configs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in configs:
        (out / f"{c['run_id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    (out / "cohort.json").write_text(
        json.dumps({"n_runs": len(configs),
                    "runs": [c["run_id"] for c in configs]}, indent=2)
    )
    print(f"wrote {len(configs)} configs to {out}")
    for c in configs:
        print("  ", c["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
