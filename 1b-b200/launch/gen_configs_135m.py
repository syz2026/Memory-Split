#!/usr/bin/env python3
"""Rewrite the 135m-v3 configs for this capacity-block node.

The configs shipped in the repo were authored for the parallelcluster
environment. Three things make them unusable here:

  corpus paths   `dataset/base/packed/targets.bin` is relative to the process
                 CWD and there is no `dataset/` under /mnt/nvme/code. The
                 corpus lives at /mnt/nvme/corpus.
  out_dir        `outputs/135m-v3/<run_id>` resolves outside /mnt/nvme/runs,
                 which is the only tree pushed to S3. Runs would train to
                 completion and then die with the instance store.
  micro_batch    8, against the 64 the 817,434 tok/s bench was measured at.
                 tokens_per_step is pinned at 524288 either way, so this only
                 changes the accumulation factor (64 -> 8), not the batch the
                 optimizer sees.

Fails closed: if any rewritten path does not exist on disk, nothing is written.
Originals are copied to configs/135m-v3.orig on first run.
"""
import os
import shutil
import sys

import yaml

CFG_DIR = "/mnt/nvme/code/configs/135m-v3"
BACKUP = "/mnt/nvme/code/configs/135m-v3.orig"
CORPUS = "/mnt/nvme/corpus"
RUNS = "/mnt/nvme/runs/135m-v3"
POINTER = "/mnt/nvme/code/DATASET-POINTER-AWS-135M-V3.json"
MICRO_BATCH = 64
EXPECT_CONFIGS = 20


def to_corpus(path):
    if path.startswith("/"):
        return path
    if path.startswith("dataset/"):
        return os.path.join(CORPUS, path[len("dataset/"):])
    raise ValueError(f"unrecognised corpus path: {path}")


def main():
    names = sorted(n for n in os.listdir(CFG_DIR) if n.endswith(".yaml"))
    if len(names) != EXPECT_CONFIGS:
        sys.exit(f"expected {EXPECT_CONFIGS} configs in {CFG_DIR}, found {len(names)}")

    patched = {}
    missing = []

    for name in names:
        with open(os.path.join(CFG_DIR, name)) as fh:
            cfg = yaml.safe_load(fh)

        cfg["train_bin"] = [to_corpus(p) for p in cfg["train_bin"]]
        cfg["train_mask"] = [to_corpus(p) for p in cfg["train_mask"]]

        ds = cfg["dataset"]
        ds["packed_targets"] = [to_corpus(p) for p in ds["packed_targets"]]
        ds["target_weights"] = [to_corpus(p) for p in ds["target_weights"]]
        if "pointer" in ds:
            ds["pointer"] = POINTER

        cfg["out_dir"] = os.path.join(RUNS, cfg["run_id"])
        cfg["micro_batch_size"] = MICRO_BATCH
        cfg["provider"] = "aws-capacity-block"

        refs = list(cfg["train_bin"]) + list(cfg["train_mask"])
        if "pointer" in ds:
            refs.append(ds["pointer"])
        for path in refs:
            if not os.path.exists(path):
                missing.append((name, path))

        # accum must stay exact or the run silently trains a different batch
        seqs = cfg["tokens_per_step"] // cfg["ctx"]
        if seqs % MICRO_BATCH != 0:
            sys.exit(f"{name}: {seqs} seqs/step not divisible by micro_batch {MICRO_BATCH}")

        patched[name] = cfg

    if missing:
        for name, path in missing:
            print(f"MISSING {name}: {path}")
        sys.exit("refusing to write: referenced files do not exist")

    if not os.path.exists(BACKUP):
        shutil.copytree(CFG_DIR, BACKUP)
        print(f"originals backed up to {BACKUP}")

    for name, cfg in patched.items():
        with open(os.path.join(CFG_DIR, name), "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, width=200)

    sample = patched[names[0]]
    accum = sample["tokens_per_step"] // sample["ctx"] // MICRO_BATCH
    print(f"rewrote {len(patched)} configs")
    print(f"  micro_batch_size={MICRO_BATCH} accum={accum} tokens_per_step={sample['tokens_per_step']}")
    print(f"  out_dir example: {sample['out_dir']}")
    print(f"  train_bin[0]:    {sample['train_bin'][0]}")


if __name__ == "__main__":
    main()
