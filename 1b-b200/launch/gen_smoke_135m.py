#!/usr/bin/env python3
"""Emit 30-step smoke variants of the d135m seed-0 configs.

Written to /mnt/nvme/probe rather than configs/135m-v3, and pointed at an
out_dir outside runs/, for one specific reason: the production launcher starts
every run with `--resume auto`, so a smoke checkpoint left in the real out_dir
would be loaded as genuine progress and silently poison the cohort.

Everything that matters for validation is left identical to production -- the
absolute corpus paths, micro_batch_size 64, compile -- so this exercises the
parts that have never been executed on this node.
"""
import os

import yaml

SRC = "/mnt/nvme/code/configs/135m-v3"
DST = "/mnt/nvme/probe"
OUT = "/mnt/nvme/probe/smoke-135m"
STEPS = 30

os.makedirs(OUT, exist_ok=True)

for arm in ("dense", "split90"):
    with open(f"{SRC}/{arm}-s0.yaml") as fh:
        cfg = yaml.safe_load(fh)

    cfg["run_id"] = f"{cfg['run_id']}_smoke"
    cfg["out_dir"] = os.path.join(OUT, cfg["run_id"])
    cfg["max_steps"] = STEPS
    cfg["warmup_steps"] = 10
    # Trainer rejects checkpoint targets past the end of the run, so these have
    # to be scaled with max_steps. Three of them keeps the snapshot path in the
    # test rather than only the final save.
    cfg["checkpoint_updates"] = [10, 20, STEPS]

    path = f"{DST}/smoke-135m-{arm}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, width=200)

    print(f"wrote {path}")
    print(f"  out_dir     {cfg['out_dir']}")
    print(f"  micro_batch {cfg['micro_batch_size']}  steps {cfg['max_steps']}")
    print(f"  train_bin   {cfg['train_bin'][0]}")
    print(f"  train_mask  {cfg['train_mask'][0]}")
