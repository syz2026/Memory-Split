#!/usr/bin/env python3
"""NCCL all-reduce correctness + bandwidth check over the visible GPUs.

Launched under torchrun. Scope is controlled by CUDA_VISIBLE_DEVICES, so the
same script serves both the full 8-GPU collective and the isolated 2-GPU pair
groups the cohort actually trains in.

Correctness is checked before bandwidth: a fast wrong all-reduce is worthless.
Each rank contributes a known distinct value so the expected sum is exact and a
silently dropped or duplicated rank cannot pass.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    label = os.environ.get("MS_LABEL", "group")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    # The physical GPU behind this rank, not the CUDA_VISIBLE_DEVICES-relative
    # index, so disjointness across concurrent groups is actually provable.
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    phys_id = phys[local_rank] if local_rank < len(phys) and phys[0] != "" else str(local_rank)

    print(
        f"[{label}] rank={rank}/{world} local_rank={local_rank} "
        f"phys_gpu={phys_id} name={props.name} "
        f"uuid={torch.cuda.get_device_properties(dev).uuid if hasattr(props,'uuid') else 'n/a'}",
        flush=True,
    )

    # --- correctness: rank r contributes (r+1); sum must be world*(world+1)/2
    probe = torch.full((1024,), float(rank + 1), device=dev, dtype=torch.float32)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2.0
    got = probe[0].item()
    correct = abs(got - expected) < 1e-3
    if rank == 0:
        print(f"[{label}] correctness: got {got} expected {expected} -> "
              f"{'PASS' if correct else 'FAIL'}", flush=True)
    if not correct:
        dist.destroy_process_group()
        raise SystemExit(f"[{label}] all-reduce produced an incorrect sum")

    # --- bandwidth on a payload big enough to be bandwidth- not latency-bound
    nbytes = 512 * 1024 * 1024
    buf = torch.empty(nbytes // 4, device=dev, dtype=torch.float32).fill_(1.0)

    for _ in range(5):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        sec = elapsed / iters
        algbw = nbytes / sec / 1e9
        # Ring all-reduce moves 2(n-1)/n of the buffer per rank.
        busbw = algbw * (2 * (world - 1) / world)
        print(
            f"[{label}] all_reduce {nbytes // 1024**2} MiB x{iters}: "
            f"{sec*1e3:.2f} ms/iter  algbw {algbw:.1f} GB/s  busbw {busbw:.1f} GB/s",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
