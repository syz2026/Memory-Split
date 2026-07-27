#!/usr/bin/env python3
"""Assert the four concurrent pair groups were genuinely disjoint.

The training schedule puts four independent two-GPU jobs on one node at once.
If two of those groups ever landed on the same physical GPU the runs would
contend for memory and bandwidth and the timing comparison between the dense
and split90 arms would be confounded, so this is checked rather than assumed.

Parses the physical GPU each rank reported in the pair logs and requires:
  - exactly 4 pair groups
  - exactly 2 distinct physical GPUs per group
  - no physical GPU claimed by more than one group
  - all 8 GPUs covered
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RANK_RE = re.compile(r"\[(?P<label>[^\]]+)\] rank=(?P<rank>\d+)/(?P<world>\d+).*?phys_gpu=(?P<phys>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, type=Path)
    args = ap.parse_args()

    groups: dict[str, set[str]] = {}
    for log in sorted(args.logs.glob("nccl-pair*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = RANK_RE.search(line)
            if m:
                groups.setdefault(log.stem, set()).add(m.group("phys"))

    ok = True
    print(f"parsed {len(groups)} pair groups from {args.logs}")
    for name, gpus in sorted(groups.items()):
        size_ok = len(gpus) == 2
        print(f"  {name}: physical GPUs {sorted(gpus)} "
              f"-> {'ok' if size_ok else 'FAIL expected exactly 2'}")
        ok &= size_ok

    if len(groups) != 4:
        print(f"FAIL: expected 4 concurrent pair groups, parsed {len(groups)}")
        ok = False

    seen: dict[str, str] = {}
    for name, gpus in sorted(groups.items()):
        for g in gpus:
            if g in seen:
                print(f"FAIL: physical GPU {g} claimed by both {seen[g]} and {name}")
                ok = False
            else:
                seen[g] = name

    covered = set(seen)
    expected = {str(i) for i in range(8)}
    if covered != expected:
        print(f"FAIL: GPU coverage {sorted(covered)} != {sorted(expected)}")
        ok = False
    else:
        print("coverage: all 8 physical GPUs claimed exactly once")

    print("PAIR ALLOCATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
