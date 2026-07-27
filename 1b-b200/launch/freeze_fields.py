#!/usr/bin/env python3
"""Freeze the six node/AMI-derived fields this agent owns into the B200 profile.

Another agent concurrently owns container_runtime.digest,
software_floors.nccl_version and software_floors.torch_version in the same file,
so this is deliberately surgical: it re-reads immediately before writing, sets
only the six owned keys, drops only their pending_confirmation entries, and
refuses to run if any owned field already holds a conflicting value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROFILE = Path(
    "${HOME}/Documents/MemorySplit/.worktrees/aws-p6-b200-profile"
    "/cluster/profiles/aws-p6-b200.48xlarge-135m-v1.json"
)

# Every value here was read back from the live node or the EC2 API. Nothing is
# assumed: the GPU facts come from nvidia-smi on the qualified node (and agree
# with torch's independent view), the AMI facts from describe-images.
OWNED: dict[str, object] = {
    "hardware.gpu_compute_capability": "10.0",
    "hardware.gpu_memory_mib": 183_359,
    "hardware.gpu_name": "NVIDIA B200",
    "image.ami_id": "am${INSTANCE_ID}",
    "image.ami_name": (
        "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20260725"
    ),
    "image.ami_owner": "898082745236",
}

# Explicitly not ours. Named so the intent is auditable rather than implicit.
NOT_OURS = {
    "container_runtime.digest",
    "software_floors.nccl_version",
    "software_floors.torch_version",
}


def get_path(doc: dict, dotted: str):
    section, _, key = dotted.partition(".")
    return doc[section][key]


def set_path(doc: dict, dotted: str, value) -> None:
    section, _, key = dotted.partition(".")
    doc[section][key] = value


def main() -> int:
    doc = json.loads(PROFILE.read_text(encoding="utf-8"))

    before = [e["field"] for e in doc["pending_confirmation"]]
    print(f"pending before ({len(before)}):")
    for f in before:
        print(f"  {f}")

    for dotted, value in OWNED.items():
        current = get_path(doc, dotted)
        if current not in (None, value):
            print(f"REFUSING: {dotted} already holds {current!r}, not overwriting")
            return 1
        set_path(doc, dotted, value)

    doc["pending_confirmation"] = [
        e for e in doc["pending_confirmation"] if e["field"] not in OWNED
    ]

    after = [e["field"] for e in doc["pending_confirmation"]]
    print(f"\npending after ({len(after)}):")
    for f in after:
        print(f"  {f}")

    # The three survivors must be exactly the other agent's fields. If anything
    # else lingers, the division of labour was misunderstood and we stop.
    if set(after) != NOT_OURS:
        print(f"\nUNEXPECTED remaining pending set: {sorted(after)}")
        print(f"expected exactly: {sorted(NOT_OURS)}")
        return 1

    # Match the repo's own serialisation (see _write_profile in the tests).
    PROFILE.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nfrozen:")
    for dotted, value in OWNED.items():
        print(f"  {dotted} = {value!r}")
    print(f"\nwrote {PROFILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
