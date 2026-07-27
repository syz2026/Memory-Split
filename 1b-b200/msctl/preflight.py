"""Protected-launch canary receipt validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cluster.corpus_contract import sha256_file
from msctl.cohort import COHORT_ID
from msctl.manifest import write_json_no_replace
from msctl.profile import SlurmProfile
from msctl.reasoning_cohort import COHORT_ID as REASONING_COHORT_ID

CANARY_NAMES = (
    "one_update_functional",
    "exact_paired_resume",
    "one_hundred_update_throughput_oom",
)
_HEX = frozenset("0123456789abcdef")


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def load_preflight(path: Path | str) -> dict[str, Any]:
    receipt = Path(path)
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError(f"preflight receipt is missing or unsafe: {receipt}")
    if receipt.stat().st_size > 1 << 20:
        raise ValueError("preflight receipt exceeds 1 MiB")
    try:
        raw = json.loads(receipt.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("preflight receipt must be valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("preflight receipt must be an object")  # noqa: TRY004
    return raw


def validate_preflight(
    path: Path | str,
    *,
    profile: SlurmProfile,
    dataset_receipt_sha256: str,
    cohort_id: str = COHORT_ID,
) -> dict[str, Any]:
    if cohort_id not in {COHORT_ID, REASONING_COHORT_ID}:
        raise ValueError("preflight cohort is not a supported 135M contract")
    raw = load_preflight(path)
    expected_fields = {
        "schema_version",
        "cohort_id",
        "site_id",
        "profile_sha256",
        "dataset_receipt_sha256",
        "canaries",
    }
    if set(raw) != expected_fields:
        raise ValueError("preflight receipt fields do not match the protected schema")
    if (
        raw["schema_version"] != 1
        or isinstance(raw["schema_version"], bool)
        or raw["cohort_id"] != cohort_id
        or raw["site_id"] != profile.site_id
        or raw["profile_sha256"] != profile.sha256
        or raw["dataset_receipt_sha256"] != dataset_receipt_sha256
    ):
        raise ValueError("preflight receipt identities do not match this launch")
    if not _is_sha(dataset_receipt_sha256):
        raise ValueError("dataset receipt identity is not a SHA-256 digest")
    canaries = raw["canaries"]
    if not isinstance(canaries, Mapping) or set(canaries) != set(CANARY_NAMES):
        raise ValueError("preflight receipt does not contain all three canaries")
    allowed = {
        "passed",
        "evidence_sha256",
        "updates",
        "oom_detected",
        "tokens_per_second",
        "gpu_names",
        "environment_sha256",
    }
    for name in CANARY_NAMES:
        record = canaries[name]
        if (
            not isinstance(record, Mapping)
            or set(record) - allowed
            or record.get("passed") is not True
        ):
            raise ValueError(f"preflight canary did not pass: {name}")
        for field in ("evidence_sha256", "environment_sha256"):
            if field in record and not _is_sha(record[field]):
                raise ValueError(f"preflight {name} {field} is invalid")
    throughput = canaries["one_hundred_update_throughput_oom"]
    if throughput.get("oom_detected") is not False:
        raise ValueError("100-update canary recorded an OOM")
    return raw


def _load_canary(path: Path, *, mode: str, updates: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{mode} canary evidence is missing or unsafe")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{mode} canary evidence is invalid JSON") from error
    arms = evidence.get("arms") if isinstance(evidence, Mapping) else None
    if (
        evidence.get("mode") != mode
        or evidence.get("status") != "completed"
        or not isinstance(arms, Mapping)
        or set(arms) != {"dense", "split90"}
    ):
        raise ValueError(f"{mode} canary evidence is not a completed pair")
    for arm, record in arms.items():
        if (
            not isinstance(record, Mapping)
            or record.get("status") != "completed"
            or record.get("step") != updates
            or record.get("oom_detected") is not False
            or record.get("gpu_supported") is not True
        ):
            raise ValueError(f"{mode} {arm} canary did not satisfy its gate")
        if mode == "resume" and record.get("resume_exact") is not True:
            raise ValueError(f"{mode} {arm} did not prove exact resume")
    record = {
        "evidence_sha256": sha256_file(path),
        "passed": True,
        "updates": updates,
    }
    if mode == "throughput":
        values = [arms[arm].get("tokens_per_second") for arm in ("dense", "split90")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            for value in values
        ):
            raise ValueError("throughput canary has no positive measurement")
        record.update(
            {
                "oom_detected": False,
                "tokens_per_second": {
                    "dense": float(values[0]),
                    "split90": float(values[1]),
                },
            }
        )
    return record


def build_preflight_receipt(
    *,
    profile: SlurmProfile,
    dataset_receipt_sha256: str,
    functional_evidence: Path | str,
    resume_evidence: Path | str,
    throughput_evidence: Path | str,
    output: Path | str,
    cohort_id: str = COHORT_ID,
) -> Path:
    """Freeze all three measured site canaries into a no-replace receipt."""

    if cohort_id not in {COHORT_ID, REASONING_COHORT_ID}:
        raise ValueError("preflight cohort is not a supported 135M contract")
    if not _is_sha(dataset_receipt_sha256):
        raise ValueError("dataset receipt identity is not a SHA-256 digest")
    receipt = {
        "canaries": {
            "exact_paired_resume": _load_canary(
                Path(resume_evidence),
                mode="resume",
                updates=2,
            ),
            "one_hundred_update_throughput_oom": _load_canary(
                Path(throughput_evidence),
                mode="throughput",
                updates=100,
            ),
            "one_update_functional": _load_canary(
                Path(functional_evidence),
                mode="functional",
                updates=1,
            ),
        },
        "cohort_id": cohort_id,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "profile_sha256": profile.sha256,
        "schema_version": 1,
        "site_id": profile.site_id,
    }
    path = write_json_no_replace(output, receipt)
    validate_preflight(
        path,
        profile=profile,
        dataset_receipt_sha256=dataset_receipt_sha256,
        cohort_id=cohort_id,
    )
    return path
