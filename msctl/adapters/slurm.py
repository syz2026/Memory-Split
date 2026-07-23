"""Deterministic argv-only Slurm adapter for paired 135M runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from msctl.cohort import ARMS, COHORT_ID, SEEDS
from msctl.preflight import validate_preflight
from msctl.profile import SlurmProfile


TRAIN_SCRIPT = "cluster/slurm/v2_pair_train.sbatch"
EVALUATE_SCRIPT = "cluster/slurm/v2_pair_evaluate.sbatch"
MODES = ("functional", "resume", "throughput", "protected")
_HEX = frozenset("0123456789abcdef")


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def load_pair_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"pair manifest is missing or unsafe: {manifest_path}")
    if manifest_path.stat().st_size > 1 << 20:
        raise ValueError("pair manifest exceeds 1 MiB")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pair manifest must be valid UTF-8 JSON") from error
    required = {
        "schema_version",
        "cohort_id",
        "operator",
        "provider",
        "seed",
        "pair_id",
        "profile_sha256",
        "dataset",
        "arms",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("pair manifest fields do not match the protected schema")
    seed = raw["seed"]
    if (
        raw["schema_version"] != 1
        or raw["cohort_id"] != COHORT_ID
        or isinstance(seed, bool)
        or seed not in SEEDS
        or raw["pair_id"] != f"d135m_full_s{seed}"
        or not _is_sha(raw["profile_sha256"])
    ):
        raise ValueError("pair manifest identity is invalid")
    dataset = raw["dataset"]
    if (
        not isinstance(dataset, dict)
        or set(dataset)
        != {"receipt_sha256", "ordered_token_stream_sha256"}
        or not _is_sha(dataset["receipt_sha256"])
        or not _is_sha(dataset["ordered_token_stream_sha256"])
    ):
        raise ValueError("pair manifest dataset identity is invalid")
    arms = raw["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("pair manifest must contain exactly two arms")
    arm_fields = {
        "arm",
        "config_path",
        "config_sha256",
        "runtime_config",
        "runtime_config_sha256",
        "run_id",
        "out_dir",
    }
    if any(not isinstance(item, dict) or set(item) != arm_fields for item in arms):
        raise ValueError("pair manifest arm records are malformed")
    if tuple(item["arm"] for item in arms) != ARMS:
        raise ValueError("pair manifest arms must be ordered Dense then Split90")
    for item in arms:
        arm = item["arm"]
        if (
            item["run_id"] != f"d135m_{arm}_full_s{seed}"
            or not _is_sha(item["config_sha256"])
            or not _is_sha(item["runtime_config_sha256"])
        ):
            raise ValueError(f"pair manifest {arm} identity is invalid")
    return raw


def _wall_time(minutes: int) -> str:
    days, remainder = divmod(minutes, 24 * 60)
    hours, minute = divmod(remainder, 60)
    clock = f"{hours:02d}:{minute:02d}:00"
    return f"{days}-{clock}" if days else clock


def _export_value(value: Path | str, *, label: str) -> str:
    text = str(value)
    if not text or any(marker in text for marker in (",", "\n", "\r", "\x00")):
        raise ValueError(f"{label} cannot be represented in a Slurm export")
    return text


def plan_sbatch(
    pair_manifest: Path | str,
    *,
    profile: SlurmProfile,
    action: str,
    mode: str,
    venv_root: Path | str,
    preflight_path: Path | str | None = None,
) -> list[str]:
    """Render one shell-free ``sbatch`` argv array."""

    if action not in {"train", "evaluate"}:
        raise ValueError("action must be train or evaluate")
    if mode not in MODES:
        raise ValueError(f"unknown run mode: {mode}")
    pair_path = Path(pair_manifest).resolve()
    pair = load_pair_manifest(pair_path)
    if pair["profile_sha256"] != profile.sha256:
        raise ValueError("pair manifest was instantiated for a different profile")
    venv = Path(venv_root)
    if not venv.is_absolute():
        venv = venv.resolve()
    if preflight_path is not None:
        preflight = Path(preflight_path).resolve()
    else:
        preflight = None
    if mode == "protected":
        if preflight is None:
            raise ValueError("protected submission requires a preflight receipt")
        validate_preflight(
            preflight,
            profile=profile,
            dataset_receipt_sha256=pair["dataset"]["receipt_sha256"],
        )

    command = ["sbatch", f"--partition={profile.partition}"]
    if profile.account is not None:
        command.append(f"--account={profile.account}")
    if profile.qos is not None:
        command.append(f"--qos={profile.qos}")
    command.extend(
        [
            f"--gres={profile.gres}",
            f"--cpus-per-task={profile.cpus}",
            f"--mem={profile.memory_gb}G",
            f"--time={_wall_time(profile.wall_minutes)}",
            f"--job-name=ms135-{action}-s{pair['seed']}",
        ]
    )
    exports = [
        "NONE",
        f"PAIR_MANIFEST={_export_value(pair_path, label='pair manifest')}",
        f"PROFILE_PATH={_export_value(profile.source_path.resolve(), label='profile')}",
        f"RUN_MODE={mode}",
        f"MS135_VENV={_export_value(venv, label='venv root')}",
    ]
    if preflight is not None:
        exports.append(
            f"PREFLIGHT_PATH={_export_value(preflight, label='preflight receipt')}"
        )
    command.append(f"--export={','.join(exports)}")
    if action == "train":
        command.append("--requeue")
        command.append(TRAIN_SCRIPT)
    else:
        command.append(EVALUATE_SCRIPT)
    return command
