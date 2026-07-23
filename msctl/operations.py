"""High-level instantiate, submit, resume, status, evaluate, and collect APIs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from cluster.corpus_contract import sha256_file
from msctl.adapters.slurm import load_pair_manifest, plan_sbatch
from msctl.cohort import (
    ARMS,
    ROLES,
    config_path,
    load_run_config,
)
from msctl.manifest import (
    build_role_manifest,
    canonical_json_bytes,
    write_json_no_replace,
)
from msctl.profile import load_profile


MAX_CAPTURE_CHARS = 16_384


def _write_yaml_no_replace(path: Path, value: object) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace runtime config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        data = yaml.safe_dump(value, sort_keys=False).encode()
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def instantiate(
    role: str,
    *,
    dataset_root: Path | str,
    pointer_path: Path | str,
    source_lock_path: Path | str,
    profile_path: Path | str,
    runtime_root: Path | str,
    out_root: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    """Instantiate two immutable pair manifests for one frozen operator role."""

    if role not in ROLES:
        raise ValueError(f"unknown frozen role: {role}")
    root = Path(repository_root).resolve()
    dataset = Path(dataset_root).resolve()
    runtime = Path(runtime_root).resolve()
    outputs = Path(out_root).resolve()
    profile = load_profile(profile_path)
    if profile.platform != ROLES[role]["platform"]:
        raise ValueError("profile platform does not match the role assignment")
    role_manifest = build_role_manifest(
        role,
        dataset_root=dataset,
        pointer_path=pointer_path,
        source_lock_path=source_lock_path,
        repository_root=root,
    )
    role_manifest_path = write_json_no_replace(
        runtime / "role-manifest.json",
        role_manifest,
    )
    pair_paths = []
    for seed in ROLES[role]["seeds"]:
        arm_records = []
        for arm in ARMS:
            relative = config_path(arm, seed)
            static_path = root / relative
            cfg = load_run_config(static_path, root=root)
            runtime_cfg = dict(cfg)
            runtime_cfg["train_bin"] = str(
                dataset / cfg["dataset"]["packed_targets"].removeprefix("dataset/")
            )
            runtime_cfg["train_mask"] = str(
                dataset / cfg["dataset"]["target_weights"].removeprefix("dataset/")
            )
            runtime_cfg["out_dir"] = str(outputs / cfg["run_id"])
            runtime_cfg["dataset_receipt_sha256"] = role_manifest["dataset"][
                "receipt_sha256"
            ]
            runtime_cfg["ordered_token_stream_sha256"] = role_manifest["dataset"][
                "ordered_token_stream_sha256"
            ]
            runtime_cfg["profile_sha256"] = profile.sha256
            runtime_path = runtime / "configs" / f"{cfg['run_id']}.yaml"
            _write_yaml_no_replace(runtime_path, runtime_cfg)
            arm_records.append(
                {
                    "arm": arm,
                    "config_path": relative,
                    "config_sha256": sha256_file(static_path),
                    "out_dir": runtime_cfg["out_dir"],
                    "run_id": cfg["run_id"],
                    "runtime_config": str(runtime_path),
                    "runtime_config_sha256": sha256_file(runtime_path),
                }
            )
        pair = {
            "arms": arm_records,
            "cohort_id": role_manifest["cohort_id"],
            "dataset": {
                "ordered_token_stream_sha256": role_manifest["dataset"][
                    "ordered_token_stream_sha256"
                ],
                "receipt_sha256": role_manifest["dataset"]["receipt_sha256"],
            },
            "operator": role,
            "pair_id": f"d135m_full_s{seed}",
            "profile_sha256": profile.sha256,
            "provider": ROLES[role]["provider"],
            "schema_version": 1,
            "seed": seed,
        }
        pair_path = write_json_no_replace(
            runtime / "pairs" / f"pair-s{seed}.json",
            pair,
        )
        load_pair_manifest(pair_path)
        pair_paths.append(pair_path)
    return {
        "role_manifest": role_manifest_path,
        "pair_manifests": pair_paths,
        "profile_sha256": profile.sha256,
        "dataset_receipt_sha256": role_manifest["dataset"]["receipt_sha256"],
    }


def _default_runner(command: Sequence[str]):
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _result_fields(result: object) -> tuple[int, str, str]:
    if isinstance(result, bool):
        raise TypeError("boolean is not a command result")
    if isinstance(result, int):
        return result, "", ""
    returncode = getattr(result, "returncode", None)
    if returncode is None:
        raise TypeError("command result has no returncode")
    return (
        int(returncode),
        str(getattr(result, "stdout", "") or "")[:MAX_CAPTURE_CHARS],
        str(getattr(result, "stderr", "") or "")[:MAX_CAPTURE_CHARS],
    )


def submit(
    pair_manifests: Sequence[Path | str],
    *,
    profile_path: Path | str,
    mode: str,
    venv_root: Path | str,
    preflight_path: Path | str | None = None,
    apply: bool = False,
    action: str = "train",
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> dict[str, Any]:
    """Plan by default; explicit ``apply=True`` submits every pair."""

    profile = load_profile(profile_path)
    manifests = [Path(path).resolve() for path in pair_manifests]
    if not manifests or len(manifests) != len(set(manifests)):
        raise ValueError("submission requires unique pair manifests")
    commands = [
        plan_sbatch(
            path,
            profile=profile,
            action=action,
            mode=mode,
            venv_root=venv_root,
            preflight_path=preflight_path,
        )
        for path in manifests
    ]
    report: dict[str, Any] = {
        "action": action,
        "commands": commands,
        "dry_run": not apply,
        "exit_code": 0,
        "mode": mode,
        "planned": len(commands),
        "profile_sha256": profile.sha256,
        "schema_version": 1,
    }
    if not apply:
        return report
    jobs, failures = [], []
    for index, (manifest, command) in enumerate(zip(manifests, commands)):
        record: dict[str, Any] = {
            "command": command,
            "index": index,
            "pair_manifest": str(manifest),
        }
        try:
            result = runner(command)
            returncode, stdout, stderr = _result_fields(result)
            record.update(
                {
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
        except Exception as error:
            record.update(
                {
                    "returncode": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        jobs.append(record)
        if record["returncode"] != 0:
            failures.append(record)
    report.update(
        {
            "attempted": len(jobs),
            "submitted": len(jobs) - len(failures),
            "failures": failures,
            "jobs": jobs,
            "exit_code": int(bool(failures)),
        }
    )
    return report


def inspect_paired_resume(
    dense_checkpoint: Path | str,
    split90_checkpoint: Path | str,
    *,
    loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require both checkpoints to share the exact pair cursor and update."""

    dense, split = Path(dense_checkpoint), Path(split90_checkpoint)
    present = (dense.is_file() and not dense.is_symlink(), split.is_file() and not split.is_symlink())
    if present == (False, False):
        return {"cursor": 0, "epoch": 0, "mode": "fresh", "step": 0}
    if present != (True, True):
        raise ValueError("paired resume requires both arm checkpoints or neither")
    if loader is None:
        import torch

        loader = lambda path: torch.load(  # noqa: E731
            path,
            map_location="cpu",
            weights_only=False,
        )
    states = [loader(path) for path in (dense, split)]
    if any(not isinstance(state, Mapping) for state in states):
        raise ValueError("checkpoint state must be a mapping")
    steps = [state.get("step") for state in states]
    if (
        any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps)
        or steps[0] != steps[1]
    ):
        raise ValueError("paired checkpoints do not share an exact step")
    data = [state.get("data") for state in states]
    if any(not isinstance(item, Mapping) for item in data):
        raise ValueError("paired checkpoints do not contain data cursors")
    cursors = [item.get("cursor") for item in data]
    epochs = [item.get("epoch", 0) for item in data]
    if cursors[0] != cursors[1] or epochs[0] != epochs[1]:
        raise ValueError("paired checkpoints do not share an exact data cursor")
    return {
        "cursor": cursors[0],
        "epoch": epochs[0],
        "mode": "resume",
        "step": steps[0],
    }


def resume(
    pair_manifest: Path | str,
    *,
    profile_path: Path | str,
    venv_root: Path | str,
    preflight_path: Path | str,
    apply: bool = False,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> dict[str, Any]:
    pair = load_pair_manifest(pair_manifest)
    by_arm = {record["arm"]: record for record in pair["arms"]}
    resume_state = inspect_paired_resume(
        Path(by_arm["dense"]["out_dir"]) / "ckpt.pt",
        Path(by_arm["split90"]["out_dir"]) / "ckpt.pt",
    )
    report = submit(
        [pair_manifest],
        profile_path=profile_path,
        mode="protected",
        venv_root=venv_root,
        preflight_path=preflight_path,
        apply=apply,
        runner=runner,
    )
    report["resume_state"] = resume_state
    return report


def evaluate(
    pair_manifests: Sequence[Path | str],
    **kwargs,
) -> dict[str, Any]:
    return submit(pair_manifests, action="evaluate", **kwargs)


def status(
    pair_manifests: Sequence[Path | str],
    *,
    evidence_root: Path | str,
    action: str = "train",
) -> dict[str, Any]:
    root = Path(evidence_root)
    pairs = []
    for path in pair_manifests:
        pair = load_pair_manifest(path)
        evidence = root / f"{pair['pair_id']}-{action}-evidence.json"
        record = {
            "pair_id": pair["pair_id"],
            "evidence": str(evidence),
            "status": "missing",
        }
        if evidence.is_file() and not evidence.is_symlink():
            try:
                value = json.loads(evidence.read_text())
                record["status"] = value.get("status", "unknown")
                record["sha256"] = sha256_file(evidence)
            except (UnicodeDecodeError, json.JSONDecodeError):
                record["status"] = "invalid"
        pairs.append(record)
    return {
        "action": action,
        "pairs": pairs,
        "schema_version": 1,
    }


def collect(
    pair_manifests: Sequence[Path | str],
    *,
    evidence_root: Path | str,
    output: Path | str,
) -> Path:
    root = Path(evidence_root)
    records = []
    for path in pair_manifests:
        pair_path = Path(path)
        pair = load_pair_manifest(pair_path)
        evidence = {}
        for action in ("train", "evaluate"):
            evidence_path = root / f"{pair['pair_id']}-{action}-evidence.json"
            if not evidence_path.is_file() or evidence_path.is_symlink():
                raise ValueError(f"missing safe {action} evidence for {pair['pair_id']}")
            evidence[action] = {
                "path": str(evidence_path.resolve()),
                "sha256": sha256_file(evidence_path),
            }
        records.append(
            {
                "evidence": evidence,
                "pair_id": pair["pair_id"],
                "pair_manifest_sha256": sha256_file(pair_path),
                "seed": pair["seed"],
            }
        )
    document = {
        "pairs": sorted(records, key=lambda item: item["seed"]),
        "schema_version": 1,
    }
    return write_json_no_replace(output, document)
