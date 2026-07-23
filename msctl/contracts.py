"""Strict release, run-manifest, and checkpoint contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cohort import load_cohort_assignment as load_cohort_assignment
from .errors import MsctlError
from .jsonutil import (
    COMMIT_RE,
    RUN_ID_RE,
    canonical_sha256,
    load_json,
    portable_relative,
    require_exact_keys,
    require_nonnegative_number,
    require_object,
    require_sha256,
    resolve_inside,
    sha256_file,
)
from .profile import SUPPORTED_PROFILE


@dataclass(frozen=True)
class Release:
    release_id: str
    archive_sha256: str
    source_commit: str
    members_sha256: str
    value: dict[str, object]


@dataclass(frozen=True)
class Run:
    run_id: str
    arm: str
    seed: int
    config: str
    config_sha256: str
    estimated_gpu_hours: float


@dataclass(frozen=True)
class RunManifest:
    provider: str
    release_sha256: str
    dataset_sha256: str
    runs: tuple[Run, ...]
    sha256: str
    value: dict[str, object]

    @property
    def gpu_hours(self) -> float:
        return sum(run.estimated_gpu_hours for run in self.runs)


def load_release(path: Path | str) -> Release:
    value = require_object(load_json(path, label="release"), label="release")
    require_exact_keys(
        value,
        {
            "schema_version",
            "release_id",
            "provider",
            "archive",
            "source",
            "members_sha256",
        },
        label="release",
    )
    if value["schema_version"] != 1 or value["provider"] != SUPPORTED_PROFILE:
        raise MsctlError(
            "RELEASE_INVALID",
            "release is not an Illumina v1 release",
        )
    release_id = value["release_id"]
    if (
        not isinstance(release_id, str)
        or not release_id
        or RUN_ID_RE.fullmatch(release_id) is None
    ):
        raise MsctlError("RELEASE_INVALID", "release_id is invalid")
    archive = require_object(value["archive"], label="release.archive")
    require_exact_keys(
        archive,
        {"path", "sha256", "bytes"},
        label="release.archive",
    )
    portable_relative(archive["path"], label="release.archive.path")
    if (
        isinstance(archive["bytes"], bool)
        or not isinstance(archive["bytes"], int)
        or archive["bytes"] <= 0
    ):
        raise MsctlError("RELEASE_INVALID", "release archive bytes are invalid")
    source = require_object(value["source"], label="release.source")
    require_exact_keys(
        source,
        {"commit", "dirty"},
        label="release.source",
    )
    if (
        not isinstance(source["commit"], str)
        or COMMIT_RE.fullmatch(source["commit"]) is None
        or source["dirty"] is not False
    ):
        raise MsctlError(
            "RELEASE_INVALID",
            "release source must bind a clean Git commit",
        )
    return Release(
        release_id=release_id,
        archive_sha256=require_sha256(
            archive["sha256"], label="release.archive.sha256"
        ),
        source_commit=source["commit"],
        members_sha256=require_sha256(
            value["members_sha256"], label="release.members_sha256"
        ),
        value=value,
    )


def load_run_manifest(
    path: Path | str,
    *,
    repo_root: Path | str,
) -> RunManifest:
    value = require_object(
        load_json(path, label="run manifest"),
        label="run manifest",
    )
    require_exact_keys(
        value,
        {
            "schema_version",
            "provider",
            "release_sha256",
            "dataset_sha256",
            "runs",
        },
        label="run manifest",
    )
    if value["schema_version"] != 1 or value["provider"] != SUPPORTED_PROFILE:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest provider or schema is unsupported",
        )
    raw_runs = value["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 2:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "seed-0 launch requires exactly two runs",
        )
    runs: list[Run] = []
    for index, item in enumerate(raw_runs):
        row = require_object(item, label=f"run manifest.runs[{index}]")
        require_exact_keys(
            row,
            {
                "run_id",
                "arm",
                "seed",
                "config",
                "config_sha256",
                "estimated_gpu_hours",
            },
            label=f"run manifest.runs[{index}]",
        )
        run_id = row["run_id"]
        if (
            not isinstance(run_id, str)
            or RUN_ID_RE.fullmatch(run_id) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run_id is invalid",
                details={"index": index},
            )
        if row["arm"] not in {"dense", "split90"} or row["seed"] != 0:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "seed-0 runs must be Dense and Split90 at seed zero",
            )
        relative = portable_relative(
            row["config"], label=f"run manifest.runs[{index}].config"
        )
        expected_hash = require_sha256(
            row["config_sha256"],
            label=f"run manifest.runs[{index}].config_sha256",
        )
        config = resolve_inside(
            repo_root,
            relative,
            label=f"run manifest.runs[{index}].config",
        )
        if config.is_symlink() or not config.is_file():
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run config must be a regular file",
                details={"config": relative},
            )
        if sha256_file(config) != expected_hash:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run config hash mismatch",
                details={"config": relative},
            )
        runs.append(
            Run(
                run_id=run_id,
                arm=str(row["arm"]),
                seed=0,
                config=relative,
                config_sha256=expected_hash,
                estimated_gpu_hours=require_nonnegative_number(
                    row["estimated_gpu_hours"],
                    label=(
                        f"run manifest.runs[{index}].estimated_gpu_hours"
                    ),
                ),
            )
        )
    if (
        {run.arm for run in runs} != {"dense", "split90"}
        or len({run.run_id for run in runs}) != 2
        or len({run.config for run in runs}) != 2
    ):
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run IDs, configs, and Dense/Split90 arms must be unique",
        )
    return RunManifest(
        provider=str(value["provider"]),
        release_sha256=require_sha256(
            value["release_sha256"],
            label="run manifest.release_sha256",
        ),
        dataset_sha256=require_sha256(
            value["dataset_sha256"],
            label="run manifest.dataset_sha256",
        ),
        runs=tuple(sorted(runs, key=lambda run: run.run_id)),
        sha256=canonical_sha256(value),
        value=value,
    )


def bind_release(release: Release, manifest: RunManifest) -> None:
    if release.archive_sha256 != manifest.release_sha256:
        raise MsctlError(
            "RELEASE_RUN_MISMATCH",
            "run manifest does not bind the supplied release",
        )


def verify_checkpoint_receipt(
    path: Path | str,
    *,
    release: Release,
    manifest: RunManifest,
) -> dict[str, object]:
    receipt_path = Path(path)
    value = require_object(
        load_json(receipt_path, label="checkpoint receipt"),
        label="checkpoint receipt",
    )
    require_exact_keys(
        value,
        {
            "schema_version",
            "provider",
            "release_sha256",
            "run_manifest_sha256",
            "dataset_sha256",
            "checkpoints",
        },
        label="checkpoint receipt",
    )
    if (
        value["schema_version"] != 1
        or value["provider"] != manifest.provider
        or value["release_sha256"] != release.archive_sha256
        or value["run_manifest_sha256"] != manifest.sha256
        or value["dataset_sha256"] != manifest.dataset_sha256
    ):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt does not bind the run artifacts",
        )
    raw = value["checkpoints"]
    if not isinstance(raw, list) or len(raw) != len(manifest.runs):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt must contain the complete paired run",
        )
    by_id = {run.run_id: run for run in manifest.runs}
    seen: set[str] = set()
    for index, item in enumerate(raw):
        row = require_object(item, label=f"checkpoint[{index}]")
        require_exact_keys(
            row,
            {
                "run_id",
                "path",
                "sha256",
                "config_sha256",
                "step",
                "world_size",
            },
            label=f"checkpoint[{index}]",
        )
        run_id = row["run_id"]
        if (
            not isinstance(run_id, str)
            or run_id not in by_id
            or run_id in seen
            or row["config_sha256"] != by_id[run_id].config_sha256
            or row["world_size"] != 3
            or isinstance(row["step"], bool)
            or not isinstance(row["step"], int)
            or row["step"] <= 0
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint metadata does not match its run",
                details={"index": index},
            )
        seen.add(run_id)
        relative = portable_relative(
            row["path"], label=f"checkpoint[{index}].path"
        )
        checkpoint = resolve_inside(
            receipt_path.parent,
            relative,
            label=f"checkpoint[{index}].path",
        )
        expected_hash = require_sha256(
            row["sha256"], label=f"checkpoint[{index}].sha256"
        )
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != expected_hash
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint bytes do not match their receipt",
                details={"run_id": run_id},
            )
    if seen != set(by_id):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt is missing a paired run",
        )
    return value
