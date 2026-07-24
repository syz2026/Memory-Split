"""Build the exact Task3/5 paired-launch manifest after bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.aws.p5.profile import (
    AWS_P5_V3_PROFILE_ID,
    AWS_P6_B300_V3_PROFILE_ID,
    LEGACY_AWS_P5_PROFILE_ID,
    load_aws_gpu_profile,
)
from msctl.jsonutil import canonical_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_FIELDS = {"arm", "config", "config_sha256"}
_V3_PROVIDERS = {AWS_P5_V3_PROFILE_ID, AWS_P6_B300_V3_PROFILE_ID}
_V3_BINDING_FIELDS = (
    "run_manifest_sha256",
    "preregistration_sha256",
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "sealed_evaluation_sha256",
    "fleet_plan_sha256",
)


class LaunchManifestError(ValueError):
    """The post-bootstrap launcher handoff is invalid."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchManifestError(f"JSON repeats field: {key}")
        result[key] = value
    return result


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LaunchManifestError(f"{label} must be a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                LaunchManifestError(f"{label} contains non-finite {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchManifestError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise LaunchManifestError(f"{label} must be a JSON object")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LaunchManifestError(f"{label} must be lowercase SHA-256")
    return value


def _hash_regular(path: Path, *, label: str) -> str:
    before = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or path.is_symlink()
    ):
        raise LaunchManifestError(f"{label} must be singly linked and regular")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise LaunchManifestError(f"{label} changed while hashing")
    return digest


def _relative_inside(path: Path, root: Path, *, label: str) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise LaunchManifestError(f"{label} is outside scratch") from error


def build_launcher_manifest(
    *,
    out: Path,
    scratch_root: Path,
    seed: int,
    profile_sha256: str,
    release_sha256: str,
    release_members_sha256: str,
    cohort_assignment_sha256: str,
    code_commit: str,
    bootstrap_receipt: Path,
    corpus_receipt: Path,
    runs: Sequence[dict[str, object]],
    profile: object | None = None,
    run_manifest_sha256: str | None = None,
    preregistration_sha256: str | None = None,
    hardware_amendment_sha256: str | None = None,
    provider_selection_sha256: str | None = None,
    sealed_evaluation_sha256: str | None = None,
    fleet_plan_sha256: str | None = None,
    fleet_wave: int | None = None,
) -> dict[str, object]:
    """Resolve dynamic receipt hashes into Task3/5's closed manifest schema."""

    provider = getattr(profile, "provider", LEGACY_AWS_P5_PROFILE_ID)
    instance_type = getattr(profile, "instance_type", "p5.48xlarge")
    gres = getattr(profile, "gres", "gpu:h100:8")
    assigned_seeds = tuple(
        getattr(profile, "assigned_seeds", (1, 2, 3, 4))
    )
    cpu_affinity_halves = tuple(
        getattr(profile, "cpu_affinity_halves", ((0, 95), (96, 191)))
    )
    receipt_type = getattr(profile, "bootstrap_receipt_type", "aws-p5-bootstrap")
    is_v3 = provider in _V3_PROVIDERS
    if (
        not isinstance(provider, str)
        or not provider
        or (
            profile is not None
            and getattr(profile, "profile_id", None) != provider
        )
        or not isinstance(instance_type, str)
        or not instance_type
        or not isinstance(gres, str)
        or not gres.endswith(":8")
        or len(cpu_affinity_halves) != 2
        or any(len(group) != 2 for group in cpu_affinity_halves)
        or (
            profile is not None
            and profile_sha256 != getattr(profile, "sha256", None)
        )
    ):
        raise LaunchManifestError("profile launch geometry is invalid")
    if type(seed) is not int or seed not in assigned_seeds:
        raise LaunchManifestError("seed must be assigned to AWS")
    for label, value in (
        ("profile", profile_sha256),
        ("release", release_sha256),
        ("release members", release_members_sha256),
        ("cohort assignment", cohort_assignment_sha256),
    ):
        _sha256(value, label=label)
    v3_bindings = {
        "run_manifest_sha256": run_manifest_sha256,
        "preregistration_sha256": preregistration_sha256,
        "hardware_amendment_sha256": hardware_amendment_sha256,
        "provider_selection_sha256": provider_selection_sha256,
        "sealed_evaluation_sha256": sealed_evaluation_sha256,
        "fleet_plan_sha256": fleet_plan_sha256,
    }
    if is_v3:
        for field, value in v3_bindings.items():
            _sha256(value, label=field.replace("_", " "))
        if type(fleet_wave) is not int or fleet_wave < 0:
            raise LaunchManifestError("fleet wave must be a nonnegative integer")
    elif any(value is not None for value in (*v3_bindings.values(), fleet_wave)):
        raise LaunchManifestError("v3 bindings cannot be added to a legacy launch")
    if not isinstance(code_commit, str) or _COMMIT_RE.fullmatch(code_commit) is None:
        raise LaunchManifestError("code commit must be a full lowercase Git commit")
    scratch = scratch_root.resolve(strict=True)
    bootstrap_hash = _hash_regular(
        bootstrap_receipt,
        label="bootstrap receipt",
    )
    bootstrap = _load_object(bootstrap_receipt, label="bootstrap receipt")
    if (
        bootstrap.get("provider") != provider
        or bootstrap.get("instance_type") != instance_type
        or bootstrap.get("profile_sha256") != profile_sha256
        or bootstrap.get("receipt_type") != receipt_type
        or (
            profile is not None
            and bootstrap.get("scratch_root")
            != getattr(profile, "scratch_root", str(scratch))
        )
    ):
        raise LaunchManifestError(
            "bootstrap receipt does not bind the selected profile"
        )
    corpus_hash = _hash_regular(corpus_receipt, label="corpus receipt")
    corpus = _load_object(corpus_receipt, label="corpus receipt")
    ordered_sha256 = _sha256(
        corpus.get("ordered_stream_sha256"),
        label="corpus ordered stream",
    )
    if len(runs) != 2:
        raise LaunchManifestError("launcher requires one complete pair")
    by_arm: dict[str, dict[str, object]] = {}
    for row in runs:
        if not isinstance(row, dict) or set(row) != _RUN_FIELDS:
            raise LaunchManifestError("run binding fields do not match")
        arm = row["arm"]
        config = row["config"]
        if (
            arm not in {"dense", "split90"}
            or arm in by_arm
            or not isinstance(config, str)
            or not config
            or config.startswith("/")
            or "\\" in config
            or any(part in {"", ".", ".."} for part in config.split("/"))
        ):
            raise LaunchManifestError("run binding identity is invalid")
        _sha256(row["config_sha256"], label=f"{arm} config")
        by_arm[str(arm)] = row
    if set(by_arm) != {"dense", "split90"}:
        raise LaunchManifestError("launcher pair must contain Dense and Split90")
    base_port = 29_500 + seed * 2
    launch_runs = []
    for arm, port, affinity in (
        ("dense", base_port, list(cpu_affinity_halves[0])),
        ("split90", base_port + 1, list(cpu_affinity_halves[1])),
    ):
        binding = by_arm[arm]
        launch_runs.append(
            {
                "arm": arm,
                "checkpoint": f"runs/seed-{seed}/{arm}/run/ckpt.pt",
                "config": binding["config"],
                "config_sha256": binding["config_sha256"],
                "cpu_affinity": affinity,
                "data_loader_workers": 16,
                "master_port": port,
                "rank_zero_pid_file": (
                    f"runs/seed-{seed}/{arm}/rank-zero.pid"
                ),
            }
        )
    manifest = {
        "bootstrap_receipt": {
            "path": _relative_inside(
                bootstrap_receipt,
                scratch,
                label="bootstrap receipt",
            ),
            "sha256": bootstrap_hash,
        },
        "code_commit": code_commit,
        "cohort_assignment_sha256": cohort_assignment_sha256,
        "cohort_id": (
            "memorysplit-confirmatory-v3-360m-n10-aws"
            if is_v3
            else "memorysplit-confirmatory-v2-360m-n5"
        ),
        "corpus_receipt": {
            "ordered_stream_sha256": ordered_sha256,
            "path": _relative_inside(
                corpus_receipt,
                scratch,
                label="corpus receipt",
            ),
            "sha256": corpus_hash,
        },
        "profile_sha256": profile_sha256,
        "provider": provider,
        "release_members_sha256": release_members_sha256,
        "release_sha256": release_sha256,
        "runs": launch_runs,
        "schema_version": 3 if is_v3 else 1,
        "seed": seed,
    }
    if is_v3:
        manifest.update(v3_bindings)
        manifest["fleet_wave"] = fleet_wave
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        out,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        payload = canonical_json(manifest) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def _run_binding(value: str) -> dict[str, object]:
    try:
        row = json.loads(value, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("run binding must be JSON") from error
    if not isinstance(row, dict):
        raise argparse.ArgumentTypeError("run binding must be an object")
    return row


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--release-members-sha256", required=True)
    parser.add_argument("--cohort-assignment-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-manifest-sha256")
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--hardware-amendment-sha256")
    parser.add_argument("--provider-selection-sha256")
    parser.add_argument("--sealed-evaluation-sha256")
    parser.add_argument("--fleet-plan-sha256")
    parser.add_argument("--fleet-wave", type=int)
    parser.add_argument("--bootstrap-receipt", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--run", type=_run_binding, action="append", required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=root / "cluster" / "profiles" / "aws-p5.48xlarge.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        profile = load_aws_gpu_profile(arguments.profile)
        manifest = build_launcher_manifest(
            out=arguments.out,
            scratch_root=arguments.scratch_root,
            seed=arguments.seed,
            profile_sha256=arguments.profile_sha256,
            release_sha256=arguments.release_sha256,
            release_members_sha256=arguments.release_members_sha256,
            cohort_assignment_sha256=arguments.cohort_assignment_sha256,
            code_commit=arguments.code_commit,
            bootstrap_receipt=arguments.bootstrap_receipt,
            corpus_receipt=arguments.corpus_receipt,
            runs=arguments.run,
            profile=profile,
            run_manifest_sha256=arguments.run_manifest_sha256,
            preregistration_sha256=arguments.preregistration_sha256,
            hardware_amendment_sha256=arguments.hardware_amendment_sha256,
            provider_selection_sha256=arguments.provider_selection_sha256,
            sealed_evaluation_sha256=arguments.sealed_evaluation_sha256,
            fleet_plan_sha256=arguments.fleet_plan_sha256,
            fleet_wave=arguments.fleet_wave,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        return 0
    except (LaunchManifestError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"ok": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
