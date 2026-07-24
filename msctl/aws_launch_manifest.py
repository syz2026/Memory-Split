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

from msctl.aws_contracts import (
    ARMS,
    COHORT_ID,
    CONFIG_ROOT,
    DATASET_RECEIPT_PATH,
    PROVIDER,
    SEEDS,
)
from msctl.aws_lifecycle import (
    AuthenticatedProviderLifecycle,
    admit_provider_lifecycle,
)
from msctl.jsonutil import canonical_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_FIELDS = {"arm", "config", "config_sha256"}


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


def _build_launcher_manifest(
    *,
    out: Path,
    scratch_root: Path,
    seed: int,
    profile_sha256: str,
    release_sha256: str,
    release_members_sha256: str,
    release_receipt_sha256: str | None = None,
    environment_receipt_sha256: str | None = None,
    run_manifest_sha256: str | None = None,
    cohort_assignment_sha256: str,
    code_commit: str,
    source_tree: str | None = None,
    bootstrap_receipt: Path,
    corpus_receipt: Path,
    runs: Sequence[dict[str, object]],
    lifecycle: AuthenticatedProviderLifecycle | None = None,
) -> dict[str, object]:
    """Resolve dynamic receipt hashes into Task3/5's closed manifest schema."""

    if type(seed) is not int or seed not in SEEDS:
        raise LaunchManifestError("seed must be assigned to AWS")
    for label, value in (
        ("profile", profile_sha256),
        ("release", release_sha256),
        ("release members", release_members_sha256),
        ("cohort assignment", cohort_assignment_sha256),
    ):
        _sha256(value, label=label)
    if not isinstance(code_commit, str) or _COMMIT_RE.fullmatch(code_commit) is None:
        raise LaunchManifestError("code commit must be a full lowercase Git commit")
    v3_values = {
        "environment_receipt_sha256": environment_receipt_sha256,
        "release_receipt_sha256": release_receipt_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "source_tree": source_tree,
    }
    if any(value is not None for value in v3_values.values()):
        if any(value is None for value in v3_values.values()):
            raise LaunchManifestError(
                "v3 launcher mirror context must be complete"
            )
        for label, digest in (
            ("release receipt", release_receipt_sha256),
            ("environment receipt", environment_receipt_sha256),
            ("run manifest", run_manifest_sha256),
        ):
            _sha256(digest, label=label)
        if (
            not isinstance(source_tree, str)
            or _COMMIT_RE.fullmatch(source_tree) is None
        ):
            raise LaunchManifestError(
                "source tree must be a full lowercase Git object"
            )
    scratch = scratch_root.resolve(strict=True)
    bootstrap_hash = _hash_regular(
        bootstrap_receipt,
        label="bootstrap receipt",
    )
    corpus_hash = _hash_regular(corpus_receipt, label="corpus receipt")
    corpus = _load_object(corpus_receipt, label="corpus receipt")
    build_id = _sha256(
        corpus.get("build_id"),
        label="corpus build ID",
    )
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
            arm not in ARMS
            or arm in by_arm
            or not isinstance(config, str)
            or not config
            or config.startswith("/")
            or "\\" in config
            or any(part in {"", ".", ".."} for part in config.split("/"))
        ):
            raise LaunchManifestError("run binding identity is invalid")
        expected_config = f"{CONFIG_ROOT}/{arm}-s{seed}.yaml"
        if config != expected_config:
            raise LaunchManifestError(
                f"{arm} config must match the v3 assigned seed"
            )
        _sha256(row["config_sha256"], label=f"{arm} config")
        by_arm[str(arm)] = row
    if set(by_arm) != set(ARMS):
        raise LaunchManifestError("launcher pair must contain Dense and Split90")
    corpus_relative = _relative_inside(
        corpus_receipt,
        scratch,
        label="corpus receipt",
    )
    if corpus_relative != DATASET_RECEIPT_PATH:
        raise LaunchManifestError(
            "corpus receipt must be dataset/receipt.json under scratch"
        )
    base_port = 29_500 + seed * 2
    launch_runs = []
    for arm, port, affinity in (
        ("dense", base_port, [0, 95]),
        ("split90", base_port + 1, [96, 191]),
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
        "corpus_receipt": {
            "build_id": build_id,
            "ordered_stream_sha256": ordered_sha256,
            "path": corpus_relative,
            "sha256": corpus_hash,
        },
        "release_members_sha256": release_members_sha256,
        "release_sha256": release_sha256,
        "runs": launch_runs,
    }
    if lifecycle is None:
        manifest.update(
            {
                "cohort_id": COHORT_ID,
                "profile_sha256": profile_sha256,
                "provider": PROVIDER,
                "schema_version": 2 if source_tree is not None else 1,
                "seed": seed,
            }
        )
    else:
        if (
            not isinstance(lifecycle, AuthenticatedProviderLifecycle)
            or lifecycle.binding.seed != seed
            or lifecycle.binding.profile_sha256 != profile_sha256
        ):
            raise LaunchManifestError(
                "authenticated provider lifecycle differs from launcher seed/profile"
            )
        manifest.update(
            {
                **lifecycle.binding.to_dict(),
                "schema_version": 3,
            }
        )
    if source_tree is not None:
        manifest.update(
            {
                "dataset_receipt_sha256": corpus_hash,
                "environment_receipt_sha256": environment_receipt_sha256,
                "release_receipt_sha256": release_receipt_sha256,
                "run_manifest_sha256": run_manifest_sha256,
                "source_tree": source_tree,
            }
        )
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


def build_launcher_manifest(
    *,
    out: Path,
    scratch_root: Path,
    seed: int,
    profile_sha256: str,
    release_sha256: str,
    release_members_sha256: str,
    release_receipt_sha256: str | None = None,
    environment_receipt_sha256: str | None = None,
    run_manifest_sha256: str | None = None,
    cohort_assignment_sha256: str,
    code_commit: str,
    source_tree: str | None = None,
    bootstrap_receipt: Path,
    corpus_receipt: Path,
    runs: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build the explicit legacy-compatible P5 launcher manifest."""

    return _build_launcher_manifest(
        out=out,
        scratch_root=scratch_root,
        seed=seed,
        profile_sha256=profile_sha256,
        release_sha256=release_sha256,
        release_members_sha256=release_members_sha256,
        release_receipt_sha256=release_receipt_sha256,
        environment_receipt_sha256=environment_receipt_sha256,
        run_manifest_sha256=run_manifest_sha256,
        cohort_assignment_sha256=cohort_assignment_sha256,
        code_commit=code_commit,
        source_tree=source_tree,
        bootstrap_receipt=bootstrap_receipt,
        corpus_receipt=corpus_receipt,
        runs=runs,
    )


def build_authenticated_launcher_manifest(
    *,
    out: Path,
    scratch_root: Path,
    seed: int,
    release_sha256: str,
    release_members_sha256: str,
    release_receipt_sha256: str,
    run_manifest_sha256: str,
    cohort_assignment_sha256: str,
    code_commit: str,
    source_tree: str,
    bootstrap_receipt: Path,
    corpus_receipt: Path,
    runs: Sequence[dict[str, object]],
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    runtime_sbom_path: Path | str,
    objective_controls_amendment_path: Path | str,
    store: object,
    account_id: str,
    instance_id: str,
    boot_id: str,
    expected_selection_version_id: str,
    identity_verifier: object,
    approval_verifier: object,
    trusted_public_key_sha256: str,
) -> dict[str, object]:
    """Build a selected-provider manifest after exact authority re-entry."""

    lifecycle = admit_provider_lifecycle(
        authority_root=authority_root,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        runtime_sbom_path=runtime_sbom_path,
        objective_controls_amendment_path=(
            objective_controls_amendment_path
        ),
        store=store,
        account_id=account_id,
        instance_id=instance_id,
        boot_id=boot_id,
        seed=seed,
        expected_selection_version_id=expected_selection_version_id,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    return _build_launcher_manifest(
        out=out,
        scratch_root=scratch_root,
        seed=seed,
        profile_sha256=lifecycle.binding.profile_sha256,
        release_sha256=release_sha256,
        release_members_sha256=release_members_sha256,
        release_receipt_sha256=release_receipt_sha256,
        environment_receipt_sha256=(
            lifecycle.binding.qualification_environment_receipt_sha256
        ),
        run_manifest_sha256=run_manifest_sha256,
        cohort_assignment_sha256=cohort_assignment_sha256,
        code_commit=code_commit,
        source_tree=source_tree,
        bootstrap_receipt=bootstrap_receipt,
        corpus_receipt=corpus_receipt,
        runs=runs,
        lifecycle=lifecycle,
    )


def _run_binding(value: str) -> dict[str, object]:
    try:
        row = json.loads(value, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("run binding must be JSON") from error
    if not isinstance(row, dict):
        raise argparse.ArgumentTypeError("run binding must be an object")
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--release-members-sha256", required=True)
    parser.add_argument("--release-receipt-sha256")
    parser.add_argument("--environment-receipt-sha256")
    parser.add_argument("--run-manifest-sha256")
    parser.add_argument("--cohort-assignment-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--source-tree")
    parser.add_argument("--bootstrap-receipt", type=Path, required=True)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--run", type=_run_binding, action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        manifest = build_launcher_manifest(
            out=arguments.out,
            scratch_root=arguments.scratch_root,
            seed=arguments.seed,
            profile_sha256=arguments.profile_sha256,
            release_sha256=arguments.release_sha256,
            release_members_sha256=arguments.release_members_sha256,
            release_receipt_sha256=arguments.release_receipt_sha256,
            environment_receipt_sha256=(
                arguments.environment_receipt_sha256
            ),
            run_manifest_sha256=arguments.run_manifest_sha256,
            cohort_assignment_sha256=arguments.cohort_assignment_sha256,
            code_commit=arguments.code_commit,
            source_tree=arguments.source_tree,
            bootstrap_receipt=arguments.bootstrap_receipt,
            corpus_receipt=arguments.corpus_receipt,
            runs=arguments.run,
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
