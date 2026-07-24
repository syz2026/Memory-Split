"""Deterministic, non-provisioning AWS v3 fleet plans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .aws_selection import (
    HardwareAmendment,
    ProviderSelection,
    V3_PROFILE_IDS,
    V3_SEEDS,
    load_hardware_amendment,
    load_provider_selection,
)
from .contracts import RunManifest, load_run_manifest
from .errors import MsctlError
from .fsutil import open_directory, rename_noreplace_at
from .jsonutil import canonical_json, canonical_sha256, require_sha256


FLEET_PLAN_TYPE = "memorysplit-aws-explicit-fleet-v3"
_MAX_PLAN_BYTES = 262_144
_INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PLAN_FIELDS = {
    "schema_version",
    "plan_type",
    "provider",
    "selected_profile_id",
    "profile_sha256",
    "provider_selection_sha256",
    "hardware_amendment_sha256",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "instance_type",
    "region",
    "ami_id",
    "container_image",
    "container_digest",
    "release_sha256",
    "dataset_sha256",
    "sealed_evaluation_sha256",
    "source_commit",
    "mixed_profiles",
    "protected_outcomes_inspected",
    "scheduling",
    "manifests",
    "instances",
}
_SCHEDULING_FIELDS = {
    "strategy",
    "seed_order",
    "instance_order",
    "max_active_pairs_per_instance",
    "wave_order",
}
_MANIFEST_FIELDS = {"seed", "path", "sha256", "instance_id", "wave"}
_INSTANCE_FIELDS = {"instance_id", "seeds", "max_active_pairs", "waves"}
_WAVE_FIELDS = {"wave", "seed", "manifest_sha256"}


@dataclass(frozen=True)
class FleetManifestBinding:
    seed: int
    path: str
    sha256: str
    instance_id: str
    wave: int


@dataclass(frozen=True)
class FleetInstance:
    instance_id: str
    seeds: tuple[int, ...]
    max_active_pairs: int
    waves: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FleetPlan:
    provider: str
    selected_profile_id: str
    profile_sha256: str
    provider_selection_sha256: str
    hardware_amendment_sha256: str
    cohort_assignment_sha256: str
    preregistration_sha256: str
    instance_type: str
    release_sha256: str
    dataset_sha256: str
    sealed_evaluation_sha256: str
    source_commit: str
    manifests: tuple[FleetManifestBinding, ...]
    instances: tuple[FleetInstance, ...]
    sha256: str
    path: Path | None
    value: dict[str, object]

    def binding_for_seed(self, seed: int) -> FleetManifestBinding:
        matches = [binding for binding in self.manifests if binding.seed == seed]
        if len(matches) != 1:
            _fail(
                "fleet plan does not contain exactly one binding for the seed",
                details={"seed": seed},
            )
        return matches[0]

    def instance_for_seed(self, seed: int) -> str:
        return self.binding_for_seed(seed).instance_id


def _fail(
    message: str,
    *,
    details: dict[str, object] | None = None,
    code: str = "FLEET_PLAN_INVALID",
) -> None:
    raise MsctlError(code, message, details=details)


def _exact(
    value: dict[str, object],
    fields: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != fields:
        _fail(
            f"{label} has missing or unknown fields",
            details={
                "missing": sorted(fields - actual),
                "unknown": sorted(actual - fields),
            },
        )


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object")
    return dict(value)


def _sha256(value: object, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except MsctlError as error:
        raise MsctlError(
            "FLEET_PLAN_INVALID",
            f"{label} must be lowercase SHA-256",
        ) from error


def _read_plan(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        before = path.stat(follow_symlinks=False)
        if path.is_symlink() or not path.is_file() or before.st_nlink != 1:
            _fail("fleet plan must be one singly linked regular file")
        if before.st_size <= 0 or before.st_size > _MAX_PLAN_BYTES:
            _fail("fleet plan exceeds the bounded plan size")
        data = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise MsctlError(
            "FLEET_PLAN_INVALID",
            "fleet plan cannot be read",
        ) from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(data) != after.st_size:
        _fail("fleet plan changed while being read")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail("fleet plan contains a duplicate field")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: _fail(
                f"fleet plan contains non-finite {value}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "FLEET_PLAN_INVALID",
            "fleet plan must contain one valid UTF-8 JSON object",
        ) from error
    value = _object(decoded, label="fleet plan")
    if data != canonical_json(value) + b"\n":
        _fail("fleet plan must use canonical JSON plus one newline")
    return data, value


def _validate_profile_selection(
    profile: object,
    selection: ProviderSelection,
) -> None:
    profile_id = getattr(profile, "profile_id", None)
    if (
        profile_id not in V3_PROFILE_IDS
        or getattr(profile, "provider", None) != selection.provider
        or profile_id != selection.selected_profile_id
        or getattr(profile, "sha256", None) != selection.profile_sha256
        or getattr(profile, "instance_type", None) != selection.instance_type
        or tuple(getattr(profile, "assigned_seeds", ())) != V3_SEEDS
        or tuple(getattr(profile, "train_groups", ())) != (4, 4)
        or selection.seeds != V3_SEEDS
        or selection.value.get("mixed_profiles") is not False
        or selection.value.get("protected_outcomes_inspected") is not False
    ):
        _fail("fleet profile conflicts with the provider selection")


def _instance_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(sorted(values))
    if (
        not 1 <= len(ids) <= 4
        or len(ids) != len(set(ids))
        or any(
            not isinstance(instance_id, str)
            or _INSTANCE_ID_RE.fullmatch(instance_id) is None
            for instance_id in ids
        )
    ):
        _fail("fleet plan requires one to four distinct explicit EC2 instance IDs")
    return ids


def _load_manifests(
    manifest_paths: Sequence[Path | str],
    *,
    repo_root: Path | str,
    profile: object,
    selection: ProviderSelection,
) -> tuple[tuple[str, RunManifest], ...]:
    paths = tuple(Path(path) for path in manifest_paths)
    if len(paths) != 10:
        _fail("fleet plan requires exactly ten run manifest paths")
    rendered_paths = tuple(str(path.resolve()) for path in paths)
    if len(set(rendered_paths)) != len(rendered_paths):
        _fail("fleet plan contains a duplicate run manifest path")
    loaded: list[tuple[str, RunManifest]] = []
    for path, rendered in zip(paths, rendered_paths, strict=True):
        try:
            manifest = load_run_manifest(path, repo_root=repo_root)
        except MsctlError as error:
            raise MsctlError(
                "FLEET_PLAN_INVALID",
                "fleet run manifest is invalid",
                details={"path": str(path), "cause": error.code},
            ) from error
        loaded.append((rendered, manifest))
    seeds = [manifest.seed for _, manifest in loaded]
    if sorted(seeds) != list(V3_SEEDS) or len(set(seeds)) != len(seeds):
        _fail(
            "fleet manifests must contain each seed exactly once",
            details={"seeds": sorted(seeds)},
        )
    _validate_profile_selection(profile, selection)
    expected_common: tuple[object, ...] | None = None
    for _, manifest in loaded:
        common = (
            manifest.source_commit,
            manifest.release_sha256,
            manifest.dataset_sha256,
            manifest.cohort_assignment_sha256,
            manifest.preregistration_sha256,
            manifest.hardware_amendment_sha256,
            manifest.provider_selection_sha256,
            manifest.profile_sha256,
            manifest.sealed_evaluation_sha256,
        )
        if (
            manifest.schema_version != 3
            or manifest.provider != selection.provider
            or manifest.profile_sha256 != selection.profile_sha256
            or manifest.provider_selection_sha256 != selection.sha256
            or manifest.hardware_amendment_sha256
            != selection.amendment_sha256
            or manifest.cohort_assignment_sha256
            != selection.cohort_assignment_sha256
            or manifest.preregistration_sha256
            != selection.preregistration_sha256
            or len(manifest.runs) != 2
            or {run.arm for run in manifest.runs} != {"dense", "split90"}
        ):
            _fail(
                "fleet manifest conflicts with the selected profile or receipt",
                details={"seed": manifest.seed},
            )
        if expected_common is None:
            expected_common = common
        elif common != expected_common:
            _fail("fleet manifests do not share one immutable provenance set")
    return tuple(sorted(loaded, key=lambda item: item[1].seed))


def create_fleet_plan(
    *,
    profile: object,
    selection: ProviderSelection,
    manifest_paths: Sequence[Path | str],
    instance_ids: Iterable[str],
    repo_root: Path | str,
) -> dict[str, object]:
    """Create one deterministic plan without calling any AWS API."""

    ids = _instance_ids(instance_ids)
    loaded = _load_manifests(
        manifest_paths,
        repo_root=repo_root,
        profile=profile,
        selection=selection,
    )
    first = loaded[0][1]
    manifest_rows: list[dict[str, object]] = []
    for path, manifest in loaded:
        manifest_rows.append(
            {
                "seed": manifest.seed,
                "path": path,
                "sha256": manifest.sha256,
                "instance_id": ids[manifest.seed % len(ids)],
                "wave": manifest.seed // len(ids),
            }
        )
    by_seed = {int(row["seed"]): row for row in manifest_rows}
    instance_rows = []
    for index, instance_id in enumerate(ids):
        seeds = list(V3_SEEDS[index:: len(ids)])
        instance_rows.append(
            {
                "instance_id": instance_id,
                "seeds": seeds,
                "max_active_pairs": 1,
                "waves": [
                    {
                        "wave": int(by_seed[seed]["wave"]),
                        "seed": seed,
                        "manifest_sha256": by_seed[seed]["sha256"],
                    }
                    for seed in seeds
                ],
            }
        )
    return {
        "schema_version": 3,
        "plan_type": FLEET_PLAN_TYPE,
        "provider": selection.provider,
        "selected_profile_id": selection.selected_profile_id,
        "profile_sha256": selection.profile_sha256,
        "provider_selection_sha256": selection.sha256,
        "hardware_amendment_sha256": selection.amendment_sha256,
        "cohort_assignment_sha256": selection.cohort_assignment_sha256,
        "preregistration_sha256": selection.preregistration_sha256,
        "instance_type": selection.instance_type,
        "region": selection.region,
        "ami_id": selection.ami_id,
        "container_image": selection.container_image,
        "container_digest": selection.container_digest,
        "release_sha256": first.release_sha256,
        "dataset_sha256": first.dataset_sha256,
        "sealed_evaluation_sha256": first.sealed_evaluation_sha256,
        "source_commit": first.source_commit,
        "mixed_profiles": False,
        "protected_outcomes_inspected": False,
        "scheduling": {
            "strategy": "sorted-seed-round-robin",
            "seed_order": list(V3_SEEDS),
            "instance_order": list(ids),
            "max_active_pairs_per_instance": 1,
            "wave_order": "ascending",
        },
        "manifests": manifest_rows,
        "instances": instance_rows,
    }


def _validate_structure(
    value: object,
    *,
    profile: object,
    selection: ProviderSelection,
    path: Path | None,
) -> FleetPlan:
    plan = _object(value, label="fleet plan")
    _exact(plan, _PLAN_FIELDS, label="fleet plan")
    _validate_profile_selection(profile, selection)
    scheduling = _object(plan["scheduling"], label="fleet scheduling")
    _exact(scheduling, _SCHEDULING_FIELDS, label="fleet scheduling")
    raw_instances = plan["instances"]
    if not isinstance(raw_instances, list):
        _fail("fleet plan instances must be a list")
    instance_ids = _instance_ids(
        str(_object(row, label="fleet instance").get("instance_id"))
        for row in raw_instances
    )
    if [row.get("instance_id") for row in raw_instances if isinstance(row, dict)] != list(
        instance_ids
    ):
        _fail("fleet plan instances must use sorted instance order")
    if (
        plan["schema_version"] != 3
        or plan["plan_type"] != FLEET_PLAN_TYPE
        or plan["provider"] != selection.provider
        or plan["selected_profile_id"] != selection.selected_profile_id
        or plan["profile_sha256"] != selection.profile_sha256
        or plan["provider_selection_sha256"] != selection.sha256
        or plan["hardware_amendment_sha256"] != selection.amendment_sha256
        or plan["cohort_assignment_sha256"]
        != selection.cohort_assignment_sha256
        or plan["preregistration_sha256"]
        != selection.preregistration_sha256
        or plan["instance_type"] != selection.instance_type
        or plan["region"] != selection.region
        or plan["ami_id"] != selection.ami_id
        or plan["container_image"] != selection.container_image
        or plan["container_digest"] != selection.container_digest
        or plan["mixed_profiles"] is not False
        or plan["protected_outcomes_inspected"] is not False
        or scheduling
        != {
            "strategy": "sorted-seed-round-robin",
            "seed_order": list(V3_SEEDS),
            "instance_order": list(instance_ids),
            "max_active_pairs_per_instance": 1,
            "wave_order": "ascending",
        }
    ):
        _fail("fleet plan conflicts with the selected profile or receipt")
    for field in (
        "profile_sha256",
        "provider_selection_sha256",
        "hardware_amendment_sha256",
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "release_sha256",
        "dataset_sha256",
        "sealed_evaluation_sha256",
    ):
        _sha256(plan[field], label=f"fleet plan.{field}")
    if (
        not isinstance(plan["source_commit"], str)
        or _COMMIT_RE.fullmatch(plan["source_commit"]) is None
    ):
        _fail("fleet plan source commit is invalid")

    raw_manifests = plan["manifests"]
    if not isinstance(raw_manifests, list) or len(raw_manifests) != len(V3_SEEDS):
        _fail("fleet plan must bind exactly ten manifests")
    bindings: list[FleetManifestBinding] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for index, raw in enumerate(raw_manifests):
        row = _object(raw, label=f"fleet manifest[{index}]")
        _exact(row, _MANIFEST_FIELDS, label=f"fleet manifest[{index}]")
        seed = row["seed"]
        manifest_path = row["path"]
        digest = _sha256(
            row["sha256"],
            label=f"fleet manifest[{index}].sha256",
        )
        expected_instance = instance_ids[index % len(instance_ids)]
        expected_wave = index // len(instance_ids)
        if (
            isinstance(seed, bool)
            or seed != index
            or not isinstance(manifest_path, str)
            or not manifest_path
            or manifest_path in seen_paths
            or digest in seen_hashes
            or row["instance_id"] != expected_instance
            or isinstance(row["wave"], bool)
            or row["wave"] != expected_wave
        ):
            _fail(
                "fleet manifest bindings are incomplete or non-deterministic",
                details={"index": index},
            )
        seen_paths.add(manifest_path)
        seen_hashes.add(digest)
        bindings.append(
            FleetManifestBinding(
                seed=index,
                path=manifest_path,
                sha256=digest,
                instance_id=expected_instance,
                wave=expected_wave,
            )
        )

    instances: list[FleetInstance] = []
    for index, raw in enumerate(raw_instances):
        row = _object(raw, label=f"fleet instance[{index}]")
        _exact(row, _INSTANCE_FIELDS, label=f"fleet instance[{index}]")
        expected_seeds = list(V3_SEEDS[index:: len(instance_ids)])
        waves = row["waves"]
        if (
            row["instance_id"] != instance_ids[index]
            or row["seeds"] != expected_seeds
            or row["max_active_pairs"] != 1
            or not isinstance(waves, list)
            or len(waves) != len(expected_seeds)
        ):
            _fail("fleet instance assignment is not deterministic")
        parsed_waves: list[tuple[int, int]] = []
        for wave_index, raw_wave in enumerate(waves):
            wave = _object(
                raw_wave,
                label=f"fleet instance[{index}].waves[{wave_index}]",
            )
            _exact(
                wave,
                _WAVE_FIELDS,
                label=f"fleet instance[{index}].waves[{wave_index}]",
            )
            seed = expected_seeds[wave_index]
            binding = bindings[seed]
            if wave != {
                "wave": binding.wave,
                "seed": seed,
                "manifest_sha256": binding.sha256,
            }:
                _fail("fleet wave does not bind its manifest assignment")
            parsed_waves.append((binding.wave, seed))
        instances.append(
            FleetInstance(
                instance_id=instance_ids[index],
                seeds=tuple(expected_seeds),
                max_active_pairs=1,
                waves=tuple(parsed_waves),
            )
        )
    return FleetPlan(
        provider=selection.provider,
        selected_profile_id=selection.selected_profile_id,
        profile_sha256=selection.profile_sha256,
        provider_selection_sha256=selection.sha256,
        hardware_amendment_sha256=selection.amendment_sha256,
        cohort_assignment_sha256=selection.cohort_assignment_sha256,
        preregistration_sha256=selection.preregistration_sha256,
        instance_type=selection.instance_type,
        release_sha256=str(plan["release_sha256"]),
        dataset_sha256=str(plan["dataset_sha256"]),
        sealed_evaluation_sha256=str(plan["sealed_evaluation_sha256"]),
        source_commit=str(plan["source_commit"]),
        manifests=tuple(bindings),
        instances=tuple(instances),
        sha256=canonical_sha256(plan),
        path=path,
        value=plan,
    )


def validate_fleet_plan(
    value: object,
    *,
    profile: object,
    selection: ProviderSelection,
    manifest_paths: Sequence[Path | str] | None = None,
    repo_root: Path | str | None = None,
    path: Path | None = None,
) -> FleetPlan:
    """Validate a decoded plan and, when supplied, all ten manifest bytes."""

    plan = _validate_structure(
        value,
        profile=profile,
        selection=selection,
        path=path,
    )
    if manifest_paths is not None:
        if repo_root is None:
            _fail("fleet manifest validation requires a repository root")
        expected = create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=manifest_paths,
            instance_ids=[
                instance.instance_id for instance in plan.instances
            ],
            repo_root=repo_root,
        )
        if canonical_json(expected) != canonical_json(plan.value):
            _fail("fleet plan does not match the supplied manifest bytes")
    return plan


def validate_fleet_manifest(
    plan: FleetPlan,
    manifest: RunManifest,
    *,
    instance_id: str | None = None,
) -> FleetManifestBinding:
    """Resolve one seed only after checking all plan/manifest provenance."""

    binding = plan.binding_for_seed(manifest.seed)
    if (
        manifest.schema_version != 3
        or manifest.provider != plan.provider
        or manifest.sha256 != binding.sha256
        or manifest.profile_sha256 != plan.profile_sha256
        or manifest.provider_selection_sha256
        != plan.provider_selection_sha256
        or manifest.hardware_amendment_sha256
        != plan.hardware_amendment_sha256
        or manifest.cohort_assignment_sha256
        != plan.cohort_assignment_sha256
        or manifest.preregistration_sha256 != plan.preregistration_sha256
        or manifest.release_sha256 != plan.release_sha256
        or manifest.dataset_sha256 != plan.dataset_sha256
        or manifest.sealed_evaluation_sha256
        != plan.sealed_evaluation_sha256
        or manifest.source_commit != plan.source_commit
        or (instance_id is not None and instance_id != binding.instance_id)
    ):
        _fail("fleet plan does not bind this seed manifest and instance")
    return binding


def load_fleet_plan(
    path: Path | str,
    *,
    profile: object,
    selection: ProviderSelection,
    manifest_paths: Sequence[Path | str] | None = None,
    repo_root: Path | str | None = None,
) -> FleetPlan:
    """Load one duplicate-safe canonical plan."""

    plan_path = Path(path)
    _, value = _read_plan(plan_path)
    return validate_fleet_plan(
        value,
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=repo_root,
        path=plan_path,
    )


def write_fleet_plan(path: Path | str, value: dict[str, object]) -> Path:
    """Publish canonical plan bytes with exclusive-create semantics."""

    destination = Path(path)
    directory_fd = open_directory(
        destination.parent,
        label="fleet plan output",
        create=True,
    )
    temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        payload = canonical_json(value) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            rename_noreplace_at(
                directory_fd,
                temporary,
                directory_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise MsctlError(
                "FLEET_PLAN_EXISTS",
                "refusing to replace an existing fleet plan",
                details={"path": str(destination)},
            ) from error
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return destination


def plan_fleet(
    *,
    profile: object,
    amendment_path: Path | str,
    provider_selection_path: Path | str,
    manifest_paths: Sequence[Path | str],
    instance_ids: Sequence[str],
    repo_root: Path | str,
    out: Path | str,
    apply: bool,
) -> dict[str, object]:
    """Render, validate, and optionally publish an explicit fleet plan."""

    amendment: HardwareAmendment = load_hardware_amendment(amendment_path)
    selection = load_provider_selection(
        provider_selection_path,
        amendment=amendment,
        profile=profile,
    )
    value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        instance_ids=instance_ids,
        repo_root=repo_root,
    )
    validated = validate_fleet_plan(
        value,
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=repo_root,
    )
    result = {
        "plan": value,
        "plan_sha256": validated.sha256,
        "out": str(Path(out)),
        "published": False,
    }
    if apply:
        write_fleet_plan(out, value)
        result["published"] = True
    return result
