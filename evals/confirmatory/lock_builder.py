"""Receipts-driven StudyLockV3 construction and no-replace publication.

The builder consumes exactly ten local Task 3F collection receipt bodies,
their Task 3D finalization and Task 3C paired checkpoint receipt bodies, and
the 100 locally collected snapshot files. Every payload is rehashed against
its receipt identity and re-parsed through the reviewed production parsers;
no caller-provided identity is trusted. Snapshot-only study identity is
extracted through the authoritative combined snapshot parser with the exact
receipt-derived lifecycle operational metadata.

Publication installs exactly one ``study-lock.json`` member into one
content-addressed ``study-lock-{sha256}`` directory using private staging,
an atomic no-replace rename, and a final composite parent/name/descriptor/
membership/content check. Collisions are never reused or replaced, failed
staging or installs are atomically renamed to intact unpredictable
quarantines, and no authority path ever unlinks or removes a pathname. The
returned SHA-256 is the authoritative commitment; paths are informational
and must be reopened and reverified downstream.

Torch, the trainer, and the Task 3C/3D/3F receipt modules are imported
lazily inside functions so importing this module stays light.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any

from evals.confirmatory.aggregate import (
    STUDY_LOCK_FILE_NAME,
    CollectionReceiptEvidence,
    SnapshotEvaluationPlan,
    plan_snapshot_evaluations,
)
from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.sealing import (
    PATH_AUTHORITY,
    _DirectorySnapshot,
    _PinnedFile,
    _StagedRelease,
    _assert_directory_entry,
    _assert_directory_path,
    _assert_directory_snapshot,
    _assert_final_directory_binding,
    _assert_pinned_file,
    _capture_directory_snapshot,
    _close_staged,
    _file_flags,
    _lock_output,
    _make_staging,
    _open_directory,
    _quarantine_preserve_directory,
    _quarantine_staged_directory,
    _read_descriptor,
    _regular_file_state,
    _reject_existing_release,
    _reject_quarantine_blockers,
    _rename_noreplace_at,
    _safe_absolute_path,
    _write_all,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    ProviderSelectionBinding,
    SeedLifecycleBinding,
    StudyLockV3,
    _reconstructed_lifecycle_binding,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    snapshot_object_key,
)
from msctl.aws_hardware import PROVIDER_SELECTION_S3_KEY
from msctl.aws_lifecycle import (
    ProviderLifecycleBinding,
    lifecycle_operational_metadata,
)


STUDY_LOCK_DIRECTORY_PREFIX = "study-lock-"
STUDY_LOCK_MEMBER_MODE = 0o444
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPECTED_RUNS = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
_EXPECTED_SNAPSHOT_SLOTS = tuple(
    (seed, arm, step)
    for seed in SEEDS
    for arm in ARMS
    for step in SNAPSHOT_STEPS
)
# Checkpoint-receipt provenance fields that must equal the collection (and
# therefore the finalization) receipt byte-for-byte.
_CHECKPOINT_PROVENANCE_FIELDS = (
    "dataset_build_id",
    "dataset_receipt_sha256",
    "environment_receipt_sha256",
    "ordered_stream_sha256",
    "release_receipt_sha256",
    "release_sha256",
    "run_manifest_sha256",
    "source_commit",
    "source_tree",
)
_OBJECT_REFERENCE_FIELDS = ("uri", "sha256", "bytes", "version_id")


class LockBuildError(ValueError):
    """A fail-closed study-lock construction or publication error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "LOCK_BUILDER_INVALID",
    ) -> None:
        super().__init__(message)
        self.code = code


def _fail(message: str, *, code: str = "LOCK_BUILDER_INVALID") -> LockBuildError:
    return LockBuildError(f"study lock builder {message}", code=code)


def _run_mutation_hook(event: str, **context: object) -> None:
    """Deterministic race boundary; production intentionally performs no action."""

    del event, context


def _sha256_text(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SeedCollectionEvidence:
    """One seed's local collection, finalization, and checkpoint payloads."""

    collection: CollectionReceiptEvidence
    finalization_payload: bytes
    checkpoint_receipt_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.collection, CollectionReceiptEvidence):
            raise _fail(
                "seed evidence requires exact collection receipt evidence"
            )
        for label, payload in (
            ("finalization", self.finalization_payload),
            ("checkpoint receipt", self.checkpoint_receipt_payload),
        ):
            if not isinstance(payload, bytes) or not payload:
                raise _fail(
                    f"seed evidence {label} payload bytes are missing"
                )


@dataclass(frozen=True)
class RunStudyIdentity:
    """Snapshot-extracted study identity for one seed and arm run."""

    seed: int
    arm: str
    training_config_sha256: str
    model_config_sha256: str
    model_identity: str
    data_provenance_sha256: str
    config_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise _fail(
                "run study identity seed must be an exact integer 0 through 9"
            )
        if self.arm not in ARMS:
            raise _fail("run study identity arm must be dense or split90")
        for field in (
            "training_config_sha256",
            "model_config_sha256",
            "data_provenance_sha256",
            "config_fingerprint",
        ):
            _sha256_text(getattr(self, field), f"run study identity {field}")
        if not isinstance(self.model_identity, str) or not self.model_identity:
            raise _fail(
                "run study identity model identity must be a non-empty string"
            )


@dataclass(frozen=True)
class PublishedStudyLock:
    """SHA-256 authority plus informational reopen-and-verify paths."""

    authoritative_commitment: str
    output_dir: Path
    study_lock_path: Path


@dataclass(frozen=True)
class _AdmittedSeedEvidence:
    seed: int
    collection: CollectionReceiptEvidence
    collection_value: Mapping[str, Any]
    selection: ProviderSelectionBinding
    lifecycle: SeedLifecycleBinding
    arm_rows: tuple[Mapping[str, Any], Mapping[str, Any]]
    snapshot_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _AdmittedCohortEvidence:
    selection: ProviderSelectionBinding
    seeds: tuple[_AdmittedSeedEvidence, ...]


def _derive_provider_selection(
    value: Mapping[str, Any],
    *,
    seed: int,
) -> ProviderSelectionBinding:
    try:
        return ProviderSelectionBinding(
            cohort_id=value["cohort_id"],
            provider_selection_s3_key=PROVIDER_SELECTION_S3_KEY,
            provider_selection_sha256=value["provider_selection_sha256"],
            provider_selection_s3_version_id=value[
                "provider_selection_version_id"
            ],
            hardware_amendment_sha256=value["hardware_amendment_sha256"],
            selected_provider=value["provider"],
            profile_id=value["profile_id"],
            profile_sha256=value["profile_sha256"],
            runtime_lock_sha256=value["runtime_lock_sha256"],
            qualification_evidence_sha256=value[
                "qualification_evidence_sha256"
            ],
            environment_receipt_sha256=value["environment_receipt_sha256"],
            canary_receipt_sha256=value["canary_receipt_sha256"],
            approval_receipt_sha256=value[
                "qualification_approval_receipt_sha256"
            ],
            approval_public_key_sha256=value[
                "qualification_approval_public_key_sha256"
            ],
        )
    except (TypeError, ValueError) as error:
        raise _fail(
            f"seed {seed} collection receipt cannot derive one cohort "
            f"provider selection: {error}"
        ) from error


def _derive_seed_lifecycle(
    *,
    seed: int,
    collection: CollectionReceiptEvidence,
    collection_value: Mapping[str, Any],
    checkpoint_receipt: Any,
    finalization_arms: Sequence[Mapping[str, Any]],
    run_reference: Mapping[str, Any],
) -> SeedLifecycleBinding:
    try:
        return SeedLifecycleBinding(
            seed=seed,
            account_id=checkpoint_receipt.account_id,
            availability_zone=checkpoint_receipt.availability_zone,
            boot_id=collection_value["boot_id"],
            instance_id=collection_value["instance_id"],
            region=checkpoint_receipt.region,
            purchase_model=checkpoint_receipt.purchase_model,
            runtime_sbom_sha256=collection_value["runtime_sbom_sha256"],
            objective_controls_contract_sha256=collection_value[
                "objective_controls_contract_sha256"
            ],
            source_commit=collection_value["source_commit"],
            source_tree=collection_value["source_tree"],
            run_manifest_sha256=collection_value["run_manifest_sha256"],
            dense_operational_config_sha256=finalization_arms[0][
                "config_sha256"
            ],
            split90_operational_config_sha256=finalization_arms[1][
                "config_sha256"
            ],
            finalization_receipt_sha256=run_reference["sha256"],
            finalization_receipt_s3_uri=run_reference["uri"],
            finalization_receipt_bytes=run_reference["bytes"],
            finalization_receipt_s3_version_id=run_reference["version_id"],
            collection_receipt_sha256=collection.sha256,
            collection_receipt_s3_uri=collection.uri,
            collection_receipt_s3_version_id=collection.version_id,
        )
    except (TypeError, ValueError) as error:
        raise _fail(
            f"seed {seed} receipts cannot derive one seed lifecycle "
            f"binding: {error}"
        ) from error


def _checkpoint_lifecycle_binding(
    receipt: Any,
    *,
    seed: int,
) -> ProviderLifecycleBinding:
    try:
        return ProviderLifecycleBinding(
            cohort_id=receipt.cohort_id,
            provider=receipt.provider,
            profile_id=receipt.profile_id,
            profile_sha256=receipt.profile_sha256,
            hardware_amendment_sha256=receipt.hardware_amendment_sha256,
            provider_selection_sha256=receipt.provider_selection_sha256,
            provider_selection_version_id=(
                receipt.provider_selection_version_id
            ),
            runtime_lock_sha256=receipt.runtime_lock_sha256,
            runtime_sbom_sha256=receipt.runtime_sbom_sha256,
            qualification_evidence_sha256=(
                receipt.qualification_evidence_sha256
            ),
            qualification_environment_receipt_sha256=(
                receipt.qualification_environment_receipt_sha256
            ),
            qualification_canary_receipt_sha256=(
                receipt.qualification_canary_receipt_sha256
            ),
            qualification_approval_receipt_sha256=(
                receipt.qualification_approval_receipt_sha256
            ),
            qualification_approval_public_key_sha256=(
                receipt.qualification_approval_public_key_sha256
            ),
            objective_controls_contract_sha256=(
                receipt.objective_controls_contract_sha256
            ),
            account_id=receipt.account_id,
            instance_id=receipt.instance_id,
            boot_id=receipt.boot_id,
            region=receipt.region,
            availability_zone=receipt.availability_zone,
            purchase_model=receipt.purchase_model,
            seed=receipt.seed,
            arms=tuple(receipt.arms),
        )
    except (TypeError, ValueError) as error:
        raise _fail(
            f"seed {seed} checkpoint receipt cannot reconstruct one "
            f"authenticated provider lifecycle: {error}"
        ) from error


def _admit_seed_collection_evidence(
    evidence: SeedCollectionEvidence,
    *,
    seed: int,
) -> _AdmittedSeedEvidence:
    from cluster.aws.p5.run_finalization import (
        _RECEIPT_BINDING_FIELDS,
        _RECEIPT_FIELDS,
        parse_run_finalization_receipt_bytes,
    )
    from msctl.aws_collect import (
        COLLECTION_RECEIPT_FIELDS,
        parse_seed_collection_receipt_bytes,
    )
    from msctl.contracts import parse_paired_checkpoint_receipt_v3
    from msctl.errors import MsctlError

    collection = evidence.collection
    if hashlib.sha256(collection.payload).hexdigest() != collection.sha256:
        raise _fail(
            f"seed {seed} collection receipt hash does not match its "
            "payload bytes"
        )
    try:
        collection_value = parse_seed_collection_receipt_bytes(
            collection.payload,
            receipt_uri=collection.uri,
            receipt_sha256=collection.sha256,
            receipt_version_id=collection.version_id,
        )
    except ValueError as error:
        raise _fail(
            f"seed {seed} collection receipt is invalid: {error}"
        ) from error
    if collection_value["seed"] != seed:
        raise _fail(
            "evidence must be ordered by ascending seed 0 through 9; "
            f"position {seed} holds seed {collection_value['seed']}"
        )

    run_reference = collection_value["run_receipt"]
    finalization_payload = evidence.finalization_payload
    if (
        hashlib.sha256(finalization_payload).hexdigest()
        != run_reference["sha256"]
        or len(finalization_payload) != run_reference["bytes"]
    ):
        raise _fail(
            f"seed {seed} finalization payload hash or bytes differ from "
            "the collected run receipt reference"
        )
    try:
        finalization_value = parse_run_finalization_receipt_bytes(
            finalization_payload,
            receipt_uri=run_reference["uri"],
            receipt_sha256=run_reference["sha256"],
            receipt_version_id=run_reference["version_id"],
        )
    except ValueError as error:
        raise _fail(
            f"seed {seed} finalization receipt is invalid: {error}"
        ) from error
    if finalization_value["seed"] != seed:
        raise _fail(
            f"seed {seed} finalization receipt seed identity is crossed"
        )

    checkpoint_reference = collection_value["checkpoint_receipt"]
    checkpoint_payload = evidence.checkpoint_receipt_payload
    if (
        hashlib.sha256(checkpoint_payload).hexdigest()
        != checkpoint_reference["sha256"]
        or len(checkpoint_payload) != checkpoint_reference["bytes"]
    ):
        raise _fail(
            f"seed {seed} checkpoint receipt payload hash or bytes differ "
            "from the collected checkpoint receipt reference"
        )
    try:
        checkpoint_receipt = parse_paired_checkpoint_receipt_v3(
            checkpoint_payload,
            receipt_uri=checkpoint_reference["uri"],
            receipt_sha256=checkpoint_reference["sha256"],
            receipt_version_id=checkpoint_reference["version_id"],
        )
    except MsctlError as error:
        raise _fail(
            f"seed {seed} checkpoint receipt is invalid: {error}"
        ) from error
    if checkpoint_receipt.seed != seed:
        raise _fail(
            f"seed {seed} checkpoint receipt seed identity is crossed"
        )
    if checkpoint_receipt.provider_selection_sha256 is None:
        raise _fail(
            f"seed {seed} checkpoint receipt is legacy; the study lock "
            "requires the selected provider-aware checkpoint receipt"
        )

    shared_fields = sorted(
        (COLLECTION_RECEIPT_FIELDS & _RECEIPT_FIELDS)
        - {"checkpoint_receipt", "receipt_type", "request_id"}
    )
    for field in shared_fields:
        if collection_value[field] != finalization_value[field]:
            raise _fail(
                f"seed {seed} finalization {field} differs from its "
                "collection receipt"
            )
    finalization_checkpoint = finalization_value["checkpoint_receipt"]
    if any(
        finalization_checkpoint[field] != checkpoint_reference[field]
        for field in ("uri", "sha256", "version_id")
    ):
        raise _fail(
            f"seed {seed} finalization checkpoint receipt reference "
            "differs from the collection"
        )

    rows = collection_value["objects"]
    snapshot_rows = (*rows[0:5], *rows[6:11])
    log_rows = (rows[5], rows[11])
    checkpoint_object_rows = (rows[12], rows[13])
    finalization_arms = finalization_value["arms"]
    for arm_index, arm in enumerate(ARMS):
        arm_row = finalization_arms[arm_index]
        for step_index, item in enumerate(arm_row["snapshots"]):
            row = snapshot_rows[arm_index * len(SNAPSHOT_STEPS) + step_index]
            reference = item["object"]
            if any(
                reference[field] != row[field]
                for field in _OBJECT_REFERENCE_FIELDS
            ):
                raise _fail(
                    f"seed {seed} {arm} finalization snapshot step "
                    f"{item['step']} row was not collected pairwise"
                )
        log_reference = arm_row["log"]
        if any(
            log_reference[field] != log_rows[arm_index][field]
            for field in _OBJECT_REFERENCE_FIELDS
        ):
            raise _fail(
                f"seed {seed} {arm} finalization log row was not "
                "collected pairwise"
            )
        checkpoint_row = checkpoint_receipt.checkpoints[arm_index]
        if (
            checkpoint_row.arm != arm
            or checkpoint_row.seed != seed
            or checkpoint_row.run_id != arm_row["run_id"]
            or checkpoint_row.config_sha256 != arm_row["config_sha256"]
            or checkpoint_row.config_fingerprint
            != arm_row["config_fingerprint"]
            or checkpoint_row.world_size != arm_row["world_size"]
        ):
            raise _fail(
                f"seed {seed} {arm} checkpoint receipt arm run/config/"
                "fingerprint/world identity differs from the finalized arm"
            )
        collected_object = checkpoint_object_rows[arm_index]
        if (
            checkpoint_row.object.uri != collected_object["uri"]
            or checkpoint_row.object.sha256 != collected_object["sha256"]
            or checkpoint_row.object.bytes != collected_object["bytes"]
            or checkpoint_row.object.version_id
            != collected_object["version_id"]
        ):
            raise _fail(
                f"seed {seed} {arm} checkpoint object placement differs "
                "from the collected terminal checkpoint row"
            )

    for field, binding_field in _RECEIPT_BINDING_FIELDS.items():
        if collection_value[field] != getattr(
            checkpoint_receipt,
            binding_field,
        ):
            raise _fail(
                f"seed {seed} checkpoint receipt {binding_field} lifecycle "
                "differs from the collection receipt"
            )
    for field in _CHECKPOINT_PROVENANCE_FIELDS:
        if collection_value[field] != getattr(checkpoint_receipt, field):
            raise _fail(
                f"seed {seed} checkpoint receipt {field} provenance "
                "differs from the collection receipt"
            )

    selection = _derive_provider_selection(collection_value, seed=seed)
    lifecycle = _derive_seed_lifecycle(
        seed=seed,
        collection=collection,
        collection_value=collection_value,
        checkpoint_receipt=checkpoint_receipt,
        finalization_arms=finalization_arms,
        run_reference=run_reference,
    )
    receipt_binding = _checkpoint_lifecycle_binding(
        checkpoint_receipt,
        seed=seed,
    )
    if receipt_binding != _reconstructed_lifecycle_binding(
        selection,
        lifecycle,
    ):
        raise _fail(
            f"seed {seed} checkpoint receipt lifecycle differs from the "
            "cohort selection and derived seed lifecycle"
        )

    return _AdmittedSeedEvidence(
        seed=seed,
        collection=collection,
        collection_value=collection_value,
        selection=selection,
        lifecycle=lifecycle,
        arm_rows=(finalization_arms[0], finalization_arms[1]),
        snapshot_rows=snapshot_rows,
    )


def _admit_collection_evidence(
    evidence: Sequence[SeedCollectionEvidence],
) -> _AdmittedCohortEvidence:
    if (
        isinstance(evidence, (str, bytes))
        or not isinstance(evidence, Sequence)
    ):
        raise _fail(
            "requires an ordered sequence of seed collection evidence"
        )
    rows = tuple(evidence)
    if any(not isinstance(row, SeedCollectionEvidence) for row in rows):
        raise _fail(
            "requires SeedCollectionEvidence for every seed evidence row"
        )
    if len(rows) != len(SEEDS):
        raise _fail(
            "requires exactly ten seed collections, seeds 0 through 9 "
            "ascending"
        )
    admitted = tuple(
        _admit_seed_collection_evidence(row, seed=seed)
        for seed, row in zip(SEEDS, rows, strict=True)
    )
    if len({record.selection for record in admitted}) != 1:
        raise _fail(
            "collection receipts disagree on one cohort provider selection"
        )
    if len({
        (
            record.collection_value["release_sha256"],
            record.collection_value["release_receipt_sha256"],
        )
        for record in admitted
    }) != 1:
        raise _fail(
            "collection receipts disagree on one training release identity"
        )
    return _AdmittedCohortEvidence(
        selection=admitted[0].selection,
        seeds=admitted,
    )


def _read_local_payload(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> bytes:
    """Read one local regular singly-linked file through a pinned descriptor."""

    absolute = _safe_absolute_path(path, label)
    try:
        descriptor = os.open(absolute, _file_flags())
    except OSError as error:
        raise _fail(
            f"{label} is missing, a symlink, or cannot be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        state = _regular_file_state(before, label)
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise _fail(
                f"{label} byte count differs from its receipt reference"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise _fail(f"{label} changed while being read")
            digest.update(chunk)
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, before.st_size):
            raise _fail(f"{label} grew while being read")
        if _regular_file_state(os.fstat(descriptor), label) != state:
            raise _fail(f"{label} changed while being read")
        if (
            expected_sha256 is not None
            and digest.hexdigest() != expected_sha256
        ):
            raise _fail(
                f"{label} stream hash differs from its receipt reference"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_local_evidence_bytes(path: Path | str, *, label: str) -> bytes:
    """Descriptor-read one local evidence payload without trusting its path."""

    return _read_local_payload(path, label=label)


def _expected_snapshot_metadata(
    selection: ProviderSelectionBinding,
    admitted: _AdmittedSeedEvidence,
    arm_row: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    binding = _reconstructed_lifecycle_binding(selection, admitted.lifecycle)
    value = admitted.collection_value
    try:
        return lifecycle_operational_metadata(
            binding,
            run_id=arm_row["run_id"],
            arm=arm,
            config_sha256=arm_row["config_sha256"],
            dataset_receipt_sha256=value["dataset_receipt_sha256"],
            dataset_build_id=value["dataset_build_id"],
            ordered_stream_sha256=value["ordered_stream_sha256"],
            source_commit=value["source_commit"],
            source_tree=value["source_tree"],
        )
    except (TypeError, ValueError) as error:
        raise _fail(
            f"seed {admitted.seed} {arm} receipts cannot derive the "
            f"expected snapshot operational metadata: {error}"
        ) from error


def extract_run_study_identities(
    evidence: Sequence[SeedCollectionEvidence],
    *,
    snapshot_paths: Mapping[tuple[int, str, int], Path],
) -> tuple[RunStudyIdentity, ...]:
    """Extract twenty step-invariant run identities from local snapshots."""

    cohort = _admit_collection_evidence(evidence)
    if not isinstance(snapshot_paths, Mapping):
        raise _fail(
            "snapshot paths must map (seed, arm, step) slots to local files"
        )
    if set(snapshot_paths) != set(_EXPECTED_SNAPSHOT_SLOTS):
        raise _fail(
            "snapshot paths must cover exactly the 100 canonical "
            "(seed, arm, step) slots"
        )

    from train.trainer import parse_model_snapshot_bytes

    identities: list[RunStudyIdentity] = []
    for admitted in cohort.seeds:
        value = admitted.collection_value
        for arm_index, arm in enumerate(ARMS):
            arm_row = admitted.arm_rows[arm_index]
            expected_metadata = _expected_snapshot_metadata(
                cohort.selection,
                admitted,
                arm_row,
                arm,
            )
            run_identity: RunStudyIdentity | None = None
            for step_index, step in enumerate(SNAPSHOT_STEPS):
                row = admitted.snapshot_rows[
                    arm_index * len(SNAPSHOT_STEPS) + step_index
                ]
                label = f"seed {admitted.seed} {arm} snapshot step {step}"
                payload = _read_local_payload(
                    snapshot_paths[(admitted.seed, arm, step)],
                    label=label,
                    expected_sha256=row["sha256"],
                    expected_bytes=row["bytes"],
                )
                try:
                    state = parse_model_snapshot_bytes(
                        payload,
                        expected_operational_metadata=expected_metadata,
                        require_study_identity=True,
                    )
                except ValueError as error:
                    raise _fail(f"{label} is invalid: {error}") from error
                if state["step"] != step:
                    raise _fail(
                        f"{label} embeds step {state['step']} instead of "
                        "its slot step"
                    )
                if state["world_size"] != 4:
                    raise _fail(f"{label} world size is not four")
                identity_value = state["study_identity"]
                if (
                    identity_value["seed"] != admitted.seed
                    or identity_value["arm"] != arm
                    or identity_value["run_id"] != arm_row["run_id"]
                    or identity_value["data_receipt_sha256"]
                    != value["dataset_receipt_sha256"]
                    or identity_value["data_build_id"]
                    != value["dataset_build_id"]
                    or identity_value["ordered_stream_sha256"]
                    != value["ordered_stream_sha256"]
                ):
                    raise _fail(
                        f"{label} study identity differs from the admitted "
                        "run/config/dataset receipts"
                    )
                if state["config_fingerprint"] != arm_row[
                    "config_fingerprint"
                ]:
                    raise _fail(
                        f"{label} config fingerprint differs from the "
                        "finalized arm fingerprint"
                    )
                extracted = RunStudyIdentity(
                    seed=admitted.seed,
                    arm=arm,
                    training_config_sha256=identity_value["config_sha256"],
                    model_config_sha256=identity_value["model_cfg_sha256"],
                    model_identity=identity_value["model_identity"],
                    data_provenance_sha256=identity_value[
                        "data_provenance_sha256"
                    ],
                    config_fingerprint=state["config_fingerprint"],
                )
                if run_identity is None:
                    run_identity = extracted
                elif run_identity != extracted:
                    raise _fail(
                        f"seed {admitted.seed} {arm} snapshots drift their "
                        "study identity across the five steps (cross-step "
                        "invariant)"
                    )
            assert run_identity is not None
            identities.append(run_identity)
    return tuple(identities)


def _validated_run_identities(
    run_study_identities: Sequence[RunStudyIdentity],
    cohort: _AdmittedCohortEvidence,
) -> dict[tuple[int, str], RunStudyIdentity]:
    if (
        isinstance(run_study_identities, (str, bytes))
        or not isinstance(run_study_identities, Sequence)
    ):
        raise _fail("run study identities must be an ordered sequence")
    rows = tuple(run_study_identities)
    if any(not isinstance(row, RunStudyIdentity) for row in rows):
        raise _fail(
            "run study identities must be RunStudyIdentity values"
        )
    if len(rows) != len(_EXPECTED_RUNS):
        raise _fail(
            "requires exactly twenty run study identities in frozen "
            "seed and arm order"
        )
    if tuple((row.seed, row.arm) for row in rows) != _EXPECTED_RUNS:
        raise _fail(
            "run study identities are not in frozen seed and arm order"
        )
    identities: dict[tuple[int, str], RunStudyIdentity] = {}
    for row in rows:
        arm_row = cohort.seeds[row.seed].arm_rows[ARMS.index(row.arm)]
        if row.config_fingerprint != arm_row["config_fingerprint"]:
            raise _fail(
                f"seed {row.seed} {row.arm} run study identity fingerprint "
                "differs from the finalized arm fingerprint"
            )
        identities[(row.seed, row.arm)] = row
    return identities


def _prove_built_lock(
    lock: StudyLockV3,
    *,
    cohort: _AdmittedCohortEvidence,
) -> tuple[SnapshotEvaluationPlan, ...]:
    from msctl.aws_collect import parse_seed_collection_receipt_bytes

    for admitted in cohort.seeds:
        collection = admitted.collection
        try:
            parse_seed_collection_receipt_bytes(
                collection.payload,
                receipt_uri=collection.uri,
                receipt_sha256=collection.sha256,
                receipt_version_id=collection.version_id,
                expected_binding=lock.lifecycle_binding(admitted.seed),
            )
        except ValueError as error:
            raise _fail(
                f"seed {admitted.seed} collection receipt does not "
                f"re-prove the built lock lifecycle: {error}"
            ) from error
    lock_sha256 = canonical_sha256(lock.to_dict())
    plans = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=tuple(
            admitted.collection for admitted in cohort.seeds
        ),
    )
    if len(plans) != 100:
        raise _fail(
            "built study lock did not produce exactly 100 snapshot "
            "evaluation plans"
        )
    return plans


def build_study_lock_v3(
    *,
    sealed_evaluation_release_sha256: str,
    evidence: Sequence[SeedCollectionEvidence],
    run_study_identities: Sequence[RunStudyIdentity],
) -> StudyLockV3:
    """Construct and receipt-prove the production StudyLockV3."""

    cohort = _admit_collection_evidence(evidence)
    identities = _validated_run_identities(run_study_identities, cohort)
    slots: list[dict[str, Any]] = []
    for admitted in cohort.seeds:
        value = admitted.collection_value
        checkpoint_reference = value["checkpoint_receipt"]
        receipt_key = checkpoint_receipt_key(
            admitted.seed,
            checkpoint_reference["sha256"],
        )
        for arm_index, arm in enumerate(ARMS):
            arm_row = admitted.arm_rows[arm_index]
            identity = identities[(admitted.seed, arm)]
            for step_index, step in enumerate(SNAPSHOT_STEPS):
                row = admitted.snapshot_rows[
                    arm_index * len(SNAPSHOT_STEPS) + step_index
                ]
                slots.append(
                    {
                        "seed": admitted.seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": row["sha256"],
                        "s3_object_key": snapshot_object_key(
                            admitted.seed,
                            arm,
                            step,
                            row["sha256"],
                        ),
                        "s3_version_id": row["version_id"],
                        "checkpoint_receipt_sha256": checkpoint_reference[
                            "sha256"
                        ],
                        "checkpoint_receipt_s3_object_key": receipt_key,
                        "checkpoint_receipt_s3_version_id": (
                            checkpoint_reference["version_id"]
                        ),
                        "provider_selection_sha256": (
                            cohort.selection.provider_selection_sha256
                        ),
                        "provider_selection_s3_version_id": (
                            cohort.selection.provider_selection_s3_version_id
                        ),
                        "snapshot_version": 2,
                        "training_run_id": arm_row["run_id"],
                        "config_fingerprint": identity.config_fingerprint,
                        "training_config_sha256": (
                            identity.training_config_sha256
                        ),
                        "model_config_sha256": identity.model_config_sha256,
                        "model_identity": identity.model_identity,
                        "data_provenance_sha256": (
                            identity.data_provenance_sha256
                        ),
                        "data_receipt_sha256": value["dataset_receipt_sha256"],
                        "data_build_id": value["dataset_build_id"],
                        "ordered_stream_sha256": value[
                            "ordered_stream_sha256"
                        ],
                        "world_size": 4,
                        "tokens_per_step": STUDY_TARGETS_PER_UPDATE,
                    }
                )
    try:
        lock = StudyLockV3.from_dict(
            {
                "record_type": STUDY_LOCK_SCHEMA_V3,
                "schema_version": STUDY_CONTRACT_VERSION,
                "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
                "sealed_evaluation_release_sha256": (
                    sealed_evaluation_release_sha256
                ),
                "provider_selection": cohort.selection.to_dict(),
                "snapshots": slots,
                "seed_lifecycles": [
                    admitted.lifecycle.to_dict()
                    for admitted in cohort.seeds
                ],
            }
        )
    except (TypeError, ValueError) as error:
        raise _fail(
            f"admitted evidence cannot construct one v3 study lock: {error}"
        ) from error
    _prove_built_lock(lock, cohort=cohort)
    return lock


def _write_lock_member(
    directory_fd: int,
    name: str,
    content: bytes,
) -> _PinnedFile:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        _write_all(descriptor, content)
        os.fchmod(descriptor, STUDY_LOCK_MEMBER_MODE)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        state = _regular_file_state(details, f"staged {name}")
        pinned = _PinnedFile(
            parent_fd=directory_fd,
            name=name,
            descriptor=descriptor,
            content=content,
            identity=(details.st_dev, details.st_ino),
            state=state,
        )
        _assert_lock_member(pinned)
        return pinned
    except BaseException:
        os.close(descriptor)
        raise


def _assert_lock_member(pinned: _PinnedFile) -> None:
    _assert_pinned_file(pinned, f"staged {pinned.name}")
    details = os.fstat(pinned.descriptor)
    if stat.S_IMODE(details.st_mode) != STUDY_LOCK_MEMBER_MODE:
        raise _fail(f"staged {pinned.name} has an unsafe mode")
    observed = _read_descriptor(
        pinned.descriptor,
        details.st_size,
        f"staged {pinned.name}",
    )
    if observed != pinned.content:
        raise _fail(f"staged {pinned.name} content verification failed")
    _assert_pinned_file(pinned, f"staged {pinned.name}")


def _stage_lock(
    output_fd: int,
    directory_name: str,
    lock_bytes: bytes,
) -> _StagedRelease:
    staging_name, staging_fd = _make_staging(output_fd, directory_name)
    files: list[_PinnedFile] = []
    try:
        files.append(
            _write_lock_member(staging_fd, STUDY_LOCK_FILE_NAME, lock_bytes)
        )
        for pinned in files:
            _assert_lock_member(pinned)
        _assert_directory_entry(
            output_fd,
            staging_name,
            staging_fd,
            "private study-lock staging",
        )
        os.fsync(staging_fd)
        snapshot = _capture_directory_snapshot(
            staging_fd,
            "private study-lock staging",
            expected_names=(STUDY_LOCK_FILE_NAME,),
            exact_mode=0o700,
        )
        return _StagedRelease(staging_name, staging_fd, tuple(files), snapshot)
    except BaseException:
        for pinned in reversed(files):
            try:
                os.close(pinned.descriptor)
            except OSError:
                pass
        try:
            _quarantine_preserve_directory(
                output_fd,
                staging_name,
                staging_fd,
                label="failed private study-lock staging",
            )
        finally:
            os.close(staging_fd)
        raise


def _verify_installed_lock(
    output_fd: int,
    directory_name: str,
    staged: _StagedRelease,
) -> _DirectorySnapshot:
    _assert_directory_entry(
        output_fd,
        directory_name,
        staged.descriptor,
        "installed study lock",
    )
    snapshot = _capture_directory_snapshot(
        staged.descriptor,
        "installed study lock",
        expected_names=(STUDY_LOCK_FILE_NAME,),
        exact_mode=0o700,
    )
    for pinned in staged.files:
        _assert_lock_member(pinned)
    os.fsync(staged.descriptor)
    _assert_directory_entry(
        output_fd,
        directory_name,
        staged.descriptor,
        "installed study lock",
    )
    _assert_directory_snapshot(
        staged.descriptor,
        snapshot,
        "installed study lock",
    )
    return snapshot


def publish_study_lock(
    lock: StudyLockV3,
    *,
    collection_receipts: Sequence[CollectionReceiptEvidence],
    output_root: Path,
) -> PublishedStudyLock:
    """Publish exactly one canonical study-lock.json with no-replace authority."""

    if not isinstance(lock, StudyLockV3):
        raise _fail("publication requires one StudyLockV3")
    lock_bytes = canonical_json_bytes(lock.to_dict())
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    plans = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=collection_receipts,
    )
    if len(plans) != 100:
        raise _fail(
            "study lock did not produce exactly 100 snapshot evaluation "
            "plans"
        )
    directory_name = f"{STUDY_LOCK_DIRECTORY_PREFIX}{lock_sha256}"
    output_path, output_fd = _open_directory(
        output_root,
        "study-lock output root",
    )
    staged: _StagedRelease | None = None
    published = False
    try:
        _lock_output(output_fd)
        _reject_quarantine_blockers(output_fd)
        _reject_existing_release(output_fd, directory_name)
        staged = _stage_lock(output_fd, directory_name, lock_bytes)
        _assert_directory_path(
            output_path,
            output_fd,
            "study-lock output root",
        )
        _assert_directory_entry(
            output_fd,
            staged.name,
            staged.descriptor,
            "private study-lock staging",
        )
        _assert_directory_snapshot(
            staged.descriptor,
            staged.snapshot,
            "private study-lock staging",
        )
        for pinned in staged.files:
            _assert_lock_member(pinned)
        _rename_noreplace_at(output_fd, staged.name, directory_name)
        installed_snapshot = _verify_installed_lock(
            output_fd,
            directory_name,
            staged,
        )
        os.fsync(output_fd)
        result = PublishedStudyLock(
            authoritative_commitment=lock_sha256,
            output_dir=output_path / directory_name,
            study_lock_path=(
                output_path / directory_name / STUDY_LOCK_FILE_NAME
            ),
        )
        _run_mutation_hook(
            "publish_before_final_binding",
            release_fd=staged.descriptor,
            parent_fd=output_fd,
            release_name=directory_name,
        )
        for pinned in staged.files:
            _assert_lock_member(pinned)
        _assert_final_directory_binding(
            parent_path=output_path,
            parent_fd=output_fd,
            directory_name=directory_name,
            directory_fd=staged.descriptor,
            snapshot=installed_snapshot,
            label="installed study lock",
        )
        published = True
        return result
    finally:
        if staged is not None:
            try:
                if not published:
                    _quarantine_staged_directory(output_fd, staged)
                    try:
                        os.fsync(output_fd)
                    except OSError:
                        pass
            finally:
                _close_staged(staged)
        os.close(output_fd)


__all__ = [
    "LockBuildError",
    "PATH_AUTHORITY",
    "PublishedStudyLock",
    "RunStudyIdentity",
    "STUDY_LOCK_DIRECTORY_PREFIX",
    "STUDY_LOCK_MEMBER_MODE",
    "SeedCollectionEvidence",
    "build_study_lock_v3",
    "extract_run_study_identities",
    "publish_study_lock",
    "read_local_evidence_bytes",
]
